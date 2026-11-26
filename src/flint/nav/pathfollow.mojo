"""Path Corridors, Time-Sliced Path Following, Off-Mesh Traversal States & Multi-Radius NavMesh Sets
(`flint.nav.pathfollow`, v5).

Implements:
1. `PathCorridor`: waypoint/poly corridor with progress tracking, visibility optimisation
   (`raycast_walkable` shortcutting for uniform filters), and validity checks after navmesh rebuilds.
2. `NavPathFollower`: game-agnostic agent steering state machine on top of `NavMesh`:
   - time-sliced A* (`NavPathQuery`, `max_iters_per_update` polygons per frame) or synchronous search,
   - automatic repath when the mesh `epoch` changes (obstacle carve / area mark) and the corridor
     is no longer walkable, when the agent leaves the corridor, or when it is stuck,
   - off-mesh link traversal states (`NAV_FOLLOW_OFFMESH_WAIT` -> `NAV_FOLLOW_OFFMESH_TRAVERSE`) with
     `NavFollowEvent`s so a game can play a jump/climb animation and report completion,
   - desired velocity output (apply it to a `CharacterController`, ORCA agent or raw transform).
3. `NavMeshSet`: one `NavMesh` per agent radius (Recast bakes per-radius erosion) with radius lookup.
"""
from std.collections import List
from std.math import sqrt, min, max, isfinite
from flint.core.math import Vec3
from flint.physics.world import PhysicsWorld
from flint.nav.navmesh import (
    NavMesh,
    NavMeshConfig,
    NavPath,
    NavPathQuery,
    NavQueryFilter,
    NavConvexVolume,
    NavObstacle,
    OFFMESH_JUMP,
    OFFMESH_DROP,
    NAV_QUERY_IN_PROGRESS,
    NAV_QUERY_SUCCESS,
    NAV_QUERY_FAILED,
)

comptime NAV_FOLLOW_IDLE: Int = 0
comptime NAV_FOLLOW_PLANNING: Int = 1
comptime NAV_FOLLOW_MOVING: Int = 2
comptime NAV_FOLLOW_OFFMESH_WAIT: Int = 3  # reached an off-mesh link start; waiting for `begin_offmesh()`
comptime NAV_FOLLOW_OFFMESH_TRAVERSE: Int = 4  # crossing the link (auto timed or until `complete_offmesh()`)
comptime NAV_FOLLOW_ARRIVED: Int = 5
comptime NAV_FOLLOW_FAILED: Int = 6

comptime NAV_EVT_PATH_READY: Int = 0
comptime NAV_EVT_PATH_FAILED: Int = 1
comptime NAV_EVT_REPATH: Int = 2
comptime NAV_EVT_STUCK: Int = 3
comptime NAV_EVT_OFFMESH_REACHED: Int = 4
comptime NAV_EVT_OFFMESH_BEGIN: Int = 5
comptime NAV_EVT_OFFMESH_END: Int = 6
comptime NAV_EVT_ARRIVED: Int = 7

comptime NAV_REPATH_NONE: Int = 0
comptime NAV_REPATH_MESH_CHANGED: Int = 1
comptime NAV_REPATH_OFF_CORRIDOR: Int = 2
comptime NAV_REPATH_STUCK: Int = 3


@fieldwise_init
struct NavFollowEvent(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var link_id: Int  # off-mesh link id (or -1)
    var link_kind: UInt8  # OFFMESH_JUMP / OFFMESH_DROP / OFFMESH_CLIMB
    var a: Vec3  # link start / event position
    var b: Vec3  # link end
    var reason: Int  # NAV_REPATH_* for NAV_EVT_REPATH


struct PathCorridor(Copyable, Movable):
    """Waypoint corridor along a `NavPath` with progress index and optimisation helpers."""
    var path: NavPath
    var index: Int  # current target waypoint
    var target: Vec3

    def __init__(out self):
        self.path = NavPath()
        self.index = 0
        self.target = Vec3()

    def reset(mut self, var path: NavPath, target: Vec3):
        self.path = path^
        self.index = 1 if len(self.path.waypoints) > 1 else 0
        self.target = target

    def clear(mut self):
        self.path = NavPath()
        self.index = 0

    @always_inline
    def valid_path(self) -> Bool:
        return self.path.found and len(self.path.waypoints) > 0

    def corner(self) -> Vec3:
        var n = len(self.path.waypoints)
        if n == 0:
            return self.target
        return self.path.waypoints[min(self.index, n - 1)]

    def at_last(self) -> Bool:
        return self.index >= len(self.path.waypoints) - 1

    def link_slot_at(self, wp_index: Int) -> Int:
        """Index into `path.used_off_mesh_links` of the link starting at waypoint `wp_index`, or -1."""
        for k in range(len(self.path.link_waypoint_idx)):
            if self.path.link_waypoint_idx[k] == wp_index:
                return k
        return -1

    def advance(mut self, pos: Vec3, radius: Float32):
        """Moves past reached corners (stops in front of off-mesh link starts so the follower can react)."""
        var n = len(self.path.waypoints)
        var r2 = radius * radius
        while self.index < n - 1:
            if self.link_slot_at(self.index) >= 0:
                break
            var wp = self.path.waypoints[self.index]
            var dx = wp.x - pos.x
            var dz = wp.z - pos.z
            if dx * dx + dz * dz <= r2:
                self.index += 1
            else:
                break

    def optimize_visibility(mut self, nav: NavMesh, pos: Vec3, max_skips: Int = 2) -> Int:
        """Skips intermediate corners directly visible from `pos` (never across off-mesh links)."""
        var skipped = 0
        var n = len(self.path.waypoints)
        while skipped < max_skips and self.index + 1 < n:
            if self.link_slot_at(self.index) >= 0 or self.link_slot_at(self.index - 1) >= 0:
                break
            if nav.raycast_walkable(pos, self.path.waypoints[self.index + 1]):
                self.index += 1
                skipped += 1
            else:
                break
        return skipped

    def is_walkable(self, nav: NavMesh, pos: Vec3, max_segments: Int = 6) -> Bool:
        """True if `pos -> corner -> ...` (up to `max_segments`, stopping at off-mesh links) is still walkable."""
        var n = len(self.path.waypoints)
        if n == 0:
            return False
        var prev = pos
        var k = min(self.index, n - 1)
        var seg = 0
        while k < n and seg < max_segments:
            if not nav.raycast_walkable(prev, self.path.waypoints[k]):
                return False
            if self.link_slot_at(k) >= 0:
                break
            prev = self.path.waypoints[k]
            k += 1
            seg += 1
        return True

    def distance_to_corner(self, pos: Vec3) -> Float32:
        var c = self.corner()
        var dx = c.x - pos.x
        var dz = c.z - pos.z
        return sqrt(dx * dx + dz * dz)

    def distance_from_segment(self, pos: Vec3) -> Float32:
        """XZ distance from `pos` to the corridor segment `[wp[index-1], wp[index]]`."""
        var n = len(self.path.waypoints)
        if n == 0:
            return 0.0
        var b = self.path.waypoints[min(self.index, n - 1)]
        var a = self.path.waypoints[max(0, min(self.index, n - 1) - 1)]
        var abx = b.x - a.x
        var abz = b.z - a.z
        var l2 = abx * abx + abz * abz
        var t = Float32(0.0)
        if l2 > Float32(1.0e-8):
            t = max(Float32(0.0), min(Float32(1.0), ((pos.x - a.x) * abx + (pos.z - a.z) * abz) / l2))
        var dx = pos.x - (a.x + abx * t)
        var dz = pos.z - (a.z + abz * t)
        return sqrt(dx * dx + dz * dz)


struct NavPathFollower(Copyable, Movable):
    """Per-agent path following state machine (see module docs). Call `update(nav, pos, dt)` every
    frame; read `desired_velocity` (or `offmesh_position()` while traversing a link) and drain
    `events` via `take_events()`."""
    var state: Int
    var corridor: PathCorridor
    var query: NavPathQuery
    var filter: NavQueryFilter
    var goal: Vec3
    var has_goal: Bool
    var max_speed: Float32
    var arrival_radius: Float32
    var corner_radius: Float32
    var max_search_dist: Float32
    var sliced: Bool
    var max_iters_per_update: Int
    var nav_epoch: Int
    var optimize: Bool
    var off_corridor_dist: Float32

    # Off-mesh link traversal
    var auto_offmesh: Bool  # True: begin immediately & time the crossing; False: wait for begin/complete calls
    var offmesh_speed: Float32
    var offmesh_jump_height: Float32
    var offmesh_t: Float32
    var offmesh_duration: Float32
    var offmesh_slot: Int
    var offmesh_from: Vec3
    var offmesh_to: Vec3
    var offmesh_kind: UInt8
    var offmesh_link_id: Int
    var offmesh_landed: Bool  # set by `complete_offmesh()` so integrators can snap to the link end

    # Stuck detection
    var stuck_window: Float32
    var stuck_min_progress: Float32
    var stuck_timer: Float32
    var stuck_ref_dist: Float32
    var stuck_ref_index: Int
    var stuck_count: Int
    var max_stuck_repaths: Int

    var repath_count: Int
    var last_repath_reason: Int
    var desired_velocity: Vec3
    var position: Vec3
    var events: List[NavFollowEvent]

    def __init__(out self, max_speed: Float32 = 3.5, arrival_radius: Float32 = 0.25, sliced: Bool = False):
        self.state = NAV_FOLLOW_IDLE
        self.corridor = PathCorridor()
        self.query = NavPathQuery()
        self.filter = NavQueryFilter()
        self.goal = Vec3()
        self.has_goal = False
        self.max_speed = max_speed
        self.arrival_radius = arrival_radius
        self.corner_radius = 0.3
        self.max_search_dist = 10.0
        self.sliced = sliced
        self.max_iters_per_update = 16
        self.nav_epoch = -1
        self.optimize = True
        self.off_corridor_dist = 1.5
        self.auto_offmesh = True
        self.offmesh_speed = 4.0
        self.offmesh_jump_height = 0.8
        self.offmesh_t = 0.0
        self.offmesh_duration = 0.0
        self.offmesh_slot = -1
        self.offmesh_from = Vec3()
        self.offmesh_to = Vec3()
        self.offmesh_kind = OFFMESH_JUMP
        self.offmesh_link_id = -1
        self.offmesh_landed = False
        self.stuck_window = 1.0
        self.stuck_min_progress = 0.15
        self.stuck_timer = 0.0
        self.stuck_ref_dist = 0.0
        self.stuck_ref_index = 0
        self.stuck_count = 0
        self.max_stuck_repaths = 3
        self.repath_count = 0
        self.last_repath_reason = NAV_REPATH_NONE
        self.desired_velocity = Vec3()
        self.position = Vec3()
        self.events = List[NavFollowEvent]()

    def _emit(mut self, kind: Int, reason: Int = NAV_REPATH_NONE):
        self.events.append(NavFollowEvent(kind, self.offmesh_link_id, self.offmesh_kind, self.offmesh_from, self.offmesh_to, reason))

    def take_events(mut self) -> List[NavFollowEvent]:
        var out = self.events^
        self.events = List[NavFollowEvent]()
        return out^

    def has_event(self, kind: Int) -> Bool:
        for i in range(len(self.events)):
            if self.events[i].kind == kind:
                return True
        return False

    def _start_plan(mut self, nav: NavMesh, pos: Vec3):
        self.nav_epoch = nav.epoch
        self.stuck_timer = 0.0
        if self.sliced:
            self.query = nav.begin_path_query(pos, self.goal, self.filter, self.max_search_dist)
            self.state = NAV_FOLLOW_PLANNING
            self._poll_plan(nav)
        else:
            var p = nav.find_path_filtered(pos, self.goal, self.filter, self.max_search_dist)
            self._accept_path(p^)

    def _poll_plan(mut self, nav: NavMesh):
        if self.query.status == NAV_QUERY_IN_PROGRESS:
            _ = nav.update_path_query(self.query, self.max_iters_per_update)
        if self.query.status == NAV_QUERY_SUCCESS:
            var p = nav.finish_path_query(self.query)
            self._accept_path(p^)
        elif self.query.status == NAV_QUERY_FAILED:
            if self.query.epoch != nav.epoch:
                # Mesh changed mid-search: restart transparently
                var cur_pos = self.position
                self._start_plan(nav, cur_pos)
                return
            self.corridor.clear()
            self.state = NAV_FOLLOW_FAILED
            self._emit(NAV_EVT_PATH_FAILED)

    def _accept_path(mut self, var p: NavPath):
        if not p.found:
            self.corridor.clear()
            self.state = NAV_FOLLOW_FAILED
            self._emit(NAV_EVT_PATH_FAILED)
            return
        self.corridor.reset(p^, self.goal)
        self.state = NAV_FOLLOW_MOVING
        self.stuck_ref_index = self.corridor.index
        self.stuck_ref_dist = self.corridor.distance_to_corner(self.position)
        self.stuck_timer = 0.0
        self._emit(NAV_EVT_PATH_READY)

    def set_goal(mut self, nav: NavMesh, pos: Vec3, goal: Vec3):
        self.goal = goal
        self.has_goal = True
        self.position = pos
        self.stuck_count = 0
        self._start_plan(nav, pos)

    def stop(mut self):
        self.has_goal = False
        self.state = NAV_FOLLOW_IDLE
        self.corridor.clear()
        self.desired_velocity = Vec3()

    def repath(mut self, nav: NavMesh, reason: Int):
        self.repath_count += 1
        self.last_repath_reason = reason
        self._emit(NAV_EVT_REPATH, reason)
        var cur_pos = self.position
        self._start_plan(nav, cur_pos)

    def begin_offmesh(mut self):
        """Game acknowledges `NAV_EVT_OFFMESH_REACHED` (e.g. jump animation started)."""
        if self.state == NAV_FOLLOW_OFFMESH_WAIT:
            self.state = NAV_FOLLOW_OFFMESH_TRAVERSE
            self.offmesh_t = 0.0
            self._emit(NAV_EVT_OFFMESH_BEGIN)

    def complete_offmesh(mut self):
        """Game reports the traversal animation finished; resumes walking from the link end."""
        if self.state == NAV_FOLLOW_OFFMESH_TRAVERSE or self.state == NAV_FOLLOW_OFFMESH_WAIT:
            self.position = self.offmesh_to
            self.corridor.index = min(len(self.corridor.path.waypoints) - 1, self.corridor.path.link_waypoint_idx[self.offmesh_slot] + 2) if self.offmesh_slot >= 0 else self.corridor.index
            self.state = NAV_FOLLOW_MOVING
            self.offmesh_t = 1.0
            self.offmesh_landed = True
            self.stuck_timer = 0.0
            self.stuck_ref_index = self.corridor.index
            self.stuck_ref_dist = self.corridor.distance_to_corner(self.position)
            self._emit(NAV_EVT_OFFMESH_END)
            self.offmesh_slot = -1

    def offmesh_position(self) -> Vec3:
        """Interpolated position while traversing a link (parabolic arc for jumps/drops)."""
        var t = max(Float32(0.0), min(Float32(1.0), self.offmesh_t))
        var p = self.offmesh_from + (self.offmesh_to - self.offmesh_from) * t
        if self.offmesh_kind == OFFMESH_JUMP or self.offmesh_kind == OFFMESH_DROP:
            p.y += Float32(4.0) * self.offmesh_jump_height * t * (Float32(1.0) - t)
        return p

    def is_traversing_offmesh(self) -> Bool:
        return self.state == NAV_FOLLOW_OFFMESH_WAIT or self.state == NAV_FOLLOW_OFFMESH_TRAVERSE

    def update(mut self, nav: NavMesh, pos: Vec3, dt: Float32) -> Vec3:
        """Advances the follower; returns (and stores) the desired XZ velocity."""
        var safe_dt = dt if (isfinite(dt) and dt > Float32(0.0)) else Float32(0.0)
        self.position = pos
        if self.offmesh_landed:
            # Link completed but the caller has not teleported the agent to the link end yet
            var ldx = pos.x - self.offmesh_to.x
            var ldz = pos.z - self.offmesh_to.z
            if ldx * ldx + ldz * ldz > self.corner_radius * self.corner_radius:
                self.position = self.offmesh_to
            else:
                self.offmesh_landed = False
        self.desired_velocity = Vec3()
        if not self.has_goal:
            return Vec3()

        if self.state == NAV_FOLLOW_PLANNING:
            self._poll_plan(nav)
            if self.state != NAV_FOLLOW_MOVING:
                return Vec3()

        if self.state == NAV_FOLLOW_OFFMESH_WAIT:
            if self.auto_offmesh:
                self.begin_offmesh()
            else:
                return Vec3()
        if self.state == NAV_FOLLOW_OFFMESH_TRAVERSE:
            if self.auto_offmesh:
                self.offmesh_t += safe_dt / max(Float32(1.0e-3), self.offmesh_duration)
                if self.offmesh_t >= Float32(1.0):
                    self.complete_offmesh()
                else:
                    self.desired_velocity = (self.offmesh_to - self.offmesh_from) * (Float32(1.0) / max(Float32(1.0e-3), self.offmesh_duration))
                    return self.desired_velocity
            else:
                return Vec3()

        if self.state == NAV_FOLLOW_ARRIVED or self.state == NAV_FOLLOW_FAILED or self.state == NAV_FOLLOW_IDLE:
            return Vec3()

        # 1. Mesh changed (obstacle carve / area mark): keep the corridor if still walkable, else repath
        if nav.epoch != self.nav_epoch:
            self.nav_epoch = nav.epoch
            if not self.corridor.is_walkable(nav, self.position):
                self.repath(nav, NAV_REPATH_MESH_CHANGED)
                if self.state != NAV_FOLLOW_MOVING:
                    return Vec3()

        # 2. Corridor progress + optimisation
        self.corridor.advance(self.position, self.corner_radius)
        if self.optimize and self.filter.is_uniform():
            _ = self.corridor.optimize_visibility(nav, self.position)

        # 3. Left the corridor (pushed by crowd / physics): repath
        if self.corridor.distance_from_segment(self.position) > self.off_corridor_dist:
            self.repath(nav, NAV_REPATH_OFF_CORRIDOR)
            if self.state != NAV_FOLLOW_MOVING:
                return Vec3()

        var n_wp = len(self.corridor.path.waypoints)
        if n_wp == 0:
            return Vec3()

        # 4. Off-mesh link start reached?
        var slot = self.corridor.link_slot_at(self.corridor.index)
        if slot >= 0 and self.corridor.distance_to_corner(self.position) <= self.corner_radius:
            var wi = self.corridor.path.link_waypoint_idx[slot]
            self.offmesh_slot = slot
            self.offmesh_from = self.corridor.path.waypoints[wi]
            self.offmesh_to = self.corridor.path.waypoints[min(n_wp - 1, wi + 1)]
            self.offmesh_kind = self.corridor.path.link_kinds[slot] if slot < len(self.corridor.path.link_kinds) else OFFMESH_JUMP
            self.offmesh_link_id = self.corridor.path.used_off_mesh_links[slot]
            self.offmesh_duration = max(Float32(0.2), (self.offmesh_to - self.offmesh_from).length() / max(Float32(0.1), self.offmesh_speed))
            self.offmesh_t = 0.0
            self.state = NAV_FOLLOW_OFFMESH_WAIT
            self._emit(NAV_EVT_OFFMESH_REACHED)
            return Vec3()

        # 5. Arrival
        var corner = self.corridor.corner()
        var dx = corner.x - self.position.x
        var dz = corner.z - self.position.z
        var d2 = dx * dx + dz * dz
        var at_final = self.corridor.at_last()
        if at_final and d2 <= self.arrival_radius * self.arrival_radius:
            self.state = NAV_FOLLOW_ARRIVED
            self._emit(NAV_EVT_ARRIVED)
            return Vec3()

        # 6. Stuck detection (no progress toward the current corner over `stuck_window`)
        self.stuck_timer += safe_dt
        if self.stuck_timer >= self.stuck_window:
            var dcur = sqrt(d2)
            var progressed = self.corridor.index != self.stuck_ref_index or (self.stuck_ref_dist - dcur) >= self.stuck_min_progress * self.stuck_window
            self.stuck_timer = 0.0
            self.stuck_ref_index = self.corridor.index
            self.stuck_ref_dist = dcur
            if not progressed:
                self.stuck_count += 1
                self._emit(NAV_EVT_STUCK)
                if self.stuck_count > self.max_stuck_repaths:
                    self.state = NAV_FOLLOW_FAILED
                    self._emit(NAV_EVT_PATH_FAILED)
                    return Vec3()
                self.repath(nav, NAV_REPATH_STUCK)
                if self.state != NAV_FOLLOW_MOVING:
                    return Vec3()
            else:
                self.stuck_count = 0

        var d = sqrt(max(Float32(1.0e-12), d2))
        var spd = min(self.max_speed, d * Float32(3.0)) if at_final else self.max_speed
        self.desired_velocity = Vec3((dx / d) * spd, Float32(0.0), (dz / d) * spd)
        return self.desired_velocity

    def step_kinematic(mut self, nav: NavMesh, mut pos: Vec3, dt: Float32):
        """Convenience integrator for tests / simple agents: updates and moves `pos` along the path,
        snapping to the navmesh surface (or the off-mesh arc while traversing a link)."""
        if self.offmesh_landed:
            self.offmesh_landed = False
            pos = self.offmesh_to
        var v = self.update(nav, pos, dt)
        if self.state == NAV_FOLLOW_OFFMESH_TRAVERSE:
            pos = self.offmesh_position()
            return
        if self.offmesh_landed:
            self.offmesh_landed = False
            pos = self.offmesh_to
        var np = pos + v * dt
        var near = nav.find_nearest_poly(np, Float32(2.0))
        if near[0] >= 0:
            np.y = near[1].y
        pos = np


struct NavMeshSet(Copyable, Movable):
    """One `NavMesh` per agent radius (Recast erodes walkable area by the agent radius at bake time).
    `mesh_index_for_radius(r)` picks the smallest baked radius `>= r` (or the largest available)."""
    var radii: List[Float32]
    var meshes: List[NavMesh]

    def __init__(out self):
        self.radii = List[Float32]()
        self.meshes = List[NavMesh]()

    @staticmethod
    def build_from_physics(world: PhysicsWorld, base_cfg: NavMeshConfig, radii: List[Float32]) -> NavMeshSet:
        var out = NavMeshSet()
        # Insertion-sort radii ascending
        var rs = radii.copy()
        for i in range(1, len(rs)):
            var key = rs[i]
            var j = i - 1
            while j >= 0 and rs[j] > key:
                rs[j + 1] = rs[j]
                j -= 1
            rs[j + 1] = key
        for i in range(len(rs)):
            var cfg = base_cfg
            cfg.walkable_radius = max(Float32(0.0), rs[i])
            out.radii.append(cfg.walkable_radius)
            out.meshes.append(NavMesh.build_from_physics(world, cfg))
        return out^

    def count(self) -> Int:
        return len(self.meshes)

    def mesh_index_for_radius(self, radius: Float32) -> Int:
        if len(self.radii) == 0:
            return -1
        for i in range(len(self.radii)):
            if self.radii[i] >= radius - Float32(1.0e-4):
                return i
        return len(self.radii) - 1

    def find_path(self, radius: Float32, start: Vec3, goal: Vec3, filter: NavQueryFilter = NavQueryFilter(), max_search_dist: Float32 = 10.0) -> NavPath:
        var mi = self.mesh_index_for_radius(radius)
        if mi < 0:
            return NavPath()
        return self.meshes[mi].find_path_filtered(start, goal, filter, max_search_dist)

    def add_obstacle(mut self, obs: NavObstacle):
        for i in range(len(self.meshes)):
            _ = self.meshes[i].add_obstacle(obs)

    def remove_obstacle(mut self, obs_id: Int):
        for i in range(len(self.meshes)):
            self.meshes[i].remove_obstacle(obs_id)

    def mark_area(mut self, vol: NavConvexVolume):
        for i in range(len(self.meshes)):
            _ = self.meshes[i].mark_area(vol.copy())
