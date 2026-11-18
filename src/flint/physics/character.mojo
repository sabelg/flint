"""Kinematic Character Controller (`KCC`, `flint.physics.character`, v5).

Implements `CharacterControllerConfig` & `CharacterController` (`move_and_slide`):
1. Moving & rotating kinematic/dynamic platform frame attachment (inheriting both
   orbital world displacement `b.pos + dq * r` and turntable heading `yaw`).
2. Multi-iteration discrete capsule overlap depenetration recovery.
3. Speed-preserving walkable slope-tangent projection (`n . up >= cos(max_slope_deg)`).
4. 3-sweep (`up -> forward -> down`) `step_height` stair auto-climbing.
5. Steep-cliff horizontal deflection clamp (`n_tilde = normalize(n - (n . up) up)`),
   preventing upward wall-surfing on `> max_slope_deg` cliffs while allowing gravity
   down-slope sliding.
6. Multi-plane active-set solver: 2-plane `n0 x n1` crease line projection and
   3-plane acute-corner lock (zero jitter in acute V-corners).
7. Downward ground/ledge snapping and mass-clamped dynamic `RigidBody` impulse pushing.
"""
from std.collections import List
from std.math import sqrt, abs, min, max
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_cos, det_atan2
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.physics.filter import CollisionFilter, CAT_DEBRIS, CAT_ALL
from flint.physics.narrowphase import (
    ContactManifold,
    collide_sphere_capsule,
    collide_capsule_capsule,
    collide_capsule_box,
    collide_convex_convex,
)
from flint.physics.world import (
    PhysicsWorld,
    SceneCastHit,
    BODY_SHAPE_SPHERE,
    BODY_SHAPE_CAPSULE,
    BODY_SHAPE_BOX,
    BODY_SHAPE_MESH_BVH8,
    BODY_SHAPE_HEIGHTFIELD,
    BODY_MOTION_DYNAMIC,
    _make_capsule_hull,
)

comptime KCC_DEG_TO_RAD: Float32 = 0.01745329252
comptime KCC_BLOB_TAG: UInt32 = fourcc("KCC5")
comptime KCC_BLOB_VERSION: Int = 1


@always_inline
def _quat_twist_about_axis(q: Quat, axis: Vec3) -> Float32:
    var qn = q.normalized()
    var proj = qn.x * axis.x + qn.y * axis.y + qn.z * axis.z
    var w = qn.w
    if w < Float32(0.0):
        proj = -proj
        w = -w
    return Float32(2.0) * det_atan2(proj, max(Float32(1.0e-7), w))


@always_inline
def _project_walkable_speed_preserving(horiz_move: Vec3, ground_n: Vec3, up: Vec3) -> Vec3:
    """Projects horizontal displacement `horiz_move` onto walkable plane `ground_n` along `up`
    and normalizes to `|horiz_move|` so uphill/downhill ramps preserve full movement speed."""
    var h_len = horiz_move.length()
    if h_len <= Float32(1.0e-7):
        return Vec3()
    var n_up = ground_n.dot(up)
    if n_up <= Float32(1.0e-4):
        return horiz_move
    var slope_vec = horiz_move - up * (horiz_move.dot(ground_n) / n_up)
    var s_len = slope_vec.length()
    if s_len <= Float32(1.0e-7):
        return Vec3()
    return slope_vec * (h_len / s_len)


@fieldwise_init
struct CharacterControllerConfig(Copyable, ImplicitlyCopyable, Movable):
    var radius: Float32
    var half_height: Float32
    var skin_width: Float32
    var max_slope_deg: Float32
    var step_height: Float32
    var snap_down_dist: Float32
    var max_slides: Int
    var depenetration_iters: Int
    var mass: Float32
    var push_force_scale: Float32
    var max_push_mass_ratio: Float32
    var gravity: Float32
    var up: Vec3
    var filter: CollisionFilter
    var ignore_body_id: Int

    def __init__(out self):
        self.radius = 0.35
        self.half_height = 0.55
        self.skin_width = 0.02
        self.max_slope_deg = 50.0
        self.step_height = 0.35
        self.snap_down_dist = 0.30
        self.max_slides = 4
        self.depenetration_iters = 4
        self.mass = 80.0
        self.push_force_scale = 1.0
        self.max_push_mass_ratio = 4.0
        self.gravity = 9.81
        self.up = Vec3(0.0, 1.0, 0.0)
        self.filter = CollisionFilter(CAT_DEBRIS, CAT_ALL, 0)
        self.ignore_body_id = -1


struct CharacterController(Copyable, Movable):
    var cfg: CharacterControllerConfig
    var pos: Vec3
    var yaw: Float32
    var velocity: Vec3
    var vertical_vel: Float32
    var grounded: Bool
    var on_steep_slope: Bool
    var stepped_up: Bool
    var hit_wall: Bool
    var hit_ceiling: Bool
    var ground_normal: Vec3
    var ground_body_id: Int
    var ground_prev_pos: Vec3
    var ground_prev_rot: Quat
    var has_ground_frame: Bool
    var platform_velocity: Vec3
    var pushed_body_ids: List[Int]

    def __init__(
        out self,
        pos: Vec3 = Vec3(0.0, 1.0, 0.0),
        cfg: CharacterControllerConfig = CharacterControllerConfig(),
    ):
        self.cfg = cfg
        self.pos = pos
        self.yaw = 0.0
        self.velocity = Vec3()
        self.vertical_vel = 0.0
        self.grounded = False
        self.on_steep_slope = False
        self.stepped_up = False
        self.hit_wall = False
        self.hit_ceiling = False
        self.ground_normal = cfg.up.normalized()
        self.ground_body_id = -1
        self.ground_prev_pos = Vec3()
        self.ground_prev_rot = Quat()
        self.has_ground_frame = False
        self.platform_velocity = Vec3()
        self.pushed_body_ids = List[Int]()

    def foot_position(self) -> Vec3:
        var up = self.cfg.up.normalized()
        return self.pos - up * (self.cfg.half_height + self.cfg.radius)

    def _capsule_rot(self) -> Quat:
        return Quat()

    def _sweep_capsule(self, world: PhysicsWorld, start: Vec3, delta: Vec3) -> SceneCastHit:
        return world.shapecast_capsule(
            start,
            self.cfg.radius,
            self.cfg.half_height,
            self._capsule_rot(),
            delta,
            self.cfg.filter,
            self.cfg.ignore_body_id,
        )

    def _collide_capsule_vs_slot(self, world: PhysicsWorld, center: Vec3, r_eff: Float32, slot: Int) -> ContactManifold:
        """Returns manifold with outward normal pointing `Obstacle -> KCC` (`B -> KCC`)."""
        var up = self.cfg.up.normalized()
        var cap_a = center - up * self.cfg.half_height
        var cap_b = center + up * self.cfg.half_height
        ref b = world.bodies[slot]

        if b.shape == BODY_SHAPE_SPHERE:
            var m = collide_sphere_capsule(b.pos, b.radius, cap_a, cap_b, r_eff, slot, -1)
            return m^
        if b.shape == BODY_SHAPE_CAPSULE:
            var eb = b.capsule_endpoints()
            var m = collide_capsule_capsule(eb[0], eb[1], b.radius, cap_a, cap_b, r_eff, slot, -1)
            return m^
        if b.shape == BODY_SHAPE_BOX:
            var m = collide_capsule_box(cap_a, cap_b, r_eff, b.pos, b.rot, b.half_extents, -1, slot)
            if m.has_collision:
                m.normal = -m.normal
                if m.count >= 1: m.p0.normal = -m.p0.normal
                if m.count >= 2: m.p1.normal = -m.p1.normal
                if m.count >= 3: m.p2.normal = -m.p2.normal
                if m.count >= 4: m.p3.normal = -m.p3.normal
            return m^
        if b.shape == BODY_SHAPE_MESH_BVH8 and b.shape_index >= 0 and b.shape_index < len(world.meshes):
            var inv_r = b.rot.conjugate()
            var la = inv_r.rotate(cap_a - b.pos)
            var lb = inv_r.rotate(cap_b - b.pos)
            var lm = world.meshes[b.shape_index].collide_capsule(la, lb, r_eff)
            var wm = ContactManifold()
            for pi in range(lm.count):
                var cp = lm.get_point(pi)
                wm.add_point(b.pos + b.rot.rotate(cp.point), b.rot.rotate(cp.normal).normalized(), cp.depth, cp.id)
            return wm^
        if b.shape == BODY_SHAPE_HEIGHTFIELD and b.shape_index >= 0 and b.shape_index < len(world.heightfields):
            var inv_r = b.rot.conjugate()
            var la = inv_r.rotate(cap_a - b.pos)
            var lb = inv_r.rotate(cap_b - b.pos)
            var lm = world.heightfields[b.shape_index].collide_capsule(la, lb, r_eff)
            var wm = ContactManifold()
            for pi in range(lm.count):
                var cp = lm.get_point(pi)
                wm.add_point(b.pos + b.rot.rotate(cp.point), b.rot.rotate(cp.normal).normalized(), cp.depth, cp.id)
            return wm^

        var kcc_h = _make_capsule_hull(r_eff, self.cfg.half_height)
        var bh = world._body_as_hull(slot)
        return collide_convex_convex(bh, b.pos, b.rot, kcc_h, center, Quat(), slot, -1)

    def _depenetrate(mut self, world: PhysicsWorld):
        var r_eff = self.cfg.radius + self.cfg.skin_width * Float32(0.05)
        var ext = Vec3(r_eff + Float32(0.05), self.cfg.half_height + r_eff + Float32(0.05), r_eff + Float32(0.05))
        for _ in range(self.cfg.depenetration_iters):
            var slots = world.overlap_aabb(self.pos - ext, self.pos + ext, self.cfg.filter, self.cfg.ignore_body_id)
            var max_push = Float32(0.0)
            for i in range(len(slots)):
                var slot = slots[i]
                ref b = world.bodies[slot]
                if not b.active or b.motion_mode == BODY_MOTION_DYNAMIC:
                    continue
                var m = self._collide_capsule_vs_slot(world, self.pos, r_eff, slot)
                if m.has_collision and m.count > 0:
                    var cp = m.p0
                    if cp.depth > Float32(1.0e-4):
                        var n = cp.normal.normalized()
                        self.pos = self.pos + n * cp.depth
                        if cp.depth > max_push:
                            max_push = cp.depth
            if max_push <= Float32(1.0e-4):
                break

    def _push_dynamic_body(mut self, mut world: PhysicsWorld, hit: SceneCastHit, desired_vel: Vec3, up: Vec3):
        var slot = world.find_slot_by_id(hit.body_id)
        if slot < 0 or slot >= len(world.bodies):
            return
        if world.bodies[slot].motion_mode != BODY_MOTION_DYNAMIC or world.bodies[slot].is_static or world.bodies[slot].inv_mass <= Float32(0.0):
            return

        # Horizontalize push direction so character pushes crates sideways without driving them through floors
        var push_dir = -hit.normal
        var vert_comp = push_dir.dot(up)
        if abs(vert_comp) < Float32(0.85):
            push_dir = (push_dir - up * vert_comp).normalized()
        if push_dir.length_squared() < Float32(1.0e-6):
            return

        var body_v = world.bodies[slot].linear_vel
        var rel_approach = (desired_vel - body_v).dot(push_dir)
        if rel_approach <= Float32(0.01):
            return

        var b_mass = max(Float32(0.05), world.bodies[slot].mass)
        var eff_char_mass = min(self.cfg.mass * self.cfg.push_force_scale, b_mass * self.cfg.max_push_mass_ratio)
        var impulse_mag = (eff_char_mass * b_mass / (eff_char_mass + b_mass)) * rel_approach
        var contact_pt = Vec3(hit.point.x, world.bodies[slot].pos.y, hit.point.z)
        world.apply_impulse_at(slot, push_dir * impulse_mag, contact_pt)

        var seen = False
        for k in range(len(self.pushed_body_ids)):
            if self.pushed_body_ids[k] == hit.body_id:
                seen = True
                break
        if not seen:
            self.pushed_body_ids.append(hit.body_id)

    def _apply_platform_attachment(mut self, world: PhysicsWorld, dt: Float32, up: Vec3):
        self.platform_velocity = Vec3()
        if not self.grounded or self.ground_body_id < 0 or dt <= Float32(1.0e-7):
            return
        var slot = world.find_slot_by_id(self.ground_body_id)
        if slot < 0 or not world.bodies[slot].active:
            self.ground_body_id = -1
            self.has_ground_frame = False
            return

        ref b = world.bodies[slot]
        var plat_move = Vec3()
        var dyaw = Float32(0.0)

        if self.has_ground_frame:
            var dp_pose = b.pos - self.ground_prev_pos
            var dq_pose = (b.rot * self.ground_prev_rot.conjugate()).normalized()
            var pose_moved = dp_pose.length_squared() > Float32(1.0e-10) or (Float32(1.0) - abs(dq_pose.w)) > Float32(1.0e-7)
            if pose_moved:
                var r_prev = self.pos - self.ground_prev_pos
                var attached_pos = b.pos + dq_pose.rotate(r_prev)
                plat_move = attached_pos - self.pos
                dyaw = _quat_twist_about_axis(dq_pose, up)
            elif b.linear_vel.length_squared() > Float32(1.0e-10) or b.angular_vel.length_squared() > Float32(1.0e-10):
                var ang_step = b.angular_vel.dot(up) * dt
                var dq_vel = Quat.from_axis_angle(up, ang_step)
                var r_cur = self.pos - b.pos
                var attached_pos = b.pos + b.linear_vel * dt + dq_vel.rotate(r_cur)
                plat_move = attached_pos - self.pos
                dyaw = ang_step
        elif b.linear_vel.length_squared() > Float32(1.0e-10) or b.angular_vel.length_squared() > Float32(1.0e-10):
            var ang_step = b.angular_vel.dot(up) * dt
            var dq_vel = Quat.from_axis_angle(up, ang_step)
            var r_cur = self.pos - b.pos
            var attached_pos = b.pos + b.linear_vel * dt + dq_vel.rotate(r_cur)
            plat_move = attached_pos - self.pos
            dyaw = ang_step

        self.yaw += dyaw
        var pm_len = plat_move.length()
        if pm_len > Float32(1.0e-7):
            var dir = plat_move * (Float32(1.0) / pm_len)
            var sh = self._sweep_capsule(world, self.pos, dir * (pm_len + self.cfg.skin_width))
            if sh.hit:
                var safe_d = max(Float32(0.0), sh.distance - self.cfg.skin_width)
                var actual = dir * safe_d
                self.pos = self.pos + actual
                self.platform_velocity = actual * (Float32(1.0) / dt)
            else:
                self.pos = self.pos + plat_move
                self.platform_velocity = plat_move * (Float32(1.0) / dt)

    def _try_stair_climb(
        mut self,
        world: PhysicsWorld,
        rem_horiz: Vec3,
        cos_walk: Float32,
        up: Vec3,
    ) -> Tuple[Bool, Vec3]:
        """3-sweep (`up -> forward -> down`) auto-step over vertical risers `<= cfg.step_height`."""
        var h_vec = rem_horiz - up * rem_horiz.dot(up)
        var h_len = h_vec.length()
        if h_len <= Float32(1.0e-5) or self.cfg.step_height <= Float32(1.0e-4):
            return (False, rem_horiz)
        var h_dir = h_vec * (Float32(1.0) / h_len)

        # 1. Sweep 1: Up by step_height + skin_width
        var up_target = self.cfg.step_height + self.cfg.skin_width
        var up_hit = self._sweep_capsule(world, self.pos, up * up_target)
        var up_dist = max(Float32(0.0), up_hit.distance - self.cfg.skin_width) if up_hit.hit else self.cfg.step_height
        if up_dist <= Float32(0.01):
            return (False, rem_horiz)
        var p_up = self.pos + up * up_dist

        # 2. Sweep 2: Forward by remaining horizontal travel (plus tread probe past capsule rim)
        var probe_fwd = max(h_len, self.cfg.radius * Float32(0.45))
        var fwd_hit = self._sweep_capsule(world, p_up, h_dir * (probe_fwd + self.cfg.skin_width))
        var max_fwd = max(Float32(0.0), fwd_hit.distance - self.cfg.skin_width) if fwd_hit.hit else probe_fwd
        if max_fwd <= Float32(0.015):
            return (False, rem_horiz)

        # Verify actual tread surface beyond the stair lip is genuinely walkable (rejects steep 65 deg cliffs!)
        var min_tread_reach = self.cfg.radius + self.cfg.skin_width + Float32(0.05)
        var ray_origin = p_up + h_dir * min(min_tread_reach, max_fwd + self.cfg.radius + Float32(0.03))
        var tread_ray = world.raycast(
            ray_origin,
            -up,
            up_dist + self.cfg.half_height + self.cfg.radius + self.cfg.skin_width * Float32(3.0),
            self.cfg.filter,
            self.cfg.ignore_body_id,
        )
        if not tread_ray.hit or tread_ray.normal.dot(up) < cos_walk:
            return (False, rem_horiz)
        var tread_rise = tread_ray.point.dot(up) - self.foot_position().dot(up)
        if tread_rise <= Float32(0.002) or tread_rise > self.cfg.step_height + Float32(0.02):
            return (False, rem_horiz)

        var step_fwd = min(h_len, max_fwd)
        var p_fwd = p_up + h_dir * step_fwd

        # 3. Sweep 3: Down onto stair lip/tread at `p_fwd`
        var down_max = up_dist + self.cfg.skin_width * Float32(2.0)
        var down_hit = self._sweep_capsule(world, p_fwd, -up * down_max)
        if not down_hit.hit:
            return (False, rem_horiz)

        var drop_dist = max(Float32(0.0), down_hit.distance - self.cfg.skin_width)
        var rise = up_dist - drop_dist
        if rise <= Float32(-0.005) or rise > self.cfg.step_height + Float32(0.02):
            return (False, rem_horiz)

        self.pos = p_fwd - up * drop_dist
        self.stepped_up = True
        self.grounded = True
        self.on_steep_slope = False
        self.ground_normal = tread_ray.normal
        self.ground_body_id = down_hit.body_id
        var rem_left = h_dir * max(Float32(0.0), h_len - step_fwd)
        return (True, rem_left)

    @staticmethod
    def _solve_multi_plane_velocity(
        vec_in: Vec3,
        planes: List[Vec3],
        plane_walkable: List[Bool],
        up: Vec3,
    ) -> Vec3:
        """Quake/Jolt multi-plane active-set velocity projection:
        - 1 plane: speed-preserving tangent on walkable slopes, tangent slide on walls.
        - 2 planes: exact `n0 x n1` crease projection when single-plane slide violates `n0`.
        - 3+ non-coplanar planes: 3-plane acute-corner zero-velocity lock."""
        var n_planes = len(planes)
        if n_planes == 0:
            return vec_in

        if n_planes == 1:
            var n0 = planes[0]
            if plane_walkable[0]:
                return _project_walkable_speed_preserving(vec_in, n0, up)
            var d0 = vec_in.dot(n0)
            return (vec_in - n0 * d0) if d0 < Float32(0.0) else vec_in

        # Try single-plane projection against newest plane first; check if it respects all prior planes
        var newest = planes[n_planes - 1]
        var cand: Vec3
        if plane_walkable[n_planes - 1]:
            cand = _project_walkable_speed_preserving(vec_in, newest, up)
        else:
            var dn = vec_in.dot(newest)
            cand = (vec_in - newest * dn) if dn < Float32(0.0) else vec_in

        var violates_idx = -1
        for i in range(n_planes - 1):
            if cand.dot(planes[i]) < Float32(-1.0e-5):
                violates_idx = i
                break
        if violates_idx < 0:
            return cand

        # 2-plane active crease: project strictly onto line n_a x n_b!
        var na = planes[violates_idx]
        var nb = newest
        var crease = na.cross(nb)
        var c_len2 = crease.length_squared()
        if c_len2 <= Float32(1.0e-6):
            # Opposing parallel squeeze
            return Vec3()
        var c_hat = crease * (Float32(1.0) / sqrt(c_len2))
        var crease_vec = c_hat * vec_in.dot(c_hat)

        # If 3 or more planes are present and crease_vec violates any third plane -> 3-plane corner lock!
        for k in range(n_planes):
            if k != violates_idx and k != n_planes - 1:
                if crease_vec.dot(planes[k]) < Float32(-1.0e-5):
                    return Vec3()
        return crease_vec

    def move_and_slide(
        mut self,
        mut world: PhysicsWorld,
        desired_vel: Vec3,
        dt: Float32,
        jump_speed: Float32 = 0.0,
    ) -> Vec3:
        """Executes full KCC movement tick of duration `dt`, returning new world `self.pos`."""
        self.stepped_up = False
        self.hit_wall = False
        self.hit_ceiling = False
        self.pushed_body_ids.clear()
        if dt <= Float32(1.0e-7):
            return self.pos

        var up = self.cfg.up.normalized()
        var cos_walk = det_cos(self.cfg.max_slope_deg * KCC_DEG_TO_RAD)
        var start_pos = self.pos
        var was_grounded = self.grounded

        # 1. Recover from any initial penetration
        self._depenetrate(world)

        # 2. Moving & rotating platform attachment
        self._apply_platform_attachment(world, dt, up)
        var pos_after_plat = self.pos

        # 3. Decompose desired velocity into horizontal intent & vertical state
        var v_up_in = desired_vel.dot(up)
        var v_horiz = desired_vel - up * v_up_in

        if jump_speed > Float32(0.0):
            self.vertical_vel = jump_speed
            self.grounded = False
            was_grounded = False
            self.has_ground_frame = False
        elif was_grounded and v_up_in > Float32(0.25):
            self.vertical_vel = v_up_in
            self.grounded = False
            was_grounded = False
            self.has_ground_frame = False
        elif was_grounded:
            self.vertical_vel = Float32(0.0)
        else:
            if abs(self.vertical_vel) < Float32(1.0e-6) and v_up_in < Float32(-0.01):
                self.vertical_vel = v_up_in
            self.vertical_vel -= self.cfg.gravity * dt

        # 4. Horizontal / Walkable-Slope Multi-Plane Slide + 3-Sweep Stair Auto-Climb
        var rem_orig = v_horiz * dt
        var rem = rem_orig
        var planes = List[Vec3]()
        var plane_walkable = List[Bool]()

        if was_grounded and self.ground_normal.dot(up) >= cos_walk:
            planes.append(self.ground_normal)
            plane_walkable.append(True)
            rem = _project_walkable_speed_preserving(rem_orig, self.ground_normal, up)

        for _ in range(self.cfg.max_slides):
            var r_len = rem.length()
            if r_len <= Float32(1.0e-6):
                break
            var r_dir = rem * (Float32(1.0) / r_len)
            var hit = self._sweep_capsule(world, self.pos, r_dir * (r_len + self.cfg.skin_width * Float32(4.0)))
            if not hit.hit:
                self.pos = self.pos + rem
                break

            var cos_inc = max(Float32(0.20), -r_dir.dot(hit.normal))
            var skin_pullback = self.cfg.skin_width / cos_inc
            if hit.distance - skin_pullback >= r_len:
                self.pos = self.pos + rem
                break

            var safe_d = max(Float32(0.0), hit.distance - skin_pullback)
            self.pos = self.pos + r_dir * safe_d
            var frac_left = max(Float32(0.0), Float32(1.0) - safe_d / r_len)
            var rem_left = rem * frac_left
            rem_orig = rem_orig * frac_left

            # Push dynamic RigidBodies on contact
            var hit_slot = world.find_slot_by_id(hit.body_id)
            var is_dyn_body = (
                hit_slot >= 0
                and world.bodies[hit_slot].motion_mode == BODY_MOTION_DYNAMIC
                and not world.bodies[hit_slot].is_static
            )
            if is_dyn_body:
                self._push_dynamic_body(world, hit, desired_vel, up)

            var n_dot_up = hit.normal.dot(up)

            # Attempt 3-sweep stair climb on non-walkable vertical/steep risers (static/kinematic obstacles)
            if (not is_dyn_body) and (was_grounded or self.grounded) and n_dot_up >= Float32(-0.05) and n_dot_up < cos_walk:
                var stair_res = self._try_stair_climb(world, rem_left, cos_walk, up)
                if stair_res[0]:
                    planes.clear()
                    plane_walkable.clear()
                    planes.append(self.ground_normal)
                    plane_walkable.append(True)
                    rem_orig = stair_res[1]
                    rem = _project_walkable_speed_preserving(rem_orig, self.ground_normal, up)
                    continue

            var cond_n: Vec3
            var is_walk = False
            if n_dot_up >= cos_walk:
                cond_n = hit.normal.normalized()
                is_walk = True
                self.grounded = True
                self.on_steep_slope = False
                self.ground_normal = cond_n
                self.ground_body_id = hit.body_id
            elif n_dot_up >= Float32(-0.05):
                # Steep cliff (> max_slope_deg) or vertical wall: horizontalize normal to prevent wall-surfing!
                self.hit_wall = True
                var h_n = hit.normal - up * n_dot_up
                cond_n = h_n.normalized() if h_n.length_squared() > Float32(1.0e-8) else hit.normal.normalized()
            else:
                self.hit_ceiling = True
                cond_n = hit.normal.normalized()

            var duplicate = False
            for pi in range(len(planes)):
                if planes[pi].dot(cond_n) > Float32(0.999):
                    duplicate = True
                    break
            if not duplicate:
                planes.append(cond_n)
                plane_walkable.append(is_walk)

            rem = Self._solve_multi_plane_velocity(rem_orig, planes, plane_walkable, up)
            if rem.length_squared() <= Float32(1.0e-10):
                break

        # 5. Vertical Sweep (Jumping / Falling / Steep-Cliff Down-Slide / Ledge Ground Snapping)
        if self.vertical_vel > Float32(0.0):
            var up_move = self.vertical_vel * dt
            var ceil_hit = self._sweep_capsule(world, self.pos, up * (up_move + self.cfg.skin_width))
            if ceil_hit.hit:
                var safe_u = max(Float32(0.0), ceil_hit.distance - self.cfg.skin_width)
                self.pos = self.pos + up * safe_u
                if ceil_hit.normal.dot(up) < Float32(-0.05):
                    self.hit_ceiling = True
                    self.vertical_vel = Float32(0.0)
            else:
                self.pos = self.pos + up * up_move
            self.grounded = False
        elif not self.stepped_up:
            var fall_dist = abs(self.vertical_vel) * dt
            var probe_down = self.cfg.snap_down_dist if (was_grounded or self.grounded) else max(fall_dist, Float32(0.02))
            var down_hit = self._sweep_capsule(world, self.pos, -up * (probe_down + self.cfg.skin_width))

            if down_hit.hit:
                var safe_down = max(Float32(0.0), down_hit.distance - self.cfg.skin_width)
                var surf_n = down_hit.normal
                var n_up = surf_n.dot(up)
                if n_up < cos_walk:
                    # Distinguish a flat ledge/stair lip (spherical cap contact on 90 deg edge) from a genuine steep cliff!
                    var into_ledge = -(surf_n - up * n_up)
                    if into_ledge.length_squared() > Float32(1.0e-6):
                        into_ledge = into_ledge.normalized()
                        var ledge_ray = world.raycast(
                            down_hit.point + into_ledge * Float32(0.02) + up * Float32(0.10),
                            -up,
                            Float32(0.25),
                            self.cfg.filter,
                            self.cfg.ignore_body_id,
                        )
                        if ledge_ray.hit and ledge_ray.normal.dot(up) >= cos_walk:
                            surf_n = ledge_ray.normal
                            n_up = surf_n.dot(up)

                if n_up >= cos_walk:
                    self.pos = self.pos - up * safe_down
                    self.grounded = True
                    self.on_steep_slope = False
                    self.vertical_vel = Float32(0.0)
                    self.ground_normal = surf_n
                    self.ground_body_id = down_hit.body_id
                else:
                    # Landed on a non-walkable steep cliff (e.g. 65 deg): do NOT ground; slide downhill under gravity!
                    self.pos = self.pos - up * min(safe_down, max(fall_dist, self.cfg.skin_width))
                    self.grounded = False
                    self.on_steep_slope = True
                    self.ground_body_id = -1
                    self.has_ground_frame = False
                    if was_grounded and abs(self.vertical_vel) < Float32(1.0e-5):
                        self.vertical_vel = -self.cfg.gravity * dt
                        fall_dist = abs(self.vertical_vel) * dt

                    var grav_vec = -up * fall_dist
                    var steep_planes = List[Vec3]()
                    var steep_walk = List[Bool]()
                    steep_planes.append(down_hit.normal.normalized())
                    steep_walk.append(False)
                    var slide_rem = grav_vec - steep_planes[0] * min(Float32(0.0), grav_vec.dot(steep_planes[0]))

                    for _ in range(3):
                        var s_len = slide_rem.length()
                        if s_len <= Float32(1.0e-6):
                            break
                        var s_dir = slide_rem * (Float32(1.0) / s_len)
                        var sh = self._sweep_capsule(world, self.pos, s_dir * (s_len + self.cfg.skin_width))
                        if not sh.hit:
                            self.pos = self.pos + slide_rem
                            break
                        var sd = max(Float32(0.0), sh.distance - self.cfg.skin_width)
                        self.pos = self.pos + s_dir * sd
                        var f_left = max(Float32(0.0), Float32(1.0) - sd / s_len)
                        grav_vec = grav_vec * f_left
                        if sh.normal.dot(up) >= cos_walk:
                            self.grounded = True
                            self.on_steep_slope = False
                            self.vertical_vel = Float32(0.0)
                            self.ground_normal = sh.normal
                            self.ground_body_id = sh.body_id
                            break
                        steep_planes.append(sh.normal.normalized())
                        steep_walk.append(False)
                        slide_rem = Self._solve_multi_plane_velocity(grav_vec, steep_planes, steep_walk, up)
            else:
                # Over open air or walked off sheer ledge > snap_down_dist
                if was_grounded and fall_dist <= Float32(1.0e-6):
                    self.vertical_vel = -self.cfg.gravity * dt
                    fall_dist = abs(self.vertical_vel) * dt
                self.pos = self.pos - up * fall_dist
                self.grounded = False
                self.on_steep_slope = False
                self.ground_body_id = -1
                self.has_ground_frame = False

        # 6. Snapshot ground body pose for next tick's moving/rotating platform attachment
        if self.grounded and self.ground_body_id >= 0:
            var g_slot = world.find_slot_by_id(self.ground_body_id)
            if g_slot >= 0 and world.bodies[g_slot].active:
                self.ground_prev_pos = world.bodies[g_slot].pos
                self.ground_prev_rot = world.bodies[g_slot].rot
                self.has_ground_frame = True
            else:
                self.has_ground_frame = False
        else:
            self.has_ground_frame = False

        self.velocity = (self.pos - pos_after_plat) * (Float32(1.0) / dt) + self.platform_velocity
        _ = start_pos
        return self.pos

    def serialize(self, mut w: BlobWriter):
        w.begin(KCC_BLOB_TAG, KCC_BLOB_VERSION)
        w.vec3(self.pos)
        w.f32(self.yaw)
        w.vec3(self.velocity)
        w.f32(self.vertical_vel)
        w.boolean(self.grounded)
        w.boolean(self.on_steep_slope)
        w.boolean(self.stepped_up)
        w.boolean(self.hit_wall)
        w.boolean(self.hit_ceiling)
        w.vec3(self.ground_normal)
        w.integer(self.ground_body_id)
        w.vec3(self.ground_prev_pos)
        w.quat(self.ground_prev_rot)
        w.boolean(self.has_ground_frame)
        w.vec3(self.platform_velocity)
        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        _ = r.enter(KCC_BLOB_TAG, KCC_BLOB_VERSION)
        self.pos = r.vec3()
        self.yaw = r.f32()
        self.velocity = r.vec3()
        self.vertical_vel = r.f32()
        self.grounded = r.boolean()
        self.on_steep_slope = r.boolean()
        self.stepped_up = r.boolean()
        self.hit_wall = r.boolean()
        self.hit_ceiling = r.boolean()
        self.ground_normal = r.vec3()
        self.ground_body_id = r.integer()
        self.ground_prev_pos = r.vec3()
        self.ground_prev_rot = r.quat()
        self.has_ground_frame = r.boolean()
        self.platform_velocity = r.vec3()
        r.leave()
