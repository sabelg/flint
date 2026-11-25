"""Motion-matching / learned-motion-matching character controller (`flint.anim.mm_controller`).

The update loop of Holden's reference controller (https://github.com/orangeduck/Motion-Matching,
`controller.cpp`, MIT) on Flint types:

1. Input (`MmInput`, or `mm_input_from_map` reading an `InputMap`): left stick -> desired
   velocity (forward / side / back speeds, walk-run gait spring), facing from the stick or,
   when strafing, from the right stick / camera.
2. The *simulation object* follows the desired velocity and facing with critically damped
   springs; its future positions and facings 20/40/60 frames ahead form the trajectory part
   of the query. With `update_kcc` the simulation object is moved by a
   `flint.physics.character.CharacterController`, so walls and steps act on it.
3. Every `search_time` seconds (or when the input changes sharply, or a clip ends) the query
   (current foot/hip features + desired trajectory) is matched against the database
   (`MmDatabase.search`, AABB tree) or projected with the learned projector. A better frame
   starts an inertialized transition (`MmInertializer`).
4. Playback advances the frame index (database mode, fixed 60 Hz) or runs the stepper and the
   decompressor (learned mode).
5. The shown root is pulled toward the simulation object (velocity-limited damping, then a
   distance / angle clamp), as in the reference.

The output is an `MmPose` (`pose()`); `write_model_pose` / `placement` turn it into a
`flint.anim.pose.Pose` for `mm_stick_biped_model(db)` plus a `Placement`, which is exactly what
`PhysicalAnimator.update` takes, so physics-driven hit reactions and falls compose on top.

`mm_stick_biped_skeleton` is the 21-joint humanoid (identity rest rotations, mirror-symmetric
offsets) that `tools/learned_mm/build_database.mojo` retargets 100STYLE onto.
"""
from std.math import sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.detmath import det_atan2
from flint.assets.model import NativeModel, NativeNode
from flint.anim.pose import Pose
from flint.anim.procedural import Placement
from flint.anim.retarget import RetargetSkeleton
from flint.input.input_map import InputMap
from flint.physics.character import CharacterController
from flint.physics.world import PhysicsWorld
from flint.anim.mm_springs import (
    MmPose, MmInertializer, mm_quat_abs, mm_quat_mul, mm_quat_mul_inv, mm_quat_inv_mul_vec, mm_quat_mul_vec,
    mm_quat_to_scaled_angle_axis, mm_quat_from_scaled_angle_axis, mm_yaw_quat, mm_quat_yaw,
    mm_simple_spring_scalar, mm_simple_spring_quat, mm_spring_character_update, mm_damp_adjustment_vec,
    mm_damp_adjustment_quat,
)
from flint.anim.mm_database import MmDatabase, MM_BIG
from flint.anim.lmm import LmmNetworks

comptime _RAD2DEG: Float32 = 57.29577951308232


# ----------------------------------------------------------------------------- stick biped

def mm_stick_biped_skeleton() -> RetargetSkeleton:
    """21 humanoid joints with retarget role names; +Y up, facing +Z, left = +X."""
    var s = RetargetSkeleton()
    var hips = s.add("Hips", -1, Vec3(0, 0.95, 0))
    var spine = s.add("Spine", hips, Vec3(0, 0.13, 0))
    var chest = s.add("Chest", spine, Vec3(0, 0.20, 0))
    var neck = s.add("Neck", chest, Vec3(0, 0.22, 0))
    _ = s.add("Head", neck, Vec3(0, 0.10, 0))
    for side in range(2):
        var sx: Float32 = 1 if side == 0 else -1
        var pre = String("Left") if side == 0 else String("Right")
        var sh = s.add(pre + "Shoulder", chest, Vec3(0.04 * sx, 0.17, 0))
        var ua = s.add(pre + "UpperArm", sh, Vec3(0.15 * sx, 0, 0))
        var la = s.add(pre + "LowerArm", ua, Vec3(0.29 * sx, 0, 0))
        _ = s.add(pre + "Hand", la, Vec3(0.25 * sx, 0, 0))
    for side in range(2):
        var sx: Float32 = 1 if side == 0 else -1
        var pre = String("Left") if side == 0 else String("Right")
        var ul = s.add(pre + "UpperLeg", hips, Vec3(0.10 * sx, 0, 0))
        var ll = s.add(pre + "LowerLeg", ul, Vec3(0, -0.43, 0))
        var ft = s.add(pre + "Foot", ll, Vec3(0, -0.43, 0))
        _ = s.add(pre + "Toes", ft, Vec3(0, -0.07, 0.14))
    return s^


def mm_stick_biped_model(db: MmDatabase) -> NativeModel:
    """A `NativeModel` whose node i is database bone i (node 0 = simulation bone), with the
    first frame as the rest pose."""
    var m = NativeModel()
    for i in range(db.nbones):
        var nd = NativeNode()
        nd.name = db.names[i]
        var t = Vec3() if i == 0 else db.pos[i]
        var r = Quat() if i == 0 else db.rot[i]
        nd.translation = t
        nd.base_translation = t
        nd.rotation = r
        nd.base_rotation = r
        nd.scale = Vec3(1, 1, 1)
        nd.base_scale = Vec3(1, 1, 1)
        m.nodes.append(nd^)
        var p = db.parents[i]
        if p >= 0:
            m.nodes[p].children.append(i)
        else:
            m.roots.append(i)
    return m^


# ----------------------------------------------------------------------------- input

@fieldwise_init
struct MmInput(Copyable, ImplicitlyCopyable, Movable):
    var move: Vec3            # left stick in camera space: x = left, z = forward, |move| <= 1
    var look: Vec3            # right stick, same frame (strafe facing)
    var walk: Bool            # held = walk gait, else run
    var strafe: Bool
    var camera_azimuth: Float32

    @staticmethod
    def idle() -> Self:
        return Self(Vec3(), Vec3(), False, False, 0)


def mm_input_from_map(imap: InputMap, camera_azimuth: Float32 = 0, move: String = "Move", look: String = "Look",
                      walk: String = "Walk", strafe: String = "Strafe") -> MmInput:
    """Reads an `InputMap`: `Move` / `Look` 2D axes (x right, y forward), `Walk` and `Strafe`
    buttons. Unregistered actions read as zero / released."""
    var m = imap.axis_2d(move)
    var l = imap.axis_2d(look)
    var mv = Vec3(-m[0], 0, m[1])
    var ml = mv.length()
    if ml > 1:
        mv = mv * (Float32(1) / ml)
    return MmInput(mv, Vec3(-l[0], 0, l[1]), imap.is_down(walk), imap.is_down(strafe), camera_azimuth)


# ----------------------------------------------------------------------------- config

struct MmControllerConfig(Copyable, ImplicitlyCopyable, Movable):
    var run_fwrd_speed: Float32
    var run_side_speed: Float32
    var run_back_speed: Float32
    var walk_fwrd_speed: Float32
    var walk_side_speed: Float32
    var walk_back_speed: Float32
    var velocity_halflife: Float32
    var rotation_halflife: Float32
    var gait_halflife: Float32
    var search_time: Float32
    var inertialize_halflife: Float32
    var transition_cost: Float32
    var ignore_range_end: Int
    var ignore_surrounding: Int
    var velocity_change_threshold: Float32
    var rotation_change_threshold: Float32
    var adjustment: Bool
    var adjustment_by_velocity: Bool
    var adjust_position_halflife: Float32
    var adjust_rotation_halflife: Float32
    var adjust_position_max_ratio: Float32
    var adjust_rotation_max_ratio: Float32
    var clamping: Bool
    var clamp_max_distance: Float32
    var clamp_max_angle: Float32

    def __init__(out self):
        self.run_fwrd_speed = 4.0
        self.run_side_speed = 3.0
        self.run_back_speed = 2.5
        self.walk_fwrd_speed = 1.75
        self.walk_side_speed = 1.5
        self.walk_back_speed = 1.25
        self.velocity_halflife = 0.27
        self.rotation_halflife = 0.27
        self.gait_halflife = 0.1
        self.search_time = 0.1
        self.inertialize_halflife = 0.1
        self.transition_cost = 0.0
        self.ignore_range_end = 20
        self.ignore_surrounding = 20
        self.velocity_change_threshold = 50.0
        self.rotation_change_threshold = 50.0
        self.adjustment = True
        self.adjustment_by_velocity = True
        self.adjust_position_halflife = 0.1
        self.adjust_rotation_halflife = 0.2
        self.adjust_position_max_ratio = 0.5
        self.adjust_rotation_max_ratio = 0.5
        self.clamping = True
        self.clamp_max_distance = 0.15
        self.clamp_max_angle = 0.5 * 3.14159265

    @staticmethod
    def for_100style() -> Self:
        """Speeds that the 100STYLE Neutral clips cover (p90 of the simulation-bone speed:
        walk ~0.9 m/s, run ~1.5 m/s); the defaults above are the reference's LaFAN-era ones."""
        var c = Self()
        c.run_fwrd_speed = 1.5
        c.run_side_speed = 1.3
        c.run_back_speed = 1.1
        c.walk_fwrd_speed = 0.85
        c.walk_side_speed = 0.8
        c.walk_back_speed = 0.7
        return c


def _adjust_position_by_velocity(p: Vec3, v: Vec3, sim: Vec3, max_ratio: Float32, halflife: Float32, dt: Float32) -> Vec3:
    var adj = mm_damp_adjustment_vec(sim - p, halflife, dt)
    var max_len = max_ratio * v.length() * dt
    var l = adj.length()
    if l > max_len:
        adj = adj * (max_len / l) if l > Float32(1.0e-12) else Vec3()
    return adj + p


def _adjust_rotation_by_velocity(r: Quat, w: Vec3, sim: Quat, max_ratio: Float32, halflife: Float32, dt: Float32) -> Quat:
    var adj = mm_damp_adjustment_quat(mm_quat_abs(mm_quat_mul_inv(sim, r).normalized()), halflife, dt)
    var max_len = max_ratio * w.length() * dt
    var sa = mm_quat_to_scaled_angle_axis(adj)
    var l = sa.length()
    if l > max_len:
        adj = mm_quat_from_scaled_angle_axis(sa * (max_len / l)) if l > Float32(1.0e-12) else Quat()
    return mm_quat_mul(adj, r).normalized()


def _clamp_position(p: Vec3, sim: Vec3, max_distance: Float32) -> Vec3:
    var d = p - sim
    var l = d.length()
    if l > max_distance:
        return d * (max_distance / l) + sim
    return p


def _clamp_rotation(r: Quat, sim: Quat, max_angle: Float32) -> Quat:
    var diff = mm_quat_abs(mm_quat_mul_inv(r, sim).normalized())
    var sa = mm_quat_to_scaled_angle_axis(diff)
    var a = sa.length()
    if a > max_angle:
        return mm_quat_mul(mm_quat_from_scaled_angle_axis(sa * (max_angle / a)), sim).normalized()
    return r


# ----------------------------------------------------------------------------- controller

struct MmController(Movable):
    var db: MmDatabase
    var cfg: MmControllerConfig
    var nets: List[LmmNetworks]   # empty, or one set of learned networks
    var lmm_enabled: Bool
    var frame_index: Int
    var curr: MmPose              # playing animation pose (animation space)
    var trns: MmPose              # transition destination
    var out: MmPose               # shown pose (root in world space)
    var curr_contacts: List[Bool]
    var inert: MmInertializer
    var search_timer: Float32
    var force_search_timer: Float32
    var desired_velocity: Vec3
    var desired_rotation: Quat
    var dvel_change_curr: Vec3
    var dvel_change_prev: Vec3
    var drot_change_curr: Vec3
    var drot_change_prev: Vec3
    var desired_gait: Float32
    var desired_gait_velocity: Float32
    var sim_position: Vec3
    var sim_velocity: Vec3
    var sim_acceleration: Vec3
    var sim_rotation: Quat
    var sim_angular_velocity: Vec3
    var traj_desired_vel: List[Vec3]
    var traj_desired_rot: List[Quat]
    var traj_pos: List[Vec3]
    var traj_vel: List[Vec3]
    var traj_acc: List[Vec3]
    var traj_rot: List[Quat]
    var traj_ang: List[Vec3]
    var features_curr: List[Float32]
    var latent_curr: List[Float32]
    var features_proj: List[Float32]
    var latent_proj: List[Float32]
    var query: List[Float32]
    var searches: Int
    var transitions: Int
    var last_cost: Float32
    var just_transitioned: Bool

    def __init__(out self, var db: MmDatabase, cfg: MmControllerConfig = MmControllerConfig()):
        var nb = db.nbones
        self.cfg = cfg
        self.nets = List[LmmNetworks]()
        self.lmm_enabled = False
        self.frame_index = db.range_starts[0] if len(db.range_starts) > 0 else 0
        self.curr = MmPose(nb)
        self.trns = MmPose(nb)
        self.out = MmPose(nb)
        self.curr_contacts = List[Bool](length=2, fill=False)
        self.inert = MmInertializer(nb)
        self.search_timer = cfg.search_time
        self.force_search_timer = cfg.search_time
        self.desired_velocity = Vec3()
        self.desired_rotation = Quat()
        self.dvel_change_curr = Vec3()
        self.dvel_change_prev = Vec3()
        self.drot_change_curr = Vec3()
        self.drot_change_prev = Vec3()
        self.desired_gait = 0
        self.desired_gait_velocity = 0
        self.sim_position = Vec3()
        self.sim_velocity = Vec3()
        self.sim_acceleration = Vec3()
        self.sim_rotation = Quat()
        self.sim_angular_velocity = Vec3()
        self.traj_desired_vel = List[Vec3](length=4, fill=Vec3())
        self.traj_desired_rot = List[Quat](length=4, fill=Quat())
        self.traj_pos = List[Vec3](length=4, fill=Vec3())
        self.traj_vel = List[Vec3](length=4, fill=Vec3())
        self.traj_acc = List[Vec3](length=4, fill=Vec3())
        self.traj_rot = List[Quat](length=4, fill=Quat())
        self.traj_ang = List[Vec3](length=4, fill=Vec3())
        self.features_curr = List[Float32](length=db.nfeatures, fill=0)
        self.latent_curr = List[Float32]()
        self.features_proj = List[Float32](length=db.nfeatures, fill=0)
        self.latent_proj = List[Float32]()
        self.query = List[Float32](length=db.nfeatures, fill=0)
        self.searches = 0
        self.transitions = 0
        self.last_cost = 0
        self.just_transitioned = False
        self.db = db^
        self.reset(Vec3(), Quat())

    def attach_networks(mut self, var nets: LmmNetworks) raises:
        if nets.nfeatures != self.db.nfeatures:
            raise Error("lmm: networks expect " + String(nets.nfeatures) + " features, database has " + String(self.db.nfeatures))
        self.latent_curr = List[Float32](length=nets.nlatent, fill=0)
        self.latent_proj = List[Float32](length=nets.nlatent, fill=0)
        self.nets.clear()
        self.nets.append(nets^)

    def has_networks(self) -> Bool:
        return len(self.nets) > 0

    def reset(mut self, position: Vec3, rotation: Quat, frame: Int = -1):
        """Starts playing `frame` (default: the first frame) at `position` / `rotation`."""
        self.frame_index = frame if frame >= 0 else (self.db.range_starts[0] if len(self.db.range_starts) > 0 else 0)
        self.db.pose_at(self.frame_index, self.curr)
        self.db.pose_at(self.frame_index, self.out)
        self.curr_contacts[0] = self.db.contact(self.frame_index, 0)
        self.curr_contacts[1] = self.db.contact(self.frame_index, 1)
        for j in range(self.db.nfeatures):
            self.features_curr[j] = self.db.feature(self.frame_index, j)
        for j in range(len(self.latent_curr)):
            self.latent_curr[j] = 0
        self.inert = MmInertializer(self.db.nbones)
        self.inert.reset(self.curr.pos[0], self.curr.rot[0])
        # place the animation's root frame at the requested transform
        self.inert.src_pos = self.curr.pos[0]
        self.inert.src_rot = self.curr.rot[0]
        self.inert.dst_pos = position
        self.inert.dst_rot = rotation
        self.inert.update(self.out, self.curr, self.cfg.inertialize_halflife, 0)
        self.sim_position = position
        self.sim_rotation = rotation
        self.sim_velocity = Vec3()
        self.sim_acceleration = Vec3()
        self.sim_angular_velocity = Vec3()
        self.desired_rotation = rotation
        self.desired_velocity = Vec3()
        self.search_timer = self.cfg.search_time
        self.force_search_timer = self.cfg.search_time

    # ------------------------------------------------------------------ desired motion

    def _desired_velocity(self, stick: Vec3, azimuth: Float32, facing: Quat, fwrd: Float32, side: Float32, back: Float32) -> Vec3:
        var g = mm_yaw_quat(azimuth).rotate(stick)
        var l = mm_quat_inv_mul_vec(facing, g)
        var lv = Vec3(side * l.x, 0, (fwrd if l.z > 0 else back) * l.z)
        return facing.rotate(lv)

    def _desired_rotation(self, current: Quat, inp: MmInput, desired_velocity: Vec3) -> Quat:
        if inp.strafe:
            var d = mm_yaw_quat(inp.camera_azimuth).rotate(Vec3(0, 0, 1))
            if inp.look.length() > Float32(0.01):
                d = mm_yaw_quat(inp.camera_azimuth).rotate(inp.look.normalized())
            return mm_yaw_quat(det_atan2(d.x, d.z))
        if inp.move.length() > Float32(0.01) and desired_velocity.length() > Float32(1.0e-6):
            var d = desired_velocity.normalized()
            return mm_yaw_quat(det_atan2(d.x, d.z))
        return current

    def _speeds(self) -> Tuple[Float32, Float32, Float32]:
        var g = self.desired_gait
        var c = self.cfg
        return (
            c.run_fwrd_speed + (c.walk_fwrd_speed - c.run_fwrd_speed) * g,
            c.run_side_speed + (c.walk_side_speed - c.run_side_speed) * g,
            c.run_back_speed + (c.walk_back_speed - c.run_back_speed) * g,
        )

    def _predict(mut self, inp: MmInput):
        var step = Float32(20.0) / Float32(60.0)
        var sp = self._speeds()
        self.traj_desired_rot[0] = self.desired_rotation
        for i in range(1, 4):
            self.traj_desired_rot[i] = self._desired_rotation(self.traj_desired_rot[i - 1], inp, self.traj_desired_vel[i])
        for i in range(4):
            self.traj_rot[i] = self.sim_rotation
            self.traj_ang[i] = self.sim_angular_velocity
        for i in range(1, 4):
            mm_simple_spring_quat(self.traj_rot[i], self.traj_ang[i], self.traj_desired_rot[i], self.cfg.rotation_halflife, Float32(i) * step)
        self.traj_desired_vel[0] = self.desired_velocity
        for i in range(1, 4):
            self.traj_desired_vel[i] = self._desired_velocity(inp.move, inp.camera_azimuth, self.traj_rot[i], sp[0], sp[1], sp[2])
        self.traj_pos[0] = self.sim_position
        self.traj_vel[0] = self.sim_velocity
        self.traj_acc[0] = self.sim_acceleration
        for i in range(1, 4):
            self.traj_pos[i] = self.traj_pos[i - 1]
            self.traj_vel[i] = self.traj_vel[i - 1]
            self.traj_acc[i] = self.traj_acc[i - 1]
            mm_spring_character_update(self.traj_pos[i], self.traj_vel[i], self.traj_acc[i], self.traj_desired_vel[i], self.cfg.velocity_halflife, step)

    def _build_query(mut self):
        var nf = self.db.nfeatures
        for j in range(15):
            if self.lmm_enabled:
                self.query[j] = self.features_curr[j] * self.db.features_scale[j] + self.db.features_offset[j]
            else:
                self.query[j] = self.db.denormalized_feature(self.frame_index, j)
        var rp = self.out.pos[0]
        var rr = self.out.rot[0]
        var k = 15
        for i in range(1, 4):
            var t = mm_quat_inv_mul_vec(rr, self.traj_pos[i] - rp)
            self.query[k] = t.x
            self.query[k + 1] = t.z
            k += 2
        for i in range(1, 4):
            var d = mm_quat_inv_mul_vec(rr, self.traj_rot[i].rotate(Vec3(0, 0, 1)))
            self.query[k] = d.x
            self.query[k + 1] = d.z
            k += 2
        _ = nf

    # ------------------------------------------------------------------ per frame

    def _animate(mut self, inp: MmInput, dt: Float32):
        """Everything up to (and including) the inertializer update; the simulation object is
        advanced by the caller."""
        mm_simple_spring_scalar(self.desired_gait, self.desired_gait_velocity, Float32(1) if inp.walk else Float32(0), self.cfg.gait_halflife, dt)
        var sp = self._speeds()
        var dv = self._desired_velocity(inp.move, inp.camera_azimuth, self.sim_rotation, sp[0], sp[1], sp[2])
        var dr = self._desired_rotation(self.desired_rotation, inp, dv)
        var inv_dt = Float32(1) / max(dt, Float32(1.0e-6))
        self.dvel_change_prev = self.dvel_change_curr
        self.dvel_change_curr = (dv - self.desired_velocity) * inv_dt
        self.desired_velocity = dv
        self.drot_change_prev = self.drot_change_curr
        self.drot_change_curr = mm_quat_to_scaled_angle_axis(mm_quat_abs(mm_quat_mul_inv(dr, self.desired_rotation))) * inv_dt
        self.desired_rotation = dr
        var force = False
        var vt = self.cfg.velocity_change_threshold
        var rt = self.cfg.rotation_change_threshold
        if self.force_search_timer <= 0 and (
            (self.dvel_change_prev.length() >= vt and self.dvel_change_curr.length() < vt)
            or (self.drot_change_prev.length() >= rt and self.drot_change_curr.length() < rt)
        ):
            force = True
            self.force_search_timer = self.cfg.search_time
        elif self.force_search_timer > 0:
            self.force_search_timer -= dt

        self._predict(inp)
        self._build_query()

        self.just_transitioned = False
        var end_of_anim = self.db.trajectory_index_clamp(self.frame_index, 1) == self.frame_index
        if force or self.search_timer <= 0 or end_of_anim:
            self.searches += 1
            var qn = self.db.normalize_query(self.query)
            if self.lmm_enabled and len(self.nets) > 0:
                var res = self.nets[0].project(self.features_proj, self.latent_proj, qn, self.features_curr, self.cfg.transition_cost)
                self.last_cost = res[1]
                if res[0]:
                    var root_p = self.curr.pos[0]
                    var root_r = self.curr.rot[0]
                    var tc = List[Bool](length=2, fill=False)
                    self.nets[0].decompress(self.trns, tc, self.features_proj, self.latent_proj, root_p, root_r, dt)
                    self.inert.transition(self.out, self.curr, self.trns)
                    for j in range(len(self.features_curr)):
                        self.features_curr[j] = self.features_proj[j]
                    for j in range(len(self.latent_curr)):
                        self.latent_curr[j] = self.latent_proj[j]
                    self.transitions += 1
                    self.just_transitioned = True
            else:
                var best = -1 if end_of_anim else self.frame_index
                var cost = MM_BIG
                self.db.search(qn, best, cost, self.cfg.transition_cost, self.cfg.ignore_range_end, self.cfg.ignore_surrounding)
                self.last_cost = cost
                if best != self.frame_index and best >= 0:
                    self.db.pose_at(best, self.trns)
                    self.inert.transition(self.out, self.curr, self.trns)
                    self.frame_index = best
                    self.transitions += 1
                    self.just_transitioned = True
            self.search_timer = self.cfg.search_time
        self.search_timer -= dt

        if self.lmm_enabled and len(self.nets) > 0:
            self.nets[0].step(self.features_curr, self.latent_curr, dt)
            var root_p = self.curr.pos[0]
            var root_r = self.curr.rot[0]
            self.nets[0].decompress(self.curr, self.curr_contacts, self.features_curr, self.latent_curr, root_p, root_r, dt)
        else:
            self.frame_index = self.db.trajectory_index_clamp(self.frame_index, 1)
            self.db.pose_at(self.frame_index, self.curr)
            self.curr_contacts[0] = self.db.contact(self.frame_index, 0)
            self.curr_contacts[1] = self.db.contact(self.frame_index, 1)
        self.inert.update(self.out, self.curr, self.cfg.inertialize_halflife, dt)

    def _adjust(mut self, dt: Float32):
        if self.cfg.adjustment:
            var ap = self.out.pos[0]
            var ar = self.out.rot[0]
            if self.cfg.adjustment_by_velocity:
                ap = _adjust_position_by_velocity(self.out.pos[0], self.out.vel[0], self.sim_position, self.cfg.adjust_position_max_ratio, self.cfg.adjust_position_halflife, dt)
                ar = _adjust_rotation_by_velocity(self.out.rot[0], self.out.ang[0], self.sim_rotation, self.cfg.adjust_rotation_max_ratio, self.cfg.adjust_rotation_halflife, dt)
            else:
                ap = mm_damp_adjustment_vec(self.sim_position - ap, self.cfg.adjust_position_halflife, dt) + ap
                ar = mm_quat_mul(mm_damp_adjustment_quat(mm_quat_abs(mm_quat_mul_inv(self.sim_rotation, ar).normalized()), self.cfg.adjust_rotation_halflife, dt), ar).normalized()
            self.inert.root_adjust(self.out.pos[0], self.out.rot[0], ap, ar)
        if self.cfg.clamping:
            var cp = _clamp_position(self.out.pos[0], self.sim_position, self.cfg.clamp_max_distance)
            var cr = _clamp_rotation(self.out.rot[0], self.sim_rotation, self.cfg.clamp_max_angle)
            self.inert.root_adjust(self.out.pos[0], self.out.rot[0], cp, cr)

    def update(mut self, inp: MmInput, dt: Float32 = 1.0 / 60.0):
        """One 60 Hz tick with a free-moving simulation object."""
        self._animate(inp, dt)
        mm_spring_character_update(self.sim_position, self.sim_velocity, self.sim_acceleration, self.desired_velocity, self.cfg.velocity_halflife, dt)
        mm_simple_spring_quat(self.sim_rotation, self.sim_angular_velocity, self.desired_rotation, self.cfg.rotation_halflife, dt)
        self._adjust(dt)

    def update_kcc(mut self, inp: MmInput, mut kcc: CharacterController, mut world: PhysicsWorld, dt: Float32 = 1.0 / 60.0):
        """One tick where the simulation object is a kinematic `CharacterController`: the
        spring proposes the move, the controller collides it, and the result (its foot point)
        becomes the simulation position the animation is pulled toward."""
        self._animate(inp, dt)
        var prev = self.sim_position
        var p = prev
        mm_spring_character_update(p, self.sim_velocity, self.sim_acceleration, self.desired_velocity, self.cfg.velocity_halflife, dt)
        var move = (p - prev) * (Float32(1) / max(dt, Float32(1.0e-6)))
        _ = kcc.move_and_slide(world, Vec3(move.x, 0, move.z), dt)
        var foot = kcc.foot_position()
        self.sim_position = foot
        kcc.yaw = mm_quat_yaw(self.sim_rotation) * _RAD2DEG
        mm_simple_spring_quat(self.sim_rotation, self.sim_angular_velocity, self.desired_rotation, self.cfg.rotation_halflife, dt)
        self._adjust(dt)

    def update_from_input(mut self, imap: InputMap, camera_azimuth: Float32 = 0, dt: Float32 = 1.0 / 60.0):
        self.update(mm_input_from_map(imap, camera_azimuth), dt)

    # ------------------------------------------------------------------ output

    def pose(self) -> ref [self.out] MmPose:
        return self.out

    def placement(self) -> Placement:
        """Root position and heading (degrees) for drawing / `PhysicalAnimator`."""
        return Placement(self.out.pos[0], Vec3(1, 1, 1), mm_quat_yaw(self.out.rot[0]) * _RAD2DEG)

    def write_model_pose(self, mut pose: Pose):
        """Local node transforms for `mm_stick_biped_model(db)` under `placement()`."""
        var n = self.db.nbones
        pose.resize(n)
        var yaw = mm_yaw_quat(mm_quat_yaw(self.out.rot[0]))
        pose.t[0] = Vec3()
        pose.r[0] = yaw.conjugate().multiply(self.out.rot[0]).normalized()
        pose.s[0] = Vec3(1, 1, 1)
        for i in range(1, n):
            pose.t[i] = self.out.pos[i]
            pose.r[i] = self.out.rot[i]
            pose.s[i] = Vec3(1, 1, 1)

    def global_positions(self) -> List[Vec3]:
        var gp = List[Vec3](length=self.db.nbones, fill=Vec3())
        var gr = List[Quat](length=self.db.nbones, fill=Quat())
        for i in range(self.db.nbones):
            var p = self.db.parents[i]
            if p < 0:
                gp[i] = self.out.pos[i]
                gr[i] = self.out.rot[i]
            else:
                gp[i] = gr[p].rotate(self.out.pos[i]) + gp[p]
                gr[i] = gr[p].multiply(self.out.rot[i]).normalized()
        return gp^
