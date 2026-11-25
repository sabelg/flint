"""Physics-driven animation blending (`flint.anim.physical_anim`).

`PhysicalAnimator` sits between an animation source (clips, a graph, motion matching or the
MotionBricks runtime) and the skeleton that is drawn:

* **Idle**: the output is the animation, bit for bit. The articulated ragdoll mirrors the
  animated pose every frame (with its velocities), so a hit always starts from a matching
  physical state.
* **Blend**: every limb is simulated. World-space drives pull each limb toward its animated
  world transform (following locomotion and root motion), local PD motors hold the posture,
  and the drawn pose blends each limb's local rotation from the animation toward the
  simulation by a per-limb weight.
* **Hits** raise a per-limb hit level that spreads along the limb hierarchy (by joint hops)
  and decays with a half-life. A raised level weakens that limb's world drive and motor
  and raises its physics weight, so a punch to the arm swings the arm and twists the chest
  while the legs keep walking. As levels decay the pose eases back to the animation and
  the controller returns to idle.
* **Falls**: a knockdown impulse, or the pelvis straying more than `fall_distance` from its
  animated position, hands the body to `RagdollStateMachine` (limp, settle, supine/prone
  get-up with zero pelvis snap). Once it is back on the animation, blending resumes.

Limbs are configured on `pa.rsm.ragdoll` (`add_limb_from_model`, root limb first); get-up
clips on `pa.rsm` (`supine_getup_clip`, `prone_getup_clip`). Everything uses
`flint.core.detmath`, so identical inputs give identical poses.
"""
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.detmath import det_exp
from flint.core.ik import angle_axis
from flint.assets.model import NativeModel
from flint.anim.pose import Pose, SkeletonInstance
from flint.anim.procedural import Placement, DEG
from flint.anim.flanim_ecs import RagdollStateMachine, RAGDOLL_SM_ANIMATED, RAGDOLL_SM_LIMP
from flint.physics.world import PhysicsWorld

comptime PA_IDLE: Int = 0
comptime PA_BLEND: Int = 1
comptime PA_FALLEN: Int = 2

comptime _LN2: Float32 = 0.69314718055994531


struct PhysicalAnimator(Copyable, Movable):
    """Per-limb physics/animation blending with hit reactions and fall + get-up."""
    var rsm: RagdollStateMachine
    var state: Int
    var hit_level: List[Float32]
    var base_weight: List[Float32]
    var hit_halflife: Float32
    var weight_gain: Float32
    var strength_drop: Float32
    var motor_drop: Float32
    var impulse_ref: Float32
    var reach: Float32
    var knockdown_impulse: Float32
    var fall_distance: Float32
    var recover_seconds: Float32
    var substeps: Int
    var falls: Int
    var hits: Int
    var anim_skel: SkeletonInstance
    var prev_place: Placement
    var has_prev: Bool
    var recover_timer: Float32
    var recovering: Bool
    var hops: List[Int]
    var targets_pos: List[Vec3]
    var targets_rot: List[Quat]
    var out_pose: Pose
    var last_pelvis_error: Float32

    def __init__(out self):
        self.rsm = RagdollStateMachine()
        self.state = PA_IDLE
        self.hit_level = List[Float32]()
        self.base_weight = List[Float32]()
        self.hit_halflife = Float32(0.3)
        self.weight_gain = Float32(1.5)
        self.strength_drop = Float32(0.95)
        self.motor_drop = Float32(0.6)
        self.impulse_ref = Float32(40.0)
        self.reach = Float32(1.0)
        self.knockdown_impulse = Float32(160.0)
        self.fall_distance = Float32(0.45)
        self.recover_seconds = Float32(0.6)
        self.substeps = 4
        self.falls = 0
        self.hits = 0
        self.anim_skel = SkeletonInstance()
        self.prev_place = Placement()
        self.has_prev = False
        self.recover_timer = Float32(0.0)
        self.recovering = False
        self.hops = List[Int]()
        self.targets_pos = List[Vec3]()
        self.targets_rot = List[Quat]()
        self.out_pose = Pose()
        self.last_pelvis_error = Float32(0.0)

    # ------------------------------------------------------------------ configuration

    def limb_count(self) -> Int:
        return self.rsm.ragdoll.limb_count()

    def _ensure_limbs(mut self):
        var n = self.limb_count()
        if len(self.hit_level) != n:
            self.hit_level = List[Float32](length=n, fill=Float32(0.0))
        if len(self.base_weight) != n:
            self.base_weight = List[Float32](length=n, fill=Float32(0.0))
        if len(self.hops) != n * n:
            self._build_hops()

    def _build_hops(mut self):
        """Joint hops between every pair of limbs along the limb tree."""
        var n = self.limb_count()
        self.hops = List[Int](length=n * n, fill=Int(1 << 20))
        for a in range(n):
            # walk a's ancestors, recording their depth from a; then every limb's path to a
            # meets one of them
            var up = List[Int](length=n, fill=-1)
            var cur = a
            var d = 0
            while cur >= 0 and d <= n:
                up[cur] = d
                cur = self.rsm.ragdoll.limbs[cur].parent_limb
                d += 1
            for b in range(n):
                var c = b
                var db = 0
                while c >= 0 and up[c] < 0 and db <= n:
                    c = self.rsm.ragdoll.limbs[c].parent_limb
                    db += 1
                if c >= 0 and up[c] >= 0:
                    self.hops[a * n + b] = up[c] + db

    def set_base_weight(mut self, limb: Int, weight: Float32):
        """Physics weight of `limb` with no hit (0 = animation, 1 = simulation). A non-zero
        base weight keeps the controller blending (secondary motion such as loose arms)."""
        self._ensure_limbs()
        if limb >= 0 and limb < len(self.base_weight):
            self.base_weight[limb] = max(Float32(0.0), min(Float32(1.0), weight))

    def physics_weight(self, limb: Int) -> Float32:
        if limb < 0 or limb >= len(self.hit_level):
            return Float32(0.0)
        var b = self.base_weight[limb] if limb < len(self.base_weight) else Float32(0.0)
        return min(Float32(1.0), b + self.weight_gain * self.hit_level[limb])

    def _blending_needed(self) -> Bool:
        for i in range(len(self.hit_level)):
            if self.hit_level[i] > Float32(0.0):
                return True
        for i in range(len(self.base_weight)):
            if self.base_weight[i] > Float32(0.0):
                return True
        return False

    # ------------------------------------------------------------------ events

    def hit(mut self, bone_node: Int, world_impulse: Vec3, world_point: Vec3):
        """A ballistic / melee impulse (N*s) at `world_point` on `bone_node` (or its nearest
        limb ancestor). Raises the hit levels around that limb; at `knockdown_impulse` or more
        the body falls."""
        self._ensure_limbs()
        if self.state == PA_FALLEN:
            self.rsm.ragdoll.apply_impulse_at_bone(bone_node, world_impulse, world_point)
            return
        var limb = self.rsm.ragdoll.limb_for_bone(bone_node)
        if limb < 0:
            return
        var n = self.limb_count()
        var lvl = world_impulse.length() / max(Float32(1.0e-3), self.impulse_ref)
        for j in range(n):
            var f = det_exp(-Float32(self.hops[limb * n + j]) / max(Float32(1.0e-3), self.reach))
            self.hit_level[j] = min(Float32(1.0), max(self.hit_level[j], lvl * f))
        self.rsm.ragdoll.apply_impulse_at_bone(bone_node, world_impulse, world_point)
        self.hits += 1
        if self.state == PA_IDLE:
            self.state = PA_BLEND
            self.rsm.ragdoll.set_active_pd(1.0)
        if world_impulse.length() >= self.knockdown_impulse:
            self.fall()

    def fall(mut self):
        """Hands the body to the ragdoll state machine: limp, settle, then get up."""
        self._ensure_limbs()
        self.rsm.ragdoll.go_limp()
        self.rsm.ragdoll.clear_world_targets()
        self.rsm.state = RAGDOLL_SM_LIMP
        self.rsm.limp_timer = Float32(0.0)
        for i in range(len(self.hit_level)):
            self.hit_level[i] = Float32(1.0)
        self.state = PA_FALLEN
        self.recovering = False
        self.falls += 1

    # ------------------------------------------------------------------ per frame

    def _animated_targets(mut self, placement: Placement):
        """World CoM and rotation of every limb in the animated pose (`anim_skel`)."""
        var n = self.rsm.ragdoll.limb_count()
        if len(self.targets_pos) != n:
            self.targets_pos = List[Vec3](length=n, fill=Vec3())
            self.targets_rot = List[Quat](length=n, fill=Quat())
        var q_place = angle_axis(placement.yaw * DEG, Vec3(0.0, 1.0, 0.0))
        for i in range(n):
            var node = self.rsm.ragdoll.limbs[i].bone_node
            var q = q_place.multiply(self.anim_skel.global_rotation(node)).normalized()
            var piv = placement.to_world(self.anim_skel.global_position(node))
            self.targets_pos[i] = piv + q.rotate(self.rsm.ragdoll.limbs[i].local_com_offset * self.rsm.ragdoll.scale)
            self.targets_rot[i] = q

    def update(
        mut self,
        world: PhysicsWorld,
        mut skel: SkeletonInstance,
        mut placement: Placement,
        model: NativeModel,
        anim_pose: Pose,
        dt: Float32,
    ):
        """Advances one frame: `anim_pose` is this frame's animated pose (model space under
        `placement`); the blended result is written to `skel`."""
        self._ensure_limbs()
        skel.ensure_topology(model)
        if self.limb_count() == 0 or anim_pose.count() != len(model.nodes):
            skel.apply_pose(anim_pose, model)
            return

        if self.state == PA_FALLEN:
            self.rsm.update(world, skel, placement, model, anim_pose, dt, True, self.substeps)
            if self.rsm.state == RAGDOLL_SM_ANIMATED:
                if not self.recovering:
                    self.recovering = True
                    self.recover_timer = Float32(0.0)
                self.recover_timer += max(Float32(0.0), dt)
                if self.recover_timer >= self.recover_seconds:
                    for i in range(len(self.hit_level)):
                        self.hit_level[i] = Float32(0.0)
                    self.state = PA_IDLE
                    self.has_prev = False
                    self.recovering = False
            return

        self.anim_skel.ensure_topology(model)
        if self.has_prev:
            self.anim_skel.snapshot_prev_matrices()
        self.anim_skel.apply_pose(anim_pose, model)

        if not self.has_prev or (self.state == PA_IDLE and not self._blending_needed()):
            # mirror the animation (with its velocities) so a hit starts from it
            if self.has_prev:
                self.rsm.ragdoll.handoff_from_skeleton(self.anim_skel, placement, model, dt, self.prev_place)
            else:
                self.rsm.ragdoll.handoff_from_skeleton(self.anim_skel, placement, model, dt)
            self._animated_targets(placement)
            self.rsm.ragdoll.set_world_targets(self.targets_pos, self.targets_rot, dt, True)
            self.prev_place = placement
            self.has_prev = True
            skel.apply_pose(anim_pose, model)
            if self._blending_needed():
                self.state = PA_BLEND
                self.rsm.ragdoll.set_active_pd(1.0)
            return

        # ---- blend
        self.rsm.sync_physics_environment(world)
        self.rsm.ragdoll.set_target_pose(anim_pose, placement.yaw * DEG)
        self._animated_targets(placement)
        self.rsm.ragdoll.set_world_targets(self.targets_pos, self.targets_rot, dt)
        var n = self.rsm.ragdoll.limb_count()
        for i in range(n):
            var lvl = self.hit_level[i]
            self.rsm.ragdoll.drive_strength[i] = max(Float32(0.0), Float32(1.0) - self.strength_drop * lvl)
            self.rsm.ragdoll.limb_motor[i] = max(Float32(0.0), Float32(1.0) - self.motor_drop * lvl)
        self.rsm.ragdoll.step(dt, self.substeps)
        self.prev_place = placement
        self.has_prev = True

        self.last_pelvis_error = (self.rsm.ragdoll.states[0].com_pos - self.targets_pos[0]).length()
        if self.last_pelvis_error > self.fall_distance * self.rsm.ragdoll.scale:
            self.fall()
            self.rsm.ragdoll.write_to_skeleton(skel, model, placement)
            return

        var phys = self.rsm.ragdoll.pose_in_frame(placement.position, placement.yaw * DEG, anim_pose)
        self.out_pose.assign(anim_pose)
        for i in range(n):
            var u = self.rsm.ragdoll.limbs[i].bone_node
            if u < 0 or u >= self.out_pose.count():
                continue
            var w = self.physics_weight(i)
            if w <= Float32(0.0):
                continue
            self.out_pose.r[u] = anim_pose.r[u].nlerp(phys.r[u], w)
            if self.rsm.ragdoll.limbs[i].parent_limb < 0:
                self.out_pose.t[u] = anim_pose.t[u] + (phys.t[u] - anim_pose.t[u]) * w
        skel.apply_pose(self.out_pose, model)

        # hit levels decay; with none left the controller goes back to mirroring
        var keep = det_exp(-_LN2 * max(Float32(0.0), dt) / max(Float32(1.0e-3), self.hit_halflife))
        for i in range(n):
            var v = self.hit_level[i] * keep
            self.hit_level[i] = v if v > Float32(2.0e-3) else Float32(0.0)
        if not self._blending_needed():
            self.state = PA_IDLE
