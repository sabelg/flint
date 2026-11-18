"""Mecha & Reverse-Joint Walker Kinematics (`flint.anim.mecha_ik`).

Provides unified access and rig-level drivers for:
1. Closed-loop 4-bar Freudenstein planar/3D linkages (`solve_four_bar_linkage`,
   `FourBarLinkageDriver`) — e.g. Kawamori VF-1 Valkyrie nosecone-to-chest folding
   and hip-intake swing bars with zero mesh self-intersection.
2. 3-bone digitigrade / reverse-knee chicken-walker IK (`three_bone_digitigrade_ik`,
   `DigitigradeLegDriver`) — e.g. Zentraedi `Regult` / `Glaug` battle pods and
   VF-1 `GERWALK` postures.
3. Stateful terrain & carrier-deck `FootPlantSolver` re-export and digitigrade
   foot-lock helper.
"""
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.ik import (
    solve_two_bone_pole,
    solve_four_bar_linkage,
    three_bone_digitigrade_ik,
    shortest_arc_quat,
)
from flint.assets.model import NativeModel
from flint.anim.pose import SkeletonInstance
from flint.anim.procedural import Placement, FootPlantLeg, FootPlantSolver, _clamped_surface_tilt
from flint.core.springs import SpringFloat, decay_spring


struct FourBarLinkageDriver(Copyable, ImplicitlyCopyable, Movable):
    """Drives crank, coupler, and rocker bones on a `SkeletonInstance` from an input
    transformation progress `crank_angle` using exact circle-circle closure."""
    var crank_node: Int
    var coupler_node: Int
    var rocker_node: Int
    var ground_a: Vec3
    var ground_d: Vec3
    var len_crank_ab: Float32
    var len_coupler_bc: Float32
    var len_rocker_cd: Float32
    var plane_normal: Vec3
    var sign_branch: Float32

    def __init__(
        out self,
        crank_node: Int,
        coupler_node: Int,
        rocker_node: Int,
        ground_a: Vec3,
        ground_d: Vec3,
        len_crank_ab: Float32,
        len_coupler_bc: Float32,
        len_rocker_cd: Float32,
        plane_normal: Vec3 = Vec3(1.0, 0.0, 0.0),
        sign_branch: Float32 = 1.0,
    ):
        self.crank_node = crank_node
        self.coupler_node = coupler_node
        self.rocker_node = rocker_node
        self.ground_a = ground_a
        self.ground_d = ground_d
        self.len_crank_ab = len_crank_ab
        self.len_coupler_bc = len_coupler_bc
        self.len_rocker_cd = len_rocker_cd
        self.plane_normal = plane_normal
        self.sign_branch = sign_branch

    def evaluate(self, crank_angle: Float32) -> Tuple[Vec3, Vec3, Float32, Float32]:
        return solve_four_bar_linkage(
            self.ground_a,
            self.ground_d,
            self.len_crank_ab,
            self.len_coupler_bc,
            self.len_rocker_cd,
            crank_angle,
            self.plane_normal,
            self.sign_branch,
        )

    def apply_to_skeleton(
        self,
        mut skel: SkeletonInstance,
        model: NativeModel,
        crank_angle: Float32,
        rest_dir_ab: Vec3 = Vec3(0.0, 0.0, 1.0),
        rest_dir_bc: Vec3 = Vec3(0.0, 0.0, 1.0),
        rest_dir_dc: Vec3 = Vec3(0.0, 1.0, 0.0),
    ) -> Tuple[Vec3, Vec3, Float32, Float32]:
        var res = self.evaluate(crank_angle)
        var joint_b = res[0]
        var joint_c = res[1]
        skel.ensure_topology(model)
        if self.crank_node >= 0 and self.crank_node < skel.node_count:
            skel.pose.t[self.crank_node] = self.ground_a
            skel.pose.r[self.crank_node] = shortest_arc_quat(rest_dir_ab, joint_b - self.ground_a)
        if self.coupler_node >= 0 and self.coupler_node < skel.node_count:
            var q_ab = skel.pose.r[self.crank_node] if (self.crank_node >= 0 and self.crank_node < skel.node_count) else Quat()
            var q_bc_world = shortest_arc_quat(rest_dir_bc, joint_c - joint_b)
            if skel.parents[self.coupler_node] == self.crank_node:
                skel.pose.r[self.coupler_node] = q_ab.conjugate().multiply(q_bc_world).normalized()
            else:
                skel.pose.t[self.coupler_node] = joint_b
                skel.pose.r[self.coupler_node] = q_bc_world
        if self.rocker_node >= 0 and self.rocker_node < skel.node_count:
            skel.pose.t[self.rocker_node] = self.ground_d
            skel.pose.r[self.rocker_node] = shortest_arc_quat(rest_dir_dc, joint_c - self.ground_d)
        skel.update_matrices(model)
        return res


struct DigitigradeLegDriver(Copyable, Movable):
    """Applies `three_bone_digitigrade_ik` + world stance-lock foot placement to a
    4-node (`hip -> knee1 -> ankle_rev -> foot`) reverse-joint walker leg."""
    var hip: Int
    var knee1: Int
    var ankle_rev: Int
    var foot: Int
    var foot_height: Float32
    var crouch_bias: Float32
    var locked: Bool
    var locked_world_pos: Vec3
    var release_offset: Vec3
    var release_vel: Vec3
    var tilt_blend: SpringFloat

    def __init__(
        out self,
        hip: Int,
        knee1: Int,
        ankle_rev: Int,
        foot: Int,
        foot_height: Float32 = 0.08,
        crouch_bias: Float32 = 0.5,
    ):
        self.hip = hip
        self.knee1 = knee1
        self.ankle_rev = ankle_rev
        self.foot = foot
        self.foot_height = foot_height
        self.crouch_bias = crouch_bias
        self.locked = False
        self.locked_world_pos = Vec3()
        self.release_offset = Vec3()
        self.release_vel = Vec3()
        self.tilt_blend = SpringFloat(0.0)

    def apply_to_skeleton(
        mut self,
        mut skel: SkeletonInstance,
        model: NativeModel,
        placement: Placement,
        in_stance: Bool,
        hit_y: Float32,
        hit_normal: Vec3,
        dt: Float32,
        forward_model: Vec3 = Vec3(0.0, 0.0, 1.0),
    ):
        skel.ensure_topology(model)
        skel.update_matrices(model)
        var scale_y = max(Float32(0.01), abs(placement.scale.y))
        var foot_h_w = self.foot_height * scale_y
        var p_anim_w = placement.to_world(skel.global_position(self.foot))
        var swing_w = Vec3(p_anim_w.x, max(p_anim_w.y, hit_y + foot_h_w), p_anim_w.z)
        var tgt_w: Vec3
        if in_stance:
            if not self.locked:
                var p0 = swing_w + self.release_offset
                self.locked_world_pos = Vec3(p0.x, hit_y + foot_h_w, p0.z)
                self.locked = True
                self.release_offset = Vec3()
                self.release_vel = Vec3()
            else:
                self.locked_world_pos.y = hit_y + foot_h_w
            tgt_w = self.locked_world_pos
        else:
            if self.locked:
                self.locked = False
                self.release_offset = self.locked_world_pos - swing_w
                self.release_vel = Vec3()
            if dt > Float32(0.0):
                decay_spring(self.release_offset, self.release_vel, Float32(0.07), dt)
            tgt_w = swing_w + self.release_offset

        # Reset leg local rotations to rest before computing hierarchical digitigrade IK
        skel.pose.r[self.hip] = model.nodes[self.hip].base_rotation
        skel.pose.r[self.knee1] = model.nodes[self.knee1].base_rotation
        skel.pose.r[self.ankle_rev] = model.nodes[self.ankle_rev].base_rotation
        skel.pose.r[self.foot] = model.nodes[self.foot].base_rotation
        skel.update_matrices(model)

        var p_hip = skel.global_position(self.hip)
        var p_k1 = skel.global_position(self.knee1)
        var p_ar = skel.global_position(self.ankle_rev)
        var p_ft = skel.global_position(self.foot)
        var tgt_m = placement.to_model(tgt_w)

        var q_res = three_bone_digitigrade_ik(p_hip, p_k1, p_ar, p_ft, tgt_m, forward_model, self.crouch_bias)
        var parent_hip = skel.parents[self.hip]
        var gp = skel.global_rotation(parent_hip) if parent_hip >= 0 else Quat()
        skel.pose.r[self.hip] = gp.conjugate().multiply(q_res[0]).multiply(gp).multiply(skel.pose.r[self.hip]).normalized()
        skel.pose.r[self.knee1] = q_res[1].multiply(skel.pose.r[self.knee1]).normalized()
        skel.pose.r[self.ankle_rev] = q_res[2].multiply(skel.pose.r[self.ankle_rev]).normalized()
        skel.update_matrices(model)

        self.tilt_blend.update(Float32(1.0) if in_stance else Float32(0.0), Float32(0.06), dt)
        var tilt_w = Quat().slerp(_clamped_surface_tilt(hit_normal, Float32(0.65)), self.tilt_blend.value)
        var wy = placement.world_yaw_quat()
        skel.rotate_node_model(self.foot, wy.conjugate().multiply(tilt_w).multiply(wy).normalized(), model)


def make_biped_foot_solver(
    pelvis: Int,
    hip_l: Int, knee_l: Int, ankle_l: Int,
    hip_r: Int, knee_r: Int, ankle_r: Int,
    ankle_height: Float32 = 0.10,
) -> FootPlantSolver:
    """Constructs a configured two-leg `FootPlantSolver` for humanoid or Battloid rigs."""
    var solver = FootPlantSolver(pelvis)
    solver.add_leg(hip_l, knee_l, ankle_l, ankle_height, Vec3(0.0, 0.5, 1.0))
    solver.add_leg(hip_r, knee_r, ankle_r, ankle_height, Vec3(0.0, 0.5, 1.0))
    return solver^


def solve_two_bone_mecha_limb(
    mut skel: SkeletonInstance,
    model: NativeModel,
    upper: Int,
    middle: Int,
    end: Int,
    target_model: Vec3,
    pole_model: Vec3,
):
    """Convenience wrapper invoking `solve_two_bone_pole` on a `SkeletonInstance`."""
    skel.ensure_topology(model)
    var q_pair = solve_two_bone_pole(
        skel.global_position(upper),
        skel.global_position(middle),
        skel.global_position(end),
        target_model,
        pole_model,
        skel.global_rotation(upper),
        skel.global_rotation(middle),
    )
    skel.pose.r[upper] = skel.pose.r[upper].multiply(q_pair[0]).normalized()
    skel.pose.r[middle] = skel.pose.r[middle].multiply(q_pair[1]).normalized()
    skel.update_matrices(model)
