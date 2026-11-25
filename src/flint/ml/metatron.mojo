"""NVIDIA Metatron-v1 whole-body humanoid controller for the Unitree G1 (`flint.ml.metatron`).

Metatron-v1 (Hugging Face `nvidia/Metatron-v1`, NVIDIA Open Model License; ProtoMotions code
Apache-2.0) is a MaskedMimic-style transformer student: a deterministic learned prior plus an
action decoder that maps the robot's sensor readings, its own action history and sparse, masked
goals (future body targets, a planar velocity / heading command, style samples, scene boxes) to
29 joint PD targets with their stiffness and damping, at 50 Hz. Flint runs the released
single-robot browser export `demo/model.onnx` (2914 nodes, 39 inputs, ~403 MB FP32) on
`flint.ml.onnx.OnnxModel`, on a MAX GPU when present. The graph contains the observation
processing (forward kinematics from the joints and the torso pose, root-heading frames,
normalisation), the prior / decoder and the affine action map; the host supplies only raw sensor
values, history and goals. The weights stay outside the repository (`tools/metatron/fetch.sh`).

Contract (`protomotions.masked_mimic.g1.v5_raw_context_scene`, batch 1, quaternions xyzw, metres,
radians, rad/s, z up, joints in `G1_JOINTS` order, bodies in `G1_BODIES` order):

  current_root_pos [3], current_root_rot [4]   pelvis (floating base) position / orientation
  noisy_anchor_pos [3], noisy_anchor_rot [4]   torso_link (anchor, body 16) position / orientation
  noisy_dof_pos [29], noisy_dof_vel [29]       joint readings
  noisy_root_local_ang_vel [3]                 pelvis angular velocity in the pelvis frame (IMU)
  noisy_historical_{anchor_rot [8,4], dof_pos [8,29], dof_vel [8,29], root_local_ang_vel [8,3]}
                                               s(t-1) .. s(t-8), most recent first
  noisy_historical_raw_actions [8,29]          a(t-2) .. a(t-9): paired with s(t-k) is a(t-k-1)
  raw_action_history [9,29]                    a(t-1) .. a(t-9) (row 0 = the incoming action)
  masked_mimic_ref_pos [5,33,3], ref_rot [5,33,4]   global body targets at time_offsets [5] s
  masked_mimic_target_bodies_masks [50]        per slot, 5 conditionable bodies (torso, right
                                               foot, left foot, left hand, right hand) x
                                               (position, rotation)
  masked_mimic_target_poses_masks [5], anchor_translation_mask [5,2] (xy, z),
                                   anchor_rotation_mask [5,2] (full rotation, heading)
  masked_mimic_local_{ref_pos, ref_rot, target_bodies_masks, target_poses_masks, time_offsets}
                                               the same in the local (heading-free) channel
  masked_mimic_style_pose_samples [225] (5 frames x 5 bodies x (local xy, floor z, tangent,
                                   normal)), style_velocity_samples [150], style_condition_masks
                                   [5,11], style_frame_masks [5]
  masked_mimic_average_velocity [2]            commanded planar velocity, world frame (m/s)
  masked_mimic_velocity_command_mask [1]
  scene_boxes_* (8 inputs)                     one object, two boxes; hidden = all false
  outputs: action [29] (PD targets, rad), stiffness [29], damping [29], raw_action [29]
           (action = offset + scale * raw, `G1_ACTION_OFFSET` / `G1_ACTION_SCALE`)

The host loop (`MetatronInputs`, `MetatronTask`) is a port of ProtoMotions' native Metatron v1
runtime (deployment/native/src/metatron_v1_{contract,rest,task}.cpp and examples/
metatron_goals.py, Apache-2.0): idle (hands-down calibrated rest pose, held heading) and steering
(future torso / foot / hand handrail from a velocity + heading command). Recovery, teleop and
motion tracking are not ported. The physics side is `flint.physics.mjcf` (`MjcfModel` for the
G1 MJCF, `MjcfArticulation` in `PhysicsWorld`); `examples/metatron` closes the loop.
"""

from std.math import sin, cos, atan2, exp, sqrt
from flint.ml.onnx import OnnxModel
from flint.ml.tensor import Tensor, DT_BOOL

comptime MT_INPUTS: Int = 39
comptime MT_DOF: Int = 29
comptime MT_BODIES: Int = 33
comptime MT_SLOTS: Int = 5
comptime MT_HISTORY: Int = 8
comptime MT_ANCHOR: Int = 16
comptime MT_RIGHT_FOOT: Int = 13
comptime MT_LEFT_FOOT: Int = 7
comptime MT_LEFT_HAND: Int = 24
comptime MT_RIGHT_HAND: Int = 32
comptime MT_CONTROL_DT: Float32 = 0.02
comptime MT_PI: Float32 = 3.14159265358979323846

# input indices (manifest order)
comptime MI_ROOT_POS: Int = 0
comptime MI_ROOT_ROT: Int = 1
comptime MI_ANCHOR_ROT_MASK: Int = 2
comptime MI_ANCHOR_TRANS_MASK: Int = 3
comptime MI_AVG_VEL: Int = 4
comptime MI_LOCAL_REF_POS: Int = 5
comptime MI_LOCAL_REF_ROT: Int = 6
comptime MI_LOCAL_BODY_MASKS: Int = 7
comptime MI_LOCAL_POSE_MASKS: Int = 8
comptime MI_LOCAL_TIMES: Int = 9
comptime MI_REF_POS: Int = 10
comptime MI_REF_ROT: Int = 11
comptime MI_STYLE_COND_MASKS: Int = 12
comptime MI_STYLE_FRAME_MASKS: Int = 13
comptime MI_STYLE_POSE: Int = 14
comptime MI_STYLE_VEL: Int = 15
comptime MI_BODY_MASKS: Int = 16
comptime MI_POSE_MASKS: Int = 17
comptime MI_TIMES: Int = 18
comptime MI_VEL_CMD_MASK: Int = 19
comptime MI_ANCHOR_POS: Int = 20
comptime MI_ANCHOR_ROT: Int = 21
comptime MI_DOF_POS: Int = 22
comptime MI_DOF_VEL: Int = 23
comptime MI_ROOT_ANG_VEL: Int = 24
comptime MI_H_ANCHOR_ROT: Int = 25
comptime MI_H_DOF_POS: Int = 26
comptime MI_H_DOF_VEL: Int = 27
comptime MI_H_RAW_ACTIONS: Int = 28
comptime MI_H_ROOT_ANG_VEL: Int = 29
comptime MI_RAW_ACTION_HISTORY: Int = 30
comptime MI_SCENE_OBJECT_ROT: Int = 34


def metatron_input_names() -> List[String]:
    return [
        "current_root_pos", "current_root_rot", "masked_mimic_anchor_rotation_mask",
        "masked_mimic_anchor_translation_mask", "masked_mimic_average_velocity", "masked_mimic_local_ref_pos",
        "masked_mimic_local_ref_rot", "masked_mimic_local_target_bodies_masks", "masked_mimic_local_target_poses_masks",
        "masked_mimic_local_time_offsets", "masked_mimic_ref_pos", "masked_mimic_ref_rot",
        "masked_mimic_style_condition_masks", "masked_mimic_style_frame_masks", "masked_mimic_style_pose_samples",
        "masked_mimic_style_velocity_samples", "masked_mimic_target_bodies_masks", "masked_mimic_target_poses_masks",
        "masked_mimic_time_offsets", "masked_mimic_velocity_command_mask", "noisy_anchor_pos", "noisy_anchor_rot",
        "noisy_dof_pos", "noisy_dof_vel", "noisy_root_local_ang_vel", "noisy_historical_anchor_rot",
        "noisy_historical_dof_pos", "noisy_historical_dof_vel", "noisy_historical_raw_actions",
        "noisy_historical_root_local_ang_vel", "raw_action_history", "scene_boxes_centers", "scene_boxes_extents",
        "scene_boxes_object_pos", "scene_boxes_object_rot", "scene_boxes_position_offset", "scene_boxes_valid",
        "scene_boxes_visible", "scene_boxes_yaw_offset",
    ]


def metatron_input_shape(k: Int) -> List[Int]:
    """Shape of input `k` without the batch dimension (`[]` for the scalar yaw offset)."""
    if k == 0: return [3]
    if k == 1: return [4]
    if k == 2 or k == 3: return [5, 2]
    if k == 4: return [2]
    if k == 5 or k == 10: return [5, 33, 3]
    if k == 6 or k == 11: return [5, 33, 4]
    if k == 7 or k == 16: return [50]
    if k == 8 or k == 9 or k == 13 or k == 17 or k == 18: return [5]
    if k == 12: return [5, 11]
    if k == 14: return [225]
    if k == 15: return [150]
    if k == 19: return [1]
    if k == 20 or k == 24 or k == 35: return [3]
    if k == 21: return [4]
    if k == 22 or k == 23: return [29]
    if k == 25: return [8, 4]
    if k == 26 or k == 27 or k == 28: return [8, 29]
    if k == 29: return [8, 3]
    if k == 30: return [9, 29]
    if k == 31 or k == 32: return [1, 2, 3]
    if k == 33: return [1, 3]
    if k == 34: return [1, 4]
    if k == 36: return [1, 2]
    if k == 37: return [1, 1]
    return List[Int]()


def metatron_input_is_bool(k: Int) -> Bool:
    return k == 2 or k == 3 or k == 7 or k == 8 or k == 12 or k == 13 or k == 16 or k == 17 or k == 19 \
        or k == 36 or k == 37


def metatron_input_numel(k: Int) -> Int:
    var n = 1
    for d in metatron_input_shape(k):
        n *= d
    return n


def metatron_input_offsets() -> List[Int]:
    """Offsets of the 39 inputs in the flat buffer (manifest order), plus the total at the end."""
    var out = List[Int](capacity=MT_INPUTS + 1)
    var o = 0
    for k in range(MT_INPUTS):
        out.append(o)
        o += metatron_input_numel(k)
    out.append(o)
    return out^


def g1_joint_names() -> List[String]:
    return [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
        "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]


def g1_body_names() -> List[String]:
    return [
        "pelvis", "head", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
        "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link",
        "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link", "waist_yaw_link",
        "waist_roll_link", "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
        "left_wrist_yaw_link", "left_rubber_hand", "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
        "right_wrist_yaw_link", "right_rubber_hand",
    ]


def g1_action_offset() -> List[Float32]:
    """The policy's action offset = the G1 default standing pose."""
    return [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, 0.0, 0.0, 0.0,
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]


def g1_action_scale() -> List[Float32]:
    var a: Float32 = 0.5475465059280396
    var b: Float32 = 0.3506614565849304
    var c: Float32 = 0.4385773241519928
    var d: Float32 = 0.07450086623430252
    return [a, b, a, b, c, c, a, b, a, b, c, c, a, c, c, c, c, c, c, c, d, d, c, c, c, c, c, d, d]


struct G1Gains(Copyable, Movable):
    """Per-joint actuator profile of the G1 (ProtoMotions robot_configs / metatron_v1_mujoco_model)."""
    var kp: List[Float32]
    var kd: List[Float32]
    var effort: List[Float32]
    var armature: List[Float32]

    def __init__(out self):
        self.kp = List[Float32]()
        self.kd = List[Float32]()
        self.effort = List[Float32]()
        self.armature = List[Float32]()
        for name in g1_joint_names():
            var kp: Float64 = 14.25062309787429
            var kd: Float64 = 0.907222843292423
            var eff: Float64 = 25.0
            var arm: Float64 = 0.003609725
            if "hip_pitch" in name or "hip_yaw" in name or "waist_yaw" in name:
                kp = 40.17923847137318; kd = 2.5578897650279457; eff = 88.0; arm = 0.010177520
            elif "hip_roll" in name or "knee" in name:
                kp = 99.09842777666113; kd = 6.3088018534966395; eff = 139.0; arm = 0.025101925
            elif "ankle" in name or "waist_roll" in name or "waist_pitch" in name:
                kp = 28.50124619574858; kd = 1.814445686584846; eff = 50.0; arm = 0.007219450
            elif "wrist_pitch" in name or "wrist_yaw" in name:
                kp = 16.77832748089279; kd = 1.06814150219; eff = 5.0; arm = 0.004250000
            self.kp.append(Float32(kp))
            self.kd.append(Float32(kd))
            self.effort.append(Float32(eff))
            self.armature.append(Float32(arm))


# ------------------------------------------------------------------------------------------ math

@always_inline
def _wrap(a: Float32) -> Float32:
    """std::remainder(a, 2 pi): the shortest equivalent angle in [-pi, pi]."""
    var t = Float64(a) / (2.0 * Float64(MT_PI))
    var n = round(t)
    return Float32(Float64(a) - n * 2.0 * Float64(MT_PI))


def mt_yaw_quat(yaw: Float32) -> List[Float32]:
    return [0.0, 0.0, sin(yaw * 0.5), cos(yaw * 0.5)]


def mt_yaw_of(q: List[Float32], o: Int = 0) -> Float32:
    """Heading of an xyzw quaternion stored at q[o..o+3]."""
    var x = q[o]
    var y = q[o + 1]
    var z = q[o + 2]
    var w = q[o + 3]
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def mt_quat_mul(a: List[Float32], b: List[Float32], ao: Int = 0, bo: Int = 0) -> List[Float32]:
    var ax = a[ao]; var ay = a[ao + 1]; var az = a[ao + 2]; var aw = a[ao + 3]
    var bx = b[bo]; var by = b[bo + 1]; var bz = b[bo + 2]; var bw = b[bo + 3]
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def mt_quat_rotate(q: List[Float32], qo: Int, v0: Float32, v1: Float32, v2: Float32) -> List[Float32]:
    var ux = q[qo]; var uy = q[qo + 1]; var uz = q[qo + 2]; var w = q[qo + 3]
    var dot = ux * v0 + uy * v1 + uz * v2
    var uu = ux * ux + uy * uy + uz * uz
    var cx = uy * v2 - uz * v1
    var cy = uz * v0 - ux * v2
    var cz = ux * v1 - uy * v0
    return [(w * w - uu) * v0 + 2.0 * dot * ux + 2.0 * w * cx,
            (w * w - uu) * v1 + 2.0 * dot * uy + 2.0 * w * cy,
            (w * w - uu) * v2 + 2.0 * dot * uz + 2.0 * w * cz]


@always_inline
def _rot_yaw(yaw: Float32, v0: Float32, v1: Float32, v2: Float32) -> Tuple[Float32, Float32, Float32]:
    var c = cos(yaw)
    var s = sin(yaw)
    return (c * v0 - s * v1, s * v0 + c * v1, v2)


# ------------------------------------------------------------------------------------- policy

struct MetatronOutputs(Copyable, Movable):
    var action: List[Float32]
    var stiffness: List[Float32]
    var damping: List[Float32]
    var raw_action: List[Float32]

    def __init__(out self):
        self.action = List[Float32](length=MT_DOF, fill=0.0)
        self.stiffness = List[Float32](length=MT_DOF, fill=0.0)
        self.damping = List[Float32](length=MT_DOF, fill=0.0)
        self.raw_action = List[Float32](length=MT_DOF, fill=0.0)

    def flat(self) -> List[Float32]:
        """action, stiffness, damping, raw_action (the reference dump's `outputs` row)."""
        var o = List[Float32](capacity=4 * MT_DOF)
        for v in self.action: o.append(v)
        for v in self.stiffness: o.append(v)
        for v in self.damping: o.append(v)
        for v in self.raw_action: o.append(v)
        return o^


struct MetatronPolicy(Movable):
    """`demo/model.onnx` on `OnnxModel`; `run(flat)` takes the 39 inputs flattened in manifest
    order (bools as 0 / 1), see `metatron_input_offsets`."""
    var model: OnnxModel
    var on_gpu: Bool
    var offsets: List[Int]

    def __init__(out self, var model: OnnxModel, on_gpu: Bool):
        self.model = model^
        self.on_gpu = on_gpu
        self.offsets = metatron_input_offsets()

    @staticmethod
    def load(path: String, gpu: Bool = True) raises -> MetatronPolicy:
        var m = OnnxModel.load(path)
        var names = metatron_input_names()
        var have = m.input_names()
        if len(have) != MT_INPUTS:
            raise Error("Metatron: " + path + " has " + String(len(have)) + " inputs, expected the 39-input demo graph")
        for n in names:
            if not m.has(n):
                raise Error("Metatron: " + path + " has no input " + n)
        var g = False
        if gpu:
            g = m.use_gpu()
        return MetatronPolicy(m^, g)

    def run(mut self, flat: List[Float32]) raises -> MetatronOutputs:
        if len(flat) != self.offsets[MT_INPUTS]:
            raise Error("Metatron: flat inputs have " + String(len(flat)) + " values, expected "
                        + String(self.offsets[MT_INPUTS]))
        var names = metatron_input_names()
        for k in range(MT_INPUTS):
            var shape: List[Int] = [1]
            for d in metatron_input_shape(k):
                shape.append(d)
            var n = metatron_input_numel(k)
            var o = self.offsets[k]
            if metatron_input_is_bool(k):
                var v = List[Int](capacity=n)
                for e in range(n):
                    v.append(1 if flat[o + e] != 0.0 else 0)
                self.model.set_input(names[k], Tensor.ints(DT_BOOL, shape^, v^))
            else:
                var v = List[Float32](capacity=n)
                for e in range(n):
                    v.append(flat[o + e])
                self.model.set_input(names[k], Tensor.floats(shape^, v^))
        self.model.run()
        var out = MetatronOutputs()
        var a = self.model.output("action")
        var s = self.model.output("stiffness")
        var d = self.model.output("damping")
        var r = self.model.output("raw_action")
        for j in range(MT_DOF):
            out.action[j] = a.fget(j)
            out.stiffness[j] = s.fget(j)
            out.damping[j] = d.fget(j)
            out.raw_action[j] = r.fget(j)
        return out^


# --------------------------------------------------------------------------------- host inputs

struct MetatronReading(Copyable, Movable):
    """One synchronized sensor sample (no simulator body state)."""
    var dof_pos: List[Float32]
    var dof_vel: List[Float32]
    var root_local_ang_vel: List[Float32]
    var anchor_rot: List[Float32]
    var anchor_pos: List[Float32]
    var root_pos: List[Float32]
    var root_rot: List[Float32]

    def __init__(out self):
        self.dof_pos = List[Float32](length=MT_DOF, fill=0.0)
        self.dof_vel = List[Float32](length=MT_DOF, fill=0.0)
        self.root_local_ang_vel = List[Float32](length=3, fill=0.0)
        self.anchor_rot = [0.0, 0.0, 0.0, 1.0]
        self.anchor_pos = List[Float32](length=3, fill=0.0)
        self.root_pos = List[Float32](length=3, fill=0.0)
        self.root_rot = [0.0, 0.0, 0.0, 1.0]


struct MetatronRest(Copyable, Movable):
    """The calibrated hands-down idle pose (`metatron_v1_rest.cpp`): body poses with the pelvis at
    the origin and heights above the lowest body, plus its zero-yaw style features."""
    var pos: List[Float32]      # [33*3]
    var rot: List[Float32]      # [33*4] xyzw
    var style: List[Float32]    # [225]
    var upright_height: Float32

    def __init__(out self):
        self.pos = List[Float32](length=MT_BODIES * 3, fill=0.0)
        self.rot = List[Float32](length=MT_BODIES * 4, fill=0.0)
        self.style = List[Float32](length=225, fill=0.0)
        self.upright_height = 0.0


def g1_rest_dof() -> List[Float32]:
    """Joint angles of the calibrated rest pose (elbows at 1.4 rad, shoulders 0 pitch)."""
    var d = List[Float32](length=MT_DOF, fill=0.0)
    d[0] = -0.312; d[6] = -0.312
    d[3] = 0.669; d[9] = 0.669
    d[4] = -0.363; d[10] = -0.363
    d[16] = 0.2; d[23] = -0.2
    d[18] = 1.4; d[25] = 1.4
    return d^


def metatron_rest_from_fk(body_pos: List[Float32], body_rot: List[Float32]) -> MetatronRest:
    """`make_rest_pose` from the FK of `g1_rest_dof()` with the pelvis at the origin (identity):
    shift heights so the lowest body is at 0, then the five-frame style template."""
    var r = MetatronRest()
    var floor = body_pos[2]
    for b in range(1, MT_BODIES):
        floor = min(floor, body_pos[b * 3 + 2])
    for b in range(MT_BODIES):
        r.pos[b * 3] = body_pos[b * 3]
        r.pos[b * 3 + 1] = body_pos[b * 3 + 1]
        r.pos[b * 3 + 2] = body_pos[b * 3 + 2] - floor
        for c in range(4):
            r.rot[b * 4 + c] = body_rot[b * 4 + c]
    r.upright_height = r.pos[MT_ANCHOR * 3 + 2]
    var cond: List[Int] = [MT_ANCHOR, MT_RIGHT_FOOT, MT_LEFT_FOOT, MT_LEFT_HAND, MT_RIGHT_HAND]
    var tx = r.pos[MT_ANCHOR * 3]
    var ty = r.pos[MT_ANCHOR * 3 + 1]
    for frame in range(5):
        for s in range(5):
            var b = cond[s]
            var base = (frame * 5 + s) * 9
            r.style[base] = r.pos[b * 3] - tx
            r.style[base + 1] = r.pos[b * 3 + 1] - ty
            r.style[base + 2] = r.pos[b * 3 + 2]
            var t = mt_quat_rotate(r.rot, b * 4, 1.0, 0.0, 0.0)
            var n = mt_quat_rotate(r.rot, b * 4, 0.0, 0.0, 1.0)
            for a in range(3):
                r.style[base + 3 + a] = t[a]
                r.style[base + 6 + a] = n[a]
    return r^


struct MetatronInputs(Movable):
    """Persistent policy inputs and the incoming-action history (`Inputs` in the native runtime).

    `prepare(reading)` writes the sensors and history; the task writes the goals; the policy runs
    on `flat`; `commit(before_step, raw_action)` advances the history once the command was applied
    (reset repeats the first reading and zero actions)."""
    var flat: List[Float32]
    var off: List[Int]
    var states: List[MetatronReading]
    var actions: List[List[Float32]]   # [9][29], row 0 = a(t-1)
    var initialized: Bool

    def __init__(out self):
        self.off = metatron_input_offsets()
        self.flat = List[Float32](length=self.off[MT_INPUTS], fill=0.0)
        self.states = List[MetatronReading]()
        self.actions = List[List[Float32]]()
        self.initialized = False
        self.reset()

    def reset(mut self):
        for e in range(len(self.flat)):
            self.flat[e] = 0.0
        for f in range(MT_SLOTS * MT_BODIES):
            self.flat[self.off[MI_LOCAL_REF_ROT] + f * 4 + 3] = 1.0
            self.flat[self.off[MI_REF_ROT] + f * 4 + 3] = 1.0
        self.flat[self.off[MI_SCENE_OBJECT_ROT] + 3] = 1.0
        self.states = List[MetatronReading]()
        self.actions = List[List[Float32]]()
        for _ in range(MT_HISTORY + 1):
            self.actions.append(List[Float32](length=MT_DOF, fill=0.0))
        self.initialized = False

    @always_inline
    def at(self, k: Int, e: Int) -> Float32:
        return self.flat[self.off[k] + e]

    @always_inline
    def put(mut self, k: Int, e: Int, v: Float32):
        self.flat[self.off[k] + e] = v

    @always_inline
    def flag(self, k: Int, e: Int) -> Bool:
        return self.flat[self.off[k] + e] != 0.0

    @always_inline
    def set_flag(mut self, k: Int, e: Int, v: Bool):
        self.flat[self.off[k] + e] = 1.0 if v else 0.0

    def _copy(mut self, k: Int, src: List[Float32], base: Int = 0):
        for e in range(len(src)):
            self.flat[self.off[k] + base + e] = src[e]

    def prepare(mut self, r: MetatronReading):
        if not self.initialized:
            self.states = List[MetatronReading]()
            for _ in range(MT_HISTORY):
                self.states.append(r.copy())
            self.initialized = True
        self._copy(MI_ROOT_POS, r.root_pos)
        self._copy(MI_ROOT_ROT, r.root_rot)
        self._copy(MI_ANCHOR_POS, r.anchor_pos)
        self._copy(MI_ANCHOR_ROT, r.anchor_rot)
        self._copy(MI_DOF_POS, r.dof_pos)
        self._copy(MI_DOF_VEL, r.dof_vel)
        self._copy(MI_ROOT_ANG_VEL, r.root_local_ang_vel)
        for row in range(MT_HISTORY):
            self._copy(MI_H_ANCHOR_ROT, self.states[row].anchor_rot, row * 4)
            self._copy(MI_H_DOF_POS, self.states[row].dof_pos, row * MT_DOF)
            self._copy(MI_H_DOF_VEL, self.states[row].dof_vel, row * MT_DOF)
            self._copy(MI_H_ROOT_ANG_VEL, self.states[row].root_local_ang_vel, row * 3)
            self._copy(MI_H_RAW_ACTIONS, self.actions[row + 1], row * MT_DOF)
        for row in range(MT_HISTORY + 1):
            self._copy(MI_RAW_ACTION_HISTORY, self.actions[row], row * MT_DOF)

    def commit(mut self, before_step: MetatronReading, raw_action: List[Float32]):
        self.states.insert(0, before_step.copy())
        while len(self.states) > MT_HISTORY:
            _ = self.states.pop()
        self.actions.insert(0, raw_action.copy())
        while len(self.actions) > MT_HISTORY + 1:
            _ = self.actions.pop()


def metatron_raw_from_action(action: List[Float32]) -> List[Float32]:
    """Inverse affine action map: the raw action that produces PD target `action`."""
    var off = g1_action_offset()
    var sc = g1_action_scale()
    var out = List[Float32](length=MT_DOF, fill=0.0)
    for j in range(MT_DOF):
        out[j] = (action[j] - off[j]) / sc[j]
    return out^


comptime MT_MODE_IDLE: Int = 0
comptime MT_MODE_STEERING: Int = 1


struct MetatronTask(Movable):
    """Idle / steering goal construction (`MetatronTask::build` + `MetatronGoals::apply`).

    `set_idle()` stands on the calibrated rest pose and holds the current heading (slowly
    following it, 0.5 s time constant); `set_steering(heading, vx, vy)` walks at a world-frame
    planar velocity while the torso heading slews to `heading` (at most pi rad/s). With zero
    velocity, steering plants the feet and turns in place."""
    var rest: MetatronRest
    var mode: Int
    var target_heading: Float32
    var vel_x: Float32
    var vel_y: Float32
    var stationary: Bool
    var hold_heading: Float32
    var heading_ready: Bool
    var gait_phase: Float32
    var support_ready: Bool
    var support_feet: List[Float32]   # right xyz, left xyz
    var support_anchor: List[Float32]
    var support_heading: Float32
    var torso_height_sent: Float32
    var torso_height_valid: Bool

    def __init__(out self, var rest: MetatronRest):
        self.rest = rest^
        self.mode = MT_MODE_IDLE
        self.target_heading = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.stationary = True
        self.hold_heading = 0.0
        self.heading_ready = False
        self.gait_phase = 0.0
        self.support_ready = False
        self.support_feet = List[Float32](length=6, fill=0.0)
        self.support_anchor = List[Float32](length=3, fill=0.0)
        self.support_heading = 0.0
        self.torso_height_sent = 0.0
        self.torso_height_valid = False

    def set_idle(mut self):
        if self.mode != MT_MODE_IDLE:
            self.mode = MT_MODE_IDLE
            self.support_ready = False
            self.heading_ready = False
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.stationary = True
        self.torso_height_valid = False

    def set_steering(mut self, heading: Float32, vx: Float32, vy: Float32):
        if self.mode != MT_MODE_STEERING:
            self.support_ready = False
            self.mode = MT_MODE_STEERING
        self.target_heading = _wrap(heading)
        self.vel_x = vx
        self.vel_y = vy

    def _rp(self, b: Int, a: Int) -> Float32:
        return self.rest.pos[b * 3 + a]

    def build(mut self, mut inp: MetatronInputs, r: MetatronReading, body_pos: List[Float32], body_rot: List[Float32]):
        """Write this step's goals. `body_pos` [33*3] / `body_rot` [33*4] are the measured body
        poses (the FK of the reading; a simulator may pass its own body state)."""
        var heading = mt_yaw_of(r.anchor_rot)
        for slot in range(MT_SLOTS):
            for e in range(MT_BODIES * 3):
                inp.put(MI_REF_POS, slot * MT_BODIES * 3 + e, body_pos[e])
            for e in range(MT_BODIES * 4):
                inp.put(MI_REF_ROT, slot * MT_BODIES * 4 + e, body_rot[e])
            inp.put(MI_TIMES, slot, Float32(slot + 1) * 0.1)
            inp.set_flag(MI_POSE_MASKS, slot, True)
        var speed = sqrt(self.vel_x * self.vel_x + self.vel_y * self.vel_y)
        self.stationary = (speed < 0.07) if self.stationary else (speed <= 0.03)
        var idle = self.mode == MT_MODE_IDLE
        var standing = idle or (speed <= 0.03 and abs(_wrap(self.target_heading - heading)) <= MT_PI / 9.0)
        if idle:
            if not self.heading_ready:
                self.hold_heading = heading
                self.heading_ready = True
            else:
                var alpha = 1.0 - exp(Float32(-0.02 / 0.5))
                self.hold_heading = _wrap(self.hold_heading + alpha * _wrap(heading - self.hold_heading))
            var q = mt_yaw_quat(self.hold_heading)
            for c in range(4):
                inp.put(MI_REF_ROT, (4 * MT_BODIES + MT_ANCHOR) * 4 + c, q[c])
            inp.set_flag(MI_BODY_MASKS, 4 * 10 + 1, True)
            inp.set_flag(MI_ANCHOR_ROT_MASK, 4 * 2 + 1, True)
        else:
            self._steering(inp, r, body_pos, heading)
        self._apply(inp, standing)

    def _steering(mut self, mut inp: MetatronInputs, r: MetatronReading, body_pos: List[Float32], heading: Float32):
        var speed = sqrt(self.vel_x * self.vel_x + self.vel_y * self.vel_y)
        var st = self.stationary
        if st and self.support_ready:
            for side in range(2):
                var b = MT_RIGHT_FOOT if side == 0 else MT_LEFT_FOOT
                var dx = body_pos[b * 3] - self.support_feet[side * 3]
                var dy = body_pos[b * 3 + 1] - self.support_feet[side * 3 + 1]
                if sqrt(dx * dx + dy * dy) > 0.12:
                    self.support_ready = False
                    break
        if st and not self.support_ready:
            for a in range(3):
                self.support_feet[a] = body_pos[MT_RIGHT_FOOT * 3 + a]
                self.support_feet[3 + a] = body_pos[MT_LEFT_FOOT * 3 + a]
                self.support_anchor[a] = body_pos[MT_ANCHOR * 3 + a]
            self.support_ready = True
            self.support_heading = heading
        if not st:
            self.support_ready = False
            self.torso_height_valid = False
        var dtx = self._rp(MT_ANCHOR, 0)
        var dty = self._rp(MT_ANCHOR, 1)
        var dtz = self._rp(MT_ANCHOR, 2)
        var fcx = 0.5 * (self._rp(MT_RIGHT_FOOT, 0) + self._rp(MT_LEFT_FOOT, 0)) - dtx
        var fcy = 0.5 * (self._rp(MT_RIGHT_FOOT, 1) + self._rp(MT_LEFT_FOOT, 1)) - dty
        var fcz = 0.5 * (self._rp(MT_RIGHT_FOOT, 2) + self._rp(MT_LEFT_FOOT, 2)) - dtz
        var default_floor = min(self._rp(MT_RIGHT_FOOT, 2), self._rp(MT_LEFT_FOOT, 2))
        var half_width = max(Float32(0.07), 0.5 * abs(self._rp(MT_LEFT_FOOT, 1) - self._rp(MT_RIGHT_FOOT, 1)))
        var cadence = min(Float32(2.0), 0.83 + 0.93 * speed)
        var swing: Float32 = 0.0 if st else 0.13 + 0.065 * speed
        var move_heading = heading if st else atan2(self.vel_y, self.vel_x)
        var ground = body_pos[2]
        for b in range(1, MT_BODIES):
            ground = min(ground, body_pos[b * 3 + 2])
        var sf = self.support_feet.copy()
        for slot in range(MT_SLOTS):
            var time = Float32(slot + 1) * 0.1
            var dh = _wrap(self.target_heading - heading)
            dh = max(-MT_PI * time, min(MT_PI * time, dh))
            var yaw = _wrap(heading + dh)
            var yq = mt_yaw_quat(yaw)
            var tx: Float32
            var ty: Float32
            var tz: Float32
            if st:
                tx = 0.5 * (sf[0] + sf[3])
                ty = 0.5 * (sf[1] + sf[4])
                var fl = min(sf[2], sf[5])
                var th = self.rest.upright_height - default_floor
                if slot == 0:
                    if self.torso_height_valid:
                        var budget: Float32 = 0.4 * 0.02
                        th = max(self.torso_height_sent - budget, min(self.torso_height_sent + budget, th))
                    self.torso_height_sent = th
                    self.torso_height_valid = True
                else:
                    th = self.torso_height_sent
                tz = fl + th
            else:
                tx = r.anchor_pos[0] + self.vel_x * time
                ty = r.anchor_pos[1] + self.vel_y * time
                tz = ground + self.rest.upright_height
            var pb = slot * MT_BODIES
            inp.put(MI_REF_POS, (pb + MT_ANCHOR) * 3, tx)
            inp.put(MI_REF_POS, (pb + MT_ANCHOR) * 3 + 1, ty)
            inp.put(MI_REF_POS, (pb + MT_ANCHOR) * 3 + 2, tz)
            for c in range(4):
                inp.put(MI_REF_ROT, (pb + MT_ANCHOR) * 4 + c, yq[c])
            var turn = st and abs(_wrap(yaw - self.support_heading)) > MT_PI / 6.0
            var feet = List[Float32](length=6, fill=0.0)
            for side in range(2):
                var b = MT_RIGHT_FOOT if side == 0 else MT_LEFT_FOOT
                var ox = self._rp(b, 0) - dtx
                var oy = self._rp(b, 1) - dty
                var oz = self._rp(b, 2) - dtz
                var sign: Float32 = -1.0 if side == 0 else 1.0
                var fx: Float32
                var fy: Float32
                var fz: Float32
                if st and not turn:
                    fx = sf[side * 3]; fy = sf[side * 3 + 1]; fz = sf[side * 3 + 2]
                elif st:
                    var rel = _rot_yaw(yaw, ox - fcx, oy - fcy, oz - fcz)
                    fx = tx + rel[0]; fy = ty + rel[1]; fz = sf[side * 3 + 2]
                else:
                    var rel = _wrap(move_heading - yaw)
                    oy = sign * (half_width + 0.12 * abs(sin(rel)))
                    var phase = self.gait_phase + 2.0 * MT_PI * cadence * time + (MT_PI if side == 1 else 0.0)
                    var lateral = min(Float32(1.0), 0.12 / swing) * abs(cos(rel))
                    ox += swing * cos(phase) * 0.75 * cos(rel)
                    oy += swing * cos(phase) * lateral * sin(rel)
                    oz += 0.08 * max(Float32(0.0), -sin(phase))
                    var rot = _rot_yaw(yaw, ox, oy, oz)
                    fx = tx + rot[0]; fy = ty + rot[1]; fz = tz + rot[2]
                feet[side * 3] = fx; feet[side * 3 + 1] = fy; feet[side * 3 + 2] = fz
                inp.put(MI_REF_POS, (pb + b) * 3, fx)
                inp.put(MI_REF_POS, (pb + b) * 3 + 1, fy)
                inp.put(MI_REF_POS, (pb + b) * 3 + 2, fz)
                var q = mt_quat_mul(yq, self.rest.rot, 0, b * 4)
                for c in range(4):
                    inp.put(MI_REF_ROT, (pb + b) * 4 + c, q[c])
            var sx = 0.5 * (feet[0] + feet[3])
            var sy = 0.5 * (feet[1] + feet[4])
            var sz = min(feet[2], feet[5])
            for h in range(2):
                var b = MT_LEFT_HAND if h == 0 else MT_RIGHT_HAND
                var rel = _rot_yaw(yaw, self._rp(b, 0) - dtx - fcx, self._rp(b, 1) - dty - fcy, self._rp(b, 2) - dtz - fcz)
                inp.put(MI_REF_POS, (pb + b) * 3, sx + rel[0])
                inp.put(MI_REF_POS, (pb + b) * 3 + 1, sy + rel[1])
                inp.put(MI_REF_POS, (pb + b) * 3 + 2, sz + rel[2])
            var m = slot * 10
            inp.set_flag(MI_BODY_MASKS, m, True)
            inp.set_flag(MI_BODY_MASKS, m + 1, True)
            inp.set_flag(MI_BODY_MASKS, m + 4, not st or turn)
            inp.set_flag(MI_BODY_MASKS, m + 2, not st or turn)
            inp.set_flag(MI_BODY_MASKS, m + 6, st or slot == 4)
            inp.set_flag(MI_BODY_MASKS, m + 8, st or slot == 4)
            inp.set_flag(MI_ANCHOR_TRANS_MASK, slot * 2, not st)
            inp.set_flag(MI_ANCHOR_TRANS_MASK, slot * 2 + 1, True)
            inp.set_flag(MI_ANCHOR_ROT_MASK, slot * 2, False)
            inp.set_flag(MI_ANCHOR_ROT_MASK, slot * 2 + 1, True)
        inp.put(MI_AVG_VEL, 0, self.vel_x)
        inp.put(MI_AVG_VEL, 1, self.vel_y)
        inp.set_flag(MI_VEL_CMD_MASK, 0, True)
        if st:
            self.gait_phase = 0.0
        else:
            var p = Float64(self.gait_phase) + 2.0 * Float64(MT_PI) * Float64(cadence) * 0.02
            var tp = 2.0 * Float64(MT_PI)
            self.gait_phase = Float32(p - tp * Float64(Int(p / tp)))

    def _apply(mut self, mut inp: MetatronInputs, standing: Bool):
        """`MetatronGoals::apply`: route body goals to the local channel, keep only the global
        heading, and replace the local goals with the static idle when standing."""
        for slot in range(MT_SLOTS):
            var any = False
            for b in range(5):
                for c in range(2):
                    var g = inp.flag(MI_BODY_MASKS, slot * 10 + b * 2 + c)
                    if b == 0 and c == 0:
                        g = g and inp.flag(MI_ANCHOR_TRANS_MASK, slot * 2 + 1)
                    if b == 0 and c == 1:
                        g = g and inp.flag(MI_ANCHOR_ROT_MASK, slot * 2)
                    inp.set_flag(MI_LOCAL_BODY_MASKS, slot * 10 + b * 2 + c, g)
                    any = any or g
            inp.set_flag(MI_LOCAL_POSE_MASKS, slot, inp.flag(MI_POSE_MASKS, slot) and any)
            inp.put(MI_LOCAL_TIMES, slot, inp.at(MI_TIMES, slot))
        for e in range(MT_SLOTS * MT_BODIES * 3):
            inp.put(MI_LOCAL_REF_POS, e, inp.at(MI_REF_POS, e))
        for e in range(MT_SLOTS * MT_BODIES * 4):
            inp.put(MI_LOCAL_REF_ROT, e, inp.at(MI_REF_ROT, e))
        for slot in range(MT_SLOTS):
            var heading = inp.flag(MI_BODY_MASKS, slot * 10 + 1) and inp.flag(MI_ANCHOR_ROT_MASK, slot * 2 + 1)
            for e in range(10):
                inp.set_flag(MI_BODY_MASKS, slot * 10 + e, False)
            inp.set_flag(MI_BODY_MASKS, slot * 10 + 1, heading)
            inp.set_flag(MI_ANCHOR_TRANS_MASK, slot * 2, False)
            inp.set_flag(MI_ANCHOR_TRANS_MASK, slot * 2 + 1, False)
            inp.set_flag(MI_ANCHOR_ROT_MASK, slot * 2, False)
            inp.set_flag(MI_POSE_MASKS, slot, inp.flag(MI_POSE_MASKS, slot) and heading)
        if standing:
            for slot in range(MT_SLOTS):
                for e in range(MT_BODIES * 3):
                    inp.put(MI_LOCAL_REF_POS, slot * MT_BODIES * 3 + e, self.rest.pos[e])
                for e in range(MT_BODIES * 4):
                    inp.put(MI_LOCAL_REF_ROT, slot * MT_BODIES * 4 + e, self.rest.rot[e])
                for b in range(5):
                    inp.set_flag(MI_LOCAL_BODY_MASKS, slot * 10 + b * 2, True)
                    inp.set_flag(MI_LOCAL_BODY_MASKS, slot * 10 + b * 2 + 1, b == 0)
                inp.set_flag(MI_LOCAL_POSE_MASKS, slot, True)
                inp.put(MI_LOCAL_TIMES, slot, Float32(slot + 1) * 0.1)
                inp.set_flag(MI_STYLE_FRAME_MASKS, slot, True)
                for c in range(11):
                    inp.set_flag(MI_STYLE_COND_MASKS, slot * 11 + c, True)
            for e in range(225):
                inp.put(MI_STYLE_POSE, e, self.rest.style[e])
            for e in range(150):
                inp.put(MI_STYLE_VEL, e, 0.0)
            inp.put(MI_AVG_VEL, 0, 0.0)
            inp.put(MI_AVG_VEL, 1, 0.0)
            inp.set_flag(MI_VEL_CMD_MASK, 0, True)
