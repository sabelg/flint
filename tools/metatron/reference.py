"""MuJoCo + onnxruntime reference host for NVIDIA Metatron-v1 (`demo/model.onnx`), dumping the
policy inputs/outputs and the MuJoCo trajectory for `tests/test_flint_v5_metatron.mojo` and
`examples/metatron` parity / sim-fidelity checks.

usage: python tools/metatron/reference.py G1_MJCF MODEL_ONNX OUT_DIR [--script idle|walk|turn]
           [--steps N] [--open-loop]

  G1_MJCF     ProtoMotions `protomotions/data/assets/mjcf/g1_holo_compat.xml` (Unitree BSD-3-Clause)
  MODEL_ONNX  Hugging Face `nvidia/Metatron-v1` `demo/model.onnx` (NVIDIA Open Model License)
  --script    idle: stand still; walk: idle 1 s, then steer forward at 0.5 m/s;
              turn: idle 1 s, then steer with a 90-degree heading change and 0.3 m/s forward
  --steps     number of 50 Hz policy steps (default 250 = 5 s)

OUT_DIR receives `<name>.f32` (raw little-endian float32) plus `manifest.txt` ("name dim0 ..."):
  inputs   [N, 4002]   the 39 policy inputs of each step, flattened in manifest order (bools 0/1)
  outputs  [N, 116]    action, stiffness, damping, raw_action
  qpos     [N+1, 36]   MuJoCo qpos before step k (pelvis xyz, pelvis quat wxyz, 29 joints)
  qvel     [N+1, 35]   MuJoCo qvel (pelvis linear world, pelvis angular local, 29 joints)
  xpos     [N+1, 33, 3], xquat [N+1, 33, 4] (xyzw)   body poses in the policy's body order
  sub_qpos [20*S+1, 36]  1 kHz qpos of the first S=10 policy steps (substep-level comparison)
  rest_pos [33,3], rest_rot [33,4], rest_style [225]   the calibrated idle goal (C++ make_rest_pose)

The host follows ProtoMotions' native Metatron v1 runtime (Apache-2.0,
deployment/native/src/metatron_v1_{contract,rest,task}.cpp, examples/metatron_goals.py,
tools/metatron_v1_mujoco_model.py): 1 kHz MuJoCo, 20 substeps per command, implicit position
actuators whose kp/kv are the policy's stiffness/damping, torque limits from the G1 profile,
history committed from the inverse affine map of the applied PD target. Recovery and teleop are
not modelled. MuJoCo is a reference tool only; Flint never links it.
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

args = sys.argv[1:]
if len(args) < 3:
    print(__doc__)
    sys.exit(1)
XML, ONNX, OUT = (os.path.abspath(a) for a in args[:3])
opts = {"script": "idle", "steps": "250"}
i = 3
while i < len(args):
    a = args[i]
    if a == "--open-loop":
        opts["open-loop"] = "1"
        i += 1
        continue
    opts[a.lstrip("-")] = args[i + 1]
    i += 2

import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402

INPUTS = [
    ("current_root_pos", (3,), False), ("current_root_rot", (4,), False),
    ("masked_mimic_anchor_rotation_mask", (5, 2), True), ("masked_mimic_anchor_translation_mask", (5, 2), True),
    ("masked_mimic_average_velocity", (2,), False), ("masked_mimic_local_ref_pos", (5, 33, 3), False),
    ("masked_mimic_local_ref_rot", (5, 33, 4), False), ("masked_mimic_local_target_bodies_masks", (50,), True),
    ("masked_mimic_local_target_poses_masks", (5,), True), ("masked_mimic_local_time_offsets", (5,), False),
    ("masked_mimic_ref_pos", (5, 33, 3), False), ("masked_mimic_ref_rot", (5, 33, 4), False),
    ("masked_mimic_style_condition_masks", (5, 11), True), ("masked_mimic_style_frame_masks", (5,), True),
    ("masked_mimic_style_pose_samples", (225,), False), ("masked_mimic_style_velocity_samples", (150,), False),
    ("masked_mimic_target_bodies_masks", (50,), True), ("masked_mimic_target_poses_masks", (5,), True),
    ("masked_mimic_time_offsets", (5,), False), ("masked_mimic_velocity_command_mask", (1,), True),
    ("noisy_anchor_pos", (3,), False), ("noisy_anchor_rot", (4,), False), ("noisy_dof_pos", (29,), False),
    ("noisy_dof_vel", (29,), False), ("noisy_root_local_ang_vel", (3,), False),
    ("noisy_historical_anchor_rot", (8, 4), False), ("noisy_historical_dof_pos", (8, 29), False),
    ("noisy_historical_dof_vel", (8, 29), False), ("noisy_historical_raw_actions", (8, 29), False),
    ("noisy_historical_root_local_ang_vel", (8, 3), False), ("raw_action_history", (9, 29), False),
    ("scene_boxes_centers", (1, 2, 3), False), ("scene_boxes_extents", (1, 2, 3), False),
    ("scene_boxes_object_pos", (1, 3), False), ("scene_boxes_object_rot", (1, 4), False),
    ("scene_boxes_position_offset", (3,), False), ("scene_boxes_valid", (1, 2), True),
    ("scene_boxes_visible", (1, 1), True), ("scene_boxes_yaw_offset", (), False),
]
IDX = {n: k for k, (n, _, _) in enumerate(INPUTS)}
JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
DEFAULT_DOF = np.array([-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0, 0, 0, 0,
                        0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0], np.float32)
ACTION_SCALE = np.array([0.5475465059280396, 0.3506614565849304, 0.5475465059280396, 0.3506614565849304,
                         0.4385773241519928, 0.4385773241519928, 0.5475465059280396, 0.3506614565849304,
                         0.5475465059280396, 0.3506614565849304, 0.4385773241519928, 0.4385773241519928,
                         0.5475465059280396, 0.4385773241519928, 0.4385773241519928, 0.4385773241519928,
                         0.4385773241519928, 0.4385773241519928, 0.4385773241519928, 0.4385773241519928,
                         0.07450086623430252, 0.07450086623430252, 0.4385773241519928, 0.4385773241519928,
                         0.4385773241519928, 0.4385773241519928, 0.4385773241519928, 0.07450086623430252,
                         0.07450086623430252], np.float32)
PD = {  # kp, kd, effort, armature (ProtoMotions tools/metatron_v1_mujoco_model.py)
    "hip_pitch": (40.17923847137318, 2.5578897650279457, 88.0, 0.010177520),
    "hip_yaw": (40.17923847137318, 2.5578897650279457, 88.0, 0.010177520),
    "hip_roll": (99.09842777666113, 6.3088018534966395, 139.0, 0.025101925),
    "knee": (99.09842777666113, 6.3088018534966395, 139.0, 0.025101925),
    "ankle": (28.50124619574858, 1.814445686584846, 50.0, 0.007219450),
    "waist_yaw": (40.17923847137318, 2.5578897650279457, 88.0, 0.010177520),
    "waist_roll": (28.50124619574858, 1.814445686584846, 50.0, 0.007219450),
    "waist_pitch": (28.50124619574858, 1.814445686584846, 50.0, 0.007219450),
    "shoulder": (14.25062309787429, 0.907222843292423, 25.0, 0.003609725),
    "elbow": (14.25062309787429, 0.907222843292423, 25.0, 0.003609725),
    "wrist_roll": (14.25062309787429, 0.907222843292423, 25.0, 0.003609725),
    "wrist_pitch": (16.77832748089279, 1.06814150219, 5.0, 0.004250000),
    "wrist_yaw": (16.77832748089279, 1.06814150219, 5.0, 0.004250000),
}
HORIZONS = [0.1, 0.2, 0.3, 0.4, 0.5]
TORSO, RFOOT, LFOOT, LHAND, RHAND = 16, 13, 7, 24, 32
CONDITIONABLE = [16, 13, 7, 24, 32]
PI = math.pi


def wrap(a):
    return math.remainder(a, 2 * PI)


def yaw_quat(y):
    return np.array([0, 0, math.sin(y / 2), math.cos(y / 2)], np.float32)


def yaw_of(q):
    return math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]))


def rot_yaw(y, v):
    c, s = math.cos(y), math.sin(y)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], np.float32)


def qmul(a, b):
    return np.array([a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1], a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
                     a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3], a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]], np.float32)


def qrot(q, v):
    u = q[:3]
    return (q[3] * q[3] - u @ u) * v + 2 * (u @ v) * u + 2 * q[3] * np.cross(u, v)


def build_model(path):
    """The native preparation (implicit PD position actuators + floor) without meshes."""
    tree = ET.parse(path)
    root = tree.getroot()
    for asset in root.findall("asset"):
        for m in asset.findall("mesh"):
            asset.remove(m)
    comp = root.find("compiler")
    if comp is not None:
        comp.attrib.pop("meshdir", None)
    for parent in root.iter():
        for g in list(parent.findall("geom")):
            if g.get("type") == "mesh":
                parent.remove(g)
    wb = root.find("worldbody")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="20 20 .1", pos="0 0 0", contype="1", conaffinity="1")
    act = root.find("actuator")
    motors = list(act)
    act.clear()
    for m in motors:
        j = m.get("joint")
        jn = root.find(f".//joint[@name='{j}']")
        key = next(k for k in PD if k in j)
        kp, kd, eff, arm = PD[key]
        jn.set("stiffness", "0")
        jn.set("damping", "0")
        jn.set("armature", str(arm))
        ET.SubElement(act, "position", name=j, joint=j, kp=str(kp), kv=str(kd), ctrllimited="false",
                      forcerange=f"{-eff} {eff}", forcelimited="true")
    s = root.find("sensor")
    if s is not None:
        root.remove(s)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = 0.001
    for g in range(model.ngeom):  # ProtoMotions _set_robot_geom_friction (default_robot_friction 1.0)
        if model.geom_bodyid[g] != 0:
            model.geom_friction[g, 0] = 1.0
    return model


def main():
    model = build_model(XML)
    data = mujoco.MjData(model)
    jq = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS]
    jv = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS]
    act_of = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS]
    bodies = [model.body(b).name for b in range(1, model.nbody)]
    assert len(bodies) == 33, bodies
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def fk(dof, pelvis_pos, pelvis_wxyz):
        d = mujoco.MjData(model)
        d.qpos[:3] = pelvis_pos
        d.qpos[3:7] = pelvis_wxyz
        d.qpos[jq] = dof
        mujoco.mj_kinematics(model, d)
        return d.xpos[1:].astype(np.float32).copy(), d.xquat[1:][:, [1, 2, 3, 0]].astype(np.float32).copy()

    # ---- calibrated rest pose (metatron_v1_rest.cpp): pelvis at the origin, floor-relative heights
    rest_dof = np.zeros(29, np.float32)
    rest_dof[[0, 6]] = -0.312
    rest_dof[[3, 9]] = 0.669
    rest_dof[[4, 10]] = -0.363
    rest_dof[16] = 0.2
    rest_dof[23] = -0.2
    rest_dof[[18, 25]] = 1.4
    rpos, rrot = fk(rest_dof, [0, 0, 0], [1, 0, 0, 0])
    floor = rpos[:, 2].min()
    rpos[:, 2] -= floor
    upright = float(rpos[TORSO, 2])
    style = np.zeros(225, np.float32)
    for f in range(5):
        for s, b in enumerate(CONDITIONABLE):
            base = (f * 5 + s) * 9
            style[base:base + 3] = [rpos[b, 0] - rpos[TORSO, 0], rpos[b, 1] - rpos[TORSO, 1], rpos[b, 2]]
            style[base + 3:base + 6] = qrot(rrot[b], np.array([1, 0, 0], np.float32))
            style[base + 6:base + 9] = qrot(rrot[b], np.array([0, 0, 1], np.float32))
    rest_pos5 = np.broadcast_to(rpos, (5, 33, 3)).copy()
    rest_rot5 = np.broadcast_to(rrot, (5, 33, 4)).copy()

    # ---- initial state: default standing pose, lowest collision point on the floor
    mujoco.mj_resetData(model, data)
    data.qpos[jq] = DEFAULT_DOF
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    low = 1e9
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or model.geom_contype[g] == 0:
            continue
        low = min(low, data.geom_xpos[g, 2] - (model.geom_size[g, 0] if model.geom_type[g] in (2, 3) else 0))
        if model.geom_type[g] == 3:  # capsule: endpoints along local z
            ax = data.geom_xmat[g].reshape(3, 3)[:, 2] * model.geom_size[g, 1]
            low = min(low, data.geom_xpos[g, 2] - abs(ax[2]) - model.geom_size[g, 0])
    data.qpos[2] = 1.0 - low + 0.0005
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # ---- persistent policy inputs (Inputs::reset)
    vals = [np.zeros(shape, bool if b else np.float32) for (_, shape, b) in INPUTS]
    vals[IDX["masked_mimic_local_ref_rot"]][..., 3] = 1
    vals[IDX["masked_mimic_ref_rot"]][..., 3] = 1
    vals[IDX["scene_boxes_object_rot"]][0, 3] = 1
    hist_states = None
    hist_actions = np.zeros((9, 29), np.float32)

    def reading():
        q, v = data.qpos, data.qvel
        tq = data.xquat[torso_id]
        return dict(dof_pos=q[jq].astype(np.float32), dof_vel=v[jv].astype(np.float32),
                    root_local_ang_vel=v[3:6].astype(np.float32),
                    anchor_rot=np.array([tq[1], tq[2], tq[3], tq[0]], np.float32),
                    anchor_pos=data.xpos[torso_id].astype(np.float32),
                    root_pos=q[:3].astype(np.float32), root_rot=np.array([q[4], q[5], q[6], q[3]], np.float32))

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    task = dict(mode="idle", heading=0.0, vel=np.zeros(2, np.float32), stationary=True, hold=0.0, hold_ready=False,
                gait=0.0, support_ready=False, support_feet=np.zeros(6, np.float32), support_anchor=None,
                support_heading=0.0, torso_sent=float("nan"))

    def build(r):
        # MetatronTask::build for idle / steering (no recovery, no teleop, no tracking)
        mpos = data.xpos[1:].astype(np.float32)
        mrot = data.xquat[1:][:, [1, 2, 3, 0]].astype(np.float32)
        heading = yaw_of(r["anchor_rot"])
        for slot in range(5):
            vals[IDX["masked_mimic_ref_pos"]][slot] = mpos
            vals[IDX["masked_mimic_ref_rot"]][slot] = mrot
            vals[IDX["masked_mimic_time_offsets"]][slot] = HORIZONS[slot]
            vals[IDX["masked_mimic_target_poses_masks"]][slot] = True
        speed = float(np.hypot(*task["vel"]))
        task["stationary"] = speed < 0.07 if task["stationary"] else speed <= 0.03
        idle = task["mode"] == "idle"
        standing = idle or (speed <= 0.03 and abs(wrap(task["heading"] - heading)) <= PI / 9)
        tbm = vals[IDX["masked_mimic_target_bodies_masks"]]
        arm = vals[IDX["masked_mimic_anchor_rotation_mask"]]
        atm = vals[IDX["masked_mimic_anchor_translation_mask"]]
        if idle:
            if not task["hold_ready"]:
                task["hold"] = heading
                task["hold_ready"] = True
            else:
                alpha = 1 - math.exp(-0.02 / 0.5)
                task["hold"] = wrap(task["hold"] + alpha * wrap(heading - task["hold"]))
            vals[IDX["masked_mimic_ref_rot"]][4, TORSO] = yaw_quat(task["hold"])
            tbm[4 * 10 + 1] = True
            arm[4, 1] = True
        else:
            steering(r, mpos, heading, tbm, arm, atm)
        apply_goals(standing)

    def steering(r, mpos, heading, tbm, arm, atm):
        vel = task["vel"]
        speed = float(np.hypot(*vel))
        st = task["stationary"]
        if st and task["support_ready"]:
            for side, b in enumerate((RFOOT, LFOOT)):
                f = mpos[b]
                if math.hypot(f[0] - task["support_feet"][side * 3], f[1] - task["support_feet"][side * 3 + 1]) > 0.12:
                    task["support_ready"] = False
                    break
        if st and not task["support_ready"]:
            task["support_feet"] = np.concatenate([mpos[RFOOT], mpos[LFOOT]]).astype(np.float32)
            task["support_anchor"] = mpos[TORSO].copy()
            task["support_ready"] = True
            task["support_heading"] = heading
        if not st:
            task["support_ready"] = False
            task["torso_sent"] = float("nan")
        sf = task["support_feet"]
        dt_ = rest_pos5[0, TORSO]
        rd, ld = rest_pos5[0, RFOOT], rest_pos5[0, LFOOT]
        foot_center = 0.5 * (rd + ld) - dt_
        default_floor = min(rd[2], ld[2])
        half_width = max(0.07, 0.5 * abs(ld[1] - rd[1]))
        cadence = min(2.0, 0.83 + 0.93 * speed)
        swing = 0.0 if st else 0.13 + 0.065 * speed
        move_heading = heading if st else math.atan2(vel[1], vel[0])
        ground = mpos[:, 2].min()
        rp, rr = vals[IDX["masked_mimic_ref_pos"]], vals[IDX["masked_mimic_ref_rot"]]
        for slot in range(5):
            t = HORIZONS[slot]
            yaw = wrap(heading + float(np.clip(wrap(task["heading"] - heading), -PI * t, PI * t)))
            yq = yaw_quat(yaw)
            if st:
                torso = task["support_anchor"].copy()
                torso[0] = 0.5 * (sf[0] + sf[3])
                torso[1] = 0.5 * (sf[1] + sf[4])
                fl = min(sf[2], sf[5])
                th = upright - default_floor
                if slot == 0:
                    if not math.isnan(task["torso_sent"]):
                        b = 0.4 * 0.02
                        th = min(max(th, task["torso_sent"] - b), task["torso_sent"] + b)
                    task["torso_sent"] = th
                else:
                    th = task["torso_sent"]
                torso[2] = fl + th
            else:
                torso = np.array([r["anchor_pos"][0] + vel[0] * t, r["anchor_pos"][1] + vel[1] * t,
                                  ground + upright], np.float32)
            rp[slot, TORSO] = torso
            rr[slot, TORSO] = yq
            turn = st and abs(wrap(yaw - task["support_heading"])) > PI / 6
            feet = []
            for side, b in enumerate((RFOOT, LFOOT)):
                off = rest_pos5[0, b] - dt_
                sign = -1.0 if side == 0 else 1.0
                if st and not turn:
                    foot = sf[side * 3:side * 3 + 3].copy()
                elif st:
                    rel = rot_yaw(yaw, off - foot_center)
                    foot = np.array([torso[0] + rel[0], torso[1] + rel[1], sf[side * 3 + 2]], np.float32)
                else:
                    off = off.copy()
                    rel = wrap(move_heading - yaw)
                    off[1] = sign * (half_width + 0.12 * abs(math.sin(rel)))
                    ph = task["gait"] + 2 * PI * cadence * t + (PI if side == 1 else 0)
                    lat = min(1.0, 0.12 / swing) * abs(math.cos(rel))
                    off[0] += swing * math.cos(ph) * 0.75 * math.cos(rel)
                    off[1] += swing * math.cos(ph) * lat * math.sin(rel)
                    off[2] += 0.08 * max(0.0, -math.sin(ph))
                    foot = torso + rot_yaw(yaw, off)
                feet.append(foot)
                rp[slot, b] = foot
                rr[slot, b] = qmul(yq, rest_rot5[0, b])
            sup = np.array([0.5 * (feet[0][0] + feet[1][0]), 0.5 * (feet[0][1] + feet[1][1]), min(feet[0][2], feet[1][2])],
                           np.float32)
            for b in (LHAND, RHAND):
                rel = rot_yaw(yaw, rest_pos5[0, b] - dt_ - foot_center)
                rp[slot, b] = sup + rel
            m = slot * 10
            tbm[m] = True
            tbm[m + 1] = True
            tbm[m + 4] = not st or turn
            tbm[m + 2] = tbm[m + 4]
            tbm[m + 6] = st or slot == 4
            tbm[m + 8] = tbm[m + 6]
            atm[slot, 0] = not st
            atm[slot, 1] = True
            arm[slot, 0] = False
            arm[slot, 1] = True
        vals[IDX["masked_mimic_average_velocity"]][:] = vel
        vals[IDX["masked_mimic_velocity_command_mask"]][0] = True
        task["gait"] = 0.0 if st else math.fmod(task["gait"] + 2 * PI * cadence * 0.02, 2 * PI)

    def apply_goals(standing):
        # examples/metatron_goals.py MetatronGoals.apply (idle / steering)
        g = vals[IDX["masked_mimic_target_bodies_masks"]].reshape(5, 5, 2)
        lm = vals[IDX["masked_mimic_local_target_bodies_masks"]].reshape(5, 5, 2)
        atm = vals[IDX["masked_mimic_anchor_translation_mask"]]
        arm = vals[IDX["masked_mimic_anchor_rotation_mask"]]
        lm[:] = g
        lm[:, 0, 0] &= atm[:, 1]
        lm[:, 0, 1] &= arm[:, 0]
        vals[IDX["masked_mimic_local_ref_pos"]][:] = vals[IDX["masked_mimic_ref_pos"]]
        vals[IDX["masked_mimic_local_ref_rot"]][:] = vals[IDX["masked_mimic_ref_rot"]]
        vals[IDX["masked_mimic_local_time_offsets"]][:] = vals[IDX["masked_mimic_time_offsets"]]
        vals[IDX["masked_mimic_local_target_poses_masks"]][:] = vals[IDX["masked_mimic_target_poses_masks"]] & lm.any(axis=(1, 2))
        heading = g[:, 0, 1] & arm[:, 1]
        g.fill(False)
        g[:, 0, 1] = heading
        atm.fill(False)
        arm[:, 0] = False
        vals[IDX["masked_mimic_target_poses_masks"]] &= heading
        if standing:
            vals[IDX["masked_mimic_local_ref_pos"]][:] = rest_pos5
            vals[IDX["masked_mimic_local_ref_rot"]][:] = rest_rot5
            lm.fill(False)
            lm[:, :, 0] = True
            lm[:, 0, 1] = True
            vals[IDX["masked_mimic_local_target_poses_masks"]].fill(True)
            vals[IDX["masked_mimic_local_time_offsets"]][:] = HORIZONS
            vals[IDX["masked_mimic_style_pose_samples"]][:] = style
            vals[IDX["masked_mimic_style_velocity_samples"]].fill(0)
            vals[IDX["masked_mimic_style_condition_masks"]].fill(True)
            vals[IDX["masked_mimic_style_frame_masks"]].fill(True)
            vals[IDX["masked_mimic_average_velocity"]].fill(0)
            vals[IDX["masked_mimic_velocity_command_mask"]].fill(True)

    def script(k):
        s = opts["script"]
        t = k * 0.02
        if s == "idle" or t < 1.0:
            return "idle", 0.0, (0.0, 0.0)
        if s == "walk":
            return "steering", 0.0, (0.5, 0.0)
        if s == "turn":
            return "steering", PI / 2, (0.3 * math.cos(min(PI / 2, (t - 1.0) * PI / 2)),
                                        0.3 * math.sin(min(PI / 2, (t - 1.0) * PI / 2)))
        raise SystemExit("unknown script " + s)

    N = int(opts["steps"])
    rec = {k: [] for k in ("inputs", "outputs", "qpos", "qvel", "xpos", "xquat", "sub_qpos")}
    rec["sub_qpos"].append(data.qpos.astype(np.float32).copy())
    out_names = ["action", "stiffness", "damping", "raw_action"]
    for k in range(N):
        mode, heading, vel = script(k)
        if mode != task["mode"]:
            if mode == "steering":
                task["support_ready"] = False
            task["mode"] = mode
            task["hold_ready"] = False
        task["heading"] = wrap(heading)
        task["vel"] = np.array(vel, np.float32)
        r = reading()
        if hist_states is None:
            hist_states = [dict(r) for _ in range(8)]
        # Inputs::prepare
        vals[IDX["current_root_pos"]][:] = r["root_pos"]
        vals[IDX["current_root_rot"]][:] = r["root_rot"]
        vals[IDX["noisy_anchor_pos"]][:] = r["anchor_pos"]
        vals[IDX["noisy_anchor_rot"]][:] = r["anchor_rot"]
        vals[IDX["noisy_dof_pos"]][:] = r["dof_pos"]
        vals[IDX["noisy_dof_vel"]][:] = r["dof_vel"]
        vals[IDX["noisy_root_local_ang_vel"]][:] = r["root_local_ang_vel"]
        for row in range(8):
            vals[IDX["noisy_historical_anchor_rot"]][row] = hist_states[row]["anchor_rot"]
            vals[IDX["noisy_historical_dof_pos"]][row] = hist_states[row]["dof_pos"]
            vals[IDX["noisy_historical_dof_vel"]][row] = hist_states[row]["dof_vel"]
            vals[IDX["noisy_historical_root_local_ang_vel"]][row] = hist_states[row]["root_local_ang_vel"]
            vals[IDX["noisy_historical_raw_actions"]][row] = hist_actions[row + 1]
        vals[IDX["raw_action_history"]][:] = hist_actions
        build(r)
        feeds = {n: vals[IDX[n]][None] if shape != () else vals[IDX[n]].reshape(1) for (n, shape, _) in INPUTS}
        outs = sess.run(out_names, feeds)
        action, kp, kd, raw = (o[0] for o in outs)
        rec["inputs"].append(np.concatenate([vals[j].astype(np.float32).ravel() for j in range(len(INPUTS))]))
        rec["outputs"].append(np.concatenate([action, kp, kd, raw]).astype(np.float32))
        rec["qpos"].append(data.qpos.astype(np.float32).copy())
        rec["qvel"].append(data.qvel.astype(np.float32).copy())
        rec["xpos"].append(data.xpos[1:].astype(np.float32).copy())
        rec["xquat"].append(data.xquat[1:][:, [1, 2, 3, 0]].astype(np.float32).copy())
        # apply the command: kp/kv per actuator, ctrl = PD target
        for j in range(29):
            a = act_of[j]
            model.actuator_gainprm[a, 0] = kp[j]
            model.actuator_biasprm[a, 1] = -kp[j]
            model.actuator_biasprm[a, 2] = -kd[j]
            data.ctrl[a] = action[j]
        for _ in range(20):
            mujoco.mj_step(model, data)
            if k < 10:
                rec["sub_qpos"].append(data.qpos.astype(np.float32).copy())
        # Inputs::commit with the inverse affine map of the applied target
        hist_states.insert(0, r)
        del hist_states[8:]
        hist_actions[1:] = hist_actions[:-1].copy()
        hist_actions[0] = (action - DEFAULT_DOF) / ACTION_SCALE
    rec["qpos"].append(data.qpos.astype(np.float32).copy())
    rec["qvel"].append(data.qvel.astype(np.float32).copy())
    rec["xpos"].append(data.xpos[1:].astype(np.float32).copy())
    rec["xquat"].append(data.xquat[1:][:, [1, 2, 3, 0]].astype(np.float32).copy())

    os.makedirs(OUT, exist_ok=True)
    lines = []

    def dump(name, arr):
        arr = np.ascontiguousarray(arr, np.float32)
        arr.tofile(os.path.join(OUT, name + ".f32"))
        lines.append(name + " " + " ".join(str(d) for d in arr.shape))

    for k, v in rec.items():
        dump(k, np.stack(v))
    dump("rest_pos", rpos)
    dump("rest_rot", rrot)
    dump("rest_style", style)
    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    q = np.stack(rec["qpos"])
    print(f"{opts['script']}: {N} steps, pelvis z {q[0, 2]:.3f} -> {q[-1, 2]:.3f}, xy drift "
          f"{np.hypot(q[-1, 0] - q[0, 0], q[-1, 1] - q[0, 1]):.3f} m, upright torso {upright:.4f} m")


main()
