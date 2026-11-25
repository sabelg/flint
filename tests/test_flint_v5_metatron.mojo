"""Flint v5 Metatron suite (`test_flint_v5_metatron.mojo`).

Verifies, without model files:
  1. `XmlDoc`: elements, attributes, self-closing tags, comments, the prolog and entities.
  2. `MjcfModel` on an inline robot: default classes and `childclass`, degree angles, euler /
     fromto / fullinertia, free + hinge joints, qpos / qvel layout, actuators, contact pairs,
     excludes and fixed tendons.
  3. Kinematics: forward kinematics against hand values, `body_velocities` against finite
     differences, the symmetric eigen solver, the z-up -> y-up frame maps.
  4. `MjcfArticulation`: welded jointless bodies (composite mass / COM), state round trip, the
     revolute PD drive tracking its target with an armature rotor, the torque cap and limits.
  5. The Metatron contract: 39 inputs / 4002 floats, the G1 tables, the affine action map and
     the observation history of `MetatronInputs`.
  6. With `METATRON_DIR` (from `tools/metatron/fetch.sh`): the G1 MJCF layout and rest pose, and
     the policy standing the G1 in Flint physics for 4 s. With `METATRON_REF` too (dumps of
     `tools/metatron/reference.py`): the host rebuilds MuJoCo's inputs, the network matches
     onnxruntime, forward kinematics matches MuJoCo, and an open-loop replay of MuJoCo's commands
     stays near MuJoCo's trajectory for 1 s. `METATRON_CPU=1` keeps the network on the CPU.
"""
from std.math import abs, sqrt, sin, cos
from std.testing import assert_true, assert_equal
from flint.core.math import Vec3, Quat
from flint.physics.world import PhysicsWorld
from flint.physics.mjcf import (
    XmlDoc, MjcfModel, MjcfArticulation, Mat3, symmetric_eigen, mj_rotate, mj_to_flint, flint_to_mj,
    mj_to_flint_quat, flint_to_mj_quat, MJ_JNT_FREE, MJ_JNT_HINGE, MJ_GEOM_CAPSULE, MJ_GEOM_BOX,
    MJ_GEOM_SPHERE, MJ_ACT_POSITION, MJ_ACT_MOTOR,
)
from flint.ml.metatron import (
    metatron_input_names, metatron_input_offsets, metatron_input_shape, metatron_input_is_bool,
    g1_joint_names, g1_body_names, g1_action_offset, g1_action_scale, G1Gains, metatron_raw_from_action,
    MetatronInputs, MetatronReading, MetatronTask, MetatronPolicy, MetatronHost, MetatronOutputs, G1Sim,
    g1_check_model, g1_rest_from_model, g1_default_qpos, metatron_reading_mj, MT_INPUTS, MT_DOF,
    MT_BODIES, MT_PI, MI_H_DOF_POS, MI_RAW_ACTION_HISTORY, MI_H_RAW_ACTIONS, MI_DOF_POS, MI_REF_ROT,
    MI_LOCAL_REF_ROT,
)
from flint.ml.tensor import f32_from_bits
from flint.platform.os import env_value, read_bytes


comptime ROBOT = """<?xml version="1.0"?>
<!-- a two-link leg on a floating base -->
<mujoco model="leg">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint armature="0.01" damping="0.1"/>
    <default class="leg">
      <joint range="-90 90" frictionloss="0.2"/>
      <geom type="capsule" size="0.02"/>
    </default>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="2" diaginertia="0.1 0.2 0.3"/>
      <geom name="torso" type="box" size="0.1 0.1 0.1"/>
      <body name="thigh" pos="0 0 -0.1" euler="0 0 90" childclass="leg">
        <joint name="hip" axis="0 1 0"/>
        <inertial pos="0 0 -0.2" mass="1" fullinertia="0.02 0.03 0.01 0.001 0 0"/>
        <geom name="thigh_g" fromto="0 0 0 0 0 -0.4"/>
        <body name="shin" pos="0 0 -0.4">
          <joint name="knee" axis="0 1 0" range="0 150"/>
          <inertial pos="0 0 -0.2" mass="0.5" diaginertia="0.01 0.01 0.001"/>
          <geom name="shin_g" fromto="0 0 0 0 0 -0.4" class="leg"/>
          <body name="toe" pos="0 0 -0.4">
            <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
            <geom type="sphere" size="0.03"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hip_a" joint="hip" kp="50" kv="2" forcerange="-20 20"/>
    <motor name="knee_m" joint="knee" gear="1" ctrlrange="-5 5"/>
  </actuator>
  <contact>
    <pair geom1="shin_g" geom2="floor" friction="0.7 0.7"/>
    <exclude body1="base" body2="thigh"/>
  </contact>
  <tendon>
    <fixed name="t" range="-1 1"><joint joint="hip" coef="1"/><joint joint="knee" coef="-1"/></fixed>
  </tendon>
</mujoco>
"""


def near(a: Float64, b: Float64, tol: Float64) -> Bool:
    return abs(a - b) <= tol


def vnear(a: Vec3, b: Vec3, tol: Float64) -> Bool:
    return near(Float64(a.x), Float64(b.x), tol) and near(Float64(a.y), Float64(b.y), tol) and near(Float64(a.z), Float64(b.z), tol)


def read_f32(path: String) raises -> List[Float32]:
    var b = read_bytes(path)
    var n = len(b) // 4
    var out = List[Float32](capacity=n)
    for i in range(n):
        var u = UInt32(b[4 * i]) | (UInt32(b[4 * i + 1]) << 8) | (UInt32(b[4 * i + 2]) << 16) | (UInt32(b[4 * i + 3]) << 24)
        out.append(f32_from_bits(u))
    return out^


# ----------------------------------------------------------------------------------------- 1

def test_xml() raises:
    var d = XmlDoc.parse("<?xml version='1.0'?><!DOCTYPE x><a k=\"1 &amp; 2\"><!-- <b/> --><b x='&lt;'/><c>text</c><b/></a>")
    assert_equal(d.count(), 4)
    assert_equal(d.tag[0], "a")
    assert_equal(d.attr(0, "k"), "1 & 2")
    assert_equal(len(d.children(0, "b")), 2)
    assert_equal(d.attr(d.children(0, "b")[0], "x"), "<")
    assert_equal(d.attr(0, "missing", "dflt"), "dflt")
    assert_equal(d.parent[3], 0)
    print("  [PASS] XmlDoc")


# ----------------------------------------------------------------------------------------- 2

def test_model() raises:
    var m = MjcfModel.parse(ROBOT)
    assert_equal(m.name, "leg")
    assert_equal(len(m.bodies), 5)
    assert_equal(len(m.joints), 3)
    assert_equal(m.nq, 9)
    assert_equal(m.nv, 8)
    assert_equal(len(m.geoms), 5)
    assert_true(near(m.timestep, 0.002, 1e-12))
    assert_equal(m.joints[0].kind, MJ_JNT_FREE)
    var hip = m.joints[m.joint_index("hip")].copy()
    var knee = m.joints[m.joint_index("knee")].copy()
    assert_equal(hip.kind, MJ_JNT_HINGE)
    assert_equal(hip.qpos_adr, 7)
    assert_equal(knee.dof_adr, 7)
    assert_true(hip.limited and near(hip.upper, Float64(MT_PI) / 2, 1e-6) and near(hip.lower, -Float64(MT_PI) / 2, 1e-6))
    assert_true(near(knee.upper, 150.0 * 3.141592653589793 / 180.0, 1e-9))
    assert_true(near(hip.armature, 0.01, 1e-12) and near(hip.frictionloss, 0.2, 1e-12) and near(hip.damping, 0.1, 1e-12))
    ref shin_g = m.geoms[m.geom_index("shin_g")]
    assert_equal(shin_g.kind, MJ_GEOM_CAPSULE)
    assert_true(near(Float64(shin_g.size.x), 0.02, 1e-7) and near(Float64(shin_g.size.y), 0.2, 1e-6))
    assert_true(vnear(shin_g.pos, Vec3(0, 0, -0.2), 1e-6))
    assert_equal(m.geoms[m.geom_index("torso")].kind, MJ_GEOM_BOX)
    var thigh = m.bodies[m.body_index("thigh")].copy()
    assert_true(near(thigh.mass, 1.0, 1e-12))
    var it = thigh.inertia
    assert_true(near(Float64(it.x + it.y + it.z), 0.06, 1e-6))   # trace kept by the diagonalisation
    assert_equal(len(m.actuators), 2)
    assert_equal(m.actuators[0].kind, MJ_ACT_POSITION)
    assert_true(near(m.actuators[0].kp, 50, 1e-9) and near(m.actuators[0].kv, 2, 1e-9))
    assert_true(m.actuators[0].force_limited and near(m.actuators[0].force_upper, 20, 1e-9))
    assert_equal(m.actuators[1].kind, MJ_ACT_MOTOR)
    assert_true(m.actuators[1].ctrl_limited and not m.actuators[1].force_limited)
    assert_equal(len(m.pair_geom1), 1)
    assert_equal(m.pair_geom1[0], m.geom_index("shin_g"))
    assert_equal(m.pair_geom2[0], m.geom_index("floor"))
    assert_true(near(m.pair_friction[0], 0.7, 1e-12))
    assert_equal(m.exclude_body2[0], m.body_index("thigh"))
    assert_equal(len(m.tendons), 1)
    assert_true(m.tendons[0].limited and near(m.tendons[0].coefs[1], -1, 1e-12))
    assert_true(near(m.total_mass(), 3.6, 1e-9))
    var q0 = m.qpos0()
    assert_true(near(q0[2], 1.0, 1e-7) and near(q0[3], 1.0, 1e-7) and near(q0[8], 0.0, 0.0))
    print("  [PASS] MjcfModel: defaults, joints, geoms, actuators, contact, tendons")


# ----------------------------------------------------------------------------------------- 3

def test_kinematics() raises:
    var m = MjcfModel.parse(ROBOT)
    var q = m.qpos0()
    q[7] = 3.141592653589793 / 2    # hip
    var fk = m.forward_kinematics(q)
    var pos = fk[0].copy()
    var rot = fk[1].copy()
    # the thigh is yawed 90 degrees, so its y hinge axis is world -x; turning it +90 swings the
    # knee from straight down to world -y
    assert_true(vnear(pos[m.body_index("thigh")], Vec3(0, 0, 0.9), 1e-6))
    assert_true(vnear(pos[m.body_index("shin")], Vec3(0, -0.4, 0.9), 1e-5))
    assert_true(vnear(mj_rotate(rot[m.body_index("thigh")], Vec3(0, 1, 0)), Vec3(-1, 0, 0), 1e-5))
    # velocities against finite differences (free-joint linear velocity + both hinges)
    var q1 = m.qpos0()
    q1[7] = 0.3
    q1[8] = 0.7
    var v = List[Float64](length=m.nv, fill=0.0)
    v[0] = 0.2
    v[2] = -0.1
    v[6] = 1.5
    v[7] = -0.8
    var vel = m.body_velocities(q1, v)
    var lin = vel[0].copy()
    var ang = vel[1].copy()
    var eps = 1.0e-4
    var qa = q1.copy()
    var qb = q1.copy()
    qa[0] -= v[0] * eps; qa[2] -= v[2] * eps; qa[7] -= v[6] * eps; qa[8] -= v[7] * eps
    qb[0] += v[0] * eps; qb[2] += v[2] * eps; qb[7] += v[6] * eps; qb[8] += v[7] * eps
    var fa = m.forward_kinematics(qa)
    var fb = m.forward_kinematics(qb)
    var pa = fa[0].copy()
    var pb = fb[0].copy()
    for b in range(1, len(m.bodies)):
        var fd = (pb[b] - pa[b]) * Float32(1.0 / (2 * eps))
        assert_true(vnear(fd, lin[b], 2e-3))
    var shin = m.body_index("shin")
    var axis = mj_rotate(m.forward_kinematics(q1)[1][shin], Vec3(0, 1, 0))
    assert_true(vnear(ang[shin], axis * Float32(1.5 - 0.8), 1e-5))
    # eigen decomposition reconstructs the tensor
    var t = Mat3()
    t.m[0] = 0.02; t.m[4] = 0.03; t.m[8] = 0.01
    t.m[1] = 0.004; t.m[3] = 0.004; t.m[2] = -0.002; t.m[6] = -0.002; t.m[5] = 0.001; t.m[7] = 0.001
    var e = symmetric_eigen(t)
    var r = Mat3.rotation(e[1])
    var back = r.mul(Mat3.diag(Float64(e[0].x), Float64(e[0].y), Float64(e[0].z))).mul(r.transposed())
    for k in range(9):
        assert_true(near(back.m[k], t.m[k], 1e-6))
    # z-up -> y-up: positions and orientations agree
    var p = Vec3(0.1, 0.2, 0.3)
    assert_true(vnear(mj_to_flint(p), Vec3(0.1, 0.3, -0.2), 1e-7))
    assert_true(vnear(flint_to_mj(mj_to_flint(p)), p, 1e-7))
    var qq = Quat(0.1, -0.3, 0.2, 0.9).normalized()
    var local = Vec3(0.3, -0.5, 0.7)
    assert_true(vnear(mj_rotate(mj_to_flint_quat(qq), local), mj_to_flint(mj_rotate(qq, local)), 1e-6))
    var qb2 = flint_to_mj_quat(mj_to_flint_quat(qq))
    assert_true(abs(Float64(qb2.x * qq.x + qb2.y * qq.y + qb2.z * qq.z + qb2.w * qq.w)) > 1 - 1e-6)
    print("  [PASS] kinematics: FK, velocities vs finite differences, eigen, frames")


# ----------------------------------------------------------------------------------------- 4

def test_articulation() raises:
    var m = MjcfModel.parse(ROBOT)
    var w = PhysicsWorld(0.0)
    w.settings.substeps = 8
    var art = MjcfArticulation.build(w, m, 0.8, 2000.0, True, True, False, False)
    assert_equal(len(art.links), 3)             # the toe is welded into the shin
    var shin = art.links[art.body_link[m.body_index("shin")]].copy()
    assert_equal(art.body_link[m.body_index("toe")], art.body_link[m.body_index("shin")])
    assert_true(near(shin.mass, 0.6, 1e-9))
    assert_true(vnear(shin.com, Vec3(0, 0, Float32(-(0.5 * 0.2 + 0.1 * 0.4) / 0.6)), 1e-6))
    # state round trip
    var q = m.qpos0()
    q[0] = 0.5; q[1] = -0.2; q[2] = 1.3
    var h = Quat(0.1, 0.2, -0.1, 0.97).normalized()
    q[3] = Float64(h.w); q[4] = Float64(h.x); q[5] = Float64(h.y); q[6] = Float64(h.z)
    q[7] = 0.4; q[8] = 1.1
    var v = List[Float64](length=m.nv, fill=0.0)
    v[0] = 0.3; v[4] = -0.6; v[6] = 2.0; v[7] = -1.0
    art.set_state(w, m, q, v)
    var q2 = art.qpos(w, m)
    var v2 = art.qvel(w, m)
    for k in range(m.nq):
        assert_true(near(q2[k], q[k], 2e-5) or (k >= 3 and k < 7 and near(-q2[k], q[k], 2e-5)))
    for k in range(m.nv):
        assert_true(near(v2[k], v[k], 2e-4))
    var bp = art.body_poses(w, m)
    var fk = m.forward_kinematics(q)
    var bpos = bp[0].copy()
    var fpos = fk[0].copy()
    for b in range(1, len(m.bodies)):
        assert_true(vnear(bpos[b], fpos[b], 1e-4))
    # PD drive tracks its target (no gravity, floating base), armature rotor and frictionloss on
    art.set_state(w, m, m.qpos0(), List[Float64](length=m.nv, fill=0.0))
    art.set_efforts([Float32(20), Float32(0)])
    art.set_command(w, [Float32(0.5), Float32(1.0)], [Float32(50), Float32(5)], [Float32(2), Float32(0.5)])
    for _ in range(2000):
        w.step(0.001)
    var qt = art.qpos(w, m)
    assert_true(near(qt[7], 0.5, 0.02))
    assert_true(near(qt[8], 1.0, 0.06))       # frictionloss 0.2 N m against kp 5 leaves ~0.04 rad
    # the hip limit holds a target beyond it
    art.set_command(w, [Float32(2.5), Float32(1.0)], [Float32(50), Float32(5)], [Float32(2), Float32(0.5)])
    for _ in range(1000):
        w.step(0.001)
    qt = art.qpos(w, m)
    assert_true(qt[7] < Float64(MT_PI) / 2 + 0.03 and qt[7] > 1.3)
    # the torque cap: 20 N m at most
    assert_true(abs(Float64(art.drive_torque(w, 0, Float32(0.001) / 8))) <= 20.0 + 1e-3)
    print("  [PASS] MjcfArticulation: welded links, state round trip, PD drive, limits, torque cap")


# ----------------------------------------------------------------------------------------- 5

def test_contract() raises:
    var names = metatron_input_names()
    assert_equal(len(names), MT_INPUTS)
    var off = metatron_input_offsets()
    assert_equal(off[MT_INPUTS], 4002)
    assert_equal(names[MI_REF_ROT], "masked_mimic_ref_rot")
    var s = metatron_input_shape(MI_REF_ROT)
    assert_true(len(s) == 3 and s[0] == 5 and s[1] == MT_BODIES and s[2] == 4)
    var bools = 0
    for k in range(MT_INPUTS):
        if metatron_input_is_bool(k):
            bools += 1
    assert_true(bools > 10)
    assert_equal(len(g1_joint_names()), MT_DOF)
    assert_equal(len(g1_body_names()), MT_BODIES)
    assert_equal(g1_body_names()[16], "torso_link")
    var g = G1Gains()
    assert_true(near(Float64(g.effort[3]), 139, 1e-6) and near(Float64(g.effort[27]), 5, 1e-6))
    # the action map round trip
    var raw = List[Float32]()
    for j in range(MT_DOF):
        raw.append(Float32(j) * 0.1 - 1.0)
    var act = List[Float32]()
    var o = g1_action_offset()
    var sc = g1_action_scale()
    for j in range(MT_DOF):
        act.append(o[j] + sc[j] * raw[j])
    var back = metatron_raw_from_action(act)
    for j in range(MT_DOF):
        assert_true(near(Float64(back[j]), Float64(raw[j]), 1e-5))
    # history: reset repeats the first reading with zero actions, commit pushes (state, action)
    var inp = MetatronInputs()
    assert_true(near(Float64(inp.at(MI_LOCAL_REF_ROT, 3)), 1.0, 0))
    var r0 = MetatronReading()
    r0.dof_pos[2] = 0.25
    inp.prepare(r0)
    for row in range(8):
        assert_true(near(Float64(inp.at(MI_H_DOF_POS, row * MT_DOF + 2)), 0.25, 0))
    inp.commit(r0, raw)
    var r1 = MetatronReading()
    r1.dof_pos[2] = 0.5
    inp.prepare(r1)
    assert_true(near(Float64(inp.at(MI_DOF_POS, 2)), 0.5, 0))
    assert_true(near(Float64(inp.at(MI_H_DOF_POS, 2)), 0.25, 0))
    assert_true(near(Float64(inp.at(MI_RAW_ACTION_HISTORY, 5)), Float64(raw[5]), 0))
    assert_true(near(Float64(inp.at(MI_RAW_ACTION_HISTORY, MT_DOF + 5)), 0.0, 0))
    assert_true(near(Float64(inp.at(MI_H_RAW_ACTIONS, 5)), 0.0, 0))       # a(t-2) is still zero
    inp.commit(r1, act)
    inp.prepare(r1)
    assert_true(near(Float64(inp.at(MI_H_RAW_ACTIONS, 5)), Float64(raw[5]), 0))
    print("  [PASS] Metatron contract: 39 inputs / 4002 floats, G1 tables, action map, history")


# ----------------------------------------------------------------------------------------- 6

def script_command(mut task: MetatronTask, script: String, k: Int):
    var t = Float32(k) * 0.02
    if script == "idle" or t < 1.0:
        task.set_idle()
    elif script == "walk":
        task.set_steering(0.0, 0.5, 0.0)
    else:
        var a = min(MT_PI / 2, (t - 1.0) * MT_PI / 2)
        task.set_steering(MT_PI / 2, 0.3 * cos(a), 0.3 * sin(a))


def test_g1() raises:
    var dir = env_value("METATRON_DIR")
    if dir.byte_length() == 0:
        print("  [SKIP] G1 + policy (set METATRON_DIR to the tools/metatron/fetch.sh directory)")
        return
    var model = MjcfModel.load(dir + "/g1_holo_compat.xml")
    g1_check_model(model)
    var rest = g1_rest_from_model(model)
    assert_true(near(Float64(rest.upright_height), 0.7657, 2e-4))
    var gpu = env_value("METATRON_CPU") != "1"
    var rdir = env_value("METATRON_REF")
    if rdir.byte_length() > 0:
        var script = env_value("METATRON_SCRIPT")
        if script.byte_length() == 0:
            script = "walk"
        var inp = read_f32(rdir + "/inputs.f32")
        var outp = read_f32(rdir + "/outputs.f32")
        var qp = read_f32(rdir + "/qpos.f32")
        var qv = read_f32(rdir + "/qvel.f32")
        var xp = read_f32(rdir + "/xpos.f32")
        var xq = read_f32(rdir + "/xquat.f32")
        var rp = read_f32(rdir + "/rest_pos.f32")
        for e in range(MT_BODIES * 3):
            assert_true(near(Float64(rp[e]), Float64(rest.pos[e]), 1e-5))
        var W = metatron_input_offsets()[MT_INPUTS]
        var n = len(outp) // 116
        # forward kinematics vs MuJoCo at the start state
        var q0 = List[Float64]()
        var v0 = List[Float64]()
        for e in range(36):
            q0.append(Float64(qp[e]))
        for e in range(35):
            v0.append(Float64(qv[e]))
        var fk = model.forward_kinematics(q0)
        var fpos = fk[0].copy()
        for b in range(MT_BODIES):
            assert_true(vnear(fpos[b + 1], Vec3(xp[b * 3], xp[b * 3 + 1], xp[b * 3 + 2]), 1e-5))
        # the host rebuilds the recorded inputs; the network matches onnxruntime
        var mi = MetatronInputs()
        var task = MetatronTask(rest.copy())
        var worst = 0.0
        for k in range(n):
            script_command(task, script, k)
            var q = List[Float64]()
            var v = List[Float64]()
            for e in range(36):
                q.append(Float64(qp[k * 36 + e]))
            for e in range(35):
                v.append(Float64(qv[k * 35 + e]))
            var bp = List[Float32]()
            var bq = List[Float32]()
            for e in range(99):
                bp.append(xp[k * 99 + e])
            for e in range(132):
                bq.append(xq[k * 132 + e])
            var r = metatron_reading_mj(q, v, bp, bq)
            mi.prepare(r)
            task.build(mi, r, bp, bq)
            for e in range(W):
                worst = max(worst, abs(Float64(mi.flat[e] - inp[k * W + e])))
            var a = List[Float32]()
            for j in range(MT_DOF):
                a.append(outp[k * 116 + j])
            mi.commit(r, metatron_raw_from_action(a))
        print("    host inputs vs MuJoCo host over", n, "ticks: max |diff|", worst)
        assert_true(worst < 1e-5)
        var policy = MetatronPolicy.load(dir + "/model.onnx", gpu)
        var net = 0.0
        for k in [0, n // 2, n - 1]:
            var row = List[Float32](capacity=W)
            for e in range(W):
                row.append(inp[k * W + e])
            var f = policy.run(row).flat()
            for e in range(116):
                net = max(net, abs(Float64(f[e]) - Float64(outp[k * 116 + e])))
        print("    network vs onnxruntime (" + ("GPU" if policy.on_gpu else "CPU") + "): max |diff|", net)
        assert_true(net < 1e-4)
        # open-loop replay of MuJoCo's commands from the same state
        var sim = G1Sim(MjcfModel.load(dir + "/g1_holo_compat.xml"))
        sim.set_state(q0, v0)
        var pelvis = 0.0
        var joints = 0.0
        for k in range(50):
            var cmd = _outputs_at(outp, k)
            sim.apply(cmd)
            var q = sim.qpos()
            pelvis = sqrt((q[0] - Float64(qp[(k + 1) * 36])) ** 2 + (q[1] - Float64(qp[(k + 1) * 36 + 1])) ** 2 + (q[2] - Float64(qp[(k + 1) * 36 + 2])) ** 2)
            var s = 0.0
            for e in range(7, 36):
                s += (q[e] - Float64(qp[(k + 1) * 36 + e])) ** 2
            joints = sqrt(s / 29)
        print("    open-loop replay, 1 s: pelvis error", pelvis, "m, joint RMS error", joints, "rad")
        assert_true(pelvis < 0.05 and joints < 0.08)
    # closed loop: the policy stands the G1 in Flint physics
    var sim = G1Sim(model^)
    var host = MetatronHost(MetatronPolicy.load(dir + "/model.onnx", gpu), rest^)
    var q0 = sim.qpos()
    for _ in range(200):
        host.task.set_idle()
        _ = sim.tick(host)
    var q = sim.qpos()
    var drift = sqrt((q[0] - q0[0]) ** 2 + (q[1] - q0[1]) ** 2)
    print("    closed loop, 4 s idle: pelvis z", q[2], "m, drift", drift, "m")
    assert_true(q[2] > 0.72 and drift < 0.2)
    print("  [PASS] G1 MJCF, rest pose, policy standing in Flint physics")


def _outputs_at(outp: List[Float32], k: Int) -> MetatronOutputs:
    var o = MetatronOutputs()
    for j in range(MT_DOF):
        o.action[j] = outp[k * 116 + j]
        o.stiffness[j] = outp[k * 116 + 29 + j]
        o.damping[j] = outp[k * 116 + 58 + j]
        o.raw_action[j] = outp[k * 116 + 87 + j]
    return o^


def main() raises:
    print("=== Flint v5 Metatron Suite ===")
    test_xml()
    test_model()
    test_kinematics()
    test_articulation()
    test_contract()
    test_g1()
    print("=== ALL FLINT v5 METATRON TESTS PASSED ===")
