"""Flint v5 Make-It-Animatable suite (`test_flint_v5_make_it_animatable.mojo`).

Verifies, without model files:
  1. The fixed 52-joint Mixamo skeleton: parents precede children, 11 tree levels, SOMA names.
  2. Farthest point sampling (first point first, ties to the lowest index) and the two
     normalisations; the hand-box clip keeps exactly the area inside the box.
  3. The Hips frame: a character turned 90 degrees about Y is turned back to face +Z.
  4. Skin-weight clean-up (head override, left / right exclusion, 4 influences) and identity
     rest transforms for identity ortho6d rotations; `top_influences`.
  5. Retargeting a SOMA-named clip onto a rig skeleton, and the GLB writer's animations
     surviving a write / parse round trip.
  6. With `MIA_DIR` (graphs from tools/make_it_animatable/export.py) and `MIA_REF` (dumps from
     tools/make_it_animatable/reference.py): every stage against PyTorch on the dumped inputs
     (FPS picks exactly; joints, rest rotations and skin weights within tolerance), GPU when
     present (`MIA_CPU=1` forces the CPU). With `MIA_MESH` too: the whole GLB -> rigged GLB run,
     written to `MIA_OUT` (default build/mia_rigged.glb).
"""
from std.math import abs, sqrt, cos, sin
from std.time import perf_counter_ns
from std.testing import assert_true, assert_equal
from flint.ml.make_it_animatable import (
    MIA_JOINTS, MIA_POINTS, MIA_HANDS_RATIO, mia_joint_names, mia_parents, mia_joint_index, mia_soma_name,
    mia_depths, farthest_points, latent_indices, normalize_points, apply_scale_shift, clip_to_box,
    hips_transform, mat_point, clean_weights, rest_transforms, skin_to_rest, top_influences, add_skeleton,
    retarget_clip, rig_native_model, MakeItAnimatable,
)
from flint.ml.motionbricks import MbClip, MbXform, MbVec3, MbQuat, quat_axis_angle
from flint.ml.tensor import f32_from_bits
from flint.assets.model import NativeModel, NativeMesh, NativeNode
from flint.assets.glb_writer import encode_native_glb, write_native_glb
from flint.assets.glb_native import parse_native_glb, load_native_glb
from flint.core.math import Vec3
from flint.platform.os import env_value, read_bytes


def test_skeleton() raises:
    var names = mia_joint_names()
    var par = mia_parents()
    assert_equal(len(names), MIA_JOINTS)
    assert_equal(par[0], -1)
    for j in range(1, MIA_JOINTS):
        assert_true(par[j] >= 0 and par[j] < j, "parents precede children: " + names[j])
    assert_equal(names[par[mia_joint_index("LeftHandIndex2")]], "LeftHandIndex1")
    assert_equal(names[par[mia_joint_index("RightHandThumb1")]], "RightHand")
    assert_equal(names[par[mia_joint_index("mixamorig:LeftShoulder")]], "Spine2")
    assert_equal(names[par[mia_joint_index("RightToeBase")]], "RightFoot")
    var depth = mia_depths()
    var deepest = 0
    for d in depth:
        deepest = max(deepest, d)
    assert_equal(deepest, 10)  # Hips .. finger 3: 11 levels of the causal decoders
    assert_equal(mia_soma_name("LeftUpLeg"), "LeftLeg")
    assert_equal(mia_soma_name("mixamorig:LeftLeg"), "LeftShin")
    assert_equal(mia_soma_name("Spine2"), "Chest")
    assert_equal(mia_soma_name("RightHandIndex3"), "RightHandIndex3")
    print("  skeleton: 52 Mixamo joints, 11 levels, SOMA map ok")


def test_sampling() raises:
    # points on a line: FPS starts at 0, then the far end, then the middle
    var pts = List[Float32]()
    for i in range(11):
        pts.append(Float32(i))
        pts.append(0)
        pts.append(0)
    var sel = farthest_points(pts, 0, 11, 3)
    assert_equal(sel[0], 0)
    assert_equal(sel[1], 10)
    assert_equal(sel[2], 5)
    # tie (two points equally far): the lower index wins
    var tie: List[Float32] = [0, 0, 0, 1, 0, 0, -1, 0, 0]
    assert_equal(farthest_points(tie, 0, 3, 2)[1], 1)
    # bbox normalisation: [-1, 1] along the longest axis, centred
    var box: List[Float32] = [2, 0, 0, 6, 1, 1]
    var n1 = normalize_points(box, True)
    var b1 = apply_scale_shift(box, n1[0], n1[1])
    assert_true(abs(b1[0] + 1) < 1e-6 and abs(b1[3] - 1) < 1e-6 and abs(b1[4] - 0.25) < 1e-6, "recentred")
    var n2 = normalize_points(box, False)
    assert_true(abs(n2[0] - 1.0 / 6.0) < 1e-6 and abs(n2[1][0]) == 0, "max-abs scale only")
    # hand box: a 2 x 2 square in z = 0, cube of side 1 centred at the origin -> area 1
    var v: List[Float32] = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]
    var f: List[Int] = [0, 1, 2, 0, 2, 3]
    var cv = List[Float32]()
    var cf = List[Int]()
    clip_to_box(v, f, [0.0, 0.0, 0.0], 0.5, cv, cf)
    var area = 0.0
    for t in range(len(cf) // 3):
        var a = cf[t * 3]
        var b = cf[t * 3 + 1]
        var c = cf[t * 3 + 2]
        var e1x = Float64(cv[b * 3] - cv[a * 3])
        var e1y = Float64(cv[b * 3 + 1] - cv[a * 3 + 1])
        var e2x = Float64(cv[c * 3] - cv[a * 3])
        var e2y = Float64(cv[c * 3 + 1] - cv[a * 3 + 1])
        area += 0.5 * abs(e1x * e2y - e1y * e2x)
    assert_true(abs(area - 1.0) < 1e-6, "clipped area " + String(area))
    for x in cv:
        assert_true(abs(x) <= 0.5 + 1e-6, "clipped vertex inside the box")
    var hi = 2 * MIA_POINTS // 2
    var lat = latent_indices(List[Float32](length=hi * 3, fill=0.0), MIA_HANDS_RATIO)
    assert_equal(len(lat), 512)
    assert_equal(lat[256], MIA_POINTS // 2)  # second group starts at the hand samples
    print("  sampling: FPS, normalisations, hand-box clip ok")


def test_hips_frame() raises:
    # pelvis facing +X (turned +90 degrees about Y): right hip at +z, left hip at -z
    var hips: List[Float64] = [0.3, 1.0, -0.2]
    var rup: List[Float64] = [0.3, 0.95, -0.1]
    var lup: List[Float64] = [0.3, 0.95, -0.3]
    var m = hips_transform(hips, rup, lup)
    var h = mat_point(m, hips[0], hips[1], hips[2])
    assert_true(abs(h[0]) < 1e-9 and abs(h[1]) < 1e-9 and abs(h[2]) < 1e-9, "Hips to the origin")
    var r = mat_point(m, rup[0], rup[1], rup[2])
    var l = mat_point(m, lup[0], lup[1], lup[2])
    assert_true(l[0] - r[0] > 0.19 and abs(l[1] - r[1]) < 1e-9 and abs(l[2] - r[2]) < 1e-9, "right -> left along +X")
    # forward (+X before) now +Z, up stays +Y
    var fw = mat_point(m, hips[0] + 1, hips[1], hips[2])
    assert_true(abs(fw[2] - 1) < 1e-9, "faces +Z")
    var up = mat_point(m, hips[0], hips[1] + 1, hips[2])
    assert_true(abs(up[1] - 1) < 1e-9, "Y stays up")
    print("  Hips frame ok")


def test_weights_and_rest() raises:
    var J = MIA_JOINTS
    var bw = List[Float32](length=3 * J, fill=0.0)
    var la = mia_joint_index("LeftArm")
    var ra = mia_joint_index("RightArm")
    var lf = mia_joint_index("LeftForeArm")
    var sp = mia_joint_index("Spine2")
    var hd = mia_joint_index("Head")
    var nk = mia_joint_index("Neck")
    var li = mia_joint_index("LeftHandIndex1")
    # row 0: left arm dominant with some right-arm weight (dropped) and 4 others (top 4 kept)
    var s1 = mia_joint_index("Spine1")
    var s0 = mia_joint_index("Spine")
    bw[la] = 0.5
    bw[ra] = 0.2
    bw[lf] = 0.15
    bw[sp] = 0.1
    bw[s1] = 0.03
    bw[s0] = 0.02
    # row 1: above the head -> Head
    bw[J + sp] = 0.9
    bw[J + nk] = 0.1
    # row 2: left index finger dominant; other left fingers dropped
    bw[2 * J + li] = 0.6
    bw[2 * J + mia_joint_index("LeftHandMiddle1")] = 0.3
    bw[2 * J + mia_joint_index("LeftHand")] = 0.1
    var ys: List[Float32] = [0.0, 2.0, 0.0]
    var out = clean_weights(bw, ys, 1.0)
    assert_equal(out[ra], 0.0)
    var nz = 0
    var tot: Float32 = 0
    for j in range(J):
        if out[j] > 0:
            nz += 1
        tot += out[j]
    assert_true(nz == 4, "4 influences, got " + String(nz))
    assert_true(out[la] > out[lf] and out[lf] > out[sp], "order kept")
    assert_true(out[J + hd] > 0.99, "above the head: Head")
    assert_equal(out[2 * J + mia_joint_index("LeftHandMiddle1")], 0.0)
    assert_true(abs(out[2 * J + li] - 0.6 / 0.7) < 1e-5, "finger renormalised")
    # identity ortho6d -> identity rest transforms, rest verts unchanged
    var pose6 = List[Float32]()
    var joints = List[Float32]()
    for j in range(J):
        for e in range(6):
            pose6.append(Float32(1.0) if e == 0 or e == 4 else Float32(0.0))
            joints.append(Float32(j) * 0.01)
    var g = rest_transforms(pose6, joints)
    for j in range(J):
        for e in range(16):
            var want = 1.0 if e % 5 == 0 else 0.0
            assert_true(abs(g[j * 16 + e] - want) < 1e-12, "identity rest transform")
    var pv: List[Float32] = [0.1, 0.2, 0.3, -1, 0, 1, 0, 0, 0]
    var rv = skin_to_rest(pv, out, g)
    for i in range(len(pv)):
        assert_true(abs(rv[i] - pv[i]) < 1e-6, "identity rest: vertices unchanged")
    # a 90-degree global turn of LeftArm about Z: the hand (child) follows about the arm joint
    var turn = pose6.copy()
    turn[la * 6 + 0] = 0
    turn[la * 6 + 1] = 1   # x axis -> +Y
    turn[la * 6 + 3] = -1  # y axis -> -X
    turn[la * 6 + 4] = 0
    for e in range(3):
        joints[la * 6 + e] = 0
        joints[lf * 6 + e] = 0
    joints[lf * 6] = 1  # forearm one unit along +X from the arm
    var g2 = rest_transforms(turn, joints)
    var pf = mat_point(List[Float64](g2[lf * 16:(lf + 1) * 16]), 1, 0, 0)
    assert_true(abs(pf[0]) < 1e-9 and abs(pf[1] - 1) < 1e-9, "child carried by the parent's rotation")
    var jt = List[UInt16]()
    var jw = List[Float32]()
    top_influences(out, jt, jw)
    assert_equal(Int(jt[0]), la)
    assert_true(abs(jw[0] + jw[1] + jw[2] + jw[3] - 1) < 1e-6, "top-4 sum to 1")
    print("  skin-weight clean-up, rest transforms ok")


def _t_pose() -> List[Vec3]:
    var out = List[Vec3]()
    var par = mia_parents()
    var names = mia_joint_names()
    for j in range(MIA_JOINTS):
        var p = Vec3(0, 1, 0) if par[j] < 0 else out[par[j]]
        ref n = names[j]
        var d = Vec3(0, 0.1, 0)
        if n.startswith("Left") and not n.endswith("Leg") and not n.endswith("Foot") and not n.endswith("ToeBase"):
            d = Vec3(0.1, 0, 0)
        elif n.startswith("Right") and not n.endswith("Leg") and not n.endswith("Foot") and not n.endswith("ToeBase"):
            d = Vec3(-0.1, 0, 0)
        elif n.endswith("UpLeg"):
            d = Vec3(Float32(0.1) if n.startswith("Left") else Float32(-0.1), -0.05, 0)
        elif n.endswith("Leg") or n.endswith("Foot"):
            d = Vec3(0, -0.4, 0)
        elif n.endswith("ToeBase"):
            d = Vec3(0, 0, 0.1)
        out.append(Vec3(p.x + d.x, p.y + d.y, p.z + d.z))
    return out^


def test_retarget_and_glb() raises:
    var model = NativeModel()
    var mesh = NativeMesh()
    mesh.positions = [0, 0, 0, 0.1, 0, 0, 0, 0.1, 0]
    mesh.indices = [0, 1, 2]
    mesh.joints = List[UInt16](length=12, fill=0)
    mesh.weights = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    mesh.skin = 0
    model.meshes.append(mesh^)
    var mn = NativeNode()
    mn.name = "body"
    mn.mesh = 0
    model.nodes.append(mn^)
    model.roots.append(0)
    add_skeleton(model, _t_pose())
    model.max_influences = 4
    # a SOMA-named clip: Hips moves up 0.1, LeftLeg (= Mixamo LeftUpLeg) turns 90 degrees about X
    var clip = MbClip()
    var soma: List[String] = ["Hips", "Spine1", "LeftLeg", "LeftShin"]
    var sp: List[Int] = [-1, 0, 0, 2]
    for i in range(4):
        clip.skeleton.names.append(soma[i])
        clip.skeleton.parents.append(sp[i])
        clip.skeleton.reference.append(MbXform(MbVec3(0, Float32(0.9) if i == 0 else Float32(0.1), 0), MbQuat.identity()))
    for k in range(2):
        var pose = List[MbXform]()
        for i in range(4):
            pose.append(clip.skeleton.reference[i])
        if k == 1:
            pose[0] = MbXform(MbVec3(0, 1.0, 0), MbQuat.identity())
            pose[2] = MbXform(pose[2].p, quat_axis_angle(MbVec3(1, 0, 0), 1.5707963))
        clip.frames.append(pose^)
    var anim = retarget_clip(clip, model, "test")
    assert_equal(len(anim.channels), 1 + MIA_JOINTS)
    var hips_y = anim.channels[0].values[4]
    assert_true(abs(hips_y - (1.1 + 0.1 * 1.1 / 0.9)) < 1e-5, "hips path scaled by hip height: " + String(hips_y))
    var lu = mia_joint_index("LeftUpLeg")
    ref q = anim.channels[1 + lu].values
    assert_true(abs(q[4] - sin(Float32(0.785398))) < 1e-4 and abs(q[7] - cos(Float32(0.785398))) < 1e-4, "thigh turned")
    var ll = mia_joint_index("LeftLeg")
    ref q2 = anim.channels[1 + ll].values
    assert_true(abs(q2[7] - 1) < 1e-5, "shin local stays identity (follows the thigh)")
    model.animations.append(anim^)
    var bytes = encode_native_glb(model)
    var back = parse_native_glb(bytes)
    assert_equal(len(back.animations), 1)
    assert_equal(len(back.animations[0].channels), 1 + MIA_JOINTS)
    assert_true(abs(back.animations[0].duration - 1.0 / 30.0) < 1e-5, "duration")
    assert_equal(len(back.skins), 1)
    assert_equal(len(back.skins[0].joints), MIA_JOINTS)
    assert_equal(back.nodes[back.skins[0].joints[0]].name, "mixamorig:Hips")
    print("  retarget + GLB animation round trip ok")


# ------------------------------------------------------------------------------- reference parity

def read_f32(path: String) raises -> List[Float32]:
    var b = read_bytes(path)
    if len(b) == 0:
        raise Error("cannot read " + path)
    var n = len(b) // 4
    var out = List[Float32](capacity=n)
    for i in range(n):
        var u = UInt32(b[4 * i]) | (UInt32(b[4 * i + 1]) << 8) | (UInt32(b[4 * i + 2]) << 16) | (UInt32(b[4 * i + 3]) << 24)
        out.append(f32_from_bits(u))
    return out^


def read_i32(path: String) raises -> List[Int]:
    var b = read_bytes(path)
    var out = List[Int]()
    for i in range(len(b) // 4):
        var u = Int(b[4 * i]) | (Int(b[4 * i + 1]) << 8) | (Int(b[4 * i + 2]) << 16) | (Int(b[4 * i + 3]) << 24)
        out.append(u)
    return out^


def max_diff(a: List[Float32], b: List[Float32]) -> Float64:
    var m = 0.0
    if len(a) != len(b):
        return 1e30
    for i in range(len(a)):
        m = max(m, abs(Float64(a[i]) - Float64(b[i])))
    return m


def report(name: String, a: List[Float32], want: List[Float32]) -> Float64:
    var d = max_diff(a, want)
    var mx = 0.0
    for x in want:
        mx = max(mx, abs(Float64(x)))
    print("    " + name + ": max |diff| " + String(d) + " (max |ref| " + String(mx) + ", n " + String(len(want)) + ")")
    return d


def _f32(v: List[Float64]) -> List[Float32]:
    var o = List[Float32](capacity=len(v))
    for x in v:
        o.append(Float32(x))
    return o^


def _ms(t0: Int) -> String:
    return String(Float64(perf_counter_ns() - t0) / 1e6) + " ms"


def test_reference() raises:
    var dir = env_value("MIA_DIR")
    var rdir = env_value("MIA_REF")
    if dir == "" or rdir == "":
        print("  reference parity: skipped (set MIA_DIR and MIA_REF)")
        return
    var t0 = perf_counter_ns()
    var mia = MakeItAnimatable.load(dir, env_value("MIA_CPU") != "1")
    print("    models loaded on " + mia.device() + " in " + _ms(t0))
    var pts1 = read_f32(rdir + "/pts1.f32")
    var pts2 = read_f32(rdir + "/pts2.f32")
    var qv = read_f32(rdir + "/qverts.f32")
    # farthest point sampling: exactly the same picks
    var fps1 = latent_indices(pts1, 0.0)
    var fps2 = latent_indices(pts2, MIA_HANDS_RATIO)
    var want1 = read_i32(rdir + "/fps1.i32")
    var want2 = read_i32(rdir + "/fps2.i32")
    var bad = 0
    for i in range(512):
        if fps1[i] != want1[i]:
            bad += 1
        if fps2[i] != want2[i]:
            bad += 1
    print("    FPS picks differing: " + String(bad) + " of 1024")
    assert_equal(bad, 0)
    # stage 1
    t0 = perf_counter_ns()
    var coarse = mia.coarse_joints(pts1, fps1)
    print("    coarse joints in " + _ms(t0))
    assert_true(report("coarse joints", coarse, read_f32(rdir + "/coarse.f32")) < 2e-3, "coarse joints")
    var rc = read_f32(rdir + "/coarse.f32")
    var h = mia_joint_index("Hips") * 6
    var r = mia_joint_index("RightUpLeg") * 6
    var l = mia_joint_index("LeftUpLeg") * 6
    var H = hips_transform([Float64(rc[h]), Float64(rc[h + 1]), Float64(rc[h + 2])],
                           [Float64(rc[r]), Float64(rc[r + 1]), Float64(rc[r + 2])],
                           [Float64(rc[l]), Float64(rc[l + 1]), Float64(rc[l + 2])])
    assert_true(report("Hips transform", _f32(H), read_f32(rdir + "/hips.f32")) < 1e-5, "Hips transform")
    var raw2 = read_f32(rdir + "/pts2_raw.f32")
    var n2 = normalize_points(raw2, False)
    assert_true(abs(Float64(n2[0]) - Float64(read_f32(rdir + "/scale2.f32")[0])) < 1e-6, "stage-2 scale")
    assert_true(report("stage-2 points", apply_scale_shift(raw2, n2[0], n2[1]), pts2) < 1e-6, "stage-2 points")
    # stage 2 networks
    t0 = perf_counter_ns()
    var bw_raw = mia.skin_weights(pts2, fps2, qv)
    var nv = len(qv) // 3
    print("    skin weights for " + String(nv) + " vertices in " + _ms(t0))
    var d_bw = report("skin weights (network)", bw_raw, read_f32(rdir + "/bw_raw.f32"))
    t0 = perf_counter_ns()
    var joints = mia.fine_joints(pts2, fps2)
    print("    joints in " + _ms(t0))
    var rj = read_f32(rdir + "/joints.f32")
    var d_j = report("joints (head, tail)", joints, rj)
    t0 = perf_counter_ns()
    var pose6 = mia.rest_pose(pts2, fps2, rj)
    print("    rest pose in " + _ms(t0))
    var rp6 = read_f32(rdir + "/pose6.f32")
    var d_p = report("rest rotations (ortho6d)", pose6, rp6)
    # post-processing on the reference network outputs
    var ys = List[Float32](length=nv, fill=0.0)
    for i in range(nv):
        ys[i] = qv[i * 3 + 1]
    var rbw = read_f32(rdir + "/bw_raw.f32")
    var bw = clean_weights(rbw, ys, rj[mia_joint_index("Head") * 6 + 4])
    var d_bwc = report("skin weights (cleaned)", bw, read_f32(rdir + "/bw.f32"))
    var g = rest_transforms(rp6, rj)
    var d_g = report("rest transforms", _f32(g), read_f32(rdir + "/pose.f32"))
    var d_rv = report("rest-pose vertices", skin_to_rest(qv, bw, g), read_f32(rdir + "/rest_verts.f32"))
    # end to end from these network outputs (Flint's own joints / weights)
    var bw_mine = clean_weights(bw_raw, ys, joints[mia_joint_index("Head") * 6 + 4])
    var d_bwe = report("skin weights (cleaned, Flint network)", bw_mine, read_f32(rdir + "/bw.f32"))
    assert_true(d_bw < 2e-3, "network skin weights")
    assert_true(d_j < 2e-3, "joints")
    assert_true(d_p < 5e-3, "rest rotations")
    assert_true(d_bwc < 1e-5, "skin-weight clean-up")
    assert_true(d_g < 1e-4, "rest transforms")
    assert_true(d_rv < 1e-4, "rest-pose vertices")
    _ = d_bwe
    var mesh = env_value("MIA_MESH")
    if mesh == "":
        return
    var src = load_native_glb(mesh)
    assert_true(len(src.meshes) > 0, "mesh loads")
    mia.verbose = True
    t0 = perf_counter_ns()
    var out = rig_native_model(mia, src, True, 0)
    print("    rigged " + mesh + " in " + _ms(t0))
    var path = env_value("MIA_OUT")
    if path == "":
        path = "build/mia_rigged.glb"
    assert_true(write_native_glb(path, out), "write " + path)
    print("    wrote " + path)


def main() raises:
    print("Make-It-Animatable (flint.ml.make_it_animatable)")
    test_skeleton()
    test_sampling()
    test_hips_frame()
    test_weights_and_rest()
    test_retarget_and_glb()
    test_reference()
    print("Make-It-Animatable: all passed")
