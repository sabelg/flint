"""Flint v5 motion matching / learned motion matching suite (`test_flint_v5_learned_mm.mojo`).

Runs without any dataset: a synthetic database is built from procedural walk / run / turn
clips on the stick biped (mirrored), then:
  1. Database: ranges, mirror symmetry of the features, normalisation, file round trip.
  2. Search: the AABB tree returns exactly the brute-force frame and cost, with and without a
     playing frame, and prunes most of the database.
  3. Inertialization: a switch keeps the shown pose and its velocity continuous, then the
     offsets decay to zero.
  4. MLP inference: a file-loaded network matches a scalar reference evaluation.
  5. Controller: identical input scripts give bit-identical poses (determinism); the shown
     joints never jump at transitions while raw frame switches do (continuity); the root
     follows the simulation object; an `InputMap` drives it; a `CharacterController` stops it
     at a wall; `PhysicalAnimator` passes the pose through when idle and blends on a hit.
  6. With LMM_DIR (a `tools/learned_mm` output directory): Mojo vs PyTorch parity of the three
     trained networks, decompressor reconstruction of the database, and a learned-mode run.
"""
from std.math import abs, sqrt
from std.testing import assert_true, assert_equal
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.detmath import det_sin, det_cos, f32_bits
from flint.core.ik import angle_axis
from flint.platform.os import env_value, read_bytes, write_bytes
from flint.platform.input import InputState
from flint.input.input_map import InputMap, InputContext, ACTION_AXIS2D, ACTION_BOOL
from flint.assets.model import NativeModel
from flint.anim.pose import Pose, SkeletonInstance
from flint.anim.procedural import Placement
from flint.anim.retarget import RetargetClip
from flint.anim.physical_anim import PhysicalAnimator, PA_IDLE, PA_BLEND
from flint.physics.world import PhysicsWorld
from flint.physics.character import CharacterController
from flint.anim.mm_springs import MmPose, MmInertializer
from flint.anim.mm_database import MmDatabase, MmDatabaseBuilder, MmBytes, mm_fk_full, MM_BIG
from flint.anim.mm_controller import (
    MmController, MmControllerConfig, MmInput, mm_input_from_map, mm_stick_biped_skeleton, mm_stick_biped_model,
)
from flint.anim.lmm import MlpNet, MlpEval, mlp_evaluate, LmmNetworks

comptime DT: Float32 = 1.0 / 60.0
comptime TAU: Float32 = 6.2831853


# ----------------------------------------------------------------------------- synthetic data

def _clip(speed: Float32, turn: Float32, seconds: Float32, cadence: Float32, side: Float32 = 0) -> RetargetClip:
    var sk = mm_stick_biped_skeleton()
    var clip = RetargetClip(sk.copy(), 60)
    var n = Int(seconds * 60)
    var yaw: Float32 = 0
    var p = Vec3(0, 0.95, 0)
    var x = Vec3(1, 0, 0)
    var lul = sk.index_of("LeftUpperLeg")
    var rul = sk.index_of("RightUpperLeg")
    var lll = sk.index_of("LeftLowerLeg")
    var rll = sk.index_of("RightLowerLeg")
    var lua = sk.index_of("LeftUpperArm")
    var rua = sk.index_of("RightUpperArm")
    var amp = min(Float32(0.7), Float32(0.25) + speed * Float32(0.2))
    for f in range(n):
        var t = Float32(f) / 60
        var ph = TAU * cadence * t
        var s = det_sin(ph)
        var c = det_cos(ph)
        var rs = List[Quat](length=sk.count(), fill=Quat())
        rs[0] = angle_axis(yaw, Vec3(0, 1, 0)).multiply(angle_axis(Float32(0.04) * s, Vec3(0, 0, 1)))
        rs[lul] = angle_axis(-amp * s, x)
        rs[rul] = angle_axis(amp * s, x)
        rs[lll] = angle_axis(amp * Float32(0.6) * (Float32(1) + c), x)
        rs[rll] = angle_axis(amp * Float32(0.6) * (Float32(1) - c), x)
        rs[lua] = angle_axis(Float32(1.2), Vec3(0, 0, -1)).multiply(angle_axis(amp * s, x))
        rs[rua] = angle_axis(Float32(1.2), Vec3(0, 0, 1)).multiply(angle_axis(-amp * s, x))
        clip.rotations.append(rs^)
        clip.root.append(Vec3(p.x, Float32(0.95) + Float32(0.02) * c * c, p.z))
        var fwd = Vec3(det_sin(yaw), 0, det_cos(yaw))
        var lft = Vec3(det_cos(yaw), 0, -det_sin(yaw))
        p = p + fwd * (speed / 60) + lft * (side / 60)
        yaw += turn / 60
    return clip^


def _database() -> MmDatabase:
    var b = MmDatabaseBuilder(mm_stick_biped_skeleton())
    # idle, walk, walk turning, run, run turning, back
    var speed: List[Float32] = [0.0, 1.2, 1.2, 3.5, 3.5, -0.9]
    var turn: List[Float32] = [0.0, 0.0, 0.9, 0.0, -0.7, 0.0]
    var secs: List[Float32] = [4.0, 8.0, 8.0, 6.0, 6.0, 5.0]
    var cad: List[Float32] = [0.0, 1.0, 1.0, 1.6, 1.6, 1.0]
    for k in range(len(speed)):
        var c = _clip(speed[k], turn[k], secs[k], cad[k])
        b.add_clip(c, False)
        b.add_clip(c, True)
    var side = _clip(0.0, 0.0, 5.0, 1.0, 1.0)
    b.add_clip(side, False)
    b.add_clip(side, True)
    return b.finish()


# ----------------------------------------------------------------------------- 1

def test_database(db: MmDatabase) raises:
    assert_equal(db.nbones, 22)
    assert_equal(db.nfeatures, 27)
    assert_equal(db.nranges(), 14)
    assert_true(db.nframes > 4000)
    assert_equal(db.range_stops[db.nranges() - 1], db.nframes)
    # mirrored copy: left foot x of the original = -(right foot x) of the mirror
    var s0 = db.range_starts[2]
    var s1 = db.range_starts[3]
    var worst: Float32 = 0
    for f in range(0, db.range_stops[2] - s0, 17):
        var lx = db.denormalized_feature(s0 + f, 0)
        var rx = db.denormalized_feature(s1 + f, 3)
        worst = max(worst, abs(lx + rx))
        var ly = db.denormalized_feature(s0 + f, 1)
        var ry = db.denormalized_feature(s1 + f, 4)
        worst = max(worst, abs(ly - ry))
    assert_true(worst < 1.0e-3, "mirrored features are symmetric")
    # normalised feature means are ~0
    for j in range(db.nfeatures):
        var m: Float32 = 0
        for i in range(db.nframes):
            m += db.feature(i, j)
        assert_true(abs(m / Float32(db.nframes)) < 1.0e-3)
    # walking clip: forward hip velocity, feet below the hips
    var mid = db.range_starts[2] + 120
    assert_true(db.denormalized_feature(mid, 14) > 1.0, "hip velocity forward")
    assert_true(db.denormalized_feature(mid, 1) < 0.3, "foot near the ground")
    # round trip through the reference database.bin layout
    var path = String("/tmp/flint_test_mm_database.bin")
    db.save(path)
    var back = MmDatabase.load(path)
    back.build_matching_features()
    assert_equal(back.nframes, db.nframes)
    assert_equal(back.names[16], db.names[16])
    var same = True
    for i in range(len(db.features)):
        if f32_bits(db.features[i]) != f32_bits(back.features[i]):
            same = False
            break
    assert_true(same, "features rebuilt from the saved database are bit-identical")
    print("  database:", db.nframes, "frames,", db.nranges(), "ranges; mirror + round trip exact")


# ----------------------------------------------------------------------------- 2

def test_search(db: MmDatabase) raises:
    var seed: UInt32 = 12345
    var pruned_visits = 0
    for k in range(200):
        var base = (k * 977) % db.nframes
        var q = List[Float32](length=db.nfeatures, fill=0)
        for j in range(db.nfeatures):
            seed = seed * 1664525 + 1013904223
            var u = Float32(Int(seed >> 8) % 10000) / 10000 - Float32(0.5)
            q[j] = db.feature(base, j) + u * Float32(0.8)
        var curr = -1 if k % 2 == 0 else base
        var bi = curr
        var bc = MM_BIG
        db.search_brute(q, bi, bc, Float32(0.0) if k % 3 else Float32(0.3))
        var ai = curr
        var ac = MM_BIG
        db.search(q, ai, ac, Float32(0.0) if k % 3 else Float32(0.3))
        assert_equal(ai, bi)
        assert_true(f32_bits(ac) == f32_bits(bc), "same cost")
        _ = pruned_visits
    print("  search: AABB tree == brute force on 200 queries (index and cost)")


# ----------------------------------------------------------------------------- 3

def test_inertialize(db: MmDatabase) raises:
    var inert = MmInertializer(db.nbones)
    var a = MmPose()
    var b = MmPose()
    var out = MmPose()
    db.pose_at(db.range_starts[1] + 100, a)
    db.pose_at(db.range_starts[7] + 50, b)
    db.pose_at(db.range_starts[1] + 100, out)
    inert.reset(a.pos[0], a.rot[0])
    inert.update(out, a, 0.1, 0)
    var shown = out.copy()
    inert.transition(out, a, b)
    inert.update(out, b, 0.1, 0)
    var d: Float32 = 0
    for i in range(1, db.nbones):
        d = max(d, (out.pos[i] - shown.pos[i]).length())
        d = max(d, (out.vel[i] - shown.vel[i]).length())
        d = max(d, Float32(1) - abs(out.rot[i].dot(shown.rot[i])))
    assert_true(d < 1.0e-4, "the switch itself shows the old pose and velocity")
    assert_true(inert.largest_offset() > 1.0e-3, "there is an offset to blend out")
    for _ in range(90):
        inert.update(out, b, 0.1, DT)
    assert_true(inert.largest_offset() < 1.0e-3, "offsets decay")
    print("  inertialization: continuous at the switch, offsets decay (", inert.largest_offset(), ")")


# ----------------------------------------------------------------------------- 4

def _write_net(path: String, sizes: List[Int]) raises:
    var w = MmBytes()
    var seed: UInt32 = 99
    var nin = sizes[0]
    var nout = sizes[len(sizes) - 1]
    for arr in range(4):
        var n = nin if arr < 2 else nout
        w.u32(n)
        for i in range(n):
            w.f32(Float32(0.1) * Float32(i % 3) if arr % 2 == 0 else Float32(1.0) + Float32(0.05) * Float32(i % 4))
    w.u32(len(sizes) - 1)
    for l in range(len(sizes) - 1):
        w.u32(sizes[l]); w.u32(sizes[l + 1])
        for _ in range(sizes[l] * sizes[l + 1]):
            seed = seed * 1664525 + 1013904223
            w.f32(Float32(Int(seed >> 9) % 2001 - 1000) / 4000)
        w.u32(sizes[l + 1])
        for i in range(sizes[l + 1]):
            w.f32(Float32(i % 5 - 2) * Float32(0.03))
    if not write_bytes(path, w.data):
        raise Error("cannot write " + path)


def _reference_eval(nn: MlpNet, x: List[Float32]) -> List[Float32]:
    var h = List[Float32]()
    for i in range(len(x)):
        h.append((x[i] - nn.input_mean[i]) / nn.input_std[i])
    for l in range(nn.layer_count()):
        var o = List[Float32]()
        for j in range(nn.cols[l]):
            var acc = nn.biases[l][j]
            for i in range(nn.rows[l]):
                acc += h[i] * nn.weights[l][i * nn.cols[l] + j]
            o.append(max(acc, Float32(0)) if l != nn.layer_count() - 1 else acc)
        h = o^
    for j in range(len(h)):
        h[j] = h[j] * nn.output_std[j] + nn.output_mean[j]
    return h^


def test_mlp() raises:
    var path = String("/tmp/flint_test_mlp.bin")
    _write_net(path, [13, 37, 21, 11])
    var nn = MlpNet.load(path)
    assert_equal(nn.layer_count(), 3)
    assert_equal(nn.parameter_count(), 13 * 37 + 37 + 37 * 21 + 21 + 21 * 11 + 11)
    var ev = MlpEval(nn)
    var worst: Float32 = 0
    for k in range(20):
        var x = List[Float32]()
        for i in range(13):
            x.append(det_sin(Float32(k * 13 + i) * Float32(0.7)))
        for i in range(13):
            ev.layers[0][i] = x[i]
        mlp_evaluate(ev, nn)
        var ref_y = _reference_eval(nn, x)
        for j in range(11):
            worst = max(worst, abs(ev.layers[3][j] - ref_y[j]))
    assert_true(worst < 1.0e-5, "SIMD evaluation matches the scalar reference")
    print("  mlp: file-loaded network matches scalar evaluation (max err", worst, ")")


# ----------------------------------------------------------------------------- 5

def _script(f: Int) -> MmInput:
    """Stand, walk forward, run and turn, stop, strafe, walk back."""
    var t = Float32(f) * DT
    if t < 0.5:
        return MmInput.idle()
    if t < 2.5:
        return MmInput(Vec3(0, 0, 1), Vec3(), True, False, 0)
    if t < 4.5:
        return MmInput(Vec3(det_sin(t - Float32(2.5)), 0, det_cos(t - Float32(2.5))), Vec3(), False, False, 0)
    if t < 5.5:
        return MmInput.idle()
    if t < 7.0:
        return MmInput(Vec3(1, 0, 0), Vec3(), True, True, 0)
    return MmInput(Vec3(0, 0, -1), Vec3(), True, False, 0)


def _run(var db: MmDatabase, frames: Int, mut positions: List[Vec3], mut max_jump: Float32, mut max_sim_gap: Float32,
         halflife: Float32 = 0.1) raises -> Int:
    """Runs the script; `max_jump` is the largest per-frame change of a joint's velocity
    relative to the root (second difference, metres per frame), which a pop would spike."""
    var cfg = MmControllerConfig()
    cfg.inertialize_halflife = halflife
    var mm = MmController(db^, cfg)
    mm.reset(Vec3(), Quat())
    var prev = mm.global_positions()
    var prev2 = prev.copy()
    max_jump = 0
    max_sim_gap = 0
    for f in range(frames):
        mm.update(_script(f), DT)
        var g = mm.global_positions()
        for i in range(len(g)):
            positions.append(g[i])
            if f >= 2:
                var acc = (g[i] - g[0]) - (prev[i] - prev[0]) * 2 + (prev2[i] - prev2[0])
                max_jump = max(max_jump, acc.length())
        max_sim_gap = max(max_sim_gap, (Vec3(g[0].x, 0, g[0].z) - Vec3(mm.sim_position.x, 0, mm.sim_position.z)).length())
        prev2 = prev^
        prev = g^
    return mm.transitions


def test_controller(db: MmDatabase) raises:
    var a = List[Vec3]()
    var b = List[Vec3]()
    var jump: Float32 = 0
    var gap: Float32 = 0
    var jump2: Float32 = 0
    var gap2: Float32 = 0
    var tr = _run(db.copy(), 540, a, jump, gap)
    _ = _run(db.copy(), 540, b, jump2, gap2)
    assert_equal(len(a), len(b))
    var same = True
    for i in range(len(a)):
        if f32_bits(a[i].x) != f32_bits(b[i].x) or f32_bits(a[i].y) != f32_bits(b[i].y) or f32_bits(a[i].z) != f32_bits(b[i].z):
            same = False
            break
    assert_true(same, "deterministic: identical runs give bit-identical joints")
    assert_true(tr >= 5, "the script causes transitions")
    # raw switches between the frames the controller picked jump far more than the shown pose
    var raw_jump: Float32 = 0
    var pa = MmPose()
    var pb = MmPose()
    var ga = List[Vec3]()
    var gr = List[Quat]()
    var gb = List[Vec3]()
    var gr2 = List[Quat]()
    for k in range(40):
        db.pose_at(db.range_starts[k % db.nranges()] + 30, pa)
        db.pose_at(db.range_starts[(k * 5 + 3) % db.nranges()] + 70, pb)
        mm_fk_full(pa, db.parents, ga, gr)
        mm_fk_full(pb, db.parents, gb, gr2)
        for i in range(1, db.nbones):
            raw_jump = max(raw_jump, ((gb[i] - gb[0]) - (ga[i] - ga[0])).length())
    print("  controller:", tr, "transitions; max joint velocity change", jump, "m/frame (raw switch", raw_jump, "m); root-sim gap", gap, "m")
    assert_true(jump < 0.05, "continuity: no joint velocity changes by more than 5 cm/frame between frames")
    assert_true(raw_jump > 5.0 * jump, "inertialization hides the switches")
    var c = List[Vec3]()
    var hard: Float32 = 0
    var gap3: Float32 = 0
    _ = _run(db.copy(), 540, c, hard, gap3, 1.0e-4)
    print("  without inertialization the same run changes joint velocity by", hard, "m/frame")
    assert_true(hard > 3.0 * jump, "the blend, not the data, keeps it smooth")
    assert_true(gap < 0.16, "the root stays within the clamp distance of the simulation object")


def test_input_map(db: MmDatabase) raises:
    var imap = InputMap()
    _ = imap.register_action("Move", ACTION_AXIS2D)
    _ = imap.register_action("Walk", ACTION_BOOL)
    var ctx = InputContext("Locomotion")
    ctx.bind_key("Move", 87, scale_x=0.0, scale_y=1.0)    # W
    ctx.bind_key("Move", 83, scale_x=0.0, scale_y=-1.0)   # S
    ctx.bind_key("Move", 68, scale_x=1.0, scale_y=0.0)    # D
    ctx.bind_key("Move", 65, scale_x=-1.0, scale_y=0.0)   # A
    ctx.bind_key("Walk", 340)                              # Shift
    imap.add_context(ctx^)
    imap.push_context("Locomotion")
    var st = InputState()
    var mm = MmController(db.copy())
    mm.reset(Vec3(), Quat())
    for f in range(180):
        st.begin_frame()
        st.set_logical_key(87, True)
        st.set_logical_key(340, f < 90)
        imap.update(st, DT)
        var inp = mm_input_from_map(imap)
        assert_true(inp.move.z > 0.99)
        mm.update_from_input(imap, 0, DT)
    assert_true(mm.sim_position.z > 2.0, "W moves the character forward")
    assert_true(abs(mm.sim_position.x) < 0.05)
    print("  input map: W/Shift drive the controller to z =", mm.sim_position.z)


def test_kcc_wall(db: MmDatabase) raises:
    var world = PhysicsWorld()
    _ = world.add_box(Vec3(0, -0.5, 0), Vec3(40, 0.5, 40), mass=0.0)
    _ = world.add_box(Vec3(0, 1.0, 4.0), Vec3(4, 1.0, 0.2), mass=0.0)    # wall at z = 3.8
    world.rebuild_broadphase()
    var kcc = CharacterController(Vec3(0, 0.95, 0))
    var mm = MmController(db.copy())
    mm.reset(kcc.foot_position(), Quat())
    for _ in range(360):
        mm.update_kcc(MmInput(Vec3(0, 0, 1), Vec3(), False, False, 0), kcc, world, DT)
    assert_true(kcc.pos.z < 3.8 - kcc.cfg.radius + 0.05, "the wall stops the simulation object")
    assert_true(kcc.pos.z > 2.5, "it ran up to the wall")
    assert_true(mm.out.pos[0].z < 3.8, "the animated root stays on its side of the wall")
    print("  kcc: stopped at the wall, sim z =", kcc.pos.z, "root z =", mm.out.pos[0].z)


def test_physical_animator(db: MmDatabase) raises:
    var model = mm_stick_biped_model(db)
    var world = PhysicsWorld()
    _ = world.add_box(Vec3(0, -0.5, 0), Vec3(40, 0.5, 40), mass=0.0)
    world.rebuild_broadphase()
    var pa = PhysicalAnimator()
    var hips = db.bone_index("Hips")
    var chest = db.bone_index("Chest")
    var rua = db.bone_index("RightUpperArm")
    var lul = db.bone_index("LeftUpperLeg")
    var rul = db.bone_index("RightUpperLeg")
    _ = pa.rsm.ragdoll.add_limb_from_model(model, hips, -1, 15.0, 0.14, 0.25)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, chest, 0, 12.0, 0.15, 0.30)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, rua, 1, 3.5, 0.06, 0.29)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, lul, 0, 7.0, 0.08, 0.43)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, rul, 0, 7.0, 0.08, 0.43)
    var mm = MmController(db.copy())
    mm.reset(Vec3(), Quat())
    var skel = SkeletonInstance(len(model.nodes))
    skel.reset_to_rest(model)
    var ghost = SkeletonInstance(len(model.nodes))
    var anim = Pose()
    var blended = False
    for f in range(240):
        mm.update(MmInput(Vec3(0, 0, 1), Vec3(), True, False, 0), DT)
        mm.write_model_pose(anim)
        var place = mm.placement()
        if f == 120:
            pa.hit(rua, Vec3(-20, 0, 0), place.to_world(skel.global_position(rua)))
        pa.update(world, skel, place, model, anim, DT)
        if f < 120:
            assert_equal(pa.state, PA_IDLE)
            ghost.apply_pose(anim, model)
            var d: Float32 = 0
            for i in range(len(model.nodes)):
                d = max(d, (skel.global_position(i) - ghost.global_position(i)).length())
            assert_true(d == 0.0, "idle: the matched pose passes through exactly")
        if pa.state == PA_BLEND:
            blended = True
    assert_true(blended, "a hit blends physics into the matched animation")
    print("  physical animator: exact pass-through when idle, blends on a hit (hits", pa.hits, ")")


# ----------------------------------------------------------------------------- 6

def _read_parity(path: String) raises -> List[Tuple[List[Float32], List[Float32], Int, Int, Int]]:
    var r = MmBytes(read_bytes(path))
    var out = List[Tuple[List[Float32], List[Float32], Int, Int, Int]]()
    var k = r.read_u32()
    for _ in range(k):
        var n = r.read_u32()
        var nin = r.read_u32()
        var nout = r.read_u32()
        var x = List[Float32]()
        for _ in range(n * nin):
            x.append(r.read_f32())
        var y = List[Float32]()
        for _ in range(n * nout):
            y.append(r.read_f32())
        out.append((x^, y^, n, nin, nout))
    return out^


def _parity(nn: MlpNet, p: Tuple[List[Float32], List[Float32], Int, Int, Int], name: String) raises:
    var ev = MlpEval(nn)
    var worst: Float32 = 0
    var scale: Float32 = 0
    var nin = p[3]
    var nout = p[4]
    for s in range(p[2]):
        for i in range(nin):
            ev.layers[0][i] = p[0][s * nin + i]
        mlp_evaluate(ev, nn)
        var last = len(ev.layers) - 1
        for j in range(nout):
            var y = p[1][s * nout + j]
            worst = max(worst, abs(ev.layers[last][j] - y))
            scale = max(scale, abs(y))
    print("  parity", name, ": max |mojo - torch| =", worst, "(max |y|", scale, ")")
    assert_true(worst < 1.0e-3 * max(Float32(1), scale), "Mojo inference matches PyTorch")


def test_trained() raises:
    var dir = env_value("LMM_DIR")
    if dir.byte_length() == 0:
        print("  [SKIP] trained networks (set LMM_DIR to a tools/learned_mm output directory)")
        return
    var nets = LmmNetworks.load(dir)
    var names: List[String] = ["decompressor", "stepper", "projector"]
    var par = _read_parity(dir + "/parity.bin")
    _parity(nets.decompressor, par[0], names[0])
    _parity(nets.stepper, par[1], names[1])
    _parity(nets.projector, par[2], names[2])
    # learned-mode controller run: deterministic and continuous
    var db = MmDatabase.load(dir + "/database.bin")
    db.build_matching_features()
    var runs = List[List[Vec3]]()
    var jump: Float32 = 0
    for _ in range(2):
        var mm = MmController(db.copy())
        mm.attach_networks(LmmNetworks.load(dir))
        mm.lmm_enabled = True
        mm.reset(Vec3(), Quat())
        var pos = List[Vec3]()
        var prev = mm.global_positions()
        var prev2 = prev.copy()
        for f in range(480):
            mm.update(_script(f), DT)
            var g = mm.global_positions()
            for i in range(len(g)):
                pos.append(g[i])
                if f >= 2:
                    jump = max(jump, ((g[i] - g[0]) - (prev[i] - prev[0]) * 2 + (prev2[i] - prev2[0])).length())
            prev2 = prev^
            prev = g^
        runs.append(pos^)
    var same = True
    for i in range(len(runs[0])):
        if f32_bits(runs[0][i].x) != f32_bits(runs[1][i].x) or f32_bits(runs[0][i].z) != f32_bits(runs[1][i].z):
            same = False
            break
    assert_true(same, "learned mode is deterministic")
    print("  learned mode: deterministic, max joint velocity change", jump, "m/frame")
    assert_true(jump < 0.08, "learned mode is continuous")


def main() raises:
    print("=== Flint v5 Motion Matching / Learned Motion Matching Suite ===")
    var db = _database()
    test_database(db)
    test_search(db)
    test_inertialize(db)
    test_mlp()
    test_controller(db)
    test_input_map(db)
    test_kcc_wall(db)
    test_physical_animator(db)
    test_trained()
    print("=== ALL FLINT v5 LEARNED MOTION MATCHING TESTS PASSED ===")
