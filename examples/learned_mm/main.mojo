"""Motion matching / learned motion matching demo (`examples/learned_mm`).

A stick biped retargeted from 100STYLE is driven by scripted keyboard input through an
`InputMap` (W/A/S/D, Shift = walk); its simulation object is a `CharacterController` that
collides with a wall; the matched pose goes through `PhysicalAnimator`, which takes a punch
and a shove on the way. Writes one PNG per two frames (30 fps): the desired trajectory (dots),
the matched animation as a grey ghost, the drawn (physics-blended) pose in colour, the
simulation object (ring), and a top bar (blue = motion matching, violet = learned, amber =
physics blending, flashes white on a transition).

usage: learned_mm DB_DIR OUT_DIR [SECONDS] [--lmm] [--nets DIR] [--eval]

DB_DIR holds `database.bin` from `tools/learned_mm/build_database.mojo` (and, for `--lmm`,
the networks from `tools/learned_mm/train.py`). `--eval` prints quality and timing numbers
instead of rendering: database search (AABB tree vs brute force), projector vs exact search,
and decompressor reconstruction of the database (needs `latent.bin`).
"""
from std.sys import argv
from std.time import perf_counter_ns
from std.math import sqrt, cos, sin
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.detmath import f32_bits
from flint.assets.image import write_png
from flint.platform.os import read_bytes, file_size
from flint.platform.input import InputState
from flint.input.input_map import InputMap, InputContext, ACTION_AXIS2D, ACTION_BOOL
from flint.anim.pose import Pose, SkeletonInstance
from flint.anim.procedural import Placement
from flint.anim.physical_anim import PhysicalAnimator, PA_IDLE, PA_BLEND, PA_FALLEN
from flint.physics.world import PhysicsWorld
from flint.physics.character import CharacterController
from flint.anim.mm_springs import MmPose
from flint.anim.mm_database import MmDatabase, MmBytes, mm_fk_full, MM_BIG
from flint.anim.mm_controller import MmController, MmControllerConfig, mm_input_from_map, mm_stick_biped_model
from flint.anim.lmm import LmmNetworks

comptime W: Int = 640
comptime H: Int = 360
comptime DT: Float32 = 1.0 / 60.0
comptime PX: Float32 = 105.0
comptime KEY_W: Int = 87
comptime KEY_A: Int = 65
comptime KEY_S: Int = 83
comptime KEY_D: Int = 68
comptime KEY_SHIFT: Int = 340
comptime WALL_Z: Float32 = 5.5


struct Canvas(Movable):
    var px: List[UInt32]
    var cam: Vec3

    def __init__(out self):
        self.px = List[UInt32](length=W * H, fill=UInt32(0xFF1B1F27))
        self.cam = Vec3()

    def project(self, p: Vec3) -> Tuple[Float32, Float32]:
        var c = Float32(0.819)
        var s = Float32(0.574)
        var q = p - self.cam
        var sx = Float32(W) * 0.5 + (q.z * c - q.x * s) * PX
        var sy = Float32(H) * 0.78 - (q.y + (q.z * s + q.x * c) * 0.30) * PX
        return (sx, sy)

    def clear(mut self):
        for i in range(W * H):
            self.px[i] = UInt32(0xFF1B1F27)
        # ground grid, 1 m
        var gx = Int(self.cam.x)
        var gz = Int(self.cam.z)
        for k in range(-8, 9):
            self.line(Vec3(Float32(gx + k), 0, Float32(gz - 8)), Vec3(Float32(gx + k), 0, Float32(gz + 8)), 0, UInt32(0xFF2B3240))
            self.line(Vec3(Float32(gx - 8), 0, Float32(gz + k)), Vec3(Float32(gx + 8), 0, Float32(gz + k)), 0, UInt32(0xFF2B3240))

    def disc(mut self, cx: Float32, cy: Float32, r: Float32, col: UInt32):
        var ir = Int(r) + 1
        for dy in range(-ir, ir + 1):
            for dx in range(-ir, ir + 1):
                if Float32(dx * dx + dy * dy) <= r * r + 0.5:
                    var x = Int(cx) + dx
                    var y = Int(cy) + dy
                    if x >= 0 and x < W and y >= 0 and y < H:
                        self.px[y * W + x] = col

    def line(mut self, a: Vec3, b: Vec3, r: Float32, col: UInt32):
        var pa = self.project(a)
        var pb = self.project(b)
        var dx = pb[0] - pa[0]
        var dy = pb[1] - pa[1]
        var n = Int(max(abs(dx), abs(dy))) + 1
        if n > 4000:
            return
        for k in range(n + 1):
            var t = Float32(k) / Float32(n)
            self.disc(pa[0] + dx * t, pa[1] + dy * t, r, col)

    def ring(mut self, c: Vec3, radius: Float32, col: UInt32):
        var prev = c + Vec3(radius, 0, 0)
        for k in range(1, 25):
            var a = Float32(k) * Float32(6.2831853 / 24.0)
            var p = c + Vec3(radius * cos(a), 0, radius * sin(a))
            self.line(prev, p, 0, col)
            prev = p

    def figure(mut self, pts: List[Vec3], parents: List[Int], head: Int, r: Float32, col: UInt32, head_col: UInt32):
        for i in range(1, len(pts)):
            var p = parents[i]
            if p > 0:
                self.line(pts[p], pts[i], r, col)
        var hp = self.project(pts[head])
        self.disc(hp[0], hp[1] - 6, 9, head_col)

    def bar(mut self, col: UInt32, x0: Int, x1: Int):
        for y in range(8):
            for x in range(x0, x1):
                self.px[y * W + x] = col


def _skel_points(skel: SkeletonInstance, place: Placement, n: Int) -> List[Vec3]:
    var out = List[Vec3]()
    for i in range(n):
        out.append(place.to_world(skel.global_position(i)))
    return out^


def _keys(t: Float32, mut st: InputState):
    """The input script: idle, walk, run, turn right, turn back, strafe-ish diagonal, idle."""
    var w = (t >= 1.0 and t < 6.5) or (t >= 11.0 and t < 13.0)
    var d = t >= 6.5 and t < 8.5
    var s = t >= 8.5 and t < 11.0
    var a = t >= 11.0 and t < 13.0
    st.set_logical_key(KEY_W, w)
    st.set_logical_key(KEY_D, d)
    st.set_logical_key(KEY_S, s)
    st.set_logical_key(KEY_A, a)
    st.set_logical_key(KEY_SHIFT, t < 3.5 or t >= 11.0)


def _input_map() -> InputMap:
    var imap = InputMap()
    _ = imap.register_action("Move", ACTION_AXIS2D)
    _ = imap.register_action("Walk", ACTION_BOOL)
    var ctx = InputContext("Locomotion")
    ctx.bind_key("Move", KEY_W, scale_x=0.0, scale_y=1.0)
    ctx.bind_key("Move", KEY_S, scale_x=0.0, scale_y=-1.0)
    ctx.bind_key("Move", KEY_D, scale_x=1.0, scale_y=0.0)
    ctx.bind_key("Move", KEY_A, scale_x=-1.0, scale_y=0.0)
    ctx.bind_key("Walk", KEY_SHIFT)
    imap.add_context(ctx^)
    imap.push_context("Locomotion")
    return imap^


def _features_match_file(db: MmDatabase, path: String) raises -> Int:
    """-1 when there is no file, else the number of feature floats that differ bitwise."""
    if file_size(path) <= 0:
        return -1
    var r = MmBytes(read_bytes(path))
    var nf = r.read_u32()
    var nx = r.read_u32()
    if nf != db.nframes or nx != db.nfeatures:
        return nf * nx
    var bad = 0
    for i in range(nf * nx):
        if f32_bits(r.read_f32()) != f32_bits(db.features[i]):
            bad += 1
    return bad


def evaluate(db: MmDatabase, dir: String, has_nets: Bool) raises:
    var nq = 400
    var seed: UInt32 = 2024
    var queries = List[List[Float32]]()
    var currs = List[Int]()
    for k in range(nq):
        seed = seed * 1664525 + 1013904223
        var base = Int(seed >> 4) % db.nframes
        var q = List[Float32](length=db.nfeatures, fill=0)
        for j in range(db.nfeatures):
            seed = seed * 1664525 + 1013904223
            q[j] = db.feature(base, j) + (Float32(Int(seed >> 8) % 10000) / 10000 - 0.5) * 0.6
        queries.append(q^)
        currs.append(base if k % 2 == 1 else -1)
    var t0 = perf_counter_ns()
    var bi = List[Int]()
    var bc = List[Float32]()
    for k in range(nq):
        var i = currs[k]
        var c = MM_BIG
        db.search_brute(queries[k], i, c)
        bi.append(i)
        bc.append(c)
    var t1 = perf_counter_ns()
    var agree = 0
    for k in range(nq):
        var i = currs[k]
        var c = MM_BIG
        db.search(queries[k], i, c)
        if i == bi[k] and f32_bits(c) == f32_bits(bc[k]):
            agree += 1
    var t2 = perf_counter_ns()
    print('{"search":{"frames":' + String(db.nframes) + ',"queries":' + String(nq) + ',"agree":' + String(agree)
          + ',"brute_ms":' + String(Float64(t1 - t0) / 1.0e6 / Float64(nq)) + ',"aabb_ms":' + String(Float64(t2 - t1) / 1.0e6 / Float64(nq)) + "}}")
    if not has_nets:
        return
    var nets = LmmNetworks.load(dir)
    # projector vs exact nearest neighbour (features in normalised units)
    var pf = List[Float32](length=db.nfeatures, fill=0)
    var pl = List[Float32](length=nets.nlatent, fill=0)
    var proj_err: Float64 = 0
    var exact_cost: Float64 = 0
    var proj_cost: Float64 = 0
    var t3 = perf_counter_ns()
    for k in range(nq):
        var cf = pf.copy()
        _ = nets.project(pf, pl, queries[k], cf, -1.0)
        var c: Float64 = 0
        var e: Float64 = 0
        var nn = -1
        var nc = MM_BIG
        db.search_brute(queries[k], nn, nc, 0, 0, 0)
        for j in range(db.nfeatures):
            var d = Float64(queries[k][j] - pf[j])
            c += d * d
            var g = Float64(pf[j] - db.feature(nn, j))
            e += g * g
        proj_cost += sqrt(c)
        exact_cost += sqrt(Float64(nc))
        proj_err += sqrt(e)
    var t4 = perf_counter_ns()
    print('{"projector":{"mean_query_dist_projected":' + String(proj_cost / Float64(nq)) + ',"mean_query_dist_exact":' + String(exact_cost / Float64(nq))
          + ',"mean_dist_to_exact_match":' + String(proj_err / Float64(nq)) + ',"us_per_projection_incl_brute":' + String(Float64(t4 - t3) / 1.0e3 / Float64(nq)) + "}}")
    # decompressor reconstruction with the trainer's per-frame latents
    var lat_path = dir + "/latent.bin"
    if file_size(lat_path) <= 0:
        print('{"decompressor":"no latent.bin"}')
        return
    var r = MmBytes(read_bytes(lat_path))
    var nf = r.read_u32()
    var nl = r.read_u32()
    var lat = List[Float32](capacity=nf * nl)
    for _ in range(nf * nl):
        lat.append(r.read_f32())
    var feats = List[Float32](length=db.nfeatures, fill=0)
    var z = List[Float32](length=nl, fill=0)
    var pose = MmPose(db.nbones)
    var gt = MmPose(db.nbones)
    var contacts = List[Bool](length=2, fill=False)
    var gp = List[Vec3]()
    var gr = List[Quat]()
    var gp2 = List[Vec3]()
    var gr2 = List[Quat]()
    var err: Float64 = 0
    var worst: Float64 = 0
    var cont_ok = 0
    var n = 0
    var t5 = perf_counter_ns()
    for i in range(0, db.nframes, 7):
        for j in range(db.nfeatures):
            feats[j] = db.feature(i, j)
        for j in range(nl):
            z[j] = lat[i * nl + j]
        db.pose_at(i, gt)
        nets.decompress(pose, contacts, feats, z, Vec3(), Quat())
        pose.pos[0] = gt.pos[0]
        pose.rot[0] = gt.rot[0]
        mm_fk_full(pose, db.parents, gp, gr)
        mm_fk_full(gt, db.parents, gp2, gr2)
        var e: Float64 = 0
        for b in range(1, db.nbones):
            e += Float64((gp[b] - gp2[b]).length())
        e /= Float64(db.nbones - 1)
        err += e
        worst = max(worst, e)
        if contacts[0] == db.contact(i, 0) and contacts[1] == db.contact(i, 1):
            cont_ok += 1
        n += 1
    var t6 = perf_counter_ns()
    print('{"decompressor":{"frames":' + String(n) + ',"mean_joint_error_cm":' + String(err / Float64(n) * 100) + ',"worst_frame_cm":' + String(worst * 100)
          + ',"contact_accuracy":' + String(Float64(cont_ok) / Float64(n)) + ',"us_per_decompress":' + String(Float64(t6 - t5) / 1.0e3 / Float64(n))
          + ',"params":' + String(nets.decompressor.parameter_count() + nets.stepper.parameter_count() + nets.projector.parameter_count()) + "}}")


def main() raises:
    var av = argv()
    var args = List[String]()
    var use_lmm = False
    var do_eval = False
    var nets_dir = String("")
    var i = 1
    while i < len(av):
        var a = String(av[i])
        if a == "--lmm":
            use_lmm = True
        elif a == "--eval":
            do_eval = True
        elif a == "--nets" and i + 1 < len(av):
            nets_dir = String(av[i + 1])
            i += 1
        else:
            args.append(a)
        i += 1
    if len(args) < 1 or (len(args) < 2 and not do_eval):
        print("usage: learned_mm DB_DIR OUT_DIR [SECONDS] [--lmm] [--nets DIR] [--eval]")
        return
    var dir = args[0]
    if nets_dir.byte_length() == 0:
        nets_dir = dir
    var db = MmDatabase.load(dir + "/database.bin")
    db.build_matching_features()
    var mismatch = _features_match_file(db, dir + "/features.bin")
    print("database:", db.nframes, "frames,", db.nbones, "bones,", db.nranges(), "ranges; features vs features.bin:",
          "no file" if mismatch < 0 else ("bit-identical" if mismatch == 0 else String(mismatch) + " differ"))
    var has_nets = file_size(nets_dir + "/projector.bin") > 0
    if do_eval:
        evaluate(db, nets_dir, has_nets)
        return
    var out = args[1]
    var seconds = Float32(14.0)
    if len(args) > 2:
        seconds = Float32(atof(args[2]))

    var model = mm_stick_biped_model(db)
    var mm = MmController(db.copy(), MmControllerConfig.for_100style())
    if use_lmm:
        if not has_nets:
            raise Error("--lmm needs decompressor.bin / stepper.bin / projector.bin in " + nets_dir)
        mm.attach_networks(LmmNetworks.load(nets_dir))
        mm.lmm_enabled = True
    var world = PhysicsWorld()
    _ = world.add_box(Vec3(0, -0.5, 0), Vec3(200, 0.5, 200), mass=0.0)
    _ = world.add_box(Vec3(0, 1.0, WALL_Z + 0.15), Vec3(3.0, 1.0, 0.15), mass=0.0)
    world.rebuild_broadphase()
    var kcc = CharacterController(Vec3(0, 0.95, 0))
    mm.reset(kcc.foot_position(), Quat())

    var pa = PhysicalAnimator()
    var hips = db.bone_index("Hips")
    var chest = db.bone_index("Chest")
    var lua = db.bone_index("LeftUpperArm")
    var rua = db.bone_index("RightUpperArm")
    _ = pa.rsm.ragdoll.add_limb_from_model(model, hips, -1, 15.0, 0.14, 0.25)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, chest, 0, 12.0, 0.15, 0.30)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, lua, 1, 3.5, 0.06, 0.29)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, rua, 1, 3.5, 0.06, 0.29)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, db.bone_index("LeftUpperLeg"), 0, 7.0, 0.08, 0.43)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, db.bone_index("RightUpperLeg"), 0, 7.0, 0.08, 0.43)

    var imap = _input_map()
    var st = InputState()
    var skel = SkeletonInstance(len(model.nodes))
    skel.reset_to_rest(model)
    var ghost = SkeletonInstance(len(model.nodes))
    var anim = Pose()
    var cv = Canvas()
    cv.cam = mm.placement().position
    var frames = Int(seconds / DT)
    var shot = 0
    var flash = 0
    var search_ns: Int = 0
    var head = db.bone_index("Head")
    for f in range(frames):
        var t = Float32(f) * DT
        st.begin_frame()
        _keys(t, st)
        imap.update(st, DT)
        var t0 = perf_counter_ns()
        mm.update_kcc(mm_input_from_map(imap), kcc, world, DT)
        search_ns += perf_counter_ns() - t0
        if mm.just_transitioned:
            flash = 2
        mm.write_model_pose(anim)
        var place = mm.placement()
        if f == 300:
            pa.hit(rua, Vec3(-22, 4, 0), place.to_world(skel.global_position(rua)))
        if f == 560:
            pa.hit(chest, Vec3(30, 0, -20), place.to_world(skel.global_position(chest)))
        pa.update(world, skel, place, model, anim, DT)
        ghost.apply_pose(anim, model)
        if f % 2 == 0:
            cv.cam = cv.cam + (place.position - cv.cam) * 0.12
            cv.clear()
            # wall
            for k in range(-6, 7):
                var x = Float32(k) * 0.5
                cv.line(Vec3(x, 0, WALL_Z), Vec3(x, 2.0, WALL_Z), 1, UInt32(0xFF5D4037))
            cv.line(Vec3(-3, 2.0, WALL_Z), Vec3(3, 2.0, WALL_Z), 1, UInt32(0xFF8D6E63))
            for k in range(1, 4):
                var tp = cv.project(mm.traj_pos[k])
                cv.disc(tp[0], tp[1], 4, UInt32(0xFFFF7043))
            cv.ring(mm.sim_position, 0.3, UInt32(0xFF80CBC4))
            cv.figure(_skel_points(ghost, place, len(model.nodes)), db.parents, head, 1, UInt32(0xFF59606E), UInt32(0xFF59606E))
            cv.figure(_skel_points(skel, place, len(model.nodes)), db.parents, head, 3, UInt32(0xFF4FC3F7), UInt32(0xFFFFD54F))
            var mode_col = UInt32(0xFFB388FF) if use_lmm else UInt32(0xFF42A5F5)
            if flash > 0:
                mode_col = UInt32(0xFFFFFFFF)
            cv.bar(mode_col, 0, W // 2)
            cv.bar(UInt32(0xFF66BB6A) if pa.state == PA_IDLE else (UInt32(0xFFFFB300) if pa.state == PA_BLEND else UInt32(0xFFE53935)), W // 2, W)
            var name = String(shot)
            while name.byte_length() < 4:
                name = "0" + name
            _ = write_png(out + "/f" + name + ".png", cv.px, W, H, False)
            shot += 1
        if flash > 0:
            flash -= 1
        if f % 60 == 0:
            var pl = mm.placement()
            print("t", t, "root", pl.position.x, pl.position.z, "yaw", pl.yaw, "frame", mm.frame_index, "transitions", mm.transitions, "pa", pa.state)
    print("frames", shot, "searches", mm.searches, "transitions", mm.transitions, "hits", pa.hits, "falls", pa.falls,
          "controller us/frame", Float64(search_ns) / 1.0e3 / Float64(frames), "wall stop z", kcc.pos.z)
