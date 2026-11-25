"""Build a motion-matching database from 100STYLE BVH clips (pure Mojo).

  build_database OUT_DIR BVH_DIR [BVH_DIR...] [--no-mirror] [--no-ik]

Every `*.bvh` under each BVH_DIR (one directory level deep, as `fetch_100style.py` lays them out)
is loaded, its 100STYLE joint names are mapped to retarget roles, and it is retargeted onto
Flint's stick biped (`flint.anim.mm_controller.mm_stick_biped_skeleton`) with
`flint.anim.retarget` (role mapping, rest alignment, hip-height scaling, foot IK with contact
pinning). Each clip is added once as-is and once mirrored. Writes, in OUT_DIR:

  database.bin   bones, velocities, ranges, contacts (+ bone names), the reference layout
  features.bin   normalised matching features, offsets and scales

Keep OUT_DIR outside the repository: it is derived from CC BY 4.0 data (see licenses/README.md).
"""
from std.sys import argv
from flint.core.math import Vec3, Quat
from flint.platform.os import list_directory, make_dirs
from flint.anim.retarget import RetargetSkeleton, RetargetClip, RetargetConfig, retarget, foot_contact_error
from flint.anim.mm_database import MmDatabaseBuilder, MmDatabase
from flint.anim.mm_controller import mm_stick_biped_skeleton
from flint.ml.motionbricks import load_bvh


def _style_name(n: String) -> String:
    """100STYLE joint names -> names `flint.anim.retarget` resolves unambiguously."""
    var src: List[String] = ["Chest", "Chest2", "Chest3", "Chest4"]
    var dst: List[String] = ["Spine", "Mid2", "Mid3", "Chest"]
    for i in range(len(src)):
        if n == src[i]:
            return dst[i]
    for side in [String("Left"), String("Right")]:
        if n.startswith(side):
            var rest = String(n[byte=side.byte_length():])
            var a: List[String] = ["Collar", "Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle", "Toe"]
            var b: List[String] = ["Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot", "Toes"]
            for i in range(len(a)):
                if rest == a[i]:
                    return side + b[i]
    return n


def clip_from_bvh(path: String) raises -> RetargetClip:
    var c = load_bvh(path)
    var s = RetargetSkeleton()
    for j in range(c.skeleton.num_joints()):
        var x = c.skeleton.reference[j]
        _ = s.add(_style_name(c.skeleton.names[j]), c.skeleton.parents[j], Vec3(x.p.x, x.p.y, x.p.z), Quat(x.q.x, x.q.y, x.q.z, x.q.w))
    var out = RetargetClip(s^, c.sample_rate)
    for f in range(c.num_frames()):
        var rs = List[Quat]()
        for j in range(c.skeleton.num_joints()):
            var q = c.frames[f][j].q
            rs.append(Quat(q.x, q.y, q.z, q.w))
        out.rotations.append(rs^)
        var p = c.frames[f][0].p
        out.root.append(Vec3(p.x, p.y, p.z))
    return out^


def main() raises:
    var av = argv()
    var args = List[String]()
    var mirror = True
    var ik = True
    for i in range(1, len(av)):
        var a = String(av[i])
        if a == "--no-mirror":
            mirror = False
        elif a == "--no-ik":
            ik = False
        else:
            args.append(a)
    if len(args) < 2:
        print("usage: build_database OUT_DIR BVH_DIR [BVH_DIR...] [--no-mirror] [--no-ik]")
        return
    var out_dir = args[0]
    _ = make_dirs(out_dir)
    var files = List[String]()
    for k in range(1, len(args)):
        var d = args[k]
        for name in list_directory(d):
            if name.startswith("."):
                continue
            if name.endswith(".bvh"):
                files.append(d + "/" + name)
            else:
                for sub in list_directory(d + "/" + name):
                    if sub.endswith(".bvh") and not sub.startswith("."):
                        files.append(d + "/" + name + "/" + sub)
    if len(files) == 0:
        raise Error("no .bvh files found")
    var target = mm_stick_biped_skeleton()
    var b = MmDatabaseBuilder(target.copy())
    var cfg = RetargetConfig()
    cfg.foot_ik = ik
    for path in files:
        var clip = clip_from_bvh(path)
        var res = retarget(clip, target, cfg)
        var err = foot_contact_error(clip, res)
        print('{"clip":"' + path + '","frames":' + String(clip.frame_count()) + ',"scale":' + String(res.scale)
              + ',"mapped_src":' + String(res.source_map.mapped()) + ',"mapped_tgt":' + String(res.target_map.mapped())
              + ',"foot_error":"' + String(err) + '"}')
        b.add_clip(res.clip, False)
        if mirror:
            b.add_clip(res.clip, True)
    var db = b.finish()
    db.save(out_dir + "/database.bin")
    db.save_features(out_dir + "/features.bin")
    var contacts = 0
    for c in db.contacts:
        if c:
            contacts += 1
    print('{"database":"' + out_dir + '/database.bin","frames":' + String(db.nframes) + ',"bones":' + String(db.nbones)
          + ',"ranges":' + String(db.nranges()) + ',"features":' + String(db.nfeatures)
          + ',"contact_ratio":' + String(Float32(contacts) / Float32(max(1, len(db.contacts)))) + "}")
