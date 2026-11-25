"""Flint geometry kit CLI (pure Mojo, no model weights).

  geometry_kit stats   in.glb
  geometry_kit lod     in.glb out_prefix [--ratios 0.5,0.25,0.125] [--lock-boundaries]
  geometry_kit meshlets in.glb
  geometry_kit unwrap  in.glb out.glb [--res 1024] [--pad 2]
  geometry_kit remesh  in.glb out.glb [--verts N] [--iters 6] [--unwrap]
  geometry_kit retarget clip.bvh rigged.glb out.glb [--no-ik] [--hand-ik]
  geometry_kit iso     sphere|box|torus out.glb [--n 48] [--dc] [--sparse]

Every mesh of the model is processed; nodes, skins, materials and animations are kept.
Each command prints one JSON line with before/after numbers.
"""
from std.sys import argv
from flint.core.math import Vec3, Quat
from flint.assets.model import NativeModel, NativeMesh
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_writer import write_native_glb
from flint.assets.geometry_kit import KitOptions, process_model
from flint.assets.mesh_kit import mesh_topology
from flint.procedural.isosurface import (grid_from_sphere, grid_from_box, grid_from_torus, marching_cubes,
    dual_contouring, sparse_from_dense, marching_cubes_sparse, dual_contouring_sparse, SdfGrid)
from flint.anim.retarget import (RetargetSkeleton, RetargetClip, RetargetConfig, retarget, skeleton_from_model,
    detect_humanoid_roles, foot_contact_error, clip_to_animation, ROLE_COUNT)
from flint.ml.motionbricks import load_bvh, MbClip


def _flag(a: List[String], name: String) -> Bool:
    for i in range(len(a)):
        if a[i] == name:
            return True
    return False


def _opt(a: List[String], name: String, default: String) -> String:
    for i in range(len(a) - 1):
        if a[i] == name:
            return a[i + 1]
    return default


def _ratios(s: String) raises -> List[Float32]:
    var out = List[Float32]()
    for part in s.split(","):
        out.append(Float32(atof(String(part))))
    return out^


def clip_from_bvh(path: String) raises -> RetargetClip:
    var c = load_bvh(path)
    var s = RetargetSkeleton()
    for j in range(c.skeleton.num_joints()):
        var x = c.skeleton.reference[j]
        _ = s.add(c.skeleton.names[j], c.skeleton.parents[j], Vec3(x.p.x, x.p.y, x.p.z), Quat(x.q.x, x.q.y, x.q.z, x.q.w))
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


def _save(path: String, mut model: NativeModel) raises:
    if not write_native_glb(path, model):
        raise Error("failed to write " + path)


def _usage():
    print("usage: geometry_kit <stats|lod|meshlets|unwrap|remesh|retarget|iso> ... (see examples/geometry_kit/main.mojo)")


def main() raises:
    var av = argv()
    var a = List[String]()
    for i in range(len(av)):
        a.append(String(av[i]))
    if len(a) < 3:
        _usage()
        return
    var cmd = a[1]
    if cmd == "stats" or cmd == "meshlets":
        var model = load_native_glb(a[2])
        print(process_model(model, KitOptions(cmd)))
    elif cmd == "lod":
        if len(a) < 4:
            _usage()
            return
        var ratios = _ratios(_opt(a, "--ratios", "0.5,0.25,0.125,0.0625"))
        var base = load_native_glb(a[2])
        for i in range(len(ratios)):
            var model = base.copy()
            var o = KitOptions("lod")
            o.ratio = ratios[i]
            o.lock_boundaries = _flag(a, "--lock-boundaries")
            var rep = process_model(model, o)
            var path = a[3] + "_lod" + String(i + 1) + ".glb"
            _save(path, model)
            print('{"level":' + String(i + 1) + ',"ratio":' + String(ratios[i]) + ',"path":"' + path + '","report":' + rep + "}")
            var ml = KitOptions("meshlets")
            print('{"level":' + String(i + 1) + ',"meshlets":' + process_model(model, ml) + "}")
    elif cmd == "unwrap" or cmd == "remesh":
        if len(a) < 4:
            _usage()
            return
        var model = load_native_glb(a[2])
        var o = KitOptions(cmd)
        o.resolution = atol(_opt(a, "--res", "1024"))
        o.padding = atol(_opt(a, "--pad", "2"))
        o.target_vertices = atol(_opt(a, "--verts", "0"))
        o.iterations = atol(_opt(a, "--iters", "6"))
        print(process_model(model, o))
        if cmd == "remesh" and _flag(a, "--unwrap"):
            var u = KitOptions("unwrap")
            u.resolution = o.resolution
            u.padding = o.padding
            print(process_model(model, u))
        _save(a[3], model)
    elif cmd == "retarget":
        if len(a) < 5:
            _usage()
            return
        var src = clip_from_bvh(a[2])
        var model = load_native_glb(a[3])
        var tgt = skeleton_from_model(model, 0)
        var cfg = RetargetConfig()
        cfg.foot_ik = False
        var raw = retarget(src, tgt, cfg)
        var before = foot_contact_error(src, raw)
        cfg.foot_ik = not _flag(a, "--no-ik")
        cfg.hand_ik = _flag(a, "--hand-ik")
        var res = retarget(src, tgt, cfg)
        var after = foot_contact_error(src, res)
        var roles = 0
        for r in range(ROLE_COUNT):
            if res.target_map.joint[r] >= 0:
                roles += 1
        model.animations.append(clip_to_animation(res.clip, "retarget"))
        _save(a[4], model)
        print('{"ok":true,"frames":' + String(len(res.clip.rotations)) + ',"target_roles":' + String(roles)
              + ',"scale":' + String(res.scale) + ',"contact_frames":' + String(after.contact_frames)
              + ',"foot_slide_no_ik":' + String(before.max_slide) + ',"foot_slide":' + String(after.max_slide)
              + ',"foot_height_error":' + String(after.max_height_error) + "}")
    elif cmd == "iso":
        if len(a) < 4:
            _usage()
            return
        var n = atol(_opt(a, "--n", "48"))
        var g: SdfGrid
        if a[2] == "box":
            g = grid_from_box(n, Vec3(0.6, 0.4, 0.35), 1.0)
        elif a[2] == "torus":
            g = grid_from_torus(n, 0.6, 0.22, 1.0)
        else:
            g = grid_from_sphere(n, 0.75, 1.0)
        var dc = _flag(a, "--dc")
        var mesh: NativeMesh
        if _flag(a, "--sparse"):
            var sp = sparse_from_dense(g, 3 * 2.0 / Float32(n))
            mesh = dual_contouring_sparse(sp) if dc else marching_cubes_sparse(sp)
        else:
            mesh = dual_contouring(g) if dc else marching_cubes(g)
        var t = mesh_topology(mesh)
        var model = NativeModel()
        model.meshes.append(mesh^)
        _save(a[3], model)
        print('{"ok":true,"method":"' + ("dual_contouring" if dc else "marching_cubes") + '","triangles":' + String(t.triangles)
              + ',"watertight":' + ("true" if t.watertight() else "false") + ',"euler":' + String(t.euler) + "}")
    else:
        _usage()
