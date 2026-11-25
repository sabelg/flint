"""Monocular depth / geometry CLI for `flint.ml.depth` (Depth Anything V2 / 3, MoGe-2).

usage: mojo run -I src examples/depth/main.mojo MODEL_DIR OUT_PREFIX IMAGE [IMAGE ...]
           [--cpu] [--glb mesh|points|none] [--step 2] [--jump 0.06] [--focal PX] [--grey]
           [--parity REF_DIR] [--bench N]

MODEL_DIR is a Hugging Face folder: depth-anything/Depth-Anything-V2-Small-hf, depth-anything/
DA3-SMALL | DA3-BASE | DA3MONO-LARGE | DA3METRIC-LARGE, or Ruicheng/moge-2-vit{s,b,l}-normal.
Several images are views of one scene (DA3-SMALL / DA3-BASE only: cross-view attention, one
camera per view, one fused world-space GLB).

Writes per view: OUT_PREFIX[_vN]_depth.png (Turbo inverse depth), _depth.exr (float "Z"),
_normal.png / _normal.exr (MoGe), _points.exr (camera-space XYZ) and OUT_PREFIX.glb (a vertex-
coloured mesh blockout, or a point cloud of tiny triangles). `--focal` turns DA3METRIC-LARGE
output into metres (focal length in pixels of the original image). `--parity` compares against
a `tools/depth/reference.py` dump; `--bench N` times N extra forward passes.
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.ml.depth.model import DepthModel, DepthResult, FAMILY_DA2, FAMILY_DA3, FAMILY_MOGE
from flint.ml.depth.image import Rgb8, load_rgb, da3_resize
from flint.ml.depth.export import (
    write_depth_png, write_depth_exr, write_normal_png, write_vec3_exr, camera_points, to_world, ColoredMesh,
    append_depth_mesh, append_point_splats, write_glb,
)
from flint.ml.depth.parity import check_parity, check_preprocess


def _secs(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e9


def main() raises:
    var a = argv()
    if len(a) < 4:
        print("usage: depth MODEL_DIR OUT_PREFIX IMAGE [IMAGE ...] [--cpu] [--glb mesh|points|none] [--step N]"
              " [--jump F] [--focal PX] [--grey] [--parity REF_DIR] [--bench N]")
        return
    var model_dir = String(a[1])
    var prefix = String(a[2])
    var images = List[String]()
    var gpu = True
    var glb = String("mesh")
    var step = 2
    var jump: Float32 = 0.06
    var focal = 0.0
    var grey = False
    var parity = String("")
    var bench = 0
    var i = 3
    while i < len(a):
        var s = String(a[i])
        var has_v = i + 1 < len(a)
        if s == "--cpu":
            gpu = False
        elif s == "--grey":
            grey = True
        elif s == "--glb" and has_v:
            glb = String(a[i + 1])
            i += 1
        elif s == "--step" and has_v:
            step = max(1, Int(String(a[i + 1])))
            i += 1
        elif s == "--jump" and has_v:
            jump = Float32(Float64(String(a[i + 1])))
            i += 1
        elif s == "--focal" and has_v:
            focal = Float64(String(a[i + 1]))
            i += 1
        elif s == "--parity" and has_v:
            parity = String(a[i + 1])
            i += 1
        elif s == "--bench" and has_v:
            bench = Int(String(a[i + 1]))
            i += 1
        else:
            images.append(s)
        i += 1
    var t0 = perf_counter_ns()
    var m = DepthModel.load(model_dir, gpu)
    print("[depth]", m.name, "on", m.device(), "loaded in", _secs(t0), "s")
    if parity != "":
        var ds = check_parity(m, parity)
        for d in ds:
            print("[depth] parity", d.line())
        if len(images) > 0:
            print("[depth] parity", check_preprocess(m, parity, images[0]).line())
        return
    var imgs = List[Rgb8]()
    for p in images:
        imgs.append(load_rgb(p))
    var t1 = perf_counter_ns()
    var r = m.predict_views(imgs, focal)
    print("[depth]", len(imgs), "view(s)", r.w, "x", r.h, "in", _secs(t1), "s")
    for _ in range(bench):
        var tb = perf_counter_ns()
        _ = m.predict_views(imgs, focal)
        print("[depth] bench forward", _secs(tb), "s")
    if r.has_camera and r.family == FAMILY_DA3:
        print("[depth] reference view", r.ref_view)
        for v in range(r.views):
            print("[depth] view", v, "fx", r.intrinsics[9 * v], "fy", r.intrinsics[9 * v + 4], "t",
                  r.extrinsics[12 * v + 3], r.extrinsics[12 * v + 7], r.extrinsics[12 * v + 11])
    if r.family == FAMILY_MOGE:
        print("[depth] fx", r.intrinsics[0], "fy", r.intrinsics[4], "metric", r.metric, "scale", r.scale)
    var cm = ColoredMesh()
    for v in range(r.views):
        var tag = prefix if r.views == 1 else prefix + "_v" + String(v)
        write_depth_png(tag + "_depth.png", r, v, grey)
        write_depth_exr(tag + "_depth.exr", r, v)
        if len(r.normal) == r.w * r.h * 3 * r.views:
            write_normal_png(tag + "_normal.png", r.normal, r.w, r.h, v * r.w * r.h * 3)
            write_vec3_exr(tag + "_normal.exr", r.normal, r.w, r.h, v)
        var pts = camera_points(r, v)
        write_vec3_exr(tag + "_points.exr", pts, r.w, r.h)
        var col = imgs[v].copy() if r.family != FAMILY_DA3 else da3_resize(imgs[v], m.input_size)
        var wp = to_world(pts^, r, v)
        if glb == "points":
            append_point_splats(cm, wp, col, r.w, r.h, step)
        elif glb == "mesh":
            append_depth_mesh(cm, wp, col, r.w, r.h, step, jump)
    if glb != "none":
        write_glb(prefix + ".glb", cm, m.name)
        print("[depth] wrote", prefix + ".glb", cm.mesh.vertex_count(), "vertices", cm.mesh.triangle_count(), "triangles")
    print("[depth] done in", _secs(t0), "s")
