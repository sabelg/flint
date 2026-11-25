"""Make-It-Animatable auto-rigging CLI: a static humanoid GLB in, a skinned GLB out (52-joint
Mixamo skeleton `mixamorig:*`, 4 weights per vertex), by default re-posed into the predicted T-pose.

usage: mojo run -I src examples/make_it_animatable/main.mojo MODEL_DIR in.glb out.glb
           [--cpu] [--input-pose] [--seed N] [--clip motion.bvh] [--png]

  MODEL_DIR     the five graphs written by tools/make_it_animatable/export.py
  --input-pose  keep the mesh in its input pose and placement (bind there) instead of the T-pose
  --clip        retarget a SOMA (MotionBricks / GEM-X) or Mixamo-named BVH onto the rig and store
                it as the GLB's animation (T-pose output only)
  --png         keep textures lossless (PNG); by default they are written as JPEG (quality 94),
                since re-encoded PNGs can make the file many times larger than its input
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_writer import encode_native_glb
from flint.platform.os import write_bytes
from flint.ml.motionbricks import load_bvh
from flint.ml.make_it_animatable import MakeItAnimatable, rig_native_model, retarget_clip


def main() raises:
    var a = argv()
    if len(a) < 4:
        print("usage: make_it_animatable MODEL_DIR in.glb out.glb [--cpu] [--input-pose] [--seed N] [--clip motion.bvh] [--png]")
        return
    var gpu = True
    var rest = True
    var seed: UInt64 = 0
    var clip_path = String()
    var png = False
    var i = 4
    while i < len(a):
        var s = String(a[i])
        if s == "--cpu":
            gpu = False
        elif s == "--input-pose":
            rest = False
        elif s == "--seed" and i + 1 < len(a):
            seed = UInt64(Int(String(a[i + 1])))
            i += 1
        elif s == "--png":
            png = True
        elif s == "--clip" and i + 1 < len(a):
            clip_path = String(a[i + 1])
            i += 1
        i += 1
    var t0 = perf_counter_ns()
    var mia = MakeItAnimatable.load(String(a[1]), gpu)
    mia.verbose = True
    print("[make-it-animatable] models loaded on", mia.device(), "in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    var src = load_native_glb(String(a[2]))
    if len(src.meshes) == 0:
        raise Error("could not load a mesh from " + String(a[2]))
    t0 = perf_counter_ns()
    var out = rig_native_model(mia, src, rest, seed)
    print("[make-it-animatable] rigged:", len(out.skins[0].joints), "joints,", len(out.meshes), "meshes in",
          Float64(perf_counter_ns() - t0) / 1e9, "s", "(T-pose)" if rest else "(input pose)")
    if clip_path.byte_length() > 0:
        if not rest:
            raise Error("--clip needs the T-pose output (drop --input-pose)")
        var clip = load_bvh(clip_path)
        out.animations.append(retarget_clip(clip, out, clip_path))
        print("[make-it-animatable] retargeted", clip.num_frames(), "frames from", clip_path)
    var mask: UInt64 = 0 if png else UInt64(0xFFFFFFFFFFFFFFFF)
    var bytes = encode_native_glb(out, mask)
    if not write_bytes(String(a[3]), bytes):
        raise Error("could not write " + String(a[3]))
    print("[make-it-animatable] wrote", String(a[3]))
