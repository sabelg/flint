"""UniRig auto-rigging CLI: a static GLB in, a skinned GLB out (skeleton + 4 weights per vertex).

usage: mojo run -I src examples/unirig/main.mojo MODEL_DIR in.glb out.glb [--cpu] [--greedy] [--seed N]
MODEL_DIR holds the five converted graphs written by `tools/unirig/export.py`.
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_writer import write_native_glb
from flint.ml.unirig import UrGenConfig
from flint.ml.unirig_rig import UniRig, rig_native_model


def main() raises:
    var a = argv()
    if len(a) < 4:
        print("usage: unirig MODEL_DIR in.glb out.glb [--cpu] [--greedy] [--seed N]")
        return
    var gpu = True
    var cfg = UrGenConfig()
    var seed: UInt64 = 0
    var i = 4
    while i < len(a):
        var s = String(a[i])
        if s == "--cpu":
            gpu = False
        elif s == "--greedy":
            cfg = UrGenConfig.greedy()
        elif s == "--seed" and i + 1 < len(a):
            seed = UInt64(Int(String(a[i + 1])))
            i += 1
        i += 1
    cfg.seed = seed
    var t0 = perf_counter_ns()
    var rig = UniRig.load(String(a[1]), gpu)
    rig.verbose = True
    print("[unirig] models loaded on", rig.device(), "in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    var src = load_native_glb(String(a[2]))
    if len(src.meshes) == 0:
        raise Error("could not load a mesh from " + String(a[2]))
    t0 = perf_counter_ns()
    var out = rig_native_model(rig, src, cfg)
    print("[unirig] rigged:", len(out.skins[0].joints), "joints,", len(out.meshes), "meshes in",
          Float64(perf_counter_ns() - t0) / 1e9, "s")
    if not write_native_glb(String(a[3]), out):
        raise Error("could not write " + String(a[3]))
    print("[unirig] wrote", String(a[3]))
