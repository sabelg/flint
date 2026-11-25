"""MagicArticulate / Puppeteer auto-rigging CLI: a static GLB in, a skinned GLB out.

usage: mojo run -I src examples/magicarticulate/main.mojo MODEL_DIR UNIRIG_DIR in.glb out.glb
           [--puppeteer | --spatial] [--cpu] [--seed N] [--no-watertight]

MODEL_DIR holds `<variant>_encoder.onnx` / `<variant>_opt.onnx` from `tools/magicarticulate/export.py`
(`magicarticulate` for the hierarchical checkpoint, also used by `--spatial` with the spatial one;
`puppeteer` with `--puppeteer`). The skeleton comes from that model; skin weights come from the
UniRig skin network in UNIRIG_DIR (`tools/unirig/export.py`), since neither release ships a
commercially usable skinning network.
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_writer import write_native_glb
from flint.ml.magicarticulate import MagicArticulate, rig_native_model_ma


def main() raises:
    var a = argv()
    if len(a) < 5:
        print("usage: magicarticulate MODEL_DIR UNIRIG_DIR in.glb out.glb [--puppeteer | --spatial] [--cpu] [--seed N] [--no-watertight]")
        return
    var gpu = True
    var variant = String("magicarticulate")
    var seed: UInt64 = 0
    var watertight = True
    var i = 5
    while i < len(a):
        var s = String(a[i])
        if s == "--cpu":
            gpu = False
        elif s == "--puppeteer":
            variant = "puppeteer"
        elif s == "--spatial":
            variant = "magicarticulate-spatial"
        elif s == "--no-watertight":
            watertight = False
        elif s == "--seed" and i + 1 < len(a):
            seed = UInt64(Int(String(a[i + 1])))
            i += 1
        i += 1
    var t0 = perf_counter_ns()
    var m = MagicArticulate.load(String(a[1]), variant, gpu, String(a[2]))
    m.verbose = True
    m.cfg.seed = seed
    m.cfg.watertight = watertight
    print("[" + variant + "] models loaded on", m.device(), "in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    var src = load_native_glb(String(a[3]))
    if len(src.meshes) == 0:
        raise Error("could not load a mesh from " + String(a[3]))
    t0 = perf_counter_ns()
    var out = rig_native_model_ma(m, src)
    print("[" + variant + "] rigged:", len(out.skins[0].joints), "joints,", len(out.meshes), "meshes in",
          Float64(perf_counter_ns() - t0) / 1e9, "s")
    if not write_native_glb(String(a[4]), out):
        raise Error("could not write " + String(a[4]))
    print("[" + variant + "] wrote", String(a[4]))
