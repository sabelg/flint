"""Image -> normal map with Marigold-Normals v1.1 (pure Mojo on flint.ml.diffusion).

usage: marigold MODEL_DIR IN.png|.jpg OUT_normal.png [--steps N] [--ensemble E] [--res R] [--seed S]
                [--mean] [--uncertainty U.png] [--directx] [--tile] [--native] [--cpu]

MODEL_DIR is filled by `bash tools/marigold/fetch.sh MODEL_DIR`. Defaults are the model's own:
4 DDIM steps (1..50), 768 px processing resolution (longer side), one ensemble member. With
--ensemble E > 1 the members are ensembled per pixel (closest to the mean, or --mean); E >= 3 also
gives an uncertainty map (--uncertainty). --res 0 processes at the input size. The PNG is the
usual tangent-space encoding rgb = 0.5 + 0.5 * n with +Y up (OpenGL / glTF); --directx flips green.
--tile runs every convolution with wrap-around padding, for seamless maps of tileable textures.
--native keeps the processing resolution instead of resizing back to the input size.
Licence: CreativeML Open RAIL++-M (licenses/marigold-LICENSE-MODEL.txt): commercial use is
allowed; the Attachment A use restrictions apply to the outputs' use too.
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.platform.os import read_bytes
from flint.assets.jpeg import decode_image_auto
from flint.assets.image import write_png
from flint.ml.diffusion.nn import Runtime
from flint.ml.diffusion.imageproc import image_to_rgb
from flint.ml.marigold import (
    MarigoldNormals, normals_to_pixels, uncertainty_to_pixels, ENSEMBLE_CLOSEST, ENSEMBLE_MEAN,
)


def main() raises:
    var a = argv()
    if len(a) < 4:
        print("usage: marigold MODEL_DIR IN.png OUT_normal.png [--steps N] [--ensemble E] [--res R] [--seed S]"
              + " [--mean] [--uncertainty U.png] [--directx] [--tile] [--native] [--cpu]")
        return
    var model = String(a[1])
    var src = String(a[2])
    var dst = String(a[3])
    var steps = 0
    var ensemble = 1
    var res = -1
    var seed = 0
    var reduction = ENSEMBLE_CLOSEST
    var unc_path = String()
    var directx = False
    var tile = False
    var native = False
    var gpu = True
    var i = 4
    while i < len(a):
        var s = String(a[i])
        var has = i + 1 < len(a)
        if s == "--steps" and has:
            steps = Int(String(a[i + 1]))
            i += 1
        elif s == "--ensemble" and has:
            ensemble = Int(String(a[i + 1]))
            i += 1
        elif s == "--res" and has:
            res = Int(String(a[i + 1]))
            i += 1
        elif s == "--seed" and has:
            seed = Int(String(a[i + 1]))
            i += 1
        elif s == "--uncertainty" and has:
            unc_path = String(a[i + 1])
            i += 1
        elif s == "--mean":
            reduction = ENSEMBLE_MEAN
        elif s == "--directx":
            directx = True
        elif s == "--tile":
            tile = True
        elif s == "--native":
            native = True
        elif s == "--cpu":
            gpu = False
        i += 1
    var img = decode_image_auto(read_bytes(src), 8192)
    if not img.valid:
        raise Error("marigold: cannot decode " + src)
    var rgb = image_to_rgb(img)
    var rt = Runtime(gpu)
    var t0 = perf_counter_ns()
    var mg = MarigoldNormals.load(rt, model)
    mg.core.tile = tile
    print("[marigold] loaded on", rt.device_name(), "in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    t0 = perf_counter_ns()
    var r = mg.predict(rt, rgb, img.height, img.width, steps, ensemble, res, seed, reduction, not native)
    print("[marigold]", img.width, "x", img.height, "->", r.w, "x", r.h, "normals in",
          Float64(perf_counter_ns() - t0) / 1e9, "s (", steps if steps > 0 else mg.core.default_steps, "steps, ensemble",
          ensemble, ")")
    var px = normals_to_pixels(r.normals, r.w * r.h, directx)
    if not write_png(dst, px, r.w, r.h):
        raise Error("marigold: cannot write " + dst)
    print("[marigold] wrote", dst)
    if unc_path.byte_length() > 0:
        if len(r.uncertainty) == 0:
            print("[marigold] no uncertainty map: needs --ensemble 3 or more")
        else:
            var up = uncertainty_to_pixels(r.uncertainty, r.w * r.h)
            _ = write_png(unc_path, up, r.w, r.h)
            print("[marigold] wrote", unc_path)
