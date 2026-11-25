"""Text / photo -> tileable PBR material (pure-Mojo StableMaterials / MatFuse on flint.ml.diffusion).

usage: materials WEIGHTS_DIR OUT_DIR "prompt" [--model stablematerials|matfuse] [--image photo.png|.jpg]
                 [--name NAME] [--steps N] [--seed S] [--guidance G] [--no-tile] [--palette RRGGBB,...]
                 [--sketch edges.png] [--repeat N] [--cpu]

WEIGHTS_DIR is filled by `python tools/diffusion/fetch.py WEIGHTS_DIR` (stablematerials/, matfuse/).
Writes OUT_DIR/NAME_{basecolor,normal,height,roughness,metallic,orm}.png, NAME.glb (a textured quad
the renderer and any glTF viewer load; --repeat tiles the UVs) and NAME.material.json.

StableMaterials (default): LCM UNet, 4 steps, guidance 10, 512 x 512, tileable. A photo (--image)
replaces the prompt as the condition. MatFuse: 20 DDIM steps, CFG 4, 256 x 256; text, photo,
palette and sketch conditions combine freely. Model licences: licenses/stablematerials-LICENSE.txt
(OpenRAIL-M use restrictions apply to the outputs' use too) and licenses/matfuse-LICENSE.txt.
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.platform.os import read_bytes
from flint.assets.jpeg import decode_image_auto
from flint.ml.diffusion.nn import Runtime
from flint.ml.diffusion.imageproc import image_to_rgb, resize_bicubic_aa
from flint.ml.diffusion.materials import StableMaterials, MatFuse, PbrMaps, write_material


def _hex(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    return 0


def _palette(spec: String) -> List[Float32]:
    var out = List[Float32]()
    for part in spec.split(","):
        var b = String(part).strip().as_bytes()
        var o = 1 if len(b) == 7 else 0
        if len(b) - o != 6:
            continue
        for k in range(3):
            out.append(Float32(_hex(Int(b[o + 2 * k])) * 16 + _hex(Int(b[o + 2 * k + 1]))) / 255.0)
    while len(out) > 0 and len(out) < 15:
        out.append(out[len(out) - 3])
    return out^


def main() raises:
    var a = argv()
    if len(a) < 4:
        print('usage: materials WEIGHTS_DIR OUT_DIR "prompt" [--model stablematerials|matfuse] [--image photo]'
              + " [--name NAME] [--steps N] [--seed S] [--guidance G] [--no-tile] [--palette RRGGBB,...]"
              + " [--sketch edges.png] [--repeat N] [--cpu]")
        return
    var weights = String(a[1])
    var out_dir = String(a[2])
    var prompt = String(a[3])
    var model = String("stablematerials")
    var image_path = String()
    var sketch_path = String()
    var name = String("material")
    var steps = -1
    var seed = 42
    var guidance: Float32 = -1.0
    var tile = True
    var gpu = True
    var palette = List[Float32]()
    var repeat = 1
    var i = 4
    while i < len(a):
        var s = String(a[i])
        var has = i + 1 < len(a)
        if s == "--model" and has:
            model = String(a[i + 1])
            i += 1
        elif s == "--image" and has:
            image_path = String(a[i + 1])
            i += 1
        elif s == "--sketch" and has:
            sketch_path = String(a[i + 1])
            i += 1
        elif s == "--name" and has:
            name = String(a[i + 1])
            i += 1
        elif s == "--steps" and has:
            steps = Int(String(a[i + 1]))
            i += 1
        elif s == "--seed" and has:
            seed = Int(String(a[i + 1]))
            i += 1
        elif s == "--guidance" and has:
            guidance = Float32(Float64(String(a[i + 1])))
            i += 1
        elif s == "--palette" and has:
            palette = _palette(String(a[i + 1]))
            i += 1
        elif s == "--repeat" and has:
            repeat = max(1, Int(String(a[i + 1])))
            i += 1
        elif s == "--no-tile":
            tile = False
        elif s == "--cpu":
            gpu = False
        i += 1

    var rgb = List[Float32]()
    var ih = 0
    var iw = 0
    if image_path.byte_length() > 0:
        var img = decode_image_auto(read_bytes(image_path), 4096)
        if not img.valid:
            raise Error("materials: cannot decode " + image_path)
        rgb = image_to_rgb(img)
        ih = img.height
        iw = img.width

    var rt = Runtime(gpu)
    var t0 = perf_counter_ns()
    var maps: PbrMaps
    if model == "matfuse":
        var mf = MatFuse.load(rt, weights + "/matfuse")
        print("[materials] MatFuse loaded on", rt.device_name(), "in", Float64(perf_counter_ns() - t0) / 1e9, "s")
        var sketch = List[Float32]()
        if sketch_path.byte_length() > 0:
            var sk = decode_image_auto(read_bytes(sketch_path), 4096)
            if not sk.valid:
                raise Error("materials: cannot decode " + sketch_path)
            var g = image_to_rgb(sk)
            var gray = List[Float32](length=sk.width * sk.height, fill=0.0)
            for p in range(len(gray)):
                gray[p] = 0.299 * g[p * 3] + 0.587 * g[p * 3 + 1] + 0.114 * g[p * 3 + 2]
            sketch = resize_bicubic_aa(gray, sk.height, sk.width, 1, 256, 256, False)
        var text = String() if prompt == "-" else prompt
        t0 = perf_counter_ns()
        maps = mf.generate(rt, text, rgb, ih, iw, palette, sketch, steps if steps > 0 else 20,
                           seed, guidance if guidance > 0 else 4.0, tile)
    else:
        var sm = StableMaterials.load(rt, weights + "/stablematerials")
        print("[materials] StableMaterials loaded on", rt.device_name(), "in", Float64(perf_counter_ns() - t0) / 1e9, "s")
        t0 = perf_counter_ns()
        var cond = sm.image_condition(rt, rgb, ih, iw) if len(rgb) > 0 else sm.text_condition(rt, prompt)
        maps = sm.generate(rt, cond^, steps if steps > 0 else 4, seed, guidance if guidance > 0 else 10.0, tile)
    print("[materials] generated", maps.size, "x", maps.size, "in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    var glb = write_material(maps, out_dir, name, "MatFuse" if model == "matfuse" else "StableMaterials",
                             prompt if len(rgb) == 0 else image_path, repeat)
    print("[materials] wrote", glb, "and", out_dir + "/" + name + "_*.png")
