"""SigLIP 2 CLI: text-to-image asset search, zero-shot labels, benchmarks, tokenizer dumps.

usage:
  main MODEL_DIR search ASSETS_DIR QUERY [QUERY ...] [--k N] [--cache FILE] [--thumbs DIR] [--cpu]
  main MODEL_DIR classify IMAGE.png LABEL [LABEL ...] [--cpu]
  main MODEL_DIR bench [--batch N] [--iters N] [--cpu]
  main TOKENIZER_FILE tokenize LINES_FILE          (ids per line; `\\t` / `\\n` escapes)
  main DIR demo-assets                             (bakes Flint's procedural GLB presets into DIR)

MODEL_DIR is a Hugging Face `google/siglip2-*` folder (tools/siglip/fetch.py). `search` indexes
every PNG / GLB under ASSETS_DIR (GLBs are rendered to thumbnails), keeps the embeddings in
`--cache` (default ASSETS_DIR/.siglip_<model>.flsg) so only new or changed files are embedded on
the next run, and prints the best matches for each query in any language, e.g.

  mojo run -I src examples/siglip/main.mojo ~/siglip2-base-patch16-224 search assets \\
      "wooden treasure chest" "rostiges Schwert" "赤い車"
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.platform.os import read_bytes
from flint.assets.image import decode_png
from flint.ml.sentencepiece import SentencePiece
from flint.ml.siglip.model import SiglipModel, sigmoid
from flint.ml.siglip.image import siglip_preprocess
from flint.ml.siglip.search import AssetIndex, ASSET_GLB
from flint.platform.os import make_dirs
from flint.studio.studio_app import bake_procedural_to_glb


def _median(var xs: List[Float64]) -> Float64:
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j] < xs[j - 1]:
            var t = xs[j]
            xs[j] = xs[j - 1]
            xs[j - 1] = t
            j -= 1
    return xs[len(xs) // 2]


def _short(x: Float32) -> String:
    """`x` with 4 decimals (no exponent)."""
    var v = Int(abs(Float64(x)) * 10000.0 + 0.5)
    var frac = String(v % 10000)
    while frac.byte_length() < 4:
        frac = "0" + frac
    return ("-" if x < 0 and v > 0 else "") + String(v // 10000) + "." + frac


def _tokenize(path: String, lines_file: String) raises:
    var sp = SentencePiece.load(path)
    var data = read_bytes(lines_file)
    var cur = List[UInt8]()
    data.append(10)
    for c in data:
        if c != 10:
            cur.append(c)
            continue
        var line = String(unsafe_from_utf8=cur).replace("\\t", "\t").replace("\\n", "\n")
        cur = List[UInt8]()
        var ids = sp.encode(line)
        var s = String("")
        for k in range(len(ids)):
            s += (" " if k > 0 else "") + String(ids[k])
        print(s)


def main() raises:
    var a = argv()
    if len(a) < 3:
        print("usage: main MODEL_DIR search|classify|bench ... | main TOKENIZER_FILE tokenize LINES_FILE")
        return
    var model_dir = String(a[1])
    var cmd = String(a[2])
    if cmd == "tokenize":
        _tokenize(model_dir, String(a[3]))
        return
    if cmd == "demo-assets":
        _ = make_dirs(model_dir)
        for preset in ["material_orb", "catmull_torus", "colonnade", "reference_grid", "pine", "oak", "bollard"]:
            var r = bake_procedural_to_glb(preset, model_dir + "/" + preset + ".glb", 1)
            print(preset + ".glb: " + String(r[2]) + " triangles" + ("" if r[0] else " (failed)"))
        return
    var gpu = True
    var k = 5
    var batch = 16
    var iters = 5
    var cache = String("")
    var thumbs = String("")
    var pos = List[String]()
    var i = 3
    while i < len(a):
        var s = String(a[i])
        if s == "--cpu":
            gpu = False
        elif s == "--k" and i + 1 < len(a):
            k = Int(String(a[i + 1]))
            i += 1
        elif s == "--batch" and i + 1 < len(a):
            batch = Int(String(a[i + 1]))
            i += 1
        elif s == "--iters" and i + 1 < len(a):
            iters = Int(String(a[i + 1]))
            i += 1
        elif s == "--cache" and i + 1 < len(a):
            cache = String(a[i + 1])
            i += 1
        elif s == "--thumbs" and i + 1 < len(a):
            thumbs = String(a[i + 1])
            i += 1
        else:
            pos.append(s)
        i += 1
    var t0 = perf_counter_ns()
    var m = SiglipModel.load(model_dir, gpu)
    print("[siglip] " + m.name + " on " + m.rt.device_name() + ", loaded in "
          + String(Float64(perf_counter_ns() - t0) / 1e9) + " s")

    if cmd == "search":
        if len(pos) < 2:
            print("search needs ASSETS_DIR and at least one query")
            return
        var root = pos[0]
        if cache == "":
            cache = root + "/.siglip_" + m.name + ".flsg"
        var index = AssetIndex.open(cache, m)
        var rep = index.update(m, root, batch, thumbs)
        _ = index.save(cache)
        print("[siglip] " + String(rep.scanned) + " assets: " + String(rep.embedded) + " embedded, "
              + String(rep.reused) + " cached, " + String(rep.removed) + " removed, " + String(rep.failed)
              + " unreadable (" + String(rep.seconds) + " s) -> " + cache)
        for q in range(1, len(pos)):
            var tq = perf_counter_ns()
            var hits = index.search(m, pos[q], k)
            print("\n" + pos[q] + "   (" + String(Float64(perf_counter_ns() - tq) / 1e6) + " ms)")
            for h in hits:
                print("  " + _short(h.prob) + "  logit " + _short(h.logit) + "  "
                      + ("[glb] " if h.kind == ASSET_GLB else "") + h.path)
        return

    if cmd == "classify":
        if len(pos) < 2:
            print("classify needs IMAGE.png and labels")
            return
        var img = decode_png(read_bytes(pos[0]), 1 << 14)
        if img.width <= 0:
            print("cannot decode " + pos[0] + " (8-bit RGB / RGBA PNG expected)")
            return
        var ie = m.embed_pixels(siglip_preprocess(img, m.image_size()), 1)
        var labels = List[String]()
        for q in range(1, len(pos)):
            labels.append(pos[q])
        var te = m.embed_texts(labels)
        var lg = m.logits(te, ie)
        for q in range(len(labels)):
            print("  " + _short(sigmoid(lg[q])) + "  " + labels[q])
        return

    if cmd == "bench":
        var S = m.image_size()
        var px = List[Float32](length=batch * S * S * 3, fill=0.0)
        for j in range(len(px)):
            px[j] = Float32((j * 7919) % 255) / 127.5 - 1.0
        var texts = List[String]()
        for j in range(batch):
            texts.append("a photo of asset number " + String(j))
        _ = m.embed_pixels(px, batch)       # warm-up (GPU kernel load, weight upload)
        _ = m.embed_texts(texts)
        var tv = List[Float64]()
        var tt = List[Float64]()
        for _ in range(iters):
            var s0 = perf_counter_ns()
            _ = m.embed_pixels(px, batch)
            m.rt.sync()
            var s1 = perf_counter_ns()
            _ = m.embed_texts(texts)
            m.rt.sync()
            var s2 = perf_counter_ns()
            tv.append(Float64(s1 - s0) / 1e6)
            tt.append(Float64(s2 - s1) / 1e6)
        var mv = _median(tv^)
        var mt = _median(tt^)
        print("[siglip] batch " + String(batch) + ": images " + String(mv) + " ms (" + String(mv / Float64(batch))
              + " ms/image), texts " + String(mt) + " ms (" + String(mt / Float64(batch)) + " ms/text)")
        return
    print("unknown command " + cmd)
