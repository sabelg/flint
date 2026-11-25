"""Flint v5 SigLIP 2 suite (`test_flint_v5_siglip.mojo`).

Always (no weights):
  1. `SentencePiece` from synthetic `tokenizer.json` files: Unigram Viterbi, unknown-character
     fusion, byte fallback, BPE merge ranks, Metaspace prefixes, added tokens, `encode_fixed`
     truncation / padding, `decode`; the GELU-tanh activation.
  2. PIL-exact resizing: BILINEAR / BICUBIC 8-bit outputs against values from Pillow 13,
     RGBA compositing over white, SigLIP normalisation.
  3. GLB thumbnails: a procedural GLB renders with the object centred on the backdrop.
  4. `AssetIndex`: add / replace / remove, top-k ranking with the sigmoid head, cache round trip.
  5. MCP v2 `assets.semantic_search`: listed in tools/list, sandboxed paths, a clear error with no
     model; with weights, a search through the tool over a workspace copy of the images.
With `FLINT_SIGLIP_DIR` (a Hugging Face folder from tools/siglip/fetch.py) and
`FLINT_SIGLIP_REF` (tools/siglip/reference.py output for that folder): tokenizer ids
(`tokenizer.model` and `tokenizer.json`), pixel values, vision / text pooled embeddings,
hidden states and logits against Transformers, plus an asset search over the reference images.
`FLINT_SIGLIP_GPU=0` forces the CPU.
"""
from std.math import sqrt, tanh
from std.os import getenv
from std.time import perf_counter_ns
from std.testing import assert_true, assert_equal
from flint.platform.os import read_bytes, write_bytes, make_dirs
from flint.assets.json import parse_json
from flint.assets.image import NativeImage, decode_png
from flint.assets.glb_native import load_native_glb
from flint.studio.studio_app import bake_procedural_to_glb
from flint.ml.sentencepiece import SentencePiece
from flint.ml.diffusion.nn import Runtime
from flint.ml.diffusion.kernels import ACT_GELU_TANH
from flint.ml.siglip.model import SiglipModel, l2_normalize, sigmoid
from flint.ml.siglip.image import (
    pil_resize_rgb8, image_rgb8, siglip_preprocess, siglip_preprocess_rgb8, render_glb_thumbnail,
    RESAMPLE_BILINEAR, RESAMPLE_BICUBIC,
)
from flint.ml.siglip.search import AssetIndex, ASSET_PNG, ASSET_GLB, load_asset_image
from flint.devtools.mcp_v2 import McpServerV2
from flint.studio.studio_app import FlintStudioState
from flint.assets.server import AssetServer
from flint.core.jobs import JobPool


def _ids_str(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        if i > 0:
            s += ","
        s += String(ids[i])
    return s + "]"


def _expect_ids(got: List[Int], want: List[Int], what: String) raises:
    var ok = len(got) == len(want)
    if ok:
        for i in range(len(got)):
            if got[i] != want[i]:
                ok = False
    if not ok:
        raise Error(what + ": got " + _ids_str(got) + " expected " + _ids_str(want))


def _write(path: String, text: String) raises:
    var b = List[UInt8](text.as_bytes())
    if not write_bytes(path, b):
        raise Error("cannot write " + path)


# ----------------------------------------------------------------------------- 1. tokenizer

comptime UNIGRAM_JSON = '{"added_tokens":[{"id":0,"content":"<pad>","special":true},{"id":1,"content":"</s>","special":true},{"id":2,"content":"<unk>","special":true},{"id":9,"content":"<tag>","special":false}],"normalizer":{"type":"Sequence","normalizers":[{"type":"Replace","pattern":{"Regex":" {2,}"},"content":" "}]},"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},"model":{"type":"Unigram","unk_id":2,"byte_fallback":false,"vocab":[["<pad>",0.0],["</s>",0.0],["<unk>",0.0],["▁",-2.0],["▁ab",-3.0],["▁a",-2.5],["b",-2.5],["c",-1.0],["abc",-1.5],["<tag>",0.0]]}}'

comptime BPE_JSON = '{"added_tokens":[{"id":0,"content":"<pad>","special":true},{"id":1,"content":"<eos>","special":true},{"id":2,"content":"<unk>","special":true}],"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},"pre_tokenizer":null,"model":{"type":"BPE","unk_token":"<unk>","byte_fallback":true,"fuse_unk":true,"vocab":{"<pad>":0,"<eos>":1,"<unk>":2,"<0x21>":3,"<0xC3>":4,"<0xA9>":5,"a":6,"b":7,"c":8,"▁":9,"ab":10,"bc":11,"abc":12,"▁a":13},"merges":[["b","c"],["a","b"],["a","bc"],["▁","a"]]}}'


def test_tokenizer() raises:
    var dir = getenv("TMPDIR", "/tmp") + "/flint_siglip_tok"
    _ = make_dirs(dir)
    _write(dir + "/uni.json", UNIGRAM_JSON)
    var u = SentencePiece.load(dir + "/uni.json")
    assert_equal(u.eos_id, 1)
    assert_equal(u.pad_id, 0)
    # "ab": ▁ab (-3.0) beats ▁a + b (-5.0)
    _expect_ids(u.encode("ab"), [4], "unigram ▁ab")
    # "abc": ▁ + abc (-3.5) beats ▁ab + c (-4.0) and ▁a + b + c (-6.0)
    _expect_ids(u.encode("abc"), [3, 8], "unigram ▁ abc")
    # two words, collapsed spaces, Metaspace split
    _expect_ids(u.encode("ab   c"), [4, 3, 7], "unigram two words")
    # unknown characters fuse into one <unk>; added tokens are matched verbatim
    _expect_ids(u.encode("zz"), [3, 2], "unigram fused unk")
    _expect_ids(u.encode("ab<tag>"), [4, 9], "added token")
    _expect_ids(u.encode_fixed("ab", 4), [4, 1, 0, 0], "fixed length + eos + pad")
    _expect_ids(u.encode_fixed("ab ab ab", 3), [4, 4, 1], "truncation keeps eos")
    assert_true(u.decode([4, 3, 7]) == "ab c", "unigram decode: '" + u.decode([4, 3, 7]) + "'")

    _write(dir + "/bpe.json", BPE_JSON)
    var b = SentencePiece.load(dir + "/bpe.json")
    assert_equal(b.eos_id, 1)
    # rank order: b+c first, then a+bc -> abc (not ab + c)
    _expect_ids(b.encode("abc"), [12], "bpe abc")
    _expect_ids(b.encode("ab"), [10], "bpe ab")
    _expect_ids(b.encode("a abc"), [6, 9, 12], "bpe a+bc outranks ▁+a")
    _expect_ids(b.encode("a a"), [6, 13], "bpe ▁a")
    # byte fallback: é = C3 A9 -> <0xC3> <0xA9>; '!' -> <0x21>; 'z' has no byte piece -> <unk>
    _expect_ids(b.encode("é!"), [4, 5, 3], "bpe byte fallback")
    _expect_ids(b.encode("z"), [2], "bpe unk")
    assert_true(b.decode([6, 9, 4, 5]) == "a é", "bpe decode with bytes: '" + b.decode([6, 9, 4, 5]) + "'")
    # GELU tanh (gelu_pytorch_tanh) through the CPU runtime
    var rt = Runtime(False)
    var x = rt.upload(1, 1, 1, 4, [-2.0, -0.5, 0.5, 3.0])
    rt.activate(x, ACT_GELU_TANH)
    var y = rt.download(x)
    var want: List[Float64] = [-0.04540230591222, -0.15428599017, 0.34571400982, 2.996362607918227]
    for i in range(4):
        assert_true(abs(Float64(y[i]) - want[i]) < 1e-5, "gelu tanh " + String(i) + ": " + String(y[i]))
    print("✓ SentencePiece unigram / BPE / byte fallback / added tokens; GELU-tanh")


# ----------------------------------------------------------------------------- 2. image

def test_resize() raises:
    # 5 x 3 RGB ramp, values from Pillow 13 `Image.resize`
    var src = List[UInt8]()
    for y in range(3):
        for x in range(5):
            src.append(UInt8(x * 50 + y * 7))
            src.append(UInt8(255 - x * 40))
            src.append(UInt8((x * 31 + y * 53) % 256))
    var bl = pil_resize_rgb8(src, 5, 3, 2, 2, RESAMPLE_BILINEAR)
    var want_bl: List[Int] = [53, 215, 51, 153, 135, 113, 61, 215, 117, 161, 135, 179]
    for i in range(12):
        assert_true(Int(bl[i]) == want_bl[i], "bilinear down " + String(i) + ": " + String(bl[i]))
    var up = pil_resize_rgb8(src, 5, 3, 7, 4, RESAMPLE_BILINEAR)
    var want_up: List[Int] = [0, 255, 0, 29, 232, 18, 64, 204, 40, 100, 175, 62]
    for i in range(12):
        assert_true(Int(up[i]) == want_up[i], "bilinear up " + String(i) + ": " + String(up[i]))
    var bc = pil_resize_rgb8(src, 5, 3, 3, 2, RESAMPLE_BICUBIC)
    var want_bc: List[Int] = [20, 241, 27, 102, 175, 78, 184, 109, 129, 30, 241, 101, 112, 175, 152, 194, 109, 203]
    for i in range(18):
        assert_true(Int(bc[i]) == want_bc[i], "bicubic " + String(i) + ": " + String(bc[i]))
    # identity size is a copy; a constant image stays constant
    var same = pil_resize_rgb8(src, 5, 3, 5, 3)
    for i in range(len(src)):
        assert_equal(Int(same[i]), Int(src[i]))
    var flat = List[UInt8](length=40 * 30 * 3, fill=77)
    var f2 = pil_resize_rgb8(flat, 40, 30, 13, 9)
    for i in range(len(f2)):
        assert_equal(Int(f2[i]), 77)
    # RGBA over white (Pillow alpha_composite), then SigLIP normalisation (x / 255 - 0.5) / 0.5
    var img = NativeImage()
    img.width = 2
    img.height = 1
    img.pixels = [UInt32(0x80FF0000), UInt32(0x00000000)]
    var rgb = image_rgb8(img)
    var want_rgb: List[Int] = [255, 127, 127, 255, 255, 255]
    for i in range(6):
        assert_true(Int(rgb[i]) == want_rgb[i], "alpha composite " + String(i) + ": " + String(rgb[i]))
    var px = siglip_preprocess_rgb8([UInt8(0), UInt8(255), UInt8(128)], 1, 1, 1)
    assert_true(abs(px[0] + 1.0) < 1e-7 and abs(px[1] - 1.0) < 1e-7 and abs(px[2] - 0.003921628) < 1e-6,
                "normalisation")
    print("✓ PIL-exact BILINEAR / BICUBIC, RGBA over white, SigLIP normalisation")


def test_thumbnail() raises:
    var dir = getenv("TMPDIR", "/tmp") + "/flint_siglip_thumb"
    _ = make_dirs(dir)
    var path = dir + "/orb.glb"
    var baked = bake_procedural_to_glb("material_orb", path, 3)
    assert_true(baked[0], "procedural GLB baked")
    var model = load_native_glb(path)
    var img = render_glb_thumbnail(model, 128)
    assert_equal(img.width, 128)
    assert_equal(len(img.pixels), 128 * 128)
    var bg = img.pixels[0]
    var fg = 0
    var cx: Float64 = 0
    var cy: Float64 = 0
    for y in range(128):
        for x in range(128):
            if img.pixels[y * 128 + x] != bg:
                fg += 1
                cx += Float64(x)
                cy += Float64(y)
    assert_true(fg > 1500 and fg < 128 * 128 - 1500, "object covers part of the frame: " + String(fg))
    cx /= Float64(fg)
    cy /= Float64(fg)
    assert_true(abs(cx - 64.0) < 16.0 and abs(cy - 64.0) < 16.0, "object centred: " + String(cx) + "," + String(cy))
    var kinds = load_asset_image(path, ASSET_GLB, dir + "/thumbs", "orb.glb")
    assert_equal(kinds.width, 448)
    assert_true(len(read_bytes(dir + "/thumbs/orb.glb.png")) > 0, "thumbnail PNG written")
    print("✓ GLB thumbnail render (" + String(fg) + " object pixels)")


# ----------------------------------------------------------------------------- 4. index

def _unit(a: Float32, b: Float32, c: Float32) -> List[Float32]:
    var v: List[Float32] = [a, b, c]
    l2_normalize(v, 3)
    return v^


def test_index() raises:
    var idx = AssetIndex("toy", 3, 224, 2.0, -1.0)
    idx.add("red.png", ASSET_PNG, 10, 100, _unit(1, 0, 0))
    idx.add("green.png", ASSET_PNG, 11, 101, _unit(0, 1, 0))
    idx.add("tree.glb", ASSET_GLB, 12, 102, _unit(0.2, 1, 0.1))
    idx.add("blue.png", ASSET_PNG, 13, 103, _unit(0, 0, 1))
    assert_equal(idx.count(), 4)
    var q = _unit(0.1, 1, 0)
    var hits = idx.search_embedding(q, 2)
    assert_equal(len(hits), 2)
    assert_true(hits[0].path == "green.png" and hits[1].path == "tree.glb", "top-2 order: " + hits[0].path + " " + hits[1].path)
    var cosv: Float64 = 0
    for d in range(3):
        cosv += Float64(q[d]) * Float64(_unit(0, 1, 0)[d])
    var want = cosv * 7.38905609893065 - 1.0
    assert_true(abs(Float64(hits[0].logit) - want) < 1e-4, "sigmoid-head logit")
    assert_true(abs(hits[0].prob - sigmoid(hits[0].logit)) < 1e-7, "probability")
    idx.add("green.png", ASSET_PNG, 20, 200, _unit(0, 0, 1))      # replaced in place
    assert_equal(idx.count(), 4)
    idx.remove_at(idx.find("red.png"))
    assert_equal(idx.count(), 3)
    assert_equal(idx.find("red.png"), -1)
    var blob = idx.serialize()
    var back = AssetIndex.deserialize(blob^)
    assert_equal(back.count(), 3)
    assert_true(back.model_name == "toy" and back.dim == 3 and back.logit_scale == 2.0, "header round trip")
    for i in range(3):
        var j = back.find(idx.paths[i])
        assert_true(j >= 0, "path kept")
        assert_equal(back.sizes[j], idx.sizes[i])
        assert_equal(back.mtimes[j], idx.mtimes[i])
        for d in range(3):
            assert_true(back.emb[j * 3 + d] == idx.emb[i * 3 + d], "embedding bits kept")
    var top = back.search_embedding(_unit(0, 0, 1), 1)
    assert_true(top[0].path == "green.png" or top[0].path == "blue.png", "search after reload")
    print("✓ AssetIndex add / replace / remove / top-k / cache round trip")


# ----------------------------------------------------------------------------- 5. MCP

def _mcp_call(mut mcp: McpServerV2, args: String, mut studio: FlintStudioState, mut assets: AssetServer,
              mut pool: JobPool) raises -> String:
    var r = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"assets.semantic_search","arguments":'
                               + args + "}}", studio, assets, pool)
    var d = parse_json(List[UInt8](r.as_bytes()))
    var content = d.at(d.object_get(d.object_get(d.root, "result"), "content"), 0)
    return d.text(d.object_get(content, "text"))


def test_mcp(model_dir: String, refdir: String) raises:
    var ws = getenv("TMPDIR", "/tmp") + "/flint_siglip_mcp"
    _ = make_dirs(ws + "/assets")
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=ws)
    var lst = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":1,"method":"tools/list"}', studio, assets, pool)
    assert_true("assets.semantic_search" in lst, "tool listed")
    var bad = _mcp_call(mcp, '{"query":"a cat","assets_dir":"/etc"}', studio, assets, pool)
    assert_true("sandbox rejected" in bad, "assets_dir outside the workspace is rejected: " + bad)
    var dotdot = _mcp_call(mcp, '{"query":"a cat","assets_dir":"assets","model_dir":"../x"}', studio, assets, pool)
    assert_true("sandbox rejected" in dotdot, "model_dir with .. is rejected: " + dotdot)
    if model_dir == "":
        var none = _mcp_call(mcp, '{"query":"a cat","assets_dir":"assets"}', studio, assets, pool)
        assert_true("ok\":false" in none or "no model" in none or "error" in none, "no model -> error: " + none)
        print("✓ MCP assets.semantic_search listed, sandboxed, errors without a model")
        return
    var img_dir = refdir + "/images"
    for name in ["circle.png", "square.png", "gradient.png"]:
        var b = read_bytes(img_dir + "/" + name)
        _ = write_bytes(ws + "/assets/" + name, b)
    var out = _mcp_call(mcp, '{"query":"ein roter Kreis auf blauem Grund","assets_dir":"assets","k":2}', studio, assets, pool)
    var d = parse_json(List[UInt8](out.as_bytes()))
    assert_true(d.boolean(d.object_get(d.root, "ok"), False), "tool ok: " + out)
    var res = d.object_get(d.root, "results")
    assert_equal(d.array_size(res), 2)
    var top = d.text(d.object_get(d.at(res, 0), "path"))
    print("  MCP search (German query) -> " + top)
    assert_true(top == "circle.png", "MCP search finds the circle")
    var again = _mcp_call(mcp, '{"query":"a green square","assets_dir":"assets","k":1}', studio, assets, pool)
    var d2 = parse_json(List[UInt8](again.as_bytes()))
    assert_equal(d2.integer(d2.object_get(d2.object_get(d2.root, "index"), "embedded"), -1), 0)
    assert_true(d2.text(d2.object_get(d2.at(d2.object_get(d2.root, "results"), 0), "path")) == "square.png", "second query")
    print("✓ MCP assets.semantic_search (sandbox, cached index, multilingual query)")


# ----------------------------------------------------------------------------- with weights

def _f32(path: String) raises -> List[Float32]:
    var b = read_bytes(path)
    if len(b) == 0:
        raise Error("missing reference " + path)
    var out = List[Float32](length=len(b) // 4, fill=0.0)
    var p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(b.unsafe_ptr()))
    for i in range(len(out)):
        out[i] = p.unsafe_offset(i).unsafe_load[alignment=1]()
    _ = b^
    return out^


def _i32(path: String) raises -> List[Int]:
    var b = read_bytes(path)
    var out = List[Int](length=len(b) // 4, fill=0)
    var p = Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(b.unsafe_ptr()))
    for i in range(len(out)):
        out[i] = Int(p.unsafe_offset(i).unsafe_load[alignment=1]())
    _ = b^
    return out^


def _cmp(a: List[Float32], b: List[Float32], dim: Int, what: String, max_abs: Float64, min_cos: Float64) raises:
    if len(a) != len(b):
        raise Error(what + ": " + String(len(a)) + " values vs reference " + String(len(b)))
    var md: Float64 = 0
    var worst_cos: Float64 = 1
    for r in range(len(a) // dim):
        var ab: Float64 = 0
        var aa: Float64 = 0
        var bb: Float64 = 0
        for d in range(dim):
            var x = Float64(a[r * dim + d])
            var y = Float64(b[r * dim + d])
            md = max(md, abs(x - y))
            ab += x * y
            aa += x * x
            bb += y * y
        worst_cos = min(worst_cos, ab / max(sqrt(aa * bb), 1e-30))
    print("  " + what + ": max |diff| " + String(md) + ", min cosine " + String(worst_cos))
    assert_true(md <= max_abs and worst_cos >= min_cos, what + " outside tolerance")


def test_model(dir: String, refdir: String) raises:
    var gpu = getenv("FLINT_SIGLIP_GPU", "1") != "0"
    var idx_doc = parse_json(read_bytes(refdir + "/index.json"))
    var root = idx_doc.root
    # tokenizer parity: both on-disk forms
    var corpus_node = idx_doc.object_get(root, "corpus")
    var tok_ids = _i32(refdir + "/tok_ids.i32")
    var tok_len = _i32(refdir + "/tok_len.i32")
    for form in ["/tokenizer.model", "/tokenizer.json"]:
        var sp = SentencePiece.load(dir + form)
        var at = 0
        var bad = 0
        for i in range(len(tok_len)):
            var got = sp.encode(idx_doc.text(idx_doc.at(corpus_node, i)))
            var want = List[Int]()
            for k in range(tok_len[i]):
                want.append(tok_ids[at + k])
            at += tok_len[i]
            if len(got) != len(want):
                bad += 1
                continue
            for k in range(len(got)):
                if got[k] != want[k]:
                    bad += 1
                    break
        print("  tokenizer" + form + ": " + String(len(tok_len) - bad) + "/" + String(len(tok_len)) + " lines identical")
        assert_equal(bad, 0)
    var t0 = perf_counter_ns()
    var m = SiglipModel.load(dir, gpu)
    var t1 = perf_counter_ns()
    print("  loaded " + m.name + " on " + m.rt.device_name() + " in " + String(Float64(t1 - t0) / 1e9) + " s")
    var S = m.image_size()
    var D = m.dim
    assert_equal(S, idx_doc.integer(idx_doc.object_get(root, "image_size"), 0))
    assert_equal(D, idx_doc.integer(idx_doc.object_get(root, "dim"), 0))
    assert_true(abs(Float64(m.logit_scale) - idx_doc.number(idx_doc.object_get(root, "logit_scale"), 0)) < 1e-6, "logit_scale")
    # preprocessing
    var images = idx_doc.object_get(root, "images")
    var n = idx_doc.array_size(images)
    var pixels = List[Float32]()
    for i in range(n):
        var rel = idx_doc.text(idx_doc.at(images, i))
        var p = rel if rel.startswith("/") else refdir + "/" + rel
        var img = decode_png(read_bytes(p), 1 << 14)
        assert_true(img.width > 0, "decoded " + p)
        pixels.extend(siglip_preprocess(img, S))
    var ref_px = _f32(refdir + "/pixels.f32")
    _cmp(pixels, ref_px, S * S * 3, "pixel_values", 1e-6, 0.9999999)
    # vision tower
    var tv0 = perf_counter_ns()
    var vis = m.vision.encode(m.rt, ref_px, n, True)
    m.rt.sync()
    var tv1 = perf_counter_ns()
    var T = idx_doc.integer(idx_doc.object_get(root, "tokens"), 0)
    var h0 = List[Float32]()
    for k in range(T * D):
        h0.append(vis[1][k])
    _cmp(h0, _f32(refdir + "/vis_hidden0.f32"), D, "vision hidden (image 0)", 2e-2, 0.99999)
    _cmp(vis[0], _f32(refdir + "/img_pooled.f32"), D, "image embeddings", 1e-2, 0.99999)
    # text tower on the processor's ids
    var ids = _i32(refdir + "/ids.i32")
    var texts = idx_doc.object_get(root, "texts")
    var nt = idx_doc.array_size(texts)
    var own = List[Int]()
    for t in range(nt):
        own.extend(m.token_ids(idx_doc.text(idx_doc.at(texts, t))))
    _expect_ids(own, ids, "processor input_ids")
    var tt0 = perf_counter_ns()
    var txt = m.text.encode(m.rt, ids, nt, True)
    m.rt.sync()
    var tt1 = perf_counter_ns()
    var th0 = List[Float32]()
    for k in range(m.text.max_len * D):
        th0.append(txt[1][k])
    _cmp(th0, _f32(refdir + "/txt_hidden0.f32"), D, "text hidden (text 0)", 2e-2, 0.99999)
    _cmp(txt[0], _f32(refdir + "/txt_pooled.f32"), D, "text embeddings", 1e-2, 0.99999)
    # sigmoid head
    var ie = vis[0].copy()
    var te = txt[0].copy()
    l2_normalize(ie, D)
    l2_normalize(te, D)
    var lg = m.logits(te, ie)
    var ref_lg = _f32(refdir + "/logits.f32")
    var md: Float64 = 0
    var agree = 0
    for t in range(nt):
        var a = 0
        var b = 0
        for i in range(n):
            md = max(md, abs(Float64(lg[t * n + i]) - Float64(ref_lg[t * n + i])))
            if lg[t * n + i] > lg[t * n + a]:
                a = i
            if ref_lg[t * n + i] > ref_lg[t * n + b]:
                b = i
        if a == b:
            agree += 1
    print("  logits_per_text: max |diff| " + String(md) + ", argmax agreement " + String(agree) + "/" + String(nt))
    assert_true(md < 5e-2, "logits outside tolerance")
    assert_equal(agree, nt)
    print("  timing: " + String(n) + " images " + String(Float64(tv1 - tv0) / 1e6) + " ms, " + String(nt)
          + " texts " + String(Float64(tt1 - tt0) / 1e6) + " ms (" + m.rt.device_name() + ")")
    # search over the reference images
    var index = AssetIndex(m.name, D, S, m.logit_scale, m.logit_bias)
    var rep = index.update(m, refdir + "/images", 8)
    assert_equal(rep.embedded, 5)
    var q = String(idx_doc.text(idx_doc.at(texts, 0)))
    var hits = index.search(m, q, 3)
    print("  search '" + q + "' -> " + hits[0].path + " p=" + String(hits[0].prob))
    assert_true(hits[0].path == "circle.png", "search finds the circle")
    var again = index.update(m, refdir + "/images", 8)
    assert_equal(again.embedded, 0)
    assert_equal(again.reused, rep.embedded)
    print("✓ SigLIP parity against Transformers (" + m.name + ")")


def main() raises:
    test_tokenizer()
    test_resize()
    test_thumbnail()
    test_index()
    var dir = getenv("FLINT_SIGLIP_DIR", "")
    var refdir = getenv("FLINT_SIGLIP_REF", "")
    if dir != "" and refdir != "":
        test_model(dir, refdir)
        test_mcp(dir, refdir)
    else:
        print("  (set FLINT_SIGLIP_DIR and FLINT_SIGLIP_REF for the Transformers parity run)")
        test_mcp("", "")
    print("✓ test_flint_v5_siglip passed")
