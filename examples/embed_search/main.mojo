"""Semantic search over game content with Qwen3-Embedding (`flint.ml.llm.embed`).

usage:
  embed_search demo   MODEL_DIR [--cpu] [--fmt fp32|int8|int4] [--dim N] [--save PATH] [QUERY ...]
  embed_search parity MODEL_DIR REF_DIR [--cpu] [--fmt fp32|int8|int4]

MODEL_DIR is a Qwen3-Embedding checkpoint directory (config.json, model.safetensors,
tokenizer.json), e.g. Qwen3-Embedding-0.6B. Default: fp32 weights on the GPU when present.

`demo` embeds a small library of assets, scripts, quests and dialogue lines into an
`EmbeddingIndex` (MRL-truncated to `--dim`, default 1024), optionally saves it, reloads it and
answers instruct-formatted queries with their top-3 hits (overall and per kind).
`parity` compares against `tools/llm/embed_reference.py` output: token ids, cosine and max
|difference| to the sentence-transformers embeddings at full size and at MRL sizes 32..768,
and whether the query -> document ranking agrees.
"""
from std.sys import argv
from std.math import abs
from std.time import perf_counter_ns
from flint.platform.os import read_bytes
from flint.ml.llm import Q_F32, Q_INT8, Q_INT4
from flint.ml.llm.embed import TextEmbedder, EmbeddingIndex, instruct_query, mrl_truncate, cosine

comptime TASK: String = "Given a player's request, retrieve the game asset, script, quest or dialogue line it refers to"


def library() -> List[String]:
    var d = List[String]()
    d.append("asset: sfx/crypt_wind_howl.qoa - low howling wind with distant chains, for tombs and crypts")
    d.append("asset: models/weapons/longsword_steel.glb - a steel longsword sold by Brakka the blacksmith")
    d.append("asset: textures/grass_meadow_albedo.png - green meadow grass, tiling")
    d.append("script: scripts/castle/drawbridge.mojo - lowers the castle drawbridge when the lever is pulled")
    d.append("script: scripts/ai/goblin_patrol.mojo - goblins walk a patrol route and alert their friends")
    d.append("quest: The Miller's Daughter - Elsa has not come home from the old mill; find her before nightfall")
    d.append("quest: Ember Tithe - collect five ember stones for the village shrine")
    d.append("dialogue: Brakka (no gold): \"No coin, no steel. Come back when your purse is heavier.\"")
    d.append("dialogue: Elsa (rescued): \"I thought the wolves would find me first. Thank you!\"")
    d.append("dialogue: Guard (night): \"Gates are shut after dark. Move along.\"")
    return d^


def kind_of(doc: String) -> String:
    var c = doc.find(":")
    return String(doc[byte=0:c]) if c > 0 else String("")


def opts(a: List[String], start: Int, mut gpu: Bool, mut fmt: Int, mut dim: Int, mut save: String,
         mut rest: List[String]) raises:
    var i = start
    while i < len(a):
        var s = a[i]
        if s == "--cpu":
            gpu = False
        elif s == "--fmt" and i + 1 < len(a):
            fmt = Q_INT8 if a[i + 1] == "int8" else (Q_INT4 if a[i + 1] == "int4" else Q_F32)
            i += 1
        elif s == "--dim" and i + 1 < len(a):
            dim = Int(a[i + 1])
            i += 1
        elif s == "--save" and i + 1 < len(a):
            save = a[i + 1]
            i += 1
        else:
            rest.append(s)
        i += 1


def demo(a: List[String]) raises:
    var gpu = True
    var fmt = Q_F32
    var dim = 0
    var save = String("")
    var queries = List[String]()
    opts(a, 3, gpu, fmt, dim, save, queries)
    if len(queries) == 0:
        queries.append("where can I buy a better sword")
        queries.append("the quest about the missing miller's daughter")
        queries.append("a spooky sound for the crypt")
        queries.append("what does the blacksmith say when you have no money")
        queries.append("script that opens the drawbridge")
    var t0 = perf_counter_ns()
    var em = TextEmbedder(a[2], dim, fmt, gpu)
    print("[embed] loaded", em.model.format_name(), "on", em.model.device(), "in",
          Float64(perf_counter_ns() - t0) / 1e9, "s; dim", em.dim, "of", em.full_dim())
    var idx = EmbeddingIndex(em.dim)
    var docs = library()
    var t1 = perf_counter_ns()
    var ntok = 0
    for d in docs:
        ntok += len(em.tokens(d))
        _ = idx.add(d, kind_of(d), em.embed(d))
    var dt = Float64(perf_counter_ns() - t1) / 1e9
    print("[embed] indexed", len(idx), "entries (", ntok, "tokens ) in", dt, "s:", Float64(len(idx)) / dt, "texts/s")
    if save.byte_length() > 0:
        idx.save(save)
        idx = EmbeddingIndex.load(save)
        print("[embed] saved and reloaded", save, "(", len(idx), "entries )")
    for q in queries:
        var t2 = perf_counter_ns()
        var qv = em.embed_query(q, TASK)
        var qs = Float64(perf_counter_ns() - t2) / 1e3
        print("\nquery:", q, "(", Int(qs), "us )")
        for h in idx.search(qv, 3):
            print("  ", h.score, h.key)
        var top_quest = idx.search(qv, 1, "quest")
        if len(top_quest) > 0:
            print("   best quest:", top_quest[0].key)


def floats(path: String) -> List[Float32]:
    var b = read_bytes(path)
    var n = len(b) // 4
    var out = List[Float32](capacity=n)
    var p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(b.unsafe_ptr()))
    for i in range(n):
        out.append(p.unsafe_offset(i).unsafe_load())
    _ = len(b)
    return out^


def lines(path: String) -> List[String]:
    var s = String(unsafe_from_utf8=read_bytes(path))
    var out = List[String]()
    for l in s.split("\n"):
        if l.byte_length() > 0:
            out.append(String(l))
    return out^


def row(v: List[Float32], r: Int, d: Int) -> List[Float32]:
    var out = List[Float32](capacity=d)
    for i in range(d):
        out.append(v[r * d + i])
    return out^


def parity(a: List[String]) raises:
    if len(a) < 4:
        print("usage: embed_search parity MODEL_DIR REF_DIR [--cpu] [--fmt fp32|int8|int4]")
        return
    var gpu = True
    var fmt = Q_F32
    var dim = 0
    var save = String("")
    var rest = List[String]()
    opts(a, 4, gpu, fmt, dim, save, rest)
    var ref_dir = a[3]
    var em = TextEmbedder(a[2], 0, fmt, gpu)
    var D = em.full_dim()
    print("[parity] Qwen3-Embedding", em.model.format_name(), "on", em.model.device(), "dim", D)
    var inputs = lines(ref_dir + "/inputs.txt")
    var ids_ref = lines(ref_dir + "/ids.txt")
    var st = floats(ref_dir + "/st.f32")
    var n = len(inputs)
    var ours = List[List[Float32]]()
    var ids_ok = 0
    var tsum: Float64 = 0
    var ntok = 0
    for i in range(n):
        var text = String(inputs[i].replace("\\n", "\n"))
        var ids = em.tokens(text)
        var want = List[Int]()
        for p in ids_ref[i].split(" "):
            if p.byte_length() > 0:
                want.append(Int(String(p)))
        var same = len(ids) == len(want)
        if same:
            for j in range(len(ids)):
                if ids[j] != want[j]:
                    same = False
        if same:
            ids_ok += 1
        em.model.reset()   # no prefix reuse while timing
        var t = perf_counter_ns()
        ours.append(em.embed_full(text))
        tsum += Float64(perf_counter_ns() - t) / 1e9
        ntok += len(ids)
    print("[parity] token ids identical for", ids_ok, "/", n, "inputs")
    var dims = List[Int]()
    for d in [32, 64, 128, 256, 512, 768]:
        dims.append(d)
    dims.append(D)
    for d in dims:
        var min_cos: Float32 = 2
        var max_abs: Float32 = 0
        for i in range(n):
            var x = mrl_truncate(ours[i], d)
            var y = mrl_truncate(row(st, i, D), d)
            min_cos = min(min_cos, cosine(x, y))
            for j in range(d):
                max_abs = max(max_abs, abs(x[j] - y[j]))
        print("[parity] dim", d, ": min cos(ours, sentence-transformers) =", min_cos, "(1 - cos", 1.0 - min_cos, "); max |diff| =", max_abs)
    # retrieval agreement: queries are the inputs starting with "Instruct:"
    var nq = 0
    for i in range(n):
        if inputs[i].startswith("Instruct:"):
            nq += 1
    var agree = 0
    var max_sdiff: Float32 = 0
    for qi in range(nq):
        var bo = -1
        var br = -1
        var so: Float32 = -2
        var sr: Float32 = -2
        for di in range(nq, n):
            var a1 = cosine(ours[qi], ours[di])
            var b1 = cosine(row(st, qi, D), row(st, di, D))
            max_sdiff = max(max_sdiff, abs(a1 - b1))
            if a1 > so:
                so = a1
                bo = di
            if b1 > sr:
                sr = b1
                br = di
        if bo == br:
            agree += 1
    print("[parity] query -> document top-1 agrees for", agree, "/", nq, "queries; max |similarity diff| =", max_sdiff)
    print("[bench] embed", em.model.format_name(), em.model.device(), ":", Float64(n) / tsum, "texts/s,",
          Float64(ntok) / tsum, "tokens/s (", ntok // n, "tokens per text )")


def main() raises:
    var raw = argv()
    var a = List[String]()
    for x in raw:
        a.append(String(x))
    if len(a) < 3:
        print("usage: embed_search demo MODEL_DIR [--cpu] [--fmt F] [--dim N] [--save PATH] [QUERY ...]")
        print("       embed_search parity MODEL_DIR REF_DIR [--cpu] [--fmt F]")
        return
    if a[1] == "parity":
        parity(a)
    else:
        demo(a)
