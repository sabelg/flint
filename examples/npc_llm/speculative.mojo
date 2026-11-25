"""Speculative decoding benchmark: a small draft model proposes, the target verifies.

usage: speculative TARGET_DIR DRAFT_DIR [--cpu] [--fmt fp32|int8|int4] [--draft-fmt F] [--k 2,4,6]
                   [--max N] [--temp T] [--seed N] [--min-prob P]

TARGET_DIR is Qwen3-1.7B / 4B / 8B, DRAFT_DIR Qwen3-0.6B (same tokenizer). For each NPC prompt
the target first decodes on its own, then with the draft proposing `k` tokens per round.
Greedy (default, `--temp 0`): checks that every speculative run is token-for-token identical to
the plain greedy run and prints the decode speed-up. Sampling (`--temp T > 0`): standard
acceptance rule; prints acceptance and speed (outputs differ from the plain run by design). `--min-prob P` stops a
round's drafting once the draft's probability for its proposal drops below P (default 0.4;
0 drafts all `k`).
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.ml.llm import LlmModel, BpeTokenizer, Sampler, Q_F32, Q_INT8, Q_INT4
from flint.ml.llm.chat import ChatTemplate, ChatMessage, generate
from flint.ml.llm.speculative import generate_speculative

comptime SYSTEM: String = "You are Brakka, a gruff but kind dwarven blacksmith in the village of Emberfall. Stay in character."


def _fmt(s: String) -> Int:
    return Q_INT8 if s == "int8" else (Q_INT4 if s == "int4" else Q_F32)


def main() raises:
    var raw = argv()
    var a = List[String]()
    for x in raw:
        a.append(String(x))
    if len(a) < 3:
        print("usage: speculative TARGET_DIR DRAFT_DIR [--cpu] [--fmt F] [--draft-fmt F] [--k 2,4,6] [--max N] [--temp T] [--seed N]")
        return
    var gpu = True
    var fmt = Q_INT8
    var dfmt = -1
    var ks = List[Int]()
    var max_tokens = 128
    var temp: Float32 = 0.0
    var seed: UInt64 = 7
    var min_prob: Float64 = 0.4
    var i = 3
    while i < len(a):
        var s = a[i]
        var has = i + 1 < len(a)
        if s == "--cpu":
            gpu = False
        elif s == "--fmt" and has:
            fmt = _fmt(a[i + 1])
            i += 1
        elif s == "--draft-fmt" and has:
            dfmt = _fmt(a[i + 1])
            i += 1
        elif s == "--k" and has:
            for p in a[i + 1].split(","):
                ks.append(Int(String(p)))
            i += 1
        elif s == "--max" and has:
            max_tokens = Int(a[i + 1])
            i += 1
        elif s == "--temp" and has:
            temp = Float32(Float64(a[i + 1]))
            i += 1
        elif s == "--min-prob" and has:
            min_prob = Float64(a[i + 1])
            i += 1
        elif s == "--seed" and has:
            seed = UInt64(Int(a[i + 1]))
            i += 1
        else:
            raise Error("unknown option " + s)
        i += 1
    if len(ks) == 0:
        ks.append(2)
        ks.append(4)
        ks.append(6)
    if dfmt < 0:
        dfmt = fmt
    var t0 = perf_counter_ns()
    var target = LlmModel.load(a[1], fmt, gpu, 2048)
    var draft = LlmModel.load(a[2], dfmt, gpu, 2048)
    var tok = BpeTokenizer.load(a[1] + "/tokenizer.json")
    var tmpl = ChatTemplate.for_model(target.cfg, tok)
    print("[spec] target", target.cfg.layers, "layers x", target.cfg.hidden, target.format_name(), "| draft",
          draft.cfg.layers, "layers x", draft.cfg.hidden, draft.format_name(), "| on", target.device(), "| loaded in",
          Float64(perf_counter_ns() - t0) / 1e9, "s")
    var users = List[String]()
    users.append("Hello there! Can you repair my sword before the goblins attack tonight?")
    users.append("Tell me the story of how you lost your left eye, in a few sentences.")
    users.append("List the five things I should buy before travelling through the Ashen Pass, and why.")
    # warm up both models (kernel loading, first allocations)
    var warm = List[ChatMessage]()
    warm.append(ChatMessage("user", "Hi"))
    var wids = tmpl.render(tok, warm)
    var ws = Sampler.greedy()
    _ = generate(target, tok, wids, ws, 4, tmpl.stops)
    _ = generate(draft, tok, wids, ws, 4, tmpl.stops)
    var plain_tok = 0
    var plain_s: Float64 = 0
    var spec_tok = List[Int](length=len(ks), fill=0)
    var spec_s = List[Float64](length=len(ks), fill=0.0)
    var all_identical = True
    for ui in range(len(users)):
        var msgs = List[ChatMessage]()
        msgs.append(ChatMessage("system", SYSTEM))
        msgs.append(ChatMessage("user", users[ui]))
        var ids = tmpl.render(tok, msgs)
        target.reset()
        var ps = Sampler(temp, 0.9, 40, seed)
        var base = generate(target, tok, ids, ps, max_tokens, tmpl.stops)
        plain_tok += len(base.tokens) - 1
        plain_s += base.decode_s
        print("\n[prompt ", ui + 1, "] ", len(ids), " tokens; target alone: ", len(base.tokens), " tokens at ",
              base.decode_tps(), " tok/s", sep="")
        print("  ", base.text.replace("\n", " "))
        for ki in range(len(ks)):
            target.reset()
            draft.reset()
            var ss = Sampler(temp, 0.9, 40, seed)
            var g = generate_speculative(target, draft, tok, ids, ss, max_tokens, tmpl.stops, ks[ki], min_prob)
            spec_tok[ki] += len(g.tokens) - 1
            spec_s[ki] += g.decode_s
            var same = len(g.tokens) == len(base.tokens)
            if same:
                for j in range(len(g.tokens)):
                    if g.tokens[j] != base.tokens[j]:
                        same = False
            if temp <= 0 and not same:
                all_identical = False
            var verdict = String("")
            if temp <= 0:
                verdict = "IDENTICAL" if same else "DIVERGES"
            print("  k=", ks[ki], ": ", len(g.tokens), " tokens at ", g.decode_tps(), " tok/s (x",
                  g.decode_tps() / max(base.decode_tps(), 1e-9), "), acceptance ", g.stats.acceptance(), ", ",
                  g.stats.tokens_per_round(), " tokens per target pass ", verdict, sep="")
    var ptps = Float64(plain_tok) / plain_s
    print("\n[bench] target alone:", ptps, "tok/s decode")
    for ki in range(len(ks)):
        var stps = Float64(spec_tok[ki]) / spec_s[ki]
        print("[bench] speculative k=", ks[ki], ": ", stps, " tok/s decode, speed-up x", stps / ptps, sep="")
    if temp <= 0:
        print("[spec] greedy outputs", "IDENTICAL to the target's own greedy output" if all_identical else "DIVERGE")
