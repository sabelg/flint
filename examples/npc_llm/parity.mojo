"""Logit / greedy parity against the Hugging Face reference, plus decode throughput.

usage: parity MODEL_DIR REF_DIR [--cpu] [--fmt fp32|int8|int4] [--steps N]
REF_DIR is written by `tools/llm/reference.py` (ids.txt, logits.f32, greedy.txt).
Prints the max |logit - reference| over every prompt position, the share of positions whose
argmax matches, whether the greedy continuation is identical, and prefill / decode tokens/s.
"""
from std.sys import argv
from std.math import abs, exp, log
from std.time import perf_counter_ns
from flint.platform.os import read_bytes
from flint.ml.llm import LlmModel, Q_F32, Q_INT8, Q_INT4, argmax


def _ids(path: String) raises -> List[Int]:
    var s = String(unsafe_from_utf8=read_bytes(path))
    var out = List[Int]()
    for p in s.split(" "):
        var t = String(p).strip()
        if t.byte_length() > 0:
            out.append(Int(String(t)))
    return out^


def main() raises:
    var a = argv()
    if len(a) < 3:
        print("usage: parity MODEL_DIR REF_DIR [--cpu] [--fmt fp32|int8|int4] [--steps N]")
        return
    var gpu = True
    var fmt = Q_F32
    var steps = -1
    var i = 3
    while i < len(a):
        var s = String(a[i])
        if s == "--cpu":
            gpu = False
        elif s == "--fmt" and i + 1 < len(a):
            var f = String(a[i + 1])
            fmt = Q_INT8 if f == "int8" else (Q_INT4 if f == "int4" else Q_F32)
            i += 1
        elif s == "--steps" and i + 1 < len(a):
            steps = Int(String(a[i + 1]))
            i += 1
        i += 1
    var ref_dir = String(a[2])
    var ids = _ids(ref_dir + "/ids.txt")
    var greedy = _ids(ref_dir + "/greedy.txt")
    if steps < 0:
        steps = len(greedy)
    var t0 = perf_counter_ns()
    var model = LlmModel.load(String(a[1]), fmt, gpu, 1024)
    print("[parity] loaded", model.cfg.family, model.format_name(), "on", model.device(), "in",
          Float64(perf_counter_ns() - t0) / 1e9, "s;", model.weight_bytes // (1 << 20), "MiB of weights")
    var V = model.cfg.vocab
    var t1 = perf_counter_ns()
    var lg = model.eval(ids, True)
    var t_pre = Float64(perf_counter_ns() - t1) / 1e9
    var refb = read_bytes(ref_dir + "/logits.f32")
    var rp = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(refb.unsafe_ptr()))
    var max_diff: Float32 = 0
    var max_ref: Float32 = 0
    var agree = 0
    var kl_sum: Float64 = 0
    for r in range(len(ids)):
        var best = 0
        var best_ref = 0
        var mx: Float64 = -1e30
        var my: Float64 = -1e30
        for v in range(V):
            mx = max(mx, Float64(lg[r * V + v]))
            my = max(my, Float64(rp.unsafe_offset(r * V + v).unsafe_load()))
        var zx: Float64 = 0
        var zy: Float64 = 0
        for v in range(V):
            zx += exp(Float64(lg[r * V + v]) - mx)
            zy += exp(Float64(rp.unsafe_offset(r * V + v).unsafe_load()) - my)
        var kl: Float64 = 0
        for v in range(V):
            var ly = Float64(rp.unsafe_offset(r * V + v).unsafe_load()) - my - log(zy)
            var lx = Float64(lg[r * V + v]) - mx - log(zx)
            kl += exp(ly) * (ly - lx)
        kl_sum += kl
        for v in range(V):
            var x = lg[r * V + v]
            var y = rp.unsafe_offset(r * V + v).unsafe_load()
            max_diff = max(max_diff, abs(x - y))
            max_ref = max(max_ref, abs(y))
            if x > lg[r * V + best]:
                best = v
            if y > rp.unsafe_offset(r * V + best_ref).unsafe_load():
                best_ref = v
        if best == best_ref:
            agree += 1
    _ = len(refb)
    print("[parity] prompt", len(ids), "tokens: max |dlogit| =", max_diff, "(max |logit|", max_ref, "); argmax agrees at",
          agree, "/", len(ids), "positions; mean KL(ref || ours) =", kl_sum / Float64(len(ids)), "nats")
    # greedy continuation from the cached prompt
    var last = List[Float32](length=V, fill=0.0)
    for v in range(V):
        last[v] = lg[(len(ids) - 1) * V + v]
    var gen = List[Int]()
    var t2 = perf_counter_ns()
    for s in range(steps):
        var t = argmax(last)
        gen.append(t)
        if s + 1 < steps:
            last = model.push(t)
    var t_dec = Float64(perf_counter_ns() - t2) / 1e9
    var same = 0
    while same < min(len(gen), len(greedy)) and gen[same] == greedy[same]:
        same += 1
    print("[parity] greedy: first", same, "of", min(len(gen), len(greedy)), "tokens identical",
          "(IDENTICAL)" if same == min(len(gen), len(greedy)) else "(DIVERGES)")
    print("[bench]", model.cfg.family, model.format_name(), model.device(), ": prefill",
          Float64(len(ids)) / t_pre, "tok/s, decode", Float64(steps - 1) / t_dec, "tok/s")
