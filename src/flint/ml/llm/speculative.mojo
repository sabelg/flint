"""Speculative decoding: a small draft model proposes, the target verifies (`flint.ml.llm.speculative`).

A draft (e.g. Qwen3-0.6B) and a target (Qwen3-1.7B / 4B / 8B) that share a tokenizer run side by
side, each with its own KV cache (on the GPU when present). One round:

  1. draft: the draft catches up on the accepted tokens it has not seen (prefix reuse, usually
     one or two tokens) and proposes `k` tokens autoregressively, keeping the distribution `q`
     each was drawn from;
  2. verify: the target evaluates the last accepted token plus the `k` proposals in one batched
     pass (`eval(..., all_logits=True)`, `k + 1` rows), giving its distribution `p` after every
     prefix;
  3. accept: proposals are accepted left to right. Greedy (`temperature <= 0`): while the
     target's own greedy choice equals the proposal. Sampling: the standard rule (Leviathan et
     al. 2023, Chen et al. 2023): accept `x ~ q` with probability `min(1, p(x) / q(x))`, else
     emit a token drawn from `norm(max(0, p - q))` and stop. When every proposal is accepted the
     target's last row gives one bonus token, so a round emits between 1 and `k + 1` tokens;
  4. rollback: `LlmModel.rewind` drops the rejected positions from both caches. K / V rows are
     addressed by position, so nothing is copied; the next round overwrites them.

Greedy output is the target's own greedy output token for token (every kernel computes a row
independently of the other rows in its batch, so the verify rows are bit-identical to one-token
decoding steps), and sampled output follows the target's sampling distribution exactly (top-k /
top-p / temperature / repetition penalty included, via `Sampler.distribution`).
"""

from std.math import exp
from std.time import perf_counter_ns
from flint.ml.llm.tokenizer import BpeTokenizer
from flint.ml.llm.model import LlmModel
from flint.ml.llm.sample import Sampler, SparseDist, Rng


struct SpecPick(Copyable, Movable):
    var accepted: Bool
    var token: Int

    def __init__(out self, accepted: Bool, token: Int):
        self.accepted = accepted
        self.token = token


def spec_accept(p: SparseDist, q: SparseDist, x: Int, mut rng: Rng) -> SpecPick:
    """The speculative sampling rule for one proposal `x ~ q` against the target `p`: accepted
    with probability min(1, p(x) / q(x)); otherwise a token drawn from norm(max(0, p - q)).
    The emitted token is distributed exactly as `p`."""
    var px = p.prob(x)
    var qx = q.prob(x)
    # (qx == 0: q could not have proposed x; treated as a rejection, which keeps the rule total)
    if qx > 0.0 and (px >= qx or rng.uniform() * qx < px):
        return SpecPick(True, x)
    var res = SparseDist()
    var tot: Float64 = 0
    for i in range(len(p.ids)):
        var r = p.p[i] - q.prob(p.ids[i])
        if r > 0:
            res.ids.append(p.ids[i])
            res.p.append(r)
            tot += r
    if tot <= 0:
        return SpecPick(False, p.draw(rng))
    for i in range(len(res.p)):
        res.p[i] /= tot
    return SpecPick(False, res.draw(rng))


struct SpecStats(Copyable, Movable):
    var rounds: Int          # target verify passes after the prompt
    var drafted: Int         # proposals made
    var accepted: Int        # proposals accepted
    var emitted: Int         # tokens produced by the rounds

    def __init__(out self):
        self.rounds = 0
        self.drafted = 0
        self.accepted = 0
        self.emitted = 0

    def acceptance(self) -> Float64:
        if self.drafted == 0:
            return 0
        return Float64(self.accepted) / Float64(self.drafted)

    def tokens_per_round(self) -> Float64:
        if self.rounds == 0:
            return 0
        return Float64(self.emitted) / Float64(self.rounds)


struct SpecGeneration(Copyable, Movable):
    var tokens: List[Int]
    var text: String
    var prompt_tokens: Int
    var prefill_s: Float64
    var decode_s: Float64
    var stopped: Bool
    var stats: SpecStats

    def __init__(out self):
        self.tokens = List[Int]()
        self.text = ""
        self.prompt_tokens = 0
        self.prefill_s = 0
        self.decode_s = 0
        self.stopped = False
        self.stats = SpecStats()

    def decode_tps(self) -> Float64:
        if self.decode_s <= 0 or len(self.tokens) < 2:
            return 0
        return Float64(len(self.tokens) - 1) / self.decode_s


def _is_stop(t: Int, stops: List[Int]) -> Bool:
    for s in stops:
        if t == s:
            return True
    return False


def _max_prob(logits: List[Float32]) -> Float64:
    """Softmax probability of the most likely token (temperature 1)."""
    var mx = logits[0]
    for v in logits:
        mx = max(mx, v)
    var z: Float64 = 0
    for v in logits:
        var d = v - mx
        if d > -30.0:   # exp(-30) ~ 1e-13: the rest cannot move the result
            z += exp(Float64(d))
    return 1.0 / z


def _row(all: List[Float32], r: Int, v: Int) -> List[Float32]:
    var out = List[Float32](capacity=v)
    for i in range(v):
        out.append(all[r * v + i])
    return out^


def speculative_tokens(mut target: LlmModel, mut draft: LlmModel, prompt: List[Int], mut sampler: Sampler,
                       max_tokens: Int, stops: List[Int], k: Int = 4, min_prob: Float64 = 0.0) raises -> SpecGeneration:
    """Samples up to `max_tokens` after `prompt` from `target`, with `draft` proposing up to `k`
    tokens per round; stops at any of `stops` (not included). `text` is left empty.

    `min_prob` > 0 ends a round's drafting early once the draft's own probability for its
    proposal falls below it (dynamic draft length: an unsure draft is likely to be rejected, so
    its further proposals are not worth a draft step). It never changes the output."""
    if target.cfg.vocab != draft.cfg.vocab:
        raise Error("speculative: the draft and target vocabularies differ (" + String(draft.cfg.vocab) + " vs "
                    + String(target.cfg.vocab) + ")")
    if k < 1 or k + 1 > 128:
        raise Error("speculative: k must be in 1..127")
    var V = target.cfg.vocab
    var greedy = sampler.is_greedy()
    var dsampler = sampler.copy()
    var g = SpecGeneration()
    g.prompt_tokens = len(prompt)
    var t0 = perf_counter_ns()
    var logits = target.prompt(prompt)
    _ = draft.prompt(prompt)
    var t1 = perf_counter_ns()
    g.prefill_s = Float64(t1 - t0) / 1e9
    var first = sampler.sample(logits^, g.tokens)
    if _is_stop(first, stops) or max_tokens <= 0:
        g.stopped = _is_stop(first, stops)
        g.decode_s = Float64(perf_counter_ns() - t1) / 1e9
        return g^
    g.tokens.append(first)
    var seq = prompt.copy()
    seq.append(first)
    while len(g.tokens) < max_tokens and not g.stopped:
        var kk = min(k, max_tokens - len(g.tokens) - 1)
        # 1. draft
        var drafts = List[Int]()
        var qs = List[SparseDist]()
        if kk > 0:
            var dl = draft.prompt(seq)
            var recent = g.tokens.copy()
            for i in range(kk):
                var d: Int
                var conf: Float64 = 1.0
                if greedy:
                    if min_prob > 0:
                        conf = _max_prob(dl)
                    d = dsampler.sample(dl^, recent)
                    dl = List[Float32]()
                else:
                    var q = dsampler.distribution(dl^, recent)
                    dl = List[Float32]()
                    d = q.draw(sampler.rng)
                    conf = q.prob(d)
                    qs.append(q^)
                drafts.append(d)
                recent.append(d)
                if _is_stop(d, stops) or conf < min_prob:
                    break
                if i + 1 < kk:
                    dl = draft.push(d)
        g.stats.drafted += len(drafts)
        # 2. verify: the last accepted token and every proposal in one pass
        target.rewind(len(seq) - 1)
        var batch = List[Int]()
        batch.append(seq[len(seq) - 1])
        for d in drafts:
            batch.append(d)
        var rows = target.eval(batch, True)
        g.stats.rounds += 1
        # 3. accept left to right
        var all_ok = True
        for j in range(len(drafts)):
            var tok_j: Int
            var ok: Bool
            if greedy:
                tok_j = sampler.sample(_row(rows, j, V), g.tokens)
                ok = tok_j == drafts[j]
            else:
                var p = sampler.distribution(_row(rows, j, V), g.tokens)
                var pick = spec_accept(p, qs[j], drafts[j], sampler.rng)
                tok_j = pick.token
                ok = pick.accepted
            if ok:
                g.stats.accepted += 1
            if _is_stop(tok_j, stops):
                g.stopped = True
                all_ok = False
                break
            g.tokens.append(tok_j)
            seq.append(tok_j)
            g.stats.emitted += 1
            if not ok:
                all_ok = False
                break
        if all_ok:
            var bonus = sampler.sample(_row(rows, len(drafts), V), g.tokens)
            if _is_stop(bonus, stops):
                g.stopped = True
            else:
                g.tokens.append(bonus)
                seq.append(bonus)
                g.stats.emitted += 1
        # 4. rollback happens lazily: the next round rewinds the target to len(seq) - 1 and the
        # draft's `prompt(seq)` keeps only the common prefix
    target.rewind(len(seq) - 1)
    g.decode_s = Float64(perf_counter_ns() - t1) / 1e9
    return g^


def generate_speculative(mut target: LlmModel, mut draft: LlmModel, mut tok: BpeTokenizer, prompt: List[Int],
                         mut sampler: Sampler, max_tokens: Int, stops: List[Int], k: Int = 4,
                         min_prob: Float64 = 0.0) raises -> SpecGeneration:
    """`speculative_tokens` plus the decoded text (a drop-in for `chat.generate`)."""
    var g = speculative_tokens(target, draft, prompt, sampler, max_tokens, stops, k, min_prob)
    g.text = tok.decode(g.tokens)
    return g^
