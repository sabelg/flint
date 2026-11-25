"""UniRig automatic rigging runtime (`flint.ml.unirig`).

A pure-Mojo port of UniRig (VAST-AI-Research / Tsinghua, MIT licence; see `licenses/`):
a mesh goes in, a skeleton and per-vertex skin weights come out.

  1. Skeleton: the mesh is normalised into [-1, 1]^3 (Z up), point-sampled, encoded into 1024
     latent tokens (`unirig_skel_encoder.onnx`, run by `flint.ml.onnx`) and an OPT-350M decoder
     (`flint.ml.opt`) writes the skeleton as tokens: quantised joint positions (256 steps per
     axis), branch markers and part labels. `UniRigTokenizer` limits each step to the tokens the
     grammar allows and turns the result back into joints, parents and tails.
  2. Skin: see `flint.ml.unirig_skin`.

The model files are converted once from the released checkpoints (`tools/unirig/export.py`)
and stay outside the repository.
"""

from std.math import exp, log, sqrt
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.opt import OptDecoder

comptime UR_DISCRETE: Int = 256
comptime UR_BRANCH: Int = 256
comptime UR_BOS: Int = 257
comptime UR_EOS: Int = 258
comptime UR_PAD: Int = 259
comptime UR_SPRING: Int = 260
comptime UR_PART_BODY: Int = 261
comptime UR_PART_HAND: Int = 262
comptime UR_CLS_NONE: Int = 263
comptime UR_CLS_VROID: Int = 264
comptime UR_CLS_MIXAMO: Int = 265
comptime UR_CLS_ARTICULATION: Int = 266
comptime UR_VOCAB: Int = 267

# grammar states
comptime _S_BOS: Int = 0
comptime _S_CLS_PART_JOINT: Int = 1
comptime _S_PART_JOINT: Int = 2
comptime _S_JOINT2: Int = 3
comptime _S_JOINT3: Int = 4
comptime _S_BRANCH_PART_JOINT: Int = 5
comptime _S_JOINT: Int = 6


def _is_cls(t: Int) -> Bool:
    return t >= UR_CLS_NONE and t <= UR_CLS_ARTICULATION


def grammar_state(seq: List[Int]) -> Int:
    """State of UniRig's skeleton grammar after `seq` (which starts with BOS)."""
    var st = _S_BOS
    for t in seq:
        if st == _S_BOS:
            st = _S_CLS_PART_JOINT
        elif st == _S_CLS_PART_JOINT:
            if t < UR_DISCRETE:
                st = _S_JOINT2
            elif _is_cls(t):
                st = _S_PART_JOINT
            else:
                st = _S_JOINT
        elif st == _S_PART_JOINT:
            st = _S_JOINT2 if t < UR_DISCRETE else _S_PART_JOINT
        elif st == _S_JOINT2:
            st = _S_JOINT3
        elif st == _S_JOINT3:
            st = _S_BRANCH_PART_JOINT
        elif st == _S_BRANCH_PART_JOINT:
            if t == UR_BRANCH:
                st = _S_JOINT
            elif t < UR_DISCRETE:
                st = _S_JOINT2
            else:
                st = _S_JOINT
        else:
            st = _S_JOINT2
    return st


def allowed_tokens(state: Int) -> List[Bool]:
    """Which tokens may follow a grammar state (`next_posible_token`)."""
    var ok = List[Bool](length=UR_VOCAB, fill=False)
    var part = False
    var cls = False
    var branch = False
    var eos = False
    if state == _S_BOS:
        ok[UR_BOS] = True
        return ok^
    elif state == _S_CLS_PART_JOINT:
        cls = True
        part = True
    elif state == _S_PART_JOINT:
        part = True
        eos = True
    elif state == _S_BRANCH_PART_JOINT:
        part = True
        branch = True
        eos = True
    # every non-BOS state allows joint coordinates
    for t in range(UR_DISCRETE):
        ok[t] = True
    if part:
        ok[UR_SPRING] = True
        ok[UR_PART_BODY] = True
        ok[UR_PART_HAND] = True
    if cls:
        for t in range(UR_CLS_NONE, UR_CLS_ARTICULATION + 1):
            ok[t] = True
    if branch:
        ok[UR_BRANCH] = True
    if eos:
        ok[UR_EOS] = True
    return ok^


@always_inline
def undiscretize(t: Int) -> Float32:
    """Token -> coordinate in [-1, 1] (bin centre)."""
    return (Float32(t) + 0.5) / Float32(UR_DISCRETE) * 2.0 - 1.0


@fieldwise_init
struct UrVec3(Copyable, Movable, ImplicitlyCopyable):
    var x: Float32
    var y: Float32
    var z: Float32

    def __add__(self, o: UrVec3) -> UrVec3:
        return UrVec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: UrVec3) -> UrVec3:
        return UrVec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: Float32) -> UrVec3:
        return UrVec3(self.x * s, self.y * s, self.z * s)

    def dot(self, o: UrVec3) -> Float32:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def length(self) -> Float32:
        return sqrt(self.dot(self))


struct UrSkeleton(Copyable, Movable):
    """Joints (bone heads), tails and parents (-1 for the root), in the space they were made in."""
    var joints: List[UrVec3]
    var tails: List[UrVec3]
    var parents: List[Int]
    var names: List[String]
    var cls: Int
    var tokens: List[Int]

    def __init__(out self):
        self.joints = List[UrVec3]()
        self.tails = List[UrVec3]()
        self.parents = List[Int]()
        self.names = List[String]()
        self.cls = UR_CLS_NONE
        self.tokens = List[Int]()

    def num_bones(self) -> Int:
        return len(self.joints)

    def transformed(self, center: UrVec3, scale: Float32) -> UrSkeleton:
        """p * scale + center for every joint and tail (undoes the [-1, 1] normalisation)."""
        var s = self.copy()
        for i in range(len(s.joints)):
            s.joints[i] = s.joints[i] * scale + center
            s.tails[i] = s.tails[i] * scale + center
        return s^


def make_skeleton(joints: List[UrVec3], p_joints: List[UrVec3], tails_in: List[Int], tails_pos: List[UrVec3],
                  extrude_scale: Float32 = 0.5) -> UrSkeleton:
    """UniRig's `make_skeleton` with leaf and branch tails extruded: each joint's parent is the
    nearest earlier joint to its recorded parent position (the latest one on ties)."""
    var n = len(joints)
    var sk = UrSkeleton()
    var heads = List[UrVec3]()
    for i in range(n):
        if i == 0:
            sk.parents.append(-1)
            heads.append(joints[0])
            continue
        var best = 999999.0
        var pid = -1
        var j = i - 1
        while j >= 0:
            var d = joints[j] - p_joints[i]
            var dd = Float64(d.dot(d))
            if dd < best:
                best = dd
                pid = j
            j -= 1
        sk.parents.append(pid)
    var children = List[List[Int]]()
    for _ in range(n):
        children.append(List[Int]())
    for i in range(n):
        if sk.parents[i] >= 0:
            children[sk.parents[i]].append(i)
    var tails = List[UrVec3](length=n, fill=UrVec3(0, 0, 0))
    var has_tail = List[Bool](length=n, fill=False)
    for k in range(len(tails_in)):
        tails[tails_in[k]] = tails_pos[k]
        has_tail[tails_in[k]] = True
    for i in range(n):
        if len(children[i]) != 0:
            continue
        var pid = sk.parents[i]
        var d = joints[i] - joints[pid] if pid >= 0 else UrVec3(0, 0, 0)
        if pid < 0 or d.length() <= 1e-9:
            d = UrVec3(0, 0, 1)
        tails[i] = joints[i] + d * extrude_scale
    for i in range(n):
        if len(children[i]) <= 1:
            continue
        var pid = sk.parents[i]
        if pid < 0:
            var av: Float32 = 0
            for c in children[i]:
                av += (joints[i] - joints[c]).length()
            av /= Float32(len(children[i]))
            # UniRig adds the root's own position here: tail = 1.5 * joint + (0, 0, 0.25 * av)
            var dd = joints[i] + UrVec3(0, 0, extrude_scale * av)
            tails[i] = joints[i] + dd * extrude_scale
        else:
            var d = joints[i] - joints[pid]
            if d.length() <= 1e-9:
                d = UrVec3(0, 0, 1)
            tails[i] = joints[i] + d * extrude_scale
    for i in range(n):
        if len(children[i]) == 1:
            tails[i] = joints[children[i][0]]
    sk.joints = joints.copy()
    sk.tails = tails^
    for i in range(n):
        sk.names.append("bone_" + String(i))
    return sk^


def detokenize(ids: List[Int]) raises -> UrSkeleton:
    """BOS ... EOS token ids -> skeleton in the normalised [-1, 1] space."""
    if len(ids) < 2 or ids[0] != UR_BOS:
        raise Error("UniRig: token sequence does not start with BOS")
    var end = len(ids)
    while end > 0 and ids[end - 1] == UR_PAD:
        end -= 1
    if ids[end - 1] != UR_EOS:
        raise Error("UniRig: token sequence does not end with EOS")
    var joints = List[UrVec3]()
    var p_joints = List[UrVec3]()
    var tail_ids = List[Int]()
    var tail_pos = List[UrVec3]()
    var is_branch = False
    var has_last = False
    var last = UrVec3(0, 0, 0)
    var cls = UR_CLS_NONE
    var i = 1
    var stop = end - 1
    while i < stop:
        var t = ids[i]
        if t < UR_DISCRETE:
            var cur: UrVec3
            if is_branch:
                if i + 6 > stop:
                    raise Error("UniRig: truncated branch")
                var pj = UrVec3(undiscretize(ids[i]), undiscretize(ids[i + 1]), undiscretize(ids[i + 2]))
                cur = UrVec3(undiscretize(ids[i + 3]), undiscretize(ids[i + 4]), undiscretize(ids[i + 5]))
                joints.append(cur)
                p_joints.append(pj)
                i += 6
            else:
                if i + 3 > stop:
                    raise Error("UniRig: truncated joint")
                cur = UrVec3(undiscretize(ids[i]), undiscretize(ids[i + 1]), undiscretize(ids[i + 2]))
                joints.append(cur)
                if len(p_joints) == 0:
                    p_joints.append(cur)
                else:
                    p_joints.append(last)
                i += 3
            if has_last:
                tail_ids.append(len(joints) - 2)
                tail_pos.append(cur)
            last = cur
            has_last = True
            is_branch = False
        elif t == UR_BRANCH:
            is_branch = True
            has_last = False
            i += 1
        elif _is_cls(t):
            cls = t
            i += 1
        else:
            i += 1      # part / spring markers
    if len(joints) == 0:
        raise Error("UniRig: no joints generated")
    var sk = make_skeleton(joints, p_joints, tail_ids, tail_pos)
    sk.cls = cls
    sk.tokens = ids.copy()
    return sk^


struct UrRng(Copyable, Movable):
    """SplitMix64: a small seedable generator for point sampling and beam sampling."""
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next_u64(mut self) -> UInt64:
        self.s += 0x9E3779B97F4A7C15
        var z = self.s
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def uniform(mut self) -> Float64:
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def below(mut self, n: Int) -> Int:
        return Int(self.next_u64() % UInt64(n))


struct UrGenConfig(Copyable, Movable):
    """UniRig's `generate_kwargs` (configs/system/ar_inference_articulationxl.yaml)."""
    var max_new_tokens: Int
    var num_beams: Int
    var do_sample: Bool
    var top_k: Int
    var top_p: Float32
    var repetition_penalty: Float32
    var temperature: Float32
    var cls: Int
    var seed: UInt64

    def __init__(out self):
        self.max_new_tokens = 2048
        self.num_beams = 15
        self.do_sample = True
        self.top_k = 5
        self.top_p = 0.95
        self.repetition_penalty = 3.0
        self.temperature = 1.5
        self.cls = UR_CLS_ARTICULATION
        self.seed = 0

    @staticmethod
    def greedy() -> UrGenConfig:
        var c = UrGenConfig()
        c.num_beams = 1
        c.do_sample = False
        c.repetition_penalty = 1.0
        c.temperature = 1.0
        c.top_k = 0
        c.top_p = 1.0
        return c^


def _log_softmax(mut row: List[Float32], off: Int, n: Int):
    var m: Float32 = -3.4e38
    for i in range(n):
        m = max(m, row[off + i])
    var s: Float64 = 0
    for i in range(n):
        s += Float64(exp(row[off + i] - m))
    var lse = m + Float32(log(s))
    for i in range(n):
        row[off + i] = row[off + i] - lse


def _process(mut scores: List[Float32], off: Int, seq: List[Int], gen_from: Int, cfg: UrGenConfig):
    """HF logits processors in UniRig's order: repetition penalty (on generated tokens), the
    grammar mask, then temperature, top-k and top-p when sampling (min 2 kept for beams)."""
    var n = UR_VOCAB
    var NEG = Float32(-3.4e38)
    if cfg.repetition_penalty != 1.0:
        var seen = List[Bool](length=n, fill=False)
        for j in range(gen_from, len(seq)):
            seen[seq[j]] = True
        for t in range(n):
            if seen[t]:
                var v = scores[off + t]
                scores[off + t] = v * cfg.repetition_penalty if v < 0 else v / cfg.repetition_penalty
    var ok = allowed_tokens(grammar_state(seq))
    for t in range(n):
        if not ok[t]:
            scores[off + t] = NEG
    if not cfg.do_sample:
        return
    if cfg.temperature != 1.0:
        for t in range(n):
            if scores[off + t] > NEG:
                scores[off + t] = scores[off + t] / cfg.temperature
    var keep_min = 2 if cfg.num_beams > 1 else 1
    if cfg.top_k > 0:
        var k = min(max(cfg.top_k, keep_min), n)
        var vals = List[Float32]()
        for t in range(n):
            vals.append(scores[off + t])
        # k-th largest
        for a in range(k):
            var best = a
            for b in range(a + 1, n):
                if vals[b] > vals[best]:
                    best = b
            var tmp = vals[a]
            vals[a] = vals[best]
            vals[best] = tmp
        var kth = vals[k - 1]
        for t in range(n):
            if scores[off + t] < kth:
                scores[off + t] = NEG
    if cfg.top_p < 1.0:
        # ascending sort; drop the low tail whose cumulative probability <= 1 - top_p
        var idx = List[Int]()
        for t in range(n):
            idx.append(t)
        for a in range(1, n):
            var j = a
            while j > 0 and scores[off + idx[j - 1]] > scores[off + idx[j]]:
                var tmp = idx[j - 1]
                idx[j - 1] = idx[j]
                idx[j] = tmp
                j -= 1
        var m: Float32 = -3.4e38
        for t in range(n):
            m = max(m, scores[off + t])
        var tot: Float64 = 0
        for t in range(n):
            if scores[off + t] > NEG:
                tot += Float64(exp(scores[off + t] - m))
        var cum: Float64 = 0
        for r in range(n - keep_min):
            var t = idx[r]
            if scores[off + t] > NEG:
                cum += Float64(exp(scores[off + t] - m)) / tot
            if cum <= Float64(1.0 - cfg.top_p):
                scores[off + t] = NEG
            else:
                break


def generate_skeleton(mut dec: OptDecoder, cond: Tensor, cfg: UrGenConfig) raises -> UrSkeleton:
    """Generates a skeleton from mesh latents `cond` [M, hidden] (normalised space)."""
    var hid = dec.hidden
    var M = cond.shape[0]
    var start = List[Int]()
    start.append(UR_BOS)
    start.append(cfg.cls)
    var pre = Tensor.zeros([M + len(start), hid])
    var pc = fptr_at(cond.fptr())
    for i in range(M * hid):
        pre.f[i] = pc.unsafe_offset(i).unsafe_load[alignment=1]()
    for s in range(len(start)):
        var e = dec.embed(start[s])
        for i in range(hid):
            pre.f[(M + s) * hid + i] = e[i]
    var B = max(cfg.num_beams, 1)
    var steps_cap = min(cfg.max_new_tokens, dec.max_pos - (M + len(start)))
    var first = dec.start(pre, steps_cap, B)
    var gen_from = len(start)
    var NEG = Float32(-3.4e38)
    if B == 1:
        var seq = start.copy()
        var logits = first^
        var anc = List[Int32]()
        for step in range(steps_cap):
            _log_softmax(logits, 0, UR_VOCAB)
            _process(logits, 0, seq, gen_from, cfg)
            var best = 0
            for t in range(1, UR_VOCAB):
                if logits[t] > logits[best]:
                    best = t
            seq.append(best)
            if best == UR_EOS:
                break
            anc.append(0)
            var toks = List[Int]()
            toks.append(best)
            logits = dec.step(toks, step, anc)
        if seq[len(seq) - 1] != UR_EOS:
            seq.append(UR_EOS)
        var out = List[Int]()
        out.append(UR_BOS)
        for j in range(1, len(seq)):
            out.append(seq[j])
        return detokenize(out)

    # ---- beam search (HF `_beam_search`, sampling continuations when do_sample)
    var rng = UrRng(cfg.seed * 2654435761 + 12345)
    var running = List[List[Int]]()
    var run_score = List[Float32]()
    var slot_hist = List[List[Int32]]()   # per running beam: slot beam of each generated step
    for b in range(B):
        running.append(start.copy())
        run_score.append(Float32(0.0) if b == 0 else Float32(-1e9))
        slot_hist.append(List[Int32]())
    var done = List[List[Int]]()
    var done_score = List[Float32]()
    var logits = List[Float32](length=B * UR_VOCAB, fill=0.0)
    for b in range(B):
        for t in range(UR_VOCAB):
            logits[b * UR_VOCAB + t] = first[t]
    var keep = 2 * B
    var cur_len = 0
    for step in range(steps_cap):
        var acc = List[Float32](length=B * UR_VOCAB, fill=NEG)
        for b in range(B):
            _log_softmax(logits, b * UR_VOCAB, UR_VOCAB)
            _process(logits, b * UR_VOCAB, running[b], gen_from, cfg)
            for t in range(UR_VOCAB):
                var v = logits[b * UR_VOCAB + t]
                acc[b * UR_VOCAB + t] = v + run_score[b] if v > NEG else NEG
        # top `keep` continuations: sampled without replacement from softmax(acc), or the best
        var picks = List[Int]()
        if cfg.do_sample:
            var m: Float32 = NEG
            for v in acc:
                m = max(m, v)
            var w = List[Float64](length=len(acc), fill=0.0)
            var tot: Float64 = 0
            for i in range(len(acc)):
                if acc[i] > NEG:
                    w[i] = Float64(exp(acc[i] - m))
                    tot += w[i]
            for _ in range(keep):
                if tot <= 0:
                    break
                var r = rng.uniform() * tot
                var pick = -1
                for i in range(len(w)):
                    if w[i] > 0:
                        pick = i
                        r -= w[i]
                        if r <= 0:
                            break
                if pick < 0:
                    break
                picks.append(pick)
                tot -= w[pick]
                w[pick] = 0
        else:
            var taken = List[Bool](length=len(acc), fill=False)
            for _ in range(keep):
                var best = -1
                for i in range(len(acc)):
                    if not taken[i] and acc[i] > NEG and (best < 0 or acc[i] > acc[best]):
                        best = i
                if best < 0:
                    break
                taken[best] = True
                picks.append(best)
        # finished candidates (EOS among the first `B` picks) -> hypotheses, length normalised
        var cand_seq = List[List[Int]]()
        var cand_score = List[Float32]()
        var cand_src = List[Int]()
        var cand_tok = List[Int]()
        var cand_eos = List[Bool]()
        for r in range(len(picks)):
            var src = picks[r] // UR_VOCAB
            var tok = picks[r] % UR_VOCAB
            var sq = running[src].copy()
            sq.append(tok)
            cand_seq.append(sq^)
            cand_score.append(acc[picks[r]])
            cand_src.append(src)
            cand_tok.append(tok)
            cand_eos.append(tok == UR_EOS)
        for r in range(min(B, len(picks))):
            if cand_eos[r]:
                done.append(cand_seq[r].copy())
                done_score.append(cand_score[r] / Float32(cur_len + 1))
        # keep the best `B` finished hypotheses
        while len(done) > B:
            var worst = 0
            for i in range(1, len(done)):
                if done_score[i] < done_score[worst]:
                    worst = i
            _ = done.pop(worst)
            _ = done_score.pop(worst)
        # next running beams: best non-EOS candidates
        var order = List[Int]()
        var used = List[Bool](length=len(picks), fill=False)
        for _ in range(B):
            var best = -1
            for r in range(len(picks)):
                if not used[r] and not cand_eos[r] and (best < 0 or cand_score[r] > cand_score[best]):
                    best = r
            if best < 0:
                break
            used[best] = True
            order.append(best)
        if len(order) == 0:
            break
        # stop when no running beam can beat the worst kept hypothesis (early_stopping=False)
        cur_len += 1
        if len(done) == B:
            var worst_done: Float32 = 3.4e38
            for v in done_score:
                worst_done = min(worst_done, v)
            if cand_score[order[0]] / Float32(cur_len) <= worst_done:
                break
        var new_running = List[List[Int]]()
        var new_score = List[Float32]()
        var new_hist = List[List[Int32]]()
        var toks = List[Int]()
        for b in range(len(order)):
            var r = order[b]
            new_running.append(cand_seq[r].copy())
            new_score.append(cand_score[r])
            var h = slot_hist[cand_src[r]].copy()
            h.append(Int32(b))
            new_hist.append(h^)
            toks.append(cand_tok[r])
        while len(new_running) < B:
            new_running.append(new_running[0].copy())
            new_score.append(-1e9)
            var h = new_hist[0].copy()
            h[len(h) - 1] = Int32(len(new_running) - 1)
            new_hist.append(h^)
            toks.append(toks[0])
        running = new_running^
        run_score = new_score^
        slot_hist = new_hist^
        # the cache slot for this step is written per new beam row; history pointers for earlier
        # steps refer to the slots their ancestors wrote
        var anc = List[Int32]()
        for b in range(B):
            for g in range(len(slot_hist[b])):
                anc.append(slot_hist[b][g])
        logits = dec.step(toks, step, anc)
    if len(done) == 0:
        var s = running[0].copy()
        s.append(UR_EOS)
        done.append(s^)
        done_score.append(run_score[0])
    var best = 0
    for i in range(1, len(done)):
        if done_score[i] > done_score[best]:
            best = i
    return detokenize(done[best])
