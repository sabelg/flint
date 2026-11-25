"""Next-token sampling (`flint.ml.llm.sample`).

`Sampler` turns logits into a token: greedy when `temperature <= 0`, otherwise softmax at
`temperature` restricted to the `top_k` most likely tokens (default 64; 0 means 1024) and then to
the smallest set whose probability reaches `top_p`. A repetition penalty (Hugging Face style:
positive logits divided, negative multiplied) applies to recently generated tokens. An optional
`allowed` mask (from a grammar) removes every other token first. The generator is a seeded
SplitMix64, so a seed reproduces a run.

`Sampler.distribution` returns the exact distribution `sample` draws from as a `SparseDist`
(token ids with normalised probabilities), which speculative decoding needs for its acceptance
rule (`flint.ml.llm.speculative`).
"""

from std.math import exp


struct Rng(Copyable, Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64 = 0x9E3779B97F4A7C15):
        self.state = seed

    def next_u64(mut self) -> UInt64:
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def uniform(mut self) -> Float64:
        """In [0, 1)."""
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)


def argmax(logits: List[Float32]) -> Int:
    var best = 0
    for i in range(1, len(logits)):
        if logits[i] > logits[best]:
            best = i
    return best


def argmax_allowed(logits: List[Float32], allowed: List[Bool]) -> Int:
    """-1 when nothing is allowed; ids past the end of the mask are not allowed."""
    var best = -1
    for i in range(min(len(logits), len(allowed))):
        if allowed[i] and (best < 0 or logits[i] > logits[best]):
            best = i
    return best


def top_candidates(logits: List[Float32], allowed: List[Bool], use_mask: Bool, k: Int) -> List[Int]:
    """Indices of the k largest (allowed) logits, sorted by decreasing logit."""
    var heap = List[Int]()   # min-heap by logit
    var n = min(len(logits), len(allowed)) if use_mask else len(logits)
    for i in range(n):
        if use_mask and not allowed[i]:
            continue
        var v = logits[i]
        if len(heap) < k:
            heap.append(i)
            var c = len(heap) - 1
            while c > 0:
                var p = (c - 1) // 2
                if logits[heap[p]] <= logits[heap[c]]:
                    break
                var t = heap[p]
                heap[p] = heap[c]
                heap[c] = t
                c = p
        elif v > logits[heap[0]]:
            heap[0] = i
            var c = 0
            while True:
                var l = 2 * c + 1
                var r = l + 1
                var s = c
                if l < len(heap) and logits[heap[l]] < logits[heap[s]]:
                    s = l
                if r < len(heap) and logits[heap[r]] < logits[heap[s]]:
                    s = r
                if s == c:
                    break
                var t = heap[s]
                heap[s] = heap[c]
                heap[c] = t
                c = s
    # sort descending (insertion sort; k is small)
    for i in range(1, len(heap)):
        var key = heap[i]
        var j = i - 1
        while j >= 0 and logits[heap[j]] < logits[key]:
            heap[j + 1] = heap[j]
            j -= 1
        heap[j + 1] = key
    return heap^


struct SparseDist(Copyable, Movable):
    """A distribution over a few token ids: `ids[i]` has probability `p[i]` (sums to 1)."""
    var ids: List[Int]
    var p: List[Float64]

    def __init__(out self):
        self.ids = List[Int]()
        self.p = List[Float64]()

    @staticmethod
    def one_hot(token: Int) -> SparseDist:
        var d = SparseDist()
        d.ids.append(token)
        d.p.append(1.0)
        return d^

    def prob(self, token: Int) -> Float64:
        for i in range(len(self.ids)):
            if self.ids[i] == token:
                return self.p[i]
        return 0.0

    def draw(self, mut rng: Rng) -> Int:
        """-1 when empty."""
        if len(self.ids) == 0:
            return -1
        var u = rng.uniform()
        for i in range(len(self.ids)):
            u -= self.p[i]
            if u < 0:
                return self.ids[i]
        return self.ids[len(self.ids) - 1]


struct Sampler(Copyable, Movable):
    var temperature: Float32
    var top_k: Int
    var top_p: Float32
    var repetition_penalty: Float32
    var penalty_window: Int
    var rng: Rng

    def __init__(out self, temperature: Float32 = 0.0, top_p: Float32 = 1.0, top_k: Int = 64, seed: UInt64 = 1,
                 repetition_penalty: Float32 = 1.0, penalty_window: Int = 64):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.penalty_window = penalty_window
        self.rng = Rng(seed)

    @staticmethod
    def greedy() -> Sampler:
        return Sampler(0.0)

    def sample(mut self, var logits: List[Float32], recent: List[Int]) -> Int:
        var none = List[Bool]()
        return self._pick(logits^, recent, none, False)

    def sample_masked(mut self, var logits: List[Float32], recent: List[Int], allowed: List[Bool]) -> Int:
        """-1 when the mask allows nothing."""
        return self._pick(logits^, recent, allowed, True)

    def is_greedy(self) -> Bool:
        return self.temperature <= 0

    def distribution(self, var logits: List[Float32], recent: List[Int]) -> SparseDist:
        """The distribution `sample(logits, recent)` draws from: a one-hot on the argmax when
        greedy, else the softmax at `temperature` over the top-k / top-p set, renormalised."""
        self._penalise(logits, recent)
        if self.temperature <= 0:
            return SparseDist.one_hot(argmax(logits))
        var none = List[Bool]()
        var k = self.top_k if self.top_k > 0 else 1024
        var cand = top_candidates(logits, none, False, k)
        var d = SparseDist()
        if len(cand) == 0:
            return d^
        var mx = logits[cand[0]]
        var probs = List[Float64](capacity=len(cand))
        var tot: Float64 = 0
        for c in cand:
            var p = exp(Float64((logits[c] - mx) / self.temperature))
            probs.append(p)
            tot += p
        var keep = len(cand)
        if self.top_p < 1.0:
            var acc: Float64 = 0
            for i in range(len(cand)):
                acc += probs[i] / tot
                if acc >= Float64(self.top_p):
                    keep = i + 1
                    break
        var mass: Float64 = 0
        for i in range(keep):
            mass += probs[i]
        for i in range(keep):
            d.ids.append(cand[i])
            d.p.append(probs[i] / mass)
        return d^

    def _penalise(self, mut logits: List[Float32], recent: List[Int]):
        if self.repetition_penalty != 1.0:
            var start = max(0, len(recent) - self.penalty_window)
            for i in range(start, len(recent)):
                var t = recent[i]
                if t >= 0 and t < len(logits):
                    var v = logits[t]
                    logits[t] = v / self.repetition_penalty if v > 0 else v * self.repetition_penalty

    def _pick(mut self, var logits: List[Float32], recent: List[Int], allowed: List[Bool], use_mask: Bool) -> Int:
        if self.repetition_penalty != 1.0:
            var start = max(0, len(recent) - self.penalty_window)
            for i in range(start, len(recent)):
                var t = recent[i]
                if t >= 0 and t < len(logits):
                    var v = logits[t]
                    logits[t] = v / self.repetition_penalty if v > 0 else v * self.repetition_penalty
        if self.temperature <= 0:
            if use_mask:
                return argmax_allowed(logits, allowed)
            return argmax(logits)
        var k = self.top_k if self.top_k > 0 else 1024
        var cand = top_candidates(logits, allowed, use_mask, k)
        if len(cand) == 0:
            return -1
        var mx = logits[cand[0]]
        var probs = List[Float64](capacity=len(cand))
        var tot: Float64 = 0
        for c in cand:
            var p = exp(Float64((logits[c] - mx) / self.temperature))
            probs.append(p)
            tot += p
        var keep = len(cand)
        if self.top_p < 1.0:
            var acc: Float64 = 0
            for i in range(len(cand)):
                acc += probs[i] / tot
                if acc >= Float64(self.top_p):
                    keep = i + 1
                    break
        var mass: Float64 = 0
        for i in range(keep):
            mass += probs[i]
        var u = self.rng.uniform() * mass
        for i in range(keep):
            u -= probs[i]
            if u <= 0:
                return cand[i]
        return cand[keep - 1]
