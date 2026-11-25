"""Chat templates and the generation loop (`flint.ml.llm.chat`).

`ChatTemplate.for_model(cfg, tok)` reproduces each family's Hugging Face template in
non-thinking mode (`enable_thinking=False`), token for token:

  - Qwen3 (ChatML): `<|im_start|>role\\ncontent<|im_end|>\\n` per message; the generation
    prompt is `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`.
  - SmolLM3: a system header `## Metadata ... Today Date: DATE ... Reasoning Mode: /no_think`
    plus `## Custom Instructions`, then ChatML turns; the generation prompt ends with an empty
    `<think>\\n\\n</think>\\n` block.

Special tokens are inserted by id and message text is encoded with `allow_special=False`, so a
player typing `<|im_end|>` cannot break out of their turn.

`generate` runs the model from a prompt with a `Sampler`, an optional `GrammarMatcher`, a token
budget and the end-of-turn ids, reusing the cached prefix of the previous call.
"""

from std.time import perf_counter_ns
from flint.ml.llm.tokenizer import BpeTokenizer
from flint.ml.llm.model import LlmModel, LlmConfig
from flint.ml.llm.sample import Sampler
from flint.ml.llm.constrain import GrammarMatcher


@fieldwise_init
struct ChatMessage(Copyable, Movable):
    var role: String        # "system", "user", "assistant"
    var content: String


struct ChatTemplate(Copyable, Movable):
    var family: String
    var date: String
    var im_start: Int
    var im_end: Int
    var think: Int
    var end_think: Int
    var stops: List[Int]

    def __init__(out self, family: String, tok: BpeTokenizer, date: String = "01 January 2026") raises:
        self.family = family
        self.date = date
        self.im_start = tok.token_id("<|im_start|>")
        self.im_end = tok.token_id("<|im_end|>")
        self.think = tok.token_id("<think>")
        self.end_think = tok.token_id("</think>")
        if self.im_start < 0 or self.im_end < 0:
            raise Error("chat: the tokenizer has no <|im_start|> / <|im_end|> tokens")
        self.stops = List[Int]()
        self.stops.append(self.im_end)
        var eot = tok.token_id("<|endoftext|>")
        if eot >= 0:
            self.stops.append(eot)
        var eot3 = tok.token_id("<|end_of_text|>")
        if eot3 >= 0:
            self.stops.append(eot3)

    @staticmethod
    def for_model(cfg: LlmConfig, tok: BpeTokenizer, date: String = "01 January 2026") raises -> ChatTemplate:
        return ChatTemplate(cfg.family, tok, date)

    def _text(self, mut tok: BpeTokenizer, s: String, mut out: List[Int]) raises:
        for t in tok.encode(s, False):
            out.append(t)

    def _empty_think(self, mut tok: BpeTokenizer, tail: String, mut out: List[Int]) raises:
        if self.think >= 0 and self.end_think >= 0:
            out.append(self.think)
            self._text(tok, "\n\n", out)
            out.append(self.end_think)
            self._text(tok, tail, out)
        else:
            self._text(tok, "<think>\n\n</think>" + tail, out)

    def render(self, mut tok: BpeTokenizer, msgs: List[ChatMessage], add_generation_prompt: Bool = True) raises -> List[Int]:
        var out = List[Int]()
        if self.family == "smollm3":
            var custom = String("")
            if len(msgs) > 0 and msgs[0].role == "system":
                custom = String(msgs[0].content.replace("/no_think", "").replace("/think", "").rstrip())
            var hdr = String("system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: ") + self.date \
                + "\nReasoning Mode: /no_think\n\n## Custom Instructions\n\n"
            if custom.byte_length() > 0:
                hdr += custom + "\n\n"
            else:
                hdr += "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n"
            out.append(self.im_start)
            self._text(tok, hdr, out)
            for m in msgs:
                if m.role == "user":
                    out.append(self.im_start)
                    self._text(tok, "user\n" + m.content, out)
                    out.append(self.im_end)
                    self._text(tok, "\n", out)
                elif m.role == "assistant":
                    out.append(self.im_start)
                    self._text(tok, "assistant\n", out)
                    self._empty_think(tok, "\n" + String(m.content.lstrip("\n")), out)
                    out.append(self.im_end)
                    self._text(tok, "\n", out)
            if add_generation_prompt:
                out.append(self.im_start)
                self._text(tok, "assistant\n", out)
                self._empty_think(tok, "\n", out)
            return out^
        # Qwen3 / ChatML
        for m in msgs:
            out.append(self.im_start)
            self._text(tok, m.role + "\n" + m.content, out)
            out.append(self.im_end)
            self._text(tok, "\n", out)
        if add_generation_prompt:
            out.append(self.im_start)
            self._text(tok, "assistant\n", out)
            self._empty_think(tok, "\n\n", out)
        return out^


struct Generation(Copyable, Movable):
    var tokens: List[Int]
    var text: String
    var prompt_tokens: Int
    var reused_tokens: Int
    var prefill_s: Float64
    var decode_s: Float64
    var stopped: Bool      # an end-of-turn token or a complete grammar ended it

    def __init__(out self):
        self.tokens = List[Int]()
        self.text = ""
        self.prompt_tokens = 0
        self.reused_tokens = 0
        self.prefill_s = 0
        self.decode_s = 0
        self.stopped = False

    def decode_tps(self) -> Float64:
        if self.decode_s <= 0 or len(self.tokens) < 2:
            return 0
        return Float64(len(self.tokens) - 1) / self.decode_s


def generate(mut model: LlmModel, mut tok: BpeTokenizer, prompt: List[Int], mut sampler: Sampler, max_tokens: Int,
             stops: List[Int]) raises -> Generation:
    """Samples up to `max_tokens` after `prompt`, stopping at any of `stops` (not included)."""
    var g = Generation()
    var reuse = 0
    while reuse < len(model.past) and reuse < len(prompt) and model.past[reuse] == prompt[reuse]:
        reuse += 1
    g.prompt_tokens = len(prompt)
    g.reused_tokens = reuse
    var t0 = perf_counter_ns()
    var logits = model.prompt(prompt)
    var t1 = perf_counter_ns()
    g.prefill_s = Float64(t1 - t0) / 1e9
    for _ in range(max_tokens):
        var t = sampler.sample(logits^, g.tokens)
        var is_stop = False
        for s in stops:
            if t == s:
                is_stop = True
        if is_stop:
            g.stopped = True
            break
        g.tokens.append(t)
        logits = model.push(t)
    g.decode_s = Float64(perf_counter_ns() - t1) / 1e9
    g.text = tok.decode(g.tokens)
    return g^


def generate_constrained(mut model: LlmModel, mut tok: BpeTokenizer, prompt: List[Int], mut sampler: Sampler,
                         max_tokens: Int, stops: List[Int], mut matcher: GrammarMatcher) raises -> Generation:
    """Like `generate`, but every token is drawn from `matcher.mask`; stops as soon as the
    grammar is complete with nothing more to add (the closing token is not fed back)."""
    var g = Generation()
    var reuse = 0
    while reuse < len(model.past) and reuse < len(prompt) and model.past[reuse] == prompt[reuse]:
        reuse += 1
    g.prompt_tokens = len(prompt)
    g.reused_tokens = reuse
    var t0 = perf_counter_ns()
    var logits = model.prompt(prompt)
    var t1 = perf_counter_ns()
    g.prefill_s = Float64(t1 - t0) / 1e9
    for _ in range(max_tokens):
        var mask = matcher.mask(tok, stops)
        var t = sampler.sample_masked(logits^, g.tokens, mask)
        if t < 0:
            break
        var is_stop = False
        for s in stops:
            if t == s:
                is_stop = True
        if is_stop:
            g.stopped = True
            break
        matcher.feed(tok.piece(t))
        g.tokens.append(t)
        if matcher.done():
            g.stopped = True
            break
        logits = model.push(t)
    g.decode_s = Float64(perf_counter_ns() - t1) / 1e9
    g.text = matcher.text()
    return g^
