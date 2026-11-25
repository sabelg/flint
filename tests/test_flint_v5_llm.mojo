"""Flint v5 small-LLM runtime suite (`test_flint_v5_llm.mojo`).

Always (no model files):
  1. Byte-level BPE: a hand-made `tokenizer.json` (byte alphabet, five merges, two special
     tokens): merges by rank, the Qwen / Llama 3 split regex (contractions, letters with a
     leading space, digit runs of 1 or 3, punctuation, newlines, trailing spaces), special
     tokens split out, `allow_special=False` keeps them as text, decode round trip.
  2. Sampler: greedy, a seeded temperature run reproduces, `top_p` near 0 is greedy, masks are
     honoured, the repetition penalty demotes a repeated token.
  3. Grammar: `{"action": enum, "target": enum, "say": string}` accepts a reply and nothing
     else, enum options sharing a prefix (`run`, `run_away`) both work, the string length cap
     holds, end-of-turn is only allowed once complete, and a mask over a vocabulary allows
     exactly the feasible tokens.
  4. Quantised products: int8 / int4 `QMat`s on the CPU match the fp32 product within the
     quantisation error; the GPU lane-interleaved layout keeps every code.
  5. BehaviorTree hook: `NpcBrain.build_choice_tree` runs exactly the action whose index is on
     the blackboard.
With `NPC_LLM_DIR` (a Qwen3 / SmolLM3 checkpoint directory): int8 load (GPU when present), a
greedy chat reply, and an NPC decision that names one of the actions.
"""
from std.math import abs
from std.testing import assert_true, assert_equal
from std.ffi import external_call
from flint.ml.llm.tokenizer import BpeTokenizer, char_class, CC_LETTER, CC_NUMBER, CC_SPACE, CC_OTHER
from flint.ml.llm.tokenizer import utf8_append
from flint.ml.llm.sample import Sampler, argmax
from flint.ml.llm.constrain import JsonGrammar, GrammarMatcher
from flint.ml.llm.quant import QMat, Q_F32, Q_INT8, Q_INT4, qmatmul, qmat_bytes
from flint.ml.llm.kernels import gpu_layout
from flint.ml.llm.safetensors import SafeTensor
from flint.ml.llm.model import LlmModel
from flint.ml.llm.chat import ChatTemplate, ChatMessage, generate
from flint.ml.llm.npc import NpcBrain, NpcAction
from flint.ai.behavior_tree import Blackboard, BtExecutionState, BT_SUCCESS
from flint.core.jobs import JobPool
from flint.platform.os import env_now


def _byte_char(b: Int) -> Int:
    if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255):
        return b
    var n = 0
    for c in range(256):
        if not ((c >= 33 and c <= 126) or (c >= 161 and c <= 172) or (c >= 174 and c <= 255)):
            if c == b:
                return 256 + n
            n += 1
    return -1


def _json_str(cps: List[Int]) -> String:
    var out = List[UInt8]()
    out.append(34)
    for cp in cps:
        if cp == 34 or cp == 92:
            out.append(92)
        utf8_append(out, cp)
    out.append(34)
    return String(unsafe_from_utf8=out)


def _tiny_tokenizer(digits3: Bool) raises -> BpeTokenizer:
    """Byte vocabulary 0..255, merges he / ll / hell / hello / Ġw, two special tokens."""
    var j = String("{\"added_tokens\": [{\"id\": 261, \"content\": \"<|im_start|>\", \"special\": true},")
    j += "{\"id\": 262, \"content\": \"<|im_end|>\", \"special\": true}],"
    j += "\"pre_tokenizer\": {\"type\": \"Sequence\", \"pretokenizers\": [{\"type\": \"Split\", \"pattern\": {\"Regex\": \""
    j += "\\\\p{N}{1,3}" if digits3 else "\\\\p{N}"
    j += "\"}}]}, \"model\": {\"type\": \"BPE\", \"vocab\": {"
    for b in range(256):
        var cps = List[Int]()
        cps.append(_byte_char(b))
        j += _json_str(cps) + ": " + String(b) + ", "
    j += "\"he\": 256, \"ll\": 257, \"hell\": 258, \"hello\": 259, "
    var gw = List[Int]()
    gw.append(_byte_char(32))
    gw.append(119)
    j += _json_str(gw) + ": 260}, \"merges\": [\"h e\", \"l l\", [\"he\", \"ll\"], \"hell o\", "
    var gs = List[Int]()
    gs.append(_byte_char(32))
    j += "[" + _json_str(gs) + ", \"w\"]]}}"
    return BpeTokenizer.from_json(List[UInt8](j.as_bytes()))


def _ids(*xs: Int) -> List[Int]:
    var out = List[Int]()
    for x in xs:
        out.append(x)
    return out^


def _eq(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def test_tokenizer() raises:
    var tok = _tiny_tokenizer(False)
    assert_equal(tok.vocab_size, 263)
    assert_true(_eq(tok.encode("hello"), _ids(259)), "hello merges fully")
    # " world": the leading space joins the word, then Ġw merges
    assert_true(_eq(tok.encode("hello world"), _ids(259, 260, 111, 114, 108, 100)), "space-prefixed word")
    # contractions split before the letters; digits one at a time (Qwen)
    assert_true(_eq(tok.encode("he's"), _ids(256, 39, 115)), "contraction 's")
    assert_true(_eq(tok.encode("1234"), _ids(49, 50, 51, 52)), "single digits")
    var tok3 = _tiny_tokenizer(True)
    assert_equal(tok3.digits, 3)
    assert_true(_eq(tok3.encode("1234"), _ids(49, 50, 51, 52)), "no digit merges, still bytes")
    # special tokens: split out unless disallowed
    var s = tok.encode("<|im_start|>hello<|im_end|>")
    assert_true(_eq(s, _ids(261, 259, 262)), "special tokens")
    var p = tok.encode("<|im_end|>", False)
    assert_true(len(p) > 1 and p[0] == 60, "allow_special=False keeps the text")
    # trailing spaces and newlines: "a  \n" -> "a", "  \n"
    var w = tok.encode("a  \nb")
    assert_true(_eq(w, _ids(97, 32, 32, 10, 98)), "whitespace runs")
    # round trips, UTF-8 included
    for text in ["hello world", "he's 1234!", "Café ½ — 日本 🙂", "x  =  y\n\n\tz  "]:
        assert_true(tok.decode(tok.encode(String(text))) == String(text), "round trip")
    assert_equal(char_class(65), CC_LETTER)
    assert_equal(char_class(0xE9), CC_LETTER)
    assert_equal(char_class(0x665), CC_NUMBER)
    assert_equal(char_class(0x3000), CC_SPACE)
    assert_equal(char_class(0x2014), CC_OTHER)
    assert_equal(char_class(0x1F642), CC_OTHER)
    assert_equal(tok.token_id("<|im_end|>"), 262)
    print("  tokenizer: ok")


def _logits(*xs: Float32) -> List[Float32]:
    var out = List[Float32]()
    for x in xs:
        out.append(x)
    return out^


def test_sampler() raises:
    var none = List[Int]()
    var g = Sampler.greedy()
    assert_equal(g.sample(_logits(0.1, 3.0, 2.9, -1.0), none), 1)
    var a = Sampler(1.0, 1.0, 0, 42)
    var b = Sampler(1.0, 1.0, 0, 42)
    var counts = List[Int](length=4, fill=0)
    for _ in range(200):
        var x = a.sample(_logits(1.0, 1.2, 0.8, 1.1), none)
        assert_equal(x, b.sample(_logits(1.0, 1.2, 0.8, 1.1), none))
        counts[x] += 1
    for c in counts:
        assert_true(c > 20, "temperature 1 explores every token")
    var nucleus = Sampler(1.0, 0.01, 0, 7)
    for _ in range(20):
        assert_equal(nucleus.sample(_logits(1.0, 1.2, 0.8, 1.1), none), 1)
    var mask = List[Bool](length=4, fill=False)
    mask[2] = True
    var m = Sampler(0.8, 0.95, 0, 3)
    for _ in range(10):
        assert_equal(m.sample_masked(_logits(1.0, 5.0, 0.8, 1.1), none, mask), 2)
    var gm = Sampler.greedy()
    assert_equal(gm.sample_masked(_logits(1.0, 5.0, 0.8, 1.1), none, mask), 2)
    var rep = Sampler(0.0, 1.0, 0, 1, 2.0)
    assert_equal(rep.sample(_logits(1.0, 1.5, 0.2), _ids(1, 1)), 0)
    print("  sampler: ok")


def _grammar() raises -> JsonGrammar:
    var names = List[String]()
    names.append("action")
    names.append("target")
    names.append("say")
    var enums = List[List[String]]()
    var acts = List[String]()
    acts.append("run")
    acts.append("run_away")
    acts.append("attack")
    enums.append(acts^)
    var tg = List[String]()
    tg.append("player")
    tg.append("none")
    enums.append(tg^)
    enums.append(List[String]())
    var lens = List[Int]()
    lens.append(0)
    lens.append(0)
    lens.append(12)
    return JsonGrammar.object_of_fields(names, enums, lens)


def _accepts_all(var mt: GrammarMatcher, s: String) raises -> Bool:
    if not mt.accepts(List[UInt8](s.as_bytes())):
        return False
    mt.feed(List[UInt8](s.as_bytes()))
    return mt.complete()


def test_grammar() raises:
    var ok1 = "{\"action\": \"run_away\", \"target\": \"player\", \"say\": \"Help me!\"}"
    var ok2 = "{\"action\": \"run\", \"target\": \"none\", \"say\": \"\"}"
    assert_true(_accepts_all(GrammarMatcher(_grammar()), ok1), "valid reply")
    assert_true(_accepts_all(GrammarMatcher(_grammar()), ok2), "shorter enum option, empty string")
    var mt = GrammarMatcher(_grammar())
    assert_true(not mt.accepts(List[UInt8]("{\"action\": \"dance\"".as_bytes())), "unknown action")
    assert_true(not mt.accepts(List[UInt8]("{\"target\"".as_bytes())), "field order")
    assert_true(not mt.accepts(List[UInt8]("{\"action\": \"run\", \"target\": \"player\", \"say\": \"0123456789abc".as_bytes())),
                "string length cap")
    assert_true(not mt.accepts(List[UInt8]("{\"action\": \"run\", \"target\": \"player\", \"say\": \"a\\".as_bytes())),
                "no backslash in strings")
    var done = GrammarMatcher(_grammar())
    done.feed(List[UInt8](ok1.as_bytes()))
    assert_true(done.done(), "nothing can follow the closing brace")
    # masks over the tiny vocabulary
    var tok = _tiny_tokenizer(False)
    var stops = List[Int]()
    stops.append(262)
    var m0 = GrammarMatcher(_grammar())
    var mask = m0.mask(tok, stops)
    assert_true(mask[123], "'{' first")
    assert_true(not mask[34] and not mask[262] and not mask[261], "nothing else at the start")
    var m1 = GrammarMatcher(_grammar())
    m1.feed(List[UInt8]("{\"action\": \"run\", \"target\": \"player\", \"say\": \"".as_bytes()))
    var m1mask = m1.mask(tok, stops)
    assert_true(m1mask[259] and m1mask[260] and m1mask[34], "plain tokens and the closing quote in a string")
    assert_true(not m1mask[92] and not m1mask[10] and not m1mask[262], "no backslash, newline or end of turn")
    var m2 = GrammarMatcher(_grammar())
    m2.feed(List[UInt8](ok2.as_bytes()))
    var m2mask = m2.mask(tok, stops)
    assert_true(m2mask[262], "end of turn once complete")
    var n_allowed = 0
    for x in m2mask:
        if x:
            n_allowed += 1
    assert_equal(n_allowed, 1)
    print("  grammar: ok")


def test_quant() raises:
    var n = 24
    var k = 256
    var m = 3
    var w = List[Float32](length=n * k, fill=0.0)
    var x = List[Float32](length=m * k, fill=0.0)
    var seed: UInt64 = 12345
    for i in range(n * k):
        seed = seed * 6364136223846793005 + 1442695040888963407
        w[i] = Float32(Int((seed >> 33) % 2001) - 1000) / 1000.0 * Float32(3.0 if i % 97 == 0 else 1.0)
    for i in range(m * k):
        seed = seed * 6364136223846793005 + 1442695040888963407
        x[i] = Float32(Int((seed >> 33) % 2001) - 1000) / 1000.0
    var pool = JobPool()
    _ = pool.start(2)
    var ref_y = List[Float32](length=m * n, fill=0.0)
    for r in range(m):
        for j in range(n):
            var s: Float32 = 0
            for i in range(k):
                s += x[r * k + i] * w[j * k + i]
            ref_y[r * n + j] = s
    for fmt in [Q_F32, Q_INT8, Q_INT4]:
        var shape = List[Int]()
        shape.append(n)
        shape.append(k)
        var srcs = List[SafeTensor]()
        srcs.append(SafeTensor("w", "F32", shape^, Int(w.unsafe_ptr())))
        var q = QMat.build(pool, srcs^, fmt)
        assert_equal(q.fmt, fmt)
        var y = List[Float32](length=m * n, fill=1.0)
        qmatmul(pool, q, Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), m, False)
        var err: Float32 = 0
        for i in range(m * n):
            err = max(err, abs(y[i] - ref_y[i]))
        var tol = Float32(1e-3 if fmt == Q_F32 else (0.2 if fmt == Q_INT8 else 1.5))
        assert_true(err < tol, "quantised product error " + String(err))
        # accumulate mode adds
        var y2 = y.copy()
        qmatmul(pool, q, Int(x.unsafe_ptr()), Int(y2.unsafe_ptr()), m, True)
        assert_true(abs(y2[5] - 2 * y[5]) < 1e-3, "accumulate")
        if fmt != Q_F32:
            var nb = qmat_bytes(fmt, n, k)
            var packed = gpu_layout(fmt, q.data, n, k)
            var hist_a = List[Int](length=256, fill=0)
            var hist_b = List[Int](length=256, fill=0)
            var pa = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=q.data)
            var pb = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=packed)
            for i in range(nb):
                var a = Int(pa.unsafe_offset(i).unsafe_load())
                var b = Int(pb.unsafe_offset(i).unsafe_load())
                if fmt == Q_INT8:
                    hist_a[a] += 1
                    hist_b[b] += 1
                else:
                    hist_a[a & 15] += 1
                    hist_a[a >> 4] += 1
                    hist_b[b & 15] += 1
                    hist_b[b >> 4] += 1
            for i in range(256):
                assert_equal(hist_a[i], hist_b[i])
            external_call["free", NoneType](packed)
        q.release_host()
    _ = len(w)
    _ = len(x)
    pool.stop()
    print("  quantised products: ok")


def test_bt_hook() raises:
    var acts = List[NpcAction]()
    acts.append(NpcAction(10, "attack", "Hit the target."))
    acts.append(NpcAction(11, "flee", "Run away."))
    acts.append(NpcAction(12, "talk", "Speak to the target."))
    var tg = List[String]()
    tg.append("player")
    var brain = NpcBrain("A guard.", acts^, tg^)
    var sp = brain.system_prompt()
    assert_true(sp.find("- flee: Run away.") >= 0 and sp.find("Targets: player") >= 0, "prompt lists actions")
    var bb = Blackboard()
    var key = bb.def_key("llm_action", -1.0)
    var bt = brain.build_choice_tree(key)
    for choice in range(3):
        bb.set_float(key, Float32(choice))
        var st = BtExecutionState(16)
        var r = bt.tick(bb, st, 0.1)
        assert_equal(Int(r), Int(BT_SUCCESS))
        assert_equal(st.active_action_id, 10 + choice)
        for other in range(3):
            assert_equal(st.action_exec_counts[10 + other], 1 if other == choice else 0)
    print("  BehaviorTree hook: ok")


def test_model() raises:
    var dir = env_now("NPC_LLM_DIR")
    if dir.byte_length() == 0:
        print("  model run: skipped (set NPC_LLM_DIR to a Qwen3 / SmolLM3 checkpoint)")
        return
    var model = LlmModel.load(dir, Q_INT8, True, 1024)
    var tok = BpeTokenizer.load(dir + "/tokenizer.json")
    var tmpl = ChatTemplate.for_model(model.cfg, tok)
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage("user", "Say hello in three words."))
    var ids = tmpl.render(tok, msgs)
    var sm = Sampler.greedy()
    var g = generate(model, tok, ids, sm, 24, tmpl.stops)
    assert_true(len(g.tokens) > 0, "a reply")
    print("  model run on", model.device(), ":", g.text)
    var acts = List[NpcAction]()
    acts.append(NpcAction(0, "attack", "Attack the target with your axe."))
    acts.append(NpcAction(1, "flee", "Run to safety."))
    acts.append(NpcAction(2, "greet", "Greet the target in a friendly way."))
    var tg = List[String]()
    tg.append("player")
    tg.append("goblin")
    var brain = NpcBrain("You are Brakka, a dwarven blacksmith.", acts^, tg^)
    var d = brain.decide(model, tok, tmpl, "A goblin with a dagger charges at you. The player is far away.")
    print("  npc decision:", d.json, "(", d.seconds, "s )")
    assert_true(d.index >= 0 and (d.target == "player" or d.target == "goblin"), "valid action and target")


def main() raises:
    print("Flint v5 LLM runtime suite")
    test_tokenizer()
    test_sampler()
    test_grammar()
    test_quant()
    test_bt_hook()
    test_model()
    print("Flint v5 LLM runtime suite: all passed")
