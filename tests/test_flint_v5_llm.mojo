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
  6. Speculative decoding on two tiny random Qwen3 checkpoints written to a temp directory
     (target, and a perturbed copy as the draft): `rewind` restores the cache exactly, greedy
     speculative output equals the target's plain greedy output for k = 1..7 (with and without
     a repetition penalty and dynamic draft length; CPU and GPU when present), a draft identical
     to the target is always accepted, seeded sampling reproduces; the acceptance rule
     `spec_accept` emits tokens distributed as the target (chi-square-style bound over 200k
     draws, including draft support the target lacks).
  7. Embeddings: `eval_hidden` on a checkpoint saved without the `model.` prefix equals the
     prefixed one, `TextEmbedder` returns unit vectors whose MRL truncation equals truncating
     the full vector, prefix reuse between texts changes nothing, the instruct query format;
     `EmbeddingIndex` top-k order, kind filter, MRL on insert and save / load round trip.
With `NPC_LLM_DIR` (a Qwen3 / SmolLM3 checkpoint directory): int8 load (GPU when present), a
greedy chat reply, and an NPC decision that names one of the actions. With `NPC_LLM_DRAFT_DIR`
too (Qwen3-0.6B for a Qwen3-1.7B+ target): speculative greedy output equals plain greedy.
With `QWEN3_EMBED_DIR` (Qwen3-Embedding-0.6B): game-content queries find their documents.
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
from flint.platform.os import env_now, make_dirs, write_bytes
from flint.ml.llm.sample import SparseDist, Rng
from flint.ml.llm.speculative import speculative_tokens, spec_accept
from flint.ml.llm.embed import TextEmbedder, EmbeddingIndex, instruct_query, mrl_truncate, l2_normalize
from std.memory import bitcast


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
    return BpeTokenizer.from_json(List[UInt8](_tiny_tokenizer_json(digits3).as_bytes()))


def _tiny_tokenizer_json(digits3: Bool) -> String:
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
    return j^


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


# ---------------------------------------------------------------- tiny checkpoints

comptime T_H = 128
comptime T_L = 2
comptime T_HEADS = 2
comptime T_KV = 1
comptime T_HD = 64
comptime T_F = 256
comptime T_V = 384


struct _StWriter(Movable):
    var names: List[String]
    var shapes: List[List[Int]]
    var data: List[Float32]

    def __init__(out self):
        self.names = List[String]()
        self.shapes = List[List[Int]]()
        self.data = List[Float32]()

    def add(mut self, name: String, var shape: List[Int], vals: List[Float32]):
        self.names.append(name)
        self.shapes.append(shape^)
        for v in vals:
            self.data.append(v)

    def write(self, path: String) raises:
        var hdr = String("{")
        var off = 0
        for i in range(len(self.names)):
            var n = 1
            var sh = String("[")
            for j in range(len(self.shapes[i])):
                n *= self.shapes[i][j]
                sh += ("," if j > 0 else "") + String(self.shapes[i][j])
            sh += "]"
            hdr += ("," if i > 0 else "") + "\"" + self.names[i] + "\":{\"dtype\":\"F32\",\"shape\":" + sh \
                + ",\"data_offsets\":[" + String(off) + "," + String(off + n * 4) + "]}"
            off += n * 4
        hdr += "}"
        while hdr.byte_length() % 8 != 0:
            hdr += " "
        var b = List[UInt8]()
        var hl = hdr.byte_length()
        for i in range(8):
            b.append(UInt8((hl >> (8 * i)) & 255))
        for c in hdr.as_bytes():
            b.append(c)
        for v in self.data:
            var u = bitcast[DType.uint32](v)
            for i in range(4):
                b.append(UInt8((u >> UInt32(8 * i)) & 255))
        if not write_bytes(path, b):
            raise Error("cannot write " + path)


def _rand(mut seed: UInt64, n: Int, scale: Float32, base: Float32 = 0.0) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        seed = seed * 6364136223846793005 + 1442695040888963407
        out.append(base + scale * (Float32(Int((seed >> 33) % 20001) - 10000) / 10000.0))
    return out^


def _shape(a: Int, b: Int = -1) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    if b >= 0:
        s.append(b)
    return s^


def _perturbed(v: List[Float32], mut seed: UInt64, rel: Float32) -> List[Float32]:
    var noise = _rand(seed, len(v), 1.0)
    var out = List[Float32](capacity=len(v))
    for i in range(len(v)):
        out.append(v[i] * (1.0 + rel * noise[i]))
    return out^


def _write_tiny(dir: String, prefix: String, noise: Float32, noise_seed: UInt64) raises:
    """A random 2-layer Qwen3 (hidden 128, 2 query heads of 64, 1 KV head, vocab 384, tied
    embeddings) in float32; `noise` > 0 multiplies every weight by (1 + noise * U[-1, 1])."""
    _ = make_dirs(dir)
    var cfg = String("{\"model_type\": \"qwen3\", \"hidden_size\": ") + String(T_H) + ", \"num_hidden_layers\": " \
        + String(T_L) + ", \"num_attention_heads\": " + String(T_HEADS) + ", \"num_key_value_heads\": " + String(T_KV) \
        + ", \"head_dim\": " + String(T_HD) + ", \"intermediate_size\": " + String(T_F) + ", \"vocab_size\": " \
        + String(T_V) + ", \"rms_norm_eps\": 1e-06, \"rope_theta\": 10000.0, \"tie_word_embeddings\": true, " \
        + "\"eos_token_id\": 262, \"rope_scaling\": null}"
    var cb = List[UInt8](cfg.as_bytes())
    _ = write_bytes(dir + "/config.json", cb)
    var tj = _tiny_tokenizer_json(False)
    var tb = List[UInt8](tj.as_bytes())
    _ = write_bytes(dir + "/tokenizer.json", tb)
    var seed: UInt64 = 2024
    var ns = noise_seed
    var w = _StWriter()

    def put(mut w: _StWriter, name: String, var shape: List[Int], vals: List[Float32], noise: Float32,
            mut ns: UInt64):
        if noise > 0:
            w.add(name, shape^, _perturbed(vals, ns, noise))
        else:
            w.add(name, shape^, vals)

    put(w, prefix + "embed_tokens.weight", _shape(T_V, T_H), _rand(seed, T_V * T_H, 1.0), noise, ns)
    for l in range(T_L):
        var p = prefix + "layers." + String(l) + "."
        var sa = Float32(1.7) / Float32(11.3)
        put(w, p + "input_layernorm.weight", _shape(T_H), _rand(seed, T_H, 0.1, 1.0), noise, ns)
        put(w, p + "post_attention_layernorm.weight", _shape(T_H), _rand(seed, T_H, 0.1, 1.0), noise, ns)
        put(w, p + "self_attn.q_proj.weight", _shape(T_HEADS * T_HD, T_H), _rand(seed, T_HEADS * T_HD * T_H, sa), noise, ns)
        put(w, p + "self_attn.k_proj.weight", _shape(T_KV * T_HD, T_H), _rand(seed, T_KV * T_HD * T_H, sa), noise, ns)
        put(w, p + "self_attn.v_proj.weight", _shape(T_KV * T_HD, T_H), _rand(seed, T_KV * T_HD * T_H, sa), noise, ns)
        put(w, p + "self_attn.o_proj.weight", _shape(T_H, T_HEADS * T_HD), _rand(seed, T_H * T_HEADS * T_HD, sa), noise, ns)
        put(w, p + "self_attn.q_norm.weight", _shape(T_HD), _rand(seed, T_HD, 0.1, 1.0), noise, ns)
        put(w, p + "self_attn.k_norm.weight", _shape(T_HD), _rand(seed, T_HD, 0.1, 1.0), noise, ns)
        put(w, p + "mlp.gate_proj.weight", _shape(T_F, T_H), _rand(seed, T_F * T_H, sa), noise, ns)
        put(w, p + "mlp.up_proj.weight", _shape(T_F, T_H), _rand(seed, T_F * T_H, sa), noise, ns)
        put(w, p + "mlp.down_proj.weight", _shape(T_H, T_F), _rand(seed, T_H * T_F, Float32(1.7) / Float32(16.0)), noise, ns)
    put(w, prefix + "norm.weight", _shape(T_H), _rand(seed, T_H, 0.1, 1.0), noise, ns)
    w.write(dir + "/model.safetensors")


def _tmp_root() -> String:
    var t = env_now("TMPDIR")
    if t.byte_length() == 0:
        t = "/tmp"
    return t + "/flint-llm-test-" + String(Int(external_call["getpid", Int32]()))


def _same_f(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _plain_greedy(mut m: LlmModel, prompt: List[Int], n: Int, mut s: Sampler, stops: List[Int]) raises -> List[Int]:
    m.reset()
    var out = List[Int]()
    var lg = m.prompt(prompt)
    for _ in range(n):
        var t = s.sample(lg^, out)
        var stop = False
        for x in stops:
            if t == x:
                stop = True
        if stop:
            break
        out.append(t)
        lg = m.push(t)
    return out^


def test_speculative_tiny() raises:
    var root = _tmp_root()
    _write_tiny(root + "/target", "model.", 0.0, 1)
    _write_tiny(root + "/draft", "model.", 0.6, 99)
    var stops = List[Int]()
    stops.append(262)
    var prompt = _ids(5, 17, 99, 260, 3, 42, 42, 7)
    var devices = List[Bool]()
    devices.append(False)
    devices.append(True)
    for gpu in devices:
        var target = LlmModel.load(root + "/target", Q_F32, gpu, 256)
        var draft = LlmModel.load(root + "/draft", Q_F32, gpu, 256)
        if gpu and not target.on_gpu():
            print("  speculative (GPU): skipped, no GPU")
            break
        # rewind: evaluating past a rewind point gives exactly a fresh model's logits
        var ext = prompt.copy()
        ext.append(11)
        ext.append(12)
        _ = target.prompt(ext)
        target.rewind(len(prompt) - 1)
        assert_equal(target.cached(), len(prompt) - 1)
        var after = target.push(prompt[len(prompt) - 1])
        target.reset()
        var fresh = target.prompt(prompt)
        assert_true(_same_f(after, fresh), "rewind restores the cache exactly")
        for pen in [Float32(1.0), Float32(1.3)]:
            var ps = Sampler(0.0, 1.0, 0, 1, pen)
            var base = _plain_greedy(target, prompt, 40, ps, stops)
            assert_true(len(base) > 8, "the tiny target generates")
            var accepted_some = False
            var rejected_some = False
            for k in range(1, 8):
                for mp in [0.0, 0.5]:
                    target.reset()
                    draft.reset()
                    var ss = Sampler(0.0, 1.0, 0, 1, pen)
                    var g = speculative_tokens(target, draft, prompt, ss, 40, stops, k, mp)
                    assert_true(_eq(g.tokens, base), "greedy speculative == plain greedy (k " + String(k) + ")")
                    if g.stats.accepted > 0:
                        accepted_some = True
                    if g.stats.accepted < g.stats.drafted:
                        rejected_some = True
                    assert_equal(g.stats.emitted, len(g.tokens) - 1)
            assert_true(accepted_some and rejected_some, "the perturbed draft is partly accepted")
            # a draft identical to the target is always accepted
            target.reset()
            var same = LlmModel.load(root + "/target", Q_F32, gpu, 256)
            var ss2 = Sampler(0.0, 1.0, 0, 1, pen)
            var g2 = speculative_tokens(target, same, prompt, ss2, 40, stops, 4)
            assert_true(_eq(g2.tokens, base), "self-draft output")
            assert_equal(g2.stats.accepted, g2.stats.drafted)
        # sampling: valid tokens, seeded runs reproduce, target cache ends consistent
        var runs = List[List[Int]]()
        for _ in range(2):
            target.reset()
            draft.reset()
            var sm = Sampler(0.9, 0.95, 40, 1234)
            var g = speculative_tokens(target, draft, prompt, sm, 48, stops, 4)
            for t in g.tokens:
                assert_true(t >= 0 and t < T_V, "sampled token in range")
            assert_equal(target.cached(), len(prompt) + len(g.tokens) - 1)
            runs.append(g.tokens.copy())
        assert_true(_eq(runs[0], runs[1]), "seeded speculative sampling reproduces")
        print("  speculative decoding (", target.device(), "): greedy-exact for k = 1..7, rewind exact, sampling ok")
    _ = root.byte_length()


def _dist(ids: List[Int], p: List[Float64]) -> SparseDist:
    var d = SparseDist()
    for i in range(len(ids)):
        d.ids.append(ids[i])
        d.p.append(p[i])
    return d^


def test_spec_accept() raises:
    # target p over 0..4, draft q puts mass on token 5 which p lacks and misses token 4
    var p = _dist(_ids(0, 1, 2, 3, 4), [0.4, 0.25, 0.2, 0.1, 0.05])
    var q = _dist(_ids(0, 1, 2, 3, 5), [0.1, 0.4, 0.2, 0.2, 0.1])
    var rng = Rng(77)
    var n = 200000
    var hist = List[Int](length=6, fill=0)
    var acc = 0
    for _ in range(n):
        var x = q.draw(rng)
        var pick = spec_accept(p, q, x, rng)
        if pick.accepted:
            acc += 1
            assert_equal(pick.token, x)
        else:
            assert_true(pick.token != x or p.prob(x) > 0, "a rejection resamples elsewhere")
        hist[pick.token] += 1
    assert_equal(hist[5], 0)
    for t in range(5):
        var f = Float64(hist[t]) / Float64(n)
        var want = p.prob(t)
        var sd = (want * (1.0 - want) / Float64(n)) ** 0.5
        assert_true(abs(f - want) < 5.0 * sd + 1e-4, "token " + String(t) + " frequency " + String(f) + " vs " + String(want))
    # acceptance rate = sum min(p, q) = 0.1 + 0.25 + 0.2 + 0.1 = 0.65
    var ar = Float64(acc) / Float64(n)
    assert_true(abs(ar - 0.65) < 0.005, "acceptance rate " + String(ar))
    # distribution() is what sample() draws from
    var sm = Sampler(0.7, 0.9, 3, 5)
    var lg = _logits(1.0, 2.0, 0.5, 1.8, -1.0)
    var none = List[Int]()
    var d = sm.distribution(lg.copy(), none)
    assert_equal(len(d.ids), 3)
    var tot: Float64 = 0
    for x in d.p:
        tot += x
    assert_true(abs(tot - 1.0) < 1e-9, "normalised")
    var counts = List[Int](length=5, fill=0)
    for _ in range(40000):
        counts[sm.sample(lg.copy(), none)] += 1
    for i in range(len(d.ids)):
        assert_true(abs(Float64(counts[d.ids[i]]) / 40000.0 - d.p[i]) < 0.015, "sample matches distribution")
    assert_equal(Sampler.greedy().distribution(lg.copy(), none).ids[0], 1)
    print("  speculative acceptance rule: emits the target distribution, acceptance = sum min(p, q)")


def test_embed_tiny() raises:
    var root = _tmp_root()
    _write_tiny(root + "/causal", "model.", 0.0, 1)
    _write_tiny(root + "/bare", "", 0.0, 1)
    var gpus = List[Bool]()
    gpus.append(False)
    gpus.append(True)
    for gpu in gpus:
        var a = LlmModel.load(root + "/causal", Q_F32, gpu, 256)
        if gpu and not a.on_gpu():
            break
        var b = LlmModel.load(root + "/bare", Q_F32, gpu, 256, False)
        assert_equal(b.prefix, "")
        var ids = _ids(3, 9, 27, 81, 243)
        var ha = a.eval_hidden(ids)
        var hb = b.eval_hidden(ids)
        assert_equal(len(ha), T_H)
        assert_true(_same_f(ha, hb), "bare checkpoint == prefixed checkpoint")
        var raised = False
        try:
            _ = b.eval(ids, False)
        except:
            raised = True
        assert_true(raised, "a head-less model refuses logits")
        var em = TextEmbedder(root + "/bare", 0, Q_F32, gpu, 64)
        var full = em.embed("hello world")
        var n2: Float64 = 0
        for x in full:
            n2 += Float64(x) * Float64(x)
        assert_true(abs(n2 - 1.0) < 1e-5, "unit length")
        var em32 = TextEmbedder(root + "/bare", 32, Q_F32, gpu, 64)
        var v32 = em32.embed("hello world")
        var t32 = mrl_truncate(full, 32)
        assert_equal(len(v32), 32)
        for i in range(32):
            assert_true(abs(v32[i] - t32[i]) < 1e-6, "MRL truncation")
        # prefix reuse between texts does not change the result
        var q1 = em.embed_query("hello", "Find it")
        _ = em.embed_query("hell", "Find it")
        em.model.reset()
        var q1b = em.embed_query("hello", "Find it")
        for i in range(len(q1)):
            assert_true(abs(q1[i] - q1b[i]) < 1e-6, "prefix reuse")
        var bad = False
        try:
            _ = TextEmbedder(root + "/bare", 16, Q_F32, gpu, 64)
        except:
            bad = True
        assert_true(bad, "dimension below 32 rejected")
    assert_true(instruct_query("Find quests", "lost girl") == "Instruct: Find quests\nQuery:lost girl", "instruct format")
    # index
    var idx = EmbeddingIndex(4)
    _ = idx.add("sword", "asset", _logits(1.0, 0.0, 0.0, 0.0, 9.0))   # MRL-truncated to 4 on insert
    _ = idx.add("mill quest", "quest", _logits(0.0, 1.0, 0.0, 0.0))
    _ = idx.add("forge line", "dialogue", _logits(0.7, 0.7, 0.0, 0.0))
    _ = idx.add("crypt wind", "asset", _logits(0.0, 0.0, 1.0, 0.2))
    var hits = idx.search(_logits(1.0, 0.1, 0.0, 0.0), 3)
    assert_equal(len(hits), 3)
    assert_true(hits[0].key == "sword" and hits[1].key == "forge line" and hits[2].key == "mill quest", "top-k order")
    assert_true(hits[0].score > hits[1].score and hits[1].score > hits[2].score, "scores descend")
    assert_true(abs(hits[0].score - Float32(1.0 / (1.01 ** 0.5))) < 1e-5, "cosine score")
    var only = idx.search(_logits(1.0, 0.1, 0.0, 0.0), 5, "asset")
    assert_true(len(only) == 2 and only[0].key == "sword" and only[1].key == "crypt wind", "kind filter")
    assert_equal(len(idx.search(_logits(1.0, 0.0, 0.0, 0.0), 0)), 0)
    _ = make_dirs(root)
    idx.save(root + "/index.flemb")
    var back = EmbeddingIndex.load(root + "/index.flemb")
    assert_equal(len(back), 4)
    assert_true(back.keys[3] == "crypt wind" and back.kinds[2] == "dialogue", "keys / kinds round trip")
    for i in range(len(idx.vecs)):
        assert_equal(back.vecs[i], idx.vecs[i])
    var junk = List[UInt8](length=20, fill=1)
    var rejected = False
    try:
        _ = EmbeddingIndex.from_bytes(junk)
    except:
        rejected = True
    assert_true(rejected, "bad index file rejected")
    print("  embeddings: head-less Qwen3, last-token pooling, MRL, prefix reuse, cosine index: ok")


def test_embed_model() raises:
    var dir = env_now("QWEN3_EMBED_DIR")
    if dir.byte_length() == 0:
        print("  Qwen3-Embedding run: skipped (set QWEN3_EMBED_DIR to Qwen3-Embedding-0.6B)")
        return
    var em = TextEmbedder(dir, 256)
    var idx = EmbeddingIndex(256)
    _ = idx.add("longsword", "asset", em.embed("asset: a steel longsword sold by the blacksmith"))
    _ = idx.add("crypt_wind", "asset", em.embed("asset: low howling wind for tombs and crypts"))
    _ = idx.add("millers_daughter", "quest", em.embed("quest: Elsa has not come home from the old mill"))
    _ = idx.add("no_gold", "dialogue", em.embed("dialogue: No coin, no steel. Come back when your purse is heavier."))
    var task = String("Given a player's request, retrieve the game asset, quest or dialogue line it refers to")
    assert_equal(idx.search(em.embed_query("where can I buy a sword", task), 1)[0].key, "longsword")
    assert_equal(idx.search(em.embed_query("spooky crypt sound", task), 1)[0].key, "crypt_wind")
    assert_equal(idx.search(em.embed_query("the missing girl at the mill", task), 1)[0].key, "millers_daughter")
    print("  Qwen3-Embedding on", em.model.device(), ": game-content queries find their documents")


def test_speculative_model() raises:
    var dir = env_now("NPC_LLM_DIR")
    var ddir = env_now("NPC_LLM_DRAFT_DIR")
    if dir.byte_length() == 0 or ddir.byte_length() == 0:
        print("  speculative model run: skipped (set NPC_LLM_DIR and NPC_LLM_DRAFT_DIR)")
        return
    var target = LlmModel.load(dir, Q_INT8, True, 1024)
    var draft = LlmModel.load(ddir, Q_INT8, True, 1024)
    var tok = BpeTokenizer.load(dir + "/tokenizer.json")
    var tmpl = ChatTemplate.for_model(target.cfg, tok)
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage("user", "Describe a blacksmith's forge in two sentences."))
    var ids = tmpl.render(tok, msgs)
    var sm = Sampler.greedy()
    var base = generate(target, tok, ids, sm, 48, tmpl.stops)
    target.reset()
    var ss = Sampler.greedy()
    var g = speculative_tokens(target, draft, ids, ss, 48, tmpl.stops, 4, 0.4)
    assert_true(_eq(g.tokens, base.tokens), "speculative greedy == plain greedy on the real models")
    print("  speculative on", target.device(), ": identical,", g.stats.acceptance(), "acceptance,",
          g.decode_tps(), "tok/s vs", base.decode_tps())


def main() raises:
    print("Flint v5 LLM runtime suite")
    test_tokenizer()
    test_sampler()
    test_grammar()
    test_quant()
    test_bt_hook()
    test_spec_accept()
    test_speculative_tiny()
    test_embed_tiny()
    test_model()
    test_speculative_model()
    test_embed_model()
    print("Flint v5 LLM runtime suite: all passed")
