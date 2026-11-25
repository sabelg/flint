"""Small-LLM NPCs in Flint: a chat CLI and an NPC that picks BehaviorTree actions.

usage:
  npc_llm chat MODEL_DIR [--cpu] [--fmt fp32|int8|int4] [--temp T] [--top-p P] [--seed N]
                         [--max N] [--system TEXT]
  npc_llm npc  MODEL_DIR [--cpu] [--fmt fp32|int8|int4] [--temp T] [--seed N]

MODEL_DIR is a Hugging Face checkpoint directory (config.json, *.safetensors, tokenizer.json)
of Qwen3-0.6B / Qwen3-1.7B / SmolLM3-3B. Default: int8 weights on the GPU when present.

`chat` streams replies; the conversation stays in the KV cache, so each turn only evaluates the
new message. `npc` runs a guard through scripted world states: the blackboard is turned into a
text observation, the LLM answers with grammar-constrained JSON naming one action, the choice
goes on the blackboard and the BehaviorTree runs it. A designer branch in front of the LLM
(health below 15: flee) shows the tree keeps the last word.
"""
from std.sys import argv
from std.collections import Optional
from std.time import perf_counter_ns
from std.ffi import external_call
from flint.ml.llm import LlmModel, BpeTokenizer, Sampler, Q_F32, Q_INT8, Q_INT4
from flint.ml.llm.chat import ChatTemplate, ChatMessage
from flint.ml.llm.npc import NpcBrain, NpcAction
from flint.ai.behavior_tree import Blackboard, BehaviorTree, BtExecutionState, BT_CMP_LT, BT_CMP_EQ


struct Opts(Copyable, Movable):
    var gpu: Bool
    var fmt: Int
    var temp: Float32
    var top_p: Float32
    var seed: UInt64
    var max_tokens: Int
    var system: String

    def __init__(out self):
        self.gpu = True
        self.fmt = Q_INT8
        self.temp = -1.0
        self.top_p = 0.9
        self.seed = 1
        self.max_tokens = 256
        self.system = "You are Brakka, a gruff but kind dwarven blacksmith in the village of Emberfall. Stay in character and keep answers short."


def parse(a: List[String]) raises -> Opts:
    var o = Opts()
    var i = 3
    while i < len(a):
        var s = a[i]
        var has = i + 1 < len(a)
        if s == "--cpu":
            o.gpu = False
        elif s == "--fmt" and has:
            o.fmt = Q_INT8 if a[i + 1] == "int8" else (Q_INT4 if a[i + 1] == "int4" else Q_F32)
            i += 1
        elif s == "--temp" and has:
            o.temp = Float32(Float64(a[i + 1]))
            i += 1
        elif s == "--top-p" and has:
            o.top_p = Float32(Float64(a[i + 1]))
            i += 1
        elif s == "--seed" and has:
            o.seed = UInt64(Int(a[i + 1]))
            i += 1
        elif s == "--max" and has:
            o.max_tokens = Int(a[i + 1])
            i += 1
        elif s == "--system" and has:
            o.system = a[i + 1]
            i += 1
        else:
            raise Error("unknown option " + s)
        i += 1
    return o^


def read_line(prompt: String) -> Optional[String]:
    """One line from stdin with read(2) (None at end of input)."""
    print(prompt, end="", flush=True)
    var bytes = List[UInt8]()
    var c = List[UInt8](length=1, fill=0)
    while True:
        var n = external_call["read", Int](Int32(0), c.unsafe_ptr(), Int(1))
        if n <= 0:
            if len(bytes) == 0:
                return None
            break
        if c[0] == 10:
            break
        bytes.append(c[0])
    return String(unsafe_from_utf8=bytes)


def _complete_utf8(bytes: List[UInt8]) -> Int:
    """Length of the longest prefix that does not end inside a UTF-8 sequence."""
    var n = len(bytes)
    var i = n - 1
    var back = 0
    while i >= 0 and back < 4:
        var b = Int(bytes[i])
        if b < 0x80:
            return n
        if b >= 0xC0:
            var need = 2 if b < 0xE0 else (3 if b < 0xF0 else 4)
            return n if n - i >= need else i
        i -= 1
        back += 1
    return n


def chat(dir: String, o: Opts) raises:
    var t0 = perf_counter_ns()
    var model = LlmModel.load(dir, o.fmt, o.gpu, 4096)
    var tok = BpeTokenizer.load(dir + "/tokenizer.json")
    var tmpl = ChatTemplate.for_model(model.cfg, tok)
    print("[npc_llm] ", model.cfg.family, " ", model.format_name(), " on ", model.device(), ", loaded in ",
          Float64(perf_counter_ns() - t0) / 1e9, " s. Empty line or Ctrl-D quits.", sep="")
    var temp = o.temp if o.temp >= 0 else Float32(0.7)
    var sampler = Sampler(temp, o.top_p, 40, o.seed, 1.05)
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage("system", o.system))
    while True:
        var got = read_line("\nyou> ")
        if not got:
            break
        var line = got.value()
        if line.byte_length() == 0:
            break
        msgs.append(ChatMessage("user", line))
        var ids = tmpl.render(tok, msgs)
        var reuse = 0
        while reuse < len(model.past) and reuse < len(ids) and model.past[reuse] == ids[reuse]:
            reuse += 1
        var t1 = perf_counter_ns()
        var logits = model.prompt(ids)
        var t2 = perf_counter_ns()
        print("npc> ", end="")
        var out = List[Int]()
        var pending = List[UInt8]()
        for _ in range(o.max_tokens):
            var t = sampler.sample(logits^, out)
            var stop = False
            for s in tmpl.stops:
                if t == s:
                    stop = True
            if stop:
                break
            out.append(t)
            for b in tok.piece(t):
                pending.append(b)
            var cut = _complete_utf8(pending)
            if cut > 0:
                var head = List[UInt8]()
                for i in range(cut):
                    head.append(pending[i])
                print(String(unsafe_from_utf8=head), end="", flush=True)
                var rest = List[UInt8]()
                for i in range(cut, len(pending)):
                    rest.append(pending[i])
                pending = rest^
            logits = model.push(t)
        var t3 = perf_counter_ns()
        print("\n  [", len(ids) - reuse, " new prompt tokens (", reuse, " cached) in ",
              Float64(t2 - t1) / 1e6, " ms; ", len(out), " tokens at ",
              Float64(max(len(out) - 1, 0)) / max(Float64(t3 - t2) / 1e9, 1e-9), " tok/s]", sep="")
        msgs.append(ChatMessage("assistant", tok.decode(out)))


@fieldwise_init
struct Scene(Copyable, Movable):
    var health: Float32
    var enemy: String
    var enemy_dist: Float32
    var player_dist: Float32
    var player_said: String


def _observation(bb: Blackboard, k_health: Int, k_enemy_dist: Int, k_player_dist: Int, s: Scene) -> String:
    var o = String("Your health: ") + String(Int(bb.get_float(k_health))) + "/100. "
    if s.enemy.byte_length() > 0:
        o += "A hostile " + s.enemy + " is " + String(Int(bb.get_float(k_enemy_dist))) + " m away. "
    else:
        o += "No enemies in sight. "
    o += "The player is " + String(Int(bb.get_float(k_player_dist))) + " m away"
    if s.player_said.byte_length() > 0:
        o += " and says: \"" + s.player_said + "\""
    return o + "."


def npc(dir: String, o: Opts) raises:
    var t0 = perf_counter_ns()
    var model = LlmModel.load(dir, o.fmt, o.gpu, 2048)
    var tok = BpeTokenizer.load(dir + "/tokenizer.json")
    var tmpl = ChatTemplate.for_model(model.cfg, tok)
    print("[npc_llm] ", model.cfg.family, " ", model.format_name(), " on ", model.device(), ", loaded in ",
          Float64(perf_counter_ns() - t0) / 1e9, " s", sep="")
    var names = ["attack", "flee", "greet", "trade", "warn", "patrol", "heal"]
    var descs = ["Fight the target with your hammer.", "Run to the forge and bar the door.",
                 "Greet the target warmly.", "Offer the target your wares.",
                 "Shout a warning to the target about danger.", "Walk your usual route around the village square.",
                 "Drink a healing draught."]
    var acts = List[NpcAction]()
    for i in range(len(names)):
        acts.append(NpcAction(100 + i, String(names[i]), String(descs[i])))
    var targets = List[String]()
    for t in ["player", "goblin", "wolf", "none"]:
        targets.append(String(t))
    var temp = o.temp if o.temp >= 0 else Float32(0.0)
    var brain = NpcBrain("You are Brakka, a gruff but kind dwarven blacksmith and the village's last defender.",
                         acts.copy(), targets^, 120, 0, Sampler(temp, 0.9, 40, o.seed))
    var bb = Blackboard()
    var k_choice = bb.def_key("llm_action", -1.0)
    var k_health = bb.def_key("health", 100.0)
    var k_enemy_dist = bb.def_key("enemy_dist", 999.0)
    var k_player_dist = bb.def_key("player_dist", 5.0)
    # root: designer safety branch first, then whatever the LLM chose
    var FLEE = 101
    var bt = BehaviorTree()
    _ = bt.begin_selector()
    _ = bt.begin_sequence()
    _ = bt.add_blackboard_cmp(k_health, BT_CMP_LT, 15.0)
    _ = bt.add_action(FLEE)
    bt.end_composite()
    _ = bt.begin_selector()
    for i in range(len(acts)):
        _ = bt.begin_sequence()
        _ = bt.add_blackboard_cmp(k_choice, BT_CMP_EQ, Float32(i))
        _ = bt.add_action(acts[i].id)
        bt.end_composite()
    bt.end_composite()
    bt.end_composite()
    var scenes = List[Scene]()
    scenes.append(Scene(100, "", 999, 4, "Good morning, Brakka! Lovely weather today."))
    scenes.append(Scene(100, "", 999, 2, "I have 40 gold. Do you have a sword for sale?"))
    scenes.append(Scene(90, "goblin", 30, 6, ""))
    scenes.append(Scene(85, "goblin", 3, 20, "Watch out behind you!"))
    scenes.append(Scene(35, "wolf", 2, 10, ""))
    scenes.append(Scene(10, "wolf", 1, 10, "Brakka, you're bleeding!"))
    scenes.append(Scene(60, "", 999, 40, ""))
    var total_s: Float64 = 0
    for si in range(len(scenes)):
        ref s = scenes[si]
        bb.set_float(k_health, s.health)
        bb.set_float(k_enemy_dist, s.enemy_dist)
        bb.set_float(k_player_dist, s.player_dist)
        var obs = _observation(bb, k_health, k_enemy_dist, k_player_dist, s)
        var d = brain.decide(model, tok, tmpl, obs)
        brain.write_blackboard(d, bb, k_choice)
        var st = BtExecutionState(128)
        _ = bt.tick(bb, st, 0.5)
        var ran = st.active_action_id
        var ran_name = String("?")
        for a in acts:
            if a.id == ran:
                ran_name = a.name
        total_s += d.seconds
        print("\n[scene ", si + 1, "] ", obs, sep="")
        print("  llm  -> ", d.json, sep="")
        print("  tree -> runs '", ran_name, "' (action id ", ran, ")",
              "  [designer override]" if ran_name != d.action else "", sep="")
        print("  ", d.prompt_tokens - d.reused_tokens, " new prompt tokens (", d.reused_tokens, " cached), ", d.tokens,
              " generated, ", Int(d.seconds * 1000), " ms", sep="")
    print("\n[npc_llm] ", len(scenes), " decisions, mean ", Int(total_s / Float64(len(scenes)) * 1000), " ms each", sep="")


def main() raises:
    var raw = argv()
    var a = List[String]()
    for x in raw:
        a.append(String(x))
    if len(a) < 3 or (a[1] != "chat" and a[1] != "npc"):
        print("usage: npc_llm chat|npc MODEL_DIR [--cpu] [--fmt fp32|int8|int4] [--temp T] [--top-p P] [--seed N] [--max N] [--system TEXT]")
        return
    var o = parse(a)
    if a[1] == "chat":
        chat(a[2], o)
    else:
        npc(a[2], o)
