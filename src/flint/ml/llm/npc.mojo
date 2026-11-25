"""An LLM that drives a BehaviorTree (`flint.ml.llm.npc`).

`NpcBrain` turns what an NPC perceives (a short text observation) into one of a fixed set of
BehaviorTree actions. The model sees a persona, the action list with descriptions and the
allowed targets, and must answer with exactly

    {"action": "<action name>", "target": "<target>", "say": "<one spoken line>"}

Decoding is grammar-constrained (`flint.ml.llm.constrain`), so the reply is always valid JSON
with an action and a target from the lists; there is nothing to parse defensively. The
system prompt is identical every turn, so its KV cache is reused and a decision costs only the
observation plus the reply.

`decide` returns an `NpcDecision`; `write_blackboard` stores the chosen action's index under a
blackboard key, and `build_choice_tree` makes the matching tree: a selector whose branch i is
`Sequence(BlackboardCmp(key == i), Action(actions[i].id))`. Designers can wrap those branches
with their own guards and cooldowns; the LLM only chooses, the tree still executes.
"""

from std.time import perf_counter_ns
from flint.ai.behavior_tree import Blackboard, BehaviorTree, BT_CMP_EQ
from flint.assets.json import parse_json
from flint.ml.llm.tokenizer import BpeTokenizer
from flint.ml.llm.model import LlmModel
from flint.ml.llm.sample import Sampler
from flint.ml.llm.chat import ChatMessage, ChatTemplate, generate_constrained
from flint.ml.llm.constrain import JsonGrammar, GrammarMatcher


@fieldwise_init
struct NpcAction(Copyable, Movable):
    var id: Int              # the BehaviorTree action id
    var name: String         # what the model writes, e.g. "attack"
    var description: String  # one line for the prompt


struct NpcDecision(Copyable, Movable):
    var index: Int           # into NpcBrain.actions (-1 if generation ran out of tokens)
    var action_id: Int
    var action: String
    var target: String
    var say: String
    var json: String
    var tokens: Int
    var prompt_tokens: Int
    var reused_tokens: Int
    var seconds: Float64

    def __init__(out self):
        self.index = -1
        self.action_id = -1
        self.action = ""
        self.target = ""
        self.say = ""
        self.json = ""
        self.tokens = 0
        self.prompt_tokens = 0
        self.reused_tokens = 0
        self.seconds = 0


struct NpcBrain(Movable):
    var persona: String
    var actions: List[NpcAction]
    var targets: List[String]
    var max_say: Int
    var max_tokens: Int
    var memory_turns: Int
    var sampler: Sampler
    var history: List[ChatMessage]
    var grammar: JsonGrammar

    def __init__(out self, persona: String, var actions: List[NpcAction], var targets: List[String],
                 max_say: Int = 160, memory_turns: Int = 0, var sampler: Sampler = Sampler.greedy()) raises:
        self.persona = persona
        self.actions = actions^
        self.targets = targets^
        self.max_say = max_say
        self.max_tokens = 96
        self.memory_turns = memory_turns
        self.sampler = sampler^
        self.history = List[ChatMessage]()
        var names = List[String]()
        names.append("action")
        names.append("target")
        names.append("say")
        var enums = List[List[String]]()
        var an = List[String]()
        for a in self.actions:
            an.append(a.name)
        enums.append(an^)
        enums.append(self.targets.copy())
        enums.append(List[String]())
        var lens = List[Int]()
        lens.append(0)
        lens.append(0)
        lens.append(max_say)
        self.grammar = JsonGrammar.object_of_fields(names, enums, lens)

    def system_prompt(self) -> String:
        var s = self.persona + "\n\nYou control this character in a game. Each message describes what the character" \
            + " perceives right now. Decide what the character does next.\nActions:\n"
        for a in self.actions:
            s += "- " + a.name + ": " + a.description + "\n"
        s += "Targets: "
        for i in range(len(self.targets)):
            s += ("" if i == 0 else ", ") + self.targets[i]
        s += "\nReply with one JSON object only: {\"action\": \"<action>\", \"target\": \"<target>\", \"say\": " \
            + "\"<one short line the character says out loud>\"}"
        return s

    def prompt_messages(self, observation: String) -> List[ChatMessage]:
        var msgs = List[ChatMessage]()
        msgs.append(ChatMessage("system", self.system_prompt()))
        for m in self.history:
            msgs.append(m.copy())
        msgs.append(ChatMessage("user", observation))
        return msgs^

    def decide(mut self, mut model: LlmModel, mut tok: BpeTokenizer, tmpl: ChatTemplate, observation: String) raises -> NpcDecision:
        var t0 = perf_counter_ns()
        var ids = tmpl.render(tok, self.prompt_messages(observation), True)
        var matcher = GrammarMatcher(self.grammar.copy())
        var g = generate_constrained(model, tok, ids, self.sampler, self.max_tokens, tmpl.stops, matcher)
        var d = NpcDecision()
        d.json = g.text
        d.tokens = len(g.tokens)
        d.prompt_tokens = g.prompt_tokens
        d.reused_tokens = g.reused_tokens
        if matcher.complete():
            var doc = parse_json(List[UInt8](g.text.as_bytes()))
            d.action = doc.text(doc.object_get(doc.root, "action"))
            d.target = doc.text(doc.object_get(doc.root, "target"))
            d.say = doc.text(doc.object_get(doc.root, "say"))
            for i in range(len(self.actions)):
                if self.actions[i].name == d.action:
                    d.index = i
                    d.action_id = self.actions[i].id
        if self.memory_turns > 0:
            self.history.append(ChatMessage("user", observation))
            self.history.append(ChatMessage("assistant", g.text))
            while len(self.history) > 2 * self.memory_turns:
                _ = self.history.pop(0)
        d.seconds = Float64(perf_counter_ns() - t0) / 1e9
        return d^

    def write_blackboard(self, d: NpcDecision, mut bb: Blackboard, key: Int):
        bb.set_float(key, Float32(d.index))

    def build_choice_tree(self, bb_key: Int) -> BehaviorTree:
        """Selector over `Sequence(BlackboardCmp(bb_key == i), Action(actions[i].id))`."""
        var bt = BehaviorTree()
        _ = bt.begin_selector()
        for i in range(len(self.actions)):
            _ = bt.begin_sequence()
            _ = bt.add_blackboard_cmp(bb_key, BT_CMP_EQ, Float32(i))
            _ = bt.add_action(self.actions[i].id)
            bt.end_composite()
        bt.end_composite()
        return bt^
