"""A small Tidal-flavoured Haskell subset interpreter (`flint.cycles.script`).

Understands the code people actually type into a Tidal buffer:

    setcps (130/60/4)
    d1 $ every 3 (fast 2) $ s "bd*2 [~ sn] bd sn:2" # gain 1.1
    d2 $ jux rev $ n (scale "minPent" "0 .. 7") # s "pluck" # room 0.3
    d3 $ sometimes (# speed 2) $ s "hh*8" # pan sine
    hush

Supported: function application and currying, `$`, `.` composition, operator
sections `(# speed 2)` / `(0.25 <~)`, lists `[a, b]`, numeric arithmetic,
`#` `|+|` `|+` `+|` `+` `-` `*` `/` `|<` `|-` `|*` `|/` `|%` `<~` `~>`, `--`
comments, and multi-line statements (continuation lines are indented).
A statement that fails to parse is reported and skipped; the rest still run.
"""
from flint.cycles.pattern import (
    Frac, Pattern, Params, parse_mini, parse_number, param_id, scale_index, bjorklund, MAX_PATTERN_COUNT,
    P_PAN, P_SPEED, P_SOUND, P_BEGIN, P_END, K_ATOM, K_ALT_ABS, K_STACK,
    ST_LEFT, ST_RIGHT, ST_BOTH, arp_mode, Params,
    CH_EVERY, CH_WHENMOD, CH_SOMECYCLES,
    OP_SET, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_KEEP, OP_MOD,
    SIG_SINE, SIG_COSINE, SIG_SAW, SIG_ISAW, SIG_TRI, SIG_SQUARE, SIG_RAND, SIG_PERLIN,
)

# ---------------------------------------------------------------- tokens

comptime T_NUM: Int = 0
comptime T_STR: Int = 1
comptime T_ID: Int = 2
comptime T_OP: Int = 3
comptime T_LP: Int = 4
comptime T_RP: Int = 5
comptime T_LB: Int = 6
comptime T_RB: Int = 7
comptime T_COMMA: Int = 8
comptime T_END: Int = 9
comptime T_LAMBDA: Int = 10


@fieldwise_init
struct Token(Copyable, Movable):
    var kind: Int
    var text: String
    var num: Float64


def _is_op_char(c: Int) -> Bool:
    return (
        c == ord("$") or c == ord("#") or c == ord("|") or c == ord("+") or c == ord("-") or c == ord("*")
        or c == ord("/") or c == ord("<") or c == ord(">") or c == ord("~") or c == ord(".") or c == ord("%")
        or c == ord("=")
    )


def _is_id_char(c: Int) -> Bool:
    return (
        (c >= ord("a") and c <= ord("z")) or (c >= ord("A") and c <= ord("Z")) or (c >= ord("0") and c <= ord("9"))
        or c == ord("_") or c == ord("'")
    )


def tokenize(text: String) raises -> List[Token]:
    var b = text.as_bytes()
    var n = len(b)
    var out = List[Token]()
    var i = 0
    while i < n:
        var c = Int(b[i])
        if c == ord(" ") or c == ord("\t") or c == ord("\n") or c == ord("\r"):
            i += 1
            continue
        if c == ord("\""):
            var start = i + 1
            i += 1
            while i < n and Int(b[i]) != ord("\""):
                i += 1
            if i >= n:
                raise Error("unterminated string")
            out.append(Token(T_STR, String(text[byte=start:i]), 0.0))
            i += 1
            continue
        if c >= ord("0") and c <= ord("9"):
            var start = i
            while i < n and Int(b[i]) >= ord("0") and Int(b[i]) <= ord("9"):
                i += 1
            if i + 1 < n and Int(b[i]) == ord(".") and Int(b[i + 1]) >= ord("0") and Int(b[i + 1]) <= ord("9"):
                i += 1
                while i < n and Int(b[i]) >= ord("0") and Int(b[i]) <= ord("9"):
                    i += 1
            if i < n and (Int(b[i]) == ord("e")) and i + 1 < n and (Int(b[i + 1]) == ord("-") or (Int(b[i + 1]) >= ord("0") and Int(b[i + 1]) <= ord("9"))):
                i += 2
                while i < n and Int(b[i]) >= ord("0") and Int(b[i]) <= ord("9"):
                    i += 1
            var word = String(text[byte=start:i])
            out.append(Token(T_NUM, word, parse_number(word)[0]))
            continue
        if _is_id_char(c):
            var start = i
            while i < n and _is_id_char(Int(b[i])):
                i += 1
            out.append(Token(T_ID, String(text[byte=start:i]), 0.0))
            continue
        if c == ord("("):
            out.append(Token(T_LP, "(", 0.0))
            i += 1
            continue
        if c == ord(")"):
            out.append(Token(T_RP, ")", 0.0))
            i += 1
            continue
        if c == ord("["):
            out.append(Token(T_LB, "[", 0.0))
            i += 1
            continue
        if c == ord("]"):
            out.append(Token(T_RB, "]", 0.0))
            i += 1
            continue
        if c == ord(","):
            out.append(Token(T_COMMA, ",", 0.0))
            i += 1
            continue
        if c == ord("\\"):
            out.append(Token(T_LAMBDA, "\\", 0.0))
            i += 1
            continue
        if _is_op_char(c):
            var start = i
            while i < n and _is_op_char(Int(b[i])):
                i += 1
            out.append(Token(T_OP, String(text[byte=start:i]), 0.0))
            continue
        raise Error("unexpected character '" + chr(c) + "'")
    out.append(Token(T_END, "", 0.0))
    return out^


# ---------------------------------------------------------------- syntax tree

comptime A_NUM: Int = 0
comptime A_STR: Int = 1
comptime A_VAR: Int = 2
comptime A_APP: Int = 3
comptime A_OP: Int = 4
comptime A_LIST: Int = 5
comptime A_SECR: Int = 6   # (op rhs)
comptime A_SECL: Int = 7   # (lhs op)
comptime A_LAMBDA: Int = 8 # \\params -> body: items = param A_VARs, a = body


struct Ast(Copyable, Movable):
    var kind: Int
    var text: String
    var num: Float64
    var a: Int
    var b: Int
    var items: List[Int]

    def __init__(out self, kind: Int, text: String = "", num: Float64 = 0.0, a: Int = -1, b: Int = -1):
        self.kind = kind
        self.text = text
        self.num = num
        self.a = a
        self.b = b
        self.items = List[Int]()


def _precedence(op: String) -> Int:
    """Haskell fixities; -1 means not a binary operator here."""
    if op == "$":
        return 0
    if op == ".":
        return 9
    if op == "*" or op == "/":
        return 7
    if op == "+" or op == "-":
        return 6
    if op == "<~" or op == "~>":
        return 5
    if op == "#" or op.startswith("|") or op.endswith("|"):
        return 1
    return -1


def _right_assoc(op: String) -> Bool:
    return op == "$" or op == "."


comptime MAX_SCRIPT_PARSE_DEPTH: Int = 128
comptime MAX_SCRIPT_EVAL_DEPTH: Int = 512


struct Parser:
    var toks: List[Token]
    var pos: Int
    var nodes: List[Ast]
    var depth: Int

    def __init__(out self, var toks: List[Token]):
        self.toks = toks^
        self.pos = 0
        self.nodes = List[Ast]()
        self.depth = 0

    def add(mut self, var a: Ast) -> Int:
        self.nodes.append(a^)
        return len(self.nodes) - 1

    def kind(self) -> Int:
        return self.toks[self.pos].kind

    def text(self) -> String:
        return self.toks[self.pos].text

    def parse_expr(mut self, min_prec: Int) raises -> Int:
        if self.depth >= MAX_SCRIPT_PARSE_DEPTH:
            raise Error("expression nesting too deep (max " + String(MAX_SCRIPT_PARSE_DEPTH) + ")")
        self.depth += 1
        var res = self._parse_expr_inner(min_prec)
        self.depth -= 1
        return res

    def _parse_expr_inner(mut self, min_prec: Int) raises -> Int:
        var left = self.parse_app()
        while self.kind() == T_OP:
            var op = self.text()
            if op == "->" or op == "=":
                break
            var prec = _precedence(op)
            if prec < 0:
                raise Error("unknown operator '" + op + "'")
            if prec < min_prec:
                break
            # `(lhs op)` left section: stop and let the paren parser see it.
            if self.toks[self.pos + 1].kind == T_RP:
                break
            self.pos += 1
            var right = self.parse_expr(prec if _right_assoc(op) else prec + 1)
            if op == "$":
                left = self.add(Ast(A_APP, "", 0.0, left, right))
            else:
                left = self.add(Ast(A_OP, op, 0.0, left, right))
        return left

    def starts_atom(self) -> Bool:
        var k = self.kind()
        return k == T_NUM or k == T_STR or k == T_ID or k == T_LP or k == T_LB or k == T_LAMBDA

    def parse_app(mut self) raises -> Int:
        if not self.starts_atom():
            if self.kind() == T_END:
                raise Error("statement ends where an expression was expected")
            raise Error("expected an expression, found '" + self.text() + "'")
        var f = self.parse_atom()
        while self.starts_atom():
            var x = self.parse_atom()
            f = self.add(Ast(A_APP, "", 0.0, f, x))
        return f

    def parse_atom(mut self) raises -> Int:
        var k = self.kind()
        if k == T_LAMBDA:
            self.pos += 1
            var lam = Ast(A_LAMBDA)
            while self.kind() == T_ID:
                lam.items.append(self.add(Ast(A_VAR, self.text())))
                self.pos += 1
            if len(lam.items) == 0 or not (self.kind() == T_OP and self.text() == "->"):
                raise Error("lambda needs `\\x -> ...`")
            self.pos += 1
            lam.a = self.parse_expr(0)
            return self.add(lam^)
        if k == T_NUM:
            var v = self.toks[self.pos].num
            self.pos += 1
            return self.add(Ast(A_NUM, "", v))
        if k == T_STR:
            var s = self.text()
            self.pos += 1
            return self.add(Ast(A_STR, s))
        if k == T_ID:
            var s = self.text()
            self.pos += 1
            return self.add(Ast(A_VAR, s))
        if k == T_LB:
            self.pos += 1
            var lst = Ast(A_LIST)
            if self.kind() != T_RB:
                while True:
                    lst.items.append(self.parse_expr(0))
                    if self.kind() == T_COMMA:
                        self.pos += 1
                        continue
                    break
            if self.kind() != T_RB:
                raise Error("missing ']'")
            self.pos += 1
            return self.add(lst^)
        if k == T_LP:
            self.pos += 1
            if self.kind() == T_OP:
                var op = self.text()
                # `(-0.25)` is a negative number, `(- 1)` too.
                if op == "-" and self.toks[self.pos + 1].kind == T_NUM and self.toks[self.pos + 2].kind == T_RP:
                    var v = -self.toks[self.pos + 1].num
                    self.pos += 3
                    return self.add(Ast(A_NUM, "", v))
                self.pos += 1
                if self.kind() == T_RP:
                    raise Error("bare operator section (" + op + ") is not supported")
                var rhs = self.parse_expr(0)
                if self.kind() != T_RP:
                    raise Error("missing ')'")
                self.pos += 1
                return self.add(Ast(A_SECR, op, 0.0, rhs))
            var inner = self.parse_expr(0)
            if self.kind() == T_COMMA:
                # a tuple like (0.25, 0.75): evaluated as a list
                var tup = Ast(A_LIST)
                tup.items.append(inner)
                while self.kind() == T_COMMA:
                    self.pos += 1
                    tup.items.append(self.parse_expr(0))
                if self.kind() != T_RP:
                    raise Error("missing ')' after tuple")
                self.pos += 1
                return self.add(tup^)
            if self.kind() == T_OP and self.toks[self.pos + 1].kind == T_RP:
                var op = self.text()
                self.pos += 2
                return self.add(Ast(A_SECL, op, 0.0, inner))
            if self.kind() != T_RP:
                raise Error("missing ')' (found '" + self.text() + "')")
            self.pos += 1
            return inner
        raise Error("unexpected '" + self.text() + "'")


# ---------------------------------------------------------------- values

comptime V_NONE: Int = 0
comptime V_NUM: Int = 1
comptime V_STR: Int = 2
comptime V_PAT: Int = 3
comptime V_FUN: Int = 4
comptime V_LIST: Int = 5
comptime V_USER: Int = 6  # let-bound function / lambda: text=name, pat=body ast, items=param asts, args=bound


struct Val(Copyable, Movable):
    var kind: Int
    var num: Float64
    var text: String   # string literal, or function name
    var pat: Int
    var items: List[Int]  # function args so far, or list items
    var args: List[Int]   # V_USER: arguments applied so far

    def __init__(out self, kind: Int):
        self.kind = kind
        self.num = 0.0
        self.text = String()
        self.pat = -1
        self.items = List[Int]()
        self.args = List[Int]()


@fieldwise_init
struct Binding(Copyable, Movable):
    var name: String
    var val: Int


@fieldwise_init
struct Track(Copyable, Movable):
    var name: String
    var root: Int


def builtin_arity(name: String) -> Int:
    """Arguments a builtin takes; -1 if `name` is not a builtin."""
    if param_id(name) >= 0:
        return 1
    if name.startswith("d") and name.byte_length() <= 3:
        var num = parse_number(String(name[byte=1:name.byte_length()]))
        if num[1] and num[0] >= 1 and num[0] <= 16:
            return 1
    var one: List[String] = ["rev", "palindrome", "degrade", "stack", "cat", "slowcat", "fastcat", "fastCat",
        "randcat", "setcps", "setbpm", "irand", "run", "brak", "id"]
    var two: List[String] = ["fast", "density", "slow", "sparsity", "hurry", "degradeBy", "unDegradeBy", "jux",
        "superimpose", "overlay", "iter", "iter'", "ply", "scale", "segment", "sometimes", "often", "rarely",
        "almostNever", "almostAlways", "always", "never", "somecycles", "struct", "rotL", "rotR", "fastGap",
        "chop", "striate", "arp", "linger", "swing", "randslice", "const"]
    var three: List[String] = ["every", "off", "euclid", "euclidInv", "range", "sometimesBy", "someCyclesBy",
        "inside", "outside", "compose", "secR", "secL", "every'", "chunk", "chunk'", "within", "within'",
        "fix", "unfix", "swingBy", "striateBy", "splice", "slice"]
    var four: List[String] = ["whenmod", "euclidOff", "euclidOffFull"]
    for s in one:
        if s == name:
            return 1
    for s in two:
        if s == name:
            return 2
    for s in three:
        if s == name:
            return 3
    for s in four:
        if s == name:
            return 4
    var zero: List[String] = ["silence", "hush", "sine", "cosine", "saw", "isaw", "tri", "square", "rand", "perlin",
        "envL", "envR"]
    for s in zero:
        if s == name:
            return 0
    return -1


struct Program(Movable):
    """The evaluated state of a whole Tidal file: tracks + tempo."""

    var pat: Pattern
    var vals: List[Val]
    var tracks: List[Track]
    var cps: Float64
    var cps_set: Bool
    var hushed: Bool
    var errors: List[String]
    var seed: Int
    var asts: List[Ast]
    var env: List[Binding]
    var eval_depth: Int

    def __init__(out self):
        self.pat = Pattern()
        self.vals = List[Val]()
        self.tracks = List[Track]()
        self.cps = 0.5625  # Tidal's default
        self.cps_set = False
        self.hushed = False
        self.errors = List[String]()
        self.seed = 1000
        self.asts = List[Ast]()
        self.env = List[Binding]()
        self.eval_depth = 0

    # ---- value helpers

    def push(mut self, var v: Val) -> Int:
        self.vals.append(v^)
        return len(self.vals) - 1

    def num_val(mut self, x: Float64) -> Int:
        var v = Val(V_NUM)
        v.num = x
        return self.push(v^)

    def pat_val(mut self, id: Int) -> Int:
        var v = Val(V_PAT)
        v.pat = id
        return self.push(v^)

    def fun_val(mut self, name: String, var args: List[Int]) -> Int:
        var v = Val(V_FUN)
        v.text = name
        v.items = args^
        return self.push(v^)

    def to_pattern(mut self, id: Int) raises -> Int:
        var k = self.vals[id].kind
        if k == V_PAT:
            return self.vals[id].pat
        if k == V_NUM:
            return self.pat.pure_number(self.vals[id].num)
        if k == V_STR:
            var text = self.vals[id].text
            return parse_mini(text, self.pat)
        if k == V_LIST:
            raise Error("expected a pattern, got a list (did you mean `stack [...]`?)")
        if k == V_FUN:
            raise Error("expected a pattern, got the function '" + self.vals[id].text + "' (missing an argument?)")
        raise Error("expected a pattern")

    def to_numbers(mut self, id: Int) raises -> List[Float64]:
        """A number, or a per-cycle sequence for a patterned arg like "<1 2>"."""
        var k = self.vals[id].kind
        if k == V_NUM:
            return [self.vals[id].num]
        if k != V_STR and k != V_PAT:
            raise Error("expected a number")
        var p = self.to_pattern(id)
        var per = List[Float64]()
        for c in range(24):
            var evs = self.pat.query_node(p, Frac.of(c), Frac.of(c + 1))
            var v = 0.0
            var found = False
            for e in range(len(evs)):
                if evs[e].has_onset() and evs[e].value.has_num:
                    v = evs[e].value.num
                    found = True
                    break
            if not found and len(per) > 0:
                v = per[len(per) - 1]
            per.append(v)
        for period in range(1, 25):
            var ok = True
            for i in range(24):
                if per[i] != per[i % period]:
                    ok = False
                    break
            if ok:
                var out = List[Float64]()
                for i in range(period):
                    out.append(per[i])
                return out^
        return per^

    def to_int(mut self, id: Int) raises -> Int:
        var v = self.to_numbers(id)[0]
        if v != v or v < -1.0e9 or v > 1.0e9:
            raise Error("integer argument out of range")
        return Int(v)

    def per_cycle(mut self, nums: List[Float64], ids: List[Int]) -> Int:
        if len(ids) == 1:
            return ids[0]
        return self.pat.with_kids(K_ALT_ABS, ids, List[Frac]())

    def lookup(self, name: String) -> Int:
        for i in range(len(self.env) - 1, -1, -1):
            if self.env[i].name == name:
                return self.env[i].val
        return -1

    def apply(mut self, f: Int, x: Int) raises -> Int:
        if self.vals[f].kind == V_USER:
            var params = self.vals[f].items.copy()
            var bound = self.vals[f].args.copy()
            bound.append(x)
            if len(bound) < len(params):
                var v = Val(V_USER)
                v.text = self.vals[f].text
                v.pat = self.vals[f].pat
                v.items = params^
                v.args = bound^
                return self.push(v^)
            var body = self.vals[f].pat
            var depth = len(self.env)
            if depth > 4000:
                raise Error("recursion too deep in '" + self.vals[f].text + "'")
            for i in range(len(params)):
                self.env.append(Binding(self.asts[params[i]].text, bound[i]))
            var result = self.eval(body)
            while len(self.env) > depth:
                _ = self.env.pop()
            return result
        if self.vals[f].kind != V_FUN:
            raise Error("tried to apply something that is not a function")
        var name = self.vals[f].text
        var args = self.vals[f].items.copy()
        args.append(x)
        if len(args) >= builtin_arity(name):
            return self.call(name, args)
        return self.fun_val(name, args^)

    def apply_fn(mut self, f: Int, pat_id: Int) raises -> Int:
        """Apply a function value to a pattern node; returns a node id."""
        var r = self.apply(f, self.pat_val(pat_id))
        return self.to_pattern(r)

    def control(mut self, pid: Int, x: Float64) -> Int:
        return self.pat.control(pid, self.pat.pure_number(x))

    def numeric_op(self, op: String) -> Int:
        if op == "#" or op == "|>" or op == "|>|" or op == ">|":
            return OP_SET
        if op == "<|":
            return OP_KEEP
        if op == "+" or op == "|+|" or op == "|+" or op == "+|":
            return OP_ADD
        if op == "-" or op == "|-|" or op == "|-" or op == "-|":
            return OP_SUB
        if op == "*" or op == "|*|" or op == "|*" or op == "*|":
            return OP_MUL
        if op == "/" or op == "|/|" or op == "|/" or op == "/|":
            return OP_DIV
        if op == "|<" or op == "|<|" or op == "<|":
            return OP_KEEP
        if op == "|%" or op == "|%|" or op == "%|":
            return OP_MOD
        return -1

    def binop(mut self, op: String, l: Int, r: Int) raises -> Int:
        if op == ".":
            return self.fun_val("compose", [l, r])
        var lk = self.vals[l].kind
        var rk = self.vals[r].kind
        if op == "<~" or op == "~>":
            var amounts = self.to_numbers(l)
            var p = self.to_pattern(r)
            var ids = List[Int]()
            for a in amounts:
                var fr = Frac.from_float(a)
                ids.append(self.pat.shift_left(fr if op == "<~" else -fr, p))
            return self.pat_val(self.per_cycle(amounts, ids))
        var code = self.numeric_op(op)
        if code < 0:
            raise Error("unknown operator '" + op + "'")
        if lk == V_NUM and rk == V_NUM:
            var a = self.vals[l].num
            var b = self.vals[r].num
            var v: Float64
            if code == OP_ADD:
                v = a + b
            elif code == OP_SUB:
                v = a - b
            elif code == OP_MUL:
                v = a * b
            elif code == OP_DIV:
                v = a / b if b != 0 else 0.0
            elif code == OP_KEEP:
                v = a
            else:
                v = b
            return self.num_val(v)
        var lp = self.to_pattern(l)
        var rp = self.to_pattern(r)
        return self.pat_val(self.pat.union(code, lp, rp, self.structure_of(op)))

    def structure_of(self, op: String) -> Int:
        """Tidal: `|op` keeps the left's structure, `op|` the right's, `|op|`
        and bare arithmetic both; `#` is `|>`."""
        if op == "#":
            return ST_LEFT
        var lbar = op.startswith("|")
        var rbar = op.endswith("|") and op.byte_length() > 1
        if lbar and rbar:
            return ST_BOTH
        if lbar:
            return ST_LEFT
        if rbar:
            return ST_RIGHT
        return ST_BOTH

    # ---- builtins

    def call(mut self, name: String, args: List[Int]) raises -> Int:
        var pid = param_id(name)
        if pid >= 0:
            return self.pat_val(self.pat.control(pid, self.to_pattern(args[0])))
        if name.startswith("d") and builtin_arity(name) == 1 and name.byte_length() <= 3:
            var root = self.to_pattern(args[0])
            for i in range(len(self.tracks)):
                if self.tracks[i].name == name:
                    self.tracks[i].root = root
                    return self.push(Val(V_NONE))
            self.tracks.append(Track(name, root))
            return self.push(Val(V_NONE))
        if name == "silence":
            return self.pat_val(self.pat.silence())
        if name == "hush":
            self.hushed = True
            self.tracks.clear()
            return self.push(Val(V_NONE))
        if name == "sine":
            return self.pat_val(self.pat.signal(SIG_SINE))
        if name == "cosine":
            return self.pat_val(self.pat.signal(SIG_COSINE))
        if name == "saw" or name == "envL":
            return self.pat_val(self.pat.signal(SIG_SAW))
        if name == "isaw" or name == "envR":
            return self.pat_val(self.pat.signal(SIG_ISAW))
        if name == "tri":
            return self.pat_val(self.pat.signal(SIG_TRI))
        if name == "square":
            return self.pat_val(self.pat.signal(SIG_SQUARE))
        if name == "rand":
            self.seed += 1
            return self.pat_val(self.pat.signal(SIG_RAND, self.seed))
        if name == "perlin":
            self.seed += 1
            return self.pat_val(self.pat.signal(SIG_PERLIN, self.seed))
        if name == "id":
            return args[0]
        if name == "const":
            return args[0]
        if name == "setcps":
            self.cps = self.to_numbers(args[0])[0]
            self.cps_set = True
            return self.push(Val(V_NONE))
        if name == "setbpm":
            self.cps = self.to_numbers(args[0])[0] / 240.0
            self.cps_set = True
            return self.push(Val(V_NONE))
        if name == "compose":
            var inner = self.apply(args[1], args[2])
            return self.apply(args[0], inner)
        if name == "secR":
            var op = self.vals[args[0]].text.copy()
            return self.binop(op, args[2], args[1])
        if name == "secL":
            var op = self.vals[args[1]].text.copy()
            return self.binop(op, args[0], args[2])
        if name == "fast" or name == "density" or name == "slow" or name == "sparsity" or name == "hurry":
            var amounts = self.to_numbers(args[0])
            var p = self.to_pattern(args[1])
            var ids = List[Int]()
            for a in amounts:
                if a != a or a > Float64(MAX_PATTERN_COUNT):
                    raise Error(name + " factor out of range (max 4096)")
                if (name == "slow" or name == "sparsity") and a > 0.0 and (1.0 / a) > Float64(MAX_PATTERN_COUNT):
                    raise Error(name + " factor too small (max 4096x)")
                var f = Frac.from_float(a)
                if name == "slow" or name == "sparsity":
                    ids.append(self.pat.slow(f, p))
                elif name == "hurry":
                    var sped = self.pat.fast(f, p)
                    ids.append(self.pat.union(OP_MUL, sped, self.control(P_SPEED, a)))
                else:
                    ids.append(self.pat.fast(f, p))
            return self.pat_val(self.per_cycle(amounts, ids))
        if name == "rev":
            return self.pat_val(self.pat.rev(self.to_pattern(args[0])))
        if name == "palindrome":
            var p = self.to_pattern(args[0])
            return self.pat_val(self.pat.choose(CH_EVERY, 2, 1, 0.0, p, self.pat.rev(p)))
        if name == "every" or name == "every'":
            var n = self.to_int(args[0])
            var offset = 0
            var fi = 1
            if name == "every'":
                offset = self.to_int(args[1])
                fi = 2
            var p = self.to_pattern(args[fi + 1])
            var t = self.apply_fn(args[fi], p)
            return self.pat_val(self.pat.choose(CH_EVERY, n, offset, 0.0, p, t))
        if name == "whenmod":
            var a = self.to_int(args[0])
            var b = self.to_int(args[1])
            var p = self.to_pattern(args[3])
            var t = self.apply_fn(args[2], p)
            return self.pat_val(self.pat.choose(CH_WHENMOD, a, b, 0.0, p, t))
        if name == "degradeBy" or name == "unDegradeBy":
            self.seed += 1
            var prob = self.to_numbers(args[0])[0]
            return self.pat_val(self.pat.degrade(prob, self.seed, self.to_pattern(args[1]), name == "unDegradeBy"))
        if name == "degrade":
            self.seed += 1
            return self.pat_val(self.pat.degrade(0.5, self.seed, self.to_pattern(args[0])))
        if (name == "sometimesBy" or name == "sometimes" or name == "often" or name == "rarely"
                or name == "almostNever" or name == "almostAlways"):
            var prob = 0.5
            var fi = 0
            if name == "sometimesBy":
                prob = self.to_numbers(args[0])[0]
                fi = 1
            elif name == "often":
                prob = 0.75
            elif name == "rarely":
                prob = 0.25
            elif name == "almostNever":
                prob = 0.1
            elif name == "almostAlways":
                prob = 0.9
            var p = self.to_pattern(args[fi + 1])
            self.seed += 1
            var keep = self.pat.degrade(prob, self.seed, p)
            var hit = self.pat.degrade(prob, self.seed, p, True)
            var t = self.apply_fn(args[fi], hit)
            return self.pat_val(self.pat.stack([keep, t]))
        if name == "always":
            return self.apply(args[0], args[1])
        if name == "never":
            return args[1]
        if name == "someCyclesBy" or name == "somecycles":
            var prob = 0.5
            var fi = 0
            if name == "someCyclesBy":
                prob = self.to_numbers(args[0])[0]
                fi = 1
            var p = self.to_pattern(args[fi + 1])
            return self.pat_val(self.pat.choose(CH_SOMECYCLES, 0, 0, prob, p, self.apply_fn(args[fi], p)))
        if name == "jux":
            var p = self.to_pattern(args[1])
            var t = self.apply_fn(args[0], p)
            var left = self.pat.union(OP_SET, p, self.control(P_PAN, 0.0))
            var right = self.pat.union(OP_SET, t, self.control(P_PAN, 1.0))
            return self.pat_val(self.pat.stack([left, right]))
        if name == "superimpose":
            var p = self.to_pattern(args[1])
            return self.pat_val(self.pat.stack([p, self.apply_fn(args[0], p)]))
        if name == "off":
            var amount = Frac.from_float(self.to_numbers(args[0])[0])
            var p = self.to_pattern(args[2])
            var shifted = self.pat.shift_left(-amount, p)
            return self.pat_val(self.pat.stack([p, self.apply_fn(args[1], shifted)]))
        if name == "overlay":
            return self.pat_val(self.pat.stack([self.to_pattern(args[0]), self.to_pattern(args[1])]))
        if name == "stack" or name == "cat" or name == "slowcat" or name == "fastcat" or name == "fastCat" or name == "randcat":
            if self.vals[args[0]].kind != V_LIST:
                raise Error(name + " expects a list like [a, b]")
            var items = self.vals[args[0]].items.copy()
            var ids = List[Int]()
            for it in items:
                ids.append(self.to_pattern(it))
            if len(ids) == 0:
                return self.pat_val(self.pat.silence())
            if name == "stack":
                return self.pat_val(self.pat.stack(ids))
            if name == "randcat":
                return self.pat_val(self.pat.randcat(ids))
            var cat = self.pat.slowcat(ids)
            if name == "fastcat" or name == "fastCat":
                return self.pat_val(self.pat.fast(Frac.of(len(ids)), cat))
            return self.pat_val(cat)
        if name == "iter" or name == "iter'":
            var count = self.to_int(args[0])
            if count > MAX_PATTERN_COUNT:
                raise Error(name + " count too large (max 4096)")
            return self.pat_val(self.pat.iter(count, self.to_pattern(args[1]), name == "iter'"))
        if name == "ply":
            var count = self.to_int(args[0])
            if count > MAX_PATTERN_COUNT:
                raise Error("ply count too large (max 4096)")
            return self.pat_val(self.pat.ply(count, self.to_pattern(args[1])))
        if name == "segment":
            var count = self.to_int(args[0])
            if count > MAX_PATTERN_COUNT:
                raise Error("segment count too large (max 4096)")
            return self.pat_val(self.pat.segment(count, self.to_pattern(args[1])))
        if name == "range":
            var lo = self.to_numbers(args[0])[0]
            var hi = self.to_numbers(args[1])[0]
            return self.pat_val(self.pat.range(lo, hi, self.to_pattern(args[2])))
        if name == "irand":
            return self.pat_val(self.pat.irand(self.to_int(args[0])))
        if name == "run":
            var count = self.to_int(args[0])
            if count > MAX_PATTERN_COUNT:
                raise Error("run count too large (max 4096)")
            var ids = List[Int]()
            for i in range(max(1, count)):
                ids.append(self.pat.pure_number(Float64(i)))
            return self.pat_val(self.pat.seq(ids))
        if name == "scale":
            if self.vals[args[0]].kind != V_STR:
                raise Error("scale expects a scale name string like \"major\"")
            var sname = self.vals[args[0]].text
            var idx = scale_index(sname)
            if idx < 0:
                raise Error("unknown scale '" + sname + "'")
            return self.pat_val(self.pat.scale(idx, self.to_pattern(args[1])))
        if name == "euclid" or name == "euclidInv" or name == "euclidOff" or name == "euclidOffFull":
            var k = self.to_int(args[0])
            var n = self.to_int(args[1])
            if n > MAX_PATTERN_COUNT:
                raise Error("euclid step count too large (max 4096)")
            var rot = 0
            var pi = 2
            if name == "euclidOff" or name == "euclidOffFull":
                rot = self.to_int(args[2])
                pi = 3
            var p = self.to_pattern(args[pi])
            if n <= 0:
                return self.pat_val(self.pat.silence())
            var bits = bjorklund(k, n)
            var slots = List[Int]()
            var rest = self.pat.silence()
            for i in range(n):
                var bit = bits[((i + rot) % n + n) % n]
                if name == "euclidInv":
                    bit = not bit
                slots.append(p if bit else rest)
            return self.pat_val(self.pat.seq(slots))
        if name == "struct":
            var shape = self.to_pattern(args[0])
            return self.pat_val(self.pat.union(OP_SET, shape, self.to_pattern(args[1])))
        if name == "rotL" or name == "rotR":
            var fr = Frac.from_float(self.to_numbers(args[0])[0])
            return self.pat_val(self.pat.shift_left(fr if name == "rotL" else -fr, self.to_pattern(args[1])))
        if name == "inside" or name == "outside":
            var n = self.to_numbers(args[0])[0]
            if name == "outside":
                n = 1.0 / n
            var fr = Frac.from_float(n)
            var p = self.to_pattern(args[2])
            var t = self.apply_fn(args[1], self.pat.slow(fr, p))
            return self.pat_val(self.pat.fast(fr, t))
        if name == "brak":
            # every other cycle, squash into the first half and shift by a quarter
            var p = self.to_pattern(args[0])
            var squashed = self.pat.seq([p, self.pat.silence()])
            var b = self.pat.shift_left(Frac(-1, 4), squashed)
            return self.pat_val(self.pat.choose(CH_EVERY, 2, 1, 0.0, p, b))
        if name == "fastGap":
            var f = self.to_numbers(args[0])[0]
            var p = self.to_pattern(args[1])
            var w = Frac.from_float(f)
            var ids: List[Int] = [self.pat.fast(Frac.of(1), p), self.pat.silence()]
            var ws: List[Frac] = [Frac.of(1), w - Frac.of(1)]
            return self.pat_val(self.pat.with_kids(2, ids, ws))
        if name == "chop":
            var counts = self.to_numbers(args[0])
            var p = self.to_pattern(args[1])
            var ids = List[Int]()
            for c in counts:
                if c != c or c > Float64(MAX_PATTERN_COUNT):
                    raise Error("chop count too large (max 4096)")
                ids.append(self.pat.chop(Int(c), p))
            return self.pat_val(self.per_cycle(counts, ids))
        if name == "striate" or name == "striateBy":
            # fastcat of n copies, copy i playing the i-th part of each sample
            var n = self.to_int(args[0])
            if n > MAX_PATTERN_COUNT:
                raise Error(name + " count too large (max 4096)")
            var width = 1.0 / Float64(max(1, n))
            var pi = 1
            if name == "striateBy":
                width = self.to_numbers(args[1])[0]
                pi = 2
            var p = self.to_pattern(args[pi])
            var parts = List[Int]()
            for i in range(max(1, n)):
                var b = Float64(i) / Float64(max(1, n))
                if name == "striateBy":
                    b = Float64(i) * (1.0 - width) / Float64(max(1, n - 1)) if n > 1 else 0.0
                var with_b = self.pat.union(OP_SET, p, self.control(P_BEGIN, b))
                parts.append(self.pat.union(OP_SET, with_b, self.control(P_END, b + width)))
            var cat = self.pat.slowcat(parts)
            return self.pat_val(self.pat.fast(Frac.of(len(parts)), cat))
        if name == "slice" or name == "splice" or name == "randslice":
            var n = self.to_int(args[0])
            if n > MAX_PATTERN_COUNT:
                raise Error(name + " count too large (max 4096)")
            var idx: Int
            var p: Int
            if name == "randslice":
                idx = self.pat.segment(1, self.pat.irand(n))
                p = self.to_pattern(args[1])
            else:
                idx = self.to_pattern(args[1])
                p = self.to_pattern(args[2])
            return self.pat_val(self.pat.union(OP_SET, self.pat.slice_values(n, idx), p))
        if name == "arp":
            var modes = self.to_words(args[0])
            var p = self.to_pattern(args[1])
            var ids = List[Int]()
            for m in modes:
                var mode = arp_mode(m)
                if mode < 0:
                    raise Error("unknown arp mode '" + m + "'")
                ids.append(self.pat.arp(mode, p))
            return self.pat_val(self.per_cycle(List[Float64](), ids))
        if name == "within" or name == "within'":
            var span = self.to_span(args[0])
            var p = self.to_pattern(args[2])
            var t: Int
            if name == "within'":
                # f sees the window zoomed to a whole cycle
                var w = span[1] - span[0]
                var zoomed = self.pat.fast(Frac.from_float(1.0 / w), self.pat.shift_left(Frac.from_float(-span[0]), p))
                var done = self.apply_fn(args[1], self.pat.window(0.0, w, zoomed))
                t = self.pat.shift_left(Frac.from_float(-span[0]), self.pat.slow(Frac.from_float(1.0 / w), done))
            else:
                t = self.apply_fn(args[1], p)
            return self.pat_val(self.within(span[0], span[1], t, p))
        if name == "chunk" or name == "chunk'":
            var n_raw = self.to_int(args[0])
            if n_raw > MAX_PATTERN_COUNT:
                raise Error(name + " count too large (max 4096)")
            var n = max(1, n_raw)
            var p = self.to_pattern(args[2])
            var t = self.apply_fn(args[1], p)
            var ids = List[Int]()
            for i in range(n):
                var k = i if name == "chunk" else (n - i) % n
                ids.append(self.within(Float64(k) / Float64(n), Float64(k + 1) / Float64(n), t, p))
            return self.pat_val(self.per_cycle(List[Float64](), ids))
        if name == "fix" or name == "unfix":
            var want_pat = self.to_pattern(args[1])
            var evs = self.pat.query_node(want_pat, Frac.of(0), Frac.of(1))
            if len(evs) == 0 or evs[0].value.raw:
                raise Error(name + " needs controls to match, like (s \"hh\") or (n 1)")
            var want = evs[0].value.copy()
            var p = self.to_pattern(args[2])
            var hit = self.pat.match(want.copy(), p, name == "unfix")
            var rest = self.pat.match(want^, p, name == "fix")
            return self.pat_val(self.pat.stack([self.apply_fn(args[0], hit), rest]))
        if name == "linger":
            var amounts = self.to_numbers(args[0])
            var p = self.to_pattern(args[1])
            var ids = List[Int]()
            for a in amounts:
                var aa = -a if a < 0 else a
                if aa > 0.0 and aa < 1.0 / Float64(MAX_PATTERN_COUNT):
                    raise Error("linger amount too small (min 1/4096)")
                if a < 0:
                    # negative: the end of the cycle instead
                    var f = Frac.from_float(-a)
                    ids.append(self.pat.linger(f, self.pat.shift_left(Frac.of(1) - f, p)))
                else:
                    ids.append(self.pat.linger(Frac.from_float(a), p))
            return self.pat_val(self.per_cycle(amounts, ids))
        if name == "swingBy" or name == "swing":
            var amount = 1.0 / 3.0
            var ni = 0
            if name == "swingBy":
                amount = self.to_numbers(args[0])[0]
                ni = 1
            var n = self.to_numbers(args[ni])[0]
            if n != n or n > Float64(MAX_PATTERN_COUNT):
                raise Error(name + " count out of range (max 4096)")
            var p = self.to_pattern(args[ni + 1])
            # inside n (within (0.5, 1) (amount ~>))
            var fr = Frac.from_float(n)
            var slowed = self.pat.slow(fr, p)
            var late = self.pat.shift_left(-Frac.from_float(amount), slowed)
            return self.pat_val(self.pat.fast(fr, self.within(0.5, 1.0, late, slowed)))
        raise Error("unknown function '" + name + "'")

    def within(mut self, lo: Float64, hi: Float64, transformed: Int, plain: Int) -> Int:
        return self.pat.stack([self.pat.window(lo, hi, transformed), self.pat.window(lo, hi, plain, True)])

    def to_span(mut self, id: Int) raises -> Tuple[Float64, Float64]:
        if self.vals[id].kind != V_LIST or len(self.vals[id].items) != 2:
            raise Error("expected a pair like (0, 0.5)")
        var a = self.vals[id].items[0]
        var b = self.vals[id].items[1]
        return (self.to_numbers(a)[0], self.to_numbers(b)[0])

    def to_words(mut self, id: Int) raises -> List[String]:
        """A word, or a per-cycle sequence for "<up down>"."""
        if self.vals[id].kind != V_STR and self.vals[id].kind != V_PAT:
            raise Error("expected a string")
        var p = self.to_pattern(id)
        var per = List[String]()
        for c in range(24):
            var evs = self.pat.query_node(p, Frac.of(c), Frac.of(c + 1))
            var w = per[len(per) - 1] if len(per) > 0 else String("up")
            for e in range(len(evs)):
                if evs[e].has_onset():
                    w = evs[e].value.word
                    break
            per.append(w)
        for period in range(1, 25):
            var ok = True
            for i in range(24):
                if per[i] != per[i % period]:
                    ok = False
                    break
            if ok:
                var out = List[String]()
                for i in range(period):
                    out.append(per[i])
                return out^
        return per^

    # ---- evaluation

    def eval(mut self, id: Int) raises -> Int:
        if self.eval_depth >= MAX_SCRIPT_EVAL_DEPTH:
            raise Error("evaluation recursion too deep (max " + String(MAX_SCRIPT_EVAL_DEPTH) + ")")
        self.eval_depth += 1
        var res: Int
        try:
            res = self._eval_inner(id)
        except e:
            self.eval_depth -= 1
            raise e^
        self.eval_depth -= 1
        return res

    def _eval_inner(mut self, id: Int) raises -> Int:
        var k = self.asts[id].kind
        var text = self.asts[id].text
        var na = self.asts[id].a
        var nb = self.asts[id].b
        if k == A_NUM:
            return self.num_val(self.asts[id].num)
        if k == A_STR:
            var v = Val(V_STR)
            v.text = text
            return self.push(v^)
        if k == A_VAR:
            var bound = self.lookup(text)
            if bound >= 0:
                return bound
            var ar = builtin_arity(text)
            if ar < 0:
                raise Error("unknown name '" + text + "'")
            if ar == 0:
                return self.call(text, List[Int]())
            return self.fun_val(text, List[Int]())
        if k == A_APP:
            var f = self.eval(na)
            var x = self.eval(nb)
            return self.apply(f, x)
        if k == A_OP:
            var l = self.eval(na)
            var r = self.eval(nb)
            return self.binop(text, l, r)
        if k == A_LIST:
            var items = self.asts[id].items.copy()
            var v = Val(V_LIST)
            for it in items:
                v.items.append(self.eval(it))
            return self.push(v^)
        if k == A_SECR:
            var rhs = self.eval(na)
            var opv = Val(V_STR)
            opv.text = text
            var o = self.push(opv^)
            return self.fun_val("secR", [o, rhs])
        if k == A_SECL:
            var lhs = self.eval(na)
            var opv = Val(V_STR)
            opv.text = text
            var o = self.push(opv^)
            return self.fun_val("secL", [lhs, o])
        if k == A_LAMBDA:
            var v = Val(V_USER)
            v.text = "lambda"
            v.pat = na
            v.items = self.asts[id].items.copy()
            return self.push(v^)
        raise Error("bad syntax tree")

    def import_ast(mut self, parser: Parser, root: Int) -> Int:
        """Move a statement's syntax tree into the program (ids shifted), so
        `let` functions can be evaluated later."""
        var base = len(self.asts)
        for i in range(len(parser.nodes)):
            var a = parser.nodes[i].copy()
            if a.a >= 0:
                a.a += base
            if a.b >= 0:
                a.b += base
            for j in range(len(a.items)):
                a.items[j] += base
            self.asts.append(a^)
        return root + base

    def run_statement(mut self, text: String) raises:
        var toks = tokenize(text)
        if toks[0].kind == T_ID and toks[0].text == "let":
            self.run_let(toks^, text)
            return
        var parser = Parser(toks^)
        var root = parser.parse_expr(0)
        if parser.kind() != T_END:
            raise Error("unexpected '" + parser.text() + "'")
        var v = self.eval(self.import_ast(parser, root))
        if self.vals[v].kind == V_FUN or self.vals[v].kind == V_USER:
            raise Error("'" + self.vals[v].text + "' is missing arguments")

    def run_let(mut self, var toks: List[Token], text: String) raises:
        """`let name = expr` or `let f a b = expr` (a function)."""
        if len(toks) < 4 or toks[1].kind != T_ID:
            raise Error("expected `let name = ...`")
        var name = toks[1].text
        var params = List[String]()
        var i = 2
        while i < len(toks) and toks[i].kind == T_ID:
            params.append(toks[i].text)
            i += 1
        if i >= len(toks) or not (toks[i].kind == T_OP and toks[i].text == "="):
            raise Error("expected '=' in let")
        var body_toks = List[Token]()
        for j in range(i + 1, len(toks)):
            body_toks.append(toks[j].copy())
        var parser = Parser(body_toks^)
        var param_ids = List[Int]()
        for p in params:
            param_ids.append(parser.add(Ast(A_VAR, p)))
        var root = parser.parse_expr(0)
        if parser.kind() != T_END:
            raise Error("unexpected '" + parser.text() + "'")
        var base = len(self.asts)
        var body = self.import_ast(parser, root)
        if len(params) == 0:
            var v = self.eval(body)
            self.env.append(Binding(name, v))
            return
        var user = Val(V_USER)
        user.text = name
        user.pat = body
        for pid in param_ids:
            user.items.append(pid + base)
        self.env.append(Binding(name, self.push(user^)))


def _strip_comment(line: String) -> String:
    var b = line.as_bytes()
    var in_str = False
    for i in range(len(b)):
        if b[i] == UInt8(ord("\"")):
            in_str = not in_str
        if not in_str and b[i] == UInt8(ord("-")) and i + 1 < len(b) and b[i + 1] == UInt8(ord("-")):
            return String(line[byte=0:i])
    return line


def _indent(line: String) -> Int:
    var n = 0
    for b in line.as_bytes():
        if b == UInt8(ord(" ")):
            n += 1
        elif b == UInt8(ord("\t")):
            n += 4
        else:
            break
    return n


def _after_keyword(text: String, keyword: String) -> Int:
    """Offset of the first non-space after a leading keyword."""
    var b = text.as_bytes()
    var i = keyword.byte_length()
    while i < len(b) and (b[i] == UInt8(ord(" ")) or b[i] == UInt8(ord("\t"))):
        i += 1
    return i


def split_statements(text: String) -> List[String]:
    """Haskell-style layout, the way Tidal files are written:
      - a line starting in the block's column starts a statement; deeper lines
        (and lines starting with an operator) continue it; blank lines end it;
      - `do` opens a block whose column is set by its first line;
      - `let a = ..` followed by lines aligned with `a` are separate bindings;
      - `:{` / `:}` and `import` lines are ignored."""
    var out = List[String]()
    var current = String()
    var block = 0
    var in_do = False
    var do_pending = False
    var let_col = -1
    for raw in text.split("\n"):
        var line = _strip_comment(String(raw))
        var stripped = String(line.strip())
        if stripped.byte_length() == 0:
            if current.byte_length() > 0:
                out.append(current)
                current = String()
            in_do = False
            do_pending = False
            block = 0
            let_col = -1
            continue
        if stripped == ":{" or stripped == ":}" or stripped.startswith("import "):
            continue
        var col = _indent(line)
        if do_pending:
            block = col
            in_do = True
            do_pending = False
        elif in_do and col < block:
            in_do = False
            block = 0
        if stripped == "do" or stripped.startswith("do ") or stripped.startswith("do\t"):
            if current.byte_length() > 0:
                out.append(current)
                current = String()
            let_col = -1
            var at = _after_keyword(stripped, "do")
            if at >= stripped.byte_length():
                do_pending = True
                continue
            in_do = True
            block = col + at
            col = block
            var rest = String(stripped[byte=at:stripped.byte_length()])
            stripped = rest
        var first = Int(stripped.as_bytes()[0])
        if let_col >= 0 and col == let_col and current.byte_length() > 0:
            out.append(current)
            current = "let " + stripped
            continue
        var continues = current.byte_length() > 0 and (col > block or _is_op_char(first))
        if continues:
            current += " " + stripped
            continue
        if current.byte_length() > 0:
            out.append(current)
        current = stripped
        let_col = -1
        if stripped.startswith("let ") or stripped.startswith("let\t"):
            let_col = col + _after_keyword(stripped, "let")
    if current.byte_length() > 0:
        out.append(current)
    return out^


def run_program(text: String) -> Program:
    """Evaluate a whole file; per-statement errors land in `errors`."""
    var prog = Program()
    for stmt in split_statements(text):
        try:
            prog.run_statement(stmt)
        except e:
            prog.errors.append(String(e) + "  <-  " + stmt)
    return prog^


def eval_pattern(expr: String) raises -> Program:
    """Evaluate a single pattern expression; its root becomes track `d1`."""
    var prog = Program()
    var parser = Parser(tokenize(expr))
    var root = parser.parse_expr(0)
    if parser.kind() != T_END:
        raise Error("unexpected '" + parser.text() + "'")
    var v = prog.eval(prog.import_ast(parser, root))
    var pid = prog.to_pattern(v)
    prog.tracks.append(Track("d1", pid))
    return prog^
