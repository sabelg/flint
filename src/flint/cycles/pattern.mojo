"""Pure-Mojo cyclic pattern engine (`flint.cycles.pattern`).

A native take on the TidalCycles model: a pattern is a function from a time
span (measured in cycles, with exact rational arithmetic) to the events active
in that span. Mojo has no heap closures, so patterns are trees of nodes in an
arena and `Pattern.query` walks the tree.

  - `Frac`: exact rational time (no float drift at cycle boundaries)
  - `Params`: the control values one event carries (s, n, note, gain, ...)
  - `Pattern`: node arena + every combinator (fastcat, slowcat, stack, fast,
    rev, every, degradeBy, signals, union `#` / `|+` ...)
  - `parse_mini`: the mini-notation ("bd*2 [~ sn] <hh oh>(3,8) a@3 b! c?")
"""
from std.math import sin, floor

comptime MAX_PATTERN_COUNT: Int = 4096
comptime MAX_QUERY_EVENTS: Int = 16384
comptime MAX_QUERY_STEPS: Int = 65536
comptime _INT64_MAX_128: Int128 = Int128(0x7FFFFFFFFFFFFFFF)

# ---------------------------------------------------------------- rational time


def _gcd(a: Int, b: Int) -> Int:
    var x = a if a >= 0 else -a
    var y = b if b >= 0 else -b
    while y != 0:
        var t = x % y
        x = y
        y = t
    return x if x != 0 else 1


@fieldwise_init
struct Frac(Copyable, ImplicitlyCopyable, Movable):
    """An exact rational number of cycles; always reduced, `den > 0`."""

    var num: Int
    var den: Int

    @staticmethod
    def make(num: Int, den: Int) -> Frac:
        if den == 0:
            return Frac(0, 1)
        if num == Int.MIN or den == Int.MIN:
            return Frac.make128(Int128(num), Int128(den))
        var n = num
        var d = den
        if d < 0:
            n = -n
            d = -d
        var g = _gcd(n, d)
        return Frac(n // g, d // g)

    @staticmethod
    def make128(num: Int128, den: Int128) -> Frac:
        """Reduce a 128-bit intermediate back to an `Int` fraction."""
        if den == 0:
            return Frac(0, 1)
        if num == 0:
            return Frac(0, 1)
        var n = num
        var d = den
        if d < 0:
            n = -n
            d = -d
        var x = n if n >= 0 else -n
        var y = d
        while y != 0:
            var t = x % y
            x = y
            y = t
        if x == 0:
            x = 1
        var rn = n // x
        var rd = d // x
        var neg = rn < 0
        var abs_rn = -rn if neg else rn
        if abs_rn <= _INT64_MAX_128 and rd <= _INT64_MAX_128:
            return Frac(Int(rn), Int(rd))
        # Continued-fraction convergent / semiconvergent within [-2^63+1, 2^63-1].
        var a = abs_rn
        var b = rd
        var p0 = Int128(0)
        var q0 = Int128(1)
        var p1 = Int128(1)
        var q1 = Int128(0)
        while b != 0:
            var q = a // b
            var r = a % b
            if (p1 > 0 and q > (_INT64_MAX_128 - p0) // p1) or (q1 > 0 and q > (_INT64_MAX_128 - q0) // q1):
                var k_max = q
                if p1 > 0:
                    var kp = (_INT64_MAX_128 - p0) // p1
                    if kp < k_max:
                        k_max = kp
                if q1 > 0:
                    var kq = (_INT64_MAX_128 - q0) // q1
                    if kq < k_max:
                        k_max = kq
                if k_max > 0:
                    p1 = k_max * p1 + p0
                    q1 = k_max * q1 + q0
                break
            var next_p = q * p1 + p0
            var next_q = q * q1 + q0
            p0 = p1
            q0 = q1
            p1 = next_p
            q1 = next_q
            a = b
            b = r
        if q1 == 0:
            q1 = Int128(1)
            p1 = _INT64_MAX_128
        var out_n = -Int(p1) if neg else Int(p1)
        return Frac(out_n, Int(q1))

    @staticmethod
    def of(n: Int) -> Frac:
        return Frac(n, 1)

    @staticmethod
    def from_float(x: Float64) -> Frac:
        """Nearest fraction with a small denominator (exact for 1/2, 1/3, 0.125, ...)."""
        if x != x:
            return Frac(0, 1)
        var cx = max(-1.0e9, min(1.0e9, x))
        var best_n = Int(floor(cx * 1048576.0 + 0.5))
        var best = Frac.make(best_n, 1048576)
        for d in range(1, 65):
            var n = Int(floor(cx * Float64(d) + 0.5))
            var err = Float64(n) / Float64(d) - cx
            if err < 0:
                err = -err
            if err < 1.0e-9:
                return Frac.make(n, d)
        return best

    def __add__(self, o: Frac) -> Frac:
        if self.den == o.den:
            return Frac.make128(Int128(self.num) + Int128(o.num), Int128(self.den))
        return Frac.make128(
            Int128(self.num) * Int128(o.den) + Int128(o.num) * Int128(self.den), Int128(self.den) * Int128(o.den))

    def __sub__(self, o: Frac) -> Frac:
        if self.den == o.den:
            return Frac.make128(Int128(self.num) - Int128(o.num), Int128(self.den))
        return Frac.make128(
            Int128(self.num) * Int128(o.den) - Int128(o.num) * Int128(self.den), Int128(self.den) * Int128(o.den))

    def __mul__(self, o: Frac) -> Frac:
        return Frac.make128(Int128(self.num) * Int128(o.num), Int128(self.den) * Int128(o.den))

    def __truediv__(self, o: Frac) -> Frac:
        if o.num == 0:
            return Frac(0, 1)
        return Frac.make128(Int128(self.num) * Int128(o.den), Int128(self.den) * Int128(o.num))

    def __neg__(self) -> Frac:
        if self.num == Int.MIN:
            return Frac.make128(-Int128(self.num), Int128(self.den))
        return Frac(-self.num, self.den)

    def __eq__(self, o: Frac) -> Bool:
        return self.num == o.num and self.den == o.den

    def __ne__(self, o: Frac) -> Bool:
        return not (self == o)

    def __lt__(self, o: Frac) -> Bool:
        return Int128(self.num) * Int128(o.den) < Int128(o.num) * Int128(self.den)

    def __le__(self, o: Frac) -> Bool:
        return Int128(self.num) * Int128(o.den) <= Int128(o.num) * Int128(self.den)

    def __gt__(self, o: Frac) -> Bool:
        return o < self

    def __ge__(self, o: Frac) -> Bool:
        return o <= self

    def sam(self) -> Int:
        """The cycle this time falls in (floor)."""
        return self.num // self.den

    def next_sam(self) -> Int:
        return self.sam() + 1

    def to_float(self) -> Float64:
        return Float64(self.num) / Float64(self.den)

    def to_string(self) -> String:
        if self.den == 1:
            return String(self.num)
        return String(self.num) + "/" + String(self.den)


def fmin(a: Frac, b: Frac) -> Frac:
    return a if a < b else b


def fmax(a: Frac, b: Frac) -> Frac:
    return a if a > b else b


# ---------------------------------------------------------------- control values

comptime P_N: Int = 0
comptime P_NOTE: Int = 1
comptime P_GAIN: Int = 2
comptime P_PAN: Int = 3
comptime P_SPEED: Int = 4
comptime P_CUTOFF: Int = 5
comptime P_RESONANCE: Int = 6
comptime P_HCUTOFF: Int = 7
comptime P_ROOM: Int = 8
comptime P_SIZE: Int = 9
comptime P_SHAPE: Int = 10
comptime P_CRUSH: Int = 11
comptime P_LEGATO: Int = 12
comptime P_ATTACK: Int = 13
comptime P_RELEASE: Int = 14
comptime P_DELAY: Int = 15
comptime P_DELAYTIME: Int = 16
comptime P_DELAYFB: Int = 17
comptime P_COARSE: Int = 18
comptime P_BEGIN: Int = 19
comptime P_END: Int = 20
comptime P_SUSTAIN: Int = 21
comptime P_COUNT: Int = 22
comptime P_SOUND: Int = 100  # pseudo id: the `s` string


def param_id(name: String) -> Int:
    """Control name (with Tidal aliases) to id; -1 when unknown."""
    if name == "s" or name == "sound":
        return P_SOUND
    if name == "n":
        return P_N
    if name == "note" or name == "up":
        return P_NOTE
    if name == "gain":
        return P_GAIN
    if name == "pan":
        return P_PAN
    if name == "speed":
        return P_SPEED
    if name == "cutoff" or name == "lpf":
        return P_CUTOFF
    if name == "resonance" or name == "lpq":
        return P_RESONANCE
    if name == "hcutoff" or name == "hpf":
        return P_HCUTOFF
    if name == "room":
        return P_ROOM
    if name == "size" or name == "sz":
        return P_SIZE
    if name == "shape":
        return P_SHAPE
    if name == "crush":
        return P_CRUSH
    if name == "legato":
        return P_LEGATO
    if name == "attack" or name == "att":
        return P_ATTACK
    if name == "release" or name == "rel":
        return P_RELEASE
    if name == "delay":
        return P_DELAY
    if name == "delaytime" or name == "delayt":
        return P_DELAYTIME
    if name == "delayfeedback" or name == "delayfb":
        return P_DELAYFB
    if name == "coarse":
        return P_COARSE
    if name == "begin":
        return P_BEGIN
    if name == "end":
        return P_END
    if name == "sustain" or name == "sus":
        return P_SUSTAIN
    return -1


def param_name(id: Int) -> String:
    var names: List[String] = [
        "n", "note", "gain", "pan", "speed", "cutoff", "resonance", "hcutoff",
        "room", "size", "shape", "crush", "legato", "attack", "release", "delay",
        "delaytime", "delayfeedback", "coarse", "begin", "end", "sustain",
    ]
    if id == P_SOUND:
        return "s"
    if id >= 0 and id < len(names):
        return names[id]
    return "?"


struct Params(Copyable, Movable):
    """What one event carries. Before a control (`s`, `n`, ...) names it, a
    mini-notation atom is *raw*: just `word` and, if numeric, `num`."""

    var raw: Bool
    var word: String
    var num: Float64
    var has_num: Bool
    var sound: String
    var has_sound: Bool
    var vals: List[Float64]
    var mask: Int

    def __init__(out self):
        self.raw = True
        self.word = String()
        self.num = 0.0
        self.has_num = False
        self.sound = String()
        self.has_sound = False
        self.vals = List[Float64](length=P_COUNT, fill=0.0)
        self.mask = 0

    def has(self, id: Int) -> Bool:
        if id == P_SOUND:
            return self.has_sound
        return (self.mask >> id) & 1 == 1

    def get(self, id: Int, fallback: Float64) -> Float64:
        if self.has(id):
            return self.vals[id]
        return fallback

    def set(mut self, id: Int, value: Float64):
        self.vals[id] = value
        self.mask |= 1 << id
        self.raw = False

    def set_sound(mut self, name: String):
        self.sound = name
        self.has_sound = True
        self.raw = False

    def describe(self) -> String:
        if self.raw:
            return self.word
        var out = String()
        if self.has_sound:
            out += "s=" + self.sound
        for id in range(P_COUNT):
            if self.has(id):
                if out.byte_length() > 0:
                    out += " "
                out += param_name(id) + "=" + _fmt(self.vals[id])
        return out


def _fmt(x: Float64) -> String:
    var r = floor(x + 0.5)
    if (x - r) < 1.0e-9 and (r - x) < 1.0e-9:
        return String(Int(r))
    return String(Float64(Int(x * 1000.0 + (0.5 if x >= 0 else -0.5))) / 1000.0)


def raw_atom(word: String) -> Params:
    var p = Params()
    p.word = word
    var parsed = parse_number(word)
    if parsed[1]:
        p.num = parsed[0]
        p.has_num = True
    else:
        var note = parse_note_name(word)
        if note[1]:
            p.num = note[0]
            p.has_num = True
    return p^


def raw_number(x: Float64) -> Params:
    var p = Params()
    p.word = _fmt(x)
    p.num = x
    p.has_num = True
    return p^


def parse_number(text: String) -> Tuple[Float64, Bool]:
    """Decimal like `-0.25`, `3`, `1e-3`; (0, False) if not a number."""
    var b = text.as_bytes()
    var n = len(b)
    if n == 0:
        return (0.0, False)
    var i = 0
    var sign = 1.0
    if b[0] == UInt8(ord("-")):
        sign = -1.0
        i = 1
    elif b[0] == UInt8(ord("+")):
        i = 1
    var value = 0.0
    var digits = 0
    while i < n and b[i] >= UInt8(ord("0")) and b[i] <= UInt8(ord("9")):
        value = value * 10.0 + Float64(Int(b[i]) - 48)
        i += 1
        digits += 1
    if i < n and b[i] == UInt8(ord(".")):
        i += 1
        var scale = 0.1
        while i < n and b[i] >= UInt8(ord("0")) and b[i] <= UInt8(ord("9")):
            value += Float64(Int(b[i]) - 48) * scale
            scale *= 0.1
            i += 1
            digits += 1
    if digits == 0:
        return (0.0, False)
    if i < n and (b[i] == UInt8(ord("e")) or b[i] == UInt8(ord("E"))):
        i += 1
        var es = 1
        if i < n and b[i] == UInt8(ord("-")):
            es = -1
            i += 1
        elif i < n and b[i] == UInt8(ord("+")):
            i += 1
        var e = 0
        var ed = 0
        while i < n and b[i] >= UInt8(ord("0")) and b[i] <= UInt8(ord("9")):
            if e <= 10000:
                e = e * 10 + Int(b[i]) - 48
            i += 1
            ed += 1
        if ed == 0:
            return (0.0, False)
        if value == 0.0:
            value = 0.0
        elif es > 0 and e > 308:
            value = Float64(1e308) * Float64(1e308)
        elif es < 0 and e > 324:
            value = 0.0
        else:
            for _ in range(e):
                value = value * 10.0 if es > 0 else value / 10.0
    if i != n:
        return (0.0, False)
    return (sign * value, True)


def parse_note_name(text: String) -> Tuple[Float64, Bool]:
    """Tidal note names: `c` `e` `a5` `cs4` `ef` `f#3` — `c5` is 0."""
    var b = text.as_bytes()
    var n = len(b)
    if n == 0:
        return (0.0, False)
    var letter = Int(b[0])
    var semis: List[Int] = [9, 11, 0, 2, 4, 5, 7]  # a b c d e f g
    if letter < ord("a") or letter > ord("g"):
        return (0.0, False)
    var value = semis[letter - ord("a")]
    var i = 1
    while i < n:
        var c = Int(b[i])
        if c == ord("s") or c == ord("#"):
            value += 1
        elif c == ord("f") or c == ord("b"):
            value -= 1  # Tidal spells flats `ef`; `eb` is accepted too
        elif c == ord("n"):
            pass
        else:
            break
        i += 1
    var octave = 5
    if i < n:
        var oct = parse_number(String(text[byte=i:n]))
        if not oct[1]:
            return (0.0, False)
        octave = Int(oct[0])
    return (Float64(value + (octave - 5) * 12), True)


# ---------------------------------------------------------------- events


struct Event(Copyable, Movable):
    """`whole` is the event's full extent; `part` the fragment inside the query.
    An event *starts* in a query when `part_b == whole_b`."""

    var whole_b: Frac
    var whole_e: Frac
    var part_b: Frac
    var part_e: Frac
    var value: Params

    def __init__(out self, whole_b: Frac, whole_e: Frac, part_b: Frac, part_e: Frac, var value: Params):
        self.whole_b = whole_b
        self.whole_e = whole_e
        self.part_b = part_b
        self.part_e = part_e
        self.value = value^

    def has_onset(self) -> Bool:
        return self.part_b == self.whole_b

    def describe(self) -> String:
        return "(" + self.whole_b.to_string() + ">" + self.whole_e.to_string() + ") " + self.value.describe()


# ---------------------------------------------------------------- node arena

comptime K_SILENCE: Int = 0
comptime K_ATOM: Int = 1      # value: atoms[a]
comptime K_SEQ: Int = 2       # weighted fastcat of kids
comptime K_STACK: Int = 3
comptime K_ALT: Int = 4       # slowcat `<a b c>`
comptime K_FAST: Int = 5      # factor f, child a
comptime K_SHIFT: Int = 6     # child a queried at t + f (rotL)
comptime K_REV: Int = 7
comptime K_CHOOSE: Int = 8    # cycle-chooser: a=plain, b=transformed, c=mode, d=n, e=offset
comptime K_DEGRADE: Int = 9   # keep with prob 1-f (as float in x), seed c, child a
comptime K_CONTROL: Int = 10  # name raw values as param c, child a
comptime K_UNION: Int = 11    # structure from a, values from b; op c
comptime K_SIGNAL: Int = 12   # continuous: c = shape
comptime K_SCALE: Int = 13    # child a, scale index c
comptime K_RANGE: Int = 14    # child a, lo x, hi y
comptime K_PLY: Int = 15      # child a, repeat each event c times
comptime K_ITER: Int = 16     # child a, c subdivisions, d=1 backwards
comptime K_SEGMENT: Int = 17  # child a sampled c times per cycle
comptime K_IRAND: Int = 18    # c = n ints, continuous
comptime K_RANDCAT: Int = 19  # kids, pick one per cycle
comptime K_ALT_ABS: Int = 20  # kids, pick kid (cycle % n) without shifting time
comptime K_CHOP: Int = 21     # child a cut into c pieces, begin/end subdivided
comptime K_ARP: Int = 22      # child a, stacked events spread in time; c = mode
comptime K_WINDOW: Int = 23   # child a kept where cyclePos(onset) in [x, y); d=1 inverts
comptime K_MATCH: Int = 24    # child a kept where its values match atoms[c]; d=1 inverts
comptime K_LINGER: Int = 25   # child a: the first f of each cycle, repeated
comptime K_SLICEVAL: Int = 26 # child a raw numbers i -> begin i/c, end (i+1)/c

comptime ST_LEFT: Int = 0     # `|+`  `#`: structure from the left
comptime ST_RIGHT: Int = 1    # `+|`: structure from the right
comptime ST_BOTH: Int = 2     # `|+|`, bare `+`: structure from both

comptime ARP_UP: Int = 0
comptime ARP_DOWN: Int = 1
comptime ARP_UPDOWN: Int = 2
comptime ARP_DOWNUP: Int = 3
comptime ARP_UP_AND_DOWN: Int = 4
comptime ARP_DOWN_AND_UP: Int = 5
comptime ARP_CONVERGE: Int = 6
comptime ARP_DIVERGE: Int = 7
comptime ARP_DISCONVERGE: Int = 8
comptime ARP_PINKYUP: Int = 9
comptime ARP_PINKYUPDOWN: Int = 10
comptime ARP_THUMBUP: Int = 11
comptime ARP_THUMBUPDOWN: Int = 12


def arp_mode(name: String) -> Int:
    var names = arp_mode_names()
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


def arp_mode_names() -> List[String]:
    return [
        "up", "down", "updown", "downup", "up&down", "down&up", "converge", "diverge",
        "disconverge", "pinkyup", "pinkyupdown", "thumbup", "thumbupdown",
    ]


def _converge(xs: List[Int]) -> List[Int]:
    """x0, xn, x1, xn-1, ... (Tidal's converge)."""
    var out = List[Int]()
    var lo = 0
    var hi = len(xs) - 1
    var from_front = True
    while lo <= hi:
        if from_front:
            out.append(xs[lo])
            lo += 1
        else:
            out.append(xs[hi])
            hi -= 1
        from_front = not from_front
    return out^


def _reversed(xs: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(xs) - 1, -1, -1):
        out.append(xs[i])
    return out^


def _init(xs: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(xs) - 1):
        out.append(xs[i])
    return out^


def _concat(a: List[Int], b: List[Int]) -> List[Int]:
    var out = a.copy()
    for v in b:
        out.append(v)
    return out^


def _pinkyup(xs: List[Int]) -> List[Int]:
    var out = List[Int]()
    if len(xs) == 0:
        return out^
    for i in range(len(xs) - 1):
        out.append(xs[i])
        out.append(xs[len(xs) - 1])
    return out^


def _thumbup(xs: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(1, len(xs)):
        out.append(xs[0])
        out.append(xs[i])
    return out^


def arp_order(mode: Int, count: Int) -> List[Int]:
    """Indices into a chord of `count` notes, in the order `mode` plays them."""
    var xs = List[Int]()
    for i in range(count):
        xs.append(i)
    var out: List[Int]
    if mode == ARP_DOWN:
        out = _reversed(xs)
    elif mode == ARP_UPDOWN:
        out = _concat(_init(xs), _init(_reversed(xs)))
    elif mode == ARP_DOWNUP:
        out = _concat(_init(_reversed(xs)), _init(xs))
    elif mode == ARP_UP_AND_DOWN:
        out = _concat(xs, _reversed(xs))
    elif mode == ARP_DOWN_AND_UP:
        out = _concat(_reversed(xs), xs)
    elif mode == ARP_CONVERGE:
        out = _converge(xs)
    elif mode == ARP_DIVERGE:
        out = _reversed(_converge(xs))
    elif mode == ARP_DISCONVERGE:
        var c = _converge(xs)
        var r = _reversed(c)
        var tail = List[Int]()
        for i in range(1, len(r)):
            tail.append(r[i])
        out = _concat(c, tail)
    elif mode == ARP_PINKYUP:
        out = _pinkyup(xs)
    elif mode == ARP_PINKYUPDOWN:
        var pu = _pinkyup(xs)
        out = _concat(_init(pu), _init(_reversed(pu)))
    elif mode == ARP_THUMBUP:
        out = _thumbup(xs)
    elif mode == ARP_THUMBUPDOWN:
        var tu = _thumbup(xs)
        out = _concat(_init(tu), _init(_reversed(tu)))
    else:
        out = xs.copy()
    if len(out) == 0:
        return xs^
    return out^


def chord_table(name: String) -> List[Int]:
    """Tidal chord names (`c'maj`, `e'min7`, ...)."""
    if name == "major" or name == "maj" or name == "M":
        return [0, 4, 7]
    if name == "aug" or name == "plus" or name == "sharp5":
        return [0, 4, 8]
    if name == "six" or name == "6" or name == "sixby9" or name == "6by9":
        return [0, 4, 7, 9]
    if name == "sixNine" or name == "six9" or name == "m6by9":
        return [0, 4, 7, 9, 14]
    if name == "major7" or name == "maj7":
        return [0, 4, 7, 11]
    if name == "major9" or name == "maj9":
        return [0, 4, 7, 11, 14]
    if name == "add9" or name == "major9add":
        return [0, 4, 7, 14]
    if name == "major11" or name == "maj11":
        return [0, 4, 7, 11, 14, 17]
    if name == "add11":
        return [0, 4, 7, 17]
    if name == "major13" or name == "maj13":
        return [0, 4, 7, 11, 14, 21]
    if name == "add13":
        return [0, 4, 7, 21]
    if name == "dom7":
        return [0, 4, 7, 10]
    if name == "dom9":
        return [0, 4, 7, 14]
    if name == "dom11":
        return [0, 4, 7, 17]
    if name == "dom13":
        return [0, 4, 7, 21]
    if name == "sevenFlat5" or name == "7f5":
        return [0, 4, 6, 10]
    if name == "sevenSharp5" or name == "7s5":
        return [0, 4, 8, 10]
    if name == "sevenFlat9" or name == "7f9":
        return [0, 4, 7, 10, 13]
    if name == "nine":
        return [0, 4, 7, 10, 14]
    if name == "eleven" or name == "11":
        return [0, 4, 7, 10, 14, 17]
    if name == "thirteen" or name == "13":
        return [0, 4, 7, 10, 14, 17, 21]
    if name == "minor" or name == "min" or name == "m":
        return [0, 3, 7]
    if name == "diminished" or name == "dim":
        return [0, 3, 6]
    if name == "minorSharp5" or name == "msharp5" or name == "mS5":
        return [0, 3, 8]
    if name == "minor6" or name == "min6" or name == "m6":
        return [0, 3, 7, 9]
    if name == "minorSixNine" or name == "minor69" or name == "min69" or name == "minSixNine" or name == "m69":
        return [0, 3, 9, 7, 14]
    if name == "minor7flat5" or name == "min7flat5" or name == "m7flat5" or name == "m7f5":
        return [0, 3, 6, 10]
    if name == "minorMajor7" or name == "minMaj7" or name == "mmaj7":
        return [0, 3, 7, 11]
    if name == "diminished7" or name == "dim7":
        return [0, 3, 6, 9]
    if name == "minor7" or name == "min7" or name == "m7":
        return [0, 3, 7, 10]
    if name == "minor9" or name == "min9" or name == "m9":
        return [0, 3, 7, 10, 14]
    if name == "minor11" or name == "min11" or name == "m11":
        return [0, 3, 7, 10, 14, 17]
    if name == "minor13" or name == "min13" or name == "m13":
        return [0, 3, 7, 10, 14, 17, 21]
    if name == "one" or name == "1":
        return [0]
    if name == "five" or name == "5":
        return [0, 7]
    if name == "sus2":
        return [0, 2, 7]
    if name == "sus4":
        return [0, 5, 7]
    if name == "sevenSus2" or name == "7sus2":
        return [0, 2, 7, 10]
    if name == "sevenSus4" or name == "7sus4":
        return [0, 5, 7, 10]
    if name == "nineSus4" or name == "ninesus4" or name == "9sus4":
        return [0, 5, 7, 10, 14]
    if name == "sevenFlat10" or name == "7f10":
        return [0, 4, 7, 10, 15]
    if name == "nineSharp5" or name == "9sharp5" or name == "9s5":
        return [0, 1, 13]
    if name == "minor9sharp5" or name == "minor9s5" or name == "min9sharp5" or name == "min9s5" or name == "m9sharp5" or name == "m9s5":
        return [0, 1, 14]
    if name == "sevenSharp5flat9" or name == "7s5f9":
        return [0, 4, 8, 10, 13]
    if name == "minor7sharp9" or name == "m7sharp9" or name == "m7s9":
        return [0, 3, 7, 10, 14]
    if name == "elevenSharp" or name == "11s":
        return [0, 4, 7, 10, 14, 18]
    if name == "minor11sharp" or name == "m11sharp" or name == "m11s":
        return [0, 3, 7, 10, 14, 18]
    return List[Int]()


def chord_notes(word: String) -> Tuple[List[Float64], Bool]:
    """`c'maj`, `e4'min7'ii`, `0'major'5`, `a'm9'o`: root, chord, then modifiers
    (`i` inversions, `o` open voicing, a number = how many notes, extended by octaves)."""
    var parts = word.split("'")
    var out = List[Float64]()
    if len(parts) < 2:
        return (out^, False)
    var root_word = String(parts[0])
    var root = parse_number(root_word)
    if not root[1]:
        root = parse_note_name(root_word)
        if not root[1]:
            return (out^, False)
    var table = chord_table(String(parts[1]))
    if len(table) == 0:
        return (out^, False)
    var notes = List[Int]()
    for v in table:
        notes.append(v)
    for m in range(2, len(parts)):
        var mod = String(parts[m])
        var count = parse_number(mod)
        if count[1]:
            var want_f = count[0]
            if want_f != want_f or want_f < 0.0 or want_f > Float64(MAX_PATTERN_COUNT):
                return (List[Float64](), False)
            var want = min(MAX_PATTERN_COUNT, max(0, Int(want_f)))
            var base = len(table)
            var ext = List[Int](capacity=want)
            for i in range(want):
                ext.append(table[i % base] + 12 * (i // base))
            notes = ext^
            continue
        for b in mod.as_bytes():
            if b == UInt8(ord("i")) and len(notes) > 0:
                # inversion: the lowest note moves up an octave, to the top
                var rotated = List[Int]()
                for i in range(1, len(notes)):
                    rotated.append(notes[i])
                rotated.append(notes[0] + 12)
                notes = rotated^
            elif b == UInt8(ord("o")) and len(notes) > 2:
                # open voicing: 1st and 3rd an octave down, in front
                var opened: List[Int] = [notes[0] - 12, notes[2] - 12]
                for i in range(1, len(notes)):
                    if i != 2:
                        opened.append(notes[i])
                notes = opened^
            elif b == UInt8(ord("d")) and len(notes) > 1:
                # drop: the top note an octave down, in front
                var dropped: List[Int] = [notes[len(notes) - 1] - 12]
                for i in range(len(notes) - 1):
                    dropped.append(notes[i])
                notes = dropped^
    for v in notes:
        out.append(root[0] + Float64(v))
    return (out^, True)

comptime CH_EVERY: Int = 0    # n, offset: (cycle - offset) % n == 0
comptime CH_WHENMOD: Int = 1  # n, offset: cycle % n >= offset
comptime CH_SOMECYCLES: Int = 2  # prob x per cycle

comptime OP_SET: Int = 0     # `#`  right values win
comptime OP_ADD: Int = 1     # `|+` / `+`
comptime OP_SUB: Int = 2
comptime OP_MUL: Int = 3
comptime OP_DIV: Int = 4
comptime OP_KEEP: Int = 5    # `|<` left values win
comptime OP_MOD: Int = 6

comptime SIG_SINE: Int = 0
comptime SIG_COSINE: Int = 1
comptime SIG_SAW: Int = 2
comptime SIG_ISAW: Int = 3
comptime SIG_TRI: Int = 4
comptime SIG_SQUARE: Int = 5
comptime SIG_RAND: Int = 6
comptime SIG_PERLIN: Int = 7


@fieldwise_init
struct Node(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var a: Int
    var b: Int
    var c: Int
    var d: Int
    var e: Int
    var f: Frac
    var x: Float64
    var y: Float64
    var kid_start: Int
    var kid_count: Int


def node(kind: Int) -> Node:
    return Node(kind, -1, -1, 0, 0, 0, Frac(0, 1), 0.0, 0.0, 0, 0)


def scale_table(name: String) -> List[Int]:
    if name == "major" or name == "ionian":
        return [0, 2, 4, 5, 7, 9, 11]
    if name == "minor" or name == "aeolian":
        return [0, 2, 3, 5, 7, 8, 10]
    if name == "dorian":
        return [0, 2, 3, 5, 7, 9, 10]
    if name == "phrygian":
        return [0, 1, 3, 5, 7, 8, 10]
    if name == "lydian":
        return [0, 2, 4, 6, 7, 9, 11]
    if name == "mixolydian":
        return [0, 2, 4, 5, 7, 9, 10]
    if name == "locrian":
        return [0, 1, 3, 5, 6, 8, 10]
    if name == "majPent":
        return [0, 2, 4, 7, 9]
    if name == "minPent":
        return [0, 3, 5, 7, 10]
    if name == "ritusen":
        return [0, 2, 5, 7, 9]
    if name == "egyptian":
        return [0, 2, 5, 7, 10]
    if name == "harmonicMinor":
        return [0, 2, 3, 5, 7, 8, 11]
    if name == "melodicMinor":
        return [0, 2, 3, 5, 7, 9, 11]
    if name == "blues":
        return [0, 3, 5, 6, 7, 10]
    if name == "whole" or name == "wholetone":
        return [0, 2, 4, 6, 8, 10]
    if name == "hexMajor7":
        return [0, 2, 4, 7, 9, 11]
    if name == "chromatic":
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    return List[Int]()


def time_hash(t: Frac, seed: Int) -> Float64:
    """Deterministic pseudo-random in [0, 1) keyed by a time and a seed (Tidal's `rand`)."""
    var x = UInt64(t.num * 2654435761 + t.den * 40503 + seed * 97531 + 12345)
    x ^= x >> 33
    x *= 0xFF51AFD7ED558CCD
    x ^= x >> 33
    x *= 0xC4CEB9FE1A85EC53
    x ^= x >> 33
    return Float64(x >> 11) / 9007199254740992.0


def _combine(op: Int, l: Float64, r: Float64) -> Float64:
    if op == OP_ADD:
        return l + r
    if op == OP_SUB:
        return l - r
    if op == OP_MUL:
        return l * r
    if op == OP_DIV:
        return l / r if r != 0 else 0.0
    if op == OP_MOD:
        if r == 0:
            return 0.0
        return l - r * floor(l / r)
    if op == OP_KEEP:
        return l
    return r


def merge_params(op: Int, left: Params, right: Params) -> Params:
    var out = left.copy()
    if left.raw and right.raw:
        if left.has_num and right.has_num:
            out = raw_number(_combine(op, left.num, right.num))
        elif op == OP_SET:
            out = right.copy()
        elif op == OP_ADD:
            out.word = left.word + right.word  # string patterns concatenate
        return out^
    if right.raw:
        # `n "0 2" + "12"`: arithmetic with a bare number touches n and note.
        if right.has_num and op != OP_SET and op != OP_KEEP:
            for id in [P_N, P_NOTE]:
                if out.has(id):
                    out.set(id, _combine(op, out.vals[id], right.num))
        return out^
    if left.raw:
        if op == OP_SET or op == OP_KEEP:
            return right.copy()
        var res = right.copy()
        if left.has_num:
            for id in [P_N, P_NOTE]:
                if res.has(id):
                    res.set(id, _combine(op, left.num, res.vals[id]))
        return res^
    if right.has_sound:
        if not out.has_sound or op == OP_SET:
            out.set_sound(right.sound)
        elif op == OP_ADD:
            out.set_sound(out.sound + right.sound)
    for id in range(P_COUNT):
        if right.has(id):
            if out.has(id):
                out.set(id, _combine(op, out.vals[id], right.vals[id]))
            else:
                out.set(id, right.vals[id])
    return out^


def _signal_value(shape: Int, t: Frac, seed: Int) -> Float64:
    var x = t.to_float()
    var ph = x - floor(x)
    if shape == SIG_SINE:
        return 0.5 + 0.5 * sin(ph * 6.283185307179586)
    if shape == SIG_COSINE:
        return 0.5 + 0.5 * sin((ph + 0.25) * 6.283185307179586)
    if shape == SIG_SAW:
        return ph
    if shape == SIG_ISAW:
        return 1.0 - ph
    if shape == SIG_TRI:
        return ph * 2.0 if ph < 0.5 else 2.0 - ph * 2.0
    if shape == SIG_SQUARE:
        return 0.0 if ph < 0.5 else 1.0
    if shape == SIG_PERLIN:
        var i = Int(floor(x))
        var a = time_hash(Frac(i, 1), seed)
        var b = time_hash(Frac(i + 1, 1), seed)
        var s = ph * ph * (3.0 - 2.0 * ph)
        return a + (b - a) * s
    return time_hash(t, seed)


struct Pattern(Copyable, Movable):
    """A node arena; `root` is the pattern. Builders return node ids so
    several patterns can share subtrees inside one arena."""

    var nodes: List[Node]
    var kids: List[Int]
    var weights: List[Frac]
    var atoms: List[Params]
    var root: Int

    def __init__(out self):
        self.nodes = List[Node]()
        self.kids = List[Int]()
        self.weights = List[Frac]()
        self.atoms = List[Params]()
        self.root = -1
        self.root = self.silence()

    # ---- builders

    def add(mut self, n: Node) -> Int:
        self.nodes.append(n)
        return len(self.nodes) - 1

    def silence(mut self) -> Int:
        return self.add(node(K_SILENCE))

    def atom(mut self, var value: Params) -> Int:
        self.atoms.append(value^)
        var n = node(K_ATOM)
        n.a = len(self.atoms) - 1
        return self.add(n)

    def pure_number(mut self, x: Float64) -> Int:
        return self.atom(raw_number(x))

    def with_kids(mut self, kind: Int, ids: List[Int], ws: List[Frac]) -> Int:
        var n = node(kind)
        n.kid_start = len(self.kids)
        n.kid_count = len(ids)
        for i in range(len(ids)):
            self.kids.append(ids[i])
            self.weights.append(ws[i] if i < len(ws) else Frac(1, 1))
        return self.add(n)

    def seq(mut self, ids: List[Int]) -> Int:
        if len(ids) == 1:
            return ids[0]
        return self.with_kids(K_SEQ, ids, List[Frac]())

    def stack(mut self, ids: List[Int]) -> Int:
        if len(ids) == 1:
            return ids[0]
        return self.with_kids(K_STACK, ids, List[Frac]())

    def slowcat(mut self, ids: List[Int]) -> Int:
        if len(ids) == 1:
            return ids[0]
        return self.with_kids(K_ALT, ids, List[Frac]())

    def randcat(mut self, ids: List[Int]) -> Int:
        return self.with_kids(K_RANDCAT, ids, List[Frac]())

    def fast(mut self, factor: Frac, child: Int) -> Int:
        if factor.num <= 0:
            return self.silence()
        var eff_factor = factor
        var eff_child = child
        if child >= 0 and child < len(self.nodes) and self.nodes[child].kind == K_FAST:
            eff_factor = factor * self.nodes[child].f
            eff_child = self.nodes[child].a
        var n = node(K_FAST)
        n.a = eff_child
        if Int128(eff_factor.num) > Int128(MAX_PATTERN_COUNT) * Int128(eff_factor.den):
            n.f = Frac.of(MAX_PATTERN_COUNT)
        else:
            n.f = eff_factor
        return self.add(n)

    def slow(mut self, factor: Frac, child: Int) -> Int:
        if factor.num <= 0:
            return self.silence()
        return self.fast(Frac.make(factor.den, factor.num), child)

    def shift_left(mut self, amount: Frac, child: Int) -> Int:
        var n = node(K_SHIFT)
        n.a = child
        n.f = amount
        return self.add(n)

    def rev(mut self, child: Int) -> Int:
        var n = node(K_REV)
        n.a = child
        return self.add(n)

    def choose(mut self, mode: Int, count: Int, offset: Int, prob: Float64, plain: Int, transformed: Int) -> Int:
        var n = node(K_CHOOSE)
        n.a = plain
        n.b = transformed
        n.c = mode
        n.d = count
        n.e = offset
        n.x = prob
        return self.add(n)

    def degrade(mut self, prob: Float64, seed: Int, child: Int, invert: Bool = False) -> Int:
        var n = node(K_DEGRADE)
        n.a = child
        n.x = prob
        n.c = seed
        n.d = 1 if invert else 0
        return self.add(n)

    def control(mut self, pid: Int, child: Int) -> Int:
        var n = node(K_CONTROL)
        n.a = child
        n.c = pid
        return self.add(n)

    def union(mut self, op: Int, left: Int, right: Int, structure: Int = ST_LEFT) -> Int:
        var n = node(K_UNION)
        n.a = left
        n.b = right
        n.c = op
        n.d = structure
        return self.add(n)

    def chop(mut self, count: Int, child: Int) -> Int:
        var n = node(K_CHOP)
        n.a = child
        n.c = min(MAX_PATTERN_COUNT, max(1, count))
        return self.add(n)

    def arp(mut self, mode: Int, child: Int) -> Int:
        var n = node(K_ARP)
        n.a = child
        n.c = mode
        return self.add(n)

    def window(mut self, lo: Float64, hi: Float64, child: Int, invert: Bool = False) -> Int:
        var n = node(K_WINDOW)
        n.a = child
        n.x = lo
        n.y = hi
        n.d = 1 if invert else 0
        return self.add(n)

    def match(mut self, var want: Params, child: Int, invert: Bool = False) -> Int:
        self.atoms.append(want^)
        var n = node(K_MATCH)
        n.a = child
        n.c = len(self.atoms) - 1
        n.d = 1 if invert else 0
        return self.add(n)

    def linger(mut self, amount: Frac, child: Int) -> Int:
        if amount.num <= 0:
            return child
        var n = node(K_LINGER)
        n.a = child
        if Int128(amount.den) > Int128(MAX_PATTERN_COUNT) * Int128(amount.num):
            n.f = Frac.make(1, MAX_PATTERN_COUNT)
        else:
            n.f = amount
        return self.add(n)

    def slice_values(mut self, count: Int, child: Int) -> Int:
        var n = node(K_SLICEVAL)
        n.a = child
        n.c = min(MAX_PATTERN_COUNT, max(1, count))
        return self.add(n)

    def signal(mut self, shape: Int, seed: Int = 0) -> Int:
        var n = node(K_SIGNAL)
        n.c = shape
        n.d = seed
        return self.add(n)

    def irand(mut self, count: Int) -> Int:
        var n = node(K_IRAND)
        n.c = count
        return self.add(n)

    def scale(mut self, table: Int, child: Int) -> Int:
        var n = node(K_SCALE)
        n.a = child
        n.c = table
        return self.add(n)

    def range(mut self, lo: Float64, hi: Float64, child: Int) -> Int:
        var n = node(K_RANGE)
        n.a = child
        n.x = lo
        n.y = hi
        return self.add(n)

    def ply(mut self, count: Int, child: Int) -> Int:
        var n = node(K_PLY)
        n.a = child
        n.c = min(MAX_PATTERN_COUNT, max(1, count))
        return self.add(n)

    def iter(mut self, count: Int, child: Int, backwards: Bool = False) -> Int:
        var n = node(K_ITER)
        n.a = child
        n.c = min(MAX_PATTERN_COUNT, max(1, count))
        n.d = 1 if backwards else 0
        return self.add(n)

    def segment(mut self, count: Int, child: Int) -> Int:
        var n = node(K_SEGMENT)
        n.a = child
        n.c = min(MAX_PATTERN_COUNT, max(1, count))
        return self.add(n)

    # ---- query

    def query(self, begin: Frac, end: Frac) -> List[Event]:
        var out = List[Event]()
        self.query_into(self.root, begin, end, out)
        return out^

    def query_node(self, id: Int, begin: Frac, end: Frac) -> List[Event]:
        var out = List[Event]()
        self.query_into(id, begin, end, out)
        return out^

    def onsets(self, begin: Frac, end: Frac) -> List[Event]:
        """Events that start inside [begin, end) — what a scheduler plays."""
        var all = self.query(begin, end)
        var out = List[Event]()
        for i in range(len(all)):
            if all[i].has_onset():
                out.append(all[i].copy())
        return out^

    def query_into(self, id: Int, begin: Frac, end: Frac, mut out: List[Event]):
        var budget = MAX_QUERY_EVENTS
        var steps = MAX_QUERY_STEPS
        self._query_into_bounded(id, begin, end, out, budget, steps, 0)

    def _query_into_bounded(
        self,
        id: Int,
        begin: Frac,
        end: Frac,
        mut out: List[Event],
        mut budget: Int,
        mut steps: Int,
        depth: Int,
    ):
        if id < 0 or id >= len(self.nodes) or not (begin < end) or budget <= 0 or steps <= 0 or depth > 64:
            return
        steps -= 1
        var n = self.nodes[id]
        var k = n.kind
        if k == K_SILENCE:
            return
        if k == K_ATOM:
            var c = begin.sam()
            while Frac.of(c) < end and budget > 0 and steps > 0:
                steps -= 1
                budget -= 1
                var wb = Frac.of(c)
                var we = Frac.of(c + 1)
                out.append(Event(wb, we, fmax(begin, wb), fmin(end, we), self.atoms[n.a].copy()))
                c += 1
            return
        if k == K_STACK:
            for i in range(n.kid_count):
                if budget <= 0 or steps <= 0:
                    break
                self._query_into_bounded(self.kids[n.kid_start + i], begin, end, out, budget, steps, depth + 1)
            return
        if k == K_SEQ:
            if n.kid_count <= 0:
                return
            var total = Frac(0, 1)
            for i in range(n.kid_count):
                var wi = self.weights[n.kid_start + i]
                if wi.num > 0:
                    total = total + wi
            if total.num <= 0:
                return
            var c = begin.sam()
            while Frac.of(c) < end and budget > 0 and steps > 0:
                steps -= 1
                var cyc = Frac.of(c)
                var acc = Frac(0, 1)
                for i in range(n.kid_count):
                    if budget <= 0 or steps <= 0:
                        break
                    var w = self.weights[n.kid_start + i]
                    if w.num <= 0:
                        continue
                    var sb = cyc + acc / total
                    var se = cyc + (acc + w) / total
                    acc = acc + w
                    var qb = fmax(begin, sb)
                    var qe = fmin(end, se)
                    if not (qb < qe):
                        continue
                    # compressArc: the slot plays cycle `c` of the child.
                    var span = se - sb
                    var inner = List[Event]()
                    self._query_into_bounded(
                        self.kids[n.kid_start + i], cyc + (qb - sb) / span, cyc + (qe - sb) / span, inner, budget, steps, depth + 1
                    )
                    for j in range(len(inner)):
                        ref ev = inner[j]
                        out.append(Event(
                            sb + (ev.whole_b - cyc) * span, sb + (ev.whole_e - cyc) * span,
                            sb + (ev.part_b - cyc) * span, sb + (ev.part_e - cyc) * span,
                            ev.value.copy()))
                c += 1
            return
        if k == K_ALT:
            var count = n.kid_count
            if count <= 0:
                return
            var c = begin.sam()
            while Frac.of(c) < end and budget > 0 and steps > 0:
                steps -= 1
                var qb = fmax(begin, Frac.of(c))
                var qe = fmin(end, Frac.of(c + 1))
                var idx = ((c % count) + count) % count
                # The child sees its own successive cycles: cycle c//count.
                var offset = Frac.of(c - (c // count))
                var inner = List[Event]()
                self._query_into_bounded(self.kids[n.kid_start + idx], qb - offset, qe - offset, inner, budget, steps, depth + 1)
                for j in range(len(inner)):
                    ref ev = inner[j]
                    out.append(Event(ev.whole_b + offset, ev.whole_e + offset, ev.part_b + offset, ev.part_e + offset, ev.value.copy()))
                c += 1
            return
        if k == K_ALT_ABS:
            var count = n.kid_count
            if count <= 0:
                return
            var c = begin.sam()
            while Frac.of(c) < end and budget > 0 and steps > 0:
                steps -= 1
                var qb = fmax(begin, Frac.of(c))
                var qe = fmin(end, Frac.of(c + 1))
                var idx = ((c % count) + count) % count
                self._query_into_bounded(self.kids[n.kid_start + idx], qb, qe, out, budget, steps, depth + 1)
                c += 1
            return
        if k == K_RANDCAT:
            var count = n.kid_count
            if count <= 0:
                return
            var c = begin.sam()
            while Frac.of(c) < end and budget > 0 and steps > 0:
                steps -= 1
                var qb = fmax(begin, Frac.of(c))
                var qe = fmin(end, Frac.of(c + 1))
                var raw_idx = Int(time_hash(Frac.of(c), 77) * Float64(count))
                var idx = ((raw_idx % count) + count) % count
                self._query_into_bounded(self.kids[n.kid_start + idx], qb, qe, out, budget, steps, depth + 1)
                c += 1
            return
        if k == K_FAST:
            if n.f.num <= 0 or n.f.den <= 0:
                return
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin * n.f, end * n.f, inner, budget, steps, depth + 1)
            for j in range(len(inner)):
                ref ev = inner[j]
                out.append(Event(ev.whole_b / n.f, ev.whole_e / n.f, ev.part_b / n.f, ev.part_e / n.f, ev.value.copy()))
            return
        if k == K_SHIFT:
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin + n.f, end + n.f, inner, budget, steps, depth + 1)
            for j in range(len(inner)):
                ref ev = inner[j]
                out.append(Event(ev.whole_b - n.f, ev.whole_e - n.f, ev.part_b - n.f, ev.part_e - n.f, ev.value.copy()))
            return
        if k == K_REV:
            var c = begin.sam()
            while Frac.of(c) < end and budget > 0 and steps > 0:
                steps -= 1
                var cyc = Frac.of(c)
                var nxt = Frac.of(c + 1)
                var qb = fmax(begin, cyc)
                var qe = fmin(end, nxt)
                var inner = List[Event]()
                self._query_into_bounded(n.a, cyc + (nxt - qe), cyc + (nxt - qb), inner, budget, steps, depth + 1)
                for j in range(len(inner)):
                    ref ev = inner[j]
                    out.append(Event(
                        cyc + (nxt - ev.whole_e), cyc + (nxt - ev.whole_b),
                        cyc + (nxt - ev.part_e), cyc + (nxt - ev.part_b), ev.value.copy()))
                c += 1
            return
        if k == K_CHOOSE:
            var c = begin.sam()
            while Frac.of(c) < end and budget > 0 and steps > 0:
                steps -= 1
                var qb = fmax(begin, Frac.of(c))
                var qe = fmin(end, Frac.of(c + 1))
                var pick: Bool
                if n.c == CH_EVERY:
                    pick = n.d > 0 and (c - n.e) % n.d == 0
                elif n.c == CH_WHENMOD:
                    pick = n.d > 0 and c % n.d >= n.e
                else:
                    pick = time_hash(Frac.of(c), 4242) < n.x
                self._query_into_bounded(n.b if pick else n.a, qb, qe, out, budget, steps, depth + 1)
                c += 1
            return
        if k == K_DEGRADE:
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin, end, inner, budget, steps, depth + 1)
            for j in range(len(inner)):
                var r = time_hash(inner[j].whole_b, n.c)
                var keep = r >= n.x
                if n.d == 1:
                    keep = not keep
                if keep:
                    out.append(inner[j].copy())
            return
        if k == K_CONTROL:
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin, end, inner, budget, steps, depth + 1)
            for j in range(len(inner)):
                ref ev = inner[j]
                var p = Params()
                if not ev.value.raw:
                    p = ev.value.copy()
                elif n.c == P_SOUND:
                    # `s "bd:3"` is sound bd, n 3.
                    var w = ev.value.word
                    var wb = w.as_bytes()
                    var colon = -1
                    for q in range(len(wb)):
                        if wb[q] == UInt8(ord(":")):
                            colon = q
                    if colon >= 0:
                        p.set_sound(String(w[byte=0:colon]))
                        var nn = parse_number(String(w[byte=colon + 1:len(wb)]))
                        if nn[1]:
                            p.set(P_N, nn[0])
                    else:
                        p.set_sound(w)
                elif ev.value.has_num:
                    p.set(n.c, ev.value.num)
                else:
                    continue
                out.append(Event(ev.whole_b, ev.whole_e, ev.part_b, ev.part_e, p^))
            return
        if k == K_UNION:
            if n.d == ST_BOTH:
                # Tidal's appBoth: every pair of overlapping events, wholes intersected.
                var left = List[Event]()
                var right = List[Event]()
                self._query_into_bounded(n.a, begin, end, left, budget, steps, depth + 1)
                self._query_into_bounded(n.b, begin, end, right, budget, steps, depth + 1)
                for j in range(len(left)):
                    if budget <= 0 or steps <= 0:
                        break
                    for q in range(len(right)):
                        if budget <= 0 or steps <= 0:
                            break
                        steps -= 1
                        var wb = fmax(left[j].whole_b, right[q].whole_b)
                        var we = fmin(left[j].whole_e, right[q].whole_e)
                        if not (wb < we):
                            continue
                        var pb = fmax(left[j].part_b, right[q].part_b)
                        var pe = fmin(left[j].part_e, right[q].part_e)
                        if not (pb < pe):
                            continue
                        budget -= 1
                        out.append(Event(wb, we, pb, pe, merge_params(n.c, left[j].value, right[q].value)))
                return
            var structure = n.a if n.d == ST_LEFT else n.b
            var other = n.b if n.d == ST_LEFT else n.a
            var shaped = List[Event]()
            self._query_into_bounded(structure, begin, end, shaped, budget, steps, depth + 1)
            for j in range(len(shaped)):
                if budget <= 0 or steps <= 0:
                    break
                ref ev = shaped[j]
                # Values come from whatever the other side holds at the onset;
                # where it holds nothing the event is dropped (as in Tidal).
                var t = ev.whole_b
                var probe = List[Event]()
                self._query_into_bounded(other, t, t + Frac(1, 1 << 16), probe, budget, steps, depth + 1)
                # Every event the other side holds there counts: `struct "t" $ n "c'maj"`
                # is three notes, as in Tidal.
                for q in range(len(probe)):
                    if budget <= 0:
                        break
                    if not (probe[q].part_b <= t and t < probe[q].part_e):
                        continue
                    budget -= 1
                    if n.d == ST_LEFT:
                        out.append(Event(ev.whole_b, ev.whole_e, ev.part_b, ev.part_e, merge_params(n.c, ev.value, probe[q].value)))
                    else:
                        out.append(Event(ev.whole_b, ev.whole_e, ev.part_b, ev.part_e, merge_params(n.c, probe[q].value, ev.value)))
            return
        if k == K_CHOP:
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin, end, inner, budget, steps, depth + 1)
            for j in range(len(inner)):
                if budget <= 0 or steps <= 0:
                    break
                ref ev = inner[j]
                var step = (ev.whole_e - ev.whole_b) / Frac.of(n.c)
                var b0 = ev.value.get(P_BEGIN, 0.0)
                var e0 = ev.value.get(P_END, 1.0)
                for i in range(n.c):
                    if budget <= 0 or steps <= 0:
                        break
                    steps -= 1
                    var wb = ev.whole_b + step * Frac.of(i)
                    var we = wb + step
                    var pb = fmax(fmax(wb, ev.part_b), begin)
                    var pe = fmin(fmin(we, ev.part_e), end)
                    if not (pb < pe):
                        continue
                    var v = ev.value.copy()
                    if not v.raw:
                        v.set(P_BEGIN, b0 + (e0 - b0) * Float64(i) / Float64(n.c))
                        v.set(P_END, b0 + (e0 - b0) * Float64(i + 1) / Float64(n.c))
                    budget -= 1
                    out.append(Event(wb, we, pb, pe, v^))
            return
        if k == K_ARP:
            # Query whole cycles so each chord is complete, group by whole, spread out.
            var qb = Frac.of(begin.sam())
            var qe = Frac.of(end.sam() + (0 if Frac.of(end.sam()) == end else 1))
            var inner = List[Event]()
            self._query_into_bounded(n.a, qb, qe, inner, budget, steps, depth + 1)
            for i in range(1, len(inner)):  # stable sort on whole
                var j = i
                while j > 0 and (inner[j].whole_b < inner[j - 1].whole_b or (inner[j].whole_b == inner[j - 1].whole_b and inner[j].whole_e < inner[j - 1].whole_e)):
                    inner.swap_elements(j, j - 1)
                    j -= 1
            var g = 0
            while g < len(inner) and budget > 0:
                var h = g
                while h < len(inner) and inner[h].whole_b == inner[g].whole_b and inner[h].whole_e == inner[g].whole_e:
                    h += 1
                var order = arp_order(n.c, h - g)
                var slot = (inner[g].whole_e - inner[g].whole_b) / Frac.of(len(order))
                for r in range(len(order)):
                    if budget <= 0:
                        break
                    ref src = inner[g + order[r]]
                    var wb = src.whole_b + slot * Frac.of(r)
                    var we = wb + slot
                    var pb = fmax(fmax(wb, src.part_b), begin)
                    var pe = fmin(fmin(we, src.part_e), end)
                    if pb < pe:
                        budget -= 1
                        out.append(Event(wb, we, pb, pe, src.value.copy()))
                g = h
            return
        if k == K_WINDOW:
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin, end, inner, budget, steps, depth + 1)
            for j in range(len(inner)):
                var t = inner[j].whole_b
                var pos = (t - Frac.of(t.sam())).to_float()
                var inside = pos >= n.x and pos < n.y
                if inside != (n.d == 1):
                    out.append(inner[j].copy())
            return
        if k == K_MATCH:
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin, end, inner, budget, steps, depth + 1)
            ref want = self.atoms[n.c]
            for j in range(len(inner)):
                ref v = inner[j].value
                var ok = True
                if want.has_sound and (not v.has_sound or v.sound != want.sound):
                    ok = False
                for id in range(P_COUNT):
                    if want.has(id) and (not v.has(id) or v.vals[id] != want.vals[id]):
                        ok = False
                if ok != (n.d == 1):
                    out.append(inner[j].copy())
            return
        if k == K_LINGER:
            var c = begin.sam()
            while Frac.of(c) < end and budget > 0 and steps > 0:
                steps -= 1
                var cyc = Frac.of(c)
                var r = 0
                while cyc + n.f * Frac.of(r) < fmin(end, cyc + Frac.of(1)) and budget > 0 and steps > 0:
                    steps -= 1
                    var sb = cyc + n.f * Frac.of(r)
                    var se = fmin(sb + n.f, cyc + Frac.of(1))
                    var qb = fmax(begin, sb)
                    var qe = fmin(end, se)
                    if qb < qe:
                        var shift = n.f * Frac.of(r)
                        var inner = List[Event]()
                        self._query_into_bounded(n.a, qb - shift, qe - shift, inner, budget, steps, depth + 1)
                        for j in range(len(inner)):
                            ref ev = inner[j]
                            var we = fmin(ev.whole_e + shift, se)
                            var pe = fmin(ev.part_e + shift, se)
                            if ev.part_b + shift < pe:
                                out.append(Event(ev.whole_b + shift, we, ev.part_b + shift, pe, ev.value.copy()))
                    r += 1
                c += 1
            return
        if k == K_SLICEVAL:
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin, end, inner, budget, steps, depth + 1)
            for j in range(len(inner)):
                ref ev = inner[j]
                var i = ev.value.num if ev.value.raw else ev.value.get(P_N, 0.0)
                var p = Params()
                var idx = floor(i)
                p.set(P_BEGIN, idx / Float64(n.c))
                p.set(P_END, (idx + 1.0) / Float64(n.c))
                out.append(Event(ev.whole_b, ev.whole_e, ev.part_b, ev.part_e, p^))
            return
        if k == K_SIGNAL:
            budget -= 1
            out.append(Event(begin, end, begin, end, raw_number(_signal_value(n.c, begin, n.d))))
            return
        if k == K_IRAND:
            budget -= 1
            var v = floor(time_hash(begin, 31) * Float64(n.c))
            out.append(Event(begin, end, begin, end, raw_number(v)))
            return
        if k == K_SCALE:
            var table = _scale_by_index(n.c)
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin, end, inner, budget, steps, depth + 1)
            for j in range(len(inner)):
                var ev = inner[j].copy()
                var deg = ev.value.num if ev.value.raw else ev.value.get(P_N, ev.value.get(P_NOTE, 0.0))
                var steps_sc = len(table)
                var di = Int(floor(deg))
                var octv = di // steps_sc
                var semis = Float64(table[di - octv * steps_sc] + octv * 12) + (deg - Float64(di))
                if ev.value.raw:
                    ev.value = raw_number(semis)
                else:
                    if ev.value.has(P_N):
                        ev.value.set(P_N, semis)
                    if ev.value.has(P_NOTE):
                        ev.value.set(P_NOTE, semis)
                out.append(ev^)
            return
        if k == K_RANGE:
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin, end, inner, budget, steps, depth + 1)
            for j in range(len(inner)):
                var ev = inner[j].copy()
                if ev.value.raw and ev.value.has_num:
                    ev.value = raw_number(n.x + ev.value.num * (n.y - n.x))
                out.append(ev^)
            return
        if k == K_PLY:
            var inner = List[Event]()
            self._query_into_bounded(n.a, begin, end, inner, budget, steps, depth + 1)
            for j in range(len(inner)):
                if budget <= 0 or steps <= 0:
                    break
                ref ev = inner[j]
                var step = (ev.whole_e - ev.whole_b) / Frac.of(n.c)
                for r in range(n.c):
                    if budget <= 0 or steps <= 0:
                        break
                    steps -= 1
                    var wb = ev.whole_b + step * Frac.of(r)
                    var we = wb + step
                    var pb = fmax(wb, ev.part_b)
                    var pe = fmin(we, ev.part_e)
                    if pb < pe:
                        budget -= 1
                        out.append(Event(wb, we, pb, pe, ev.value.copy()))
            return
        if k == K_ITER:
            var c = begin.sam()
            while Frac.of(c) < end and budget > 0 and steps > 0:
                steps -= 1
                var qb = fmax(begin, Frac.of(c))
                var qe = fmin(end, Frac.of(c + 1))
                var step = c % n.c
                if n.d == 1:
                    step = (n.c - step) % n.c
                var amount = Frac.make(step, n.c)
                var inner = List[Event]()
                self._query_into_bounded(n.a, qb + amount, qe + amount, inner, budget, steps, depth + 1)
                for j in range(len(inner)):
                    ref ev = inner[j]
                    out.append(Event(ev.whole_b - amount, ev.whole_e - amount, ev.part_b - amount, ev.part_e - amount, ev.value.copy()))
                c += 1
            return
        if k == K_SEGMENT:
            var step = Frac.make(1, n.c)
            var slot = (begin / step).sam()
            while step * Frac.of(slot) < end and budget > 0 and steps > 0:
                steps -= 1
                var wb = step * Frac.of(slot)
                var we = wb + step
                var probe = List[Event]()
                self._query_into_bounded(n.a, wb, we, probe, budget, steps, depth + 1)
                if len(probe) > 0:
                    var pb = fmax(begin, wb)
                    var pe = fmin(end, we)
                    budget -= 1
                    out.append(Event(wb, we, pb, pe, probe[0].value.copy()))
                slot += 1
            return


def scale_names() -> List[String]:
    return [
        "major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian",
        "majPent", "minPent", "ritusen", "egyptian", "harmonicMinor", "melodicMinor",
        "blues", "whole", "hexMajor7", "chromatic", "ionian", "aeolian", "wholetone",
    ]


def scale_index(name: String) -> Int:
    var names = scale_names()
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


def _scale_by_index(i: Int) -> List[Int]:
    var names = scale_names()
    if i < 0 or i >= len(names):
        return scale_table("chromatic")
    return scale_table(names[i])


# ---------------------------------------------------------------- euclid


def bjorklund(k: Int, n: Int) -> List[Bool]:
    """Bjorklund's algorithm exactly as Tidal implements it."""
    if n <= 0:
        return List[Bool]()
    var nn = min(n, MAX_PATTERN_COUNT)
    var a = List[List[Bool]]()
    var b = List[List[Bool]]()
    var kk = max(0, min(k, nn))
    for _ in range(kk):
        a.append([True])
    for _ in range(nn - kk):
        b.append([False])
    while len(b) > 1 and len(a) > 0:
        var m = min(len(a), len(b))
        var na = List[List[Bool]]()
        for i in range(m):
            var joined = a[i].copy()
            for v in b[i]:
                joined.append(v)
            na.append(joined^)
        var nb = List[List[Bool]]()
        if len(a) > m:
            for i in range(m, len(a)):
                nb.append(a[i].copy())
        else:
            for i in range(m, len(b)):
                nb.append(b[i].copy())
        a = na^
        b = nb^
    var out = List[Bool]()
    for g in a:
        for v in g:
            out.append(v)
    for g in b:
        for v in g:
            out.append(v)
    return out^


# ---------------------------------------------------------------- mini-notation


comptime MAX_MINI_PARSE_DEPTH: Int = 128


struct MiniParser:
    """Recursive-descent parser for Tidal mini-notation into a `Pattern` arena.

    sequence  := group ('.' group)*        (dots split into equal groups)
    group     := step*                     ([...] or | alternatives inside brackets)
    step      := term modifiers*
    term      := word | '~' | '[' stack ']' | '<' stack '>' | '{' poly '}' ('%' n)?
    modifier  := '*' n | '/' n | '!' n? | '@' n | '?' n? | ':' n | '(' k ',' n (',' r)? ')' | '..' n
    """

    var src: List[UInt8]
    var text: String
    var pos: Int
    var error: String
    var seed: Int
    var depth: Int

    def __init__(out self, text: String):
        self.text = text
        self.src = List[UInt8]()
        for b in text.as_bytes():
            self.src.append(b)
        self.pos = 0
        self.error = String()
        self.seed = 0
        self.depth = 0

    def peek(self) -> Int:
        if self.pos < len(self.src):
            return Int(self.src[self.pos])
        return -1

    def skip_ws(mut self):
        while self.pos < len(self.src):
            var c = Int(self.src[self.pos])
            if c == ord(" ") or c == ord("\t") or c == ord("\n") or c == ord("\r"):
                self.pos += 1
            else:
                break

    def fail(mut self, message: String):
        if self.error.byte_length() == 0:
            self.error = message + " at column " + String(self.pos + 1) + " in \"" + self.text + "\""

    @staticmethod
    def is_word_char(c: Int) -> Bool:
        return (
            (c >= ord("a") and c <= ord("z")) or (c >= ord("A") and c <= ord("Z")) or (c >= ord("0") and c <= ord("9"))
            or c == ord("-") or c == ord("#") or c == ord("_") or c == ord("'") or c == ord("^")
        )

    def read_number(mut self) -> Float64:
        self.skip_ws()
        var start = self.pos
        if self.peek() == ord("-"):
            self.pos += 1
        while self.pos < len(self.src):
            var c = Int(self.src[self.pos])
            if (c >= ord("0") and c <= ord("9")) or c == ord("."):
                # stop before a `..` range
                if c == ord(".") and self.pos + 1 < len(self.src) and Int(self.src[self.pos + 1]) == ord("."):
                    break
                self.pos += 1
            else:
                break
        var parsed = parse_number(String(self.text[byte=start:self.pos]))
        if not parsed[1]:
            self.fail("expected a number")
            return 0.0
        return parsed[0]

    def parse(mut self, mut pat: Pattern) -> Int:
        var id = self.parse_stack(pat, -1)
        self.skip_ws()
        if self.pos < len(self.src):
            self.fail("unexpected '" + chr(self.peek()) + "'")
        return id

    def parse_stack(mut self, mut pat: Pattern, close: Int) -> Int:
        """Comma-separated layers (stack) or `|`-separated choices (random)."""
        var layers = List[Int]()
        var mode = 0  # 0 stack, 1 random choice
        while True:
            layers.append(self.parse_sequence(pat, close))
            if self.error.byte_length() > 0:
                break
            self.skip_ws()
            var c = self.peek()
            if c == ord(","):
                self.pos += 1
                continue
            if c == ord("|"):
                self.pos += 1
                mode = 1
                continue
            break
        if mode == 1:
            return pat.randcat(layers)
        return pat.stack(layers)

    def parse_sequence(mut self, mut pat: Pattern, close: Int) -> Int:
        var groups = List[Int]()
        var steps = List[Int]()
        var ws = List[Frac]()
        var dotted = False
        while True:
            self.skip_ws()
            var c = self.peek()
            if c == -1 or c == close or c == ord(",") or c == ord("|") or c == ord("]") or c == ord(">") or c == ord("}"):
                break
            if c == ord(".") and not (self.pos + 1 < len(self.src) and Int(self.src[self.pos + 1]) == ord(".")):
                self.pos += 1
                dotted = True
                groups.append(self._finish_seq(pat, steps, ws))
                steps = List[Int]()
                ws = List[Frac]()
                continue
            if c == ord(".") and len(steps) > 0 and self.pos + 1 < len(self.src) and Int(self.src[self.pos + 1]) == ord("."):
                # `0 .. 7` expands the previous number into a run
                self.pos += 2
                var hi_f = self.read_number()
                if self.error.byte_length() > 0:
                    break
                if hi_f != hi_f or hi_f < -1.0e9 or hi_f > 1.0e9:
                    self.fail("integer range too large (max 4096 elements)")
                    break
                var hi = Int(hi_f)
                var last = steps[len(steps) - 1]
                if pat.nodes[last].kind == K_ATOM and pat.atoms[pat.nodes[last].a].has_num:
                    var lo = Int(pat.atoms[pat.nodes[last].a].num)
                    var diff = Int128(hi) - Int128(lo)
                    if diff < Int128(-4095) or diff > Int128(4095):
                        self.fail("integer range too large (max 4096 elements)")
                        break
                    var w = ws[len(ws) - 1]
                    _ = steps.pop()
                    _ = ws.pop()
                    var step = 1 if hi >= lo else -1
                    var v = lo
                    while True:
                        if len(steps) >= MAX_PATTERN_COUNT:
                            self.fail("too many steps in sequence (max 4096)")
                            break
                        steps.append(pat.pure_number(Float64(v)))
                        ws.append(w)
                        if v == hi:
                            break
                        v += step
                continue
            if c == ord("_"):
                # `a _ _` elongates the previous step.
                self.pos += 1
                if len(ws) > 0:
                    ws[len(ws) - 1] = ws[len(ws) - 1] + Frac(1, 1)
                continue
            if c == ord("!") and len(steps) > 0:
                # bare `!` repeats the previous step
                self.pos += 1
                if len(steps) >= MAX_PATTERN_COUNT:
                    self.fail("too many steps in sequence (max 4096)")
                    break
                steps.append(steps[len(steps) - 1])
                ws.append(ws[len(ws) - 1])
                continue
            var before = self.pos
            self.parse_step(pat, steps, ws)
            if self.error.byte_length() > 0:
                break
            if self.pos == before:
                self.fail("unexpected '" + chr(c) + "'")
                break
        if dotted:
            groups.append(self._finish_seq(pat, steps, ws))
            return pat.seq(groups)
        return self._finish_seq(pat, steps, ws)

    def _finish_seq(mut self, mut pat: Pattern, steps: List[Int], ws: List[Frac]) -> Int:
        if len(steps) == 0:
            return pat.silence()
        var uniform = True
        for w in ws:
            if w != Frac(1, 1):
                uniform = False
        if len(steps) == 1:
            return steps[0]
        if uniform:
            return pat.seq(steps)
        return pat.with_kids(K_SEQ, steps, ws)

    def parse_term(mut self, mut pat: Pattern) -> Int:
        if self.depth >= MAX_MINI_PARSE_DEPTH:
            self.fail("pattern nesting too deep (max " + String(MAX_MINI_PARSE_DEPTH) + ")")
            return pat.silence()
        self.depth += 1
        var res = self._parse_term_inner(pat)
        self.depth -= 1
        return res

    def _parse_term_inner(mut self, mut pat: Pattern) -> Int:
        self.skip_ws()
        var c = self.peek()
        if c == ord("~"):
            self.pos += 1
            return pat.silence()
        if c == ord("["):
            self.pos += 1
            var id = self.parse_stack(pat, ord("]"))
            self.skip_ws()
            if self.peek() != ord("]"):
                self.fail("missing ']'")
            else:
                self.pos += 1
            return id
        if c == ord("<"):
            self.pos += 1
            var layers = List[Int]()
            while True:
                var steps = List[Int]()
                var ws = List[Frac]()
                while True:
                    self.skip_ws()
                    var d = self.peek()
                    if d == -1 or d == ord(">") or d == ord(","):
                        break
                    var before = self.pos
                    self.parse_step(pat, steps, ws)
                    if self.error.byte_length() > 0 or self.pos == before:
                        self.fail("bad step inside <>")
                        return pat.silence()
                # `<a b c>` is slowcat; `<a@3 b>` keeps the weights as cycle counts.
                var alt = List[Int]()
                for i in range(len(steps)):
                    var w = ws[i]
                    var reps = max(1, w.num // w.den)
                    if reps > MAX_PATTERN_COUNT or len(alt) + reps > MAX_PATTERN_COUNT:
                        self.fail("step repetition count too large (max 4096)")
                        return pat.silence()
                    for _ in range(reps):
                        alt.append(steps[i])
                layers.append(pat.slowcat(alt) if len(alt) > 0 else pat.silence())
                self.skip_ws()
                if self.peek() == ord(","):
                    self.pos += 1
                    continue
                break
            if self.peek() != ord(">"):
                self.fail("missing '>'")
            else:
                self.pos += 1
            return pat.stack(layers)
        if c == ord("{"):
            self.pos += 1
            var layer_steps = List[List[Int]]()
            while True:
                var steps = List[Int]()
                var ws = List[Frac]()
                while True:
                    self.skip_ws()
                    var d = self.peek()
                    if d == -1 or d == ord("}") or d == ord(","):
                        break
                    var before = self.pos
                    self.parse_step(pat, steps, ws)
                    if self.error.byte_length() > 0 or self.pos == before:
                        self.fail("bad step inside {}")
                        return pat.silence()
                layer_steps.append(steps^)
                self.skip_ws()
                if self.peek() == ord(","):
                    self.pos += 1
                    continue
                break
            if self.peek() != ord("}"):
                self.fail("missing '}'")
                return pat.silence()
            self.pos += 1
            var base = len(layer_steps[0]) if len(layer_steps) > 0 else 1
            if self.peek() == ord("%"):
                self.pos += 1
                var base_f = self.read_number()
                if base_f != base_f or base_f < 1.0 or base_f > Float64(MAX_PATTERN_COUNT):
                    self.fail("polymeter step count out of range (max 4096)")
                    return pat.silence()
                base = Int(base_f)
            var layers = List[Int]()
            for i in range(len(layer_steps)):
                var count = len(layer_steps[i])
                if count == 0:
                    continue
                var s = pat.seq(layer_steps[i])
                layers.append(pat.fast(Frac.make(base, count), s))
            if len(layers) == 0:
                return pat.silence()
            return pat.stack(layers)
        if MiniParser.is_word_char(c):
            var start = self.pos
            while self.pos < len(self.src):
                var d = Int(self.src[self.pos])
                if MiniParser.is_word_char(d):
                    self.pos += 1
                elif d == ord(".") and self.pos + 1 < len(self.src) and Int(self.src[self.pos + 1]) >= ord("0") and Int(self.src[self.pos + 1]) <= ord("9") and self.pos > start:
                    self.pos += 1  # decimal point
                elif d == ord(":") and self.pos + 1 < len(self.src) and MiniParser.is_word_char(Int(self.src[self.pos + 1])):
                    self.pos += 1  # `bd:3`
                else:
                    break
            if self.pos == start:
                return pat.silence()
            var word = String(self.text[byte=start:self.pos])
            var chord = chord_notes(word)
            if chord[1]:
                var notes = List[Int]()
                for v in chord[0]:
                    notes.append(pat.pure_number(v))
                return pat.stack(notes)
            return pat.atom(raw_atom(word))
        self.fail("unexpected '" + (chr(c) if c >= 0 else "end") + "'")
        return pat.silence()

    def parse_step(mut self, mut pat: Pattern, mut steps: List[Int], mut ws: List[Frac]):
        var id = self.parse_term(pat)
        var weight = Frac(1, 1)
        var repeats = 1
        while self.error.byte_length() == 0:
            var c = self.peek()
            if c == ord("*") or c == ord("/"):
                self.pos += 1
                var fs = self._factors()
                var variants = List[Int]()
                for f in fs:
                    if (
                        (c == ord("*") and Int128(f.num) > Int128(MAX_PATTERN_COUNT) * Int128(f.den))
                        or (c == ord("/") and f.num > 0 and Int128(f.den) > Int128(MAX_PATTERN_COUNT) * Int128(f.num))
                    ):
                        self.fail("speed factor too large (max 4096)")
                        return
                    variants.append(pat.fast(f, id) if c == ord("*") else pat.slow(f, id))
                id = variants[0] if len(variants) == 1 else pat.with_kids(K_ALT_ABS, variants, List[Frac]())
            elif c == ord("@"):
                self.pos += 1
                var w_f = self.read_number()
                if w_f != w_f or w_f < 0.0 or w_f > Float64(MAX_PATTERN_COUNT):
                    self.fail("step weight out of range (max 4096)")
                    return
                weight = Frac.from_float(w_f)
            elif c == ord("!"):
                if self.pos + 1 < len(self.src) and Int(self.src[self.pos + 1]) >= ord("0") and Int(self.src[self.pos + 1]) <= ord("9"):
                    self.pos += 1
                    var r_f = self.read_number()
                    if r_f != r_f or r_f < 0.0 or r_f > Float64(MAX_PATTERN_COUNT):
                        self.fail("repeat count too large (max 4096)")
                        return
                    repeats = Int(r_f)
                else:
                    break
            elif c == ord("?"):
                self.pos += 1
                var prob = 0.5
                var d = self.peek()
                if d >= ord("0") and d <= ord("9") or d == ord("."):
                    prob = self.read_number()
                self.seed += 1
                id = pat.degrade(prob, self.seed, id)
            elif c == ord("("):
                # bd(3,8) / bd(3,8,2); each argument may be `<...>` per cycle
                self.pos += 1
                var ks = self._factors()
                self.skip_ws()
                if self.peek() != ord(","):
                    self.fail("expected ',' in euclid")
                    return
                self.pos += 1
                var ns = self._factors()
                var rots: List[Frac] = [Frac(0, 1)]
                self.skip_ws()
                if self.peek() == ord(","):
                    self.pos += 1
                    rots = self._factors()
                self.skip_ws()
                if self.peek() != ord(")"):
                    self.fail("expected ')' in euclid")
                    return
                self.pos += 1
                var count = len(ks)
                while count % len(ns) != 0 or count % len(rots) != 0:
                    count += len(ks)
                    if count > MAX_PATTERN_COUNT:
                        self.fail("euclid cycle count too large (max 4096)")
                        return
                var variants = List[Int]()
                for v in range(count):
                    var k = ks[v % len(ks)].sam()
                    var n = ns[v % len(ns)].sam()
                    var rot = rots[v % len(rots)].sam()
                    var rest = pat.silence()
                    if n <= 0:
                        variants.append(rest)
                        continue
                    if n > MAX_PATTERN_COUNT:
                        self.fail("euclid step count too large (max 4096)")
                        return
                    var bits = bjorklund(k if k >= 0 else -k, n)
                    var slots = List[Int]()
                    for i in range(n):
                        var bit = bits[((i + rot) % n + n) % n]
                        if k < 0:
                            bit = not bit
                        slots.append(id if bit else rest)
                    variants.append(pat.seq(slots))
                id = variants[0] if len(variants) == 1 else pat.with_kids(K_ALT_ABS, variants, List[Frac]())
            else:
                break
        if self.error.byte_length() > 0:
            return
        var add_reps = max(1, repeats)
        if len(steps) + add_reps > MAX_PATTERN_COUNT:
            self.fail("too many steps in sequence (max 4096)")
            return
        for _ in range(add_reps):
            steps.append(id)
            ws.append(weight)

    def _factors(mut self) -> List[Frac]:
        """`*2`, `*1.5`, or a per-cycle choice `*<2 3 4>`."""
        self.skip_ws()
        var out = List[Frac]()
        if self.peek() == ord("<"):
            self.pos += 1
            while True:
                self.skip_ws()
                var c = self.peek()
                if c == ord(">"):
                    self.pos += 1
                    break
                if c == -1:
                    self.fail("missing '>'")
                    break
                out.append(Frac.from_float(self.read_number()))
                if self.error.byte_length() > 0:
                    break
        else:
            out.append(Frac.from_float(self.read_number()))
        if len(out) == 0:
            out.append(Frac(1, 1))
        return out^


def parse_mini(text: String, mut pat: Pattern) raises -> Int:
    """Parse mini-notation into `pat` and return the new node id."""
    var parser = MiniParser(text)
    var id = parser.parse(pat)
    if parser.error.byte_length() > 0:
        raise Error(parser.error)
    return id


def mini(text: String) raises -> Pattern:
    """Convenience: a standalone pattern from mini-notation."""
    var pat = Pattern()
    pat.root = parse_mini(text, pat)
    return pat^
