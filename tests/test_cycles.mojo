"""Tests for `flint.cycles` (pattern engine, mini-notation, script, synth)."""
from std.testing import assert_equal, assert_true, assert_false
from flint.cycles import Frac, mini, run_program, eval_pattern, CyclesEngine, bjorklund, split_statements
from flint.cycles.pattern import Pattern, Event, P_N, P_GAIN, P_PAN, P_SPEED


def by_time(var evs: List[Event]) -> List[Event]:
    for i in range(1, len(evs)):
        var j = i
        while j > 0 and evs[j].whole_b < evs[j - 1].whole_b:
            evs.swap_elements(j, j - 1)
            j -= 1
    return evs^


def onsets(src: String, cycles: Int = 1) raises -> List[Event]:
    var p = mini(src)
    return by_time(p.onsets(Frac.of(0), Frac.of(cycles)))


def expr_onsets(src: String, cycles: Int = 1) raises -> List[Event]:
    var prog = eval_pattern(src)
    var all = prog.pat.query_node(prog.tracks[0].root, Frac.of(0), Frac.of(cycles))
    var out = List[Event]()
    for i in range(len(all)):
        if all[i].has_onset():
            out.append(all[i].copy())
    return by_time(out^)


def starts(evs: List[Event]) -> String:
    var s = String()
    for i in range(len(evs)):
        if i > 0:
            s += " "
        s += evs[i].whole_b.to_string()
    return s


def words(evs: List[Event]) -> String:
    var s = String()
    for i in range(len(evs)):
        if i > 0:
            s += " "
        s += evs[i].value.describe()
    return s


def test_frac() raises:
    var a = Frac.make(2, 4)
    assert_equal(a.to_string(), "1/2")
    assert_equal((Frac.make(1, 3) + Frac.make(1, 6)).to_string(), "1/2")
    assert_equal(Frac.make(-1, 3).sam(), -1)
    assert_true(Frac.make(1, 3) < Frac.make(1, 2))
    assert_equal(Frac.from_float(0.125).to_string(), "1/8")
    assert_equal(Frac.from_float(130.0 / 60.0 / 4.0).to_string(), "13/24")
    # large denominators stay exact through 128-bit intermediates
    var big = Frac.make(1, 1 << 40) + Frac.make(1, 3)
    assert_true(big > Frac.make(1, 3))


def test_sequences() raises:
    assert_equal(starts(onsets("bd sn")), "0 1/2")
    assert_equal(starts(onsets("bd*2 [~ sn]")), "0 1/4 3/4")
    assert_equal(words(onsets("<a b c>", 4)), "a b c a")
    assert_equal(words(onsets("<a <b c>>", 4)), "a b a c")
    assert_equal(starts(onsets("a@3 b")), "0 3/4")
    assert_equal(starts(onsets("a!3 b")), "0 1/4 1/2 3/4")
    assert_equal(starts(onsets("a ! b")), "0 1/3 2/3")
    assert_equal(starts(onsets("a _ _ b")), "0 3/4")
    assert_equal(starts(onsets("a b . c d e")), "0 1/4 1/2 2/3 5/6")
    assert_equal(len(onsets("[a, b c]")), 3)
    assert_equal(words(onsets("0 .. 3")), "0 1 2 3")
    assert_equal(starts(onsets("a/2", 2)), "0")
    assert_equal(starts(onsets("a*<2 3>", 2)), "0 1/2 1 4/3 5/3")


def test_euclid_polymeter() raises:
    assert_equal(starts(onsets("bd(3,8)")), "0 3/8 3/4")
    assert_equal(starts(onsets("bd(3,8,2)")), "1/8 1/2 3/4")
    assert_equal(starts(onsets("bd(<3 5>,8)", 2)), "0 3/8 3/4 1 5/4 11/8 13/8 7/4")
    assert_equal(starts(onsets("bd(3,8,<0 2>)", 2)), "0 3/8 3/4 9/8 3/2 7/4")
    var b = bjorklund(5, 8)
    var s = String()
    for v in b:
        s += "x" if v else "."
    assert_equal(s, "x.xx.xx.")
    assert_equal(words(onsets("{a b c}%4", 1)), "a b c a")
    assert_equal(len(onsets("{a b, c d e}")), 4)


def test_degrade_is_deterministic() raises:
    var a = onsets("hh*16?", 4)
    var b = onsets("hh*16?", 4)
    assert_equal(starts(a), starts(b))
    assert_true(len(a) > 16 and len(a) < 48)


def test_script_transforms() raises:
    assert_equal(starts(expr_onsets("fast 2 $ s \"bd sn\"")), "0 1/4 1/2 3/4")
    assert_equal(starts(expr_onsets("rev $ s \"a b ~ ~\"")), "1/2 3/4")
    assert_equal(words(expr_onsets("rev $ s \"a b\"")), "s=b s=a")
    # every 2 applies on cycle 0, 2, ...
    assert_equal(len(expr_onsets("every 2 (fast 2) $ s \"bd\"", 2)), 3)
    assert_equal(starts(expr_onsets("0.25 <~ s \"a b c d\"")), "0 1/4 1/2 3/4")
    assert_equal(words(expr_onsets("0.25 <~ s \"a b c d\"")), "s=b s=c s=d s=a")
    assert_equal(starts(expr_onsets("(1/4) ~> s \"a\"", 2)), "1/4 5/4")
    assert_equal(words(expr_onsets("s \"bd:3\"")), "s=bd n=3")
    assert_equal(words(expr_onsets("n \"0 1\" # s \"x\" # gain 1.2")), "s=x n=0 gain=1.2 s=x n=1 gain=1.2")
    assert_equal(words(expr_onsets("n \"0 2\" + n \"<12 7>\"", 2)), "n=12 n=14 n=7 n=9")
    assert_equal(words(expr_onsets("n (\"0 2\" + \"10\")")), "n=10 n=12")
    assert_equal(words(expr_onsets("n (scale \"major\" \"0 2 7\")")), "n=0 n=4 n=12")
    assert_equal(words(expr_onsets("s \"a*2\" # speed (range 1 3 saw)")), "s=a speed=1 s=a speed=2")
    var jux = expr_onsets("jux rev $ s \"a b\"")
    assert_equal(len(jux), 4)
    assert_equal(len(expr_onsets("off 0.125 (# crush 4) $ s \"a\"", 2)), 4)
    assert_equal(len(expr_onsets("stack [s \"a\", s \"b*2\"]")), 3)
    assert_equal(words(expr_onsets("cat [s \"a\", s \"b\"]", 2)), "s=a s=b")
    assert_equal(len(expr_onsets("whenmod 4 2 (fast 2) $ s \"a\"", 4)), 6)
    assert_equal(len(expr_onsets("ply 3 $ s \"a b\"")), 6)
    assert_equal(words(expr_onsets("iter 4 $ s \"a b c d\"", 2)), "s=a s=b s=c s=d s=b s=c s=d s=a")
    assert_equal(len(expr_onsets("every 2 (fast 2 . rev) $ s \"a b\"", 2)), 6)
    assert_equal(words(expr_onsets("sometimesBy 1 (# speed 2) $ s \"a\"")), "s=a speed=2")
    assert_equal(len(expr_onsets("euclid 3 8 $ s \"bd\"")), 3)
    assert_equal(len(expr_onsets("fast \"<1 2>\" $ s \"a\"", 2)), 3)


def test_program() raises:
    var src = String("""
-- comment
setcps (120/60/4)
d1 $ s "bd*4"
d2 $ n "0 3"
   # s "saw"   -- continuation line
d3 $ nonsense "x"
""")
    var prog = run_program(src)
    assert_equal(len(prog.tracks), 2)
    assert_equal(len(prog.errors), 1)
    assert_true(prog.cps > 0.4999 and prog.cps < 0.5001)
    assert_equal(len(split_statements("d1 $ s \"a\"\n  # n 1\n\nhush")), 2)
    var hushed = run_program("d1 $ s \"bd\"\nhush")
    assert_equal(len(hushed.tracks), 0)


def test_engine_renders_audio() raises:
    var engine = CyclesEngine(48000)
    var errs = engine.load("setcps 1\nd1 $ s \"bd sn hh cp\" # room 0.2\nd2 $ n \"0 7\" # s \"supersaw\" # cutoff 2000")
    assert_equal(len(errs), 0)
    var peak = Float32(0)
    for _ in range(93):  # just under 1 s = one cycle
        var block = engine.render(512)
        for x in block:
            assert_true(x == x)  # no NaN
            peak = max(peak, abs(x))
    assert_true(peak > 0.05)
    assert_true(peak <= 1.0)
    assert_equal(engine.events_played, 6)
    # A broken reload keeps the last good program playing.
    var bad = engine.load("d1 $ s \"bd\" #")
    assert_true(len(bad) > 0)
    assert_equal(len(engine.program.tracks), 2)


def test_clock_is_continuous_across_blocks() raises:
    # 37-frame blocks at a cps that does not divide the rate: each onset fires once.
    var engine = CyclesEngine(48000)
    _ = engine.load("setcps 0.7\nd1 $ s \"hh*16\"")
    var frames = 0
    while frames < 48000 * 3:
        _ = engine.render(37)
        frames += 37
    var cycles = engine.now().to_float()
    var expected = Int(cycles * 16.0) + 1
    assert_true(engine.events_played == expected or engine.events_played == expected - 1)


def test_structure_rules() raises:
    # `|+` left, `+|` right, `|+|` and bare `+` both (Tidal semantics)
    assert_equal(words(expr_onsets("n \"0 1\" |+ n \"10 20 30\"")), "n=10 n=21")
    assert_equal(words(expr_onsets("n \"0 1\" +| n \"10 20 30\"")), "n=10 n=20 n=31")
    assert_equal(starts(expr_onsets("n \"0 1\" |+| n \"10 20 30\"")), "0 1/3 1/2 2/3")
    assert_equal(starts(expr_onsets("s \"a*4\" + n \"1 2\"")), "0 1/4 1/2 3/4")
    # `#` drops events where the right side is silent
    assert_equal(len(expr_onsets("s \"a*4\" # n \"1 ~\"")), 2)
    assert_equal(words(expr_onsets("n \"1 2\" |< n \"5\"")), "n=1 n=2")
    # every stacked value on the other side is kept (chords through `#` / struct)
    assert_equal(words(expr_onsets("struct \"~ t\" $ n \"c'maj\"")), "n=0 n=4 n=7")
    assert_equal(len(expr_onsets("s \"x*2\" # n \"[0,4]\"")), 4)


def test_chop_striate_slice() raises:
    assert_equal(words(expr_onsets("chop 2 $ s \"x\"")), "s=x begin=0 end=0.5 s=x begin=0.5 end=1")
    assert_equal(words(expr_onsets("striate 2 $ s \"a b\"")),
        "s=a begin=0 end=0.5 s=b begin=0 end=0.5 s=a begin=0.5 end=1 s=b begin=0.5 end=1")
    assert_equal(words(expr_onsets("slice 4 \"0 2\" $ s \"bev\"")), "s=bev begin=0 end=0.25 s=bev begin=0.5 end=0.75")


def test_arp_and_chords() raises:
    assert_equal(words(onsets("c'maj")), "0 4 7")
    assert_equal(words(expr_onsets("n \"ef3 eb3 bf\"")), "n=-21 n=-21 n=10")
    assert_equal(words(onsets("e'min7")), "4 7 11 14")
    assert_equal(words(onsets("c'maj'ii")), "7 12 16")
    assert_equal(words(onsets("c'maj'5")), "0 4 7 12 16")
    assert_equal(words(onsets("0'major'o")), "-12 -5 4")
    assert_equal(words(onsets("c'maj'd")), "-5 0 4")
    assert_equal(words(expr_onsets("arp \"up\" $ n \"c'maj\"")), "n=0 n=4 n=7")
    assert_equal(words(expr_onsets("arp \"down\" $ n \"c'maj\"")), "n=7 n=4 n=0")
    assert_equal(words(expr_onsets("arp \"updown\" $ n \"c'maj7\"")), "n=0 n=4 n=7 n=11 n=7 n=4")
    assert_equal(words(expr_onsets("arp \"converge\" $ n \"[0,1,2,3]\"")), "n=0 n=3 n=1 n=2")
    assert_equal(words(expr_onsets("arp \"thumbup\" $ n \"[0,1,2]\"")), "n=0 n=1 n=0 n=2")
    assert_equal(words(expr_onsets("arp \"pinkyup\" $ n \"[0,1,2]\"")), "n=0 n=2 n=1 n=2")
    assert_equal(words(expr_onsets("arp \"<up down>\" $ n \"[0,1]\"", 2)), "n=0 n=1 n=1 n=0")


def test_within_chunk_fix_linger_swing() raises:
    assert_equal(words(expr_onsets("within (0, 0.5) (fast 2) $ s \"x y\"")), "s=x s=y s=y")
    assert_equal(starts(expr_onsets("chunk 4 (hurry 2) $ s \"a b c d\"")), "0 1/8 1/4 1/2 3/4")
    assert_equal(starts(expr_onsets("chunk 4 (# speed 2) $ s \"a b c d\"", 2)), "0 1/4 1/2 3/4 1 5/4 3/2 7/4")
    var ch = expr_onsets("chunk 4 (# speed 2) $ s \"a b c d\"", 2)
    assert_true(ch[0].value.has(P_SPEED) and not ch[1].value.has(P_SPEED))
    assert_true(ch[5].value.has(P_SPEED) and not ch[4].value.has(P_SPEED))
    assert_equal(words(expr_onsets("fix (# crush 4) (s \"hh\") $ s \"bd hh\"")), "s=bd s=hh crush=4")
    assert_equal(words(expr_onsets("unfix (# crush 4) (s \"hh\") $ s \"bd hh\"")), "s=bd crush=4 s=hh")
    assert_equal(words(expr_onsets("linger 0.25 $ n \"0 1 2 3\"")), "n=0 n=0 n=0 n=0")
    assert_equal(words(expr_onsets("linger (-0.25) $ n \"0 1 2 3\"")), "n=3 n=3 n=3 n=3")
    assert_equal(starts(expr_onsets("swingBy (1/3) 4 $ s \"hh*8\"")), "0 5/24 1/4 11/24 1/2 17/24 3/4 23/24")


def test_let_do_lambda() raises:
    var src = String("""
let pat = "bd sn"
    fx = (# speed 2)
let twice f = f . f
do
  d1 $ s pat
  d2 $ twice (fast 2) $ s "hh"
      # pan 0.2
  d3 $ fx $ s "cp"
d4 $ (\\x -> x # gain 0.5) $ s "lam"
do d5 $ s "one"
   d6 $ s "two"
""")
    var prog = run_program(src)
    assert_equal(len(prog.errors), 0)
    assert_equal(len(prog.tracks), 6)
    var d2 = prog.pat.onsets(Frac.of(0), Frac.of(1))
    _ = d2
    for t in prog.tracks:
        var evs = prog.pat.query_node(t.root, Frac.of(0), Frac.of(1))
        if t.name == "d2":
            assert_equal(len(evs), 4)
            assert_true(evs[0].value.has(P_PAN))
        if t.name == "d3":
            assert_true(evs[0].value.has(P_SPEED))
        if t.name == "d4":
            assert_true(evs[0].value.has(P_GAIN))


def main() raises:
    test_frac()
    test_sequences()
    test_euclid_polymeter()
    test_degrade_is_deterministic()
    test_script_transforms()
    test_program()
    test_engine_renders_audio()
    test_clock_is_continuous_across_blocks()
    test_structure_rules()
    test_chop_striate_slice()
    test_arp_and_chords()
    test_within_chunk_fix_linger_swing()
    test_let_do_lambda()
    print("[cycles] all tests passed")
