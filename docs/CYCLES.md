# flint.cycles — pure-Mojo live coding in the TidalCycles style

`flint.cycles` is a native Mojo pattern engine, a small Tidal-syntax interpreter
and a synth/sampler, all built on Flint's own audio stack (`flint.platform.dsp`,
`flint.platform.audio`, `flint.platform.pulse`). It has no Haskell, no
SuperCollider and no C libraries.

```bash
bash tools/flint.sh cycles live examples/cycles/demo.tidal       # play; edit + save to change it
bash tools/flint.sh cycles render examples/cycles/demo.tidal out.wav 30
bash tools/flint.sh cycles render-loop "$PWD/song.tidal" music.wav 32   # seamless game loop
bash tools/flint.sh cycles query 'every 2 (fast 2) $ s "bd [~ sn]"' 2
```

In **live** mode the file is re-evaluated on every save. The clock keeps running
through a reload. If a save has an error, the error is printed and the last good
version keeps playing. Every `dN` in the file is the full current state, so
deleting a line silences that track, and `hush` silences them all.

## Mini-notation (inside `"..."`)

| syntax | meaning |
|---|---|
| `bd sn` | sequence: steps share the cycle |
| `[bd sn] hh` | subsequence in one step |
| `bd . hh hh hh` | `.` splits the cycle into equal groups |
| `~` | rest |
| `bd*2` `bd/2` `bd*<2 3>` | faster / slower (per-cycle factor choice) |
| `<a b c>` | one per cycle (nests: `<a <b c>>`) |
| `[a, b c]` | stack layers |
| `{a b c}%4` `{a b, c d e}` | polymeter |
| `a@3 b` `a _ _ b` | weight / elongate |
| `a!3` `a ! !` | replicate |
| `a?` `a?0.3` | random drop (deterministic per event time) |
| `bd(3,8)` `bd(3,8,2)` | Euclidean rhythm (with rotation) |
| `bd:3` | sample/variant 3 (`s` + `n`) |
| `0 .. 7` | numeric run |
| `c e g` `cs4` `ef` | note names (`c5` = 0) |
| `[a | b | c]` | random choice per cycle |
| `c'maj e'min7` | chords (stacked notes); modifiers `'i` inversion, `'o` open, `'d` drop, `'5` number of notes: `c'maj'ii'5` |

## Functions

Controls: `s`/`sound`, `n`, `note`/`up`, `gain`, `pan`, `speed`, `cutoff`/`lpf`,
`resonance`, `hcutoff`/`hpf`, `room`, `size`, `shape`, `crush`, `coarse`,
`legato`, `sustain`, `attack`, `release`, `delay`, `delaytime`, `delayfeedback`,
`begin`, `end`.

Time: `fast` `slow` `hurry` `rev` `palindrome` `iter` `iter'` `ply` `<~` `~>`
`rotL` `rotR` `inside` `outside` `brak` `fastGap` `segment` `linger` `swingBy`
`swing`.

Parts of a cycle: `within (0, 0.5) f`, `within'` (f sees the window as a whole
cycle), `chunk n f`, `chunk' n f`.

Sample slicing: `chop n` (each event cut into n parts of its sample), `striate n`
(interleaved slices), `striateBy n width`, `slice n "0 2"`, `splice`,
`randslice n`. These set `begin`/`end`, so they only change the sound of
sample-based voices. Synth voices are just retriggered.

Chords and arpeggios: `arp "<up down>"` with the modes `up` `down` `updown`
`downup` `up&down` `down&up` `converge` `diverge` `disconverge` `pinkyup`
`pinkyupdown` `thumbup` `thumbupdown`.

Selecting events: `fix f (s "hh")` applies `f` only to matching events;
`unfix` applies it to the others.

Conditionals and randomness: `every` `every'` `whenmod` `sometimesBy`
`sometimes` `often` `rarely` `almostNever` `almostAlways` `always` `never`
`someCyclesBy` `somecycles` `degrade` `degradeBy` `unDegradeBy`.

Layering: `stack` `overlay` `cat`/`slowcat` `fastcat` `randcat` `jux`
`superimpose` `off` `euclid` `euclidInv` `euclidOff` `struct`.

Values: `scale` (major minor dorian phrygian lydian mixolydian locrian majPent
minPent ritusen egyptian harmonicMinor melodicMinor blues whole hexMajor7
chromatic), `range`, `run`, `irand`, plus the signals `sine` `cosine` `saw` `isaw`
`tri` `square` `rand` `perlin`.

Operators: `#`, `|<`, `|+| |+ +| +`, the matching `-` `*` `/` and `|%` forms,
`$`, `.` (compose), sections like `(# speed 2)`, `(|+ n 12)` and `(0.25 <~)`,
lists `[a, b]`, and number arithmetic `(130/60/4)`.

Statements: `dN $ ...` (d1–d16), `setcps x`, `setbpm x` (4 beats per cycle),
`hush`, and `--` comments. Layout follows Haskell. A statement continues on
deeper-indented lines and on lines that start with an operator. A `do` block
runs each line aligned with its first line as a statement.

Definitions:

```haskell
let pat = "bd*2 [~ sn]"
    fx  = (# room 0.3) . (|* speed 1.5)   -- several bindings, aligned
let twice f = f . f                       -- functions with arguments
do
  d1 $ s pat
  d2 $ twice (fast 2) $ fx $ s "hh"
  d3 $ (\x -> x # gain 0.8) $ s "cp*2"   -- lambdas
```

### Structure

As in Tidal, which side of an operator sets the rhythm depends on the bars:
`|+` (and `#`, which is `|>`) keeps the left side's events, `+|` keeps the right
side's, and `|+|` or a bare `+` keeps both, intersecting where they overlap.
With `#` and `|op`, an event is dropped where the other side is silent:
`s "a*4" # n "1 ~"` plays two.

### Not implemented yet

A string passed to a numeric argument (`fast "<1 2>"`) is read once per cycle,
with up to 24 cycles of period. `once`, `solo`, transitions (`xfade` and
friends), `nudge`, `vowel`, `loopAt`, `rot`, `squeeze`/`select`, `where`,
`case`/`if` and type annotations are not supported yet. Using one reports an
error for that statement only.

## Sounds

A sound is looked up in this order:

1. A sample folder `<root>/<s>/*.wav`, laid out like SuperDirt (`n` picks the
   file). `<root>` is `$FLINT_CYCLES_SAMPLES`, otherwise SuperCollider's
   `Dirt-Samples` quark if it is installed, otherwise `./samples`. If you already
   have Dirt-Samples, your usual sample names just work. The WAV files must be
   16-bit PCM.
2. A built-in synthesized voice:
   - drums: `bd` `sn` `hh` `oh` `cp` `cb` `lt`/`mt`/`ht` `rim` `cr`
   - pitched: `sine` `saw` `square` `tri` `supersaw` `bass` (filter-swept saw),
     `pluck`/`arpy` (Karplus-Strong), `pad`, `fm`/`superpiano`, `bell`/`gong`
     (modal), `timpani` (modal membrane), `noise`

   SuperDirt aliases such as `superpiano`, `supersquare` and `808bd` map to the
   nearest built-in voice. For pitched voices, `n` adds to `note`.

Each event can have its own `shape`, `crush`, `coarse` and SVF `cutoff`/`hcutoff`.
`room` sends to Flint's FDN reverb, and `delay` sends to a ping-pong delay. The
master bus goes through a soft clipper.

## Game music loops

`render-loop <song> <out.wav> <cycles>` renders exactly `cycles` cycles. It
renders one pass first and throws it away, so the reverb and delay tails that
cross the loop point are already in the captured audio and the WAV repeats
without a seam. Write the arrangement to repeat every `cycles`. For example,
gate sections with `whenmod` and `const`:

```haskell
let section a b p = whenmod 32 a (const (whenmod 32 b (const silence) p)) silence
d1 $ section 8 16 $ s "bd*4"   -- only in cycles 8..15 of every 32
```

Then load the WAV like any other `WavClip`, for example with
`NativeAudio.set_music_clip`.

## Embedding in a game

```mojo
from flint.cycles import CyclesEngine
var music = CyclesEngine(48000)
_ = music.load('setcps 0.55\nd1 $ s "bd*2 [~ sn]" # room 0.2')
var stereo = music.render(512)   # interleaved Float32, mix into your bus
```

The source is in `src/flint/cycles/`: `pattern.mojo` (rational time, combinators,
mini-notation), `script.mojo` (the interpreter) and `synth.mojo` (voices, mixer,
clock, WAV output).
