"""`flint.cycles` — a pure-Mojo live-coding pattern engine in the TidalCycles style.

    from flint.cycles import CyclesEngine
    var engine = CyclesEngine(48000)
    _ = engine.load('d1 $ s "bd*2 [~ sn]" # room 0.2')
    var stereo = engine.render(512)

See `docs/CYCLES.md` for the language reference.
"""
from flint.cycles.pattern import Frac, Pattern, Params, Event, mini, parse_mini, bjorklund
from flint.cycles.script import Program, run_program, eval_pattern, split_statements
from flint.cycles.synth import CyclesEngine, render_to_wav, render_loop_to_wav, encode_wav16, voice_kind
