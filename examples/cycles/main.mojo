"""`flint-cycles` — pure-Mojo TidalCycles-style live coding.

Usage:
  flint-cycles live   <song.tidal>                 play, and re-evaluate on every save
  flint-cycles render <song.tidal> <out.wav> [sec]  offline render (default 16 s)
  flint-cycles query  '<pattern expr>' [cycles]     print the events of an expression

Live workflow: open the .tidal file in any editor, run `flint-cycles live`,
edit and save. The clock never resets; a file with errors is reported and the
last good version keeps playing.
"""
from std.sys import argv
from std.ffi import external_call
from flint.cycles import CyclesEngine, render_to_wav, render_loop_to_wav, eval_pattern, Frac
from flint.cycles.pattern import parse_number
from flint.platform.os import read_bytes, file_stamp, get_time
from flint.platform.pulse import PulseClient

comptime RATE: Int = 48000
comptime BLOCK: Int = 512


def read_text(path: String) -> String:
    var bytes = read_bytes(path)
    return String(unsafe_from_utf8=bytes)


def usage():
    print("usage:")
    print("  flint-cycles live   <song.tidal>")
    print("  flint-cycles render <song.tidal> <out.wav> [seconds]")
    print("  flint-cycles render-loop <song.tidal> <out.wav> <cycles>   seamless game-music loop")
    print("  flint-cycles query  '<pattern expr>' [cycles]")


def report(errors: List[String]):
    for e in errors:
        print("  [error] " + e)


def cmd_live(path: String) raises:
    var engine = CyclesEngine(RATE)
    var stamp = file_stamp(path)
    if stamp[0] < 0:
        raise Error("cannot read " + path)
    var errs = engine.load(read_text(path))
    if len(errs) > 0:
        report(errs)
    print("[cycles] " + path + ": " + String(len(engine.program.tracks)) + " track(s), cps " + String(engine.cps))
    var pulse = PulseClient()
    if not pulse.connect(RATE, BLOCK * 4):
        raise Error("no sound server (" + pulse.error + "); try `flint-cycles render` instead")
    print("[cycles] playing — edit and save the file to change the music, Ctrl+C to stop")
    var pending = List[Int16]()
    var last_check = get_time()
    var last_cycle = -1
    while True:
        pulse.pump()
        if not pulse.connected():
            raise Error("sound server went away: " + pulse.error)
        while pulse.requested_bytes >= BLOCK * 4:
            if len(pending) == 0:
                pending = engine.render_i16(BLOCK)
            if not pulse.write_samples(pending):
                break
            pending.clear()
        var now = get_time()
        if now - last_check > 0.15:
            last_check = now
            var s = file_stamp(path)
            if s[0] >= 0 and (s[0] != stamp[0] or s[1] != stamp[1]):
                stamp = s
                var e = engine.load(read_text(path))
                if len(e) > 0:
                    print("[cycles] reload failed, still playing the last good version:")
                    report(e)
                else:
                    print("[cycles] reloaded: " + String(len(engine.program.tracks)) + " track(s), cps " + String(engine.cps))
            var cyc = engine.now().sam()
            if cyc != last_cycle:
                last_cycle = cyc
                print("  cycle " + String(cyc) + "  voices " + String(len(engine.voices)))
        _ = external_call["usleep", Int32](UInt32(2000))


def main() raises:
    var args = argv()
    if len(args) < 3:
        usage()
        return
    var cmd = String(args[1])
    if cmd == "live" or cmd == "play":
        cmd_live(String(args[2]))
    elif cmd == "render":
        if len(args) < 4:
            usage()
            return
        var seconds = 16.0
        if len(args) >= 5:
            seconds = parse_number(String(args[4]))[0]
        var count = render_to_wav(read_text(String(args[2])), seconds, String(args[3]), RATE)
        print("[cycles] wrote " + String(args[3]) + " (" + String(seconds) + " s, " + String(count) + " events)")
    elif cmd == "render-loop":
        if len(args) < 5:
            usage()
            return
        var cycles = Int(parse_number(String(args[4]))[0])
        var frames = render_loop_to_wav(read_text(String(args[2])), cycles, String(args[3]), RATE)
        print("[cycles] wrote seamless loop " + String(args[3]) + " (" + String(cycles) + " cycles, "
            + String(Float64(frames) / Float64(RATE)) + " s)")
    elif cmd == "query":
        var cycles = 1
        if len(args) >= 4:
            cycles = Int(parse_number(String(args[3]))[0])
        var prog = eval_pattern(String(args[2]))
        var evs = prog.pat.query_node(prog.tracks[0].root, Frac.of(0), Frac.of(cycles))
        for i in range(len(evs)):
            if evs[i].has_onset():
                print(evs[i].describe())
    else:
        usage()
