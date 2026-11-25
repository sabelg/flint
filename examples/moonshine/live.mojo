"""Live voice commands from the microphone with Moonshine.

usage: mojo run -I src examples/moonshine/live.mojo MODEL_DIR [--cpu] [--seconds N]

Captures 16 kHz mono from the desktop sound server (PulseAudio / PipeWire native protocol,
`flint.platform.pulse_record`), cuts utterances with `EnergyVad`, prints a partial transcript
about every 0.5 s while someone is talking and the final one when they stop, then matches it
against a small command list (`VoiceCommands`). Audio stays in memory and is never written out.
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.ml.moonshine import Moonshine, VoiceCommands
from flint.ml.moonshine_audio import EnergyVad
from flint.platform.pulse_record import PulseRecorder


def main() raises:
    var a = argv()
    if len(a) < 2:
        print("usage: live MODEL_DIR [--cpu] [--seconds N]")
        return
    var gpu = True
    var seconds = 60
    var i = 2
    while i < len(a):
        var s = String(a[i])
        if s == "--cpu":
            gpu = False
        elif s == "--seconds" and i + 1 < len(a):
            seconds = Int(String(a[i + 1]))
            i += 1
        i += 1
    var m = Moonshine.load(String(a[1]), gpu)
    var warm = List[Float32](length=16000, fill=0.0)
    _ = m.transcribe(warm)   # uploads weights before the first utterance
    var cmds = VoiceCommands()
    for p in ["open the gate", "close the gate", "light the torches", "hold position", "fire",
              "save the level", "start play session", "select the camera", "undo", "redo"]:
        _ = cmds.add(p)
    var mic = PulseRecorder()
    if not mic.open(16000):
        print("[live] no microphone:", mic.error)
        return
    print("[live] listening on", m.device(), "for", seconds, "s; say e.g. \"open the gate\"")
    var vad = EnergyVad()
    var t0 = perf_counter_ns()
    var last_partial = 0
    while Float64(perf_counter_ns() - t0) / 1e9 < Float64(seconds):
        var chunk = mic.read(20)
        if not mic.is_open():
            print("[live] capture stopped:", mic.error)
            break
        if vad.push(chunk):
            var r = m.transcribe(vad.utterance)
            var k = cmds.match(r.text)
            print("[final]", r.text, " (" + String(Int(r.total_ms())) + " ms)",
                  "-> command: " + cmds.phrases[k] if k >= 0 else "-> no command")
            vad.utterance = List[Float32]()
            last_partial = 0
        elif vad.active and len(vad.utterance) - last_partial >= 8000:
            last_partial = len(vad.utterance)
            var p = m.transcribe(vad.utterance)
            print("[partial]", p.text)
    mic.close()
