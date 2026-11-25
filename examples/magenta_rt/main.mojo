"""Magenta RealTime 2 adaptive-music demo: a text prompt, a style change mid-way, 48 kHz out.

usage: magenta_rt MRT_DIR out.wav|out.qoa [--prompt "..."] [--prompt2 "..."] [--seconds 20]
                  [--switch 10] [--morph 4] [--seed 0] [--temperature 1.3] [--topk 40]
                  [--cpu] [--fp32] [--live]

MRT_DIR is the google/magenta-realtime-2 download laid out by `tools/magenta_rt/fetch.sh`
(checkpoints/mrt2_small.safetensors + resources/musiccoca/musiccoca_text.safetensors).

The music goes through the engine's audio path: `MagentaRtStream.pump` fills a lock-free
`PcmStreamRing`, `AudioServer.add_pcm_stream` mixes it into the music bus, and the server's
master output (limiter, LUFS meter) is what is written. At `--switch` seconds the style
morphs to `--prompt2` over `--morph` seconds. `--live` plays through PulseAudio while
generating instead of rendering offline. Defaults: int8 LM weights, SpectroStream on the GPU
(`--cpu` keeps everything on the CPU, `--fp32` uses float32 LM weights).
"""
from std.sys import argv
from std.ffi import external_call
from std.time import perf_counter_ns
from flint.ml.magenta_rt import MagentaRtStream, MRT_SAMPLE_RATE
from flint.ml.llm.quant import Q_F32, Q_INT8
from flint.audio.server import AudioServer, BUS_MUSIC
from flint.audio.pcm_stream import PcmStreamRing
from flint.audio.qoa import encode_qoa
from flint.cycles.synth import encode_wav16
from flint.platform.os import write_bytes


def main() raises:
    var a = argv()
    if len(a) < 3:
        print('usage: magenta_rt MRT_DIR out.wav|out.qoa [--prompt "..."] [--prompt2 "..."] [--seconds 20] [--switch 10] [--morph 4] [--seed 0] [--cpu] [--fp32] [--live]')
        return
    var prompt = String("upbeat synthwave with punchy drums")
    var prompt2 = String("dark ambient drone with deep strings")
    var seconds = 20.0
    var switch_s = 10.0
    var morph = 4.0
    var seed = 0
    var temperature: Float32 = 1.3
    var topk = 40
    var gpu = True
    var fmt = Q_INT8
    var live = False
    var i = 3
    while i < len(a):
        var s = String(a[i])
        var has = i + 1 < len(a)
        if s == "--prompt" and has:
            prompt = String(a[i + 1])
            i += 1
        elif s == "--prompt2" and has:
            prompt2 = String(a[i + 1])
            i += 1
        elif s == "--seconds" and has:
            seconds = Float64(String(a[i + 1]))
            i += 1
        elif s == "--switch" and has:
            switch_s = Float64(String(a[i + 1]))
            i += 1
        elif s == "--morph" and has:
            morph = Float64(String(a[i + 1]))
            i += 1
        elif s == "--seed" and has:
            seed = Int(String(a[i + 1]))
            i += 1
        elif s == "--temperature" and has:
            temperature = Float32(Float64(String(a[i + 1])))
            i += 1
        elif s == "--topk" and has:
            topk = Int(String(a[i + 1]))
            i += 1
        elif s == "--cpu":
            gpu = False
        elif s == "--fp32":
            fmt = Q_F32
        elif s == "--live":
            live = True
        i += 1

    var t0 = perf_counter_ns()
    var music = MagentaRtStream.load(String(a[1]), fmt)
    var device = String("cpu")
    if gpu:
        device = music.mrt.enable_gpu_codec()
    music.seed = seed
    music.mrt.set_sampling(temperature, topk)
    music.mrt.reset(seed)
    music.set_prompt(prompt)
    print("[magenta_rt] loaded in", Float64(perf_counter_ns() - t0) / 1e9, "s; codec on", device,
          "; LM weights", "int8" if fmt == Q_INT8 else "fp32")
    print("[magenta_rt] style A:", prompt, "-> B at", switch_s, "s:", prompt2, "(morph", morph, "s)")

    var server = AudioServer(MRT_SAMPLE_RATE, 480)
    var ring = PcmStreamRing.create(MRT_SAMPLE_RATE)   # 1 s of buffer
    server.add_pcm_stream(ring, BUS_MUSIC, 1.0)
    if live and not server.start_realtime_thread():
        print("[magenta_rt] no PulseAudio sink; rendering offline")
        live = False

    var total_frames = Int(seconds * Float64(MRT_SAMPLE_RATE))
    var switch_frame = Int(switch_s * 25.0)
    var codec_frames = (total_frames + 1919) // 1920
    var pcm = List[Int16](capacity=total_frames * 2)
    var quantum = List[Float32](length=480 * 2, fill=0.0)
    var made = 0
    var gen_ns = 0
    while made < codec_frames:
        if made == switch_frame:
            music.set_prompt(prompt2, morph)
        var g0 = perf_counter_ns()
        var n = music.pump(ring, 1)
        gen_ns += Int(perf_counter_ns() - g0)
        made += n
        if live:
            if n == 0:
                _ = external_call["usleep", Int32](UInt32(5000))   # ring full: the mixer is behind us
            continue
        # offline: render the server's master bus from what is buffered
        while ring.buffered_frames() >= 480:
            for j in range(len(quantum)):
                quantum[j] = 0.0
            server.render_offline_quantum(quantum, 480)
            for j in range(len(quantum)):
                pcm.append(Int16(max(Float32(-32767.0), min(Float32(32767.0), quantum[j] * 32767.0))))
        if made % 125 == 0:
            print("[magenta_rt]", Float64(made) / 25.0, "s generated, RTF",
                  Float64(gen_ns) / 1e9 / (Float64(made) / 25.0))
    var gen_s = Float64(gen_ns) / 1e9
    print("[magenta_rt]", seconds, "s of 48 kHz stereo in", gen_s, "s (RTF", gen_s / (Float64(codec_frames) / 25.0),
          "); LM", Float64(music.mrt.lm_ns) / 1e6 / Float64(codec_frames), "ms/frame, codec",
          Float64(music.mrt.codec_ns) / 1e6 / Float64(codec_frames), "ms/frame; underruns", ring.underrun_frames())
    if live:
        while ring.buffered_frames() > 0:
            _ = external_call["usleep", Int32](UInt32(20000))
        server.stop()
        ring.destroy()
        return
    print("[magenta_rt] master LUFS (integrated)", server.integrated_lufs())
    var out = String(a[2])
    var bytes = encode_qoa(pcm, 2, MRT_SAMPLE_RATE) if out.endswith(".qoa") else encode_wav16(pcm, MRT_SAMPLE_RATE, 2)
    if not write_bytes(out, bytes):
        raise Error("could not write " + out)
    print("[magenta_rt] wrote", out, "(", len(bytes), "bytes )")
    server.stop()
    ring.destroy()
