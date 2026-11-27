"""Flint v5 audio gap-closure suite (headless; never opens a real device).

  (a) Disk streaming: WAV16 & QOA streamed through the bounded ring are sample-identical
      to the in-memory paths (one-shot + looping, > 4x the ring length), channels recycle.
  (b) Playback control: per-voice pause/resume (position frozen), fade-out stop,
      global pause with a pause-exempt UI bus, per-bus pause.
  (c) Output robustness: injected device loss -> reconnect with backoff while the mixer
      clock keeps running; backoff schedule; device enumeration API.
  (d) Sound design: random(no-repeat)/sequence/switch containers, RTPC curves -> voice gain,
      mixer snapshots blended over time, JSON bank loading.
  (e) Attenuation presets (linear/log/inverse/custom + cone) and the physics occlusion provider.
  (f) Pure-Mojo Ogg Vorbis decoder: bit-exact (+-1 LSB) against a libvorbis reference decode,
      corrupted/truncated input rejected without crashing, `.ogg` loading through AudioServer.
"""
from std.collections import List
from std.ffi import external_call
from std.math import sin, sqrt, abs
from std.testing import assert_true, assert_equal, assert_false
from flint.core.math import Vec3
from flint.platform.os import write_bytes, remove_file, env_value, read_bytes
from flint.platform.audio import decode_wav
from flint.audio.vorbis import decode_ogg_vorbis, is_ogg_vorbis
from flint.assets.asset_ref import AssetRef
from flint.assets.cooker_v2 import cook_single_source_bytes
from flint.assets.flpk_v3 import AssetMeta, ASSET_KIND_AUDIO, write_flpk_v3
from flint.assets.server import AssetServer
from flint.core.serial import BlobReader, BlobWriter
from flint.audio.qoa import encode_qoa, decode_qoa
from flint.audio.server import (
    AudioServer,
    BUS_SFX,
    BUS_MUSIC,
    BUS_UI,
    next_reconnect_backoff_ms,
)
from flint.audio.stream import STREAM_RING_FRAMES, probe_stream_source, STREAM_FMT_WAV16, STREAM_FMT_QOA, stream_ring_sample
from flint.audio.design import SoundBank, GameParameter, MixerSnapshot
from flint.audio.plugin import (
    AudioPlugin,
    AudioEmitterComponent,
    AudioListenerComponent,
    AudioSyncStateResource,
    evaluate_physics_occlusion,
)
from flint.audio.acoustics import AcousticMaterialTable
from flint.app import AppConfig, EngineApp
from flint.physics.world import PhysicsWorld
from flint.physics.plugin import PhysicsPlugin, RigidBodyComponent, ColliderComponent
from flint.scene import GlobalTransform, LocalTransform, TransformPlugin
from flint.audio.attenuation import (
    AttenuationPreset,
    ATTEN_INVERSE,
    ATTEN_LINEAR,
    ATTEN_LOG,
    ATTEN_CUSTOM,
)


def _tmp_path(name: String) -> String:
    var d = env_value("TMPDIR")
    if d.byte_length() == 0:
        d = "/tmp"
    return d + "/flint_audio_gaps_" + name


def _sleep_ms(ms: Int):
    var ts = Array[Int64, 2](fill=0)
    ts[1] = Int64(ms) * 1_000_000
    _ = external_call["nanosleep", Int32](Pointer(to=ts), Int(0))
    _ = ts[1]


def _test_pcm(frames: Int, channels: Int) -> List[Int16]:
    """Deterministic non-periodic stereo test signal (two detuned sines + slow ramp)."""
    var pcm = List[Int16](capacity=frames * channels)
    for i in range(frames):
        var t = Float64(i) / 48000.0
        var l = 0.45 * sin(2.0 * 3.141592653589793 * 330.0 * t) + 0.2 * sin(2.0 * 3.141592653589793 * 47.0 * t)
        var r = 0.40 * sin(2.0 * 3.141592653589793 * 523.25 * t) + 0.15 * (Float64((i * 7919) % 1000) / 1000.0 - 0.4995)
        pcm.append(Int16(Int(l * 32000.0)))
        if channels > 1:
            pcm.append(Int16(Int(r * 32000.0)))
    return pcm^


def put32(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))
    b.append(UInt8((v >> 16) & 0xFF))
    b.append(UInt8((v >> 24) & 0xFF))


def put16(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))


def _wav_bytes(pcm: List[Int16], channels: Int, rate: Int) -> List[UInt8]:
    var data_bytes = len(pcm) * 2
    var out = List[UInt8](capacity=44 + data_bytes + 16)
    out.append(82); out.append(73); out.append(70); out.append(70)  # RIFF
    put32(out, 36 + 12 + data_bytes)
    out.append(87); out.append(65); out.append(86); out.append(69)  # WAVE
    # A LIST chunk before fmt to exercise the chunk walker
    out.append(76); out.append(73); out.append(83); out.append(84)
    put32(out, 4)
    out.append(1); out.append(2); out.append(3); out.append(4)
    out.append(102); out.append(109); out.append(116); out.append(32)  # fmt
    put32(out, 16)
    put16(out, 1)
    put16(out, channels)
    put32(out, rate)
    put32(out, rate * channels * 2)
    put16(out, channels * 2)
    put16(out, 16)
    out.append(100); out.append(97); out.append(116); out.append(97)  # data
    put32(out, data_bytes)
    for i in range(len(pcm)):
        var u = Int(pcm[i]) & 0xFFFF
        put16(out, u)
    return out^


def _render_n(mut srv: AudioServer, quanta: Int, frames: Int = 256) -> List[Float32]:
    var all = List[Float32](capacity=quanta * frames * 2)
    var buf = List[Float32](length=frames * 2, fill=0.0)
    for _ in range(quanta):
        srv.render_offline_quantum(buf, frames)
        for i in range(frames * 2):
            all.append(buf[i])
    return all^


def _rms(buf: List[Float32], start: Int = 0, count: Int = -1) -> Float32:
    var n = len(buf) - start if count < 0 else count
    if n <= 0:
        return 0.0
    var acc = Float64(0.0)
    for i in range(start, start + n):
        acc += Float64(buf[i]) * Float64(buf[i])
    return Float32(sqrt(acc / Float64(n)))


def _max_abs_diff(a: List[Float32], b: List[Float32]) -> Float32:
    var m = Float32(0.0)
    for i in range(min(len(a), len(b))):
        m = max(m, abs(a[i] - b[i]))
    return m


# ---------------------------------------------------------------------------
# (a) Disk streaming
# ---------------------------------------------------------------------------

def _compare_stream_vs_memory(
    mem_aid_is_qoa: Bool, path: String, var pcm: List[Int16], looping: Bool, quanta: Int
) raises -> Float32:
    var srv_mem = AudioServer(48000, 256)
    var srv_str = AudioServer(48000, 256)
    var mem_aid: Int
    if mem_aid_is_qoa:
        mem_aid = srv_mem.register_qoa_asset(encode_qoa(pcm, 2, 48000))
    else:
        mem_aid = srv_mem.register_pcm16_asset(pcm, 2, 48000)
    var str_aid = srv_str.load_audio_cached(path, stream=True)
    assert_true(str_aid >= 0, "stream asset registered")
    assert_true(srv_str.is_stream_asset(str_aid))
    assert_equal(srv_str.load_audio_cached(path, stream=True), str_aid, "stream path cache")
    _ = srv_mem.play_2d(mem_aid, 1.0, 1.0, 0.0, BUS_MUSIC, looping, True)
    var h = srv_str.play_2d(str_aid, 1.0, 1.0, 0.0, BUS_MUSIC, looping, True)
    assert_true(h.is_valid())
    assert_equal(srv_str.active_stream_count(), 1, "one channel claimed")
    var a = _render_n(srv_mem, quanta)
    var b = _render_n(srv_str, quanta)
    assert_true(_rms(b) > Float32(0.05), "streamed audio is audible")
    assert_equal(srv_str.stream_underruns(), 0, "offline streaming never underruns")
    if looping:
        assert_equal(srv_str.active_stream_count(), 1, "looping stream keeps its channel")
        _ = srv_str.stop_sound(h)
        _ = _render_n(srv_str, 2)
    assert_equal(srv_str.active_stream_count(), 0, "channel recycled after the sound ended")
    return _max_abs_diff(a, b)


def test_streaming() raises:
    var frames = 48000 * 2  # 2 s stereo, ~5.9x the stream ring
    assert_true(frames > STREAM_RING_FRAMES * 4)
    var pcm = _test_pcm(frames, 2)
    var wav_path = _tmp_path("stream.wav")
    var wav = _wav_bytes(pcm, 2, 48000)
    assert_true(write_bytes(wav_path, wav))
    var info = probe_stream_source(wav_path)
    assert_true(info.valid and info.format == STREAM_FMT_WAV16)
    assert_equal(info.total_frames, frames)
    assert_equal(info.channels, 2)

    var qoa_path = _tmp_path("stream.qoa")
    var qbytes = encode_qoa(pcm, 2, 48000)
    assert_true(write_bytes(qoa_path, qbytes))
    var qinfo = probe_stream_source(qoa_path)
    assert_true(qinfo.valid and qinfo.format == STREAM_FMT_QOA)
    assert_equal(qinfo.total_frames, frames)

    # One-shot: entire file plus tail (sound ends -> channel released)
    var d1 = _compare_stream_vs_memory(False, wav_path, pcm.copy(), False, frames // 256 + 4)
    assert_true(d1 < Float32(1.0e-6), "WAV stream == in-memory PCM (max diff " + String(d1) + ")")
    var d2 = _compare_stream_vs_memory(True, qoa_path, pcm.copy(), False, frames // 256 + 4)
    assert_true(d2 < Float32(1.0e-6), "QOA stream == in-memory QOA (max diff " + String(d2) + ")")

    # Looping across the file end several times (short 0.5 s file)
    var short = _test_pcm(24000, 2)
    var loop_path = _tmp_path("loop.wav")
    var lw = _wav_bytes(short, 2, 48000)
    assert_true(write_bytes(loop_path, lw))
    var d3 = _compare_stream_vs_memory(False, loop_path, short.copy(), True, 24000 * 3 // 256)
    assert_true(d3 < Float32(1.0e-5), "looping WAV stream == in-memory loop (max diff " + String(d3) + ")")

    # Background streamer pthread with the headless RT mixer
    var srv = AudioServer(48000, 256)
    var aid = srv.load_audio_cached(wav_path, stream=True)
    assert_true(srv.start_headless_rt_thread())
    var h = srv.play_2d(aid, 1.0, 1.0, 0.0, BUS_MUSIC, True, True)
    var q0 = srv.quantums_rendered()
    _sleep_ms(150)
    assert_true(srv.quantums_rendered() > q0 + 20, "RT mixer advanced")
    assert_equal(srv.active_stream_count(), 1)
    _ = srv.stop_sound(h)
    var released = False
    for _ in range(100):
        _sleep_ms(5)
        if srv.active_stream_count() == 0:
            released = True
            break
    assert_true(released, "streamer thread released the channel")
    srv.stop_realtime_thread()

    # Missing / non-streamable files are rejected cleanly
    assert_equal(srv.load_audio_cached(_tmp_path("does_not_exist.wav"), stream=True), -1)
    _ = remove_file(wav_path)
    _ = remove_file(qoa_path)
    _ = remove_file(loop_path)
    print("[PASS] (a) Disk streaming: WAV16/QOA bounded-ring stream == in-memory (one-shot, looping, RT streamer)")


# ---------------------------------------------------------------------------
# (b) Playback control
# ---------------------------------------------------------------------------

def test_packed_stream() raises:
    var pcm = _test_pcm(96000, 2)
    for encoding in range(2):
        var path = "music/test.wav" if encoding == 0 else "music/test.qoa"
        var raw = _wav_bytes(pcm, 2, 48000) if encoding == 0 else encode_qoa(pcm, 2, 48000)
        var meta = AssetMeta()
        var aref = AssetRef.from_path(path, ASSET_KIND_AUDIO)
        meta.uuid = aref.uuid
        var cooked = cook_single_source_bytes(raw, meta, path)
        assert_true(cooked.valid and len(cooked.chunks) == 2)
        var pack_path = _tmp_path("stream-" + String(external_call["getpid", Int32]()) + ".flpk")
        assert_true(write_flpk_v3(pack_path, cooked.chunks))

        var app = EngineApp(AppConfig(title="PackAudio", headless=True, worker_threads=0))
        app.add_plugin(TransformPlugin())
        app.add_plugin(AudioPlugin(headless=True))
        var assets = AssetServer(enable_watcher=False)
        assert_true(assets.mount_flpk(pack_path))
        app.insert_resource[AssetServer](assets^)
        app.startup()
        ref srv = app.resources.get_ptr_mut[AudioServer]().value()[]
        srv.stop_realtime_thread()
        var entity = app.world.spawn()
        _ = app.world.insert[GlobalTransform](entity, GlobalTransform())
        var emitter = AudioEmitterComponent(asset_path=path, streaming=True, play_on_spawn=True, spatial=False)
        var bw = BlobWriter()
        emitter.write_bin(bw)
        var br = BlobReader(bw.data.copy())
        var restored = AudioEmitterComponent.read_bin(br, AudioEmitterComponent.SCHEMA_VERSION)
        assert_true(restored.streaming)
        _ = app.world.insert[AudioEmitterComponent](entity, restored^)
        app.step_one_frame(Float32(1.0 / 60.0))
        ref em = app.world.get_ptr[AudioEmitterComponent](entity).value()[]
        assert_true(srv.is_stream_asset(em.asset_id))
        var h = em.handle
        assert_true(h.is_valid())
        _ = _render_n(srv, 1)
        ref mounted = app.resources.get_ptr_mut[AssetServer]().value()[]
        assert_equal(mounted.audios.find_slot_by_path(path), -1)
        mounted.mounts.clear()
        assert_true(remove_file(pack_path))
        var qclip = decode_qoa(raw) if encoding == 1 else decode_qoa(List[UInt8]())
        for frame in [75000, 113]:
            assert_true(srv.seek_stream(h, frame))
            _ = _render_n(srv, 1)
            ref voice = srv._rt_core()[].emitters[Int(h.slot)]
            assert_equal(Int(voice.virtual_frame_pos), frame + 256)
            var check = frame + 128
            var expected = pcm[check * 2] if encoding == 0 else qclip.samples[check * 2]
            var value = stream_ring_sample(srv._stream_pool()[].ctrl_addr, voice.stream_ch, check, 0)
            assert_equal(value, Float32(expected) * Float32(1.0 / 32768.0))
        assert_false(srv.seek_stream(h, 96000))
        app.shutdown()
    print("[PASS] packed PCM/QOA scene emitters survive unmount and seek sample-accurately in both directions")


def _emitter_pos(srv: AudioServer, slot: Int32) -> Float64:
    return srv._rt_core()[].emitters[Int(slot)].virtual_frame_pos


def test_playback_control() raises:
    var tone = _test_pcm(48000, 2)
    var srv = AudioServer(48000, 256)
    _ = srv.set_dc_blocker(False)  # isolate voice output from the master DC-blocker tail
    var aid = srv.register_pcm16_asset(tone, 2, 48000)
    var h = srv.play_2d(aid, 1.0, 1.0, 0.0, BUS_SFX, True, True)
    var a = _render_n(srv, 4)
    assert_true(_rms(a, 512 * 2) > Float32(0.05))

    # Per-voice pause: one quantum of fade, then silence with a frozen position
    assert_true(srv.pause_sound(h))
    var fade = _render_n(srv, 2)  # 256-frame fade + 64-frame limiter lookahead tail
    var p0 = _emitter_pos(srv, h.slot)
    var paused = _render_n(srv, 6)
    assert_true(_rms(paused) < Float32(1.0e-3), "paused voice is silent (rms " + String(_rms(paused)) + ")")
    assert_true(abs(_emitter_pos(srv, h.slot) - p0) < 1.0e-9, "paused position frozen")
    # The fade is click-free: no sample-to-sample jump larger than the signal's own slope
    var max_step = Float32(0.0)
    for i in range(2, len(fade), 2):
        max_step = max(max_step, abs(fade[i] - fade[i - 2]))
    assert_true(max_step < Float32(0.08), "click-free pause fade (max step " + String(max_step) + ")")
    assert_equal(srv.active_logical_count(), 1, "paused voice stays alive")
    assert_true(srv.resume_sound(h))
    var resumed = _render_n(srv, 3)
    assert_true(_rms(resumed, 512) > Float32(0.05), "resumed")
    assert_true(_emitter_pos(srv, h.slot) > p0 + 500.0, "position advances after resume")

    # Fade-out stop: decays over ~0.1 s then frees the voice
    assert_true(srv.stop_sound(h, 0.1))
    var r1 = _rms(_render_n(srv, 3))
    var r2 = _rms(_render_n(srv, 6))
    assert_true(r2 < r1, "fading")
    assert_true(srv.active_logical_count() >= 1 or r2 < Float32(0.05))
    _ = _render_n(srv, 16)
    assert_equal(srv.active_logical_count(), 0, "faded voice released")

    # Global pause with a pause-exempt UI bus
    var sfx = srv.play_2d(aid, 1.0, 1.0, 0.0, BUS_SFX, True, True)
    var ui = srv.play_2d(aid, 1.0, 1.0, 0.0, BUS_UI, True, True)
    _ = _render_n(srv, 2)
    assert_true(srv.set_bus_pause_exempt(BUS_UI, True))
    assert_true(srv.pause_all())
    _ = _render_n(srv, 2)
    var sfx_p = _emitter_pos(srv, sfx.slot)
    var ui_p = _emitter_pos(srv, ui.slot)
    var menu = _render_n(srv, 4)
    assert_true(abs(_emitter_pos(srv, sfx.slot) - sfx_p) < 1.0e-9, "SFX frozen in pause menu")
    assert_true(_emitter_pos(srv, ui.slot) > ui_p + 1000.0, "UI keeps playing")
    assert_true(_rms(menu) > Float32(0.03), "menu audio audible")
    assert_true(srv.resume_all())
    _ = _render_n(srv, 2)
    assert_true(_emitter_pos(srv, sfx.slot) > sfx_p + 300.0, "SFX resumes")

    # Per-bus pause
    var mus = srv.play_2d(aid, 1.0, 1.0, 0.0, BUS_MUSIC, True, True)
    _ = _render_n(srv, 2)
    assert_true(srv.set_bus_paused(BUS_MUSIC, True))
    _ = _render_n(srv, 2)
    var mp = _emitter_pos(srv, mus.slot)
    var sp = _emitter_pos(srv, sfx.slot)
    _ = _render_n(srv, 3)
    assert_true(abs(_emitter_pos(srv, mus.slot) - mp) < 1.0e-9, "music bus paused")
    assert_true(_emitter_pos(srv, sfx.slot) > sp + 500.0, "SFX unaffected by music bus pause")
    assert_true(srv.set_bus_paused(BUS_MUSIC, False))
    _ = _render_n(srv, 2)
    assert_true(_emitter_pos(srv, mus.slot) > mp + 300.0)

    # Fade-out stop_all
    srv.stop_all(0.05)
    _ = _render_n(srv, 14)
    assert_equal(srv.active_logical_count(), 0, "stop_all fade released everything")
    print("[PASS] (b) Playback control: voice pause/resume, fade-out stop, pause menu (UI exempt), bus pause")


# ---------------------------------------------------------------------------
# (c) Output robustness
# ---------------------------------------------------------------------------

def test_output_robustness() raises:
    assert_equal(next_reconnect_backoff_ms(0), 100)
    assert_equal(next_reconnect_backoff_ms(100), 200)
    assert_equal(next_reconnect_backoff_ms(3200), 5000)
    assert_equal(next_reconnect_backoff_ms(5000), 5000)

    var srv = AudioServer(48000, 256)
    assert_equal(srv.audio_backend_name(), "none")
    assert_true(srv.start_headless_rt_thread())
    assert_equal(srv.audio_backend_name(), "headless")
    _sleep_ms(10)
    assert_true(srv.simulate_device_loss())
    var saw_lost = False
    for _ in range(200):
        _sleep_ms(2)
        if srv.device_lost_count() >= 1:
            saw_lost = True
            break
    assert_true(saw_lost, "device loss observed")
    var q_lost = srv.quantums_rendered()
    var back = False
    for _ in range(400):
        _sleep_ms(5)
        if srv.reconnect_count() >= 1 and srv.audio_backend_name() == "headless":
            back = True
            break
    assert_true(back, "reconnected after backoff")
    assert_true(srv.quantums_rendered() > q_lost, "mixer clock kept running during the outage")
    srv.stop_realtime_thread()
    assert_equal(srv.audio_backend_name(), "none")

    # Device enumeration never crashes (may be empty in CI); selection is remembered.
    var devs = srv.list_output_devices()
    for i in range(len(devs)):
        assert_true(devs[i].id.startswith("pulse:") or devs[i].id.startswith("alsa:"))
    assert_true(srv.select_output_device("alsa:/dev/snd/pcmC9D9p"))
    assert_equal(srv.output_device(), "alsa:/dev/snd/pcmC9D9p")
    assert_true(srv.select_output_device(""))
    print("[PASS] (c) Output robustness: device-loss reconnect w/ backoff, live mixer clock, device enumeration/selection (" + String(len(devs)) + " devices)")


# ---------------------------------------------------------------------------
# (e) Attenuation presets
# ---------------------------------------------------------------------------

def test_attenuation() raises:
    var lin = AttenuationPreset(ATTEN_LINEAR, 1.0, 11.0)
    assert_true(abs(lin.distance_gain(6.0) - 0.5) < 1.0e-5)
    assert_true(lin.distance_gain(0.5) == 1.0 and lin.distance_gain(12.0) == 0.0)
    var lg = AttenuationPreset(ATTEN_LOG, 1.0, 4.0)
    assert_true(abs(lg.distance_gain(2.0) - 0.1) < 1.0e-3, "log: -20 dB at 1/3 range")
    var inv = AttenuationPreset(ATTEN_INVERSE, 2.0, 100.0, 1.0)
    assert_true(abs(inv.distance_gain(4.0) - 0.5) < 1.0e-5)
    var cust = AttenuationPreset(ATTEN_CUSTOM, 0.0, 10.0)
    assert_true(cust.add_point(0.0, 1.0) and cust.add_point(0.5, 0.8) and cust.add_point(1.0, 0.0))
    assert_true(abs(cust.distance_gain(2.5) - 0.9) < 1.0e-3)
    assert_false(cust.add_point(0.2, 0.5), "custom points must be increasing")
    var cone = AttenuationPreset(ATTEN_LINEAR, 1.0, 50.0)
    cone.set_cone(60.0, 120.0, 0.25)
    var fwd = Vec3(0.0, 0.0, 1.0)
    assert_true(abs(cone.cone_gain(fwd, Vec3(0.0, 0.0, 5.0)) - 1.0) < 1.0e-5, "inside inner cone")
    assert_true(abs(cone.cone_gain(fwd, Vec3(0.0, 0.0, -5.0)) - 0.25) < 1.0e-5, "behind")
    var mid = cone.cone_gain(fwd, Vec3(1.0, 0.0, 1.0))  # 45 deg: between 30 and 60 deg half-angles
    assert_true(mid < 1.0 and mid > 0.25)

    # End-to-end: an emitter at 8 m is audible with the default curve, silent past a 5 m preset.
    var tone = _test_pcm(48000, 1)
    var srv_a = AudioServer(48000, 256)
    var srv_b = AudioServer(48000, 256)
    _ = srv_b.set_dc_blocker(False)
    var aa = srv_a.register_pcm16_asset(tone, 1, 48000)
    var ab = srv_b.register_pcm16_asset(tone, 1, 48000)
    var short_range = srv_b.register_attenuation(AttenuationPreset(ATTEN_LINEAR, 1.0, 5.0))
    srv_a.set_listener(Vec3(), Vec3(), Vec3(0.0, 0.0, -1.0), Vec3(1.0, 0.0, 0.0))
    srv_b.set_listener(Vec3(), Vec3(), Vec3(0.0, 0.0, -1.0), Vec3(1.0, 0.0, 0.0))
    _ = srv_a.play_3d(aa, Vec3(0.0, 0.0, -8.0), looping=True)
    _ = srv_b.play_3d(ab, Vec3(0.0, 0.0, -8.0), looping=True, attenuation_id=short_range)
    var ra = _rms(_render_n(srv_a, 6), 1024)
    var rb = _rms(_render_n(srv_b, 6), 1024)
    assert_true(ra > Float32(0.02), "default curve audible at 8 m")
    assert_true(rb < Float32(1.0e-4), "5 m linear preset silences an 8 m emitter")

    # Cone: facing away attenuates to the outer gain
    var srv_c = AudioServer(48000, 256)
    var srv_d = AudioServer(48000, 256)
    var ac = srv_c.register_pcm16_asset(tone, 1, 48000)
    var ad = srv_d.register_pcm16_asset(tone, 1, 48000)
    var coned = AttenuationPreset(ATTEN_LINEAR, 1.0, 40.0)
    coned.set_cone(60.0, 90.0, 0.25)
    var pc = srv_c.register_attenuation(coned)
    var pd = srv_d.register_attenuation(coned)
    _ = srv_c.play_3d(ac, Vec3(0.0, 0.0, -4.0), looping=True, attenuation_id=pc, cone_dir=Vec3(0.0, 0.0, 1.0))
    _ = srv_d.play_3d(ad, Vec3(0.0, 0.0, -4.0), looping=True, attenuation_id=pd, cone_dir=Vec3(0.0, 0.0, -1.0))
    var rc = _rms(_render_n(srv_c, 8), 2048)
    var rd = _rms(_render_n(srv_d, 8), 2048)
    var ratio = rd / max(Float32(1.0e-9), rc)
    assert_true(ratio > Float32(0.2) and ratio < Float32(0.3), "cone outer gain 0.25 (got " + String(ratio) + ")")
    print("[PASS] (e1) Attenuation presets: linear/log/inverse/custom curves, cone, per-voice binding")


# ---------------------------------------------------------------------------
# (d) Sound-design layer
# ---------------------------------------------------------------------------

comptime _BANK_JSON = """{
  "parameters": [
    {"name": "speed", "default": 0, "min": 0, "max": 100},
    {"name": "surface", "default": 0, "min": 0, "max": 1},
    {"name": "tension", "default": 0, "min": 0, "max": 100, "slew": 50}
  ],
  "attenuations": [
    {"name": "steps", "curve": "custom", "min": 1, "max": 30, "points": [[0, 1], [0.5, 0.4], [1, 0]],
     "cone": {"inner": 360, "outer": 360, "outer_gain": 1}}
  ],
  "events": [
    {"name": "step", "type": "random", "avoid_repeat": 1, "bus": "SFX", "volume": 0.9,
     "volume_random": 0.1, "pitch_random": 0.05, "spatial": true, "attenuation": "steps",
     "children": [{"clip": "s1"}, {"clip": "s2"}, {"clip": "s3", "weight": 2}]},
    {"name": "combo", "type": "sequence", "bus": "UI",
     "children": [{"clip": "s1"}, {"clip": "s2"}, {"clip": "s3"}]},
    {"name": "foot", "type": "switch", "switch": "surface",
     "children": [{"event": "step", "min": 0, "max": 0.5}, {"clip": "s3", "min": 0.5, "max": 1}]},
    {"name": "engine", "type": "single", "looping": true, "bus": "SFX",
     "children": [{"clip": "s1"}],
     "rtpc": [{"param": "speed", "target": "volume", "curve": [[0, 0.25], [100, 1.0]]},
              {"param": "speed", "target": "pitch", "curve": [[0, 0.8], [100, 1.6]]},
              {"param": "tension", "target": "lowpass", "curve": [[0, 20000], [100, 500]]}]}
  ],
  "snapshots": [
    {"name": "pause_menu", "buses": {"SFX": 0.2, "Music": 0.5}}
  ]
}"""


def test_sound_design() raises:
    var srv = AudioServer(48000, 256)
    _ = srv.set_dc_blocker(False)
    var bank = SoundBank(seed=1234)
    var s1 = srv.register_pcm16_asset(_test_pcm(48000, 1), 1, 48000)
    var s2 = srv.register_pcm16_asset(_test_pcm(24000, 1), 1, 48000)
    var s3 = srv.register_pcm16_asset(_test_pcm(12000, 1), 1, 48000)
    bank.bind_clip("s1", s1)
    bank.bind_clip("s2", s2)
    bank.bind_clip("s3", s3)
    assert_equal(bank.load_json(_BANK_JSON, srv), 4)

    # Random container: never the same clip twice in a row, every clip used
    var prev = -1
    var used = List[Int](length=3, fill=0)
    for _ in range(60):
        var h = bank.post_event(srv, "step", Vec3(0.0, 0.0, -3.0))
        assert_true(h.is_valid())
        var a = bank.last_asset_id
        assert_true(a != prev, "random container repeated a clip")
        prev = a
        used[a - s1] += 1
        _ = srv.stop_sound(h)
    assert_true(used[0] > 0 and used[1] > 0 and used[2] > 0)
    assert_true(used[2] > used[0], "weight 2 child picked more often")

    # Sequence container cycles in order
    var seq = List[Int]()
    for _ in range(6):
        _ = bank.post_event(srv, "combo")
        seq.append(bank.last_asset_id)
    assert_true(seq[0] == s1 and seq[1] == s2 and seq[2] == s3 and seq[3] == s1 and seq[5] == s3)

    # Switch container by game parameter (nested random event below 0.5)
    assert_true(bank.set_parameter("surface", 0.9))
    _ = bank.post_event(srv, "foot")
    assert_equal(bank.last_asset_id, s3)
    assert_true(bank.set_parameter("surface", 0.1))
    _ = bank.post_event(srv, "foot")
    assert_true(bank.last_asset_id >= s1 and bank.last_asset_id <= s3)
    assert_false(bank.post_event(srv, "nope").is_valid())
    srv.stop_all()
    _ = _render_n(srv, 1)

    # RTPC: speed -> volume & pitch, tension (slewed) -> low-pass
    assert_true(bank.set_parameter("speed", 0.0))
    var eh = bank.post_event(srv, "engine")
    _ = _render_n(srv, 2)
    bank.update(srv, 0.016)
    var slow = _rms(_render_n(srv, 8), 1024)
    ref em0 = srv._rt_core()[].emitters[Int(eh.slot)]
    assert_true(abs(em0.rtpc_gain - 0.25) < 1.0e-5 and abs(em0.rtpc_pitch - 0.8) < 1.0e-5)
    assert_true(bank.set_parameter("speed", 100.0))
    bank.update(srv, 0.016)
    var fast = _rms(_render_n(srv, 8), 1024)
    ref em1 = srv._rt_core()[].emitters[Int(eh.slot)]
    assert_true(abs(em1.rtpc_gain - 1.0) < 1.0e-5 and abs(em1.rtpc_pitch - 1.6) < 1.0e-5)
    var ratio = fast / max(Float32(1.0e-9), slow)
    assert_true(ratio > Float32(3.0) and ratio < Float32(5.0), "RTPC volume 0.25 -> 1.0 (ratio " + String(ratio) + ")")
    assert_true(bank.set_parameter("tension", 100.0))
    bank.update(srv, 1.0)  # slew 50/s -> halfway
    assert_true(abs(bank.get_parameter("tension") - 50.0) < 1.0e-3, "slewed parameter")
    _ = _render_n(srv, 1)
    ref em2 = srv._rt_core()[].emitters[Int(eh.slot)]
    assert_true(abs(em2.user_cutoff_hz - 10250.0) < 1.0, "lowpass curve at 50 -> 10250 Hz")
    assert_equal(bank.live_instance_count(), 1)

    # Mixer snapshot blended over one second
    assert_true(bank.activate_snapshot("pause_menu", 1.0))
    bank.update(srv, 0.5)
    _ = _render_n(srv, 1)
    assert_true(abs(bank.bus_gain(BUS_SFX) - 0.6) < 1.0e-4)
    assert_true(abs(srv._rt_core()[].buses[BUS_SFX].gain - 0.6) < 1.0e-4, "bus gain pushed to the mixer")
    assert_true(abs(srv._rt_core()[].buses[BUS_MUSIC].gain - 0.75) < 1.0e-4)
    bank.update(srv, 0.6)
    _ = _render_n(srv, 1)
    assert_true(abs(srv._rt_core()[].buses[BUS_SFX].gain - 0.2) < 1.0e-4)
    bank.deactivate_snapshot(0.0)
    bank.update(srv, 0.016)
    _ = _render_n(srv, 1)
    assert_true(abs(srv._rt_core()[].buses[BUS_SFX].gain - 1.0) < 1.0e-4)

    # Finished instances are pruned
    _ = bank.stop_event(srv, "engine")
    _ = _render_n(srv, 1)
    bank.update(srv, 0.016)
    assert_equal(bank.live_instance_count(), 0)

    # Malformed banks are rejected
    var bad = SoundBank()
    var raised = False
    try:
        _ = bad.load_json("{\"events\": [{\"name\": \"x\", \"attenuation\": \"missing\"}]}", srv)
    except:
        raised = True
    assert_true(raised, "unknown attenuation reference raises")
    print("[PASS] (d) Sound design: random(no-repeat, weighted)/sequence/switch containers, RTPC volume/pitch/lowpass + slew, snapshot blend, JSON bank")


# ---------------------------------------------------------------------------
# (e2) Physics occlusion provider
# ---------------------------------------------------------------------------

def test_physics_occlusion() raises:
    var phys = PhysicsWorld(gravity=0.0)
    var wall = phys.add_box(Vec3(0.0, 1.0, -5.0), Vec3(5.0, 5.0, 0.2), 0.0)
    var own = phys.add_box(Vec3(0.0, 1.0, -10.0), Vec3(0.5, 0.5, 0.5), 0.0)
    phys.rebuild_broadphase()
    var mats = AcousticMaterialTable()
    var blocked = evaluate_physics_occlusion(phys, mats, Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, -10.0), own)
    assert_true(blocked.direct_blocked and blocked.wall_hits == 1, "wall blocks the direct path")
    assert_true(blocked.occlusion_gain < Float32(0.1) and blocked.occlusion_cutoff_hz < Float32(1000.0))
    var clear = evaluate_physics_occlusion(phys, mats, Vec3(0.0, 1.0, 0.0), Vec3(8.0, 1.0, 0.0))
    assert_false(clear.direct_blocked)
    assert_true(clear.occlusion_gain == 1.0)
    _ = wall

    # Plugin integration: occlusion refreshed every 2 frames through PhysicsWorld
    var app = EngineApp(AppConfig(title="AudioOcclusion", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.add_plugin(AudioPlugin(headless=True, physics_occlusion=True, occlusion_every_n_frames=2))
    app.startup()
    var aid = app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].register_pcm16_asset(_test_pcm(24000, 1), 1, 48000)
    var wall_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](wall_e, LocalTransform(Vec3(0.0, 1.0, -5.0)))
    _ = app.world.insert[GlobalTransform](wall_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](wall_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](wall_e, ColliderComponent.box(Vec3(5.0, 5.0, 0.2)))
    var lis_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](lis_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](lis_e, GlobalTransform())
    _ = app.world.insert[AudioListenerComponent](lis_e, AudioListenerComponent(active=True))
    var hid_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](hid_e, LocalTransform(Vec3(0.0, 1.0, -10.0)))
    _ = app.world.insert[GlobalTransform](hid_e, GlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        hid_e, AudioEmitterComponent(asset_id=aid, looping=True, play_on_spawn=True, enable_occlusion=True)
    )
    var vis_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](vis_e, LocalTransform(Vec3(8.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](vis_e, GlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        vis_e, AudioEmitterComponent(asset_id=aid, looping=True, play_on_spawn=True, enable_occlusion=True)
    )
    for _ in range(8):
        app.step_one_frame(Float32(1.0 / 60.0))
    ref hid = app.world.get_ptr[AudioEmitterComponent](hid_e).value()[unsafe_offset=0]
    ref vis = app.world.get_ptr[AudioEmitterComponent](vis_e).value()[unsafe_offset=0]
    assert_true(hid.occlusion_gain < Float32(0.1), "occluded emitter (gain " + String(hid.occlusion_gain) + ")")
    assert_true(hid.cutoff_hz < Float32(1000.0))
    assert_true(vis.occlusion_gain > Float32(0.99), "clear emitter")
    var queries = app.resources.get_ptr[AudioSyncStateResource]().value()[unsafe_offset=0].occlusion_queries
    assert_true(queries >= 4 and queries <= 10, "queried every 2nd frame (" + String(queries) + ")")
    app.shutdown()
    print("[PASS] (e2) Physics occlusion provider: PhysicsWorld raycast_all every N frames -> per-emitter occlusion")


def test_vorbis() raises:
    var ogg = read_bytes("tests/fixtures/audio/vorbis_stereo_44k.ogg")
    var ref_raw = read_bytes("tests/fixtures/audio/vorbis_stereo_44k_ref.wav")
    assert_true(len(ogg) > 0 and len(ref_raw) > 0, "vorbis fixtures present (run from repo root)")
    assert_true(is_ogg_vorbis(ogg))
    var clip = decode_ogg_vorbis(ogg)
    assert_true(clip.valid, "vorbis decode: " + clip.error)
    var refw = decode_wav(ref_raw)
    assert_true(refw.valid)
    assert_equal(clip.channels, 2)
    assert_equal(clip.sample_rate, 44100)
    assert_equal(clip.channels, refw.channels)
    # Granule end-trim: exact length of the libvorbis reference.
    assert_equal(len(clip.samples), len(refw.samples))
    var max_lsb = 0
    var energy = 0.0
    for i in range(len(refw.samples)):
        var d = abs(Int(clip.samples[i]) - Int(refw.samples[i]))
        max_lsb = max(max_lsb, d)
        energy += Float64(refw.samples[i]) * Float64(refw.samples[i])
    assert_true(energy > 1.0e9, "reference is not silent")
    assert_true(max_lsb <= 1, "vorbis vs libvorbis reference within 1 LSB (got " + String(max_lsb) + ")")

    # Corruption: truncated streams and flipped bytes are rejected or decode partially, never crash.
    var cut = List[UInt8]()
    for i in range(len(ogg) // 3):
        cut.append(ogg[i])
    var part = decode_ogg_vorbis(cut)
    assert_true(len(part.samples) < len(clip.samples))
    var bad = ogg.copy()
    bad[0] = 0
    assert_false(decode_ogg_vorbis(bad).valid)
    var flipped = ogg.copy()
    for i in range(200, len(flipped), 97):
        flipped[i] ^= 0x5A  # CRC mismatch -> damaged pages dropped
    _ = decode_ogg_vorbis(flipped)

    # AudioServer integration: `.ogg` loads through the normal cached path and plays.
    var srv = AudioServer(48000, 256)
    _ = srv.set_dc_blocker(False)
    var aid = srv.load_audio_cached("tests/fixtures/audio/vorbis_stereo_44k.ogg")
    assert_true(aid >= 0)
    assert_equal(srv.load_audio_cached("tests/fixtures/audio/vorbis_stereo_44k.ogg"), aid)
    var sid = srv.load_audio_cached("tests/fixtures/audio/vorbis_stereo_44k.ogg", stream=True)
    assert_equal(sid, aid)  # not disk-streamable: falls back to the cached in-memory decode
    _ = srv.play_2d(aid, 1.0)
    var out = _render_n(srv, 40)
    assert_true(_rms(out) > Float32(1.0e-3), "decoded vorbis is audible")
    print("[PASS] (f) Ogg Vorbis: within " + String(max_lsb) + " LSB of libvorbis, corruption-safe, AudioServer .ogg loading")


def main() raises:
    test_streaming()
    test_packed_stream()
    test_playback_control()
    test_output_robustness()
    test_attenuation()
    test_sound_design()
    test_physics_occlusion()
    test_vorbis()
    print("ALL FLINT V5 AUDIO GAP TESTS PASSED")
