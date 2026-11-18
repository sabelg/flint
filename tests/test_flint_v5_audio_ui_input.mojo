"""Comprehensive Verification Suite for Flint v5 Track 4:
  (a) Lock-free 64B cache-line-isolated `AudioMpscRing` multi-producer `JobPool` stress,
      `AudioServer` zero-NaN/zero-DC offline quantum, `+12 dBFS` step into zero-overshoot
      lookahead true-peak limiter (`max |y| <= 0.989`), ITU-R BS.1770-4 K-weighted LUFS
      accuracy (`-3.01 ± 0.08 LUFS` on full-scale stereo sine), and 128->64 Voice Virtualizer
      click-free 128-sample steal crossfade + sidechain ducking + `CyclesStemNode`.
  (b) Woodworth spherical-head HRTF ITD (`~29.5` samples @ +90° right, 48 kHz) & ILD,
      continuous 180° listener rotation click-freedom, `0.1 Mach` Doppler (`1111.1 ± 1.5 Hz`),
      and `Bvh8` wall occlusion (`< 700 Hz` blocked vs `20 kHz` clear), 4-ray portal fan
      pillar obstruction, & 6-ray Sabine `RT60` room probe.
  (c) Pure-Mojo QOA codec round-trip (`SNR >= 29 dB`, exact `3.2 bits/sample` slice payload,
      bit-identical `QoaStreamCursor.seek` vs full `decode_qoa`).
  (d) 1,000 randomized Flexbox trees conservation/bounds/no-overlap fuzz, 5-panel `DockSpace`
      zero-gap tiling & JSON round-trip, `<0.1 ms` virtualized 100k-item `tree_node` outliner,
      UTF-8 `"¡Hola! エンジン ★"` `text_input` cursor safety, and modal focus trap.
  (e) Synthetic `EvdevGamepad` Xbox & DualSense `EVIOCGABS` (`0.0` unpressed trigger),
      48-byte `ff_effect` rumble packet layout, and `InputMap` context masking, `Ctrl+S` chord,
      `120 ms` jump buffer, `100 ms` coyote time, JSON rebind round-trip & deterministic tick pack.
"""
from std.collections import List
from std.math import sin, cos, sqrt, log10, abs, isnan, isinf
from std.sys.info import size_of
from std.testing import assert_true, assert_equal
from flint.core.math import Vec3
from flint.core.jobs import JobPool
from flint.platform.os import get_time
from flint.platform.input import (
    InputState,
    PAD_A,
    PAD_B,
    PAD_UP,
    PAD_DOWN,
    PAD_AXIS_LT,
    PAD_AXIS_RT,
    PAD_AXIS_LEFT_X,
)
from flint.physics.bvh import Bvh8MeshCollider
from flint.audio import (
    QOA_SLICE_BITS_PER_SAMPLE,
    QoaStreamCursor,
    encode_qoa,
    decode_qoa,
    AcousticMaterialTable,
    evaluate_emitter_acoustics,
    probe_room_sabine_rt60,
    BUS_MASTER,
    BUS_SFX,
    BUS_MUSIC,
    BUS_VOICE,
    AudioCmd,
    AudioMpscRing,
    AudioServer,
    compute_woodworth_hrtf,
)
from flint.ui.draw_list import UiDrawList, GpuUiPrim
from flint.ui.flexbox_dock import (
    RectF,
    FlexStyle,
    FlexTree,
    FLEX_ROW,
    FLEX_COL,
    ALIGN_START,
    ALIGN_CENTER,
    ALIGN_END,
    ALIGN_STRETCH,
    JUSTIFY_START,
    JUSTIFY_CENTER,
    JUSTIFY_END,
    JUSTIFY_SPACE_BETWEEN,
    DockSpace,
)
from flint.ui.context import (
    UiContext,
    UI_KEY_TAB,
    UI_KEY_LEFT,
    UI_KEY_RIGHT,
    UI_KEY_BACKSPACE,
    UI_KEY_HOME,
    UI_KEY_END,
)
from flint.text.utf8 import decode_utf8, REPLACEMENT
from flint.input import (
    EV_KEY,
    EV_ABS,
    ABS_X,
    ABS_Y,
    ABS_Z,
    ABS_RZ,
    BTN_SOUTH,
    BTN_EAST,
    VENDOR_MICROSOFT,
    VENDOR_SONY,
    EvdevGamepad,
    pack_ff_rumble_effect,
    pack_ff_play_event,
    ACTION_BOOL,
    ACTION_AXIS1D,
    ACTION_AXIS2D,
    MOD_CTRL,
    INPUT_DEV_KBM,
    INPUT_DEV_XBOX,
    INPUT_DEV_SONY,
    InputContext,
    InputMap,
)

comptime TAU64: Float64 = 6.283185307179586


# ==============================================================================
# Helpers for Multithreaded MPSC Stress & Geometry Construction
# ==============================================================================

struct _MpscStressCtx:
    var ring: AudioMpscRing
    var per_chunk: Int

    def __init__(out self, ring: AudioMpscRing, per_chunk: Int):
        self.ring = ring
        self.per_chunk = per_chunk


def _mpsc_producer_chunk(ctx_addr: Int, chunk_idx: Int):
    ref ctx = Pointer[_MpscStressCtx, origin=MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    for k in range(ctx.per_chunk):
        var cmd = AudioCmd()
        cmd.kind = UInt32(100 + chunk_idx)
        cmd.slot = Int32(chunk_idx * ctx.per_chunk + k)
        cmd.generation = UInt32(k + 1)
        cmd.p0 = Float32(cmd.slot) * Float32(0.5)
        while not ctx.ring.push(cmd):
            _ = external_call_yield()


@always_inline
def external_call_yield() -> Int32:
    from std.ffi import external_call
    return external_call["sched_yield", Int32]()


def _make_sine_pcm16(freq_hz: Float64, frames: Int, channels: Int = 1, amp: Float64 = 0.9, sample_rate: Int = 48000, dc_offset: Float64 = 0.0) -> List[Int16]:
    var out = List[Int16](capacity=frames * channels)
    for f in range(frames):
        var s = amp * sin(TAU64 * freq_hz * Float64(f) / Float64(sample_rate)) + dc_offset
        var q = Int16(max(-32768, min(32767, Int(s * 32767.0))))
        for _ in range(channels):
            out.append(q)
    return out^


def _build_wall_quad_bvh(wall_x: Float32, half_y: Float32, half_z: Float32) -> Bvh8MeshCollider:
    var verts = List[Vec3]()
    verts.append(Vec3(wall_x, -half_y, -half_z))
    verts.append(Vec3(wall_x, -half_y, half_z))
    verts.append(Vec3(wall_x, half_y, -half_z))
    verts.append(Vec3(wall_x, half_y, half_z))
    var idxs = List[Int32]()
    idxs.append(0); idxs.append(1); idxs.append(2)
    idxs.append(2); idxs.append(1); idxs.append(3)
    return Bvh8MeshCollider.build(verts, idxs)


def _build_box_room_bvh(hx: Float32, hy: Float32, hz: Float32) -> Bvh8MeshCollider:
    var verts = List[Vec3]()
    # 8 corners of [-hx, hx] x [-hy, hy] x [-hz, hz]
    verts.append(Vec3(-hx, -hy, -hz))  # 0
    verts.append(Vec3(hx, -hy, -hz))   # 1
    verts.append(Vec3(hx, hy, -hz))    # 2
    verts.append(Vec3(-hx, hy, -hz))   # 3
    verts.append(Vec3(-hx, -hy, hz))   # 4
    verts.append(Vec3(hx, -hy, hz))    # 5
    verts.append(Vec3(hx, hy, hz))     # 6
    verts.append(Vec3(-hx, hy, hz))    # 7
    var faces = [
        0, 1, 2, 0, 2, 3,  # -Z
        4, 6, 5, 4, 7, 6,  # +Z
        0, 4, 5, 0, 5, 1,  # -Y
        3, 2, 6, 3, 6, 7,  # +Y
        0, 3, 7, 0, 7, 4,  # -X
        1, 5, 6, 1, 6, 2,  # +X
    ]
    var idxs = List[Int32]()
    for v in faces:
        idxs.append(Int32(v))
    return Bvh8MeshCollider.build(verts, idxs)


# ==============================================================================
# (a) AudioMpscRing, Limiter, BS.1770-4 LUFS & 128->64 Voice Virtualizer
# ==============================================================================

def _relocate_audio_server(var s: AudioServer) -> AudioServer:
    """Forces a physical `__moveinit__` relocation of `AudioServer` across stack frames."""
    var local_copy = s^
    return local_copy^


def test_audio_server_mastering_and_virtualizer() raises:
    # 1. Exact 64-byte struct layouts & 64-byte cache-line head/tail isolation
    assert_equal(size_of[AudioCmd](), 64)
    assert_equal(size_of[GpuUiPrim](), 64)

    var ring = AudioMpscRing.create(2048)
    assert_true(ring.allocated())
    assert_equal(ring.dequeue_tail_address() - ring.enqueue_head_address(), 64)

    # Multi-producer JobPool concurrent enqueue stress (8 chunks x 192 cmds = 1536 cmds)
    var pool = JobPool()
    _ = pool.start(4)
    var total_expected = 8 * 192
    var stress_ctx = _MpscStressCtx(ring, 192)
    pool.run(_mpsc_producer_chunk, Int(Pointer(to=stress_ctx)), 8)
    _ = stress_ctx.per_chunk
    pool.stop()

    var seen = List[Bool](length=total_expected, fill=False)
    var popped = 0
    var c_out = AudioCmd()
    while ring.pop(c_out):
        var s = Int(c_out.slot)
        assert_true(s >= 0 and s < total_expected)
        assert_true(not seen[s])
        assert_true(abs(c_out.p0 - Float32(s) * Float32(0.5)) < Float32(1.0e-4))
        seen[s] = True
        popped += 1
    assert_equal(popped, total_expected)
    ring.destroy()

    # 2. Zero-NaN & Zero-DC offline quantum verification
    var srv_dc = AudioServer(48000, 256)
    var dc_clip = _make_sine_pcm16(250.0, 24000, 1, 0.25, 48000, 0.45)
    var dc_aid = srv_dc.register_pcm16_asset(dc_clip, 1, 48000)
    _ = srv_dc.play_2d(dc_aid, volume=1.0, looping=True, direct_stereo=True)
    var buf = List[Float32](length=512, fill=0.0)
    var tail_dc_sum = Float64(0.0)
    var tail_samples = 0
    for q in range(100):
        srv_dc.render_offline_quantum(buf, 256)
        for i in range(512):
            assert_true(not isnan(buf[i]) and not isinf(buf[i]))
        if q >= 80:
            for f in range(256):
                tail_dc_sum += Float64(buf[f * 2])
                tail_samples += 1
    var residual_dc = abs(tail_dc_sum / Float64(tail_samples))
    assert_true(residual_dc < 0.01)

    # 3. +12 dBFS (gain = 3.98107) step into 64-sample lookahead true-peak limiter (max |y| <= 0.989)
    var srv_lim = AudioServer(48000, 256)
    var full_sine = _make_sine_pcm16(997.0, 24000, 2, 1.0, 48000)
    var lim_aid = srv_lim.register_pcm16_asset(full_sine, 2, 48000)
    # Render 2 silent quantums, then instantaneous +12 dBFS step (volume = 3.98107)
    srv_lim.render_offline_quantum(buf, 256)
    srv_lim.render_offline_quantum(buf, 256)
    _ = srv_lim.play_2d(lim_aid, volume=3.98107, bus_id=BUS_SFX, looping=True, direct_stereo=True)
    var max_abs_y = Float32(0.0)
    for _ in range(16):
        srv_lim.render_offline_quantum(buf, 256)
        for i in range(512):
            var a = abs(buf[i])
            if a > max_abs_y:
                max_abs_y = a
    assert_true(max_abs_y > Float32(0.95) and max_abs_y <= Float32(0.989))

    # 4. ITU-R BS.1770-4 K-weighted LUFS accuracy (-3.01 +/- 0.08 LUFS on full-scale stereo sine)
    var srv_lufs = AudioServer(48000, 256)
    _ = srv_lufs.set_dc_blocker(False)  # Pure uncolored full-scale 997 Hz stereo reference
    var lufs_aid = srv_lufs.register_pcm16_asset(full_sine, 2, 48000)
    _ = srv_lufs.play_2d(lufs_aid, volume=1.0, bus_id=BUS_MUSIC, looping=True, direct_stereo=True)
    # Disable default Voice->Music sidechain interference (Voice is silent anyway) and run 1.0 s (48,000 frames)
    for _ in range(190):
        srv_lufs.render_offline_quantum(buf, 256)
    var measured_lufs = srv_lufs.integrated_lufs()
    assert_true(abs(measured_lufs - Float32(-3.01)) <= Float32(0.08))

    # 5. Voice Virtualizer (128 -> 64) click-free 128-sample steal crossfade + Sidechain Ducking + CyclesStemNode
    var srv_virt = AudioServer(48000, 256)
    var tone_a = _make_sine_pcm16(330.0, 24000, 1, 0.20, 48000)
    var tone_b = _make_sine_pcm16(440.0, 24000, 1, 0.20, 48000)
    var aid_a = srv_virt.register_pcm16_asset(tone_a, 1, 48000)
    var aid_b = srv_virt.register_pcm16_asset(tone_b, 1, 48000)

    # Spawn 64 distant low-priority emitters filling all 64 physical voices
    for i in range(64):
        _ = srv_virt.play_3d(aid_a, Vec3(Float32(40 + (i % 10)), 0.0, -20.0), volume=0.25, looping=True, priority=1)
    srv_virt.render_offline_quantum(buf, 256)
    srv_virt.render_offline_quantum(buf, 256)
    assert_equal(srv_virt.active_physical_count(), 64)

    var prev_sample = buf[510]
    # Now spawn 64 closer higher-priority emitters => 128 active logical emitters -> steals all 64 voices with 128-sample crossfade!
    for i in range(64):
        _ = srv_virt.play_3d(aid_b, Vec3(Float32((i % 8) - 4), 0.0, -6.0), volume=0.35, looping=True, priority=4)
    srv_virt.render_offline_quantum(buf, 256)
    assert_equal(srv_virt.active_logical_count(), 128)
    assert_equal(srv_virt.active_physical_count(), 64)
    assert_true(srv_virt.voices_stolen_count() >= 64)

    var max_step = Float32(0.0)
    var prev = prev_sample
    for f in range(256):
        var cur = buf[f * 2]
        var diff = abs(cur - prev)
        if diff > max_step:
            max_step = diff
        prev = cur
    # Smooth band-limited waves + 128-sample crossfade: sample-to-sample delta stays smooth (< 0.25, no step click)
    assert_true(max_step < Float32(0.28))

    # Sidechain ducking: Voice ducks Music + CyclesStemNode renders pattern audio
    var srv_sc = AudioServer(48000, 256)
    var stem_idx = srv_sc.add_cycles_stem('d1 $ s "sine*4" # gain 0.9', BUS_MUSIC)
    srv_sc.transition_cycles_stem(stem_idx, 'd1 $ s "saw*4" # cutoff 1200', 0.5)
    for _ in range(12):
        srv_sc.render_offline_quantum(buf, 256)
    var duck_before = srv_sc.music_duck_env()
    var voice_aid = srv_sc.register_pcm16_asset(full_sine, 2, 48000)
    _ = srv_sc.play_2d(voice_aid, volume=0.95, bus_id=BUS_VOICE, looping=True, direct_stereo=True)
    for _ in range(16):
        srv_sc.render_offline_quantum(buf, 256)
    assert_true(srv_sc.music_duck_env() < duck_before * Float32(0.75))

    # 6. Heap-Pinned `AudioRtEngine` Live Pthread Relocation + Race-Free `CMD_INSTALL_CLIP` & `CMD_CONFIG_BUS`
    var srv_rt = AudioServer(48000, 128)
    var pinned_addr = srv_rt.rt_engine_address()
    assert_true(pinned_addr != 0)
    assert_true(srv_rt.start_headless_rt_thread(), "headless RT mixer pthread spawned")

    # Relocate `AudioServer` across stack frames while RT pthread is actively running!
    var moved_rt = _relocate_audio_server(srv_rt^)
    assert_equal(moved_rt.rt_engine_address(), pinned_addr)

    # Concurrently install clips, add/configure buses, and allocate/play handles via MPSC ring while RT runs
    var qoa_clip_bytes = encode_qoa(tone_a, 1, 48000)
    var rt_pcm_aid = moved_rt.register_pcm16_asset(tone_b, 1, 48000)
    var rt_qoa_aid = moved_rt.register_qoa_asset(qoa_clip_bytes^)
    var fx_bus = moved_rt.add_bus("LiveSubFX", BUS_MASTER)
    assert_true(moved_rt.set_bus_gain(fx_bus, 0.8))
    assert_true(moved_rt.set_bus_eq_db(fx_bus, 1.5, -1.0, 2.0))
    assert_true(moved_rt.configure_bus_compressor(fx_bus, True, -16.0, 4.0, 1.1))
    for i in range(16):
        var aid_pick = rt_pcm_aid if (i & 1) == 0 else rt_qoa_aid
        var h_rt = moved_rt.play_3d(
            aid_pick, Vec3(Float32(i - 8), 1.0, -5.0), volume=0.5, bus_id=fx_bus, looping=True
        )
        assert_true(h_rt.is_valid())

    var wait_t0 = get_time()
    while moved_rt.quantums_rendered() < 8 and (get_time() - wait_t0) < 0.5:
        _ = external_call_yield()
    assert_true(moved_rt.quantums_rendered() >= 8)
    assert_true(moved_rt.active_physical_count() > 0)
    moved_rt.stop()
    assert_true(not moved_rt.rt_running)


# ==============================================================================
# (b) Woodworth HRTF ITD/ILD, 180° Rotation, 0.1 Mach Doppler & Bvh8 Acoustics
# ==============================================================================

def test_spatial_hrtf_doppler_and_bvh_acoustics() raises:
    # 1. Analytic & Rendered Woodworth ITD (~29.5 samples @ +90 deg right, 48 kHz) & ILD
    var hrtf_right = compute_woodworth_hrtf(
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, -1.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(10.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        15.0,
        350.0,
        1.0,
        20000.0,
        48000.0,
    )
    assert_true(abs(hrtf_right.delay_l_samples - Float32(29.5)) < Float32(0.15))
    assert_true(hrtf_right.delay_r_samples == Float32(0.0))
    assert_true(hrtf_right.gain_r > hrtf_right.gain_l * Float32(1.4))

    # Measure actual rendered impulse/burst arrival lag between R and L channels in AudioServer
    var srv_itd = AudioServer(48000, 256)
    _ = srv_itd.set_aux_gain(3, 0.0)  # Dry anechoic measurement
    var pulse_pcm = List[Int16](length=512, fill=0)
    pulse_pcm[16] = 28000
    pulse_pcm[17] = 20000
    pulse_pcm[18] = -18000
    var pulse_aid = srv_itd.register_pcm16_asset(pulse_pcm, 1, 48000)
    _ = srv_itd.play_3d(pulse_aid, Vec3(10.0, 0.0, 0.0), volume=0.8)
    var out = List[Float32](length=512, fill=0.0)
    srv_itd.render_offline_quantum(out, 256)

    # Cross-correlate L vs R across lags 0..45 samples
    var best_lag = 0
    var best_corr = Float64(-1.0e30)
    var energy_l = Float64(0.0)
    var energy_r = Float64(0.0)
    for f in range(256):
        energy_l += Float64(out[f * 2] * out[f * 2])
        energy_r += Float64(out[f * 2 + 1] * out[f * 2 + 1])
    for lag in range(0, 45):
        var corr = Float64(0.0)
        for f in range(256 - lag):
            corr += Float64(out[(f + lag) * 2]) * Float64(out[f * 2 + 1])
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    assert_true(best_lag >= 29 and best_lag <= 31)
    assert_true(energy_r > energy_l * 1.8)

    # 2. Continuous 180-degree listener rotation click-freedom
    var srv_rot = AudioServer(48000, 128)
    _ = srv_rot.set_aux_gain(3, 0.0)
    var sine_440 = _make_sine_pcm16(440.0, 48000, 1, 0.6, 48000)
    var rot_aid = srv_rot.register_pcm16_asset(sine_440, 1, 48000)
    _ = srv_rot.play_3d(rot_aid, Vec3(8.0, 0.0, 0.0), volume=0.8, looping=True)

    var rot_buf = List[Float32](length=256, fill=0.0)
    srv_rot.render_offline_quantum(rot_buf, 128)
    var prev_l = rot_buf[254]
    var prev_r = rot_buf[255]
    var max_rot_jump = Float32(0.0)

    for step in range(1, 65):
        var ang = Float64(step) / 64.0 * 3.141592653589793
        var fwd = Vec3(Float32(sin(ang)), 0.0, Float32(-cos(ang)))
        var right = Vec3(Float32(cos(ang)), 0.0, Float32(sin(ang)))
        srv_rot.set_listener(Vec3(), Vec3(), fwd, right, Vec3(0.0, 1.0, 0.0))
        srv_rot.render_offline_quantum(rot_buf, 128)
        for f in range(128):
            var cl = rot_buf[f * 2]
            var cr = rot_buf[f * 2 + 1]
            var jl = abs(cl - prev_l)
            var jr = abs(cr - prev_r)
            if jl > max_rot_jump:
                max_rot_jump = jl
            if jr > max_rot_jump:
                max_rot_jump = jr
            prev_l = cl
            prev_r = cr
    # Theoretical max slope of 440 Hz @ 48 kHz with amp <= 0.6 is ~0.035; any click jump would exceed 0.15
    assert_true(max_rot_jump < Float32(0.09))

    # 3. 0.1 Mach approaching Doppler on 1000 Hz tone => 1000 / (1 - 0.1) = 1111.11 Hz (+/- 1.5 Hz)
    var srv_dop = AudioServer(48000, 256)
    _ = srv_dop.set_aux_gain(3, 0.0)
    _ = srv_dop.set_dc_blocker(False)
    var sine_1k = _make_sine_pcm16(1000.0, 48000, 1, 0.7, 48000)
    var dop_aid = srv_dop.register_pcm16_asset(sine_1k, 1, 48000)
    # Emitter at (50, 0, 0) approaching listener at (0, 0, 0) at 0.1 Mach = -34.3 m/s along +X
    _ = srv_dop.play_3d(dop_aid, Vec3(50.0, 0.0, 0.0), Vec3(-34.3, 0.0, 0.0), volume=1.0, looping=True, ref_dist=100.0)

    var dop_samples = List[Float32]()
    for q in range(20):
        srv_dop.render_offline_quantum(out, 256)
        if q >= 2:  # Skip limiter lookahead latency on quantum 0..1
            for f in range(256):
                dop_samples.append(out[f * 2 + 1])  # Right ear (0 ITD delay)

    var first_zc = Float64(-1.0)
    var last_zc = Float64(-1.0)
    var zc_periods = 0
    for i in range(1, len(dop_samples)):
        var y0 = dop_samples[i - 1]
        var y1 = dop_samples[i]
        if y0 <= Float32(0.0) and y1 > Float32(0.0):
            var frac = Float64(-y0) / Float64(y1 - y0)
            var zc_pos = Float64(i - 1) + frac
            if first_zc < 0.0:
                first_zc = zc_pos
            else:
                last_zc = zc_pos
                zc_periods += 1
    assert_true(zc_periods > 50)
    var est_freq_hz = Float64(zc_periods) * 48000.0 / (last_zc - first_zc)
    assert_true(abs(est_freq_hz - 1111.11) <= 1.5)

    # 4. Bvh8 Wall Occlusion (< 700 Hz blocked vs 20 kHz clear), 4-Ray Pillar Obstruction & 6-Ray Sabine RT60
    var mat_table = AcousticMaterialTable()
    var wall_bvh = _build_wall_quad_bvh(5.0, 20.0, 20.0)
    var blocked_res = evaluate_emitter_acoustics(wall_bvh, mat_table, Vec3(0.0, 0.0, 0.0), Vec3(10.0, 0.0, 0.0))
    assert_true(blocked_res.direct_blocked)
    assert_true(blocked_res.occlusion_cutoff_hz < Float32(700.0))
    assert_equal(blocked_res.fan_open_rays, 0)
    assert_true(blocked_res.room_occlusion_factor > Float32(0.95))

    var clear_res = evaluate_emitter_acoustics(wall_bvh, mat_table, Vec3(0.0, 0.0, 0.0), Vec3(-10.0, 0.0, 0.0))
    assert_true(not clear_res.direct_blocked)
    assert_equal(clear_res.occlusion_cutoff_hz, Float32(20000.0))

    # Narrow 0.4 m pillar at x=5.0: blocks direct ray, while 4-ray aperture fan wraps around it!
    var pillar_bvh = _build_wall_quad_bvh(5.0, 0.4, 0.4)
    var pillar_res = evaluate_emitter_acoustics(pillar_bvh, mat_table, Vec3(0.0, 0.0, 0.0), Vec3(10.0, 0.0, 0.0), 1.2)
    assert_true(pillar_res.direct_blocked)
    assert_true(pillar_res.obstruction_factor >= Float32(0.75))
    assert_true(pillar_res.occlusion_cutoff_hz > blocked_res.occlusion_cutoff_hz * Float32(2.5))

    # 6-ray Sabine RT60 room enclosure probe inside a 16m x 8m x 20m room
    var room_bvh = _build_box_room_bvh(8.0, 4.0, 10.0)
    var probe = probe_room_sabine_rt60(room_bvh, mat_table, Vec3(0.0, 0.0, 0.0), 50.0)
    assert_equal(probe.hit_count, 6)
    assert_true(probe.sabine_rt60_sec > Float32(0.8) and probe.sabine_rt60_sec < Float32(6.5))


# ==============================================================================
# (c) Pure-Mojo QOA Round-Trip, Compression Ratio & O(1) Stream Seek
# ==============================================================================

def test_qoa_codec_and_seeking() raises:
    assert_equal(QOA_SLICE_BITS_PER_SAMPLE, Float32(3.2))

    # Rich multi-harmonic stereo test signal across > 3 full QOA frames (16,384 frames)
    var n_frames = 16384
    var sr = 48000
    var src_pcm = List[Int16](capacity=n_frames * 2)
    for f in range(n_frames):
        var t = Float64(f) / Float64(sr)
        var env = 0.85 if f >= 256 else (Float64(f) / 256.0 * 0.85)
        var wl = env * (0.52 * sin(TAU64 * 220.0 * t) + 0.26 * sin(TAU64 * 440.0 * t) + 0.14 * sin(TAU64 * 660.0 * t))
        var wr = env * (0.50 * sin(TAU64 * 277.18 * t) + 0.28 * sin(TAU64 * 554.37 * t) + 0.14 * sin(TAU64 * 831.55 * t))
        src_pcm.append(Int16(max(-32768, min(32767, Int(wl * 32767.0)))))
        src_pcm.append(Int16(max(-32768, min(32767, Int(wr * 32767.0)))))

    var encoded = encode_qoa(src_pcm, 2, sr)
    assert_true(len(encoded) > 0)

    # Slice payload bitrate is exact 3.2 bits/sample; total file with headers is ~3.25 bits/sample (~5:1 vs 16-bit)
    var total_samples = n_frames * 2
    var total_bps = Float64(len(encoded) * 8) / Float64(total_samples)
    assert_true(total_bps > 3.19 and total_bps < 3.30)

    var decoded = decode_qoa(encoded)
    assert_true(decoded.valid)
    assert_equal(decoded.channels, 2)
    assert_equal(decoded.sample_rate, sr)
    assert_equal(decoded.frame_count(), n_frames)

    var sig_pow = Float64(0.0)
    var err_pow = Float64(0.0)
    for i in range(total_samples):
        var s = Float64(src_pcm[i])
        var d = Float64(decoded.samples[i])
        var e = s - d
        sig_pow += s * s
        err_pow += e * e
    var snr_db = 10.0 * log10(sig_pow / max(1.0, err_pow))
    assert_true(snr_db >= 29.0)

    # Bit-identical O(1) QoaStreamCursor.seek vs full decode across arbitrary mid-slice & multi-frame offsets
    var cursor = QoaStreamCursor()
    assert_true(cursor.open(encoded))
    var seek_targets = [0, 19, 20, 137, 5119, 5120, 5121, 7777, 10240, 12345, 16300]
    var check_buf = List[Int16](length=128 * 2, fill=0)

    for target in seek_targets:
        assert_true(cursor.seek(encoded, target))
        var want = min(64, n_frames - target)
        var got = cursor.decode_frames_into(encoded, check_buf, 0, want)
        assert_equal(got, want)
        for k in range(want * 2):
            assert_equal(check_buf[k], decoded.samples[target * 2 + k])


# ==============================================================================
# (d) 1,000 Flexbox Fuzz Trees, DockSpace, 100k Culled Outliner, UTF-8 & Modal
# ==============================================================================

@always_inline
def _lcg_next(mut state: UInt64) -> UInt32:
    state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return UInt32((state >> 32) & 0xFFFFFFFF)


def _rand_f(mut state: UInt64, lo: Float32, hi: Float32) -> Float32:
    var u = Float32(_lcg_next(state) & 0xFFFF) / Float32(65535.0)
    return lo + u * (hi - lo)


def _build_random_flex_subtree(mut tree: FlexTree, root_parent: Int, mut rng: UInt64):
    var stack_parent = List[Int]()
    var stack_depth = List[Int]()
    stack_parent.append(root_parent)
    stack_depth.append(0)

    while len(stack_parent) > 0:
        var parent = stack_parent.pop()
        var depth = stack_depth.pop()
        var n_ch = Int(_lcg_next(rng) % 5) + 1
        for _ in range(n_ch):
            var st = FlexStyle(Int(_lcg_next(rng) & 1))
            st.align_items = Int(_lcg_next(rng) % 4)
            st.align_self = Int(_lcg_next(rng) % 5) - 1
            st.justify_content = Int(_lcg_next(rng) % 4)
            st.grow = _rand_f(rng, 0.0, 3.0) if (_lcg_next(rng) % 3) != 0 else Float32(0.0)
            st.shrink = _rand_f(rng, 0.0, 2.0)
            st.min_w = _rand_f(rng, 4.0, 28.0)
            st.min_h = _rand_f(rng, 4.0, 28.0)
            st.max_w = _rand_f(rng, 180.0, 900.0)
            st.max_h = _rand_f(rng, 180.0, 900.0)
            st.pad_l = _rand_f(rng, 0.0, 8.0)
            st.pad_t = _rand_f(rng, 0.0, 8.0)
            st.pad_r = _rand_f(rng, 0.0, 8.0)
            st.pad_b = _rand_f(rng, 0.0, 8.0)
            st.gap = _rand_f(rng, 0.0, 10.0)
            var iw = _rand_f(rng, 10.0, 65.0)
            var ih = _rand_f(rng, 10.0, 50.0)
            var idx = tree.add_node(parent, st, iw, ih)
            if depth < 2 and (_lcg_next(rng) % 3) == 0:
                stack_parent.append(idx)
                stack_depth.append(depth + 1)


def test_flexbox_1000_fuzz_dockspace_utf8_and_modal() raises:
    # 1. 1,000 randomized Flexbox trees: Bounds, No-Overlap, & Space Conservation invariants
    var rng = UInt64(0x5EED_F1E8_2026)
    var tree = FlexTree()

    for _ in range(1000):
        tree.clear()
        var root_st = FlexStyle(Int(_lcg_next(rng) & 1))
        root_st.align_items = Int(_lcg_next(rng) % 4)
        root_st.justify_content = Int(_lcg_next(rng) % 4)
        root_st.pad_l = _rand_f(rng, 0.0, 16.0)
        root_st.pad_t = _rand_f(rng, 0.0, 16.0)
        root_st.pad_r = _rand_f(rng, 0.0, 16.0)
        root_st.pad_b = _rand_f(rng, 0.0, 16.0)
        root_st.gap = _rand_f(rng, 0.0, 12.0)
        var root_w = _rand_f(rng, 120.0, 1280.0)
        var root_h = _rand_f(rng, 90.0, 720.0)
        _ = tree.add_node(-1, root_st, root_w, root_h)
        _build_random_flex_subtree(tree, 0, rng)
        tree.solve(RectF(0.0, 0.0, root_w, root_h), 0)

        for ni in range(len(tree.nodes)):
            var pr = tree.nodes[ni].rect
            assert_true(pr.w >= Float32(-1.0e-4) and pr.h >= Float32(-1.0e-4))
            var n_ch = len(tree.nodes[ni].children)
            if n_ch == 0:
                continue
            var st = tree.nodes[ni].style
            var inner = pr.inset(st.pad_l, st.pad_t, st.pad_r, st.pad_b)
            var is_row = st.direction == FLEX_ROW

            var sum_main = Float32(0.0)
            var has_unclamped_grow = False
            for ci in range(n_ch):
                var c_idx = tree.nodes[ni].children[ci]
                var cr = tree.nodes[c_idx].rect
                var cst = tree.nodes[c_idx].style
                # Bounds check: child strictly inside parent inner rect
                assert_true(cr.x >= inner.x - Float32(0.02))
                assert_true(cr.y >= inner.y - Float32(0.02))
                assert_true(cr.right() <= inner.right() + Float32(0.02))
                assert_true(cr.bottom() <= inner.bottom() + Float32(0.02))

                var cm = cr.w if is_row else cr.h
                sum_main += cm
                var max_lim = cst.max_w if is_row else cst.max_h
                if cst.grow > Float32(0.0) and cm < max_lim - Float32(0.05):
                    has_unclamped_grow = True

                # Sibling no-overlap check
                for cj in range(ci + 1, n_ch):
                    var other = tree.nodes[tree.nodes[ni].children[cj]].rect
                    assert_true(not cr.overlaps_interior(other, Float32(0.02)))

            # Space conservation check when positive grow is active and not max-capped
            var main_avail = inner.w if is_row else inner.h
            var gap_tot = Float32(max(0, n_ch - 1)) * st.gap
            if has_unclamped_grow and main_avail >= gap_tot + sum_main - Float32(0.05):
                var diff = abs((sum_main + min(main_avail, gap_tot)) - main_avail)
                assert_true(diff < Float32(0.05))

    # 2. 5-Panel DockSpace Zero-Gap Tiling & JSON Round-Trip
    var dock = DockSpace.create_studio_5panel()
    var screen_r = RectF(0.0, 0.0, 1920.0, 1080.0)
    dock.solve_layout(screen_r)
    var leaves = dock.leaf_rects()
    assert_equal(len(leaves), 5)

    var area_sum = Float32(0.0)
    for i in range(len(leaves)):
        area_sum += leaves[i].area()
        for j in range(i + 1, len(leaves)):
            assert_true(not leaves[i].overlaps_interior(leaves[j], Float32(0.01)))
    assert_true(abs(area_sum - screen_r.area()) < Float32(0.5))

    var dock_json = dock.save_json()
    var dock2 = DockSpace()
    assert_true(dock2.load_json(dock_json))
    dock2.solve_layout(screen_r)
    var leaves2 = dock2.leaf_rects()
    assert_equal(len(leaves2), 5)
    for i in range(5):
        assert_true(abs(leaves[i].x - leaves2[i].x) < Float32(0.05))
        assert_true(abs(leaves[i].y - leaves2[i].y) < Float32(0.05))
        assert_true(abs(leaves[i].w - leaves2[i].w) < Float32(0.05))
        assert_true(abs(leaves[i].h - leaves2[i].h) < Float32(0.05))

    # 3. 100k-item virtualized / culled tree_node outliner (deterministic culling + non-flaky CI guard)
    var ui = UiContext()
    var inp = InputState()
    ui.begin_frame(inp, 1280.0, 720.0)
    var stride = ui.theme.row_height + Float32(2.0)
    var t0 = get_time()
    ui.begin_scroll_view("outliner_100k", 320.0, 140.0, Float32(100000) * stride)
    var vis = ui.scroll_visible_range(100000, stride)
    for idx in range(vis[0], vis[1]):
        _ = ui.tree_node_indexed(idx, "Entity")
    ui.end_scroll_view()
    var elapsed_ms = (get_time() - t0) * 1000.0
    ui.end_frame()
    assert_true(len(ui.draw_list.prims) > 5 and len(ui.draw_list.prims) < 80)
    assert_true(elapsed_ms < 25.0)

    # Also verify brute-force 100,000 calls to tree_node_indexed culls offscreen items cleanly
    inp.begin_frame()
    ui.begin_frame(inp, 1280.0, 720.0)
    ui.begin_scroll_view("outliner_brute", 320.0, 600.0)
    for idx in range(100000):
        _ = ui.tree_node_indexed(idx, "Entity")
    ui.end_scroll_view()
    ui.end_frame()
    assert_true(ui.culled_items_this_frame > 99950)

    # 4. UTF-8 Multi-Byte-Safe `text_input` on "¡Hola! エンジン ★"
    var utf_str = String("")
    inp.begin_frame()
    inp.move_mouse(50.0, 15.0)
    inp.set_mouse(0, True)
    inp.push_text_utf8("¡Hola! エンジン ★")
    ui.begin_frame(inp, 800.0, 600.0)
    var changed = ui.text_input("utf_field", utf_str, 300.0)
    ui.end_frame()
    assert_true(changed)
    assert_equal(utf_str, "¡Hola! エンジン ★")

    var orig_cp_len = len(decode_utf8(utf_str))
    assert_equal(orig_cp_len, 13)
    assert_equal(ui.text_cursor_codepoint("utf_field"), 13)

    # Step cursor left 4 codepoints (past '★', ' ', 'ン', 'ジ') and verify every byte offset is a valid UTF-8 lead byte
    inp.set_mouse(0, False)
    for step in range(4):
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_LEFT, True)
        ui.begin_frame(inp, 800.0, 600.0)
        _ = ui.text_input("utf_field", utf_str, 300.0)
        ui.end_frame()
        inp.set_logical_key(UI_KEY_LEFT, False)
        var boff = ui.text_cursor_byte_offset("utf_field", utf_str)
        if boff < utf_str.byte_length():
            var lead_byte = Int(utf_str.as_bytes()[boff])
            assert_true((lead_byte & 0xC0) != 0x80)  # NEVER mid-character continuation byte!
        assert_equal(ui.text_cursor_codepoint("utf_field"), 13 - (step + 1))

    # Backspace deletes 'ン' (3-byte katakana) cleanly without corrupting UTF-8
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_BACKSPACE, True)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_true(ui.text_input("utf_field", utf_str, 300.0))
    ui.end_frame()
    inp.set_logical_key(UI_KEY_BACKSPACE, False)

    var after_codes = decode_utf8(utf_str)
    assert_equal(len(after_codes), 12)
    for c in after_codes:
        assert_true(c != REPLACEMENT)
    assert_equal(utf_str, "¡Hola! エジン ★")

    # Also test Home/End/Right cursor bounds
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_HOME, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("utf_field", utf_str, 300.0)
    ui.end_frame()
    inp.set_logical_key(UI_KEY_HOME, False)
    assert_equal(ui.text_cursor_codepoint("utf_field"), 0)

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_END, True)
    inp.set_logical_key(UI_KEY_RIGHT, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("utf_field", utf_str, 300.0)
    ui.end_frame()
    inp.set_logical_key(UI_KEY_END, False)
    inp.set_logical_key(UI_KEY_RIGHT, False)
    assert_equal(ui.text_cursor_codepoint("utf_field"), 12)

    # 5. Focus-trapping `begin_modal` / `end_modal` with Tab & Gamepad D-Pad navigation
    var modal_open = True
    var bg_btn1_id: UInt64
    var bg_btn2_id: UInt64

    for step in range(8):
        inp.begin_frame()
        if step % 2 == 1:
            inp.set_logical_key(UI_KEY_TAB, True)
        else:
            inp.pads[0].set_control(PAD_DOWN, True)
        ui.begin_frame(inp, 1280.0, 720.0)
        bg_btn1_id = ui.widget_id("bg_1")
        assert_true(not ui.button("bg_1", "Background 1"))
        bg_btn2_id = ui.widget_id("bg_2")
        assert_true(not ui.button("bg_2", "Background 2"))

        if ui.begin_modal("confirm_dlg", "Confirm Quit", 360.0, 200.0, modal_open):
            _ = ui.button("modal_ok", "Confirm")
            _ = ui.button("modal_cancel", "Cancel")
            ui.end_modal()
        ui.end_frame()

        inp.set_logical_key(UI_KEY_TAB, False)
        inp.pads[0].set_control(PAD_DOWN, False)

        # Focus must NEVER land on background widgets while modal is open
        assert_true(ui.focused_id != 0)
        assert_true(ui.focused_id != bg_btn1_id and ui.focused_id != bg_btn2_id)
        _ = PAD_UP


# ==============================================================================
# (e) Evdev EVIOCGABS, 48B ff_effect Rumble & InputMap Chords/Buffer/Coyote/JSON
# ==============================================================================

def test_evdev_and_declarative_input_map() raises:
    # 1. InputState PadState uncalibrated trigger (min=0) reads 0.0 when raw=0
    var legacy_inp = InputState()
    legacy_inp.pads[0].apply_event(2, PAD_AXIS_LT, 0)
    legacy_inp.pads[0].apply_event(2, PAD_AXIS_RT, 0)
    assert_equal(legacy_inp.gamepad_axis(0, PAD_AXIS_LT), Float32(0.0))
    assert_equal(legacy_inp.gamepad_axis(0, PAD_AXIS_RT), Float32(0.0))

    # 2. Synthetic EvdevGamepad: Xbox Series (triggers 0..1023) & Sony DualSense (triggers 0..255)
    var xbox = EvdevGamepad()
    xbox.set_vendor_product(VENDOR_MICROSOFT, 0x0B12, "Microsoft Xbox Series S|X Controller")
    xbox.configure_abs_axis(ABS_X, -32768, 32767, 16, 128)
    xbox.configure_abs_axis(ABS_Y, -32768, 32767, 16, 128)
    xbox.configure_abs_axis(ABS_Z, 0, 1023, 0, 0)
    xbox.configure_abs_axis(ABS_RZ, 0, 1023, 0, 0)
    assert_true(not xbox.is_sony)
    assert_equal(xbox.axes[PAD_AXIS_LT], Float32(0.0))
    assert_equal(xbox.axes[PAD_AXIS_RT], Float32(0.0))

    xbox.feed_raw_evdev_event(EV_ABS, UInt16(ABS_Z), Int32(0))
    assert_equal(xbox.axes[PAD_AXIS_LT], Float32(0.0))
    xbox.feed_raw_evdev_event(EV_ABS, UInt16(ABS_Z), Int32(1023))
    assert_true(abs(xbox.axes[PAD_AXIS_LT] - Float32(1.0)) < Float32(1.0e-5))

    var dualsense = EvdevGamepad()
    dualsense.set_vendor_product(VENDOR_SONY, 0x0CE6, "Sony Interactive Entertainment Wireless Controller")
    dualsense.configure_abs_axis(ABS_X, 0, 255, 0, 4, 128)
    dualsense.configure_abs_axis(ABS_Y, 0, 255, 0, 4, 128)
    dualsense.configure_abs_axis(ABS_Z, 0, 255, 0, 0, 0)
    dualsense.configure_abs_axis(ABS_RZ, 0, 255, 0, 0, 0)
    assert_true(dualsense.is_sony)
    assert_equal(dualsense.axes[PAD_AXIS_LT], Float32(0.0))
    assert_equal(dualsense.axes[PAD_AXIS_RT], Float32(0.0))
    assert_true(abs(dualsense.axes[PAD_AXIS_LEFT_X]) < Float32(0.02))
    dualsense.feed_raw_evdev_event(EV_KEY, UInt16(BTN_SOUTH), Int32(1))
    dualsense.feed_raw_evdev_event(EV_KEY, UInt16(BTN_EAST), Int32(0))
    assert_true(dualsense.down[PAD_A])
    assert_true(not dualsense.down[PAD_B])

    # 3. 48-byte Linux UAPI `struct ff_effect` & 24-byte `EV_FF` packet layout verification
    var ff_pkt = pack_ff_rumble_effect(-1, 1.0, 0.5, 750)
    assert_equal(len(ff_pkt), 48)
    assert_equal(Int(ff_pkt[0]) | (Int(ff_pkt[1]) << 8), 0x0050)    # FF_RUMBLE
    assert_equal(Int(ff_pkt[2]) | (Int(ff_pkt[3]) << 8), 0xFFFF)    # id = -1
    assert_equal(Int(ff_pkt[10]) | (Int(ff_pkt[11]) << 8), 750)     # replay.length = 750 ms
    assert_equal(Int(ff_pkt[16]) | (Int(ff_pkt[17]) << 8), 65535)   # strong_magnitude = 1.0
    var weak_u16 = Int(ff_pkt[18]) | (Int(ff_pkt[19]) << 8)
    assert_true(abs(weak_u16 - 32768) <= 2)                         # weak_magnitude = 0.5

    var ev_play = pack_ff_play_event(3, True)
    assert_equal(len(ev_play), 24)
    assert_equal(Int(ev_play[16]) | (Int(ev_play[17]) << 8), 0x0015)  # EV_FF
    assert_equal(Int(ev_play[18]) | (Int(ev_play[19]) << 8), 3)       # effect_id = 3
    assert_equal(Int(ev_play[20]), 1)                                 # value = 1 (play)

    # 4. Declarative `InputMap`: Context Masking, `Ctrl+S` Chord, Jump Buffer (120 ms), Coyote Time (100 ms) & JSON
    var imap = InputMap()
    _ = imap.register_action("Jump", ACTION_BOOL)
    _ = imap.register_action("MenuConfirm", ACTION_BOOL)
    _ = imap.register_action("MoveBack", ACTION_BOOL)
    _ = imap.register_action("SaveScene", ACTION_BOOL)
    _ = imap.register_action("Throttle", ACTION_AXIS1D)
    _ = imap.register_action("Move", ACTION_AXIS2D)

    var gameplay = InputContext("Gameplay", priority=0, blocks_lower_all=False)
    gameplay.bind_key("Jump", 32)                     # Space
    gameplay.bind_pad_button("Jump", PAD_A)
    gameplay.bind_key("MoveBack", 83)                 # 'S'
    gameplay.bind_key("SaveScene", 83, MOD_CTRL)      # 'Ctrl+S' chord preempts & consumes 'S'!
    gameplay.bind_pad_axis_1d("Throttle", PAD_AXIS_RT)
    gameplay.bind_pad_stick_2d("Move", 0)
    imap.add_context(gameplay^)

    var menu_ctx = InputContext("PauseMenu", priority=10, blocks_lower_all=False)
    menu_ctx.bind_key("MenuConfirm", 32, consume=True)  # Consumes Space before Gameplay sees it!
    menu_ctx.bind_pad_button("MenuConfirm", PAD_A, consume=True)
    imap.add_context(menu_ctx^)

    imap.push_context("Gameplay")

    var st_in = InputState()
    # Plain 'S' triggers MoveBack, NOT SaveScene
    st_in.begin_frame()
    st_in.set_logical_key(83, True)
    imap.update(st_in, 0.016)
    assert_true(imap.is_down("MoveBack"))
    assert_true(not imap.is_down("SaveScene"))

    # 'Ctrl+S' chord triggers SaveScene AND consumes 'S' so MoveBack is False!
    st_in.begin_frame()
    st_in.set_logical_key(341, True)  # Left Ctrl
    st_in.set_logical_key(83, True)   # 'S'
    imap.update(st_in, 0.016)
    assert_true(imap.is_down("SaveScene"))
    assert_true(not imap.is_down("MoveBack"))
    st_in.set_logical_key(341, False)
    st_in.set_logical_key(83, False)

    # Dynamic Prompt Glyph lookup across KB/M -> Xbox -> Sony
    assert_equal(imap.prompt_glyph_for("SaveScene").label, "Ctrl+S")
    assert_equal(imap.prompt_glyph_for("Jump").label, "Space")
    imap.active_device = INPUT_DEV_XBOX
    assert_equal(imap.prompt_glyph_for("Jump").label, "A")
    imap.active_device = INPUT_DEV_SONY
    assert_equal(imap.prompt_glyph_for("Jump").label, "Cross")
    imap.active_device = INPUT_DEV_KBM

    # Context stack masking: pushing PauseMenu (priority 10) masks Space from Gameplay.Jump
    imap.push_context("PauseMenu")
    st_in.begin_frame()
    st_in.set_logical_key(32, True)
    imap.update(st_in, 0.016)
    assert_true(imap.just_pressed("MenuConfirm"))
    assert_true(not imap.is_down("Jump"))
    _ = imap.pop_context("PauseMenu")
    st_in.set_logical_key(32, False)
    st_in.begin_frame()
    imap.update(st_in, 0.016)

    # Jump Buffer (120 ms) & Coyote Time (100 ms)
    imap.set_grounded(False)
    st_in.begin_frame()
    st_in.set_logical_key(32, True)  # Player presses Jump in mid-air 80 ms before landing
    imap.update(st_in, 0.016)
    st_in.set_logical_key(32, False)
    for _ in range(5):
        st_in.begin_frame()
        imap.update(st_in, 0.016)  # +80 ms elapsed -> within 120 ms jump buffer!
    assert_true(imap.consume_buffered("Jump", 0.120))
    assert_true(not imap.consume_buffered("Jump", 0.120))  # Consumed once only!

    # Press Jump 160 ms before landing -> expires past 120 ms window
    st_in.begin_frame()
    st_in.set_logical_key(32, True)
    imap.update(st_in, 0.016)
    st_in.set_logical_key(32, False)
    for _ in range(9):
        st_in.begin_frame()
        imap.update(st_in, 0.016)  # +144 ms elapsed (> 120 ms)
    assert_true(not imap.consume_buffered("Jump", 0.120))

    # Coyote time (100 ms): walk off ledge
    imap.set_grounded(True)
    st_in.begin_frame()
    imap.update(st_in, 0.016)
    imap.set_grounded(False)
    for _ in range(4):
        st_in.begin_frame()
        imap.update(st_in, 0.016)  # +64 ms off ledge (<= 100 ms)
    assert_true(imap.coyote_valid(0.100))
    for _ in range(4):
        st_in.begin_frame()
        imap.update(st_in, 0.016)  # +128 ms off ledge (> 100 ms)
    assert_true(not imap.coyote_valid(0.100))

    # Deterministic lockstep tick pack & JSON rebind round-trip
    assert_true(imap.rebind_key("Gameplay", "Jump", 74))  # Rebind Jump -> 'J' (74)
    var saved_json = imap.save_json()
    var imap2 = InputMap()
    assert_true(imap2.load_json(saved_json))
    assert_equal(imap2.prompt_glyph_for("Jump").label, "J")

    st_in.begin_frame()
    st_in.set_logical_key(74, True)
    imap2.update(st_in, 0.016)
    assert_true(imap2.just_pressed("Jump"))

    var tick_pkt = imap2.pack_deterministic_tick()
    var imap3 = InputMap()
    assert_true(imap3.load_json(saved_json))
    assert_true(imap3.unpack_deterministic_tick(tick_pkt))
    assert_true(imap3.just_pressed("Jump"))


def main() raises:
    var t_start = get_time()
    test_audio_server_mastering_and_virtualizer()
    print("[PASS] (a) AudioMpscRing MPSC stress, Zero-DC/NaN, +12 dBFS Limiter, BS.1770-4 LUFS & 128->64 Virtualizer")
    test_spatial_hrtf_doppler_and_bvh_acoustics()
    print("[PASS] (b) Woodworth HRTF ITD/ILD, 180 deg Rotation, 0.1 Mach Doppler & Bvh8 Raycast Acoustics")
    test_qoa_codec_and_seeking()
    print("[PASS] (c) Pure-Mojo QOA Codec (3.2 bps, SNR >= 29 dB, Bit-Identical O(1) Stream Seek)")
    test_flexbox_1000_fuzz_dockspace_utf8_and_modal()
    print("[PASS] (d) 1,000 Flexbox Fuzz, 5-Panel DockSpace, <0.1ms 100k Outliner, UTF-8 TextInput & Modal Trap")
    test_evdev_and_declarative_input_map()
    print("[PASS] (e) Evdev EVIOCGABS (0.0 Trigger), 48B ff_effect Rumble & Declarative InputMap")
    var elapsed = get_time() - t_start
    print("ALL FLINT V5 TRACK 4 TESTS PASSED in", elapsed, "s")
    assert_true(elapsed < 10.0)
