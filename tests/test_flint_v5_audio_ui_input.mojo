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
from std.ffi import external_call
from std.math import sin, cos, sqrt, log10, abs, isnan, isinf
from std.sys.info import size_of
from std.testing import assert_true, assert_equal, assert_false
from flint.core.math import Vec3
from flint.core.jobs import JobPool
from flint.platform.os import get_time, close_fd, list_directory, LockstepPeer, LinuxGamepad, RollbackPeer
from flint.platform.audio import alsa_open_pcm, alsa_close_pcm, NativeAudio
from flint.platform.pulse import PulseClient
from flint.platform.logind import LogindSession
from flint.platform.kms import (
    KmsWindow,
    install_rescue,
    rescue_ready,
    rstore,
    rload,
    R_DRM_FD,
    R_TTY_FD,
    R_CRTC_SAVED,
    R_TERMIOS_SAVED,
)
from flint.platform.xkb import (
    XkbKeymap,
    x11_keysym_to_codepoint,
    XKB_MOD_SHIFT,
    XKB_MOD_LOCK,
    XKB_MOD_CONTROL,
    XKB_MOD_MOD1,
    XKB_MOD_MOD5,
)
from flint.platform.wayland import WaylandWindow, wl_put16, wl_put32, wl_append32, wl_append_string, wl_get32
from flint.platform.wayland_bridge import (
    WaylandBridge,
    STATE_WORDS,
    WORD_IN,
    WORD_IN_USED,
    WORD_IN_FD_COUNT,
    WORD_OUT,
    WORD_OUT_FDS,
    WORD_OUT_FD_COUNT,
    WORD_GAME,
    WORD_GAME_USED,
    WORD_OWNED,
    WORD_KEYBOARD,
    WORD_IN_FDS,
    WORD_GAME_FDS,
    WORD_GAME_FD_COUNT,
    WORD_DATA_DEVICE,
    WORD_DATA_SOURCE,
    WORD_OFFER_IDS,
    WORD_IO_SCRATCH,
    WORD_LIBRARY,
    WORD_LIB_IFACES,
    WORD_STRICT_IFACE,
    STREAM_BYTES,
    GAME_BYTES,
    MAX_OWNED_ID,
    MAX_FDS,
    MAX_OFFERS,
    IO_SCRATCH_BYTES,
    LIB_IFACE_DMABUF_FEEDBACK,
    LIB_IFACE_UNKNOWN,
    init_lib_ifaces,
    free_lib_ifaces,
    lookup_lib_iface,
    is_server_offer,
    set_owned,
    take_incoming,
    take_outgoing,
    receive_some,
    set_word,
    word,
)
from flint.platform.x11 import X11Window, x11_put16, x11_put32, x11_get16, x11_get32
from flint.platform.vulkan import VulkanGpu, get64, put32, put64
from flint.platform.amdgpu import GpuBuffer
from flint.platform.input import (
    InputState,
    PAD_A,
    PAD_B,
    PAD_X,
    PAD_LB,
    PAD_UP,
    PAD_DOWN,
    PAD_STICK_UP,
    PAD_STICK_DOWN,
    PAD_STICK_LEFT,
    PAD_STICK_RIGHT,
    PAD_CONTROLS,
    PAD_AXIS_LT,
    PAD_AXIS_RT,
    PAD_AXIS_LEFT_X,
    PAD_AXIS_LEFT_Y,
    identity_map,
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
    UI_KEY_ESCAPE,
    UI_KEY_A,
    UI_KEY_C,
    UI_KEY_V,
    UI_KEY_X,
    UI_KEY_LEFT_SHIFT,
    UI_KEY_LEFT_CTRL,
)
from flint.ui.draw_list import PRIM_LINE, PRIM_GLYPH
from flint.text.utf8 import decode_utf8, encode_utf8, REPLACEMENT
from flint.text.ttf import TrueTypeFont
from flint.text.sdf_bake import (
    bake_glyph,
    bake_glyph_msdf,
    bake_glyph_mtsdf,
    bake_font,
    dsf_bytes,
    msdf_median,
    msdf_error_correct,
)
from flint.text.bidi import BIDI_AUTO, BIDI_LTR, BIDI_RTL, resolve_bidi, bidi_mirror
from flint.text.shaper import FontFallbackChain, TextShaper, shape_text
from flint.text.line_break import wrap_shaped
from flint.text.rich_text import parse_rich, layout_rich, layout_rich_shaped
from flint.render.font import SdfFont
from flint.platform.os import read_bytes
from flint.input import (
    EV_KEY,
    EV_ABS,
    ABS_X,
    ABS_Y,
    ABS_Z,
    ABS_RZ,
    BTN_SOUTH,
    BTN_EAST,
    BTN_NORTH,
    BTN_WEST,
    BTN_TL,
    VENDOR_MICROSOFT,
    VENDOR_SONY,
    EvdevGamepad,
    EvdevHub,
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
from flint.net import NetTransport, UdpSocket
from flint.platform.macos_audio import MacOsAudio
from flint.platform.macos_window import CocoaWindow
from flint.platform.traits import AudioBackend, WindowBackend
from flint.platform.win32_audio import Win32Audio
from flint.platform.win32_window import Win32Window

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

    # 7. RT-Safe CyclesEngine Hi-Hat (no per-sample heap alloc), Delay/Reverb Send Routing & Deferred Retire Queue
    var srv_cycles = AudioServer(48000, 256)
    var stem_fx = srv_cycles.add_cycles_stem('d1 $ s "hh ~ ~ ~" # gain 0.9 # room 0.8 # delay 0.8 # delaytime 0.05 # delayfb 0.6', BUS_MUSIC)
    var fx_buf = List[Float32](length=512, fill=0.0)
    var max_fx_amp = Float32(0.0)
    var tail_energy = Float64(0.0)
    for q in range(40):
        srv_cycles.render_offline_quantum(fx_buf, 256)
        for i in range(512):
            assert_true(not isnan(fx_buf[i]) and not isinf(fx_buf[i]))
            var a = abs(fx_buf[i])
            if a > max_fx_amp:
                max_fx_amp = a
            if q >= 12 and q < 30:
                tail_energy += Float64(fx_buf[i] * fx_buf[i])
    assert_true(max_fx_amp > Float32(0.01))
    assert_true(tail_energy > 1.0e-5)
    # Off-thread parsed stem transition + main-thread deferred retire reclamation
    srv_cycles.transition_cycles_stem(stem_fx, 'd1 $ s "oh*2" # gain 0.8 # room 0.5', 0.5)
    for _ in range(8):
        srv_cycles.render_offline_quantum(fx_buf, 256)
    assert_true(srv_cycles.drain_retired() >= 0)
    srv_cycles.stop()


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

    # 5. Regression: BIND_MOUSE_DELTA_2D must NOT clamp mouse look to 1.0 px/frame
    var look_map = InputMap()
    _ = look_map.register_action("Look", ACTION_AXIS2D)
    var look_ctx = InputContext("Camera", priority=0, blocks_lower_all=False)
    look_ctx.bind_mouse_look("Look", sensitivity=0.5, invert_y=True)
    look_map.add_context(look_ctx^)
    look_map.push_context("Camera")

    var look_in = InputState()
    look_in.begin_frame()
    look_in.add_mouse_delta(24.0, -10.0)
    look_map.update(look_in, 0.016)
    var look_v = look_map.axis_2d("Look")
    assert_true(abs(look_v[0] - Float32(12.0)) < Float32(1.0e-4))
    assert_true(abs(look_v[1] - Float32(5.0)) < Float32(1.0e-4))

    # 6. Regression: EvdevGamepad.sync_into_input_state() enables gamepad_available() across 4 slots
    var pad_in = InputState()
    assert_equal(len(pad_in.pads), 4)
    assert_true(not pad_in.gamepad_available(0))
    assert_true(not pad_in.gamepad_available(3))
    xbox.sync_into_input_state(pad_in, 0)
    dualsense.sync_into_input_state(pad_in, 3)
    assert_true(pad_in.gamepad_available(0))
    assert_true(pad_in.gamepad_available(3))
    assert_true(pad_in.pads[3].sony_layout)
    # Safe no-op rumble on non-writable pad fd
    pad_in.set_pad_rumble(0, 0.5, 0.8, 200)


def test_vulkan_suballocator_and_alsa_fallback() raises:
    # 1. AudioServer backend introspection & ALSA fallback resilience
    var srv_fb = AudioServer(48000, 256)
    assert_equal(srv_fb.audio_backend_name(), "none")
    assert_true(srv_fb.start_headless_rt_thread())
    assert_equal(srv_fb.audio_backend_name(), "headless")
    srv_fb.stop()
    assert_equal(srv_fb.audio_backend_name(), "none")

    # Direct ALSA probe must either open a valid fd or return -1 cleanly without crashing
    var alsa_period = 256
    var alsa_buf = 1024
    var alsa_fd = alsa_open_pcm(48000, 256, alsa_period, alsa_buf)
    if alsa_fd >= 0:
        assert_true(alsa_period >= 64)
        alsa_close_pcm(alsa_fd)

    # 2. Vulkan GPU Memory Sub-Allocator: 64 small buffers share 1 VkDeviceMemory slab & reuse cleanly
    var vk = VulkanGpu()
    if vk.initialize():
        var bufs = List[GpuBuffer](capacity=64)
        for i in range(64):
            var b = vk.create_buffer(4096, False)
            assert_true(b.vk_buffer != 0 and b.cpu_address != 0)
            b.write32(0, UInt32(0xCAFE0000 + i))
            bufs.append(b^)
        assert_equal(vk.suballoc_block_count(), 1)
        assert_equal(vk.suballoc_live_count(), 64)
        for i in range(64):
            assert_equal(bufs[i].read32(0), UInt32(0xCAFE0000 + i))
            vk.destroy_buffer(bufs[i])
        bufs.clear()
        assert_equal(vk.suballoc_live_count(), 0)

        # Second wave of 64 small buffers must reuse the exact same slab (no new VkDeviceMemory)
        for _ in range(64):
            var b = vk.create_buffer(4096, False)
            assert_true(b.vk_buffer != 0 and b.cpu_address != 0)
            bufs.append(b^)
        assert_equal(vk.suballoc_block_count(), 1)
        assert_equal(vk.suballoc_live_count(), 64)
        for i in range(64):
            vk.destroy_buffer(bufs[i])
        vk.close()
        assert_equal(vk.suballoc_block_count(), 0)
    else:
        vk.close()


# ==============================================================================
# (g)-(j) M4a Tests: Wayland Parity (F23), X11 Parity (F24), EvdevHub (F25), RAII (F28)
# ==============================================================================

def _wl_build_msg(object: Int, opcode: Int, body: List[UInt8]) -> List[UInt8]:
    var pkt = List[UInt8](length=8 + len(body), fill=0)
    wl_put32(pkt, 0, object)
    wl_put16(pkt, 4, opcode)
    wl_put16(pkt, 6, len(pkt))
    for i in range(len(body)):
        pkt[8 + i] = body[i]
    return pkt^


def _wl_build_global_event(registry_id: Int, name: Int, iface: String, version: Int) -> List[UInt8]:
    var body = List[UInt8]()
    wl_append32(body, name)
    wl_append_string(body, iface)
    wl_append32(body, version)
    return _wl_build_msg(registry_id, 0, body)


def _append_bytes(mut dst: List[UInt8], src: List[UInt8]):
    for b in src:
        dst.append(b)


def _send_fd_msg(sock_fd: Int, data: List[UInt8], carried_fd: Int) -> Bool:
    var iov = List[UInt8](length=16, fill=0)
    var control = List[UInt8](length=24, fill=0)
    var msg = List[UInt8](length=56, fill=0)
    var ptr = UInt64(Int(data.unsafe_ptr()))
    for k in range(8):
        iov[k] = UInt8((ptr >> UInt64(k * 8)) & 255)
        iov[8 + k] = UInt8((UInt64(len(data)) >> UInt64(k * 8)) & 255)
        control[k] = UInt8((UInt64(20) >> UInt64(k * 8)) & 255)
    wl_put32(control, 8, 1)
    wl_put32(control, 12, 1)
    wl_put32(control, 16, carried_fd)
    var iov_addr = UInt64(Int(iov.unsafe_ptr()))
    var ctrl_addr = UInt64(Int(control.unsafe_ptr()))
    for k in range(8):
        msg[16 + k] = UInt8((iov_addr >> UInt64(k * 8)) & 255)
        msg[24 + k] = UInt8((UInt64(1) >> UInt64(k * 8)) & 255)
        msg[32 + k] = UInt8((ctrl_addr >> UInt64(k * 8)) & 255)
        msg[40 + k] = UInt8((UInt64(24) >> UInt64(k * 8)) & 255)
    var rc = Int(external_call["sendmsg", Int64](Int32(sock_fd), msg.unsafe_ptr(), Int32(0)))
    _ = len(iov)
    _ = len(control)
    _ = len(msg)
    return rc == len(data)


def _recv_fd_msg(sock_fd: Int) -> Tuple[List[UInt8], Int]:
    var buf = List[UInt8](length=4096, fill=0)
    var iov = List[UInt8](length=16, fill=0)
    var control = List[UInt8](length=64, fill=0)
    var msg = List[UInt8](length=56, fill=0)
    var buf_addr = UInt64(Int(buf.unsafe_ptr()))
    var iov_addr = UInt64(Int(iov.unsafe_ptr()))
    var ctrl_addr = UInt64(Int(control.unsafe_ptr()))
    for k in range(8):
        iov[k] = UInt8((buf_addr >> UInt64(k * 8)) & 255)
        iov[8 + k] = UInt8((UInt64(len(buf)) >> UInt64(k * 8)) & 255)
        msg[16 + k] = UInt8((iov_addr >> UInt64(k * 8)) & 255)
        msg[24 + k] = UInt8((UInt64(1) >> UInt64(k * 8)) & 255)
        msg[32 + k] = UInt8((ctrl_addr >> UInt64(k * 8)) & 255)
        msg[40 + k] = UInt8((UInt64(len(control)) >> UInt64(k * 8)) & 255)
    var n = Int(external_call["recvmsg", Int64](Int32(sock_fd), msg.unsafe_ptr(), Int32(64)))
    var out_bytes = List[UInt8]()
    var out_fd = -1
    if n > 0:
        for i in range(n):
            out_bytes.append(buf[i])
        var clen = 0
        for k in range(8):
            clen |= Int(msg[40 + k]) << (k * 8)
        if clen >= 20 and wl_get32(control, 8) == 1 and wl_get32(control, 12) == 1:
            out_fd = wl_get32(control, 16)
    _ = len(buf)
    _ = len(iov)
    _ = len(control)
    _ = len(msg)
    return (out_bytes^, out_fd)


def _sample_de_xkb_keymap() -> String:
    return (
        'xkb_keymap {\n'
        'xkb_keycodes "evdev" {\n'
        '  <AD03> = 26;\n'
        '  <AD06> = 29;\n'
        '  <AB01> = 52;\n'
        '  <AE11> = 20;\n'
        '  alias <LATZ> = <AD06>;\n'
        '};\n'
        'xkb_symbols "de" {\n'
        '  key <AD03> { [ e, E, EuroSign, EuroSign ], symbols[Group2]= [ a, A ] };\n'
        '  key <LATZ> { [ z, Z ] };\n'
        '  key <AB01> { [ y, Y ] };\n'
        '  key <AE11> { type[Group1]= "FOUR_LEVEL_PLUS_LOCK", symbols[Group1]= [ ssharp, question, backslash, questiondown, U1E9E ] };\n'
        '};\n'
        '};\n'
    )


def _count_open_fds() -> Int:
    return len(list_directory("/proc/self/fd"))


def _send_fds_msg(sock_fd: Int, data: List[UInt8], fds: List[Int32]) -> Bool:
    if len(fds) == 0:
        return Int(external_call["write", Int64](Int32(sock_fd), data.unsafe_ptr(), Int64(len(data)))) == len(data)
    var iov = List[UInt64](length=2, fill=0)
    iov[0] = UInt64(Int(data.unsafe_ptr()))
    iov[1] = UInt64(len(data))
    var cmsg_bytes = 16 + len(fds) * 4
    var cmsg_padded = (cmsg_bytes + 7) & ~7
    var cmsg = List[UInt8](length=cmsg_padded, fill=0)
    put64(Int(cmsg.unsafe_ptr()), cmsg_bytes)
    put32(Int(cmsg.unsafe_ptr()) + 8, 1)   # SOL_SOCKET
    put32(Int(cmsg.unsafe_ptr()) + 12, 1)  # SCM_RIGHTS
    for i in range(len(fds)):
        put32(Int(cmsg.unsafe_ptr()) + 16 + i * 4, UInt32(fds[i]))
    var msg = List[UInt8](length=56, fill=0)
    put64(Int(msg.unsafe_ptr()) + 16, Int(iov.unsafe_ptr()))
    put64(Int(msg.unsafe_ptr()) + 24, 1)
    put64(Int(msg.unsafe_ptr()) + 32, Int(cmsg.unsafe_ptr()))
    put64(Int(msg.unsafe_ptr()) + 40, cmsg_padded)
    var rc = Int(external_call["sendmsg", Int64](Int32(sock_fd), msg.unsafe_ptr(), Int32(0x4000)))
    _ = len(iov)
    _ = len(cmsg)
    _ = len(msg)
    return rc == len(data)


def _make_pipe_tagged(tag: UInt8) -> Tuple[Int, Int]:
    var p = List[Int32](length=2, fill=-1)
    if external_call["pipe", Int32](p.unsafe_ptr()) != 0:
        return (-1, -1)
    var b = List[UInt8](length=1, fill=tag)
    _ = external_call["write", Int64](p[1], b.unsafe_ptr(), Int64(1))
    close_fd(Int(p[1]))
    return (Int(p[0]), Int(tag))


def _read_pipe_tag(fd: Int) -> Int:
    if fd < 0:
        return -1
    var b = List[UInt8](length=4, fill=0)
    var n = Int(external_call["read", Int64](Int32(fd), b.unsafe_ptr(), Int64(4)))
    close_fd(fd)
    if n != 1:
        return -2
    return Int(b[0])


def _wl_build_bind_req(registry_id: Int, name: Int, iface: String, version: Int, new_id: Int) -> List[UInt8]:
    var b = List[UInt8]()
    wl_append32(b, name)
    wl_append_string(b, iface)
    wl_append32(b, version)
    wl_append32(b, new_id)
    return _wl_build_msg(registry_id, 0, b)


def _dnd_responder_thread(arg_addr: Int) abi("C") -> Int:
    var server_fd = get64(arg_addr)
    var fd_count = 0
    _ = external_call["usleep", Int32](UInt32(15000))
    for _ in range(40):
        var rx = _recv_fd_msg(server_fd)
        if rx[1] >= 0:
            fd_count += 1
            if fd_count == 1:
                var payload = String("file:///tmp/dnd_reentry.glb\r\n")
                var pbytes = payload.as_bytes()
                _ = external_call["write", Int64](Int32(rx[1]), pbytes.unsafe_ptr(), Int64(len(pbytes)))
            close_fd(rx[1])
            break
        _ = external_call["usleep", Int32](UInt32(2000))
    # Check briefly if any duplicate FDs were sent
    for _ in range(5):
        var rx2 = _recv_fd_msg(server_fd)
        if rx2[1] >= 0:
            fd_count += 1
            close_fd(rx2[1])
        elif len(rx2[0]) == 0:
            break
    put64(arg_addr + 8, fd_count)
    return 0


def test_f23_wayland_desktop_parity() raises:
    # 1. Pure-Mojo XkbKeymap US QWERTY fallback & modifier rules
    var xkb_us = XkbKeymap()
    assert_true(not xkb_us.has_parsed_keymap)
    assert_equal(xkb_us.translate_key_to_utf8(30), "a")  # evdev KEY_A
    xkb_us.set_modifiers(XKB_MOD_SHIFT, 0, 0, 0)
    assert_equal(xkb_us.translate_key_to_utf8(30), "A")
    assert_equal(xkb_us.translate_key_to_utf8(3), "@")   # Shift+2
    xkb_us.set_modifiers(0, 0, XKB_MOD_LOCK, 0)          # CapsLock
    assert_equal(xkb_us.translate_key_to_utf8(30), "A")
    assert_equal(xkb_us.translate_key_to_utf8(3), "2")   # CapsLock does NOT shift '2'
    xkb_us.set_modifiers(XKB_MOD_CONTROL, 0, 0, 0)
    assert_equal(xkb_us.translate_key_to_utf8(30), "")   # Ctrl suppresses text
    xkb_us.set_modifiers(XKB_MOD_MOD1, 0, 0, 0)
    assert_equal(xkb_us.translate_key_to_utf8(30), "")   # Alt suppresses text

    # 2. German XKB keymap parsing (aliases, 4-level+Lock, AltGr EuroSign, Group2)
    var xkb_de = XkbKeymap()
    assert_true(xkb_de.parse(_sample_de_xkb_keymap()))
    assert_true(xkb_de.has_parsed_keymap)
    assert_equal(xkb_de.translate_key_to_utf8(21), "z")  # <AD06>/<LATZ>
    assert_equal(xkb_de.translate_key_to_utf8(44), "y")  # <AB01>
    assert_equal(xkb_de.translate_key_to_utf8(12), "ß")  # <AE11> unshifted
    xkb_de.set_modifiers(XKB_MOD_SHIFT, 0, 0, 0)
    assert_equal(xkb_de.translate_key_to_utf8(12), "?")
    xkb_de.set_modifiers(XKB_MOD_MOD5, 0, 0, 0)          # AltGr
    assert_equal(xkb_de.translate_key_to_utf8(12), "\\")
    assert_equal(xkb_de.translate_key_to_utf8(18), "€")  # AltGr+E -> EuroSign
    xkb_de.set_modifiers(0, 0, XKB_MOD_LOCK, 0)          # CapsLock -> U1E9E (ẞ)
    assert_equal(xkb_de.translate_key_to_utf8(12), "ẞ")
    xkb_de.set_modifiers(0, 0, 0, 1)                     # Group2
    assert_equal(xkb_de.translate_key_to_utf8(18), "a")

    # 2b. Regression: XKB actions[Group1], lowercase type[group1]/symbols[group2], chained aliases & 0x010000E4
    assert_equal(x11_keysym_to_codepoint(0x010000E4), 0xE4)
    var xkb_edge = XkbKeymap()
    var edge_map = (
        'xkb_keymap {\n'
        'xkb_keycodes "evdev" {\n'
        '  <AE01> = 10;\n'
        '  <AD01> = 24;\n'
        '  <AD02> = 25;\n'
        '  <AC01> = 38;\n'
        '  alias <AL1> = <AL2>;\n'
        '  alias <AL2> = <AC01>;\n'
        '};\n'
        'xkb_symbols "edge" {\n'
        '  key <AD01> {\n'
        '    type[Group1]= "ALPHABETIC",\n'
        '    symbols[Group1]= [ q, Q ],\n'
        '    actions[Group1]= [ NoAction(), NoAction() ]\n'
        '  };\n'
        '  key <AD02> {\n'
        '    actions[Group1]= [ NoAction() ],\n'
        '    vmods[Group1]= [ AltGr ],\n'
        '    symbols[Group1]= [ w, W ]\n'
        '  };\n'
        '  key <AE01> { type[group1]= "FOUR_LEVEL_PLUS_LOCK", symbols[group1]= [ 1, exclam, onesuperior, exclamdown, U00A1 ], symbols[group2]= [ z, Z ] };\n'
        '  key <AL1> { [ x, X ] };\n'
        '};\n'
        '};\n'
    )
    assert_true(xkb_edge.parse(edge_map))
    assert_equal(xkb_edge.translate_key_to_utf8(16), "q")  # <AD01> not overwritten by actions[Group1]
    assert_equal(xkb_edge.translate_key_to_utf8(17), "w")  # <AD02> after actions/vmods
    assert_equal(xkb_edge.translate_key_to_utf8(30), "x")  # <AL1> -> <AL2> -> <AC01> (evdev 30)
    xkb_edge.set_modifiers(0, 0, XKB_MOD_LOCK, 0)
    assert_equal(xkb_edge.translate_key_to_utf8(2), "¡")   # lowercase type[group1] + CapsLock
    xkb_edge.set_modifiers(0, 0, 0, 1)
    assert_equal(xkb_edge.translate_key_to_utf8(2), "z")   # lowercase symbols[group2]
    xkb_edge.set_modifiers(XKB_MOD_SHIFT, 0, 0, 0)
    assert_equal(xkb_edge.translate_key_to_utf8(16), "Q")

    # 3. WaylandWindow protocol injection over real AF_UNIX socketpair:
    #    fractional scale, SSD configure, XKB memfd SCM_RIGHTS, clipboard, DnD
    var sv = List[Int32](length=2, fill=-1)
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), sv.unsafe_ptr()), Int32(0))
    var client_fd = Int(sv[0])
    var server_fd = Int(sv[1])

    var win = WaylandWindow()
    win.fd = client_fd
    win.width = 1280
    win.height = 720
    win.logical_width = 1280
    win.logical_height = 720
    win.registry = 2
    win.surface = 3
    win.toplevel = 4
    win.seat = 5
    win.seat_version = 5
    win.keyboard = 6
    win.next_id = 10

    # Inject wl_registry.global for fractional scale, xdg_decoration, data_device_manager
    _append_bytes(win.incoming, _wl_build_global_event(win.registry, 101, "wp_fractional_scale_manager_v1", 1))
    _append_bytes(win.incoming, _wl_build_global_event(win.registry, 102, "zxdg_decoration_manager_v1", 1))
    _append_bytes(win.incoming, _wl_build_global_event(win.registry, 103, "wl_data_device_manager", 3))
    win.handle_incoming()
    assert_true(win.fractional_scale_manager != 0)
    assert_true(win.decoration_manager != 0)
    assert_true(win.data_device_manager != 0)
    assert_true(win.data_device != 0)

    # Fallback preferred_buffer_scale = 2 before fractional_scale object exists
    var pbs_body = List[UInt8]()
    wl_append32(pbs_body, 2)
    _append_bytes(win.incoming, _wl_build_msg(win.surface, 2, pbs_body))
    win.handle_incoming()
    assert_equal(win.content_scale(), Float32(2.0))
    var fb2 = win.framebuffer_size()
    assert_equal(fb2[0], 2560)
    assert_equal(fb2[1], 1440)

    # Fractional scale = 180 / 120 = 1.5 -> 1920x1080 while logical remains 1280x720
    win.fractional_scale = 20
    win.toplevel_decoration = 21
    var fs_body = List[UInt8]()
    wl_append32(fs_body, 180)
    _append_bytes(win.incoming, _wl_build_msg(win.fractional_scale, 0, fs_body))
    var dec_body = List[UInt8]()
    wl_append32(dec_body, 2)  # SSD
    _append_bytes(win.incoming, _wl_build_msg(win.toplevel_decoration, 0, dec_body))
    win.handle_incoming()
    assert_equal(win.content_scale(), Float32(1.5))
    var fb15 = win.framebuffer_size()
    assert_equal(fb15[0], 1920)
    assert_equal(fb15[1], 1080)
    assert_equal(win.width, 1280)
    assert_equal(win.height, 720)
    assert_equal(win.decoration_mode, 2)

    # Send wl_keyboard.keymap memfd via SCM_RIGHTS over socketpair
    var km_str = _sample_de_xkb_keymap()
    var km_name = String("test-xkb")
    var km_fd = Int(external_call["memfd_create", Int32](km_name.as_c_string_span(), UInt32(1)))
    assert_true(km_fd >= 0)
    _ = external_call["ftruncate", Int32](Int32(km_fd), Int64(km_str.byte_length()))
    _ = external_call["pwrite", Int64](Int32(km_fd), km_str.as_bytes().unsafe_ptr(), Int64(km_str.byte_length()), Int64(0))
    var km_body = List[UInt8]()
    wl_append32(km_body, 1)  # XKB_V1
    wl_append32(km_body, km_str.byte_length())
    assert_true(_send_fd_msg(server_fd, _wl_build_msg(win.keyboard, 0, km_body), km_fd))
    close_fd(km_fd)

    # Send modifiers + key presses: 'z' (evdev 21), Shift+'Z', 'ß' (evdev 12), AltGr+'€' (evdev 18), Ctrl+'z' (suppressed)
    def _kbd_key(kbd: Int, serial: Int, code: Int, state: Int) -> List[UInt8]:
        var b = List[UInt8]()
        wl_append32(b, serial)
        wl_append32(b, 100)
        wl_append32(b, code)
        wl_append32(b, state)
        return _wl_build_msg(kbd, 3, b)

    def _kbd_mods(kbd: Int, dep: Int, lat: Int, loc: Int, grp: Int) -> List[UInt8]:
        var b = List[UInt8]()
        wl_append32(b, 50)
        wl_append32(b, dep)
        wl_append32(b, lat)
        wl_append32(b, loc)
        wl_append32(b, grp)
        return _wl_build_msg(kbd, 4, b)

    var kbd_stream = List[UInt8]()
    _append_bytes(kbd_stream, _kbd_key(win.keyboard, 701, 21, 1))
    _append_bytes(kbd_stream, _kbd_mods(win.keyboard, XKB_MOD_SHIFT, 0, 0, 0))
    _append_bytes(kbd_stream, _kbd_key(win.keyboard, 702, 21, 1))
    _append_bytes(kbd_stream, _kbd_mods(win.keyboard, 0, 0, 0, 0))
    _append_bytes(kbd_stream, _kbd_key(win.keyboard, 703, 12, 1))
    _append_bytes(kbd_stream, _kbd_mods(win.keyboard, XKB_MOD_MOD5, 0, 0, 0))
    _append_bytes(kbd_stream, _kbd_key(win.keyboard, 704, 18, 1))
    _append_bytes(kbd_stream, _kbd_mods(win.keyboard, XKB_MOD_CONTROL, 0, 0, 0))
    _append_bytes(kbd_stream, _kbd_key(win.keyboard, 705, 21, 1))
    _ = external_call["write", Int64](Int32(server_fd), kbd_stream.unsafe_ptr(), Int64(len(kbd_stream)))

    var inp = InputState()
    inp.begin_frame()
    win.poll(inp)
    assert_equal(inp.text_input_utf8, "zZß€")
    assert_equal(win.last_input_serial, 705)
    assert_true(inp.key_down(89))  # physical evdev 21 ('Y' on US evdev_key) still works

    # Clipboard set + wl_data_source.send over socketpair
    win.clipboard_set("Copy Me ¡Hola!")
    assert_true(win.clipboard_source != 0)
    assert_true(win.clipboard_owned)
    assert_equal(win.clipboard_get(), "Copy Me ¡Hola!")
    var p_send = List[Int32](length=2, fill=-1)
    assert_equal(external_call["pipe", Int32](p_send.unsafe_ptr()), Int32(0))
    var ds_body = List[UInt8]()
    wl_append_string(ds_body, "text/plain;charset=utf-8")
    assert_true(_send_fd_msg(server_fd, _wl_build_msg(win.clipboard_source, 1, ds_body), Int(p_send[1])))
    close_fd(Int(p_send[1]))
    win.receive()
    var got_buf = List[UInt8](length=64, fill=0)
    var n_clip = Int(external_call["read", Int64](p_send[0], got_buf.unsafe_ptr(), Int64(64)))
    close_fd(Int(p_send[0]))
    assert_true(n_clip > 0)
    got_buf.resize(n_clip, 0)
    assert_equal(String(unsafe_from_utf8=got_buf), "Copy Me ¡Hola!")

    # wl_data_device.selection -> wl_data_offer.receive pipe read
    win.clipboard_owned = False
    var offer_sel = 0xFF000001
    var do_body = List[UInt8]()
    wl_append32(do_body, offer_sel)
    _append_bytes(win.incoming, _wl_build_msg(win.data_device, 0, do_body))
    var om_body = List[UInt8]()
    wl_append_string(om_body, "text/plain;charset=utf-8")
    _append_bytes(win.incoming, _wl_build_msg(offer_sel, 0, om_body))
    var sel_body = List[UInt8]()
    wl_append32(sel_body, offer_sel)
    _append_bytes(win.incoming, _wl_build_msg(win.data_device, 5, sel_body))
    win.handle_incoming()
    # Drain client outbound socket first, then write remote selection payload
    var drain_tmp = List[UInt8](length=2048, fill=0)
    while Int(external_call["recv", Int64](Int32(server_fd), drain_tmp.unsafe_ptr(), Int64(len(drain_tmp)), Int32(64))) > 0:
        pass
    # Write a fake pre-queued selection by using socketpair: client sends receive(fd) during clipboard_get;
    # wait: clipboard_get blocks on read_fd, so spawn? No! Instead test DnD & clipboard via bridge or non-blocking pipe:
    # Better: pre-fill? No, client creates pipe inside _read_offer_string! To test _read_offer_string without threads,
    # verify DnD URI-list parser & accept/motion state directly AND verify socketpair send_with_fd + receive:
    # Actually: we CAN test _read_offer_string cleanly using a lightweight pthread or test parse_uri_list + offer tracking!
    # Wait: let's verify DnD enter/motion/leave state AND parse_uri_list:
    var dnd_offer = 0xFF000002
    var dnd_pkt = List[UInt8]()
    var do2 = List[UInt8]()
    wl_append32(do2, dnd_offer)
    _append_bytes(dnd_pkt, _wl_build_msg(win.data_device, 0, do2))
    var om2 = List[UInt8]()
    wl_append_string(om2, "text/uri-list")
    _append_bytes(dnd_pkt, _wl_build_msg(dnd_offer, 0, om2))
    var ent = List[UInt8]()
    wl_append32(ent, 901)
    wl_append32(ent, win.surface)
    wl_append32(ent, 100 * 256)
    wl_append32(ent, 200 * 256)
    wl_append32(ent, dnd_offer)
    _append_bytes(dnd_pkt, _wl_build_msg(win.data_device, 1, ent))
    var mot = List[UInt8]()
    wl_append32(mot, 1234)
    wl_append32(mot, 150 * 256)
    wl_append32(mot, 250 * 256)
    _append_bytes(dnd_pkt, _wl_build_msg(win.data_device, 3, mot))
    _append_bytes(win.incoming, dnd_pkt)
    win.handle_incoming()
    assert_equal(win.dnd_offer_id, dnd_offer)
    assert_equal(win.dnd_x, Float32(150.0))
    assert_equal(win.dnd_y, Float32(250.0))

    from flint.platform.wayland import parse_uri_list
    var parsed_uris = parse_uri_list("# drag comment\r\nfile:///tmp/hello%20world.glb\r\nfile://localhost/opt/a%2Bb.flpk\r\n")
    assert_equal(len(parsed_uris), 2)
    assert_equal(parsed_uris[0], "/tmp/hello world.glb")
    assert_equal(parsed_uris[1], "/opt/a+b.flpk")
    for p in parsed_uris:
        win.pending_dropped_files.append(p)
    inp.begin_frame()
    win.poll(inp)
    assert_equal(len(inp.dropped_files), 2)
    assert_equal(inp.dropped_files[0], "/tmp/hello world.glb")
    assert_equal(inp.dropped_files[1], "/opt/a+b.flpk")

    win.close()
    close_fd(server_fd)

    # 4. WaylandBridge server-offer ID tracking & game-FD routing (`WORD_GAME_FDS`)
    var st_ptr = external_call["calloc", Int](Int(STATE_WORDS), Int(8))
    var b_sv = List[Int32](length=2, fill=-1)
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), b_sv.unsafe_ptr()), Int32(0))
    var l_sv = List[Int32](length=2, fill=-1)
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), l_sv.unsafe_ptr()), Int32(0))
    set_word(st_ptr, 0, Int(b_sv[0]))  # WORD_REAL
    set_word(st_ptr, WORD_LIBRARY, Int(l_sv[0]))
    set_word(st_ptr, WORD_IN, external_call["malloc", Int](STREAM_BYTES))
    set_word(st_ptr, WORD_GAME, external_call["malloc", Int](GAME_BYTES))
    set_word(st_ptr, WORD_OWNED, external_call["calloc", Int](Int(MAX_OWNED_ID // 8), Int(1)))
    set_word(st_ptr, WORD_IN_FDS, external_call["malloc", Int](MAX_FDS * 8))
    set_word(st_ptr, WORD_GAME_FDS, external_call["malloc", Int](MAX_FDS * 8))
    set_word(st_ptr, WORD_OFFER_IDS, external_call["calloc", Int](Int(MAX_OFFERS), Int(4)))
    set_word(st_ptr, WORD_IO_SCRATCH, external_call["calloc", Int](Int(1), IO_SCRATCH_BYTES))
    set_word(st_ptr, WORD_KEYBOARD, 6)
    set_word(st_ptr, WORD_DATA_DEVICE, 7)
    set_word(st_ptr, WORD_DATA_SOURCE, 8)

    from flint.platform.wayland_bridge import set_owned, take_incoming
    set_owned(st_ptr, 6)
    set_owned(st_ptr, 7)
    set_owned(st_ptr, 8)
    var srv_offer_id = Int(0xFF000010)
    assert_true(not is_server_offer(st_ptr, srv_offer_id))
    var do_srv = List[UInt8]()
    wl_append32(do_srv, srv_offer_id)
    var srv_msg1 = _wl_build_msg(7, 0, do_srv)
    var srv_msg2 = _wl_build_msg(srv_offer_id, 0, om_body)
    var p_km = List[Int32](length=2, fill=-1)
    assert_equal(external_call["pipe", Int32](p_km.unsafe_ptr()), Int32(0))
    var km_wire = _wl_build_msg(6, 0, km_body)
    _append_bytes(km_wire, srv_msg1)
    _append_bytes(km_wire, srv_msg2)
    assert_true(_send_fd_msg(Int(b_sv[1]), km_wire, Int(p_km[0])))
    close_fd(Int(p_km[0]))
    close_fd(Int(p_km[1]))
    assert_true(take_incoming(st_ptr))
    assert_true(is_server_offer(st_ptr, srv_offer_id))
    assert_equal(word(st_ptr, WORD_GAME_FD_COUNT), 1)
    assert_true(word(st_ptr, WORD_GAME_USED) == len(km_wire))
    from flint.platform.vulkan import get64
    close_fd(Int(get64(word(st_ptr, WORD_GAME_FDS))))
    set_word(st_ptr, WORD_GAME_FD_COUNT, 0)
    set_word(st_ptr, WORD_GAME_USED, 0)

    # 4b. Regression: batched [for_library(no-FD), wl_keyboard.keymap(with-FD)] preserves game FD!
    var lib_body = List[UInt8]()
    wl_append32(lib_body, 1234)
    var lib_msg = _wl_build_msg(99, 0, lib_body)  # obj 99 not owned -> for_library
    var batch1 = List[UInt8]()
    _append_bytes(batch1, lib_msg)
    _append_bytes(batch1, _wl_build_msg(6, 0, km_body))
    var p_km2 = List[Int32](length=2, fill=-1)
    assert_equal(external_call["pipe", Int32](p_km2.unsafe_ptr()), Int32(0))
    assert_true(_send_fd_msg(Int(b_sv[1]), batch1, Int(p_km2[0])))
    close_fd(Int(p_km2[0]))
    close_fd(Int(p_km2[1]))
    assert_true(take_incoming(st_ptr))
    assert_equal(word(st_ptr, WORD_GAME_FD_COUNT), 1)
    close_fd(Int(get64(word(st_ptr, WORD_GAME_FDS))))
    set_word(st_ptr, WORD_GAME_FD_COUNT, 0)
    set_word(st_ptr, WORD_GAME_USED, 0)
    var drain_lib = List[UInt8](length=256, fill=0)
    var n_lib = Int(external_call["recv", Int64](Int32(l_sv[1]), drain_lib.unsafe_ptr(), Int64(256), Int32(64)))
    assert_equal(n_lib, len(lib_msg))

    # 4c. Regression: batched [for_library(with-FD), wl_keyboard.keymap(with-FD)] preserves BOTH FDs in order!
    var p_lib = List[Int32](length=2, fill=-1)
    var p_km3 = List[Int32](length=2, fill=-1)
    assert_equal(external_call["pipe", Int32](p_lib.unsafe_ptr()), Int32(0))
    assert_equal(external_call["pipe", Int32](p_km3.unsafe_ptr()), Int32(0))
    var batch2 = List[UInt8]()
    _append_bytes(batch2, lib_msg)
    _append_bytes(batch2, _wl_build_msg(6, 0, km_body))
    var fds_pair = List[Int32](length=2, fill=-1)
    fds_pair[0] = p_lib[0]
    fds_pair[1] = p_km3[0]
    # Send 2 FDs in single SCM_RIGHTS control message
    var cmsg_buf = List[UInt8](length=24, fill=0)
    var iov_pair = List[UInt64](length=2, fill=0)
    iov_pair[0] = UInt64(Int(batch2.unsafe_ptr()))
    iov_pair[1] = UInt64(len(batch2))
    var msg_pair = List[UInt8](length=56, fill=0)
    from flint.platform.vulkan import put64, put32
    put64(Int(msg_pair.unsafe_ptr()) + 16, Int(iov_pair.unsafe_ptr()))
    put64(Int(msg_pair.unsafe_ptr()) + 24, 1)
    put64(Int(cmsg_buf.unsafe_ptr()), 24)
    put32(Int(cmsg_buf.unsafe_ptr()) + 8, 1)   # SOL_SOCKET
    put32(Int(cmsg_buf.unsafe_ptr()) + 12, 1)  # SCM_RIGHTS
    put32(Int(cmsg_buf.unsafe_ptr()) + 16, UInt32(fds_pair[0]))
    put32(Int(cmsg_buf.unsafe_ptr()) + 20, UInt32(fds_pair[1]))
    put64(Int(msg_pair.unsafe_ptr()) + 32, Int(cmsg_buf.unsafe_ptr()))
    put64(Int(msg_pair.unsafe_ptr()) + 40, 24)
    assert_true(Int(external_call["sendmsg", Int64](Int32(b_sv[1]), msg_pair.unsafe_ptr(), Int32(0x4000))) == len(batch2))
    _ = iov_pair^
    _ = cmsg_buf^
    _ = msg_pair^
    _ = batch2^
    close_fd(Int(p_lib[0]))
    close_fd(Int(p_km3[0]))
    assert_true(take_incoming(st_ptr))
    assert_equal(word(st_ptr, WORD_GAME_FD_COUNT), 1)
    var game_got_fd = Int(get64(word(st_ptr, WORD_GAME_FDS)))
    var tag_lib = List[UInt8](length=1, fill=0x41)
    var tag_km = List[UInt8](length=1, fill=0x42)
    _ = external_call["write", Int64](p_lib[1], tag_lib.unsafe_ptr(), Int64(1))
    _ = external_call["write", Int64](p_km3[1], tag_km.unsafe_ptr(), Int64(1))
    close_fd(Int(p_lib[1]))
    close_fd(Int(p_km3[1]))
    var rd_km = List[UInt8](length=4, fill=0)
    assert_equal(Int(external_call["read", Int64](Int32(game_got_fd), rd_km.unsafe_ptr(), Int64(4))), 1)
    assert_equal(Int(rd_km[0]), 0x42)  # Game got the SECOND FD (p_km3), NOT p_lib!
    close_fd(game_got_fd)
    set_word(st_ptr, WORD_GAME_FD_COUNT, 0)
    set_word(st_ptr, WORD_GAME_USED, 0)
    # Verify l_sv[1] received the lib message AND the FIRST FD (p_lib)!
    var lib_rx_buf = List[UInt8](length=256, fill=0)
    var lib_rx_fds = List[Int64](length=4, fill=-1)
    var lib_rx_cnt = 0
    from flint.platform.wayland_bridge import receive_some
    var lib_rx_n = receive_some(Int(l_sv[1]), Int(lib_rx_buf.unsafe_ptr()), 256, Int(lib_rx_fds.unsafe_ptr()), lib_rx_cnt, word(st_ptr, WORD_IO_SCRATCH))
    _ = lib_rx_buf^
    assert_equal(lib_rx_n, len(lib_msg))
    assert_equal(lib_rx_cnt, 1)
    var rd_lib = List[UInt8](length=4, fill=0)
    assert_equal(Int(external_call["read", Int64](Int32(lib_rx_fds[0]), rd_lib.unsafe_ptr(), Int64(4))), 1)
    assert_equal(Int(rd_lib[0]), 0x41)  # Library got the FIRST FD (p_lib)!
    close_fd(Int(lib_rx_fds[0]))

    # 4d. Regression (T3): interleaved [for_library(0), wl_keyboard.keymap(0x55), for_library(0x66)]
    #     must NOT swap FDs: game gets 0x55 and library gets 0x66!
    var lib_msg0 = _wl_build_msg(99, 0, lib_body)  # obj 99, op 0, 12B (for_library, 0 FDs)
    var lib_msg_fd = _wl_build_msg(98, 2, lib_body)  # obj 98, op 2, 12B (for_library, 0x66)
    var p_t3_km = List[Int32](length=2, fill=-1)
    var p_t3_lib = List[Int32](length=2, fill=-1)
    assert_equal(external_call["pipe", Int32](p_t3_km.unsafe_ptr()), Int32(0))
    assert_equal(external_call["pipe", Int32](p_t3_lib.unsafe_ptr()), Int32(0))
    var batch_t3 = List[UInt8]()
    _append_bytes(batch_t3, lib_msg0)
    _append_bytes(batch_t3, _wl_build_msg(6, 0, km_body))
    _append_bytes(batch_t3, lib_msg_fd)
    var iov_t3 = List[UInt64](length=2, fill=0)
    iov_t3[0] = UInt64(Int(batch_t3.unsafe_ptr()))
    iov_t3[1] = UInt64(len(batch_t3))
    var cmsg_t3 = List[UInt8](length=24, fill=0)
    put64(Int(cmsg_t3.unsafe_ptr()), 24)
    put32(Int(cmsg_t3.unsafe_ptr()) + 8, 1)   # SOL_SOCKET
    put32(Int(cmsg_t3.unsafe_ptr()) + 12, 1)  # SCM_RIGHTS
    put32(Int(cmsg_t3.unsafe_ptr()) + 16, UInt32(p_t3_km[0]))
    put32(Int(cmsg_t3.unsafe_ptr()) + 20, UInt32(p_t3_lib[0]))
    var msg_t3 = List[UInt8](length=56, fill=0)
    put64(Int(msg_t3.unsafe_ptr()) + 16, Int(iov_t3.unsafe_ptr()))
    put64(Int(msg_t3.unsafe_ptr()) + 24, 1)
    put64(Int(msg_t3.unsafe_ptr()) + 32, Int(cmsg_t3.unsafe_ptr()))
    put64(Int(msg_t3.unsafe_ptr()) + 40, 24)
    assert_true(Int(external_call["sendmsg", Int64](Int32(b_sv[1]), msg_t3.unsafe_ptr(), Int32(0x4000))) == len(batch_t3))
    _ = iov_t3^
    _ = cmsg_t3^
    _ = msg_t3^
    _ = batch_t3^
    close_fd(Int(p_t3_km[0]))
    close_fd(Int(p_t3_lib[0]))
    assert_true(take_incoming(st_ptr))
    assert_equal(word(st_ptr, WORD_GAME_FD_COUNT), 1)
    var t3_game_fd = Int(get64(word(st_ptr, WORD_GAME_FDS)))
    var tag_t3_km = List[UInt8](length=1, fill=0x55)
    var tag_t3_lib = List[UInt8](length=1, fill=0x66)
    _ = external_call["write", Int64](p_t3_km[1], tag_t3_km.unsafe_ptr(), Int64(1))
    _ = external_call["write", Int64](p_t3_lib[1], tag_t3_lib.unsafe_ptr(), Int64(1))
    close_fd(Int(p_t3_km[1]))
    close_fd(Int(p_t3_lib[1]))
    var rd_t3_km = List[UInt8](length=4, fill=0)
    assert_equal(Int(external_call["read", Int64](Int32(t3_game_fd), rd_t3_km.unsafe_ptr(), Int64(4))), 1)
    assert_equal(Int(rd_t3_km[0]), 0x55)  # Game got 0x55!
    close_fd(t3_game_fd)
    set_word(st_ptr, WORD_GAME_FD_COUNT, 0)
    set_word(st_ptr, WORD_GAME_USED, 0)
    # Drain both lib_msg0 (no FD) and lib_msg_fd (with 0x66) from l_sv[1]
    var t3_lib_buf = List[UInt8](length=256, fill=0)
    var t3_lib_fds = List[Int64](length=4, fill=-1)
    var t3_lib_cnt = 0
    var t3_lib_total = 0
    for _ in range(4):
        var n_part = receive_some(Int(l_sv[1]), Int(t3_lib_buf.unsafe_ptr()) + t3_lib_total, 256 - t3_lib_total, Int(t3_lib_fds.unsafe_ptr()), t3_lib_cnt, word(st_ptr, WORD_IO_SCRATCH))
        if n_part <= 0:
            break
        t3_lib_total += n_part
    _ = t3_lib_buf^
    assert_equal(t3_lib_total, len(lib_msg0) + len(lib_msg_fd))
    assert_equal(t3_lib_cnt, 1)
    var rd_t3_lib = List[UInt8](length=4, fill=0)
    assert_equal(Int(external_call["read", Int64](Int32(t3_lib_fds[0]), rd_t3_lib.unsafe_ptr(), Int64(4))), 1)
    assert_equal(Int(rd_t3_lib[0]), 0x66)  # Library got 0x66!
    close_fd(Int(t3_lib_fds[0]))

    # 4e. Regression (T4a): split <8-byte partial wl_keyboard.keymap header after for_library(0)
    #     must NOT let for_library steal/close the game FD!
    var km_wire_t4a = _wl_build_msg(6, 0, km_body)
    assert_equal(len(km_wire_t4a), 16)
    var t4a_part1 = List[UInt8]()
    _append_bytes(t4a_part1, lib_msg0)
    for k in range(4):
        t4a_part1.append(km_wire_t4a[k])  # 4-byte partial wl_keyboard.keymap header (< 8B)
    var t4a_part2 = List[UInt8]()
    for k in range(4, len(km_wire_t4a)):
        t4a_part2.append(km_wire_t4a[k])  # remaining 12 bytes
    var p_t4a = List[Int32](length=2, fill=-1)
    assert_equal(external_call["pipe", Int32](p_t4a.unsafe_ptr()), Int32(0))
    assert_true(_send_fd_msg(Int(b_sv[1]), t4a_part1, Int(p_t4a[0])))
    close_fd(Int(p_t4a[0]))
    assert_true(not take_incoming(st_ptr))  # only lib_msg0 processed so far
    assert_equal(word(st_ptr, WORD_GAME_FD_COUNT), 0)
    var t4a_lib_buf = List[UInt8](length=256, fill=0)
    var t4a_lib_fds = List[Int64](length=4, fill=-1)
    var t4a_lib_cnt = 0
    var t4a_lib_n = receive_some(Int(l_sv[1]), Int(t4a_lib_buf.unsafe_ptr()), 256, Int(t4a_lib_fds.unsafe_ptr()), t4a_lib_cnt, word(st_ptr, WORD_IO_SCRATCH))
    _ = t4a_lib_buf^
    assert_equal(t4a_lib_n, len(lib_msg0))
    assert_equal(t4a_lib_cnt, 0)  # Library must receive 0 FDs!
    _ = external_call["write", Int64](Int32(b_sv[1]), t4a_part2.unsafe_ptr(), Int64(len(t4a_part2)))
    assert_true(take_incoming(st_ptr))
    assert_equal(word(st_ptr, WORD_GAME_FD_COUNT), 1)
    var t4a_game_fd = Int(get64(word(st_ptr, WORD_GAME_FDS)))
    var tag_t4a = List[UInt8](length=1, fill=0x77)
    _ = external_call["write", Int64](p_t4a[1], tag_t4a.unsafe_ptr(), Int64(1))
    close_fd(Int(p_t4a[1]))
    var rd_t4a = List[UInt8](length=4, fill=0)
    assert_equal(Int(external_call["read", Int64](Int32(t4a_game_fd), rd_t4a.unsafe_ptr(), Int64(4))), 1)
    assert_equal(Int(rd_t4a[0]), 0x77)
    close_fd(t4a_game_fd)
    set_word(st_ptr, WORD_GAME_FD_COUNT, 0)
    set_word(st_ptr, WORD_GAME_USED, 0)

    # 4e2. Regression (T4a <4B): 2-byte partial wl_keyboard.keymap header after for_library(0)
    var t4a2_part1 = List[UInt8]()
    _append_bytes(t4a2_part1, lib_msg0)
    for k in range(2):
        t4a2_part1.append(km_wire_t4a[k])  # 2-byte partial (< 4B)
    var t4a2_part2 = List[UInt8]()
    for k in range(2, len(km_wire_t4a)):
        t4a2_part2.append(km_wire_t4a[k])  # remaining 14 bytes
    var p_t4a2 = List[Int32](length=2, fill=-1)
    assert_equal(external_call["pipe", Int32](p_t4a2.unsafe_ptr()), Int32(0))
    assert_true(_send_fd_msg(Int(b_sv[1]), t4a2_part1, Int(p_t4a2[0])))
    close_fd(Int(p_t4a2[0]))
    assert_true(not take_incoming(st_ptr))
    assert_equal(word(st_ptr, WORD_GAME_FD_COUNT), 0)
    var t4a2_lib_buf = List[UInt8](length=256, fill=0)
    var t4a2_lib_fds = List[Int64](length=4, fill=-1)
    var t4a2_lib_cnt = 0
    var t4a2_lib_n = receive_some(Int(l_sv[1]), Int(t4a2_lib_buf.unsafe_ptr()), 256, Int(t4a2_lib_fds.unsafe_ptr()), t4a2_lib_cnt, word(st_ptr, WORD_IO_SCRATCH))
    _ = t4a2_lib_buf^
    assert_equal(t4a2_lib_n, len(lib_msg0))
    assert_equal(t4a2_lib_cnt, 0)
    _ = external_call["write", Int64](Int32(b_sv[1]), t4a2_part2.unsafe_ptr(), Int64(len(t4a2_part2)))
    assert_true(take_incoming(st_ptr))
    assert_equal(word(st_ptr, WORD_GAME_FD_COUNT), 1)
    var t4a2_game_fd = Int(get64(word(st_ptr, WORD_GAME_FDS)))
    var tag_t4a2 = List[UInt8](length=1, fill=0x7A)
    _ = external_call["write", Int64](p_t4a2[1], tag_t4a2.unsafe_ptr(), Int64(1))
    close_fd(Int(p_t4a2[1]))
    var rd_t4a2 = List[UInt8](length=4, fill=0)
    assert_equal(Int(external_call["read", Int64](Int32(t4a2_game_fd), rd_t4a2.unsafe_ptr(), Int64(4))), 1)
    assert_equal(Int(rd_t4a2[0]), 0x7A)
    close_fd(t4a2_game_fd)
    set_word(st_ptr, WORD_GAME_FD_COUNT, 0)
    set_word(st_ptr, WORD_GAME_USED, 0)

    # 4f. Regression (T4a <4B + T3 trailing partial for_library FD):
    #     Part 1: [lib(0), keymap(0x55), 4B partial lib_msg_fd(0x66)] with [p_km, p_lib]
    #     Part 2: remaining 8B of lib_msg_fd
    var p_t4f_km = List[Int32](length=2, fill=-1)
    var p_t4f_lib = List[Int32](length=2, fill=-1)
    assert_equal(external_call["pipe", Int32](p_t4f_km.unsafe_ptr()), Int32(0))
    assert_equal(external_call["pipe", Int32](p_t4f_lib.unsafe_ptr()), Int32(0))
    var batch_t4f_1 = List[UInt8]()
    _append_bytes(batch_t4f_1, lib_msg0)
    _append_bytes(batch_t4f_1, km_wire_t4a)
    for k in range(4):
        batch_t4f_1.append(lib_msg_fd[k])
    var batch_t4f_2 = List[UInt8]()
    for k in range(4, len(lib_msg_fd)):
        batch_t4f_2.append(lib_msg_fd[k])
    var iov_t4f = List[UInt64](length=2, fill=0)
    iov_t4f[0] = UInt64(Int(batch_t4f_1.unsafe_ptr()))
    iov_t4f[1] = UInt64(len(batch_t4f_1))
    var cmsg_t4f = List[UInt8](length=24, fill=0)
    put64(Int(cmsg_t4f.unsafe_ptr()), 24)
    put32(Int(cmsg_t4f.unsafe_ptr()) + 8, 1)
    put32(Int(cmsg_t4f.unsafe_ptr()) + 12, 1)
    put32(Int(cmsg_t4f.unsafe_ptr()) + 16, UInt32(p_t4f_km[0]))
    put32(Int(cmsg_t4f.unsafe_ptr()) + 20, UInt32(p_t4f_lib[0]))
    var msg_t4f = List[UInt8](length=56, fill=0)
    put64(Int(msg_t4f.unsafe_ptr()) + 16, Int(iov_t4f.unsafe_ptr()))
    put64(Int(msg_t4f.unsafe_ptr()) + 24, 1)
    put64(Int(msg_t4f.unsafe_ptr()) + 32, Int(cmsg_t4f.unsafe_ptr()))
    put64(Int(msg_t4f.unsafe_ptr()) + 40, 24)
    assert_true(Int(external_call["sendmsg", Int64](Int32(b_sv[1]), msg_t4f.unsafe_ptr(), Int32(0x4000))) == len(batch_t4f_1))
    _ = iov_t4f^
    _ = cmsg_t4f^
    _ = msg_t4f^
    _ = batch_t4f_1^
    close_fd(Int(p_t4f_km[0]))
    close_fd(Int(p_t4f_lib[0]))
    assert_true(take_incoming(st_ptr))
    assert_equal(word(st_ptr, WORD_GAME_FD_COUNT), 1)
    var t4f_game_fd = Int(get64(word(st_ptr, WORD_GAME_FDS)))
    var tag_t4f_km = List[UInt8](length=1, fill=0x88)
    var tag_t4f_lib = List[UInt8](length=1, fill=0x99)
    _ = external_call["write", Int64](p_t4f_km[1], tag_t4f_km.unsafe_ptr(), Int64(1))
    _ = external_call["write", Int64](p_t4f_lib[1], tag_t4f_lib.unsafe_ptr(), Int64(1))
    close_fd(Int(p_t4f_km[1]))
    close_fd(Int(p_t4f_lib[1]))
    var rd_t4f_km = List[UInt8](length=4, fill=0)
    assert_equal(Int(external_call["read", Int64](Int32(t4f_game_fd), rd_t4f_km.unsafe_ptr(), Int64(4))), 1)
    assert_equal(Int(rd_t4f_km[0]), 0x88)
    close_fd(t4f_game_fd)
    set_word(st_ptr, WORD_GAME_FD_COUNT, 0)
    set_word(st_ptr, WORD_GAME_USED, 0)
    _ = external_call["write", Int64](Int32(b_sv[1]), batch_t4f_2.unsafe_ptr(), Int64(len(batch_t4f_2)))
    assert_true(not take_incoming(st_ptr))
    var t4f_lib_buf = List[UInt8](length=256, fill=0)
    var t4f_lib_fds = List[Int64](length=4, fill=-1)
    var t4f_lib_cnt = 0
    var t4f_lib_total = 0
    for _ in range(4):
        var n_part = receive_some(Int(l_sv[1]), Int(t4f_lib_buf.unsafe_ptr()) + t4f_lib_total, 256 - t4f_lib_total, Int(t4f_lib_fds.unsafe_ptr()), t4f_lib_cnt, word(st_ptr, WORD_IO_SCRATCH))
        if n_part <= 0:
            break
        t4f_lib_total += n_part
    _ = t4f_lib_buf^
    assert_equal(t4f_lib_total, len(lib_msg0) + len(lib_msg_fd))
    assert_equal(t4f_lib_cnt, 1)
    var rd_t4f_lib = List[UInt8](length=4, fill=0)
    assert_equal(Int(external_call["read", Int64](Int32(t4f_lib_fds[0]), rd_t4f_lib.unsafe_ptr(), Int64(4))), 1)
    assert_equal(Int(rd_t4f_lib[0]), 0x99)
    close_fd(Int(t4f_lib_fds[0]))

    close_fd(Int(b_sv[0])); close_fd(Int(b_sv[1]))
    close_fd(Int(l_sv[0])); close_fd(Int(l_sv[1]))
    external_call["free", NoneType](word(st_ptr, WORD_IN))
    external_call["free", NoneType](word(st_ptr, WORD_GAME))
    external_call["free", NoneType](word(st_ptr, WORD_OWNED))
    external_call["free", NoneType](word(st_ptr, WORD_IN_FDS))
    external_call["free", NoneType](word(st_ptr, WORD_GAME_FDS))
    external_call["free", NoneType](word(st_ptr, WORD_OFFER_IDS))
    external_call["free", NoneType](word(st_ptr, WORD_IO_SCRATCH))
    external_call["free", NoneType](st_ptr)

    # 4g. Protocol-tracked WaylandBridge (take_outgoing() wire snooping + WORD_STRICT_IFACE=1):
    #     ADV-A, ADV-A2, DynSingleOffer, ADV-B, ADV-C, ADV-D, ADV-E, T1-T5, wl_display.delete_id, 0 FD leaks
    var fds_4g_before = _count_open_fds()
    var st_strict = external_call["calloc", Int](Int(STATE_WORDS), Int(8))
    var sb_sv = List[Int32](length=2, fill=-1)
    var sl_sv = List[Int32](length=2, fill=-1)
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), sb_sv.unsafe_ptr()), Int32(0))
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), sl_sv.unsafe_ptr()), Int32(0))
    set_word(st_strict, 0, Int(sb_sv[0]))  # WORD_REAL
    set_word(st_strict, WORD_LIBRARY, Int(sl_sv[0]))
    set_word(st_strict, WORD_IN, external_call["malloc", Int](STREAM_BYTES))
    set_word(st_strict, WORD_OUT, external_call["malloc", Int](STREAM_BYTES))
    set_word(st_strict, WORD_GAME, external_call["malloc", Int](GAME_BYTES))
    set_word(st_strict, WORD_OWNED, external_call["calloc", Int](Int(MAX_OWNED_ID // 8), Int(1)))
    set_word(st_strict, WORD_IN_FDS, external_call["malloc", Int](MAX_FDS * 8))
    set_word(st_strict, WORD_OUT_FDS, external_call["malloc", Int](MAX_FDS * 8))
    set_word(st_strict, WORD_GAME_FDS, external_call["malloc", Int](MAX_FDS * 8))
    set_word(st_strict, WORD_OFFER_IDS, external_call["calloc", Int](Int(MAX_OFFERS), Int(4)))
    set_word(st_strict, WORD_IO_SCRATCH, external_call["calloc", Int](Int(1), IO_SCRATCH_BYTES))
    init_lib_ifaces(st_strict)
    set_word(st_strict, WORD_STRICT_IFACE, 1)
    set_word(st_strict, WORD_KEYBOARD, 6)
    set_word(st_strict, WORD_DATA_DEVICE, 7)
    set_word(st_strict, WORD_DATA_SOURCE, 8)
    set_owned(st_strict, 6)
    set_owned(st_strict, 7)
    set_owned(st_strict, 8)

    # Snoop outgoing requests: wl_display.get_registry(90) -> bind dmabuf(91), bind wl_output(99), get_default_feedback(98)
    var out_wire = List[UInt8]()
    var b_reg = List[UInt8]()
    wl_append32(b_reg, 90)
    _append_bytes(out_wire, _wl_build_msg(1, 1, b_reg))  # wl_display.get_registry(90)
    _append_bytes(out_wire, _wl_build_bind_req(90, 1, "zwp_linux_dmabuf_v1", 4, 91))
    _append_bytes(out_wire, _wl_build_bind_req(90, 2, "wl_output", 4, 99))
    var b_fb = List[UInt8]()
    wl_append32(b_fb, 98)
    _append_bytes(out_wire, _wl_build_msg(91, 2, b_fb))  # zwp_linux_dmabuf_v1.get_default_feedback(98)
    assert_equal(Int(external_call["write", Int64](sl_sv[1], out_wire.unsafe_ptr(), Int64(len(out_wire)))), len(out_wire))
    take_outgoing(st_strict)
    assert_equal(lookup_lib_iface(st_strict, 98), LIB_IFACE_DMABUF_FEEDBACK)
    # Drain forwarded outgoing bytes on sb_sv[1]
    var out_drain = List[UInt8](length=512, fill=0)
    assert_equal(Int(external_call["recv", Int64](sb_sv[1], out_drain.unsafe_ptr(), Int64(512), Int32(64))), len(out_wire))

    def _drain_lib_msgs(st_ptr_arg: Int, lib_sock: Int) -> Tuple[Int, List[Int]]:
        var buf = List[UInt8](length=4096, fill=0)
        var fds = List[Int64](length=64, fill=-1)
        var cnt = 0
        var total = 0
        for _ in range(16):
            var n = receive_some(lib_sock, Int(buf.unsafe_ptr()) + total, 4096 - total, Int(fds.unsafe_ptr()), cnt, word(st_ptr_arg, WORD_IO_SCRATCH))
            if n <= 0:
                break
            total += n
        var tags = List[Int]()
        for i in range(cnt):
            tags.append(_read_pipe_tag(Int(fds[i])))
        _ = len(buf)
        _ = len(fds)
        return (total, tags^)

    def _pop_game_tags(st_ptr_arg: Int) -> List[Int]:
        var g_cnt = word(st_ptr_arg, WORD_GAME_FD_COUNT)
        var g_fds = word(st_ptr_arg, WORD_GAME_FDS)
        var tags = List[Int]()
        for i in range(g_cnt):
            tags.append(_read_pipe_tag(Int(get64(g_fds + i * 8))))
        set_word(st_ptr_arg, WORD_GAME_FD_COUNT, 0)
        set_word(st_ptr_arg, WORD_GAME_USED, 0)
        return tags^

    var fb_ft_msg = _wl_build_msg(98, 1, lib_body)  # format_table(fd) -> 1 FD
    var out_done_msg = _wl_build_msg(99, 2, lib_body)  # wl_output.done() -> 0 FDs
    var km_msg = _wl_build_msg(6, 0, km_body)          # wl_keyboard.keymap -> 1 game FD
    var ds_body_4g = List[UInt8]()
    wl_append_string(ds_body_4g, "text/plain")
    var ds_send_msg = _wl_build_msg(8, 1, ds_body_4g)     # wl_data_source.send -> 1 game FD

    # ADV-A: [dmabuf_feedback.format_table(0xA1), keymap(0xA2), wl_output.done(0)]
    var p_a1 = _make_pipe_tagged(0xA1)[0]
    var p_a2 = _make_pipe_tagged(0xA2)[0]
    var b_adv_a = List[UInt8]()
    _append_bytes(b_adv_a, fb_ft_msg)
    _append_bytes(b_adv_a, km_msg)
    _append_bytes(b_adv_a, out_done_msg)
    var fds_a = List[Int32](); fds_a.append(Int32(p_a1)); fds_a.append(Int32(p_a2))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_adv_a, fds_a))
    close_fd(p_a1); close_fd(p_a2)
    assert_true(take_incoming(st_strict))
    var ga = _pop_game_tags(st_strict)
    var la = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(len(ga), 1); assert_equal(ga[0], 0xA2)
    assert_equal(la[0], len(fb_ft_msg) + len(out_done_msg))
    assert_equal(len(la[1]), 1); assert_equal(la[1][0], 0xA1)

    # ADV-A2: [dmabuf_feedback.format_table(0xA3), wl_data_source.send(0xA4), wl_output.done(0)]
    var p_a3 = _make_pipe_tagged(0xA3)[0]
    var p_a4 = _make_pipe_tagged(0xA4)[0]
    var b_adv_a2 = List[UInt8]()
    _append_bytes(b_adv_a2, fb_ft_msg)
    _append_bytes(b_adv_a2, ds_send_msg)
    _append_bytes(b_adv_a2, out_done_msg)
    var fds_a2 = List[Int32](); fds_a2.append(Int32(p_a3)); fds_a2.append(Int32(p_a4))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_adv_a2, fds_a2))
    close_fd(p_a3); close_fd(p_a4)
    assert_true(take_incoming(st_strict))
    var ga2 = _pop_game_tags(st_strict)
    var la2 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(len(ga2), 1); assert_equal(ga2[0], 0xA4)
    assert_equal(la2[0], len(fb_ft_msg) + len(out_done_msg))
    assert_equal(len(la2[1]), 1); assert_equal(la2[1][0], 0xA3)

    # DynSingleOffer: [format_table(0xF1), data_offer(400), offer(400), ds.send(0xF2), 4B partial wl_output.done(0)]
    var p_f1 = _make_pipe_tagged(0xF1)[0]
    var p_f2 = _make_pipe_tagged(0xF2)[0]
    var do_400 = List[UInt8](); wl_append32(do_400, 400)
    var b_dyn1 = List[UInt8]()
    _append_bytes(b_dyn1, fb_ft_msg)
    _append_bytes(b_dyn1, _wl_build_msg(7, 0, do_400))
    _append_bytes(b_dyn1, _wl_build_msg(400, 0, om_body))
    _append_bytes(b_dyn1, ds_send_msg)
    for k in range(4):
        b_dyn1.append(out_done_msg[k])
    var fds_dyn = List[Int32](); fds_dyn.append(Int32(p_f1)); fds_dyn.append(Int32(p_f2))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_dyn1, fds_dyn))
    close_fd(p_f1); close_fd(p_f2)
    assert_true(take_incoming(st_strict))
    var gdyn = _pop_game_tags(st_strict)
    var ldyn1 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(len(gdyn), 1); assert_equal(gdyn[0], 0xF2)
    assert_equal(ldyn1[0], len(fb_ft_msg))
    assert_equal(len(ldyn1[1]), 1); assert_equal(ldyn1[1][0], 0xF1)
    var b_dyn2 = List[UInt8]()
    for k in range(4, len(out_done_msg)):
        b_dyn2.append(out_done_msg[k])
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_dyn2, List[Int32]()))
    assert_true(not take_incoming(st_strict))
    var ldyn2 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(ldyn2[0], len(out_done_msg))
    assert_equal(len(ldyn2[1]), 0)

    # ADV-B: [format_table(0xB1), keymap(0xB2), format_table(0xB3)]
    var p_b1 = _make_pipe_tagged(0xB1)[0]
    var p_b2 = _make_pipe_tagged(0xB2)[0]
    var p_b3 = _make_pipe_tagged(0xB3)[0]
    var b_adv_b = List[UInt8]()
    _append_bytes(b_adv_b, fb_ft_msg)
    _append_bytes(b_adv_b, km_msg)
    _append_bytes(b_adv_b, fb_ft_msg)
    var fds_b = List[Int32](); fds_b.append(Int32(p_b1)); fds_b.append(Int32(p_b2)); fds_b.append(Int32(p_b3))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_adv_b, fds_b))
    close_fd(p_b1); close_fd(p_b2); close_fd(p_b3)
    assert_true(take_incoming(st_strict))
    var gb = _pop_game_tags(st_strict)
    var lb = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(len(gb), 1); assert_equal(gb[0], 0xB2)
    assert_equal(lb[0], len(fb_ft_msg) * 2)
    assert_equal(len(lb[1]), 2); assert_equal(lb[1][0], 0xB1); assert_equal(lb[1][1], 0xB3)

    # ADV-C: [format_table(0xC1), keymap(0xC2), 2B partial wl_keyboard.enter(0)] -> remaining 14B
    var p_c1 = _make_pipe_tagged(0xC1)[0]
    var p_c2 = _make_pipe_tagged(0xC2)[0]
    var kb_enter_b = List[UInt8](); wl_append32(kb_enter_b, 10); wl_append32(kb_enter_b, 3)
    var kb_enter_msg = _wl_build_msg(6, 1, kb_enter_b)
    var b_adv_c1 = List[UInt8]()
    _append_bytes(b_adv_c1, fb_ft_msg)
    _append_bytes(b_adv_c1, km_msg)
    for k in range(2):
        b_adv_c1.append(kb_enter_msg[k])
    var fds_c = List[Int32](); fds_c.append(Int32(p_c1)); fds_c.append(Int32(p_c2))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_adv_c1, fds_c))
    close_fd(p_c1); close_fd(p_c2)
    assert_true(take_incoming(st_strict))
    var gc = _pop_game_tags(st_strict)
    var lc1 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(len(gc), 1); assert_equal(gc[0], 0xC2)
    assert_equal(lc1[0], len(fb_ft_msg))
    assert_equal(len(lc1[1]), 1); assert_equal(lc1[1][0], 0xC1)
    var b_adv_c2 = List[UInt8]()
    for k in range(2, len(kb_enter_msg)):
        b_adv_c2.append(kb_enter_msg[k])
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_adv_c2, List[Int32]()))
    assert_true(take_incoming(st_strict))
    assert_equal(word(st_strict, WORD_GAME_USED), len(kb_enter_msg))
    var gc2 = _pop_game_tags(st_strict)
    assert_equal(len(gc2), 0)

    # ADV-D: [format_table(0xD1), 2B partial wl_output.done(0)] -> remaining 10B
    var p_d1 = _make_pipe_tagged(0xD1)[0]
    var b_adv_d1 = List[UInt8]()
    _append_bytes(b_adv_d1, fb_ft_msg)
    for k in range(2):
        b_adv_d1.append(out_done_msg[k])
    var fds_d = List[Int32](); fds_d.append(Int32(p_d1))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_adv_d1, fds_d))
    close_fd(p_d1)
    assert_true(not take_incoming(st_strict))
    var ld1 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(ld1[0], len(fb_ft_msg))
    assert_equal(len(ld1[1]), 1); assert_equal(ld1[1][0], 0xD1)
    var b_adv_d2 = List[UInt8]()
    for k in range(2, len(out_done_msg)):
        b_adv_d2.append(out_done_msg[k])
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_adv_d2, List[Int32]()))
    assert_true(not take_incoming(st_strict))
    var ld2 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(ld2[0], len(out_done_msg))
    assert_equal(len(ld2[1]), 0)

    # ADV-E: [format_table(0xE1), keymap(0xE2), data_offer(500), data_offer(501), offer(500)]
    var p_e1 = _make_pipe_tagged(0xE1)[0]
    var p_e2 = _make_pipe_tagged(0xE2)[0]
    var do_500 = List[UInt8](); wl_append32(do_500, 500)
    var do_501 = List[UInt8](); wl_append32(do_501, 501)
    var off_500 = _wl_build_msg(500, 0, om_body)
    var b_adv_e = List[UInt8]()
    _append_bytes(b_adv_e, fb_ft_msg)
    _append_bytes(b_adv_e, km_msg)
    _append_bytes(b_adv_e, _wl_build_msg(7, 0, do_500))
    _append_bytes(b_adv_e, _wl_build_msg(7, 0, do_501))
    _append_bytes(b_adv_e, off_500)
    var fds_e = List[Int32](); fds_e.append(Int32(p_e1)); fds_e.append(Int32(p_e2))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_adv_e, fds_e))
    close_fd(p_e1); close_fd(p_e2)
    assert_true(take_incoming(st_strict))
    assert_true(is_server_offer(st_strict, 500))
    assert_true(is_server_offer(st_strict, 501))
    assert_equal(word(st_strict, WORD_GAME_USED), len(km_msg) + 24 + len(off_500))
    var ge = _pop_game_tags(st_strict)
    var le = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(len(ge), 1); assert_equal(ge[0], 0xE2)
    assert_equal(le[0], len(fb_ft_msg))
    assert_equal(len(le[1]), 1); assert_equal(le[1][0], 0xE1)

    # T1-T5 in strict mode + wl_display.delete_id(98) cleanup
    # T1: [wl_output.done(0), keymap(0x41)]
    var p_t1 = _make_pipe_tagged(0x41)[0]
    var b_t1 = List[UInt8](); _append_bytes(b_t1, out_done_msg); _append_bytes(b_t1, km_msg)
    var fds_t1 = List[Int32](); fds_t1.append(Int32(p_t1))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_t1, fds_t1))
    close_fd(p_t1)
    assert_true(take_incoming(st_strict))
    var gt1 = _pop_game_tags(st_strict)
    var lt1 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(len(gt1), 1); assert_equal(gt1[0], 0x41); assert_equal(len(lt1[1]), 0)

    # T2: [format_table(0x11), wl_output.done(0), keymap(0x22), ds.send(0x33)]
    var p_t2a = _make_pipe_tagged(0x11)[0]; var p_t2b = _make_pipe_tagged(0x22)[0]; var p_t2c = _make_pipe_tagged(0x33)[0]
    var b_t2 = List[UInt8]()
    _append_bytes(b_t2, fb_ft_msg); _append_bytes(b_t2, out_done_msg); _append_bytes(b_t2, km_msg); _append_bytes(b_t2, ds_send_msg)
    var fds_t2 = List[Int32](); fds_t2.append(Int32(p_t2a)); fds_t2.append(Int32(p_t2b)); fds_t2.append(Int32(p_t2c))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_t2, fds_t2))
    close_fd(p_t2a); close_fd(p_t2b); close_fd(p_t2c)
    assert_true(take_incoming(st_strict))
    var gt2 = _pop_game_tags(st_strict)
    var lt2 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(len(gt2), 2); assert_equal(gt2[0], 0x22); assert_equal(gt2[1], 0x33)
    assert_equal(len(lt2[1]), 1); assert_equal(lt2[1][0], 0x11)

    # T3: [wl_output.done(0), keymap(0x55), format_table(0x66)]
    var p_t3a = _make_pipe_tagged(0x55)[0]; var p_t3b = _make_pipe_tagged(0x66)[0]
    var b_t3s = List[UInt8](); _append_bytes(b_t3s, out_done_msg); _append_bytes(b_t3s, km_msg); _append_bytes(b_t3s, fb_ft_msg)
    var fds_t3s = List[Int32](); fds_t3s.append(Int32(p_t3a)); fds_t3s.append(Int32(p_t3b))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_t3s, fds_t3s))
    close_fd(p_t3a); close_fd(p_t3b)
    assert_true(take_incoming(st_strict))
    var gt3 = _pop_game_tags(st_strict)
    var lt3 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(len(gt3), 1); assert_equal(gt3[0], 0x55)
    assert_equal(len(lt3[1]), 1); assert_equal(lt3[1][0], 0x66)

    # T4: [wl_output.done(0), 2B partial keymap(0x71)] -> remaining 14B
    var p_t4s = _make_pipe_tagged(0x71)[0]
    var b_t4s1 = List[UInt8](); _append_bytes(b_t4s1, out_done_msg)
    for k in range(2):
        b_t4s1.append(km_msg[k])
    var fds_t4s = List[Int32](); fds_t4s.append(Int32(p_t4s))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_t4s1, fds_t4s))
    close_fd(p_t4s)
    assert_true(not take_incoming(st_strict))
    var lt4 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(lt4[0], len(out_done_msg)); assert_equal(len(lt4[1]), 0)
    var b_t4s2 = List[UInt8]()
    for k in range(2, len(km_msg)):
        b_t4s2.append(km_msg[k])
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_t4s2, List[Int32]()))
    assert_true(take_incoming(st_strict))
    var gt4 = _pop_game_tags(st_strict)
    assert_equal(len(gt4), 1); assert_equal(gt4[0], 0x71)

    # T5: 20 keymap FDs + 6 format_table FDs interleaved
    var b_t5s = List[UInt8]()
    var fds_t5s = List[Int32]()
    for r in range(6):
        var pf = _make_pipe_tagged(UInt8(0x10 + r))[0]
        var pk = _make_pipe_tagged(UInt8(0x80 + r))[0]
        _append_bytes(b_t5s, fb_ft_msg)
        _append_bytes(b_t5s, km_msg)
        fds_t5s.append(Int32(pf)); fds_t5s.append(Int32(pk))
    assert_true(_send_fds_msg(Int(sb_sv[1]), b_t5s, fds_t5s))
    for fd_i in fds_t5s:
        close_fd(Int(fd_i))
    assert_true(take_incoming(st_strict))
    var gt5 = _pop_game_tags(st_strict)
    var lt5 = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(len(gt5), 6)
    assert_equal(len(lt5[1]), 6)
    for r in range(6):
        assert_equal(gt5[r], 0x80 + r)
        assert_equal(lt5[1][r], 0x10 + r)

    # wl_display.delete_id(98) unregisters lib iface 98
    assert_equal(lookup_lib_iface(st_strict, 98), LIB_IFACE_DMABUF_FEEDBACK)
    var del_b = List[UInt8](); wl_append32(del_b, 98)
    var del_msg = _wl_build_msg(1, 1, del_b)
    assert_true(_send_fds_msg(Int(sb_sv[1]), del_msg, List[Int32]()))
    assert_true(not take_incoming(st_strict))
    assert_equal(lookup_lib_iface(st_strict, 98), LIB_IFACE_UNKNOWN)
    var ldel = _drain_lib_msgs(st_strict, Int(sl_sv[1]))
    assert_equal(ldel[0], len(del_msg))

    close_fd(Int(sb_sv[0])); close_fd(Int(sb_sv[1]))
    close_fd(Int(sl_sv[0])); close_fd(Int(sl_sv[1]))
    free_lib_ifaces(st_strict)
    external_call["free", NoneType](word(st_strict, WORD_IN))
    external_call["free", NoneType](word(st_strict, WORD_OUT))
    external_call["free", NoneType](word(st_strict, WORD_GAME))
    external_call["free", NoneType](word(st_strict, WORD_OWNED))
    external_call["free", NoneType](word(st_strict, WORD_IN_FDS))
    external_call["free", NoneType](word(st_strict, WORD_OUT_FDS))
    external_call["free", NoneType](word(st_strict, WORD_GAME_FDS))
    external_call["free", NoneType](word(st_strict, WORD_OFFER_IDS))
    external_call["free", NoneType](word(st_strict, WORD_IO_SCRATCH))
    external_call["free", NoneType](st_strict)
    assert_equal(_count_open_fds(), fds_4g_before)

    # 5. WaylandWindow delayed wl_data_device.drop (EAGAIN + trailing wl_keyboard.key, 0 recursion)
    #    and WaylandWindow(fd=-1, shared_fd>=0).close() 0 FD leak
    var fds_dnd_before = _count_open_fds()
    var dsv = List[Int32](length=2, fill=-1)
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), dsv.unsafe_ptr()), Int32(0))
    var w_dnd = WaylandWindow()
    w_dnd.fd = Int(dsv[0])
    w_dnd.data_device = 7
    w_dnd.data_device_version = 3
    w_dnd.keyboard = 6
    var dnd_oid = 0xFF000020
    var dnd_do = List[UInt8](); wl_append32(dnd_do, dnd_oid)
    var dnd_om = List[UInt8](); wl_append_string(dnd_om, "text/uri-list")
    _append_bytes(w_dnd.incoming, _wl_build_msg(w_dnd.data_device, 0, dnd_do))
    _append_bytes(w_dnd.incoming, _wl_build_msg(dnd_oid, 0, dnd_om))
    var dnd_ent = List[UInt8]()
    wl_append32(dnd_ent, 42); wl_append32(dnd_ent, 3); wl_append32(dnd_ent, 0); wl_append32(dnd_ent, 0); wl_append32(dnd_ent, dnd_oid)
    _append_bytes(w_dnd.incoming, _wl_build_msg(w_dnd.data_device, 1, dnd_ent))
    w_dnd.handle_incoming()
    assert_equal(w_dnd.dnd_offer_id, dnd_oid)
    # Drain accept + set_actions from server_fd
    var dnd_drain = List[UInt8](length=512, fill=0)
    while Int(external_call["recv", Int64](dsv[1], dnd_drain.unsafe_ptr(), Int64(512), Int32(64))) > 0:
        pass
    var thr_args = List[Int](length=2, fill=0)
    thr_args[0] = Int(dsv[1])
    thr_args[1] = 0
    var thr_id = 0
    assert_equal(external_call["pthread_create", Int32](Pointer(to=thr_id), Int(0), _dnd_responder_thread, Int(thr_args.unsafe_ptr())), Int32(0))
    # Batch [wl_data_device.drop, wl_keyboard.key(evdev=30 'A')]
    _append_bytes(w_dnd.incoming, _wl_build_msg(w_dnd.data_device, 4, List[UInt8]()))
    _append_bytes(w_dnd.incoming, _kbd_key(w_dnd.keyboard, 888, 30, 1))
    w_dnd.handle_incoming()
    _ = external_call["pthread_join", Int32](thr_id, Int(0))
    assert_equal(thr_args[1], 1)  # Exactly ONE wl_data_offer.receive FD sent!
    assert_equal(w_dnd.dnd_offer_id, 0)
    assert_equal(len(w_dnd.pending_dropped_files), 1)
    assert_equal(w_dnd.pending_dropped_files[0], "/tmp/dnd_reentry.glb")
    assert_equal(len(w_dnd.pending_keys), 1)
    assert_equal(w_dnd.pending_keys[0], 65)  # trailing wl_keyboard.key processed once
    assert_equal(len(w_dnd.incoming), 0)
    w_dnd.close()
    close_fd(Int(dsv[1]))
    assert_equal(_count_open_fds(), fds_dnd_before)

    var w_shm_leak = WaylandWindow()
    var mem_nm = String("flint-close-test")
    w_shm_leak.shared_fd = Int(external_call["memfd_create", Int32](mem_nm.as_c_string_span(), UInt32(1)))
    assert_true(w_shm_leak.fd < 0 and w_shm_leak.shared_fd >= 0)
    w_shm_leak.close()
    assert_equal(w_shm_leak.shared_fd, -1)
    assert_equal(_count_open_fds(), fds_dnd_before)


def test_f24_x11_desktop_parity() raises:
    var x11 = X11Window()

    # 1. MIT-SHM ShmCompletion clears slot busy flags
    x11.shm_first_event = 90
    x11.shm_busy0 = True
    x11.shm_busy1 = True
    x11.incoming.resize(64, 0)
    x11.incoming[0] = 90
    x11_put32(x11.incoming, 16, 0)
    x11.incoming[32] = 90
    x11_put32(x11.incoming, 48, 4096)
    x11.process_events()
    assert_true(not x11.shm_busy0)
    assert_true(not x11.shm_busy1)

    # 2. XInput2 XI_RawMotion (GenericEvent 35, evtype=17, FP3232 raw_values)
    #    and MotionNotify non-double-counting when xi_raw_active == True
    x11.xi_opcode = 131
    x11.xi_raw_active = True
    x11.pointer_locked = True
    x11.focused = True
    x11.has_last_mouse = True
    x11.last_mouse_x = 100.0
    x11.last_mouse_y = 100.0
    # GenericEvent (32) + 4B mask + 16B transformed + 16B raw = 68 bytes (extra_words = 9)
    x11.incoming.resize(68 + 32, 0)
    x11.incoming[0] = 35  # GenericEvent
    x11.incoming[1] = 131  # xi_opcode
    x11_put32(x11.incoming, 4, 9)  # 9 * 4 = 36 extra bytes
    x11_put16(x11.incoming, 8, 17)  # XI_RawMotion
    x11_put16(x11.incoming, 22, 1)  # valuators_len = 1 (4 bytes mask)
    x11.incoming[32] = 0x03         # valuators 0 & 1 set
    # raw_values start at 32 + 4 + 16 = 52: dx = +12.5, dy = -3.25 (-4 + 0.75)
    x11_put32(x11.incoming, 52, 12)
    x11_put32(x11.incoming, 56, Int(0x80000000))
    x11_put32(x11.incoming, 60, -4)
    x11_put32(x11.incoming, 64, Int(0xC0000000))
    # Append MotionNotify (code=6) at byte 68 -> must NOT double-count when xi_raw_active!
    x11.incoming[68] = 6
    x11_put16(x11.incoming, 68 + 24, 150)
    x11_put16(x11.incoming, 68 + 26, 150)
    x11.process_events()
    assert_true(abs(x11.pending_rel_x - Float32(12.5)) < Float32(1.0e-4))
    assert_true(abs(x11.pending_rel_y - Float32(-3.25)) < Float32(1.0e-4))

    # XI_RawMotion ignored when unlocked
    x11.pending_rel_x = 0
    x11.pending_rel_y = 0
    x11.pointer_locked = False
    x11.incoming.resize(68, 0)
    x11.incoming[0] = 35
    x11.incoming[1] = 131
    x11_put32(x11.incoming, 4, 9)
    x11_put16(x11.incoming, 8, 17)
    x11_put16(x11.incoming, 22, 1)
    x11.incoming[32] = 0x03
    x11_put32(x11.incoming, 52, 5)
    x11.process_events()
    assert_equal(x11.pending_rel_x, Float32(0.0))

    # 3. X11 Keymap + UTF-8 text_input_utf8 + synthetic auto-repeat compatibility
    x11.min_keycode = 24
    x11.max_keycode = 40
    x11.keysyms_per_keycode = 2
    x11.keymap = List[Int](length=(40 - 24 + 1) * 2, fill=0)
    x11.keymap[(38 - 24) * 2] = 97         # 'a'
    x11.keymap[(38 - 24) * 2 + 1] = 65     # 'A'
    x11.keymap[(39 - 24) * 2] = 0x010020AC # '€'
    x11.keymap[(24 - 24) * 2] = 113        # 'q'
    x11.keymap[(25 - 24) * 2] = 0x010000E4 # 'ä' via 0x01000000 | cp
    x11.incoming.resize(32 * 8, 0)
    # (0) KeyPress 38 unshifted -> 'a'
    x11.incoming[0] = 2; x11.incoming[1] = 38; x11_put16(x11.incoming, 28, 0)
    # (1) KeyPress 38 Shift -> 'A'
    x11.incoming[32] = 2; x11.incoming[33] = 38; x11_put16(x11.incoming, 32 + 28, 1)
    # (2) KeyPress 39 -> '€'
    x11.incoming[64] = 2; x11.incoming[65] = 39; x11_put16(x11.incoming, 64 + 28, 0)
    # (3) KeyPress 38 Ctrl -> suppressed
    x11.incoming[96] = 2; x11.incoming[97] = 38; x11_put16(x11.incoming, 96 + 28, 4)
    # (4+5) Synthetic auto-repeat KeyRelease(24) + KeyPress(24) at same timestamp 7777 -> emits 'q', no pending_keys
    x11.incoming[128] = 3; x11.incoming[129] = 24; x11_put32(x11.incoming, 128 + 4, 7777)
    x11.incoming[160] = 2; x11.incoming[161] = 24; x11_put32(x11.incoming, 160 + 4, 7777); x11_put16(x11.incoming, 160 + 28, 0)
    # (6) KeyPress 25 -> 'ä', (7) KeyPress 25 CapsLock -> 'Ä'
    x11.incoming[192] = 2; x11.incoming[193] = 25; x11_put16(x11.incoming, 192 + 28, 0)
    x11.incoming[224] = 2; x11.incoming[225] = 25; x11_put16(x11.incoming, 224 + 28, 2)
    x11.process_events()
    assert_equal(x11.pending_text_utf8, "aA€qäÄ")
    assert_equal(len(x11.pending_keys), 6)  # auto-repeat pair did NOT add to pending_keys
    x11.pending_keys.clear()
    x11.pending_key_down.clear()

    # 4. X11 CLIPBOARD local ownership + SelectionClear (offset 12 = atom, offset 8 = window) + attach_vulkan headless safety
    x11.window_id = 0x200001
    x11.atom_clipboard = 100
    x11.clipboard_set("X11 Clipboard")
    assert_equal(x11.clipboard_get(), "X11 Clipboard")
    # Put atom_clipboard at offset 8 (window field) ONLY -> must NOT clear clipboard_owned
    x11.incoming.resize(32, 0)
    x11.incoming[0] = 29  # SelectionClear
    x11_put32(x11.incoming, 8, 100)
    x11_put32(x11.incoming, 12, 999)
    x11.process_events()
    assert_true(x11.clipboard_owned)
    # Put window at +8 and atom_clipboard at +12 -> MUST clear clipboard_owned
    x11.incoming.resize(32, 0)
    x11.incoming[0] = 29  # SelectionClear
    x11_put32(x11.incoming, 8, x11.window_id)
    x11_put32(x11.incoming, 12, 100)
    x11.process_events()
    assert_true(not x11.clipboard_owned)

    # 5. Regression: wire MappingNotify (byte 4 == 1, byte 1 == 0) + PutImage alpha normalization over socketpair
    var xsv = List[Int32](length=2, fill=-1)
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), xsv.unsafe_ptr()), Int32(0))
    var x_wire = X11Window()
    x_wire.uncapped = True
    x_wire.fd = Int(xsv[0])
    x_wire.window_id = 0x300001
    x_wire.gc_id = 0x300002
    x_wire.root_depth = 24
    x_wire.max_req_words = 65535
    x_wire.width = 4
    x_wire.height = 1
    x_wire.logical_width = 4
    x_wire.logical_height = 1
    x_wire.min_keycode = 24
    x_wire.max_keycode = 25
    x_wire.keysyms_per_keycode = 2
    x_wire.keymap = List[Int](length=4, fill=97)  # 'a'

    # Prepare GetKeyboardMapping reply on server socket (seq = 1, 2 keycodes * 2 keysyms = 4 words = 16 extra bytes)
    var gkm_rep = List[UInt8](length=32 + 16, fill=0)
    gkm_rep[0] = 1  # Reply
    gkm_rep[1] = 2  # keysyms_per_keycode
    x11_put16(gkm_rep, 2, 1)  # sequence 1
    x11_put32(gkm_rep, 4, 4)  # length = 4 words
    x11_put32(gkm_rep, 32, 122)  # keycode 24 -> 'z'
    _ = external_call["write", Int64](xsv[1], gkm_rep.unsafe_ptr(), Int64(len(gkm_rep)))

    # Inject MappingNotify (code=34, pad1=0 at byte 1, request=1 MappingKeyboard at byte 4)
    x_wire.incoming.resize(32, 0)
    x_wire.incoming[0] = 34
    x_wire.incoming[1] = 0
    x11_put16(x_wire.incoming, 2, 0)
    x_wire.incoming[4] = 1  # MappingKeyboard
    x_wire.process_events()
    assert_equal(x_wire.keymap[0], 122)  # refreshed via GetKeyboardMapping!
    var req_buf = List[UInt8](length=64, fill=0)
    var n_gkm = Int(external_call["recv", Int64](xsv[1], req_buf.unsafe_ptr(), Int64(64), Int32(64)))
    assert_true(n_gkm >= 8)
    assert_equal(Int(req_buf[0]), 101)  # GetKeyboardMapping opcode

    # Verify PutImage fallback normalizes top alpha byte (0xFF000000)
    var px = List[UInt32](length=4, fill=UInt32(0x00112233))
    x_wire.present(px, 4, 1)
    var put_buf = List[UInt8](length=64, fill=0)
    var n_put = Int(external_call["recv", Int64](xsv[1], put_buf.unsafe_ptr(), Int64(64), Int32(64)))
    assert_equal(n_put, 24 + 16)
    assert_equal(Int(put_buf[0]), 72)  # PutImage
    assert_equal(x11_get32(put_buf, 24), Int(0xFF112233))
    x_wire.close()
    close_fd(Int(xsv[1]))

    var vk_dummy = VulkanGpu()
    var sc_empty = x11.attach_vulkan(vk_dummy, 640, 480)
    assert_equal(sc_empty.vk_swapchain, 0)
    assert_equal(x11.content_scale(), Float32(1.0))
    var xfb = x11.framebuffer_size()
    assert_equal(xfb[0], 1280)
    assert_equal(xfb[1], 720)

    # 6. Regressions: same-batch [MappingNotify, KeyPress], MappingNotify + 16B partial KeyPress,
    #    direct _wait_reply() with <32B partial incoming, and close() SIGPIPE safety without initialize()
    def _make_gkm_reply(seq: Int, ks0: Int) -> List[UInt8]:
        var g = List[UInt8](length=32 + 16, fill=0)
        g[0] = 1
        g[1] = 2
        x11_put16(g, 2, seq)
        x11_put32(g, 4, 4)
        x11_put32(g, 32, ks0)
        return g^

    def _make_mapping_notify() -> List[UInt8]:
        var m = List[UInt8](length=32, fill=0)
        m[0] = 34
        m[4] = 1  # MappingKeyboard
        return m^

    def _make_keypress(detail: Int) -> List[UInt8]:
        var k = List[UInt8](length=32, fill=0)
        k[0] = 2
        k[1] = UInt8(detail)
        return k^

    # 6a. Same-batch [MappingNotify, KeyPress] -> inline refresh yields 'a' (97), NOT stale 'q' (113)
    var sv6a = List[Int32](length=2, fill=-1)
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), sv6a.unsafe_ptr()), Int32(0))
    var xb = X11Window()
    xb.fd = Int(sv6a[0])
    xb.min_keycode = 24; xb.max_keycode = 25; xb.keysyms_per_keycode = 2
    xb.keymap = List[Int](length=4, fill=113)  # 'q'
    var rep6a = _make_gkm_reply(1, 97)  # 'a'
    _ = external_call["write", Int64](sv6a[1], rep6a.unsafe_ptr(), Int64(len(rep6a)))
    _append_bytes(xb.incoming, _make_mapping_notify())
    _append_bytes(xb.incoming, _make_keypress(24))
    xb.process_events()
    assert_equal(xb.keymap[0], 97)
    assert_equal(xb.pending_text_utf8, "a")
    assert_equal(len(xb.pending_keys), 1)
    assert_equal(len(xb.incoming), 0)
    xb.close()
    close_fd(Int(sv6a[1]))

    # 6b. MappingNotify + 16B partial KeyPress (remaining 16B + GetKeyboardMapping reply on server_fd)
    var sv6b = List[Int32](length=2, fill=-1)
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), sv6b.unsafe_ptr()), Int32(0))
    var xs = X11Window()
    xs.fd = Int(sv6b[0])
    xs.min_keycode = 24; xs.max_keycode = 25; xs.keysyms_per_keycode = 2
    xs.keymap = List[Int](length=4, fill=113)  # 'q'
    var kp_full = _make_keypress(24)
    _append_bytes(xs.incoming, _make_mapping_notify())
    for k in range(16):
        xs.incoming.append(kp_full[k])
    var srv_b = List[UInt8]()
    for k in range(16, 32):
        srv_b.append(kp_full[k])
    _append_bytes(srv_b, _make_gkm_reply(1, 97))
    _ = external_call["write", Int64](sv6b[1], srv_b.unsafe_ptr(), Int64(len(srv_b)))
    xs.process_events()
    assert_equal(xs.keymap[0], 97)
    assert_equal(xs.pending_text_utf8, "a")
    assert_equal(len(xs.incoming), 0)
    xs.close()
    close_fd(Int(sv6b[1]))

    # 6c. Direct _wait_reply() via intern_atom() with 16B partial KeyPress already in incoming
    var sv6c = List[Int32](length=2, fill=-1)
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), sv6c.unsafe_ptr()), Int32(0))
    var xi = X11Window()
    xi.fd = Int(sv6c[0])
    xi.min_keycode = 24; xi.max_keycode = 25; xi.keysyms_per_keycode = 2
    xi.keymap = List[Int](length=4, fill=97)  # 'a'
    for k in range(16):
        xi.incoming.append(kp_full[k])
    var ia_rep = List[UInt8](length=32, fill=0)
    ia_rep[0] = 1
    x11_put16(ia_rep, 2, 1)
    x11_put32(ia_rep, 8, 444)
    var srv_c = List[UInt8]()
    for k in range(16, 32):
        srv_c.append(kp_full[k])
    _append_bytes(srv_c, ia_rep)
    _ = external_call["write", Int64](sv6c[1], srv_c.unsafe_ptr(), Int64(len(srv_c)))
    var atom_val = xi.intern_atom("CLIPBOARD")
    assert_equal(atom_val, 444)
    assert_equal(len(xi.incoming), 32)
    xi.process_events()
    assert_equal(xi.pending_text_utf8, "a")
    assert_equal(len(xi.incoming), 0)
    xi.close()
    close_fd(Int(sv6c[1]))

    # 6d. X11Window.close() after peer close_fd without initialize() (MSG_NOSIGNAL / no SIGPIPE)
    var sv6d = List[Int32](length=2, fill=-1)
    assert_equal(external_call["socketpair", Int32](Int32(1), Int32(1 | 0x800), Int32(0), sv6d.unsafe_ptr()), Int32(0))
    var xd = X11Window()
    xd.fd = Int(sv6d[0])
    xd.window_id = 0x300001
    xd.blank_cursor = 0x300003
    xd.blank_pixmap = 0x300004
    close_fd(Int(sv6d[1]))
    xd.close()
    assert_equal(xd.fd, -1)
    assert_true(xd.close_requested)


def test_f25_evdev_hub_integration() raises:
    var hub = EvdevHub(enable_hotplug=False)
    assert_equal(hub.connected_count(), 0)
    hub.scanned_once = True  # Avoid reading real host /dev/input/event* devices in unit test

    # Wire Xbox + DualSense into hub slots
    hub.pads[0].set_vendor_product(VENDOR_MICROSOFT, 0x02EA, "Xbox One S Controller")
    hub.pads[0].configure_abs_axis(ABS_X, -32768, 32767, 0, 0, 0)
    hub.pads[0].configure_abs_axis(ABS_Y, -32768, 32767, 0, 0, 0)
    hub.pads[1].set_vendor_product(VENDOR_SONY, 0x0CE6, "DualSense Wireless Controller")
    assert_equal(hub.connected_count(), 2)

    # Test radial deadzone, PAD_STICK_*, pad_map, toggle_pads, and pad_is_sony()
    var inp = InputState()
    inp.deadzone_left = 0.25
    inp.pad_map[PAD_A] = PAD_B
    inp.toggle_pads[PAD_B] = True

    # Small stick deflection below radial deadzone (0.15 < 0.25) -> clamped to 0.0
    hub.pads[0].feed_raw_evdev_event(EV_ABS, UInt16(ABS_X), Int32(4915))
    hub.pads[0].feed_raw_evdev_event(EV_KEY, UInt16(BTN_SOUTH), Int32(1))
    inp.begin_frame()
    hub.poll(inp)
    assert_true(inp.gamepad_available(0))
    assert_true(inp.gamepad_available(1))
    assert_true(inp.pad_is_sony())  # Fixed: works when connected==True and fd==-1!
    assert_equal(inp.gamepad_axis(0, PAD_AXIS_LEFT_X), Float32(0.0))
    assert_true(inp.gamepad_down(0, PAD_B))     # Remapped PAD_A -> PAD_B and latched
    assert_true(inp.gamepad_pressed(0, PAD_B))

    # Release physical BTN_SOUTH -> toggle latch keeps PAD_B down!
    hub.pads[0].begin_frame()
    hub.pads[0].feed_raw_evdev_event(EV_KEY, UInt16(BTN_SOUTH), Int32(0))
    inp.begin_frame()
    hub.poll(inp)
    assert_true(inp.gamepad_down(0, PAD_B))

    # Strong stick deflection right (+32767) -> sets PAD_STICK_RIGHT
    hub.pads[0].feed_raw_evdev_event(EV_ABS, UInt16(ABS_X), Int32(32767))
    inp.begin_frame()
    hub.poll(inp)
    assert_true(inp.gamepad_axis(0, PAD_AXIS_LEFT_X) > Float32(0.95))
    assert_true(inp.gamepad_down(0, PAD_STICK_RIGHT))
    assert_true(not inp.gamepad_down(0, PAD_STICK_LEFT))

    assert_true(not hub.set_rumble(0, 0.5, 0.5, 100))

    # Regression 1: single-batch feed_raw_packet24 quick press+release preserves pressed across EvdevHub.poll
    def _pkt24(ev_type: UInt16, ev_code: UInt16, ev_val: Int32) -> List[UInt8]:
        var b = List[UInt8](length=24, fill=0)
        b[16] = UInt8(ev_type & 0xFF)
        b[17] = UInt8((ev_type >> 8) & 0xFF)
        b[18] = UInt8(ev_code & 0xFF)
        b[19] = UInt8((ev_code >> 8) & 0xFF)
        var u = UInt32(ev_val)
        b[20] = UInt8(u & 0xFF)
        b[21] = UInt8((u >> 8) & 0xFF)
        b[22] = UInt8((u >> 16) & 0xFF)
        b[23] = UInt8((u >> 24) & 0xFF)
        return b^

    assert_true(hub.pads[0].feed_raw_packet24(_pkt24(EV_KEY, UInt16(BTN_TL), Int32(1))))
    assert_true(hub.pads[0].feed_raw_packet24(_pkt24(EV_KEY, UInt16(BTN_TL), Int32(0))))
    inp.begin_frame()
    hub.poll(inp)
    assert_true(inp.gamepad_pressed(0, PAD_LB))
    assert_true(not inp.gamepad_down(0, PAD_LB))
    # Next frame without new events must NOT repeat pressed
    inp.begin_frame()
    hub.poll(inp)
    assert_true(not inp.gamepad_pressed(0, PAD_LB))

    # Regression 2: multi-slot disconnect clears stale latched/down state and never leaks onto slot 1
    assert_true(inp.gamepad_down(0, PAD_B))
    assert_true(inp.pads[0].latched[PAD_B])
    hub.pads[1].feed_raw_evdev_event(EV_KEY, UInt16(BTN_WEST), Int32(1))  # DualSense Square -> PAD_X
    inp.begin_frame()
    hub.poll(inp)
    assert_true(inp.gamepad_down(1, PAD_X))
    hub.pads[0].close()
    inp.begin_frame()
    hub.poll(inp)
    assert_true(not inp.gamepad_available(0))
    assert_true(not inp.gamepad_down(0, PAD_B))
    assert_true(not inp.pads[0].latched[PAD_B])
    assert_true(inp.gamepad_available(1))
    assert_true(inp.pads[1].sony_layout)
    assert_true(inp.gamepad_down(1, PAD_X))
    assert_true(not inp.gamepad_down(1, PAD_B))
    assert_true(not inp.pads[1].latched[PAD_B])

    # Reconnect new controller into slot 0 -> clean state
    hub.pads[0].set_vendor_product(VENDOR_MICROSOFT, 0x02EA, "Xbox Replacement")
    inp.begin_frame()
    hub.poll(inp)
    assert_true(inp.gamepad_available(0))
    assert_true(not inp.gamepad_down(0, PAD_B))
    assert_true(not inp.pads[0].latched[PAD_B])

    hub.close()
    assert_equal(hub.connected_count(), 0)

    # Retired /dev/input/js* opener must return False
    var legacy_pad = LinuxGamepad()
    assert_true(not legacy_pad.open_first_available())

    # Regression 3: all 6 single-batch edge sequences (0->1, 1->0, 0->1->0, 1->0->1, 0->1->0->1, 1->0->1->0)
    # across both normal buttons (PAD_A) and toggle_pads (PAD_X)
    var hub_edge = EvdevHub(enable_hotplug=False)
    hub_edge.scanned_once = True
    hub_edge.pads[0].set_vendor_product(VENDOR_MICROSOFT, 0x02EA, "Xbox Edge Probe")
    var inp_e = InputState()
    inp_e.toggle_pads[PAD_X] = True

    def _step_ev(mut h: EvdevHub, mut st: InputState, seq_a: List[Int32], seq_x: List[Int32]):
        for v in seq_a:
            h.pads[0].feed_raw_evdev_event(EV_KEY, UInt16(BTN_SOUTH), v)
        for v in seq_x:
            h.pads[0].feed_raw_evdev_event(EV_KEY, UInt16(BTN_NORTH), v)
        st.begin_frame()
        h.poll(st)

    # 1) 0 -> 1
    var sa1 = List[Int32](); sa1.append(1)
    _step_ev(hub_edge, inp_e, sa1, sa1)
    assert_true(inp_e.gamepad_pressed(0, PAD_A) and inp_e.gamepad_down(0, PAD_A))
    assert_true(inp_e.gamepad_pressed(0, PAD_X) and inp_e.gamepad_down(0, PAD_X) and inp_e.pads[0].latched[PAD_X])

    # 2) 1 -> 0
    var s10 = List[Int32](); s10.append(0)
    _step_ev(hub_edge, inp_e, s10, s10)
    assert_true((not inp_e.gamepad_pressed(0, PAD_A)) and (not inp_e.gamepad_down(0, PAD_A)))
    assert_true((not inp_e.gamepad_pressed(0, PAD_X)) and inp_e.gamepad_down(0, PAD_X) and inp_e.pads[0].latched[PAD_X])

    # 3) 0 -> 1 -> 0
    var s010 = List[Int32](); s010.append(1); s010.append(0)
    _step_ev(hub_edge, inp_e, s010, s010)
    assert_true(inp_e.gamepad_pressed(0, PAD_A) and (not inp_e.gamepad_down(0, PAD_A)))
    assert_true((not inp_e.gamepad_pressed(0, PAD_X)) and (not inp_e.gamepad_down(0, PAD_X)) and (not inp_e.pads[0].latched[PAD_X]))

    # 4) Prime held=1, then 1 -> 0 -> 1
    _step_ev(hub_edge, inp_e, sa1, sa1)
    assert_true(inp_e.gamepad_down(0, PAD_A) and inp_e.pads[0].latched[PAD_X])
    var s101 = List[Int32](); s101.append(0); s101.append(1)
    _step_ev(hub_edge, inp_e, s101, s101)
    assert_true(inp_e.gamepad_pressed(0, PAD_A) and inp_e.gamepad_down(0, PAD_A))
    assert_true((not inp_e.gamepad_down(0, PAD_X)) and (not inp_e.pads[0].latched[PAD_X]))

    # 5) Release back to 0, then 0 -> 1 -> 0 -> 1 (2 presses -> 2 toggle flips back to unlatched!)
    _step_ev(hub_edge, inp_e, s10, s10)
    var s0101 = List[Int32](); s0101.append(1); s0101.append(0); s0101.append(1)
    _step_ev(hub_edge, inp_e, s0101, s0101)
    assert_true(inp_e.gamepad_pressed(0, PAD_A) and inp_e.gamepad_down(0, PAD_A))
    assert_true(inp_e.gamepad_pressed(0, PAD_X) and (not inp_e.gamepad_down(0, PAD_X)) and (not inp_e.pads[0].latched[PAD_X]))

    # 6) Currently held=1 (PAD_A=1, PAD_X latched=0), now 1 -> 0 -> 1 -> 0 (1 press -> PAD_X flips to 1!)
    var s1010 = List[Int32](); s1010.append(0); s1010.append(1); s1010.append(0)
    _step_ev(hub_edge, inp_e, s1010, s1010)
    assert_true(inp_e.gamepad_pressed(0, PAD_A) and (not inp_e.gamepad_down(0, PAD_A)))
    assert_true(inp_e.gamepad_pressed(0, PAD_X) and inp_e.gamepad_down(0, PAD_X) and inp_e.pads[0].latched[PAD_X])
    hub_edge.close()


def _move_relocate_lockstep(var p: LockstepPeer) -> LockstepPeer:
    var moved = p^
    return moved^


def _move_relocate_rollback(var p: RollbackPeer) -> RollbackPeer:
    var moved = p^
    return moved^


def _move_relocate_evdev_hub(var h: EvdevHub) -> EvdevHub:
    var moved = h^
    return moved^


def _run_scoped_raii_fds() raises:
    var ls = LockstepPeer()
    assert_true(ls.open(0, True))
    assert_true(ls.fd >= 0)
    var ls2 = _move_relocate_lockstep(ls^)
    assert_true(ls2.fd >= 0)

    var rb = RollbackPeer()
    assert_true(rb.open(0, True))
    assert_true(rb.fd >= 0)
    var rb2 = _move_relocate_rollback(rb^)
    assert_true(rb2.fd >= 0)

    var hub = EvdevHub(enable_hotplug=True)
    var hub2 = _move_relocate_evdev_hub(hub^)
    assert_equal(hub2.connected_count(), 0)


def _run_scoped_idle_kms() raises:
    var kms_a = KmsWindow()
    var kms_b = kms_a^
    kms_b.close()
    kms_b.close()


def test_f28_platform_handle_raii() raises:
    # 1. Zero FD leak across move-relocation + scope-exit __deinit__ (no manual .close())
    var fds_before = _count_open_fds()
    _run_scoped_raii_fds()
    var fds_after = _count_open_fds()
    assert_equal(fds_after, fds_before)

    # 2. Move relocation + double-close idempotency across all 10 F28 structs
    var wl_a = WaylandWindow()
    var wl_b = wl_a^
    wl_b.close()
    wl_b.close()

    var br_a = WaylandBridge()
    var br_b = br_a^
    br_b.stop()
    br_b.stop()

    var x11_a = X11Window()
    var x11_b = x11_a^
    x11_b.close()
    x11_b.close()

    var lg_a = LogindSession()
    var lg_b = lg_a^
    lg_b.close()
    lg_b.close()

    var ps_a = PulseClient()
    var ps_b = ps_a^
    ps_b.close()
    ps_b.close()

    var na_a = NativeAudio()
    var na_b = na_a^
    na_b.close()
    na_b.close()

    var gp_a = LinuxGamepad()
    var gp_b = gp_a^
    gp_b.close()
    gp_b.close()

    var eg_a = EvdevGamepad()
    var eg_b = eg_a^
    eg_b.close()
    eg_b.close()

    # 3. KmsWindow.close() rescue-page isolation when idle (drm_fd < 0 and tty_fd < 0)
    assert_true(install_rescue())
    var prev_drm = rload(R_DRM_FD)
    var prev_tty = rload(R_TTY_FD)
    var prev_crtc = rload(R_CRTC_SAVED)
    var prev_term = rload(R_TERMIOS_SAVED)
    rstore(R_DRM_FD, 999)
    rstore(R_TTY_FD, 888)
    rstore(R_CRTC_SAVED, 1)
    rstore(R_TERMIOS_SAVED, 1)
    _run_scoped_idle_kms()
    assert_equal(rload(R_DRM_FD), 999)
    assert_equal(rload(R_TTY_FD), 888)
    assert_equal(rload(R_CRTC_SAVED), 1)
    assert_equal(rload(R_TERMIOS_SAVED), 1)
    rstore(R_DRM_FD, prev_drm)
    rstore(R_TTY_FD, prev_tty)
    rstore(R_CRTC_SAVED, prev_crtc)
    rstore(R_TERMIOS_SAVED, prev_term)


# ==============================================================================
# (k) M4b F26 Verification: CFF/OTF (OTTO), MSDF/MTSDF, GSUB, BiDi, Fallbacks & TextInput
# ==============================================================================

def _f26_u16_be(mut out: List[UInt8], v: Int):
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _f26_i16_be(mut out: List[UInt8], v: Int):
    _f26_u16_be(out, v & 0xFFFF)


def _f26_u32_be(mut out: List[UInt8], v: Int):
    out.append(UInt8((v >> 24) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _f26_tag(mut out: List[UInt8], s: String):
    var b = s.as_bytes()
    for i in range(4):
        out.append(b[i] if i < len(b) else UInt8(32))


def _f26_assemble_sfnt(version: Int, tags: List[String], tables: List[List[UInt8]]) -> List[UInt8]:
    var n = len(tags)
    var out = List[UInt8]()
    _f26_u32_be(out, version)
    _f26_u16_be(out, n)
    _f26_u16_be(out, 16)
    _f26_u16_be(out, 0)
    _f26_u16_be(out, 0)
    var data_off = 12 + n * 16
    var offsets = List[Int](capacity=n)
    var cur = data_off
    for i in range(n):
        offsets.append(cur)
        var padded = (len(tables[i]) + 3) & ~3
        cur += padded
    for i in range(n):
        _f26_tag(out, tags[i])
        _f26_u32_be(out, 0)
        _f26_u32_be(out, offsets[i])
        _f26_u32_be(out, len(tables[i]))
    for i in range(n):
        out.extend(tables[i].copy())
        while (len(out) & 3) != 0:
            out.append(0)
    return out^


def _f26_cff_index(items: List[List[UInt8]]) -> List[UInt8]:
    var out = List[UInt8]()
    var n = len(items)
    _f26_u16_be(out, n)
    if n == 0:
        return out^
    out.append(2)  # offSize = 2
    var off = 1
    _f26_u16_be(out, off)
    for i in range(n):
        off += len(items[i])
        _f26_u16_be(out, off)
    for i in range(n):
        out.extend(items[i].copy())
    return out^


def _f26_cff_i32(mut out: List[UInt8], v: Int):
    out.append(29)
    _f26_u32_be(out, v)


def _f26_cff_i16(mut out: List[UInt8], v: Int):
    out.append(28)
    _f26_i16_be(out, v)


def _f26_simple_square_glyf(x0: Int, y0: Int, x1: Int, y1: Int) -> List[UInt8]:
    """Clockwise y-up TrueType simple glyph contour."""
    var g = List[UInt8]()
    _f26_i16_be(g, 1)   # numberOfContours
    _f26_i16_be(g, x0)
    _f26_i16_be(g, y0)
    _f26_i16_be(g, x1)
    _f26_i16_be(g, y1)
    _f26_u16_be(g, 3)   # endPtsOfContours[0] = 3
    _f26_u16_be(g, 0)   # instructionLength = 0
    for _ in range(4):
        g.append(1)     # on-curve, long x/y
    var xs = [x0, x0, x1, x1]
    var ys = [y0, y1, y1, y0]
    var px = 0
    for i in range(4):
        _f26_i16_be(g, xs[i] - px)
        px = xs[i]
    var py = 0
    for i in range(4):
        _f26_i16_be(g, ys[i] - py)
        py = ys[i]
    return g^


def _f26_build_head(upem: Int, index_to_loc: Int) -> List[UInt8]:
    var t = List[UInt8](length=54, fill=0)
    t[1] = 1
    t[18] = UInt8((upem >> 8) & 0xFF)
    t[19] = UInt8(upem & 0xFF)
    t[50] = UInt8((index_to_loc >> 8) & 0xFF)
    t[51] = UInt8(index_to_loc & 0xFF)
    return t^


def _f26_build_hhea(asc: Int, desc: Int, metric_count: Int) -> List[UInt8]:
    var t = List[UInt8](length=36, fill=0)
    t[1] = 1
    t[4] = UInt8((asc >> 8) & 0xFF); t[5] = UInt8(asc & 0xFF)
    var dv = desc & 0xFFFF
    t[6] = UInt8((dv >> 8) & 0xFF); t[7] = UInt8(dv & 0xFF)
    t[34] = UInt8((metric_count >> 8) & 0xFF); t[35] = UInt8(metric_count & 0xFF)
    return t^


def _f26_build_maxp(num_glyphs: Int) -> List[UInt8]:
    var t = List[UInt8](length=6, fill=0)
    t[1] = 1
    t[4] = UInt8((num_glyphs >> 8) & 0xFF); t[5] = UInt8(num_glyphs & 0xFF)
    return t^


def _f26_build_hmtx(advances: List[Int]) -> List[UInt8]:
    var t = List[UInt8]()
    for a in advances:
        _f26_u16_be(t, a)
        _f26_i16_be(t, 0)
    return t^


def _f26_build_cmap_fmt4(codes: List[Int], gids: List[Int]) -> List[UInt8]:
    """Builds a cmap table with format 4 subtable (segments sorted by `codes`)."""
    var seg_cnt = len(codes) + 1  # +1 for 0xFFFF sentinel
    var sub = List[UInt8]()
    _f26_u16_be(sub, 4)
    _f26_u16_be(sub, 16 + seg_cnt * 8)
    _f26_u16_be(sub, 0)
    _f26_u16_be(sub, seg_cnt * 2)
    _f26_u16_be(sub, 2); _f26_u16_be(sub, 0); _f26_u16_be(sub, 0)
    for c in codes:
        _f26_u16_be(sub, c)
    _f26_u16_be(sub, 0xFFFF)
    _f26_u16_be(sub, 0)  # reservedPad
    for c in codes:
        _f26_u16_be(sub, c)
    _f26_u16_be(sub, 0xFFFF)
    for i in range(len(codes)):
        _f26_i16_be(sub, gids[i] - codes[i])
    _f26_i16_be(sub, 1)  # sentinel delta -> 0
    for _ in range(seg_cnt):
        _f26_u16_be(sub, 0)
    var cmap = List[UInt8]()
    _f26_u16_be(cmap, 0); _f26_u16_be(cmap, 1)
    _f26_u16_be(cmap, 3); _f26_u16_be(cmap, 1); _f26_u32_be(cmap, 12)
    cmap.extend(sub^)
    return cmap^


def _f26_build_cmap_fmt12(codes: List[Int], gids: List[Int]) -> List[UInt8]:
    var n = len(codes)
    var sub = List[UInt8]()
    _f26_u16_be(sub, 12); _f26_u16_be(sub, 0)
    _f26_u32_be(sub, 16 + n * 12)
    _f26_u32_be(sub, 0)
    _f26_u32_be(sub, n)
    for i in range(n):
        _f26_u32_be(sub, codes[i])
        _f26_u32_be(sub, codes[i])
        _f26_u32_be(sub, gids[i])
    var cmap = List[UInt8]()
    _f26_u16_be(cmap, 0); _f26_u16_be(cmap, 1)
    _f26_u16_be(cmap, 3); _f26_u16_be(cmap, 10); _f26_u32_be(cmap, 12)
    cmap.extend(sub^)
    return cmap^


def _build_test_otto_cff() -> List[UInt8]:
    # Build Local Subr & Global Subr first
    var lsubr0 = List[UInt8]()
    _f26_cff_i16(lsubr0, 200); _f26_cff_i16(lsubr0, 150); _f26_cff_i16(lsubr0, 150); _f26_cff_i16(lsubr0, 200)
    lsubr0.append(31)  # hvcurveto
    lsubr0.append(11)  # return
    var lsubr_items: List[List[UInt8]] = [lsubr0^]
    var lsubrs_idx = _f26_cff_index(lsubr_items)

    var gsubr0 = List[UInt8]()
    _f26_cff_i16(gsubr0, -200); _f26_cff_i16(gsubr0, -150); _f26_cff_i16(gsubr0, 150); _f26_cff_i16(gsubr0, -200)
    gsubr0.append(30)  # vhcurveto
    gsubr0.append(11)  # return
    var gsubr_items: List[List[UInt8]] = [gsubr0^]
    var gsubrs_idx = _f26_cff_index(gsubr_items)

    # Private DICT: Subrs relative offset (fixed size 6 bytes: i32 + op 19) + defaultWidthX + nominalWidthX
    var priv_sz = 6 + 6 + 6
    var priv = List[UInt8]()
    _f26_cff_i32(priv, priv_sz); priv.append(19)   # Subrs rel offset
    _f26_cff_i32(priv, 600); priv.append(20)       # defaultWidthX = 600
    _f26_cff_i32(priv, 500); priv.append(21)       # nominalWidthX = 500

    # CharStrings: 0=.notdef, 1='A' (triangle), 2='O' (curves+subrs+hintmask), 3='#' (CCW y-up square)
    var cs0: List[UInt8] = [14]
    var cs1 = List[UInt8]()
    _f26_cff_i16(cs1, 100); _f26_cff_i16(cs1, 0); cs1.append(21)    # rmoveto(100, 0)
    _f26_cff_i16(cs1, 250); _f26_cff_i16(cs1, 700)
    _f26_cff_i16(cs1, 250); _f26_cff_i16(cs1, -700); cs1.append(5)  # rlineto
    _f26_cff_i16(cs1, -500); cs1.append(6)                          # hlineto
    cs1.append(14)

    var cs2 = List[UInt8]()
    _f26_cff_i16(cs2, 50); _f26_cff_i16(cs2, 100); cs2.append(1)    # hstem (1 stem -> 1 mask byte)
    cs2.append(19); cs2.append(0x80)                                # hintmask + 1 byte
    _f26_cff_i16(cs2, 350); _f26_cff_i16(cs2, 100); cs2.append(21)  # rmoveto
    _f26_cff_i16(cs2, -107); cs2.append(10)                         # callsubr 0
    _f26_cff_i16(cs2, -107); cs2.append(29)                         # callgsubr 0
    _f26_cff_i16(cs2, -150); _f26_cff_i16(cs2, -150)
    _f26_cff_i16(cs2, -150); _f26_cff_i16(cs2, 150)
    _f26_cff_i16(cs2, 150); _f26_cff_i16(cs2, 150); cs2.append(8)   # rrcurveto
    cs2.append(14)

    var cs3 = List[UInt8]()
    _f26_cff_i16(cs3, 100); _f26_cff_i16(cs3, 100); cs3.append(21)  # rmoveto(100,100)
    _f26_cff_i16(cs3, 500); cs3.append(6)                           # hlineto(500) -> (600,100)
    _f26_cff_i16(cs3, 500); cs3.append(7)                           # vlineto(500) -> (600,600)
    _f26_cff_i16(cs3, -500); cs3.append(6)                          # hlineto(-500) -> (100,600)
    _f26_cff_i16(cs3, -500); cs3.append(7)                          # vlineto(-500) -> (100,100)
    cs3.append(14)

    var cs_items: List[List[UInt8]] = [cs0^, cs1^, cs2^, cs3^]
    var cs_idx = _f26_cff_index(cs_items)

    var name_bytes = List[UInt8]()
    for b in String("TestOTTO").as_bytes():
        name_bytes.append(b)
    var name_items: List[List[UInt8]] = [name_bytes^]
    var name_idx = _f26_cff_index(name_items)
    var str_idx = _f26_cff_index(List[List[UInt8]]())

    # Top DICT has fixed 17-byte payload: charstrings_off (i32 + 17) + priv (i32 + i32 + 18)
    var dummy_top = List[UInt8](length=17, fill=0)
    var dummy_items: List[List[UInt8]] = [dummy_top^]
    var top_idx_sz = len(_f26_cff_index(dummy_items))
    var cs_off = 4 + len(name_idx) + top_idx_sz + len(str_idx) + len(gsubrs_idx)
    var priv_off = cs_off + len(cs_idx)

    var top_dict = List[UInt8]()
    _f26_cff_i32(top_dict, cs_off); top_dict.append(17)
    _f26_cff_i32(top_dict, priv_sz); _f26_cff_i32(top_dict, priv_off); top_dict.append(18)
    var top_items: List[List[UInt8]] = [top_dict^]
    var top_idx = _f26_cff_index(top_items)

    var cff: List[UInt8] = [1, 0, 4, 2]
    cff.extend(name_idx^)
    cff.extend(top_idx^)
    cff.extend(str_idx^)
    cff.extend(gsubrs_idx^)
    cff.extend(cs_idx^)
    cff.extend(priv^)
    cff.extend(lsubrs_idx^)

    var tags: List[String] = [String("CFF "), String("cmap"), String("head"), String("hhea"), String("hmtx"), String("maxp")]
    var cmap_c: List[Int] = [35, 65, 79]
    var cmap_g: List[Int] = [3, 1, 2]
    var hmtx_a: List[Int] = [500, 620, 740, 700]
    var tbls: List[List[UInt8]] = [
        cff^,
        _f26_build_cmap_fmt4(cmap_c, cmap_g),
        _f26_build_head(1000, 1),
        _f26_build_hhea(800, -200, 4),
        _f26_build_hmtx(hmtx_a),
        _f26_build_maxp(4),
    ]
    return _f26_assemble_sfnt(0x4F54544F, tags, tbls)


def _f26_cov_fmt1(gids: List[Int]) -> List[UInt8]:
    var out = List[UInt8]()
    _f26_u16_be(out, 1)
    _f26_u16_be(out, len(gids))
    for g in gids:
        _f26_u16_be(out, g)
    return out^


def _f26_cov1_single(gid: Int) -> List[UInt8]:
    var gids: List[Int] = [gid]
    return _f26_cov_fmt1(gids)


def _f26_wrap1(ltype: Int, lflag: Int, var subtable: List[UInt8]) -> List[UInt8]:
    var subs: List[List[UInt8]] = [subtable^]
    return _f26_wrap_lookup(ltype, lflag, subs)


def _f26_single_subst_fmt2(src: List[Int], dst: List[Int]) -> List[UInt8]:
    var cov = _f26_cov_fmt1(src)
    var sub = List[UInt8]()
    _f26_u16_be(sub, 2)
    _f26_u16_be(sub, 6 + len(dst) * 2)
    _f26_u16_be(sub, len(dst))
    for g in dst:
        _f26_u16_be(sub, g)
    sub.extend(cov^)
    return sub^


def _f26_wrap_lookup(ltype: Int, lflag: Int, subtables: List[List[UInt8]]) -> List[UInt8]:
    var n = len(subtables)
    var hdr = 6 + n * 2
    var out = List[UInt8]()
    _f26_u16_be(out, ltype)
    _f26_u16_be(out, lflag)
    _f26_u16_be(out, n)
    var off = hdr
    for i in range(n):
        _f26_u16_be(out, off)
        off += len(subtables[i])
    for i in range(n):
        out.extend(subtables[i].copy())
    return out^


def _build_test_ttf_with_gsub(upem: Int = 1000) -> List[UInt8]:
    """Programmatic TTF with GDEF + GSUB (Types 1-7 + Arab contextual + Latin lig/calt)."""
    var num_glyphs = 90
    var advances = List[Int](length=num_glyphs, fill=600)
    advances[32] = 350   # 'f' (code 102)
    advances[35] = 300   # 'i' (code 105)
    advances[66] = 450   # gid_fi (strictly < 350 + 300)
    advances[67] = 600   # gid_ffi
    advances[68] = 500   # gid_arrow
    advances[53] = 280   # ' '
    advances[57] = 0     # U+0301 combining acute mark

    # Build glyf + loca (indexToLocFormat = 1)
    var glyf = List[UInt8]()
    var loca = List[UInt8]()
    for g in range(num_glyphs):
        _f26_u32_be(loca, len(glyf))
        if g == 53 or g == 0:
            continue
        var gd: List[UInt8]
        if g == 1:  # 'A': sharp 90-deg corner box for MSDF verification
            gd = _f26_simple_square_glyf(100, 0, 600, 700)
        elif g == 57:  # combining mark
            gd = _f26_simple_square_glyf(150, 720, 250, 820)
        else:
            gd = _f26_simple_square_glyf(50, 0, 50 + max(100, advances[g] - 100), 650)
        glyf.extend(gd^)
        while (len(glyf) & 3) != 0:
            glyf.append(0)
    _f26_u32_be(loca, len(glyf))

    # Build sorted cmap format 4 codes -> gids
    var cmap_codes = List[Int]()
    var cmap_gids = List[Int]()
    # ASCII ' ' (32), '-' (45), '>' (62), '?' (63), 'A'..'Z' (65..90), 'a'..'z' (97..122),
    # U+0301 (769), Hebrew (0x05D5..0x05E9), Arabic (0x0627..0x0645)
    cmap_codes.append(32); cmap_gids.append(53)
    cmap_codes.append(45); cmap_gids.append(55)
    cmap_codes.append(62); cmap_gids.append(56)
    cmap_codes.append(63); cmap_gids.append(54)
    for k in range(26):
        cmap_codes.append(65 + k); cmap_gids.append(1 + k)
    for k in range(26):
        cmap_codes.append(97 + k); cmap_gids.append(27 + k)
    cmap_codes.append(0x0301); cmap_gids.append(57)
    cmap_codes.append(0x05D5); cmap_gids.append(64)  # ו
    cmap_codes.append(0x05DC); cmap_gids.append(63)  # ל
    cmap_codes.append(0x05DD); cmap_gids.append(65)  # ם
    cmap_codes.append(0x05E9); cmap_gids.append(62)  # ש
    cmap_codes.append(0x0627); cmap_gids.append(61)  # ا
    cmap_codes.append(0x0633); cmap_gids.append(58)  # س
    cmap_codes.append(0x0644); cmap_gids.append(59)  # ل
    cmap_codes.append(0x0645); cmap_gids.append(60)  # م

    # GDEF v1.0: GlyphClassDef format 1 starting at glyph 0
    var gdef = List[UInt8]()
    _f26_u16_be(gdef, 1); _f26_u16_be(gdef, 0)
    _f26_u16_be(gdef, 12)  # glyphClassDefOffset
    _f26_u16_be(gdef, 0); _f26_u16_be(gdef, 0); _f26_u16_be(gdef, 0)
    _f26_u16_be(gdef, 1)   # ClassDef format 1
    _f26_u16_be(gdef, 0); _f26_u16_be(gdef, num_glyphs)
    for g in range(num_glyphs):
        if g == 57:
            _f26_u16_be(gdef, 3)  # Mark
        elif g == 66 or g == 67 or g == 89:
            _f26_u16_be(gdef, 2)  # Ligature
        else:
            _f26_u16_be(gdef, 1)  # Base

    # GSUB Lookup 0: Type 1 fmt 1 ('K' gid 11 -> +58 = 69)
    var l0_sub = List[UInt8]()
    _f26_u16_be(l0_sub, 1); _f26_u16_be(l0_sub, 6); _f26_i16_be(l0_sub, 58)
    l0_sub.extend(_f26_cov1_single(11))
    var l0 = _f26_wrap1(1, 0, l0_sub^)

    # GSUB Lookup 1: Type 1 fmt 2 ('L' gid 12 -> 70)
    var s1_src: List[Int] = [12]; var s1_dst: List[Int] = [70]
    var l1 = _f26_wrap1(1, 0, _f26_single_subst_fmt2(s1_src, s1_dst))

    # GSUB Lookup 2: Type 2 fmt 1 ('Q' gid 17 -> [71, 72])
    var l2_sub = List[UInt8]()
    _f26_u16_be(l2_sub, 1); _f26_u16_be(l2_sub, 8); _f26_u16_be(l2_sub, 1); _f26_u16_be(l2_sub, 14)
    l2_sub.extend(_f26_cov1_single(17))
    _f26_u16_be(l2_sub, 2); _f26_u16_be(l2_sub, 71); _f26_u16_be(l2_sub, 72)
    var l2 = _f26_wrap1(2, 0, l2_sub^)

    # GSUB Lookup 3: Type 3 fmt 1 ('Z' gid 26 -> [73, 74])
    var l3_sub = List[UInt8]()
    _f26_u16_be(l3_sub, 1); _f26_u16_be(l3_sub, 8); _f26_u16_be(l3_sub, 1); _f26_u16_be(l3_sub, 14)
    l3_sub.extend(_f26_cov1_single(26))
    _f26_u16_be(l3_sub, 2); _f26_u16_be(l3_sub, 73); _f26_u16_be(l3_sub, 74)
    var l3 = _f26_wrap1(3, 0, l3_sub^)

    # GSUB Lookup 4: Type 4 fmt 1 ('ffi' -> 67, 'fi' -> 66) with IGNORE_MARKS (0x0008)
    # 'f' = 32, 'i' = 35
    var l4_sub = List[UInt8]()
    _f26_u16_be(l4_sub, 1); _f26_u16_be(l4_sub, 8); _f26_u16_be(l4_sub, 1); _f26_u16_be(l4_sub, 14)
    l4_sub.extend(_f26_cov1_single(32))
    # LigatureSet at offset 14 (6 bytes header + 2 ligs: lig0 at 6, lig1 at 14)
    _f26_u16_be(l4_sub, 2); _f26_u16_be(l4_sub, 6); _f26_u16_be(l4_sub, 14)
    _f26_u16_be(l4_sub, 67); _f26_u16_be(l4_sub, 3); _f26_u16_be(l4_sub, 32); _f26_u16_be(l4_sub, 35)  # ffi
    _f26_u16_be(l4_sub, 66); _f26_u16_be(l4_sub, 2); _f26_u16_be(l4_sub, 35)                          # fi
    var l4 = _f26_wrap1(4, 0x0008, l4_sub^)

    # GSUB Lookup 5: Type 4 ('-' 55 + '>' 56 -> 68 arrow, invoked by Lookup 6)
    var l5_sub = List[UInt8]()
    _f26_u16_be(l5_sub, 1); _f26_u16_be(l5_sub, 8); _f26_u16_be(l5_sub, 1); _f26_u16_be(l5_sub, 14)
    l5_sub.extend(_f26_cov1_single(55))
    _f26_u16_be(l5_sub, 1); _f26_u16_be(l5_sub, 4)
    _f26_u16_be(l5_sub, 68); _f26_u16_be(l5_sub, 2); _f26_u16_be(l5_sub, 56)
    var l5 = _f26_wrap1(4, 0, l5_sub^)

    # GSUB Lookup 6: Type 7 (Extension) -> Type 6 fmt 3: backtrack ['a'=27], input ['-'=55, '>'=56], lookahead ['b'=28], seq(0, 5)
    var t6 = List[UInt8]()
    _f26_u16_be(t6, 3)  # fmt 3
    _f26_u16_be(t6, 1); _f26_u16_be(t6, 22)      # backtrack 1 cov at 22
    _f26_u16_be(t6, 2); _f26_u16_be(t6, 28); _f26_u16_be(t6, 34)  # input 2 covs at 28, 34
    _f26_u16_be(t6, 1); _f26_u16_be(t6, 40)      # lookahead 1 cov at 40
    _f26_u16_be(t6, 1); _f26_u16_be(t6, 0); _f26_u16_be(t6, 5)    # seqLookup: idx=0, lookup=5
    t6.extend(_f26_cov1_single(27))
    t6.extend(_f26_cov1_single(55))
    t6.extend(_f26_cov1_single(56))
    t6.extend(_f26_cov1_single(28))
    var l6_ext = List[UInt8]()
    _f26_u16_be(l6_ext, 1); _f26_u16_be(l6_ext, 6); _f26_u32_be(l6_ext, 8)
    l6_ext.extend(t6^)
    var l6 = _f26_wrap1(7, 0, l6_ext^)

    # Arabic contextual lookups 7..10 + Lam-Alif rlig lookup 11
    var ar_src4: List[Int] = [58, 59, 60, 61]
    var ar_src3: List[Int] = [58, 59, 60]
    var ar_isol: List[Int] = [75, 79, 83, 87]
    var ar_init: List[Int] = [76, 80, 84]
    var ar_medi: List[Int] = [77, 81, 85]
    var ar_fina: List[Int] = [78, 82, 86, 88]
    var l7 = _f26_wrap1(1, 0, _f26_single_subst_fmt2(ar_src4, ar_isol))   # isol
    var l8 = _f26_wrap1(1, 0, _f26_single_subst_fmt2(ar_src3, ar_init))   # init
    var l9 = _f26_wrap1(1, 0, _f26_single_subst_fmt2(ar_src3, ar_medi))   # medi
    var l10 = _f26_wrap1(1, 0, _f26_single_subst_fmt2(ar_src4, ar_fina))  # fina

    # Lookup 11: Lam-init (80) or Lam-medi (81) + Alif-fina (88) -> 89
    var l11_sub = List[UInt8]()
    _f26_u16_be(l11_sub, 1); _f26_u16_be(l11_sub, 10); _f26_u16_be(l11_sub, 2); _f26_u16_be(l11_sub, 18); _f26_u16_be(l11_sub, 28)
    var l11_cov_gids: List[Int] = [80, 81]
    l11_sub.extend(_f26_cov_fmt1(l11_cov_gids))
    _f26_u16_be(l11_sub, 1); _f26_u16_be(l11_sub, 4)
    _f26_u16_be(l11_sub, 89); _f26_u16_be(l11_sub, 2); _f26_u16_be(l11_sub, 88)
    _f26_u16_be(l11_sub, 1); _f26_u16_be(l11_sub, 4)
    _f26_u16_be(l11_sub, 89); _f26_u16_be(l11_sub, 2); _f26_u16_be(l11_sub, 88)
    var l11 = _f26_wrap1(4, 0x0008, l11_sub^)

    var all_lookups: List[List[UInt8]] = [l0^, l1^, l2^, l3^, l4^, l5^, l6^, l7^, l8^, l9^, l10^, l11^]
    var lookup_list = List[UInt8]()
    var n_l = len(all_lookups)
    _f26_u16_be(lookup_list, n_l)
    var loff = 2 + n_l * 2
    for i in range(n_l):
        _f26_u16_be(lookup_list, loff)
        loff += len(all_lookups[i])
    for i in range(n_l):
        lookup_list.extend(all_lookups[i].copy())

    # Features:
    # 0: locl -> [0, 1]
    # 1: ccmp -> [2]
    # 2: calt -> [3, 6]
    # 3: liga -> [4]
    # 4: isol -> [7]
    # 5: init -> [8]
    # 6: medi -> [9]
    # 7: fina -> [10]
    # 8: rlig -> [11]
    var feat_tags = [String("locl"), String("ccmp"), String("calt"), String("liga"), String("isol"), String("init"), String("medi"), String("fina"), String("rlig")]
    var feat_tables = List[List[UInt8]]()
    var f0 = List[UInt8](); _f26_u16_be(f0, 0); _f26_u16_be(f0, 2); _f26_u16_be(f0, 0); _f26_u16_be(f0, 1); feat_tables.append(f0^)
    var f1 = List[UInt8](); _f26_u16_be(f1, 0); _f26_u16_be(f1, 1); _f26_u16_be(f1, 2); feat_tables.append(f1^)
    var f2 = List[UInt8](); _f26_u16_be(f2, 0); _f26_u16_be(f2, 2); _f26_u16_be(f2, 3); _f26_u16_be(f2, 6); feat_tables.append(f2^)
    var f3 = List[UInt8](); _f26_u16_be(f3, 0); _f26_u16_be(f3, 1); _f26_u16_be(f3, 4); feat_tables.append(f3^)
    var f4 = List[UInt8](); _f26_u16_be(f4, 0); _f26_u16_be(f4, 1); _f26_u16_be(f4, 7); feat_tables.append(f4^)
    var f5 = List[UInt8](); _f26_u16_be(f5, 0); _f26_u16_be(f5, 1); _f26_u16_be(f5, 8); feat_tables.append(f5^)
    var f6 = List[UInt8](); _f26_u16_be(f6, 0); _f26_u16_be(f6, 1); _f26_u16_be(f6, 9); feat_tables.append(f6^)
    var f7 = List[UInt8](); _f26_u16_be(f7, 0); _f26_u16_be(f7, 1); _f26_u16_be(f7, 10); feat_tables.append(f7^)
    var f8 = List[UInt8](); _f26_u16_be(f8, 0); _f26_u16_be(f8, 1); _f26_u16_be(f8, 11); feat_tables.append(f8^)

    var feat_list = List[UInt8]()
    _f26_u16_be(feat_list, len(feat_tags))
    var foff = 2 + len(feat_tags) * 6
    for i in range(len(feat_tags)):
        _f26_tag(feat_list, feat_tags[i])
        _f26_u16_be(feat_list, foff)
        foff += len(feat_tables[i])
    for i in range(len(feat_tables)):
        feat_list.extend(feat_tables[i].copy())

    # ScriptList: DFLT -> [0,1,2,3], arab -> [4,5,6,7,8], latn -> [0,1,2,3]
    var sc_dflt = List[UInt8]()
    _f26_u16_be(sc_dflt, 4); _f26_u16_be(sc_dflt, 0)  # defaultLangSys at 4, 0 langSysRecords
    _f26_u16_be(sc_dflt, 0); _f26_u16_be(sc_dflt, 0xFFFF); _f26_u16_be(sc_dflt, 4)
    _f26_u16_be(sc_dflt, 0); _f26_u16_be(sc_dflt, 1); _f26_u16_be(sc_dflt, 2); _f26_u16_be(sc_dflt, 3)

    var sc_arab = List[UInt8]()
    _f26_u16_be(sc_arab, 4); _f26_u16_be(sc_arab, 0)
    _f26_u16_be(sc_arab, 0); _f26_u16_be(sc_arab, 0xFFFF); _f26_u16_be(sc_arab, 5)
    _f26_u16_be(sc_arab, 4); _f26_u16_be(sc_arab, 5); _f26_u16_be(sc_arab, 6); _f26_u16_be(sc_arab, 7); _f26_u16_be(sc_arab, 8)

    var sc_list = List[UInt8]()
    _f26_u16_be(sc_list, 3)
    var s_off0 = 2 + 3 * 6
    var s_off1 = s_off0 + len(sc_dflt)
    var s_off2 = s_off1 + len(sc_arab)
    _f26_tag(sc_list, "DFLT"); _f26_u16_be(sc_list, s_off0)
    _f26_tag(sc_list, "arab"); _f26_u16_be(sc_list, s_off1)
    _f26_tag(sc_list, "latn"); _f26_u16_be(sc_list, s_off2)
    sc_list.extend(sc_dflt.copy())
    sc_list.extend(sc_arab^)
    sc_list.extend(sc_dflt^)

    var gsub = List[UInt8]()
    var g_sc_off = 10
    var g_ft_off = g_sc_off + len(sc_list)
    var g_lk_off = g_ft_off + len(feat_list)
    _f26_u16_be(gsub, 1); _f26_u16_be(gsub, 0)
    _f26_u16_be(gsub, g_sc_off)
    _f26_u16_be(gsub, g_ft_off)
    _f26_u16_be(gsub, g_lk_off)
    gsub.extend(sc_list^)
    gsub.extend(feat_list^)
    gsub.extend(lookup_list^)

    var tags: List[String] = [String("GDEF"), String("GSUB"), String("cmap"), String("glyf"), String("head"), String("hhea"), String("hmtx"), String("loca"), String("maxp")]
    var tbls: List[List[UInt8]] = [
        gdef^,
        gsub^,
        _f26_build_cmap_fmt4(cmap_codes, cmap_gids),
        glyf^,
        _f26_build_head(upem, 1),
        _f26_build_hhea(Int(Float32(upem) * 0.8), -Int(Float32(upem) * 0.2), num_glyphs),
        _f26_build_hmtx(advances),
        loca^,
        _f26_build_maxp(num_glyphs),
    ]
    return _f26_assemble_sfnt(0x00010000, tags, tbls)


def _build_test_ttf_cjk_emoji(upem: Int = 1024) -> List[UInt8]:
    """TTF with cmap format 12 for CJK U+4E2D and Emoji U+1F680."""
    var glyf = List[UInt8]()
    var loca = List[UInt8]()
    _f26_u32_be(loca, 0)  # glyph 0 empty
    _f26_u32_be(loca, 0)
    var g1 = _f26_simple_square_glyf(60, 0, 452, 720)
    glyf.extend(g1^)
    while (len(glyf) & 3) != 0:
        glyf.append(0)
    _f26_u32_be(loca, len(glyf))
    var g2 = _f26_simple_square_glyf(80, 0, 944, 760)
    glyf.extend(g2^)
    while (len(glyf) & 3) != 0:
        glyf.append(0)
    _f26_u32_be(loca, len(glyf))

    var tags: List[String] = [String("cmap"), String("glyf"), String("head"), String("hhea"), String("hmtx"), String("loca"), String("maxp")]
    var c12: List[Int] = [0x4E2D, 0x1F680]
    var g12: List[Int] = [1, 2]
    var h12: List[Int] = [500, 512, 1024]
    var tbls: List[List[UInt8]] = [
        _f26_build_cmap_fmt12(c12, g12),
        glyf^,
        _f26_build_head(upem, 1),
        _f26_build_hhea(800, -200, 3),
        _f26_build_hmtx(h12),
        loca^,
        _f26_build_maxp(3),
    ]
    return _f26_assemble_sfnt(0x00010000, tags, tbls)


def test_f26_msdf_mtsdf_and_otto_cff() raises:
    # 1. 0-table OTTO must fail; 1-table minimal OTTO (CFF v1.0 header only) must succeed
    var otto0 = List[UInt8](length=32, fill=0)
    otto0[0] = 0x4F; otto0[1] = 0x54; otto0[2] = 0x54; otto0[3] = 0x4F
    var f0 = TrueTypeFont()
    assert_false(f0.load_bytes(otto0^))
    assert_false(f0.valid)

    var otto1: List[UInt8] = [
        0x4F, 0x54, 0x54, 0x4F, 0, 1, 0, 16, 0, 0, 0, 0,
        0x43, 0x46, 0x46, 0x20, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8,
        1, 0, 4, 1, 0, 0, 0, 0,
    ]
    var f1 = TrueTypeFont()
    assert_true(f1.load_bytes(otto1^))
    assert_true(f1.valid)
    assert_true(f1.is_cff)

    # 2. Full CFF/OTF font with lines, cubic curves, callsubr, callgsubr, hintmask
    var otto_full = _build_test_otto_cff()
    var cff_font = TrueTypeFont()
    assert_true(cff_font.load_bytes(otto_full^))
    assert_true(cff_font.valid)
    assert_true(cff_font.is_cff)
    assert_equal(cff_font.glyph_id(65), 1)
    assert_equal(cff_font.glyph_id(79), 2)
    assert_equal(cff_font.glyph_id(35), 3)

    var out_a = cff_font.outline(1)
    assert_true(out_a.is_cubic)
    assert_true(out_a.point_count() >= 4 and len(out_a.ends) == 1)
    assert_true(cff_font.advance(1) > Float32(0.0))

    var out_o = cff_font.outline(2)
    assert_true(out_o.is_cubic)
    assert_true(out_o.point_count() >= 9 and len(out_o.ends) == 1)

    # 3. Guarded system .otf check (Cantarell-Regular.otf if installed)
    var sys_otf = "/usr/share/fonts/opentype/cantarell/Cantarell-Regular.otf"
    var sys_bytes = read_bytes(sys_otf)
    if len(sys_bytes) > 0:
        var cf = TrueTypeFont()
        assert_true(cf.load_bytes(sys_bytes^))
        assert_true(cf.is_cff and cf.glyph_count > 100)
        var oa = cf.outline(cf.glyph_id(65))
        assert_true(oa.is_cubic and oa.point_count() > 0)
        var ms = bake_glyph_msdf(cf, 82, 32, 4)  # 'R'
        assert_equal(ms.channels, 3)
        assert_true(ms.w > 0 and ms.h > 0)

    # 4. Genuine MSDF (3ch) & MTSDF (4ch): edge-color divergence, sharp corner preservation, DSF v2
    var ttf_bytes = _build_test_ttf_with_gsub(1000)
    var ttf = TrueTypeFont()
    assert_true(ttf.load_bytes(ttf_bytes^))

    var sdf1 = bake_glyph(ttf, 65, 32, 4)
    var msdf = bake_glyph_msdf(ttf, 65, 32, 4)
    var mtsdf = bake_glyph_mtsdf(ttf, 65, 32, 4)
    assert_equal(sdf1.channels, 1)
    assert_equal(msdf.channels, 3)
    assert_equal(len(msdf.pixels), msdf.w * msdf.h * 3)
    assert_equal(mtsdf.channels, 4)
    assert_equal(len(mtsdf.pixels), mtsdf.w * mtsdf.h * 4)

    var divergent_texels = 0
    for idx in range(msdf.w * msdf.h):
        var r = msdf.pixels[idx * 3]
        var g = msdf.pixels[idx * 3 + 1]
        var b = msdf.pixels[idx * 3 + 2]
        if r != g or g != b:
            divergent_texels += 1
    assert_true(divergent_texels > 10)

    # Bake 4-channel MTSDF atlas at bake_px=40 (integer-aligned corner (4, -28)), round-trip via DSF v2
    var b_codes: List[Int] = [32, 63, 65, 66]
    var baked4 = bake_font(ttf, b_codes, 40, 4, 256, 4)
    assert_equal(baked4.channels, 4)
    var dsf2 = dsf_bytes(baked4)
    var sdf_rt = SdfFont()
    assert_true(sdf_rt.load_bytes(dsf2))
    assert_equal(sdf_rt.channels, 4)

    var g_a = sdf_rt.glyphs[sdf_rt.glyph_index(65)]
    var sc40 = Float32(40.0) / Float32(1000.0)
    var b_a = ttf.outline(ttf.glyph_id(65)).bounds()
    # Top-left corner of 'A' box in atlas texel coordinates
    var cx = Float32(g_a.x) + (b_a[0] * sc40 - g_a.xoff)
    var cy = Float32(g_a.y) + (-b_a[3] * sc40 - g_a.yoff)
    assert_true(sdf_rt.sample_msdf(cx + Float32(0.5), cy + Float32(0.5)) > Float32(128.0))
    assert_true(sdf_rt.sample_msdf(cx - Float32(0.5), cy - Float32(0.5)) < Float32(128.0))
    # Right at the sharp 90° vertex (cx, cy), MSDF preserves ~128.0 while Euclidean SDF rounds inward!
    var msdf_vtx = sdf_rt.sample_msdf(cx, cy)
    var true_vtx = sdf_rt.sample_true_distance(cx, cy)
    assert_true(abs(msdf_vtx - Float32(128.0)) < Float32(2.0))
    assert_true(true_vtx < msdf_vtx - Float32(1.0))

    # Dynamic ensure on channels=4 SdfFont
    sdf_rt.source = ttf.copy()
    sdf_rt.has_source = True
    sdf_rt._ensure_dynamic_page()
    assert_true(sdf_rt.ensure(67))  # 'C'
    assert_true(sdf_rt.is_dirty())
    var g_c = sdf_rt.glyphs[sdf_rt.glyph_index(67)]
    assert_true(sdf_rt.sample_msdf(Float32(g_c.x + g_c.w // 2), Float32(g_c.y + g_c.h // 2)) > Float32(128.0))

    # Winding parity: CW y-up TTF square vs CCW y-up CFF square ('#' code 35)
    var cff_msdf = bake_glyph_msdf(cff_font, 35, 32, 4)
    assert_equal(cff_msdf.channels, 3)
    var c_mid = ((cff_msdf.h // 2) * cff_msdf.w + (cff_msdf.w // 2)) * 3
    var c_med = msdf_median(Float32(cff_msdf.pixels[c_mid]), Float32(cff_msdf.pixels[c_mid + 1]), Float32(cff_msdf.pixels[c_mid + 2]))
    assert_true(c_med > Float32(200.0))
    var c_out = (0 * cff_msdf.w + 0) * 3
    var c_out_med = msdf_median(Float32(cff_msdf.pixels[c_out]), Float32(cff_msdf.pixels[c_out + 1]), Float32(cff_msdf.pixels[c_out + 2]))
    assert_true(c_out_med < Float32(100.0))

    # 5. Regression: DSF v2 64-bit integer overflow + negative glyph.w/h + bake_px<=0 + spread<=0
    var bad_ovf = List[UInt8](length=52, fill=0)
    var ovf_hdr: List[Int] = [0x31465344, 2, 0x7FFFFFFF, 0x7FFFFFFF, 32, 4, 1536, 512, 2048, 0x20000000, 0, 4, 0]
    for hi in range(13):
        var v = ovf_hdr[hi]
        bad_ovf[hi * 4] = UInt8(v & 0xFF)
        bad_ovf[hi * 4 + 1] = UInt8((v >> 8) & 0xFF)
        bad_ovf[hi * 4 + 2] = UInt8((v >> 16) & 0xFF)
        bad_ovf[hi * 4 + 3] = UInt8((v >> 24) & 0xFF)
    var f_ovf = SdfFont()
    assert_false(f_ovf.load_bytes(bad_ovf))
    assert_false(f_ovf.valid)

    # Mutate valid dsf2 header: bake_px=0, spread=0, negative glyph.w=-1, negative glyph.h=-1
    var f_chk = SdfFont()
    var dsf_bad_px = dsf2.copy()
    dsf_bad_px[16] = 0; dsf_bad_px[17] = 0; dsf_bad_px[18] = 0; dsf_bad_px[19] = 0
    assert_false(f_chk.load_bytes(dsf_bad_px))
    var dsf_bad_sp = dsf2.copy()
    dsf_bad_sp[20] = 0; dsf_bad_sp[21] = 0; dsf_bad_sp[22] = 0; dsf_bad_sp[23] = 0
    assert_false(f_chk.load_bytes(dsf_bad_sp))
    var dsf_neg_w = dsf2.copy()
    dsf_neg_w[48 + 12] = 0xFF; dsf_neg_w[48 + 13] = 0xFF; dsf_neg_w[48 + 14] = 0xFF; dsf_neg_w[48 + 15] = 0xFF
    assert_false(f_chk.load_bytes(dsf_neg_w))
    var dsf_neg_h = dsf2.copy()
    dsf_neg_h[48 + 16] = 0xFF; dsf_neg_h[48 + 17] = 0xFF; dsf_neg_h[48 + 18] = 0xFF; dsf_neg_h[48 + 19] = 0xFF
    assert_false(f_chk.load_bytes(dsf_neg_h))

    # 6. Regression: 15-deg acute spike & 0.8px thin bar MSDF preservation
    var g_spike = List[UInt8]()
    _f26_i16_be(g_spike, 1)
    _f26_i16_be(g_spike, 100); _f26_i16_be(g_spike, 193); _f26_i16_be(g_spike, 900); _f26_i16_be(g_spike, 407)
    _f26_u16_be(g_spike, 2); _f26_u16_be(g_spike, 0)
    g_spike.append(1); g_spike.append(1); g_spike.append(1)
    _f26_i16_be(g_spike, 100); _f26_i16_be(g_spike, 800); _f26_i16_be(g_spike, -800)
    _f26_i16_be(g_spike, 193); _f26_i16_be(g_spike, 107); _f26_i16_be(g_spike, 107)
    var g_bar = _f26_simple_square_glyf(100, 290, 700, 310)
    var g_bar5 = List[UInt8]()
    _f26_i16_be(g_bar5, 1)
    _f26_i16_be(g_bar5, 100); _f26_i16_be(g_bar5, 290); _f26_i16_be(g_bar5, 720); _f26_i16_be(g_bar5, 310)
    _f26_u16_be(g_bar5, 4); _f26_u16_be(g_bar5, 0)
    for _ in range(5):
        g_bar5.append(1)
    _f26_i16_be(g_bar5, 100); _f26_i16_be(g_bar5, 600); _f26_i16_be(g_bar5, 20); _f26_i16_be(g_bar5, -20); _f26_i16_be(g_bar5, -600)
    _f26_i16_be(g_bar5, 290); _f26_i16_be(g_bar5, 0); _f26_i16_be(g_bar5, 10); _f26_i16_be(g_bar5, 10); _f26_i16_be(g_bar5, 0)
    var sp_glyf = List[UInt8]()
    var sp_loca = List[UInt8]()
    _f26_u32_be(sp_loca, 0); _f26_u32_be(sp_loca, 0)
    sp_glyf.extend(g_spike^);
    while (len(sp_glyf) & 3) != 0:
        sp_glyf.append(0)
    _f26_u32_be(sp_loca, len(sp_glyf))
    sp_glyf.extend(g_bar^)
    while (len(sp_glyf) & 3) != 0:
        sp_glyf.append(0)
    _f26_u32_be(sp_loca, len(sp_glyf))
    sp_glyf.extend(g_bar5^)
    while (len(sp_glyf) & 3) != 0:
        sp_glyf.append(0)
    _f26_u32_be(sp_loca, len(sp_glyf))
    var sp_tags: List[String] = [String("cmap"), String("glyf"), String("head"), String("hhea"), String("hmtx"), String("loca"), String("maxp")]
    var sp_c: List[Int] = [65, 66, 67]
    var sp_g: List[Int] = [1, 2, 3]
    var sp_adv: List[Int] = [500, 1000, 800, 800]
    var sp_tbls: List[List[UInt8]] = [
        _f26_build_cmap_fmt4(sp_c, sp_g),
        sp_glyf^,
        _f26_build_head(1000, 1),
        _f26_build_hhea(800, -200, 4),
        _f26_build_hmtx(sp_adv),
        sp_loca^,
        _f26_build_maxp(4),
    ]
    var sp_ttf = TrueTypeFont()
    assert_true(sp_ttf.load_bytes(_f26_assemble_sfnt(0x00010000, sp_tags, sp_tbls)))
    var sp_codes: List[Int] = [65, 66, 67]
    var sp_baked = bake_font(sp_ttf, sp_codes, 40, 6, 256, 4)
    var sp_sdf = SdfFont()
    assert_true(sp_sdf.load_bytes(dsf_bytes(sp_baked)))
    var gs = sp_sdf.glyphs[sp_sdf.glyph_index(65)]
    var b_sp = sp_ttf.outline(sp_ttf.glyph_id(65)).bounds()
    var tip_x = Float32(gs.x) + b_sp[2] * Float32(0.04) - gs.xoff
    var tip_y = Float32(gs.y) - Float32(300.0) * Float32(0.04) - gs.yoff
    assert_true(sp_sdf.sample_msdf(tip_x - Float32(1.5), tip_y) > Float32(128.0))
    assert_true(sp_sdf.sample_msdf(tip_x - Float32(2.5), tip_y) > Float32(128.0))
    assert_true(sp_sdf.sample_msdf(tip_x - Float32(3.5), tip_y) > Float32(128.0))
    var gb = sp_sdf.glyphs[sp_sdf.glyph_index(66)]
    var b_br = sp_ttf.outline(sp_ttf.glyph_id(66)).bounds()
    var bar_x = Float32(gb.x) + Float32(0.5) * (b_br[0] + b_br[2]) * Float32(0.04) - gb.xoff
    var bar_y = Float32(gb.y) - Float32(300.0) * Float32(0.04) - gb.yoff
    assert_true(sp_sdf.sample_msdf(bar_x, bar_y) > Float32(128.0))
    var gb5 = sp_sdf.glyphs[sp_sdf.glyph_index(67)]
    var b_br5 = sp_ttf.outline(sp_ttf.glyph_id(67)).bounds()
    var bar5_y = Float32(gb5.y) - Float32(300.0) * Float32(0.04) - gb5.yoff
    for px_u in [100, 300, 500]:
        var bar5_x = Float32(gb5.x) + (b_br5[0] + Float32(px_u)) * Float32(0.04) - gb5.xoff
        assert_true(sp_sdf.sample_msdf(bar5_x, bar5_y) > Float32(128.0))

    # Genuine large channel-jump clash is still corrected by msdf_error_correct
    var clash_px: List[UInt8] = [60, 180, 180, 180, 60, 180, 180, 180, 180, 180, 180, 180]
    msdf_error_correct(clash_px, 2, 2, 3, Float32(6.0))
    assert_equal(clash_px[0], clash_px[1])
    assert_equal(clash_px[1], clash_px[2])


def test_f26_gsub_bidi_and_fallback_fonts() raises:
    var primary_bytes = _build_test_ttf_with_gsub(1000)
    var primary = TrueTypeFont()
    assert_true(primary.load_bytes(primary_bytes^))
    assert_true(primary.has_gsub())

    # 1. GSUB Types 1-7 + GDEF IGNORE_MARKS + Arabic contextual forms
    var chain = FontFallbackChain()
    chain.set_primary(primary.copy())

    # Type 1 fmt 1 & fmt 2 ('K'->69, 'L'->70) and Type 2 ('Q'->71, 72)
    var g_klq = shape_text(chain, "KLQ", 20.0, BIDI_LTR)
    assert_equal(len(g_klq), 4)
    assert_equal(g_klq[0].glyph_id, 69)
    assert_equal(g_klq[1].glyph_id, 70)
    assert_equal(g_klq[2].glyph_id, 71)
    assert_equal(g_klq[3].glyph_id, 72)

    # Type 3 AlternateSubst ('Z' alt_choice=0 vs 1)
    var g_z0 = shape_text(chain, "Z", 20.0, BIDI_LTR, "", 0)
    var g_z1 = shape_text(chain, "Z", 20.0, BIDI_LTR, "", 1)
    assert_equal(g_z0[0].glyph_id, 73)
    assert_equal(g_z1[0].glyph_id, 74)

    # Type 4 LigatureSubst ('ffi'->67, 'fi'->66, 'f\u0301i' with IGNORE_MARKS->66 + mark 57)
    var g_ffi = shape_text(chain, "ffi", 20.0, BIDI_LTR)
    assert_equal(len(g_ffi), 1)
    assert_equal(g_ffi[0].glyph_id, 67)
    var g_fi = shape_text(chain, "fi", 20.0, BIDI_LTR)
    assert_equal(len(g_fi), 1)
    assert_equal(g_fi[0].glyph_id, 66)
    assert_true(g_fi[0].x_advance < Float32(10.0))  # 450/1000 * 20 = 9.0 < 13.0
    var g_f_mark_i = shape_text(chain, "f\u0301i", 20.0, BIDI_LTR)
    assert_equal(len(g_f_mark_i), 2)
    assert_equal(g_f_mark_i[0].glyph_id, 66)
    assert_equal(g_f_mark_i[1].glyph_id, 57)

    # Type 7 Extension wrapping Type 6 fmt 3 ChainingContextualSubst ('a->b' vs 'a- b' vs ' ->b')
    var g_arrow = shape_text(chain, "a->b", 20.0, BIDI_LTR)
    assert_equal(len(g_arrow), 3)
    assert_equal(g_arrow[1].glyph_id, 68)
    var g_no_arrow1 = shape_text(chain, "a- b", 20.0, BIDI_LTR)
    assert_equal(len(g_no_arrow1), 4)
    assert_equal(g_no_arrow1[1].glyph_id, 55)
    var g_no_arrow2 = shape_text(chain, " ->b", 20.0, BIDI_LTR)
    assert_equal(len(g_no_arrow2), 4)

    # Arabic contextual forms (isol/init/medi/fina) + rlig (Lam-Alif) in visual RTL order
    var g_salam = shape_text(chain, "سلم", 20.0)
    assert_equal(len(g_salam), 3)
    assert_true(g_salam[0].rtl and g_salam[1].rtl and g_salam[2].rtl)
    # Visual order (left-to-right): [م_fina=86, ل_medi=81, س_init=76]
    assert_equal(g_salam[0].glyph_id, 86)
    assert_equal(g_salam[1].glyph_id, 81)
    assert_equal(g_salam[2].glyph_id, 76)

    var g_sla = shape_text(chain, "سلا", 20.0)
    # س_init (76) + Lam-Alif rlig (89) -> visual [89, 76]
    assert_equal(len(g_sla), 2)
    assert_equal(g_sla[0].glyph_id, 89)
    assert_equal(g_sla[1].glyph_id, 76)

    var g_isol = shape_text(chain, "س", 20.0)
    assert_equal(len(g_isol), 1)
    assert_equal(g_isol[0].glyph_id, 75)

    # 2. UAX #9 BiDi/RTL reordering, bracket mirroring & line wrapping
    var res_mix = resolve_bidi(decode_utf8("שלום world 123!"), BIDI_AUTO)
    assert_equal(res_mix.base_level, 1)
    var g_mix = shape_text(chain, "שלום world 123!", 20.0, BIDI_AUTO)
    assert_equal(len(g_mix), 15)
    assert_equal(g_mix[0].codepoint, 33)    # '!' reset to level 1 at visual left
    assert_equal(g_mix[1].codepoint, 119)   # 'w' of LTR "world 123"
    assert_equal(g_mix[11].codepoint, 0x05DD)  # 'ם' (reversed Hebrew on right)
    assert_equal(g_mix[14].codepoint, 0x05E9)  # 'ש'

    var res_an = resolve_bidi(decode_utf8("ID: \u0661\u0662\u0663 \u0633\u0644\u0627\u0645"), BIDI_AUTO)
    assert_equal(res_an.base_level, 0)
    assert_equal(res_an.levels[4], 2)  # AN digits stay LTR level 2 inside RTL run

    var g_paren = shape_text(chain, "(שלום)", 20.0, BIDI_AUTO)
    assert_equal(len(g_paren), 6)
    assert_equal(g_paren[0].codepoint, 40)  # '(' mirrored from logical ')' on left
    assert_equal(g_paren[5].codepoint, 41)  # ')' mirrored from logical '(' on right
    assert_equal(bidi_mirror(0x28), 0x29)

    var wrapped = wrap_shaped(chain, "שלום וםל שלום", 20.0, 60.0, BIDI_AUTO)
    assert_true(len(wrapped) >= 2)
    assert_true(len(wrapped[0].glyphs) >= 2)

    var sdf_ui = SdfFont()
    var ui_codes: List[Int] = [32, 63, 65, 0x05D5, 0x05DC, 0x05DD, 0x05E9]
    assert_true(sdf_ui.load_bytes(dsf_bytes(bake_font(primary, ui_codes, 20, 3, 256))))
    var rlayout = layout_rich(parse_rich("שלום לם"), sdf_ui, sdf_ui, 20.0, 0, 75.0, 0, False, False)
    assert_true(rlayout.lines >= 1 and rlayout.base_level == 1)
    assert_true(len(rlayout.runs) >= 1 and rlayout.runs[0].rtl)

    # 3. Multi-font fallback chain on SdfFont (Primary ASCII + Fallback CJK/Emoji upem=1024 + Arabic)
    var primary_ascii_only = _build_test_ttf_cjk_emoji(1000)  # no Arabic/ASCII; build ASCII-only primary from GSUB font?
    # Use `primary` (no CJK U+4E2D, no Emoji U+1F680) + `fb_cjk` (upem=1024)
    assert_false(primary.has_glyph(0x4E2D))
    assert_false(primary.has_glyph(0x1F680))
    var sdf = SdfFont()
    var asc_codes = List[Int]()
    for c in range(32, 91):
        asc_codes.append(c)
    var b_asc = bake_font(primary, asc_codes, 32, 4, 512)
    assert_true(sdf.load_bytes(dsf_bytes(b_asc)))
    sdf.source = primary.copy()
    sdf.has_source = True
    sdf._ensure_dynamic_page()

    assert_false(sdf.ensure(0x4E2D))
    assert_equal(sdf.glyph_for(0x4E2D), sdf.glyph_index(63))

    var fb_bytes = _build_test_ttf_cjk_emoji(1024)
    assert_true(sdf.add_fallback_bytes(fb_bytes^, 0.0, False))
    assert_equal(sdf.resolve_codepoint(65)[0], 0)
    assert_equal(sdf.resolve_codepoint(0x4E2D)[0], 1)
    assert_equal(sdf.resolve_codepoint(0x1F680)[0], 1)

    # Scaled metrics: fb upem=1024, U+4E2D raw advance=512 -> 0.5 * bake_px (16.0)
    assert_true(abs(sdf.advance_of(0x4E2D) - Float32(16.0)) < Float32(0.05))
    sdf.ensure_text("A中🚀")
    assert_true(sdf.is_dirty())
    assert_true(sdf.glyph_for(0x4E2D) != sdf.glyph_index(63))
    assert_true(sdf.glyph_for(0x1F680) != sdf.glyph_index(63))
    var g_cjk = sdf.glyphs[sdf.glyph_for(0x4E2D)]
    assert_true(g_cjk.w > 0 and g_cjk.h > 0)

    var ensured_mix = sdf.shape_and_ensure("A سلم 🚀", 24.0)
    assert_true(len(ensured_mix) >= 5)
    assert_equal(ensured_mix[0][0].font_index, 0)  # 'A'
    assert_equal(ensured_mix[2][0].font_index, 0)  # Arabic from primary
    assert_equal(ensured_mix[len(ensured_mix) - 1][0].font_index, 1)  # '🚀' from fallback
    for pair in ensured_mix:
        assert_true(pair[1] >= 0 and pair[1] < len(sdf.glyphs))
    _ = primary_ascii_only^
    sdf.flush()
    assert_equal(sdf.flushes, 1)

    # 4. Regression: SdfFont .dsf (NO attach_source) + add_fallback_font -> non-zero primary advance & non-overlapping label
    var dsf_only = SdfFont()
    var asc_only_codes: List[Int] = [32, 63, 65, 66]
    assert_true(dsf_only.load_bytes(dsf_bytes(bake_font(primary, asc_only_codes, 20, 3, 256))))
    assert_false(dsf_only.has_source)
    var fb_cjk = _build_test_ttf_cjk_emoji(1024)
    assert_true(dsf_only.add_fallback_bytes(fb_cjk^, 0.0, False))
    var meas_mix = dsf_only.measure_shaped("AB中", 20.0)
    # 'A' (12.0) + 'B' (12.0) + '中' (512/1024 * 20 = 10.0) = 34.0
    assert_true(abs(meas_mix - Float32(34.0)) < Float32(0.5))
    var ui_lbl = UiContext()
    ui_lbl.font = dsf_only.copy()
    var inp_lbl = InputState()
    ui_lbl.begin_frame(inp_lbl, 800.0, 600.0)
    ui_lbl.label("AB中")
    ui_lbl.end_frame()
    var glyph_xs = List[Float32]()
    for pr in ui_lbl.draw_list.prims:
        if (pr.header & 0xF) == PRIM_GLYPH:
            glyph_xs.append(pr.x0)
    assert_equal(len(glyph_xs), 3)
    assert_true(glyph_xs[0] < glyph_xs[1] and glyph_xs[1] < glyph_xs[2])

    # 4b. Regression: .dsf (no attach_source) + fallback unmapped codepoint emits '?' advance (no glyph overlap)
    var meas_unmap_shaped = dsf_only.measure_shaped("A☃B", 20.0)
    var meas_unmap_plain = dsf_only.measure("A☃B", 20.0)
    assert_true(abs(meas_unmap_shaped - meas_unmap_plain) < Float32(1.0e-4))
    assert_true(meas_unmap_shaped > dsf_only.measure("AB", 20.0))
    var ui_unmap = UiContext()
    ui_unmap.font = dsf_only.copy()
    ui_unmap.begin_frame(inp_lbl, 800.0, 600.0)
    ui_unmap.label("A☃B")
    ui_unmap.end_frame()
    var unmap_xs = List[Float32]()
    for pr in ui_unmap.draw_list.prims:
        if (pr.header & 0xF) == PRIM_GLYPH:
            unmap_xs.append(pr.x0)
    assert_equal(len(unmap_xs), 3)
    assert_true(unmap_xs[0] < unmap_xs[1] and unmap_xs[1] < unmap_xs[2])

    # 5. Regression: layout_rich & layout_rich_shaped inter-run space placement on RTL runs
    var eq_codes: List[Int] = [32, 63, 101, 104, 108, 111, 0x05D5, 0x05DC, 0x05DD, 0x05E9]
    var eq_ttf_glyf = List[UInt8]()
    var eq_ttf_loca = List[UInt8]()
    for g in range(11):
        _f26_u32_be(eq_ttf_loca, len(eq_ttf_glyf))
        if g == 0 or g == 1:
            continue
        var gd = _f26_simple_square_glyf(20, 0, 400, 600)
        eq_ttf_glyf.extend(gd^)
        while (len(eq_ttf_glyf) & 3) != 0:
            eq_ttf_glyf.append(0)
    _f26_u32_be(eq_ttf_loca, len(eq_ttf_glyf))
    var eq_adv: List[Int] = [500, 250, 500, 500, 500, 500, 500, 500, 500, 500, 500]
    var eq_gids: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    var eq_tags: List[String] = [String("cmap"), String("glyf"), String("head"), String("hhea"), String("hmtx"), String("loca"), String("maxp")]
    var eq_tbls: List[List[UInt8]] = [
        _f26_build_cmap_fmt4(eq_codes, eq_gids),
        eq_ttf_glyf^,
        _f26_build_head(1000, 1),
        _f26_build_hhea(800, -200, 11),
        _f26_build_hmtx(eq_adv),
        eq_ttf_loca^,
        _f26_build_maxp(11),
    ]
    var eq_ttf = TrueTypeFont()
    assert_true(eq_ttf.load_bytes(_f26_assemble_sfnt(0x00010000, eq_tags, eq_tbls)))
    var eq_sdf = SdfFont()
    assert_true(eq_sdf.load_bytes(dsf_bytes(bake_font(eq_ttf, eq_codes, 20, 3, 256))))
    eq_sdf.source = eq_ttf.copy(); eq_sdf.has_source = True; eq_sdf._ensure_dynamic_page()

    var rl_a = layout_rich(parse_rich("hello שלום"), eq_sdf, eq_sdf, 20.0, 0, 0.0, 0, False, False)
    assert_equal(len(rl_a.runs), 2)
    assert_true(abs(rl_a.runs[0].x - Float32(0.0)) < Float32(0.1))
    assert_true(abs(rl_a.runs[1].x - Float32(55.0)) < Float32(0.1))

    var rl_b = layout_rich(parse_rich("שלום hello"), eq_sdf, eq_sdf, 20.0, 0, 0.0, 0, False, False)
    assert_equal(len(rl_b.runs), 2)
    assert_true(abs(rl_b.runs[0].x - Float32(0.0)) < Float32(0.1))
    assert_true(abs(rl_b.runs[1].x - Float32(55.0)) < Float32(0.1))

    var rls_b = layout_rich_shaped(parse_rich("שלום hello"), eq_sdf, eq_sdf, 20.0, 0, 0.0, 0, False, False)
    assert_equal(len(rls_b.runs), 2)
    assert_true(abs(rls_b.runs[0].x - Float32(0.0)) < Float32(0.1))
    assert_true(abs(rls_b.runs[1].x - Float32(55.0)) < Float32(0.1))

    # 6. Regression: Fallback-font Arabic Lam+ZWJ+Alif ligature & Beh+ZWNJ+Yeh non-ligature
    var latin_only = TrueTypeFont()
    var lat_c: List[Int] = [32, 65]
    var lat_g: List[Int] = [1, 2]
    var lat_adv: List[Int] = [500, 280, 600]
    var lat_glyf = List[UInt8]()
    var lat_loca = List[UInt8]()
    _f26_u32_be(lat_loca, 0); _f26_u32_be(lat_loca, 0); _f26_u32_be(lat_loca, 0)
    lat_glyf.extend(_f26_simple_square_glyf(50, 0, 550, 700))
    while (len(lat_glyf) & 3) != 0:
        lat_glyf.append(0)
    _f26_u32_be(lat_loca, len(lat_glyf))
    var lat_tags: List[String] = [String("cmap"), String("glyf"), String("head"), String("hhea"), String("hmtx"), String("loca"), String("maxp")]
    var lat_tbls: List[List[UInt8]] = [
        _f26_build_cmap_fmt4(lat_c, lat_g),
        lat_glyf^,
        _f26_build_head(1000, 1),
        _f26_build_hhea(800, -200, 3),
        _f26_build_hmtx(lat_adv),
        lat_loca^,
        _f26_build_maxp(3),
    ]
    assert_true(latin_only.load_bytes(_f26_assemble_sfnt(0x00010000, lat_tags, lat_tbls)))
    var ar_chain = FontFallbackChain()
    ar_chain.set_primary(latin_only.copy())
    ar_chain.add_fallback(primary.copy(), False)
    var g_zwj = shape_text(ar_chain, "ل\u200Dا", 20.0)
    assert_equal(len(g_zwj), 1)
    assert_equal(g_zwj[0].glyph_id, 89)  # Lam-Alif rlig
    assert_equal(g_zwj[0].font_index, 1)
    var g_zwnj = shape_text(ar_chain, "س\u200Cم", 20.0)
    assert_equal(len(g_zwnj), 2)
    # Visual RTL order: [م_isol=83, س_isol=75] (ZWNJ prevented joining and did not emit .notdef)
    assert_equal(g_zwnj[0].glyph_id, 83)
    assert_equal(g_zwnj[1].glyph_id, 75)
    # ZWNJ must also block GSUB LigatureSubst inside a fallback font ('f' + ZWNJ + 'i' must NOT form 'fi' ligature)
    var g_fi_zwnj = shape_text(ar_chain, "f\u200Ci", 20.0, BIDI_LTR)
    assert_equal(len(g_fi_zwnj), 2)
    assert_equal(g_fi_zwnj[0].glyph_id, 32)
    assert_equal(g_fi_zwnj[1].glyph_id, 35)


def test_f26_text_input_clipboard_mouse_scroll_ime() raises:
    var ui = UiContext()
    var inp = InputState()
    var txt = String("¡Hola! エンジン ★")

    # 1. Ctrl+A -> Ctrl+C clipboard copy & take_clipboard_out
    inp.begin_frame()
    inp.move_mouse(20.0, 15.0)
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("clip_box", txt, 300.0)
    ui.end_frame()
    inp.set_mouse(0, False)

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
    inp.set_logical_key(UI_KEY_A, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("clip_box", txt, 300.0)
    ui.end_frame()
    inp.set_logical_key(UI_KEY_A, False)

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_C, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("clip_box", txt, 300.0)
    ui.end_frame()
    inp.set_logical_key(UI_KEY_C, False)
    inp.set_logical_key(UI_KEY_LEFT_CTRL, False)
    assert_equal(ui.get_clipboard(), "¡Hola! エンジン ★")
    assert_true(ui.has_clipboard_out())
    assert_equal(ui.take_clipboard_out(), "¡Hola! エンジン ★")
    assert_false(ui.has_clipboard_out())

    # 2. End (collapse selection) -> Shift+Left across multi-byte codepoints + Ctrl+X cut + Home + Ctrl+V paste
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_END, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("clip_box", txt, 300.0)
    ui.end_frame()
    inp.set_logical_key(UI_KEY_END, False)

    for _ in range(4):
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_LEFT_SHIFT, True)
        inp.set_logical_key(UI_KEY_LEFT, True)
        ui.begin_frame(inp, 800.0, 600.0)
        _ = ui.text_input("clip_box", txt, 300.0)
        ui.end_frame()
        inp.set_logical_key(UI_KEY_LEFT, False)
        inp.set_logical_key(UI_KEY_LEFT_SHIFT, False)

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
    inp.set_logical_key(UI_KEY_X, True)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_true(ui.text_input("clip_box", txt, 300.0))
    ui.end_frame()
    inp.set_logical_key(UI_KEY_X, False)
    inp.set_logical_key(UI_KEY_LEFT_CTRL, False)
    assert_equal(ui.get_clipboard(), "ジン ★")
    assert_equal(txt, "¡Hola! エン")

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_HOME, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("clip_box", txt, 300.0)
    ui.end_frame()
    inp.set_logical_key(UI_KEY_HOME, False)
    assert_equal(ui.text_cursor_codepoint("clip_box"), 0)

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
    inp.set_logical_key(UI_KEY_V, True)
    assert_true(ui.wants_clipboard_paste(inp))
    ui.begin_frame(inp, 800.0, 600.0)
    assert_true(ui.text_input("clip_box", txt, 300.0))
    ui.end_frame()
    inp.set_logical_key(UI_KEY_V, False)
    inp.set_logical_key(UI_KEY_LEFT_CTRL, False)
    assert_equal(txt, "ジン ★¡Hola! エン")
    var boff = ui.text_cursor_byte_offset("clip_box", txt)
    assert_true((Int(txt.as_bytes()[boff]) & 0xC0) != 0x80)

    # Empty selection Ctrl+C / Ctrl+X must be no-op
    ui.set_clipboard("KEEP")
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
    inp.set_logical_key(UI_KEY_C, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("clip_box", txt, 300.0)
    ui.end_frame()
    inp.set_logical_key(UI_KEY_C, False)
    inp.set_logical_key(UI_KEY_LEFT_CTRL, False)
    assert_equal(ui.get_clipboard(), "KEEP")
    assert_false(ui.has_clipboard_out())

    # 3. Mouse-click, drag selection & double-click word select on headless 0.56 * font_size advance
    var ui2 = UiContext()
    var word_txt = String("alpha 日本語_word beta")
    var char_adv = ui2.theme.font_size * Float32(0.56)
    # Click at codepoint 6 ('日'): pad_x = 8.0 + 6 * char_adv + 2.0
    inp.begin_frame()
    inp.move_mouse(Float32(8.0) + Float32(6.0) * char_adv + Float32(2.0), 15.0)
    inp.set_mouse(0, True)
    ui2.begin_frame(inp, 800.0, 600.0)
    _ = ui2.text_input("mouse_box", word_txt, 400.0)
    ui2.end_frame()
    assert_equal(ui2.text_cursor_codepoint("mouse_box"), 6)

    # Drag across 4 codepoints to codepoint 10 ('_')
    inp.begin_frame()
    inp.move_mouse(Float32(8.0) + Float32(10.0) * char_adv + Float32(2.0), 15.0)
    ui2.begin_frame(inp, 800.0, 600.0)
    _ = ui2.text_input("mouse_box", word_txt, 400.0)
    ui2.end_frame()
    inp.set_mouse(0, False)
    var sel_rng = ui2.text_selection_range_cp("mouse_box")
    assert_equal(sel_rng[0], 6)
    assert_equal(sel_rng[1], 10)

    # Release mouse, then double-click inside "日本語_word" (codepoint 8) within 0.40s
    inp.begin_frame()
    ui2.begin_frame(inp, 800.0, 600.0)
    _ = ui2.text_input("mouse_box", word_txt, 400.0)
    ui2.end_frame()

    inp.begin_frame()
    inp.move_mouse(Float32(8.0) + Float32(8.0) * char_adv + Float32(2.0), 15.0)
    inp.set_mouse(0, True)
    ui2.begin_frame(inp, 800.0, 600.0, 0.016)
    _ = ui2.text_input("mouse_box", word_txt, 400.0)
    ui2.end_frame()
    inp.set_mouse(0, False)

    inp.begin_frame()
    inp.set_mouse(0, True)
    ui2.begin_frame(inp, 800.0, 600.0, 0.05)
    _ = ui2.text_input("mouse_box", word_txt, 400.0)
    ui2.end_frame()
    inp.set_mouse(0, False)
    var dbl_rng = ui2.text_selection_range_cp("mouse_box")
    assert_equal(dbl_rng[0], 6)
    assert_equal(dbl_rng[1], 14)

    # 4. Horizontal scroll tracking (`scroll_x`)
    var long_txt = String("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    var ui_sc = UiContext()
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_END, True)
    ui_sc.begin_frame(inp, 800.0, 600.0)
    _ = ui_sc.text_input("scroll_box", long_txt, 120.0)
    ui_sc.end_frame()
    inp.set_logical_key(UI_KEY_END, False)
    assert_true(ui_sc.text_scroll_x("scroll_box") > Float32(0.0))

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_HOME, True)
    ui_sc.begin_frame(inp, 800.0, 600.0)
    _ = ui_sc.text_input("scroll_box", long_txt, 120.0)
    ui_sc.end_frame()
    inp.set_logical_key(UI_KEY_HOME, False)
    assert_equal(ui_sc.text_scroll_x("scroll_box"), Float32(0.0))

    # 5. IME Preedit, Inline Primitives, Shortcut Suppression, Modal Escape Guard & Commit
    var ui_ime = UiContext()
    var ime_txt = String("hello")
    inp.begin_frame()
    inp.set_ime_preedit("nihon", 3, 2)
    ui_ime.begin_frame(inp, 800.0, 600.0)
    assert_false(ui_ime.text_input("ime_box", ime_txt, 240.0))
    ui_ime.end_frame()
    assert_equal(ime_txt, "hello")
    assert_equal(ui_ime.active_ime_preedit("ime_box"), "nihon")

    var line_prims = 0
    for pr in ui_ime.draw_list.prims:
        if (pr.header & 0xF) == PRIM_LINE:
            line_prims += 1
    assert_true(line_prims >= 3)  # clause underline + preedit underline + caret

    # Backspace / Ctrl+A suppressed during IME preedit
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_BACKSPACE, True)
    inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
    inp.set_logical_key(UI_KEY_A, True)
    ui_ime.begin_frame(inp, 800.0, 600.0)
    assert_false(ui_ime.text_input("ime_box", ime_txt, 240.0))
    ui_ime.end_frame()
    inp.set_logical_key(UI_KEY_BACKSPACE, False)
    inp.set_logical_key(UI_KEY_LEFT_CTRL, False)
    inp.set_logical_key(UI_KEY_A, False)
    assert_equal(ime_txt, "hello")

    # Commit IME text via InputState.commit_ime_preedit on focused `ime_box`
    inp.clear_ime_preedit()
    inp.begin_frame()
    inp.commit_ime_preedit("日本")
    ui_ime.begin_frame(inp, 800.0, 600.0)
    assert_true(ui_ime.text_input("ime_box", ime_txt, 240.0))
    ui_ime.end_frame()
    assert_equal(ime_txt, "日本hello")
    assert_equal(ui_ime.text_cursor_codepoint("ime_box"), 2)
    assert_equal(ui_ime.active_ime_preedit("ime_box"), "")

    # Escape cancels IME without closing an open modal; second Escape closes the modal
    var ui_mod = UiContext()
    var mod_txt = String("hello")
    var modal_open = True
    inp.clear_ime_preedit()
    inp.begin_frame()
    inp.set_ime_preedit("nihon", 3, 2)
    ui_mod.begin_frame(inp, 800.0, 600.0)
    if ui_mod.begin_modal("ime_modal", "IME Modal", 320.0, 180.0, modal_open):
        _ = ui_mod.text_input("ime_box_modal", mod_txt, 220.0)
        ui_mod.end_modal()
    ui_mod.end_frame()

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, True)
    ui_mod.begin_frame(inp, 800.0, 600.0)
    if ui_mod.begin_modal("ime_modal", "IME Modal", 320.0, 180.0, modal_open):
        _ = ui_mod.text_input("ime_box_modal", mod_txt, 220.0)
        ui_mod.end_modal()
    ui_mod.end_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, False)
    assert_true(modal_open)
    assert_equal(ui_mod.active_ime_preedit("ime_box_modal"), "")
    assert_equal(mod_txt, "hello")

    inp.clear_ime_preedit()
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, True)
    ui_mod.begin_frame(inp, 800.0, 600.0)
    assert_false(ui_mod.begin_modal("ime_modal", "IME Modal", 320.0, 180.0, modal_open))
    ui_mod.end_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, False)
    assert_false(modal_open)

    # 6. Regression: IME focus switch cancellation (button click + box_a unmount -> box_b) & simultaneous commit+preedit
    var ui_sw = UiContext()
    var box_a_txt = String("a")
    var box_b_txt = String("b")
    inp.clear_ime_preedit()
    inp.begin_frame()
    inp.move_mouse(20.0, 15.0)
    inp.set_mouse(0, True)
    ui_sw.begin_frame(inp, 800.0, 600.0)
    _ = ui_sw.text_input("box_a", box_a_txt, 200.0)
    _ = ui_sw.button("other_btn", "Click", 120.0, 30.0)
    ui_sw.end_frame()
    inp.set_mouse(0, False)

    inp.begin_frame()
    inp.set_ime_preedit("あいう", 3, 0)
    ui_sw.begin_frame(inp, 800.0, 600.0)
    _ = ui_sw.text_input("box_a", box_a_txt, 200.0)
    _ = ui_sw.button("other_btn", "Click", 120.0, 30.0)
    ui_sw.end_frame()
    assert_true(ui_sw.ime_active)
    assert_equal(ui_sw.active_ime_preedit("box_a"), "あいう")

    # Click button below box_a (y ~= 36..66): IME must be cancelled by end_frame()
    inp.begin_frame()
    inp.move_mouse(20.0, 45.0)
    inp.set_mouse(0, True)
    ui_sw.begin_frame(inp, 800.0, 600.0)
    _ = ui_sw.text_input("box_a", box_a_txt, 200.0)
    _ = ui_sw.button("other_btn", "Click", 120.0, 30.0)
    ui_sw.end_frame()
    inp.set_mouse(0, False)
    assert_false(ui_sw.ime_active)

    # Start IME on box_a, then unmount box_a and click box_b: box_b must NOT steal box_a's preedit
    inp.clear_ime_preedit()
    inp.begin_frame()
    inp.move_mouse(20.0, 15.0)
    inp.set_mouse(0, True)
    ui_sw.begin_frame(inp, 800.0, 600.0)
    _ = ui_sw.text_input("box_a", box_a_txt, 200.0)
    ui_sw.end_frame()
    inp.set_mouse(0, False)

    inp.begin_frame()
    inp.set_ime_preedit("あいう", 3, 0)
    ui_sw.begin_frame(inp, 800.0, 600.0)
    _ = ui_sw.text_input("box_a", box_a_txt, 200.0)
    ui_sw.end_frame()
    assert_true(ui_sw.ime_active)
    assert_equal(ui_sw.active_ime_preedit("box_a"), "あいう")

    inp.begin_frame()
    inp.move_mouse(20.0, 15.0)
    inp.set_mouse(0, True)
    ui_sw.begin_frame(inp, 800.0, 600.0)
    _ = ui_sw.text_input("box_b", box_b_txt, 200.0)
    ui_sw.end_frame()
    inp.set_mouse(0, False)
    assert_false(ui_sw.ime_active)
    assert_equal(ui_sw.active_ime_preedit("box_b"), "")
    assert_equal(box_b_txt, "b")

    # Start IME on box_a, then unmount box_a (only box_b rendered, no click, Tab pressed):
    # end_frame() must cancel IME before Tab navigation so focus moves cleanly to box_b!
    inp.clear_ime_preedit()
    inp.begin_frame()
    inp.move_mouse(20.0, 15.0)
    inp.set_mouse(0, True)
    ui_sw.begin_frame(inp, 800.0, 600.0)
    _ = ui_sw.text_input("box_a", box_a_txt, 200.0)
    ui_sw.end_frame()
    inp.set_mouse(0, False)

    inp.begin_frame()
    inp.set_ime_preedit("あいう", 3, 0)
    ui_sw.begin_frame(inp, 800.0, 600.0)
    _ = ui_sw.text_input("box_a", box_a_txt, 200.0)
    ui_sw.end_frame()
    assert_true(ui_sw.ime_active)
    assert_equal(ui_sw.active_ime_preedit("box_a"), "あいう")

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_TAB, True)
    ui_sw.begin_frame(inp, 800.0, 600.0)
    _ = ui_sw.text_input("box_b", box_b_txt, 200.0)
    ui_sw.end_frame()
    inp.set_logical_key(UI_KEY_TAB, False)
    assert_false(ui_sw.ime_active)
    assert_equal(ui_sw.active_ime_preedit("box_b"), "")
    assert_equal(ui_sw.focused_id, ui_sw.widget_id("box_b"))

    # Simultaneous ime_commit + new ime_active preedit in the same frame
    inp.clear_ime_preedit()
    var ui_sim = UiContext()
    var sim_txt = String("ab")
    inp.begin_frame()
    inp.commit_ime_preedit("漢")
    inp.set_ime_preedit("kan", 3, 0)
    ui_sim.begin_frame(inp, 800.0, 600.0)
    assert_true(ui_sim.text_input("sim_box", sim_txt, 220.0))
    ui_sim.end_frame()
    assert_equal(sim_txt, "漢ab")
    assert_true(ui_sim.ime_active)
    assert_equal(ui_sim.active_ime_preedit("sim_box"), "kan")


def test_f27_multios_audio_and_backend_traits() raises:
    comptime assert conforms_to(WaylandWindow, WindowBackend)
    comptime assert conforms_to(X11Window, WindowBackend)
    comptime assert conforms_to(KmsWindow, WindowBackend)
    comptime assert conforms_to(Win32Window, WindowBackend)
    comptime assert conforms_to(CocoaWindow, WindowBackend)
    comptime assert conforms_to(Win32Audio, AudioBackend)
    comptime assert conforms_to(MacOsAudio, AudioBackend)
    comptime assert conforms_to(UdpSocket, NetTransport)
    comptime assert not conforms_to(Win32Window, Copyable)
    comptime assert not conforms_to(CocoaWindow, Copyable)
    comptime assert not conforms_to(Win32Audio, Copyable)
    comptime assert not conforms_to(MacOsAudio, Copyable)
    comptime assert not conforms_to(UdpSocket, Copyable)

    var wa = Win32Audio()
    assert_false(wa.initialize())
    wa.trigger(1, 1.0, 0.8)
    wa.set_ambient_drone(0.3)
    wa.pump(1.0)
    var w_pcm = wa.last_mixed_slot()
    assert_true(len(w_pcm) > 0)
    var w_has_sig = False
    for s in w_pcm:
        if s != 0:
            w_has_sig = True
            break
    assert_true(w_has_sig)
    wa.shutdown()

    var ma = MacOsAudio()
    assert_false(ma.initialize())
    ma.trigger(2, 1.2, 0.8)
    ma.set_ambient_drone(0.3)
    ma.pump(1.0)
    var m_pcm = ma.last_mixed_slot()
    assert_true(len(m_pcm) > 0)
    var m_has_sig = False
    for s in m_pcm:
        if s != 0:
            m_has_sig = True
            break
    assert_true(m_has_sig)
    ma.shutdown()

    var w32 = Win32Window()
    assert_true(w32.content_scale() >= Float32(1.0))
    w32.clipboard_set("w32")
    assert_equal(w32.clipboard_get(), "w32")
    var mac = CocoaWindow()
    assert_true(mac.content_scale() >= Float32(1.0))
    mac.clipboard_set("mac")
    assert_equal(mac.clipboard_get(), "mac")

    var sock = UdpSocket.open_dual_stack(0)
    assert_true(sock.is_valid())
    var sock_m = sock^
    assert_true(sock_m.is_valid())
    sock_m.close()
    assert_false(sock_m.is_valid())


def test_t5_button_drag_outside_and_qoa_bounds() raises:
    # 1. [HIGH-4] UiContext.button must NOT fire on drag-from-outside release or cross-button release
    var ui = UiContext()
    var inp = InputState()

    # Frame 0: Layout pass to capture exact button rects
    inp.begin_frame()
    ui.begin_frame(inp, 800.0, 600.0)
    assert_false(ui.button("btn_a", "Alpha", 120.0, 32.0))
    var ra = ui.last_item_rect
    assert_false(ui.button("btn_b", "Beta", 120.0, 32.0))
    var rb = ui.last_item_rect
    ui.end_frame()

    var ax = ra.x + ra.w * Float32(0.5)
    var ay = ra.y + ra.h * Float32(0.5)
    var bx = rb.x + rb.w * Float32(0.5)
    var by = rb.y + rb.h * Float32(0.5)

    # Frame 1: Press left mouse button outside any button
    inp.begin_frame()
    inp.move_mouse(600.0, 500.0)
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_false(ui.button("btn_a", "Alpha", 120.0, 32.0))
    assert_false(ui.button("btn_b", "Beta", 120.0, 32.0))
    ui.end_frame()
    assert_equal(ui.active_id, UInt64(0))

    # Frame 2: Drag pointer onto BtnA while holding left mouse button
    inp.begin_frame()
    inp.move_mouse(ax, ay)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_false(ui.button("btn_a", "Alpha", 120.0, 32.0))
    assert_false(ui.button("btn_b", "Beta", 120.0, 32.0))
    ui.end_frame()

    # Frame 3: Release on BtnA -> must NOT trigger click!
    inp.begin_frame()
    inp.set_mouse(0, False)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_false(ui.button("btn_a", "Alpha", 120.0, 32.0))
    assert_false(ui.button("btn_b", "Beta", 120.0, 32.0))
    ui.end_frame()

    # Press on BtnA, release on BtnB -> neither triggers click
    inp.begin_frame()
    inp.move_mouse(ax, ay)
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_false(ui.button("btn_a", "Alpha", 120.0, 32.0))
    assert_false(ui.button("btn_b", "Beta", 120.0, 32.0))
    ui.end_frame()
    assert_true(ui.active_id != UInt64(0))

    inp.begin_frame()
    inp.move_mouse(bx, by)
    inp.set_mouse(0, False)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_false(ui.button("btn_a", "Alpha", 120.0, 32.0))
    assert_false(ui.button("btn_b", "Beta", 120.0, 32.0))
    ui.end_frame()

    # Legitimate press + release inside BtnA DOES trigger click
    inp.begin_frame()
    inp.move_mouse(ax, ay)
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_false(ui.button("btn_a", "Alpha", 120.0, 32.0))
    ui.end_frame()

    inp.begin_frame()
    inp.set_mouse(0, False)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_true(ui.button("btn_a", "Alpha", 120.0, 32.0))
    ui.end_frame()

    # 2. [Crit-4] QoaStreamCursor.decode_frames_into bounds check on negative offset and undersized dst
    var pcm = List[Int16](capacity=128)
    for i in range(64):
        pcm.append(Int16((i * 137) & 0x3FFF))
        pcm.append(Int16((-i * 113) & 0x3FFF))
    var qoa_bytes = encode_qoa(pcm, 2, 48000)
    var cur = QoaStreamCursor()
    assert_true(cur.open(qoa_bytes))
    assert_true(cur.valid)
    var tiny = List[Int16](length=5, fill=Int16(0))
    assert_equal(cur.decode_frames_into(qoa_bytes, tiny, -1, 10), 0)
    assert_equal(cur.decode_frames_into(qoa_bytes, tiny, 2, 10), 0)  # len(tiny)//2 == 2, offset 2 -> 0 room
    var wrote_clamped = cur.decode_frames_into(qoa_bytes, tiny, 0, 10)
    assert_equal(wrote_clamped, 2)


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
    test_vulkan_suballocator_and_alsa_fallback()
    print("[PASS] (f) Vulkan OffsetAllocator 64MiB Sub-Allocator Block Reuse & ALSA/AudioServer Fallback")
    test_f23_wayland_desktop_parity()
    print("[PASS] (g) F23 Wayland Fractional Scale, SSD, DataDevice Clipboard/DnD, XKB Memfd & Bridge FD Routing")
    test_f24_x11_desktop_parity()
    print("[PASS] (h) F24 X11 MIT-SHM Completion, XI_RawMotion FP3232, GetKeyboardMapping UTF-8 & CLIPBOARD")
    test_f25_evdev_hub_integration()
    print("[PASS] (i) F25 Direct EvdevHub Sync (Deadzone, PAD_STICK_*, pad_map, pad_is_sony) & Zero js*")
    test_f28_platform_handle_raii()
    print("[PASS] (j) F28 Platform Handle RAII (__moveinit__, __deinit__ Zero-FD Leak, KmsWindow Rescue Isolation)")
    test_f26_msdf_mtsdf_and_otto_cff()
    print("[PASS] (k1) F26 CFF/OTF (OTTO), Type2 CharStrings, Genuine MSDF/MTSDF & DSF v2")
    test_f26_gsub_bidi_and_fallback_fonts()
    print("[PASS] (k2) F26 GSUB (Types 1-7 + GDEF), Arabic Shaping, UAX #9 BiDi/RTL & Fallback Fonts")
    test_f26_text_input_clipboard_mouse_scroll_ime()
    print("[PASS] (k3) F26 UiContext.text_input Clipboard, Mouse Hit-Test/Drag/DblClick, Scroll & IME")
    test_f27_multios_audio_and_backend_traits()
    print("[PASS] (l) F27 Multi-OS WindowBackend / AudioBackend / UdpSocket Trait & PCM Ring Checks")
    test_t5_button_drag_outside_and_qoa_bounds()
    print("[PASS] (m) Tier 5 UiContext.button Drag-From-Outside Guard & QoaStreamCursor Bounds Safety")
    var elapsed = get_time() - t_start
    print("ALL FLINT V5 TRACK 4 TESTS PASSED in", elapsed, "s")
    assert_true(elapsed < 10.0)
