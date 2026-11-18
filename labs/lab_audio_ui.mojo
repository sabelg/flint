"""Flint v5 Track 4 Standalone Lab Showcase (`labs/lab_audio_ui.mojo`).

Demonstrates end-to-end integration of:
  1. Pure-Mojo QOA 5:1 lossy codec (`encode_qoa`, `decode_qoa`, O(1) `QoaStreamCursor.seek`)
  2. `Bvh8` raycast spatial acoustics (wall transmission loss, 4-ray portal pillar diffraction,
     and 6-ray Sabine `RT60` enclosure probe)
  3. Multithreaded `AudioServer` (`AudioMpscRing`, 512->64 Voice Virtualizer, Woodworth HRTF ITD,
     0.1 Mach Doppler, `CyclesStemNode`, Voice->Music sidechain ducking, BS.1770-4 LUFS &
     zero-overshoot `-0.1 dBFS` lookahead true-peak limiter)
  4. 5-panel Studio `DockSpace` + two-pass Flexbox + GPU `UiContext` (100k-entity `<0.1 ms`
     outliner, UTF-8 `text_input`, `curve_editor`, `DragDrop`, modal focus trap)
  5. `EvdevGamepad` (`EVIOCGABS`, 48B `ff_effect` rumble) & declarative `InputMap`
     (`Ctrl+S` chord, `120 ms` jump buffer, `100 ms` coyote time, deterministic lockstep tick).
"""
from std.collections import List
from std.math import sin, log10, abs
from flint.core.math import Vec3
from flint.platform.os import get_time
from flint.platform.input import InputState, PAD_A, PAD_AXIS_LT
from flint.physics.bvh import Bvh8MeshCollider
from flint.audio import (
    QOA_SLICE_BITS_PER_SAMPLE,
    QoaStreamCursor,
    encode_qoa,
    decode_qoa,
    AcousticMaterialTable,
    evaluate_emitter_acoustics,
    probe_room_sabine_rt60,
    BUS_SFX,
    BUS_MUSIC,
    BUS_VOICE,
    AudioServer,
    compute_woodworth_hrtf,
)
from flint.ui.flexbox_dock import RectF, FlexStyle, DockSpace
from flint.ui.context import UiContext
from flint.input import (
    ABS_Z,
    ABS_RZ,
    VENDOR_SONY,
    EvdevGamepad,
    pack_ff_rumble_effect,
    ACTION_BOOL,
    ACTION_AXIS2D,
    MOD_CTRL,
    InputContext,
    InputMap,
)


def _build_demo_hall_bvh() -> Bvh8MeshCollider:
    var verts = List[Vec3]()
    var hx = Float32(12.0)
    var hy = Float32(5.0)
    var hz = Float32(15.0)
    verts.append(Vec3(-hx, -hy, -hz))
    verts.append(Vec3(hx, -hy, -hz))
    verts.append(Vec3(hx, hy, -hz))
    verts.append(Vec3(-hx, hy, -hz))
    verts.append(Vec3(-hx, -hy, hz))
    verts.append(Vec3(hx, -hy, hz))
    verts.append(Vec3(hx, hy, hz))
    verts.append(Vec3(-hx, hy, hz))
    # Interior dividing wall at x = 4.0
    verts.append(Vec3(4.0, -hy, -hz))
    verts.append(Vec3(4.0, -hy, hz))
    verts.append(Vec3(4.0, hy, -hz))
    verts.append(Vec3(4.0, hy, hz))

    var faces = [
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        3, 2, 6, 3, 6, 7,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
        8, 9, 10, 10, 9, 11,
    ]
    var idxs = List[Int32]()
    for i in faces:
        idxs.append(Int32(i))
    return Bvh8MeshCollider.build(verts, idxs)


def main() raises:
    var t0 = get_time()
    print("=== Flint v5 Lab: RT AudioServer, QOA, Flexbox/DockSpace UI & Evdev InputMap ===")

    # 1. QOA Encode / Decode / O(1) Seek Demo
    var n_frames = 12000
    var pcm = List[Int16](capacity=n_frames * 2)
    for f in range(n_frames):
        var t = Float64(f) / 48000.0
        var l = 0.55 * sin(6.283185307179586 * 220.0 * t) + 0.25 * sin(6.283185307179586 * 440.0 * t)
        var r = 0.55 * sin(6.283185307179586 * 330.0 * t) + 0.25 * sin(6.283185307179586 * 660.0 * t)
        pcm.append(Int16(Int(l * 32767.0)))
        pcm.append(Int16(Int(r * 32767.0)))

    var qoa_bytes = encode_qoa(pcm, 2, 48000)
    var qoa_dec = decode_qoa(qoa_bytes)
    var sig_e = Float64(0.0)
    var err_e = Float64(0.0)
    for i in range(len(pcm)):
        var d = Float64(pcm[i]) - Float64(qoa_dec.samples[i])
        sig_e += Float64(pcm[i]) * Float64(pcm[i])
        err_e += d * d
    var snr_db = 10.0 * log10(sig_e / max(1.0, err_e))

    var cur = QoaStreamCursor()
    _ = cur.open(qoa_bytes)
    _ = cur.seek(qoa_bytes, 6400)
    var s_l = Int16(0)
    var s_r = Int16(0)
    _ = cur.read_next_frame_i16(qoa_bytes, s_l, s_r)
    print(
        "[QOA]   Encoded", n_frames, "stereo frames ->", len(qoa_bytes), "bytes (slice:",
        QOA_SLICE_BITS_PER_SAMPLE, "bps), SNR =", snr_db, "dB, seekexact =",
        s_l == qoa_dec.samples[6400 * 2],
    )

    # 2. Bvh8 Acoustics + RT AudioServer Render
    var hall_bvh = _build_demo_hall_bvh()
    var mats = AcousticMaterialTable()
    var occ = evaluate_emitter_acoustics(hall_bvh, mats, Vec3(0.0, 0.0, 0.0), Vec3(8.0, 0.0, 0.0))
    var room = probe_room_sabine_rt60(hall_bvh, mats, Vec3(0.0, 0.0, 0.0))
    var hrtf90 = compute_woodworth_hrtf(
        Vec3(), Vec3(), Vec3(0, 0, -1), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(10, 0, 0), Vec3(-34.3, 0, 0)
    )
    print(
        "[ACOUSTICS] Wall blocked cutoff =", occ.occlusion_cutoff_hz,
        "Hz | Sabine RT60 =", room.sabine_rt60_sec,
        "s | Woodworth +90d ITD =", hrtf90.delay_l_samples,
        "samples | 0.1M Doppler =", Float32(1000.0) * hrtf90.doppler_ratio, "Hz",
    )

    var srv = AudioServer(48000, 256)
    var qoa_aid = srv.register_qoa_asset(qoa_bytes.copy())
    _ = srv.add_cycles_stem('d1 $ s "bd*2 [~ sn]" # gain 0.85', BUS_MUSIC)
    var h3d = srv.play_3d(qoa_aid, Vec3(8.0, 0.0, -2.0), Vec3(-12.0, 0.0, 0.0), volume=0.9, bus_id=BUS_SFX, looping=True)
    _ = srv.set_emitter_occlusion(h3d, occ.occlusion_gain, occ.occlusion_cutoff_hz, occ.obstruction_factor)
    _ = srv.play_2d(qoa_aid, volume=0.8, bus_id=BUS_VOICE, looping=True)

    var quantum_out = List[Float32](length=512, fill=0.0)
    var peak_out = Float32(0.0)
    for _ in range(32):
        srv.render_offline_quantum(quantum_out, 256)
        for s in quantum_out:
            if abs(s) > peak_out:
                peak_out = abs(s)
    print(
        "[AUDIO] Rendered 32 quantums | Peak =", peak_out,
        "(<= -0.1 dBFS) | Momentary LUFS =", srv.momentary_lufs(),
        "| Music Duck Env =", srv.music_duck_env(),
    )

    # 3. 5-Panel Studio DockSpace + 100k Virtualized Outliner + UTF-8 TextInput + Curve Editor
    var dock = DockSpace.create_studio_5panel()
    var ui = UiContext()
    var inp = InputState()
    inp.move_mouse(120.0, 80.0)
    inp.push_text_utf8("¡Hola! エンジン ★")

    var ui_t0 = get_time()
    ui.begin_frame(inp, 1920.0, 1080.0)
    ui.update_dock_space(dock, RectF(0.0, 0.0, 1920.0, 1080.0))

    if ui.begin_dock_panel(dock, "Outliner"):
        var stride = ui.theme.row_height + Float32(2.0)
        ui.begin_scroll_view("scene_tree", 360.0, 640.0, Float32(100000) * stride)
        var vr = ui.scroll_visible_range(100000, stride)
        for row in range(vr[0], vr[1]):
            _ = ui.tree_node_indexed(row, "MeshEntity")
        ui.end_scroll_view()
        ui.end_dock_panel()

    var gain_val = Float32(0.85)
    var pos_val = Vec3(1.5, 2.0, -4.0)
    var hrtf_on = True
    var bez_x = Float32(0.35)
    var bez_y = Float32(0.75)
    var search_txt = String("¡Hola! エンジン ★")

    if ui.begin_dock_panel(dock, "Inspector"):
        ui.begin_box("insp_col", FlexStyle.col(gap=6.0, pad=4.0))
        _ = ui.text_input("filter", search_txt, 260.0, "Filter...")
        _ = ui.checkbox("hrtf_chk", "Enable Woodworth HRTF", hrtf_on)
        _ = ui.slider_float("bus_gain", "Master Gain", gain_val, 0.0, 1.0, 240.0)
        _ = ui.drag_float3("emit_pos", "Position", pos_val)
        _ = ui.curve_editor("attenuation_curve", bez_x, bez_y, 220.0, 110.0)
        ui.end_box()
        ui.end_dock_panel()

    ui.end_frame()
    var ui_ms = (get_time() - ui_t0) * 1000.0
    print(
        "[UI]    5-Panel DockSpace + 100k Outliner built in", ui_ms,
        "ms | GPU GpuUiPrims emitted =", ui.draw_list.count(),
    )

    # 4. Evdev DualSense Calibration + Rumble + InputMap
    var pad = EvdevGamepad()
    pad.set_vendor_product(VENDOR_SONY, 0x0CE6, "DualSense Wireless Controller")
    pad.configure_abs_axis(ABS_Z, 0, 255)
    pad.configure_abs_axis(ABS_RZ, 0, 255)
    var rumble48 = pack_ff_rumble_effect(-1, 0.8, 0.4, 500)

    var imap = InputMap()
    _ = imap.register_action("Jump", ACTION_BOOL)
    _ = imap.register_action("Save", ACTION_BOOL)
    _ = imap.register_action("Move", ACTION_AXIS2D)
    var ctx = InputContext("StudioGameplay", 0)
    ctx.bind_key("Jump", 32)
    ctx.bind_pad_button("Jump", PAD_A)
    ctx.bind_key("Save", 83, MOD_CTRL)
    imap.add_context(ctx^)
    imap.push_context("StudioGameplay")

    inp.begin_frame()
    inp.set_logical_key(341, True)
    inp.set_logical_key(83, True)
    imap.update(inp, 0.016)
    var pkt = imap.pack_deterministic_tick()
    print(
        "[INPUT] DualSense unpressed L2 =", pad.axes[PAD_AXIS_LT],
        "| ff_effect bytes =", len(rumble48),
        "| Ctrl+S active =", imap.is_down("Save"),
        "| Lockstep pkt bytes =", len(pkt),
    )

    var total_sec = get_time() - t0
    print("=== Lab Complete in", total_sec, "s ===")
