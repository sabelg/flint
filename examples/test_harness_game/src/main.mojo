"""Flint Engine v5 Standalone Playable Test-Harness Game (`examples/test_harness_game/`)."""
from std.ffi import external_call
from std.math import max, min
from std.sys import argv
from flint.ai.behavior_tree import AiPlugin, register_ai_codecs
from flint.anim.plugin import AnimPlugin, register_anim_codecs
from flint.app import (
    AppConfig,
    EngineApp,
    Resource,
    Resources,
)
from flint.app.flow import GameFlow
from flint.app.window_plugin import WindowPlugin, WindowStateResource
from flint.assets.image import write_png
from flint.assets.server import AssetServer
from flint.audio.plugin import AudioPlugin, register_audio_codecs
from flint.core.build_info import check_flint_version_pin
from flint.core.log import flush_log, start_log
from flint.core.replay import (
    ByteReader,
    CHECK_SECTIONS,
    ReplayBlob,
    TICK_BYTES,
    decode_header,
    encode_header,
    is_crash_tick,
    read_input,
    release_replay_pads,
    section_name,
    skip_dev_records,
)
from flint.core.session_replay import SessionRecorder
from flint.input.plugin import InputPlugin, InputStateResource
from flint.nav.plugin import NavPlugin, register_nav_codecs
from flint.physics.plugin import PhysicsPlugin, register_physics_codecs
from flint.platform.kms import on_virtual_console
from flint.platform.os import (
    env_flag,
    env_value,
    file_size,
    get_time,
    is_directory,
    make_dirs,
    parent_directory,
    read_bytes,
)
from flint.render.image_diff import compare_with_png
from flint.render.plugin import register_render_codecs
from flint.scene import ComponentRegistry, Scene, TransformPlugin
from gameplay_core import (
    HarnessStateResource,
    compute_sim_checksums,
    register_harness_systems,
    resolve_harness_assets,
    resolve_harness_entities,
    setup_harness_input,
)


struct GameConfig(Copyable, Movable):
    var app_name: String
    var entry_scene: String
    var flint_version: String
    var gameplay_output: String
    var window_title: String
    var window_width: Int
    var window_height: Int
    var vsync: Bool
    var fixed_hz: Int
    var net_tick_hz: Int
    var net_max_rollback_ticks: Int
    var net_mtu_bytes: Int
    var pack_output: String

    def __init__(out self):
        self.app_name = "test_harness_game"
        self.entry_scene = "assets/main.flscn"
        self.flint_version = ""
        self.gameplay_output = "build/gameplay.so"
        self.window_title = "Test Harness Game"
        self.window_width = 1280
        self.window_height = 720
        self.vsync = True
        self.fixed_hz = 60
        self.net_tick_hz = 60
        self.net_max_rollback_ticks = 8
        self.net_mtu_bytes = 1180
        self.pack_output = "build/assets.flpk"


def _trim_ascii(bytes: Span[UInt8, _], start: Int, end: Int) -> String:
    var s = start
    var e = end
    while s < e and (bytes[s] == 32 or bytes[s] == 9):
        s += 1
    while e > s and (bytes[e - 1] == 32 or bytes[e - 1] == 9 or bytes[e - 1] == 13):
        e -= 1
    var out = List[UInt8](capacity=max(0, e - s))
    for i in range(s, e):
        out.append(bytes[i])
    return String(unsafe_from_utf8=out)


def _unquote(text: String) -> String:
    var b = text.as_bytes()
    var n = len(b)
    if n >= 2 and ((b[0] == 34 and b[n - 1] == 34) or (b[0] == 39 and b[n - 1] == 39)):
        var out = List[UInt8](capacity=n - 2)
        for i in range(1, n - 1):
            out.append(b[i])
        return String(unsafe_from_utf8=out)
    return text


def _parse_int(text: String, fallback: Int) -> Int:
    var b = text.as_bytes()
    var n = len(b)
    if n == 0:
        return fallback
    var i = 0
    var sign = 1
    if b[0] == 45:
        sign = -1
        i = 1
    if i >= n:
        return fallback
    var val = 0
    while i < n:
        if b[i] < 48 or b[i] > 57:
            return fallback
        val = val * 10 + Int(b[i] - 48)
        i += 1
    return sign * val


def _load_flint_toml() -> GameConfig:
    var cfg = GameConfig()
    var raw = read_bytes("flint.toml")
    if len(raw) == 0:
        return cfg^
    var section = String("")
    var assets_pack_output = String("")
    var project_pak_path = String("")
    var assets_pak_path = String("")
    var line_start = 0
    var n = len(raw)
    for idx in range(n + 1):
        if idx < n and raw[idx] != 10:
            continue
        var line_end = idx
        var quote_char: UInt8 = 0
        var escaped = False
        for c in range(line_start, line_end):
            var ch = raw[c]
            if quote_char == 0:
                if ch == 34 or ch == 39:
                    quote_char = ch
                    escaped = False
                elif ch == 35:
                    line_end = c
                    break
            else:
                if quote_char == 34 and escaped:
                    escaped = False
                elif quote_char == 34 and ch == 92:
                    escaped = True
                elif ch == quote_char:
                    quote_char = 0
        var trimmed = _trim_ascii(raw, line_start, line_end)
        line_start = idx + 1
        var tb = trimmed.as_bytes()
        if len(tb) == 0:
            continue
        if len(tb) >= 2 and tb[0] == 91 and tb[len(tb) - 1] == 93:
            section = _trim_ascii(tb, 1, len(tb) - 1)
            continue
        var eq_pos = -1
        for c in range(len(tb)):
            if tb[c] == 61:
                eq_pos = c
                break
        if eq_pos <= 0:
            continue
        var key = _trim_ascii(tb, 0, eq_pos)
        var val = _unquote(_trim_ascii(tb, eq_pos + 1, len(tb)))
        if section == "project":
            if key == "name" and val.byte_length() > 0:
                cfg.app_name = val
            elif key == "entry_scene" and val.byte_length() > 0:
                cfg.entry_scene = val
            elif key == "gameplay_output" and val.byte_length() > 0:
                cfg.gameplay_output = val
            elif key == "pak_path" and val.byte_length() > 0 and project_pak_path.byte_length() == 0:
                project_pak_path = val
        elif section == "engine" and key == "flint_version":
            cfg.flint_version = val
        elif section == "window":
            if key == "title" and val.byte_length() > 0:
                cfg.window_title = val
            elif key == "width":
                cfg.window_width = max(64, _parse_int(val, cfg.window_width))
            elif key == "height":
                cfg.window_height = max(64, _parse_int(val, cfg.window_height))
            elif key == "vsync":
                cfg.vsync = val == "true"
        elif section == "physics" and key == "fixed_hz":
            cfg.fixed_hz = max(1, _parse_int(val, cfg.fixed_hz))
        elif section == "net":
            if key == "tick_hz":
                cfg.net_tick_hz = max(1, _parse_int(val, cfg.net_tick_hz))
            elif key == "max_rollback_ticks":
                cfg.net_max_rollback_ticks = max(0, _parse_int(val, cfg.net_max_rollback_ticks))
            elif key == "mtu_bytes":
                cfg.net_mtu_bytes = max(256, _parse_int(val, cfg.net_mtu_bytes))
        elif section == "assets":
            if key == "pack_output" and val.byte_length() > 0 and assets_pack_output.byte_length() == 0:
                assets_pack_output = val
            elif key == "pak_path" and val.byte_length() > 0 and assets_pak_path.byte_length() == 0:
                assets_pak_path = val
    if assets_pack_output.byte_length() > 0:
        cfg.pack_output = assets_pack_output
    elif project_pak_path.byte_length() > 0:
        cfg.pack_output = project_pak_path
    elif assets_pak_path.byte_length() > 0:
        cfg.pack_output = assets_pak_path
    return cfg^


def _looks_like_json_scene(raw: List[UInt8]) -> Bool:
    for i in range(len(raw)):
        var b = raw[i]
        if b == 32 or b == 9 or b == 10 or b == 13:
            continue
        return b == 123
    return False


def _create_configured_app(
    cfg: GameConfig,
    effective_headless: Bool,
) -> Tuple[Bool, EngineApp, String]:
    var assets = AssetServer(enable_watcher=False)
    if file_size(cfg.pack_output) > 0:
        _ = assets.mount_flpk(cfg.pack_output)
    elif file_size("assets.flpk") > 0:
        _ = assets.mount_flpk("assets.flpk")
    if is_directory("assets", follow_symlinks=True):
        assets.loose_root = "assets"
        assets.index_loose_directory()

    var app_cfg = AppConfig(
        title=cfg.window_title,
        headless=effective_headless,
        fixed_hz=Float32(cfg.fixed_hz),
        worker_threads=0,
        enable_crash_handler=True,
    )
    var app = EngineApp(app_cfg^)
    app.add_plugin(TransformPlugin(snapshot_prev_in_fixed=True))
    app.add_plugin(
        WindowPlugin(
            headless=effective_headless,
            width=cfg.window_width,
            height=cfg.window_height,
            title=cfg.window_title,
        )
    )
    app.add_plugin(InputPlugin(enable_evdev=not effective_headless))
    app.add_plugin(PhysicsPlugin(gravity=Float32(0.0)))
    app.add_plugin(NavPlugin(draw_debug=True))
    app.add_plugin(AiPlugin())
    app.add_plugin(AudioPlugin(headless=effective_headless))
    app.add_plugin(AnimPlugin())

    var flow = GameFlow(initial_state="Menu")
    _ = flow.fsm.allow_transition("Menu", "Play")
    _ = flow.fsm.allow_transition("Play", "Pause")
    _ = flow.fsm.allow_transition("Pause", "Play")
    _ = flow.fsm.allow_transition("Play", "Victory")
    _ = flow.fsm.transition_to("Play")
    app.insert_resource[GameFlow](flow^)

    app.insert_resource[HarnessStateResource](
        HarnessStateResource(effective_headless, cfg.window_width, cfg.window_height)
    )
    setup_harness_input(app)
    register_harness_systems(app)

    var reg = ComponentRegistry.with_builtins()
    register_render_codecs(reg)
    register_physics_codecs(reg)
    register_nav_codecs(reg)
    register_ai_codecs(reg)
    register_audio_codecs(reg)
    register_anim_codecs(reg)
    reg.ensure_registered_in_world(app.world)

    var scene_bytes = List[UInt8]()
    var rh = assets.load_raw(cfg.entry_scene)
    var rptr = assets.raws.get_ptr(rh)
    if rptr and len(rptr.value()[unsafe_offset=0].bytes) > 0:
        scene_bytes = rptr.value()[unsafe_offset=0].bytes.copy()
    elif file_size(cfg.entry_scene) > 0:
        scene_bytes = read_bytes(cfg.entry_scene)

    if len(scene_bytes) == 0:
        assets.close()
        return (False, app^, "entry_scene not found or empty: " + cfg.entry_scene)

    var scene = Scene(cfg.entry_scene)
    try:
        if cfg.entry_scene.endswith(".flscnb") or not _looks_like_json_scene(scene_bytes):
            _ = scene.load_flscnb_bytes(app.world, reg, scene_bytes, clear_existing=True)
        else:
            _ = scene.load_flscn_bytes(app.world, reg, scene_bytes, clear_existing=True)
    except err:
        assets.close()
        return (False, app^, "failed to load entry_scene '" + cfg.entry_scene + "': " + String(err))

    app.insert_resource[AssetServer](assets^)
    app.startup()

    var st_opt = app.resources.get_ptr_mut[HarnessStateResource]()
    if st_opt:
        ref st = st_opt.value()[unsafe_offset=0]
        resolve_harness_entities(app.world, st)
        resolve_harness_assets(app.world, app.resources, st)
        var plugin_path = env_value("FLINT_GAMEPLAY_PLUGIN")
        if plugin_path.byte_length() == 0:
            plugin_path = cfg.gameplay_output
        if plugin_path.byte_length() > 0 and file_size(plugin_path) > 0:
            _ = st.hot_reload_host.load_or_reload(plugin_path, app.world, app.resources)

    return (True, app^, String(""))


def _shutdown_app(mut app: EngineApp):
    var st_opt = app.resources.get_ptr_mut[HarnessStateResource]()
    if st_opt:
        if st_opt.value()[unsafe_offset=0].hot_reload_host.is_loaded():
            _ = st_opt.value()[unsafe_offset=0].hot_reload_host.unload(app.world, app.resources, is_reload=False)
    var as_opt = app.resources.get_ptr_mut[AssetServer]()
    if as_opt:
        as_opt.value()[unsafe_offset=0].close()
    app.shutdown()


def _resolve_golden_ref_path() -> String:
    if file_size("assets/golden_frame.pngraw") > 0:
        return "assets/golden_frame.pngraw"
    if file_size("golden/golden_shot.png") > 0:
        return "golden/golden_shot.png"
    return String("")


def _run_golden_check(cfg: GameConfig) -> Int:
    var built = _create_configured_app(cfg, True)
    if not built[0]:
        print("[" + cfg.app_name + "] ERROR: " + built[2])
        _shutdown_app(built[1])
        return 1
    ref app = built[1]
    var dt = Float32(1.0) / Float32(max(1, cfg.fixed_hz))
    for _ in range(4):
        app.step_one_frame(dt)

    var ref_path = _resolve_golden_ref_path()
    if ref_path.byte_length() == 0:
        print("[" + cfg.app_name + "] ERROR: golden reference missing (expected assets/golden_frame.pngraw or golden/golden_shot.png)")
        _shutdown_app(app)
        return 1

    var st_opt = app.resources.get_ptr[HarnessStateResource]()
    if not st_opt:
        print("[" + cfg.app_name + "] ERROR: missing HarnessStateResource")
        _shutdown_app(app)
        return 1
    ref st = st_opt.value()[unsafe_offset=0]
    var diff = compare_with_png(st.canvas.pixels, st.canvas.width, st.canvas.height, ref_path, 4)
    if not diff.passes(1000, Float64(0.02)):
        print("[" + cfg.app_name + "] FAIL: Golden FLIP check mismatch (" + diff.summary() + ") ref=" + ref_path)
        _shutdown_app(app)
        return 1

    print("[" + cfg.app_name + "] [PASS] Golden FLIP check OK (" + diff.summary() + ", tris=" + String(st.canvas.triangles_drawn) + ", ref=" + ref_path + ")")
    _shutdown_app(app)
    return 0


def _run_record_golden(cfg: GameConfig) -> Int:
    var built = _create_configured_app(cfg, True)
    if not built[0]:
        print("[" + cfg.app_name + "] ERROR: " + built[2])
        _shutdown_app(built[1])
        return 1
    ref app = built[1]
    var dt = Float32(1.0) / Float32(max(1, cfg.fixed_hz))
    for _ in range(4):
        app.step_one_frame(dt)

    var st_opt = app.resources.get_ptr_mut[HarnessStateResource]()
    if not st_opt:
        _shutdown_app(app)
        return 1
    ref st = st_opt.value()[unsafe_offset=0]
    _ = make_dirs("assets")
    _ = make_dirs("golden")
    var ok1 = write_png("assets/golden_frame.pngraw", st.canvas.pixels, st.canvas.width, st.canvas.height, stamped=False)
    var ok2 = write_png("golden/golden_shot.png", st.canvas.pixels, st.canvas.width, st.canvas.height, stamped=False)
    _shutdown_app(app)
    if not ok1 or not ok2:
        print("[" + cfg.app_name + "] ERROR: failed writing golden PNG files")
        return 1
    print("[" + cfg.app_name + "] Recorded golden frames -> assets/golden_frame.pngraw & golden/golden_shot.png (tris=" + String(st.canvas.triangles_drawn) + ")")
    return 0


def _run_record_replay(cfg: GameConfig) -> Int:
    var built = _create_configured_app(cfg, True)
    if not built[0]:
        print("[" + cfg.app_name + "] ERROR: " + built[2])
        _shutdown_app(built[1])
        return 1
    ref app = built[1]
    var dt = Float32(1.0) / Float32(max(1, cfg.fixed_hz))
    _ = make_dirs("assets")
    var rec = SessionRecorder()
    var hdr = encode_header(0, 0, List[ReplayBlob]())
    if not rec.start("assets/test_input.sdrp", hdr^, for_crashes=False):
        print("[" + cfg.app_name + "] ERROR: failed to open assets/test_input.sdrp for recording")
        _shutdown_app(app)
        return 1

    # Deterministic 130-tick playthrough:
    # - Ticks 0..49: hold D (68) -> move +X across Pickup trigger
    # - Ticks 50..65: attack Enemy twice with Space (32) on ticks 52 and 60
    # - Ticks 66..129: hold D (68) -> move +X into Goal trigger -> GameFlow "Victory"
    for tick in range(130):
        var in_opt = app.resources.get_ptr_mut[InputStateResource]()
        if in_opt:
            ref inp = in_opt.value()[unsafe_offset=0].input
            inp.set_logical_key(68, tick < 50 or tick >= 66)
            inp.set_logical_key(32, tick == 52 or tick == 60)
            rec.begin_tick(inp)
        app.step_one_frame(dt)
        var st_opt = app.resources.get_ptr[HarnessStateResource]()
        if st_opt:
            var checks = compute_sim_checksums(app.world, app.resources, st_opt.value()[unsafe_offset=0])
            rec.end_tick(checks)

    rec.finish()

    var st_fin = app.resources.get_ptr[HarnessStateResource]()
    var flow_opt = app.resources.get_ptr[GameFlow]()
    var score = st_fin.value()[unsafe_offset=0].score if st_fin else 0
    var pickup_ok = st_fin.value()[unsafe_offset=0].pickup_collected if st_fin else False
    var goal_ok = st_fin.value()[unsafe_offset=0].goal_reached if st_fin else False
    var nav_ok = st_fin.value()[unsafe_offset=0].nav_path_verified if st_fin else False
    var enemy_hp = st_fin.value()[unsafe_offset=0].enemy_hp if st_fin else -1
    var flow_st = flow_opt.value()[unsafe_offset=0].fsm.current_state() if flow_opt else String("")
    _shutdown_app(app)

    if score <= 0 or not pickup_ok or not goal_ok or not nav_ok or flow_st != "Victory":
        print(
            "["
            + cfg.app_name
            + "] ERROR: recorded replay invariants failed (score="
            + String(score)
            + ", pickup="
            + String(pickup_ok)
            + ", goal="
            + String(goal_ok)
            + ", enemy_hp="
            + String(enemy_hp)
            + ", nav="
            + String(nav_ok)
            + ", flow="
            + flow_st
            + ")"
        )
        return 1
    print(
        "["
        + cfg.app_name
        + "] Recorded SDRP replay -> assets/test_input.sdrp (130 ticks, score="
        + String(score)
        + ", enemy_hp="
        + String(enemy_hp)
        + ", state="
        + flow_st
        + ")"
    )
    return 0


def _run_replay_verify(cfg: GameConfig) -> Int:
    var sdrp_path = String("assets/test_input.sdrp")
    var raw = read_bytes(sdrp_path)
    if len(raw) == 0:
        print("[" + cfg.app_name + "] ERROR: replay file missing or empty: " + sdrp_path)
        return 1
    var reader = ByteReader(raw^)
    var header = decode_header(reader)
    if not header.valid:
        print("[" + cfg.app_name + "] ERROR: invalid SDRP header in " + sdrp_path)
        return 1

    var built = _create_configured_app(cfg, True)
    if not built[0]:
        print("[" + cfg.app_name + "] ERROR: " + built[2])
        _shutdown_app(built[1])
        return 1
    ref app = built[1]
    var dt = Float32(1.0) / Float32(max(1, cfg.fixed_hz))
    var ticks_replayed = 0

    while reader.remaining() >= TICK_BYTES:
        _ = skip_dev_records(reader)
        if reader.remaining() < TICK_BYTES:
            break
        var in_opt = app.resources.get_ptr_mut[InputStateResource]()
        if in_opt:
            read_input(reader, in_opt.value()[unsafe_offset=0].input)
        var expected = List[UInt64](capacity=CHECK_SECTIONS)
        for _ in range(CHECK_SECTIONS):
            expected.append(reader.u64())
        if reader.failed:
            print("[" + cfg.app_name + "] ERROR: truncated SDRP record at tick " + String(ticks_replayed))
            _shutdown_app(app)
            return 1
        if is_crash_tick(expected):
            print("[" + cfg.app_name + "] ERROR: crash tick encountered at tick " + String(ticks_replayed))
            _shutdown_app(app)
            return 1

        app.step_one_frame(dt)
        var st_opt = app.resources.get_ptr[HarnessStateResource]()
        if not st_opt:
            _shutdown_app(app)
            return 1
        var actual = compute_sim_checksums(app.world, app.resources, st_opt.value()[unsafe_offset=0])
        for s in range(CHECK_SECTIONS):
            if actual[s] != expected[s]:
                print(
                    "["
                    + cfg.app_name
                    + "] FAIL: REPLAY_CHECKSUM_MISMATCH tick="
                    + String(ticks_replayed)
                    + " section="
                    + section_name(s)
                    + " expected="
                    + String(expected[s])
                    + " actual="
                    + String(actual[s])
                )
                _shutdown_app(app)
                return 1
        ticks_replayed += 1

    _ = skip_dev_records(reader)
    if reader.failed or reader.remaining() != 0:
        print(
            "["
            + cfg.app_name
            + "] ERROR: truncated SDRP record at tick "
            + String(ticks_replayed)
            + " (trailing_bytes="
            + String(reader.remaining())
            + ")"
        )
        _shutdown_app(app)
        return 1

    var in_end = app.resources.get_ptr_mut[InputStateResource]()
    if in_end:
        release_replay_pads(in_end.value()[unsafe_offset=0].input)

    var st_fin = app.resources.get_ptr[HarnessStateResource]()
    var flow_opt = app.resources.get_ptr[GameFlow]()
    var score = st_fin.value()[unsafe_offset=0].score if st_fin else 0
    var pickup_ok = st_fin.value()[unsafe_offset=0].pickup_collected if st_fin else False
    var goal_ok = st_fin.value()[unsafe_offset=0].goal_reached if st_fin else False
    var nav_ok = st_fin.value()[unsafe_offset=0].nav_path_verified if st_fin else False
    var flow_st = flow_opt.value()[unsafe_offset=0].fsm.current_state() if flow_opt else String("")
    _shutdown_app(app)

    if ticks_replayed == 0 or score <= 0 or not pickup_ok or not goal_ok or not nav_ok or flow_st != "Victory":
        print(
            "["
            + cfg.app_name
            + "] FAIL: Replay outcome check failed (ticks="
            + String(ticks_replayed)
            + ", score="
            + String(score)
            + ", pickup="
            + String(pickup_ok)
            + ", goal="
            + String(goal_ok)
            + ", nav="
            + String(nav_ok)
            + ", state="
            + flow_st
            + ")"
        )
        return 1

    print(
        "["
        + cfg.app_name
        + "] [PASS] Replay SDRP check OK ("
        + String(ticks_replayed)
        + " ticks, score="
        + String(score)
        + ", state="
        + flow_st
        + ")"
    )
    return 0


def _run_game() -> Int:
    var cfg = _load_flint_toml()
    var ver_check = check_flint_version_pin(cfg.flint_version)
    if not ver_check[0]:
        print("[" + cfg.app_name + "] ERROR: " + ver_check[1])
        return 1

    _ = start_log(cfg.app_name)
    var headless_smoke = False
    var shots_mode = False
    var verify_golden = False
    var verify_replay = False
    var test_harness = False
    var record_golden = False
    var record_replay = False
    var out_png = String("")

    var args = argv()
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--smoke" or a == "--headless-smoke":
            headless_smoke = True
        elif a == "--shots":
            shots_mode = True
        elif a == "--verify-golden" or a == "--golden-check":
            verify_golden = True
        elif a == "--replay" or a == "--input-replay" or a == "--verify-replay":
            verify_replay = True
        elif a == "--test-harness":
            test_harness = True
        elif a == "--record-golden":
            record_golden = True
        elif a == "--record-replay":
            record_replay = True
        elif a == "--out" and ai + 1 < len(args):
            out_png = String(args[ai + 1])
            ai += 1
        elif a.startswith("--out="):
            out_png = String(a[byte=6:])
        ai += 1

    if record_golden:
        var rc = _run_record_golden(cfg)
        _ = flush_log()
        return rc
    if record_replay:
        var rc = _run_record_replay(cfg)
        _ = flush_log()
        return rc
    if test_harness:
        var rc_g = _run_golden_check(cfg)
        if rc_g != 0:
            _ = flush_log()
            return rc_g
        var rc_r = _run_replay_verify(cfg)
        _ = flush_log()
        return rc_r
    if verify_golden:
        var rc = _run_golden_check(cfg)
        _ = flush_log()
        return rc
    if verify_replay:
        var rc = _run_replay_verify(cfg)
        _ = flush_log()
        return rc

    var no_display = (
        env_flag("FLINT_HEADLESS")
        or (
            env_value("FLINT_BACKEND").byte_length() == 0
            and env_value("WAYLAND_DISPLAY").byte_length() == 0
            and env_value("DISPLAY").byte_length() == 0
            and not on_virtual_console()
        )
    )
    var effective_headless = headless_smoke or shots_mode or no_display

    var built = _create_configured_app(cfg, effective_headless)
    if not built[0]:
        print("[" + cfg.app_name + "] ERROR: " + built[2])
        _shutdown_app(built[1])
        _ = flush_log()
        return 1
    ref app = built[1]

    if headless_smoke or shots_mode or no_display:
        var dt = Float32(1.0) / Float32(max(1, cfg.fixed_hz))
        for _ in range(4):
            app.step_one_frame(dt)

        var shot_target = out_png
        if shots_mode and shot_target.byte_length() == 0:
            shot_target = "build/shots/" + cfg.app_name + ".png"
        if shot_target.byte_length() > 0:
            var pdir = parent_directory(shot_target)
            if pdir.byte_length() > 0 and pdir != ".":
                _ = make_dirs(pdir)
            var st_opt = app.resources.get_ptr_mut[HarnessStateResource]()
            if st_opt:
                ref st = st_opt.value()[unsafe_offset=0]
                if not write_png(shot_target, st.canvas.pixels, st.canvas.width, st.canvas.height, stamped=False):
                    print("[" + cfg.app_name + "] ERROR: failed writing PNG -> " + shot_target)
                    _shutdown_app(app)
                    _ = flush_log()
                    return 1

        print(
            "["
            + cfg.app_name
            + "] Headless v5 frame loop OK ("
            + String(cfg.window_width)
            + "x"
            + String(cfg.window_height)
            + ", fixed_hz="
            + String(cfg.fixed_hz)
            + ", entities="
            + String(len(app.world.entities))
            + ")."
        )
        _shutdown_app(app)
        _ = flush_log()
        return 0

    var win_open = False
    var win_opt = app.resources.get_ptr[WindowStateResource]()
    if win_opt:
        win_open = win_opt.value()[unsafe_offset=0].is_open()
    if not win_open:
        print("[" + cfg.app_name + "] No display available; exiting cleanly. Use '--headless-smoke' or '--shots'.")
        _shutdown_app(app)
        _ = flush_log()
        return 0

    var last_t = get_time()
    while True:
        var open_now = False
        var wo = app.resources.get_ptr[WindowStateResource]()
        if wo:
            open_now = wo.value()[unsafe_offset=0].is_open()
        if not open_now:
            break
        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - last_t)))
        last_t = now
        app.step_one_frame(dt)

    _shutdown_app(app)
    _ = flush_log()
    return 0


def main():
    var rc = _run_game()
    if rc != 0:
        external_call["exit", NoneType](Int32(rc))
