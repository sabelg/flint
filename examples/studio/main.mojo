"""Flint v5 Visual Editor (`Flint Studio`, `examples/studio/main.mojo`).

Standalone AAA Visual Scene Editor, Procedural Kit, Code-to-Scene Workspace & Visual AI Agent Host
powered by the unified Engine v5 spine (`EngineApp`, `WindowPlugin`, `TransformPlugin`, `InputPlugin`,
`PhysicsPlugin`, `RenderPlugin`, `flint.ecs.World`, `flint.scene.Scene`, `PhysicsWorld`,
`ScalableRenderer`):
  - **Interactive Window Mode**:
    - Non-stalling `FRAMES_IN_FLIGHT` async 40-bit `VisBuffer` viewport entity picking (`VisBufferGpuPicker`)
    - Multi-select (`Ctrl`/`Shift`), centroid analytical 3D gizmos (`W` / `E` / `R`, grid/15° snap)
    - `DockSpace` v2 + `StudioHeaderBar` + Floating `ViewportHud` + `CommandPalette` (`Ctrl+K`)
    - Native GPU `CodeDrawer` (Mojo syntax highlight, unified/split diffs, Problems, 1-click Build & Hot-Reload)
    - Visual Agent Studio (`AgentMissionControlPanel` + 3D `AgentTheatre` + live ACP/MCP v2 bridge)
    - Reversible coalesced `EditorTransaction` Undo/Redo (`Ctrl+Z` / `Ctrl+Y`) & `FlintMcpUnixListener`
  - **`--headless-smoke` / `--shots` (`--out <png>`)**:
    Renders a deterministic `1280x720` Studio editor frame (`UI_ANIM_INSTANT`, `ui_time_sec = 0.0`).
  - **`--gallery <dir>`**:
    Renders all 6 deterministic `1280x720` visual showcase shots (`01_edit_default.png` .. `06_play_hud.png`).
  - **`--mcp-stdio`**:
    Runs `McpServerV2` in spec-compliant Model Context Protocol (`2024-11-05`) JSON-RPC 2.0 `stdio` mode.
  - **`--mcp-connect <sock_path>`**:
    Pure-Mojo bidirectional `stdio <-> AF_UNIX` MCP proxy for live Studio Unix socket sessions.
"""
from std.collections import List
from std.ffi import external_call
from std.math import cos, max, min, sin
from std.sys import argv
from flint.app.engine_app import STAGE_FIRST
from flint.app.window_plugin import WindowStateResource
from flint.assets.asset_plugin import SceneAssetResolver
from flint.assets.image import write_png
from flint.assets.server import AssetServer
from flint.core.build_info import check_flint_version_pin, parse_flint_version_from_toml_bytes
from flint.core.jobs import JobPool, default_worker_count
from flint.core.math import Quat, Vec3
from flint.devtools.mcp_v2 import (
    MCP_GATE_ALL,
    MCP_GATE_AUTO,
    MCP_GATE_MUTATING,
    FlintMcpUnixListener,
    McpServerV2,
    McpToolEvent,
    default_mcp_socket_path,
    run_mcp_connect_stdio,
)
from flint.input.plugin import InputStateResource
from flint.platform.input import InputState
from flint.platform.os import (
    env_value,
    file_size,
    get_time,
    is_directory,
    make_dirs,
    parent_directory,
    read_bytes,
    real_path,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    ScalableRenderer,
)
from flint.studio import (
    APPROVAL_AUTO,
    APPROVAL_EDITS,
    APPROVAL_STEP,
    GIZMO_AXIS_NONE,
    GIZMO_AXIS_X,
    GIZMO_AXIS_Y,
    GIZMO_AXIS_Z,
    GIZMO_MODE_ROTATE,
    GIZMO_MODE_SCALE,
    GIZMO_MODE_TRANSLATE,
    GIZMO_PLANE_XY,
    GIZMO_PLANE_XZ,
    GIZMO_PLANE_YZ,
    PIE_PLAYING,
    PIE_STOPPED,
    WORKSPACE_AGENT_STUDIO,
    WORKSPACE_CODE_SCENE,
    WORKSPACE_EDIT,
    WORKSPACE_LOOKDEV,
    WORKSPACE_PLAY,
    BridgedMcpEvent,
    FlintStudioState,
    SelectedTransformSnap,
    ViewportRay,
    VisBufferGpuPicker,
    fill_instance_entity_maps,
    gizmo_rotation_delta_rad,
    gizmo_scale_drag,
    gizmo_world_scale,
    pick_gizmo_handle,
    ray_axis_closest_param,
    ray_plane_hit,
    snap_angle_deg,
    snap_translation,
)
from flint.studio.studio_app import (
    populate_default_studio_scene,
    upload_default_studio_gpu_assets,
)
from flint.ui.animator import UI_ANIM_INSTANT, UI_ANIM_NORMAL, UI_ANIM_REDUCED
from flint.ui.theme import (
    THEME_FLINT_OBSIDIAN,
    THEME_MIDNIGHT_STUDIO,
    THEME_OKABE_ITO,
    THEME_PAPER_LIGHT,
)


def _fill_instance_entity_maps(
    studio: FlintStudioState,
    mut out_inst_to_entity: List[Int],
    mut out_inst_to_bits: List[UInt64],
):
    fill_instance_entity_maps(studio, out_inst_to_entity, out_inst_to_bits)


def _parse_workspace_arg(raw: String) -> Int:
    var s = String(raw.strip()).lower()
    if s == "agent" or s == "agent_studio" or s == "agents":
        return WORKSPACE_AGENT_STUDIO
    if s == "code" or s == "code_scene" or s == "code+scene":
        return WORKSPACE_CODE_SCENE
    if s == "lookdev" or s == "look":
        return WORKSPACE_LOOKDEV
    if s == "play":
        return WORKSPACE_PLAY
    if s == "edit" or s == "default":
        return WORKSPACE_EDIT
    return -1


def _parse_theme_arg(raw: String) -> Int:
    var s = String(raw.strip()).lower()
    if s == "obsidian" or s == "flint_obsidian" or s == "dark":
        return THEME_FLINT_OBSIDIAN
    if s == "midnight" or s == "midnight_studio":
        return THEME_MIDNIGHT_STUDIO
    if s == "okabe" or s == "okabe_ito" or s == "cvd":
        return THEME_OKABE_ITO
    if s == "paper" or s == "paper_light" or s == "light":
        return THEME_PAPER_LIGHT
    return -1


def _parse_anim_mode_arg(raw: String) -> Int:
    var s = String(raw.strip()).lower()
    if s == "normal":
        return UI_ANIM_NORMAL
    if s == "reduced":
        return UI_ANIM_REDUCED
    if s == "instant":
        return UI_ANIM_INSTANT
    return -1


def _sync_mcp_gate_from_bus(mut mcp: McpServerV2, studio: FlintStudioState):
    """Aligns `McpServerV2` approval gate mode with the selected `AgentTelemetryBus` session policy."""
    var si = studio.agent_bus.find_session_index(studio.agent_bus.selected_agent_id)
    if si < 0 or si >= len(studio.agent_bus.sessions):
        mcp.set_gate_mode(MCP_GATE_AUTO)
        return
    var pol = studio.agent_bus.sessions[si].approval_policy
    if pol == APPROVAL_STEP:
        mcp.set_gate_mode(MCP_GATE_ALL)
    elif pol == APPROVAL_EDITS:
        mcp.set_gate_mode(MCP_GATE_MUTATING)
    else:
        mcp.set_gate_mode(MCP_GATE_AUTO)


def _drain_mcp_events_to_studio(mut mcp: McpServerV2, mut studio: FlintStudioState):
    """Bridges live `McpToolEvent`s from `McpServerV2` into `FlintStudioState`'s Visual Agent Theatre."""
    if mcp.peek_tool_event_count() <= 0:
        return
    var raw = mcp.take_tool_events()
    var bridged = List[BridgedMcpEvent](capacity=len(raw))
    for i in range(len(raw)):
        ref r = raw[i]
        var b = BridgedMcpEvent(
            seq=r.seq,
            tool_name=r.tool_name,
            params_json=r.params_json,
            result_json=r.result_json,
            elapsed_ms=r.elapsed_ms,
            ok=r.ok,
            is_mutating=r.is_mutating,
            blocked_by_gate=r.blocked_by_gate,
            agent_id=studio.agent_bus.selected_agent_id,
        )
        b.affected_entities = r.affected_entities.copy()
        bridged.append(b^)
    _ = studio.ingest_mcp_events(bridged^)


def _render_deterministic_png(
    mut studio: FlintStudioState,
    assets: AssetServer,
    out_w: Int,
    out_h: Int,
    target_png: String,
    pinned_ui_time: Float32,
) -> Tuple[Bool, Int, Int]:
    """Renders 4 deterministic TSR-converged headless subframes and writes `target_png`."""
    var pdir = parent_directory(target_png)
    if pdir.byte_length() > 0 and pdir != ".":
        _ = make_dirs(pdir)
    var empty_input = InputState()
    studio.set_anim_mode(UI_ANIM_INSTANT)
    studio.ui_time_sec = pinned_ui_time
    for sub in range(4):
        studio.ui_time_sec = pinned_ui_time
        studio.update_camera(Float32(0.0))
        var rend_opt = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if rend_opt and rend_opt.value()[unsafe_offset=0].uses_vulkan():
            rend_opt.value()[unsafe_offset=0].gpu_slot[0].tsr_history_weight = (
                Float32(0.0) if sub == 0 else Float32(0.78)
            )
            rend_opt.value()[unsafe_offset=0].ui_time_sec = pinned_ui_time
        var gms = 4.2
        studio.render_editor_frame(
            out_w,
            out_h,
            gpu_ms=gms,
            mounted_packs=len(assets.mounts),
            mcp_clients=0,
            input=empty_input,
            dt=Float32(1.0 / 60.0),
        )

    var rend_fin = studio.app.resources.get_ptr_mut[ScalableRenderer]()
    var px = List[UInt32]()
    var culled_draws = 0
    if rend_fin and rend_fin.value()[unsafe_offset=0].uses_vulkan():
        ref r = rend_fin.value()[unsafe_offset=0]
        px = r.readback_pixels()
        culled_draws = r.gpu_slot[0].last_culled_meshlet_draws
    var wrote = write_png(target_png, px, out_w, out_h)
    return (wrote, len(studio.ui_ctx.draw_list.prims), culled_draws)


def _run_studio() raises -> Int:
    var mcp_stdio = False
    var mcp_connect_mode = False
    var mcp_connect_sock = String("")
    var headless_smoke = False
    var shots_mode = False
    var gallery_mode = False
    var gallery_dir = String("")
    var out_png = String("")
    var has_out_flag = False
    var project_arg = String("")
    var has_project_flag = False
    var workspace_opt = -1
    var theme_opt = -1
    var anim_mode_opt = -1

    var args = argv()
    var exe_path = String(args[0]) if len(args) > 0 else String("flint-studio")
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--mcp-stdio":
            mcp_stdio = True
        elif a == "--mcp-connect":
            if ai + 1 >= len(args):
                print("[Flint Studio] ERROR: --mcp-connect requires a socket path argument.")
                return 1
            ai += 1
            mcp_connect_sock = String(args[ai])
            mcp_connect_mode = True
        elif a.startswith("--mcp-connect="):
            mcp_connect_sock = String(a[byte=14:])
            mcp_connect_mode = True
        elif a == "--headless-smoke":
            headless_smoke = True
        elif a == "--shots":
            shots_mode = True
        elif a == "--gallery":
            if ai + 1 >= len(args):
                print("[Flint Studio] ERROR: --gallery requires an output directory argument.")
                return 1
            ai += 1
            gallery_dir = String(args[ai])
            gallery_mode = True
        elif a.startswith("--gallery="):
            gallery_dir = String(a[byte=10:])
            gallery_mode = True
        elif a == "--out":
            if ai + 1 >= len(args):
                print("[Flint Studio] ERROR: --out requires a file path argument.")
                return 1
            ai += 1
            out_png = String(args[ai])
            has_out_flag = True
        elif a.startswith("--out="):
            out_png = String(a[byte=6:])
            has_out_flag = True
        elif a == "--project":
            if ai + 1 >= len(args):
                print("[Flint Studio] ERROR: --project requires a directory argument.")
                return 1
            ai += 1
            project_arg = String(args[ai])
            has_project_flag = True
        elif a.startswith("--project="):
            project_arg = String(a[byte=10:])
            has_project_flag = True
        elif a == "--workspace":
            if ai + 1 >= len(args):
                print("[Flint Studio] ERROR: --workspace requires one of: edit, agent, code, lookdev, play.")
                return 1
            ai += 1
            workspace_opt = _parse_workspace_arg(String(args[ai]))
            if workspace_opt < 0:
                print("[Flint Studio] ERROR: unknown --workspace value: " + String(args[ai]))
                return 1
        elif a.startswith("--workspace="):
            var wraw = String(a[byte=12:])
            workspace_opt = _parse_workspace_arg(wraw)
            if workspace_opt < 0:
                print("[Flint Studio] ERROR: unknown --workspace value: " + wraw)
                return 1
        elif a == "--theme":
            if ai + 1 >= len(args):
                print("[Flint Studio] ERROR: --theme requires one of: obsidian, midnight, okabe, paper.")
                return 1
            ai += 1
            theme_opt = _parse_theme_arg(String(args[ai]))
            if theme_opt < 0:
                print("[Flint Studio] ERROR: unknown --theme value: " + String(args[ai]))
                return 1
        elif a.startswith("--theme="):
            var traw = String(a[byte=8:])
            theme_opt = _parse_theme_arg(traw)
            if theme_opt < 0:
                print("[Flint Studio] ERROR: unknown --theme value: " + traw)
                return 1
        elif a == "--anim-mode":
            if ai + 1 >= len(args):
                print("[Flint Studio] ERROR: --anim-mode requires one of: normal, reduced, instant.")
                return 1
            ai += 1
            anim_mode_opt = _parse_anim_mode_arg(String(args[ai]))
            if anim_mode_opt < 0:
                print("[Flint Studio] ERROR: unknown --anim-mode value: " + String(args[ai]))
                return 1
        elif a.startswith("--anim-mode="):
            var mraw = String(a[byte=12:])
            anim_mode_opt = _parse_anim_mode_arg(mraw)
            if anim_mode_opt < 0:
                print("[Flint Studio] ERROR: unknown --anim-mode value: " + mraw)
                return 1
        ai += 1

    # Pure-Mojo stdio <-> AF_UNIX proxy exits immediately with zero GPU/Window initialization
    if mcp_connect_mode:
        return run_mcp_connect_stdio(mcp_connect_sock)

    if has_out_flag:
        if out_png.byte_length() == 0 or is_directory(out_png, follow_symlinks=True):
            print("[Flint Studio] ERROR: --out path must be a file path, not empty or an existing directory: " + out_png)
            return 1

    if gallery_mode and gallery_dir.byte_length() == 0:
        print("[Flint Studio] ERROR: --gallery directory path must be non-empty.")
        return 1

    var has_project_env = False
    if not has_project_flag:
        var env_proj = env_value("FLINT_STUDIO_PROJECT")
        if env_proj.byte_length() > 0:
            project_arg = env_proj
            has_project_env = True

    var project_root = String(".")
    if has_project_flag or has_project_env:
        if project_arg.byte_length() == 0 or not is_directory(project_arg, follow_symlinks=True):
            print("[Flint Studio] ERROR: invalid --project directory: " + project_arg)
            return 1
        var rp = real_path(project_arg)
        if not rp[0] or not is_directory(rp[1], follow_symlinks=True):
            print("[Flint Studio] ERROR: invalid --project directory: " + project_arg)
            return 1
        var clean_p = rp[1]
        while clean_p.byte_length() > 1 and clean_p.as_bytes()[clean_p.byte_length() - 1] == 47:
            var trimmed = String(clean_p[byte=0 : clean_p.byte_length() - 1])
            clean_p = trimmed^
        project_root = clean_p
        if file_size(project_root + "/flint.toml") > 0:
            var toml_bytes = read_bytes(project_root + "/flint.toml")
            var ver_pin = parse_flint_version_from_toml_bytes(toml_bytes)
            var ver_check = check_flint_version_pin(ver_pin)
            if not ver_check[0]:
                print("[Flint Studio] ERROR: " + ver_check[1])
                return 1

    var out_w = 1280
    var out_h = 720
    var is_headless = mcp_stdio or headless_smoke or shots_mode or gallery_mode
    var render_tier = RENDER_TIER_0_HEADLESS_SIMD if mcp_stdio else RENDER_TIER_2_AAA_VISBUFFER_RT
    var proj_arg = project_root if (has_project_flag or has_project_env) else ""
    var studio = FlintStudioState(
        headless=is_headless,
        render_tier=render_tier,
        width=out_w,
        height=out_h,
        worker_threads=max(1, default_worker_count()),
        project_root=proj_arg,
    )
    if has_project_flag or has_project_env:
        studio.asset_browser_dir = project_root
    populate_default_studio_scene(studio)

    if theme_opt >= 0:
        studio.set_studio_theme(theme_opt)
    if anim_mode_opt >= 0:
        studio.set_anim_mode(anim_mode_opt)
    if workspace_opt >= 0:
        _ = studio.apply_workspace_preset(workspace_opt, Float32(out_w), Float32(out_h))

    var assets_ptr = studio.app.resources.get_ptr_mut[AssetServer]()
    if not assets_ptr:
        print("[Flint Studio] ERROR: AssetServer resource missing.")
        studio.shutdown()
        return 1
    ref assets = assets_ptr.value()[unsafe_offset=0]
    var mcp = McpServerV2(workspace_root=project_root)

    if mcp_stdio:
        var mcp_pool = JobPool()
        _ = mcp_pool.start(max(1, default_worker_count()))
        _ = mcp.serve_stdio_loop(studio, assets, mcp_pool)
        mcp_pool.stop()
        studio.shutdown()
        return 0

    var ren_check = studio.app.resources.get_ptr[ScalableRenderer]()
    if not ren_check or not ren_check.value()[unsafe_offset=0].uses_vulkan():
        print("[Flint Studio] ERROR: Vulkan 1.3 driver unavailable.")
        studio.shutdown()
        return 0

    upload_default_studio_gpu_assets(studio)
    var picker = VisBufferGpuPicker()
    var inst_to_ent = List[Int]()
    var inst_to_bits = List[UInt64]()

    if gallery_mode:
        var clean_dir = gallery_dir
        while clean_dir.byte_length() > 1 and clean_dir.as_bytes()[clean_dir.byte_length() - 1] == 47:
            var trimmed = String(clean_dir[byte=0 : clean_dir.byte_length() - 1])
            clean_dir = trimmed^
        _ = make_dirs(clean_dir)
        var shot_names = List[String](capacity=6)
        shot_names.append("01_edit_default.png")
        shot_names.append("02_command_palette.png")
        shot_names.append("03_agent_theatre.png")
        shot_names.append("04_code_scene_split.png")
        shot_names.append("05_pie_running.png")
        shot_names.append("06_play_hud.png")

        for si in range(6):
            studio.configure_gallery_shot(si, out_w, out_h)
            if theme_opt >= 0:
                studio.set_studio_theme(theme_opt)
            var pinned_t = studio.ui_time_sec
            var out_path = clean_dir + "/" + shot_names[si]
            var res = _render_deterministic_png(studio, assets, out_w, out_h, out_path, pinned_t)
            if not res[0]:
                print("[Flint Studio] ERROR: Failed to write gallery PNG -> " + out_path)
                var rf = studio.app.resources.get_ptr_mut[ScalableRenderer]()
                if rf and rf.value()[unsafe_offset=0].uses_vulkan():
                    picker.shutdown(rf.value()[unsafe_offset=0].gpu_slot[0])
                studio.shutdown()
                return 1
            print(
                "[Flint Studio] Gallery ["
                + String(si + 1)
                + "/6] -> "
                + out_path
                + " (UIPrims="
                + String(res[1])
                + ", CulledDraws="
                + String(res[2])
                + ")"
            )

        var rf_done = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if rf_done and rf_done.value()[unsafe_offset=0].uses_vulkan():
            picker.shutdown(rf_done.value()[unsafe_offset=0].gpu_slot[0])
        # Restore canonical Edit layout before shutdown so --gallery never leaves Play saved on disk
        studio.workspace_mgr.reset_preset(WORKSPACE_EDIT)
        _ = studio.apply_workspace_preset(WORKSPACE_EDIT, Float32(out_w), Float32(out_h))
        studio.shutdown()
        return 0

    if headless_smoke or shots_mode:
        var target_png = out_png
        if target_png.byte_length() == 0:
            if headless_smoke:
                target_png = "/tmp/flint_studio_smoke.png"
            elif has_project_flag or has_project_env:
                target_png = project_root + "/build/shots/flint_studio.png"
            else:
                target_png = "build/shots/flint_studio.png"

        # Enforce canonical deterministic steady-state frame so `--shots` and `--headless-smoke` match
        if workspace_opt < 0:
            studio.workspace_mgr.reset_preset(WORKSPACE_EDIT)
            _ = studio.apply_workspace_preset(WORKSPACE_EDIT, Float32(out_w), Float32(out_h))
        studio.set_anim_mode(UI_ANIM_INSTANT)
        studio.ui_time_sec = Float32(0.0)
        var res = _render_deterministic_png(studio, assets, out_w, out_h, target_png, Float32(0.0))

        var rend_fin = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if rend_fin and rend_fin.value()[unsafe_offset=0].uses_vulkan():
            picker.shutdown(rend_fin.value()[unsafe_offset=0].gpu_slot[0])
        if not res[0]:
            print("[Flint Studio] ERROR: Failed to write Studio frame PNG -> " + target_png)
            studio.shutdown()
            return 1

        print(
            "[Flint Studio] Saved 1280x720 Studio frame -> "
            + target_png
            + " (UIPrims=" + String(res[1])
            + ", Entities=" + String(studio.live_entity_count())
            + ", CulledDraws=" + String(res[2]) + ")"
        )
        studio.shutdown()
        return 0

    # Interactive Window Mode via `WindowPlugin` + `ScalableRenderer`
    var win_ok = False
    var win_opt = studio.app.resources.get_ptr_mut[WindowStateResource]()
    if win_opt and win_opt.value()[unsafe_offset=0].initialized and len(win_opt.value()[unsafe_offset=0].windows) > 0:
        var rend_i = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if rend_i:
            _ = rend_i.value()[unsafe_offset=0].attach_window_surface(win_opt.value()[unsafe_offset=0].windows[0])
            if rend_i.value()[unsafe_offset=0].uses_vulkan():
                ref gpu = rend_i.value()[unsafe_offset=0].gpu_slot[0]
                if gpu.vk.swapchain == 0:
                    gpu.deferred_readback = True
            win_ok = True

    if not win_ok:
        print("[Flint Studio] No Wayland/X11/KMS display detected. Use '--headless-smoke', '--shots', or '--gallery'.")
        var rend_s = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if rend_s and rend_s.value()[unsafe_offset=0].uses_vulkan():
            picker.shutdown(rend_s.value()[unsafe_offset=0].gpu_slot[0])
        studio.shutdown()
        return 0

    var mcp_sock = FlintMcpUnixListener(default_mcp_socket_path())
    if mcp_sock.active and mcp_sock.socket_path.byte_length() > 0:
        studio.agent_runtime.acp.configure_studio_mcp_socket(exe_path, mcp_sock.socket_path)
    var mcp_pool = JobPool()
    _ = mcp_pool.start(2)
    var dragging_gizmo = False
    var drag_start_ray = ViewportRay()
    var drag_start_pivot = Vec3()
    var drag_snaps = List[SelectedTransformSnap]()
    var drag_seq = UInt64(100)
    var pending_pick_ctrls = List[Tuple[UInt64, Bool]]()
    var resume_swapchain = False
    var t_last = get_time()

    while True:
        studio.app.schedule.run_stage(
            STAGE_FIRST, studio.app.world, studio.app.resources, studio.app.pool, studio.app.commands
        )
        studio.app.world.flush_commands(studio.app.commands)

        var w_opt = studio.app.resources.get_ptr_mut[WindowStateResource]()
        if not w_opt or not w_opt.value()[unsafe_offset=0].initialized:
            break
        ref win_res = w_opt.value()[unsafe_offset=0]
        if win_res.close_requested:
            if studio.dirty and not studio.quit_requested:
                win_res.close_requested = False
                if len(win_res.windows) > 0:
                    win_res.windows[0].close_requested = False
                    win_res.windows[0].x11.close_requested = False
                    win_res.windows[0].kms.close_requested = False
                _ = studio.request_quit()
            else:
                break
        if studio.quit_requested:
            break
        if win_res.resized:
            out_w = max(64, win_res.width)
            out_h = max(64, win_res.height)
            var r_opt = studio.app.resources.get_ptr_mut[ScalableRenderer]()
            if r_opt:
                _ = r_opt.value()[unsafe_offset=0].resize(out_w, out_h)

        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - t_last)))
        t_last = now

        var inp_opt = studio.app.resources.get_ptr[InputStateResource]()
        var frame_input = inp_opt.value()[unsafe_offset=0].input.copy() if inp_opt else InputState()

        ref window = win_res.windows[0]
        if studio.source_editor.present(window, frame_input, scene_hotkey=False):
            # Track held keys across the focus boundary so returning with Ctrl+F8
            # cannot trigger the scene's F8 binding again on the following frame.
            studio.input_map.update(frame_input, dt)
            var editor_renderer = studio.app.resources.get_ptr_mut[ScalableRenderer]()
            if editor_renderer:
                ref renderer = editor_renderer.value()[unsafe_offset=0]
                if studio.source_editor.visible and renderer.surface.is_zero_copy_gpu():
                    # The text workspace presents a cached, pixel-aligned frame.
                    _ = window.detach_vulkan(renderer.gpu_slot[0].vk)
                    _ = renderer.surface.initialize_headless(out_w, out_h)
                    resume_swapchain = True
                elif not studio.source_editor.visible and resume_swapchain:
                    _ = renderer.attach_window_surface(window)
                    resume_swapchain = False
                _ = assets.tick(now)
                var res_ed = studio.app.resources.get_ptr_mut[SceneAssetResolver]()
                if res_ed and assets.hot_reloads_applied > res_ed.value()[].hot_reloads_applied:
                    res_ed.value()[].resolve_world(studio.app.world, assets, studio.app.resources)
                    studio.sync_flmat_studio_styles()
                    studio.asset_browser_scanned = False
                _sync_mcp_gate_from_bus(mcp, studio)
                _ = mcp_sock.poll_once_with_renderer(mcp, studio, assets, mcp_pool, renderer.gpu_slot[0])
                _drain_mcp_events_to_studio(mcp, studio)
            dragging_gizmo = False
            studio.active_gizmo_handle = GIZMO_AXIS_NONE
            continue

        # Harvest completed non-stalling async GPU viewport picks from previous frames
        var r_pick = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if r_pick and r_pick.value()[unsafe_offset=0].uses_vulkan():
            while True:
                var harvested = picker.poll_completed_pick_entry(r_pick.value()[unsafe_offset=0].gpu_slot[0])
                if not harvested:
                    break
                var entry = harvested.value()
                var h_req_id = entry[0]
                var pres = entry[1]
                var pick_ctrl = False
                var kept = List[Tuple[UInt64, Bool]](capacity=len(pending_pick_ctrls))
                for pi in range(len(pending_pick_ctrls)):
                    if pending_pick_ctrls[pi][0] == h_req_id:
                        pick_ctrl = pending_pick_ctrls[pi][1]
                    else:
                        kept.append(pending_pick_ctrls[pi])
                pending_pick_ctrls = kept^
                if (
                    pres.hit
                    and pres.entity_id >= 0
                    and studio.is_alive_id(pres.entity_id)
                    and (pres.entity_bits == 0 or studio.resolve_entity(pres.entity_id).to_bits() == pres.entity_bits)
                ):
                    studio.select_entity(pres.entity_id, multi_select=pick_ctrl)
                elif not pick_ctrl:
                    studio.clear_selection()

        if studio.handle_command_menu(frame_input, out_w, out_h):
            # Record held keys without sending menu gestures to the scene.
            studio.input_map.update(frame_input, dt)
            frame_input = InputState()
        elif not studio.process_editor_input(frame_input, dt):
            break
        if studio.quit_requested:
            break

        var vp_rect = studio.viewport_rect(out_w, out_h)
        var vp_x = vp_rect.x
        var vp_y = vp_rect.y
        var vp_w = vp_rect.w
        var vp_h = vp_rect.h

        var mx = frame_input.last_mouse_x
        var my = frame_input.last_mouse_y
        var in_vp = vp_rect.contains(mx, my)

        var ui_blocks_vp = (
            studio.dock.dragging_node >= 0
            or studio.dock.hovered_splitter_node >= 0
            or studio.wants_mouse_at(mx, my)
            or studio.ui_ctx.wants_mouse_at(mx, my)
            or studio.menu.bar.group >= 0
            or studio.pie.mode != PIE_STOPPED
        )
        var gizmo_rot = studio.selection_gizmo_rotation()
        if frame_input.mouse_pressed(0) and in_vp and not ui_blocks_vp:
            var g_piv = studio.selection_gizmo_pivot()
            var g_hit = GIZMO_AXIS_NONE
            if g_piv[0]:
                g_hit = pick_gizmo_handle(
                    mx, my, g_piv[1],
                    studio.gizmo_mode, vp_x, vp_y, vp_w, vp_h,
                    studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
                    gizmo_rot=gizmo_rot,
                )
            if g_hit != GIZMO_AXIS_NONE:
                dragging_gizmo = True
                studio.active_gizmo_handle = g_hit
                drag_seq += 1
                drag_start_ray = ViewportRay.from_screen(
                    mx, my, vp_x, vp_y, vp_w, vp_h,
                    studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
                )
                drag_start_pivot = g_piv[1]
                drag_snaps = studio.capture_selected_transforms()
            else:
                var ctrl_down = frame_input.key_down(341) or frame_input.key_down(345)
                if picker.has_pending_pick:
                    var stale_id = picker.queued_request_id
                    var kept = List[Tuple[UInt64, Bool]](capacity=len(pending_pick_ctrls))
                    for pi in range(len(pending_pick_ctrls)):
                        if pending_pick_ctrls[pi][0] != stale_id:
                            kept.append(pending_pick_ctrls[pi])
                    pending_pick_ctrls = kept^
                var req_id = picker.request_pick(Int(mx), Int(my))
                pending_pick_ctrls.append((req_id, ctrl_down))

        if (
            studio.pie.mode != PIE_STOPPED
            or studio.ui_ctx.modal_depth > 0
            or studio.ui_ctx.had_modal_prev_frame
            or (frame_input.key_pressed(256) and not studio.wants_keyboard())
        ):
            dragging_gizmo = False
            studio.active_gizmo_handle = GIZMO_AXIS_NONE

        if dragging_gizmo and frame_input.mouse_down(0) and len(drag_snaps) > 0:
            var cur_ray = ViewportRay.from_screen(
                mx, my, vp_x, vp_y, vp_w, vp_h,
                studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
            )
            var h_id = studio.active_gizmo_handle
            var wscale = gizmo_world_scale(drag_start_pivot, studio.cam_pos, studio.cam_fwd, studio.fov_y_deg, Int(vp_h))
            if studio.gizmo_mode == GIZMO_MODE_TRANSLATE:
                var delta_p = Vec3()
                if h_id == GIZMO_AXIS_X or h_id == GIZMO_AXIS_Y or h_id == GIZMO_AXIS_Z:
                    var ax = gizmo_rot.rotate(Vec3(1, 0, 0) if h_id == GIZMO_AXIS_X else (Vec3(0, 1, 0) if h_id == GIZMO_AXIS_Y else Vec3(0, 0, 1)))
                    var t0 = ray_axis_closest_param(drag_start_ray, drag_start_pivot, ax)
                    var t1 = ray_axis_closest_param(cur_ray, drag_start_pivot, ax)
                    if t0[0] and t1[0]:
                        delta_p = ax * (t1[1] - t0[1])
                elif h_id == GIZMO_PLANE_XY or h_id == GIZMO_PLANE_XZ or h_id == GIZMO_PLANE_YZ:
                    var pn = gizmo_rot.rotate(Vec3(0, 0, 1) if h_id == GIZMO_PLANE_XY else (Vec3(0, 1, 0) if h_id == GIZMO_PLANE_XZ else Vec3(1, 0, 0)))
                    var p0 = ray_plane_hit(drag_start_ray, drag_start_pivot, pn)
                    var p1 = ray_plane_hit(cur_ray, drag_start_pivot, pn)
                    if p0[0] and p1[0]:
                        delta_p = p1[1] - p0[1]
                if studio.snap_enabled:
                    var snapped_piv = snap_translation(drag_start_pivot + delta_p, studio.grid_snap_m)
                    delta_p = snapped_piv - drag_start_pivot
                _ = studio.apply_gizmo_drag_selected(
                    drag_snaps,
                    drag_start_pivot,
                    delta_p,
                    Quat(0.0, 0.0, 0.0, 1.0),
                    Vec3(1.0, 1.0, 1.0),
                    GIZMO_MODE_TRANSLATE,
                    coalesce_key=drag_seq,
                )
            elif studio.gizmo_mode == GIZMO_MODE_ROTATE:
                var rax = gizmo_rot.rotate(Vec3(1, 0, 0) if h_id == GIZMO_AXIS_X else (Vec3(0, 1, 0) if h_id == GIZMO_AXIS_Y else Vec3(0, 0, 1)))
                var drad = gizmo_rotation_delta_rad(drag_start_ray, cur_ray, drag_start_pivot, rax, studio.cam_fwd, wscale)
                if studio.snap_enabled:
                    drad = snap_angle_deg(drad, studio.angle_snap_deg)
                var dq = Quat(rax.x * sin(drad * Float32(0.5)), rax.y * sin(drad * Float32(0.5)), rax.z * sin(drad * Float32(0.5)), cos(drad * Float32(0.5)))
                _ = studio.apply_gizmo_drag_selected(
                    drag_snaps,
                    drag_start_pivot,
                    Vec3(),
                    dq,
                    Vec3(1.0, 1.0, 1.0),
                    GIZMO_MODE_ROTATE,
                    coalesce_key=drag_seq,
                )
            elif studio.gizmo_mode == GIZMO_MODE_SCALE:
                var sf = gizmo_scale_drag(drag_start_ray, cur_ray, drag_start_pivot, h_id, wscale, studio.cam_fwd, studio.cam_right, Vec3(1.0, 1.0, 1.0), gizmo_rot=gizmo_rot)
                _ = studio.apply_gizmo_drag_selected(
                    drag_snaps,
                    drag_start_pivot,
                    Vec3(),
                    Quat(0.0, 0.0, 0.0, 1.0),
                    sf,
                    GIZMO_MODE_SCALE,
                    coalesce_key=drag_seq,
                )

        if not frame_input.mouse_down(0):
            dragging_gizmo = False
            studio.active_gizmo_handle = GIZMO_AXIS_NONE

        _ = assets.tick(now)
        var r_mcp = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        var gms = 0.0
        if r_mcp and r_mcp.value()[unsafe_offset=0].uses_vulkan():
            _sync_mcp_gate_from_bus(mcp, studio)
            _ = mcp_sock.poll_once_with_renderer(mcp, studio, assets, mcp_pool, r_mcp.value()[unsafe_offset=0].gpu_slot[0])
            _drain_mcp_events_to_studio(mcp, studio)
            gms = r_mcp.value()[unsafe_offset=0].gpu_slot[0].render_graph.total_gpu_ms

        if studio.pie.mode == PIE_PLAYING:
            _ = studio.step_pie_frame(dt)

        studio.render_editor_frame(
            out_w,
            out_h,
            gpu_ms=gms,
            mounted_packs=len(assets.mounts),
            mcp_clients=mcp_sock.client_count(),
            input=frame_input,
            dt=dt,
        )
        _fill_instance_entity_maps(studio, inst_to_ent, inst_to_bits)

        var r_post = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if r_post:
            ref renderer = r_post.value()[unsafe_offset=0]
            if renderer.uses_vulkan():
                _ = picker.dispatch_pending_pick(renderer.gpu_slot[0], inst_to_ent, inst_to_bits)
            if not renderer.surface.is_zero_copy_gpu():
                var frame_px = renderer.readback_pixels()
                var w_p = studio.app.resources.get_ptr_mut[WindowStateResource]()
                if w_p and len(w_p.value()[unsafe_offset=0].windows) > 0:
                    w_p.value()[unsafe_offset=0].windows[0].present(frame_px, renderer.width, renderer.height)

    mcp_sock.close()
    mcp_pool.stop()
    var r_end = studio.app.resources.get_ptr_mut[ScalableRenderer]()
    if r_end and r_end.value()[unsafe_offset=0].uses_vulkan():
        picker.shutdown(r_end.value()[unsafe_offset=0].gpu_slot[0])
    studio.shutdown()
    return 0


def main() raises:
    var rc = _run_studio()
    if rc != 0:
        external_call["exit", NoneType](Int32(rc))
