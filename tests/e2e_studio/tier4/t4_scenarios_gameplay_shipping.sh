#!/usr/bin/env bash
# AREA: r1 m0 m2 m3
# Tier 4 Real-World Application Scenarios (Scenarios 3 & 5):
#   S3: Behaviour loop — gameplay-so + HotReloadHost (FlintPluginApiV2/PodComponentSchema)
#       + KCC + Trigger (OnEnter/OnExit, entity_from_body) + BehaviorTreeComponent
#       + NavPlugin/NavMesh + AudioPlugin/AudioServer + game_hud() + Shift+F1 camera possession
#   S5: Full shipping acceptance — examples/test_harness_game/ (and flint.sh new) -> package
#       -> relocated clean-dir run.sh (headless golden-image FLIP + input-replay) under run_in_clean_env

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t4_gameplay_ship"
resolve_mojo_env || {
    fail_test "t4_gameplay_ship_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

# ---------------------------------------------------------------------------
# Scenario 3: Behaviour Loop — gameplay-so + HotReloadHost (FlintPluginApiV2)
#   + KCC + Trigger (OnEnter/OnExit, entity_from_body) + BehaviorTreeComponent
#   + NavPlugin/NavMesh + AudioPlugin/AudioServer + game_hud() + Shift+F1 PIE camera
# ---------------------------------------------------------------------------
test_t4_s3_gameplay_so_hot_reload_kcc_bt_nav_audio_hud_shift_f1() {
    local proj="${TMPD}/s3_gameplay_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t4_s3_gameplay_so_hot_reload_kcc_bt_nav_audio_hud_shift_f1" "flint.sh new failed"
        return
    fi

    # 1. Verify tools/flint.sh gameplay-so builds a project gameplay .so
    if ! grep -q "gameplay-so" "${ROOT}/tools/flint.sh"; then
        fail_test "t4_s3_gameplay_so_hot_reload_kcc_bt_nav_audio_hud_shift_f1" "M2 F22 contract pending: tools/flint.sh missing 'gameplay-so' subcommand"
        return
    fi

    local gp_src="${proj}/src/gameplay.mojo"
    cat >"$gp_src" <<'MOJO'
from flint.app.hot_reload import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2, PodComponentSchema

@export("flint_plugin_api_version")
def flint_plugin_api_version() abi("C") -> UInt32:
    return FLINT_PLUGIN_API_VERSION

@export("flint_plugin_on_load")
def flint_plugin_on_load(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_step")
def flint_plugin_on_step(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _dt: Float32) abi("C") -> Int32:
    return 1

@export("flint_plugin_on_unload")
def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> None:
    pass
MOJO

    local so_out so_rc=0
    so_out="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" gameplay-so 2>&1)" || so_rc=$?
    if [[ "$so_rc" -ne 0 ]]; then
        fail_test "t4_s3_gameplay_so_hot_reload_kcc_bt_nav_audio_hud_shift_f1" "flint.sh gameplay-so failed (rc=$so_rc): $so_out"
        return
    fi

    local so_file
    so_file="$(find "${proj}/build" -maxdepth 3 -type f -name '*.so' -size +0c -print -quit 2>/dev/null || true)"
    if [[ -z "$so_file" || ! -f "$so_file" ]]; then
        fail_test "t4_s3_gameplay_so_hot_reload_kcc_bt_nav_audio_hud_shift_f1" "flint.sh gameplay-so did not produce a non-empty .so in ${proj}/build"
        return
    fi

    # 2. Compile & execute headless Mojo PIE behaviour loop exercising all Scenario 3 subsystems
    local probe_src="${TMPD}/s3_behaviour_probe.mojo"
    local probe_bin="${TMPD}/bin/s3_behaviour_probe"
    cat >"$probe_src" <<'MOJO'
from std.ffi import external_call
from std.sys import argv
from flint.ai import (
    BT_ABORT_LOWER_PRIORITY,
    BT_CMP_GT,
    BT_RUNNING,
    BT_SUCCESS,
    BehaviorTree,
    Blackboard,
    BtExecutionState,
)
from flint.ai.behavior_tree import BehaviorTreeComponent
from flint.app.engine_app import AppConfig, EngineApp
from flint.app.hot_reload import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2, HotReloadHost, PodComponentSchema
from flint.audio.plugin import AudioPlugin
from flint.audio.server import AudioServer, BUS_SFX
from flint.core.math import Vec3
from flint.nav.navmesh import NavMesh
from flint.nav.plugin import NavPlugin
from flint.physics.character import CharacterControllerConfig
from flint.physics.plugin import (
    CharacterControllerComponent,
    ColliderComponent,
    OnEnter,
    OnExit,
    PhysicsPlugin,
    RigidBodyComponent,
    TriggerEventsResource,
    entity_from_body,
)
from flint.physics.world import PhysicsWorld
from flint.platform.audio import decode_wav
from flint.platform.input import InputState
from flint.render.plugin import CameraComponent, RenderPluginStateResource
from flint.scene.transform_hierarchy import GlobalTransform, LocalTransform, PrevGlobalTransform, TransformPlugin
from flint.studio.studio_app import COLLIDER_BOX, FlintStudioState, PIE_PLAYING
from flint.ui.hud import game_hud
from e2e_studio.common.studio_test_utils import read_bytes, write_minimal_wav, write_text

def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error("usage: s3_behaviour_probe <proj_dir> <flint_sh> <gameplay.so> <tmp_wav>")
    var proj_dir = String(args[1])
    var flint_sh = String(args[2])
    var so_path = String(args[3])
    var wav_path = String(args[4])
    if Int(FLINT_PLUGIN_API_VERSION) != 2:
        raise Error("expected FLINT_PLUGIN_API_VERSION == 2")
    if not write_minimal_wav(wav_path, 0.15, 48000, 2):
        raise Error("failed writing s3 wav")
    var wav = decode_wav(read_bytes(wav_path))
    if not wav.valid:
        raise Error("invalid s3 wav")

    # Verify NavPlugin & AudioPlugin startup lifecycle on EngineApp (before startup())
    var aux_app = EngineApp(AppConfig(title="s3_nav_aud_aux", headless=True, worker_threads=0))
    aux_app.add_plugin(TransformPlugin())
    aux_app.add_plugin(PhysicsPlugin(gravity=0.0))
    aux_app.add_plugin(NavPlugin())
    aux_app.add_plugin(AudioPlugin(headless=True))
    aux_app.startup()
    var aux_floor = aux_app.world.spawn()
    _ = aux_app.world.insert[LocalTransform](aux_floor, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = aux_app.world.insert[GlobalTransform](aux_floor, GlobalTransform())
    _ = aux_app.world.insert[PrevGlobalTransform](aux_floor, PrevGlobalTransform())
    _ = aux_app.world.insert[ColliderComponent](aux_floor, ColliderComponent.box(Vec3(10.0, 0.5, 10.0)))
    _ = aux_app.world.insert[RigidBodyComponent](aux_floor, RigidBodyComponent.static_body())
    aux_app.step_one_frame(Float32(1.0 / 60.0))
    var aux_pw = aux_app.resources.get_ptr[PhysicsWorld]()
    if not aux_pw:
        raise Error("aux_app missing PhysicsWorld")
    var aux_nm = NavMesh.build_from_physics(aux_pw.value()[unsafe_offset=0])
    var aux_path = aux_nm.find_path(Vec3(-4.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0))
    if aux_nm.poly_count() <= 0 or not aux_path.found or len(aux_path.waypoints) < 2:
        raise Error("NavPlugin/NavMesh bake & query failed in aux_app")
    aux_app.shutdown()

    var studio = FlintStudioState(headless=True, width=320, height=180, worker_threads=0)
    studio.look_at(Vec3(-6.5, 4.2, -9.5), Vec3(0.0, 1.0, 0.0))

    _ = studio.spawn_entity(
        "S3Floor",
        position=Vec3(0.0, -0.5, 0.0),
        scale=Vec3(20.0, 1.0, 20.0),
        collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(10.0, 0.5, 10.0),
        mass=0.0,
        record_undo=False,
    )

    var schema = PodComponentSchema("EnemyStats", 8)
    var host = HotReloadHost()
    if not host.load_or_reload(so_path, studio.app.world, studio.app.resources):
        raise Error("HotReloadHost.load_or_reload (v1) failed: " + host.last_error)
    var api_v2 = FlintPluginApiV2()
    if host.reload_generation != 1 or host.step(studio.app.world, studio.app.resources, Float32(1.0 / 60.0)) != 1:
        raise Error("gameplay.so v1 generation or step mismatch")

    var cam_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(0.0, 4.0, -8.0)))
    _ = studio.app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = studio.app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=60.0, active=True))

    var goal_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](goal_e, LocalTransform(Vec3(2.0, 1.0, 0.0)))
    _ = studio.app.world.insert[GlobalTransform](goal_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](goal_e, PrevGlobalTransform())
    _ = studio.app.world.insert[ColliderComponent](goal_e, ColliderComponent.box(Vec3(0.8, 1.5, 0.8), is_trigger=True))

    var kcc_cfg = CharacterControllerConfig()
    kcc_cfg.gravity = 0.0
    var player_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](player_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = studio.app.world.insert[GlobalTransform](player_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](player_e, PrevGlobalTransform())
    _ = studio.app.world.insert[CharacterControllerComponent](
        player_e, CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), kcc_cfg, Vec3(6.0, 0.0, 0.0))
    )

    var enemy_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](enemy_e, LocalTransform(Vec3(5.0, 0.0, 0.0)))
    _ = studio.app.world.insert[GlobalTransform](enemy_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](enemy_e, PrevGlobalTransform())
    _ = studio.app.world.insert[BehaviorTreeComponent](enemy_e, BehaviorTreeComponent())

    var bb = Blackboard()
    var k_alert = bb.def_key("alert", Float32(0.0))
    var bt = BehaviorTree()
    _ = bt.begin_selector()
    _ = bt.begin_sequence()
    _ = bt.add_blackboard_cmp(k_alert, BT_CMP_GT, Float32(0.5), BT_ABORT_LOWER_PRIORITY)
    _ = bt.add_action(10)
    bt.end_composite()
    _ = bt.add_action(2)
    bt.end_composite()
    var bt_st = BtExecutionState()
    bt_st.set_action_status(2, BT_RUNNING)
    bt_st.set_action_status(10, BT_SUCCESS)
    var bt_r1 = bt.tick(bb, bt_st, Float32(1.0 / 60.0))
    bb.set_float(k_alert, Float32(1.0))
    var bt_r2 = bt.tick(bb, bt_st, Float32(1.0 / 60.0))
    if not studio.app.world.has[BehaviorTreeComponent](enemy_e) or bt_r1 != BT_RUNNING or bt_r2 != BT_SUCCESS:
        raise Error("BehaviorTreeComponent/BT tick failed")

    # Bake & query NavMesh on Studio world before PIE
    studio.render_editor_frame(320, 180)
    var st_pw = studio.app.resources.get_ptr[PhysicsWorld]()
    if not st_pw:
        raise Error("studio missing PhysicsWorld")
    var st_nm = NavMesh.build_from_physics(st_pw.value()[unsafe_offset=0])
    var st_path = st_nm.find_path(Vec3(-4.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0))
    if st_nm.poly_count() <= 0 or not st_path.found or len(st_path.waypoints) < 2:
        raise Error("Studio NavMesh bake/query failed")

    if not studio.pie_play() or studio.pie.mode != PIE_PLAYING:
        raise Error("pie_play failed")

    # Exercise PIE loop: KCC triggers (OnEnter + OnExit + entity_from_body)
    var cursor = 0
    var saw_enter = False
    var saw_exit = False
    for _ in range(60):
        _ = studio.step_pie_frame(Float32(1.0 / 60.0))
        studio.render_editor_frame(320, 180, dt=Float32(1.0 / 60.0))
        var tevs = studio.app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(cursor)
        for i in range(len(tevs)):
            if OnEnter(tevs[i], goal_e, studio.app.world, studio.app.resources):
                if entity_from_body(tevs[i].body_id, studio.app.resources) == player_e:
                    saw_enter = True
            if OnExit(tevs[i], goal_e, studio.app.world, studio.app.resources):
                if entity_from_body(tevs[i].body_id, studio.app.resources) == player_e:
                    saw_exit = True
        if saw_enter and saw_exit:
            break
    if not saw_enter or not saw_exit:
        raise Error("KCC trigger OnEnter/OnExit failed: enter=" + String(saw_enter) + " exit=" + String(saw_exit))

    # Perform real edit -> flint.sh gameplay-so rebuild -> HotReloadHost.load_or_reload (v1 -> v2) during PIE without restarting Studio
    var ents_before_reload = studio.app.world.entity_count()
    var v2_src = String("from flint.app.hot_reload import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2, PodComponentSchema\n\n")
    v2_src += "@export(\"flint_plugin_api_version\")\n"
    v2_src += "def flint_plugin_api_version() abi(\"C\") -> UInt32:\n"
    v2_src += "    return FLINT_PLUGIN_API_VERSION\n\n"
    v2_src += "@export(\"flint_plugin_on_load\")\n"
    v2_src += "def flint_plugin_on_load(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi(\"C\") -> Int32:\n"
    v2_src += "    return 0\n\n"
    v2_src += "@export(\"flint_plugin_on_step\")\n"
    v2_src += "def flint_plugin_on_step(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _dt: Float32) abi(" + '"C"' + ") -> Int32:\n"
    v2_src += "    return 2\n\n"
    v2_src += "@export(\"flint_plugin_on_unload\")\n"
    v2_src += "def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi(\"C\") -> None:\n"
    v2_src += "    pass\n"
    if not write_text(proj_dir + "/src/gameplay.mojo", v2_src):
        raise Error("failed writing v2 gameplay.mojo")
    var rebuild_cmd = "cd '" + proj_dir + "' && bash '" + flint_sh + "' gameplay-so >/dev/null 2>&1"
    if external_call["system", Int32](rebuild_cmd.as_c_string_span()) != 0:
        raise Error("flint.sh gameplay-so rebuild (v2) failed")
    if not host.load_or_reload(so_path, studio.app.world, studio.app.resources):
        raise Error("HotReloadHost.load_or_reload (v2) failed: " + host.last_error)
    if (
        host.reload_generation != 2
        or host.step(studio.app.world, studio.app.resources, Float32(1.0 / 60.0)) != 2
        or studio.app.world.entity_count() != ents_before_reload
        or not studio.app.world.is_alive(player_e)
    ):
        raise Error("v2 hot-reload during PIE failed to update step or preserve world state")

    # Verify Shift+F1 releases game camera possession back to editor camera override in PIE
    var rps = studio.app.resources.get_ptr[RenderPluginStateResource]()
    if not rps or rps.value()[unsafe_offset=0].editor_camera_override:
        raise Error("PIE did not possess game camera (expected editor_camera_override == False)")
    var inp = InputState()
    inp.begin_frame()
    inp.set_logical_key(340, True)  # Left Shift
    inp.set_logical_key(290, True)  # F1
    _ = studio.process_editor_input(inp, Float32(1.0 / 60.0))
    studio.render_editor_frame(320, 180, dt=Float32(1.0 / 60.0))
    var rps_after = studio.app.resources.get_ptr[RenderPluginStateResource]()
    if not rps_after or not rps_after.value()[unsafe_offset=0].editor_camera_override:
        raise Error("Shift+F1 did not restore editor_camera_override == True during PIE")

    # Verify AudioServer playback before stop_all()
    var audio_srv = studio.app.resources.get_ptr_mut[AudioServer]()
    if not audio_srv:
        raise Error("studio missing AudioServer")
    var aid = audio_srv.value()[unsafe_offset=0].register_wav_asset(wav)
    var sh = audio_srv.value()[unsafe_offset=0].play_2d(aid, volume=0.5, bus_id=BUS_SFX, looping=True)
    if not studio.app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].rt_running:
        var out_lr = List[Float32]()
        studio.app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].render_offline_quantum(out_lr, 256)
    for _ in range(10):
        _ = studio.step_pie_frame(Float32(1.0 / 60.0))
        if studio.app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() >= 1:
            break
        if studio.app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].rt_running:
            _ = external_call["usleep", Int32](UInt32(2000))
    if not sh.is_valid() or studio.app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() < 1:
        raise Error("AudioServer.play_2d did not start sound before stop_all()")
    studio.app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].stop_all()
    if not studio.app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].rt_running:
        var out_lr = List[Float32]()
        studio.app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].render_offline_quantum(out_lr, 256)
    for _ in range(10):
        _ = studio.step_pie_frame(Float32(1.0 / 60.0))
        if studio.app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() == 0:
            break
        if studio.app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].rt_running:
            _ = external_call["usleep", Int32](UInt32(2000))
    if studio.app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() != 0:
        raise Error("AudioServer.stop_all() did not clear active logical emitters")

    _ = game_hud(studio.ui_ctx, 100, 100, 0)

    if not studio.pie_stop():
        raise Error("pie_stop failed")
    if not host.unload(studio.app.world, studio.app.resources):
        raise Error("HotReloadHost.unload failed: " + host.last_error)
    studio.shutdown()
    if schema.byte_size != 8 or api_v2.version != 2:
        raise Error("Scenario 3 schema/api_v2 assertions failed")
MOJO
    local compile_log="${TMPD}/s3_probe.compile.log"
    if ! "$MOJO_BIN" build -I "${ROOT}/src" -I "${ROOT}/tests" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >"$compile_log" 2>&1; then
        local c_err
        c_err="$(tail -n 6 "$compile_log" | tr '\n' ' ')"
        fail_test "t4_s3_gameplay_so_hot_reload_kcc_bt_nav_audio_hud_shift_f1" "M2 contract pending (s3_probe compile failed): ${c_err}"
        return
    fi

    local probe_out probe_rc=0
    probe_out="$("$probe_bin" "$proj" "${ROOT}/tools/flint.sh" "$so_file" "${TMPD}/s3_sfx.wav" 2>&1)" || probe_rc=$?
    if [[ "$probe_rc" -ne 0 ]]; then
        fail_test "t4_s3_gameplay_so_hot_reload_kcc_bt_nav_audio_hud_shift_f1" "s3_probe failed (rc=$probe_rc): $probe_out"
        return
    fi

    pass_test "t4_s3_gameplay_so_hot_reload_kcc_bt_nav_audio_hud_shift_f1"
}

# ---------------------------------------------------------------------------
# Scenario 5a: flint.sh new -> assets/main.flscn -> flint.sh package ->
#   completely isolated clean-dir run.sh (--headless-smoke + --shots FLIP check)
# ---------------------------------------------------------------------------
test_t4_s5_scaffold_new_package_relocated_clean_run() {
    local proj="${TMPD}/s5_new_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t4_s5_scaffold_new_package_relocated_clean_run" "flint.sh new failed"
        return
    fi

    if [[ ! -f "${proj}/assets/main.flscn" ]] || ! grep -qE 'main\.flscn|entry_scene' "${proj}/src/main.mojo"; then
        fail_test "t4_s5_scaffold_new_package_relocated_clean_run" "M3 F26 contract pending: flint.sh new does not scaffold/load assets/main.flscn"
        return
    fi

    local stage="${TMPD}/s5_stage"
    local pkg_out pkg_rc=0
    pkg_out="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" package --out "$stage" 2>&1)" || pkg_rc=$?
    if [[ "$pkg_rc" -ne 0 ]]; then
        fail_test "t4_s5_scaffold_new_package_relocated_clean_run" "flint.sh package failed (rc=$pkg_rc): $pkg_out"
        return
    fi

    local reloc="${TMPD}/s5_isolated_machine/game"
    mkdir -p "$(dirname "$reloc")"
    cp -a "$stage" "$reloc"
    rm -rf "$stage" "$proj"

    if grep -Fq "/home/user/flint" "${reloc}/run.sh" "${reloc}/flint.toml" || grep -Fq "${ROOT}" "${reloc}/run.sh" "${reloc}/flint.toml"; then
        fail_test "t4_s5_scaffold_new_package_relocated_clean_run" "relocated bundle leaks repo path"
        return
    fi
    if ! validate_flpk_v3 "${reloc}/assets.flpk" "main.flscn"; then
        fail_test "t4_s5_scaffold_new_package_relocated_clean_run" "assets.flpk missing main.flscn"
        return
    fi

    local smoke_out smoke_rc=0
    smoke_out="$(FLINT_STATE_DIR="${TMPD}/s5_state" XDG_STATE_HOME="${TMPD}/s5_xdg" run_in_clean_env "$reloc" ./run.sh --headless-smoke 2>&1)" || smoke_rc=$?
    if [[ "$smoke_rc" -ne 0 ]]; then
        fail_test "t4_s5_scaffold_new_package_relocated_clean_run" "clean-dir ./run.sh --headless-smoke failed (rc=$smoke_rc): $smoke_out"
        return
    fi

    local shot="${reloc}/s5_shot.png"
    local shots_rc=0
    FLINT_STATE_DIR="${TMPD}/s5_state" XDG_STATE_HOME="${TMPD}/s5_xdg" run_in_clean_env "$reloc" ./run.sh --shots --out "$shot" >/dev/null 2>&1 || shots_rc=$?
    if [[ "$shots_rc" -ne 0 ]] || ! validate_png_dims "$shot" 1280 720; then
        fail_test "t4_s5_scaffold_new_package_relocated_clean_run" "clean-dir ./run.sh --shots failed or invalid PNG"
        return
    fi

    pass_test "t4_s5_scaffold_new_package_relocated_clean_run"
}

# ---------------------------------------------------------------------------
# Scenario 5b: Studio-authored examples/test_harness_game/ -> flint.sh package
#   -> isolated clean directory -> run.sh headless golden-image FLIP + input-replay
# ---------------------------------------------------------------------------
test_t4_s5_test_harness_game_clean_dir_golden_and_replay() {
    local th_dir="${ROOT}/examples/test_harness_game"
    if [[ ! -d "$th_dir" || ! -f "${th_dir}/flint.toml" ]]; then
        fail_test "t4_s5_test_harness_game_clean_dir_golden_and_replay" "M3 F28 contract pending: ${th_dir} does not exist"
        return
    fi

    local th_copy="${TMPD}/th_project"
    rm -rf "$th_copy"
    mkdir -p "$th_copy"
    cp -a "${th_dir}/." "$th_copy/"
    rm -rf "${th_copy}/build" "${th_copy}/dist"

    local th_stage="${TMPD}/th_stage"
    local pkg_out pkg_rc=0
    pkg_out="$(cd "$th_copy" && bash "${ROOT}/tools/flint.sh" package --out "$th_stage" 2>&1)" || pkg_rc=$?
    if [[ "$pkg_rc" -ne 0 ]]; then
        fail_test "t4_s5_test_harness_game_clean_dir_golden_and_replay" "packaging test_harness_game failed (rc=$pkg_rc): $pkg_out"
        return
    fi

    local reloc="${TMPD}/clean_machine/test_harness_game"
    mkdir -p "$(dirname "$reloc")"
    cp -a "$th_stage" "$reloc"
    rm -rf "$th_stage"

    if grep -Fq "/home/user/flint" "${reloc}/run.sh" "${reloc}/flint.toml" || grep -Fq "${ROOT}" "${reloc}/run.sh" "${reloc}/flint.toml"; then
        fail_test "t4_s5_test_harness_game_clean_dir_golden_and_replay" "bundle run.sh or flint.toml leaks repo root"
        return
    fi
    if ! validate_flpk_v3 "${reloc}/assets.flpk"; then
        fail_test "t4_s5_test_harness_game_clean_dir_golden_and_replay" "bundle assets.flpk invalid"
        return
    fi

    local smoke_out smoke_rc=0
    smoke_out="$(FLINT_STATE_DIR="${TMPD}/th_state" XDG_STATE_HOME="${TMPD}/th_xdg" run_in_clean_env "$reloc" ./run.sh --headless-smoke 2>&1)" || smoke_rc=$?
    if [[ "$smoke_rc" -ne 0 ]]; then
        fail_test "t4_s5_test_harness_game_clean_dir_golden_and_replay" "clean-dir ./run.sh --headless-smoke failed (rc=$smoke_rc): $smoke_out"
        return
    fi

    # Verify golden-image FLIP check & input-replay check through clean-dir run.sh
    local golden_ok=0 replay_ok=0 check_out
    for gflag in "--verify-golden" "--golden-check" "--test-harness"; do
        if check_out="$(FLINT_STATE_DIR="${TMPD}/th_state" XDG_STATE_HOME="${TMPD}/th_xdg" run_in_clean_env "$reloc" ./run.sh "$gflag" 2>&1)"; then
            if grep -qiE "golden|flip" <<<"$check_out" && grep -qE '\b(PASS|OK)\b|\[PASS\]' <<<"$check_out"; then
                golden_ok=1
                break
            fi
        fi
    done
    if [[ "$golden_ok" -eq 0 ]]; then
        # Fallback: capture shot via --shots and compare against shipped golden PNG via studio_test_utils
        local shot_out="${reloc}/harness_shot.png"
        local golden_ref=""
        golden_ref="$(find "$th_dir" "$reloc" -iname '*golden*.png' -print -quit 2>/dev/null || true)"
        if [[ -n "$golden_ref" ]] && FLINT_STATE_DIR="${TMPD}/th_state" XDG_STATE_HOME="${TMPD}/th_xdg" run_in_clean_env "$reloc" ./run.sh --shots --out "$shot_out" >/dev/null 2>&1; then
            local flip_src="${TMPD}/th_flip_check.mojo"
            local flip_bin="${TMPD}/bin/th_flip_check"
            cat >"$flip_src" <<'MOJO'
from std.sys import argv
from flint.assets.image import decode_png
from flint.platform.os import read_bytes
from e2e_studio.common.studio_test_utils import verify_studio_golden

def main() raises:
    var args = argv()
    var dec = decode_png(read_bytes(String(args[1])), 16384)
    if not dec.valid:
        raise Error("invalid shot PNG")
    var res = verify_studio_golden(dec.pixels, dec.width, dec.height, String(args[2]))
    if not res[0]:
        raise Error("golden FLIP mismatch: " + res[1])
MOJO
            if "$MOJO_BIN" build -I "${ROOT}/src" -I "${ROOT}/tests" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$flip_src" -o "$flip_bin" >/dev/null 2>&1; then
                if "$flip_bin" "$shot_out" "$golden_ref" >/dev/null 2>&1; then
                    golden_ok=1
                fi
            fi
        fi
    fi

    for rflag in "--replay" "--input-replay" "--verify-replay" "--test-harness"; do
        if check_out="$(FLINT_STATE_DIR="${TMPD}/th_state" XDG_STATE_HOME="${TMPD}/th_xdg" run_in_clean_env "$reloc" ./run.sh "$rflag" 2>&1)"; then
            if grep -qiE "replay|sdrp" <<<"$check_out" && grep -qE '\b(PASS|OK)\b|\[PASS\]' <<<"$check_out"; then
                replay_ok=1
                break
            fi
        fi
    done

    if [[ "$golden_ok" -ne 1 || "$replay_ok" -ne 1 ]]; then
        fail_test "t4_s5_test_harness_game_clean_dir_golden_and_replay" "test_harness_game clean-dir run.sh missing golden-image (ok=$golden_ok) or input-replay (ok=$replay_ok) verification"
        return
    fi

    pass_test "t4_s5_test_harness_game_clean_dir_golden_and_replay"
}

test_t4_s3_gameplay_so_hot_reload_kcc_bt_nav_audio_hud_shift_f1
test_t4_s5_scaffold_new_package_relocated_clean_run
test_t4_s5_test_harness_game_clean_dir_golden_and_replay

print_summary_and_exit
