#!/usr/bin/env bash
# AREA: r1 m0 m1 m2 m3
# Tier 3 Cross-Feature Pairwise Tests: CLI / MCP / Studio / Cooker / Packer / Clean-Dir Shipping

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t3_mcp_ship"
resolve_mojo_env || {
    fail_test "t3_mcp_ship_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

# ---------------------------------------------------------------------------
# Pair 1: Studio --project + MCP entity.set_transform + studio.undo_redo
#         + scene.save + restart fresh Studio + scene.load
# ---------------------------------------------------------------------------
test_t3_pair1_studio_project_transform_undo_save_restart_load() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_test "t3_pair1_studio_project_transform_undo_save_restart_load" "could not resolve flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"

    local proj="${TMPD}/pair1_proj"
    mkdir -p "${proj}/scenes" "${proj}/assets"

    local py_out py_rc=0
    py_out="$(
        FLINT_STATE_DIR="${TMPD}/p1_state" XDG_STATE_HOME="${TMPD}/p1_xdg" \
        python3 -B - "$studio_bin" "$proj" "${SCRIPT_DIR}/../common" "$TMPD" <<'PY' 2>&1
import os, sys
sys.path.insert(0, sys.argv[3])
from mcp_client import McpStdioClient

studio_bin, proj, other_cwd = sys.argv[1], sys.argv[2], sys.argv[4]

with McpStdioClient([studio_bin, "--project", proj, "--mcp-stdio"], cwd=other_cwd, framing="ndjson", timeout=30.0, allow_stderr=False) as c1:
    c1.initialize()
    c1.notify_initialized()
    tools = {t["name"] for t in c1.list_tools()}
    needed = {"entity.set_transform", "studio.undo_redo", "scene.save", "scene.load", "ecs.spawn_entity", "ecs.query_entities"}
    missing = sorted(needed - tools)
    if missing:
        raise SystemExit(f"M0/M3 contract pending; missing MCP tools: {missing}")

    sp = c1.call_tool("ecs.spawn_entity", {"name": "Pair1Actor", "position": [0.0, 1.0, 0.0], "mass": 2.0})
    eid = sp["entity_id"]

    c1.call_tool("entity.set_transform", {"entity_id": eid, "position": [5.0, 3.0, -2.0], "scale": [2.0, 2.0, 2.0]})
    c1.call_tool("studio.undo_redo", {"action": "undo", "steps": 1})
    q_undo = c1.call_tool("ecs.query_entities", {"name_contains": "Pair1Actor"})
    pos_u = q_undo["entities"][0]["position"]
    if abs(pos_u[0] - 0.0) > 1e-3 or abs(pos_u[1] - 1.0) > 1e-3:
        raise SystemExit(f"undo did not revert entity.set_transform: {pos_u}")

    c1.call_tool("studio.undo_redo", {"action": "redo", "steps": 1})
    q_redo = c1.call_tool("ecs.query_entities", {"name_contains": "Pair1Actor"})
    pos_r = q_redo["entities"][0]["position"]
    if abs(pos_r[0] - 5.0) > 1e-3 or abs(pos_r[1] - 3.0) > 1e-3 or abs(pos_r[2] - (-2.0)) > 1e-3:
        raise SystemExit(f"redo did not restore entity.set_transform: {pos_r}")

    c1.call_tool("scene.save", {"path": "scenes/pair1.flscn"})
    if not os.path.isfile(os.path.join(proj, "scenes", "pair1.flscn")):
        raise SystemExit("scene.save did not resolve relative path under --project")
    if os.path.exists(os.path.join(other_cwd, "scenes", "pair1.flscn")):
        raise SystemExit("scene.save leaked relative path into cwd instead of --project")

# Restart completely fresh Studio instance on the same --project (still with cwd != proj) and load saved scene
with McpStdioClient([studio_bin, "--project", proj, "--mcp-stdio"], cwd=other_cwd, framing="ndjson", timeout=30.0, allow_stderr=False) as c2:
    c2.initialize()
    c2.notify_initialized()
    c2.call_tool("scene.load", {"path": "scenes/pair1.flscn"})
    q2 = c2.call_tool("ecs.query_entities", {"name_contains": "Pair1Actor"})
    if q2.get("count", 0) < 1:
        raise SystemExit(f"Pair1Actor missing after restart + scene.load: {q2}")
    pos2 = q2["entities"][0]["position"]
    if abs(pos2[0] - 5.0) > 1e-3 or abs(pos2[1] - 3.0) > 1e-3 or abs(pos2[2] - (-2.0)) > 1e-3:
        raise SystemExit(f"Pair1Actor transform drifted after restart + scene.load: {pos2}")
PY
    )" || py_rc=$?
    if [[ "$py_rc" -ne 0 ]]; then
        fail_test "t3_pair1_studio_project_transform_undo_save_restart_load" "$py_out"
        return
    fi

    pass_test "t3_pair1_studio_project_transform_undo_save_restart_load"
}

# ---------------------------------------------------------------------------
# Pair 2: procedural.generate_asset + material.assign + prefab.spawn
#         + scene.save + assets.list_and_cook + render.capture_viewport
# ---------------------------------------------------------------------------
test_t3_pair2_procedural_material_prefab_save_cook_capture() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_test "t3_pair2_procedural_material_prefab_save_cook_capture" "could not resolve flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"

    local proj="${TMPD}/pair2_proj"
    mkdir -p "${proj}/assets/materials" "${proj}/assets/prefabs" "${proj}/scenes" "${proj}/build"
    cat > "${proj}/assets/materials/copper.flmat" <<'JSON'
{"base_color":[0.85,0.45,0.22],"roughness":0.25,"metallic":0.9,"emissive":[0.0,0.0,0.0]}
JSON
    cat > "${proj}/assets/prefabs/crate.flprefab" <<'JSON'
{"name":"crate","entities":[{"name":"PrefabCrate","position":[2.0,1.0,0.0],"mass":1.0}]}
JSON

    local py_out py_rc=0
    py_out="$(
        FLINT_STATE_DIR="${TMPD}/p2_state" XDG_STATE_HOME="${TMPD}/p2_xdg" \
        python3 -B - "$studio_bin" "$proj" "${SCRIPT_DIR}/../common" <<'PY' 2>&1
import os, sys
sys.path.insert(0, sys.argv[3])
from mcp_client import McpStdioClient

studio_bin, proj = sys.argv[1], sys.argv[2]

with McpStdioClient([studio_bin, "--project", proj, "--mcp-stdio"], cwd=proj, framing="cl", timeout=30.0, allow_stderr=True) as c:
    c.initialize()
    c.notify_initialized()
    tools = {t["name"] for t in c.list_tools()}

    gen = c.call_tool("procedural.generate_asset", {
        "preset": "material_orb",
        "output_glb_path": "assets/orb.glb",
        "seed": 7,
        "spawn_in_scene": True,
        "position": [0.0, 1.5, 0.0],
    })
    if gen.get("glb_bytes", 0) <= 0 or gen.get("spawned_entity_id", -1) < 0:
        raise SystemExit(f"procedural.generate_asset failed to spawn/write GLB: {gen}")

    needed = {"material.assign", "prefab.spawn", "scene.save"}
    missing = sorted(needed - tools)
    if missing:
        raise SystemExit(f"M0/M1 contract pending; missing MCP tools: {missing}")

    c.call_tool("material.assign", {"entity_id": gen["spawned_entity_id"], "material_path": "assets/materials/copper.flmat"})
    c.call_tool("prefab.spawn", {"path": "assets/prefabs/crate.flprefab", "position": [2.0, 1.0, 0.0]})
    c.call_tool("scene.save", {"path": "scenes/pair2.flscn"})

    cook = c.call_tool("assets.list_and_cook", {
        "source_dir": "assets",
        "output_flpk": "build/pair2.flpk",
        "cache_dir": "build/pair2_cache",
    })
    cr = cook.get("cook_report") or {}
    if not cr.get("ok") or cr.get("total_assets", 0) < 3:
        raise SystemExit(f"assets.list_and_cook unexpected report: {cook}")

    cap = c.call_tool("render.capture_viewport", {
        "output_path": "build/pair2_cap.png",
        "width": 320,
        "height": 180,
    })
    if cap.get("width") != 320 or cap.get("height") != 180:
        raise SystemExit(f"unexpected capture dimensions: {cap}")
PY
    )" || py_rc=$?
    if [[ "$py_rc" -ne 0 ]]; then
        fail_test "t3_pair2_procedural_material_prefab_save_cook_capture" "$py_out"
        return
    fi

    if ! validate_flpk_v3 "${proj}/build/pair2.flpk" "orb.glb" "materials/copper.flmat" "prefabs/crate.flprefab"; then
        fail_test "t3_pair2_procedural_material_prefab_save_cook_capture" "pair2.flpk missing expected assets"
        return
    fi
    if ! validate_png_dims "${proj}/build/pair2_cap.png" 320 180; then
        fail_test "t3_pair2_procedural_material_prefab_save_cook_capture" "pair2_cap.png invalid"
        return
    fi

    pass_test "t3_pair2_procedural_material_prefab_save_cook_capture"
}

# ---------------------------------------------------------------------------
# Pair 3: flint.sh new + cook_workspace (.flscn/.flmat/.flprefab/.flrig/.json/.qoa)
#         + incremental flint.sh package (2nd run cache_hits>0) + validate_flpk_v3 + clean-dir run.sh
# ---------------------------------------------------------------------------
test_t3_pair3_new_cook_incremental_package_and_clean_run() {
    local proj="${TMPD}/pair3_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t3_pair3_new_cook_incremental_package_and_clean_run" "flint.sh new failed"
        return
    fi
    cat > "${proj}/src/main.mojo" <<'MOJO'
from flint.app import AppConfig, EngineApp
from flint.assets.server import AssetServer
from flint.core.log import LOG_CORE, flush_log, log_info, start_log

def main() raises:
    _ = start_log("pair3_proj")
    var assets = AssetServer(enable_watcher=False)
    if not assets.mount_flpk("assets.flpk"):
        raise Error("failed to mount assets.flpk")
    var h = assets.load_raw("sub/meta.json")
    if not assets.raws.get_ptr(h):
        raise Error("failed to read sub/meta.json")
    var app = EngineApp(AppConfig(title="pair3_proj", headless=True, enable_crash_handler=True))
    app.startup()
    app.step_one_frame(Float32(1.0 / 60.0))
    log_info(LOG_CORE, "pair3_ok")
    _ = flush_log()
    assets.close()
    app.shutdown()
    print("[pair3_proj] Headless v5 frame loop OK")
MOJO

    mkdir -p "${proj}/assets/sub"
    python3 -B - "${proj}/assets/sub" <<'PY'
import os, sys
d = sys.argv[1]
files = {
    "level.flscn": b'{"format":"flscn","version":1,"name":"level","entities":[]}\n',
    "gold.flmat": b'{"base_color":[0.9,0.75,0.2],"metallic":0.95,"roughness":0.2}\n',
    "crate.flprefab": b'{"name":"crate","entities":[]}\n',
    "hero.flrig": b"FRIG\x01\x00\x00\x00",
    "meta.json": b'{"pair":3}\n',
    "bgm.qoa": b"qoaf\x00\x00\x00\x10" + b"\x00" * 16,
}
for name, data in files.items():
    with open(os.path.join(d, name), "wb") as f:
        f.write(data)
PY

    local d1="${TMPD}/p3_dist1"
    local d2="${TMPD}/p3_dist2"
    local out1 out2 rc=0
    out1="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" package --out "$d1" 2>&1)" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t3_pair3_new_cook_incremental_package_and_clean_run" "first package failed (rc=$rc): $out1"
        return
    fi

    sleep 1
    out2="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" package --out "$d2" 2>&1)" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t3_pair3_new_cook_incremental_package_and_clean_run" "second package failed (rc=$rc): $out2"
        return
    fi
    if ! grep -qE 'cooked=0, cache_hits=[1-9][0-9]*' <<<"$out2"; then
        fail_test "t3_pair3_new_cook_incremental_package_and_clean_run" "second package did not hit cook cache: $out2"
        return
    fi

    if ! validate_flpk_v3 "${d2}/assets.flpk" "sub/level.flscn" "sub/gold.flmat" "sub/crate.flprefab" "sub/hero.flrig" "sub/meta.json" "sub/bgm.qoa"; then
        fail_test "t3_pair3_new_cook_incremental_package_and_clean_run" "assets.flpk missing cooked entries"
        return
    fi

    local reloc="${TMPD}/p3_clean_reloc/app"
    mkdir -p "$(dirname "$reloc")"
    cp -a "$d2" "$reloc"
    rm -rf "$d1" "$d2"

    local state_dir="${TMPD}/p3_state"
    local run_out run_rc=0
    run_out="$(FLINT_STATE_DIR="$state_dir" XDG_STATE_HOME="${TMPD}/p3_xdg" run_in_clean_env "$reloc" ./run.sh --headless-smoke 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 ]] || ! grep -q "\[pair3_proj\] Headless v5 frame loop OK" <<<"$run_out"; then
        fail_test "t3_pair3_new_cook_incremental_package_and_clean_run" "clean-env run.sh failed (rc=$run_rc): $run_out"
        return
    fi
    if [[ ! -s "${state_dir}/logs/pair3_proj.log" ]]; then
        fail_test "t3_pair3_new_cook_incremental_package_and_clean_run" "missing log ${state_dir}/logs/pair3_proj.log"
        return
    fi

    pass_test "t3_pair3_new_cook_incremental_package_and_clean_run"
}

# ---------------------------------------------------------------------------
# Pair 4: project.package MCP + scene.save + isolated clean-dir run.sh
#         headless golden-image & input-replay
# ---------------------------------------------------------------------------
test_t3_pair4_mcp_project_package_scene_save_and_clean_harness() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_test "t3_pair4_mcp_project_package_scene_save_and_clean_harness" "could not resolve flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"

    local proj="${TMPD}/pair4_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t3_pair4_mcp_project_package_scene_save_and_clean_harness" "flint.sh new failed"
        return
    fi
    mkdir -p "${proj}/assets" "${proj}/scenes"

    local py_out py_rc=0
    py_out="$(
        FLINT_STATE_DIR="${TMPD}/p4_mcp_state" XDG_STATE_HOME="${TMPD}/p4_mcp_xdg" \
        python3 -B - "$studio_bin" "$proj" "${SCRIPT_DIR}/../common" <<'PY' 2>&1
import sys
sys.path.insert(0, sys.argv[3])
from mcp_client import McpStdioClient

studio_bin, proj = sys.argv[1], sys.argv[2]

with McpStdioClient([studio_bin, "--project", proj, "--mcp-stdio"], cwd=proj, framing="cl", timeout=30.0, allow_stderr=True) as c:
    c.initialize()
    c.notify_initialized()
    tools = {t["name"] for t in c.list_tools()}
    missing = sorted({"scene.save", "project.package"} - tools)
    if missing:
        raise SystemExit(f"M0/M3 contract pending; missing MCP tools: {missing}")

    c.call_tool("ecs.spawn_entity", {"name": "ShippedActor", "position": [0.0, 2.0, 0.0], "mass": 1.0})
    c.call_tool("scene.save", {"path": "assets/main.flscn"})
    c.call_tool("project.package", {"output_dir": "dist/p4_pkg"}, timeout=900.0)
PY
    )" || py_rc=$?
    if [[ "$py_rc" -ne 0 ]]; then
        fail_test "t3_pair4_mcp_project_package_scene_save_and_clean_harness" "$py_out"
        return
    fi

    local pkg_dir="${proj}/dist/p4_pkg"
    if [[ ! -x "${pkg_dir}/run.sh" || ! -f "${pkg_dir}/assets.flpk" ]]; then
        fail_test "t3_pair4_mcp_project_package_scene_save_and_clean_harness" "MCP project.package did not produce ${pkg_dir}/run.sh or assets.flpk"
        return
    fi
    if ! validate_flpk_v3 "${pkg_dir}/assets.flpk" "main.flscn"; then
        fail_test "t3_pair4_mcp_project_package_scene_save_and_clean_harness" "packaged assets.flpk failed validate_flpk_v3 (main.flscn)"
        return
    fi

    local reloc="${TMPD}/p4_isolated/bundle"
    mkdir -p "$(dirname "$reloc")"
    cp -a "$pkg_dir" "$reloc"
    rm -rf "$proj"

    if grep -Fq "/home/user/flint" "${reloc}/run.sh" "${reloc}/flint.toml" || grep -Fq "${ROOT}" "${reloc}/run.sh" "${reloc}/flint.toml"; then
        fail_test "t3_pair4_mcp_project_package_scene_save_and_clean_harness" "packaged bundle leaks repo path"
        return
    fi

    local smoke_out smoke_rc=0
    smoke_out="$(FLINT_STATE_DIR="${TMPD}/p4_run_state" XDG_STATE_HOME="${TMPD}/p4_run_xdg" run_in_clean_env "$reloc" ./run.sh --headless-smoke 2>&1)" || smoke_rc=$?
    if [[ "$smoke_rc" -ne 0 ]]; then
        fail_test "t3_pair4_mcp_project_package_scene_save_and_clean_harness" "clean-env run.sh --headless-smoke failed (rc=$smoke_rc): $smoke_out"
        return
    fi

    local shot_out="${reloc}/p4_shot.png"
    local run_out run_rc=0
    run_out="$(FLINT_STATE_DIR="${TMPD}/p4_run_state" XDG_STATE_HOME="${TMPD}/p4_run_xdg" run_in_clean_env "$reloc" ./run.sh --shots --out "$shot_out" 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 || ! -s "$shot_out" ]]; then
        fail_test "t3_pair4_mcp_project_package_scene_save_and_clean_harness" "clean-env run.sh --shots failed (rc=$run_rc): $run_out"
        return
    fi
    if ! validate_png_dims "$shot_out" 1280 720; then
        fail_test "t3_pair4_mcp_project_package_scene_save_and_clean_harness" "clean-env shot PNG failed validate_png_dims"
        return
    fi

    pass_test "t3_pair4_mcp_project_package_scene_save_and_clean_harness"
}

# ---------------------------------------------------------------------------
# Pair 5: StudioPlay (studio.pie_control) + physics.trigger_events + nav.query_path
#         + render.configure_npr_cel + render.trigger_post_fx + FLIP capture
# ---------------------------------------------------------------------------
test_t3_pair5_pie_triggers_nav_and_npr_postfx() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_test "t3_pair5_pie_triggers_nav_and_npr_postfx" "could not resolve flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"

    local ws="${TMPD}/pair5_ws"
    mkdir -p "${ws}/shots"

    local py_out py_rc=0
    py_out="$(
        FLINT_STATE_DIR="${TMPD}/p5_state" XDG_STATE_HOME="${TMPD}/p5_xdg" \
        python3 -B - "$studio_bin" "$ws" "${SCRIPT_DIR}/../common" <<'PY' 2>&1
import sys
sys.path.insert(0, sys.argv[3])
from mcp_client import McpStdioClient

studio_bin, ws = sys.argv[1], sys.argv[2]

with McpStdioClient([studio_bin, "--mcp-stdio"], cwd=ws, framing="cl", timeout=30.0, allow_stderr=True) as c:
    c.initialize()
    c.notify_initialized()
    tools = {t["name"] for t in c.list_tools()}

    # Exercise live PIE, NPR/cel, post-FX, and FLIP viewport capture first
    sp = c.call_tool("ecs.spawn_entity", {"name": "NprHero", "position": [0.0, 2.0, 0.0], "mass": 1.0})
    eid = sp["entity_id"]

    c.call_tool("render.configure_npr_cel", {
        "entity_id": eid,
        "shading_mode": "npr_cel",
        "cel_outline_width_px": 2.0,
        "cel_shadow_tint": [0.3, 0.25, 0.45],
    })
    c.call_tool("render.trigger_post_fx", {
        "impact_blend": 0.45,
        "shockwave_active": True,
        "shockwave_radius": 0.35,
        "hitstop_sec": 0.05,
    })
    q_before = c.call_tool("ecs.query_entities", {"name_contains": "NprHero"})
    hash_before = q_before["state_hash"]
    cap_a = c.call_tool("render.capture_viewport", {"output_path": "shots/p5_base.png", "width": 320, "height": 180})
    cap_same = c.call_tool("render.capture_viewport", {
        "output_path": "shots/p5_same.png",
        "reference_path": "shots/p5_base.png",
        "width": 320,
        "height": 180,
    })
    if not cap_same.get("has_flip_diff") or cap_same.get("flip_mean", 1.0) > 1e-6:
        raise SystemExit(f"identical capture expected flip_mean==0: {cap_same}")

    c.call_tool("studio.pie_control", {"action": "play"})
    stepped = c.call_tool("studio.pie_control", {"action": "step", "ticks": 12})
    if stepped.get("ticks_simulated", 0) < 12:
        raise SystemExit(f"PIE step did not advance 12 ticks: {stepped}")

    # Check M2 cross-feature MCP tools while PIE is active (physics.trigger_events + nav.query_path)
    missing = sorted({"physics.trigger_events", "nav.query_path"} - tools)
    if missing:
        raise SystemExit(f"M2 contract pending; PIE/NPR/PostFX passed, missing MCP tools: {missing}")

    trig = c.call_tool("physics.trigger_events", {})
    nav = c.call_tool("nav.query_path", {"start": [0.0, 0.0, 0.0], "goal": [4.0, 0.0, 4.0]})
    if not nav.get("ok") or not trig.get("ok"):
        raise SystemExit(f"trigger/nav query failed: trig={trig} nav={nav}")

    stopped = c.call_tool("studio.pie_control", {"action": "stop"})
    if not stopped.get("restored_exact") or stopped.get("state_hash") != hash_before:
        raise SystemExit(f"PIE stop did not restore exact pre-play state_hash: {stopped} vs {hash_before}")

    # Verify PIE_STOPPED enforcement after stop
    sc_trig = c.call_tool_expect_error("physics.trigger_events", {})
    if "pie is stopped" not in str(sc_trig.get("error", "")).lower():
        raise SystemExit(f"physics.trigger_events expected 'PIE is stopped' error: {sc_trig}")
    sc_nav = c.call_tool_expect_error("nav.query_path", {"start": [0.0, 0.0, 0.0], "goal": [4.0, 0.0, 4.0]})
    if "pie is stopped" not in str(sc_nav.get("error", "")).lower():
        raise SystemExit(f"nav.query_path expected 'PIE is stopped' error: {sc_nav}")
PY
    )" || py_rc=$?
    if [[ "$py_rc" -ne 0 ]]; then
        fail_test "t3_pair5_pie_triggers_nav_and_npr_postfx" "$py_out"
        return
    fi

    if ! validate_png_dims "${ws}/shots/p5_base.png" 320 180; then
        fail_test "t3_pair5_pie_triggers_nav_and_npr_postfx" "p5_base.png failed validate_png_dims"
        return
    fi

    pass_test "t3_pair5_pie_triggers_nav_and_npr_postfx"
}

test_t3_pair1_studio_project_transform_undo_save_restart_load
test_t3_pair2_procedural_material_prefab_save_cook_capture
test_t3_pair3_new_cook_incremental_package_and_clean_run
test_t3_pair4_mcp_project_package_scene_save_and_clean_harness
test_t3_pair5_pie_triggers_nav_and_npr_postfx

print_summary_and_exit
