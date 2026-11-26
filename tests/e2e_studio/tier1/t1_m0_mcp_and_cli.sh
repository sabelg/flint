#!/usr/bin/env bash
# AREA: m0
# Tier 1 Feature Coverage: Area B — Studio CLI, Project Root & MCP v2 (F15 + F06-F14)

set -uo pipefail

export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t1_m0_cli"
resolve_mojo_env || {
    fail_test "t1_m0_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

resolve_bin "flint-studio" "studio" || {
    fail_test "t1_m0_build_studio" "resolve_bin flint-studio failed"
    print_summary_and_exit
}
STUDIO_BIN="$RESOLVED_BIN"

has_vulkan_icd() {
    local icd_dir
    for icd_dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
        if compgen -G "${icd_dir}/*.json" >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Case 1 (F15 CLI): --headless-smoke and --shots -> valid 1280x720 PNGs
# ---------------------------------------------------------------------------
test_cli_headless_smoke_and_shots() {
    local smoke_png="${TMPD}/smoke.png"
    local shots_ws="${TMPD}/shots_ws"
    local custom_shot="${TMPD}/custom_shot.png"
    mkdir -p "$shots_ws"

    if ! has_vulkan_icd; then
        local no_vk_out no_vk_rc=0
        no_vk_out="$("$STUDIO_BIN" --headless-smoke --out "$smoke_png" 2>&1)" || no_vk_rc=$?
        if [[ "$no_vk_rc" -eq 0 ]] && grep -qiE "Vulkan.*unavailable|ERROR" <<<"$no_vk_out"; then
            echo "[INFO] Vulkan ICD unavailable on host; graceful diagnostic verified: $no_vk_out"
            pass_test "t1_m0_cli_headless_smoke_and_shots"
            return
        fi
        fail_test "t1_m0_cli_headless_smoke_and_shots" "No Vulkan ICD and unexpected exit=$no_vk_rc: $no_vk_out"
        return
    fi

    local out1 rc1=0
    out1="$(cd "$shots_ws" && "$STUDIO_BIN" --headless-smoke --out "$smoke_png" 2>&1)" || rc1=$?
    if [[ "$rc1" -ne 0 ]] || ! grep -q "Saved 1280x720 Studio frame" <<<"$out1"; then
        fail_test "t1_m0_cli_headless_smoke_and_shots" "--headless-smoke failed (rc=$rc1): $out1"
        return
    fi
    if ! validate_png_dims "$smoke_png" 1280 720; then
        fail_test "t1_m0_cli_headless_smoke_and_shots" "smoke PNG validation failed"
        return
    fi
    if ! python3 -B - "$smoke_png" <<'PY'
import struct, sys, zlib

with open(sys.argv[1], "rb") as f:
    data = f.read()
assert data[:8] == b"\x89PNG\r\n\x1a\n"
pos = 8
w = h = bit_depth = color_type = 0
idat = bytearray()
while pos + 8 <= len(data):
    ln = struct.unpack(">I", data[pos:pos + 4])[0]
    ctype = data[pos + 4:pos + 8]
    chunk = data[pos + 8:pos + 8 + ln]
    pos += 12 + ln
    if ctype == b"IHDR":
        w, h, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
    elif ctype == b"IDAT":
        idat.extend(chunk)
    elif ctype == b"IEND":
        break

assert bit_depth == 8 and color_type in (2, 6), f"unexpected PNG format: bd={bit_depth} ct={color_type}"
bpp = 3 if color_type == 2 else 4
raw = zlib.decompress(bytes(idat))
stride = w * bpp
assert len(raw) >= h * (1 + stride)
prev = bytearray(stride)
colors = set()
def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)

for y in range(h):
    off = y * (1 + stride)
    filt = raw[off]
    row = bytearray(raw[off + 1:off + 1 + stride])
    if filt == 1:
        for i in range(bpp, stride):
            row[i] = (row[i] + row[i - bpp]) & 0xFF
    elif filt == 2:
        for i in range(stride):
            row[i] = (row[i] + prev[i]) & 0xFF
    elif filt == 3:
        for i in range(stride):
            a = row[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + ((a + prev[i]) >> 1)) & 0xFF
    elif filt == 4:
        for i in range(stride):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + paeth(a, b, c)) & 0xFF
    prev = row
    if y % 6 == 0:
        for x in range(0, w, 10):
            i = x * bpp
            colors.add(bytes(row[i:i + 3]))

assert len(colors) > 1, f"smoke PNG appears blank/uniform: unique_sampled_colors={len(colors)}"
PY
    then
        fail_test "t1_m0_cli_headless_smoke_and_shots" "smoke PNG pixel diversity check failed"
        return
    fi

    local out2 rc2=0
    out2="$(cd "$shots_ws" && "$STUDIO_BIN" --shots 2>&1)" || rc2=$?
    if [[ "$rc2" -ne 0 ]] || ! grep -q "Saved 1280x720 Studio frame" <<<"$out2"; then
        fail_test "t1_m0_cli_headless_smoke_and_shots" "--shots default failed (rc=$rc2): $out2"
        return
    fi
    if ! validate_png_dims "${shots_ws}/build/shots/flint_studio.png" 1280 720; then
        fail_test "t1_m0_cli_headless_smoke_and_shots" "default shots PNG validation failed"
        return
    fi

    local out3 rc3=0
    out3="$(cd "$shots_ws" && "$STUDIO_BIN" --shots --out "$custom_shot" 2>&1)" || rc3=$?
    if [[ "$rc3" -ne 0 ]] || ! validate_png_dims "$custom_shot" 1280 720; then
        fail_test "t1_m0_cli_headless_smoke_and_shots" "--shots --out PNG validation failed (rc=$rc3)"
        return
    fi

    pass_test "t1_m0_cli_headless_smoke_and_shots"
}

# ---------------------------------------------------------------------------
# Case 2 (F15 MCP v2): Content-Length + NDJSON handshake, byte-0 clean, silent notify
# ---------------------------------------------------------------------------
test_mcp_handshake_cl_and_ndjson_byte0() {
    local ws="${TMPD}/mcp_ws"
    mkdir -p "$ws"

    if ! PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - "$STUDIO_BIN" "$ws" <<'PY'
import sys
from mcp_client import McpStdioClient

bin_path, ws = sys.argv[1], sys.argv[2]
expected_baseline = {
    "ecs.query_entities",
    "ecs.spawn_entity",
    "ecs.patch_components",
    "ecs.despawn_entity",
    "render.capture_viewport",
    "render.inspect_gpu_frame",
    "assets.list_and_cook",
    "procedural.generate_asset",
    "physics.step_deterministic",
    "physics.raycast_query",
    "studio.pie_control",
    "studio.undo_redo",
    "render.configure_npr_cel",
    "render.trigger_post_fx",
}

for framing in ("cl", "ndjson"):
    with McpStdioClient([bin_path, "--mcp-stdio"], cwd=ws, framing=framing, allow_stderr=False) as client:
        init_res = client.initialize()
        assert init_res.get("protocolVersion") == "2024-11-05", f"bad protocolVersion: {init_res!r}"
        client.notify_initialized()
        ping_resp = client.request("ping", {})
        assert "error" not in ping_resp and ping_resp.get("result") == {}, f"bad ping: {ping_resp!r}"
        tools = client.list_tools()
        names = {t.get("name") for t in tools}
        missing = sorted(expected_baseline - names)
        assert not missing, f"missing baseline tools in {framing}: {missing!r}"
PY
    then
        fail_test "t1_m0_mcp_handshake_cl_and_ndjson_byte0" "Content-Length or NDJSON MCP handshake failed"
        return
    fi

    pass_test "t1_m0_mcp_handshake_cl_and_ndjson_byte0"
}

# ---------------------------------------------------------------------------
# Case 3 (F06-F08, F15): Baseline MCP PIE, raycast, capture_viewport, undo/redo, cook
# ---------------------------------------------------------------------------
test_mcp_baseline_pie_raycast_capture_undo() {
    local ws="${TMPD}/mcp_workflow_ws"
    mkdir -p "${ws}/assets" "${ws}/shots" "${ws}/out"

    if ! PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - "$STUDIO_BIN" "$ws" <<'PY'
import os, sys
from mcp_client import McpStdioClient

bin_path, ws = sys.argv[1], sys.argv[2]

with McpStdioClient([bin_path, "--mcp-stdio"], cwd=ws, framing="cl", allow_stderr=False) as client:
    client.initialize()
    client.notify_initialized()

    # Spawn dynamic cube at y=4
    sp = client.call_tool("ecs.spawn_entity", {
        "name": "DropCube",
        "position": [0.0, 4.0, 0.0],
        "mass": 1.0,
        "base_color": [0.9, 0.2, 0.2],
        "roughness": 0.3,
        "metallic": 0.1,
    })
    eid = sp["entity_id"]
    h0 = sp["state_hash"]

    # Patch + undo + redo parity
    p_res = client.call_tool("ecs.patch_components", {
        "entity_id": eid,
        "position": [0.0, 4.5, 0.0],
        "roughness": 0.7,
    })
    h_patched = p_res["state_hash"]
    assert h_patched != h0

    u_res = client.call_tool("studio.undo_redo", {"action": "undo", "steps": 1})
    assert u_res["applied_steps"] == 1
    assert u_res["state_hash"] == h0

    r_res = client.call_tool("studio.undo_redo", {"action": "redo", "steps": 1})
    assert r_res["applied_steps"] == 1
    assert r_res["state_hash"] == h_patched

    # Reset back to [0, 4, 0] via undo for capture/PIE
    client.call_tool("studio.undo_redo", {"action": "undo", "steps": 1})

    cap1 = client.call_tool("render.capture_viewport", {
        "output_path": "shots/before.png",
        "width": 320,
        "height": 180,
    })
    assert cap1["width"] == 320 and cap1["height"] == 180
    assert os.path.isfile(os.path.join(ws, "shots/before.png"))

    play_res = client.call_tool("studio.pie_control", {"action": "play"})
    assert play_res["pie_mode"] == "playing", f"unexpected pie_mode: {play_res!r}"
    step_res = client.call_tool("studio.pie_control", {"action": "step", "ticks": 30})
    assert step_res["ticks_simulated"] >= 30, f"unexpected ticks_simulated: {step_res!r}"

    cap2 = client.call_tool("render.capture_viewport", {
        "output_path": "shots/during.png",
        "reference_path": "shots/before.png",
        "width": 320,
        "height": 180,
    })
    assert cap2["has_flip_diff"] is True and cap2["flip_mean"] > 0.0, f"expected fall pixel diff: {cap2!r}"

    stop_res = client.call_tool("studio.pie_control", {"action": "stop"})
    assert stop_res["pie_mode"] == "stopped" and stop_res["restored_exact"] is True and stop_res["state_hash"] == h0

    rc_res = client.call_tool("physics.raycast_query", {
        "origin": [0.0, 10.0, 0.0],
        "direction": [0.0, -1.0, 0.0],
        "max_distance": 30.0,
    })
    assert rc_res["hit"] is True and abs(rc_res["point"][1] - 4.5) < 0.25, f"unexpected raycast: {rc_res!r}"

    gpu = client.call_tool("render.inspect_gpu_frame", {})
    assert gpu.get("backend") in ("HEADLESS-STUDIO", "VULKAN-AAA"), f"unexpected inspect_gpu_frame: {gpu!r}"

    gen = client.call_tool("procedural.generate_asset", {
        "preset": "material_orb",
        "output_glb_path": "assets/orb.glb",
        "spawn_in_scene": True,
        "position": [2.0, 1.0, 0.0],
    })
    assert os.path.isfile(os.path.join(ws, "assets/orb.glb")) and gen["spawned_entity_id"] >= 0

    cook = client.call_tool("assets.list_and_cook", {
        "source_dir": "assets",
        "output_flpk": "out/pack.flpk",
    })
    assert cook["cook_report"]["total_assets"] >= 1 and os.path.isfile(os.path.join(ws, "out/pack.flpk"))
PY
    then
        fail_test "t1_m0_mcp_baseline_pie_raycast_capture_undo" "Baseline MCP workflow failed"
        return
    fi

    if ! validate_png_dims "${ws}/shots/before.png" 320 180 || ! validate_png_dims "${ws}/shots/during.png" 320 180; then
        fail_test "t1_m0_mcp_baseline_pie_raycast_capture_undo" "captured PNG dims validation failed"
        return
    fi
    if ! validate_flpk_v3 "${ws}/out/pack.flpk" "orb.glb"; then
        fail_test "t1_m0_mcp_baseline_pie_raycast_capture_undo" "cooked FLPK validation failed"
        return
    fi

    pass_test "t1_m0_mcp_baseline_pie_raycast_capture_undo"
}

# ---------------------------------------------------------------------------
# Case 4 (F15 M0 CLI): --project <dir> & <dir>/scenes/ rooting
# ---------------------------------------------------------------------------
test_cli_project_root_and_scenes_dir() {
    local proj="${TMPD}/proj_m0"
    local caller_ws="${TMPD}/caller_cwd"
    mkdir -p "${proj}/assets" "${proj}/scenes" "$caller_ws"
    printf '[project]\nname = "m0_proj"\nentry_scene = "assets/main.flscn"\n' > "${proj}/flint.toml"

    local err_msg=""
    if ! err_msg="$(PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - "$STUDIO_BIN" "$proj" "$caller_ws" "$ROOT" 2>&1 <<'PY'
import os, sys
from mcp_client import McpStdioClient

bin_path, proj, caller_ws, repo_root = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sentinel = os.path.join(repo_root, "build", "studio_scene.flscn")
had_sentinel = os.path.exists(sentinel)

with McpStdioClient([bin_path, "--project", proj, "--mcp-stdio"], cwd=caller_ws, framing="cl", allow_stderr=False) as client:
    client.initialize()
    client.notify_initialized()
    tools = {t.get("name") for t in client.list_tools()}
    if "scene.save" not in tools:
        raise SystemExit("M0 tool scene.save not exposed yet")
    client.call_tool("ecs.spawn_entity", {"name": "ProjBox", "position": [1.0, 2.0, 3.0]})
    client.call_tool("scene.save", {"path": "scenes/level1.flscn"})
    client.call_tool("render.capture_viewport", {"output_path": "shots/proj.png", "width": 160, "height": 90})

if not os.path.isfile(os.path.join(proj, "scenes", "level1.flscn")):
    raise SystemExit("scene.save did not write inside <project>/scenes/level1.flscn")
if not os.path.isfile(os.path.join(proj, "shots", "proj.png")):
    raise SystemExit("render.capture_viewport did not write inside <project>/shots/proj.png")
if os.path.exists(os.path.join(caller_ws, "scenes", "level1.flscn")):
    raise SystemExit("wrote into caller_cwd instead of --project dir")
if not had_sentinel and os.path.exists(sentinel):
    raise SystemExit("polluted repo build/studio_scene.flscn")
PY
    )"; then
        fail_test "t1_m0_cli_project_root_and_scenes_dir" "$err_msg"
        return
    fi

    pass_test "t1_m0_cli_project_root_and_scenes_dir"
}

# ---------------------------------------------------------------------------
# Case 5 (F15 M0 MCP): 4 M0 tools (scene.save/load, scene.set_entity_enabled, entity.set_transform)
# ---------------------------------------------------------------------------
test_mcp_m0_tools_catalog_and_roundtrip() {
    local proj="${TMPD}/proj_m0_tools"
    mkdir -p "${proj}/assets" "${proj}/scenes"
    printf '[project]\nname = "m0_tools"\nentry_scene = "assets/main.flscn"\n' > "${proj}/flint.toml"

    local err_msg=""
    if ! err_msg="$(PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - "$STUDIO_BIN" "$proj" 2>&1 <<'PY'
import os, sys
from mcp_client import McpStdioClient

bin_path, proj = sys.argv[1], sys.argv[2]
m0_tools = {"scene.save", "scene.load", "scene.set_entity_enabled", "entity.set_transform"}

with McpStdioClient([bin_path, "--project", proj, "--mcp-stdio"], cwd=proj, framing="cl", allow_stderr=False) as client:
    client.initialize()
    client.notify_initialized()
    names = {t.get("name") for t in client.list_tools()}
    missing = sorted(m0_tools - names)
    if missing:
        raise SystemExit(f"missing M0 MCP tools: {missing!r}")

    sp = client.call_tool("ecs.spawn_entity", {"name": "M0Target", "position": [0.0, 1.0, 0.0]})
    eid = sp["entity_id"]

    t_res = client.call_tool("entity.set_transform", {
        "entity_id": eid,
        "position": [3.0, 2.0, -1.0],
        "rotation": [0.0, 0.7071068, 0.0, 0.7071068],
        "scale": [2.0, 2.0, 2.0],
    })
    h_xform = t_res["state_hash"]
    assert h_xform != sp["state_hash"], "entity.set_transform did not update state_hash"

    client.call_tool("scene.set_entity_enabled", {"entity_id": eid, "enabled": False})
    q_vis = client.call_tool("ecs.query_entities", {"visible_only": True})
    assert all(e["entity_id"] != eid for e in q_vis["entities"]), "disabled entity appeared in visible_only query"

    # Undo/redo parity on M0 tools
    client.call_tool("studio.undo_redo", {"action": "undo", "steps": 1})
    q_vis2 = client.call_tool("ecs.query_entities", {"visible_only": True})
    assert any(e["entity_id"] == eid for e in q_vis2["entities"]), "undo did not re-enable entity"
    client.call_tool("studio.undo_redo", {"action": "redo", "steps": 1})
    q_vis3 = client.call_tool("ecs.query_entities", {"visible_only": True})
    assert all(e["entity_id"] != eid for e in q_vis3["entities"]), "redo did not re-disable entity in visible_only query"

    client.call_tool("scene.set_entity_enabled", {"entity_id": eid, "enabled": True})

    # Second entity left disabled across save/restart/load to prove enabled=False persists
    sp2 = client.call_tool("ecs.spawn_entity", {"name": "M0Disabled", "position": [-2.0, 3.0, 1.0]})
    eid2 = sp2["entity_id"]
    client.call_tool("scene.set_entity_enabled", {"entity_id": eid2, "enabled": False})

    client.call_tool("scene.save", {"path": "scenes/rt.flscn"})
    client.call_tool("scene.save", {"path": "scenes/rt.flscnb"})
    q_saved = client.call_tool("ecs.query_entities", {})
    saved_hash = q_saved["state_hash"]

def verify_reloaded(client_x, label):
    q_all = client_x.call_tool("ecs.query_entities", {})
    assert len(q_all["entities"]) == len(q_saved["entities"]), f"{label}: entity count mismatch"
    assert q_all["state_hash"] == saved_hash, f"{label}: state_hash mismatch ({q_all['state_hash']} != {saved_hash})"
    targets = [e for e in q_all["entities"] if e.get("name") == "M0Target"]
    assert len(targets) == 1, f"{label}: missing M0Target: {q_all!r}"
    tgt = targets[0]
    assert abs(tgt["position"][0] - 3.0) < 1e-3 and abs(tgt["position"][1] - 2.0) < 1e-3 and abs(tgt["position"][2] + 1.0) < 1e-3, f"{label}: position mismatch: {tgt!r}"
    assert abs(tgt["rotation"][1] - 0.7071068) < 1e-3 and abs(tgt["rotation"][3] - 0.7071068) < 1e-3, f"{label}: rotation mismatch: {tgt!r}"
    assert all(abs(tgt["scale"][i] - 2.0) < 1e-3 for i in range(3)), f"{label}: scale mismatch: {tgt!r}"
    disabled_ents = [e for e in q_all["entities"] if e.get("name") == "M0Disabled"]
    assert len(disabled_ents) == 1 and disabled_ents[0].get("visible") is False, f"{label}: M0Disabled did not preserve enabled=False: {disabled_ents!r}"
    q_vis_only = client_x.call_tool("ecs.query_entities", {"visible_only": True})
    assert all(e.get("name") != "M0Disabled" for e in q_vis_only["entities"]), f"{label}: M0Disabled appeared in visible_only query"

# Fresh Studio restart & reload
with McpStdioClient([bin_path, "--project", proj, "--mcp-stdio"], cwd=proj, framing="ndjson", allow_stderr=False) as client2:
    client2.initialize()
    client2.notify_initialized()
    client2.call_tool("scene.load", {"path": "scenes/rt.flscn"})
    verify_reloaded(client2, "flscn")

    client2.call_tool("ecs.spawn_entity", {"name": "TempJunk"})
    client2.call_tool("scene.load", {"path": "scenes/rt.flscnb"})
    q_b = client2.call_tool("ecs.query_entities", {})
    assert not any(e.get("name") == "TempJunk" for e in q_b["entities"])
    verify_reloaded(client2, "flscnb")
PY
    )"; then
        fail_test "t1_m0_mcp_m0_tools_catalog_and_roundtrip" "$err_msg"
        return
    fi

    pass_test "t1_m0_mcp_m0_tools_catalog_and_roundtrip"
}

test_cli_headless_smoke_and_shots
test_mcp_handshake_cl_and_ndjson_byte0
test_mcp_baseline_pie_raycast_capture_undo
test_cli_project_root_and_scenes_dir
test_mcp_m0_tools_catalog_and_roundtrip

print_summary_and_exit
