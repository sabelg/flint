#!/usr/bin/env bash
# AREA: m0
# Tier 2 Boundary & Negative Cases: Area B — Studio CLI, Sandbox & MCP v2 (F15)

set -uo pipefail

export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t2_m0_cli"
resolve_mojo_env || {
    fail_test "t2_m0_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

resolve_bin "flint-studio" "studio" || {
    fail_test "t2_m0_build_studio" "resolve_bin flint-studio failed"
    print_summary_and_exit
}
STUDIO_BIN="$RESOLVED_BIN"

# ---------------------------------------------------------------------------
# Case 1 (F15 MCP): JSON-RPC protocol error codes (-32700, -32600, -32601, -32602, isError)
# ---------------------------------------------------------------------------
test_mcp_jsonrpc_protocol_errors() {
    local ws="${TMPD}/proto_ws"
    mkdir -p "$ws"

    if ! PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - "$STUDIO_BIN" "$ws" <<'PY'
import subprocess, sys
from mcp_client import McpStdioClient, encode_frame, parse_frames

bin_path, ws = sys.argv[1], sys.argv[2]

with McpStdioClient([bin_path, "--mcp-stdio"], cwd=ws, framing="cl", allow_stderr=False) as c:
    c.initialize()
    c.notify_initialized()

    # -32700: invalid JSON inside valid Content-Length frame
    bad_json = b"{not_json"
    c.expect_protocol_error(f"Content-Length: {len(bad_json)}\r\n\r\n".encode("ascii") + bad_json, -32700)

    # -32600: wrong jsonrpc version / missing method
    c.expect_protocol_error({"jsonrpc": "1.0", "id": 9001, "method": "ping"}, -32600)
    c.expect_protocol_error({"jsonrpc": "2.0", "id": 9002}, -32600)

    # -32601: unknown method
    c.expect_protocol_error({"jsonrpc": "2.0", "id": 9003, "method": "studio/nonexistent"}, -32601)

    # -32602: tools/call missing params.name
    c.expect_protocol_error({"jsonrpc": "2.0", "id": 9004, "method": "tools/call", "params": {}}, -32602)

    # Unknown tool -> tool-level isError=True, structuredContent.ok=False
    err_sc = c.call_tool_expect_error("does.not.exist", {})
    assert err_sc.get("ok") is False

    # Server stays healthy after all errors
    ok_ping = c.request("ping", {})
    assert "error" not in ok_ping and ok_ping.get("result") == {}

# Malformed framing headers (zero/negative/oversized/non-numeric Content-Length, >8KiB header, bad header, JSON array) -> -32600 / -32700
for raw_bad, exp_code in [
    (b"Content-Length: 0\r\n\r\n", -32600),
    (b"Content-Length: -5\r\n\r\n{}", -32600),
    (b"Content-Length: 999999999\r\n\r\n{}", -32600),
    (b"Content-Length: abc\r\n\r\n{}", -32600),
    (b"Content-Length: 5\r\nX-Overflow: " + b"A" * 9000, -32600),
    (b"BadHeader: 5\r\n\r\nhello", -32700),
    (b"Content-Length: 2\r\n\r\n[]", -32700),
]:
    proc = subprocess.run([bin_path, "--mcp-stdio"], input=raw_bad, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ws, timeout=15)
    assert proc.returncode == 0, f"crashed on malformed header {raw_bad[:40]!r}: rc={proc.returncode}"
    assert not proc.stderr, f"unexpected stderr: {proc.stderr!r}"
    frames = parse_frames(proc.stdout, framing="auto")
    assert len(frames) >= 1 and frames[0].get("error", {}).get("code") == exp_code, f"expected {exp_code}, got {frames!r}"
PY
    then
        fail_test "t2_m0_mcp_jsonrpc_protocol_errors" "JSON-RPC protocol error assertions failed"
        return
    fi

    pass_test "t2_m0_mcp_jsonrpc_protocol_errors"
}

# ---------------------------------------------------------------------------
# Case 2 (F15 Sandbox): Path traversal, symlinks, outside absolute paths & backslash
# ---------------------------------------------------------------------------
test_mcp_baseline_sandbox_traversal_and_symlinks() {
    local ws="${TMPD}/sandbox_ws"
    local outside="${TMPD}/outside_dir"
    mkdir -p "${ws}/assets" "${ws}/shots" "$outside"
    ln -s "$outside" "${ws}/escape_link"
    ln -s "${outside}/dangling_target.png" "${ws}/shots/dangling.png"

    if ! PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - "$STUDIO_BIN" "$ws" "$outside" <<'PY'
import os, sys
from mcp_client import McpStdioClient

bin_path, ws, outside = sys.argv[1], sys.argv[2], sys.argv[3]
abs_escape = os.path.join(outside, "evil.png")
dangling_target = os.path.join(outside, "dangling_target.png")
bad_paths = [
    "../outside.png",
    "shots/../../outside.png",
    abs_escape,
    "escape_link/evil.png",
    "bad\\path.png",
]

with McpStdioClient([bin_path, "--mcp-stdio"], cwd=ws, framing="cl", allow_stderr=False) as c:
    c.initialize()
    c.notify_initialized()
    c.call_tool_expect_error("render.capture_viewport", {"output_path": "", "width": 128, "height": 96})
    c.call_tool_expect_error("render.capture_viewport", {"output_path": "shots/dangling.png", "width": 128, "height": 96})
    c.call_tool_expect_error("assets.list_and_cook", {"source_dir": "assets", "output_flpk": ""})
    for p in bad_paths:
        c.call_tool_expect_error("render.capture_viewport", {"output_path": p, "width": 128, "height": 96})
        c.call_tool_expect_error("render.capture_viewport", {"output_path": "shots/ok.png", "reference_path": p, "width": 128, "height": 96})
        c.call_tool_expect_error("procedural.generate_asset", {"preset": "material_orb", "output_glb_path": p})
        c.call_tool_expect_error("assets.list_and_cook", {"source_dir": "assets", "output_flpk": p})
        c.call_tool_expect_error("assets.list_and_cook", {"source_dir": p, "output_flpk": "out.flpk"})
        c.call_tool_expect_error("assets.list_and_cook", {"source_dir": "assets", "output_flpk": "out.flpk", "cache_dir": p})
        c.call_tool_expect_error("physics.step_deterministic", {"ticks": 2, "output_path": p})

assert not os.path.exists(abs_escape), "symlink or traversal created outside file!"
assert not os.path.exists(dangling_target), "dangling leaf symlink wrote outside file!"
assert not os.path.exists(os.path.join(os.path.dirname(ws), "outside.png"))
PY
    then
        fail_test "t2_m0_mcp_baseline_sandbox_traversal_and_symlinks" "Baseline sandbox allowed escape path"
        return
    fi

    pass_test "t2_m0_mcp_baseline_sandbox_traversal_and_symlinks"
}

# ---------------------------------------------------------------------------
# Case 3 (F15 Existing MCP): Negative params on patch/despawn/npr/pie/undo/raycast/cook
# ---------------------------------------------------------------------------
test_mcp_existing_tools_negative_boundaries() {
    local ws="${TMPD}/neg_tools_ws"
    mkdir -p "${ws}/assets" "${ws}/shots"

    if ! PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - "$STUDIO_BIN" "$ws" <<'PY'
import sys
from mcp_client import McpStdioClient

bin_path, ws = sys.argv[1], sys.argv[2]

with McpStdioClient([bin_path, "--mcp-stdio"], cwd=ws, framing="cl", allow_stderr=False) as c:
    c.initialize()
    c.notify_initialized()

    # Empty undo stack returns applied_steps=0 cleanly
    u0 = c.call_tool("studio.undo_redo", {"action": "undo", "steps": 1})
    assert u0["applied_steps"] == 0

    sp = c.call_tool("ecs.spawn_entity", {"name": "LiveEnt", "position": [0.0, 2.0, 0.0]})
    eid = sp["entity_id"]

    # Invalid entity IDs & malformed params
    for bad_id in (-1, 999999):
        c.call_tool_expect_error("ecs.patch_components", {"entity_id": bad_id, "mass": 1.0})
        c.call_tool_expect_error("ecs.despawn_entity", {"entity_id": bad_id})
        c.call_tool_expect_error("render.configure_npr_cel", {"entity_id": bad_id, "shading_mode": "cel"})

    c.call_tool_expect_error("ecs.patch_components", {"entity_id": eid, "position": [1.0, 2.0]})
    c.call_tool_expect_error("ecs.patch_components", {"entity_id": eid, "mass": "not_a_number"})
    c.call_tool_expect_error("render.configure_npr_cel", {"entity_id": eid, "shading_mode": "bogus_mode"})
    c.call_tool_expect_error("render.configure_npr_cel", {"entity_id": eid, "shading_mode": 99})
    c.call_tool_expect_error("studio.pie_control", {"action": "explode"})
    c.call_tool_expect_error("studio.undo_redo", {"action": "clear"})
    c.call_tool_expect_error("physics.raycast_query", {"origin": [1.0, 2.0], "direction": [0.0, -1.0, 0.0]})
    c.call_tool_expect_error("assets.list_and_cook", {"source_dir": "assets"})

    # Despawned entity rejected afterwards
    c.call_tool("ecs.despawn_entity", {"entity_id": eid})
    c.call_tool_expect_error("ecs.patch_components", {"entity_id": eid, "roughness": 0.2})

    # Tiny capture dimensions clamped >= 64 and missing reference_path safe
    cap = c.call_tool("render.capture_viewport", {
        "output_path": "shots/clamped.png",
        "reference_path": "shots/does_not_exist.png",
        "width": 1,
        "height": 1,
    })
    assert cap["width"] >= 64 and cap["height"] >= 64
    assert cap["has_flip_diff"] is False
PY
    then
        fail_test "t2_m0_mcp_existing_tools_negative_boundaries" "Existing MCP negative parameter boundary failed"
        return
    fi

    pass_test "t2_m0_mcp_existing_tools_negative_boundaries"
}

# ---------------------------------------------------------------------------
# Case 4 (F15 CLI): Empty stdin --mcp-stdio, missing VK ICD, read-only out dir
# ---------------------------------------------------------------------------
test_cli_empty_stdin_vk_missing_and_ro_out() {
    local ws="${TMPD}/cli_neg_ws"
    mkdir -p "$ws"

    local empty_out="${TMPD}/empty.stdout"
    local empty_err="${TMPD}/empty.stderr"
    local rc_empty=0
    (cd "$ws" && "$STUDIO_BIN" --mcp-stdio </dev/null >"$empty_out" 2>"$empty_err") || rc_empty=$?
    if [[ "$rc_empty" -ne 0 ]] || [[ -s "$empty_out" ]] || [[ -s "$empty_err" ]]; then
        fail_test "t2_m0_cli_empty_stdin_vk_missing_and_ro_out" "empty stdin mcp-stdio rc=$rc_empty stdout=$(wc -c <"$empty_out") stderr=$(wc -c <"$empty_err")"
        return
    fi

    # Missing Vulkan ICD must fail gracefully without writing PNG
    local missing_png="${TMPD}/missing_vk.png"
    local fake_icd="${TMPD}/missing_icd.json"
    local vk_out vk_rc=0
    vk_out="$(cd "$ws" && VK_ICD_FILENAMES="$fake_icd" "$STUDIO_BIN" --headless-smoke --out "$missing_png" 2>&1)" || vk_rc=$?
    if [[ -f "$missing_png" ]]; then
        fail_test "t2_m0_cli_empty_stdin_vk_missing_and_ro_out" "wrote PNG despite missing VK_ICD_FILENAMES"
        return
    fi
    if ! grep -qiE "Vulkan|ERROR" <<<"$vk_out"; then
        fail_test "t2_m0_cli_empty_stdin_vk_missing_and_ro_out" "missing Vulkan/ERROR diagnostic on missing VK_ICD_FILENAMES (rc=$vk_rc): $vk_out"
        return
    fi

    # Read-only output directory (skip only if running as uid 0)
    if [[ "$(id -u)" -ne 0 ]]; then
        local ro_dir="${TMPD}/ro_dir"
        mkdir -p "$ro_dir"
        chmod 0555 "$ro_dir"
        local ro_png="${ro_dir}/cant_write.png"
        local ro_out ro_rc=0
        ro_out="$(cd "$ws" && "$STUDIO_BIN" --headless-smoke --out "$ro_png" 2>&1)" || ro_rc=$?
        chmod 0755 "$ro_dir"
        if [[ -f "$ro_png" ]]; then
            fail_test "t2_m0_cli_empty_stdin_vk_missing_and_ro_out" "unexpectedly wrote PNG into 0555 dir (rc=$ro_rc)"
            return
        fi
        if [[ "$ro_rc" -eq 0 ]] || grep -q "Saved 1280x720" <<<"$ro_out"; then
            fail_test "t2_m0_cli_empty_stdin_vk_missing_and_ro_out" "failed write into 0555 dir exited rc=$ro_rc or falsely claimed Saved 1280x720: $ro_out"
            return
        fi
    fi

    pass_test "t2_m0_cli_empty_stdin_vk_missing_and_ro_out"
}

# ---------------------------------------------------------------------------
# Case 5 (F15 M0 CLI): --project nonexistent/file/missing-arg & --out collisions
# ---------------------------------------------------------------------------
test_cli_invalid_project_and_out_args() {
    local ws="${TMPD}/cli_bad_proj"
    local existing_dir="${TMPD}/existing_out_dir"
    local file_proj="${TMPD}/not_a_dir.txt"
    mkdir -p "$ws" "$existing_dir"
    echo "not a dir" > "$file_proj"

    local rc=0
    if (cd "$ws" && "$STUDIO_BIN" --project "${TMPD}/does_not_exist" --mcp-stdio </dev/null >/dev/null 2>&1); then
        fail_test "t2_m0_cli_invalid_project_and_out_args" "--project /nonexistent unexpectedly exited 0"
        return
    fi
    if (cd "$ws" && "$STUDIO_BIN" --project "$file_proj" --mcp-stdio </dev/null >/dev/null 2>&1); then
        fail_test "t2_m0_cli_invalid_project_and_out_args" "--project <file> unexpectedly exited 0"
        return
    fi
    if (cd "$ws" && "$STUDIO_BIN" --project >/dev/null 2>&1); then
        fail_test "t2_m0_cli_invalid_project_and_out_args" "--project with missing value unexpectedly exited 0"
        return
    fi
    if (cd "$ws" && "$STUDIO_BIN" --headless-smoke --out >/dev/null 2>&1); then
        fail_test "t2_m0_cli_invalid_project_and_out_args" "--out with missing value unexpectedly exited 0"
        return
    fi
    if (cd "$ws" && "$STUDIO_BIN" --headless-smoke --out "$existing_dir" >/dev/null 2>&1); then
        fail_test "t2_m0_cli_invalid_project_and_out_args" "--out <existing_dir> unexpectedly exited 0"
        return
    fi

    pass_test "t2_m0_cli_invalid_project_and_out_args"
}

# ---------------------------------------------------------------------------
# Case 6 (F15 M0 MCP): scene.save/load traversal, symlinks & missing/corrupt files
# ---------------------------------------------------------------------------
test_mcp_m0_scene_load_and_traversal_boundaries() {
    local proj="${TMPD}/proj_scene_neg"
    local outside="${TMPD}/proj_outside"
    mkdir -p "${proj}/scenes" "${proj}/assets" "$outside"
    printf '[project]\nname = "scene_neg"\nentry_scene = "assets/main.flscn"\n' > "${proj}/flint.toml"
    ln -s "$outside" "${proj}/scenes/escape_link"

    : > "${proj}/scenes/empty.flscn"
    printf '{not_valid_json' > "${proj}/scenes/corrupt.flscn"
    printf 'BAD_HDR' > "${proj}/scenes/trunc.flscnb"

    local err_msg=""
    if ! err_msg="$(PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - "$STUDIO_BIN" "$proj" "$outside" 2>&1 <<'PY'
import os, sys
from mcp_client import McpStdioClient

bin_path, proj, outside = sys.argv[1], sys.argv[2], sys.argv[3]
abs_evil = os.path.join(outside, "evil.flscn")

with McpStdioClient([bin_path, "--project", proj, "--mcp-stdio"], cwd=proj, framing="cl", allow_stderr=False) as c:
    c.initialize()
    c.notify_initialized()
    tools = {t.get("name") for t in c.list_tools()}
    if not {"scene.save", "scene.load"}.issubset(tools):
        raise SystemExit("M0 tools scene.save / scene.load not exposed yet")

    for bad_p in ("", "../evil.flscn", "scenes/../../evil.flscn", abs_evil, "scenes/escape_link/evil.flscn", "scenes\\evil.flscn"):
        c.call_tool_expect_error("scene.save", {"path": bad_p})
        c.call_tool_expect_error("scene.load", {"path": bad_p})

    for bad_file in ("scenes/missing.flscn", "scenes/empty.flscn", "scenes/corrupt.flscn", "scenes/trunc.flscnb"):
        c.call_tool_expect_error("scene.load", {"path": bad_file})

    q = c.call_tool("ecs.query_entities", {})
    assert isinstance(q.get("entities"), list)

assert not os.path.exists(abs_evil)
PY
    )"; then
        fail_test "t2_m0_mcp_m0_scene_load_and_traversal_boundaries" "$err_msg"
        return
    fi

    pass_test "t2_m0_mcp_m0_scene_load_and_traversal_boundaries"
}

# ---------------------------------------------------------------------------
# Case 7 (F15 M0 MCP): entity.set_transform & scene.set_entity_enabled negative params
# ---------------------------------------------------------------------------
test_mcp_m0_entity_transform_and_enable_negative() {
    local proj="${TMPD}/proj_ent_neg"
    mkdir -p "${proj}/scenes" "${proj}/assets"

    local err_msg=""
    if ! err_msg="$(PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - "$STUDIO_BIN" "$proj" 2>&1 <<'PY'
import sys
from mcp_client import McpStdioClient

bin_path, proj = sys.argv[1], sys.argv[2]

with McpStdioClient([bin_path, "--project", proj, "--mcp-stdio"], cwd=proj, framing="cl", allow_stderr=False) as c:
    c.initialize()
    c.notify_initialized()
    tools = {t.get("name") for t in c.list_tools()}
    if not {"entity.set_transform", "scene.set_entity_enabled"}.issubset(tools):
        raise SystemExit("M0 tools entity.set_transform / scene.set_entity_enabled not exposed yet")

    sp = c.call_tool("ecs.spawn_entity", {"name": "SafeEntity", "position": [1.0, 2.0, 3.0]})
    eid = sp["entity_id"]

    for bad_id in (-1, 999999):
        c.call_tool_expect_error("entity.set_transform", {"entity_id": bad_id, "position": [0.0, 0.0, 0.0]})
        c.call_tool_expect_error("scene.set_entity_enabled", {"entity_id": bad_id, "enabled": False})

    c.call_tool_expect_error("entity.set_transform", {"position": [0.0, 0.0, 0.0]})
    c.call_tool_expect_error("entity.set_transform", {"entity_id": eid, "position": [1.0, 2.0]})
    c.call_tool_expect_error("entity.set_transform", {"entity_id": eid, "position": ["bad", 0.0, 0.0]})
    c.call_tool_expect_error("scene.set_entity_enabled", {"entity_id": eid})
    c.call_tool_expect_error("scene.set_entity_enabled", {"enabled": True})

    q = c.call_tool("ecs.query_entities", {})
    assert any(e["entity_id"] == eid and abs(e["position"][0] - 1.0) < 1e-4 for e in q["entities"])
PY
    )"; then
        fail_test "t2_m0_mcp_m0_entity_transform_and_enable_negative" "$err_msg"
        return
    fi

    pass_test "t2_m0_mcp_m0_entity_transform_and_enable_negative"
}

# ---------------------------------------------------------------------------
# Case 8 (F15 M0 Isolation): Strict isolation between proj_a and proj_b
# ---------------------------------------------------------------------------
test_mcp_project_isolation_between_workspaces() {
    local proj_a="${TMPD}/proj_a"
    local proj_b="${TMPD}/proj_b"
    mkdir -p "${proj_a}/scenes" "${proj_a}/assets" "${proj_b}/scenes" "${proj_b}/assets"

    local err_msg=""
    if ! err_msg="$(PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - "$STUDIO_BIN" "$proj_a" "$proj_b" 2>&1 <<'PY'
import os, sys
from mcp_client import McpStdioClient

bin_path, proj_a, proj_b = sys.argv[1], sys.argv[2], sys.argv[3]

with McpStdioClient([bin_path, "--project", proj_a, "--mcp-stdio"], cwd=proj_a, framing="cl", allow_stderr=False) as ca:
    ca.initialize()
    ca.notify_initialized()
    tools = {t.get("name") for t in ca.list_tools()}
    if not {"scene.save", "scene.load"}.issubset(tools):
        raise SystemExit("M0 tools scene.save / scene.load not exposed yet")
    ca.call_tool("ecs.spawn_entity", {"name": "OnlyInA", "position": [5.0, 1.0, 0.0]})
    ca.call_tool("scene.save", {"path": "scenes/a.flscn"})

with McpStdioClient([bin_path, "--project", proj_b, "--mcp-stdio"], cwd=proj_b, framing="cl", allow_stderr=False) as cb:
    cb.initialize()
    cb.notify_initialized()
    cb.call_tool_expect_error("scene.load", {"path": "scenes/a.flscn"})
    cb.call_tool_expect_error("scene.load", {"path": os.path.join(proj_a, "scenes/a.flscn")})
    qb = cb.call_tool("ecs.query_entities", {})
    assert not any(e.get("name") == "OnlyInA" for e in qb["entities"])
PY
    )"; then
        fail_test "t2_m0_mcp_project_isolation_between_workspaces" "$err_msg"
        return
    fi

    pass_test "t2_m0_mcp_project_isolation_between_workspaces"
}

test_mcp_jsonrpc_protocol_errors
test_mcp_baseline_sandbox_traversal_and_symlinks
test_mcp_existing_tools_negative_boundaries
test_cli_empty_stdin_vk_missing_and_ro_out
test_cli_invalid_project_and_out_args
test_mcp_m0_scene_load_and_traversal_boundaries
test_mcp_m0_entity_transform_and_enable_negative
test_mcp_project_isolation_between_workspaces

print_summary_and_exit
