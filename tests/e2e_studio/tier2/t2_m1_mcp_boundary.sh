#!/usr/bin/env bash
# AREA: m1
# Tier 2 Boundary & Negative Coverage: Area C — M1 MCP Tool Boundaries & Corrupt Inputs (F17, F20, F21)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t2_m1"

if ! resolve_bin "flint-studio" "studio"; then
    fail_test "t2_m1_mcp_build_studio" "failed to resolve or build flint-studio"
    print_summary_and_exit
fi
STUDIO_BIN="$RESOLVED_BIN"

EMPTY_PROJ="${TMPD}/empty_proj"
TEST_PROJ="${TMPD}/ws/boundary_proj"
mkdir -p "${EMPTY_PROJ}/assets" "${EMPTY_PROJ}/scenes"
mkdir -p "${TEST_PROJ}/assets/materials" "${TEST_PROJ}/assets/prefabs" "${TEST_PROJ}/scenes"

cat >"${EMPTY_PROJ}/flint.toml" <<'TOML'
[project]
name = "empty_m1_proj"
entry_scene = "assets/main.flscn"
TOML

cat >"${TEST_PROJ}/flint.toml" <<'TOML'
[project]
name = "boundary_m1_proj"
entry_scene = "assets/main.flscn"
TOML

RESULTS_JSON="${TMPD}/t2_m1_mcp_results.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - \
    "$STUDIO_BIN" "$EMPTY_PROJ" "$TEST_PROJ" "$TMPD" "$RESULTS_JSON" <<'PY'
import json
import os
import sys
from mcp_client import McpStdioClient

studio_bin, empty_proj, test_proj, tmpd, results_path = sys.argv[1:6]
results = []

def record(name: str, ok: bool, reason: str = "") -> None:
    results.append({"name": name, "ok": ok, "reason": reason})

def pick_prop(tool_schema: dict, candidates: list[str], default: str) -> str:
    props = tool_schema.get("inputSchema", {}).get("properties", {})
    if isinstance(props, dict):
        for c in candidates:
            if c in props:
                return c
    return default

def assert_no_unimplemented_error(tool_name: str, msg: str, payload) -> None:
    low = msg.lower()
    if "unknown tool" in low:
        raise AssertionError(f"{tool_name} returned 'unknown tool': {payload!r}")
    if "not implemented" in low or "unimplemented" in low:
        raise AssertionError(f"{tool_name} returned 'not implemented': {payload!r}")
    if "unsupported tool" in low or "unsupported method" in low or (
        "unsupported" in low
        and not any(k in low for k in ("version", "schema", "format", "header", "asset", "file", "extension"))
    ):
        raise AssertionError(f"{tool_name} returned 'unsupported' stub error: {payload!r}")

def assert_structured_tool_or_param_error(client: McpStdioClient, tool_name: str, args: dict, tools_by_name: dict) -> str:
    """Assert tool call fails cleanly with -32602 or isError=True + structuredContent.ok=False, NEVER unimplemented."""
    if tool_name not in tools_by_name:
        raise AssertionError(f"target tool {tool_name!r} missing from tools/list")
    resp = client.request("tools/call", {"name": tool_name, "arguments": args})
    if "error" in resp:
        err = resp["error"]
        code = err.get("code")
        msg = str(err.get("message", ""))
        assert_no_unimplemented_error(tool_name, msg, resp)
        if code != -32602:
            raise AssertionError(f"{tool_name} expected -32602 or isError=True, got JSON-RPC error: {resp!r}")
        return msg
    res = resp.get("result", {})
    if not res.get("isError", False):
        raise AssertionError(f"{tool_name} unexpectedly succeeded on invalid input {args!r}: {res!r}")
    sc = res.get("structuredContent")
    if not isinstance(sc, dict) or sc.get("ok") is not False:
        raise AssertionError(f"{tool_name} isError=True missing structuredContent.ok == False: {res!r}")
    err_msg = str(sc.get("error", ""))
    content = res.get("content", [])
    content_msg = ""
    if isinstance(content, list) and content and isinstance(content[0], dict):
        content_msg = str(content[0].get("text", ""))
    combined_msg = f"{err_msg} {content_msg}".strip()
    assert_no_unimplemented_error(tool_name, combined_msg, res)
    return err_msg or content_msg

required_tools = [
    "asset.import",
    "asset.list",
    "prefab.spawn",
    "material.assign",
    "scene.save",
    "scene.load",
    "procedural.generate_asset",
    "ecs.spawn_entity",
    "ecs.query_entities",
]

# Pre-flight tools/list gate on empty project + Case 1: empty project asset.list
tools_by_name = {}
try:
    with McpStdioClient([studio_bin, "--project", empty_proj, "--mcp-stdio"], cwd=empty_proj, framing="cl") as c_empty:
        c_empty.initialize()
        c_empty.notify_initialized()
        for t in c_empty.list_tools():
            if isinstance(t, dict) and "name" in t:
                tools_by_name[t["name"]] = t
        missing = [n for n in required_tools if n not in tools_by_name]
        if missing:
            raise AssertionError(f"required M1/M0 tools missing from tools/list: {missing}")
        sc = c_empty.call_tool("asset.list", {})
        assets = sc.get("assets", [])
        count = sc.get("count", len(assets) if isinstance(assets, list) else -1)
        if not isinstance(assets, list) or len(assets) != 0 or count != 0:
            raise AssertionError(f"expected empty asset.list in fresh project, got {sc!r}")
    record("t2_m1_mcp_empty_project_asset_list", True)
except Exception as exc:
    record("t2_m1_mcp_empty_project_asset_list", False, str(exc))

if not all(n in tools_by_name for n in required_tools):
    missing = [n for n in required_tools if n not in tools_by_name]
    for cname in [
        "t2_m1_mcp_asset_import_traversal_and_missing_source",
        "t2_m1_mcp_prefab_and_material_invalid_entity_and_missing_assets",
        "t2_m1_mcp_corrupt_flscn_reload_preserves_state_and_server",
        "t2_m1_mcp_corrupt_glb_malformed_flmat_flprefab_rejected",
        "t2_m1_mcp_protocol_invalid_params_and_traversal_save",
    ]:
        record(cname, False, f"prerequisite tools missing from tools/list: {missing}")
else:
    # Stage corrupt fixtures inside test_proj
    corrupt_glb = os.path.join(test_proj, "corrupt_src.glb")
    with open(corrupt_glb, "wb") as f:
        f.write(b"glTF\x02\x00\x00\x00\x01\x00\x00\x00")

    bad_flmat_rel = "assets/materials/corrupt.flmat"
    with open(os.path.join(test_proj, bad_flmat_rel), "w", encoding="utf-8") as f:
        f.write("{not valid json")

    bad_flprefab_rel = "assets/prefabs/corrupt.flprefab"
    with open(os.path.join(test_proj, bad_flprefab_rel), "w", encoding="utf-8") as f:
        f.write('{"format":"flprefab","nodes":"not_a_list"}')

    with open(os.path.join(test_proj, "scenes/bad_json.flscn"), "w", encoding="utf-8") as f:
        f.write("{bad_json")
    with open(os.path.join(test_proj, "scenes/future_ver.flscn"), "w", encoding="utf-8") as f:
        f.write('{"format":"flscn","version":999,"name":"Future","entities":[]}')
    with open(os.path.join(test_proj, "scenes/empty.flscn"), "wb") as f:
        f.write(b"")

    try:
        with McpStdioClient([studio_bin, "--project", test_proj, "--mcp-stdio"], cwd=test_proj, framing="cl") as client:
            client.initialize()
            client.notify_initialized()

            # Spawn a valid sentinel entity so we have a valid entity_id + state_hash
            spawn_sc = client.call_tool("ecs.spawn_entity", {"name": "Sentinel_M1", "position": [3.0, 1.0, -2.0]})
            valid_eid = spawn_sc.get("entity_id", -1)
            if not isinstance(valid_eid, int) or valid_eid < 0:
                raise AssertionError(f"failed spawning sentinel entity: {spawn_sc!r}")

            t_imp = tools_by_name["asset.import"]
            src_k = pick_prop(t_imp, ["source_path", "src_path", "path"], "source_path")
            dst_k = pick_prop(t_imp, ["dest_path", "rel_path", "target_path"], "dest_path")

            t_pf = tools_by_name["prefab.spawn"]
            pf_k = pick_prop(t_pf, ["prefab_path", "path", "asset_path"], "prefab_path")

            t_mat = tools_by_name["material.assign"]
            mat_k = pick_prop(t_mat, ["material_path", "path", "asset_path"], "material_path")

            t_save = tools_by_name["scene.save"]
            save_k = pick_prop(t_save, ["path", "scene_path", "file_path"], "path")

            t_load = tools_by_name["scene.load"]
            load_k = pick_prop(t_load, ["path", "scene_path", "file_path"], "path")

            # Case 2: asset.import traversal & missing source
            try:
                _ = assert_structured_tool_or_param_error(
                    client, "asset.import", {src_k: "assets/does_not_exist.glb", dst_k: "assets/meshes/miss.glb"}, tools_by_name
                )
                _ = assert_structured_tool_or_param_error(
                    client, "asset.import", {src_k: "/etc/passwd", dst_k: "assets/passwd.glb"}, tools_by_name
                )
                _ = assert_structured_tool_or_param_error(
                    client, "asset.import", {src_k: "../../escape.glb", dst_k: "../../escape.glb"}, tools_by_name
                )
                for leaked in [
                    os.path.join(test_proj, "assets/meshes/miss.glb"),
                    os.path.join(test_proj, "assets/passwd.glb"),
                    os.path.join(test_proj, "escape.glb"),
                    os.path.join(tmpd, "escape.glb"),
                    os.path.normpath(os.path.join(test_proj, "../escape.glb")),
                    os.path.normpath(os.path.join(test_proj, "../../escape.glb")),
                ]:
                    if os.path.exists(leaked):
                        raise AssertionError(f"rejected asset.import leaked file on disk: {leaked}")
                record("t2_m1_mcp_asset_import_traversal_and_missing_source", True)
            except Exception as exc:
                record("t2_m1_mcp_asset_import_traversal_and_missing_source", False, str(exc))

            # Case 3: prefab.spawn & material.assign invalid entity IDs + missing asset paths
            try:
                _ = assert_structured_tool_or_param_error(
                    client, "material.assign", {"entity_id": -1, mat_k: "assets/materials/missing.flmat"}, tools_by_name
                )
                _ = assert_structured_tool_or_param_error(
                    client, "material.assign", {"entity_id": 999999, mat_k: "assets/materials/missing.flmat"}, tools_by_name
                )
                _ = assert_structured_tool_or_param_error(
                    client, "material.assign", {"entity_id": "bad_id", mat_k: "assets/materials/missing.flmat"}, tools_by_name
                )
                _ = assert_structured_tool_or_param_error(
                    client, "material.assign", {mat_k: "assets/materials/missing.flmat"}, tools_by_name
                )
                _ = assert_structured_tool_or_param_error(
                    client, "material.assign", {"entity_id": valid_eid, mat_k: "assets/materials/missing.flmat"}, tools_by_name
                )
                _ = assert_structured_tool_or_param_error(
                    client, "prefab.spawn", {pf_k: "../escape.flprefab"}, tools_by_name
                )
                _ = assert_structured_tool_or_param_error(
                    client, "prefab.spawn", {pf_k: "assets/prefabs/missing.flprefab"}, tools_by_name
                )
                record("t2_m1_mcp_prefab_and_material_invalid_entity_and_missing_assets", True)
            except Exception as exc:
                record("t2_m1_mcp_prefab_and_material_invalid_entity_and_missing_assets", False, str(exc))

            # Case 4: Corrupt .flscn reload preserves scene state & keeps MCP server alive
            try:
                q_before = client.call_tool("ecs.query_entities", {})
                hash_before = q_before.get("state_hash")
                count_before = q_before.get("count")

                for bad_scene in ["scenes/bad_json.flscn", "scenes/future_ver.flscn", "scenes/empty.flscn"]:
                    _ = assert_structured_tool_or_param_error(
                        client, "scene.load", {load_k: bad_scene}, tools_by_name
                    )

                ping_resp = client.request("ping", {})
                if "error" in ping_resp:
                    raise AssertionError(f"server unhealthy after corrupt scene.load: {ping_resp!r}")

                q_after = client.call_tool("ecs.query_entities", {})
                if q_after.get("count") != count_before or q_after.get("state_hash") != hash_before:
                    raise AssertionError(
                        f"failed scene.load mutated live world: before=({count_before},{hash_before}), after=({q_after.get('count')},{q_after.get('state_hash')})"
                    )
                record("t2_m1_mcp_corrupt_flscn_reload_preserves_state_and_server", True)
            except Exception as exc:
                record("t2_m1_mcp_corrupt_flscn_reload_preserves_state_and_server", False, str(exc))

            # Case 5: Corrupt .glb / malformed .flmat / malformed .flprefab rejected without crash or state/disk mutation
            try:
                q_c5_before = client.call_tool("ecs.query_entities", {})
                c5_count_before = q_c5_before.get("count")
                c5_hash_before = q_c5_before.get("state_hash")

                _ = assert_structured_tool_or_param_error(
                    client, "asset.import", {src_k: corrupt_glb, dst_k: "assets/meshes/corrupt.glb"}, tools_by_name
                )
                if os.path.exists(os.path.join(test_proj, "assets/meshes/corrupt.glb")):
                    raise AssertionError("rejected corrupt asset.import wrote assets/meshes/corrupt.glb on disk")
                _ = assert_structured_tool_or_param_error(
                    client, "material.assign", {"entity_id": valid_eid, mat_k: bad_flmat_rel}, tools_by_name
                )
                sneaky_json = os.path.join(test_proj, "assets", "sneaky.json")
                with open(sneaky_json, "w", encoding="utf-8") as sf:
                    sf.write('{"name":"Sneaky"}')
                _ = assert_structured_tool_or_param_error(
                    client, "material.assign", {"entity_id": valid_eid, mat_k: "assets/sneaky.json"}, tools_by_name
                )
                _ = assert_structured_tool_or_param_error(
                    client, "prefab.spawn", {pf_k: bad_flprefab_rel}, tools_by_name
                )
                ping_resp = client.request("ping", {})
                if "error" in ping_resp:
                    raise AssertionError(f"server unhealthy after corrupt asset calls: {ping_resp!r}")
                q_c5_after = client.call_tool("ecs.query_entities", {})
                if q_c5_after.get("count") != c5_count_before or q_c5_after.get("state_hash") != c5_hash_before:
                    raise AssertionError(
                        f"corrupt material/prefab call mutated ECS world: before=({c5_count_before},{c5_hash_before}), after=({q_c5_after.get('count')},{q_c5_after.get('state_hash')})"
                    )
                record("t2_m1_mcp_corrupt_glb_malformed_flmat_flprefab_rejected", True)
            except Exception as exc:
                record("t2_m1_mcp_corrupt_glb_malformed_flmat_flprefab_rejected", False, str(exc))

            # Case 6: JSON-RPC -32602 envelope errors & traversal on scene.save / procedural.generate_asset
            try:
                _ = assert_structured_tool_or_param_error(
                    client, "scene.save", {save_k: "../escape.flscn"}, tools_by_name
                )
                _ = assert_structured_tool_or_param_error(
                    client, "procedural.generate_asset", {"preset": "material_orb", "output_glb_path": "../escape.glb"}, tools_by_name
                )
                for leaked in [
                    os.path.join(tmpd, "escape.flscn"),
                    os.path.normpath(os.path.join(test_proj, "../escape.flscn")),
                    os.path.join(test_proj, "escape.flscn"),
                    os.path.join(tmpd, "escape.glb"),
                    os.path.normpath(os.path.join(test_proj, "../escape.glb")),
                    os.path.join(test_proj, "escape.glb"),
                ]:
                    if os.path.exists(leaked):
                        raise AssertionError(f"rejected traversal save/generate leaked file on disk: {leaked}")
                _ = client.expect_protocol_error(
                    {"jsonrpc": "2.0", "id": 9001, "method": "tools/call", "params": {"arguments": {}}},
                    -32602,
                )
                _ = client.expect_protocol_error(
                    {"jsonrpc": "2.0", "id": 9002, "method": "tools/call", "params": "not_an_object"},
                    -32602,
                )
                record("t2_m1_mcp_protocol_invalid_params_and_traversal_save", True)
            except Exception as exc:
                record("t2_m1_mcp_protocol_invalid_params_and_traversal_save", False, str(exc))
    except Exception as exc:
        record("t2_m1_mcp_session_lifecycle", False, str(exc))

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
PY

if [[ -f "$RESULTS_JSON" ]]; then
    while IFS=$'\t' read -r cname cok creason; do
        if [[ "$cok" == "True" ]]; then
            pass_test "$cname"
        else
            fail_test "$cname" "$creason"
        fi
    done < <(PYTHONDONTWRITEBYTECODE=1 python3 -B - "$RESULTS_JSON" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for row in json.load(f):
        reason = str(row.get("reason", "")).replace("\t", " ").replace("\n", " ")
        print(f"{row['name']}\t{bool(row['ok'])}\t{reason}")
PY
)
else
    fail_test "t2_m1_mcp_runner" "python harness failed to produce results"
fi

print_summary_and_exit
