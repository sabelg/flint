#!/usr/bin/env bash
# AREA: m2
# Tier 1 Feature Coverage: Area D — M2 MCP Runtime Tools & Framing Parity (F25)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t1_m2_mcp"
resolve_bin "flint-studio" "studio" || {
    fail_test "t1_m2_mcp_build_studio" "resolve_bin flint-studio failed"
    print_summary_and_exit
}

PROJ_DIR="${TMPD}/mcp_proj"
mkdir -p "${PROJ_DIR}/assets" "${PROJ_DIR}/scenes"

cat >"${PROJ_DIR}/flint.toml" <<'TOML'
[project]
name = "m2_mcp_proj"
entry_scene = "scenes/main.flscn"
TOML

# Generate valid 48kHz stereo PCM16 WAV fixture + minimal scene fixture inside $TMPD
python3 -B - "${PROJ_DIR}" <<'PY'
import json, math, os, struct, sys

proj = sys.argv[1]
sr, ch, n = 48000, 2, 1200
data_sz = n * ch * 2
wav = bytearray()
wav += b"RIFF" + struct.pack("<I", 36 + data_sz) + b"WAVE"
wav += b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ch * 2, ch * 2, 16)
wav += b"data" + struct.pack("<I", data_sz)
for i in range(n):
    val = int(math.sin(2.0 * math.pi * 440.0 * i / sr) * 12000)
    s = struct.pack("<h", val)
    wav += s + s
with open(os.path.join(proj, "assets", "sfx.wav"), "wb") as f:
    f.write(wav)

with open(os.path.join(proj, "scenes", "main.flscn"), "w", encoding="utf-8") as f:
    json.dump({"format": "flscn", "version": 2, "name": "M2McpTest", "entities": []}, f)
PY

validate_wav_pcm "${PROJ_DIR}/assets/sfx.wav" 2 48000 16 || {
    fail_test "t1_m2_mcp_fixture_wav" "generated sfx.wav failed validate_wav_pcm"
    print_summary_and_exit
}

RESULTS_FILE="${TMPD}/mcp_results.tsv"

PYTHONPATH="${SCRIPT_DIR}/../common${PYTHONPATH:+:$PYTHONPATH}" python3 -B - "$RESOLVED_BIN" "$PROJ_DIR" "$RESULTS_FILE" <<'PY'
import sys
from mcp_client import McpStdioClient

studio_bin, proj_dir, out_tsv = sys.argv[1], sys.argv[2], sys.argv[3]

CASE_ORDER = [
    "t1_m2_mcp_tools_registered",
    "t1_m2_mcp_pie_and_physics_trigger_events",
    "t1_m2_mcp_anim_set_state",
    "t1_m2_mcp_audio_play",
    "t1_m2_mcp_nav_query_path",
    "t1_m2_mcp_pie_stop_restore_and_ndjson",
]
results: dict[str, tuple[bool, str]] = {}

def record(name: str, ok: bool, reason: str = "") -> None:
    if name in results and not results[name][0]:
        if not ok and reason and reason not in results[name][1]:
            results[name] = (False, f"{results[name][1]}; {reason}")
        return
    results[name] = (ok, reason)

def pick_arg(props: dict, candidates: list[str], default: str) -> str:
    for c in candidates:
        if c in props:
            return c
    return default

tool_map: dict[str, dict] = {}
tools_reg_ok = False
tools_reg_reason = ""
cl_stop_ok = False
cl_stop_reason = ""

try:
    with McpStdioClient([studio_bin, "--project", proj_dir, "--mcp-stdio"], cwd=proj_dir, framing="cl", timeout=20.0) as client:
        init_res = client.initialize()
        if init_res.get("protocolVersion") != "2024-11-05":
            raise AssertionError(f"Unexpected protocolVersion: {init_res!r}")
        client.notify_initialized()

        tools = client.list_tools()
        for t in tools:
            if isinstance(t, dict) and "name" in t:
                tool_map[t["name"]] = t.get("inputSchema", {})

        required_m2 = ["physics.trigger_events", "anim.set_state", "audio.play", "nav.query_path"]
        missing = [n for n in required_m2 if n not in tool_map]
        if missing:
            tools_reg_ok = False
            tools_reg_reason = f"missing M2 MCP tool(s) (pending M5): {', '.join(missing)}"
        else:
            schemas_ok = all(isinstance(tool_map[n], dict) for n in required_m2)
            tools_reg_ok = schemas_ok
            tools_reg_reason = "" if schemas_ok else "invalid inputSchema"

        # Dynamically spawn & configure floor + actor entity before PIE
        actor_eid = None
        pre_pie_hash = None
        prep_err = ""
        try:
            _ = client.call_tool(
                "ecs.spawn_entity",
                {
                    "name": "M2_Floor",
                    "position": [0.0, -0.5, 0.0],
                    "scale": [12.0, 1.0, 12.0],
                    "collider": "box",
                    "mass": 0.0,
                },
            )
            actor_sc = client.call_tool(
                "ecs.spawn_entity",
                {
                    "name": "M2_AnimActor",
                    "position": [0.0, 1.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                    "collider": "box",
                    "mass": 1.0,
                },
            )
            actor_eid = actor_sc.get("entity_id")
            if not isinstance(actor_eid, int) or actor_eid < 0:
                raise AssertionError(f"ecs.spawn_entity returned invalid actor entity_id: {actor_sc!r}")
            _ = client.call_tool(
                "ecs.patch_components",
                {"entity_id": actor_eid, "position": [0.5, 1.0, 0.0], "visible": True},
            )
            _ = client.call_tool("physics.step_deterministic", {"ticks": 2})
            q_pre = client.call_tool("ecs.query_entities", {})
            live_ids = [e.get("entity_id") for e in q_pre.get("entities", []) if isinstance(e, dict)]
            if actor_eid not in live_ids:
                raise AssertionError(f"spawned actor_eid={actor_eid} missing from ecs.query_entities: {q_pre!r}")
            pre_pie_hash = q_pre.get("state_hash")
        except Exception as exc:
            prep_err = f"pre-PIE entity setup failed: {exc}"

        # Start PIE & step ticks + physics.trigger_events
        try:
            play_sc = client.call_tool("studio.pie_control", {"action": "play"})
            if play_sc.get("pie_mode") != "playing":
                raise AssertionError(f"Expected pie_mode=='playing', got {play_sc!r}")
            step_sc = client.call_tool("studio.pie_control", {"action": "step", "ticks": 10})
            if step_sc.get("ticks_simulated", 0) < 10:
                raise AssertionError(f"Expected >=10 ticks simulated, got {step_sc!r}")

            if "physics.trigger_events" not in tool_map:
                record("t1_m2_mcp_pie_and_physics_trigger_events", False, "physics.trigger_events not registered (pending M5)")
            else:
                trig_sc = client.call_tool("physics.trigger_events", {})
                evs = trig_sc.get("events")
                if not isinstance(evs, list):
                    record("t1_m2_mcp_pie_and_physics_trigger_events", False, f"expected events list, got {trig_sc!r}")
                else:
                    record("t1_m2_mcp_pie_and_physics_trigger_events", True)
        except Exception as exc:
            record("t1_m2_mcp_pie_and_physics_trigger_events", False, str(exc))

        # anim.set_state on spawned actor_eid
        if "anim.set_state" not in tool_map:
            record("t1_m2_mcp_anim_set_state", False, "anim.set_state not registered (pending M5)")
        elif prep_err or actor_eid is None:
            record("t1_m2_mcp_anim_set_state", False, prep_err or "actor_eid unavailable")
        else:
            try:
                props = tool_map["anim.set_state"].get("properties", {})
                ent_k = pick_arg(props, ["entity_id", "entity", "id"], "entity_id")
                st_k = pick_arg(props, ["state", "state_name", "clip"], "state")
                anim_sc = client.call_tool("anim.set_state", {ent_k: actor_eid, st_k: "idle"})
                record("t1_m2_mcp_anim_set_state", anim_sc.get("ok") is True)
            except Exception as exc:
                record("t1_m2_mcp_anim_set_state", False, str(exc))

        # audio.play
        if "audio.play" not in tool_map:
            record("t1_m2_mcp_audio_play", False, "audio.play not registered (pending M5)")
        else:
            try:
                props = tool_map["audio.play"].get("properties", {})
                path_k = pick_arg(props, ["asset_path", "path", "sound", "clip"], "asset_path")
                aud_sc = client.call_tool("audio.play", {path_k: "assets/sfx.wav"})
                record("t1_m2_mcp_audio_play", aud_sc.get("ok") is True)
            except Exception as exc:
                record("t1_m2_mcp_audio_play", False, str(exc))

        # nav.query_path
        if "nav.query_path" not in tool_map:
            record("t1_m2_mcp_nav_query_path", False, "nav.query_path not registered (pending M5)")
        else:
            try:
                nav_sc = client.call_tool(
                    "nav.query_path",
                    {"start": [0.0, 0.0, 0.0], "goal": [4.0, 0.0, 4.0]},
                )
                found = nav_sc.get("found")
                wps = nav_sc.get("waypoints")
                ok_nav = isinstance(found, bool) and isinstance(wps, list) and (not found or len(wps) > 0)
                record(
                    "t1_m2_mcp_nav_query_path",
                    ok_nav,
                    "" if ok_nav else f"invalid nav.query_path payload (expected bool found & list waypoints): {nav_sc!r}",
                )
            except Exception as exc:
                record("t1_m2_mcp_nav_query_path", False, str(exc))

        # Stop PIE cleanly
        try:
            stop_sc = client.call_tool("studio.pie_control", {"action": "stop"})
            if stop_sc.get("pie_mode") != "stopped" or stop_sc.get("restored_exact") is not True:
                raise AssertionError(f"PIE stop failed restore check: {stop_sc!r}")
            if pre_pie_hash is not None and stop_sc.get("state_hash") != pre_pie_hash:
                raise AssertionError(
                    f"PIE stop state_hash mismatch: expected {pre_pie_hash!r}, got {stop_sc.get('state_hash')!r}"
                )
            cl_stop_ok = True
        except Exception as exc:
            cl_stop_ok = False
            cl_stop_reason = f"CL PIE stop failed: {exc}"

    # Record only AFTER `with McpStdioClient(...)` exits and close() succeeds
    record("t1_m2_mcp_tools_registered", tools_reg_ok, tools_reg_reason)
    if not cl_stop_ok:
        record("t1_m2_mcp_pie_stop_restore_and_ndjson", False, cl_stop_reason)
except Exception as exc:
    close_msg = f"CL MCP session/close failed: {exc}"
    reg_msg = f"{tools_reg_reason}; {close_msg}" if tools_reg_reason else close_msg
    record("t1_m2_mcp_tools_registered", False, reg_msg)
    record("t1_m2_mcp_pie_stop_restore_and_ndjson", False, close_msg)

# NDJSON framing parity check (only if CL stop did not already fail)
if "t1_m2_mcp_pie_stop_restore_and_ndjson" not in results:
    nd_ok = False
    nd_err = ""
    try:
        with McpStdioClient([studio_bin, "--project", proj_dir, "--mcp-stdio"], cwd=proj_dir, framing="ndjson", timeout=20.0) as nd_client:
            nd_init = nd_client.initialize()
            if nd_init.get("protocolVersion") != "2024-11-05":
                raise AssertionError(f"NDJSON initialize unexpected: {nd_init!r}")
            nd_client.notify_initialized()
            nd_tools = {t["name"] for t in nd_client.list_tools() if isinstance(t, dict) and "name" in t}
            if "nav.query_path" not in nd_tools:
                nd_ok = False
                nd_err = "nav.query_path missing under NDJSON tools/list (pending M5)"
            else:
                _ = nd_client.call_tool("studio.pie_control", {"action": "play"})
                nav_nd = nd_client.call_tool("nav.query_path", {"start": [0.0, 0.0, 0.0], "goal": [2.0, 0.0, 2.0]})
                stop_nd = nd_client.call_tool("studio.pie_control", {"action": "stop"})
                found_nd = nav_nd.get("found")
                wps_nd = nav_nd.get("waypoints")
                if not (isinstance(found_nd, bool) and isinstance(wps_nd, list) and (not found_nd or len(wps_nd) > 0)):
                    nd_ok = False
                    nd_err = f"NDJSON nav.query_path invalid payload: {nav_nd!r}"
                elif stop_nd.get("pie_mode") != "stopped" or stop_nd.get("restored_exact") is not True:
                    nd_ok = False
                    nd_err = f"NDJSON PIE stop failed restore check: {stop_nd!r}"
                else:
                    nd_ok = True
        record("t1_m2_mcp_pie_stop_restore_and_ndjson", nd_ok, nd_err)
    except Exception as exc:
        record("t1_m2_mcp_pie_stop_restore_and_ndjson", False, f"NDJSON MCP session/close failed: {exc}")

for cname in CASE_ORDER:
    if cname not in results:
        results[cname] = (False, "skipped due to earlier CL MCP session failure")

with open(out_tsv, "w", encoding="utf-8") as f:
    for cname in CASE_ORDER:
        ok, reason = results[cname]
        clean_reason = reason.replace("\t", " ").replace("\n", " ")
        f.write(f"{cname}\t{'1' if ok else '0'}\t{clean_reason}\n")
PY

while IFS=$'\t' read -r cname cok creason; do
    [[ -z "$cname" ]] && continue
    if [[ "$cok" == "1" ]]; then
        pass_test "$cname"
    else
        fail_test "$cname" "$creason"
    fi
done < "$RESULTS_FILE"

print_summary_and_exit
