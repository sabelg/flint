#!/usr/bin/env bash
# AREA: m3
# Tier 1 Feature Coverage: Area E — M3 Shipping, flint_version, .flpk-first VFS, MCP project.package & Test-Harness Game (F26-F28)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t1_m3"
resolve_mojo_env || {
    fail_test "t1_m3_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

_write_shipping_assets_dir() {
    local target_dir="$1"
    mkdir -p "$target_dir"
    python3 -B - "$target_dir" <<'PY'
import json, math, os, struct, sys, zlib

out_dir = sys.argv[1]

# 8x8 RGBA PNG
w, h = 8, 8
raw_rows = bytearray()
for y in range(h):
    raw_rows.append(0)
    for x in range(w):
        raw_rows.extend((48, 144, 216, 255))

def png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)

ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
idat = zlib.compress(bytes(raw_rows))
with open(os.path.join(out_dir, "tex.png"), "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", idat) + png_chunk(b"IEND", b""))

# 48kHz stereo PCM16 WAV
sr, ch, n = 48000, 2, 1200
data_sz = n * ch * 2
wav = bytearray()
wav += b"RIFF" + struct.pack("<I", 36 + data_sz) + b"WAVE"
wav += b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ch * 2, ch * 2, 16)
wav += b"data" + struct.pack("<I", data_sz)
for i in range(n):
    val = int(math.sin(2.0 * math.pi * 440.0 * i / sr) * 10000)
    s = struct.pack("<h", val)
    wav += s + s
with open(os.path.join(out_dir, "sfx.wav"), "wb") as f:
    f.write(wav)

# Minimal glTF 2.0 GLB
positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
normals = struct.pack("<9f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
indices = struct.pack("<3H", 0, 1, 2) + b"\x00\x00"
bin_blob = positions + normals + indices
gltf = {
    "asset": {"version": "2.0"},
    "scene": 0,
    "scenes": [{"nodes": [0]}],
    "nodes": [{"name": "tri", "mesh": 0}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2}]}],
    "buffers": [{"byteLength": len(bin_blob)}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
        {"buffer": 0, "byteOffset": 36, "byteLength": 36, "target": 34962},
        {"buffer": 0, "byteOffset": 72, "byteLength": 6, "target": 34963},
    ],
    "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
        {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
        {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
    ],
}
json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
while len(json_bytes) % 4 != 0:
    json_bytes += b" "
total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
glb = (
    struct.pack("<III", 0x46546C67, 2, total_len)
    + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    + struct.pack("<II", len(bin_blob), 0x004E4942) + bin_blob
)
with open(os.path.join(out_dir, "prop.glb"), "wb") as f:
    f.write(glb)

# Valid minimal .flscn + .flmat + .json
main_flscn = {
    "format": "flscn",
    "version": 1,
    "name": "main",
    "entities": [
        {
            "slot": 0,
            "parent": -1,
            "components": {
                "LocalTransform": {"position": [0.0, 1.5, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]}
            },
        }
    ],
}
with open(os.path.join(out_dir, "main.flscn"), "w", encoding="utf-8") as f:
    json.dump(main_flscn, f)
    f.write("\n")

with open(os.path.join(out_dir, "stone.flmat"), "w", encoding="utf-8") as f:
    f.write('{"base_color":[0.5,0.5,0.55],"roughness":0.8,"metallic":0.0}\n')

with open(os.path.join(out_dir, "game.json"), "w", encoding="utf-8") as f:
    f.write('{"title":"m3_shipping_test"}\n')
PY
}

# ---------------------------------------------------------------------------
# Case 1 (F26): flint.sh new creates assets/main.flscn, flint.toml (entry_scene,
# pak_path/pack_output, flint_version), and main.mojo loads main.flscn + crash/log
# ---------------------------------------------------------------------------
test_f26_new_scaffold_main_flscn_and_config() {
    local proj="${TMPD}/m3_new_proj"
    local new_out new_rc=0
    new_out="$(bash "${ROOT}/tools/flint.sh" new "$proj" 2>&1)" || new_rc=$?
    if [[ "$new_rc" -ne 0 ]]; then
        fail_test "t1_m3_f26_new_scaffold_main_flscn_and_config" "flint.sh new failed (rc=$new_rc): $new_out"
        return
    fi

    # Verify crash handler & start_log(cfg.app_name) are initialized
    if ! grep -q "start_log(cfg.app_name)" "${proj}/src/main.mojo" || ! grep -q "enable_crash_handler=True" "${proj}/src/main.mojo"; then
        fail_test "t1_m3_f26_new_scaffold_main_flscn_and_config" "scaffolded src/main.mojo missing start_log(cfg.app_name) or enable_crash_handler=True"
        return
    fi

    # Run --headless-smoke and confirm log path uses cfg.app_name
    local state_dir="${TMPD}/new_state"
    local smoke_out smoke_rc=0
    smoke_out="$(cd "$proj" && FLINT_ROOT="$ROOT" FLINT_STATE_DIR="$state_dir" XDG_STATE_HOME="${TMPD}/new_xdg" bash tools/run.sh --headless-smoke 2>&1)" || smoke_rc=$?
    if [[ "$smoke_rc" -ne 0 ]]; then
        fail_test "t1_m3_f26_new_scaffold_main_flscn_and_config" "tools/run.sh --headless-smoke failed (rc=$smoke_rc): $smoke_out"
        return
    fi
    if [[ ! -s "${state_dir}/logs/m3_new_proj.log" ]]; then
        fail_test "t1_m3_f26_new_scaffold_main_flscn_and_config" "expected log at ${state_dir}/logs/m3_new_proj.log"
        return
    fi

    # M3 contract checks: assets/main.flscn, entry_scene, pak_path/pack_output, flint_version, main.mojo loading main.flscn
    local missing=()
    [[ -f "${proj}/assets/main.flscn" ]] || missing+=("assets/main.flscn")
    grep -qE '^[[:space:]]*entry_scene[[:space:]]*=' "${proj}/flint.toml" || missing+=("flint.toml:entry_scene")
    grep -qE '^[[:space:]]*(pak_path|pack_output)[[:space:]]*=' "${proj}/flint.toml" || missing+=("flint.toml:pak_path/pack_output")
    grep -qE '^[[:space:]]*flint_version[[:space:]]*=' "${proj}/flint.toml" || missing+=("flint.toml:flint_version")
    grep -qE 'main\.flscn|entry_scene|load_flscn' "${proj}/src/main.mojo" || missing+=("src/main.mojo:load_entry_scene")

    if [[ "${#missing[@]}" -gt 0 ]]; then
        fail_test "t1_m3_f26_new_scaffold_main_flscn_and_config" "M3 F26 contract pending; missing: ${missing[*]}"
        return
    fi

    pass_test "t1_m3_f26_new_scaffold_main_flscn_and_config"
}

# ---------------------------------------------------------------------------
# Case 2 (F26): flint_version pin matches engine version/commit from build_info.mojo
# ---------------------------------------------------------------------------
test_f26_flint_version_pin_matches_build_info() {
    local bi_src="${TMPD}/bi_probe.mojo"
    local bi_bin="${TMPD}/bin/bi_probe"
    cat >"$bi_src" <<'MOJO'
from flint.core.build_info import build_commit, build_stamp

def main():
    print("COMMIT=" + build_commit())
    print("STAMP=" + build_stamp())
MOJO
    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$bi_src" -o "$bi_bin" >/dev/null 2>&1; then
        fail_test "t1_m3_f26_flint_version_pin_matches_build_info" "failed to compile build_info probe"
        return
    fi
    local bi_out engine_commit
    bi_out="$("$bi_bin")"
    engine_commit="$(awk -F= '/^COMMIT=/{print $2; exit}' <<<"$bi_out")"
    engine_commit="${engine_commit%+dirty}"
    if [[ -z "$engine_commit" ]]; then
        fail_test "t1_m3_f26_flint_version_pin_matches_build_info" "build_info.build_commit() returned empty"
        return
    fi

    local proj="${TMPD}/pin_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t1_m3_f26_flint_version_pin_matches_build_info" "flint.sh new failed"
        return
    fi

    local pin_line
    pin_line="$(grep -E '^[[:space:]]*(flint_version|flint_commit|engine_commit)[[:space:]]*=' "${proj}/flint.toml" || true)"
    if [[ -z "$pin_line" ]]; then
        fail_test "t1_m3_f26_flint_version_pin_matches_build_info" "flint.toml missing flint_version pin"
        return
    fi

    # Verify flint_version/commit pin actually matches engine build_info commit
    if ! grep -Fq "$engine_commit" <<<"$pin_line"; then
        fail_test "t1_m3_f26_flint_version_pin_matches_build_info" "M3 F26 contract pending: flint.toml (${pin_line}) does not pin engine commit '${engine_commit}' from build_info.mojo"
        return
    fi

    # Lightweight run check (avoids re-compiling full heavy template twice inside 180s Tier 1 budget)
    local run_rc=0
    local run_out
    run_out="$(cd "$proj" && FLINT_ROOT="$ROOT" FLINT_STATE_DIR="${TMPD}/pin_state" XDG_STATE_HOME="${TMPD}/pin_xdg" bash tools/run.sh --headless-smoke 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 ]]; then
        fail_test "t1_m3_f26_flint_version_pin_matches_build_info" "matching pin project run failed (rc=$run_rc): $run_out"
        return
    fi

    pass_test "t1_m3_f26_flint_version_pin_matches_build_info"
}

# ---------------------------------------------------------------------------
# Case 3 (F26): AssetServer .flpk-first resolution with loose filesystem fallback
# ---------------------------------------------------------------------------
test_f26_asset_server_flpk_first_and_loose_fallback() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t1_m3_f26_asset_server_flpk_first_and_loose_fallback" "could not resolve flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"

    local pack_src="${TMPD}/vfs_pack_src"
    local loose_dir="${TMPD}/vfs_loose"
    mkdir -p "${pack_src}/data" "${loose_dir}/data"
    printf '{"source":"flpk","ver":1}\n' > "${pack_src}/data/override.json"
    printf '{"source":"pack_only"}\n' > "${pack_src}/data/pack_only.json"
    printf '{"source":"loose","ver":2}\n' > "${loose_dir}/data/override.json"
    printf '{"source":"loose_only"}\n' > "${loose_dir}/data/loose_only.json"

    local vfs_flpk="${TMPD}/vfs.flpk"
    if ! "$pack_bin" --cook "$pack_src" "$vfs_flpk" --cache-dir "${TMPD}/vfs_cache" >/dev/null 2>&1; then
        fail_test "t1_m3_f26_asset_server_flpk_first_and_loose_fallback" "flint-pack --cook failed"
        return
    fi

    local probe_src="${TMPD}/vfs_probe.mojo"
    local probe_bin="${TMPD}/bin/vfs_probe"
    cat >"$probe_src" <<'MOJO'
from std.sys import argv
from flint.assets.server import AssetServer

def _as_str(bytes: List[UInt8]) -> String:
    var cp = bytes.copy()
    return String(unsafe_from_utf8=cp)

def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: vfs_probe <flpk> <loose_dir>")
    var flpk = String(args[1])
    var loose = String(args[2])

    var srv = AssetServer(enable_watcher=False)
    if not srv.mount_flpk(flpk):
        raise Error("mount_flpk failed")
    _ = srv.watch_loose_directory(loose)

    var h_ov = srv.load_raw("data/override.json")
    var p_ov = srv.raws.get_ptr(h_ov)
    if not p_ov:
        raise Error("override.json not loaded")
    var s_ov = _as_str(p_ov.value()[unsafe_offset=0].bytes)
    if s_ov.find('"flpk"') < 0 or s_ov.find('"loose"') >= 0:
        raise Error("expected .flpk priority over loose FS, got: " + s_ov)

    var h_pk = srv.load_raw("data/pack_only.json")
    var p_pk = srv.raws.get_ptr(h_pk)
    if not p_pk or _as_str(p_pk.value()[unsafe_offset=0].bytes).find('"pack_only"') < 0:
        raise Error("pack_only.json mismatch")

    var h_ls = srv.load_raw("data/loose_only.json")
    var p_ls = srv.raws.get_ptr(h_ls)
    if not p_ls or _as_str(p_ls.value()[unsafe_offset=0].bytes).find('"loose_only"') < 0:
        raise Error("loose_only.json fallback failed")

    var h_miss = srv.load_raw("data/does_not_exist.json")
    if srv.raws.get_ptr(h_miss):
        raise Error("missing file unexpectedly resolved")

    srv.close()
MOJO
    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >/dev/null 2>&1; then
        fail_test "t1_m3_f26_asset_server_flpk_first_and_loose_fallback" "failed to compile vfs_probe.mojo"
        return
    fi

    local probe_out probe_rc=0
    probe_out="$("$probe_bin" "$vfs_flpk" "$loose_dir" 2>&1)" || probe_rc=$?
    if [[ "$probe_rc" -ne 0 ]]; then
        fail_test "t1_m3_f26_asset_server_flpk_first_and_loose_fallback" "vfs_probe failed (rc=$probe_rc): $probe_out"
        return
    fi

    pass_test "t1_m3_f26_asset_server_flpk_first_and_loose_fallback"
}

# ---------------------------------------------------------------------------
# Case 4 (F27): project.package + scene.save / scene.load via MCP v2 stdio
# ---------------------------------------------------------------------------
test_f27_mcp_project_package_and_scene_save_load() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_test "t1_m3_f27_mcp_project_package_and_scene_save_load" "could not resolve flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"

    local proj="${TMPD}/mcp_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t1_m3_f27_mcp_project_package_and_scene_save_load" "flint.sh new failed"
        return
    fi
    _write_shipping_assets_dir "${proj}/assets"

    local py_out py_rc=0
    py_out="$(
        FLINT_STATE_DIR="${TMPD}/mcp_state" XDG_STATE_HOME="${TMPD}/mcp_xdg" \
        python3 -B - "$studio_bin" "$proj" "${SCRIPT_DIR}/../common" "$TMPD" <<'PY' 2>&1
import os, sys
sys.path.insert(0, sys.argv[3])
from mcp_client import McpStdioClient

studio_bin, proj_dir, other_cwd = sys.argv[1], sys.argv[2], sys.argv[4]
with McpStdioClient([studio_bin, "--project", proj_dir, "--mcp-stdio"], cwd=other_cwd, framing="ndjson", timeout=30.0, allow_stderr=False) as c:
    c.initialize()
    c.notify_initialized()
    tools = {t["name"] for t in c.list_tools()}
    required = {"scene.save", "scene.load", "project.package"}
    missing = sorted(required - tools)
    if missing:
        raise SystemExit(f"M0/M3 contract pending; missing MCP tools in tools/list: {missing}")

    sp = c.call_tool("ecs.spawn_entity", {"name": "M3Crate", "position": [1.0, 2.0, 3.0], "mass": 1.0})
    eid = sp["entity_id"]
    c.call_tool("scene.save", {"path": "scenes/m3_test.flscn"})
    if not os.path.isfile(os.path.join(proj_dir, "scenes", "m3_test.flscn")):
        raise SystemExit("scene.save did not resolve relative path against --project root")
    if os.path.exists(os.path.join(other_cwd, "scenes", "m3_test.flscn")):
        raise SystemExit("scene.save wrote relative path to cwd instead of --project root")

    c.call_tool("ecs.despawn_entity", {"entity_id": eid})
    c.call_tool("scene.load", {"path": "scenes/m3_test.flscn"})
    q = c.call_tool("ecs.query_entities", {"name_contains": "M3Crate"})
    if q.get("count", 0) < 1:
        raise SystemExit(f"scene.load did not restore M3Crate: {q}")

    pkg = c.call_tool("project.package", {"output_dir": "dist/mcp_out"}, timeout=900.0)
    print("PKG_OUT=" + pkg.get("output_dir", os.path.join(proj_dir, "dist/mcp_out")))
PY
    )" || py_rc=$?
    if [[ "$py_rc" -ne 0 ]]; then
        fail_test "t1_m3_f27_mcp_project_package_and_scene_save_load" "$py_out"
        return
    fi

    local mcp_dist="${proj}/dist/mcp_out"
    if [[ ! -x "${mcp_dist}/run.sh" || ! -f "${mcp_dist}/assets.flpk" ]]; then
        fail_test "t1_m3_f27_mcp_project_package_and_scene_save_load" "project.package bundle missing run.sh or assets.flpk in ${mcp_dist}"
        return
    fi
    if ! validate_flpk_v3 "${mcp_dist}/assets.flpk"; then
        fail_test "t1_m3_f27_mcp_project_package_and_scene_save_load" "project.package assets.flpk failed validate_flpk_v3"
        return
    fi

    pass_test "t1_m3_f27_mcp_project_package_and_scene_save_load"
}

# ---------------------------------------------------------------------------
# Case 5 (F28): flint.sh package incremental cook -> relocated clean-dir run.sh
#               + zero /home/user/flint refs + validate_flpk_v3 + --shots PNG
# ---------------------------------------------------------------------------
test_f28_package_incremental_clean_dir_and_shots() {
    local proj="${TMPD}/ship_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t1_m3_f28_package_incremental_clean_dir_and_shots" "flint.sh new failed"
        return
    fi
    _write_shipping_assets_dir "${proj}/assets"

    # Keep Case 5 compile fast (<15s vs ~70s) while exercising real EngineApp, AssetServer(.flpk), and 1280x720 PNG --shots
    cat > "${proj}/src/main.mojo" <<'MOJO'
from std.collections import List
from std.sys import argv
from flint.app import AppConfig, EngineApp
from flint.assets.image import write_png
from flint.assets.server import AssetServer
from flint.core.log import LOG_CORE, flush_log, log_info, start_log
from flint.platform.os import make_dirs, parent_directory

def main() raises:
    _ = start_log("ship_proj")
    var shots_mode = False
    var out_png = String("")
    var args = argv()
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--shots":
            shots_mode = True
        elif a == "--out" and ai + 1 < len(args):
            out_png = String(args[ai + 1])
            ai += 1
        ai += 1

    var assets = AssetServer(enable_watcher=False)
    if not assets.mount_flpk("assets.flpk"):
        raise Error("failed to mount assets.flpk")
    var rh = assets.load_raw("main.flscn")
    var rptr = assets.raws.get_ptr(rh)
    if not rptr or len(rptr.value()[unsafe_offset=0].bytes) == 0:
        raise Error("failed to load main.flscn from assets.flpk")

    var app = EngineApp(AppConfig(title="ship_proj", headless=True, enable_crash_handler=True))
    app.startup()
    app.step_one_frame(Float32(1.0 / 60.0))

    if shots_mode:
        var shot_target = out_png if out_png.byte_length() > 0 else String("build/shots/ship_proj.png")
        var pdir = parent_directory(shot_target)
        if pdir.byte_length() > 0 and pdir != ".":
            _ = make_dirs(pdir)
        var px = List[UInt32](length=1280 * 720, fill=0xFF284870)
        _ = write_png(shot_target, px, 1280, 720)

    log_info(LOG_CORE, "ship_proj_smoke_ok")
    _ = flush_log()
    assets.close()
    app.shutdown()
    print("[ship_proj] Headless v5 frame loop OK")
MOJO

    local dist1="${TMPD}/ship_dist1"
    local dist2="${TMPD}/ship_dist2"
    local pkg_out1 pkg_out2 pkg_rc=0
    pkg_out1="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" package --out "$dist1" 2>&1)" || pkg_rc=$?
    if [[ "$pkg_rc" -ne 0 ]]; then
        fail_test "t1_m3_f28_package_incremental_clean_dir_and_shots" "first flint.sh package failed (rc=$pkg_rc): $pkg_out1"
        return
    fi

    sleep 1
    pkg_out2="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" package --out "$dist2" 2>&1)" || pkg_rc=$?
    if [[ "$pkg_rc" -ne 0 ]]; then
        fail_test "t1_m3_f28_package_incremental_clean_dir_and_shots" "second flint.sh package failed (rc=$pkg_rc): $pkg_out2"
        return
    fi
    if ! grep -qE 'cooked=0, cache_hits=[1-9][0-9]*' <<<"$pkg_out2"; then
        fail_test "t1_m3_f28_package_incremental_clean_dir_and_shots" "second package was not incremental (expected cooked=0, cache_hits>0): $pkg_out2"
        return
    fi

    if ! validate_flpk_v3 "${dist2}/assets.flpk" "tex.png" "sfx.wav" "prop.glb" "main.flscn" "stone.flmat" "game.json"; then
        fail_test "t1_m3_f28_package_incremental_clean_dir_and_shots" "dist2/assets.flpk validation failed"
        return
    fi

    if grep -Fq "/home/user/flint" "${dist2}/run.sh" "${dist2}/flint.toml" || grep -Fq "${ROOT}" "${dist2}/run.sh" "${dist2}/flint.toml"; then
        fail_test "t1_m3_f28_package_incremental_clean_dir_and_shots" "run.sh or flint.toml contains repo root reference"
        return
    fi

    # Relocate bundle to an isolated clean directory and delete original dist2
    local clean_dir="${TMPD}/relocated_bundle/game"
    mkdir -p "$(dirname "$clean_dir")"
    cp -a "$dist2" "$clean_dir"
    rm -rf "$dist1" "$dist2"

    local smoke_out smoke_rc=0
    smoke_out="$(FLINT_STATE_DIR="${TMPD}/reloc_state" XDG_STATE_HOME="${TMPD}/reloc_xdg" run_in_clean_env "$clean_dir" ./run.sh --headless-smoke 2>&1)" || smoke_rc=$?
    if [[ "$smoke_rc" -ne 0 ]] || ! grep -q "\[ship_proj\] Headless v5 frame loop OK" <<<"$smoke_out"; then
        fail_test "t1_m3_f28_package_incremental_clean_dir_and_shots" "relocated clean-env --headless-smoke failed (rc=$smoke_rc): $smoke_out"
        return
    fi

    local shot_png="${clean_dir}/smoke_shot.png"
    local shots_out shots_rc=0
    shots_out="$(FLINT_STATE_DIR="${TMPD}/reloc_state" XDG_STATE_HOME="${TMPD}/reloc_xdg" run_in_clean_env "$clean_dir" ./run.sh --shots --out "$shot_png" 2>&1)" || shots_rc=$?
    if [[ "$shots_rc" -ne 0 || ! -s "$shot_png" ]]; then
        fail_test "t1_m3_f28_package_incremental_clean_dir_and_shots" "relocated clean-env --shots failed (rc=$shots_rc): $shots_out"
        return
    fi
    if ! validate_png_dims "$shot_png" 1280 720; then
        fail_test "t1_m3_f28_package_incremental_clean_dir_and_shots" "shot PNG failed validate_png_dims 1280x720"
        return
    fi

    pass_test "t1_m3_f28_package_incremental_clean_dir_and_shots"
}

# ---------------------------------------------------------------------------
# Case 6 (F28): examples/test_harness_game/ existence, packaging, clean-dir
#               run.sh golden-image & input-replay verification
# ---------------------------------------------------------------------------
test_f28_test_harness_game_package_and_checks() {
    local th_dir="${ROOT}/examples/test_harness_game"
    if [[ ! -d "$th_dir" || ! -f "${th_dir}/flint.toml" ]]; then
        fail_test "t1_m3_f28_test_harness_game_package_and_checks" "M3 F28 contract pending: ${th_dir}/flint.toml does not exist"
        return
    fi

    local th_copy="${TMPD}/th_project"
    rm -rf "$th_copy"
    mkdir -p "$th_copy"
    cp -a "${th_dir}/." "$th_copy/"
    rm -rf "${th_copy}/build" "${th_copy}/dist"

    local th_dist="${TMPD}/th_dist"
    local pkg_out pkg_rc=0
    pkg_out="$(cd "$th_copy" && bash "${ROOT}/tools/flint.sh" package --out "$th_dist" 2>&1)" || pkg_rc=$?
    if [[ "$pkg_rc" -ne 0 ]]; then
        fail_test "t1_m3_f28_test_harness_game_package_and_checks" "packaging test_harness_game failed (rc=$pkg_rc): $pkg_out"
        return
    fi

    if [[ ! -x "${th_dist}/run.sh" || ! -f "${th_dist}/assets.flpk" ]]; then
        fail_test "t1_m3_f28_test_harness_game_package_and_checks" "packaged test_harness_game missing run.sh or assets.flpk"
        return
    fi
    if ! validate_flpk_v3 "${th_dist}/assets.flpk"; then
        fail_test "t1_m3_f28_test_harness_game_package_and_checks" "packaged test_harness_game assets.flpk invalid"
        return
    fi

    local clean_th="${TMPD}/clean_th/game"
    mkdir -p "$(dirname "$clean_th")"
    cp -a "$th_dist" "$clean_th"
    rm -rf "$th_dist"

    # Run clean-dir smoke, golden-image check, and input-replay check from run.sh
    local run_out run_rc=0
    run_out="$(FLINT_STATE_DIR="${TMPD}/th_state" XDG_STATE_HOME="${TMPD}/th_xdg" run_in_clean_env "$clean_th" ./run.sh --headless-smoke 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 ]]; then
        fail_test "t1_m3_f28_test_harness_game_package_and_checks" "clean-dir ./run.sh --headless-smoke failed (rc=$run_rc): $run_out"
        return
    fi

    # Verify documented golden-image & input-replay entry points in test_harness_game/run.sh
    local has_golden=0 has_replay=0
    local check_out
    for flag in "--verify-golden" "--golden-check" "--test-harness"; do
        if check_out="$(FLINT_STATE_DIR="${TMPD}/th_state" XDG_STATE_HOME="${TMPD}/th_xdg" run_in_clean_env "$clean_th" ./run.sh "$flag" 2>&1)"; then
            if grep -qiE "golden|flip" <<<"$check_out" && grep -qE '\b(PASS|OK)\b|\[PASS\]' <<<"$check_out"; then
                has_golden=1
                break
            fi
        fi
    done
    if [[ "$has_golden" -eq 0 ]]; then
        local shot_out="${clean_th}/th_shot.png"
        local golden_ref=""
        golden_ref="$(find "$th_dir" "$clean_th" -iname '*golden*.png' -print -quit 2>/dev/null || true)"
        if [[ -n "$golden_ref" ]] && FLINT_STATE_DIR="${TMPD}/th_state" XDG_STATE_HOME="${TMPD}/th_xdg" run_in_clean_env "$clean_th" ./run.sh --shots --out "$shot_out" >/dev/null 2>&1; then
            local flip_src="${TMPD}/t1_th_flip.mojo"
            local flip_bin="${TMPD}/bin/t1_th_flip"
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
                    has_golden=1
                fi
            fi
        fi
    fi
    for flag in "--replay" "--input-replay" "--verify-replay" "--test-harness"; do
        if check_out="$(FLINT_STATE_DIR="${TMPD}/th_state" XDG_STATE_HOME="${TMPD}/th_xdg" run_in_clean_env "$clean_th" ./run.sh "$flag" 2>&1)"; then
            if grep -qiE "replay|sdrp" <<<"$check_out" && grep -qE '\b(PASS|OK)\b|\[PASS\]' <<<"$check_out"; then
                has_replay=1
                break
            fi
        fi
    done
    if [[ "$has_golden" -ne 1 || "$has_replay" -ne 1 ]]; then
        fail_test "t1_m3_f28_test_harness_game_package_and_checks" "test_harness_game run.sh missing golden-image (ok=$has_golden) or input-replay (ok=$has_replay) verification"
        return
    fi

    pass_test "t1_m3_f28_test_harness_game_package_and_checks"
}

test_f26_new_scaffold_main_flscn_and_config
test_f26_flint_version_pin_matches_build_info
test_f26_asset_server_flpk_first_and_loose_fallback
test_f27_mcp_project_package_and_scene_save_load
test_f28_package_incremental_clean_dir_and_shots
test_f28_test_harness_game_package_and_checks

print_summary_and_exit
