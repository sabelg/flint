#!/usr/bin/env bash
# AREA: r1
# Tier 1 Feature Coverage: Area A — R1 Hygiene, Crash Handler/Logs & Cooker/Packer (F01-F05, Bug #6)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t1_r1"
resolve_mojo_env || {
    fail_test "t1_r1_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

_write_test_assets_dir() {
    local target_dir="$1"
    mkdir -p "$target_dir"
    python3 - "$target_dir" <<'PY'
import json, math, os, struct, sys, zlib

out_dir = sys.argv[1]

# 1. Valid 8x8 RGBA PNG
w, h = 8, 8
raw_rows = bytearray()
for y in range(h):
    raw_rows.append(0)  # filter=0
    for x in range(w):
        raw_rows.extend((64, 160, 224, 255))

def png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)

ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
idat = zlib.compress(bytes(raw_rows))
png_bytes = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", idat) + png_chunk(b"IEND", b"")
with open(os.path.join(out_dir, "tex.png"), "wb") as f:
    f.write(png_bytes)

# 2. Valid 48kHz stereo PCM16 WAV
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
with open(os.path.join(out_dir, "sfx.wav"), "wb") as f:
    f.write(wav)

# 3. Valid minimal glTF 2.0 GLB (single triangle)
positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
normals = struct.pack("<9f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
indices = struct.pack("<3H", 0, 1, 2) + b"\x00\x00"
bin_blob = positions + normals + indices
while len(bin_blob) % 4 != 0:
    bin_blob += b"\x00"

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
with open(os.path.join(out_dir, "mesh.glb"), "wb") as f:
    f.write(glb)

# 4. Remaining R1 extensions (.qoa, .flscn, .flscnb, .flprefab, .flmat, .flanim, .flrig, .json)
files = {
    "music.qoa": b"qoaf\x00\x00\x00\x10" + b"\x00" * 16,
    "main.flscn": b'{"schema":2,"entities":[]}\n',
    "packed.flscnb": b"FLSCNB\x02\x00\x00\x00",
    "prop.flprefab": b'{"name":"prop","entities":[]}\n',
    "wood.flmat": b'{"base_color":[0.6,0.4,0.2,1.0],"roughness":0.7}\n',
    "idle.flanim": b'{"clip":"idle","fps":30}\n',
    "hero.flrig": b"FRIG\x01\x00\x00\x00",
    "config.json": b'{"title":"r1_test"}\n',
}
for name, data in files.items():
    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(data)
PY
}

# ---------------------------------------------------------------------------
# Case 1 (F01): Zero Maroon/deepsong/sharddepth/SD_* leaks or gameplay/world imports in src/flint/
# ---------------------------------------------------------------------------
test_f01_zero_leaks() {
    local matches imp_matches
    matches="$(grep -rnE "Maroon|maroon|MAROON|deepsong|Deepsong|DEEPSONG|sharddepth|Sharddepth|SHARDDEPTH|SD_[A-Z0-9_]+" "${ROOT}/src/flint/" || true)"
    imp_matches="$(grep -rnE "^from (gameplay|world)(\.| )" "${ROOT}/src/flint/" || true)"
    if [[ -z "$matches" && -z "$imp_matches" ]]; then
        pass_test "t1_r1_f01_zero_game_leaks_in_src_flint"
    else
        local combined=""
        [[ -n "$matches" ]] && combined+="$matches"$'\n'
        [[ -n "$imp_matches" ]] && combined+="$imp_matches"
        local count
        count="$(printf '%s\n' "$combined" | sed '/^$/d' | wc -l)"
        fail_test "t1_r1_f01_zero_game_leaks_in_src_flint" "found ${count} leak line(s) in src/flint/; first: $(printf '%s\n' "$combined" | sed '/^$/d' | head -n 1)"
    fi
}

# ---------------------------------------------------------------------------
# Case 2 (F02): Extended Zero-Leak Gate in sync_from_deepsong.sh & tools/flint.sh
# ---------------------------------------------------------------------------
test_f02_sync_verify_and_gate() {
    # (a) Inspect ONLY non-comment, non-echo active leak-check lines after '# Zero-Game-Leak Gate'
    local active_gate_lines
    active_gate_lines="$(
        awk '
            /# Zero-Game-Leak Gate/ { in_gate=1; next }
            in_gate {
                sub(/^[[:space:]]+/, "")
                if ($0 ~ /^#/ || $0 ~ /^echo[[:space:]]/ || $0 == "") next
                print
            }
        ' "${ROOT}/tools/sync_from_deepsong.sh"
    )"
    local tok
    for tok in "Maroon" "deepsong" "sharddepth" "SD_" "gameplay|world"; do
        if ! grep -Fq "$tok" <<<"$active_gate_lines"; then
            fail_test "t1_r1_f02_sync_verify_and_gate_leak_check" "active Zero-Game-Leak Gate lines missing token: ${tok}"
            return
        fi
    done

    # Ensure cmd_gate invokes cmd_sync --verify
    if ! sed -n '/^cmd_gate()/,/^}/p' "${ROOT}/tools/flint.sh" | grep -q "cmd_sync --verify"; then
        fail_test "t1_r1_f02_sync_verify_and_gate_leak_check" "tools/flint.sh cmd_gate does not invoke cmd_sync --verify"
        return
    fi

    # (b) Run real flint.sh sync --verify
    local sync_out rc
    sync_out="$(bash "${ROOT}/tools/flint.sh" sync --verify 2>&1)"
    rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t1_r1_f02_sync_verify_and_gate_leak_check" "flint.sh sync --verify exited $rc: $sync_out"
        return
    fi
    if ! grep -q "Zero-Game-Leak Gate" <<<"$sync_out"; then
        fail_test "t1_r1_f02_sync_verify_and_gate_leak_check" "missing '[PASS] Zero-Game-Leak Gate' output"
        return
    fi

    # (c) Isolated negative test in mock_leak_repo
    local mock_repo="${TMPD}/mock_leak_repo"
    mkdir -p "${mock_repo}/tools" "${mock_repo}/src/flint"
    cp -f "${ROOT}/tools/sync_from_deepsong.sh" "${mock_repo}/tools/sync_from_deepsong.sh"
    python3 - "${mock_repo}/tools/sync_from_deepsong.sh" "${mock_repo}/src/flint" <<'PY'
import ast, os, re, sys
sync_path, dst_root = sys.argv[1], sys.argv[2]
with open(sync_path, "r", encoding="utf-8") as fh:
    src = fh.read()
m_enh = re.search(r"FLINT_ENHANCED\s*=\s*(\{.*?\n\})", src, re.DOTALL)
m_mod = re.search(r"MODULES\s*=\s*(\{.*?\n\})", src, re.DOTALL)
enhanced = ast.literal_eval(m_enh.group(1))
modules = ast.literal_eval(m_mod.group(1))
for pkg, fname in enhanced:
    os.makedirs(os.path.join(dst_root, pkg), exist_ok=True)
    open(os.path.join(dst_root, pkg, fname), "w").close()
for pkg, files in modules.items():
    os.makedirs(os.path.join(dst_root, pkg), exist_ok=True)
    for fname in files:
        open(os.path.join(dst_root, pkg, fname), "w").close()
PY
    if ! bash "${mock_repo}/tools/sync_from_deepsong.sh" --verify >/dev/null 2>&1; then
        fail_test "t1_r1_f02_sync_verify_and_gate_leak_check" "clean mock_leak_repo baseline failed sync --verify"
        return
    fi

    local probe_file="${mock_repo}/src/flint/core/leak_probe.mojo"
    local -a leak_cases=(
        'SD_LEAK_TEST|comptime TEST_LEAK: String = "SD_LEAK_TEST"'
        'Maroon|# Maroon'
        'deepsong|# deepsong'
        'sharddepth|# sharddepth'
        'from gameplay.leak|from gameplay.leak import thing'
    )
    local entry label payload neg_rc
    for entry in "${leak_cases[@]}"; do
        label="${entry%%|*}"
        payload="${entry#*|}"
        printf '%s\n' "$payload" > "$probe_file"
        neg_rc=0
        bash "${mock_repo}/tools/sync_from_deepsong.sh" --verify >/dev/null 2>&1 || neg_rc=$?
        if [[ "$neg_rc" -eq 0 ]]; then
            fail_test "t1_r1_f02_sync_verify_and_gate_leak_check" "mock_leak_repo with '${label}' alone unexpectedly passed --verify"
            return
        fi
    done
    rm -f "$probe_file"

    pass_test "t1_r1_f02_sync_verify_and_gate_leak_check"
}

# ---------------------------------------------------------------------------
# Case 3 (F03 + F01): flint.sh new crash handler, start_log(cfg.app_name), FLINT_CRASH
# ---------------------------------------------------------------------------
test_f03_new_logs_and_crash_handler() {
    local proj="${TMPD}/proj_alpha"
    local new_out
    if ! new_out="$(bash "${ROOT}/tools/flint.sh" new "$proj" 2>&1)"; then
        fail_test "t1_r1_f03_new_logs_and_crash_handler" "flint.sh new failed: $new_out"
        return
    fi

    local req
    for req in flint.toml pixi.toml .gitignore README.md tools/run.sh src/main.mojo; do
        if [[ ! -f "${proj}/${req}" ]]; then
            fail_test "t1_r1_f03_new_logs_and_crash_handler" "missing scaffolded file ${req}"
            return
        fi
    done

    if ! python3 - "${proj}/src/main.mojo" <<'PY'
import re, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    src = fh.read()
m_log = re.search(r"\bstart_log\(\s*cfg\.app_name\s*\)", src)
m_startup = re.search(r"\bapp\.startup\(\)", src)
if not m_log:
    raise SystemExit("missing start_log(cfg.app_name)")
if not m_startup:
    raise SystemExit("missing app.startup()")
if m_log.start() >= m_startup.start():
    raise SystemExit("start_log(cfg.app_name) must precede app.startup()")
m_cfg = re.search(r"\bAppConfig\s*\(", src)
if not m_cfg:
    raise SystemExit("missing AppConfig(")
depth = 0
end_pos = -1
for idx in range(m_cfg.end() - 1, len(src)):
    ch = src[idx]
    if ch == "(":
        depth += 1
    elif ch == ")":
        depth -= 1
        if depth == 0:
            end_pos = idx + 1
            break
if end_pos == -1:
    raise SystemExit("unterminated AppConfig(")
cfg_call = src[m_cfg.start():end_pos]
if not re.search(r"\benable_crash_handler\s*=\s*True\b", cfg_call):
    raise SystemExit(f"AppConfig missing enable_crash_handler=True: {cfg_call}")
PY
    then
        fail_test "t1_r1_f03_new_logs_and_crash_handler" "scaffolded src/main.mojo failed start_log/AppConfig crash_handler order check"
        return
    fi
    if grep -qiE "sharddepth|deepsong|maroon" "${proj}/src/main.mojo"; then
        fail_test "t1_r1_f03_new_logs_and_crash_handler" "scaffolded src/main.mojo contains legacy game/engine leak"
        return
    fi

    local state_dir="${TMPD}/state_alpha"
    local smoke_out smoke_rc
    smoke_out="$(cd "$proj" && FLINT_ROOT="$ROOT" FLINT_STATE_DIR="$state_dir" XDG_STATE_HOME="${TMPD}/xdg_alpha" bash tools/run.sh --headless-smoke 2>&1)"
    smoke_rc=$?
    if [[ "$smoke_rc" -ne 0 ]] || ! grep -q "\[proj_alpha\] Headless v5 frame loop OK" <<<"$smoke_out"; then
        fail_test "t1_r1_f03_new_logs_and_crash_handler" "run.sh --headless-smoke failed (rc=$smoke_rc): $smoke_out"
        return
    fi

    local expected_log="${state_dir}/logs/proj_alpha.log"
    if [[ ! -s "$expected_log" ]]; then
        fail_test "t1_r1_f03_new_logs_and_crash_handler" "expected non-empty log at ${expected_log}; found: $(ls -la "${state_dir}/logs" 2>&1 || true)"
        return
    fi

    # Crash handler probe THROUGH EngineApp(AppConfig(..., enable_crash_handler=True))
    local probe_src="${TMPD}/crash_probe.mojo"
    local probe_bin="${TMPD}/bin/crash_probe"
    cat >"$probe_src" <<'MOJO'
from std.ffi import external_call
from flint.app import AppConfig, EngineApp
from flint.core.log import LOG_CORE, flush_log, log_info, start_log

def main() raises:
    _ = start_log("proj_alpha")
    var app = EngineApp(AppConfig(title="proj_alpha", headless=True, enable_crash_handler=True))
    app.startup()
    log_info(LOG_CORE, "about_to_abort")
    _ = flush_log()
    external_call["abort", NoneType]()
MOJO
    if ! "$MOJO_BIN" build -I "${ROOT}/src" -O3 --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >/dev/null 2>&1; then
        fail_test "t1_r1_f03_new_logs_and_crash_handler" "failed to compile EngineApp crash probe"
        return
    fi

    local crash_state="${TMPD}/state_crash"
    local crash_err="${TMPD}/crash.stderr"
    local crash_rc=0
    bash -c 'FLINT_STATE_DIR="$1" "$2" >/dev/null 2>"$3"' _ "$crash_state" "$probe_bin" "$crash_err" >/dev/null 2>&1 || crash_rc=$?
    if [[ "$crash_rc" -eq 0 ]]; then
        fail_test "t1_r1_f03_new_logs_and_crash_handler" "crash probe unexpectedly exited 0"
        return
    fi
    if ! grep -q "FLINT_CRASH" "$crash_err" || grep -q "SHARDDEPTH_CRASH" "$crash_err"; then
        fail_test "t1_r1_f03_new_logs_and_crash_handler" "stderr missing FLINT_CRASH or has SHARDDEPTH_CRASH: $(cat "$crash_err")"
        return
    fi
    local crash_txt
    crash_txt="$(find "${crash_state}/crashes" -name 'crash.txt' -print -quit 2>/dev/null || true)"
    if [[ -z "$crash_txt" || ! -s "$crash_txt" ]]; then
        fail_test "t1_r1_f03_new_logs_and_crash_handler" "no crash.txt generated under ${crash_state}/crashes"
        return
    fi
    if grep -qi "Sharddepth" "$crash_txt" || ! grep -q "Flint crash report" "$crash_txt" || ! grep -q "proj_alpha" "$crash_txt"; then
        fail_test "t1_r1_f03_new_logs_and_crash_handler" "crash.txt invalid (must contain 'Flint crash report' + 'proj_alpha', no 'Sharddepth'): $(head -n 6 "$crash_txt")"
        return
    fi

    pass_test "t1_r1_f03_new_logs_and_crash_handler"
}

# ---------------------------------------------------------------------------
# Case 4 (F04 / Bug #6a): flint.sh package uses cook_workspace + incremental cache + clean-env run.sh
# ---------------------------------------------------------------------------
test_f04_package_cook_and_cache() {
    local proj="${TMPD}/pkg_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t1_r1_f04_package_uses_cook_and_cache" "flint.sh new failed"
        return
    fi
    # Lightweight main.mojo (well within 180s) that mounts assets.flpk via AssetServer and reads packed entries
    cat > "${proj}/src/main.mojo" <<'MOJO'
from std.sys import argv
from flint.app import AppConfig, EngineApp
from flint.assets.server import AssetServer
from flint.core.log import LOG_CORE, flush_log, log_info, start_log

def main() raises:
    _ = start_log("pkg_proj")
    var headless = False
    for a in argv():
        if a == "--headless-smoke":
            headless = True
    var assets = AssetServer(enable_watcher=False)
    if not assets.mount_flpk("assets.flpk"):
        raise Error("failed to mount assets.flpk")
    var rh = assets.load_raw("sub/config.json")
    var rptr = assets.raws.get_ptr(rh)
    if not rptr or len(rptr.value()[unsafe_offset=0].bytes) == 0:
        raise Error("failed to read sub/config.json from assets.flpk")
    var th = assets.load_texture("sub/tex.png", progressive=False)
    var tptr = assets.textures.get_ptr(th)
    if not tptr or tptr.value()[unsafe_offset=0].width != 8:
        raise Error("failed to load sub/tex.png from assets.flpk")
    var app = EngineApp(AppConfig(title="pkg_proj", headless=headless, enable_crash_handler=True))
    app.startup()
    app.step_one_frame(Float32(1.0 / 60.0))
    log_info(LOG_CORE, "pkg_proj_smoke_ok")
    _ = flush_log()
    assets.close()
    app.shutdown()
    print("[pkg_proj] Headless v5 frame loop OK (raw_bytes=" + String(len(rptr.value()[unsafe_offset=0].bytes)) + ")")
MOJO
    _write_test_assets_dir "${proj}/assets/sub"

    local dist1="${TMPD}/dist_1"
    local dist2="${TMPD}/dist_2"
    local pkg_out
    if ! pkg_out="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" package --out "$dist1" 2>&1)"; then
        fail_test "t1_r1_f04_package_uses_cook_and_cache" "first flint.sh package failed: $pkg_out"
        return
    fi

    local req
    for req in "bin/pkg_proj" "run.sh" "flint.toml" "MANIFEST.txt" "assets.flpk"; do
        if [[ ! -e "${dist1}/${req}" ]]; then
            fail_test "t1_r1_f04_package_uses_cook_and_cache" "bundle missing ${req}"
            return
        fi
    done

    if grep -Fq "/home/user/flint" "${dist1}/run.sh" "${dist1}/flint.toml" || grep -Fq "${ROOT}" "${dist1}/run.sh" "${dist1}/flint.toml"; then
        fail_test "t1_r1_f04_package_uses_cook_and_cache" "dist1 run.sh or flint.toml leaks repo root path"
        return
    fi

    if ! validate_flpk_v3 "${dist1}/assets.flpk" "sub/tex.png" "sub/sfx.wav" "sub/mesh.glb" "sub/main.flscn" "sub/wood.flmat" "sub/config.json"; then
        fail_test "t1_r1_f04_package_uses_cook_and_cache" "dist1/assets.flpk missing expected cooked/raw entries (Bug #6)"
        return
    fi

    local stats_before stats_after
    stats_before="$(find "${proj}/build" -name '*.flcook' -exec stat -c '%n:%i:%Y' {} + 2>/dev/null | LC_ALL=C sort || true)"
    if [[ -z "$stats_before" ]]; then
        fail_test "t1_r1_f04_package_uses_cook_and_cache" "flint.sh package did not populate project cook cache (*.flcook) under ${proj}/build"
        return
    fi

    sleep 1
    if ! pkg_out="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" package --out "$dist2" 2>&1)"; then
        fail_test "t1_r1_f04_package_uses_cook_and_cache" "second flint.sh package failed: $pkg_out"
        return
    fi
    stats_after="$(find "${proj}/build" -name '*.flcook' -exec stat -c '%n:%i:%Y' {} + 2>/dev/null | LC_ALL=C sort || true)"
    if [[ "$stats_before" != "$stats_after" ]]; then
        fail_test "t1_r1_f04_package_uses_cook_and_cache" "cook cache .flcook file(s) changed on unchanged second package"
        return
    fi
    if ! grep -qE 'cooked=0, cache_hits=[1-9][0-9]*' <<<"$pkg_out"; then
        fail_test "t1_r1_f04_package_uses_cook_and_cache" "second package output did not report cooked=0 with cache_hits>0: $pkg_out"
        return
    fi
    if ! validate_flpk_v3 "${dist2}/assets.flpk" "sub/tex.png" "sub/sfx.wav" "sub/mesh.glb" "sub/main.flscn" "sub/wood.flmat" "sub/config.json"; then
        fail_test "t1_r1_f04_package_uses_cook_and_cache" "dist2/assets.flpk validation failed"
        return
    fi

    local run_out run_rc
    run_out="$(FLINT_STATE_DIR="${TMPD}/state_pkg" run_in_clean_env "$dist1" ./run.sh --headless-smoke 2>&1)"
    run_rc=$?
    if [[ "$run_rc" -ne 0 ]] || ! grep -q "\[pkg_proj\] Headless v5 frame loop OK" <<<"$run_out"; then
        fail_test "t1_r1_f04_package_uses_cook_and_cache" "clean-env bundle run.sh failed (rc=$run_rc): $run_out"
        return
    fi

    pass_test "t1_r1_f04_package_uses_cook_and_cache"
}

# ---------------------------------------------------------------------------
# Case 5 (F05 / Bug #6b): flint-pack --cook --cache-dir across all 11 R1 extensions
# ---------------------------------------------------------------------------
test_f05_cooker_extensions_and_cache_dir() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t1_r1_f05_cooker_extensions_and_cache_dir" "could not resolve/build flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"
    local ws="${TMPD}/ws_all"
    _write_test_assets_dir "${ws}/nested"

    local out_flpk="${TMPD}/all_r1.flpk"
    local cache_dir="${TMPD}/custom_cache"
    local cook_out cook_rc
    cook_out="$("$pack_bin" --cook "$ws" "$out_flpk" --cache-dir "$cache_dir" 2>&1)"
    cook_rc=$?
    if [[ "$cook_rc" -ne 0 ]]; then
        fail_test "t1_r1_f05_cooker_extensions_and_cache_dir" "flint-pack --cook --cache-dir failed (rc=$cook_rc): $cook_out"
        return
    fi

    local expected_rels=(
        "nested/mesh.glb"
        "nested/tex.png"
        "nested/sfx.wav"
        "nested/music.qoa"
        "nested/main.flscn"
        "nested/packed.flscnb"
        "nested/prop.flprefab"
        "nested/wood.flmat"
        "nested/idle.flanim"
        "nested/hero.flrig"
        "nested/config.json"
    )
    if ! validate_flpk_v3 "$out_flpk" "${expected_rels[@]}"; then
        fail_test "t1_r1_f05_cooker_extensions_and_cache_dir" "FLPK v3 archive validation failed for 11 R1 extensions"
        return
    fi

    local insp_out
    if ! insp_out="$("$pack_bin" --inspect "$out_flpk" 2>&1)"; then
        fail_test "t1_r1_f05_cooker_extensions_and_cache_dir" "flint-pack --inspect failed: $insp_out"
        return
    fi
    local rel
    for rel in "${expected_rels[@]}"; do
        if ! grep -Fq "$rel" <<<"$insp_out"; then
            fail_test "t1_r1_f05_cooker_extensions_and_cache_dir" "--inspect output missing ${rel}"
            return
        fi
    done

    if ! compgen -G "${cache_dir}/*.flcook" >/dev/null || ! compgen -G "${cache_dir}/workspace_stamp_*.bin" >/dev/null; then
        fail_test "t1_r1_f05_cooker_extensions_and_cache_dir" "custom --cache-dir ${cache_dir} missing .flcook or workspace_stamp_*.bin"
        return
    fi

    pass_test "t1_r1_f05_cooker_extensions_and_cache_dir"
}

# ---------------------------------------------------------------------------
# Case 6: flint-pack --help, --raw --no-page-align, --inspect round-trip
# ---------------------------------------------------------------------------
test_r1_pack_cli_raw_inspect_help() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t1_r1_pack_cli_raw_inspect_help" "could not resolve/build flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"

    if ! "$pack_bin" --help >/dev/null 2>&1; then
        fail_test "t1_r1_pack_cli_raw_inspect_help" "--help exited non-zero"
        return
    fi

    local raw_dir="${TMPD}/raw_ws"
    mkdir -p "$raw_dir"
    printf '{"hello":"raw"}\n' > "${raw_dir}/sample.json"
    local raw_flpk="${TMPD}/raw.flpk"
    if ! (cd "$raw_dir" && "$pack_bin" --raw --no-page-align "$raw_flpk" "sample.json" >/dev/null 2>&1); then
        fail_test "t1_r1_pack_cli_raw_inspect_help" "--raw --no-page-align failed"
        return
    fi
    if ! validate_flpk_v3 "$raw_flpk" "sample.json"; then
        fail_test "t1_r1_pack_cli_raw_inspect_help" "raw flpk validation failed"
        return
    fi
    if ! "$pack_bin" --inspect "$raw_flpk" | grep -Fq "sample.json"; then
        fail_test "t1_r1_pack_cli_raw_inspect_help" "--inspect missing sample.json"
        return
    fi

    pass_test "t1_r1_pack_cli_raw_inspect_help"
}

test_f01_zero_leaks
test_f02_sync_verify_and_gate
test_f03_new_logs_and_crash_handler
test_f04_package_cook_and_cache
test_f05_cooker_extensions_and_cache_dir
test_r1_pack_cli_raw_inspect_help

print_summary_and_exit
