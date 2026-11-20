#!/usr/bin/env bash
# M5/M6 Gated Opaque-Box CLI & Deliverable Verification Suite.
# Verified Feature IDs: F29, F30, F31, F32, F33, F34
#   - F29: Rebased examples/minimal_spark & examples/open_world onto EngineApp/ECS/PhysicsWorld/ScalableRenderer
#   - F30: Rebased Flint Studio (zero StudioEntityNode) + multi-select, reparenting, async picking
#   - F31: Honest multi-frame tests/e2e/ integration (ECS -> Physics -> Anim -> Render -> Audio -> Net)
#   - F32: All 5 labs/ emitting EngineBenchSuite CSVs, Flint perf_history.mojo metrics, FLIP/p99 gate
#   - F33: tools/flint.sh package bundler, clean stdout byte-0 `flint.sh mcp`, v5 `EngineApp` `flint.sh new`
#   - F34: Deletion of legacy render stack, dead-code modules, 4 orphaned tests & clean sync_from_deepsong.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ -d "${ROOT}/.pixi/envs/default" ]]; then
    export MOJO_PREFIX="${ROOT}/.pixi/envs/default"
    export MODULAR_HOME="${MOJO_PREFIX}/share/max"
    export LD_LIBRARY_PATH="${MOJO_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT

PASS=0
FAIL=0
TOTAL=0

pass_test() {
    local name="$1"
    PASS=$((PASS + 1))
    TOTAL=$((TOTAL + 1))
    echo "[PASS] ${name}"
}

fail_test() {
    local name="$1"
    local reason="${2:-assertion failed}"
    FAIL=$((FAIL + 1))
    TOTAL=$((TOTAL + 1))
    echo "[FAIL] ${name}: ${reason}" >&2
}

validate_png_dims() {
    local png_path="$1"
    local exp_w="$2"
    local exp_h="$3"
    python3 - "$png_path" "$exp_w" "$exp_h" <<'PY' 2>&1
import struct, sys, zlib
path, exp_w, exp_h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "rb") as f:
    data = f.read()
assert len(data) >= 8, f"{path}: shorter than PNG signature ({len(data)})"
assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: invalid PNG magic {data[:8]!r}"
pos = 8
chunks = []
idat_chunks = []
saw_iend = False
while pos < len(data):
    assert pos + 12 <= len(data), f"{path}: truncated PNG chunk header at {pos}"
    clen = struct.unpack_from(">I", data, pos)[0]
    ctype = data[pos + 4:pos + 8]
    cstart = pos + 8
    cend = cstart + clen
    assert cend + 4 <= len(data), f"{path}: truncated PNG chunk {ctype!r} (len={clen}) at {pos}"
    cdata = data[cstart:cend]
    ccrc = struct.unpack_from(">I", data, cend)[0]
    calc_crc = zlib.crc32(ctype + cdata) & 0xFFFFFFFF
    assert ccrc == calc_crc, f"{path}: CRC mismatch in chunk {ctype!r}: {ccrc:#x} != {calc_crc:#x}"
    chunks.append((ctype, cdata))
    if ctype == b"IDAT":
        assert clen > 0, f"{path}: empty IDAT chunk"
        idat_chunks.append(cdata)
    elif ctype == b"IEND":
        assert clen == 0, f"{path}: non-empty IEND chunk (len={clen})"
        saw_iend = True
        pos = cend + 4
        break
    pos = cend + 4
assert len(chunks) >= 3, f"{path}: expected at least IHDR+IDAT+IEND, got {len(chunks)} chunks"
assert chunks[0][0] == b"IHDR", f"{path}: first chunk is {chunks[0][0]!r}, expected IHDR"
ihdr = chunks[0][1]
assert len(ihdr) == 13, f"{path}: invalid IHDR length {len(ihdr)} != 13"
w, h, bit_depth, color_type, _comp_method, _filter_method, interlace = struct.unpack(">IIBBBBB", ihdr)
assert (w, h) == (exp_w, exp_h), f"{path}: expected {exp_w}x{exp_h}, got {w}x{h}"
assert len(idat_chunks) >= 1, f"{path}: missing IDAT chunk"
assert saw_iend and pos == len(data), f"{path}: missing IEND or trailing bytes after IEND ({pos} != {len(data)})"
raw = zlib.decompress(b"".join(idat_chunks))
assert len(raw) > 0, f"{path}: decompressed IDAT is empty"
if interlace == 0 and bit_depth == 8 and color_type in (0, 2, 3, 4, 6):
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    expected_raw_len = h * (1 + w * channels)
    assert len(raw) == expected_raw_len, (
        f"{path}: decoded scanline bytes {len(raw)} != expected {expected_raw_len} ({w}x{h}x{channels})"
    )
PY
}

# 1. F33: flint.sh package release bundle creation & standalone execution outside /home/user/flint
gated_f33_flint_package_bundle() {
    if ! grep -qE "^cmd_package\(\)" "${ROOT}/tools/flint.sh"; then
        fail_test "gated_f33_flint_package_bundle" "cmd_package() missing in tools/flint.sh"
        return
    fi
    local help_out="${TMPD}/flint_help.out"
    if ! bash "${ROOT}/tools/flint.sh" help >"$help_out" 2>"${TMPD}/flint_help.err" || \
       ! grep -qE "^[[:space:]]+package\b" "$help_out"; then
        fail_test "gated_f33_flint_package_bundle" "'package' subcommand not listed in flint.sh help"
        return
    fi
    local proj="${TMPD}/pkg_game"
    local pkg_out="${TMPD}/pkg_out"
    mkdir -p "$pkg_out"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >"${TMPD}/f33_pkg_new.out" 2>"${TMPD}/f33_pkg_new.err"; then
        fail_test "gated_f33_flint_package_bundle" "flint.sh new failed"
        return
    fi
    mkdir -p "${proj}/assets"
    printf 'flint release bundle asset\n' >"${proj}/assets/sample.txt"

    local rc=0
    (cd "$proj" && bash "${ROOT}/tools/flint.sh" package "$pkg_out") >"${TMPD}/f33_pkg.out" 2>"${TMPD}/f33_pkg.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        rc=0
        (cd "$proj" && bash "${ROOT}/tools/flint.sh" package) >"${TMPD}/f33_pkg.out" 2>"${TMPD}/f33_pkg.err" || rc=$?
    fi
    if [[ "$rc" -ne 0 ]]; then
        fail_test "gated_f33_flint_package_bundle" "flint.sh package failed (rc=${rc})"
        return
    fi

    # Locate packaged executable ONLY in $pkg_out or ${proj}/dist (never fall back to ${proj}/build)
    local exe_path=""
    exe_path="$(find "$pkg_out" -maxdepth 4 -type f -executable ! -name '*.sh' 2>/dev/null | sed -n '1p' || true)"
    if [[ -z "$exe_path" && -d "${proj}/dist" ]]; then
        exe_path="$(find "${proj}/dist" -maxdepth 4 -type f -executable ! -name '*.sh' 2>/dev/null | sed -n '1p' || true)"
    fi
    if [[ -z "$exe_path" || ! -x "$exe_path" ]]; then
        fail_test "gated_f33_flint_package_bundle" "no packaged executable found in ${pkg_out} or ${proj}/dist"
        return
    fi
    local exe_dir bundle_root rel_sub
    exe_dir="$(dirname "$exe_path")"
    if [[ "$(basename "$exe_dir")" == "bin" ]]; then
        bundle_root="$(dirname "$exe_dir")"
    else
        bundle_root="$exe_dir"
    fi
    rel_sub="${exe_path#${bundle_root}/}"
    local standalone="${TMPD}/standalone_release"
    rm -rf "$standalone"
    cp -r "$bundle_root" "$standalone"
    local rel_exe="${standalone}/${rel_sub}"

    local assets_list="${TMPD}/f33_standalone_assets.txt"
    find "$standalone" -maxdepth 4 -type f \( -name 'sample.txt' -o -name '*.flpk' \) > "$assets_list" 2>/dev/null || true
    if ! grep -q . "$assets_list"; then
        fail_test "gated_f33_flint_package_bundle" "packaged bundle missing game assets (sample.txt or *.flpk)"
        return
    fi

    local run_rc=0
    if [[ -f "${standalone}/run.sh" ]]; then
        (cd "$standalone" && env -u FLINT_ROOT -u LD_LIBRARY_PATH "${standalone}/run.sh" --headless-smoke) >"${TMPD}/f33_pkg_run.out" 2>"${TMPD}/f33_pkg_run.err" || run_rc=$?
        if [[ "$run_rc" -ne 0 ]]; then
            run_rc=0
            (cd "$standalone" && env -u FLINT_ROOT -u LD_LIBRARY_PATH "${standalone}/run.sh" --smoke) >"${TMPD}/f33_pkg_run.out" 2>"${TMPD}/f33_pkg_run.err" || run_rc=$?
        fi
    else
        (cd "$standalone" && env -u FLINT_ROOT "$rel_exe" --headless-smoke) >"${TMPD}/f33_pkg_run.out" 2>"${TMPD}/f33_pkg_run.err" || run_rc=$?
        if [[ "$run_rc" -ne 0 ]]; then
            run_rc=0
            (cd "$standalone" && env -u FLINT_ROOT "$rel_exe" --smoke) >"${TMPD}/f33_pkg_run.out" 2>"${TMPD}/f33_pkg_run.err" || run_rc=$?
        fi
    fi
    if [[ "$run_rc" -eq 0 ]]; then
        pass_test "gated_f33_flint_package_bundle"
    else
        fail_test "gated_f33_flint_package_bundle" "standalone packaged binary failed outside repo (rc=${run_rc})"
    fi
}

# 2. F33: bash tools/flint.sh mcp clean stdout byte-0 JSON-RPC check
gated_f33_mcp_clean_stdout() {
    local py_err=""
    if py_err="$(python3 - "${ROOT}/tools/flint.sh" <<'PY' 2>&1
import json, subprocess, sys

flint_sh = sys.argv[1]
req = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "gated-e2e", "version": "1.0"}},
}, separators=(",", ":")) + "\n"

p = subprocess.run(["bash", flint_sh, "mcp"], input=req.encode("utf-8"), capture_output=True, timeout=180)
assert p.returncode == 0, f"flint.sh mcp rc={p.returncode}, stderr={p.stderr!r}"
out = p.stdout
assert len(out) > 0, "empty stdout from flint.sh mcp"
assert out.startswith(b"{") or out.startswith(b"Content-Length:"), (
    f"byte-0 stdout polluted (expected '{{' or 'Content-Length:'): {out[:120]!r}"
)
assert b"[Flint]" not in out, f"stdout contains [Flint] build noise: {out[:200]!r}"
line = out.decode("utf-8").splitlines()[0]
if out.startswith(b"Content-Length:"):
    sep = out.find(b"\r\n\r\n")
    assert sep > 0
    line = out[sep + 4:].decode("utf-8")
msg = json.loads(line)
assert msg.get("jsonrpc") == "2.0" and msg.get("id") == 1, f"invalid JSON-RPC envelope: {msg}"
assert msg["result"]["protocolVersion"] == "2024-11-05", f"bad protocolVersion: {msg}"
PY
)"; then
        pass_test "gated_f33_mcp_clean_stdout"
    else
        fail_test "gated_f33_mcp_clean_stdout" "${py_err}"
    fi
}

# 3. F33: flint.sh new v5 EngineApp scaffold + compile & run in $TMPD
gated_f33_new_v5_engine_app_scaffold() {
    local game_dir="${TMPD}/v5_game"
    if ! bash "${ROOT}/tools/flint.sh" new "$game_dir" >"${TMPD}/f33_new.out" 2>"${TMPD}/f33_new.err"; then
        fail_test "gated_f33_new_v5_engine_app_scaffold" "flint.sh new failed"
        return
    fi
    if [[ ! -f "${game_dir}/flint.toml" || ! -f "${game_dir}/src/main.mojo" || ! -x "${game_dir}/tools/run.sh" ]]; then
        fail_test "gated_f33_new_v5_engine_app_scaffold" "missing scaffolded files"
        return
    fi
    if ! grep -q "EngineApp" "${game_dir}/src/main.mojo"; then
        fail_test "gated_f33_new_v5_engine_app_scaffold" "src/main.mojo does not use EngineApp"
        return
    fi
    if grep -qE "flint\.core\.ecs|WaylandWindow" "${game_dir}/src/main.mojo"; then
        fail_test "gated_f33_new_v5_engine_app_scaffold" "src/main.mojo still contains legacy flint.core.ecs or WaylandWindow boilerplate"
        return
    fi
    local rc=0
    (cd "$game_dir" && FLINT_ROOT="$ROOT" bash "${game_dir}/tools/run.sh" --headless-smoke) >"${TMPD}/f33_new_run.out" 2>"${TMPD}/f33_new_run.err" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        pass_test "gated_f33_new_v5_engine_app_scaffold"
    else
        fail_test "gated_f33_new_v5_engine_app_scaffold" "scaffolded project run.sh --headless-smoke failed (rc=${rc})"
    fi
}

# 4. F29/F30: Rebased minimal_spark, open_world, studio, pack_cli + valid PNGs
gated_f29_f30_rebased_examples_and_studio() {
    local ex_files=(
        "${ROOT}/examples/minimal_spark/main.mojo"
        "${ROOT}/examples/open_world/main.mojo"
        "${ROOT}/examples/studio/main.mojo"
    )
    local f
    for f in "${ex_files[@]}"; do
        if [[ ! -f "$f" ]]; then
            fail_test "gated_f29_f30_rebased_examples_and_studio" "missing example entry ${f}"
            return
        fi
        local code_file="${TMPD}/ex_code.mojo"
        grep -vE '^[[:space:]]*#' "$f" > "$code_file" || true
        if grep -qE "flint\.core\.ecs|NativeAudio|StudioEntityNode" "$code_file"; then
            fail_test "gated_f29_f30_rebased_examples_and_studio" "${f} still references legacy flint.core.ecs, NativeAudio, or StudioEntityNode"
            return
        fi
        local app_pat="\bEngineApp\b"
        if [[ "$f" == */examples/studio/main.mojo ]]; then
            app_pat="\b(EngineApp|FlintStudioState)\b"
        fi
        if ! grep -qE "$app_pat" "$code_file"; then
            fail_test "gated_f29_f30_rebased_examples_and_studio" "${f} does not use EngineApp"
            return
        fi
        if ! grep -qE "flint\.ecs|flint\.scene" "$code_file" || \
           ! grep -qE "PhysicsWorld|ScalableRenderer|AudioServer|InputMap|RenderPlugin|PhysicsPlugin|AudioPlugin|InputPlugin" "$code_file"; then
            fail_test "gated_f29_f30_rebased_examples_and_studio" "${f} missing v5 ECS/Scene/Subsystem plugin adoption"
            return
        fi
    done

    local pack_code_file="${TMPD}/pack_cli_code.mojo"
    grep -vE '^[[:space:]]*#' "${ROOT}/examples/pack_cli/main.mojo" > "$pack_code_file" || true
    if grep -qE "flint\.assets\.vfs" "$pack_code_file"; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "pack_cli still imports flint.assets.vfs"
        return
    fi

    # Verify Studio rebase (zero StudioEntityNode/flint.core.ecs, uses flint.ecs/scene/PhysicsWorld)
    # plus genuine multi-select, hierarchy reparenting, and non-blocking async picking
    local studio_code_file="${TMPD}/studio_code.mojo"
    grep -h -vE '^[[:space:]]*#' "${ROOT}/src/flint/studio"/*.mojo > "$studio_code_file" || true
    if grep -qE "StudioEntityNode|flint\.core\.ecs" "$studio_code_file"; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "src/flint/studio still references StudioEntityNode or flint.core.ecs"
        return
    fi
    if ! grep -qE "flint\.ecs" "$studio_code_file" || \
       ! grep -qE "flint\.scene|PhysicsWorld" "$studio_code_file"; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "src/flint/studio missing flint.ecs / flint.scene / PhysicsWorld rebase"
        return
    fi
    if ! grep -qE "selected_entities|multi_select|toggle_select" "$studio_code_file"; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "src/flint/studio missing multi-select support"
        return
    fi
    if ! grep -qE "reparent_entity|set_entity_parent|reparent\(" "$studio_code_file"; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "src/flint/studio missing hierarchy reparenting API/UI"
        return
    fi
    if ! grep -qE "pending_pick|poll_completed_pick|poll_pick|request_pick" "$studio_code_file"; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "src/flint/studio missing async picking API"
        return
    fi
    local studio_main_code_file="${TMPD}/studio_main_code.mojo"
    grep -vE '^[[:space:]]*#' "${ROOT}/examples/studio/main.mojo" > "$studio_main_code_file" || true
    if ! grep -qE "pending_pick|poll_completed_pick|poll_pick|request_pick" "$studio_main_code_file" || \
       grep -qE "pick_viewport\(|vk\.wait\(" "$studio_main_code_file"; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "examples/studio/main.mojo still uses blocking pick_viewport/vk.wait instead of async picking"
        return
    fi
    local studio_app_code_file="${TMPD}/studio_app_code.mojo"
    grep -vE '^[[:space:]]*#' "${ROOT}/src/flint/studio/studio_app.mojo" > "$studio_app_code_file" || true
    if grep -qE "pick_viewport\(|vk\.wait\(" "$studio_app_code_file"; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "studio_app.mojo calls blocking pick_viewport or vk.wait"
        return
    fi

    if ! bash "${ROOT}/tools/flint.sh" build spark world studio >"${TMPD}/f29_build.out" 2>"${TMPD}/f29_build.err"; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "flint.sh build spark world studio failed"
        return
    fi

    mkdir -p "${TMPD}/spark/build/shots" "${TMPD}/world/build/shots" "${TMPD}/studio_run"
    if ! (cd "${TMPD}/spark" && "${ROOT}/build/flint-spark" --shots) >"${TMPD}/f29_spark.out" 2>&1; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "flint-spark --shots failed"
        return
    fi
    if ! (cd "${TMPD}/world" && "${ROOT}/build/flint-world" --shots) >"${TMPD}/f29_world.out" 2>&1; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "flint-world --shots failed"
        return
    fi
    if ! (cd "${TMPD}/studio_run" && "${ROOT}/build/flint-studio" --headless-smoke --out "${TMPD}/studio_rebased.png") >"${TMPD}/f30_studio.out" 2>&1; then
        fail_test "gated_f29_f30_rebased_examples_and_studio" "flint-studio --headless-smoke failed"
        return
    fi

    local pngs=(
        "${TMPD}/spark/build/shots/flint_spark.png"
        "${TMPD}/world/build/shots/open_world_noon_vista.png"
        "${TMPD}/world/build/shots/open_world_golden_citadel.png"
        "${TMPD}/world/build/shots/open_world_night_ltc_bokeh.png"
        "${TMPD}/studio_rebased.png"
    )
    local p py_err=""
    for p in "${pngs[@]}"; do
        if [[ ! -f "$p" ]]; then
            fail_test "gated_f29_f30_rebased_examples_and_studio" "missing output PNG ${p}"
            return
        fi
        if ! py_err="$(validate_png_dims "$p" 1280 720)"; then
            fail_test "gated_f29_f30_rebased_examples_and_studio" "${py_err}"
            return
        fi
    done
    pass_test "gated_f29_f30_rebased_examples_and_studio"
}

# 5. F32: All 5 labs/ use EngineBenchSuite and emit fresh CSV outputs
gated_f32_five_labs_engine_bench_csv() {
    local labs=(physics anim render audio_ui net)
    local lab
    for lab in "${labs[@]}"; do
        local src="${ROOT}/labs/lab_${lab}.mojo"
        if [[ ! -f "$src" ]]; then
            fail_test "gated_f32_five_labs_engine_bench_csv" "missing lab source ${src}"
            return
        fi
        local lab_code_file="${TMPD}/lab_${lab}_code.mojo"
        grep -vE '^[[:space:]]*#' "$src" > "$lab_code_file" || true
        if ! grep -q "EngineBenchSuite" "$lab_code_file"; then
            fail_test "gated_f32_five_labs_engine_bench_csv" "${src} does not use EngineBenchSuite in code"
            return
        fi
        if ! grep -qE "finish_and_export|append_perf_history_csv|to_perf_table|\.csv" "$lab_code_file"; then
            fail_test "gated_f32_five_labs_engine_bench_csv" "${src} does not emit CSV output in code"
            return
        fi
        local marker="${TMPD}/marker_${lab}"
        touch "$marker"
        sleep 0.05
        local rc=0
        bash "${ROOT}/tools/flint.sh" lab "$lab" >"${TMPD}/lab_${lab}.out" 2>"${TMPD}/lab_${lab}.err" || rc=$?
        if [[ "$rc" -ne 0 ]]; then
            fail_test "gated_f32_five_labs_engine_bench_csv" "lab ${lab} exited ${rc}"
            return
        fi
        if ! grep -q "FLINT LAB SUITE:" "${TMPD}/lab_${lab}.out" || ! grep -q "RESULT: PASS" "${TMPD}/lab_${lab}.out"; then
            fail_test "gated_f32_five_labs_engine_bench_csv" "lab ${lab} missing EngineBenchSuite PASS summary"
            return
        fi
        local csv_hit=""
        csv_hit="$(find "${ROOT}/build" "${ROOT}/docs" /tmp -maxdepth 3 \( -name '*lab*.csv' -o -name 'PERF_HISTORY.csv' \) -newer "$marker" 2>/dev/null | sed -n '1p' || true)"
        if [[ -z "$csv_hit" ]]; then
            local out_csv=""
            out_csv="$(grep -oE '(/|build/|docs/|/tmp/)[^[:space:]]+\.csv' "${TMPD}/lab_${lab}.out" | sed -n '1p' || true)"
            if [[ -n "$out_csv" ]]; then
                local resolved_csv="$out_csv"
                [[ "$resolved_csv" != /* ]] && resolved_csv="${ROOT}/${resolved_csv}"
                if [[ -s "$resolved_csv" && ! "$marker" -nt "$resolved_csv" ]]; then
                    csv_hit="$resolved_csv"
                fi
            fi
        fi
        if [[ -z "$csv_hit" ]]; then
            fail_test "gated_f32_five_labs_engine_bench_csv" "lab ${lab} did not emit/update a fresh .csv artifact"
            return
        fi
    done
    pass_test "gated_f32_five_labs_engine_bench_csv"
}

# 6. F32: src/flint/core/perf_history.mojo uses Flint metrics (zero Deepsong/Sharddepth legacy)
gated_f32_perf_history_flint_metrics() {
    local ph="${ROOT}/src/flint/core/perf_history.mojo"
    if [[ ! -f "$ph" ]]; then
        fail_test "gated_f32_perf_history_flint_metrics" "missing ${ph}"
        return
    fi
    if grep -qE "voxel_initial_mesh_ms|voxel_dirty_remesh_ms|voxel_carve_ms|voxel_ao_mesh|dig_remesh_p99_ms|hum_restored_frame|systemic_r2|smooth_exposed_chunk|carry_toss|SHARDDEPTH|sharddepth|SD_GATE_ONLY|regression_gate\.sh|RX 9070|tools/run\.sh benchmark" "$ph"; then
        fail_test "gated_f32_perf_history_flint_metrics" "perf_history.mojo still contains legacy Deepsong/Sharddepth metrics"
        return
    fi
    if grep -q "perf_metrics" "$ph" && grep -q "PerfMetric" "$ph" && grep -qE "physics|render|anim|audio|net" "$ph"; then
        pass_test "gated_f32_perf_history_flint_metrics"
    else
        fail_test "gated_f32_perf_history_flint_metrics" "perf_history.mojo missing Flint perf_metrics/PerfMetric definitions"
    fi
}

# 7. F31 & F32: tools/flint.sh gate passes calibrated FLIP/p99 gates and honest tests/e2e/
gated_f31_f32_gate_flip_p99_pass() {
    local gate_body_file="${TMPD}/gate_body.sh"
    sed -n '/^cmd_gate()/,/^}/p' "${ROOT}/tools/flint.sh" | grep -vE '^[[:space:]]*#' > "$gate_body_file" || true
    if [[ ! -s "$gate_body_file" ]]; then
        fail_test "gated_f31_f32_gate_flip_p99_pass" "cmd_gate() not found in tools/flint.sh"
        return
    fi
    if ! grep -qiE "flip" "$gate_body_file" || ! grep -qE "lab|p99" "$gate_body_file"; then
        fail_test "gated_f31_f32_gate_flip_p99_pass" "cmd_gate() missing FLIP or lab/p99 checks"
        return
    fi

    # Verify F31: tests/e2e/ uses real multi-frame subsystem integration rather than synthetic stubs
    if [[ ! -d "${ROOT}/tests/e2e" ]]; then
        fail_test "gated_f31_f32_gate_flip_p99_pass" "missing ${ROOT}/tests/e2e directory"
        return
    fi
    local e2e_code_file="${TMPD}/e2e_code.mojo"
    find "${ROOT}/tests/e2e" -name '*.mojo' -exec grep -h -vE '^[[:space:]]*#' {} + > "$e2e_code_file" || true
    local sym
    for sym in EngineApp World PhysicsWorld ScalableRenderer AudioServer NetPlugin NetHub LagCompHistory; do
        if ! grep -qE "\b${sym}\b" "$e2e_code_file"; then
            fail_test "gated_f31_f32_gate_flip_p99_pass" "tests/e2e/ missing real integration usage of ${sym} (F31)"
            return
        fi
    done
    if [[ ! -f "${ROOT}/tests/test_flint_v5_net_rollback.mojo" ]] || \
       ! grep -qE "\bRollbackSession\b" "${ROOT}/tests/test_flint_v5_net_rollback.mojo"; then
        fail_test "gated_f31_f32_gate_flip_p99_pass" "tests/test_flint_v5_net_rollback.mojo missing RollbackSession coverage"
        return
    fi

    local rc=0
    bash "${ROOT}/tools/flint.sh" gate >"${TMPD}/f32_gate.out" 2>"${TMPD}/f32_gate.err" || rc=$?
    if [[ "$rc" -eq 0 ]] && grep -q "\[PASS\] FLINT GATE PASSED" "${TMPD}/f32_gate.out"; then
        pass_test "gated_f31_f32_gate_flip_p99_pass"
    else
        fail_test "gated_f31_f32_gate_flip_p99_pass" "flint.sh gate failed (rc=${rc})"
    fi
}

# 8. F34: Legacy render stack, dead-code modules, and 4 orphaned tests deleted; sync script clean
gated_f34_dead_code_deleted_and_sync_clean() {
    local deleted_paths=(
        # Legacy render stack
        "src/flint/render/scene_renderer.mojo"
        "src/flint/render/rasterizer.mojo"
        "src/flint/render/gpu_raster.mojo"
        "src/flint/render/vulkan_shaders.mojo"
        "src/flint/render/frame_passes.mojo"
        "src/flint/render/gpu_skinning.mojo"
        "src/flint/render/livelink.mojo"
        # Dead-code modules
        "src/flint/core/entity.mojo"
        "src/flint/core/check.mojo"
        "src/flint/core/crc32.mojo"
        "src/flint/core/test_history.mojo"
        "src/flint/physics/handling.mojo"
        "src/flint/physics/mass.mojo"
        "src/flint/physics/ropes.mojo"
        "src/flint/physics/wind_mirror.mojo"
        "src/flint/anim/cloth_proxy.mojo"
        "src/flint/anim/skin_limb.mojo"
        "src/flint/assets/vfs.mojo"
        "src/flint/assets/cook.mojo"
        "src/flint/assets/cooked_model.mojo"
        "src/flint/assets/cooked.mojo"
        "src/flint/assets/clip_names.mojo"
        "src/flint/platform/narration.mojo"
        "src/flint/text/lettering.mojo"
        "src/flint/ui/widgets.mojo"
        "src/flint/devtools/mcp_server.mojo"
        # 4 orphaned tests
        "tests/test_aaa_engine_v2.mojo"
        "tests/test_flint_v4_anim.mojo"
        "tests/test_flint_v4_physics.mojo"
        "tests/test_procedural_qem.mojo"
    )
    local rel
    for rel in "${deleted_paths[@]}"; do
        if [[ -e "${ROOT}/${rel}" ]]; then
            fail_test "gated_f34_dead_code_deleted_and_sync_clean" "legacy file still exists: ${rel}"
            return
        fi
    done

    local sync_sh="${ROOT}/tools/sync_from_deepsong.sh"
    local bad_refs=(
        "scene_renderer\.mojo" "rasterizer\.mojo" "gpu_raster\.mojo" "vulkan_shaders\.mojo"
        "frame_passes\.mojo" "gpu_skinning\.mojo" "livelink\.mojo"
        "entity\.mojo" "check\.mojo" "crc32\.mojo" "test_history\.mojo"
        "handling\.mojo" "mass\.mojo" "ropes\.mojo" "wind_mirror\.mojo"
        "cloth_proxy\.mojo" "skin_limb\.mojo"
        "vfs\.mojo" "cook\.mojo" "cooked_model\.mojo" "cooked\.mojo" "clip_names\.mojo"
        "narration\.mojo" "lettering\.mojo" "widgets\.mojo" "mcp_server\.mojo"
    )
    local pat
    for pat in "${bad_refs[@]}"; do
        if grep -qE "$pat" "$sync_sh"; then
            fail_test "gated_f34_dead_code_deleted_and_sync_clean" "sync_from_deepsong.sh still references ${pat}"
            return
        fi
    done
    pass_test "gated_f34_dead_code_deleted_and_sync_clean"
}

gated_f33_flint_package_bundle
gated_f33_mcp_clean_stdout
gated_f33_new_v5_engine_app_scaffold
gated_f29_f30_rebased_examples_and_studio
gated_f32_five_labs_engine_bench_csv
gated_f32_perf_history_flint_metrics
gated_f31_f32_gate_flip_p99_pass
gated_f34_dead_code_deleted_and_sync_clean

echo "[SUMMARY] ${PASS}/${TOTAL} passed"
if [[ "$FAIL" -ne 0 ]]; then
    exit 1
fi
exit 0
