#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPAQUE_DIR="${ROOT}/tests/e2e_opaque"
GATED_DIR="${OPAQUE_DIR}/gated"
BUILD_DIR="${ROOT}/build"
MOJO_BIN="${MOJO:-${ROOT}/.pixi/envs/default/bin/mojo}"

STRICT=0
FILTER=""
TIMEOUT_SECS="${FLINT_E2E_TIMEOUT:-120}"

usage() {
    cat <<'EOF'
Usage: run_e2e_opaque.sh [--strict] [--filter <substr>]
  --strict           Require all gated M1-M6 tests to compile with --Werror and pass.
  --filter <substr>  Only execute tests whose relative path contains <substr>.
  Env: FLINT_E2E_TIMEOUT (seconds per test binary/fast script, default: 120; test_t4_cli_workflows.sh min 600s)
       FLINT_E2E_CLI_TIMEOUT (seconds for test_gated_m5_m6_cli.sh, default: 1200)
EOF
}

script_timeout_for() {
    local src="$1"
    local base
    base="$(basename "$src")"
    local t="$TIMEOUT_SECS"
    case "$base" in
        test_gated_m5_m6_cli.sh)
            local cli_t="${FLINT_E2E_CLI_TIMEOUT:-1200}"
            if [[ "$t" -lt "$cli_t" ]]; then
                t="$cli_t"
            fi
            ;;
        test_t4_cli_workflows.sh)
            if [[ "$t" -lt 600 ]]; then
                t=600
            fi
            ;;
    esac
    printf '%s\n' "$t"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --strict)
            STRICT=1
            shift
            ;;
        --filter)
            if [[ $# -lt 2 ]]; then
                echo "Error: --filter requires an argument" >&2
                usage >&2
                exit 2
            fi
            FILTER="$2"
            shift 2
            ;;
        --filter=*)
            FILTER="${1#--filter=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$MOJO_BIN" ]]; then
    echo "Error: Mojo compiler not found or not executable at: $MOJO_BIN" >&2
    exit 1
fi

if [[ -d "${ROOT}/.pixi/envs/default" ]]; then
    export MOJO_PREFIX="${ROOT}/.pixi/envs/default"
    export MODULAR_HOME="${MOJO_PREFIX}/share/max"
    export LD_LIBRARY_PATH="${MOJO_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

mkdir -p "$BUILD_DIR"

cleanup_temps() {
    rm -f "${BUILD_DIR}/.tmp_e2e_opaque_"*"_$$" "${BUILD_DIR}/.e2e_opaque_gated_log_$$_"* 2>/dev/null || true
}
trap cleanup_temps EXIT

matches_filter() {
    local path="$1"
    if [[ -z "$FILTER" ]]; then
        return 0
    fi
    [[ "$path" == *"$FILTER"* ]]
}

live_total=0
live_pass=0
live_fail=0
gated_total=0
gated_pass=0
gated_pending=0
gated_fail=0

shopt -s nullglob

# Discover live tests (sorted deterministically)
mapfile -t live_files < <(
    for f in "${OPAQUE_DIR}"/test_*.mojo "${OPAQUE_DIR}"/test_*.sh; do
        printf '%s\n' "$f"
    done | LC_ALL=C sort
)

for src in "${live_files[@]}"; do
    rel_path="${src#${ROOT}/}"
    if ! matches_filter "$rel_path"; then
        continue
    fi
    live_total=$((live_total + 1))

    if [[ "$src" == *.mojo ]]; then
        stem="$(basename "$src" .mojo)"
        tmp_bin="${BUILD_DIR}/.tmp_e2e_opaque_${stem}_$$"
        out_bin="${BUILD_DIR}/e2e_opaque_${stem}"
        echo "[RUN: LIVE] ${rel_path}"
        if "$MOJO_BIN" build -I "${ROOT}/src" -I "${ROOT}/tests" -O3 --fp-mode contract=off --Werror "$src" -o "$tmp_bin" \
            && mv -f "$tmp_bin" "$out_bin" \
            && timeout -k 5 "$TIMEOUT_SECS" "$out_bin"; then
            echo "[PASS: LIVE] ${rel_path}"
            live_pass=$((live_pass + 1))
        else
            rc=$?
            rm -f "$tmp_bin" || true
            if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
                echo "[TIMEOUT: LIVE] ${rel_path} exceeded ${TIMEOUT_SECS}s" >&2
            fi
            echo "[FAIL: LIVE] ${rel_path}" >&2
            live_fail=$((live_fail + 1))
        fi
    else
        script_timeout="$(script_timeout_for "$src")"
        echo "[RUN: LIVE] ${rel_path}"
        if (cd "$ROOT" && timeout -k 5 "$script_timeout" bash "$src"); then
            echo "[PASS: LIVE] ${rel_path}"
            live_pass=$((live_pass + 1))
        else
            rc=$?
            if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
                echo "[TIMEOUT: LIVE] ${rel_path} exceeded ${script_timeout}s" >&2
            fi
            echo "[FAIL: LIVE] ${rel_path}" >&2
            live_fail=$((live_fail + 1))
        fi
    fi
done

# Discover gated M1-M6 tests (sorted deterministically)
if [[ -d "$GATED_DIR" ]]; then
    mapfile -t gated_files < <(
        for f in "${GATED_DIR}"/test_*.mojo "${GATED_DIR}"/test_*.sh; do
            printf '%s\n' "$f"
        done | LC_ALL=C sort
    )
else
    gated_files=()
fi

for src in "${gated_files[@]}"; do
    rel_path="${src#${ROOT}/}"
    if ! matches_filter "$rel_path"; then
        continue
    fi
    gated_total=$((gated_total + 1))
    log_file="$(mktemp "${BUILD_DIR}/.e2e_opaque_gated_log_$$_XXXXXX")"

    if [[ "$src" == *.mojo ]]; then
        stem="$(basename "$src" .mojo)"
        tmp_bin="${BUILD_DIR}/.tmp_e2e_opaque_gated_${stem}_$$"
        out_bin="${BUILD_DIR}/e2e_opaque_gated_${stem}"
        if "$MOJO_BIN" build -I "${ROOT}/src" -I "${ROOT}/tests" -O3 --fp-mode contract=off --Werror "$src" -o "$tmp_bin" >"$log_file" 2>&1 \
            && mv -f "$tmp_bin" "$out_bin" >>"$log_file" 2>&1 \
            && timeout -k 5 "$TIMEOUT_SECS" "$out_bin" >>"$log_file" 2>&1; then
            cat "$log_file"
            echo "[GATED: PASS] ${rel_path}"
            gated_pass=$((gated_pass + 1))
        else
            rc=$?
            rm -f "$tmp_bin" || true
            if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
                echo "[TIMEOUT: GATED] ${rel_path} exceeded ${TIMEOUT_SECS}s" >>"$log_file"
            fi
            if [[ "$STRICT" -eq 1 ]]; then
                cat "$log_file" >&2
                echo "[GATED: FAIL] ${rel_path}" >&2
                gated_fail=$((gated_fail + 1))
            else
                echo "[GATED: PENDING] ${rel_path}"
                gated_pending=$((gated_pending + 1))
            fi
        fi
    else
        script_timeout="$(script_timeout_for "$src")"
        if (cd "$ROOT" && timeout -k 5 "$script_timeout" bash "$src") >"$log_file" 2>&1; then
            cat "$log_file"
            echo "[GATED: PASS] ${rel_path}"
            gated_pass=$((gated_pass + 1))
        else
            rc=$?
            if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
                echo "[TIMEOUT: GATED] ${rel_path} exceeded ${script_timeout}s" >>"$log_file"
            fi
            if [[ "$STRICT" -eq 1 ]]; then
                cat "$log_file" >&2
                echo "[GATED: FAIL] ${rel_path}" >&2
                gated_fail=$((gated_fail + 1))
            else
                echo "[GATED: PENDING] ${rel_path}"
                gated_pending=$((gated_pending + 1))
            fi
        fi
    fi

    rm -f "$log_file" || true
done

echo "[SUMMARY] live=${live_pass}/${live_total} passed (${live_fail} failed), gated=${gated_pass}/${gated_total} passed (${gated_pending} pending, ${gated_fail} failed), strict=${STRICT}"

if [[ "$live_fail" -gt 0 ]]; then
    exit 1
fi
if [[ "$STRICT" -eq 1 && "$gated_fail" -gt 0 ]]; then
    exit 1
fi
exit 0
