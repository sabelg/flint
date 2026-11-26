#!/usr/bin/env bash
# Master E2E Test Runner for Flint Studio & Game Shipping (/home/user/flint/tests/e2e_studio/)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export ROOT
export FLINT_ROOT="$ROOT"
export PYTHONDONTWRITEBYTECODE=1

# shellcheck source=common/e2e_helpers.sh
source "${SCRIPT_DIR}/common/e2e_helpers.sh"

TIER="all"
AREA="all"
FILTER=""
LIST_ONLY=0
KEEP_TMP=0

usage() {
    cat <<'EOF'
Usage: tests/e2e_studio/run_studio_e2e.sh [options]
  --tier <1|2|3|4|all>      Select test tier (default: all)
  --area <r1|m0|m1|m2|m3|all>
                            Filter by requirement area (default: all)
  --filter <substr>         Substring match on test relative path
  --list                    List matching tests and exit 0
  --keep-tmp                Preserve /tmp/flint_e2e_studio_<pid> temp dir
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            [[ $# -ge 2 ]] || { echo "Missing value for --tier" >&2; exit 2; }
            TIER="$2"
            shift 2
            ;;
        --area)
            [[ $# -ge 2 ]] || { echo "Missing value for --area" >&2; exit 2; }
            AREA="$2"
            shift 2
            ;;
        --filter)
            [[ $# -ge 2 ]] || { echo "Missing value for --filter" >&2; exit 2; }
            FILTER="$2"
            shift 2
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        --keep-tmp)
            KEEP_TMP=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$TIER" in
    1|2|3|4|all) ;;
    *) echo "Invalid --tier: $TIER (expected 1|2|3|4|all)" >&2; exit 2 ;;
esac

case "$AREA" in
    r1|m0|m1|m2|m3|all) ;;
    *) echo "Invalid --area: $AREA (expected r1|m0|m1|m2|m3|all)" >&2; exit 2 ;;
esac

matches_area() {
    local file_path="$1"
    if [[ "$AREA" == "all" ]]; then
        return 0
    fi
    local bname
    bname="$(basename "$file_path")"
    if [[ "$bname" == *"_${AREA}_"* ]]; then
        return 0
    fi
    if head -n 20 "$file_path" | grep -qiE "(#|\"\"\")[[:space:]]*AREA:.*\b${AREA}\b"; then
        return 0
    fi
    return 1
}

declare -a TIERS=()
if [[ "$TIER" == "all" ]]; then
    TIERS=(1 2 3 4)
else
    TIERS=("$TIER")
fi

declare -a TESTS=()
shopt -s nullglob
for t in "${TIERS[@]}"; do
    tdir="${SCRIPT_DIR}/tier${t}"
    [[ -d "$tdir" ]] || continue
    mapfile -t found < <(
        for f in "${tdir}"/t"${t}"_*.sh "${tdir}"/t"${t}"_*.mojo; do
            [[ -e "$f" ]] || continue
            echo "$f"
        done | LC_ALL=C sort
    )
    for f in "${found[@]}"; do
        if [[ "$f" == *.sh && ! -x "$f" ]]; then
            echo "[WARN] Skipping non-executable test script: ${f#"${SCRIPT_DIR}/"}" >&2
            continue
        fi
        rel="${f#"${SCRIPT_DIR}/"}"
        if [[ -n "$FILTER" && "$rel" != *"$FILTER"* ]]; then
            continue
        fi
        if ! matches_area "$f"; then
            continue
        fi
        TESTS+=("$f")
    done
done
shopt -u nullglob

if [[ "$LIST_ONLY" -eq 1 ]]; then
    for f in "${TESTS[@]}"; do
        echo "${f#"${SCRIPT_DIR}/"}"
    done
    exit 0
fi

if [[ "${#TESTS[@]}" -eq 0 ]]; then
    echo "[ERROR] No tests matched (--tier ${TIER}, --area ${AREA}, --filter '${FILTER}')." >&2
    exit 1
fi

resolve_mojo_env || exit 1

RUN_TMP="/tmp/flint_e2e_studio_$$"
mkdir -p "${RUN_TMP}/bin" "${RUN_TMP}/logs" || exit 1
export FLINT_STUDIO_E2E_TMP="$RUN_TMP"
export FLINT_STUDIO_E2E_KEEP_TMP="$KEEP_TMP"

_runner_cleanup() {
    local rc=$?
    if [[ "$KEEP_TMP" -ne 1 && -n "${RUN_TMP:-}" && "$RUN_TMP" == /tmp/flint_e2e_studio_* ]]; then
        rm -rf "$RUN_TMP"
    fi
    return "$rc"
}
trap _runner_cleanup EXIT

passed=0
failed=0
total=0

default_timeout_for_tier() {
    case "$1" in
        1|2) echo 180 ;;
        3) echo 300 ;;
        4) echo 600 ;;
        *) echo 180 ;;
    esac
}

for test_file in "${TESTS[@]}"; do
    rel_path="${test_file#"${SCRIPT_DIR}/"}"
    tier_dir="${rel_path%%/*}"
    tier_num="${tier_dir#tier}"
    t_sec="${FLINT_STUDIO_E2E_TIMEOUT:-$(default_timeout_for_tier "$tier_num")}"
    stem="$(basename "$test_file")"
    stem="${stem%.*}"

    echo "=== Running ${rel_path} (timeout=${t_sec}s) ==="
    total=$((total + 1))

    if [[ "$test_file" == *.mojo ]]; then
        tmp_bin="${RUN_TMP}/bin/${tier_dir}_${stem}.tmp.$$"
        out_bin="${RUN_TMP}/bin/${tier_dir}_${stem}"
        compile_log="${RUN_TMP}/logs/${tier_dir}_${stem}.compile.log"
        if ! "$MOJO_BIN" build -I "${ROOT}/src" -I "${ROOT}/tests" -O3 --fp-mode contract=off --Werror "$test_file" -o "$tmp_bin" >"$compile_log" 2>&1; then
            rm -f "$tmp_bin"
            tail -n 20 "$compile_log" >&2 || true
            echo "[FAIL] ${rel_path} (compile/contract pending)"
            failed=$((failed + 1))
            continue
        fi
        mv -f "$tmp_bin" "$out_bin" || exit 1
        timeout -k 5 "$t_sec" "$out_bin"
        rc=$?
    else
        timeout -k 5 "$t_sec" "$test_file"
        rc=$?
    fi

    if [[ "$rc" -eq 0 ]]; then
        echo "[PASS] ${rel_path}"
        passed=$((passed + 1))
    elif [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
        echo "[TIMEOUT] ${rel_path} exceeded ${t_sec}s (rc=${rc})"
        failed=$((failed + 1))
    else
        echo "[FAIL] ${rel_path} (rc=${rc})"
        failed=$((failed + 1))
    fi
done

echo "[SUMMARY] ${passed}/${total} passed (${failed} failed)"
if [[ "$total" -gt 0 && "$failed" -eq 0 ]]; then
    exit 0
fi
exit 1
