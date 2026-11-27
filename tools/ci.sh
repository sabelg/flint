#!/usr/bin/env bash
# CPU-only CI subset of `tools/flint.sh gate`.
#
# The full gate needs a Vulkan GPU, a display and ~50 minutes of serial work. This script runs the part
# that is meaningful on a plain Linux box (e.g. a GitHub-hosted ubuntu runner):
#
#   1. Zero-Game-Leak gate        (`tools/sync_from_deepsong.sh --verify`)
#   2. --Werror -O3 builds          every core test binary, through `tools/flint.sh build`
#                                   (same flags, cache and output paths as the gate)
#   3. Suites, in parallel, each in its own process with a log under build/ci-logs/:
#        - CPU suites + every tests/test_flint_v5_*.mojo, with Vulkan and the display hidden
#        - Vulkan suites (test_flint, test_e2e) on Mesa lavapipe when it is installed, else build-only
#        - Studio suites: the 8 editor-model suites (STUDIO_TEST_SUITES in tools/flint.sh, built
#          at -O1 like the gate) and the profile-compare CLI test, headless
#
# Usage: bash tools/ci.sh [--jobs N] [--build-only] [--list] [--studio | --no-studio]
#        --studio     only the Studio group (build + run); --no-studio skips it
# Env:   FLINT_CI_JOBS        parallel compile jobs (default: min(8, nproc/2), at least 1)
#        FLINT_CI_HIDE_GPU=0  run the CPU suites with the host's Vulkan and display visible
#        FLINT_CI_VULKAN=0    never run the Vulkan suites (build them only)
#        FLINT_CI_SUITES      space-separated subset of suite names to run (see --list)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Suites run with Vulkan, MAX GPU and the display hidden (they pass on a GPU-less box and exercise
# the CPU fallbacks). Every tests/test_flint_v5_*.mojo is added automatically below, minus V5_EXCLUDE.
#   name                  command (relative to the repo root)
CPU_SUITES=(
    "embedded_ide          build/flint-embedded-ide-test"
    "hygiene_cooker        build/flint-hygiene-cooker-test"
    "cycles                build/flint-cycles-test"
    "studio_ide_launcher   tests/test_studio_ide.sh"
)
V5_EXCLUDE=()   # v5 suite basenames that must not run here (none today)

# Studio group: STUDIO_TEST_SUITES from tools/flint.sh (built at -O1 by `flint.sh test studio
# --build-only`) plus the profile-compare CLI test. Run headless like the CPU suites.
STUDIO_SUITES=("studio_profile_compare  tests/test_studio_profile_compare.sh")
mapfile -t _studio_names < <(sed -nE 's/^STUDIO_TEST_SUITES=\((.*)\)$/\1/p' tools/flint.sh | tr ' ' '\n' | sed '/^$/d')
for _n in "${_studio_names[@]}"; do
    STUDIO_SUITES+=("${_n#test_}  build/studio-tests/${_n}")
done

# Suites that need a working Vulkan device: they assert `FlintAaaVulkanRenderer.active`
# (tests/test_flint.mojo, suites V12-V15) or crash without one (tests/test_e2e.mojo: index out of
# bounds in render/aaa_renderer.mojo when the renderer is inactive). They pass on Mesa's lavapipe
# software ICD, so they run when lvp_icd.json is installed (apt: mesa-vulkan-drivers) and are
# build-only otherwise. FLINT_CI_VULKAN=0 forces build-only.
VULKAN_SUITES=(
    "unit_vulkan           build/flint-test"
    "e2e                   build/flint-e2e"
)
LVP_ICD_CANDIDATES=(/usr/share/vulkan/icd.d/lvp_icd.json /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
                    /usr/share/vulkan/icd.d/lvp_icd.aarch64.json /etc/vulkan/icd.d/lvp_icd.json)

jobs="${FLINT_CI_JOBS:-}"
build_only=0
studio=1    # 1: everything, 0: --no-studio, 2: --studio (Studio group only)
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs) jobs="${2:-}"; shift 2 ;;
        --build-only) build_only=1; shift ;;
        --studio) studio=2; shift ;;
        --no-studio) studio=0; shift ;;
        --list)
            printf 'CPU suites (+ every tests/test_flint_v5_*.mojo):\n'; printf '  %s\n' "${CPU_SUITES[@]}"
            printf 'Vulkan suites (lavapipe if installed, else build only):\n'; printf '  %s\n' "${VULKAN_SUITES[@]}"
            printf 'Studio suites (headless, -O1):\n'; printf '  %s\n' "${STUDIO_SUITES[@]}"
            exit 0 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "[CI] Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$jobs" ]]; then
    ncpu="$(nproc 2>/dev/null || echo 2)"
    jobs=$(( ncpu / 2 ))
    (( jobs > 8 )) && jobs=8
    (( jobs < 1 )) && jobs=1
fi

LOG_DIR="${ROOT}/build/ci-logs"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log
t_start=$SECONDS
fail=0

stamp() { printf '[CI %4ds] %s\n' "$(( SECONDS - t_start ))" "$*"; }

# ---------------------------------------------------------------------------
# 1. Zero-Game-Leak gate (cheap, no compiler)
# ---------------------------------------------------------------------------
stamp "Zero-Game-Leak gate"
if ! bash tools/sync_from_deepsong.sh --verify > "$LOG_DIR/leak_gate.log" 2>&1; then
    cat "$LOG_DIR/leak_gate.log"
    stamp "FAIL leak gate"
    exit 1
fi
tail -n 1 "$LOG_DIR/leak_gate.log"

# ---------------------------------------------------------------------------
# 2. --Werror builds. `flint.sh build` writes the build stamp, skips up-to-date binaries and uses the
#    gate's flags (-I src -I tests -O3 --fp-mode contract=off --Werror). The 13 v5 suites dominate, so
#    they get the parallel slots; the small targets build alongside them in one serial chain.
# ---------------------------------------------------------------------------
build_fail=0
pid_v5=""
pid_core=""
pid_studio=""
if [[ "$studio" -ne 2 ]]; then
    stamp "Building test binaries (--Werror, ${jobs} parallel v5 jobs)"
    bash tools/flint.sh build v5 -j "$jobs" > "$LOG_DIR/build_v5.log" 2>&1 &
    pid_v5=$!
    (
        bash tools/flint.sh build unit -j 3 &&
        bash tools/flint.sh build e2e &&
        bash tools/flint.sh build cycles -j 2
    ) > "$LOG_DIR/build_core.log" 2>&1 &
    pid_core=$!
fi
if [[ "$studio" -ne 0 ]]; then
    # Each Studio suite imports the whole editor; -O1 keeps them to a few minutes each.
    stamp "Building Studio suites (--Werror, -O1)"
    FLINT_STUDIO_TEST_OPT=-O1 bash tools/flint.sh test studio --build-only > "$LOG_DIR/build_studio.log" 2>&1 &
    pid_studio=$!
fi
[[ -z "$pid_v5" ]] || wait "$pid_v5" || { build_fail=1; stamp "FAIL build v5 (see build/ci-logs/build_v5.log)"; }
[[ -z "$pid_core" ]] || wait "$pid_core" || { build_fail=1; stamp "FAIL build unit/e2e/cycles (see build/ci-logs/build_core.log)"; }
[[ -z "$pid_studio" ]] || wait "$pid_studio" || { build_fail=1; stamp "FAIL build studio (see build/ci-logs/build_studio.log)"; }
if [[ "$build_fail" -ne 0 ]]; then
    grep -hE 'error|warning' "$LOG_DIR"/build_*.log | head -n 60
    exit 1
fi
stamp "Build OK"
[[ "$build_only" -eq 1 ]] && { stamp "PASS (build only)"; exit 0; }

# ---------------------------------------------------------------------------
# 3. Run CPU-safe suites in parallel.
# ---------------------------------------------------------------------------
cpu_env=()
if [[ "${FLINT_CI_HIDE_GPU:-1}" != "0" ]]; then
    # No Vulkan ICD and no display: what a GPU-less runner sees. (The MAX GPU is left visible:
    # hiding it with CUDA_VISIBLE_DEVICES= makes MAX's DeviceContext segfault instead of raising.)
    cpu_env=(env -u WAYLAND_DISPLAY -u DISPLAY VK_ICD_FILENAMES=/nonexistent/none.json
             VK_DRIVER_FILES=/nonexistent/none.json FLINT_HEADLESS=1)
fi
lvp_icd=""
if [[ "${FLINT_CI_VULKAN:-1}" != "0" ]]; then
    for c in "${LVP_ICD_CANDIDATES[@]}"; do [[ -f "$c" ]] && { lvp_icd="$c"; break; }; done
fi
vk_env=(env -u WAYLAND_DISPLAY -u DISPLAY VK_ICD_FILENAMES="$lvp_icd" VK_DRIVER_FILES="$lvp_icd" FLINT_HEADLESS=1)

# The Studio suites' async gameplay builds spawn the Mojo compiler; hand them the pixi toolchain
# (a missing compiler must fail those tests, not pass as a "failed build").
pixi_prefix="${ROOT}/.pixi/envs/default"
if [[ -x "${pixi_prefix}/bin/mojo" ]]; then
    export FLINT_MOJO_BIN="${FLINT_MOJO_BIN:-${pixi_prefix}/bin/mojo}"
    export MOJO_PREFIX="${MOJO_PREFIX:-${pixi_prefix}}"
    export PATH="${pixi_prefix}/bin:${PATH}"
    export MODULAR_HOME="${MODULAR_HOME:-${pixi_prefix}/share/max}"
    export LD_LIBRARY_PATH="${pixi_prefix}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

all_suites=()
if [[ "$studio" -ne 2 ]]; then
    all_suites+=("${CPU_SUITES[@]}")
    for src in tests/test_flint_v5_*.mojo; do
        base="$(basename "$src" .mojo)"
        [[ " ${V5_EXCLUDE[*]-} " == *" ${base} "* ]] && continue
        all_suites+=("${base#test_flint_}  build/v5/${base}")
    done
fi
if [[ "$studio" -ne 0 ]]; then
    all_suites+=("${STUDIO_SUITES[@]}")
fi
if [[ "$studio" -eq 2 ]]; then
    :
elif [[ -n "$lvp_icd" ]]; then
    for e in "${VULKAN_SUITES[@]}"; do all_suites+=("$e"); done
    stamp "Vulkan suites use lavapipe: ${lvp_icd}"
else
    vk_names=()
    for e in "${VULKAN_SUITES[@]}"; do read -r n _ <<<"$e"; vk_names+=("$n"); done
    stamp "No lavapipe ICD (or FLINT_CI_VULKAN=0): build-only ${vk_names[*]}"
fi

selected=()
for entry in "${all_suites[@]}"; do
    read -r name bin <<<"$entry"
    if [[ -n "${FLINT_CI_SUITES:-}" && " ${FLINT_CI_SUITES} " != *" ${name} "* ]]; then
        continue
    fi
    selected+=("$name $bin")
done

stamp "Running ${#selected[@]} suites in parallel"
declare -A pids=()
for entry in "${selected[@]}"; do
    read -r name bin <<<"$entry"
    (
        state_dir="$(mktemp -d "${TMPDIR:-/tmp}/flint-ci-state.XXXXXX")"
        s0=$SECONDS
        runner=("${cpu_env[@]}")
        [[ " ${VULKAN_SUITES[*]} " == *" ${bin} "* ]] && runner=("${vk_env[@]}")
        [[ "$bin" == *.sh ]] && runner+=(bash)
        XDG_STATE_HOME="$state_dir" "${runner[@]}" "${ROOT}/${bin}" > "$LOG_DIR/${name}.log" 2>&1
        rc=$?
        rm -rf "$state_dir"
        echo "rc=${rc} seconds=$(( SECONDS - s0 ))" >> "$LOG_DIR/${name}.log"
        exit "$rc"
    ) &
    pids["$name"]=$!
done

passed=()
failed=()
for entry in "${selected[@]}"; do
    read -r name _ <<<"$entry"
    if wait "${pids[$name]}"; then
        passed+=("$name")
        stamp "PASS ${name} ($(tail -n 1 "$LOG_DIR/${name}.log" | sed 's/.*seconds=//')s)"
    else
        failed+=("$name")
        stamp "FAIL ${name} ($(tail -n 1 "$LOG_DIR/${name}.log"))"
        tail -n 25 "$LOG_DIR/${name}.log" | sed "s/^/    [${name}] /"
        fail=1
    fi
done

echo "------------------------------------------------------------"
stamp "${#passed[@]} passed, ${#failed[@]} failed (logs: build/ci-logs/)"
if [[ "$fail" -ne 0 ]]; then
    stamp "FAILED: ${failed[*]}"
    exit 1
fi
stamp "PASS"
