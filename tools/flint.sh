#!/usr/bin/env bash
set -euo pipefail

CALLER_PWD="$(pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MOJO_BIN=""
FLINT_FORCE_BUILD="${FLINT_FORCE_BUILD:-0}"
FLINT_BUILD_JOBS="${FLINT_BUILD_JOBS:-1}"

# Resolve Mojo 1.2 toolchain in current shell (never inside $(...) so exports persist)
resolve_mojo() {
    if [[ -n "${MOJO:-}" && -x "${MOJO}" ]]; then
        MOJO_BIN="${MOJO}"
        return
    fi
    local candidate_prefix
    for candidate_prefix in \
        "${ROOT}/.pixi/envs/default" \
        "${ROOT}/../.pixi/envs/default" \
        "${HOME}/.pixi/envs/turbomojo"; do
        if [[ -x "${candidate_prefix}/bin/mojo" ]]; then
            export MOJO_PREFIX="${candidate_prefix}"
            export MODULAR_HOME="${candidate_prefix}/share/max"
            export LD_LIBRARY_PATH="${candidate_prefix}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            MOJO_BIN="${candidate_prefix}/bin/mojo"
            return
        fi
    done
    if command -v mojo >/dev/null 2>&1; then
        MOJO_BIN="$(command -v mojo)"
        return
    fi
    echo "[Flint] ERROR: Mojo 1.2 compiler not found. Run 'pixi install' or set MOJO=/path/to/mojo." >&2
    exit 1
}

read_flint_toml_value() {
    local file="$1"
    local section="$2"
    local key="$3"
    local skip_empty="${4:-0}"
    [[ -f "$file" ]] || return 1
    awk -v sec="[$section]" -v k="$key" -v skip_empty="$skip_empty" '
        function strip_toml_comment(s,    i, n, ch, q, esc, sq) {
            sq = sprintf("%c", 39)
            n = length(s)
            q = ""
            esc = 0
            for (i = 1; i <= n; i++) {
                ch = substr(s, i, 1)
                if (q == "") {
                    if (ch == "#") return substr(s, 1, i - 1)
                    if (ch == "\"" || ch == sq) { q = ch; esc = 0 }
                } else if (q == "\"") {
                    if (esc) esc = 0
                    else if (ch == "\\") esc = 1
                    else if (ch == "\"") q = ""
                } else if (ch == sq) {
                    q = ""
                }
            }
            return s
        }
        BEGIN { in_sec = 0; found = 0 }
        { gsub(/\r$/, "") }
        /^[[:space:]]*\[/ {
            hdr = strip_toml_comment($0)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", hdr)
            in_sec = (hdr == sec)
            next
        }
        in_sec {
            line = strip_toml_comment($0)
            eq = index(line, "=")
            if (eq > 0) {
                key_part = substr(line, 1, eq - 1)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", key_part)
                if (key_part == k) {
                    val = substr(line, eq + 1)
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
                    n = length(val)
                    q = sprintf("%c", 39)
                    if (n >= 2 && ((substr(val, 1, 1) == "\"" && substr(val, n, 1) == "\"") || (substr(val, 1, 1) == q && substr(val, n, 1) == q))) {
                        val = substr(val, 2, n - 2)
                    }
                    if (skip_empty == "1" && val == "") {
                        next
                    }
                    print val
                    found = 1
                    exit 0
                }
            }
        }
        END { if (!found) exit 1 }
    ' "$file"
}

validate_project_flint_version() {
    local toml_file="$1"
    local flint_root="${2:-$ROOT}"
    [[ -f "$toml_file" ]] || return 0
    local pin=""
    pin="$(read_flint_toml_value "$toml_file" "engine" "flint_version" || true)"
    pin="${pin#"${pin%%[![:space:]]*}"}"
    pin="${pin%"${pin##*[![:space:]]}"}"
    [[ -n "$pin" ]] || return 0

    local engine_commit="dev"
    if git -C "$flint_root" rev-parse --short HEAD >/dev/null 2>&1; then
        engine_commit="$(git -C "$flint_root" rev-parse --short HEAD)"
    elif [[ -f "${flint_root}/src/flint/core/build_stamp.mojo" ]]; then
        engine_commit="$(awk -F'"' '/BUILD_COMMIT/{print $2; exit}' "${flint_root}/src/flint/core/build_stamp.mojo" || echo "dev")"
    fi

    local op="" rest="$pin"
    case "$rest" in
        ">="*|"<="*|"=="*|"~="*) op="${rest:0:2}"; rest="${rest#??}" ;;
        ">"*|"<"*|"="*|"~"*|"^"*) op="${rest:0:1}"; rest="${rest#?}" ;;
    esac
    rest="${rest#"${rest%%[![:space:]]*}"}"
    if [[ "$rest" == v* || "$rest" == V* ]]; then
        rest="${rest#?}"
    fi

    local ver_part="$rest" commit_part=""
    if [[ "$rest" == *"+"* ]]; then
        ver_part="${rest%%+*}"
        commit_part="${rest#*+}"
    fi
    ver_part="${ver_part%%-*}"

    if [[ ! "$ver_part" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]]; then
        echo "[Flint] ERROR: flint_version invalid '${pin}' in ${toml_file}." >&2
        return 1
    fi
    local maj="" min="0" pat="0" extra_ver=""
    IFS='.' read -r maj min pat extra_ver <<<"$ver_part"
    min="${min:-0}"
    pat="${pat:-0}"
    if [[ -z "$maj" || -n "$extra_ver" || ! "$maj" =~ ^[0-9]+$ || ! "$min" =~ ^[0-9]+$ || ! "$pat" =~ ^[0-9]+$ ]]; then
        echo "[Flint] ERROR: flint_version invalid '${pin}' in ${toml_file}." >&2
        return 1
    fi

    local eng_maj=5 eng_min=0 eng_pat=0
    local cmp=0
    if (( eng_maj != 10#$maj )); then
        if (( eng_maj > 10#$maj )); then cmp=1; else cmp=-1; fi
    elif (( eng_min != 10#$min )); then
        if (( eng_min > 10#$min )); then cmp=1; else cmp=-1; fi
    elif (( eng_pat != 10#$pat )); then
        if (( eng_pat > 10#$pat )); then cmp=1; else cmp=-1; fi
    fi

    local ver_ok=0
    case "$op" in
        ">=") if (( cmp >= 0 )); then ver_ok=1; fi ;;
        ">")  if (( cmp > 0 )); then ver_ok=1; fi ;;
        "<=") if (( cmp <= 0 )); then ver_ok=1; fi ;;
        "<")  if (( cmp < 0 )); then ver_ok=1; fi ;;
        "^"|"~"|"~=") if (( 10#$maj == eng_maj && cmp >= 0 )); then ver_ok=1; fi ;;
        *)    if (( cmp == 0 )); then ver_ok=1; fi ;;
    esac
    if [[ "$ver_ok" -ne 1 ]]; then
        echo "[Flint] ERROR: flint_version mismatch: project requires '${pin}', incompatible with Flint Engine ${eng_maj}.${eng_min}.${eng_pat}+${engine_commit}." >&2
        return 1
    fi

    commit_part="${commit_part#"${commit_part%%[![:space:]]*}"}"
    commit_part="${commit_part%"${commit_part##*[![:space:]]}"}"
    if [[ -n "$commit_part" && "$commit_part" != "dirty" && "$engine_commit" != "dev" && "$commit_part" != "$engine_commit" ]]; then
        echo "[Flint] ERROR: flint_version commit mismatch: project pins '${pin}', engine commit is '${engine_commit}'." >&2
        return 1
    fi
    return 0
}

is_binary_fresh() {
    local src="$1"
    local out="$2"
    local extra_inc="${3-__DEFAULT_TESTS__}"
    local opt="${4:--O3}"
    [[ "${FLINT_FORCE_BUILD:-0}" == "1" ]] && return 1
    [[ -x "$out" ]] || return 1
    [[ -f "${out}.opt" ]] || return 1
    local opt_tag="${opt}|relinc_v1|werror=${FLINT_WERROR:-1}|fp=${FLINT_DETERMINISTIC_FP:-1}|strip=${FLINT_STRIP_SOURCE_PATHS:-0}"
    local prev_opt
    prev_opt="$(cat "${out}.opt" 2>/dev/null || true)"
    [[ "$prev_opt" == "$opt_tag" ]] || return 1
    [[ "$src" -nt "$out" ]] && return 1
    local hit
    hit="$(find "${ROOT}/src" -name '*.mojo' ! -name 'build_stamp.mojo' -newer "$out" -print -quit 2>/dev/null || true)"
    [[ -n "$hit" ]] && return 1
    if [[ "$extra_inc" == "__DEFAULT_TESTS__" && "$src" == "${ROOT}/tests/"* ]]; then
        hit="$(find "${ROOT}/tests" -name '*.mojo' -newer "$out" -print -quit 2>/dev/null || true)"
        [[ -n "$hit" ]] && return 1
    elif [[ -n "$extra_inc" && "$extra_inc" != "__DEFAULT_TESTS__" && -d "$extra_inc" ]]; then
        hit="$(find "$extra_inc" -name '*.mojo' -newer "$out" -print -quit 2>/dev/null || true)"
        [[ -n "$hit" ]] && return 1
    fi
    return 0
}

compile_mojo() {
    local src="$1"
    local out="$2"
    local opt="${3:--O3}"
    local extra_inc="${4-__DEFAULT_TESTS__}"
    if is_binary_fresh "$src" "$out" "$extra_inc" "$opt"; then
        echo "[Flint] Up-to-date: ${out#${ROOT}/}"
        return 0
    fi
    local tmp
    tmp="$(mktemp "/tmp/flint-link-$(basename "$out")-XXXXXX")"
    rm -f "$tmp" "$out" "${out}.opt"
    local fp_args=()
    if [[ "${FLINT_DETERMINISTIC_FP:-1}" != "0" ]]; then
        fp_args=("--fp-mode" "contract=off")
    fi
    local werror_args=()
    if [[ "${FLINT_WERROR:-1}" != "0" ]]; then
        werror_args=("--Werror")
    fi
    local inc_args=("-I" "src")
    if [[ "$extra_inc" == "__DEFAULT_TESTS__" ]]; then
        inc_args+=("-I" "tests")
    elif [[ -n "$extra_inc" ]]; then
        local rel_extra="$extra_inc"
        [[ "$rel_extra" == "${ROOT}/"* ]] && rel_extra="${rel_extra#${ROOT}/}"
        inc_args+=("-I" "$rel_extra")
    fi
    local build_src="$src"
    local entry_tmp_dir=""
    if [[ "${FLINT_STRIP_SOURCE_PATHS:-0}" == "1" && "$src" == "${ROOT}/"* ]]; then
        local rel_entry="${src#${ROOT}/}"
        local rel_entry_dir
        rel_entry_dir="$(dirname "$rel_entry")"
        inc_args+=("-I" "$rel_entry_dir")
        entry_tmp_dir="$(mktemp -d "/tmp/flint-entry-XXXXXX")"
        cp "$src" "${entry_tmp_dir}/$(basename "$src")"
        build_src="${entry_tmp_dir}/$(basename "$src")"
    elif [[ "$build_src" == "${ROOT}/"* ]]; then
        build_src="${build_src#${ROOT}/}"
    fi
    if ! "$MOJO_BIN" build "$build_src" -o "$tmp" "${inc_args[@]}" "$opt" "${fp_args[@]}" "${werror_args[@]}"; then
        rm -f "$tmp"
        [[ -n "$entry_tmp_dir" ]] && rm -rf "$entry_tmp_dir"
        return 1  # also when called from an `if`, where `set -e` does not stop us
    fi
    [[ -n "$entry_tmp_dir" ]] && rm -rf "$entry_tmp_dir"
    mkdir -p "$(dirname "$out")"
    rm -f "$out" "${out}.opt"
    cp "$tmp" "$out"
    chmod +x "$out"
    rm -f "$tmp"
    local opt_tmp
    opt_tmp="$(mktemp "${out}.opt.XXXXXX")"
    local opt_tag="${opt}|relinc_v1|werror=${FLINT_WERROR:-1}|fp=${FLINT_DETERMINISTIC_FP:-1}|strip=${FLINT_STRIP_SOURCE_PATHS:-0}"
    printf '%s\n' "$opt_tag" > "$opt_tmp"
    mv -f "$opt_tmp" "${out}.opt"
}

write_build_stamp() {
    local mojo_bin="$1"
    local commit="dev"
    local dirty="False"
    if git rev-parse --short HEAD >/dev/null 2>&1; then
        commit="$(git rev-parse --short HEAD)"
        if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
            dirty="True"
        fi
    fi
    local mojo_ver
    mojo_ver="$("$mojo_bin" --version 2>/dev/null | awk '{print $2}' || echo "1.2.0")"
    local stamp_file="${ROOT}/src/flint/core/build_stamp.mojo"
    local tmp_stamp
    tmp_stamp="$(mktemp "/tmp/flint-stamp-XXXXXX")"
    cat > "$tmp_stamp" <<EOF
"""Generated by tools/flint.sh — build provenance stamp."""
comptime BUILD_COMMIT: String = "${commit}"
comptime BUILD_DIRTY: Bool = ${dirty}
comptime BUILD_MOJO: String = "${mojo_ver}"
EOF
    if [[ -f "$stamp_file" ]] && cmp -s "$tmp_stamp" "$stamp_file"; then
        rm -f "$tmp_stamp"
        return
    fi
    mv -f "$tmp_stamp" "$stamp_file"
}

cmd_sync() {
    bash "${ROOT}/tools/sync_from_deepsong.sh" "$@"
}

cmd_build() {
    resolve_mojo
    write_build_stamp "$MOJO_BIN"
    mkdir -p "${ROOT}/build" "${ROOT}/build/v5"

    local target=""
    local jobs="${FLINT_BUILD_JOBS:-1}"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force|-f) FLINT_FORCE_BUILD=1; shift ;;
            -j|--jobs) jobs="${2:-1}"; shift 2 ;;
            -j*) jobs="${1#-j}"; shift ;;
            *)
                if [[ -z "$target" ]]; then
                    target="$1"
                fi
                shift
                ;;
        esac
    done

    # If invoked inside a scaffolded game repo containing flint.toml and no engine target was given, build that project!
    if [[ -z "$target" && "$CALLER_PWD" != "$ROOT" && -f "${CALLER_PWD}/flint.toml" ]]; then
        local toml_file="${CALLER_PWD}/flint.toml"
        validate_project_flint_version "$toml_file" || return 1
        local proj_name proj_entry toml_werror toml_det_fp eff_werror eff_det_fp
        proj_name="$(read_flint_toml_value "$toml_file" "project" "name" || true)"
        proj_entry="$(read_flint_toml_value "$toml_file" "project" "entry" || true)"
        [[ -z "$proj_name" ]] && proj_name="$(basename "$CALLER_PWD")"
        [[ -z "$proj_entry" ]] && proj_entry="src/main.mojo"
        toml_werror="$(read_flint_toml_value "$toml_file" "engine" "werror" || echo "true")"
        toml_det_fp="$(read_flint_toml_value "$toml_file" "engine" "deterministic_fp" || echo "true")"
        eff_werror="${FLINT_WERROR:-1}"
        eff_det_fp="${FLINT_DETERMINISTIC_FP:-1}"
        [[ "$toml_werror" == "false" ]] && eff_werror="0"
        [[ "$toml_det_fp" == "false" ]] && eff_det_fp="0"
        echo "[Flint] Building ${proj_name} (${proj_entry}) -> ${CALLER_PWD}/build/${proj_name} ..."
        FLINT_WERROR="$eff_werror" FLINT_DETERMINISTIC_FP="$eff_det_fp" \
            compile_mojo "${CALLER_PWD}/${proj_entry}" "${CALLER_PWD}/build/${proj_name}" "-O3" "${CALLER_PWD}/src"
        echo "[Flint] Build complete."
        return 0
    fi

    target="${target:-all}"
    local build_srcs=()
    local build_outs=()
    local build_labels=()

    if [[ "$target" == "all" || "$target" == "spark" ]]; then
        build_srcs+=("${ROOT}/examples/minimal_spark/main.mojo")
        build_outs+=("${ROOT}/build/flint-spark")
        build_labels+=("examples/minimal_spark/main.mojo -> build/flint-spark")
    fi
    if [[ "$target" == "all" || "$target" == "world" ]]; then
        build_srcs+=("${ROOT}/examples/open_world/main.mojo")
        build_outs+=("${ROOT}/build/flint-world")
        build_labels+=("examples/open_world/main.mojo -> build/flint-world")
    fi
    if [[ "$target" == "all" || "$target" == "studio" ]]; then
        if [[ -f "${ROOT}/examples/studio/main.mojo" ]]; then
            build_srcs+=("${ROOT}/examples/studio/main.mojo")
            build_outs+=("${ROOT}/build/flint-studio")
            build_labels+=("examples/studio/main.mojo -> build/flint-studio")
        fi
    fi
    if [[ "$target" == "all" || "$target" == "cycles" ]]; then
        build_srcs+=("${ROOT}/examples/cycles/main.mojo" "${ROOT}/tests/test_cycles.mojo")
        build_outs+=("${ROOT}/build/flint-cycles" "${ROOT}/build/flint-cycles-test")
        build_labels+=("examples/cycles/main.mojo -> build/flint-cycles" "tests/test_cycles.mojo -> build/flint-cycles-test")
    fi
    if [[ "$target" == "all" || "$target" == "kms" ]]; then
        build_srcs+=("${ROOT}/examples/kms_console/main.mojo")
        build_outs+=("${ROOT}/build/flint-kms")
        build_labels+=("examples/kms_console/main.mojo -> build/flint-kms")
    fi
    if [[ "$target" == "all" || "$target" == "kms3d" ]]; then
        build_srcs+=("${ROOT}/examples/kms_3d/main.mojo")
        build_outs+=("${ROOT}/build/flint-kms3d")
        build_labels+=("examples/kms_3d/main.mojo -> build/flint-kms3d")
    fi
    if [[ "$target" == "all" || "$target" == "pack" ]]; then
        build_srcs+=("${ROOT}/examples/pack_cli/main.mojo")
        build_outs+=("${ROOT}/build/flint-pack")
        build_labels+=("examples/pack_cli/main.mojo -> build/flint-pack")
    fi
    if [[ "$target" == "all" || "$target" == "test" || "$target" == "unit" ]]; then
        build_srcs+=("${ROOT}/tests/test_flint.mojo" "${ROOT}/tests/test_embedded_ide.mojo" "${ROOT}/tests/test_hygiene_and_cooker.mojo")
        build_outs+=("${ROOT}/build/flint-test" "${ROOT}/build/flint-embedded-ide-test" "${ROOT}/build/flint-hygiene-cooker-test")
        build_labels+=("tests/test_flint.mojo -> build/flint-test" "tests/test_embedded_ide.mojo -> build/flint-embedded-ide-test" "tests/test_hygiene_and_cooker.mojo -> build/flint-hygiene-cooker-test")
    fi
    if [[ "$target" == "all" || "$target" == "test" || "$target" == "e2e" ]]; then
        build_srcs+=("${ROOT}/tests/test_e2e.mojo")
        build_outs+=("${ROOT}/build/flint-e2e")
        build_labels+=("tests/test_e2e.mojo -> build/flint-e2e")
    fi
    if [[ "$target" == "all" || "$target" == "test" || "$target" == "v5" ]]; then
        local v5_src
        for v5_src in "${ROOT}"/tests/test_flint_v5_*.mojo; do
            [[ -f "$v5_src" ]] || continue
            local v5_base
            v5_base="$(basename "$v5_src" .mojo)"
            build_srcs+=("$v5_src")
            build_outs+=("${ROOT}/build/v5/${v5_base}")
            build_labels+=("tests/${v5_base}.mojo -> build/v5/${v5_base}")
        done
    fi

    if [[ "$jobs" -le 1 || "${#build_srcs[@]}" -le 1 ]]; then
        local i
        for ((i = 0; i < ${#build_srcs[@]}; i++)); do
            echo "[Flint] Building ${build_labels[$i]} ..."
            compile_mojo "${build_srcs[$i]}" "${build_outs[$i]}"
        done
    else
        echo "[Flint] Building ${#build_srcs[@]} targets with ${jobs} parallel jobs..."
        local pids=()
        local i fail=0
        for ((i = 0; i < ${#build_srcs[@]}; i++)); do
            echo "[Flint] Building ${build_labels[$i]} ..."
            compile_mojo "${build_srcs[$i]}" "${build_outs[$i]}" &
            pids+=($!)
            if [[ "${#pids[@]}" -ge "$jobs" ]]; then
                wait "${pids[0]}" || fail=1
                pids=("${pids[@]:1}")
            fi
        done
        local p
        for p in "${pids[@]}"; do
            wait "$p" || fail=1
        done
        if [[ "$fail" -ne 0 ]]; then
            return 1
        fi
    fi
    echo "[Flint] Build complete."
}

cmd_run() {
    if [[ ! -f "${CALLER_PWD}/flint.toml" ]]; then
        echo "[Flint] ERROR: No flint.toml found in ${CALLER_PWD}. Run from a scaffolded Flint game directory or use 'play'/'world'/'studio'." >&2
        return 1
    fi
    resolve_mojo
    write_build_stamp "$MOJO_BIN"
    local toml_file="${CALLER_PWD}/flint.toml"
    validate_project_flint_version "$toml_file" || return 1
    local proj_name proj_entry toml_werror toml_det_fp eff_werror eff_det_fp
    proj_name="$(read_flint_toml_value "$toml_file" "project" "name" || true)"
    proj_entry="$(read_flint_toml_value "$toml_file" "project" "entry" || true)"
    [[ -z "$proj_name" ]] && proj_name="$(basename "$CALLER_PWD")"
    [[ -z "$proj_entry" ]] && proj_entry="src/main.mojo"
    toml_werror="$(read_flint_toml_value "$toml_file" "engine" "werror" || echo "true")"
    toml_det_fp="$(read_flint_toml_value "$toml_file" "engine" "deterministic_fp" || echo "true")"
    eff_werror="${FLINT_WERROR:-1}"
    eff_det_fp="${FLINT_DETERMINISTIC_FP:-1}"
    [[ "$toml_werror" == "false" ]] && eff_werror="0"
    [[ "$toml_det_fp" == "false" ]] && eff_det_fp="0"
    local bin="${CALLER_PWD}/build/${proj_name}"
    echo "[Flint] Building ${proj_name} (${proj_entry}) -> ${bin} ..."
    FLINT_WERROR="$eff_werror" FLINT_DETERMINISTIC_FP="$eff_det_fp" \
        compile_mojo "${CALLER_PWD}/${proj_entry}" "$bin" "-O3" "${CALLER_PWD}/src"
    cd "$CALLER_PWD"
    exec "$bin" "$@"
}

cmd_pack() {
    cmd_build pack
    if [[ $# -eq 0 && -f "${CALLER_PWD}/flint.toml" ]]; then
        local pack_out
        pack_out="$(read_flint_toml_value "${CALLER_PWD}/flint.toml" "assets" "pack_output" 1 || true)"
        [[ -z "$pack_out" ]] && pack_out="$(read_flint_toml_value "${CALLER_PWD}/flint.toml" "project" "pak_path" 1 || true)"
        [[ -z "$pack_out" ]] && pack_out="$(read_flint_toml_value "${CALLER_PWD}/flint.toml" "assets" "pak_path" 1 || true)"
        [[ -z "$pack_out" ]] && pack_out="build/assets.flpk"
        if [[ ! -d "${CALLER_PWD}/assets" ]]; then
            echo "[Flint Pack] ERROR: No assets/ directory found in ${CALLER_PWD}." >&2
            return 1
        fi
        local asset_files=()
        local f
        while IFS= read -r -d '' f; do
            asset_files+=("${f#${CALLER_PWD}/}")
        done < <(find "${CALLER_PWD}/assets" -type f -size +0c -print0 | sort -z)
        if [[ ${#asset_files[@]} -eq 0 ]]; then
            echo "[Flint Pack] ERROR: No non-empty asset files found under ${CALLER_PWD}/assets." >&2
            return 1
        fi
        (cd "$CALLER_PWD" && mkdir -p "$(dirname "$pack_out")" && "${ROOT}/build/flint-pack" "$pack_out" "${asset_files[@]}")
        return $?
    fi
    (cd "$CALLER_PWD" && "${ROOT}/build/flint-pack" "$@")
}

run_v5_suites() {
    local v5_bin
    for v5_bin in "${ROOT}"/build/v5/test_flint_v5_*; do
        [[ -x "$v5_bin" ]] || continue
        echo "[Flint] Running $(basename "$v5_bin")..."
        "$v5_bin"
    done
}

cmd_test() {
    local sub="${1:-}"
    if [[ "$sub" == "e2e" ]]; then
        cmd_build e2e
        echo "[Flint] Running Flint Engine E2E test suite..."
        "${ROOT}/build/flint-e2e"
    elif [[ "$sub" == "unit" ]]; then
        cmd_build unit
        echo "[Flint] Running Flint Engine unit test suite..."
        "${ROOT}/build/flint-test"
        echo "[Flint] Running Flint Embedded IDE test suite..."
        "${ROOT}/build/flint-embedded-ide-test"
        echo "[Flint] Running Flint Hygiene & Cooker test suite..."
        "${ROOT}/build/flint-hygiene-cooker-test"
    elif [[ "$sub" == "v5" ]]; then
        cmd_build v5
        run_v5_suites
    else
        cmd_build test
        echo "[Flint] Running Flint Engine test suite..."
        "${ROOT}/build/flint-test"
        echo "[Flint] Running Flint Embedded IDE test suite..."
        "${ROOT}/build/flint-embedded-ide-test"
        echo "[Flint] Running Flint Hygiene & Cooker test suite..."
        "${ROOT}/build/flint-hygiene-cooker-test"
        echo "[Flint] Running Flint Engine E2E test suite..."
        "${ROOT}/build/flint-e2e"
        run_v5_suites
        echo "[Flint] Running Studio IDE & Profile Compare headless suites..."
        bash "${ROOT}/tests/test_studio_ide.sh"
        bash "${ROOT}/tests/test_studio_profile_compare.sh"
    fi
}

cmd_lab() {
    local target=""
    local opt="-O3"
    local run_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --fast) opt="-O1"; shift ;;
            -h|--help)
                echo "Usage: bash tools/flint.sh lab <physics|anim|render|audio_ui|net|all> [--fast] [-- args...]"
                echo "  Compiles labs/lab_<name>.mojo (-O3 default, -O1 with --fast) and runs subsystem benchmarks."
                return 0
                ;;
            --) shift; run_args=("$@"); break ;;
            *)
                if [[ -z "$target" ]]; then
                    target="$1"
                else
                    run_args+=("$1")
                fi
                shift
                ;;
        esac
    done

    if [[ -z "$target" ]]; then
        echo "Usage: bash tools/flint.sh lab <physics|anim|render|audio_ui|net|all> [--fast]" >&2
        return 1
    fi

    local labs=()
    case "$target" in
        physics|anim|render|audio_ui|net)
            labs=("$target")
            ;;
        all)
            labs=("physics" "anim" "render" "audio_ui" "net")
            ;;
        *)
            echo "[Flint Lab] ERROR: Unknown lab '$target'. Expected one of: physics, anim, render, audio_ui, net, all." >&2
            return 1
            ;;
    esac

    resolve_mojo
    [[ -f "${ROOT}/src/flint/core/build_stamp.mojo" ]] || write_build_stamp "$MOJO_BIN"
    mkdir -p "${ROOT}/build/labs"

    local lab_name
    for lab_name in "${labs[@]}"; do
        local src="${ROOT}/labs/lab_${lab_name}.mojo"
        if [[ ! -f "$src" ]]; then
            echo "[Flint Lab] ERROR: Missing lab source at ${src}" >&2
            return 1
        fi
        local out="${ROOT}/build/labs/flint-lab-${lab_name}"
        echo "[Flint Lab] Compiling labs/lab_${lab_name}.mojo (${opt}) ..."
        compile_mojo "$src" "$out" "$opt"
        echo "[Flint Lab] Executing lab_${lab_name} ..."
        "$out" "${run_args[@]+"${run_args[@]}"}"
    done
}

# Run the installed terminal IDE with this project's compiler, sources and session.
cmd_studio_ide() {
    local mode="$1"
    shift
    (( $# <= 1 )) || { echo "studio --ide [FILE[:LINE[:COLUMN]]]" >&2; return 2; }
    cd "$CALLER_PWD"
    local ide="${FLINT_TURBOMOJO:-}" location="${1:-}" suffix="" project="$PWD"
    local gui_args=()
    if [[ "$mode" == --moves ]]; then
        mode=--ide-gui
        gui_args=(--moves)
    fi
    local program=tmide override=FLINT_TURBOMOJO
    if [[ "$mode" == --ide-gui ]]; then
        program=tmide-gui
        override=FLINT_TURBOMOJO_GUI
        ide="${FLINT_TURBOMOJO_GUI:-}"
    fi
    if [[ -z "$ide" ]]; then
        ide="$(command -v "$program" || true)"
        [[ -n "$ide" ]] || ide="$HOME/IDE/build/$program"
    fi
    [[ -f "$ide" && -x "$ide" ]] || { echo "Install $program or set $override to its executable." >&2; return 1; }
    ide="$(realpath -- "$ide")"
    local parent="$project"
    while [[ ! -f "$parent/flint.toml" && "$parent" != / ]]; do
        parent="$(dirname -- "$parent")"
    done
    if [[ -f "$parent/flint.toml" ]]; then project="$parent"; fi
    if [[ -z "$location" ]]; then
        if [[ -f "$project/flint.toml" ]]; then
            location="$(read_flint_toml_value "$project/flint.toml" project entry || true)"
            location="${location:-src/main.mojo}"
            [[ "$location" == /* ]] || location="$project/$location"
        else
            project="$ROOT"
            location="$ROOT/examples/studio/main.mojo"
        fi
    fi
    # Match TurboMojo's FILE:LINE:COL grammar. A literal filename takes priority.
    if [[ ! -f "$location" && "$location" =~ ^(.+):([0-9]+)$ ]]; then
        location="${BASH_REMATCH[1]}"
        suffix=":${BASH_REMATCH[2]}"
        if [[ ! -f "$location" && "$location" =~ ^(.+):([0-9]+)$ ]]; then
            location="${BASH_REMATCH[1]}"
            suffix=":${BASH_REMATCH[2]}$suffix"
        fi
        if [[ "$suffix" =~ :0+(:|$) ]]; then
            echo "Source line and column numbers start at 1." >&2
            return 2
        fi
    fi
    [[ -f "$location" ]] || { echo "Source file does not exist: $location" >&2; return 1; }
    location="$(realpath -- "$location")$suffix"
    cd "$project"
    resolve_mojo
    export PATH="$(dirname "$MOJO_BIN"):$PATH"
    if [[ -f flint.toml ]]; then
        mkdir -p build/flint-ide
        local imports="build/flint-ide/engine-src"
        if [[ ! -e "$imports" || -L "$imports" ]]; then
            ln -sfnT -- "$ROOT/src" "$imports"
        else
            echo "Expected an engine source link at $project/$imports" >&2
            return 1
        fi
    fi
    local base="${FLINT_IDE_STATE_BASE:-${XDG_STATE_HOME:-$HOME/.local/state}}" key
    key="$(printf '%s' "$PWD" | sha256sum)"
    local state="$base/flint/ide/${key%% *}"
    mkdir -p -- "$state"
    if [[ "$mode" == --ide-embedded || "$mode" == --ide-gui ]]; then
        # Each live pane owns its buffers and IPC files. Reuse its history on subsequent launches.
        local slot=1 pane
        while :; do
            pane="$state/${mode#--ide-}-$slot"
            mkdir -p -- "$pane"
            exec 9>"$pane/session.lock"
            flock -n 9 && break
            slot=$((slot + 1))
        done
        exec env "TERM=xterm-256color" "COLORTERM=truecolor" "FLINT_IDE_STATE_BASE=$base" \
            "XDG_STATE_HOME=$pane" "TURBOMOJO_COMPILER=${TURBOMOJO_COMPILER:-$MOJO_BIN}" "$ide" "${gui_args[@]}" "$location"
    fi
    local cmd=(env "FLINT_IDE_STATE_BASE=$base" "XDG_STATE_HOME=$state" "PATH=$PATH"
        "TURBOMOJO_COMPILER=${TURBOMOJO_COMPILER:-$MOJO_BIN}"
        "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "MODULAR_HOME=${MODULAR_HOME:-}" "$ide" --reuse "$location")
    if [[ "$mode" == --ide && -t 0 && -t 1 ]]; then
        exec "${cmd[@]}"
    fi
    if [[ -s "$state/turbomojo/ide-state.json" ]] &&
        TERM=xterm-256color "${cmd[@]}" </dev/null >/dev/null 2>&1; then
        return 0
    fi
    # A terminal-launched --reuse falls back to a full interactive session on first use.
    if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
        echo "Open Studio --ide in an interactive terminal or launch it on a desktop." >&2
        return 1
    elif command -v gnome-terminal >/dev/null; then
        gnome-terminal --working-directory="$project" -- "${cmd[@]}"
    elif command -v x-terminal-emulator >/dev/null; then
        x-terminal-emulator -e "${cmd[@]}" </dev/null &
    else
        echo "No desktop terminal found. Run studio --ide in your terminal." >&2
        return 1
    fi
}

cmd_studio() {
    export FLINT_STUDIO_IDE_LAUNCHER="$ROOT/tools/flint.sh"
    export FLINT_STUDIO_PROJECT="$CALLER_PWD"
    export FLINT_STUDIO_IDE_FILE=""
    case "${1:-}" in
        --ide|--ide-window|--ide-embedded|--ide-gui|--moves) cmd_studio_ide "$@"; return ;;
        --takes)
            shift
            resolve_mojo
            compile_mojo "$ROOT/examples/kimodo/takes.mojo" "$ROOT/build/flint-takes" "-O3" "$ROOT/examples/kimodo"
            cd "$CALLER_PWD"
            exec "$ROOT/build/flint-takes" "$@"
            ;;
        --kimodo)
            shift
            resolve_mojo
            compile_mojo "$ROOT/examples/kimodo/main.mojo" "$ROOT/build/flint-kimodo" "-O3" "$ROOT/examples/kimodo"
            cd "$CALLER_PWD"
            exec "$ROOT/build/flint-kimodo" "$@"
            ;;
        --compare|--build-profiles)
            local action="$1" viewer="${ROOT}/build/flint-profile-studio"
            shift
            resolve_mojo
            compile_mojo "${ROOT}/examples/profile/main.mojo" "$viewer" "-O3" ""
            if [[ "$action" == "--compare" ]]; then
                (cd "$CALLER_PWD" && "$viewer" "$@")
                return
            fi
            local profiles=() selection selected
            mapfile -t profiles < <(profile_names)
            (( ${#profiles[@]} )) || { echo "No build profiles in $CALLER_PWD/flint.toml" >&2; return 1; }
            selection="$(mktemp)"
            "$viewer" --pick "$selection" "${profiles[@]}" || { rm -f "$selection"; return 1; }
            selected="$(cat "$selection")"
            rm -f "$selection"
            [[ -n "$selected" ]] || return 0
            cmd_package --profile "$selected"
            return ;;
        --capture)
            shift
            if [[ "${1:-}" == "--ssh" ]]; then
                [[ $# == 4 && -n "$2" && "$2" != -* ]] || { echo "studio --capture --ssh HOST REMOTE_TRACE LOCAL_TRACE" >&2; return 1; }
                (cd "$CALLER_PWD" && remote_capture "$2" "$3" "$4")
                return
            fi
            [[ $# == 1 ]] || { echo "studio --capture TRACE_PATH" >&2; return 1; }
            (cd "$CALLER_PWD" && request_capture "$1")
            return ;;
    esac
    if [[ "${1:-}" == "--motion" || "${1:-}" == "--combat" || "${1:-}" == "--room" ]]; then
        local tool="${1#--}"
        shift
        resolve_mojo
        local source="${ROOT}/examples/motion/main.mojo" binary="${ROOT}/build/flint-motion" include=""
        if [[ -f "${CALLER_PWD}/flint.toml" ]]; then
            local entry
            entry="$(read_flint_toml_value "${CALLER_PWD}/flint.toml" studio "${tool}_entry" || true)"
            if [[ -n "$entry" ]]; then
                source="${CALLER_PWD}/$entry"
                binary="${CALLER_PWD}/build/flint-${tool}"
                include="${CALLER_PWD}/src"
            fi
        fi
        if [[ "$tool" != motion && -z "$include" ]]; then
            echo "studio --$tool needs a project with [studio] ${tool}_entry in flint.toml" >&2
            return 1
        fi
        export FLINT_STUDIO_IDE_FILE="$source"
        compile_mojo "$source" "$binary" "-O3" "$include"
        cd "$CALLER_PWD"
        exec "$binary" "$@"
    fi
    if [[ ! -f "${ROOT}/examples/studio/main.mojo" ]]; then
        echo "[Flint Studio] ERROR: examples/studio/main.mojo not found." >&2
        exit 1
    fi
    local is_mcp=0
    local proj_raw=""
    local has_proj_flag=0
    local argv_arr=("$@")
    local idx=0
    while [[ "$idx" -lt ${#argv_arr[@]} ]]; do
        local cur="${argv_arr[$idx]}"
        if [[ "$cur" == "--mcp-stdio" ]]; then
            is_mcp=1
        elif [[ "$cur" == "--project" ]]; then
            if [[ $((idx + 1)) -ge ${#argv_arr[@]} ]]; then
                echo "[Flint Studio] ERROR: --project requires a directory argument." >&2
                return 1
            fi
            idx=$((idx + 1))
            proj_raw="${argv_arr[$idx]}"
            has_proj_flag=1
        elif [[ "$cur" == --project=* ]]; then
            proj_raw="${cur#--project=}"
            has_proj_flag=1
        fi
        idx=$((idx + 1))
    done
    if [[ "$has_proj_flag" -eq 1 ]]; then
        if [[ -z "$proj_raw" ]]; then
            echo "[Flint Studio] ERROR: --project requires a non-empty directory." >&2
            return 1
        fi
        local proj_cand="$proj_raw"
        if [[ "$proj_cand" != /* ]]; then
            proj_cand="${CALLER_PWD}/${proj_cand}"
        fi
        if [[ ! -d "$proj_cand" ]]; then
            echo "[Flint Studio] ERROR: --project directory does not exist: $proj_raw" >&2
            return 1
        fi
        FLINT_STUDIO_PROJECT="$(cd "$proj_cand" && pwd)"
        export FLINT_STUDIO_PROJECT
    fi
    if [[ -n "${FLINT_STUDIO_PROJECT:-}" && -f "${FLINT_STUDIO_PROJECT}/flint.toml" ]]; then
        validate_project_flint_version "${FLINT_STUDIO_PROJECT}/flint.toml" || return 1
    fi
    if [[ "$is_mcp" -eq 1 ]]; then
        cmd_build studio >&2
        cd "$CALLER_PWD"
    else
        cmd_build studio
        cd "$CALLER_PWD"
    fi
    exec "${ROOT}/build/flint-studio" "$@"
}

cmd_mcp() {
    cmd_studio --mcp-stdio "$@"
}

cmd_doctor() {
    echo "============================================================"
    echo " FLINT ENGINE ENVIRONMENT DOCTOR (Pure Mojo 1.2)"
    echo "============================================================"

    # 1. Mojo 1.2 Compiler
    resolve_mojo
    local mojo_ver
    mojo_ver="$("$MOJO_BIN" --version 2>/dev/null || echo "unknown")"
    if [[ "$mojo_ver" == *"1.2"* ]]; then
        echo "  [OK]   Mojo 1.2 toolchain : ${MOJO_BIN} (${mojo_ver})"
    else
        echo "  [WARN] Mojo toolchain     : ${MOJO_BIN} (${mojo_ver} — expected 1.2.x)"
    fi

    # 2. Vulkan Loader & ICDs
    local vk_lib="missing"
    local p
    for p in \
        /usr/lib/x86_64-linux-gnu/libvulkan.so.1 \
        /usr/lib/aarch64-linux-gnu/libvulkan.so.1 \
        /usr/lib64/libvulkan.so.1 \
        /usr/lib/libvulkan.so.1; do
        if [[ -e "$p" ]]; then vk_lib="$p"; break; fi
    done
    if [[ "$vk_lib" == "missing" ]] && command -v ldconfig >/dev/null 2>&1; then
        local ld_hit
        ld_hit="$(ldconfig -p 2>/dev/null | awk '/libvulkan\.so\.1/{print $NF; exit}' || true)"
        [[ -n "$ld_hit" ]] && vk_lib="$ld_hit"
    fi
    local icd_count=0
    local icd_dir
    for icd_dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
        if [[ -d "$icd_dir" ]]; then
            local c
            c="$(find "$icd_dir" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
            icd_count=$((icd_count + c))
        fi
    done
    if [[ "$vk_lib" != "missing" && $icd_count -gt 0 ]]; then
        echo "  [OK]   Vulkan ICD         : ${vk_lib} (${icd_count} ICD manifest(s))"
    else
        echo "  [WARN] Vulkan ICD         : loader=${vk_lib}, manifests=${icd_count} (CPU SimdCanvas fallback active)"
    fi

    # 3. Display Backends (Wayland / X11 / DRM-KMS)
    local disp_notes=()
    if [[ -n "${WAYLAND_DISPLAY:-}" && -n "${XDG_RUNTIME_DIR:-}" && -S "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}" ]]; then
        disp_notes+=("Wayland(${WAYLAND_DISPLAY})")
    fi
    if [[ -n "${DISPLAY:-}" || -d /tmp/.X11-unix ]]; then
        disp_notes+=("X11(${DISPLAY:-socket})")
    fi
    local dri_cards=0
    if [[ -d /dev/dri ]]; then
        dri_cards="$(find /dev/dri -maxdepth 1 -name 'card*' 2>/dev/null | wc -l)"
    fi
    if [[ $dri_cards -gt 0 ]]; then
        disp_notes+=("DRM/KMS(${dri_cards} card(s))")
    fi
    if [[ ${#disp_notes[@]} -gt 0 ]]; then
        echo "  [OK]   Display backends   : ${disp_notes[*]}"
    else
        echo "  [INFO] Display backends   : headless environment (offscreen PNG / Vulkan headless ready)"
    fi

    # 4. Audio Backends (PulseAudio / PipeWire / ALSA)
    local audio_notes=()
    if [[ -n "${PULSE_SERVER:-}" ]] || [[ -n "${XDG_RUNTIME_DIR:-}" && -S "${XDG_RUNTIME_DIR}/pulse/native" ]]; then
        audio_notes+=("PulseAudio")
    fi
    if [[ -n "${XDG_RUNTIME_DIR:-}" && -S "${XDG_RUNTIME_DIR}/pipewire-0" ]]; then
        audio_notes+=("PipeWire")
    fi
    if [[ -d /dev/snd ]] && [[ "$(find /dev/snd -maxdepth 1 -name 'pcm*' 2>/dev/null | wc -l)" -gt 0 ]]; then
        audio_notes+=("ALSA(/dev/snd)")
    fi
    if [[ ${#audio_notes[@]} -gt 0 ]]; then
        echo "  [OK]   Audio backends     : ${audio_notes[*]}"
    else
        echo "  [INFO] Audio backends     : null/offline mixer ready (no live PulseAudio/ALSA socket detected)"
    fi

    # 5. Linux evdev / Gamepad Input
    local ev_total=0
    local ev_readable=0
    if [[ -d /dev/input ]]; then
        local ev
        for ev in /dev/input/event* /dev/input/js*; do
            [[ -e "$ev" ]] || continue
            ev_total=$((ev_total + 1))
            [[ -r "$ev" ]] && ev_readable=$((ev_readable + 1))
        done
    fi
    if [[ $ev_readable -gt 0 ]]; then
        echo "  [OK]   Input (evdev)      : ${ev_readable}/${ev_total} device node(s) directly readable"
    else
        echo "  [INFO] Input (evdev)      : ${ev_total} node(s) (Wayland/X11 seat or systemd-logind handles input)"
    fi

    # 6. Linux inotify Hot-Reload Limits
    local max_watches="unknown"
    local max_inst="unknown"
    [[ -r /proc/sys/fs/inotify/max_user_watches ]] && max_watches="$(cat /proc/sys/fs/inotify/max_user_watches)"
    [[ -r /proc/sys/fs/inotify/max_user_instances ]] && max_inst="$(cat /proc/sys/fs/inotify/max_user_instances)"
    if [[ "$max_watches" =~ ^[0-9]+$ ]] && [[ "$max_watches" -ge 65536 ]]; then
        echo "  [OK]   inotify limits     : max_user_watches=${max_watches}, max_user_instances=${max_inst}"
    else
        echo "  [WARN] inotify limits     : max_user_watches=${max_watches}, max_user_instances=${max_inst} (recommend >= 65536)"
    fi
    echo "============================================================"
}

cmd_cycles() {
    cmd_build cycles
    exec "${ROOT}/build/flint-cycles" "$@"
}

cmd_play() {
    cmd_build spark
    exec "${ROOT}/build/flint-spark" "$@"
}

cmd_kms() {
    cmd_build kms
    FLINT_BACKEND="${FLINT_BACKEND:-kms}" exec "${ROOT}/build/flint-kms" "$@"
}

cmd_kms3d() {
    cmd_build kms3d
    FLINT_BACKEND="${FLINT_BACKEND:-kms}" exec "${ROOT}/build/flint-kms3d" "$@"
}

# Live preview: keep a program's window open while you edit. Every save of a .mojo file
# (the program's own tree and Flint's src/) rebuilds it in the background; a good build
# replaces the window, a failed one prints the errors and leaves the old window running.
preview_source() {
    case "$1" in
        spark|play) echo "${ROOT}/examples/minimal_spark/main.mojo" ;;
        world)      echo "${ROOT}/examples/open_world/main.mojo" ;;
        studio)     echo "${ROOT}/examples/studio/main.mojo" ;;
        cycles)     echo "${ROOT}/examples/cycles/main.mojo" ;;
        *)
            if [[ -d "$1" && -f "$1/main.mojo" ]]; then echo "$(cd "$1" && pwd)/main.mojo"
            elif [[ -d "$1" && -f "$1/src/main.mojo" ]]; then echo "$(cd "$1/src" && pwd)/main.mojo"
            elif [[ -f "$1" ]]; then echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
            fi ;;
    esac
}

cmd_preview() {
    local target="${1:-}"
    if [[ -z "$target" || "$target" == "-h" || "$target" == "--help" ]]; then
        echo "Usage: bash tools/flint.sh preview <spark|world|studio|cycles|program.mojo|project_dir> [--release] [-- program args...]"
        echo "  Rebuilds on every .mojo save and swaps the window. -O1 builds; --release uses -O3."
        echo "  Runs windowed (FLINT_WINDOWED=1); Ctrl+C stops the preview and its window."
        return 1
    fi
    shift
    local opt="-O1"
    local args=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --release) opt="-O3"; shift ;;
            --) shift; args=("$@"); break ;;
            *) args+=("$1"); shift ;;
        esac
    done
    local src
    src="$(preview_source "$target")"
    if [[ -z "$src" || ! -f "$src" ]]; then
        echo "[Preview] No program at '$target' (a target name, a .mojo file, or a directory with main.mojo / src/main.mojo)." >&2
        return 1
    fi
    resolve_mojo
    [[ -f "${ROOT}/src/flint/core/build_stamp.mojo" ]] || write_build_stamp "$MOJO_BIN"
    # A program inside a project (.../src/...) runs from the project root, where its assets live.
    local run_dir="$ROOT"
    local watch_dirs=("${ROOT}/src" "$(dirname "$src")")
    if [[ "$src" == */src/* ]]; then
        run_dir="${src%%/src/*}"
        watch_dirs+=("${run_dir}/src")
    fi
    local name
    name="$(basename "$(dirname "$src")")"
    [[ "$name" == "src" ]] && name="$(basename "$run_dir")"
    mkdir -p "${ROOT}/build/preview"
    local bin="${ROOT}/build/preview/${name}"
    local stamp="${ROOT}/build/preview/.${name}.stamp"
    local child=0

    stop_child() {
        if [[ $child -gt 0 ]] && kill -0 "$child" 2>/dev/null; then
            kill -TERM "$child" 2>/dev/null || true
            for _ in $(seq 30); do kill -0 "$child" 2>/dev/null || break; sleep 0.1; done
            kill -KILL "$child" 2>/dev/null || true
            wait "$child" 2>/dev/null || true
        fi
        child=0
    }
    launch_child() {
        ( cd "$run_dir" && FLINT_WINDOWED="${FLINT_WINDOWED:-1}" exec "$bin" "${args[@]+"${args[@]}"}" ) &
        child=$!
        echo "[Preview] Running ${name} (pid ${child}). Edit and save; Ctrl+C stops."
    }
    rebuild() {
        touch "$stamp"  # edits saved while compiling trigger the next rebuild
        local started=$SECONDS
        local preview_inc=""
        if [[ "$run_dir" != "$ROOT" && -d "${run_dir}/src" ]]; then
            preview_inc="${run_dir}/src"
        fi
        echo "[Preview] Building ${src#${run_dir}/} (${opt}) ..."
        if compile_mojo "$src" "${bin}.next" "$opt" "$preview_inc"; then
            echo "[Preview] Built in $((SECONDS - started)) s."
            stop_child
            if mv -f "${bin}.next" "$bin"; then
                mv -f "${bin}.next.opt" "${bin}.opt" 2>/dev/null || true
                launch_child
            fi
        else
            echo "[Preview] Build failed after $((SECONDS - started)) s; the previous window keeps running. Fix and save again."
        fi
    }
    changed_file() {
        find "${watch_dirs[@]}" -name '*.mojo' -newer "$stamp" \
            ! -path '*/build/*' ! -name 'build_stamp.mojo' -print -quit 2>/dev/null
    }

    trap 'echo; echo "[Preview] Stopping."; stop_child; exit 0' INT TERM
    trap 'stop_child' EXIT  # never leave a window behind, however the preview ends
    echo "[Preview] Watching: ${watch_dirs[*]}"
    rebuild
    while true; do
        sleep 0.5
        if [[ $child -gt 0 ]] && ! kill -0 "$child" 2>/dev/null; then
            wait "$child" 2>/dev/null || true
            child=0
            echo "[Preview] The window closed. Save a file to build and open it again (Ctrl+C to quit)."
        fi
        local hit
        hit="$(changed_file)"
        if [[ -n "$hit" ]]; then
            sleep 0.3  # let an editor finish writing (and catch save-all)
            echo "[Preview] Changed: ${hit#${run_dir}/}"
            rebuild
        fi
    done
}

cmd_world() {
    cmd_build world
    mkdir -p "${ROOT}/build/shots"
    exec "${ROOT}/build/flint-world" "$@"
}

cmd_shot() {
    cmd_build spark
    mkdir -p "${ROOT}/build/shots"
    "${ROOT}/build/flint-spark" --shots
}

cmd_gate() {
    echo "============================================================"
    echo " FLINT ENGINE STANDALONE VERIFICATION GATE"
    echo "============================================================"
    cmd_sync --verify
    cmd_build all
    "${ROOT}/build/flint-test"
    "${ROOT}/build/flint-embedded-ide-test"
    "${ROOT}/build/flint-hygiene-cooker-test"
    bash "${ROOT}/tests/test_studio_ide.sh"
    bash "${ROOT}/tests/test_studio_profile_compare.sh"
    # Render all showcase & Studio headless shots before E2E and FLIP checks.
    mkdir -p "${ROOT}/build/shots"
    "${ROOT}/build/flint-spark" --shots
    "${ROOT}/build/flint-world" --shots
    "${ROOT}/build/flint-studio" --headless-smoke --out "${ROOT}/build/shots/flint_studio_smoke.png"
    "${ROOT}/build/flint-studio" --shots
    "${ROOT}/build/flint-e2e"
    "${ROOT}/build/flint-cycles-test"
    run_v5_suites
    cmd_lab all
    echo "[Flint Gate] Running golden-frame FLIP and perf_history p99 verification..."
    compile_mojo "${ROOT}/tools/gate_verify.mojo" "${ROOT}/build/flint-gate-verify" "-O3"
    "${ROOT}/build/flint-gate-verify"

    echo "============================================================"
    echo " [PASS] FLINT GATE PASSED (100% Standalone + Zero Warnings)"
    echo "============================================================"
}

profile_names() {
    [[ -f "$CALLER_PWD/flint.toml" ]] || return 0
    sed -nE 's/^[[:space:]]*\[build_profiles\.([a-zA-Z0-9_-]+)\][[:space:]]*(#.*)?$/\1/p' "$CALLER_PWD/flint.toml"
}

request_capture() (
    local path="$1"
    [[ -f "$path.ready" ]] || { echo "No listening development game at $path" >&2; return 1; }
    local request token i
    if [[ -f "$path.lock" ]] && flock -n "$path.lock" true; then
        echo "The development game has stopped" >&2
        return 1
    fi
    exec 8>"$path.client.lock"
    flock -n 8 || { echo "Another Studio client is capturing this session" >&2; return 1; }
    request="$(mktemp -- "$path.request.XXXXXX")"
    token="$(basename -- "$request")"
    printf '%s' "$token" > "$request"
    if ! ln -- "$request" "$path.request" 2>/dev/null; then
        rm -f -- "$request"
        echo "Capture already requested" >&2
        return 1
    fi
    rm -f -- "$request"
    for ((i=0; i<1200; i++)); do
        if [[ -s "$path.done" && "$(cat -- "$path.done")" == "$token" && ! -e "$path.request" && -s "$path" ]]; then
            echo "Capture saved: $path"
            return 0
        fi
        [[ -f "$path.ready" ]] || { echo "Game closed before capture completed" >&2; return 1; }
        sleep 0.1
    done
    if [[ -f "$path.request" && "$(cat -- "$path.request")" == "$token" ]]; then
        rm -f -- "$path.request"
    fi
    echo "Capture timed out waiting for the game at $path" >&2
    return 1
)

remote_capture() (
    set -euo pipefail
    local host="$1" remote_file="$2" local_file="$3" temp
    temp="$(mktemp -- "$local_file.partial.XXXXXX")"
    trap 'rm -f -- "$temp"' EXIT
    # The device needs Bash and the running development build. All user paths are shell-quoted.
    {
        printf '%s\n' 'set -euo pipefail'
        declare -f request_capture
        printf 'request_capture %q >&2\ncat -- %q\n' "$remote_file" "$remote_file"
    } | ssh -T -- "$host" bash -s > "$temp"
    [[ -s "$temp" ]] || return 1
    mv -f -- "$temp" "$local_file"
    echo "Downloaded trace: $local_file"
)

cmd_package() {
    local out_dir=""
    local target=""
    local assets_dir=""
    local assets_explicit=0
    local proj_name=""
    local loose_assets=0 profile="" platform=linux opt=-O3
    local profile_section=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                echo "Usage: bash tools/flint.sh package [out_dir] [--profile NAME | --list-profiles] [--out <dir>] [--target <spark|world|studio|cycles|kms|kms3d|path.mojo>] [--assets <dir>] [--name <name>] [--loose-assets]"
                echo "  Builds a self-contained release bundle (bin/, lib/, assets.flpk, flint.toml, MANIFEST.txt, run.sh)."
                return 0
                ;;
            --list-profiles) profile_names; return ;;
            --profile)
                [[ $# -ge 2 && "$2" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Expected a build profile name" >&2; return 1; }
                profile="$2"; shift 2 ;;
            --out)
                [[ $# -ge 2 ]] || { echo "[Flint Package] ERROR: --out requires a directory argument." >&2; return 1; }
                out_dir="$2"; shift 2 ;;
            --out=*)
                out_dir="${1#--out=}"; shift ;;
            --target)
                [[ $# -ge 2 ]] || { echo "[Flint Package] ERROR: --target requires an argument." >&2; return 1; }
                target="$2"; shift 2 ;;
            --target=*)
                target="${1#--target=}"; shift ;;
            --assets)
                [[ $# -ge 2 ]] || { echo "[Flint Package] ERROR: --assets requires a directory argument." >&2; return 1; }
                assets_dir="$2"; assets_explicit=1; shift 2 ;;
            --assets=*)
                assets_dir="${1#--assets=}"; assets_explicit=1; shift ;;
            --name)
                [[ $# -ge 2 ]] || { echo "[Flint Package] ERROR: --name requires a name argument." >&2; return 1; }
                proj_name="$2"; shift 2 ;;
            --name=*)
                proj_name="${1#--name=}"; shift ;;
            --loose-assets)
                loose_assets=1; shift ;;
            -*)
                echo "[Flint Package] ERROR: Unknown option: $1" >&2
                return 1
                ;;
            *)
                if [[ -z "$out_dir" ]]; then
                    out_dir="$1"
                    shift
                else
                    echo "[Flint Package] ERROR: Unexpected argument: $1" >&2
                    return 1
                fi
                ;;
        esac
    done

    if [[ -n "$profile" ]]; then
        profile_section="build_profiles.$profile"
        local manifest="$CALLER_PWD/flint.toml"
        platform="$(read_flint_toml_value "$manifest" "$profile_section" platform)" || { echo "Missing platform for profile $profile" >&2; return 1; }
        opt="-$(read_flint_toml_value "$manifest" "$profile_section" optimization || echo O3)"
        local loose
        loose="$(read_flint_toml_value "$manifest" "$profile_section" loose_assets || echo false)"
        [[ "$opt" =~ ^-O[0-3]$ && "$loose" =~ ^(true|false)$ ]] || { echo "Invalid optimization or loose_assets in $profile" >&2; return 1; }
        [[ "$loose" == false ]] || loose_assets=1
        case "$platform" in
            linux|windows-wsl2) ;;
            *) echo "Unsupported build platform: $platform" >&2; return 1 ;;
        esac
        [[ -z "$target" ]] || { echo "Project profiles use the project entry" >&2; return 1; }
        [[ -n "$out_dir" ]] || out_dir="$CALLER_PWD/build/packages/$profile"
    fi

    if [[ -n "$assets_dir" && "$assets_dir" != /* ]]; then
        assets_dir="${CALLER_PWD}/${assets_dir}"
    fi
    if [[ "$assets_explicit" -eq 1 && ( -z "$assets_dir" || ! -d "$assets_dir" ) ]]; then
        echo "[Flint Package] ERROR: --assets directory not found: ${assets_dir:-<empty>}" >&2
        return 1
    fi

    resolve_mojo
    write_build_stamp "$MOJO_BIN"

    local mode="engine"
    local src=""
    local extra_inc=""
    local proj_version="0.1.0"
    local build_bin=""
    local eff_werror="${FLINT_WERROR:-1}"
    local eff_det_fp="${FLINT_DETERMINISTIC_FP:-1}"
    local pack_output_name="assets.flpk"

    if [[ -z "$target" && -f "${CALLER_PWD}/flint.toml" ]]; then
        mode="project"
        local toml_file="${CALLER_PWD}/flint.toml"
        validate_project_flint_version "$toml_file" || return 1
        local toml_name toml_ver toml_entry toml_werror toml_det_fp toml_pack
        toml_name="$(read_flint_toml_value "$toml_file" "project" "name" || true)"
        toml_ver="$(read_flint_toml_value "$toml_file" "project" "version" || true)"
        toml_entry="$(read_flint_toml_value "$toml_file" "project" "entry" || true)"
        toml_werror="$(read_flint_toml_value "$toml_file" "engine" "werror" || echo "true")"
        toml_det_fp="$(read_flint_toml_value "$toml_file" "engine" "deterministic_fp" || echo "true")"
        toml_pack="$(read_flint_toml_value "$toml_file" "assets" "pack_output" 1 || true)"
        [[ -z "$toml_pack" ]] && toml_pack="$(read_flint_toml_value "$toml_file" "project" "pak_path" 1 || true)"
        [[ -z "$toml_pack" ]] && toml_pack="$(read_flint_toml_value "$toml_file" "assets" "pak_path" 1 || true)"

        [[ -z "$proj_name" ]] && proj_name="${toml_name:-$(basename "$CALLER_PWD")}"
        [[ -n "$toml_ver" ]] && proj_version="$toml_ver"
        local entry_rel="${toml_entry:-src/main.mojo}"
        if [[ "$entry_rel" == /* ]]; then
            src="$entry_rel"
        else
            src="${CALLER_PWD}/${entry_rel}"
        fi
        if [[ -d "${CALLER_PWD}/src" ]]; then
            extra_inc="${CALLER_PWD}/src"
            [[ "$extra_inc" == "${ROOT}/"* ]] && extra_inc="${extra_inc#${ROOT}/}"
        fi
        [[ "$toml_werror" == "false" ]] && eff_werror="0"
        [[ "$toml_det_fp" == "false" ]] && eff_det_fp="0"
        if [[ -n "$toml_pack" ]]; then
            pack_output_name="$(basename "$toml_pack")"
            [[ -z "$pack_output_name" ]] && pack_output_name="assets.flpk"
        fi
        build_bin="${CALLER_PWD}/build/${proj_name}"
        [[ -z "$profile" ]] || build_bin="${CALLER_PWD}/build/profile-bin/$profile/${proj_name}"
        if [[ -z "$assets_dir" && -d "${CALLER_PWD}/assets" ]]; then
            assets_dir="${CALLER_PWD}/assets"
        fi
    else
        target="${target:-spark}"
        local default_name=""
        case "$target" in
            spark)  src="${ROOT}/examples/minimal_spark/main.mojo"; default_name="flint-spark" ;;
            world)  src="${ROOT}/examples/open_world/main.mojo"; default_name="flint-world" ;;
            studio) src="${ROOT}/examples/studio/main.mojo"; default_name="flint-studio" ;;
            cycles) src="${ROOT}/examples/cycles/main.mojo"; default_name="flint-cycles" ;;
            kms)    src="${ROOT}/examples/kms_console/main.mojo"; default_name="flint-kms" ;;
            kms3d)  src="${ROOT}/examples/kms_3d/main.mojo"; default_name="flint-kms3d" ;;
            *)
                if [[ "$target" == /* ]]; then
                    src="$target"
                elif [[ -f "${CALLER_PWD}/${target}" ]]; then
                    src="${CALLER_PWD}/${target}"
                elif [[ -f "${ROOT}/${target}" ]]; then
                    src="${ROOT}/${target}"
                else
                    echo "[Flint Package] ERROR: Unknown target or missing .mojo file: $target" >&2
                    return 1
                fi
                default_name="$(basename "$src" .mojo)"
                ;;
        esac
        [[ -z "$proj_name" ]] && proj_name="$default_name"
        extra_inc=""
        build_bin="${ROOT}/build/${proj_name}"
    fi

    if [[ ! -f "$src" ]]; then
        echo "[Flint Package] ERROR: Entry source not found: ${src}" >&2
        return 1
    fi

    if [[ -z "$out_dir" ]]; then
        out_dir="${CALLER_PWD}/dist/${proj_name}"
    elif [[ "$out_dir" != /* ]]; then
        out_dir="${CALLER_PWD}/${out_dir}"
    fi
    mkdir -p "${out_dir}/bin" "${out_dir}/lib"
    out_dir="$(cd "$out_dir" && pwd)"

    echo "[Flint Package] Building ${proj_name} (${src}) ..."
    mkdir -p "$(dirname "$build_bin")"
    FLINT_WERROR="$eff_werror" FLINT_DETERMINISTIC_FP="$eff_det_fp" FLINT_STRIP_SOURCE_PATHS=1 \
        compile_mojo "$src" "$build_bin" "$opt" "$extra_inc"

    cp -f "$build_bin" "${out_dir}/bin/${proj_name}"
    chmod 0755 "${out_dir}/bin/${proj_name}"
    if ! command -v chrpath >/dev/null 2>&1; then
        echo "[Flint Package] ERROR: 'chrpath' is required to rewrite ELF RUNPATH to '\$ORIGIN/../lib'." >&2
        return 1
    fi
    chrpath -r '$ORIGIN/../lib' "${out_dir}/bin/${proj_name}" >/dev/null
    if command -v readelf >/dev/null 2>&1; then
        local rpath_check
        rpath_check="$(readelf -d "${out_dir}/bin/${proj_name}" 2>/dev/null | grep -E 'RUNPATH|RPATH' || true)"
        if grep -Fq "$ROOT" <<<"$rpath_check" || grep -Fq "/.pixi/" <<<"$rpath_check"; then
            echo "[Flint Package] ERROR: Packaged binary RUNPATH still references repo/pixi: ${rpath_check}" >&2
            return 1
        fi
    fi

    local mojo_lib_dir=""
    if [[ -n "${MOJO_PREFIX:-}" && -d "${MOJO_PREFIX}/lib" ]]; then
        mojo_lib_dir="${MOJO_PREFIX}/lib"
    else
        local cand_dir
        cand_dir="$(cd "$(dirname "$MOJO_BIN")/.." && pwd)/lib"
        [[ -d "$cand_dir" ]] && mojo_lib_dir="$cand_dir"
    fi

    local so_name
    for so_name in libKGENCompilerRTShared.so libMSupportGlobals.so libAsyncRTRuntimeGlobals.so; do
        local so_src=""
        if [[ -n "$mojo_lib_dir" && -f "${mojo_lib_dir}/${so_name}" ]]; then
            so_src="${mojo_lib_dir}/${so_name}"
        elif command -v ldd >/dev/null 2>&1; then
            so_src="$(ldd "${out_dir}/bin/${proj_name}" 2>/dev/null | awk -v n="$so_name" '$1 == n && $3 ~ /^\// {print $3; exit}' || true)"
        fi
        if [[ -n "$so_src" && -f "$so_src" ]]; then
            cp -L -f "$so_src" "${out_dir}/lib/${so_name}"
            chmod 0644 "${out_dir}/lib/${so_name}"
        fi
    done

    local has_flpk=0
    local packed_flpk="${out_dir}/${pack_output_name}"
    if [[ -n "$assets_dir" && -d "$assets_dir" ]]; then
        local abs_assets cook_cache_dir
        abs_assets="$(cd "$assets_dir" && pwd)"
        if [[ "$mode" == "project" ]]; then
            cook_cache_dir="${CALLER_PWD}/build/cook_cache"
        else
            cook_cache_dir="${ROOT}/build/cook_cache/${proj_name}"
        fi

        local cookable_files=()
        local extra_loose_files=()
        local all_files=()
        local f
        while IFS= read -r -d '' f; do
            all_files+=("$f")
            case "$f" in
                *.flmeta)
                    ;;
                *.glb|*.png|*.jpg|*.jpeg|*.hdr|*.exr|*.ttf|*.wav|*.qoa|*.flscn|*.flscnb|*.flprefab|*.flmat|*.flanim|*.flrig|*.json)
                    cookable_files+=("$f")
                    ;;
                *)
                    extra_loose_files+=("$f")
                    ;;
            esac
        done < <(find "$abs_assets" -type f -print0 | sort -z)

        if [[ ${#cookable_files[@]} -gt 0 ]]; then
            echo "[Flint Package] Cooking ${#cookable_files[@]} asset file(s) -> ${packed_flpk} (cache=${cook_cache_dir}) ..."
            compile_mojo "${ROOT}/examples/pack_cli/main.mojo" "${ROOT}/build/flint-pack" "-O3" "" >&2
            "${ROOT}/build/flint-pack" --cook "$abs_assets" "$packed_flpk" --cache-dir "$cook_cache_dir"
            if [[ ! -s "$packed_flpk" ]]; then
                echo "[Flint Package] ERROR: Failed to cook assets into ${packed_flpk}" >&2
                return 1
            fi
            chmod 0644 "$packed_flpk"
            has_flpk=1
        else
            rm -f "$packed_flpk"
        fi

        if [[ "$loose_assets" -eq 1 || ${#extra_loose_files[@]} -gt 0 ]]; then
            local copy_list=()
            if [[ "$loose_assets" -eq 1 ]]; then
                copy_list=("${all_files[@]}")
            else
                copy_list=("${extra_loose_files[@]}")
            fi
            local src_f
            for src_f in "${copy_list[@]}"; do
                [[ -n "$src_f" ]] || continue
                local dest_rel="assets/${src_f#${abs_assets}/}"
                mkdir -p "${out_dir}/$(dirname "$dest_rel")"
                cp -f "$src_f" "${out_dir}/${dest_rel}"
                chmod 0644 "${out_dir}/${dest_rel}"
            done
        fi
    fi

    if [[ "$mode" == "project" && -f "${CALLER_PWD}/flint.toml" ]]; then
        if [[ "$has_flpk" -eq 1 ]]; then
            awk -v po="$pack_output_name" '
                function strip_toml_comment(s,    i, n, ch, q, esc, sq) {
                    sq = sprintf("%c", 39)
                    n = length(s)
                    q = ""
                    esc = 0
                    for (i = 1; i <= n; i++) {
                        ch = substr(s, i, 1)
                        if (q == "") {
                            if (ch == "#") return substr(s, 1, i - 1)
                            if (ch == "\"" || ch == sq) { q = ch; esc = 0 }
                        } else if (q == "\"") {
                            if (esc) esc = 0
                            else if (ch == "\\") esc = 1
                            else if (ch == "\"") q = ""
                        } else if (ch == sq) {
                            q = ""
                        }
                    }
                    return s
                }
                BEGIN { in_assets = 0; in_project = 0 }
                /^[[:space:]]*\[/ {
                    hdr = strip_toml_comment($0); gsub(/\r$/, "", hdr); gsub(/^[[:space:]]+|[[:space:]]+$/, "", hdr)
                    in_assets = (hdr == "[assets]")
                    in_project = (hdr == "[project]")
                }
                (in_assets && /^[[:space:]]*(pack_output|pak_path)[[:space:]]*=/) || (in_project && /^[[:space:]]*pak_path[[:space:]]*=/) {
                    match($0, /^[[:space:]]*/)
                    indent = substr($0, 1, RLENGTH)
                    eq = index($0, "=")
                    k = substr($0, RLENGTH + 1, eq - RLENGTH - 1)
                    gsub(/[[:space:]]+$/, "", k)
                    rhs = substr($0, eq + 1)
                    gsub(/^[[:space:]]+/, "", rhs)
                    q = (substr(rhs, 1, 1) == sprintf("%c", 39)) ? sprintf("%c", 39) : "\""
                    print indent k " = " q po q
                    next
                }
                { print }
            ' "${CALLER_PWD}/flint.toml" > "${out_dir}/flint.toml"
        else
            cp -f "${CALLER_PWD}/flint.toml" "${out_dir}/flint.toml"
        fi
    else
        cat > "${out_dir}/flint.toml" <<EOF
[project]
name = "${proj_name}"
version = "${proj_version}"
target = "${target}"

[engine]
flint_version = ">=5.0"
deterministic_fp = true
werror = true

[assets]
pack_output = "$([[ "$has_flpk" -eq 1 ]] && echo "$pack_output_name" || echo "")"
EOF
    fi
    chmod 0644 "${out_dir}/flint.toml"

    cat > "${out_dir}/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="\${HERE}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
cd "\$HERE"
exec "\${HERE}/bin/${proj_name}" "\$@"
EOF
    chmod 0755 "${out_dir}/run.sh"
    if [[ "$platform" == windows-wsl2 ]]; then
        # Rebuild on a differently architected Windows PC using the same sources and lockfile.
        local kit
        kit="$(mktemp -d -- "$out_dir/.wsl-source.XXXXXX")"
        mkdir -p "$kit/engine" "$kit/game"
        cp -a "$ROOT/src" "$kit/engine/"
        cp "$ROOT/pixi.toml" "$ROOT/pixi.lock" "$kit/engine/"
        cp -a "$ROOT/licenses" "$kit/engine/"
        cp -a "$CALLER_PWD/src" "$kit/game/"
        local entry="${src#"$CALLER_PWD/"}"
        [[ "$entry" != "$src" && "$entry" != *".."* ]] || { echo "WSL source kit requires a project-local entry" >&2; return 1; }
        if [[ "$entry" != src/* ]]; then
            mkdir -p "$kit/game/$(dirname "$entry")"
            cp "$src" "$kit/game/$entry"
        fi
        printf '%s\n' "$entry" > "$kit/entry"
        printf '%s\n' "$opt" > "$kit/optimization"
        : > "$kit/flags"
        if [[ "$eff_werror" != 0 ]]; then echo --Werror >> "$kit/flags"; fi
        if [[ "$eff_det_fp" != 0 ]]; then printf '%s\n' --fp-mode contract=off >> "$kit/flags"; fi
        uname -m > "$kit/build-arch"
        if [[ -d "$out_dir/wsl-source" && -f "$out_dir/wsl-source/entry" ]]; then
            local old_kit
            old_kit="$(mktemp -d -- "$out_dir/.wsl-previous.XXXXXX")"
            mv -T "$out_dir/wsl-source" "$old_kit"
            if ! mv -T "$kit" "$out_dir/wsl-source"; then
                mv -T "$old_kit" "$out_dir/wsl-source"
                return 1
            fi
            rm -rf -- "$old_kit"
        else
            mv -T "$kit" "$out_dir/wsl-source"
        fi
        echo "[Flint Package] Bundled project and Flint source for first-run WSL builds."
        cp "$ROOT/tools/run-wsl.sh" "$out_dir/run-wsl.sh"
        cp "$ROOT/tools/Play-WSL.cmd" "$out_dir/Play-WSL.cmd"
        cat > "$out_dir/Windows-README.txt" <<'WSL_README'
Extract the entire folder, then open Play-WSL.cmd.
This build runs on Windows through a configured WSL2 Linux distribution.
Graphical play requires WSLg and a working Vulkan driver inside that distribution.
The bundled binary runs directly on the matching CPU architecture (see MANIFEST.txt).
For a different architecture, install Pixi inside WSL and launch again. The first
run downloads the locked Linux toolchain, compiles the included project and engine
sources, and caches the player for subsequent launches.
WSL_README
    fi


    local git_commit="dev"
    if git rev-parse --short HEAD >/dev/null 2>&1; then
        git_commit="$(git rev-parse --short HEAD)"
    fi
    local mojo_ver
    mojo_ver="$("$MOJO_BIN" --version 2>/dev/null | awk '{print $2}' || echo "1.2.0")"
    local bin_sha="unknown"
    if command -v sha256sum >/dev/null 2>&1; then
        bin_sha="$(sha256sum "${out_dir}/bin/${proj_name}" | awk '{print $1}')"
    fi
    local manifest_src="$src"
    if [[ "$mode" == "project" && "$manifest_src" == "${CALLER_PWD}/"* ]]; then
        manifest_src="${manifest_src#${CALLER_PWD}/}"
    elif [[ "$manifest_src" == "${ROOT}/"* ]]; then
        manifest_src="${manifest_src#${ROOT}/}"
    fi
    cat > "${out_dir}/MANIFEST.txt" <<EOF
name=${proj_name}
version=${proj_version}
mode=${mode}
source=${manifest_src}
binary=bin/${proj_name}
binary_sha256=${bin_sha}
mojo_version=${mojo_ver}
git_commit=${git_commit}
assets_pack=$([[ "$has_flpk" -eq 1 ]] && echo "$pack_output_name" || echo "none")
platform=$platform
build_profile=$profile
optimization=$opt
architecture=$(uname -m)
packaged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
    chmod 0644 "${out_dir}/MANIFEST.txt"

    echo "[Flint Package] Release bundle ready at: ${out_dir}"
}

cmd_new() {
    local dest="${1:-}"
    if [[ -z "$dest" ]]; then
        echo "Usage: tools/flint.sh new <target_directory>" >&2
        exit 1
    fi
    if [[ "$dest" != /* ]]; then
        dest="${CALLER_PWD}/${dest}"
    fi
    mkdir -p "${dest}/src" "${dest}/assets" "${dest}/tools"
    local abs_dest app_name safe_app_name engine_commit
    abs_dest="$(cd "$dest" && pwd)"
    app_name="$(basename "$abs_dest")"
    safe_app_name="$(printf '%s' "$app_name" | sed 's/[&/\\]/\\&/g')"
    engine_commit="dev"
    if git -C "$ROOT" rev-parse --short HEAD >/dev/null 2>&1; then
        engine_commit="$(git -C "$ROOT" rev-parse --short HEAD)"
    elif [[ -f "${ROOT}/src/flint/core/build_stamp.mojo" ]]; then
        engine_commit="$(awk -F'"' '/BUILD_COMMIT/{print $2; exit}' "${ROOT}/src/flint/core/build_stamp.mojo" || echo "dev")"
    fi

    cat > "${abs_dest}/flint.toml" <<EOF
[project]
name = "${app_name}"
version = "0.1.0"
entry = "src/main.mojo"
entry_scene = "assets/main.flscn"

[engine]
flint_version = "5.0.0+${engine_commit}"
deterministic_fp = true
werror = true

[window]
title = "${app_name}"
width = 1280
height = 720
vsync = true

[physics]
fixed_hz = 60

[net]
tick_hz = 60
max_rollback_ticks = 8
mtu_bytes = 1180

[assets]
pack_output = "build/assets.flpk"
EOF

    cat > "${abs_dest}/assets/main.flscn" <<'EOF'
{
  "format": "flscn",
  "version": 2,
  "name": "main",
  "entities": [
    {
      "slot": 0,
      "parent": -1,
      "components": {
        "LocalTransform": {"translation": [0.0, -0.5, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [16.0, 1.0, 16.0]},
        "GlobalTransform": {"translation": [0.0, -0.5, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [16.0, 1.0, 16.0]},
        "RigidBodyComponent": {"motion_mode": 0, "mass": 0.0, "can_sleep": true},
        "ColliderComponent": {"shape": 2, "half_extents": [8.0, 0.5, 8.0], "is_trigger": false},
        "MeshRendererComponent": {"mesh_id": 0, "material_id": 0, "cast_shadows": true, "visible": true},
        "MaterialOverrideComponent": {"material_id": 0, "base_color": [0.18, 0.24, 0.34], "roughness": 0.82, "metallic": 0.05, "enabled": true},
        "StudioEntityMetaComponent": {"name": "Ground", "enabled": true}
      }
    },
    {
      "slot": 1,
      "parent": -1,
      "components": {
        "LocalTransform": {"translation": [0.0, 1.2, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.1, 1.1, 1.1]},
        "GlobalTransform": {"translation": [0.0, 1.2, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.1, 1.1, 1.1]},
        "RigidBodyComponent": {"motion_mode": 2, "mass": 1.0, "can_sleep": true},
        "ColliderComponent": {"shape": 2, "half_extents": [0.55, 0.55, 0.55], "is_trigger": false},
        "MeshRendererComponent": {"mesh_id": 0, "material_id": 0, "cast_shadows": true, "visible": true},
        "MaterialOverrideComponent": {"material_id": 0, "base_color": [0.95, 0.58, 0.16], "roughness": 0.28, "metallic": 0.25, "enabled": true},
        "StudioEntityMetaComponent": {"name": "Cube", "enabled": true}
      }
    },
    {
      "slot": 2,
      "parent": -1,
      "components": {
        "LocalTransform": {"translation": [6.0, 3.6, 6.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]},
        "GlobalTransform": {"translation": [6.0, 3.6, 6.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]},
        "CameraComponent": {"fov_y_deg": 55.0, "near_z": 0.10, "far_z": 250.0, "exposure": 1.0, "active": true, "look_target_offset": [-6.0, -2.6, -6.0]},
        "AudioListenerComponent": {"active": true},
        "StudioEntityMetaComponent": {"name": "MainCamera", "enabled": true}
      }
    },
    {
      "slot": 3,
      "parent": -1,
      "components": {
        "LocalTransform": {"translation": [4.0, 6.0, 4.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]},
        "GlobalTransform": {"translation": [4.0, 6.0, 4.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]},
        "PointLightComponent": {"radiance": [12.0, 10.0, 8.0], "radius": 28.0, "enabled": true},
        "StudioEntityMetaComponent": {"name": "PointLight", "enabled": true}
      }
    },
    {
      "slot": 4,
      "parent": -1,
      "components": {
        "LocalTransform": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]},
        "GlobalTransform": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]},
        "EnvironmentComponent": {"sun_dir": [0.378803, 0.797480, -0.458551], "sun_radiance": [3.2, 3.0, 2.7], "exposure": 1.0, "ibl_intensity": 1.0, "turbidity": 2.5, "enabled": true},
        "StudioEntityMetaComponent": {"name": "Environment", "enabled": true}
      }
    }
  ]
}
EOF

    cat > "${abs_dest}/pixi.toml" <<EOF
[workspace]
name = "${app_name}"
version = "0.1.0"
description = "Pure-Mojo 1.2 standalone game powered by Flint Engine v5"
channels = ["https://conda.modular.com/max", "conda-forge"]
platforms = ["linux-64", "linux-aarch64"]

[dependencies]
mojo = ">=1.2.0,<1.3"

[tasks]
play = "bash tools/run.sh"
EOF

    cat > "${abs_dest}/.gitignore" <<'EOF'
build/
dist/
.pixi/
*.flpk
*.flcook
*.png
EOF

    sed "s/__FLINT_APP_NAME__/${safe_app_name}/g" > "${abs_dest}/README.md" <<'EOF'
# __FLINT_APP_NAME__

Standalone Pure-Mojo 1.2 game project powered by **Flint Engine v5**.

## Quick Start
- Interactive run: `FLINT_ROOT=/path/to/flint bash tools/run.sh`
- Headless smoke test: `FLINT_ROOT=/path/to/flint bash tools/run.sh --headless-smoke`
- Headless PNG shot: `FLINT_ROOT=/path/to/flint bash tools/run.sh --shots --out build/shots/__FLINT_APP_NAME__.png`
- Pack assets: `bash /path/to/flint/tools/flint.sh pack`
- Live preview: `bash /path/to/flint/tools/flint.sh preview .`
- Release bundle: `bash /path/to/flint/tools/flint.sh package`

## Configuration
Edit `flint.toml` to configure `[project]`, `[engine]`, `[window]`, `[physics]`, `[net]`, and `[assets]`.
EOF

    cat > "${abs_dest}/tools/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLINT_ROOT="${FLINT_ROOT:-}"
if [[ -z "$FLINT_ROOT" ]]; then
    for cand in "${APP_ROOT}/../flint" "${HOME}/flint"; do
        if [[ -d "${cand}/src/flint" ]]; then
            FLINT_ROOT="$cand"
            break
        fi
    done
fi
if [[ -z "$FLINT_ROOT" || ! -d "${FLINT_ROOT}/src/flint" ]]; then
    echo "[Flint] ERROR: Set FLINT_ROOT=/path/to/flint before running." >&2
    exit 1
fi
mkdir -p "${APP_ROOT}/build"

MOJO_BIN="${MOJO:-}"
if [[ -z "$MOJO_BIN" ]]; then
    if [[ -x "${APP_ROOT}/.pixi/envs/default/bin/mojo" ]]; then
        PFX="${APP_ROOT}/.pixi/envs/default"
        export MOJO_PREFIX="$PFX"
        export MODULAR_HOME="${PFX}/share/max"
        export LD_LIBRARY_PATH="${PFX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        MOJO_BIN="${PFX}/bin/mojo"
    elif [[ -x "${FLINT_ROOT}/.pixi/envs/default/bin/mojo" ]]; then
        PFX="${FLINT_ROOT}/.pixi/envs/default"
        export MOJO_PREFIX="$PFX"
        export MODULAR_HOME="${PFX}/share/max"
        export LD_LIBRARY_PATH="${PFX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        MOJO_BIN="${PFX}/bin/mojo"
    else
        MOJO_BIN="mojo"
    fi
fi

_read_toml() {
    local sec="[$1]" k="$2"
    [[ -f "${APP_ROOT}/flint.toml" ]] || return 1
    awk -v sec="$sec" -v k="$k" '
        function strip_toml_comment(s,    i, n, ch, q, esc, sq) {
            sq = sprintf("%c", 39)
            n = length(s)
            q = ""
            esc = 0
            for (i = 1; i <= n; i++) {
                ch = substr(s, i, 1)
                if (q == "") {
                    if (ch == "#") return substr(s, 1, i - 1)
                    if (ch == "\"" || ch == sq) { q = ch; esc = 0 }
                } else if (q == "\"") {
                    if (esc) esc = 0
                    else if (ch == "\\") esc = 1
                    else if (ch == "\"") q = ""
                } else if (ch == sq) {
                    q = ""
                }
            }
            return s
        }
        BEGIN { in_sec = 0; found = 0 }
        { gsub(/\r$/, "") }
        /^[[:space:]]*\[/ {
            hdr = strip_toml_comment($0); gsub(/^[[:space:]]+|[[:space:]]+$/, "", hdr)
            in_sec = (hdr == sec); next
        }
        in_sec {
            line = strip_toml_comment($0)
            eq = index(line, "=")
            if (eq > 0) {
                key_part = substr(line, 1, eq - 1)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", key_part)
                if (key_part == k) {
                    val = substr(line, eq + 1)
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
                    n = length(val)
                    q = sprintf("%c", 39)
                    if (n >= 2 && ((substr(val, 1, 1) == "\"" && substr(val, n, 1) == "\"") || (substr(val, 1, 1) == q && substr(val, n, 1) == q))) {
                        val = substr(val, 2, n - 2)
                    }
                    print val; found = 1; exit 0
                }
            }
        }
        END { if (!found) exit 1 }
    ' "${APP_ROOT}/flint.toml"
}

PROJ_NAME="$(_read_toml project name || basename "$APP_ROOT")"
PROJ_ENTRY="$(_read_toml project entry || echo "src/main.mojo")"
TOML_FLINT_VER="$(_read_toml engine flint_version || true)"
TOML_WERROR="$(_read_toml engine werror || echo "true")"
TOML_DET_FP="$(_read_toml engine deterministic_fp || echo "true")"
[[ -z "$PROJ_NAME" ]] && PROJ_NAME="$(basename "$APP_ROOT")"
[[ -z "$PROJ_ENTRY" ]] && PROJ_ENTRY="src/main.mojo"

_check_flint_version() {
    local pin="$1"
    pin="${pin#"${pin%%[![:space:]]*}"}"
    pin="${pin%"${pin##*[![:space:]]}"}"
    [[ -n "$pin" ]] || return 0
    local engine_commit="dev"
    if git -C "$FLINT_ROOT" rev-parse --short HEAD >/dev/null 2>&1; then
        engine_commit="$(git -C "$FLINT_ROOT" rev-parse --short HEAD)"
    elif [[ -f "${FLINT_ROOT}/src/flint/core/build_stamp.mojo" ]]; then
        engine_commit="$(awk -F'"' '/BUILD_COMMIT/{print $2; exit}' "${FLINT_ROOT}/src/flint/core/build_stamp.mojo" || echo "dev")"
    fi
    local op="" rest="$pin"
    case "$rest" in
        ">="*|"<="*|"=="*|"~="*) op="${rest:0:2}"; rest="${rest#??}" ;;
        ">"*|"<"*|"="*|"~"*|"^"*) op="${rest:0:1}"; rest="${rest#?}" ;;
    esac
    rest="${rest#"${rest%%[![:space:]]*}"}"
    if [[ "$rest" == v* || "$rest" == V* ]]; then
        rest="${rest#?}"
    fi
    local ver_part="$rest" commit_part=""
    if [[ "$rest" == *"+"* ]]; then
        ver_part="${rest%%+*}"
        commit_part="${rest#*+}"
    fi
    ver_part="${ver_part%%-*}"
    if [[ ! "$ver_part" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]]; then
        echo "[Flint] ERROR: flint_version invalid '${pin}'." >&2
        return 1
    fi
    local maj="" min="0" pat="0" extra_ver=""
    IFS='.' read -r maj min pat extra_ver <<<"$ver_part"
    min="${min:-0}"
    pat="${pat:-0}"
    if [[ -z "$maj" || -n "$extra_ver" || ! "$maj" =~ ^[0-9]+$ || ! "$min" =~ ^[0-9]+$ || ! "$pat" =~ ^[0-9]+$ ]]; then
        echo "[Flint] ERROR: flint_version invalid '${pin}'." >&2
        return 1
    fi
    local eng_maj=5 eng_min=0 eng_pat=0 cmp=0
    if (( eng_maj != 10#$maj )); then
        if (( eng_maj > 10#$maj )); then cmp=1; else cmp=-1; fi
    elif (( eng_min != 10#$min )); then
        if (( eng_min > 10#$min )); then cmp=1; else cmp=-1; fi
    elif (( eng_pat != 10#$pat )); then
        if (( eng_pat > 10#$pat )); then cmp=1; else cmp=-1; fi
    fi
    local ver_ok=0
    case "$op" in
        ">=") if (( cmp >= 0 )); then ver_ok=1; fi ;;
        ">")  if (( cmp > 0 )); then ver_ok=1; fi ;;
        "<=") if (( cmp <= 0 )); then ver_ok=1; fi ;;
        "<")  if (( cmp < 0 )); then ver_ok=1; fi ;;
        "^"|"~"|"~=") if (( 10#$maj == eng_maj && cmp >= 0 )); then ver_ok=1; fi ;;
        *)    if (( cmp == 0 )); then ver_ok=1; fi ;;
    esac
    if [[ "$ver_ok" -ne 1 ]]; then
        echo "[Flint] ERROR: flint_version mismatch: project requires '${pin}', incompatible with Flint Engine ${eng_maj}.${eng_min}.${eng_pat}+${engine_commit}." >&2
        return 1
    fi
    commit_part="${commit_part#"${commit_part%%[![:space:]]*}"}"
    commit_part="${commit_part%"${commit_part##*[![:space:]]}"}"
    if [[ -n "$commit_part" && "$commit_part" != "dirty" && "$engine_commit" != "dev" && "$commit_part" != "$engine_commit" ]]; then
        echo "[Flint] ERROR: flint_version commit mismatch: project pins '${pin}', engine commit is '${engine_commit}'." >&2
        return 1
    fi
    return 0
}
_check_flint_version "$TOML_FLINT_VER" || exit 1

BUILD_FLAGS=("-O3")
[[ "$TOML_DET_FP" != "false" ]] && BUILD_FLAGS+=("--fp-mode" "contract=off")
[[ "$TOML_WERROR" != "false" && "${FLINT_WERROR:-1}" != "0" ]] && BUILD_FLAGS+=("--Werror")

BIN="${APP_ROOT}/build/${PROJ_NAME}"
"$MOJO_BIN" build "${APP_ROOT}/${PROJ_ENTRY}" -o "$BIN" \
    -I "${FLINT_ROOT}/src" -I "${APP_ROOT}/src" \
    "${BUILD_FLAGS[@]}"
cd "$APP_ROOT"
exec "$BIN" "$@"
EOF
    chmod +x "${abs_dest}/tools/run.sh"

    sed "s/__FLINT_APP_NAME__/${safe_app_name}/g" > "${abs_dest}/src/main.mojo" <<'EOF'
"""Flint Engine v5 game: __FLINT_APP_NAME__."""
from std.ffi import external_call
from std.math import cos, max, min, sin
from std.sys import argv
from flint.app import (
    AppConfig,
    EngineApp,
    FrameTime,
    Resource,
    Resources,
    STAGE_LAST,
    STAGE_UPDATE,
)
from flint.app.window_plugin import WindowPlugin, WindowStateResource
from flint.assets.image import write_png
from flint.assets.server import AssetServer
from flint.core.build_info import check_flint_version_pin
from flint.core.jobs import JobPool
from flint.core.log import flush_log, start_log
from flint.core.math import Quat, Vec3
from flint.ecs import Entity, ParallelCommands, World, comp_hash64
from flint.input.input_map import ACTION_BOOL, InputContext, InputMap
from flint.input.plugin import InputPlugin
from flint.platform.kms import on_virtual_console
from flint.platform.os import (
    env_flag,
    env_value,
    file_size,
    get_time,
    is_directory,
    make_dirs,
    parent_directory,
    read_bytes,
)
from flint.render.plugin import (
    CameraComponent,
    EnvironmentComponent,
    MaterialOverrideComponent,
    MeshRendererComponent,
    PointLightComponent,
    register_render_codecs,
)
from flint.render.simd_canvas import Camera3D, SimdCanvas, pack_argb
from flint.scene import (
    ComponentRegistry,
    GlobalTransform,
    LocalTransform,
    Scene,
    TransformPlugin,
)


struct GameConfig(Copyable, Movable):
    var app_name: String
    var entry_scene: String
    var flint_version: String
    var window_title: String
    var window_width: Int
    var window_height: Int
    var vsync: Bool
    var fixed_hz: Int
    var net_tick_hz: Int
    var net_max_rollback_ticks: Int
    var net_mtu_bytes: Int
    var pack_output: String

    def __init__(out self):
        self.app_name = "__FLINT_APP_NAME__"
        self.entry_scene = "assets/main.flscn"
        self.flint_version = ""
        self.window_title = "__FLINT_APP_NAME__"
        self.window_width = 1280
        self.window_height = 720
        self.vsync = True
        self.fixed_hz = 60
        self.net_tick_hz = 60
        self.net_max_rollback_ticks = 8
        self.net_mtu_bytes = 1180
        self.pack_output = "build/assets.flpk"


def _trim_ascii(bytes: Span[UInt8, _], start: Int, end: Int) -> String:
    var s = start
    var e = end
    while s < e and (bytes[s] == 32 or bytes[s] == 9):
        s += 1
    while e > s and (bytes[e - 1] == 32 or bytes[e - 1] == 9 or bytes[e - 1] == 13):
        e -= 1
    var out = List[UInt8](capacity=max(0, e - s))
    for i in range(s, e):
        out.append(bytes[i])
    return String(unsafe_from_utf8=out)


def _unquote(text: String) -> String:
    var b = text.as_bytes()
    var n = len(b)
    if n >= 2 and ((b[0] == 34 and b[n - 1] == 34) or (b[0] == 39 and b[n - 1] == 39)):
        var out = List[UInt8](capacity=n - 2)
        for i in range(1, n - 1):
            out.append(b[i])
        return String(unsafe_from_utf8=out)
    return text


def _parse_int(text: String, fallback: Int) -> Int:
    var b = text.as_bytes()
    var n = len(b)
    if n == 0:
        return fallback
    var i = 0
    var sign = 1
    if b[0] == 45:
        sign = -1
        i = 1
    if i >= n:
        return fallback
    var val = 0
    while i < n:
        if b[i] < 48 or b[i] > 57:
            return fallback
        val = val * 10 + Int(b[i] - 48)
        i += 1
    return sign * val


def _load_flint_toml() -> GameConfig:
    var cfg = GameConfig()
    var raw = read_bytes("flint.toml")
    if len(raw) == 0:
        return cfg^
    var section = String("")
    var assets_pack_output = String("")
    var project_pak_path = String("")
    var assets_pak_path = String("")
    var line_start = 0
    var n = len(raw)
    for idx in range(n + 1):
        if idx < n and raw[idx] != 10:
            continue
        var line_end = idx
        var quote_char: UInt8 = 0
        var escaped = False
        for c in range(line_start, line_end):
            var ch = raw[c]
            if quote_char == 0:
                if ch == 34 or ch == 39:
                    quote_char = ch
                    escaped = False
                elif ch == 35:
                    line_end = c
                    break
            else:
                if quote_char == 34 and escaped:
                    escaped = False
                elif quote_char == 34 and ch == 92:
                    escaped = True
                elif ch == quote_char:
                    quote_char = 0
        var trimmed = _trim_ascii(raw, line_start, line_end)
        line_start = idx + 1
        var tb = trimmed.as_bytes()
        if len(tb) == 0:
            continue
        if len(tb) >= 2 and tb[0] == 91 and tb[len(tb) - 1] == 93:
            section = _trim_ascii(tb, 1, len(tb) - 1)
            continue
        var eq_pos = -1
        for c in range(len(tb)):
            if tb[c] == 61:
                eq_pos = c
                break
        if eq_pos <= 0:
            continue
        var key = _trim_ascii(tb, 0, eq_pos)
        var val = _unquote(_trim_ascii(tb, eq_pos + 1, len(tb)))
        if section == "project":
            if key == "name" and val.byte_length() > 0:
                cfg.app_name = val
            elif key == "entry_scene" and val.byte_length() > 0:
                cfg.entry_scene = val
            elif key == "pak_path" and val.byte_length() > 0 and project_pak_path.byte_length() == 0:
                project_pak_path = val
        elif section == "engine" and key == "flint_version":
            cfg.flint_version = val
        elif section == "window":
            if key == "title" and val.byte_length() > 0:
                cfg.window_title = val
            elif key == "width":
                cfg.window_width = max(64, _parse_int(val, cfg.window_width))
            elif key == "height":
                cfg.window_height = max(64, _parse_int(val, cfg.window_height))
            elif key == "vsync":
                cfg.vsync = val == "true"
        elif section == "physics" and key == "fixed_hz":
            cfg.fixed_hz = max(1, _parse_int(val, cfg.fixed_hz))
        elif section == "net":
            if key == "tick_hz":
                cfg.net_tick_hz = max(1, _parse_int(val, cfg.net_tick_hz))
            elif key == "max_rollback_ticks":
                cfg.net_max_rollback_ticks = max(0, _parse_int(val, cfg.net_max_rollback_ticks))
            elif key == "mtu_bytes":
                cfg.net_mtu_bytes = max(256, _parse_int(val, cfg.net_mtu_bytes))
        elif section == "assets":
            if key == "pack_output" and val.byte_length() > 0 and assets_pack_output.byte_length() == 0:
                assets_pack_output = val
            elif key == "pak_path" and val.byte_length() > 0 and assets_pak_path.byte_length() == 0:
                assets_pak_path = val
    if assets_pack_output.byte_length() > 0:
        cfg.pack_output = assets_pack_output
    elif project_pak_path.byte_length() > 0:
        cfg.pack_output = project_pak_path
    elif assets_pak_path.byte_length() > 0:
        cfg.pack_output = assets_pak_path
    return cfg^


struct GameStateResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.game.GameStateResource")
    var headless: Bool
    var cam_yaw: Float32
    var cam_dist: Float32
    var camera_orbited: Bool
    var canvas: SimdCanvas

    def __init__(out self, headless: Bool, width: Int, height: Int):
        self.headless = headless
        self.cam_yaw = Float32(0.785398)
        self.cam_dist = Float32(9.2)
        self.camera_orbited = False
        self.canvas = SimdCanvas(width, height)


def _draw_box_3d(
    mut canvas: SimdCanvas,
    cam: Camera3D,
    center: Vec3,
    rot: Quat,
    scale: Vec3,
    color: UInt32,
    sun_dir: Vec3,
):
    var hx = scale.x * Float32(0.5)
    var hy = scale.y * Float32(0.5)
    var hz = scale.z * Float32(0.5)
    if scale.x >= Float32(4.0) or scale.z >= Float32(4.0):
        var nx = 6
        var nz = 6
        var dx = scale.x / Float32(nx)
        var dz = scale.z / Float32(nz)
        for ix in range(nx):
            var x0 = -hx + Float32(ix) * dx
            var x1 = x0 + dx
            for iz in range(nz):
                var z0 = -hz + Float32(iz) * dz
                var z1 = z0 + dz
                var q0 = center + rot.rotate(Vec3(x0, hy, z0))
                var q1 = center + rot.rotate(Vec3(x1, hy, z0))
                var q2 = center + rot.rotate(Vec3(x1, hy, z1))
                var q3 = center + rot.rotate(Vec3(x0, hy, z1))
                canvas.draw_tri_3d_lit(cam, q0, q1, q2, color, sun_dir)
                canvas.draw_tri_3d_lit(cam, q0, q2, q3, color, sun_dir)
        return

    var c0 = center + rot.rotate(Vec3(-hx, -hy, -hz))
    var c1 = center + rot.rotate(Vec3(hx, -hy, -hz))
    var c2 = center + rot.rotate(Vec3(hx, hy, -hz))
    var c3 = center + rot.rotate(Vec3(-hx, hy, -hz))
    var c4 = center + rot.rotate(Vec3(-hx, -hy, hz))
    var c5 = center + rot.rotate(Vec3(hx, -hy, hz))
    var c6 = center + rot.rotate(Vec3(hx, hy, hz))
    var c7 = center + rot.rotate(Vec3(-hx, hy, hz))
    canvas.draw_tri_3d_lit(cam, c0, c1, c2, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c0, c2, c3, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c5, c4, c7, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c5, c7, c6, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c4, c0, c3, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c4, c3, c7, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c1, c5, c6, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c1, c6, c2, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c3, c2, c6, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c3, c6, c7, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c4, c5, c1, color, sun_dir)
    canvas.draw_tri_3d_lit(cam, c4, c1, c0, color, sun_dir)


def _render_scene_simd(mut world: World, mut state: GameStateResource):
    var cam = Camera3D(
        screen_w=state.canvas.width,
        screen_h=state.canvas.height,
        eye=Vec3(
            cos(state.cam_yaw) * state.cam_dist,
            Float32(3.6),
            sin(state.cam_yaw) * state.cam_dist,
        ),
        target=Vec3(0.0, 0.8, 0.0),
        fov_deg=Float32(55.0),
    )
    if not state.camera_orbited:
        var cam_ents = world.query_entities(world.bit_for[CameraComponent]())
        for i in range(len(cam_ents)):
            var ce = cam_ents[i]
            var cc_opt = world.get_ptr[CameraComponent](ce)
            var lt_opt = world.get_ptr[LocalTransform](ce)
            if cc_opt and lt_opt and cc_opt.value()[unsafe_offset=0].active:
                var eye = lt_opt.value()[unsafe_offset=0].translation
                cam.eye = eye
                cam.target = eye + cc_opt.value()[unsafe_offset=0].look_target_offset
                cam.fov_rad = cc_opt.value()[unsafe_offset=0].fov_y_deg * Float32(0.0174532925)
                break
    cam.update_basis()

    var sun_dir = Vec3(0.38, 0.80, -0.46)
    var env_ents = world.query_entities(world.bit_for[EnvironmentComponent]())
    for i in range(len(env_ents)):
        var eo = world.get_ptr[EnvironmentComponent](env_ents[i])
        if eo and eo.value()[unsafe_offset=0].enabled:
            sun_dir = eo.value()[unsafe_offset=0].sun_dir
            break

    state.canvas.clear(0x0D1726, 0x1B2B44)
    var mesh_mask = world.bit_for[LocalTransform]().union(world.bit_for[MeshRendererComponent]())
    var mesh_ents = world.query_entities(mesh_mask)
    for i in range(len(mesh_ents)):
        var me = mesh_ents[i]
        var mr_opt = world.get_ptr[MeshRendererComponent](me)
        var lt_opt = world.get_ptr[LocalTransform](me)
        if not mr_opt or not lt_opt or not mr_opt.value()[unsafe_offset=0].visible:
            continue
        var col = pack_argb(180, 195, 210)
        var mo_opt = world.get_ptr[MaterialOverrideComponent](me)
        if mo_opt and mo_opt.value()[unsafe_offset=0].enabled:
            var bc = mo_opt.value()[unsafe_offset=0].base_color
            col = pack_argb(
                Int(max(Float32(0.0), min(Float32(255.0), bc.x * Float32(255.0)))),
                Int(max(Float32(0.0), min(Float32(255.0), bc.y * Float32(255.0)))),
                Int(max(Float32(0.0), min(Float32(255.0), bc.z * Float32(255.0)))),
            )
        var gt_opt = world.get_ptr[GlobalTransform](me)
        var pos = gt_opt.value()[unsafe_offset=0].translation if gt_opt else lt_opt.value()[unsafe_offset=0].translation
        var rot = gt_opt.value()[unsafe_offset=0].rotation if gt_opt else lt_opt.value()[unsafe_offset=0].rotation
        var scl = gt_opt.value()[unsafe_offset=0].scale if gt_opt else lt_opt.value()[unsafe_offset=0].scale
        _draw_box_3d(state.canvas, cam, pos, rot, scl, col, sun_dir)


def _sys_game_update(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[GameStateResource]()
    if not st_opt:
        return
    ref state = st_opt.value()[unsafe_offset=0]

    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        var fdt = ft_opt.value()[unsafe_offset=0].delta_seconds
        if fdt > Float32(0.0):
            dt = fdt

    var win_opt = resources.get_ptr_mut[WindowStateResource]()
    if win_opt:
        ref win_res = win_opt.value()[unsafe_offset=0]
        if win_res.resized and win_res.width > 0 and win_res.height > 0:
            state.canvas.resize(win_res.width, win_res.height)

    var imap_opt = resources.get_ptr[InputMap]()
    if not state.headless and imap_opt:
        ref imap = imap_opt.value()[unsafe_offset=0]
        if imap.just_pressed("quit"):
            if win_opt:
                win_opt.value()[unsafe_offset=0].request_close()
            return
        if imap.is_down("orbit_left"):
            state.cam_yaw -= Float32(1.5) * dt
            state.camera_orbited = True
        if imap.is_down("orbit_right"):
            state.cam_yaw += Float32(1.5) * dt
            state.camera_orbited = True

    _render_scene_simd(world, state)


def _sys_game_present(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[GameStateResource]()
    var win_opt = resources.get_ptr_mut[WindowStateResource]()
    if not st_opt or not win_opt:
        return
    if st_opt.value()[unsafe_offset=0].headless:
        return
    ref state = st_opt.value()[unsafe_offset=0]
    ref win_res = win_opt.value()[unsafe_offset=0]
    if len(win_res.windows) > 0:
        win_res.windows[0].present(state.canvas.pixels, state.canvas.width, state.canvas.height)


def _setup_input(mut app: EngineApp):
    var imap_opt = app.resources.get_ptr_mut[InputMap]()
    if not imap_opt:
        return
    ref imap = imap_opt.value()[unsafe_offset=0]
    _ = imap.register_action("orbit_left", ACTION_BOOL)
    _ = imap.register_action("orbit_right", ACTION_BOOL)
    _ = imap.register_action("quit", ACTION_BOOL)
    var ctx = InputContext("gameplay", priority=10)
    ctx.bind_key("orbit_left", 65)    # A
    ctx.bind_key("orbit_left", 263)   # Left
    ctx.bind_key("orbit_right", 68)   # D
    ctx.bind_key("orbit_right", 262)  # Right
    ctx.bind_key("quit", 256)         # Esc
    imap.add_context(ctx^)
    imap.push_context("gameplay")


def _looks_like_json_scene(raw: List[UInt8]) -> Bool:
    for i in range(len(raw)):
        var b = raw[i]
        if b == 32 or b == 9 or b == 10 or b == 13:
            continue
        return b == 123  # '{'
    return False


def _run_game() -> Int:
    var cfg = _load_flint_toml()
    var ver_check = check_flint_version_pin(cfg.flint_version)
    if not ver_check[0]:
        print("[" + cfg.app_name + "] ERROR: " + ver_check[1])
        return 1

    _ = start_log(cfg.app_name)
    var headless_smoke = False
    var shots_mode = False
    var out_png = String("")

    var args = argv()
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--smoke" or a == "--headless-smoke":
            headless_smoke = True
        elif a == "--shots":
            shots_mode = True
        elif a == "--out" and ai + 1 < len(args):
            out_png = String(args[ai + 1])
            ai += 1
        ai += 1

    var no_display = (
        env_flag("FLINT_HEADLESS")
        or (
            env_value("FLINT_BACKEND").byte_length() == 0
            and env_value("WAYLAND_DISPLAY").byte_length() == 0
            and env_value("DISPLAY").byte_length() == 0
            and not on_virtual_console()
        )
    )
    var effective_headless = headless_smoke or shots_mode or no_display

    var assets = AssetServer(enable_watcher=not effective_headless)
    if file_size(cfg.pack_output) > 0:
        _ = assets.mount_flpk(cfg.pack_output)
    elif file_size("assets.flpk") > 0:
        _ = assets.mount_flpk("assets.flpk")
    if is_directory("assets", follow_symlinks=True):
        assets.loose_root = "assets"
        assets.index_loose_directory()

    var app_cfg = AppConfig(
        title=cfg.window_title,
        headless=effective_headless,
        fixed_hz=Float32(cfg.fixed_hz),
        enable_crash_handler=True,
    )
    var app = EngineApp(app_cfg^)

    app.add_plugin(TransformPlugin(snapshot_prev_in_fixed=True))
    app.add_plugin(
        WindowPlugin(
            headless=effective_headless,
            width=cfg.window_width,
            height=cfg.window_height,
            title=cfg.window_title,
        )
    )
    app.add_plugin(InputPlugin(enable_evdev=not effective_headless))

    app.insert_resource[GameStateResource](
        GameStateResource(effective_headless, cfg.window_width, cfg.window_height)
    )
    _ = app.add_system(STAGE_UPDATE, "game.update", _sys_game_update, exclusive=True)
    _ = app.add_system(STAGE_LAST, "game.present", _sys_game_present, exclusive=True)

    _setup_input(app)

    var reg = ComponentRegistry.with_builtins()
    register_render_codecs(reg)
    reg.ensure_registered_in_world(app.world)

    var scene_bytes = List[UInt8]()
    var rh = assets.load_raw(cfg.entry_scene)
    var rptr = assets.raws.get_ptr(rh)
    if rptr and len(rptr.value()[unsafe_offset=0].bytes) > 0:
        scene_bytes = rptr.value()[unsafe_offset=0].bytes.copy()
    elif file_size(cfg.entry_scene) > 0:
        scene_bytes = read_bytes(cfg.entry_scene)

    if len(scene_bytes) == 0:
        print("[" + cfg.app_name + "] ERROR: entry_scene not found or empty: " + cfg.entry_scene)
        assets.close()
        _ = flush_log()
        app.shutdown()
        return 1

    var scene = Scene(cfg.entry_scene)
    try:
        if cfg.entry_scene.endswith(".flscnb") or not _looks_like_json_scene(scene_bytes):
            _ = scene.load_flscnb_bytes(app.world, reg, scene_bytes, clear_existing=True)
        else:
            _ = scene.load_flscn_bytes(app.world, reg, scene_bytes, clear_existing=True)
    except err:
        print("[" + cfg.app_name + "] ERROR: failed to load entry_scene '" + cfg.entry_scene + "': " + String(err))
        assets.close()
        _ = flush_log()
        app.shutdown()
        return 1

    if headless_smoke or shots_mode or no_display:
        app.startup()
        var steps = 4 if headless_smoke else 1
        for _ in range(steps):
            app.step_one_frame(Float32(1.0) / Float32(max(1, cfg.fixed_hz)))

        var shot_target = out_png
        if shots_mode and shot_target.byte_length() == 0:
            shot_target = "build/shots/" + cfg.app_name + ".png"
        if shot_target.byte_length() > 0:
            var pdir = parent_directory(shot_target)
            if pdir.byte_length() > 0 and pdir != ".":
                _ = make_dirs(pdir)
            var st_opt = app.resources.get_ptr_mut[GameStateResource]()
            if st_opt:
                ref st = st_opt.value()[unsafe_offset=0]
                _ = write_png(shot_target, st.canvas.pixels, st.canvas.width, st.canvas.height)

        print(
            "["
            + cfg.app_name
            + "] Headless v5 frame loop OK ("
            + String(cfg.window_width)
            + "x"
            + String(cfg.window_height)
            + ", fixed_hz="
            + String(cfg.fixed_hz)
            + ", net_mtu="
            + String(cfg.net_mtu_bytes)
            + ", entities="
            + String(len(app.world.entities))
            + ")."
        )
        assets.close()
        _ = flush_log()
        app.shutdown()
        return 0

    app.startup()
    var win_open = False
    var win_opt = app.resources.get_ptr[WindowStateResource]()
    if win_opt:
        win_open = win_opt.value()[unsafe_offset=0].is_open()

    if not win_open:
        print("[" + cfg.app_name + "] No display available; exiting cleanly. Use '--headless-smoke' or '--shots'.")
        assets.close()
        _ = flush_log()
        app.shutdown()
        return 0

    var last_t = get_time()
    while True:
        var open_now = False
        var wo = app.resources.get_ptr[WindowStateResource]()
        if wo:
            open_now = wo.value()[unsafe_offset=0].is_open()
        if not open_now:
            break
        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - last_t)))
        last_t = now
        app.step_one_frame(dt)

    assets.close()
    _ = flush_log()
    app.shutdown()
    return 0


def main():
    var rc = _run_game()
    if rc != 0:
        external_call["exit", NoneType](Int32(rc))
EOF
    echo "[Flint] Scaffolded new standalone Flint v5 project (with flint.toml) at: ${abs_dest}"
}

cmd_gameplay_so() {
    local opt="-O1"
    local out_override=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --fast|--debug)
                opt="-O0"
                shift
                ;;
            --release)
                opt="-O3"
                shift
                ;;
            --out)
                if [[ $# -lt 2 || -z "${2:-}" ]]; then
                    echo "[Flint] ERROR: --out requires a file path." >&2
                    return 1
                fi
                out_override="$2"
                shift 2
                ;;
            --out=*)
                out_override="${1#--out=}"
                if [[ -z "$out_override" ]]; then
                    echo "[Flint] ERROR: --out requires a non-empty file path." >&2
                    return 1
                fi
                shift
                ;;
            *)
                echo "[Flint] ERROR: Unknown gameplay-so option: $1" >&2
                return 1
                ;;
        esac
    done

    local toml_file="${CALLER_PWD}/flint.toml"
    if [[ ! -f "$toml_file" ]]; then
        echo "[Flint] ERROR: No flint.toml found in ${CALLER_PWD}. Run from a Flint game project directory." >&2
        return 1
    fi
    validate_project_flint_version "$toml_file" || return 1

    local gp_entry gp_out
    gp_entry="$(read_flint_toml_value "$toml_file" "project" "gameplay_entry" || true)"
    gp_out="$(read_flint_toml_value "$toml_file" "project" "gameplay_output" || true)"
    [[ -z "$gp_entry" ]] && gp_entry="src/gameplay.mojo"
    [[ -z "$gp_out" ]] && gp_out="build/gameplay.so"
    if [[ -n "$out_override" ]]; then
        gp_out="$out_override"
    fi

    local src_path out_so
    if [[ "$gp_entry" == /* ]]; then
        src_path="$gp_entry"
    else
        src_path="${CALLER_PWD}/${gp_entry}"
    fi
    if [[ "$gp_out" == /* ]]; then
        out_so="$gp_out"
    else
        out_so="${CALLER_PWD}/${gp_out}"
    fi

    if [[ ! -f "$src_path" ]]; then
        echo "[Flint] ERROR: Gameplay plugin entry not found: ${src_path}" >&2
        return 1
    fi

    local toml_werror toml_det_fp eff_werror eff_det_fp
    toml_werror="$(read_flint_toml_value "$toml_file" "engine" "werror" || echo "true")"
    toml_det_fp="$(read_flint_toml_value "$toml_file" "engine" "deterministic_fp" || echo "true")"
    eff_werror="${FLINT_WERROR:-1}"
    eff_det_fp="${FLINT_DETERMINISTIC_FP:-1}"
    [[ "$toml_werror" == "false" ]] && eff_werror="0"
    [[ "$toml_det_fp" == "false" ]] && eff_det_fp="0"

    local fp_args=()
    if [[ "$eff_det_fp" != "0" ]]; then
        fp_args=("--fp-mode" "contract=off")
    fi
    local werror_args=()
    if [[ "$eff_werror" != "0" ]]; then
        werror_args=("--Werror")
    fi
    local inc_args=("-I" "${ROOT}/src")
    if [[ -d "${CALLER_PWD}/src" ]]; then
        inc_args+=("-I" "${CALLER_PWD}/src")
    fi

    resolve_mojo
    write_build_stamp "$MOJO_BIN"

    local out_dir tmp_base tmp_so
    out_dir="$(dirname "$out_so")"
    mkdir -p "$out_dir"
    tmp_base="$(mktemp "${out_dir}/.$(basename "$out_so").tmp.XXXXXX")"
    tmp_so="${tmp_base}.so"
    rm -f "$tmp_base"

    echo "[Flint] Building gameplay plugin ${gp_entry} -> ${out_so} (${opt}) ..."
    if ! "$MOJO_BIN" build "$src_path" --emit shared-lib -o "$tmp_so" "${inc_args[@]}" "$opt" "${fp_args[@]}" "${werror_args[@]}"; then
        rm -f "$tmp_so"
        echo "[Flint] ERROR: Failed to build gameplay plugin shared library." >&2
        return 1
    fi

    chmod 0755 "$tmp_so"
    mv -f "$tmp_so" "$out_so"
    echo "[Flint] Gameplay plugin built: ${out_so}"
}

subcmd="${1:-help}"
shift || true
case "$subcmd" in
    sync)    cmd_sync "$@" ;;
    build)   cmd_build "$@" ;;
    run)     cmd_run "$@" ;;
    gameplay-so) cmd_gameplay_so "$@" ;;
    fern)
        resolve_mojo
        compile_mojo "${ROOT}/examples/fern/main.mojo" "${ROOT}/build/flint-fern" "-O3" ""
        (cd "$CALLER_PWD" && "${ROOT}/build/flint-fern" "$@")
        ;;
    pack)    cmd_pack "$@" ;;
    package) cmd_package "$@" ;;
    test)    cmd_test "$@" ;;
    lab)     cmd_lab "$@" ;;
    studio)  cmd_studio "$@" ;;
    mcp)     cmd_mcp "$@" ;;
    doctor)  cmd_doctor "$@" ;;
    play)    cmd_play "$@" ;;
    world)   cmd_world "$@" ;;
    kms)     cmd_kms "$@" ;;
    kms3d)   cmd_kms3d "$@" ;;
    preview) cmd_preview "$@" ;;
    cycles)  cmd_cycles "$@" ;;
    shot)    cmd_shot "$@" ;;
    gate)    cmd_gate "$@" ;;
    new)     cmd_new "$@" ;;
    *)
        echo "Flint Engine v5 CLI (Pure Mojo 1.2)"
        echo "Usage: bash tools/flint.sh <command>"
        echo ""
        echo "Commands:"
        echo "  doctor               Check Mojo 1.2 toolchain, Vulkan ICD, Wayland/X11/KMS, Audio, evdev & inotify"
        echo "  lab <name|all> [--fast] Compile & run subsystem lab (physics|anim|render|audio_ui|net|all)"
        echo "  studio --takes [take or recipe]     Preview and author motion, capture video and export BVH/GLB"
        echo "  studio --kimodo [PROMPT OUTPUT.bvh [FRAMES STEPS SEED]]   Generate SOMA motion"
        echo "  studio --compare A B [--text | --shot FILE]   Compare two frame captures"
        echo "  studio --capture TRACE | --build-profiles    Capture a running game or select a build"
        echo "  studio --moves                              Browse and rehearse project move presets"
        echo "  studio --ide-gui [FILE[:LINE[:COL]]]         Open the native TurboMojo desktop window"
        echo "  studio --ide [FILE[:LINE[:COL]]]             Open project source in TurboMojo"
        echo "  studio [...]         Launch the visual editor, or use --motion / --combat / --room PRESET.json"
        echo "  mcp [...]            Build and launch Flint Studio headless MCP v2 stdio server"
        echo "  build [target] [--force] [-j N] Compile standalone targets (or flint.toml project) with incremental skip"
        echo "  gameplay-so [--fast|--debug|--release] [--out path.so]  Compile hot-reloadable gameplay .so plugin"
        echo "  run [...]            Build and run a scaffolded game project using flint.toml [project] entry"
        echo "  test [unit|e2e|v5]   Build and run standalone Flint Engine verification suites"
        echo "  play                 Launch interactive 'minimal_spark' 2D+3D showcase"
        echo "  preview <program>    Live preview: rebuild on save and swap window (spark|world|studio|cycles|file|dir)"
        echo "  fern IMAGE PREFIX    Bake an upright frond into near, medium and far GLBs"
        echo "  world [--shots]      Launch 512m x 512m AAA Vulkan Open-World sandbox"
        echo "  kms [--probe]        Run bare-display DRM/KMS demo from a text console (no X11/Wayland)"
        echo "  kms3d                GPU 3D on the bare display via VK_KHR_display"
        echo "  cycles live|render|query ... Pure-Mojo TidalCycles-style live coding (docs/CYCLES.md)"
        echo "  pack <out.flpk> <files...> Bundle loose assets into a single .flpk VFS archive"
        echo "  package [out_dir] [...]    Build a self-contained release bundle (bin/, lib/, assets.flpk, flint.toml, run.sh)"
        echo "  shot                 Render deterministic headless PNG to build/shots/flint_spark.png"
        echo "  sync [deepsong_dir]  Harvest latest pure-Mojo libraries from Deepsong & run Zero-Leak Gate"
        echo "  gate                 Run full standalone verification gate (sync + build + unit/e2e/v5 + shot)"
        echo "  new <dir>            Scaffold a standalone Flint v5 game project with flint.toml"
        ;;
esac
