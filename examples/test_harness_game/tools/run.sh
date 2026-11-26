#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLINT_ROOT="${FLINT_ROOT:-}"
if [[ -z "$FLINT_ROOT" ]]; then
    for cand in "${APP_ROOT}/../.." "${APP_ROOT}/../flint" "${HOME}/flint"; do
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
                } else if (q == sq) {
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
