#!/usr/bin/env bash
# AREA: r1
# Tier 2 Boundary & Negative Coverage: Area A — R1 Hygiene, Crash/Log Env & Cooker/Packer (F01-F05, Bug #6)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t2_r1"
resolve_mojo_env || {
    fail_test "t2_r1_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

# ---------------------------------------------------------------------------
# Case 1 (F01): FLINT_STATE_DIR / XDG_STATE_HOME honored; legacy SD_STATE_DIR ignored
# ---------------------------------------------------------------------------
test_t2_env_flint_state_dir_vs_legacy_sd() {
    local probe_src="${TMPD}/env_probe.mojo"
    local probe_bin="${TMPD}/bin/env_probe"
    cat >"$probe_src" <<'MOJO'
from flint.core.log import LOG_CORE, flush_log, log_info, start_log

def main() raises:
    if not start_log("t2_env_probe"):
        raise Error("start_log failed")
    log_info(LOG_CORE, "probe_ok")
    _ = flush_log()
MOJO
    if ! "$MOJO_BIN" build -I "${ROOT}/src" -O3 --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >/dev/null 2>&1; then
        fail_test "t2_r1_flint_state_dir_ignores_legacy_sd" "failed to compile env_probe.mojo"
        return
    fi

    local bad_sd="${TMPD}/legacy_sd"
    local good_xdg="${TMPD}/good_xdg"
    if ! env -u FLINT_STATE_DIR SD_STATE_DIR="$bad_sd" XDG_STATE_HOME="$good_xdg" "$probe_bin" >/dev/null 2>&1; then
        fail_test "t2_r1_flint_state_dir_ignores_legacy_sd" "env_probe failed under XDG_STATE_HOME"
        return
    fi
    if [[ -e "$bad_sd" ]]; then
        fail_test "t2_r1_flint_state_dir_ignores_legacy_sd" "legacy SD_STATE_DIR was unexpectedly created"
        return
    fi
    if [[ ! -s "${good_xdg}/t2_env_probe/logs/t2_env_probe.log" ]]; then
        fail_test "t2_r1_flint_state_dir_ignores_legacy_sd" "missing log at ${good_xdg}/t2_env_probe/logs/t2_env_probe.log"
        return
    fi

    local flint_sd="${TMPD}/explicit_flint_sd"
    if ! FLINT_STATE_DIR="$flint_sd" SD_STATE_DIR="$bad_sd" XDG_STATE_HOME="$good_xdg" "$probe_bin" >/dev/null 2>&1; then
        fail_test "t2_r1_flint_state_dir_ignores_legacy_sd" "env_probe failed under FLINT_STATE_DIR"
        return
    fi
    if [[ ! -s "${flint_sd}/logs/t2_env_probe.log" ]]; then
        fail_test "t2_r1_flint_state_dir_ignores_legacy_sd" "FLINT_STATE_DIR not honored"
        return
    fi

    pass_test "t2_r1_flint_state_dir_ignores_legacy_sd"
}

# ---------------------------------------------------------------------------
# Case 2 (F03 boundary): flint.sh new missing arg, file collision, read-only dir
# ---------------------------------------------------------------------------
test_t2_new_negative_boundaries() {
    local out rc=0
    out="$(bash "${ROOT}/tools/flint.sh" new 2>&1)" || rc=$?
    if [[ "$rc" -eq 0 ]] || ! grep -q "Usage: tools/flint.sh new" <<<"$out"; then
        fail_test "t2_r1_new_missing_arg_file_collision_and_readonly" "missing-arg should exit non-zero with Usage; rc=$rc out=$out"
        return
    fi

    local file_col="${TMPD}/existing_file"
    printf 'occupied\n' > "$file_col"
    rc=0
    bash "${ROOT}/tools/flint.sh" new "$file_col" >/dev/null 2>&1 || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        fail_test "t2_r1_new_missing_arg_file_collision_and_readonly" "file collision target unexpectedly succeeded"
        return
    fi

    if [[ "$(id -u)" -ne 0 ]]; then
        local ro_parent="${TMPD}/ro_parent"
        mkdir -p "$ro_parent"
        chmod 0555 "$ro_parent"
        rc=0
        bash "${ROOT}/tools/flint.sh" new "${ro_parent}/child_proj" >/dev/null 2>&1 || rc=$?
        chmod 0755 "$ro_parent"
        if [[ "$rc" -eq 0 || -e "${ro_parent}/child_proj" ]]; then
            fail_test "t2_r1_new_missing_arg_file_collision_and_readonly" "read-only parent allowed creation or exited 0 (rc=$rc)"
            return
        fi
    fi

    pass_test "t2_r1_new_missing_arg_file_collision_and_readonly"
}

# ---------------------------------------------------------------------------
# Case 3 (F05 boundary): flint-pack --cook empty, nonexistent, unsupported-only, 0-byte mixed
# ---------------------------------------------------------------------------
test_t2_pack_cook_empty_nonexistent_unsupported_and_zero_byte() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t2_r1_pack_cook_empty_nonexistent_unsupported_and_zero_byte" "could not resolve flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"

    # Empty dir
    local empty_ws="${TMPD}/empty_ws"
    mkdir -p "$empty_ws"
    local rc=0
    "$pack_bin" --cook "$empty_ws" "${TMPD}/empty.flpk" >/dev/null 2>&1 || rc=$?
    if [[ "$rc" -eq 0 || -e "${TMPD}/empty.flpk" ]]; then
        fail_test "t2_r1_pack_cook_empty_nonexistent_unsupported_and_zero_byte" "empty dir exited $rc or created flpk"
        return
    fi

    # Nonexistent dir
    rc=0
    "$pack_bin" --cook "${TMPD}/no_such_dir" "${TMPD}/missing.flpk" >/dev/null 2>&1 || rc=$?
    if [[ "$rc" -eq 0 || -e "${TMPD}/missing.flpk" ]]; then
        fail_test "t2_r1_pack_cook_empty_nonexistent_unsupported_and_zero_byte" "nonexistent dir exited $rc or created flpk"
        return
    fi

    # Unsupported + .flmeta only
    local unsup_ws="${TMPD}/unsup_ws"
    mkdir -p "$unsup_ws"
    printf 'notes\n' > "${unsup_ws}/notes.txt"
    printf '{}\n' > "${unsup_ws}/ghost.png.flmeta"
    rc=0
    "$pack_bin" --cook "$unsup_ws" "${TMPD}/unsup.flpk" >/dev/null 2>&1 || rc=$?
    if [[ "$rc" -eq 0 || -e "${TMPD}/unsup.flpk" ]]; then
        fail_test "t2_r1_pack_cook_empty_nonexistent_unsupported_and_zero_byte" "unsupported-only dir exited $rc or created flpk"
        return
    fi

    # Mixed workspace: 0-byte .flscn + unsupported .tmp + valid .json
    local mixed_ws="${TMPD}/mixed_ws"
    mkdir -p "$mixed_ws"
    : > "${mixed_ws}/zero.flscn"
    printf 'ignore\n' > "${mixed_ws}/scratch.tmp"
    printf '{"valid":true}\n' > "${mixed_ws}/valid.json"
    local mixed_flpk="${TMPD}/mixed.flpk"
    if ! "$pack_bin" --cook "$mixed_ws" "$mixed_flpk" --cache-dir "${TMPD}/mixed_cache" >/dev/null 2>&1; then
        fail_test "t2_r1_pack_cook_empty_nonexistent_unsupported_and_zero_byte" "mixed workspace cook with --cache-dir failed"
        return
    fi
    if ! validate_flpk_v3 "$mixed_flpk" "valid.json" "zero.flscn"; then
        fail_test "t2_r1_pack_cook_empty_nonexistent_unsupported_and_zero_byte" "mixed.flpk missing valid.json or zero.flscn"
        return
    fi
    if "$pack_bin" --inspect "$mixed_flpk" | grep -Fq "scratch.tmp"; then
        fail_test "t2_r1_pack_cook_empty_nonexistent_unsupported_and_zero_byte" "mixed.flpk unexpectedly included scratch.tmp"
        return
    fi

    pass_test "t2_r1_pack_cook_empty_nonexistent_unsupported_and_zero_byte"
}

# ---------------------------------------------------------------------------
# Case 4 (F05 negative): symlink source_dir, symlink output_pack, dir output_pack rejected
# ---------------------------------------------------------------------------
test_t2_pack_cook_symlink_guards() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t2_r1_pack_cook_symlink_and_bad_output_rejected" "could not resolve flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"

    local real_ws="${TMPD}/real_ws"
    mkdir -p "$real_ws"
    printf '{"x":1}\n' > "${real_ws}/a.json"

    ln -s "$real_ws" "${TMPD}/sym_ws"
    local rc=0
    "$pack_bin" --cook "${TMPD}/sym_ws" "${TMPD}/sym_src.flpk" >/dev/null 2>&1 || rc=$?
    if [[ "$rc" -eq 0 || -e "${TMPD}/sym_src.flpk" ]]; then
        fail_test "t2_r1_pack_cook_symlink_and_bad_output_rejected" "symlink source_dir was not rejected"
        return
    fi

    printf 'target\n' > "${TMPD}/sym_target"
    ln -s "${TMPD}/sym_target" "${TMPD}/sym_out.flpk"
    rc=0
    "$pack_bin" --cook "$real_ws" "${TMPD}/sym_out.flpk" >/dev/null 2>&1 || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        fail_test "t2_r1_pack_cook_symlink_and_bad_output_rejected" "symlink output_pack was not rejected"
        return
    fi

    mkdir -p "${TMPD}/dir_out.flpk"
    rc=0
    "$pack_bin" --cook "$real_ws" "${TMPD}/dir_out.flpk" >/dev/null 2>&1 || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        fail_test "t2_r1_pack_cook_symlink_and_bad_output_rejected" "directory output_pack was not rejected"
        return
    fi

    pass_test "t2_r1_pack_cook_symlink_and_bad_output_rejected"
}

# ---------------------------------------------------------------------------
# Case 5 (F04/F05 negative): bad CLI flags, missing --cache-dir value, truncated --inspect
# ---------------------------------------------------------------------------
test_t2_pack_bad_cli_and_truncated_inspect() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t2_r1_pack_cli_bad_args_and_truncated_inspect" "could not resolve flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"

    local ws="${TMPD}/cli_ws"
    mkdir -p "$ws"
    printf '{"ok":1}\n' > "${ws}/a.json"

    local rc=0
    "$pack_bin" --bogus-flag >/dev/null 2>&1 || rc=$?
    [[ "$rc" -ne 0 ]] || { fail_test "t2_r1_pack_cli_bad_args_and_truncated_inspect" "--bogus-flag exited 0"; return; }

    rc=0
    "$pack_bin" --cook "$ws" >/dev/null 2>&1 || rc=$?
    [[ "$rc" -ne 0 ]] || { fail_test "t2_r1_pack_cli_bad_args_and_truncated_inspect" "--cook missing output arg exited 0"; return; }

    rc=0
    "$pack_bin" --cook "$ws" "${TMPD}/bad_cd.flpk" --cache-dir >/dev/null 2>&1 || rc=$?
    [[ "$rc" -ne 0 ]] || { fail_test "t2_r1_pack_cli_bad_args_and_truncated_inspect" "--cache-dir missing value exited 0"; return; }

    rc=0
    "$pack_bin" --inspect "${TMPD}/nonexistent.flpk" >/dev/null 2>&1 || rc=$?
    [[ "$rc" -ne 0 ]] || { fail_test "t2_r1_pack_cli_bad_args_and_truncated_inspect" "--inspect missing file exited 0"; return; }

    python3 -c "open('${TMPD}/trunc.flpk', 'wb').write(b'FLPK' + b'\x00' * 12)"
    rc=0
    "$pack_bin" --inspect "${TMPD}/trunc.flpk" >/dev/null 2>&1 || rc=$?
    [[ "$rc" -ne 0 ]] || { fail_test "t2_r1_pack_cli_bad_args_and_truncated_inspect" "--inspect truncated file exited 0"; return; }

    pass_test "t2_r1_pack_cli_bad_args_and_truncated_inspect"
}

# ---------------------------------------------------------------------------
# Case 6 (F04/F05 boundary): same-size edit & delete cache invalidation
# ---------------------------------------------------------------------------
test_t2_cook_cache_edit_and_delete_invalidation() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t2_r1_cook_cache_invalidation_on_edit_and_delete" "could not resolve flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"

    local ws="${TMPD}/mut_ws"
    local cache="${TMPD}/mut_cache"
    local flpk="${TMPD}/mut.flpk"
    mkdir -p "$ws"
    printf '{"v":"AAAA"}\n' > "${ws}/a.json"
    printf '{"mat":"stone"}\n' > "${ws}/b.flmat"

    if ! "$pack_bin" --cook "$ws" "$flpk" --cache-dir "$cache" >/dev/null 2>&1; then
        fail_test "t2_r1_cook_cache_invalidation_on_edit_and_delete" "initial cook failed"
        return
    fi
    if ! validate_flpk_v3 "$flpk" "a.json" "b.flmat"; then
        fail_test "t2_r1_cook_cache_invalidation_on_edit_and_delete" "initial archive missing a.json or b.flmat"
        return
    fi

    # Same-size edit on a.json (with sleep 1 for coarse mtime safety)
    sleep 1
    printf '{"v":"BBBB"}\n' > "${ws}/a.json"
    local recook_out
    if ! recook_out="$("$pack_bin" --cook "$ws" "$flpk" --cache-dir "$cache" 2>&1)"; then
        fail_test "t2_r1_cook_cache_invalidation_on_edit_and_delete" "re-cook after same-size edit failed: $recook_out"
        return
    fi
    if ! grep -qE 'cooked=1, cache_hits=1' <<<"$recook_out"; then
        fail_test "t2_r1_cook_cache_invalidation_on_edit_and_delete" "expected cooked=1, cache_hits=1 in stdout, got: $recook_out"
        return
    fi
    # Compression-agnostic FLPK v3 payload readback for a.json (handles comp=0 and comp=1 LZ4)
    if ! python3 - "$flpk" <<'PY'
import struct, sys

path = sys.argv[1]
with open(path, "rb") as f:
    data = f.read()

magic, ver, flags, count, toc_off, uuid_off, path_off, hcap, str_bytes, total_bytes, toc_crc64 = struct.unpack(
    "<4sIIIQQQIIQQ", data[:64]
)
str_off = path_off + hcap * 16

def lz4_decompress_block(src: bytes, raw_len: int) -> bytes:
    out = bytearray()
    ip = 0
    n = len(src)
    while ip < n:
        token = src[ip]; ip += 1
        lit_len = token >> 4
        if lit_len == 15:
            while True:
                b = src[ip]; ip += 1
                lit_len += b
                if b != 255:
                    break
        out.extend(src[ip:ip + lit_len]); ip += lit_len
        if ip >= n:
            break
        off = src[ip] | (src[ip + 1] << 8); ip += 2
        if off == 0 or off > len(out):
            raise SystemExit(f"Invalid LZ4 offset {off}")
        match_len = (token & 0xF) + 4
        if (token & 0xF) == 15:
            while True:
                b = src[ip]; ip += 1
                match_len += b
                if b != 255:
                    break
        start = len(out) - off
        for i in range(match_len):
            out.append(out[start + i])
    if len(out) != raw_len:
        raise SystemExit(f"LZ4 decoded size {len(out)} != {raw_len}")
    return bytes(out)

found_payload = None
for i in range(count):
    ebase = toc_off + i * 64
    u_hi, u_lo, pfnv, p_off, c_bytes, u_bytes, pcrc, sub_id, kind, comp, p_str = struct.unpack(
        "<QQQQQQQHBBI", data[ebase:ebase + 64]
    )
    s_off = p_str & 0xFFFFF
    s_len = (p_str >> 20) & 0xFFF
    ep = data[str_off + s_off:str_off + s_off + s_len].decode("utf-8")
    if ep == "a.json":
        raw_chunk = data[p_off:p_off + c_bytes]
        if comp == 0:
            found_payload = raw_chunk
        elif comp == 1:
            found_payload = lz4_decompress_block(raw_chunk, u_bytes)
        else:
            raise SystemExit(f"Unknown compression {comp}")
        break

if found_payload is None:
    raise SystemExit("a.json not found in FLPK TOC")
if b"BBBB" not in found_payload or b"AAAA" in found_payload:
    raise SystemExit(f"Unexpected a.json payload: {found_payload!r}")
PY
    then
        fail_test "t2_r1_cook_cache_invalidation_on_edit_and_delete" "updated a.json payload check failed (BBBB missing or AAAA present)"
        return
    fi

    # Delete b.flmat and re-cook
    rm -f "${ws}/b.flmat"
    if ! "$pack_bin" --cook "$ws" "$flpk" --cache-dir "$cache" >/dev/null 2>&1; then
        fail_test "t2_r1_cook_cache_invalidation_on_edit_and_delete" "re-cook after delete failed"
        return
    fi
    if "$pack_bin" --inspect "$flpk" | grep -Fq "b.flmat"; then
        fail_test "t2_r1_cook_cache_invalidation_on_edit_and_delete" "deleted b.flmat still present in archive"
        return
    fi
    if ! validate_flpk_v3 "$flpk" "a.json"; then
        fail_test "t2_r1_cook_cache_invalidation_on_edit_and_delete" "a.json missing after b.flmat deletion"
        return
    fi

    pass_test "t2_r1_cook_cache_invalidation_on_edit_and_delete"
}

# ---------------------------------------------------------------------------
# Case 7 (F04 negative): flint.sh package bad --assets and bad --target
# ---------------------------------------------------------------------------
test_t2_package_bad_assets_and_target() {
    local proj="${TMPD}/bad_pkg_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t2_r1_package_bad_assets_and_bad_target" "flint.sh new failed"
        return
    fi

    local out rc=0
    out="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" package --assets "${TMPD}/missing_assets" --out "${TMPD}/bad_dist1" 2>&1)" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        fail_test "t2_r1_package_bad_assets_and_bad_target" "bad --assets unexpectedly exited 0"
        return
    fi

    rc=0
    out="$(bash "${ROOT}/tools/flint.sh" package --target nonexistent_target_xyz --out "${TMPD}/bad_dist2" 2>&1)" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        fail_test "t2_r1_package_bad_assets_and_bad_target" "bad --target unexpectedly exited 0"
        return
    fi

    pass_test "t2_r1_package_bad_assets_and_bad_target"
}

test_t2_env_flint_state_dir_vs_legacy_sd
test_t2_new_negative_boundaries
test_t2_pack_cook_empty_nonexistent_unsupported_and_zero_byte
test_t2_pack_cook_symlink_guards
test_t2_pack_bad_cli_and_truncated_inspect
test_t2_cook_cache_edit_and_delete_invalidation
test_t2_package_bad_assets_and_target

print_summary_and_exit
