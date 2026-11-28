#!/usr/bin/env bash
# AREA: m3
# Tier 2 Boundary & Negative Coverage: Area E — M3 Shipping, flint_version Pin, VFS Fallback/Corruption, MCP Sandbox & Bundle Isolation (F26-F28)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t2_m3"
resolve_mojo_env || {
    fail_test "t2_m3_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

# ---------------------------------------------------------------------------
# Case 1 (F26 negative): flint_version semver/pin mismatch exits non-zero cleanly
# ---------------------------------------------------------------------------
test_t2_m3_flint_version_mismatch_rejected() {
    local proj="${TMPD}/ver_mismatch_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t2_m3_f26_flint_version_mismatch_rejected" "flint.sh new failed"
        return
    fi

    # Replace flint_version with incompatible semver + commit pin
    python3 -B - "${proj}/flint.toml" <<'PY'
import re, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as fh:
    txt = fh.read()
if re.search(r"^\s*flint_version\s*=", txt, re.M):
    txt = re.sub(r"^\s*flint_version\s*=.*$", 'flint_version = "99.0.0+deadbeef"', txt, flags=re.M)
else:
    txt += '\n[engine]\nflint_version = "99.0.0+deadbeef"\n'
with open(p, "w", encoding="utf-8") as fh:
    fh.write(txt)
PY

    local run_out run_rc=0
    run_out="$(cd "$proj" && FLINT_ROOT="$ROOT" FLINT_STATE_DIR="${TMPD}/ver_state" XDG_STATE_HOME="${TMPD}/ver_xdg" bash tools/run.sh --headless-smoke 2>&1)" || run_rc=$?

    # Use lightweight main.mojo for the package sub-check to stay well within Tier 2 180s budget
    local pkg_proj="${TMPD}/ver_mismatch_pkg_proj"
    cp -a "$proj" "$pkg_proj"
    cat > "${pkg_proj}/src/main.mojo" <<'MOJO'
from flint.app import AppConfig, EngineApp
from flint.core.log import start_log

def main() raises:
    _ = start_log("ver_mismatch_pkg_proj")
    var app = EngineApp(AppConfig(title="ver_mismatch_pkg_proj", headless=True, enable_crash_handler=True))
    app.startup()
    app.shutdown()
MOJO
    local pkg_out pkg_rc=0
    pkg_out="$(cd "$pkg_proj" && bash "${ROOT}/tools/flint.sh" package --out "${TMPD}/ver_dist" 2>&1)" || pkg_rc=$?

    if [[ "$run_rc" -eq 0 ]]; then
        fail_test "t2_m3_f26_flint_version_mismatch_rejected" "M3 F26 contract pending: tools/run.sh accepted incompatible flint_version='99.0.0+deadbeef' (run_rc=0, pkg_rc=$pkg_rc)"
        return
    fi
    if [[ "$pkg_rc" -eq 0 ]]; then
        fail_test "t2_m3_f26_flint_version_mismatch_rejected" "M3 F26 contract pending: flint.sh package accepted incompatible flint_version='99.0.0+deadbeef' (run_rc=$run_rc, pkg_rc=0)"
        return
    fi

    # Ensure both failed with version/pin diagnostics rather than unrelated crashes (e.g. SIGSEGV 139)
    if [[ "$run_rc" -ge 128 || "$pkg_rc" -ge 128 ]]; then
        fail_test "t2_m3_f26_flint_version_mismatch_rejected" "terminated by signal (run_rc=$run_rc, pkg_rc=$pkg_rc)"
        return
    fi
    if ! grep -qiE "flint_version|version|commit|mismatch|incompatible" <<<"$run_out"; then
        fail_test "t2_m3_f26_flint_version_mismatch_rejected" "tools/run.sh rejected without clear version mismatch diagnostic: run='${run_out}'"
        return
    fi
    if ! grep -qiE "flint_version|version|commit|mismatch|incompatible" <<<"$pkg_out"; then
        fail_test "t2_m3_f26_flint_version_mismatch_rejected" "flint.sh package rejected without clear version mismatch diagnostic: pkg='${pkg_out}'"
        return
    fi

    pass_test "t2_m3_f26_flint_version_mismatch_rejected"
}

# ---------------------------------------------------------------------------
# Case 2 (F26 negative): Missing entry_scene exits non-zero cleanly (no segfault)
# ---------------------------------------------------------------------------
test_t2_m3_missing_entry_scene_graceful_error() {
    local proj="${TMPD}/missing_scene_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t2_m3_f26_missing_entry_scene_graceful_error" "flint.sh new failed"
        return
    fi

    rm -f "${proj}/assets/main.flscn"
    if ! grep -qE 'entry_scene|main\.flscn|load_flscn' "${proj}/src/main.mojo"; then
        fail_test "t2_m3_f26_missing_entry_scene_graceful_error" "M3 F26 contract pending: scaffolded src/main.mojo does not load entry_scene / main.flscn"
        return
    fi

    python3 -B - "${proj}/flint.toml" <<'PY'
import re, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as fh:
    txt = fh.read()
if re.search(r"^\s*entry_scene\s*=", txt, re.M):
    txt = re.sub(r"^\s*entry_scene\s*=.*$", 'entry_scene = "assets/nonexistent_entry.flscn"', txt, flags=re.M)
else:
    txt += '\nentry_scene = "assets/nonexistent_entry.flscn"\n'
with open(p, "w", encoding="utf-8") as fh:
    fh.write(txt)
PY

    local run_out run_rc=0
    run_out="$(cd "$proj" && FLINT_ROOT="$ROOT" FLINT_STATE_DIR="${TMPD}/ms_state" XDG_STATE_HOME="${TMPD}/ms_xdg" bash tools/run.sh --headless-smoke 2>&1)" || run_rc=$?
    if [[ "$run_rc" -eq 0 ]]; then
        fail_test "t2_m3_f26_missing_entry_scene_graceful_error" "M3 F26 contract pending: missing entry_scene ('assets/nonexistent_entry.flscn') exited 0 instead of failing cleanly"
        return
    fi
    if [[ "$run_rc" -ge 128 ]]; then
        fail_test "t2_m3_f26_missing_entry_scene_graceful_error" "process crashed with signal (rc=$run_rc): $run_out"
        return
    fi
    if ! grep -qiE "scene|flscn|nonexistent|not found|missing|failed to read" <<<"$run_out"; then
        fail_test "t2_m3_f26_missing_entry_scene_graceful_error" "exited $run_rc without clear scene error message: $run_out"
        return
    fi

    pass_test "t2_m3_f26_missing_entry_scene_graceful_error"
}

# ---------------------------------------------------------------------------
# Case 3 (F27 negative): MCP project.package & scene.save/load reject path traversal
# ---------------------------------------------------------------------------
test_t2_m3_mcp_traversal_and_symlink_rejected() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_test "t2_m3_f27_mcp_traversal_and_symlink_rejected" "could not resolve flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"

    local ws="${TMPD}/mcp_sec_ws"
    local outside_dir="${TMPD}/outside_target"
    mkdir -p "${ws}/scenes" "${ws}/assets" "$outside_dir"
    ln -s "$outside_dir" "${ws}/sym_escape"
    ln -s "${outside_dir}/escape.flscn" "${ws}/sym_file.flscn"

    local py_out py_rc=0
    py_out="$(
        FLINT_STATE_DIR="${TMPD}/mcp_sec_state" XDG_STATE_HOME="${TMPD}/mcp_sec_xdg" \
        python3 -B - "$studio_bin" "$ws" "$outside_dir" "${SCRIPT_DIR}/../common" "$TMPD" <<'PY' 2>&1
import os, sys
sys.path.insert(0, sys.argv[4])
from mcp_client import McpStdioClient

studio_bin, ws, outside_dir, other_cwd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5]
with McpStdioClient([studio_bin, "--project", ws, "--mcp-stdio"], cwd=other_cwd, framing="cl", timeout=30.0, allow_stderr=False) as c:
    c.initialize()
    c.notify_initialized()
    tools = {t["name"] for t in c.list_tools()}
    missing = sorted({"project.package", "scene.save", "scene.load"} - tools)
    if missing:
        raise SystemExit(f"M0/M3 contract pending; missing MCP tools in tools/list: {missing}")

    bad_paths = [
        "../escape_pkg",
        os.path.join(outside_dir, "abs_escape"),
        "sym_escape",
        "sym_escape/via_symlink",
        ".",
    ]
    for p in bad_paths:
        c.call_tool_expect_error("project.package", {"output_dir": p})
        c.call_tool_expect_error("scene.save", {"path": p})
        c.call_tool_expect_error("scene.load", {"path": p})
        if p not in (".", "sym_escape"):
            c.call_tool_expect_error("scene.save", {"path": p + ".flscn"})
            c.call_tool_expect_error("scene.load", {"path": p + ".flscn"})
    c.call_tool_expect_error("scene.save", {"path": "sym_file.flscn"})
    c.call_tool_expect_error("scene.load", {"path": "sym_file.flscn"})

    if os.listdir(outside_dir):
        raise SystemExit(f"Traversal leaked files into outside_dir: {os.listdir(outside_dir)}")
    if (
        os.path.exists(os.path.join(os.path.dirname(ws), "escape_pkg"))
        or os.path.exists(os.path.join(os.path.dirname(ws), "escape_pkg.flscn"))
        or os.path.exists(os.path.join(other_cwd, "escape_pkg"))
        or os.path.exists(os.path.join(other_cwd, "escape_pkg.flscn"))
    ):
        raise SystemExit("Relative ../ traversal wrote outside workspace_root")
PY
    )" || py_rc=$?
    if [[ "$py_rc" -ne 0 ]]; then
        fail_test "t2_m3_f27_mcp_traversal_and_symlink_rejected" "$py_out"
        return
    fi

    pass_test "t2_m3_f27_mcp_traversal_and_symlink_rejected"
}

# ---------------------------------------------------------------------------
# Case 4 (F26 boundary): VFS loose fallback on missing optional override &
#   corrupt chunk payload CRC, reverse mount precedence, and 0-byte loose file
# ---------------------------------------------------------------------------
test_t2_m3_vfs_fallback_corrupt_chunk_precedence_and_zero_byte() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t2_m3_f26_vfs_fallback_corrupt_chunk_precedence_and_zero_byte" "could not resolve flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"

    local base_src="${TMPD}/vfs_base_src"
    local patch_src="${TMPD}/vfs_patch_src"
    local corrupt_src="${TMPD}/vfs_corrupt_src"
    local loose_dir="${TMPD}/vfs_loose_fallback"
    mkdir -p "${base_src}/cfg" "${patch_src}/cfg" "${corrupt_src}/cfg" "${loose_dir}/cfg"

    printf '{"layer":"base","id":1}\n' > "${base_src}/cfg/layer.json"
    printf '{"layer":"patch","id":2}\n' > "${patch_src}/cfg/layer.json"
    printf '{"source":"corrupt_pack_payload"}\n' > "${corrupt_src}/cfg/badchunk.json"
    printf '{"source":"loose_override_recovery"}\n' > "${loose_dir}/cfg/badchunk.json"
    printf '{"source":"loose_optional"}\n' > "${loose_dir}/cfg/optional.json"
    : > "${loose_dir}/cfg/zero.json"

    local base_flpk="${TMPD}/base.flpk"
    local patch_flpk="${TMPD}/patch.flpk"
    local corrupt_chunk_flpk="${TMPD}/corrupt_chunk.flpk"
    if ! "$pack_bin" --cook "$base_src" "$base_flpk" --cache-dir "${TMPD}/c_base" >/dev/null 2>&1 \
       || ! "$pack_bin" --cook "$patch_src" "$patch_flpk" --cache-dir "${TMPD}/c_patch" >/dev/null 2>&1 \
       || ! "$pack_bin" --cook "$corrupt_src" "$corrupt_chunk_flpk" --cache-dir "${TMPD}/c_corr" >/dev/null 2>&1; then
        fail_test "t2_m3_f26_vfs_fallback_corrupt_chunk_precedence_and_zero_byte" "flint-pack --cook setup failed"
        return
    fi

    # Flip 1 byte inside badchunk.json payload (keeping 64B header + TOC CRC64 intact so mount succeeds,
    # while read_payload_by_toc(..., verify_crc=True) detects chunk payload CRC64 mismatch).
    python3 -B - "$corrupt_chunk_flpk" <<'PY'
import struct, sys
p = sys.argv[1]
with open(p, "rb") as f:
    data = bytearray(f.read())
magic, ver, flags, count, toc_off, uuid_off, path_off, hcap, str_bytes, total_bytes, toc_crc64 = struct.unpack(
    "<4sIIIQQQIIQQ", data[:64]
)
assert magic == b"FLPK" and ver == 3 and count == 1
u_hi, u_lo, pfnv, p_off, c_bytes, u_bytes, pcrc, sub_id, kind, comp, p_str = struct.unpack(
    "<QQQQQQQHBBI", data[toc_off:toc_off + 64]
)
assert c_bytes > 0
data[p_off] ^= 0xFF
with open(p, "wb") as f:
    f.write(data)
PY

    local probe_src="${TMPD}/vfs_neg_probe.mojo"
    local probe_bin="${TMPD}/bin/vfs_neg_probe"
    cat >"$probe_src" <<'MOJO'
from std.sys import argv
from flint.assets.server import AssetServer

def _as_str(bytes: List[UInt8]) -> String:
    var cp = bytes.copy()
    return String(unsafe_from_utf8=cp)

def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error("usage: vfs_neg_probe <base.flpk> <patch.flpk> <corrupt_chunk.flpk> <loose_dir>")

    # 1. Reverse mount precedence + missing optional asset fallback + 0-byte loose boundary
    var srv1 = AssetServer(enable_watcher=False)
    if not srv1.mount_flpk(String(args[1])):
        raise Error("failed mounting base.flpk")
    if not srv1.mount_flpk(String(args[2])):
        raise Error("failed mounting patch.flpk")
    _ = srv1.watch_loose_directory(String(args[4]))

    var h_layer = srv1.load_raw("cfg/layer.json")
    var p_layer = srv1.raws.get_ptr(h_layer)
    if not p_layer or _as_str(p_layer.value()[unsafe_offset=0].bytes).find('"patch"') < 0:
        raise Error("reverse mount precedence failed (expected patch.flpk to win)")

    var h_opt = srv1.load_raw("cfg/optional.json")
    var p_opt = srv1.raws.get_ptr(h_opt)
    if not p_opt or _as_str(p_opt.value()[unsafe_offset=0].bytes).find('"loose_optional"') < 0:
        raise Error("loose FS fallback failed for cfg/optional.json")

    var h_zero = srv1.load_raw("cfg/zero.json")
    var p_zero = srv1.raws.get_ptr(h_zero)
    if not p_zero or len(p_zero.value()[unsafe_offset=0].bytes) != 0:
        raise Error("0-byte loose raw asset did not load with 0-byte parity")
    srv1.close()

    # 2. Corrupt chunk payload CRC inside .flpk triggers fallback to valid loose FS
    var srv2 = AssetServer(enable_watcher=False)
    if not srv2.mount_flpk(String(args[3])):
        raise Error("mount_flpk failed on valid-header/corrupt-chunk archive")
    _ = srv2.watch_loose_directory(String(args[4]))
    var h_bad = srv2.load_raw("cfg/badchunk.json")
    var p_bad = srv2.raws.get_ptr(h_bad)
    if not p_bad:
        raise Error("corrupt chunk payload did not fall back to loose FS")
    var s_bad = _as_str(p_bad.value()[unsafe_offset=0].bytes)
    if s_bad.find("loose_override_recovery") < 0:
        raise Error("corrupt chunk returned wrong bytes instead of loose FS fallback: " + s_bad)
    srv2.close()
MOJO
    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >/dev/null 2>&1; then
        fail_test "t2_m3_f26_vfs_fallback_corrupt_chunk_precedence_and_zero_byte" "failed to compile vfs_neg_probe.mojo"
        return
    fi

    local probe_out probe_rc=0
    probe_out="$("$probe_bin" "$base_flpk" "$patch_flpk" "$corrupt_chunk_flpk" "$loose_dir" 2>&1)" || probe_rc=$?
    if [[ "$probe_rc" -ne 0 ]]; then
        fail_test "t2_m3_f26_vfs_fallback_corrupt_chunk_precedence_and_zero_byte" "vfs_neg_probe failed (rc=$probe_rc): $probe_out"
        return
    fi

    pass_test "t2_m3_f26_vfs_fallback_corrupt_chunk_precedence_and_zero_byte"
}

# ---------------------------------------------------------------------------
# Case 5 (F28 isolation): Clean-dir bundle self-containment (ldd, strings/grep,
#   and isolated relocated run_in_clean_env execution without touching /home/user/flint)
# ---------------------------------------------------------------------------
test_t2_m3_bundle_self_containment_ldd_strings_and_relocation() {
    local proj="${TMPD}/iso_proj"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t2_m3_f28_bundle_self_containment_ldd_strings_and_relocation" "flint.sh new failed"
        return
    fi
    mkdir -p "${proj}/assets"
    printf '{"isolated":true}\n' > "${proj}/assets/cfg.json"
    cat > "${proj}/src/main.mojo" <<'MOJO'
from std.ffi import external_call
from flint.platform.os import env_flag
from flint.app import AppConfig, EngineApp
from flint.assets.server import AssetServer
from flint.core.log import flush_log, start_log

def main() raises:
    _ = start_log("iso_proj")
    var assets = AssetServer(enable_watcher=False)
    if not assets.mount_flpk("assets.flpk"):
        raise Error("failed to mount assets.flpk")
    var h = assets.load_raw("cfg.json")
    if not assets.raws.get_ptr(h):
        raise Error("failed to read cfg.json from assets.flpk")
    var app = EngineApp(AppConfig(title="iso_proj", headless=True, enable_crash_handler=True))
    app.startup()
    if env_flag("FLINT_TEST_RELEASE_CRASH"):
        _ = external_call["raise", Int32](Int32(6))
    app.step_one_frame(Float32(1.0 / 60.0))
    _ = flush_log()
    assets.close()
    app.shutdown()
    print("[iso_proj] Headless v5 frame loop OK")
MOJO

    local stage_dist="${TMPD}/iso_stage_dist"
    local pkg_out pkg_rc=0
    pkg_out="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" package --out "$stage_dist" --symbols "$TMPD/iso_private_symbols" 2>&1)" || pkg_rc=$?
    if [[ "$pkg_rc" -ne 0 ]]; then
        fail_test "t2_m3_f28_bundle_self_containment_ldd_strings_and_relocation" "flint.sh package failed (rc=$pkg_rc): $pkg_out"
        return
    fi

    local shipped sha private_symbols
    for shipped in "$stage_dist/bin/iso_proj" "$stage_dist/lib/gameplay.so"; do
        sha="$(sha256sum -- "$shipped" | cut -d' ' -f1)"
        private_symbols="$TMPD/iso_private_symbols/$sha.debug"
        if [[ ! -s "$private_symbols" ]] || [[ "$(readelf -SW "$shipped")" == *".debug_info"* ]] ||
                [[ "$(readelf -SW "$private_symbols" 2>/dev/null)" != *".debug_line"* ]]; then
            fail_test "t2_m3_f28_bundle_self_containment_ldd_strings_and_relocation" "split line tables missing or included in shipped module: $shipped"
            return
        fi
    done
    local pack_digest lock_digest
    pack_digest="$(sha256sum "$stage_dist/assets.flpk" | cut -d' ' -f1)"
    lock_digest="$(sha256sum "$stage_dist/engine-pixi.lock" | cut -d' ' -f1)"
    if [[ -z "$pack_digest" || -z "$lock_digest" ]] ||
        ! grep -Fxq "assets_pack_sha256=$pack_digest" "$stage_dist/MANIFEST.txt" ||
        ! grep -Fxq "engine_lock_sha256=$lock_digest" "$stage_dist/MANIFEST.txt" ||
        ! grep -Eq '^engine_revision=.+' "$stage_dist/MANIFEST.txt" ||
        ! grep -Eq '^game_revision=.+' "$stage_dist/MANIFEST.txt"; then
        fail_test "t2_m3_f28_bundle_self_containment_ldd_strings_and_relocation" "package provenance does not match its pack or compiler lock"
        return
    fi

    local fault_line
    fault_line="$(awk '/external_call\["raise"/ { print NR }' "$proj/src/main.mojo")"
    # Relocate to a completely isolated directory and remove project + staging dirs
    local reloc="${TMPD}/isolated_machine/nested/game_bundle"
    mkdir -p "$(dirname "$reloc")"
    cp -a "$stage_dist" "$reloc"
    rm -rf "$stage_dist" "$proj"

    # 1. Execute relocated bundle in clean env first to confirm runtime behavior
    local run_out run_rc=0
    run_out="$(FLINT_STATE_DIR="${TMPD}/iso_state" XDG_STATE_HOME="${TMPD}/iso_xdg" run_in_clean_env "$reloc" ./run.sh --headless-smoke 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 ]] || ! grep -q "\[iso_proj\] Headless v5 frame loop OK" <<<"$run_out"; then
        fail_test "t2_m3_f28_bundle_self_containment_ldd_strings_and_relocation" "relocated clean-env run.sh failed (rc=$run_rc): $run_out"
        return
    fi

    # 2. Text grep on run.sh, flint.toml, MANIFEST.txt for zero /home/user/flint or $ROOT references
    local tf
    for tf in "${reloc}/run.sh" "${reloc}/flint.toml" "${reloc}/MANIFEST.txt"; do
        if [[ ! -f "$tf" ]]; then
            fail_test "t2_m3_f28_bundle_self_containment_ldd_strings_and_relocation" "bundle missing $(basename "$tf")"
            return
        fi
        if grep -Fq "/home/user/flint" "$tf" || grep -Fq "${ROOT}" "$tf"; then
            fail_test "t2_m3_f28_bundle_self_containment_ldd_strings_and_relocation" "$(basename "$tf") leaks repo path (/home/user/flint)"
            return
        fi
    done

    # 3. Check ldd on bin/* (both without LD_LIBRARY_PATH and with ${reloc}/lib) and lib/*.so
    local ldd_issues=()
    local elf
    for elf in "${reloc}/bin/"* "${reloc}/lib/"*.so; do
        [[ -e "$elf" ]] || continue
        local ldd_out
        ldd_out="$(LD_LIBRARY_PATH="${reloc}/lib" ldd "$elf" 2>&1 || true)"
        if grep -q "not found" <<<"$ldd_out"; then
            ldd_issues+=("not_found($(basename "$elf"))")
        fi
        if grep -Fq "/home/user/flint" <<<"$ldd_out" || grep -Fq "${ROOT}" <<<"$ldd_out"; then
            local leak_line
            leak_line="$(grep -F -m 1 -e "/home/user/flint" -e "${ROOT}" <<<"$ldd_out" || true)"
            ldd_issues+=("ldd_repo_ref($(basename "$elf"): ${leak_line})")
        fi
    done

    local b
    for b in "${reloc}/bin/"*; do
        [[ -f "$b" ]] || continue
        local ldd_nold
        ldd_nold="$(env -u LD_LIBRARY_PATH ldd "$b" 2>&1 || true)"
        if grep -q "not found" <<<"$ldd_nold"; then
            ldd_issues+=("not_found_no_ld_path($(basename "$b"))")
        fi
        if grep -Fq "/home/user/flint" <<<"$ldd_nold" || grep -Fq "${ROOT}" <<<"$ldd_nold"; then
            local leak_nold
            leak_nold="$(grep -F -m 1 -e "/home/user/flint" -e "${ROOT}" <<<"$ldd_nold" || true)"
            ldd_issues+=("ldd_no_ld_path_repo_ref($(basename "$b"): ${leak_nold})")
        fi
    done

    # 4. Check strings on bin/* for zero /home/user/flint references (RUNPATH / embedded paths)
    # Capture strings output first to avoid SIGPIPE (141) under set -uo pipefail when grep -m 1 exits early
    local bin_leaks=()
    if command -v strings >/dev/null 2>&1; then
        for b in "${reloc}/bin/"*; do
            [[ -f "$b" ]] || continue
            local all_strings hit
            all_strings="$(strings "$b" 2>/dev/null || true)"
            hit="$(grep -F -m 1 -e "/home/user/flint" -e "${ROOT}" <<<"$all_strings" || true)"
            if [[ -n "$hit" ]]; then
                bin_leaks+=("$(basename "$b"):${hit}")
            fi
        done
    fi

    if [[ "${#ldd_issues[@]}" -gt 0 || "${#bin_leaks[@]}" -gt 0 ]]; then
        fail_test "t2_m3_f28_bundle_self_containment_ldd_strings_and_relocation" "bundle references /home/user/flint or unresolved libs: ldd=[${ldd_issues[*]}] bin_strings=[${bin_leaks[*]}]"
        return
    fi

    local crash_rc=0 crash_report lines
    (ulimit -c 0; FLINT_STATE_DIR="$TMPD/iso_crash_state" FLINT_TEST_RELEASE_CRASH=1 "$reloc/run.sh") >"$TMPD/iso_crash.log" 2>&1 || crash_rc=$?
    crash_report="$(find "$TMPD/iso_crash_state" -name crash.txt -print -quit)"
    if [[ "$crash_rc" != 134 || -z "$crash_report" ]] ||
            ! lines="$(python3 "$ROOT/tools/symbolize_release.py" "$crash_report" "$reloc/bin/iso_proj" "$TMPD/iso_private_symbols")" ||
            ! grep -Eq "main\.mojo:${fault_line}([^0-9]|$)" <<<"$lines"; then
        fail_test "t2_m3_f28_bundle_self_containment_ldd_strings_and_relocation" "could not resolve packaged SIGABRT to its Mojo call site: ${lines:-no frames}"
        return
    fi
    pass_test "t2_m3_f28_bundle_self_containment_ldd_strings_and_relocation"
}

# ---------------------------------------------------------------------------
# Case 6 (F26 boundary): Corrupt .flpk header/TOC (0-byte, truncated 16B,
#   bad magic, corrupted CRC64 TOC) fails gracefully in AssetServer & flint-pack
# ---------------------------------------------------------------------------
test_t2_m3_corrupt_flpk_header_and_toc_graceful() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t2_m3_f26_corrupt_flpk_header_and_toc_graceful" "could not resolve flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"

    local ws="${TMPD}/corr_hdr_ws"
    mkdir -p "$ws"
    printf '{"ok":true}\n' > "${ws}/ok.json"
    local valid_flpk="${TMPD}/valid_src.flpk"
    if ! "$pack_bin" --cook "$ws" "$valid_flpk" --cache-dir "${TMPD}/corr_hdr_cache" >/dev/null 2>&1; then
        fail_test "t2_m3_f26_corrupt_flpk_header_and_toc_graceful" "initial cook failed"
        return
    fi

    local f_zero="${TMPD}/zero.flpk"
    local f_trunc="${TMPD}/trunc16.flpk"
    local f_magic="${TMPD}/badmagic.flpk"
    local f_toc="${TMPD}/badtoc.flpk"

    python3 -B - "$valid_flpk" "$f_zero" "$f_trunc" "$f_magic" "$f_toc" <<'PY'
import sys
with open(sys.argv[1], "rb") as f:
    good = bytearray(f.read())

with open(sys.argv[2], "wb") as f:
    f.write(b"")

with open(sys.argv[3], "wb") as f:
    f.write(good[:16])

bad_m = bytearray(good)
bad_m[:4] = b"BAD!"
with open(sys.argv[4], "wb") as f:
    f.write(bad_m)

bad_t = bytearray(good)
# Flip a byte inside TOC CRC64 / header metadata
bad_t[56] ^= 0xFF
with open(sys.argv[5], "wb") as f:
    f.write(bad_t)
PY

    local probe_src="${TMPD}/flpk_corrupt_probe.mojo"
    local probe_bin="${TMPD}/bin/flpk_corrupt_probe"
    cat >"$probe_src" <<'MOJO'
from std.sys import argv
from flint.assets.flpk_v3 import FlpkV3MappedArchive
from flint.assets.server import AssetServer

def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: flpk_corrupt_probe <paths...>")
    var srv = AssetServer(enable_watcher=False)
    for i in range(1, len(args)):
        var p = String(args[i])
        var arch = FlpkV3MappedArchive.open(p)
        if arch.valid:
            arch.close()
            raise Error("FlpkV3MappedArchive.open unexpectedly succeeded on corrupt archive: " + p)
        if srv.mount_flpk(p):
            raise Error("AssetServer.mount_flpk unexpectedly succeeded on corrupt archive: " + p)
    srv.close()
MOJO
    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >/dev/null 2>&1; then
        fail_test "t2_m3_f26_corrupt_flpk_header_and_toc_graceful" "failed to compile flpk_corrupt_probe.mojo"
        return
    fi

    local probe_out probe_rc=0
    probe_out="$("$probe_bin" "$f_zero" "$f_trunc" "$f_magic" "$f_toc" 2>&1)" || probe_rc=$?
    if [[ "$probe_rc" -ne 0 ]]; then
        fail_test "t2_m3_f26_corrupt_flpk_header_and_toc_graceful" "Mojo corrupt archive probe failed (rc=$probe_rc): $probe_out"
        return
    fi

    local cf rc
    for cf in "$f_zero" "$f_trunc" "$f_magic" "$f_toc"; do
        rc=0
        "$pack_bin" --inspect "$cf" >/dev/null 2>&1 || rc=$?
        if [[ "$rc" -eq 0 || "$rc" -ge 128 ]]; then
            fail_test "t2_m3_f26_corrupt_flpk_header_and_toc_graceful" "flint-pack --inspect on $(basename "$cf") exited $rc (expected clean non-zero <128)"
            return
        fi
    done

    pass_test "t2_m3_f26_corrupt_flpk_header_and_toc_graceful"
}

test_t2_m3_scaffold_preserves_existing_game() {
    local proj="${TMPD}/existing_project" collision="${TMPD}/late_collision"
    local test_name="t2_m3_scaffold_preserves_existing_game"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null; then
        fail_test "$test_name" "initial scaffold failed"
        return
    fi
    echo "user edit" >> "$proj/src/main.mojo"
    local before after
    before="$(cd "$proj" && sha256sum flint.toml pixi.toml .gitignore README.md assets/main.flscn src/main.mojo src/gameplay.mojo tools/run.sh)"
    if bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "$test_name" "accepted existing project"
        return
    fi
    after="$(cd "$proj" && sha256sum flint.toml pixi.toml .gitignore README.md assets/main.flscn src/main.mojo src/gameplay.mojo tools/run.sh)"
    mkdir -p "$collision/tools"
    echo "existing script" > "$collision/tools/run.sh"
    if [[ "$after" != "$before" ]] || bash "${ROOT}/tools/flint.sh" new "$collision" >/dev/null 2>&1 ||
        [[ -e "$collision/flint.toml" || -e "$collision/assets" || "$(cat "$collision/tools/run.sh")" != "existing script" ]]; then
        fail_test "$test_name" "scaffold changed existing files before reporting a collision"
        return
    fi
    pass_test "$test_name"
}

test_t2_m3_scaffold_preserves_existing_game
test_t2_m3_flint_version_mismatch_rejected
test_t2_m3_missing_entry_scene_graceful_error
test_t2_m3_mcp_traversal_and_symlink_rejected
test_t2_m3_vfs_fallback_corrupt_chunk_precedence_and_zero_byte
test_t2_m3_bundle_self_containment_ldd_strings_and_relocation
test_t2_m3_corrupt_flpk_header_and_toc_graceful

print_summary_and_exit
