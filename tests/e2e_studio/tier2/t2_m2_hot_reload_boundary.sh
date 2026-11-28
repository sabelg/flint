#!/usr/bin/env bash
# AREA: m2
# Tier 2 Boundary & Negative Coverage: Area D — Hot-Reload, Reflection, BT/GameFlow & M2 MCP Negative Guards (F22, F23, F25)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t2_m2_hr"
resolve_mojo_env || {
    fail_test "t2_m2_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

# ---------------------------------------------------------------------------
# Case 1 (F22): `flint.sh gameplay-so` missing project & broken source preserves prior .so
# ---------------------------------------------------------------------------
test_f22_gameplay_so_missing_project_and_broken_source() {
    local empty_dir="${TMPD}/no_project"
    mkdir -p "$empty_dir"
    local out_empty="" rc_empty=0
    out_empty="$(cd "$empty_dir" && bash "${ROOT}/tools/flint.sh" gameplay-so 2>&1)" || rc_empty=$?
    if [[ "$rc_empty" -eq 0 ]]; then
        fail_test "t2_m2_f22_gameplay_so_missing_project_and_broken_source" "gameplay-so exited 0 in directory without flint.toml (Usage fallback trap)"
        return
    fi
    local stray_so=""
    stray_so="$(find "$empty_dir" -name '*.so' 2>/dev/null | head -n 1 || true)"
    if [[ -n "$stray_so" ]]; then
        fail_test "t2_m2_f22_gameplay_so_missing_project_and_broken_source" "unexpected .so created without flint.toml: $stray_so"
        return
    fi

    local proj="${TMPD}/proj_broken_so"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t2_m2_f22_gameplay_so_missing_project_and_broken_source" "flint.sh new failed"
        return
    fi
    cat >"${proj}/src/gameplay.mojo" <<'MOJO'
@export("flint_plugin_api_version")
def flint_plugin_api_version() abi("C") -> UInt32:
    return 2

@export("flint_plugin_on_load")
def flint_plugin_on_load(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_step")
def flint_plugin_on_step(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _dt: Float32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_unload")
def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> None:
    pass
MOJO

    local good_out="" good_rc=0
    good_out="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" gameplay-so 2>&1)" || good_rc=$?
    if [[ "$good_rc" -ne 0 ]] || grep -q "Usage:" <<<"$good_out"; then
        fail_test "t2_m2_f22_gameplay_so_missing_project_and_broken_source" "initial valid gameplay-so failed or fell through to Usage (pending M5)"
        return
    fi
    local prev_so=""
    prev_so="$(find "${proj}/build" -maxdepth 3 -type f -name '*.so' -size +0c 2>/dev/null | head -n 1 || true)"
    if [[ -z "$prev_so" ]]; then
        fail_test "t2_m2_f22_gameplay_so_missing_project_and_broken_source" "missing initial .so artifact"
        return
    fi
    local prev_sha=""
    prev_sha="$(sha256sum "$prev_so" | awk '{print $1}')"

    # Inject syntax error and verify non-zero exit + intact previous .so
    cat >"${proj}/src/gameplay.mojo" <<'MOJO'
def broken_syntax(((:
    return ???
MOJO
    local bad_rc=0
    (cd "$proj" && bash "${ROOT}/tools/flint.sh" gameplay-so >/dev/null 2>&1) || bad_rc=$?
    if [[ "$bad_rc" -eq 0 ]]; then
        fail_test "t2_m2_f22_gameplay_so_missing_project_and_broken_source" "broken gameplay source exited 0"
        return
    fi
    local after_sha=""
    after_sha="$(sha256sum "$prev_so" | awk '{print $1}')"
    if [[ "$prev_sha" != "$after_sha" ]]; then
        fail_test "t2_m2_f22_gameplay_so_missing_project_and_broken_source" "broken build overwrote/corrupted prior valid .so"
        return
    fi

    pass_test "t2_m2_f22_gameplay_so_missing_project_and_broken_source"
}

# ---------------------------------------------------------------------------
# Case 2 (F22): ABI version (v1/v99), missing entrypoints & FlintPluginApiV2 magic/size rejected
# ---------------------------------------------------------------------------
test_f22_abi_version_and_magic_size_mismatch_rejected() {
    local guest_v99="${TMPD}/guest_v99.mojo"
    local guest_v1="${TMPD}/guest_v1.mojo"
    local guest_no_step="${TMPD}/guest_no_step.mojo"
    local guest_no_ver="${TMPD}/guest_no_ver.mojo"
    local so_v99="${TMPD}/guest_v99.so"
    local so_v1="${TMPD}/guest_v1.so"
    local so_no_step="${TMPD}/guest_no_step.so"
    local so_no_ver="${TMPD}/guest_no_ver.so"

    cat >"$guest_v99" <<'MOJO'
@export("flint_plugin_api_version")
def flint_plugin_api_version() abi("C") -> UInt32:
    return 99

@export("flint_plugin_on_load")
def flint_plugin_on_load(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_step")
def flint_plugin_on_step(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _dt: Float32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_unload")
def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> None:
    pass
MOJO

    cat >"$guest_v1" <<'MOJO'
@export("flint_plugin_api_version")
def flint_plugin_api_version() abi("C") -> UInt32:
    return 1

@export("flint_plugin_on_load")
def flint_plugin_on_load(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_step")
def flint_plugin_on_step(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _dt: Float32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_unload")
def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> None:
    pass
MOJO

    cat >"$guest_no_step" <<'MOJO'
from flint.app import FLINT_PLUGIN_API_VERSION

@export("flint_plugin_api_version")
def flint_plugin_api_version() abi("C") -> UInt32:
    return FLINT_PLUGIN_API_VERSION

@export("flint_plugin_on_load")
def flint_plugin_on_load(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_unload")
def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> None:
    pass
MOJO

    cat >"$guest_no_ver" <<'MOJO'
@export("flint_plugin_on_load")
def flint_plugin_on_load(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_step")
def flint_plugin_on_step(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _dt: Float32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_unload")
def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> None:
    pass
MOJO

    local pair src_f out_f
    for pair in \
        "${guest_v99}:${so_v99}" \
        "${guest_v1}:${so_v1}" \
        "${guest_no_step}:${so_no_step}" \
        "${guest_no_ver}:${so_no_ver}"; do
        src_f="${pair%%:*}"
        out_f="${pair##*:}"
        if ! "$MOJO_BIN" build -I "${ROOT}/src" "$src_f" --emit shared-lib -o "$out_f" -O0 --Werror >"${out_f}.compile.log" 2>&1; then
            fail_test "t2_m2_f22_abi_version_and_magic_size_mismatch_rejected" "failed compiling negative guest ${src_f}"
            return
        fi
    done

    local probe_src="${TMPD}/abi_reject_probe.mojo"
    local probe_bin="${TMPD}/bin/abi_reject_probe"
    local probe_log="${TMPD}/abi_reject_probe.compile.log"

    cat >"$probe_src" <<MOJO
from std.sys import size_of
from flint.app import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2, HotReloadHost, Resources
from flint.ecs import World

def main() raises:
    if Int(FLINT_PLUGIN_API_VERSION) != 2:
        raise Error("FLINT_PLUGIN_API_VERSION must be 2")

    var w = World()
    var r = Resources()
    var host = HotReloadHost("${TMPD}/t2_abi_stage")

    # 1. ABI v99 & ABI v1 mismatch rejection
    if host.load_or_reload("${so_v99}", w, r) or host.is_loaded():
        raise Error("guest_v99.so was not rejected")
    if host.last_error.byte_length() == 0 or host.last_error.find("mismatch") < 0:
        raise Error("guest_v99.so error did not report ABI mismatch: " + host.last_error)

    if host.load_or_reload("${so_v1}", w, r) or host.is_loaded():
        raise Error("guest_v1.so was not rejected")
    if host.last_error.byte_length() == 0 or host.last_error.find("mismatch") < 0:
        raise Error("guest_v1.so error did not report ABI mismatch: " + host.last_error)

    # 2. Missing required entrypoints (no on_step / no api_version)
    if host.load_or_reload("${so_no_step}", w, r) or host.is_loaded():
        raise Error("guest_no_step.so was not rejected")
    if host.last_error.byte_length() == 0 or host.last_error.find("missing required plugin entrypoints") < 0:
        raise Error("guest_no_step.so error did not report missing entrypoints: " + host.last_error)

    if host.load_or_reload("${so_no_ver}", w, r) or host.is_loaded():
        raise Error("guest_no_ver.so was not rejected")
    if host.last_error.byte_length() == 0 or host.last_error.find("missing required plugin entrypoints") < 0:
        raise Error("guest_no_ver.so error did not report missing entrypoints: " + host.last_error)

    # 3. FlintPluginApiV2 magic & struct_size corruption checks
    var api_ok = FlintPluginApiV2()
    if not api_ok.is_valid() or Int(api_ok.struct_size) != size_of[FlintPluginApiV2]():
        raise Error("baseline FlintPluginApiV2 invalid")
    var api_bad_magic = FlintPluginApiV2()
    api_bad_magic.magic ^= UInt32(0x1)
    if api_bad_magic.is_valid():
        raise Error("corrupted FlintPluginApiV2.magic was not rejected")
    var api_bad_size = FlintPluginApiV2()
    api_bad_size.struct_size = 8
    if api_bad_size.is_valid():
        raise Error("mismatched FlintPluginApiV2.struct_size was not rejected")
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >"$probe_log" 2>&1; then
        fail_test "t2_m2_f22_abi_version_and_magic_size_mismatch_rejected" "M5 FlintPluginApiV2 contract pending"
        return
    fi

    local out="" rc=0
    out="$("$probe_bin" 2>&1)" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t2_m2_f22_abi_version_and_magic_size_mismatch_rejected" "probe failed (rc=$rc): $out"
        return
    fi
    local leaked_stage=""
    leaked_stage="$(find "${TMPD}" -maxdepth 1 -name 't2_abi_stage*.so' 2>/dev/null | head -n 1 || true)"
    if [[ -n "$leaked_stage" ]]; then
        fail_test "t2_m2_f22_abi_version_and_magic_size_mismatch_rejected" "leaked staging .so after ABI rejections: $leaked_stage"
        return
    fi
    pass_test "t2_m2_f22_abi_version_and_magic_size_mismatch_rejected"
}

# ---------------------------------------------------------------------------
# Case 3 (F22): Nonexistent .so, corrupt non-ELF bytes, and on_load non-zero error
# ---------------------------------------------------------------------------
test_f22_nonexistent_corrupt_so_and_on_load_error() {
    local bad_guest="${TMPD}/bad_onload_guest.mojo"
    local bad_so="${TMPD}/bad_onload_guest.so"
    local corrupt_so="${TMPD}/corrupt.so"
    printf 'NOT_ELF_DATA_CORRUPT_HEADER' >"$corrupt_so"

    cat >"$bad_guest" <<'MOJO'
from flint.app import FLINT_PLUGIN_API_VERSION

@export("flint_plugin_api_version")
def flint_plugin_api_version() abi("C") -> UInt32:
    return FLINT_PLUGIN_API_VERSION

@export("flint_plugin_on_load")
def flint_plugin_on_load(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> Int32:
    return -1

@export("flint_plugin_on_step")
def flint_plugin_on_step(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _dt: Float32) abi("C") -> Int32:
    return 0

@export("flint_plugin_on_unload")
def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> None:
    pass
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "$bad_guest" --emit shared-lib -o "$bad_so" -O0 --Werror >/dev/null 2>&1; then
        fail_test "t2_m2_f22_nonexistent_corrupt_so_and_on_load_error" "failed to compile bad_onload_guest.mojo"
        return
    fi

    local probe_src="${TMPD}/corrupt_so_probe.mojo"
    local probe_bin="${TMPD}/bin/corrupt_so_probe"
    cat >"$probe_src" <<MOJO
from flint.app import HotReloadHost, Resources
from flint.ecs import World

def main() raises:
    var w = World()
    var r = Resources()
    var host = HotReloadHost("${TMPD}/hr_stage_neg")

    if host.load_or_reload("${TMPD}/does_not_exist.so", w, r):
        raise Error("nonexistent .so unexpectedly loaded")
    if host.last_error.byte_length() == 0 or host.is_loaded():
        raise Error("missing last_error or loaded=True after nonexistent .so")

    if host.load_or_reload("${corrupt_so}", w, r):
        raise Error("corrupt .so unexpectedly loaded")
    if host.last_error.byte_length() == 0 or host.is_loaded():
        raise Error("missing last_error or loaded=True after corrupt .so")

    if host.load_or_reload("${bad_so}", w, r):
        raise Error("on_load=-1 .so unexpectedly loaded")
    if host.last_error.byte_length() == 0 or host.is_loaded():
        raise Error("missing last_error or loaded=True after on_load error")
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >/dev/null 2>&1; then
        fail_test "t2_m2_f22_nonexistent_corrupt_so_and_on_load_error" "failed to compile corrupt_so_probe.mojo"
        return
    fi

    local out="" rc=0
    out="$("$probe_bin" 2>&1)" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t2_m2_f22_nonexistent_corrupt_so_and_on_load_error" "probe failed (rc=$rc): $out"
        return
    fi
    local leaked_stage=""
    leaked_stage="$(find "${TMPD}" -maxdepth 1 -name 'hr_stage_neg*.so' 2>/dev/null | head -n 1 || true)"
    if [[ -n "$leaked_stage" ]]; then
        fail_test "t2_m2_f22_nonexistent_corrupt_so_and_on_load_error" "leaked staging .so after failed loads: $leaked_stage"
        return
    fi

    pass_test "t2_m2_f22_nonexistent_corrupt_so_and_on_load_error"
}

# ---------------------------------------------------------------------------
# Case 4 (F22): PodComponentSchema layout mismatch & out-of-bounds PodFieldDesc rejected
# ---------------------------------------------------------------------------
test_f22_pod_schema_layout_mismatch_rejected() {
    local probe_src="${TMPD}/pod_layout_probe.mojo"
    local probe_bin="${TMPD}/bin/pod_layout_probe"
    cat >"$probe_src" <<'MOJO'
from flint.app import PodComponentSchema, PodFieldDesc
from flint.ecs import STORAGE_SPARSE, STORAGE_TABLE, World
from flint.studio import FIELD_F32, FIELD_VEC3

def main() raises:
    var w = World()
    var tid1 = w.register_pod_component(UInt64(0xABCD123400000001), 8, 4, STORAGE_TABLE)
    var tid_mismatch = w.register_pod_component(UInt64(0xABCD123400000001), 16, 4, STORAGE_TABLE)
    var tid_mode_mismatch = w.register_pod_component(UInt64(0xABCD123400000001), 8, 4, STORAGE_SPARSE)
    if tid1 < 0 or tid_mismatch != -1 or tid_mode_mismatch != -1:
        raise Error("World.register_pod_component size/storage-mode mismatch did not return -1")

    var valid_s = PodComponentSchema("ValidPod", 8, 4)
    if not valid_s.add_field(PodFieldDesc("hp", FIELD_F32, 0, 4)) or not valid_s.validate():
        raise Error("valid PodComponentSchema rejected")

    # 1. Out-of-bounds field: offset(4) + size(12) = 16 > byte_size(8)
    var oob_s = PodComponentSchema("OutOfBoundsPod", 8, 4)
    var added_oob = oob_s.add_field(PodFieldDesc("pos", FIELD_VEC3, 4, 12))
    if added_oob and oob_s.validate():
        raise Error("out-of-bounds PodFieldDesc (offset+size > byte_size) was not rejected")

    # 2. Negative field offset
    var neg_s = PodComponentSchema("NegOffsetPod", 8, 4)
    var added_neg = neg_s.add_field(PodFieldDesc("bad", FIELD_F32, -4, 4))
    if added_neg and neg_s.validate():
        raise Error("negative offset PodFieldDesc was not rejected")

    # 3. Non-power-of-two alignment & empty schema name
    var bad_align = PodComponentSchema("BadAlign", 8, 3)
    if bad_align.validate():
        raise Error("non-power-of-two alignment schema was not rejected")
    var empty_name = PodComponentSchema("", 8, 4)
    if empty_name.validate():
        raise Error("empty schema name was not rejected")

    # 4. Invalid field type / zero size
    var bad_field = PodComponentSchema("BadFieldType", 8, 4)
    var added_type = bad_field.add_field(PodFieldDesc("unknown", -99, 0, 0))
    if added_type and bad_field.validate():
        raise Error("invalid field type/size PodFieldDesc was not rejected")
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >/dev/null 2>&1; then
        fail_test "t2_m2_f22_pod_schema_layout_mismatch_rejected" "M5 PodComponentSchema/PodFieldDesc contract pending"
        return
    fi
    local out="" rc=0
    out="$("$probe_bin" 2>&1)" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t2_m2_f22_pod_schema_layout_mismatch_rejected" "probe failed (rc=$rc): $out"
        return
    fi
    pass_test "t2_m2_f22_pod_schema_layout_mismatch_rejected"
}

# ---------------------------------------------------------------------------
# Case 5 (F23): Reflection Inspector invalid Add/Remove Component boundaries
# ---------------------------------------------------------------------------
test_f23_reflection_inspector_invalid_add_remove() {
    local probe_src="${TMPD}/refl_invalid_probe.mojo"
    local probe_bin="${TMPD}/bin/refl_invalid_probe"
    cat >"$probe_src" <<'MOJO'
from flint.app import PodComponentSchema
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD
from flint.studio import FlintStudioState

def main() raises:
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96)
    if studio.register_pod_schema(PodComponentSchema("", 8, 4)):
        raise Error("register_pod_schema accepted empty schema name")
    if studio.add_component_by_name(999999, "LocalTransform") or studio.add_component_by_name(-1, "LocalTransform"):
        raise Error("add_component_by_name succeeded on invalid entity")
    if studio.remove_component_by_name(999999, "LocalTransform") or studio.remove_component_by_name(-1, "LocalTransform"):
        raise Error("remove_component_by_name succeeded on invalid entity")

    var eid = studio.spawn_entity("LiveReflectionTarget")
    var h0 = studio.state_hash()
    if studio.add_component_by_name(eid, "NonExistentM2Component") or studio.state_hash() != h0:
        raise Error("add_component_by_name accepted unknown component or mutated state_hash")
    if studio.add_component_by_name(eid, "LocalTransform") or studio.state_hash() != h0:
        raise Error("duplicate add_component_by_name(LocalTransform) succeeded or mutated state_hash")
    if studio.remove_component_by_name(eid, "SpotLightComponent") or studio.state_hash() != h0:
        raise Error("remove_component_by_name succeeded on absent SpotLightComponent or mutated state_hash")
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >/dev/null 2>&1; then
        fail_test "t2_m2_f23_reflection_inspector_invalid_add_remove" "M5 Reflection Inspector Add/Remove Component API pending"
        return
    fi
    local out="" rc=0
    out="$("$probe_bin" 2>&1)" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t2_m2_f23_reflection_inspector_invalid_add_remove" "probe failed (rc=$rc): $out"
        return
    fi
    pass_test "t2_m2_f23_reflection_inspector_invalid_add_remove"
}

# ---------------------------------------------------------------------------
# Cases 6 & 7 (F25): M2 MCP Runtime Negative Guards (tools/list pre-gate + -32602 / isError=True)
# ---------------------------------------------------------------------------
run_m2_mcp_negative_cases() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_test "t2_m2_mcp_runtime_rejected_when_pie_stopped" "resolve_bin flint-studio failed"
        fail_test "t2_m2_mcp_runtime_invalid_entity_id_and_bad_params" "resolve_bin flint-studio failed"
        return
    fi

    local mcp_proj="${TMPD}/mcp_neg_proj"
    mkdir -p "${mcp_proj}/assets" "${mcp_proj}/scenes"
    cat >"${mcp_proj}/flint.toml" <<'TOML'
[project]
name = "m2_mcp_neg_proj"
entry_scene = "scenes/main.flscn"
TOML

    python3 -B - "${mcp_proj}" <<'PY'
import json, math, os, struct, sys

proj = sys.argv[1]
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
with open(os.path.join(proj, "assets", "sfx.wav"), "wb") as f:
    f.write(wav)

with open(os.path.join(proj, "scenes", "main.flscn"), "w", encoding="utf-8") as f:
    json.dump({"format": "flscn", "version": 2, "name": "M2McpNeg", "entities": []}, f)
PY

    validate_wav_pcm "${mcp_proj}/assets/sfx.wav" 2 48000 16 || {
        fail_test "t2_m2_mcp_runtime_rejected_when_pie_stopped" "sfx.wav fixture failed validate_wav_pcm"
        fail_test "t2_m2_mcp_runtime_invalid_entity_id_and_bad_params" "sfx.wav fixture failed validate_wav_pcm"
        return
    }

    local tsv="${TMPD}/mcp_neg_results.tsv"

    PYTHONPATH="${SCRIPT_DIR}/../common${PYTHONPATH:+:$PYTHONPATH}" python3 -B - "$RESOLVED_BIN" "$mcp_proj" "$tsv" <<'PY'
import sys
from mcp_client import McpStdioClient

studio_bin, proj_dir, out_tsv = sys.argv[1], sys.argv[2], sys.argv[3]

CASE_ORDER = [
    "t2_m2_mcp_runtime_rejected_when_pie_stopped",
    "t2_m2_mcp_runtime_invalid_entity_id_and_bad_params",
]
results: dict[str, tuple[bool, str]] = {}

def record(name: str, ok: bool, reason: str = "") -> None:
    if name in results and not results[name][0]:
        if not ok and reason and reason not in results[name][1]:
            results[name] = (False, f"{results[name][1]}; {reason}")
        return
    results[name] = (ok, reason)

def pick_arg(props: dict, candidates: list[str], default: str) -> str:
    for c in candidates:
        if c in props:
            return c
    return default

def assert_no_unimplemented_error(tool_name: str, msg: str, payload) -> None:
    low = msg.lower()
    if "unknown tool" in low:
        raise AssertionError(f"{tool_name} returned 'unknown tool': {payload!r}")
    if "not implemented" in low or "unimplemented" in low:
        raise AssertionError(f"{tool_name} returned 'not implemented': {payload!r}")
    if "unsupported tool" in low or "unsupported method" in low or (
        "unsupported" in low
        and not any(k in low for k in ("version", "schema", "format", "header", "asset", "file", "extension", "pie", "stopped"))
    ):
        raise AssertionError(f"{tool_name} returned 'unsupported' stub error: {payload!r}")

def assert_structured_tool_or_param_error(client: McpStdioClient, tool_name: str, args: dict, tools_by_name: dict) -> str:
    if tool_name not in tools_by_name:
        raise AssertionError(f"target tool {tool_name!r} missing from tools/list")
    resp = client.request("tools/call", {"name": tool_name, "arguments": args})
    if "error" in resp:
        err = resp["error"]
        code = err.get("code")
        msg = str(err.get("message", ""))
        assert_no_unimplemented_error(tool_name, msg, resp)
        if code != -32602:
            raise AssertionError(f"{tool_name} expected -32602 or isError=True, got JSON-RPC error: {resp!r}")
        return msg
    res = resp.get("result", {})
    if not res.get("isError", False):
        raise AssertionError(f"{tool_name} unexpectedly succeeded on invalid input {args!r}: {res!r}")
    sc = res.get("structuredContent")
    if not isinstance(sc, dict) or sc.get("ok") is not False:
        raise AssertionError(f"{tool_name} isError=True missing structuredContent.ok == False: {res!r}")
    err_msg = str(sc.get("error", ""))
    content = res.get("content", [])
    content_msg = str(content[0].get("text", "")) if isinstance(content, list) and content and isinstance(content[0], dict) else ""
    combined_msg = f"{err_msg} {content_msg}".strip()
    assert_no_unimplemented_error(tool_name, combined_msg, res)
    return err_msg or content_msg

required_m2 = ["physics.trigger_events", "anim.set_state", "audio.play", "nav.query_path"]
case6_ok, case6_err = False, ""
case7_ok, case7_err = False, ""

try:
    with McpStdioClient([studio_bin, "--project", proj_dir, "--mcp-stdio"], cwd=proj_dir, framing="cl", timeout=20.0) as client:
        init_res = client.initialize()
        if init_res.get("protocolVersion") != "2025-06-18":
            raise AssertionError(f"Unexpected protocolVersion: {init_res!r}")
        client.notify_initialized()

        tools_by_name = {
            t["name"]: t
            for t in client.list_tools()
            if isinstance(t, dict) and "name" in t
        }
        missing_m2 = [n for n in required_m2 if n not in tools_by_name]
        if missing_m2:
            msg = f"missing M2 MCP tool(s) (pending M5): {', '.join(missing_m2)}"
            case6_ok, case6_err = False, msg
            case7_ok, case7_err = False, msg
        else:
            anim_props = tools_by_name["anim.set_state"].get("inputSchema", {}).get("properties", {})
            aud_props = tools_by_name["audio.play"].get("inputSchema", {}).get("properties", {})
            trig_props = tools_by_name["physics.trigger_events"].get("inputSchema", {}).get("properties", {})

            ent_k = pick_arg(anim_props, ["entity_id", "entity", "id"], "entity_id")
            st_k = pick_arg(anim_props, ["state", "state_name", "clip"], "state")
            path_k = pick_arg(aud_props, ["asset_path", "path", "sound", "clip"], "asset_path")

            # Case 6: Runtime M2 tools rejected cleanly when PIE is stopped
            try:
                stopped_calls = [
                    ("physics.trigger_events", {}),
                    ("anim.set_state", {ent_k: 1, st_k: "idle"}),
                    ("audio.play", {path_k: "assets/sfx.wav"}),
                    ("nav.query_path", {"start": [0.0, 0.0, 0.0], "goal": [2.0, 0.0, 2.0]}),
                ]
                for tname, targs in stopped_calls:
                    _ = assert_structured_tool_or_param_error(client, tname, targs, tools_by_name)
                _ = client.call_tool("studio.pie_control", {"action": "play"})
                _ = client.call_tool("studio.pie_control", {"action": "stop"})
                _ = assert_structured_tool_or_param_error(client, "physics.trigger_events", {}, tools_by_name)
                case6_ok = True
            except Exception as exc:
                case6_ok, case6_err = False, str(exc)

            # Case 7: Active PIE with invalid entity_id, malformed nav.query_path, and -32602 envelope check
            try:
                _ = client.call_tool("studio.pie_control", {"action": "play"})
                bad_calls = [
                    ("anim.set_state", {ent_k: 999999, st_k: "idle"}),
                    ("anim.set_state", {ent_k: -1, st_k: "idle"}),
                    ("audio.play", {path_k: "assets/does_not_exist.wav"}),
                    ("nav.query_path", {"start": [1.0, 2.0], "goal": "bad"}),
                ]
                if "entity_id" in trig_props:
                    bad_calls.append(("physics.trigger_events", {"entity_id": -1}))
                for tname, targs in bad_calls:
                    _ = assert_structured_tool_or_param_error(client, tname, targs, tools_by_name)
                _ = client.call_tool("studio.pie_control", {"action": "stop"})
                _ = client.expect_protocol_error(
                    {"jsonrpc": "2.0", "id": 250, "method": "tools/call", "params": {"arguments": {}}},
                    -32602,
                )
                case7_ok = True
            except Exception as exc:
                case7_ok, case7_err = False, str(exc)

    # Record strictly AFTER `with McpStdioClient(...)` exits and close() succeeds
    record("t2_m2_mcp_runtime_rejected_when_pie_stopped", case6_ok, case6_err)
    record("t2_m2_mcp_runtime_invalid_entity_id_and_bad_params", case7_ok, case7_err)
except Exception as exc:
    close_msg = f"MCP session/close failed: {exc}"
    record("t2_m2_mcp_runtime_rejected_when_pie_stopped", False, f"{case6_err}; {close_msg}" if case6_err else close_msg)
    record("t2_m2_mcp_runtime_invalid_entity_id_and_bad_params", False, f"{case7_err}; {close_msg}" if case7_err else close_msg)

for cname in CASE_ORDER:
    if cname not in results:
        results[cname] = (False, "missing test outcome")

with open(out_tsv, "w", encoding="utf-8") as f:
    for cname in CASE_ORDER:
        ok, reason = results[cname]
        clean_reason = reason.replace("\t", " ").replace("\n", " ")
        f.write(f"{cname}\t{'1' if ok else '0'}\t{clean_reason}\n")
PY

    while IFS=$'\t' read -r cname cok creason; do
        [[ -z "$cname" ]] && continue
        if [[ "$cok" == "1" ]]; then
            pass_test "$cname"
        else
            fail_test "$cname" "$creason"
        fi
    done < "$tsv"
}

# ---------------------------------------------------------------------------
# Case 8 (F25): Corrupt BT JSON & GameFlow invalid save/transition/NaN boundaries
# ---------------------------------------------------------------------------
test_f25_bt_json_and_game_flow_boundary() {
    local probe_src="${TMPD}/bt_flow_boundary_probe.mojo"
    local probe_bin="${TMPD}/bin/bt_flow_boundary_probe"
    local save_dir="${TMPD}/neg_saves"
    mkdir -p "$save_dir"

    cat >"$probe_src" <<MOJO
from std.math import isfinite
from flint.ai import BT_SUCCESS, BehaviorTree, Blackboard, BtExecutionState
from flint.app import GameStateMachine, GameTimer, SaveSlotStore, Tween

def main() raises:
    # 1. Corrupt BT JSON must fail cleanly and preserve existing valid tree
    var bt = BehaviorTree()
    _ = bt.begin_sequence()
    _ = bt.add_action(7)
    bt.end_composite()

    if bt.load_json("") or bt.load_json("{bad json") or bt.load_json("{\"nodes\":\"not_a_list\"}"):
        raise Error("BehaviorTree.load_json unexpectedly accepted corrupt JSON")

    var bb = Blackboard()
    var st = BtExecutionState()
    st.set_action_status(7, BT_SUCCESS)
    if bt.tick(bb, st, Float32(1.0 / 60.0)) != BT_SUCCESS:
        raise Error("failed BehaviorTree.load_json corrupted pre-existing valid tree")

    # 2. GameStateMachine rejects unknown/disallowed transitions
    var fsm = GameStateMachine("Menu")
    fsm.allow_transition("Menu", "Play")
    if fsm.transition_to("Victory") or fsm.current_state() != "Menu":
        raise Error("GameStateMachine allowed invalid Menu -> Victory transition")

    # 3. SaveSlotStore rejects missing slot & path traversal
    var store = SaveSlotStore("${save_dir}")
    if store.load_slot("does_not_exist").byte_length() != 0:
        raise Error("SaveSlotStore returned non-empty data for missing slot")
    if store.save_slot("../escape_slot", "{}"):
        raise Error("SaveSlotStore allowed path-traversal slot name")

    # 4. GameTimer & Tween handle NaN / non-positive dt safely
    var nan_dt = Float32(0.0) / Float32(0.0)
    var timer = GameTimer(Float32(0.5), repeating=False)
    if timer.step(Float32(-0.5)) or timer.step(nan_dt):
        raise Error("GameTimer fired under negative or NaN dt")

    var tw = Tween(Float32(0.0), Float32(10.0), Float32(0.5))
    var v_neg = tw.step(Float32(-0.25))
    var v_nan = tw.step(nan_dt)
    if not isfinite(v_neg) or not isfinite(v_nan) or v_nan != Float32(0.0) or tw.is_finished():
        raise Error("Tween corrupted state under negative/NaN dt")
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >/dev/null 2>&1; then
        fail_test "t2_m2_f25_bt_json_and_game_flow_boundary" "M5 BehaviorTree.load_json / GameFlow boundary contract pending"
        return
    fi

    local out="" rc=0
    out="$("$probe_bin" 2>&1)" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t2_m2_f25_bt_json_and_game_flow_boundary" "probe failed (rc=$rc): $out"
        return
    fi
    if [[ -e "${TMPD}/escape_slot" || -e "${TMPD}/escape_slot.json" ]]; then
        fail_test "t2_m2_f25_bt_json_and_game_flow_boundary" "SaveSlotStore traversal leaked file outside save_dir"
        return
    fi
    pass_test "t2_m2_f25_bt_json_and_game_flow_boundary"
}

test_f22_gameplay_so_missing_project_and_broken_source
test_f22_abi_version_and_magic_size_mismatch_rejected
test_f22_nonexistent_corrupt_so_and_on_load_error
test_f22_pod_schema_layout_mismatch_rejected
test_f23_reflection_inspector_invalid_add_remove
run_m2_mcp_negative_cases
test_f25_bt_json_and_game_flow_boundary

print_summary_and_exit
