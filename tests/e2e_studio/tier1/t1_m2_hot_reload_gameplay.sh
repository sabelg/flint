#!/usr/bin/env bash
# AREA: m2
# Tier 1 Feature Coverage: Area D — M2 gameplay-so CLI, FlintPluginApiV2 Hot-Reload, Reflection, Triggers, BT, InputMap UI & GameFlow (F22–F25)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t1_m2_hr"
resolve_mojo_env || {
    fail_test "t1_m2_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

# ---------------------------------------------------------------------------
# Case 1 (F22): `tools/flint.sh gameplay-so` produces valid ELF ET_DYN .so (API v2)
# ---------------------------------------------------------------------------
test_f22_gameplay_so_cli_produces_valid_elf_plugin() {
    local proj="${TMPD}/proj_hr"
    local new_out new_rc=0
    new_out="$(bash "${ROOT}/tools/flint.sh" new "$proj" 2>&1)" || new_rc=$?
    if [[ "$new_rc" -ne 0 || ! -f "${proj}/flint.toml" ]]; then
        fail_test "t1_m2_f22_gameplay_so_cli_produces_valid_elf_plugin" "flint.sh new failed (rc=$new_rc): ${new_out}"
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

    local gs_out gs_rc=0
    gs_out="$(cd "$proj" && bash "${ROOT}/tools/flint.sh" gameplay-so 2>&1)" || gs_rc=$?
    if [[ "$gs_rc" -ne 0 ]]; then
        fail_test "t1_m2_f22_gameplay_so_cli_produces_valid_elf_plugin" "flint.sh gameplay-so exited ${gs_rc}: ${gs_out}"
        return
    fi
    if grep -q "Usage:" <<<"$gs_out"; then
        fail_test "t1_m2_f22_gameplay_so_cli_produces_valid_elf_plugin" "flint.sh gameplay-so fell through to Usage banner (subcommand pending M5)"
        return
    fi

    local so_path=""
    so_path="$(find "${proj}/build" -maxdepth 3 -type f -name '*.so' -size +0c 2>/dev/null | head -n 1 || true)"
    if [[ -z "$so_path" || ! -f "$so_path" ]]; then
        fail_test "t1_m2_f22_gameplay_so_cli_produces_valid_elf_plugin" "no non-empty .so produced in ${proj}/build"
        return
    fi

    local elf_err=""
    if ! elf_err="$(python3 -B - "$so_path" <<'PY' 2>&1
import ctypes, struct, sys

path = sys.argv[1]
with open(path, "rb") as f:
    hdr = f.read(20)
if len(hdr) < 20 or hdr[:4] != b"\x7fELF":
    raise SystemExit(f"Not an ELF file: {path}")
e_type = struct.unpack("<H", hdr[16:18])[0]
if e_type != 3:  # ET_DYN
    raise SystemExit(f"Expected ET_DYN (3), got e_type={e_type} in {path}")
lib = ctypes.CDLL(path)
for sym in ("flint_plugin_on_load", "flint_plugin_on_step", "flint_plugin_on_unload"):
    if not hasattr(lib, sym):
        raise SystemExit(f"Missing required plugin symbol {sym} in {path}")
ver_fn = lib.flint_plugin_api_version
ver_fn.restype = ctypes.c_uint32
ver = ver_fn()
if ver != 2:
    raise SystemExit(f"Expected flint_plugin_api_version() == 2, got {ver}")
PY
)"; then
        fail_test "t1_m2_f22_gameplay_so_cli_produces_valid_elf_plugin" "ELF/ABI check failed: ${elf_err}"
        return
    fi

    pass_test "t1_m2_f22_gameplay_so_cli_produces_valid_elf_plugin"
}

# ---------------------------------------------------------------------------
# Case 2 (F22): HotReloadHost v2 swap preserves World entities, PodComponentSchema & Resources
# ---------------------------------------------------------------------------
test_f22_hot_reload_host_v2_swap_preserves_state() {
    local probe_src="${TMPD}/hr_v2_probe.mojo"
    local probe_bin="${TMPD}/bin/hr_v2_probe"
    local probe_log="${TMPD}/hr_v2_probe.compile.log"

    cat >"$probe_src" <<MOJO
from std.collections import List
from std.ffi import external_call
from std.sys import size_of
from flint.app import (
    FLINT_PLUGIN_API_VERSION,
    FlintPluginApiV2,
    HotReloadHost,
    PodComponentSchema,
    PodFieldDesc,
    Resources,
)
from flint.ecs import STORAGE_TABLE, World
from flint.studio import FIELD_F32

def _write_text(path: String, text: String) -> Bool:
    from flint.platform.os import write_bytes
    var buf = List[UInt8](capacity=text.byte_length())
    for b in text.as_bytes():
        buf.append(b)
    return write_bytes(path, buf)

def _guest_src(rate: String) -> String:
    var s = String("from flint.app import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2, PodComponentSchema, PodFieldDesc\n")
    s += "from flint.studio import FIELD_F32\n\n"
    s += "comptime COUNTER_RES_ID: UInt64 = UInt64(0xE4E2670000000011)\n\n"
    s += "@export(\"flint_plugin_api_version\")\n"
    s += "def flint_plugin_api_version() abi(\"C\") -> UInt32:\n"
    s += "    return FLINT_PLUGIN_API_VERSION\n\n"
    s += "@export(\"flint_plugin_on_load\")\n"
    s += "def flint_plugin_on_load(api_ptr: Int, _world_ptr: Int, res_ptr: Int, is_reload: Int32) abi(\"C\") -> Int32:\n"
    s += "    if api_ptr == 0:\n"
    s += "        return -2\n"
    s += "    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
    s += "    if not api.is_valid():\n"
    s += "        return -2\n"
    s += "    var schema = PodComponentSchema(\"HostSwapEnergy\", 4, 4)\n"
    s += "    if not schema.add_field(PodFieldDesc(\"energy\", FIELD_F32, 0, 4)) or not schema.validate():\n"
    s += "        return -3\n"
    s += "    if is_reload == 0:\n"
    s += "        var init_val = List[Int64](length=1, fill=Int64(5))\n"
    s += "        var mem = api.upsert_pod_resource(res_ptr, COUNTER_RES_ID, 8, Int(init_val.unsafe_ptr()))\n"
    s += "        _ = len(init_val)\n"
    s += "        if mem == 0:\n"
    s += "            return -4\n"
    s += "    else:\n"
    s += "        if api.get_resource_ptr(res_ptr, COUNTER_RES_ID) == 0:\n"
    s += "            return -5\n"
    s += "    return 0\n\n"
    s += "@export(\"flint_plugin_on_step\")\n"
    s += "def flint_plugin_on_step(api_ptr: Int, _world_ptr: Int, res_ptr: Int, _dt: Float32) abi(\"C\") -> Int32:\n"
    s += "    if api_ptr == 0:\n"
    s += "        return -2\n"
    s += "    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
    s += "    if not api.is_valid():\n"
    s += "        return -2\n"
    s += "    var rptr = api.get_resource_ptr(res_ptr, COUNTER_RES_ID)\n"
    s += "    if rptr == 0:\n"
    s += "        return -5\n"
    s += "    var ctr = Pointer[Int64, MutAnyOrigin](unsafe_from_address=rptr)\n"
    s += "    ctr[unsafe_offset=0] += Int64(" + rate + ")\n"
    s += "    return Int32(" + rate + ")\n\n"
    s += "@export(\"flint_plugin_on_unload\")\n"
    s += "def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi(\"C\") -> None:\n"
    s += "    pass\n"
    return s^

def main() raises:
    if Int(FLINT_PLUGIN_API_VERSION) != 2:
        raise Error("FLINT_PLUGIN_API_VERSION must be 2")
    var api_v2 = FlintPluginApiV2()
    if not api_v2.is_valid() or Int(api_v2.struct_size) != size_of[FlintPluginApiV2]():
        raise Error("FlintPluginApiV2 baseline validation failed")

    var schema = PodComponentSchema("HostSwapEnergy", 4, 4)
    if not schema.add_field(PodFieldDesc("energy", FIELD_F32, 0, 4)) or not schema.validate():
        raise Error("PodComponentSchema validation failed")

    var s1 = "${TMPD}/hr_v1.mojo"
    var s2 = "${TMPD}/hr_v2.mojo"
    var so1 = "${TMPD}/hr_v1.so"
    var so2 = "${TMPD}/hr_v2.so"
    if not _write_text(s1, _guest_src("10")) or not _write_text(s2, _guest_src("100")):
        raise Error("failed writing guest sources")
    var bcmd1 = "${MOJO_BIN} build -I ${ROOT}/src " + s1 + " --emit shared-lib -o " + so1 + " -O0 --Werror"
    var bcmd2 = "${MOJO_BIN} build -I ${ROOT}/src " + s2 + " --emit shared-lib -o " + so2 + " -O0 --Werror"
    if external_call["system", Int32](bcmd1.as_c_string_span()) != 0 or external_call["system", Int32](bcmd2.as_c_string_span()) != 0:
        raise Error("failed building guest v1/v2 .so")

    var world = World()
    var tid = world.register_pod_component(UInt64(0xE4E2670000000001), 4, 4, STORAGE_TABLE)
    var ent = world.spawn_empty()
    var resources = Resources()
    var host = HotReloadHost("${TMPD}/hr_v2_stage")
    if not host.load_or_reload(so1, world, resources):
        raise Error("load v1 failed: " + host.last_error)
    if host.last_reload_ms >= 300.0 or host.step(world, resources, Float32(0.5)) != 10:
        raise Error("v1 timing or step mismatch")
    var rptr1 = resources.get_raw_ptr(UInt64(0xE4E2670000000011))
    if rptr1 == 0 or Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=rptr1)[unsafe_offset=0] != 15:
        raise Error("v1 guest resource state mismatch (expected 15)")

    if not host.load_or_reload(so2, world, resources):
        raise Error("reload v2 failed: " + host.last_error)
    if host.reload_generation != 2 or host.last_reload_ms >= 300.0 or not world.is_alive(ent) or tid < 0:
        raise Error("v2 reload did not preserve state")
    if host.step(world, resources, Float32(0.5)) != 100:
        raise Error("v2 step behavior not updated")
    var rptr2 = resources.get_raw_ptr(UInt64(0xE4E2670000000011))
    if rptr2 == 0 or Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=rptr2)[unsafe_offset=0] != 115:
        raise Error("v2 reload did not preserve resource counter + apply +100 step delta (expected 115)")
    if not host.unload(world, resources):
        raise Error("unload failed")
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >"$probe_log" 2>&1; then
        fail_test "t1_m2_f22_hot_reload_host_v2_swap_preserves_state" "M5 FlintPluginApiV2/PodComponentSchema contract pending"
        return
    fi

    local run_out="" run_rc=0
    run_out="$("$probe_bin" 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 ]]; then
        fail_test "t1_m2_f22_hot_reload_host_v2_swap_preserves_state" "probe failed (rc=$run_rc): ${run_out}"
        return
    fi

    pass_test "t1_m2_f22_hot_reload_host_v2_swap_preserves_state"
}

# ---------------------------------------------------------------------------
# Shared AOT binary for Case 3 + Case 6 (avoids compiling flint.studio twice under -O3)
# ---------------------------------------------------------------------------
STUDIO_PROBES_BIN="${TMPD}/bin/studio_probes"
STUDIO_PROBES_LOG="${TMPD}/studio_probes.compile.log"
STUDIO_PROBES_STATE=0  # 0=unbuilt, 1=ok, -1=failed

ensure_studio_probes_bin() {
    if [[ "$STUDIO_PROBES_STATE" -eq 1 ]]; then
        return 0
    fi
    if [[ "$STUDIO_PROBES_STATE" -eq -1 ]]; then
        return 1
    fi
    local probe_src="${TMPD}/studio_probes.mojo"
    cat >"$probe_src" <<'MOJO'
from std.sys import argv
from flint.app import FLINT_PLUGIN_API_VERSION, PodComponentSchema, PodFieldDesc
from flint.input import ACTION_BOOL, InputContext, InputMap
from flint.platform.input import InputState
from flint.render.font import SdfFont
from flint.render.plugin import PointLightComponent
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD
from flint.studio import (
    FIELD_F32,
    FIELD_INT,
    FlintStudioState,
    GIZMO_MODE_TRANSLATE,
    PIE_PLAYING,
    emit_studio_ui_draw_list,
)
from flint.ui.context import UI_KEY_DELETE
from flint.ui.draw_list import UiDrawList

def _run_pie_hr(so_v1: String, so_v2: String) raises:
    if Int(FLINT_PLUGIN_API_VERSION) != 2:
        raise Error("FLINT_PLUGIN_API_VERSION must be 2")
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var eid = studio.spawn_entity("ReflectionTarget")
    if not studio.is_alive_id(eid):
        raise Error("spawn_entity failed")

    # 1. PodComponentSchema + PodFieldDesc registration & validation
    var schema = PodComponentSchema("GameplayStats", 8, 4)
    if not schema.add_field(PodFieldDesc("hp", FIELD_F32, 0, 4)):
        raise Error("add_field hp failed")
    if not schema.add_field(PodFieldDesc("score", FIELD_INT, 4, 4)):
        raise Error("add_field score failed")
    if not schema.validate() or not studio.register_pod_schema(schema):
        raise Error("PodComponentSchema registration failed")

    # 2. Reflection Inspector Add/Remove Component on live entity (built-in + POD schema)
    if not studio.add_component_by_name(eid, "PointLightComponent"):
        raise Error("add_component_by_name(PointLightComponent) failed")
    var ent = studio.resolve_entity(eid)
    if not studio.app.world.has[PointLightComponent](ent):
        raise Error("PointLightComponent not present after add_component_by_name")
    if not studio.remove_component_by_name(eid, "PointLightComponent"):
        raise Error("remove_component_by_name(PointLightComponent) failed")
    if studio.app.world.has[PointLightComponent](ent):
        raise Error("PointLightComponent still present after remove_component_by_name")
    if not studio.add_component_by_name(eid, "GameplayStats"):
        raise Error("add_component_by_name(GameplayStats) failed")
    if not studio.remove_component_by_name(eid, "GameplayStats"):
        raise Error("remove_component_by_name(GameplayStats) failed")

    # 3. Studio PIE hot-reload v1 -> v2 WITHOUT recreating FlintStudioState
    if not studio.pie_play() or studio.pie.mode != PIE_PLAYING:
        raise Error("pie_play failed")
    if not studio.load_gameplay_plugin(so_v1):
        raise Error("Studio PIE load v1 .so failed")
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    var r_v1 = studio.app.resources.get_raw_ptr(UInt64(0xE4E26700000000A1))
    if r_v1 == 0 or Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=r_v1)[unsafe_offset=0] != 12:
        raise Error("v1 Studio PIE step behavior did not run (expected counter 12)")

    # Swap to v2 on the SAME FlintStudioState while PIE stays active
    if not studio.reload_gameplay_plugin(so_v2):
        raise Error("Studio PIE hot-reload v2 .so failed")
    if studio.pie.mode != PIE_PLAYING or not studio.is_alive_id(eid):
        raise Error("Studio PIE exited or entity lost on hot-reload")
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    var r_v2 = studio.app.resources.get_raw_ptr(UInt64(0xE4E26700000000A1))
    if r_v2 == 0 or Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=r_v2)[unsafe_offset=0] != 212:
        raise Error("v2 Studio PIE reload did not preserve state & apply +100 step delta (expected 212)")

    if not studio.pie_stop() or not studio.is_alive_id(eid):
        raise Error("pie_stop failed after hot-reload")
    studio.shutdown()

def _run_input_map() raises:
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=640, height=480)
    var eid = studio.spawn_entity("InputMapUiTarget")
    studio.select_entity(eid)
    studio.gizmo_mode = GIZMO_MODE_TRANSLATE

    var ctx = InputContext("Gameplay", 0, False)
    ctx.bind_key("Jump", 32)  # Space
    _ = studio.input_map.register_action("Jump", ACTION_BOOL)
    studio.input_map.add_context(ctx^)
    studio.input_map.push_context("Gameplay")

    studio.open_input_map_panel()
    if not studio.begin_input_rebind("Gameplay", "Jump"):
        raise Error("begin_input_rebind failed")

    var ui = UiDrawList()
    var font = SdfFont()
    var inp_neutral = InputState()
    inp_neutral.begin_frame()
    emit_studio_ui_draw_list(ui, studio, 0, font, 640, 480, input=inp_neutral, dt=Float32(1.0 / 60.0))
    if len(studio.ui_ctx.draw_list.prims) == 0:
        raise Error("InputMap UI panel emitted zero draw primitives")
    if not studio.ui_ctx.wants_keyboard():
        raise Error("InputMap UI rebind capture did not assert ui_ctx.wants_keyboard()")

    # Editor hotkeys must be suppressed while UI panel captures keyboard
    var inp_capture = InputState()
    inp_capture.begin_frame()
    inp_capture.set_logical_key(UI_KEY_DELETE, True)
    inp_capture.set_logical_key(69, True)  # 'E' editor rotate hotkey
    _ = studio.process_editor_input(inp_capture, Float32(1.0 / 60.0))
    if not studio.is_alive_id(eid) or studio.gizmo_mode != GIZMO_MODE_TRANSLATE:
        raise Error("editor hotkeys leaked while InputMap UI panel captured keyboard")

    # Rebind to 'K' (75) and verify persistence via save_json/load_json
    var inp_rebind = InputState()
    inp_rebind.begin_frame()
    inp_rebind.set_logical_key(75, True)
    emit_studio_ui_draw_list(ui, studio, 0, font, 640, 480, input=inp_rebind, dt=Float32(1.0 / 60.0))

    var saved_json = studio.input_map.save_json()
    var reloaded = InputMap()
    if not reloaded.load_json(saved_json):
        raise Error("rebound InputMap save_json/load_json failed")
    var inp_test = InputState()
    inp_test.begin_frame()
    inp_test.set_logical_key(75, True)
    reloaded.update(inp_test, Float32(1.0 / 60.0))
    if not reloaded.just_pressed("Jump"):
        raise Error("rebound key 'K' did not trigger Jump after InputMap UI rebind + JSON reload")
    studio.shutdown()

def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: studio_probes <pie_hr|input_map> ...")
    var mode = String(args[1])
    if mode == "pie_hr":
        if len(args) < 4:
            raise Error("usage: studio_probes pie_hr <so_v1> <so_v2>")
        _run_pie_hr(String(args[2]), String(args[3]))
    elif mode == "input_map":
        _run_input_map()
    else:
        raise Error("unknown studio_probes mode: " + mode)
MOJO

    if "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$STUDIO_PROBES_BIN" >"$STUDIO_PROBES_LOG" 2>&1; then
        STUDIO_PROBES_STATE=1
        return 0
    fi
    STUDIO_PROBES_STATE=-1
    return 1
}

# ---------------------------------------------------------------------------
# Case 3 (F22 + F23): Studio PIE hot-reload without restart & reflection Add/Remove
# ---------------------------------------------------------------------------
test_f22_studio_pie_hot_reload_without_restart() {
    local guest_v1="${TMPD}/pie_hr_v1.mojo"
    local guest_v2="${TMPD}/pie_hr_v2.mojo"
    local so_v1="${TMPD}/pie_hr_v1.so"
    local so_v2="${TMPD}/pie_hr_v2.so"
    local probe_log="${TMPD}/studio_pie_hr_probe.compile.log"

    cat >"$guest_v1" <<'MOJO'
from std.collections import List
from flint.app import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2

comptime PIE_CTR_ID: UInt64 = UInt64(0xE4E26700000000A1)

@export("flint_plugin_api_version")
def flint_plugin_api_version() abi("C") -> UInt32:
    return FLINT_PLUGIN_API_VERSION

@export("flint_plugin_on_load")
def flint_plugin_on_load(api_ptr: Int, _world_ptr: Int, res_ptr: Int, is_reload: Int32) abi("C") -> Int32:
    if api_ptr == 0:
        return -2
    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]
    if not api.is_valid():
        return -2
    if is_reload == 0:
        var init_val = List[Int64](length=1, fill=Int64(10))
        var mem = api.upsert_pod_resource(res_ptr, PIE_CTR_ID, 8, Int(init_val.unsafe_ptr()))
        _ = len(init_val)
        if mem == 0:
            return -3
    elif api.get_resource_ptr(res_ptr, PIE_CTR_ID) == 0:
        return -4
    return 0

@export("flint_plugin_on_step")
def flint_plugin_on_step(api_ptr: Int, _world_ptr: Int, res_ptr: Int, _dt: Float32) abi("C") -> Int32:
    if api_ptr == 0:
        return -2
    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]
    if not api.is_valid():
        return -2
    var rptr = api.get_resource_ptr(res_ptr, PIE_CTR_ID)
    if rptr == 0:
        return -4
    Pointer[Int64, MutAnyOrigin](unsafe_from_address=rptr)[unsafe_offset=0] += Int64(1)
    return 0

@export("flint_plugin_on_unload")
def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> None:
    pass
MOJO

    cat >"$guest_v2" <<'MOJO'
from std.collections import List
from flint.app import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2

comptime PIE_CTR_ID: UInt64 = UInt64(0xE4E26700000000A1)

@export("flint_plugin_api_version")
def flint_plugin_api_version() abi("C") -> UInt32:
    return FLINT_PLUGIN_API_VERSION

@export("flint_plugin_on_load")
def flint_plugin_on_load(api_ptr: Int, _world_ptr: Int, res_ptr: Int, is_reload: Int32) abi("C") -> Int32:
    if api_ptr == 0:
        return -2
    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]
    if not api.is_valid():
        return -2
    if is_reload == 0:
        var init_val = List[Int64](length=1, fill=Int64(10))
        var mem = api.upsert_pod_resource(res_ptr, PIE_CTR_ID, 8, Int(init_val.unsafe_ptr()))
        _ = len(init_val)
        if mem == 0:
            return -3
    elif api.get_resource_ptr(res_ptr, PIE_CTR_ID) == 0:
        return -4
    return 0

@export("flint_plugin_on_step")
def flint_plugin_on_step(api_ptr: Int, _world_ptr: Int, res_ptr: Int, _dt: Float32) abi("C") -> Int32:
    if api_ptr == 0:
        return -2
    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]
    if not api.is_valid():
        return -2
    var rptr = api.get_resource_ptr(res_ptr, PIE_CTR_ID)
    if rptr == 0:
        return -4
    Pointer[Int64, MutAnyOrigin](unsafe_from_address=rptr)[unsafe_offset=0] += Int64(100)
    return 0

@export("flint_plugin_on_unload")
def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> None:
    pass
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "$guest_v1" --emit shared-lib -o "$so_v1" -O0 --Werror >"${probe_log}.v1" 2>&1; then
        fail_test "t1_m2_f22_studio_pie_hot_reload_without_restart" "M5 FlintPluginApiV2 guest contract pending"
        return
    fi
    if ! "$MOJO_BIN" build -I "${ROOT}/src" "$guest_v2" --emit shared-lib -o "$so_v2" -O0 --Werror >"${probe_log}.v2" 2>&1; then
        fail_test "t1_m2_f22_studio_pie_hot_reload_without_restart" "M5 FlintPluginApiV2 guest v2 contract pending"
        return
    fi
    if ! ensure_studio_probes_bin; then
        fail_test "t1_m2_f22_studio_pie_hot_reload_without_restart" "M5 PIE HotReloadHost / Reflection Inspector contract pending"
        return
    fi

    local run_out="" run_rc=0
    run_out="$(FLINT_STATE_DIR="${TMPD}/flint_state" XDG_STATE_HOME="${TMPD}/xdg_state" "$STUDIO_PROBES_BIN" pie_hr "$so_v1" "$so_v2" 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 ]]; then
        fail_test "t1_m2_f22_studio_pie_hot_reload_without_restart" "probe failed (rc=$run_rc): ${run_out}"
        return
    fi

    pass_test "t1_m2_f22_studio_pie_hot_reload_without_restart"
}

# ---------------------------------------------------------------------------
# Case 4 (F24): entity_from_body, OnEnter & OnExit trigger helpers with KCC
# ---------------------------------------------------------------------------
test_f24_entity_from_body_and_on_enter_exit() {
    local probe_src="${TMPD}/trigger_helpers_probe.mojo"
    local probe_bin="${TMPD}/bin/trigger_helpers_probe"
    local probe_log="${TMPD}/trigger_helpers_probe.compile.log"

    cat >"$probe_src" <<'MOJO'
from flint.app import AppConfig, EngineApp
from flint.core.math import Vec3
from flint.ecs import NULL_ENTITY
from flint.physics.character import CharacterControllerConfig
from flint.physics.plugin import (
    CharacterControllerComponent,
    ColliderComponent,
    OnEnter,
    OnExit,
    PhysicsPlugin,
    PhysicsSyncStateResource,
    TriggerEventsResource,
    entity_from_body,
)
from flint.scene import GlobalTransform, LocalTransform, PrevGlobalTransform, TransformPlugin

def main() raises:
    var app = EngineApp(AppConfig(title="t1_trig_helpers", headless=True, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.startup()

    var trig_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(2.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](trig_e, GlobalTransform.from_local(LocalTransform(Vec3(2.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](trig_e, PrevGlobalTransform())
    _ = app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(0.8, 1.5, 0.8), is_trigger=True))

    var kcc_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](kcc_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](kcc_e, GlobalTransform.from_local(LocalTransform(Vec3(0.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](kcc_e, PrevGlobalTransform())
    _ = app.world.insert[CharacterControllerComponent](
        kcc_e,
        CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), CharacterControllerConfig(), Vec3(6.0, 0.0, 0.0)),
    )

    var cursor = 0
    var enter_count = 0
    var exit_count = 0
    var body_resolved_ok = False

    for _ in range(60):
        app.step_one_frame(Float32(1.0 / 60.0))
        var tev_opt = app.resources.get_ptr[TriggerEventsResource]()
        var sync_opt = app.resources.get_ptr[PhysicsSyncStateResource]()
        if tev_opt and sync_opt:
            ref sync_state = sync_opt.value()[unsafe_offset=0]
            if entity_from_body(sync_state, -1) != NULL_ENTITY or entity_from_body(sync_state, 999999) != NULL_ENTITY:
                raise Error("entity_from_body failed to return NULL_ENTITY for invalid body_id")
            var evs = tev_opt.value()[unsafe_offset=0].events.read_since(cursor)
            for i in range(len(evs)):
                var ev = evs[i]
                if OnEnter(ev):
                    enter_count += 1
                    if entity_from_body(sync_state, ev.body_id) == kcc_e:
                        body_resolved_ok = True
                if OnExit(ev):
                    exit_count += 1

    if enter_count != 1 or exit_count != 1 or not body_resolved_ok:
        raise Error(
            "trigger helper mismatch: enter="
            + String(enter_count)
            + " exit="
            + String(exit_count)
            + " body_resolved="
            + String(body_resolved_ok)
        )
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >"$probe_log" 2>&1; then
        fail_test "t1_m2_f24_entity_from_body_and_on_enter_exit" "M5 entity_from_body / OnEnter / OnExit contract pending"
        return
    fi

    local run_out="" run_rc=0
    run_out="$("$probe_bin" 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 ]]; then
        fail_test "t1_m2_f24_entity_from_body_and_on_enter_exit" "probe failed (rc=$run_rc): ${run_out}"
        return
    fi

    pass_test "t1_m2_f24_entity_from_body_and_on_enter_exit"
}

# ---------------------------------------------------------------------------
# Case 5 (F25): BehaviorTreeComponent + BT JSON save/reload roundtrip & reactive tick
# ---------------------------------------------------------------------------
test_f25_bt_component_json_roundtrip() {
    local probe_src="${TMPD}/bt_comp_roundtrip_probe.mojo"
    local probe_bin="${TMPD}/bin/bt_comp_roundtrip_probe"
    local probe_log="${TMPD}/bt_comp_roundtrip_probe.compile.log"

    cat >"$probe_src" <<MOJO
from flint.ai import (
    BT_ABORT_LOWER_PRIORITY,
    BT_CMP_GT,
    BT_FAILURE,
    BT_RUNNING,
    BT_SUCCESS,
    BehaviorTree,
    BehaviorTreeComponent,
    Blackboard,
    BtExecutionState,
    register_ai_codecs,
)
from flint.app import AppConfig, EngineApp
from flint.platform.os import read_bytes, write_bytes
from flint.scene import ComponentRegistry, Scene

def _write_text(path: String, text: String) -> Bool:
    var buf = List[UInt8](capacity=text.byte_length())
    for b in text.as_bytes():
        buf.append(b)
    return write_bytes(path, buf)

def _read_text(path: String) -> String:
    var b = read_bytes(path)
    return String(unsafe_from_utf8=b)

def main() raises:
    var bb = Blackboard()
    var k_alert = bb.def_key("alert", Float32(0.0))

    var bt = BehaviorTree()
    _ = bt.begin_selector()
    _ = bt.begin_sequence()
    _ = bt.add_blackboard_cmp(k_alert, BT_CMP_GT, Float32(0.5), BT_ABORT_LOWER_PRIORITY)
    _ = bt.add_action(10)
    bt.end_composite()
    _ = bt.add_action(2)
    bt.end_composite()

    var bt_json_path = "${TMPD}/enemy_bt.json"
    if not _write_text(bt_json_path, bt.save_json()):
        raise Error("failed writing BT JSON")

    var bt_reloaded = BehaviorTree()
    if not bt_reloaded.load_json(_read_text(bt_json_path)):
        raise Error("BehaviorTree.load_json failed on saved BT JSON")

    var reg = ComponentRegistry.with_builtins()
    register_ai_codecs(reg)

    var app1 = EngineApp(AppConfig(title="t1_bt_save", headless=True))
    reg.ensure_registered_in_world(app1.world)
    var e1 = app1.world.spawn_empty()
    _ = app1.world.insert[BehaviorTreeComponent](e1, BehaviorTreeComponent(bt_reloaded.copy()))

    var scn_path = "${TMPD}/bt_scene.flscn"
    var scn1 = Scene("BtLevel")
    if not scn1.save_flscn_file(scn_path, app1.world, reg):
        raise Error("failed saving scene with BehaviorTreeComponent")

    var app2 = EngineApp(AppConfig(title="t1_bt_load", headless=True))
    var scn2 = Scene()
    var loaded = scn2.load_flscn_file(scn_path, app2.world, reg)
    if len(loaded) != 1 or not app2.world.has[BehaviorTreeComponent](loaded[0]):
        raise Error("BehaviorTreeComponent lost across Scene roundtrip")

    var comp_ptr = app2.world.get_ptr_mut[BehaviorTreeComponent](loaded[0])
    if not comp_ptr:
        raise Error("BehaviorTreeComponent pointer missing")

    var st = BtExecutionState()
    st.set_action_status(2, BT_RUNNING)
    st.set_action_status(10, BT_SUCCESS)

    var r1 = comp_ptr.value()[unsafe_offset=0].tree.tick(bb, st, Float32(1.0 / 60.0))
    if r1 != BT_RUNNING or st.active_action_id != 2:
        raise Error("reloaded BehaviorTreeComponent did not run expected action 2")

    bb.set_float(k_alert, Float32(1.0))
    var r2 = comp_ptr.value()[unsafe_offset=0].tree.tick(bb, st, Float32(1.0 / 60.0))
    if r2 != BT_SUCCESS or st.aborts_triggered != 1 or st.active_action_id != 10:
        raise Error("reloaded BehaviorTreeComponent reactive abort failed")
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >"$probe_log" 2>&1; then
        fail_test "t1_m2_f25_bt_component_json_roundtrip" "M5 BehaviorTreeComponent / BT JSON roundtrip contract pending"
        return
    fi

    local run_out="" run_rc=0
    run_out="$("$probe_bin" 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 ]]; then
        fail_test "t1_m2_f25_bt_component_json_roundtrip" "probe failed (rc=$run_rc): ${run_out}"
        return
    fi

    pass_test "t1_m2_f25_bt_component_json_roundtrip"
}

# ---------------------------------------------------------------------------
# Case 6 (F25): InputMap Studio UI panel render, rebind, wants_keyboard() capture & save
# ---------------------------------------------------------------------------
test_f25_input_map_ui_panel_rebind_and_capture() {
    if ! ensure_studio_probes_bin; then
        fail_test "t1_m2_f25_input_map_ui_panel_rebind_and_capture" "M5 InputMap UI panel contract pending"
        return
    fi

    local run_out="" run_rc=0
    run_out="$(FLINT_STATE_DIR="${TMPD}/flint_state" XDG_STATE_HOME="${TMPD}/xdg_state" "$STUDIO_PROBES_BIN" input_map 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 ]]; then
        fail_test "t1_m2_f25_input_map_ui_panel_rebind_and_capture" "probe failed (rc=$run_rc): ${run_out}"
        return
    fi

    pass_test "t1_m2_f25_input_map_ui_panel_rebind_and_capture"
}

# ---------------------------------------------------------------------------
# Case 7 (F25): GameFlow primitives — FSM, scene transitions, save slots, timers & tweens
# ---------------------------------------------------------------------------
test_f25_game_flow_fsm_scene_save_timers_tweens() {
    local probe_src="${TMPD}/game_flow_probe.mojo"
    local probe_bin="${TMPD}/bin/game_flow_probe"
    local probe_log="${TMPD}/game_flow_probe.compile.log"
    local save_dir="${TMPD}/saves"
    mkdir -p "$save_dir"

    cat >"$probe_src" <<MOJO
from std.math import abs
from flint.app import (
    AppConfig,
    EngineApp,
    GameFlow,
    GameStateMachine,
    GameTimer,
    SaveSlotStore,
    Tween,
)
from flint.scene import ComponentRegistry, Scene
from flint.scene.transform_hierarchy import LocalTransform, TransformPlugin
from flint.core.math import Vec3

def main() raises:
    # 1. State machine valid & invalid transitions
    var fsm = GameStateMachine("Menu")
    fsm.allow_transition("Menu", "Play")
    fsm.allow_transition("Play", "Pause")
    fsm.allow_transition("Pause", "Play")
    fsm.allow_transition("Play", "Victory")
    if not fsm.transition_to("Play") or fsm.current_state() != "Play":
        raise Error("FSM Menu -> Play transition failed")
    if fsm.transition_to("Menu"):
        raise Error("FSM disallowed Play -> Menu transition unexpectedly succeeded")
    if not fsm.transition_to("Pause") or not fsm.transition_to("Play") or not fsm.transition_to("Victory"):
        raise Error("FSM Play -> Pause -> Play -> Victory chain failed")

    # 2. Runtime scene transition in EngineApp (Scene A -> Scene B)
    var reg = ComponentRegistry.with_builtins()
    var app_a = EngineApp(AppConfig(title="scene_a", headless=True))
    app_a.add_plugin(TransformPlugin())
    reg.ensure_registered_in_world(app_a.world)
    var ea = app_a.world.spawn_empty()
    _ = app_a.world.insert[LocalTransform](ea, LocalTransform(Vec3(1.0, 2.0, 3.0)))
    var path_a = "${TMPD}/scene_a.flscn"
    if not Scene("SceneA").save_flscn_file(path_a, app_a.world, reg):
        raise Error("failed saving scene_a")

    var app_b = EngineApp(AppConfig(title="scene_b", headless=True))
    app_b.add_plugin(TransformPlugin())
    reg.ensure_registered_in_world(app_b.world)
    var eb1 = app_b.world.spawn_empty()
    var eb2 = app_b.world.spawn_empty()
    _ = app_b.world.insert[LocalTransform](eb1, LocalTransform(Vec3(10.0, 0.0, 0.0)))
    _ = app_b.world.insert[LocalTransform](eb2, LocalTransform(Vec3(20.0, 0.0, 0.0)))
    var path_b = "${TMPD}/scene_b.flscn"
    if not Scene("SceneB").save_flscn_file(path_b, app_b.world, reg):
        raise Error("failed saving scene_b")

    var live_app = EngineApp(AppConfig(title="flow_live", headless=True))
    live_app.add_plugin(TransformPlugin())
    live_app.startup()
    var flow = GameFlow()
    if not flow.transition_to_scene(live_app.world, reg, path_a) or live_app.world.entity_count() != 1:
        raise Error("GameFlow transition to SceneA failed")
    if not flow.transition_to_scene(live_app.world, reg, path_b) or live_app.world.entity_count() != 2:
        raise Error("GameFlow transition SceneA -> SceneB did not replace entities")

    # 3. Save slots in isolated $TMPD
    var store = SaveSlotStore("${save_dir}")
    if not store.save_slot("slot_1", "{\"score\":420,\"level\":\"SceneB\"}"):
        raise Error("SaveSlotStore.save_slot failed")
    var slots = store.list_slots()
    if len(slots) != 1 or slots[0] != "slot_1":
        raise Error("SaveSlotStore.list_slots mismatch")
    var loaded_payload = store.load_slot("slot_1")
    if loaded_payload.find("420") < 0 or not store.delete_slot("slot_1"):
        raise Error("SaveSlotStore load/delete failed")

    # 4. Timers & Tweens
    var one_shot = GameTimer(Float32(0.25), repeating=False)
    var repeating = GameTimer(Float32(0.2), repeating=True)
    var one_fires = 0
    var rep_fires = 0
    for _ in range(5):
        if one_shot.step(Float32(0.1)):
            one_fires += 1
        if repeating.step(Float32(0.1)):
            rep_fires += 1
    if one_fires != 1 or rep_fires != 2:
        raise Error("GameTimer fire count mismatch: one=" + String(one_fires) + " rep=" + String(rep_fires))

    var tw = Tween(Float32(0.0), Float32(10.0), Float32(0.5))
    var mid = tw.step(Float32(0.25))
    var done = tw.step(Float32(0.5))  # overshoots 0.5s -> must clamp to 10.0
    if abs(mid - Float32(5.0)) > Float32(0.25) or abs(done - Float32(10.0)) > Float32(1.0e-4) or not tw.is_finished():
        raise Error("Tween interpolation/clamping failed")
MOJO

    if ! "$MOJO_BIN" build -I "${ROOT}/src" "${E2E_MOJO_OPT:--O3}" --fp-mode contract=off --Werror "$probe_src" -o "$probe_bin" >"$probe_log" 2>&1; then
        fail_test "t1_m2_f25_game_flow_fsm_scene_save_timers_tweens" "M5 GameFlow FSM/scene/save/timer/tween contract pending"
        return
    fi

    local run_out="" run_rc=0
    run_out="$("$probe_bin" 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 ]]; then
        fail_test "t1_m2_f25_game_flow_fsm_scene_save_timers_tweens" "probe failed (rc=$run_rc): ${run_out}"
        return
    fi

    pass_test "t1_m2_f25_game_flow_fsm_scene_save_timers_tweens"
}

test_f22_gameplay_so_cli_produces_valid_elf_plugin
test_f22_hot_reload_host_v2_swap_preserves_state
test_f22_studio_pie_hot_reload_without_restart
test_f24_entity_from_body_and_on_enter_exit
test_f25_bt_component_json_roundtrip
test_f25_input_map_ui_panel_rebind_and_capture
test_f25_game_flow_fsm_scene_save_timers_tweens

print_summary_and_exit
