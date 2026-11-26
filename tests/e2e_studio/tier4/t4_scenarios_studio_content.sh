#!/usr/bin/env bash
# AREA: r1 m0 m1 m3
# Tier 4 Real-World Application Scenarios (Scenarios 1, 2, 4, 6):
#   S1: Author 3D lit level (EnvironmentComponent, SpotLightComponent, TriggerVolume, baked NavMesh)
#       in project-rooted Studio -> save -> restart -> reload unchanged + PIE/Edit Esc/Del & UiContext safety
#   S2: Content pipeline: GLB + .flmat + .flrig + .flprefab -> skinned character idle -> save -> restart -> reload + MaterialOverrideComponent
#   S4: Headless MCP-driven full game creation, scene.save -> project.package -> validate_flpk_v3 & validate_png_dims
#   S6: Scene v1->v2 migration + unknown component opaque blob preservation + crash handler recovery

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t4_studio_content"
resolve_mojo_env || {
    fail_test "t4_studio_content_toolchain" "resolve_mojo_env failed"
    print_summary_and_exit
}

# ---------------------------------------------------------------------------
# Scenario 1: Author 3D lit level (EnvironmentComponent, SpotLightComponent,
#   TriggerVolume, baked NavMesh) in project-rooted Studio -> save -> restart
#   -> reload unchanged + verify PIE/Edit isolation (Esc/Del & UiContext)
# ---------------------------------------------------------------------------
test_t4_s1_author_lit_level_navmesh_save_restart_pie_isolation() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_test "t4_s1_author_lit_level_navmesh_save_restart_pie_isolation" "could not resolve flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"

    local proj="${TMPD}/s1_proj"
    mkdir -p "${proj}/scenes" "${proj}/assets"

    # A. Mojo E2E Probe: EnvironmentComponent + SpotLightComponent + TriggerVolume + NavPlugin/NavMesh + UiContext + Esc/Del in PIE
    local s1_src="${TMPD}/s1_probe.mojo"
    local s1_bin="${TMPD}/bin/s1_probe"
    cat >"$s1_src" <<'MOJO'
from std.sys import argv
from flint.app.engine_app import AppConfig, EngineApp
from flint.core.math import Vec3
from flint.nav.navmesh import NavMesh
from flint.nav.plugin import NavPlugin
from flint.physics.plugin import ColliderComponent, PhysicsPlugin, RigidBodyComponent
from flint.physics.world import PhysicsWorld
from flint.platform.input import InputState
from flint.render.aaa_clustered_lights import AAA_LIGHT_SPOT
from flint.render.plugin import EnvironmentComponent, SpotLightComponent
from flint.render.render_world import RenderWorld
from flint.scene.transform_hierarchy import GlobalTransform, LocalTransform, PrevGlobalTransform, TransformPlugin
from flint.studio.studio_app import COLLIDER_BOX, FlintStudioState, PIE_PLAYING
from flint.ui.context import UI_KEY_ESCAPE, UI_KEY_LEFT_CTRL, UiContext

def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: s1_probe <scene_out.flscn>")
    var scene_path = String(args[1])

    # 1. Verify NavPlugin startup lifecycle on EngineApp (before startup()) + NavMesh bake/query
    var nav_app = EngineApp(AppConfig(title="s1_nav_check", headless=True, worker_threads=0))
    nav_app.add_plugin(TransformPlugin())
    nav_app.add_plugin(PhysicsPlugin(gravity=0.0))
    nav_app.add_plugin(NavPlugin())
    nav_app.startup()
    var nav_floor = nav_app.world.spawn()
    _ = nav_app.world.insert[LocalTransform](nav_floor, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = nav_app.world.insert[GlobalTransform](nav_floor, GlobalTransform())
    _ = nav_app.world.insert[PrevGlobalTransform](nav_floor, PrevGlobalTransform())
    _ = nav_app.world.insert[ColliderComponent](nav_floor, ColliderComponent.box(Vec3(10.0, 0.5, 10.0)))
    _ = nav_app.world.insert[RigidBodyComponent](nav_floor, RigidBodyComponent.static_body())
    nav_app.step_one_frame(Float32(1.0 / 60.0))
    var npw = nav_app.resources.get_ptr[PhysicsWorld]()
    if not npw:
        raise Error("nav_app missing PhysicsWorld")
    var nm_app = NavMesh.build_from_physics(npw.value()[unsafe_offset=0])
    var p_app = nm_app.find_path(Vec3(-4.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0))
    if nm_app.poly_count() <= 0 or not p_app.found or len(p_app.waypoints) < 2:
        raise Error("NavPlugin/NavMesh bake & path query failed")
    nav_app.shutdown()

    # 2. Author 3D lit level + TriggerVolume + NavMesh in FlintStudioState
    var studio = FlintStudioState(headless=True, width=320, height=180, worker_threads=0)

    _ = studio.spawn_entity(
        "GroundFloor",
        position=Vec3(0.0, -0.5, 0.0),
        scale=Vec3(20.0, 1.0, 20.0),
        collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(10.0, 0.5, 10.0),
        mass=0.0,
        record_undo=False,
    )

    var env_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](env_e, LocalTransform(Vec3()))
    _ = studio.app.world.insert[GlobalTransform](env_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](env_e, PrevGlobalTransform())
    _ = studio.app.world.insert[EnvironmentComponent](env_e, EnvironmentComponent())

    var spot_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](spot_e, LocalTransform(Vec3(0.0, 5.0, 0.0)))
    _ = studio.app.world.insert[GlobalTransform](spot_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](spot_e, PrevGlobalTransform())
    _ = studio.app.world.insert[SpotLightComponent](spot_e, SpotLightComponent())

    var trig_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(2.0, 1.0, 0.0)))
    _ = studio.app.world.insert[GlobalTransform](trig_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](trig_e, PrevGlobalTransform())
    _ = studio.app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(1.0, 1.0, 1.0), is_trigger=True))

    # Propagate GlobalTransform & verify EnvironmentComponent + SpotLightComponent extraction BEFORE before_hash
    studio.render_editor_frame(320, 180)
    var rw_ptr = studio.app.resources.get_ptr[RenderWorld]()
    if not rw_ptr:
        raise Error("render_editor_frame missing RenderWorld")
    ref lights = rw_ptr.value()[unsafe_offset=0].cluster_lights
    var has_spot = False
    for li in range(len(lights)):
        if (
            lights[li].kind == AAA_LIGHT_SPOT
            and abs(lights[li].position.x) < Float32(0.01)
            and abs(lights[li].position.y - Float32(5.0)) < Float32(0.01)
            and abs(lights[li].position.z) < Float32(0.01)
        ):
            has_spot = True
            break
    if not has_spot:
        raise Error("render_editor_frame failed to extract SpotLightComponent light at (0,5,0)")
    if studio.registry.find_by_name("EnvironmentComponent") < 0 or studio.registry.find_by_name("SpotLightComponent") < 0:
        raise Error("EnvironmentComponent or SpotLightComponent missing from registry")

    var pw_ptr = studio.app.resources.get_ptr[PhysicsWorld]()
    if not pw_ptr:
        raise Error("studio missing PhysicsWorld")
    var nm = NavMesh.build_from_physics(pw_ptr.value()[unsafe_offset=0])
    var nav_path = nm.find_path(Vec3(-4.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0))
    if nm.poly_count() <= 0 or not nav_path.found or len(nav_path.waypoints) < 2:
        raise Error("Studio NavMesh bake/query failed")

    var before_count = studio.app.world.entity_count()
    var before_hash = studio.state_hash()

    # 3. Verify UiContext.wants_keyboard() capture barrier blocks Del/E/Ctrl+Q in studio.process_editor_input
    studio.select_entity(Int(trig_e.index))
    var name_buf = String("Level01")
    var ui_drv = InputState()
    ui_drv.begin_frame()
    ui_drv.move_mouse(800.0, 500.0)
    studio.ui_ctx.begin_frame(ui_drv, 1000.0, 700.0)
    _ = studio.ui_ctx.text_input("s1_lvl_name", name_buf, 200.0)
    var ti_rect = studio.ui_ctx.last_item_rect
    studio.ui_ctx.end_frame()

    ui_drv.begin_frame()
    ui_drv.move_mouse(ti_rect.center_x(), ti_rect.center_y())
    ui_drv.set_mouse(0, True)
    studio.ui_ctx.begin_frame(ui_drv, 1000.0, 700.0)
    _ = studio.ui_ctx.text_input("s1_lvl_name", name_buf, 200.0)
    studio.ui_ctx.end_frame()
    ui_drv.set_mouse(0, False)
    if not studio.ui_ctx.wants_keyboard() or not studio.ui_ctx.wants_mouse():
        raise Error("studio.ui_ctx text_input did not activate wants_keyboard()/wants_mouse()")

    var cap_inp = InputState()
    cap_inp.begin_frame()
    cap_inp.set_logical_key(261, True)  # Delete
    cap_inp.set_logical_key(69, True)   # 'E'
    cap_inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
    cap_inp.set_logical_key(81, True)   # Ctrl+Q
    if not studio.process_editor_input(cap_inp, Float32(1.0 / 60.0)):
        raise Error("Ctrl+Q quit while studio.ui_ctx.wants_keyboard() was True")
    if studio.app.world.entity_count() != before_count or studio.state_hash() != before_hash:
        raise Error("Delete/hotkey mutated world while studio.ui_ctx.wants_keyboard() was True")

    # Unfocus UI text input with Esc so normal editor input resumes
    ui_drv.begin_frame()
    ui_drv.set_logical_key(UI_KEY_ESCAPE, True)
    studio.ui_ctx.begin_frame(ui_drv, 1000.0, 700.0)
    _ = studio.ui_ctx.text_input("s1_lvl_name", name_buf, 200.0)
    studio.ui_ctx.end_frame()
    ui_drv.set_logical_key(UI_KEY_ESCAPE, False)
    if studio.ui_ctx.wants_keyboard():
        raise Error("Esc did not blur studio.ui_ctx text_input")

    # 4. Esc in Edit mode MUST NOT quit; Esc + Del during PIE with entity selected MUST NOT delete/quit
    var ed_inp = InputState()
    ed_inp.begin_frame()
    ed_inp.keys[UI_KEY_ESCAPE] = True
    ed_inp.pressed[UI_KEY_ESCAPE] = True
    if not studio.process_editor_input(ed_inp, Float32(1.0 / 60.0)):
        raise Error("bare Esc unexpectedly signaled quit")
    ed_inp.keys[UI_KEY_ESCAPE] = False

    studio.select_entity(Int(trig_e.index))
    if studio.selected_count() < 1:
        raise Error("failed to select trig_e before PIE")
    if not studio.pie_play() or studio.pie.mode != PIE_PLAYING:
        raise Error("pie_play failed")

    var pie_inp = InputState()
    pie_inp.begin_frame()
    pie_inp.set_logical_key(UI_KEY_ESCAPE, True)
    pie_inp.set_logical_key(261, True)  # Delete
    if not studio.process_editor_input(pie_inp, Float32(1.0 / 60.0)):
        raise Error("Esc during PIE signaled quit")
    if studio.app.world.entity_count() != before_count or studio.state_hash() != before_hash:
        raise Error("PIE Esc/Del mutated world during PIE before pie_stop()")

    if not studio.pie_stop():
        raise Error("pie_stop failed")
    if studio.state_hash() != before_hash:
        raise Error("PIE Esc/Del leaked mutation into editor world after pie_stop()")

    if not studio.scene.save_flscn_file(scene_path, studio.app.world, studio.registry):
        raise Error("save_flscn_file failed")
    studio.shutdown()

    # Restart fresh FlintStudioState and reload unchanged
    var studio2 = FlintStudioState(headless=True, width=320, height=180, worker_threads=0)
    _ = studio2.scene.load_flscn_file(scene_path, studio2.app.world, studio2.registry)
    if studio2.app.world.entity_count() != before_count or studio2.state_hash() != before_hash:
        raise Error("reloaded lit level count/hash mismatch")
    studio2.shutdown()
MOJO
    local compile_log="${TMPD}/s1_probe.compile.log"
    if ! "$MOJO_BIN" build -I "${ROOT}/src" -I "${ROOT}/tests" -O3 --fp-mode contract=off --Werror "$s1_src" -o "$s1_bin" >"$compile_log" 2>&1; then
        local c_err
        c_err="$(tail -n 6 "$compile_log" | tr '\n' ' ')"
        fail_test "t4_s1_author_lit_level_navmesh_save_restart_pie_isolation" "M0/M3 contract pending (s1_probe compile failed): ${c_err}"
        return
    fi

    local probe_out probe_rc=0
    probe_out="$("$s1_bin" "${proj}/scenes/s1_level.flscn" 2>&1)" || probe_rc=$?
    if [[ "$probe_rc" -ne 0 ]]; then
        fail_test "t4_s1_author_lit_level_navmesh_save_restart_pie_isolation" "s1_probe runtime failed (rc=$probe_rc): $probe_out"
        return
    fi

    # B. Project-rooted Studio MCP persistence check (--project + scene.save -> restart -> scene.load)
    local py_out py_rc=0
    py_out="$(
        FLINT_STATE_DIR="${TMPD}/s1_mcp_state" XDG_STATE_HOME="${TMPD}/s1_mcp_xdg" \
        python3 -B - "$studio_bin" "$proj" "${SCRIPT_DIR}/../common" <<'PY' 2>&1
import sys
sys.path.insert(0, sys.argv[3])
from mcp_client import McpStdioClient

studio_bin, proj = sys.argv[1], sys.argv[2]
with McpStdioClient([studio_bin, "--project", proj, "--mcp-stdio"], cwd=proj, framing="cl", timeout=30.0, allow_stderr=True) as c1:
    c1.initialize()
    c1.notify_initialized()
    tools = {t["name"] for t in c1.list_tools()}
    missing = sorted({"scene.save", "scene.load"} - tools)
    if missing:
        raise SystemExit(f"M0/M3 contract pending; missing MCP tools: {missing}")
    c1.call_tool("scene.load", {"path": "scenes/s1_level.flscn"})
    q1 = c1.call_tool("ecs.query_entities", {})
    c1.call_tool("scene.save", {"path": "scenes/s1_roundtrip.flscn"})
    h1 = q1["state_hash"]

with McpStdioClient([studio_bin, "--project", proj, "--mcp-stdio"], cwd=proj, framing="cl", timeout=30.0, allow_stderr=True) as c2:
    c2.initialize()
    c2.notify_initialized()
    c2.call_tool("scene.load", {"path": "scenes/s1_roundtrip.flscn"})
    q2 = c2.call_tool("ecs.query_entities", {})
    if q2["state_hash"] != h1:
        raise SystemExit(f"Studio restart state_hash mismatch: {q2['state_hash']} != {h1}")
PY
    )" || py_rc=$?
    if [[ "$py_rc" -ne 0 ]]; then
        fail_test "t4_s1_author_lit_level_navmesh_save_restart_pie_isolation" "$py_out"
        return
    fi

    pass_test "t4_s1_author_lit_level_navmesh_save_restart_pie_isolation"
}

# ---------------------------------------------------------------------------
# Scenario 2: Content Pipeline — GLB + .flmat + .flrig + .flprefab -> skinned
#   character idle -> save -> restart Studio -> reload identically + MaterialOverrideComponent
# ---------------------------------------------------------------------------
test_t4_s2_content_pipeline_glb_flmat_flrig_flprefab_reload() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_test "t4_s2_content_pipeline_glb_flmat_flrig_flprefab_reload" "could not resolve flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"

    local proj="${TMPD}/s2_proj"
    mkdir -p "${proj}/assets/chars" "${proj}/assets/materials" "${proj}/assets/prefabs" "${proj}/scenes"

    # Generate rigged glTF 2.0 hero.glb (3 joints + skin + Idle animation) via Python stdlib
    python3 -B - "${proj}/assets/chars/hero.glb" <<'PY'
import json, struct, sys

out_path = sys.argv[1]
positions = struct.pack("<9f", 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0)
normals = struct.pack("<9f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
joints0 = struct.pack("<12B", 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0)
weights0 = struct.pack("<12f", 0.5, 0.3, 0.2, 0.0, 0.5, 0.3, 0.2, 0.0, 0.5, 0.3, 0.2, 0.0)
indices = struct.pack("<3H", 0, 1, 2) + b"\x00\x00"
inv_bind_1 = struct.pack("<16f", 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)
inv_binds = inv_bind_1 * 3
anim_times = struct.pack("<2f", 0.0, 1.0)
anim_vals = struct.pack("<6f", 0.5, 0.4, 0.0, 0.5, 0.45, 0.0)

chunks = [positions, normals, joints0, weights0, indices, inv_binds, anim_times, anim_vals]
offsets = []
bin_blob = bytearray()
for c in chunks:
    while len(bin_blob) % 4 != 0:
        bin_blob.append(0)
    offsets.append((len(bin_blob), len(c)))
    bin_blob.extend(c)

gltf = {
    "asset": {"version": "2.0"},
    "scene": 0,
    "scenes": [{"nodes": [0]}],
    "nodes": [
        {"name": "Root", "mesh": 0, "skin": 0, "children": [1]},
        {"name": "Spine", "translation": [0.0, 1.0, 0.0], "children": [2]},
        {"name": "Hand.R", "translation": [0.5, 0.4, 0.0]},
    ],
    "skins": [{"inverseBindMatrices": 5, "joints": [0, 1, 2]}],
    "meshes": [{
        "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "JOINTS_0": 2, "WEIGHTS_0": 3},
            "indices": 4,
        }]
    }],
    "animations": [{
        "name": "Idle",
        "channels": [{"sampler": 0, "target": {"node": 2, "path": "translation"}}],
        "samplers": [{"input": 6, "interpolation": "LINEAR", "output": 7}],
    }],
    "buffers": [{"byteLength": len(bin_blob)}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": offsets[0][0], "byteLength": offsets[0][1]},
        {"buffer": 0, "byteOffset": offsets[1][0], "byteLength": offsets[1][1]},
        {"buffer": 0, "byteOffset": offsets[2][0], "byteLength": offsets[2][1]},
        {"buffer": 0, "byteOffset": offsets[3][0], "byteLength": offsets[3][1]},
        {"buffer": 0, "byteOffset": offsets[4][0], "byteLength": offsets[4][1]},
        {"buffer": 0, "byteOffset": offsets[5][0], "byteLength": offsets[5][1]},
        {"buffer": 0, "byteOffset": offsets[6][0], "byteLength": offsets[6][1]},
        {"buffer": 0, "byteOffset": offsets[7][0], "byteLength": offsets[7][1]},
    ],
    "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [0.5, 1.0, 0.0]},
        {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
        {"bufferView": 2, "componentType": 5121, "count": 3, "type": "VEC4"},
        {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
        {"bufferView": 4, "componentType": 5123, "count": 3, "type": "SCALAR"},
        {"bufferView": 5, "componentType": 5126, "count": 3, "type": "MAT4"},
        {"bufferView": 6, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0]},
        {"bufferView": 7, "componentType": 5126, "count": 2, "type": "VEC3"},
    ],
}
jbytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
while len(jbytes) % 4 != 0:
    jbytes += b" "
total_len = 12 + 8 + len(jbytes) + 8 + len(bin_blob)
glb = (
    struct.pack("<III", 0x46546C67, 2, total_len)
    + struct.pack("<II", len(jbytes), 0x4E4F534A) + jbytes
    + struct.pack("<II", len(bin_blob), 0x004E4942) + bytes(bin_blob)
)
with open(out_path, "wb") as f:
    f.write(glb)
PY

    cat > "${proj}/assets/materials/hero_armor.flmat" <<'JSON'
{"base_color":[0.2,0.45,0.85],"roughness":0.3,"metallic":0.7,"emissive":[0.05,0.1,0.2],"clearcoat":0.6,"opacity":1.0}
JSON
    cat > "${proj}/assets/prefabs/hero_npc.flprefab" <<'JSON'
{"name":"hero_npc","entities":[{"name":"HeroNpc","mesh":"assets/chars/hero.glb","material":"assets/materials/hero_armor.flmat","rig":"assets/chars/hero.flrig","clip":"idle"}]}
JSON

    # Mojo probe: generate genuine FRIG (.flrig) + .flanim, verify skinned idle playback + MaterialOverrideComponent isolation across save/restart/reload
    local s2_src="${TMPD}/s2_content_probe.mojo"
    local s2_bin="${TMPD}/bin/s2_content_probe"
    cat >"$s2_src" <<'MOJO'
from std.collections import List
from std.math import abs
from std.sys import argv
from flint.anim.flanim_ecs import save_flanim_bytes
from flint.anim.flrig import FrigJointDef, FrigRigDef, FrigSocketDef, RagdollBoneConfig, RagdollChain, load_flrig_file, save_flrig_file
from flint.anim.graph import AnimGraphDef, StateDef
from flint.anim.plugin import AnimAssetCache, AnimPlugin, AnimatorComponent, RigComponent, SkeletonComponent, register_anim_codecs
from flint.anim.pose import SkeletonInstance
from flint.anim.sync import ClipInfo
from flint.app.engine_app import AppConfig, EngineApp
from flint.assets.asset_plugin import AssetPlugin
from flint.assets.asset_ref import AssetRef
from flint.assets.flmat import FlmatAsset, MAT_SHADING_PBR, load_flmat_file, save_flmat_file
from flint.assets.flpk_v3 import ASSET_KIND_RAW, AssetUuid
from flint.assets.model import NativeAnimation, NativeChannel, NativeMesh, NativeModel, NativeNode, NativeSkin
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import MaterialOverrideComponent, MeshRendererComponent, RenderPlugin, RenderPluginStateResource, register_render_codecs
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, ScalableRenderer
from flint.scene.scene import ComponentRegistry, Scene
from flint.scene.transform_hierarchy import GlobalTransform, LocalTransform, PrevGlobalTransform, TransformPlugin
from e2e_studio.common.studio_test_utils import write_bytes

def _build_rigged_model() -> NativeModel:
    var model = NativeModel()
    var n0 = NativeNode(); n0.name = "Root"; n0.children.append(1); n0.mesh = 0
    model.nodes.append(n0^)
    var n1 = NativeNode(); n1.name = "Spine"; n1.translation = Vec3(0.0, 1.0, 0.0); n1.base_translation = Vec3(0.0, 1.0, 0.0); n1.children.append(2)
    model.nodes.append(n1^)
    var n2 = NativeNode(); n2.name = "Hand.R"; n2.translation = Vec3(0.5, 0.4, 0.0); n2.base_translation = Vec3(0.5, 0.4, 0.0)
    model.nodes.append(n2^)
    model.roots.append(0)
    var skin = NativeSkin()
    skin.joints.append(0); skin.joints.append(1); skin.joints.append(2)
    for _ in range(3):
        skin.inverse_bind.append(1.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0)
        skin.inverse_bind.append(0.0); skin.inverse_bind.append(1.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0)
        skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(1.0); skin.inverse_bind.append(0.0)
        skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(1.0)
    model.skins.append(skin^)
    var mesh = NativeMesh()
    mesh.skin = 0
    mesh.positions.append(0.0); mesh.positions.append(0.0); mesh.positions.append(0.0)
    mesh.positions.append(0.5); mesh.positions.append(0.0); mesh.positions.append(0.0)
    mesh.positions.append(0.0); mesh.positions.append(1.0); mesh.positions.append(0.0)
    for _ in range(3):
        mesh.normals.append(0.0); mesh.normals.append(0.0); mesh.normals.append(1.0)
        mesh.texcoords.append(0.0); mesh.texcoords.append(0.0)
        mesh.joints.append(0); mesh.joints.append(1); mesh.joints.append(2); mesh.joints.append(0)
        mesh.weights.append(0.5); mesh.weights.append(0.3); mesh.weights.append(0.2); mesh.weights.append(0.0)
    mesh.indices.append(0); mesh.indices.append(1); mesh.indices.append(2)
    model.meshes.append(mesh^)
    var idle_anim = NativeAnimation()
    idle_anim.name = "Idle"
    idle_anim.duration = Float32(1.0)
    var ch = NativeChannel()
    ch.node = 2; ch.path = 0
    ch.times.append(Float32(0.0)); ch.times.append(Float32(1.0))
    ch.values.append(Float32(0.5)); ch.values.append(Float32(0.4)); ch.values.append(Float32(0.0))
    ch.values.append(Float32(0.5)); ch.values.append(Float32(0.45)); ch.values.append(Float32(0.0))
    idle_anim.channels.append(ch^)
    model.animations.append(idle_anim^)
    return model^

def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: s2_content_probe <proj_dir> <scene_out.flscn>")
    var proj_dir = String(args[1])
    var scene_out = String(args[2])

    # 1. Build genuine FRIG (.flrig) + .flanim + .flmat
    var rig = FrigRigDef("HeroRig")
    rig.joints.append(FrigJointDef("Root", -1, Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    rig.joints.append(FrigJointDef("Spine", 0, Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    rig.joints.append(FrigJointDef("Hand.R", 1, Vec3(0.5, 0.4, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    var chain = RagdollChain("RightArm")
    chain.bones.append(RagdollBoneConfig("Spine", Float32(0.18), Float32(0.45), Float32(8.0)))
    chain.bones.append(RagdollBoneConfig("Hand.R", Float32(0.08), Float32(0.25), Float32(2.0)))
    rig.ragdoll_chains.append(chain^)
    rig.sockets.append(FrigSocketDef("WeaponSocket", "Hand.R", Vec3(0.0, 0.1, 0.25), Quat(), Vec3(1.0, 1.0, 1.0)))
    var flrig_path = proj_dir + "/assets/chars/hero.flrig"
    if not save_flrig_file(flrig_path, rig):
        raise Error("save_flrig_file failed")
    var rig_loaded = load_flrig_file(flrig_path)
    if len(rig_loaded.joints) != 3 or len(rig_loaded.sockets) != 1:
        raise Error("loaded hero.flrig missing joints or sockets")

    var g = AnimGraphDef()
    g.name = "HeroLocomotion"
    var st_idle = StateDef()
    st_idle.name = "Idle"
    st_idle.clip = 0
    g.states.append(st_idle^)
    var flanim_bytes = save_flanim_bytes(g)
    if not write_bytes(proj_dir + "/assets/chars/hero.flanim", flanim_bytes):
        raise Error("failed writing hero.flanim")
    var clips = List[ClipInfo]()
    clips.append(ClipInfo("Idle", Float32(1.0), True, Float32(1.0)))

    var mat_asset = FlmatAsset("HeroArmor")
    mat_asset.shading_mode = MAT_SHADING_PBR
    mat_asset.base_color = Vec3(0.2, 0.45, 0.85)
    mat_asset.roughness = Float32(0.3)
    mat_asset.metallic = Float32(0.7)
    mat_asset.clearcoat = Float32(0.6)
    _ = save_flmat_file(proj_dir + "/assets/materials/hero_armor.flmat", mat_asset)
    var mat_loaded = load_flmat_file(proj_dir + "/assets/materials/hero_armor.flmat")

    # 2. Run EngineApp with AnimPlugin + AssetPlugin + RenderPlugin, verify skinned idle + MaterialOverrideComponent isolation
    var reg = ComponentRegistry.with_builtins()
    register_render_codecs(reg)
    register_anim_codecs(reg)

    var app = EngineApp(AppConfig(title="s2_skinned_mat", headless=True, worker_threads=0, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AssetPlugin(loose_root=proj_dir, enable_watcher=False))
    app.add_plugin(AnimPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    reg.ensure_registered_in_world(app.world)
    app.startup()

    var rend_ptr = app.resources.get_ptr_mut[ScalableRenderer]()
    var acache_ptr = app.resources.get_ptr_mut[AnimAssetCache]()
    if not rend_ptr or not acache_ptr:
        raise Error("missing ScalableRenderer or AnimAssetCache")
    var shared_mat_id = rend_ptr.value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=mat_loaded.base_color, roughness=mat_loaded.roughness, metallic=mat_loaded.metallic)
    )
    acache_ptr.value()[unsafe_offset=0].register_model(UInt64(101), _build_rigged_model())
    acache_ptr.value()[unsafe_offset=0].register_graph(UInt64(201), g^, clips^)

    var e_hero = app.world.spawn()
    _ = app.world.insert[LocalTransform](e_hero, LocalTransform(Vec3(0.0, 0.0, -3.0)))
    _ = app.world.insert[GlobalTransform](e_hero, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](e_hero, PrevGlobalTransform())
    var rc = RigComponent()
    rc.rig_ref = AssetRef(AssetUuid.from_path("assets/chars/hero.flrig"), "assets/chars/hero.flrig", ASSET_KIND_RAW)
    _ = app.world.insert[RigComponent](e_hero, rc^)
    _ = app.world.insert[SkeletonComponent](e_hero, SkeletonComponent(UInt64(101), SkeletonInstance(3), True))
    _ = app.world.insert[AnimatorComponent](e_hero, AnimatorComponent(UInt64(201)))
    _ = app.world.insert[MeshRendererComponent](e_hero, MeshRendererComponent(0, shared_mat_id))
    var ov = MaterialOverrideComponent(-1, Vec3(0.95, 0.25, 0.15), 0.2, 0.85, Vec3(), True, opacity=0.9, clearcoat=0.7)
    ov.material_ref = AssetRef(AssetUuid.from_path("assets/materials/hero_armor.flmat"), "assets/materials/hero_armor.flmat", ASSET_KIND_RAW)
    _ = app.world.insert[MaterialOverrideComponent](e_hero, ov^)

    var e_comp = app.world.spawn()
    _ = app.world.insert[LocalTransform](e_comp, LocalTransform(Vec3(2.0, 0.0, -3.0)))
    _ = app.world.insert[GlobalTransform](e_comp, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](e_comp, PrevGlobalTransform())
    _ = app.world.insert[MeshRendererComponent](e_comp, MeshRendererComponent(0, shared_mat_id))

    for _ in range(5):
        app.step_one_frame(Float32(1.0 / 60.0))

    var state_idle = app.world.get_ptr[AnimatorComponent](e_hero).value()[].animator.state_name()
    var bones_n = len(app.world.get_ptr[SkeletonComponent](e_hero).value()[].skeleton.world_matrices)
    var rps = app.resources.get_ptr[RenderPluginStateResource]()
    if state_idle != "Idle" or bones_n != 3 or not rps or rps.value()[unsafe_offset=0].skinned_sync_count < 1:
        raise Error("skinned Idle playback failed: state=" + state_idle + " bones=" + String(bones_n))
    # Shared material must remain untouched by e_hero's MaterialOverrideComponent
    if abs(rend_ptr.value()[unsafe_offset=0].materials[shared_mat_id].base_color.x - Float32(0.2)) > Float32(1.0e-4):
        raise Error("MaterialOverrideComponent clobbered shared material slot")

    var sc = Scene("S2Content")
    if not sc.save_flscn_file(scene_out, app.world, reg):
        raise Error("S2 scene save failed")
    app.shutdown()

    # 3. Restart fresh app and reload scene
    var app2 = EngineApp(AppConfig(title="s2_skinned_reload", headless=True, worker_threads=0, fixed_hz=60.0))
    app2.add_plugin(TransformPlugin())
    app2.add_plugin(AssetPlugin(loose_root=proj_dir, enable_watcher=False))
    app2.add_plugin(AnimPlugin())
    app2.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    reg.ensure_registered_in_world(app2.world)
    app2.startup()
    var sc2 = Scene()
    var ents2 = sc2.load_flscn_file(scene_out, app2.world, reg)
    if len(ents2) != 2:
        raise Error("S2 scene reload entity count mismatch: " + String(len(ents2)))
    if not app2.world.has[RigComponent](ents2[0]) or not app2.world.has[MaterialOverrideComponent](ents2[0]) or app2.world.has[MaterialOverrideComponent](ents2[1]):
        raise Error("S2 reload lost RigComponent or MaterialOverrideComponent per-entity isolation")
    app2.shutdown()
MOJO
    local s2_log="${TMPD}/s2_content_probe.compile.log"
    if ! "$MOJO_BIN" build -I "${ROOT}/src" -I "${ROOT}/tests" -O3 --fp-mode contract=off --Werror "$s2_src" -o "$s2_bin" >"$s2_log" 2>&1; then
        local c_err
        c_err="$(tail -n 6 "$s2_log" | tr '\n' ' ')"
        fail_test "t4_s2_content_pipeline_glb_flmat_flrig_flprefab_reload" "M1 contract pending (s2_content_probe compile failed): ${c_err}"
        return
    fi
    local s2_out s2_rc=0
    s2_out="$("$s2_bin" "$proj" "${proj}/scenes/s2_mojo_roundtrip.flscn" 2>&1)" || s2_rc=$?
    if [[ "$s2_rc" -ne 0 ]]; then
        fail_test "t4_s2_content_pipeline_glb_flmat_flrig_flprefab_reload" "s2_content_probe failed (rc=$s2_rc): $s2_out"
        return
    fi

    # Check M1 Studio/MCP content pipeline end-to-end
    local py_out py_rc=0
    py_out="$(
        FLINT_STATE_DIR="${TMPD}/s2_state" XDG_STATE_HOME="${TMPD}/s2_xdg" \
        python3 -B - "$studio_bin" "$proj" "${SCRIPT_DIR}/../common" <<'PY' 2>&1
import sys
sys.path.insert(0, sys.argv[3])
from mcp_client import McpStdioClient

studio_bin, proj = sys.argv[1], sys.argv[2]

with McpStdioClient([studio_bin, "--project", proj, "--mcp-stdio"], cwd=proj, framing="cl", timeout=30.0, allow_stderr=True) as c1:
    c1.initialize()
    c1.notify_initialized()
    tools = {t["name"] for t in c1.list_tools()}
    needed = {"asset.import", "asset.list", "prefab.spawn", "material.assign", "scene.save", "scene.load"}
    missing = sorted(needed - tools)
    if missing:
        raise SystemExit(f"M1/M3 contract pending; missing MCP tools: {missing}")

    imp = c1.call_tool("asset.import", {"path": "assets/chars/hero.glb"})
    listed = c1.call_tool("asset.list", {"dir": "assets"})
    sp = c1.call_tool("prefab.spawn", {"path": "assets/prefabs/hero_npc.flprefab", "position": [0.0, 0.0, 0.0]})
    eids = sp.get("entity_ids") or ([sp["entity_id"]] if "entity_id" in sp else [])
    if not eids:
        raise SystemExit(f"prefab.spawn returned no entity ids: {sp}")
    c1.call_tool("material.assign", {"entity_id": eids[0], "material_path": "assets/materials/hero_armor.flmat"})
    c1.call_tool("scene.save", {"path": "scenes/s2_content.flscn"})
    q1 = c1.call_tool("ecs.query_entities", {})
    names1 = [e.get("name", "") for e in q1.get("entities", [])]
    if q1.get("count", 0) < 1 or "HeroNpc" not in names1:
        raise SystemExit(f"Scenario 2 HeroNpc missing before save: {q1}")
    cap1 = c1.call_tool("render.capture_viewport", {"output_path": "scenes/s2_before.png", "width": 320, "height": 180})

with McpStdioClient([studio_bin, "--project", proj, "--mcp-stdio"], cwd=proj, framing="cl", timeout=30.0, allow_stderr=True) as c2:
    c2.initialize()
    c2.notify_initialized()
    c2.call_tool("scene.load", {"path": "scenes/s2_content.flscn"})
    q2 = c2.call_tool("ecs.query_entities", {})
    names2 = [e.get("name", "") for e in q2.get("entities", [])]
    if q2.get("count", 0) < 1 or "HeroNpc" not in names2:
        raise SystemExit(f"Scenario 2 HeroNpc missing after scene.load: {q2}")
    if q2["state_hash"] != q1["state_hash"] or q2["count"] != q1["count"]:
        raise SystemExit(f"Scenario 2 restart reload mismatch: {q2} vs {q1}")
    cap2 = c2.call_tool("render.capture_viewport", {
        "output_path": "scenes/s2_after.png",
        "reference_path": "scenes/s2_before.png",
        "width": 320,
        "height": 180,
    })
    if cap2.get("has_flip_diff") is not True:
        raise SystemExit(f"Scenario 2 FLIP diff was not computed (has_flip_diff={cap2.get('has_flip_diff')}): {cap2}")
    if cap2.get("flip_mean", 1.0) > 0.02:
        raise SystemExit(f"Scenario 2 reload render diverged: flip_mean={cap2.get('flip_mean')}")
PY
    )" || py_rc=$?
    if [[ "$py_rc" -ne 0 ]]; then
        fail_test "t4_s2_content_pipeline_glb_flmat_flrig_flprefab_reload" "$py_out"
        return
    fi

    if ! validate_png_dims "${proj}/scenes/s2_after.png" 320 180; then
        fail_test "t4_s2_content_pipeline_glb_flmat_flrig_flprefab_reload" "s2_after.png invalid"
        return
    fi

    pass_test "t4_s2_content_pipeline_glb_flmat_flrig_flprefab_reload"
}

# ---------------------------------------------------------------------------
# Scenario 4: Headless MCP-driven full game creation -> scene.save ->
#   project.package -> verify cooked .flpk (validate_flpk_v3) & PNG (validate_png_dims)
# ---------------------------------------------------------------------------
test_t4_s4_headless_mcp_full_game_create_save_package() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_test "t4_s4_headless_mcp_full_game_create_save_package" "could not resolve flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"

    local proj="${TMPD}/s4_mcp_game"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t4_s4_headless_mcp_full_game_create_save_package" "flint.sh new failed"
        return
    fi
    mkdir -p "${proj}/assets" "${proj}/scenes" "${proj}/build"

    local py_out py_rc=0
    py_out="$(
        FLINT_STATE_DIR="${TMPD}/s4_state" XDG_STATE_HOME="${TMPD}/s4_xdg" \
        python3 -B - "$studio_bin" "$proj" "${SCRIPT_DIR}/../common" <<'PY' 2>&1
import sys
sys.path.insert(0, sys.argv[3])
from mcp_client import McpStdioClient

studio_bin, proj = sys.argv[1], sys.argv[2]

with McpStdioClient([studio_bin, "--project", proj, "--mcp-stdio"], cwd=proj, framing="cl", timeout=30.0, allow_stderr=True) as c:
    c.initialize()
    c.notify_initialized()
    tools = {t["name"] for t in c.list_tools()}

    # Author level geometry & props via MCP
    c.call_tool("ecs.spawn_entity", {"name": "Floor", "position": [0.0, -0.5, 0.0], "scale": [12.0, 1.0, 12.0], "collider": "box", "mass": 0.0})
    orb = c.call_tool("procedural.generate_asset", {
        "preset": "material_orb",
        "output_glb_path": "assets/orb.glb",
        "spawn_in_scene": True,
        "position": [0.0, 1.5, 0.0],
    })
    c.call_tool("render.configure_npr_cel", {"entity_id": orb["spawned_entity_id"], "shading_mode": "npr_cel"})
    c.call_tool("render.capture_viewport", {"output_path": "build/s4_preview.png", "width": 320, "height": 180})

    missing = sorted({"scene.save", "project.package"} - tools)
    if missing:
        raise SystemExit(f"M0/M3 contract pending; missing MCP tools: {missing}")

    c.call_tool("scene.save", {"path": "assets/main.flscn"})
    c.call_tool("project.package", {"output_dir": "dist/s4_game"})
PY
    )" || py_rc=$?
    if [[ "$py_rc" -ne 0 ]]; then
        fail_test "t4_s4_headless_mcp_full_game_create_save_package" "$py_out"
        return
    fi

    if ! validate_png_dims "${proj}/build/s4_preview.png" 320 180; then
        fail_test "t4_s4_headless_mcp_full_game_create_save_package" "s4_preview.png failed validate_png_dims"
        return
    fi

    local dist="${proj}/dist/s4_game"
    if [[ ! -x "${dist}/run.sh" || ! -f "${dist}/assets.flpk" ]]; then
        fail_test "t4_s4_headless_mcp_full_game_create_save_package" "packaged bundle missing run.sh or assets.flpk"
        return
    fi
    if ! validate_flpk_v3 "${dist}/assets.flpk" "orb.glb" "main.flscn"; then
        fail_test "t4_s4_headless_mcp_full_game_create_save_package" "assets.flpk missing orb.glb or main.flscn"
        return
    fi

    local reloc="${TMPD}/s4_isolated/game"
    mkdir -p "$(dirname "$reloc")"
    cp -a "$dist" "$reloc"

    local run_out run_rc=0
    run_out="$(FLINT_STATE_DIR="${TMPD}/s4_run_state" XDG_STATE_HOME="${TMPD}/s4_run_xdg" run_in_clean_env "$reloc" ./run.sh --shots --out "${reloc}/ship_shot.png" 2>&1)" || run_rc=$?
    if [[ "$run_rc" -ne 0 || ! -s "${reloc}/ship_shot.png" ]]; then
        fail_test "t4_s4_headless_mcp_full_game_create_save_package" "packaged run.sh --shots failed (rc=$run_rc): $run_out"
        return
    fi
    if ! validate_png_dims "${reloc}/ship_shot.png" 1280 720; then
        fail_test "t4_s4_headless_mcp_full_game_create_save_package" "ship_shot.png invalid"
        return
    fi

    pass_test "t4_s4_headless_mcp_full_game_create_save_package"
}

# ---------------------------------------------------------------------------
# Scenario 6: Scene v1->v2 migration + unknown component opaque blob preservation
#             + crash handler recovery (FLINT_CRASH / <state>/crashes/.../crash.txt)
# ---------------------------------------------------------------------------
test_t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler() {
    # Part A: Crash handler verification through scaffolded app_name project
    local proj="${TMPD}/s6_crash_app"
    if ! bash "${ROOT}/tools/flint.sh" new "$proj" >/dev/null 2>&1; then
        fail_test "t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler" "flint.sh new failed"
        return
    fi
    if ! grep -q "enable_crash_handler=True" "${proj}/src/main.mojo" || ! grep -q "start_log(cfg.app_name)" "${proj}/src/main.mojo"; then
        fail_test "t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler" "scaffold missing enable_crash_handler=True or start_log(cfg.app_name)"
        return
    fi

    local crash_probe_src="${TMPD}/s6_crash_probe.mojo"
    local crash_probe_bin="${TMPD}/bin/s6_crash_probe"
    cat >"$crash_probe_src" <<'MOJO'
from std.ffi import external_call
from flint.app import AppConfig, EngineApp
from flint.core.log import LOG_CORE, flush_log, log_info, start_log

def main() raises:
    _ = start_log("s6_crash_app")
    var app = EngineApp(AppConfig(title="s6_crash_app", headless=True, enable_crash_handler=True))
    app.startup()
    log_info(LOG_CORE, "s6_crash_trigger")
    _ = flush_log()
    external_call["abort", NoneType]()
MOJO
    if ! "$MOJO_BIN" build -I "${ROOT}/src" -O3 --fp-mode contract=off --Werror "$crash_probe_src" -o "$crash_probe_bin" >/dev/null 2>&1; then
        fail_test "t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler" "failed to compile s6_crash_probe.mojo"
        return
    fi

    local crash_state="${TMPD}/s6_crash_state"
    local crash_err="${TMPD}/s6_crash.stderr"
    local crash_rc=0
    bash -c 'FLINT_STATE_DIR="$1" XDG_STATE_HOME="$2" "$3" >/dev/null 2>"$4"' _ "$crash_state" "${TMPD}/s6_xdg" "$crash_probe_bin" "$crash_err" >/dev/null 2>&1 || crash_rc=$?
    if [[ "$crash_rc" -eq 0 ]]; then
        fail_test "t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler" "crash probe exited 0"
        return
    fi
    if ! grep -q "FLINT_CRASH" "$crash_err" || grep -qi "SHARDDEPTH" "$crash_err"; then
        fail_test "t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler" "stderr missing FLINT_CRASH or contains SHARDDEPTH: $(cat "$crash_err")"
        return
    fi
    local crash_txt
    crash_txt="$(find "${crash_state}/crashes" -name 'crash.txt' -print -quit 2>/dev/null || true)"
    if [[ -z "$crash_txt" || ! -s "$crash_txt" ]]; then
        fail_test "t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler" "crash.txt not created under ${crash_state}/crashes"
        return
    fi
    if grep -qi "sharddepth" "$crash_txt" || ! grep -q "s6_crash_app" "$crash_txt"; then
        fail_test "t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler" "crash.txt invalid (expected s6_crash_app, no sharddepth): $(head -n 8 "$crash_txt")"
        return
    fi

    # Part B: Scene v1 -> v2 migration + unknown component opaque blob preservation
    local v1_scene="${TMPD}/s6_v1_unknown.flscn"
    cat > "$v1_scene" <<'JSON'
{"format":"flscn","version":1,"name":"legacy_level","entities":[{"slot":0,"parent":-1,"components":{"LocalTransform":{"position":[3.0,4.0,5.0],"rotation":[0.0,0.0,0.0,1.0],"scale":[1.0,1.0,1.0]},"UnknownCustomAi":{"aggro_range":12.5,"faction":"rogue"}}}]}
JSON

    local scene_probe_src="${TMPD}/s6_scene_probe.mojo"
    local scene_probe_bin="${TMPD}/bin/s6_scene_probe"
    cat >"$scene_probe_src" <<'MOJO'
from std.sys import argv
from flint.ecs.world import World
from flint.scene.scene import SCENE_SCHEMA_VERSION, ComponentRegistry, Scene

def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: s6_scene_probe <in_v1.flscn> <out_v2.flscn>")
    var schema_ver = SCENE_SCHEMA_VERSION
    if schema_ver != 2:
        raise Error("M1 contract pending: SCENE_SCHEMA_VERSION == " + String(schema_ver) + ", expected 2")

    var reg = ComponentRegistry.with_builtins()
    var w1 = World()
    var sc = Scene()
    var ents1 = sc.load_flscn_file(String(args[1]), w1, reg)
    if len(ents1) != 1:
        raise Error("expected 1 entity from v1 scene")

    if not sc.save_flscn_file(String(args[2]), w1, reg):
        raise Error("save_flscn_file failed")

    var w2 = World()
    var sc2 = Scene()
    var ents2 = sc2.load_flscn_file(String(args[2]), w2, reg)
    if len(ents2) != 1:
        raise Error("expected 1 entity after v2 reload")
    var out_str = sc2.save_flscn_string(w2, reg)
    if out_str.find("UnknownCustomAi") < 0 or out_str.find("rogue") < 0 or out_str.find("12.5") < 0:
        raise Error("unknown component opaque blob lost during round-trip: " + out_str)
MOJO
    if ! "$MOJO_BIN" build -I "${ROOT}/src" -O3 --fp-mode contract=off --Werror "$scene_probe_src" -o "$scene_probe_bin" >/dev/null 2>&1; then
        fail_test "t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler" "failed to compile s6_scene_probe.mojo"
        return
    fi

    local scene_out scene_rc=0
    scene_out="$("$scene_probe_bin" "$v1_scene" "${TMPD}/s6_v2_out.flscn" 2>&1)" || scene_rc=$?
    if [[ "$scene_rc" -ne 0 ]]; then
        fail_test "t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler" "Scene v1->v2 opaque blob check failed (crash handler passed): $scene_out"
        return
    fi

    pass_test "t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler"
}

test_t4_s1_author_lit_level_navmesh_save_restart_pie_isolation
test_t4_s2_content_pipeline_glb_flmat_flrig_flprefab_reload
test_t4_s4_headless_mcp_full_game_create_save_package
test_t4_s6_scene_v1_v2_migration_opaque_blob_and_crash_handler

print_summary_and_exit
