#!/usr/bin/env bash
# AREA: m1
# Tier 1 Feature Coverage: Area C — M1 MCP Assets, Procedural Upload, Material/Prefab & Scene v2 Round-Trip (F17, F20, F21)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROOT

# shellcheck source=../common/e2e_helpers.sh
source "${SCRIPT_DIR}/../common/e2e_helpers.sh"

make_isolated_tmp "flint_t1_m1"

if ! resolve_bin "flint-studio" "studio"; then
    fail_test "t1_m1_mcp_build_studio" "failed to resolve or build flint-studio"
    print_summary_and_exit
fi
STUDIO_BIN="$RESOLVED_BIN"

PROJ_DIR="${TMPD}/m1_mcp_proj"
IMPORT_SRC="${TMPD}/import_src"
mkdir -p "${PROJ_DIR}/assets/materials" "${PROJ_DIR}/assets/prefabs" "${PROJ_DIR}/scenes" "${IMPORT_SRC}"

cat >"${PROJ_DIR}/flint.toml" <<'TOML'
[project]
name = "m1_mcp_proj"
entry_scene = "assets/main.flscn"
TOML

# Generate minimal .glb, .png, .wav, .flmat, and .flprefab fixtures without writing any bytecode to repo
PYTHONDONTWRITEBYTECODE=1 python3 -B - "${IMPORT_SRC}" "${PROJ_DIR}" <<'PY'
import json, math, os, struct, sys, zlib

imp_dir, proj_dir = sys.argv[1], sys.argv[2]

# 1. Valid minimal glTF 2.0 GLB (single triangle)
positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
normals = struct.pack("<9f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
indices = struct.pack("<3H", 0, 1, 2) + b"\x00\x00"
bin_blob = positions + normals + indices
while len(bin_blob) % 4 != 0:
    bin_blob += b"\x00"
gltf = {
    "asset": {"version": "2.0"},
    "scene": 0,
    "scenes": [{"nodes": [0]}],
    "nodes": [{"name": "prop_box", "mesh": 0}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2}]}],
    "buffers": [{"byteLength": len(bin_blob)}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
        {"buffer": 0, "byteOffset": 36, "byteLength": 36, "target": 34962},
        {"buffer": 0, "byteOffset": 72, "byteLength": 6, "target": 34963},
    ],
    "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
        {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
        {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
    ],
}
json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
while len(json_bytes) % 4 != 0:
    json_bytes += b" "
total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
glb = (
    struct.pack("<III", 0x46546C67, 2, total_len)
    + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    + struct.pack("<II", len(bin_blob), 0x004E4942) + bin_blob
)
with open(os.path.join(imp_dir, "prop_box.glb"), "wb") as f:
    f.write(glb)

# 2. Valid 8x8 RGBA PNG
w, h = 8, 8
raw_rows = bytearray()
for _ in range(h):
    raw_rows.append(0)
    for _ in range(w):
        raw_rows.extend((220, 180, 64, 255))

def png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)

png_bytes = (
    b"\x89PNG\r\n\x1a\n"
    + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    + png_chunk(b"IDAT", zlib.compress(bytes(raw_rows)))
    + png_chunk(b"IEND", b"")
)
with open(os.path.join(imp_dir, "albedo.png"), "wb") as f:
    f.write(png_bytes)

# 3. Valid 48kHz stereo PCM16 WAV
sr, ch, n = 48000, 2, 1200
data_sz = n * ch * 2
wav = bytearray()
wav += b"RIFF" + struct.pack("<I", 36 + data_sz) + b"WAVE"
wav += b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ch * 2, ch * 2, 16)
wav += b"data" + struct.pack("<I", data_sz)
for i in range(n):
    val = int(math.sin(2.0 * math.pi * 440.0 * i / sr) * 10000)
    s = struct.pack("<h", val)
    wav += s + s
with open(os.path.join(imp_dir, "sfx.wav"), "wb") as f:
    f.write(wav)

# 4. Shared .flmat and standalone .flprefab inside project
flmat = {
    "format": "flmat",
    "version": 1,
    "name": "GoldPBR",
    "shading_mode": "pbr",
    "base_color": [0.95, 0.78, 0.28],
    "roughness": 0.25,
    "metallic": 0.92,
    "textures": {
        "albedo": {"rel_path": "assets/textures/albedo.png"}
    },
}
with open(os.path.join(proj_dir, "assets/materials/gold.flmat"), "w", encoding="utf-8") as f:
    json.dump(flmat, f, indent=2)

flprefab = {
    "format": "flprefab",
    "version": 1,
    "name": "CrateProp",
    "nodes": [
        {
            "name": "CrateRoot",
            "parent_slot": -1,
            "components": {
                "LocalTransform": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]},
                "MeshRendererComponent": {
                    "mesh_ref": {"rel_path": "assets/meshes/prop_box.glb"},
                    "material_ref": {"rel_path": "assets/materials/gold.flmat"}
                }
            }
        }
    ],
    "nested_refs": []
}
with open(os.path.join(proj_dir, "assets/prefabs/crate.flprefab"), "w", encoding="utf-8") as f:
    json.dump(flprefab, f, indent=2)
PY

# Run MCP v2 Tier 1 checks and capture structured results per case
RESULTS_JSON="${TMPD}/t1_m1_mcp_results.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}/../common" python3 -B - \
    "$STUDIO_BIN" "$PROJ_DIR" "$IMPORT_SRC" "$TMPD" "$RESULTS_JSON" <<'PY'
import json
import os
import sys
from mcp_client import McpStdioClient

studio_bin, proj_dir, import_src, tmpd, results_path = sys.argv[1:6]
results = []

def record(name: str, ok: bool, reason: str = "") -> None:
    results.append({"name": name, "ok": ok, "reason": reason})

def pick_prop(tool_schema: dict, candidates: list[str], default: str) -> str:
    props = tool_schema.get("inputSchema", {}).get("properties", {})
    if isinstance(props, dict):
        for c in candidates:
            if c in props:
                return c
    return default

def find_tool(tools_by_name: dict, name: str) -> dict:
    if name not in tools_by_name:
        raise AssertionError(f"missing required MCP tool in tools/list: {name}")
    return tools_by_name[name]

required_tools = [
    "asset.import",
    "asset.list",
    "prefab.spawn",
    "material.assign",
    "scene.save",
    "scene.load",
    "procedural.generate_asset",
    "ecs.query_entities",
    "render.capture_viewport",
]

# Case 1: tools/list + CL & NDJSON framing
tools_by_name = {}
try:
    with McpStdioClient([studio_bin, "--project", proj_dir, "--mcp-stdio"], cwd=proj_dir, framing="ndjson") as c_nd:
        init_nd = c_nd.initialize()
        if init_nd.get("protocolVersion") != "2024-11-05":
            raise AssertionError(f"bad protocolVersion in ndjson: {init_nd!r}")
        c_nd.notify_initialized()
        nd_tools = c_nd.list_tools()
        if not nd_tools:
            raise AssertionError("ndjson tools/list returned empty")

    with McpStdioClient([studio_bin, "--project", proj_dir, "--mcp-stdio"], cwd=proj_dir, framing="cl") as c_cl:
        init_cl = c_cl.initialize()
        if init_cl.get("protocolVersion") != "2024-11-05":
            raise AssertionError(f"bad protocolVersion in cl: {init_cl!r}")
        c_cl.notify_initialized()
        cl_tools = c_cl.list_tools()
        for t in cl_tools:
            if isinstance(t, dict) and "name" in t:
                tools_by_name[t["name"]] = t
        missing = [name for name in required_tools if name not in tools_by_name]
        if missing:
            raise AssertionError(f"missing M1/M0 MCP tools: {missing}")
        for name in required_tools:
            if not isinstance(tools_by_name[name].get("inputSchema"), dict):
                raise AssertionError(f"tool {name} missing object inputSchema")
    record("t1_m1_mcp_tools_list_and_framing", True)
except Exception as exc:
    record("t1_m1_mcp_tools_list_and_framing", False, str(exc))

# Cases 2-5 run inside real MCP sessions (with server restart for Case 5)
if not all(name in tools_by_name for name in required_tools):
    missing = [name for name in required_tools if name not in tools_by_name]
    for cname in [
        "t1_m1_mcp_asset_import_and_list",
        "t1_m1_mcp_procedural_generate_asset_upload",
        "t1_m1_mcp_material_assign_and_prefab_spawn",
        "t1_m1_mcp_scene_save_restart_reload_roundtrip",
    ]:
        record(cname, False, f"prerequisite tools missing from tools/list: {missing}")
else:
    orb_eid = -1
    proc_png_rel = "scenes/proc_orb_check.png"
    before_png_rel = "scenes/before_restart.png"
    after_png_rel = "scenes/after_restart.png"
    before_flag_path = os.path.join(tmpd, "case5_before_png_ready")
    saved_entities = []
    case4_ok = False

    def check_path_str(s: str, loc: str) -> None:
        if tmpd in s or "/home/user/flint" in s:
            raise AssertionError(f"absolute tmp or repo path leaked at {loc}: {s!r}")
        if s.startswith("/") or s.startswith("\\\\") or (len(s) >= 2 and s[1] == ":" and s[0].isalpha()):
            raise AssertionError(f"absolute path leaked at {loc}: {s!r}")
        if ".." in s.replace("\\", "/").split("/"):
            raise AssertionError(f"traversal '..' segment leaked at {loc}: {s!r}")

    def scan_json_paths(obj, loc: str = "root") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str):
                    check_path_str(k, f"{loc}.key({k})")
                scan_json_paths(v, f"{loc}.{k}")
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                scan_json_paths(v, f"{loc}[{idx}]")
        elif isinstance(obj, str):
            check_path_str(obj, loc)

    def vec_close(a, b, tol: float = 1e-3) -> bool:
        if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
            return False
        try:
            return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(len(a)))
        except (TypeError, ValueError):
            return False

    def ent_sort_key(ent: dict) -> tuple:
        pos = ent.get("position", [0.0, 0.0, 0.0])
        if isinstance(pos, list) and len(pos) == 3:
            p_key = tuple(round(float(x), 3) for x in pos)
        else:
            p_key = (0.0, 0.0, 0.0)
        return (str(ent.get("name", "")), p_key)

    try:
        with McpStdioClient([studio_bin, "--project", proj_dir, "--mcp-stdio"], cwd=proj_dir, framing="cl") as client:
            client.initialize()
            client.notify_initialized()

            # Case 2: asset.import & asset.list
            try:
                t_imp = find_tool(tools_by_name, "asset.import")
                src_k = pick_prop(t_imp, ["source_path", "src_path", "path"], "source_path")
                dst_k = pick_prop(t_imp, ["dest_path", "rel_path", "target_path"], "dest_path")

                imports = [
                    (os.path.join(import_src, "prop_box.glb"), "assets/meshes/prop_box.glb", b"glTF", 12),
                    (os.path.join(import_src, "albedo.png"), "assets/textures/albedo.png", b"\x89PNG\r\n\x1a\n", 8),
                    (os.path.join(import_src, "sfx.wav"), "assets/audio/sfx.wav", b"RIFF", 44),
                ]
                for s_abs, d_rel, magic, min_len in imports:
                    sc = client.call_tool("asset.import", {src_k: s_abs, dst_k: d_rel})
                    dst_disk = os.path.join(proj_dir, d_rel)
                    if not os.path.isfile(dst_disk):
                        raise AssertionError(f"imported file missing on disk: {d_rel} (resp={sc!r})")
                    with open(dst_disk, "rb") as f:
                        raw_bytes = f.read()
                    if len(raw_bytes) < min_len or not raw_bytes.startswith(magic):
                        raise AssertionError(f"imported file {d_rel} invalid/empty (len={len(raw_bytes)}, head={raw_bytes[:12]!r})")
                    if d_rel.endswith(".wav") and raw_bytes[8:12] != b"WAVE":
                        raise AssertionError(f"imported WAV {d_rel} missing WAVE header")

                listed = client.call_tool("asset.list", {})
                items = listed.get("assets", [])
                if not isinstance(items, list):
                    raise AssertionError(f"asset.list assets is not a list: {listed!r}")
                rel_paths = []
                for it in items:
                    if isinstance(it, dict):
                        for pk in ("rel_path", "path", "asset_path", "source_path", "abs_path"):
                            pv = it.get(pk)
                            if pv is not None:
                                check_path_str(str(pv), f"asset.list[{pk}]")
                        p = it.get("rel_path") or it.get("path") or ""
                    else:
                        p = str(it)
                        check_path_str(p, "asset.list[item]")
                    if not p:
                        raise AssertionError(f"empty asset path in asset.list item: {it!r}")
                    rel_paths.append(p)
                expected_assets = {
                    "assets/meshes/prop_box.glb",
                    "assets/textures/albedo.png",
                    "assets/audio/sfx.wav",
                    "assets/materials/gold.flmat",
                    "assets/prefabs/crate.flprefab",
                }
                if len(rel_paths) != len(expected_assets) or set(rel_paths) != expected_assets:
                    raise AssertionError(f"asset.list mismatch: got {rel_paths!r}, expected exact {sorted(expected_assets)!r}")
                record("t1_m1_mcp_asset_import_and_list", True)
            except Exception as exc:
                record("t1_m1_mcp_asset_import_and_list", False, str(exc))

            # Case 3: procedural.generate_asset real GLB upload + spawn
            try:
                exp_orb_pos = [1.5, 2.0, -0.5]
                sc = client.call_tool(
                    "procedural.generate_asset",
                    {
                        "preset": "material_orb",
                        "output_glb_path": "assets/generated/orb.glb",
                        "spawn_in_scene": True,
                        "position": exp_orb_pos,
                    },
                )
                if sc.get("glb_bytes", 0) <= 0 or sc.get("vertex_count", 0) <= 0 or sc.get("triangle_count", 0) <= 0:
                    raise AssertionError(f"invalid generated mesh stats: {sc!r}")
                orb_eid = sc.get("spawned_entity_id", -1)
                if not isinstance(orb_eid, int) or isinstance(orb_eid, bool) or orb_eid < 0:
                    raise AssertionError(f"expected valid spawned_entity_id, got {sc!r}")
                # F17 contract: must upload real mesh AND bind valid project-relative asset_ref
                mesh_id = sc.get("mesh_id", -1)
                if not isinstance(mesh_id, int) or isinstance(mesh_id, bool) or mesh_id < 0:
                    raise AssertionError(f"F17 procedural.generate_asset missing valid int mesh_id >= 0: {sc!r}")
                asset_ref = sc.get("asset_ref")
                ref_str = ""
                if isinstance(asset_ref, str):
                    ref_str = asset_ref
                elif isinstance(asset_ref, dict):
                    ref_str = str(asset_ref.get("rel_path", ""))
                if not ref_str or "assets/generated/orb.glb" not in ref_str:
                    raise AssertionError(f"F17 procedural.generate_asset missing valid asset_ref for orb.glb: {sc!r}")
                check_path_str(ref_str, "procedural.generate_asset.asset_ref")

                orb_disk = os.path.join(proj_dir, "assets/generated/orb.glb")
                if not os.path.isfile(orb_disk):
                    raise AssertionError(f"missing generated GLB at {orb_disk}")
                with open(orb_disk, "rb") as f:
                    if f.read(4) != b"glTF":
                        raise AssertionError("generated orb.glb does not start with glTF magic")

                q = client.call_tool("ecs.query_entities", {"name_contains": "orb"})
                q_ents = q.get("entities", [])
                orb_matches = [
                    e for e in q_ents
                    if isinstance(e, dict) and (e.get("entity_id") == orb_eid or "orb" in str(e.get("name", "")).lower())
                ]
                if q.get("count", 0) < 1 or not orb_matches:
                    raise AssertionError(f"spawned orb missing from ecs.query_entities: {q!r}")
                if not vec_close(orb_matches[0].get("position"), exp_orb_pos):
                    raise AssertionError(f"spawned orb position mismatch: got {orb_matches[0].get('position')!r}, expected {exp_orb_pos!r}")

                cap = client.call_tool("render.capture_viewport", {"output_path": proc_png_rel, "width": 160, "height": 120})
                if not os.path.isfile(os.path.join(proj_dir, proc_png_rel)):
                    raise AssertionError(f"capture_viewport missing file: {cap!r}")
                record("t1_m1_mcp_procedural_generate_asset_upload", True)
            except Exception as exc:
                record("t1_m1_mcp_procedural_generate_asset_upload", False, str(exc))

            # Case 4: material.assign & prefab.spawn
            try:
                if orb_eid < 0:
                    raise AssertionError("skipping material.assign because orb_eid < 0")
                t_mat = find_tool(tools_by_name, "material.assign")
                mat_k = pick_prop(t_mat, ["material_path", "path", "asset_path"], "material_path")
                mat_sc = client.call_tool("material.assign", {"entity_id": orb_eid, mat_k: "assets/materials/gold.flmat"})
                if "material_path" in mat_sc and mat_sc["material_path"] != "assets/materials/gold.flmat":
                    raise AssertionError(f"material.assign returned unexpected material_path: {mat_sc!r}")

                exp_crate_pos = [-2.0, 0.5, 1.0]
                t_pf = find_tool(tools_by_name, "prefab.spawn")
                pf_k = pick_prop(t_pf, ["prefab_path", "path", "asset_path"], "prefab_path")
                pf_sc = client.call_tool(
                    "prefab.spawn",
                    {pf_k: "assets/prefabs/crate.flprefab", "position": exp_crate_pos},
                )
                if pf_sc.get("entity_id", -1) < 0 and not pf_sc.get("entity_ids"):
                    raise AssertionError(f"prefab.spawn did not return spawned entity ID(s): {pf_sc!r}")
                q_all = client.call_tool("ecs.query_entities", {})
                saved_entities = q_all.get("entities", [])
                if len(saved_entities) < 2:
                    raise AssertionError(f"expected >= 2 entities after spawn, got {q_all!r}")
                orb_ent_list = [e for e in saved_entities if isinstance(e, dict) and e.get("entity_id") == orb_eid]
                if not orb_ent_list:
                    raise AssertionError(f"orb_eid={orb_eid} missing from ecs.query_entities: {q_all!r}")
                orb_ent = orb_ent_list[0]
                for mk in ("material_path", "material_ref", "components"):
                    if mk in orb_ent and "gold.flmat" not in json.dumps(orb_ent[mk]):
                        raise AssertionError(f"material.assign did not bind gold.flmat on orb entity: {orb_ent!r}")
                if not any(isinstance(e, dict) and vec_close(e.get("position"), exp_crate_pos) for e in saved_entities):
                    raise AssertionError(f"spawned crate prefab at {exp_crate_pos!r} not found in ecs.query_entities: {saved_entities!r}")

                _ = client.call_tool("render.capture_viewport", {"output_path": before_png_rel, "width": 160, "height": 120})
                if not os.path.isfile(os.path.join(proj_dir, before_png_rel)):
                    raise AssertionError("before_restart.png was not written")
                with open(before_flag_path, "w", encoding="utf-8") as bf:
                    bf.write("ready\n")
                case4_ok = True
                record("t1_m1_mcp_material_assign_and_prefab_spawn", True)
            except Exception as exc:
                saved_entities = []
                record("t1_m1_mcp_material_assign_and_prefab_spawn", False, str(exc))

            # Case 5 part A: scene.save before restart
            if case4_ok and saved_entities:
                try:
                    t_save = find_tool(tools_by_name, "scene.save")
                    save_k = pick_prop(t_save, ["path", "scene_path", "file_path"], "path")
                    sv_sc = client.call_tool("scene.save", {save_k: "scenes/m1_roundtrip.flscn"})
                    scn_disk = os.path.join(proj_dir, "scenes/m1_roundtrip.flscn")
                    if not os.path.isfile(scn_disk):
                        raise AssertionError(f"scene.save did not write {scn_disk} ({sv_sc!r})")
                    with open(scn_disk, "r", encoding="utf-8") as f:
                        scn_txt = f.read()
                    scn_obj = json.loads(scn_txt)
                    if scn_obj.get("version", 0) < 2:
                        raise AssertionError(f"expected scene schema version >= 2, got {scn_obj.get('version')!r}")
                    scan_json_paths(scn_obj, "m1_roundtrip.flscn")
                    for req_ref in ("gold.flmat", "orb.glb", "prop_box.glb"):
                        if req_ref not in scn_txt:
                            raise AssertionError(f"saved .flscn missing required AssetRef {req_ref!r}")
                    scn_ents = scn_obj.get("entities", [])
                    if not isinstance(scn_ents, list) or not any(
                        "orb.glb" in json.dumps(ent) and "gold.flmat" in json.dumps(ent)
                        for ent in scn_ents
                    ):
                        raise AssertionError("saved .flscn lacks orb entity with both orb.glb and assigned gold.flmat")
                except Exception as exc:
                    record("t1_m1_mcp_scene_save_restart_reload_roundtrip", False, f"save phase: {exc}")
                    saved_entities = []

        # Case 5 part B: restart Studio MCP and reload scene
        if case4_ok and saved_entities:
            with McpStdioClient([studio_bin, "--project", proj_dir, "--mcp-stdio"], cwd=proj_dir, framing="cl") as client2:
                client2.initialize()
                client2.notify_initialized()
                t_load = find_tool(tools_by_name, "scene.load")
                load_k = pick_prop(t_load, ["path", "scene_path", "file_path"], "path")
                _ = client2.call_tool("scene.load", {load_k: "scenes/m1_roundtrip.flscn"})
                q_reloaded = client2.call_tool("ecs.query_entities", {})
                reloaded_ents = q_reloaded.get("entities", [])
                if len(reloaded_ents) != len(saved_entities):
                    raise AssertionError(
                        f"entity count mismatch after restart+reload: got {len(reloaded_ents)}, expected {len(saved_entities)}"
                    )
                before_sorted = sorted(saved_entities, key=ent_sort_key)
                after_sorted = sorted(reloaded_ents, key=ent_sort_key)
                for eb, ea in zip(before_sorted, after_sorted):
                    if str(eb.get("name", "")) != str(ea.get("name", "")):
                        raise AssertionError(f"entity name mismatch after reload: {ea!r} != {eb!r}")
                    if not vec_close(ea.get("position"), eb.get("position")):
                        raise AssertionError(f"entity position mismatch after reload: {ea!r} != {eb!r}")
                    if "rotation" in eb and not vec_close(ea.get("rotation"), eb.get("rotation")):
                        raise AssertionError(f"entity rotation mismatch after reload: {ea!r} != {eb!r}")
                    if "scale" in eb and not vec_close(ea.get("scale"), eb.get("scale")):
                        raise AssertionError(f"entity scale mismatch after reload: {ea!r} != {eb!r}")
                    for field in ("visible", "shading_mode", "collider", "mesh_ref", "material_ref", "components"):
                        if field in eb and ea.get(field) != eb.get(field):
                            raise AssertionError(f"entity {field} mismatch after reload: {ea!r} != {eb!r}")

                t_save2 = find_tool(tools_by_name, "scene.save")
                save_k2 = pick_prop(t_save2, ["path", "scene_path", "file_path"], "path")
                _ = client2.call_tool("scene.save", {save_k2: "scenes/m1_after_reload.flscn"})
                resaved_disk = os.path.join(proj_dir, "scenes/m1_after_reload.flscn")
                if not os.path.isfile(resaved_disk):
                    raise AssertionError("post-reload scene.save failed to write m1_after_reload.flscn")
                with open(resaved_disk, "r", encoding="utf-8") as f:
                    resaved_txt = f.read()
                resaved_obj = json.loads(resaved_txt)
                for req_ref in ("gold.flmat", "orb.glb", "prop_box.glb"):
                    if req_ref not in resaved_txt:
                        raise AssertionError(f"post-reload .flscn lost AssetRef {req_ref!r}")
                if not any(
                    "orb.glb" in json.dumps(ent) and "gold.flmat" in json.dumps(ent)
                    for ent in resaved_obj.get("entities", [])
                ):
                    raise AssertionError("post-reload .flscn lost orb.glb + gold.flmat binding")

                cap_after = client2.call_tool(
                    "render.capture_viewport",
                    {
                        "output_path": after_png_rel,
                        "reference_path": before_png_rel,
                        "width": 160,
                        "height": 120,
                    },
                )
                if not os.path.isfile(os.path.join(proj_dir, after_png_rel)):
                    raise AssertionError("after_restart.png was not written")
                if cap_after.get("width") != 160 or cap_after.get("height") != 120:
                    raise AssertionError(f"after_restart.png dimensions mismatch: {cap_after!r}")
                if cap_after.get("has_flip_diff") is False or float(cap_after.get("flip_mean", 1.0)) > 0.02:
                    raise AssertionError(f"after_restart.png FLIP mismatch vs before_restart.png: {cap_after!r}")
            record("t1_m1_mcp_scene_save_restart_reload_roundtrip", True)
        else:
            if not any(r["name"] == "t1_m1_mcp_scene_save_restart_reload_roundtrip" for r in results):
                record(
                    "t1_m1_mcp_scene_save_restart_reload_roundtrip",
                    False,
                    "skipped because Case 4 failed or pre-restart saved_entities is empty",
                )
    except Exception as exc:
        if not any(r["name"] == "t1_m1_mcp_scene_save_restart_reload_roundtrip" for r in results):
            record("t1_m1_mcp_scene_save_restart_reload_roundtrip", False, str(exc))

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
PY

# Emit standardized [PASS]/[FAIL] lines & validate generated PNGs via e2e_helpers.sh
if [[ -f "${TMPD}/case5_before_png_ready" ]]; then
    if [[ -f "${PROJ_DIR}/scenes/before_restart.png" ]] && validate_png_dims "${PROJ_DIR}/scenes/before_restart.png" 160 120 >/dev/null 2>&1; then
        pass_test "t1_m1_mcp_before_restart_png_valid"
    else
        fail_test "t1_m1_mcp_before_restart_png_valid" "missing or invalid Case 5 before_restart.png"
    fi
fi
if [[ -f "${PROJ_DIR}/scenes/after_restart.png" ]]; then
    if validate_png_dims "${PROJ_DIR}/scenes/after_restart.png" 160 120 >/dev/null 2>&1; then
        pass_test "t1_m1_mcp_after_restart_png_valid"
    else
        fail_test "t1_m1_mcp_after_restart_png_valid" "invalid PNG header/IHDR/CRC"
    fi
fi

if [[ -f "$RESULTS_JSON" ]]; then
    while IFS=$'\t' read -r cname cok creason; do
        if [[ "$cok" == "True" ]]; then
            pass_test "$cname"
        else
            fail_test "$cname" "$creason"
        fi
    done < <(PYTHONDONTWRITEBYTECODE=1 python3 -B - "$RESULTS_JSON" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for row in json.load(f):
        reason = str(row.get("reason", "")).replace("\t", " ").replace("\n", " ")
        print(f"{row['name']}\t{bool(row['ok'])}\t{reason}")
PY
)
else
    fail_test "t1_m1_mcp_runner" "python harness failed to produce results"
fi

print_summary_and_exit
