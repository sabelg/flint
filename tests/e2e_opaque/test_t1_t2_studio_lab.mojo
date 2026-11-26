"""Tiers 1 & 2 Opaque-Box E2E Suite for FA8 (FlintStudioState, McpServerV2, EngineBenchSuite).

Uses ONLY public Flint Engine v5 APIs (zero private `_` symbols from `flint.*`).
All disk outputs live under an isolated `/tmp/flint_e2e_m3_fa8_<pid>` directory cleaned on exit.
"""
from std.collections import List
from std.ffi import external_call
from std.testing import assert_equal, assert_false, assert_true

from flint.assets.image import decode_png, write_png
from flint.assets.json import JsonDocument, parse_json
from flint.assets.server import AssetServer
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.core.perf_history import PerfTable
from flint.devtools.mcp_v2 import (
    JSONRPC_INVALID_PARAMS,
    JSONRPC_INVALID_REQUEST,
    JSONRPC_METHOD_NOT_FOUND,
    JSONRPC_PARSE_ERROR,
    MCP_PROTOCOL_VERSION,
    McpServerV2,
)
from flint.lab import EngineBenchSuite, LabCaseResult, compute_latency_percentiles
from flint.platform.os import file_size, make_dirs, read_bytes, write_bytes
from flint.studio import (
    COLLIDER_BOX,
    COLLIDER_SPHERE,
    FIELD_BOOL,
    FIELD_F32,
    FIELD_INT,
    FIELD_STRING,
    PIE_PAUSED,
    PIE_PLAYING,
    PIE_STOPPED,
    FlintStudioState,
    InspectedField,
)


def _str_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _parse_json_str(s: String) -> JsonDocument:
    var raw = _str_to_bytes(s)
    return parse_json(raw)


def _rm_rf_tmp(path: String):
    if path.startswith("/tmp/flint_e2e_m3_"):
        var cmd = "rm -rf " + path
        _ = external_call["system", Int32](cmd.as_c_string_span())


# ===========================================================================
# Tier 1: Happy-Path Tests (FA8)
# ===========================================================================

def test_t1_studio_entity_transform_patch_undo_redo() raises:
    var studio = FlintStudioState()
    var h0 = studio.state_hash()
    assert_equal(studio.live_entity_count(), 0)
    assert_false(studio.can_undo())
    assert_false(studio.can_redo())

    var parent_id = studio.spawn_entity(
        "ParentRig",
        position=Vec3(1.0, 2.0, 3.0),
        collider_kind=COLLIDER_BOX,
        mass=0.0,
    )
    assert_true(parent_id >= 0)
    assert_equal(studio.live_entity_count(), 1)
    assert_equal(studio.selected_entity_id, parent_id)
    var h1 = studio.state_hash()
    assert_true(h1 != h0)

    var child_id = studio.spawn_entity(
        "ChildProp",
        position=Vec3(0.0, 1.0, 0.0),
        collider_kind=COLLIDER_SPHERE,
        mass=2.0,
        parent_id=parent_id,
    )
    assert_true(child_id >= 0)
    assert_equal(studio.live_entity_count(), 2)
    assert_equal(studio.entities[child_id].parent_id, parent_id)
    var h2 = studio.state_hash()
    assert_true(h2 != h1)

    var dup_id = studio.duplicate_entity(child_id)
    assert_true(dup_id >= 0)
    assert_equal(studio.live_entity_count(), 3)
    assert_equal(studio.entities[dup_id].name, "ChildProp_Copy")
    assert_true(studio.entities[dup_id].translation.x > studio.entities[child_id].translation.x)
    var h3 = studio.state_hash()

    assert_true(
        studio.set_entity_transform(
            dup_id,
            Vec3(4.0, 5.0, -2.0),
            Quat(0.0, 0.7071, 0.0, 0.7071),
            Vec3(1.5, 1.5, 1.5),
        )
    )
    var h4 = studio.state_hash()
    assert_true(h4 != h3)

    var f_rough = InspectedField("roughness", FIELD_F32)
    f_rough.f_val = Float32(0.85)
    assert_true(studio.patch_entity_field(dup_id, "roughness", f_rough))
    assert_true(studio.entities[dup_id].roughness > Float32(0.8))
    var h5 = studio.state_hash()

    var f_name = InspectedField("name", FIELD_STRING)
    f_name.s_val = "RenamedProp"
    assert_true(studio.patch_entity_field(dup_id, "name", f_name))
    assert_equal(studio.entities[dup_id].name, "RenamedProp")
    var h6 = studio.state_hash()

    var f_vis = InspectedField("visible", FIELD_BOOL)
    f_vis.b_val = False
    assert_true(studio.patch_entity_field(dup_id, "visible", f_vis))
    assert_false(studio.entities[dup_id].visible)
    var h7 = studio.state_hash()

    assert_true(studio.can_undo())
    assert_true(studio.undo())
    assert_equal(studio.state_hash(), h6)
    assert_true(studio.undo())
    assert_equal(studio.state_hash(), h5)
    assert_true(studio.undo())
    assert_equal(studio.state_hash(), h4)
    assert_true(studio.undo())
    assert_equal(studio.state_hash(), h3)
    assert_true(studio.undo())
    assert_equal(studio.state_hash(), h2)
    assert_true(studio.undo())
    assert_equal(studio.state_hash(), h1)
    assert_true(studio.undo())
    assert_equal(studio.state_hash(), h0)
    assert_false(studio.can_undo())

    assert_true(studio.can_redo())
    assert_true(studio.redo())
    assert_equal(studio.state_hash(), h1)
    assert_true(studio.redo())
    assert_equal(studio.state_hash(), h2)
    assert_true(studio.redo())
    assert_equal(studio.state_hash(), h3)
    assert_true(studio.redo())
    assert_equal(studio.state_hash(), h4)
    assert_true(studio.redo())
    assert_equal(studio.state_hash(), h5)
    assert_true(studio.redo())
    assert_equal(studio.state_hash(), h6)
    assert_true(studio.redo())
    assert_equal(studio.state_hash(), h7)
    assert_false(studio.can_redo())
    print("[PASS] test_t1_studio_entity_transform_patch_undo_redo")


def test_t1_studio_pie_deterministic_and_lz4_restore() raises:
    var studio = FlintStudioState()
    var ball_id = studio.spawn_entity(
        "FallingSphere",
        position=Vec3(0.0, 5.0, 0.0),
        collider_kind=COLLIDER_SPHERE,
        collider_half_extents=Vec3(0.5, 0.5, 0.5),
        mass=5.0,
    )
    assert_true(ball_id >= 0)
    var pre_hash = studio.state_hash()
    var pre_y = studio.entities[ball_id].translation.y

    assert_true(studio.pie_play())
    assert_equal(studio.pie.mode, PIE_PLAYING)
    assert_true(studio.pie_pause())
    assert_equal(studio.pie.mode, PIE_PAUSED)
    assert_true(studio.pie_pause())
    assert_equal(studio.pie.mode, PIE_PLAYING)

    var stepped = studio.pie_step(60)
    assert_equal(stepped, 60)
    assert_equal(studio.pie.ticks_simulated, 60)
    assert_true(studio.entities[ball_id].translation.y < pre_y)
    assert_true(studio.state_hash() != pre_hash)

    assert_true(studio.pie_stop())
    assert_equal(studio.pie.mode, PIE_STOPPED)
    assert_true(studio.pie.last_restore_verified)
    assert_equal(studio.state_hash(), pre_hash)
    assert_equal(studio.entities[ball_id].translation.y, pre_y)

    var tick_before = studio.sim_tick
    var direct = studio.step_physics_deterministic(10)
    assert_equal(direct, 10)
    assert_equal(studio.sim_tick, tick_before + 10)
    print("[PASS] test_t1_studio_pie_deterministic_and_lz4_restore")


def test_t1_studio_scene_save_load_and_asset_browser(tmp_root: String) raises:
    var scene_dir = tmp_root + "/studio_scene_test"
    assert_true(make_dirs(scene_dir))

    var dummy_px = List[UInt32](length=4 * 4, fill=UInt32(0xFF336699))
    assert_true(write_png(scene_dir + "/albedo.png", dummy_px, 4, 4, stamped=False))

    var studio = FlintStudioState()
    var e1 = studio.spawn_entity("HeroCrate", position=Vec3(2.0, 1.0, -4.0), mass=3.0)
    assert_true(
        studio.set_entity_transform(
            e1,
            Vec3(2.5, 1.25, -4.0),
            Quat(0.0, 0.0, 0.0, 1.0),
            Vec3(2.0, 2.0, 2.0),
        )
    )
    var scene_path = scene_dir + "/level.flscene"
    var saved_hash = studio.state_hash()
    assert_true(studio.save_scene_file(scene_path))
    assert_true(file_size(scene_path) > 0)

    var loaded = FlintStudioState()
    assert_true(loaded.load_scene_file(scene_path))
    assert_equal(loaded.state_hash(), saved_hash)
    assert_false(loaded.can_undo())
    assert_false(loaded.can_redo())

    var scanned = loaded.scan_asset_browser(dir_path=scene_dir)
    assert_true(loaded.asset_browser_scanned)
    assert_true(scanned >= 2)
    print("[PASS] test_t1_studio_scene_save_load_and_asset_browser")


def test_t1_mcp_v2_protocol_tools_and_auth(tmp_root: String) raises:
    var ws_dir = tmp_root + "/mcp_ws"
    assert_true(make_dirs(ws_dir))

    var mcp = McpServerV2(workspace_root=ws_dir, auth_token="secret-42")
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    _ = pool.start(2)

    try:
        assert_false(mcp.session_authenticated)

        var init_req = (
            '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{'
            '"protocolVersion":"2024-11-05","auth_token":"secret-42",'
            '"clientInfo":{"name":"e2e-opaque"}}}'
        )
        var init_resp = mcp.handle_jsonrpc(init_req, studio, assets, pool)
        var init_doc = _parse_json_str(init_resp)
        assert_true(init_doc.valid)
        assert_true(mcp.session_authenticated)
        assert_true(mcp.initialized)
        var init_res = init_doc.object_get(init_doc.root, "result")
        assert_equal(init_doc.text(init_doc.object_get(init_res, "protocolVersion")), MCP_PROTOCOL_VERSION)

        var notif_resp = mcp.handle_jsonrpc('{"jsonrpc":"2.0","method":"notifications/initialized"}', studio, assets, pool)
        assert_equal(notif_resp, "")

        var ping_resp = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":2,"method":"ping"}', studio, assets, pool)
        var ping_doc = _parse_json_str(ping_resp)
        assert_true(ping_doc.valid)
        assert_true(ping_doc.object_get(ping_doc.root, "result") >= 0)

        var list_resp = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":3,"method":"tools/list"}', studio, assets, pool)
        var list_doc = _parse_json_str(list_resp)
        assert_true(list_doc.valid)
        var list_res = list_doc.object_get(list_doc.root, "result")
        var tools_arr = list_doc.object_get(list_res, "tools")
        assert_true(list_doc.array_size(tools_arr) >= 14)

        var spawn_req = (
            '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{'
            '"name":"ecs.spawn_entity","arguments":{"name":"McpSphere","position":[0.0,4.0,0.0],"collider":"sphere","mass":2.0}}}'
        )
        var spawn_doc = _parse_json_str(mcp.handle_jsonrpc(spawn_req, studio, assets, pool))
        var spawn_res = spawn_doc.object_get(spawn_doc.root, "result")
        assert_false(spawn_doc.boolean(spawn_doc.object_get(spawn_res, "isError"), True))
        var spawn_sc = spawn_doc.object_get(spawn_res, "structuredContent")
        assert_true(spawn_doc.boolean(spawn_doc.object_get(spawn_sc, "ok"), False))
        var eid = spawn_doc.integer(spawn_doc.object_get(spawn_sc, "entity_id"), -1)
        assert_true(eid >= 0)

        var query_req = '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{"name_contains":"McpSphere"}}}'
        var query_doc = _parse_json_str(mcp.handle_jsonrpc(query_req, studio, assets, pool))
        var query_sc = query_doc.object_get(query_doc.object_get(query_doc.root, "result"), "structuredContent")
        assert_equal(query_doc.integer(query_doc.object_get(query_sc, "count"), 0), 1)

        var patch_req = (
            '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{'
            '"name":"ecs.patch_components","arguments":{"entity_id":' + String(eid) + ',"roughness":0.75}}}'
        )
        var patch_doc = _parse_json_str(mcp.handle_jsonrpc(patch_req, studio, assets, pool))
        var patch_sc = patch_doc.object_get(patch_doc.object_get(patch_doc.root, "result"), "structuredContent")
        assert_true(patch_doc.boolean(patch_doc.object_get(patch_sc, "ok"), False))

        var step_req = '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"physics.step_deterministic","arguments":{"ticks":5}}}'
        var step_doc = _parse_json_str(mcp.handle_jsonrpc(step_req, studio, assets, pool))
        var step_sc = step_doc.object_get(step_doc.object_get(step_doc.root, "result"), "structuredContent")
        assert_equal(step_doc.integer(step_doc.object_get(step_sc, "ticks_advanced"), 0), 5)

        var ray_req = '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"physics.raycast_query","arguments":{"origin":[0.0,10.0,0.0],"direction":[0.0,-1.0,0.0]}}}'
        var ray_doc = _parse_json_str(mcp.handle_jsonrpc(ray_req, studio, assets, pool))
        var ray_sc = ray_doc.object_get(ray_doc.object_get(ray_doc.root, "result"), "structuredContent")
        assert_true(ray_doc.boolean(ray_doc.object_get(ray_sc, "ok"), False))
        assert_true(ray_doc.boolean(ray_doc.object_get(ray_sc, "hit"), False))

        var pie_play_req = '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"play"}}}'
        var pie_play_doc = _parse_json_str(mcp.handle_jsonrpc(pie_play_req, studio, assets, pool))
        var pie_play_sc = pie_play_doc.object_get(pie_play_doc.object_get(pie_play_doc.root, "result"), "structuredContent")
        assert_equal(pie_play_doc.text(pie_play_doc.object_get(pie_play_sc, "pie_mode")), "playing")

        var pie_stop_req = '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"stop"}}}'
        var pie_stop_doc = _parse_json_str(mcp.handle_jsonrpc(pie_stop_req, studio, assets, pool))
        var pie_stop_sc = pie_stop_doc.object_get(pie_stop_doc.object_get(pie_stop_doc.root, "result"), "structuredContent")
        assert_true(pie_stop_doc.boolean(pie_stop_doc.object_get(pie_stop_sc, "restored_exact"), False))

        var undo_req = '{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"undo","steps":1}}}'
        var undo_doc = _parse_json_str(mcp.handle_jsonrpc(undo_req, studio, assets, pool))
        var undo_sc = undo_doc.object_get(undo_doc.object_get(undo_doc.root, "result"), "structuredContent")
        assert_equal(undo_doc.integer(undo_doc.object_get(undo_sc, "applied_steps"), 0), 1)

        var cap_req = (
            '{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{'
            '"name":"render.capture_viewport","arguments":{"output_path":"cap.png","width":64,"height":64}}}'
        )
        var cap_doc = _parse_json_str(mcp.handle_jsonrpc(cap_req, studio, assets, pool))
        var cap_res = cap_doc.object_get(cap_doc.root, "result")
        assert_false(cap_doc.boolean(cap_doc.object_get(cap_res, "isError"), True))
        var cap_sc = cap_doc.object_get(cap_res, "structuredContent")
        assert_true(cap_doc.boolean(cap_doc.object_get(cap_sc, "ok"), False))
        assert_true(file_size(ws_dir + "/cap.png") > 0)

        var cap_ref_req = (
            '{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{'
            '"name":"render.capture_viewport","arguments":{"output_path":"cap2.png","reference_path":"cap.png","width":64,"height":64}}}'
        )
        var cap_ref_doc = _parse_json_str(mcp.handle_jsonrpc(cap_ref_req, studio, assets, pool))
        var cap_ref_sc = cap_ref_doc.object_get(cap_ref_doc.object_get(cap_ref_doc.root, "result"), "structuredContent")
        assert_true(cap_ref_doc.boolean(cap_ref_doc.object_get(cap_ref_sc, "has_flip_diff"), False))

        var despawn_req = (
            '{"jsonrpc":"2.0","id":14,"method":"tools/call","params":{'
            '"name":"ecs.despawn_entity","arguments":{"entity_id":' + String(eid) + "}}}"
        )
        var despawn_doc = _parse_json_str(mcp.handle_jsonrpc(despawn_req, studio, assets, pool))
        var despawn_sc = despawn_doc.object_get(despawn_doc.object_get(despawn_doc.root, "result"), "structuredContent")
        assert_true(despawn_doc.boolean(despawn_doc.object_get(despawn_sc, "despawned"), False))
    finally:
        assets.close()
        pool.stop()

    print("[PASS] test_t1_mcp_v2_protocol_tools_and_auth")


def test_t1_bench_suite_percentiles_and_determinism() raises:
    var small = List[Float64]()
    small.append(5.0)
    small.append(1.0)
    small.append(4.0)
    small.append(2.0)
    small.append(3.0)
    var s5 = compute_latency_percentiles(small)
    assert_equal(s5[0], 3.0)  # p50 idx=(5-1)*50//100=2 -> 3.0
    assert_equal(s5[1], 4.0)  # p99 idx=(5-1)*99//100=3 -> 4.0
    assert_equal(s5[2], 1.0)
    assert_equal(s5[3], 5.0)
    assert_equal(s5[4], 3.0)

    var hundred = List[Float64]()
    for i in range(1, 101):
        hundred.append(Float64(101 - i))
    var s100 = compute_latency_percentiles(hundred)
    assert_equal(s100[0], 50.0)
    assert_equal(s100[1], 99.0)
    assert_equal(s100[2], 1.0)
    assert_equal(s100[3], 100.0)

    var suite = EngineBenchSuite("fa8_t1")
    assert_true(suite.record_samples("step_samples", hundred, budget_p99_ms=100.0))
    assert_true(suite.record_latency("step_latency", small, budget_p99_ms=5.0))
    assert_true(suite.record_single("single_op", 1.25, passed=True, budget_ms=2.0))
    assert_true(suite.record_determinism_gate("det_ok", UInt64(0xCAFEBABE), UInt64(0xCAFEBABE), elapsed_ms=0.5, budget_ms=1.0))

    assert_true(suite.all_passed())
    assert_equal(suite.failed_cases, 0)
    assert_equal(suite.passed_cases, 4)
    assert_true(suite.summary_report().find("PASS") >= 0)
    print("[PASS] test_t1_bench_suite_percentiles_and_determinism")


def test_t1_bench_suite_golden_flip_and_csv(tmp_root: String) raises:
    var bench_dir = tmp_root + "/bench_csv"
    assert_true(make_dirs(bench_dir))

    var suite = EngineBenchSuite("fa8_csv")
    var pixels = List[UInt32](length=16 * 16, fill=UInt32(0xFF2288CC))
    var expected = pixels.copy()

    assert_true(
        suite.verify_golden_pixels(
            "uniform_px",
            pixels,
            expected,
            16,
            16,
            heatmap_out_path=bench_dir + "/unused_heat.png",
        )
    )
    assert_equal(suite.results[0].flip_score, 0.0)

    var golden_png = bench_dir + "/golden.png"
    assert_true(
        suite.verify_golden_frame(
            "golden_boot",
            pixels,
            16,
            16,
            golden_png,
            heatmap_out_path=bench_dir + "/unused_boot_heat.png",
            bootstrap_if_missing=True,
        )
    )
    assert_true(file_size(golden_png) > 0)

    assert_true(
        suite.verify_golden_frame(
            "golden_match",
            pixels,
            16,
            16,
            golden_png,
            heatmap_out_path=bench_dir + "/unused_match_heat.png",
            bootstrap_if_missing=False,
        )
    )

    var lat = List[Float64]()
    lat.append(1.0)
    lat.append(2.0)
    lat.append(3.0)
    assert_true(suite.record_samples("op", lat, budget_p99_ms=5.0))

    var table = suite.to_perf_table(package_name="e2e-t1", date_str="2026-11-20")
    assert_true(table.column("op_p50_ms") >= 0)
    assert_true(table.column("op_p99_ms") >= 0)
    assert_equal(len(table.rows), 1)

    var csv_path = bench_dir + "/perf.csv"
    assert_true(suite.append_perf_history_csv(csv_path, package_name="e2e-t1", date_str="2026-11-20"))
    assert_true(suite.append_perf_history_csv(csv_path, package_name="e2e-t1", date_str="2026-11-20"))
    assert_true(file_size(csv_path) > 0)
    var csv_bytes = read_bytes(csv_path)
    var csv_str = String(unsafe_from_utf8=csv_bytes)
    assert_true(csv_str.find("op_p50_ms") >= 0)
    assert_true(csv_str.find("e2e-t1") >= 0)
    print("[PASS] test_t1_bench_suite_golden_flip_and_csv")


# ===========================================================================
# Tier 2: Boundary / Error / Negative Tests (FA8)
# ===========================================================================

def test_t2_studio_invalid_ids_and_empty_stacks() raises:
    var studio = FlintStudioState()
    assert_false(studio.can_undo())
    assert_false(studio.can_redo())
    assert_false(studio.undo())
    assert_false(studio.redo())

    assert_equal(studio.duplicate_entity(-1), -1)
    assert_equal(studio.duplicate_entity(999), -1)
    assert_false(studio.despawn_entity(-1))
    assert_false(studio.despawn_entity(0))
    assert_false(studio.set_entity_transform(-1, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0)))

    var e0 = studio.spawn_entity("Ephemeral")
    assert_true(e0 >= 0)
    assert_true(studio.despawn_entity(e0))
    assert_equal(studio.duplicate_entity(e0), -1)
    assert_false(studio.despawn_entity(e0))
    assert_false(studio.set_entity_transform(e0, Vec3(1.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))

    var e1 = studio.spawn_entity("Target")
    var bad_field = InspectedField("nonexistent_field", FIELD_F32)
    bad_field.f_val = Float32(1.0)
    assert_false(studio.patch_entity_field(e1, "nonexistent_field", bad_field))

    var wrong_kind = InspectedField("roughness", FIELD_INT)
    wrong_kind.i_val = 42
    assert_false(studio.patch_entity_field(e1, "roughness", wrong_kind))

    var wrong_kind2 = InspectedField("mass", FIELD_STRING)
    wrong_kind2.s_val = "heavy"
    assert_false(studio.patch_entity_field(e1, "mass", wrong_kind2))

    # Coalescing check: spawn e0 (1), despawn e0 (2), spawn e1 (3) -> now 2 coalesced transforms
    var undo_before = len(studio.undo_stack)
    assert_true(studio.set_entity_transform(e1, Vec3(1.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0), coalesce_key=42))
    assert_true(studio.set_entity_transform(e1, Vec3(2.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0), coalesce_key=42))
    assert_equal(len(studio.undo_stack), undo_before + 1)
    assert_true(studio.undo())
    assert_equal(studio.entities[e1].translation.x, Float32(0.0))
    print("[PASS] test_t2_studio_invalid_ids_and_empty_stacks")


def test_t2_studio_pie_and_scene_io_errors(tmp_root: String) raises:
    var studio = FlintStudioState()
    assert_equal(studio.pie.mode, PIE_STOPPED)
    assert_false(studio.pie_pause())
    assert_false(studio.pie_stop())

    assert_equal(studio.step_physics_deterministic(0), 0)
    assert_equal(studio.step_physics_deterministic(-5), 0)

    studio.scene_file_path = ""
    assert_false(studio.save_scene_file())
    assert_false(studio.load_scene_file())
    assert_false(studio.load_scene_file(tmp_root + "/missing_scene.flscene"))

    var corrupt_path = tmp_root + "/corrupt.flscene"
    var corrupt_bytes = List[UInt8]()
    corrupt_bytes.append(UInt8(1))
    corrupt_bytes.append(UInt8(2))
    corrupt_bytes.append(UInt8(3))
    corrupt_bytes.append(UInt8(4))
    corrupt_bytes.append(UInt8(5))
    assert_true(write_bytes(corrupt_path, corrupt_bytes))
    assert_false(studio.load_scene_file(corrupt_path))

    assert_equal(studio.scan_asset_browser(dir_path=tmp_root + "/does_not_exist_dir"), 0)
    print("[PASS] test_t2_studio_pie_and_scene_io_errors")


def test_t2_mcp_v2_jsonrpc_protocol_and_auth_errors(tmp_root: String) raises:
    var mcp = McpServerV2(workspace_root=tmp_root)
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()

    try:
        var r_parse = mcp.handle_jsonrpc("{bad json", studio, assets, pool)
        assert_true(r_parse.find(String(JSONRPC_PARSE_ERROR)) >= 0)
        assert_true(r_parse.find('"id":null') >= 0)

        var r_arr = mcp.handle_jsonrpc("[1,2,3]", studio, assets, pool)
        assert_true(r_arr.find(String(JSONRPC_PARSE_ERROR)) >= 0)

        var r_ver = mcp.handle_jsonrpc('{"jsonrpc":"1.0","id":1,"method":"ping"}', studio, assets, pool)
        assert_true(r_ver.find(String(JSONRPC_INVALID_REQUEST)) >= 0)

        var r_nomethod = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":2}', studio, assets, pool)
        assert_true(r_nomethod.find(String(JSONRPC_INVALID_REQUEST)) >= 0)

        var r_unk_method = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":3,"method":"unknown/op"}', studio, assets, pool)
        assert_true(r_unk_method.find(String(JSONRPC_METHOD_NOT_FOUND)) >= 0)

        var r_no_params = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":4,"method":"tools/call"}', studio, assets, pool)
        assert_true(r_no_params.find(String(JSONRPC_INVALID_PARAMS)) >= 0)

        var r_no_name = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"arguments":{}}}', studio, assets, pool)
        assert_true(r_no_name.find(String(JSONRPC_INVALID_PARAMS)) >= 0)

        var auth_mcp = McpServerV2(workspace_root=tmp_root, auth_token="tok-123")
        var r_unauth = auth_mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":6,"method":"tools/list"}', studio, assets, pool)
        assert_true(r_unauth.find(String(JSONRPC_INVALID_REQUEST)) >= 0)
        assert_false(auth_mcp.session_authenticated)

        var r_bad_tok = auth_mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":7,"method":"initialize","params":{"auth_token":"wrong"}}',
            studio,
            assets,
            pool,
        )
        assert_true(r_bad_tok.find(String(JSONRPC_INVALID_REQUEST)) >= 0)
        assert_false(auth_mcp.session_authenticated)

        var r_unauth_notif = auth_mcp.handle_jsonrpc('{"jsonrpc":"2.0","method":"notifications/initialized"}', studio, assets, pool)
        assert_equal(r_unauth_notif, "")
    finally:
        assets.close()
        pool.stop()

    print("[PASS] test_t2_mcp_v2_jsonrpc_protocol_and_auth_errors")


def test_t2_mcp_v2_tool_error_and_sandbox_rejection(tmp_root: String) raises:
    var ws = tmp_root + "/sandbox_ws"
    assert_true(make_dirs(ws))
    var mcp = McpServerV2(workspace_root=ws)
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()

    try:
        var requests = List[String]()
        requests.append('{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"bogus.tool","arguments":{}}}')
        requests.append('{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ecs.patch_components","arguments":{"entity_id":999}}}')
        requests.append('{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ecs.despawn_entity","arguments":{"entity_id":999}}}')
        requests.append('{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ecs.spawn_entity","arguments":{"position":"not_an_array"}}}')
        requests.append('{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"physics.raycast_query","arguments":{"origin":[1.0,2.0]}}}')
        requests.append('{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"explode"}}}')
        requests.append('{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"rewind"}}}')
        requests.append('{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"../../etc/passwd"}}}')
        requests.append('{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"/tmp/outside_sandbox.png"}}}')

        for i in range(len(requests)):
            var resp = mcp.handle_jsonrpc(requests[i], studio, assets, pool)
            var doc = _parse_json_str(resp)
            assert_true(doc.valid)
            var res = doc.object_get(doc.root, "result")
            assert_true(doc.boolean(doc.object_get(res, "isError"), False))
            var sc = doc.object_get(res, "structuredContent")
            assert_false(doc.boolean(doc.object_get(sc, "ok"), True))
    finally:
        assets.close()
        pool.stop()

    print("[PASS] test_t2_mcp_v2_tool_error_and_sandbox_rejection")


def test_t2_bench_suite_empty_and_budget_failures() raises:
    var empty_stats = compute_latency_percentiles(List[Float64]())
    assert_equal(empty_stats[0], 0.0)
    assert_equal(empty_stats[1], 0.0)
    assert_equal(empty_stats[2], 0.0)
    assert_equal(empty_stats[3], 0.0)
    assert_equal(empty_stats[4], 0.0)

    var suite = EngineBenchSuite("fa8_t2_budget")
    assert_false(suite.record_samples("empty", List[Float64]()))

    var over = List[Float64]()
    over.append(1.0)
    over.append(2.0)
    over.append(10.0)
    # For n=3, p99_idx = (3-1)*99//100 = 1 -> p99_ms = 2.0
    assert_false(suite.record_samples("over_budget", over, budget_p99_ms=1.5))
    assert_false(suite.record_single("single_over", 10.0, passed=True, budget_ms=5.0))
    assert_false(suite.record_single("explicit_fail", 1.0, passed=False))

    assert_false(suite.all_passed())
    assert_equal(suite.failed_cases, 4)
    assert_true(suite.summary_report().find("FAIL") >= 0)
    print("[PASS] test_t2_bench_suite_empty_and_budget_failures")


def test_t2_bench_suite_determinism_mismatch() raises:
    var suite = EngineBenchSuite("fa8_t2_det")
    assert_false(suite.record_determinism_gate("det_mismatch", UInt64(0x1111), UInt64(0x2222)))
    assert_false(suite.results[0].deterministic)
    assert_true(suite.results[0].summary_line().find("!=") >= 0)
    assert_false(suite.all_passed())
    print("[PASS] test_t2_bench_suite_determinism_mismatch")


def test_t2_bench_suite_golden_flip_and_size_mismatch_failures(tmp_root: String) raises:
    var flip_dir = tmp_root + "/flip_fail"
    assert_true(make_dirs(flip_dir))

    var suite = EngineBenchSuite("fa8_t2_flip")
    var black = List[UInt32](length=16 * 16, fill=UInt32(0xFF000000))
    var white = List[UInt32](length=16 * 16, fill=UInt32(0xFFFFFFFF))

    var heat1 = flip_dir + "/heat1.png"
    assert_false(
        suite.verify_golden_pixels(
            "color_diff",
            black,
            white,
            16,
            16,
            heatmap_out_path=heat1,
        )
    )
    assert_true(suite.results[0].flip_score > 0.05)
    assert_true(file_size(heat1) > 0)
    var h1_img = decode_png(read_bytes(heat1), 4096)
    assert_true(h1_img.valid)
    assert_equal(h1_img.width, 16)
    assert_equal(h1_img.height, 16)

    var small_px = List[UInt32](length=4, fill=UInt32(0))
    var heat2 = flip_dir + "/heat2.png"
    assert_false(
        suite.verify_golden_pixels(
            "size_diff",
            black,
            small_px,
            16,
            16,
            heatmap_out_path=heat2,
        )
    )
    assert_equal(suite.results[1].flip_score, 1.0)
    assert_true(suite.results[1].detail.find("size_mismatch") >= 0)

    var heat3 = flip_dir + "/heat3.png"
    assert_false(
        suite.verify_golden_frame(
            "missing_ref",
            black,
            16,
            16,
            flip_dir + "/does_not_exist.png",
            heatmap_out_path=heat3,
            bootstrap_if_missing=False,
        )
    )
    assert_equal(suite.results[2].flip_score, 1.0)
    assert_false(suite.all_passed())
    print("[PASS] test_t2_bench_suite_golden_flip_and_size_mismatch_failures")


def main() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp_root = "/tmp/flint_e2e_m3_fa8_" + String(pid)
    _rm_rf_tmp(tmp_root)
    assert_true(make_dirs(tmp_root))

    try:
        # Tier 1 (6 tests)
        test_t1_studio_entity_transform_patch_undo_redo()
        test_t1_studio_pie_deterministic_and_lz4_restore()
        test_t1_studio_scene_save_load_and_asset_browser(tmp_root)
        test_t1_mcp_v2_protocol_tools_and_auth(tmp_root)
        test_t1_bench_suite_percentiles_and_determinism()
        test_t1_bench_suite_golden_flip_and_csv(tmp_root)

        # Tier 2 (7 tests)
        test_t2_studio_invalid_ids_and_empty_stacks()
        test_t2_studio_pie_and_scene_io_errors(tmp_root)
        test_t2_mcp_v2_jsonrpc_protocol_and_auth_errors(tmp_root)
        test_t2_mcp_v2_tool_error_and_sandbox_rejection(tmp_root)
        test_t2_bench_suite_empty_and_budget_failures()
        test_t2_bench_suite_determinism_mismatch()
        test_t2_bench_suite_golden_flip_and_size_mismatch_failures(tmp_root)
    except e:
        _rm_rf_tmp(tmp_root)
        raise e^

    _rm_rf_tmp(tmp_root)
    print("[SUMMARY] 13/13 passed")
