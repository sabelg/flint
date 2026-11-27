"""Phase 2 tests for stable Studio entity identity (`tests/test_studio_identity.mojo`).

ECS slots are reused after a despawn. These tests pin that `editor_id` / `EntityRef` never follow a
reused slot to a different entity, that ids survive undo/redo, and that MCP accepts `editor_id`.
"""
from std.math import abs
from std.testing import assert_equal, assert_true

from flint.assets.server import AssetServer
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.devtools.mcp_v2 import McpServerV2
from flint.studio import FlintStudioState


def test_ids_are_unique_and_never_reused() raises:
    var studio = FlintStudioState()
    var a = studio.spawn_entity("A")
    var b = studio.spawn_entity("B")
    var ida = studio.editor_id_of(a)
    var idb = studio.editor_id_of(b)
    assert_true(ida != 0 and idb != 0 and ida != idb)
    var ref_a = studio.entity_ref(a)
    assert_true(studio.despawn_entity(a, record_undo=False))
    var c = studio.spawn_entity("C")
    studio._sync_ecs_and_physics()
    assert_true(studio.editor_id_of(c) != ida, "a new entity in a reused slot gets a new editor_id")
    assert_equal(studio.resolve_ref(ref_a), -1, "a ref to a despawned entity must not resolve to the reused slot")
    assert_equal(studio.slot_for_editor_id(idb), b)


def test_ids_survive_undo_redo() raises:
    var studio = FlintStudioState()
    var a = studio.spawn_entity("Keep")
    var ida = studio.editor_id_of(a)
    assert_true(studio.despawn_entity(a))
    assert_equal(studio.slot_for_editor_id(ida), -1)
    _ = studio.undo()
    studio._sync_ecs_and_physics()
    var back = studio.slot_for_editor_id(ida)
    assert_true(back >= 0, "undoing a delete brings back the same editor_id")
    assert_equal(studio.entities[back].name, "Keep")


def test_mcp_editor_id_targets_the_right_entity() raises:
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=".")
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"identity"}}}',
        studio, assets, pool,
    )
    var a = studio.spawn_entity("Target", position=Vec3(0.0, 0.0, 0.0))
    var ida = studio.editor_id_of(a)
    # Legacy `entity_id` points at the wrong slot on purpose; `editor_id` must win.
    var r = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"entity.set_transform","arguments":{"entity_id":999,"editor_id":'
        + String(ida) + ',"position":[4,0,0]}}}',
        studio, assets, pool,
    )
    assert_true(r.find('"isError":true') < 0, "set_transform by editor_id must succeed: " + r)
    studio._sync_ecs_and_physics()
    assert_true(abs(studio.entities[a].translation.x - 4.0) < 1.0e-4, "editor_id addressed the entity")
    var bad = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"entity.set_transform","arguments":{"editor_id":987654,"position":[1,0,0]}}}',
        studio, assets, pool,
    )
    assert_true(bad.find("unknown editor_id") >= 0, "unknown editor_id is refused, not guessed: " + bad)
    var q = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{"name_contains":"Target"}}}',
        studio, assets, pool,
    )
    assert_true(q.find("editor_id") >= 0, "query results carry editor_id")


def main() raises:
    test_ids_are_unique_and_never_reused()
    print("[PASS] test_ids_are_unique_and_never_reused")
    test_ids_survive_undo_redo()
    print("[PASS] test_ids_survive_undo_redo")
    test_mcp_editor_id_targets_the_right_entity()
    print("[PASS] test_mcp_editor_id_targets_the_right_entity")
    print("[PASS] test_studio_identity: editor ids are stable")
