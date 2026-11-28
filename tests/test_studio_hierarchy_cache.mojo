"""Outliner row cache (`tests/test_studio_hierarchy_cache.mojo`).

`FlintStudioState.cached_hierarchy_rows` serves the flattened Hierarchy rows from
`hierarchy_row_cache` and rebuilds them only when the key changes (`scene_version`, expand/collapse
version, search query, and the world's restore epoch / slot / live counts). Every mutating operation
must invalidate it, and the cached rows must always equal a fresh `hierarchy_rows_for_query` build.
"""
from std.ffi import external_call
from std.testing import assert_equal, assert_true

from flint.platform.os import make_dirs
from flint.studio import FlintStudioState
from flint.studio.studio_app import StudioEntityMetaComponent


def _run_dir() -> String:
    var pid = Int(external_call["getpid", Int32]())
    var d = "build/test_studio_hierarchy_cache_" + String(pid)
    _ = make_dirs(d)
    return d


def _rows_match_fresh(mut studio: FlintStudioState, label: String) raises:
    var q = studio.outliner_state.search_query
    var cached = studio.outliner_rows()
    var fresh = studio.hierarchy_rows_for_query(q)
    assert_equal(len(cached), len(fresh), "row count matches a fresh rebuild: " + label)
    for i in range(len(fresh)):
        assert_true(
            cached[i].entity_id == fresh[i].entity_id
            and cached[i].parent_id == fresh[i].parent_id
            and cached[i].depth == fresh[i].depth
            and cached[i].has_children == fresh[i].has_children
            and cached[i].expanded == fresh[i].expanded,
            "row " + String(i) + " matches a fresh rebuild: " + label,
        )


def _expect_rebuild(mut studio: FlintStudioState, label: String) raises:
    """The next query rebuilds once (the op invalidated the cache), a second query is served cached,
    and the rows equal a fresh build."""
    var before = studio.hierarchy_row_cache.rebuild_count
    _ = studio.outliner_rows()
    assert_equal(studio.hierarchy_row_cache.rebuild_count, before + 1, "cache invalidated by: " + label)
    _ = studio.outliner_rows()
    assert_equal(studio.hierarchy_row_cache.rebuild_count, before + 1, "unchanged key is served cached after: " + label)
    _rows_match_fresh(studio, label)


def _visible_ids(mut studio: FlintStudioState) -> List[Int]:
    var rows = studio.outliner_rows()
    var out = List[Int]()
    for i in range(len(rows)):
        out.append(rows[i].entity_id)
    return out^


def main() raises:
    var studio = FlintStudioState()
    studio.project_root = _run_dir()
    _ = studio.outliner_rows()

    # Idle frames do not rebuild (the editor's per-frame view sync included).
    var idle = studio.hierarchy_row_cache.rebuild_count
    for _ in range(5):
        studio._sync_compat_view()
        _ = studio.outliner_rows()
    assert_equal(studio.hierarchy_row_cache.rebuild_count, idle, "idle frames reuse the cached rows")

    var a = studio.spawn_entity("Alpha")
    studio._sync_ecs_and_physics()
    _expect_rebuild(studio, "spawn")
    var b = studio.spawn_entity("Bravo")
    var c = studio.spawn_entity("Charlie")
    studio._sync_ecs_and_physics()
    _expect_rebuild(studio, "spawn x2")

    assert_true(studio.reparent_entity(b, a))
    _expect_rebuild(studio, "reparent")
    assert_true(studio.parent_of(b) == a)

    var v = studio.entities[c].copy()
    v.name = "Zulu"
    assert_true(studio.commit_node_edit(c, v^, "Rename"))
    _expect_rebuild(studio, "rename")

    var vh = studio.entities[c].copy()
    vh.visible = False
    assert_true(studio.commit_node_edit(c, vh^, "Hide"))
    _expect_rebuild(studio, "visibility flag")

    assert_true(studio.set_entity_locked(c, True))
    _expect_rebuild(studio, "lock flag")

    # Collapse / expand.
    studio.set_hierarchy_expanded(a, False)
    _expect_rebuild(studio, "collapse")
    var ids_collapsed = _visible_ids(studio)
    assert_true(b not in ids_collapsed, "a collapsed parent hides its child")
    _ = studio.toggle_hierarchy_expanded(a)
    _expect_rebuild(studio, "expand toggle")
    assert_true(b in _visible_ids(studio), "an expanded parent shows its child")

    # Search filter.
    studio.outliner_state.search_query = "zul"
    _expect_rebuild(studio, "filter")
    var f_ids = _visible_ids(studio)
    assert_equal(len(f_ids), 1)
    assert_equal(f_ids[0], c, "the filter matches the renamed entity")
    studio.outliner_state.search_query = ""
    _expect_rebuild(studio, "clear filter")

    # Undo / redo.
    assert_true(studio.undo())
    _expect_rebuild(studio, "undo")
    assert_true(studio.redo())
    _expect_rebuild(studio, "redo")

    # Despawn.
    assert_true(studio.despawn_entity(c))
    _expect_rebuild(studio, "despawn")
    assert_true(c not in _visible_ids(studio))

    # Out-of-band world change picked up by the per-frame view sync (a name changed behind Studio's back).
    var meta = studio.app.world.get_ptr_mut[StudioEntityMetaComponent](studio.resolve_entity(a))
    if meta:
        meta.value()[unsafe_offset=0].name = "Renamed Externally"
    studio._sync_compat_view()
    _expect_rebuild(studio, "external rename seen by the view sync")

    # Save / New / Load.
    var path = studio.project_root + "/cache.flscnb"
    assert_true(studio.save_scene_file(path))
    assert_true(studio.new_scene(force=True))
    _expect_rebuild(studio, "new scene")
    assert_true(studio.load_scene_file(path))
    _expect_rebuild(studio, "load")
    assert_true(a in _visible_ids(studio) and b in _visible_ids(studio), "loaded rows hold the saved entities")

    print("[PASS] test_studio_hierarchy_cache: Outliner rows rebuild only on change and match a fresh build")
