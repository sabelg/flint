"""Cross and revisit 200 authored-format, cooked cells with a three-cell horizontal window."""
from std.ffi import external_call
from std.testing import assert_equal, assert_true
from std.time import monotonic
from flint.assets.cooker_v2 import cook_single_source_bytes
from flint.assets.flpk_v3 import ASSET_KIND_RAW, AssetMeta, AssetUuid, FlpkBuildChunk, write_flpk_v3
from flint.assets.server import AssetServer
from flint.core.math import Vec3
from flint.core.memory import MEM_ENTITIES, mem_live
from flint.ecs.world import World
from flint.lab.bench_suite import compute_latency_percentiles
from flint.platform.os import make_dirs, process_rss_bytes, remove_file
from flint.scene.scene import ComponentRegistry, Scene
from flint.scene.transform_hierarchy import LocalTransform, set_parent
from flint.scene.world_stream import CellActor, WorldCellStreamer


def add_cooked(mut chunks: List[FlpkBuildChunk], path: String, data: List[UInt8]) raises:
    var meta = AssetMeta()
    meta.uuid = AssetUuid.from_path(path)
    meta.asset_kind = ASSET_KIND_RAW
    var result = cook_single_source_bytes(data, meta, path)
    assert_true(result.valid and len(result.chunks) == 1)
    chunks.append(result.chunks.pop())


def make_pack(path: String, count: Int) raises:
    var chunks = List[FlpkBuildChunk]()
    var manifest = String('{"format":"flint-world-grid","version":1,"cell_size":32,"cells":[')
    var reg = ComponentRegistry.with_builtins()
    reg.register[CellActor]()
    for i in range(count):
        var name = "sector-" + String(i)
        var cell_path = "world/" + name + ".flscnb"
        var scene = Scene(name)
        var w = World()
        reg.ensure_registered_in_world(w)
        var root = w.spawn()
        _ = w.insert[LocalTransform](root, LocalTransform(Vec3(Float32(i * 32), 0, 0)))
        _ = w.insert[CellActor](root, CellActor("gate", "sector-" + String((i + 1) % count) + "/gate"))
        for j in range(7):
            var child = w.spawn()
            _ = w.insert[LocalTransform](child, LocalTransform(Vec3(Float32(j + 1), 1, 0)))
            assert_true(set_parent(w, child, root))
        add_cooked(chunks, cell_path, scene.save_flscnb_bytes(w, reg))
        if i > 0:
            manifest += ","
        manifest += '{"id":"' + name + '","x":' + String(i) + ',"z":0,"scene":"' + cell_path + '"}'
    manifest += "]}"
    var data = List[UInt8]()
    for byte in manifest.as_bytes():
        data.append(byte)
    add_cooked(chunks, "world/grid.json", data)
    assert_true(write_flpk_v3(path, chunks))


def main() raises:
    comptime CELLS = 200
    assert_true(make_dirs("build/labs/world_stream"))
    var base = "build/labs/world_stream/" + String(external_call["getpid", Int32]()) + "-" + String(monotonic())
    var pack = base + ".flpk"
    var saves = base + "-save"
    make_pack(pack, CELLS)
    var assets = AssetServer(enable_watcher=False)
    assert_true(assets.mount_flpk(pack))
    var streamer = WorldCellStreamer.from_manifest(assets, "world/grid.json", radius=1, save_directory=saves)
    var world = World()
    var reg = ComponentRegistry.with_builtins()
    var times = List[Float64]()
    var baseline_rss = process_rss_bytes()
    var sampled_rss = baseline_rss
    var max_cells = 0
    var max_entities = 0
    var state_bytes = 0
    var ecs_bytes = 0
    for step in range(CELLS * 2 - 1):
        var pos = step if step < CELLS else 2 * (CELLS - 1) - step
        var focus = Vec3(Float32(pos * 32), 0, 0)
        var ready = False
        while not ready:
            var t = monotonic()
            ready = streamer.update(world, reg, assets, focus, load_budget=1)
            times.append(Float64(monotonic() - t) / 1e6)
            if step == 0 and not ready:
                var actor = streamer.resolve(world, "sector-0/gate")
                assert_true(actor.valid())
                world.get_ptr_mut[LocalTransform](actor).value()[].translation.y = 7
                streamer.checkpoint(world, reg)
        var actor = streamer.resolve(world, "sector-" + String(pos) + "/gate")
        assert_true(actor.valid())
        var target = world.get_ptr[CellActor](actor).value()[].target
        assert_equal(streamer.resolve(world, target).valid(), pos < CELLS - 1)
        if pos == 0:
            assert_equal(world.get_ptr[LocalTransform](actor).value()[].translation.y, Float32(7))
        max_cells = max(max_cells, streamer.resident_count())
        max_entities = max(max_entities, world.entity_count())
        sampled_rss = max(sampled_rss, process_rss_bytes())
        state_bytes = max(state_bytes, assets.total_resident_bytes())
        ecs_bytes = max(ecs_bytes, mem_live(MEM_ENTITIES))
        assert_true(streamer.resident_count() <= 3)
        assert_true(world.entity_count() <= 24)
    assert_equal(streamer.loads, CELLS * 2 - 2)
    assert_equal(streamer.saves, 1)
    var fresh = WorldCellStreamer.from_manifest(assets, "world/grid.json", radius=1, save_directory=saves)
    var fresh_world = World()
    assert_true(fresh.update(fresh_world, reg, assets, Vec3(), load_budget=2))
    var restored = fresh.resolve(fresh_world, "sector-0/gate")
    assert_equal(fresh_world.get_ptr[LocalTransform](restored).value()[].translation.y, Float32(7))
    var stats = compute_latency_percentiles(times)
    print("cells", CELLS, "loads", streamer.loads, "evictions", streamer.evictions, "saved cells", streamer.saves)
    print("peak settled cells/entities", max_cells, max_entities, "settled asset bytes", state_bytes)
    print("sampled ECS tag / largest cell source bytes", ecs_bytes, streamer.peak_cell_source_bytes)
    print("process RSS baseline/sampled peak", baseline_rss, sampled_rss, "bytes")
    print("window updates", len(times), "p50/p99/max ms", stats[0], stats[1], stats[3])
    _ = remove_file(fresh._save_path(0))
    _ = external_call["rmdir", Int32](saves.as_c_string_span())
    _ = remove_file(pack)
