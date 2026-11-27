"""Streaming transition failures preserve prior cells, external handles, and saved actors."""
from std.ffi import external_call
from std.testing import assert_true, assert_equal
from std.time import monotonic
from flint.anim.plugin import AnimatorComponent, register_anim_codecs
from flint.assets.cooker_v2 import cook_single_source_bytes
from flint.assets.flpk_v3 import AssetMeta, AssetUuid, FlpkBuildChunk, write_flpk_v3
from flint.assets.server import AssetServer
from flint.core.math import Vec3
from flint.ecs.world import World
from flint.platform.os import make_dirs, read_bytes, remove_file, write_file_durable
from flint.render.plugin import PointLightComponent, register_render_codecs
from flint.scene.scene import ComponentRegistry
from flint.scene.transform_hierarchy import LocalTransform
from flint.scene.world_stream import WorldCell, WorldCellStreamer, CellActor


def text_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for byte in s.as_bytes():
        out.append(byte)
    return out^


def write_cell(path: String, target: String, duplicate: Bool = False) raises:
    var entity = '{"parent":-1,"components":{"CellActor":{"id":"well","target":"' + target + '"},"LocalTransform":{}}}'
    var ents = entity + "," + entity if duplicate else entity
    var json = '{"format":"flscn","version":2,"entities":[' + ents + ']}'
    var data = text_bytes(json)
    assert_true(write_file_durable(path, data))


def check_packed_sunspire(root: String) raises:
    var chunks = List[FlpkBuildChunk]()
    for name in ["grid.json", "south.flscn", "watch.flscn", "keystone.flscn", "landing.flscn", "north.flscn"]:
        var path = "bridge_cells/" + name
        var meta = AssetMeta()
        meta.uuid = AssetUuid.from_path(path)
        var cooked = cook_single_source_bytes(read_bytes("examples/character_course/assets/" + path), meta, path)
        assert_true(cooked.valid and len(cooked.chunks) == 1)
        chunks.append(cooked.chunks.pop())
    var pack = root + "/sunspire.flpk"
    assert_true(write_flpk_v3(pack, chunks))
    var assets = AssetServer(enable_watcher=False)
    assert_true(assets.mount_flpk(pack))
    var cells = WorldCellStreamer.from_manifest(assets, "bridge_cells/grid.json", radius=1)
    var reg = ComponentRegistry.with_builtins()
    register_anim_codecs(reg)
    register_render_codecs(reg)
    var world = World()
    var start = Vec3(0, 2.49, -24)
    assert_true(not cells.update(world, reg, assets, start))
    var marker = cells.resolve(world, "south/lantern")
    assert_true(world.has[PointLightComponent](marker))
    var link = world.get[CellActor](marker).target
    assert_true(not cells.resolve(world, link).valid())
    assert_true(cells.update(world, reg, assets, start))
    var guide = cells.resolve(world, link)
    assert_true(world.has[AnimatorComponent](guide))
    var far = Vec3(0, 2.49, 14)
    assert_true(not cells.update(world, reg, assets, far))
    assert_true(cells.update(world, reg, assets, far))
    assert_true(not world.is_alive(guide) and not cells.resolve(world, link).valid())
    assert_true(cells.update(world, reg, assets, start, load_budget=2))
    var returned = cells.resolve(world, link)
    assert_true(world.has[AnimatorComponent](returned) and returned != guide)
    assert_equal(cells.resident_entity_count(world), 3)
    assert_equal(assets.total_resident_bytes(), 0)
    _ = remove_file(pack)


def main() raises:
    var root = "/tmp/flint-cell-verify-" + String(external_call["getpid", Int32]()) + "-" + String(monotonic())
    assert_true(make_dirs(root))
    check_packed_sunspire(root)
    write_cell(root + "/a.flscn", "B/well")
    write_cell(root + "/b.flscn", "A/well")
    var assets = AssetServer(enable_watcher=False)
    _ = assets.watch_loose_directory(root)
    var stream = WorldCellStreamer(32, 0, root + "/save")
    stream.define_cell(WorldCell("A", -1, 0, "a.flscn"))
    stream.define_cell(WorldCell("B", 0, 0, "b.flscn"))
    stream.define_cell(WorldCell("C", 1, 0, "c.flscn"))
    var reg = ComponentRegistry.with_builtins()
    var world = World()
    # An external consumer keeps its original file handle even after the cell loads and drops its reference.
    var external = assets.load_raw("a.flscn")
    assert_true(stream.update(world, reg, assets, Vec3(-0.01, 0, 0)))
    var a = stream.resolve(world, "A/well")
    assert_true(a.valid() and not stream.resolve(world, "B/well").valid())
    assert_equal(assets.raws.ref_count(external), 1)
    var prop = world.spawn()
    _ = world.insert[LocalTransform](prop, LocalTransform(Vec3(0, 9, 0)))
    _ = world.insert[CellActor](prop, CellActor("new", "B/well"))
    stream.adopt(world, "A", prop)
    world.get_ptr_mut[LocalTransform](a).value()[].translation.y = 8
    assert_true(stream.update(world, reg, assets, Vec3(0, 0, 0)))
    assert_true(not world.is_alive(a))
    var b = stream.resolve(world, "B/well")
    assert_true(b.valid())

    # A missing or duplicate-ID incoming scene rolls back its provisional entities.
    for attempt in range(2):
        if attempt == 1:
            write_cell(root + "/c.flscn", "A/well", duplicate=True)
        var failed = False
        try:
            _ = stream.update(world, reg, assets, Vec3(32, 0, 0))
        except:
            failed = True
        assert_true(failed)
        assert_equal(world.entity_count(), 1)
        assert_true(stream.resolve(world, "B/well") == b)
    stream.radius = 1
    # Retry at the same grid location once the scene is repaired.
    write_cell(root + "/c.flscn", "A/well")
    assert_true(stream.update(world, reg, assets, Vec3(0, 0, 0), load_budget=2))
    var restored = stream.resolve(world, "A/well")
    assert_true(restored.valid() and restored != a)
    assert_equal(world.get_ptr[LocalTransform](restored).value()[].translation.y, Float32(8))
    var adopted = stream.resolve(world, "A/new")
    assert_equal(world.get_ptr[LocalTransform](adopted).value()[].translation.y, Float32(9))
    assert_true(stream.resolve(world, world.get_ptr[CellActor](adopted).value()[].target) == b)
    stream.unload_all(world, reg)
    assert_equal(world.entity_count(), 0)
    assert_equal(assets.raws.ref_count(external), 1)
    _ = assets.raws.release(external)
    _ = remove_file(stream._save_path(0))
    var save_dir = root + "/save"
    _ = external_call["rmdir", Int32](save_dir.as_c_string_span())
    for path in ["a.flscn", "b.flscn", "c.flscn"]:
        _ = remove_file(root + "/" + path)
    _ = external_call["rmdir", Int32](root.as_c_string_span())
    print("[PASS] world cell references, negative coordinates, rollback, adoption, unload and restore")
