"""Ship-blocker fixes for games (`tests/test_flint_v5_game_runtime.mojo`).

  - Scene transitions read through the mounted `.flpk` pack, where `flint.sh package` puts scenes.
  - `GameFlow()` with no save folder saves to the game's own player folder instead of nowhere.
  - Versioned saves round-trip their payload, version and label; old plain slots still load.
  - A whole `World` snapshot saves and restores through a slot.
  - `GameSettings` round-trips through its file, clamps bad values, and persists key bindings.
  - `register_builtin_primitives` gives `mesh_id` 0/1 real meshes on a fresh renderer.
"""
from std.ffi import external_call
from std.testing import assert_equal, assert_true

from flint.app import GameFlow
from flint.app.flow import SaveSlotStore
from flint.app.game_runtime import register_builtin_primitives
from flint.app.settings import GameSettings, QUALITY_LOW
from flint.assets.asset_ref import AssetUuid
from flint.assets.flpk_v3 import ASSET_KIND_RAW, FlpkBuildChunk, write_flpk_v3
from flint.assets.server import AssetServer
from flint.ecs import World
from flint.input.input_map import ACTION_BOOL, InputContext, InputMap
from flint.platform.os import file_size, make_dirs, player_save_directory_for, write_bytes
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, ScalableRenderer
from flint.scene import ComponentRegistry, LocalTransform


def _run_dir(tag: String) -> String:
    var d = "build/test_game_runtime_" + String(Int(external_call["getpid", Int32]())) + "/" + tag
    _ = make_dirs(d)
    return d


def _bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for b in s.as_bytes():
        out.append(b)
    return out^


comptime _LEVEL2 = """{"format":"flscn","version":2,"name":"level2","entities":[
{"slot":0,"parent":-1,"components":{"LocalTransform":{"translation":[1.0,2.0,3.0],"rotation":[0.0,0.0,0.0,1.0],"scale":[1.0,1.0,1.0]}}},
{"slot":1,"parent":-1,"components":{"LocalTransform":{"translation":[4.0,5.0,6.0],"rotation":[0.0,0.0,0.0,1.0],"scale":[1.0,1.0,1.0]}}}]}"""


def test_scene_transition_reads_from_pack() raises:
    var d = _run_dir("pack")
    var pack = d + "/assets.flpk"
    var chunks = List[FlpkBuildChunk]()
    chunks.append(FlpkBuildChunk(AssetUuid.from_path("scenes/level2.flscn"), "scenes/level2.flscn", 0, ASSET_KIND_RAW, _bytes(_LEVEL2)))
    assert_true(write_flpk_v3(pack, chunks))
    assert_true(file_size("scenes/level2.flscn") < 0, "the scene exists only inside the pack")

    var assets = AssetServer(enable_watcher=False)
    assert_true(assets.mount_flpk(pack))
    var world = World()
    var reg = ComponentRegistry.with_builtins()
    reg.ensure_registered_in_world(world)
    var flow = GameFlow(save_dir=d + "/saves")
    assert_true(not flow.transition_to_scene(world, reg, "scenes/level2.flscn"), "the disk-only path cannot see packed scenes")
    assert_true(flow.transition_to_scene(world, reg, assets, "scenes/level2.flscn"), "the asset-server path reads the pack")
    assert_equal(flow.current_scene, "scenes/level2.flscn")
    var n = 0
    for i in range(len(world.entities)):
        if world.entities[i].alive:
            n += 1
    assert_equal(n, 2)
    assets.close()


def test_default_save_folder_is_per_game() raises:
    var a = player_save_directory_for("My Game!")
    var b = player_save_directory_for("other")
    assert_true(a.endswith("/MyGame"), "ids are reduced to safe characters: " + a)
    assert_true(a != b, "two games never share a save folder")
    assert_true(player_save_directory_for("").endswith("/default"))
    assert_true(player_save_directory_for("../../etc").endswith("/etc"), "no traversal through the id")
    var flow = GameFlow(game_id="flint_test_runtime")
    assert_true(flow.saves.base_dir.byte_length() > 0, "an unset save_dir still saves somewhere real")


def test_versioned_saves() raises:
    var d = _run_dir("saves")
    var store = SaveSlotStore(d)
    assert_true(store.save_versioned("slot_1", '{"level": 3, "hp": 72}', 4, "Chapter 2"))
    var got = store.load_versioned("slot_1")
    assert_true(got.found)
    assert_equal(got.version, 4)
    assert_equal(got.label, "Chapter 2")
    assert_true(got.payload.find('"level"') >= 0 and got.payload.find("72") >= 0, got.payload)
    assert_true(got.saved_unix > 0)
    assert_true(store.save_versioned("slot_2", "not json at all", 1))
    assert_equal(store.load_versioned("slot_2").payload, "not json at all", "non-JSON payloads survive as text")
    assert_true(store.save_slot("old", '{"legacy": true}'))
    var old = store.load_versioned("old")
    assert_true(old.found and old.version == 0, "a pre-envelope slot loads as version 0")
    assert_true(not store.load_versioned("missing").found)


def test_world_snapshot_slot() raises:
    var d = _run_dir("world")
    var store = SaveSlotStore(d)
    var world = World()
    var reg = ComponentRegistry.with_builtins()
    reg.ensure_registered_in_world(world)
    var e = world.spawn()
    _ = world.insert(e, LocalTransform())
    assert_true(store.save_world("quick", world))
    var restored = World()
    reg.ensure_registered_in_world(restored)
    assert_true(store.load_world("quick", restored))
    var n = 0
    for i in range(len(restored.entities)):
        if restored.entities[i].alive:
            n += 1
    assert_equal(n, 1, "the saved world comes back")
    assert_true(not store.load_world("nope", restored))


def test_settings_round_trip_and_bindings() raises:
    var d = _run_dir("settings")
    var env_k = String("FLINT_SAVE_DIR")
    var env_v = d
    _ = external_call["setenv", Int32](env_k.as_c_string_span(), env_v.as_c_string_span(), Int32(1))
    var s = GameSettings("settings_test")
    s.width = 1920
    s.height = 1080
    s.fullscreen = True
    s.caption_size = 42
    s.music_volume = Float32(0.25)
    s.quality = QUALITY_LOW
    s.language = "de"
    assert_true(s.save())
    var back = GameSettings.load("settings_test")
    assert_equal(back.caption_size, 42)
    assert_equal(back.width, 1920)
    assert_true(back.fullscreen)
    assert_true(abs(back.music_volume - Float32(0.25)) < Float32(1.0e-4))
    assert_equal(back.quality, QUALITY_LOW)
    assert_equal(back.language, "de")

    var bad = GameSettings("x")
    _ = bad.apply_json('{"width": 5, "music_volume": 9.0, "render_scale": 0.1, "quality": 42}')
    assert_equal(bad.width, 1280, "an absurd width is ignored")
    assert_true(bad.music_volume <= Float32(1.0), "volumes are clamped")
    assert_true(bad.render_scale >= Float32(0.5))
    assert_true(bad.quality != 42)
    assert_true(not bad.apply_json("{{garbage"), "a corrupt file does not break defaults")

    var imap = InputMap()
    _ = imap.register_action("jump", ACTION_BOOL)
    var ctx = InputContext("gameplay", priority=10)
    ctx.bind_key("jump", 32)
    imap.add_context(ctx^)
    _ = imap.rebind_key("gameplay", "jump", 74)  # player rebinds Jump to J
    assert_true(s.save_bindings(imap))
    var imap2 = InputMap()
    _ = imap2.register_action("jump", ACTION_BOOL)
    var ctx2 = InputContext("gameplay", priority=10)
    ctx2.bind_key("jump", 32)
    imap2.add_context(ctx2^)
    assert_true(s.load_bindings(imap2))
    assert_true(imap2.save_json().find("74") >= 0, "the rebinding survives a restart")


def test_builtin_primitives() raises:
    var r = ScalableRenderer(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64)
    assert_true(register_builtin_primitives(r), "a fresh renderer gets the cube and sphere")
    assert_true(len(r.cpu_mesh_materials) >= 2)
    assert_true(not register_builtin_primitives(r), "a renderer that already has meshes is left alone")


def main() raises:
    test_scene_transition_reads_from_pack()
    print("[PASS] test_scene_transition_reads_from_pack")
    test_default_save_folder_is_per_game()
    print("[PASS] test_default_save_folder_is_per_game")
    test_versioned_saves()
    print("[PASS] test_versioned_saves")
    test_world_snapshot_slot()
    print("[PASS] test_world_snapshot_slot")
    test_settings_round_trip_and_bindings()
    print("[PASS] test_settings_round_trip_and_bindings")
    test_builtin_primitives()
    print("[PASS] test_builtin_primitives")
    print("[PASS] test_flint_v5_game_runtime: ship-blocker fixes hold")
