"""An interrupted Studio session can recover an unsaved project scene by its original path."""
from std.ffi import external_call
from std.sys import argv
from std.testing import assert_equal, assert_true
from std.time import monotonic
from flint.core.serial import fnv64
from flint.platform.os import file_size, list_directory, make_dirs, read_bytes, remove_file, run_process_redirected
from flint.studio.autosave import recovery_directory, recovery_original, snapshot_path
from flint.studio.palette_registry import PAL_CMD_RECOVER_AUTOSAVE
from flint.studio.studio_app import FlintStudioState


def interrupted_editor(project: String, original: String) raises:
    var studio = FlintStudioState(project_root=project)
    studio.scene_file_path = original
    _ = studio.spawn_entity("Saved")
    assert_true(studio.save_scene_file())
    _ = studio.spawn_entity("Unsaved")
    studio.autosave_interval_sec = 1
    assert_true(not studio.tick_autosave(0.5))
    assert_true(studio.tick_autosave(0.5))
    assert_true(not studio.tick_autosave(2))
    assert_true(studio.dirty)
    external_call["_exit", NoneType](Int32(77))


def main() raises:
    var args = argv()
    if len(args) == 4 and String(args[1]) == "--interrupt":
        interrupted_editor(String(args[2]), String(args[3]))
        return
    var root = "/tmp/flint-studio-autosave-" + String(external_call["getpid", Int32]()) + "-" + String(monotonic())
    var original = root + "/scenes/level.flscn"
    assert_true(make_dirs(root))
    var child: List[String] = ["/proc/self/exe", "--interrupt", root, original]
    assert_equal(run_process_redirected(child, log_path=root + "/child.log"), Int32(77))
    var original_hash = fnv64(read_bytes(original))
    var snap = snapshot_path(root, original)
    assert_true(file_size(snap) > 0)
    assert_true(snap != snapshot_path(root, root + "/other/level.flscn"))
    var fresh = FlintStudioState(project_root=root)
    assert_true(fresh.scene_file_path != original)
    assert_equal(recovery_original(root, fresh.scene_file_path), original)
    assert_true(fresh.execute_palette_command(PAL_CMD_RECOVER_AUTOSAVE))
    assert_equal(fresh.scene_file_path, original)
    assert_true(fresh.dirty)
    assert_equal(fresh.live_entity_count(), 2)
    assert_equal(fnv64(read_bytes(original)), original_hash)
    _ = fresh.spawn_entity("Third")
    assert_true(fresh.write_autosave())
    assert_equal(len(list_directory(recovery_directory(root))), 2)
    assert_true(fresh.save_scene_file())
    assert_true(not fresh.dirty)
    assert_true(not recovery_original(root, original))
    assert_true(file_size(snap) < 0 and fnv64(read_bytes(original)) != original_hash)
    assert_true(fresh.load_scene_file())
    assert_equal(fresh.live_entity_count(), 3)
    _ = remove_file(original)
    _ = remove_file(root + "/child.log")
    for folder in [recovery_directory(root), root + "/.flint", root + "/scenes", root]:
        var owned = folder.copy()
        _ = external_call["rmdir", Int32](owned.as_c_string_span())
    print("[PASS] Studio crash autosave, discovery, original path, dirty recovery and explicit save")
