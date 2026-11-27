"""Project-local scene recovery. The scene snapshot and its original path survive a restart."""
from flint.assets.flpk_v3 import AssetUuid
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.platform.os import file_size, file_stamp, list_directory, make_dirs, read_bytes, remove_file, write_file_durable


def recovery_directory(project: String) -> String:
    return (project if project else String(".")) + "/.flint/autosave"


def snapshot_path(project: String, original: String) -> String:
    return recovery_directory(project) + "/" + AssetUuid.from_path(original).format() + ".autosave.flscnb"


def write_snapshot(project: String, original: String, mut packed_scene: List[UInt8]) -> Bool:
    if not original or not packed_scene or not make_dirs(recovery_directory(project)):
        return False
    var target = snapshot_path(project, original)
    var identity = BlobWriter()
    identity.begin(fourcc("ASRC"), 1)
    identity.text(original)
    identity.end()
    # Publish identity before the first snapshot. An interruption can only leave an unused identity.
    if file_size(target + ".origin") < 0 and not write_file_durable(target + ".origin", identity.data):
        return False
    return write_file_durable(target, packed_scene)


def discard_snapshot(project: String, original: String):
    var target = snapshot_path(project, original)
    if file_size(target) >= 0 and not remove_file(target):
        return
    if file_size(target + ".origin") >= 0:
        _ = remove_file(target + ".origin")


def snapshot_original(project: String, path: String) -> String:
    var meta = BlobReader(read_bytes(path + ".origin"))
    try:
        _ = meta.enter(fourcc("ASRC"), 1)
        var original = meta.text()
        meta.leave()
        if original and snapshot_path(project, original) == path:
            return original
    except:
        pass
    return ""


def recovery_original(project: String, current: String) -> String:
    """Prefer this scene, then the newest recoverable scene in the same project."""
    var latest = String()
    var latest_stamp = -1
    var directory = recovery_directory(project)
    for name in list_directory(directory):
        if not name.endswith(".autosave.flscnb"):
            continue
        var path = directory + "/" + name
        var stamp = file_stamp(path)
        if stamp[0] <= 0:
            continue
        var original = snapshot_original(project, path)
        if not original:
            continue
        var saved = file_stamp(original)
        if saved[0] >= 0 and saved[1] > stamp[1]:
            continue
        if original == current:
            return original
        if stamp[1] > latest_stamp:
            latest_stamp = stamp[1]
            latest = original
    return latest
