"""Project-local scene recovery. The scene snapshot and its original path survive a restart."""

from flint.core.serial import BlobReader, BlobWriter, fnv64, fourcc
from flint.platform.os import file_size, file_stamp, list_directory, make_dirs, read_bytes, remove_file, write_file_durable


def recovery_directory(project: String) -> String:
    return (project if project else String(".")) + "/.flint/autosave"


def snapshot_path(project: String, original: String) -> String:
    """`<project>/.flint/autosave/<scene stem>-<fnv64 hex of the full path>.autosave.flscnb`.
    The stem keeps the file recognisable; the hash keeps `a/level.flscn` and `b/level.flscn` apart."""
    var b = original.as_bytes()
    var start = 0
    var stop = len(b)
    for i in range(len(b)):
        if b[i] == 47:  # '/'
            start = i + 1
    for i in range(len(b) - 1, start - 1, -1):
        if b[i] == 46:  # '.'
            stop = i
            break
    var name = String("untitled")
    if stop > start:
        name = String(unsafe_from_utf8=b[start:stop])
    # Never stack suffixes (`level.autosave.autosave`) if a path ever points at an autosave.
    while name.endswith(".autosave"):
        var trimmed = String(name[byte=0 : name.byte_length() - 9])
        name = trimmed^
    if name.byte_length() == 0:
        name = String("untitled")
    var key = List[UInt8](capacity=original.byte_length())
    for c in b:
        key.append(c)
    var h = fnv64(key)
    var digits = String("0123456789abcdef").as_bytes()
    var hex = List[UInt8](capacity=16)
    for k in range(16):
        hex.append(digits[Int((h >> UInt64(60 - 4 * k)) & UInt64(15))])
    return recovery_directory(project) + "/" + name + "-" + String(unsafe_from_utf8=hex) + ".autosave.flscnb"


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
