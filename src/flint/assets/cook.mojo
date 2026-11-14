"""Cooking one model: its .glb read once, written as the engine holds it.

Shared by the asset pipeline (tools/asset_pipeline.mojo cooks what it
installs) and `tools/run.sh cook` (tools/cook.mojo, every model under
assets/). A cook is skipped while the cooked file's stamp says its source
is unchanged: the same size and time, or the same bytes (then only the time
in the stamp is brought up to date, so later loads need not hash the
source). Before a cooked file is installed it is read back and compared
with the glTF model bit for bit (assets/cooked_model.mojo); a cook that
would load differently is refused, and the game goes on reading the glTF.
"""
from std.ffi import external_call
from flint.core.lz4 import hash_list
from flint.assets.cooked_model import (
    STAMP_CHECKSUM_AT,
    SourceStamp,
    stamp_checksum,
    decode_cooked_model,
    encode_cooked_model,
    model_difference,
    read_cooked_stamp,
)
from flint.assets.glb import inspect_glb
from flint.assets.glb_native import parse_native_glb
from flint.assets.vfs import cooked_model_path
from flint.platform.os import file_stamp, make_dirs, read_at, read_bytes, rename_file, write_bytes

comptime COOK_DONE: Int = 1        # cooked and installed
comptime COOK_CURRENT: Int = 2     # the cooked file already matches its source
comptime COOK_RESTAMPED: Int = 3   # same bytes, newer time: the stamp was updated
comptime COOK_REFUSED: Int = 0     # see the message
comptime STAMP_MTIME_OFFSET: Int = 48  # the source time in a cooked model file


def parent_of(path: String) -> String:
    var bytes = path.as_bytes()
    var end = len(bytes)
    while end > 0 and bytes[end - 1] != 47:
        end -= 1
    if end <= 1:
        return String()
    return String(path[byte=0 : end - 1])


def install(path: String, mut data: List[UInt8]) -> Bool:
    """Beside the target, then renamed: a reader never sees half a file."""
    var directory = parent_of(path)
    if directory.byte_length() > 0 and not make_dirs(directory):
        return False
    var temporary = path + "." + String(Int(external_call["getpid", Int32]())) + ".tmp"
    if not write_bytes(temporary, data):
        _ = external_call["unlink", Int32](temporary.as_c_string_span())
        return False
    return rename_file(temporary, path)


struct CookOutcome(Movable):
    var status: Int
    var message: String
    var raw_bytes: Int     # the .glb
    var cooked_bytes: Int  # the cooked file

    def __init__(out self, status: Int, message: String = String()):
        self.status = status
        self.message = message
        self.raw_bytes = 0
        self.cooked_bytes = 0


def cook_model(source: String, asset_id: UInt64, force: Bool = False) -> CookOutcome:
    var target = cooked_model_path(source)
    var found = file_stamp(source)
    if found[0] <= 0:
        return CookOutcome(COOK_REFUSED, source + ": missing or empty")
    var existing = read_at(target, 0, 96)
    var stamp = read_cooked_stamp(existing, file_stamp(target)[0])
    if not force and stamp.valid and stamp.asset_id == asset_id and stamp.stamp.size == found[0] and stamp.stamp.mtime == found[1]:
        return CookOutcome(COOK_CURRENT)
    var bytes = read_bytes(source)
    var hash = hash_list(bytes)
    if not force and stamp.valid and stamp.asset_id == asset_id and stamp.stamp.size == len(bytes) and stamp.stamp.hash == hash:
        var whole = read_bytes(target)
        if len(whole) > STAMP_MTIME_OFFSET + 8:
            for index in range(8):
                whole[STAMP_MTIME_OFFSET + index] = UInt8((found[1] >> (index * 8)) & 255)
            var checksum = stamp_checksum(whole)
            for index in range(8):
                whole[STAMP_CHECKSUM_AT + index] = UInt8((checksum >> UInt64(index * 8)) & 255)
            if decode_cooked_model(whole).valid and install(target, whole):
                return CookOutcome(COOK_RESTAMPED)
    var info = inspect_glb(source)
    var model = parse_native_glb(bytes)
    if not model.loaded():
        return CookOutcome(COOK_REFUSED, source + ": does not read as glTF, so there is nothing to cook")
    var data = encode_cooked_model(model, info, SourceStamp(found[0], found[1], hash), asset_id)
    var back = decode_cooked_model(data)
    if not back.valid:
        return CookOutcome(COOK_REFUSED, source + ": the cooked file does not read back: " + back.error)
    var difference = model_difference(back.model, model)
    if difference.byte_length() > 0:
        return CookOutcome(COOK_REFUSED, source + ": the cooked model differs from the glTF in " + difference)
    var size = len(data)
    if not install(target, data):
        return CookOutcome(COOK_REFUSED, source + ": cannot write " + target)
    var outcome = CookOutcome(COOK_DONE)
    outcome.raw_bytes = found[0]
    outcome.cooked_bytes = size
    return outcome^
