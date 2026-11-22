from std.ffi import external_call
from flint.ml.motionbricks import MbClip, load_bvh
from flint.ml.take_edit import validate_take
from flint.ml.kimodo_bvh import write_kimodo_bvh
from flint.platform.os import remove_file


def load_take(path: String) raises -> MbClip:
    var clip = load_bvh(path)
    validate_take(clip)
    return clip^


def save_new(clip: MbClip, var path: String) raises:
    # Publish a complete new file atomically, even if another editor exports here.
    var staging = (
        path + "." + String(external_call["getpid", Int32]()) + ".stage"
    )
    write_kimodo_bvh(clip, staging)
    var linked = external_call["link", Int32](
        staging.as_c_string_span(), path.as_c_string_span()
    )
    _ = remove_file(staging)
    if linked != 0:
        raise Error("Cannot create " + path + " (choose a new, writable path)")
