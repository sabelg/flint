"""Append one baked animation to a self-contained GLB, preserving original asset bytes."""
from std.ffi import external_call
from std.math import isfinite
from flint.assets.glb_native import little_u32
from flint.assets.glb_writer import append_f32_le, append_u32_le, pad4
from flint.assets.json import JsonDocument
from flint.assets.model import NativeAnimation
from flint.platform.os import read_bytes, remove_file, write_file_durable


@fieldwise_init
struct _Span(Copyable, ImplicitlyCopyable, Movable):
    var first: Int
    var last: Int


def _fields(data: List[UInt8], start: Int) raises -> Dict[String, _Span]:
    """Consume one JSON object, retaining the original text of every value."""
    var fields = Dict[String, _Span]()
    var doc = JsonDocument()
    doc.valid = True
    doc.index = start
    doc.skip_space(data)
    if doc.index >= len(data) or data[doc.index] != 123:
        raise Error("Expected a GLB JSON object")
    doc.index += 1
    while doc.valid:
        doc.skip_space(data)
        var key = doc.parse_string(data)
        doc.skip_space(data)
        if not doc.valid or doc.index >= len(data) or data[doc.index] != 58:
            break
        doc.index += 1
        doc.skip_space(data)
        var first = doc.index
        if doc.parse_value(data) < 0 or not doc.valid:
            break
        fields[key] = _Span(first, doc.index)
        doc.skip_space(data)
        if doc.index >= len(data):
            break
        if data[doc.index] == 125:
            return fields^
        if data[doc.index] != 44:
            break
        doc.index += 1
    raise Error("Malformed GLB metadata")


def _add_array(data: List[UInt8], span: _Span, added: String) raises -> String:
    if data[span.first] != 91 or data[span.last - 1] != 93:
        raise Error("Expected GLB array")
    var original = String(unsafe_from_utf8=data[span.first + 1 : span.last - 1])
    return "[" + original + ("," if original.strip() else "") + added + "]"


def append_take_glb(
    input: String, var output: String, anim: NativeAnimation
) raises:
    var raw = read_bytes(input)
    if (
        len(raw) < 28
        or little_u32(raw, 0) != 0x46546C67
        or little_u32(raw, 4) != 2
        or little_u32(raw, 8) != len(raw)
        or little_u32(raw, 16) != 0x4E4F534A
    ):
        raise Error("Expected a binary glTF 2.0 character")
    var h = 20 + little_u32(raw, 12)
    if (
        h + 8 > len(raw)
        or little_u32(raw, h + 4) != 0x004E4942
        or h + 8 + little_u32(raw, h) != len(raw)
    ):
        raise Error("Expected a single embedded binary chunk")
    var data = List[UInt8](raw[20:h])
    var root = _fields(data, 0)
    var binary = List[UInt8](raw[h + 8 : len(raw)])
    pad4(binary)
    var arr = JsonDocument()
    arr.valid = True
    arr.index = root["buffers"].first
    var buffers = arr.parse_value(data)
    if (
        arr.array_size(buffers) != 1
        or arr.object_get(arr.at(buffers, 0), "uri") >= 0
    ):
        raise Error("Only single-buffer embedded characters can be exported")
    if "images" in root:
        arr.index = root["images"].first
        var images = arr.parse_value(data)
        for i in range(arr.array_size(images)):
            var uri = arr.text(arr.object_get(arr.at(images, i), "uri"))
            if uri and not uri.startswith("data:"):
                raise Error(
                    "Embed external image files before exporting a take"
                )
    var counts = List[Int]()
    for key in [String("bufferViews"), String("accessors")]:
        arr.index = root[key].first
        counts.append(arr.array_size(arr.parse_value(data)))
    var views = List[String]()
    var accessors = List[String]()
    var channels = List[String]()
    var samplers = List[String]()
    if not anim.channels:
        raise Error("Animation has no channels")
    if not isfinite(anim.duration) or anim.duration < Float32(0.0):
        raise Error("Animation duration must be finite and non-negative")
    for c_idx in range(len(anim.channels)):
        ref ch = anim.channels[c_idx]
        for t in ch.times:
            if not isfinite(t):
                raise Error("Animation channel times must be finite")
        for v in ch.values:
            if not isfinite(v):
                raise Error("Animation channel values must be finite")
    # Every track from the retarget baker shares the same time samples.
    for i in range(-1, len(anim.channels)):
        var time_track = i < 0
        ref ch = anim.channels[max(i, 0)]
        var offset = len(binary)
        var kind = String(
            "SCALAR" if time_track else "VEC4" if ch.path == 1 else "VEC3"
        )
        if time_track:
            for value in ch.times:
                append_f32_le(binary, value)
        else:
            for value in ch.values:
                append_f32_le(binary, value)
        var idx = len(views)
        views.append(
            '{"buffer":0,"byteOffset":'
            + String(offset)
            + ',"byteLength":'
            + String(len(binary) - offset)
            + "}"
        )
        var bounds = (
            ',"min":[0],"max":['
            + String(Float64(anim.duration))
            + "]" if time_track else String()
        )
        accessors.append(
            '{"bufferView":'
            + String(counts[0] + idx)
            + ',"componentType":5126,"count":'
            + String(len(ch.times))
            + ',"type":"'
            + kind
            + '"'
            + bounds
            + "}"
        )
        if not time_track:
            samplers.append(
                '{"input":'
                + String(counts[1])
                + ',"output":'
                + String(counts[1] + idx)
                + "}"
            )
            channels.append(
                '{"sampler":'
                + String(i)
                + ',"target":{"node":'
                + String(ch.node)
                + ',"path":"'
                + ("rotation" if ch.path == 1 else "translation")
                + '"}}'
            )
    var addition = (
        '{"name":"Retargeted take","samplers":['
        + ",".join(samplers)
        + '],"channels":['
        + ",".join(channels)
        + "]}"
    )
    var patches = Dict[Int, Tuple[Int, String]]()
    var v = root["bufferViews"]
    patches[v.first] = (v.last, _add_array(data, v, ",".join(views)))
    var a = root["accessors"]
    patches[a.first] = (a.last, _add_array(data, a, ",".join(accessors)))
    arr.index = root["buffers"].first + 1
    arr.skip_space(data)
    var bf = _fields(data, arr.index)
    var b = bf["byteLength"]
    patches[b.first] = (b.last, String(len(binary)))
    if "animations" in root:
        var span = root["animations"]
        patches[span.first] = (span.last, _add_array(data, span, addition))
    else:
        var end = len(data) - 1
        while end > 0 and data[end] != 125:
            end -= 1
        patches[end] = (end + 1, ',"animations":[' + addition + "]}")
    var result = List[UInt8]()
    var cursor = 0
    while cursor < len(data):
        if cursor in patches:
            ref patch = patches[cursor]
            result.extend(patch[1].as_bytes())
            cursor = patch[0]
        else:
            result.append(data[cursor])
            cursor += 1
    pad4(result, 32)
    var file = List[UInt8]()
    for word in [
        0x46546C67,
        2,
        28 + len(result) + len(binary),
        len(result),
        0x4E4F534A,
    ]:
        append_u32_le(file, word)
    file.extend(result^)
    append_u32_le(file, len(binary))
    append_u32_le(file, 0x004E4942)
    file.extend(binary^)
    var staging = (
        output + "." + String(external_call["getpid", Int32]()) + ".stage"
    )
    if not write_file_durable(staging, file):
        raise Error("Cannot write " + output)
    var linked = external_call["link", Int32](
        staging.as_c_string_span(), output.as_c_string_span()
    )
    _ = remove_file(staging)
    if linked != 0:
        raise Error("Choose a new writable GLB filename")
