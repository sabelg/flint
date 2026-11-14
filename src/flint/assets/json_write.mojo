"""Writes a parsed JSON node back as text, keeping what the source said.

Numbers keep the text they were written with (so `13.1` stays `13.1`, never a
rounded binary value), object keys keep their order, and strings are escaped
the way assets/json.mojo reads them. One line per node: callers that want a
layout (the layer file, gameplay/layer_export.mojo) put lines together.
"""
from flint.assets.json import JsonDocument

comptime JSON_NULL: Int = 0
comptime JSON_BOOL: Int = 1
comptime JSON_NUMBER: Int = 2
comptime JSON_STRING: Int = 3
comptime JSON_ARRAY: Int = 4
comptime JSON_OBJECT: Int = 5


def json_quote(text: String) -> String:
    var hex = "0123456789abcdef".as_bytes()
    var bytes = List[UInt8]()
    bytes.append(34)
    for byte in text.as_bytes():
        if byte == 34 or byte == 92:
            bytes.append(92)
            bytes.append(byte)
        elif byte == 10:
            bytes.append(92)
            bytes.append(110)
        elif byte == 9:
            bytes.append(92)
            bytes.append(116)
        elif byte == 13:
            bytes.append(92)
            bytes.append(114)
        elif byte < 32:
            for escaped in [UInt8(92), UInt8(117), UInt8(48), UInt8(48), hex[Int(byte >> 4)], hex[Int(byte & 15)]]:
                bytes.append(escaped)
        else:
            bytes.append(byte)
    bytes.append(34)
    return String(unsafe_from_utf8=bytes)


def json_text(doc: JsonDocument, node: Int) -> String:
    """A node as one line: `{"a": 1, "b": [2, 3]}`."""
    if node < 0 or node >= len(doc.nodes):
        return "null"
    var kind = doc.nodes[node].kind
    if kind == JSON_NULL:
        return "null"
    if kind == JSON_BOOL:
        return "true" if doc.nodes[node].boolean else "false"
    if kind == JSON_NUMBER:
        return doc.nodes[node].text
    if kind == JSON_STRING:
        return json_quote(doc.nodes[node].text)
    var out = String()
    if kind == JSON_ARRAY:
        out += "["
        for index in range(len(doc.nodes[node].children)):
            if index > 0:
                out += ", "
            out += json_text(doc, doc.nodes[node].children[index])
        return out + "]"
    out += "{"
    for index in range(len(doc.nodes[node].children)):
        if index > 0:
            out += ", "
        out += json_quote(doc.nodes[node].keys[index]) + ": " + json_text(doc, doc.nodes[node].children[index])
    return out + "}"
