"""Read typed motion-recipe fields with errors suitable for Studio's status bar."""
from std.math import isfinite
from flint.assets.json import JsonDocument
from flint.core.math import Vec3
from flint.platform.os import parent_directory


def field(d: JsonDocument, n: Int, key: String) -> Int:
    return d.object_get(n, key)


def string(
    d: JsonDocument, n: Int, key: String, fallback: String = ""
) -> String:
    return d.text(field(d, n, key), fallback)


def num(
    d: JsonDocument, n: Int, key: String, fallback: Float64
) raises -> Float32:
    var i = field(d, n, key)
    if i < 0:
        return Float32(fallback)
    if d.nodes[i].kind != 2 or not isfinite(Float32(d.number(i))):
        raise Error("Expected a finite number for " + key)
    return Float32(d.number(i))


def integer(d: JsonDocument, n: Int, key: String, fallback: Int) raises -> Int:
    var f = num(d, n, key, Float64(fallback))
    if abs(f) > 1e8 or f != Float32(Int(f)):
        raise Error("Expected an integer for " + key)
    return Int(f)


def vector(d: JsonDocument, n: Int, key: String) raises -> Vec3:
    var ar = field(d, n, key)
    if d.array_size(ar) != 3:
        raise Error("Expected [x,y,z] for " + key)
    var vals = List[Float32]()
    for k in range(3):
        var v = d.at(ar, k)
        if d.nodes[v].kind != 2 or not isfinite(Float32(d.number(v))):
            raise Error("Invalid vector for " + key)
        vals.append(Float32(d.number(v)))
    return Vec3(vals[0], vals[1], vals[2])


def relative(file: String, path: String) raises -> String:
    if not path:
        raise Error("Missing asset path in " + file)
    return path if path.startswith("/") else parent_directory(file) + "/" + path
