"""Artist-authored, reusable one-shot particle descriptors."""
from std.math import isfinite
from flint.assets.json import JsonDocument, parse_json
from flint.core.math import Vec3
from flint.render.aaa_gpu_particles import AaaParticleEmitter


def _number(doc: JsonDocument, key: String, default: Float32) raises -> Float32:
    var node = doc.object_get(doc.root, key)
    if node < 0:
        return default
    var value = Float32(doc.number(node))
    if doc.nodes[node].kind != 2 or not isfinite(value):
        raise Error("Invalid particle number " + key)
    return value


def _vector(doc: JsonDocument, key: String, default: Vec3) raises -> Vec3:
    var node = doc.object_get(doc.root, key)
    if node < 0:
        return default
    if doc.array_size(node) != 3:
        raise Error("Invalid particle vector " + key)
    var xyz = SIMD[DType.float32, 4](0)
    for i in range(3):
        var field = doc.at(node, i)
        xyz[i] = Float32(doc.number(field))
        if doc.nodes[field].kind != 2 or not isfinite(xyz[i]):
            raise Error("Invalid particle vector " + key)
    return Vec3(xyz[0], xyz[1], xyz[2])


def load_particle_burst(bytes: List[UInt8]) raises -> AaaParticleEmitter:
    """Loads a generic burst in metres and seconds. Set its origin on each submission."""
    var doc = parse_json(bytes)
    if not doc.valid or doc.text(doc.object_get(doc.root, "format")) != "flint.particle-burst.v1":
        raise Error("Expected a Flint particle burst v1")
    var count = _number(doc, "count", 16)
    if count < 1 or count > 8191 or Float32(Int(count)) != count:
        raise Error("Particle burst count must be an integer from 1 to 8191")
    return AaaParticleEmitter.generic(
        Vec3(), _vector(doc, "velocity", Vec3(0, 1, 0)), Int(count),
        color=_vector(doc, "color", Vec3(1, 1, 1)),
        end_color=_vector(doc, "end_color", Vec3(-1, -1, -1)),
        size=_number(doc, "size", 0.05), end_size=_number(doc, "end_size", -1),
        alpha=_number(doc, "alpha", 1), end_alpha=_number(doc, "end_alpha", 0),
        lifetime=_number(doc, "lifetime", 0.5), spawn_radius=_number(doc, "radius", 0),
        velocity_spread=_number(doc, "spread", 1), gravity_scale=_number(doc, "gravity", 1),
        drag=_number(doc, "drag", 0), additive=doc.boolean(doc.object_get(doc.root, "additive"), False),
        lit=doc.boolean(doc.object_get(doc.root, "lit"), True),
    )
