"""Area E (`sota:E`): lay an eye mesh's front out flat in UV space.

The eye model (`sota_e_characters.emit_sota_e_eye_surface`) needs no eye centre or gaze attribute: it
recovers the eye frame from the surface normal, the UV tangent frame and the UV position, which works when
the UVs are the planar front projection `(u, v) = 0.5 + uv_per_radius * (x, -y) / radius` (x / y along the
eye's right / up, the gaze along +forward). Authored or generated eyeballs usually come with lat-long or
unwrapped UVs; `sota_e_eye_front_uv` rewrites them into that layout in place, so any eyeball mesh works
with the model. The back hemisphere is never seen inside a socket: its vertices are put on the rim circle
(`|uv - 0.5| = uv_per_radius`, sclera), so no second iris appears there and the projection has no seam.
Rotating the gaze is re-running this with a new `forward` (or rotating the instance).
"""
from std.math import max, sqrt
from flint.assets.model import NativeMesh
from flint.core.math import Vec3


def sota_e_eye_front_uv(
    mut m: NativeMesh, center: Vec3, forward: Vec3, up: Vec3, radius: Float32, uv_per_radius: Float32 = 0.5
) -> Int:
    """Rewrites `m.texcoords` in place; returns how many vertices were put on the back-hemisphere rim."""
    var f = forward.normalized()
    var r = f.cross(up).normalized()
    var u = r.cross(f).normalized()
    var inv_r = Float32(1.0) / max(Float32(1e-9), radius)
    var n = m.vertex_count()
    if len(m.texcoords) < n * 2:
        m.texcoords.resize(n * 2, Float32(0.0))
    var back = 0
    for i in range(n):
        var q = Vec3(m.positions[i * 3], m.positions[i * 3 + 1], m.positions[i * 3 + 2]) - center
        var x = q.dot(r) * inv_r
        var y = q.dot(u) * inv_r
        if q.dot(f) < Float32(0.0):
            var l = sqrt(x * x + y * y)
            if l < Float32(1e-6):
                x = Float32(1.0)
                y = Float32(0.0)
            else:
                x = x / l
                y = y / l
            back += 1
        m.texcoords[i * 2] = Float32(0.5) + uv_per_radius * x
        m.texcoords[i * 2 + 1] = Float32(0.5) - uv_per_radius * y
    return back
