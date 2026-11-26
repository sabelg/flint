"""Occluder dithering (package 3.2): what stands between the camera and
the character is screen-doored away around it instead of the camera pushing in.

Triangles drawn while the occluder pass is on (terrain chunks, the layer's
procedural environment, environment and scenery models) carry
`DITHER_FLAG` (render/gpu_layout.mojo). For each of their pixels the
raster shaders (CPU, amdgpu, Vulkan) compute

    value = d2 * scale + offset        d2 = (x - cx)^2 + (y - cy)^2
    drop  = inverse_z > near_inverse_z and value < bayer4(x, y)

where (x, y) is the integer pixel, (cx, cy) the character's projected chest (the
camera target) in the same pixel convention, and bayer4 the 4x4 ordered
dither level 0..15. `near_inverse_z` keeps everything at or behind the character
(less `margin`) solid, so the floor it stands on and walls behind it never
dither; `scale` and `offset` make a disc whose centre keeps `keep` of the
pixels and whose edge (at `radius` metres around it) keeps all of them.
Dropped pixels are not drawn and write no depth, so the character shows through.

The test is the same float operations in the same order on every backend
(two fused multiply-adds), so the three agree pixel for pixel. It is
presentation only: nothing here reaches the simulation.
"""
from std.math import fma
from flint.core.math import Vec3

comptime DITHER_LEVELS: Int = 16
# Never-dither values: a value no level can exceed, and a depth no pixel reaches.
comptime DITHER_OFF_OFFSET: Float32 = 1.0e9
comptime DITHER_OFF_INVERSE_Z: Float32 = 3.0e38


def bayer4(x: Int, y: Int) -> Int:
    """The 4x4 ordered-dither level of a pixel, 0..15, from bits only:
    4 * M2(x & 1, y & 1) + M2(x >> 1, y >> 1) with M2(x, y) = 2 * (x ^ y) + y."""
    var v = x ^ y
    return ((v & 1) << 3) | ((y & 1) << 2) | (v & 2) | ((y >> 1) & 1)


struct DitherView(Copyable, ImplicitlyCopyable, Movable):
    """One frame's dither disc, in the units the shaders read."""
    var center_x: Float32
    var center_y: Float32
    var scale: Float32
    var offset: Float32
    var near_inverse_z: Float32

    def __init__(out self):
        self.center_x = 0
        self.center_y = 0
        self.scale = 0
        self.offset = DITHER_OFF_OFFSET
        self.near_inverse_z = DITHER_OFF_INVERSE_Z

    def active(self) -> Bool:
        return self.offset < DITHER_OFF_OFFSET

    def drops(self, x: Int, y: Int, inverse_z: Float32) -> Bool:
        """The CPU reference of the shaders' test."""
        if not (inverse_z > self.near_inverse_z):
            return False
        var dx = Float32(x) - self.center_x
        var dy = Float32(y) - self.center_y
        var d2 = fma(dy, dy, dx * dx)
        var value = fma(d2, self.scale, self.offset)
        return value < Float32(bayer4(x & 3, y & 3))


def dither_view(
    camera: Vec3, right: Vec3, up: Vec3, forward: Vec3, focal: Float32, width: Int, height: Int,
    target: Vec3, radius: Float32, margin: Float32, keep: Float32,
) -> DitherView:
    """The disc around `target` (radius in metres at its depth): pixels of
    flagged triangles nearer than its depth less `margin` thin out toward the
    middle, where `keep` (0..1) of them stay. Off when the target is behind
    or too near the camera, or the radius is zero."""
    var view = DitherView()
    var relative = target - camera
    var depth = relative.x * forward.x + relative.y * forward.y + relative.z * forward.z
    if radius <= 0 or depth - margin < 0.3 or focal <= 0:
        return view
    var pixels = focal * radius / depth
    if pixels < 1:
        return view
    var screen_x = Float32(width) * 0.5 + (relative.x * right.x + relative.y * right.y + relative.z * right.z) * focal / depth
    var screen_y = Float32(height) * 0.5 - (relative.x * up.x + relative.y * up.y + relative.z * up.z) * focal / depth
    # Shaders evaluate at whole pixel numbers for pixel centres (+0.5).
    view.center_x = screen_x - 0.5
    view.center_y = screen_y - 0.5
    var kept = max(Float32(0), min(Float32(1), keep))
    view.offset = Float32(DITHER_LEVELS) * kept - 1.0
    view.scale = (Float32(DITHER_LEVELS - 1) - view.offset) / (pixels * pixels)
    view.near_inverse_z = 1.0 / (depth - margin)
    return view
