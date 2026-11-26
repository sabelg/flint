"""Per-Sound Distance Attenuation Presets & Directional Cones (`flint.audio.attenuation`).

`AttenuationPreset` is a fixed-size, trivially copyable description evaluated on the
real-time mixer thread for every spatial voice:
  - Curves: `ATTEN_INVERSE` (`min / (min + rolloff * (d - min))` with a horizon fade to
    zero at `max_dist`), `ATTEN_LINEAR`, `ATTEN_LOG` (linear in dB, `-60 dB` at `max_dist`),
    and `ATTEN_CUSTOM` (piecewise-linear over normalized distance, up to 8 points).
  - Cone: inner/outer full angles (degrees) and the gain applied outside the outer cone,
    measured between the emitter's forward axis and the emitter->listener direction.
"""
from std.math import exp, cos, isfinite
from flint.core.math import Vec3

comptime ATTEN_INVERSE: Int = 0
comptime ATTEN_LINEAR: Int = 1
comptime ATTEN_LOG: Int = 2
comptime ATTEN_CUSTOM: Int = 3
comptime ATTEN_MAX_POINTS: Int = 8


struct AttenuationPreset(Copyable, Movable):
    """Distance curve + optional directional cone for one class of sounds."""
    var curve: Int
    var min_dist: Float32
    var max_dist: Float32
    var rolloff: Float32
    var point_count: Int
    var point_x: Array[Float32, ATTEN_MAX_POINTS]  # normalized distance t in [0, 1]
    var point_y: Array[Float32, ATTEN_MAX_POINTS]  # linear gain
    var cone_enabled: Bool
    var cone_inner_deg: Float32
    var cone_outer_deg: Float32
    var cone_outer_gain: Float32

    def __init__(
        out self,
        curve: Int = ATTEN_INVERSE,
        min_dist: Float32 = 1.0,
        max_dist: Float32 = 100.0,
        rolloff: Float32 = 1.0,
    ):
        self.curve = curve if (curve >= ATTEN_INVERSE and curve <= ATTEN_CUSTOM) else ATTEN_INVERSE
        var mn = min_dist if isfinite(min_dist) else Float32(1.0)
        self.min_dist = max(Float32(0.01), mn)
        var mx = max_dist if isfinite(max_dist) else Float32(100.0)
        self.max_dist = max(self.min_dist + Float32(0.01), mx)
        self.rolloff = max(Float32(0.0), rolloff) if isfinite(rolloff) else Float32(1.0)
        self.point_count = 0
        self.point_x = Array[Float32, ATTEN_MAX_POINTS](fill=0.0)
        self.point_y = Array[Float32, ATTEN_MAX_POINTS](fill=0.0)
        self.cone_enabled = False
        self.cone_inner_deg = 360.0
        self.cone_outer_deg = 360.0
        self.cone_outer_gain = 1.0

    def add_point(mut self, t: Float32, gain: Float32) -> Bool:
        """Appends a custom-curve point (`t` = normalized distance, must be increasing)."""
        if self.point_count >= ATTEN_MAX_POINTS or not isfinite(t) or not isfinite(gain):
            return False
        var tc = max(Float32(0.0), min(Float32(1.0), t))
        if self.point_count > 0 and tc < self.point_x[self.point_count - 1]:
            return False
        self.point_x[self.point_count] = tc
        self.point_y[self.point_count] = max(Float32(0.0), gain)
        self.point_count += 1
        return True

    def set_cone(mut self, inner_deg: Float32, outer_deg: Float32, outer_gain: Float32):
        var i = max(Float32(0.0), min(Float32(360.0), inner_deg)) if isfinite(inner_deg) else Float32(360.0)
        var o = max(i, min(Float32(360.0), outer_deg)) if isfinite(outer_deg) else Float32(360.0)
        self.cone_inner_deg = i
        self.cone_outer_deg = o
        self.cone_outer_gain = max(Float32(0.0), min(Float32(1.0), outer_gain)) if isfinite(outer_gain) else Float32(1.0)
        self.cone_enabled = i < Float32(360.0)

    def distance_gain(self, dist: Float32) -> Float32:
        """Linear gain from the distance curve alone (`1.0` inside `min_dist`)."""
        if not isfinite(dist):
            return 0.0
        var d = max(Float32(0.0), dist)
        if d <= self.min_dist:
            return 1.0
        if d >= self.max_dist:
            return 0.0
        var span = self.max_dist - self.min_dist
        var t = (d - self.min_dist) / span
        if self.curve == ATTEN_LINEAR:
            return Float32(1.0) - t
        if self.curve == ATTEN_LOG:
            # Linear in decibels: 0 dB at min_dist, -60 dB at max_dist.
            return exp(Float32(-6.907755) * t)
        if self.curve == ATTEN_CUSTOM:
            if self.point_count == 0:
                return Float32(1.0) - t
            if t <= self.point_x[0]:
                return self.point_y[0]
            for i in range(1, self.point_count):
                if t <= self.point_x[i]:
                    var x0 = self.point_x[i - 1]
                    var x1 = self.point_x[i]
                    var w = (t - x0) / max(Float32(1.0e-6), x1 - x0)
                    return self.point_y[i - 1] + (self.point_y[i] - self.point_y[i - 1]) * w
            return self.point_y[self.point_count - 1]
        # ATTEN_INVERSE with a smooth fade over the last 20% of the range.
        var g = self.min_dist / (self.min_dist + self.rolloff * (d - self.min_dist))
        if t > Float32(0.8):
            g *= max(Float32(0.0), (Float32(1.0) - t) / Float32(0.2))
        return g

    def cone_gain(self, emitter_fwd: Vec3, to_listener: Vec3) -> Float32:
        """Directional gain from the cone (`1.0` when disabled or `emitter_fwd` is zero)."""
        if not self.cone_enabled:
            return 1.0
        var fl = emitter_fwd.length()
        var tl = to_listener.length()
        if not isfinite(fl) or not isfinite(tl) or fl < Float32(1.0e-6) or tl < Float32(1.0e-6):
            return 1.0
        var c = emitter_fwd.dot(to_listener) / (fl * tl)
        c = max(Float32(-1.0), min(Float32(1.0), c))
        var cos_inner = _cos_half_angle_deg(self.cone_inner_deg)
        var cos_outer = _cos_half_angle_deg(self.cone_outer_deg)
        if c >= cos_inner:
            return 1.0
        if c <= cos_outer:
            return self.cone_outer_gain
        var w = (cos_inner - c) / max(Float32(1.0e-6), cos_inner - cos_outer)
        return Float32(1.0) + (self.cone_outer_gain - Float32(1.0)) * w

    def evaluate(self, emitter_pos: Vec3, emitter_fwd: Vec3, listener_pos: Vec3) -> Float32:
        var rel = listener_pos - emitter_pos
        var g = self.distance_gain(rel.length()) * self.cone_gain(emitter_fwd, rel)
        return g if isfinite(g) else Float32(0.0)


@always_inline
def _cos_half_angle_deg(full_deg: Float32) -> Float32:
    var half_rad = full_deg * Float32(0.5) * Float32(0.017453292519943295)
    return cos(half_rad)


def attenuation_curve_from_name(name: String) -> Int:
    """Maps `"inverse" | "linear" | "log" | "custom"` to a curve id (`-1` if unknown)."""
    if name == "inverse":
        return ATTEN_INVERSE
    if name == "linear":
        return ATTEN_LINEAR
    if name == "log" or name == "logarithmic":
        return ATTEN_LOG
    if name == "custom":
        return ATTEN_CUSTOM
    return -1
