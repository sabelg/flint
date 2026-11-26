"""Area F render SOTA: VSM static-page caching that stays correct for dynamic scenes (`sota_perf_vsm`).

v5's `VsmPageManager` keeps rendered shadow pages across frames, but only invalidates a cascade when its
page-snapped origin shifts (or on mesh upload / opt-in skinning refits). Two cases leave stale shadows:

1. **Light direction changes.** Every cascade's light view-projection rotates, so every cached page holds depth
   from the old light. A small sun move does not necessarily shift any page origin (the origin is the camera's
   light-space position quantised to the page size, 0.5 m .. 32 m), so cascades 1-3 kept the old depth.
2. **Casters move.** Rigid instances drawn with a changed world matrix and GPU-skinned meshes changed their
   silhouettes without touching the page cache.

`SotaVsmCache.update_light` / `caster` run once per frame before the VSM requests are processed (hook in
`aaa_renderer.render_frame`, marked `# --- sota:perf hook ---`) and invalidates:
- everything when the sun direction rotated by more than `light_epsilon_deg`,
- for every moved / skinned caster, the pages under its bounding sphere at the old and the new transform,
  per cascade, padded by one page (caster shadows extend along the light direction, which in a directional
  orthographic VSM projects onto the same (u, v) footprint - depth is the only axis the shadow travels).

Static geometry keeps 0 re-rendered pages. Enabled by default (`enabled`), counters are exposed for tests.
"""
from std.collections import List
from std.math import abs, acos, max, min
from flint.core.math import Mat4, Vec3
from flint.render.aaa_vsm import (
    VSM_CASCADE_COUNT,
    VSM_CASCADE_PAGES_DIM,
    VSM_VIRTUAL_PAGES_DIM,
    VsmPageManager,
    vsm_cascade_quadrant_origin,
)


def sota_vsm_light_uv(m: List[Float32], p: Vec3) -> Tuple[Float32, Float32]:
    """Cascade-local light UV of world point `p` (same convention as the VSM shaders:
    u = clip.x * 0.5 + 0.5, v = -clip.y * 0.5 + 0.5) for a column-major `[0..1]`-Z ortho matrix."""
    var cx = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12]
    var cy = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13]
    return (cx * Float32(0.5) + Float32(0.5), -cy * Float32(0.5) + Float32(0.5))


def sota_vsm_invalidate_sphere(
    mut vsm: VsmPageManager,
    cascade_mats: List[List[Float32]],
    center: Vec3,
    radius: Float32,
    toroidal_offsets: List[Tuple[Int, Int]] = List[Tuple[Int, Int]](),  # --- sota:shadow hook --- Area G slots
) -> Int:
    """Invalidates, in every cascade, the pages covering a world sphere (+1 page of padding).
    Returns the number of virtual pages marked invalid."""
    var marked = 0
    for c in range(min(VSM_CASCADE_COUNT, len(cascade_mats))):
        ref m = cascade_mats[c]
        if len(m) < 16:
            continue
        var uv = sota_vsm_light_uv(m, center)
        # ortho scale: |row 0| = 1 / half_extent -> radius in uv = r * sx * 0.5
        var sx = Vec3(m[0], m[4], m[8]).length()
        var r_uv = radius * sx * Float32(0.5)
        var page = Float32(1.0) / Float32(VSM_CASCADE_PAGES_DIM)
        var u0 = uv[0] - r_uv - page
        var u1 = uv[0] + r_uv + page
        var v0 = uv[1] - r_uv - page
        var v1 = uv[1] + r_uv + page
        if u1 < Float32(0.0) or u0 > Float32(1.0) or v1 < Float32(0.0) or v0 > Float32(1.0):
            continue
        var q = vsm_cascade_quadrant_origin(c)
        var x0 = max(0, Int(max(Float32(0.0), u0) * Float32(VSM_CASCADE_PAGES_DIM)))
        var x1 = min(VSM_CASCADE_PAGES_DIM - 1, Int(min(Float32(0.99999), u1) * Float32(VSM_CASCADE_PAGES_DIM)))
        var y0 = max(0, Int(max(Float32(0.0), v0) * Float32(VSM_CASCADE_PAGES_DIM)))
        var y1 = min(VSM_CASCADE_PAGES_DIM - 1, Int(min(Float32(0.99999), v1) * Float32(VSM_CASCADE_PAGES_DIM)))
        # --- sota:shadow hook --- toroidal addressing: window page (px, py) lives in slot (px + ox, py + oy) mod 64
        var tox = toroidal_offsets[c][0] if c < len(toroidal_offsets) else 0
        var toy = toroidal_offsets[c][1] if c < len(toroidal_offsets) else 0
        for py in range(y0, y1 + 1):
            var row = (q[1] + (py + toy) % VSM_CASCADE_PAGES_DIM) * VSM_VIRTUAL_PAGES_DIM + q[0]
            for px in range(x0, x1 + 1):
                var col = (px + tox) % VSM_CASCADE_PAGES_DIM
                if vsm.virtual_valid[row + col]:
                    vsm.virtual_valid[row + col] = False
                    marked += 1
    return marked


def _mat_moved(a: Mat4, b: Mat4) -> Bool:
    for i in range(16):
        if abs(a.values[i] - b.values[i]) > Float32(1e-5):
            return True
    return False


struct SotaVsmCache(Copyable, Movable):
    var enabled: Bool
    var light_epsilon_deg: Float32
    var last_sun: Vec3
    var has_last: Bool
    var last_radiance: Vec3
    var light_changed_frame: Bool  # the sun moved / changed colour this frame
    var gi_boost_frames: Int  # after a light change: frames with the WRC hysteresis capped
    var gi_boost_history: Int  # the cap (probes then blend >= 1/cap of the new radiance per frame)
    var gi_boost_left: Int
    var light_cut_deg: Float32  # a jump larger than this in one frame is a lighting *cut* (also caps TSR history)
    var light_cut_frames: Int
    var tsr_cut_weight: Float32  # TSR history weight cap during the cut window (v5 default weight is 0.82)
    var tsr_cut_left: Int
    var light_cuts: Int
    var light_invalidations: Int  # total full invalidations caused by light rotation
    var caster_pages_last_frame: Int  # pages invalidated by moving / skinned casters this frame
    var moved_casters_last_frame: Int

    def __init__(out self):
        self.enabled = True
        self.light_epsilon_deg = Float32(0.05)
        self.last_sun = Vec3(0.0, 1.0, 0.0)
        self.has_last = False
        self.last_radiance = Vec3(0.0, 0.0, 0.0)
        self.light_changed_frame = False
        self.gi_boost_frames = 8
        self.gi_boost_history = 4
        self.gi_boost_left = 0
        self.light_cut_deg = Float32(2.0)
        self.light_cut_frames = 4
        self.tsr_cut_weight = Float32(0.25)
        self.tsr_cut_left = 0
        self.light_cuts = 0
        self.light_invalidations = 0
        self.caster_pages_last_frame = 0
        self.moved_casters_last_frame = 0

    def light_changed(mut self, sun_dir: Vec3) -> Bool:
        var s = sun_dir.normalized()
        if not self.has_last:
            self.last_sun = s
            self.has_last = True
            return False
        var d = max(Float32(-1.0), min(Float32(1.0), s.dot(self.last_sun)))
        var ang = acos(d) * Float32(57.29578)
        if ang > self.light_epsilon_deg:
            self.last_sun = s
            return True
        return False

    def track_light(mut self, sun_dir: Vec3, sun_radiance: Vec3) -> Bool:
        """Once per frame: did the sun rotate (> `light_epsilon_deg`) or change radiance (> 1 %)?
        Arms the GI boost window on a change."""
        self.light_changed_frame = False
        if not self.enabled:
            return False
        var first = not self.has_last
        var prev = self.last_sun
        var rotated = self.light_changed(sun_dir)
        var jump_deg = acos(max(Float32(-1.0), min(Float32(1.0), sun_dir.normalized().dot(prev)))) * Float32(57.29578)
        var lr = max(Float32(1e-4), self.last_radiance.length())
        var prev_rad = self.last_radiance
        var recoloured = not first and (sun_radiance - self.last_radiance).length() > Float32(0.01) * lr
        self.last_radiance = sun_radiance
        if rotated or recoloured:
            self.light_changed_frame = True
            self.gi_boost_left = self.gi_boost_frames
            var big_colour = (sun_radiance - prev_rad).length() > Float32(0.1) * lr
            if (rotated and jump_deg > self.light_cut_deg) or big_colour:
                self.tsr_cut_left = self.light_cut_frames
                self.light_cuts += 1
        return rotated

    def tsr_history_cap(mut self, weight: Float32) -> Float32:
        """TSR history weight to use this frame: capped for `light_cut_frames` after a lighting cut, because
        a light change moves no pixel, so neither motion-vector disocclusion nor (for diffuse changes that stay
        inside the neighbourhood box) colour clamping rejects the stale history (0.82^n decay = 12+ frames)."""
        if self.tsr_cut_left > 0:
            self.tsr_cut_left -= 1
            return min(weight, self.tsr_cut_weight)
        return weight

    def update_light(mut self, mut vsm: VsmPageManager, sun_dir: Vec3, sun_radiance: Vec3):
        if self.track_light(sun_dir, sun_radiance):
            vsm.invalidate_all()
            self.light_invalidations += 1

    def gi_history_cap(mut self, base: Int) -> Int:
        """WRC max history to use this frame (consumes one boost frame)."""
        if self.gi_boost_left > 0:
            self.gi_boost_left -= 1
            return min(base, self.gi_boost_history)
        return base

    def begin_casters(mut self):
        self.caster_pages_last_frame = 0
        self.moved_casters_last_frame = 0

    def caster(
        mut self,
        mut vsm: VsmPageManager,
        cascade_mats: List[List[Float32]],
        world: Mat4,
        prev_world: Mat4,
        local_center: Vec3,
        local_radius: Float32,
        scale: Float32,
        deforming: Bool,
        toroidal_offsets: List[Tuple[Int, Int]] = List[Tuple[Int, Int]](),  # --- sota:shadow hook ---
    ):
        """One draw instance: invalidates its old + new footprint if it moved or deforms (skinned)."""
        if not self.enabled:
            return
        var moved = _mat_moved(world, prev_world)
        if not moved and not deforming:
            return
        self.moved_casters_last_frame += 1
        var r = local_radius * max(Float32(1e-3), scale)
        self.caster_pages_last_frame += sota_vsm_invalidate_sphere(vsm, cascade_mats, world.transform_point(local_center), r, toroidal_offsets)
        if moved:
            self.caster_pages_last_frame += sota_vsm_invalidate_sphere(vsm, cascade_mats, prev_world.transform_point(local_center), r, toroidal_offsets)
