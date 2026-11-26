"""Area D round 3 (sota:camera): optical (mechanical) vignetting of the pupil, cat's-eye and oval bokeh.

Model. Seen from an image point at normalised position `p` (`|p| = 1` at the frame corner), the light that
reaches the sensor passes the aperture stop (radius `a = N_max / N` of the wide-open pupil, so stopping down
shrinks it) and two barrel clips: the front element rim, whose image in pupil space is a circle of radius
`rho` centred at `-k_f p`, and the rear element rim, a circle of radius `rho` centred at `+k_r p`. The
transmitted pupil is the intersection of the three. Off axis it narrows radially into the cat's eye; on axis
and once the lens is stopped down far enough (`a + k |p| <= rho`) it is the whole aperture again, so the
effect fades the way it does on a real fast lens (about two stops below wide open for the defaults).

The same pupil gives both effects, so they stay consistent:
  1. Bokeh shape (`sota_dof_gather_shader`): every gather tap is masked by the pupil, in pupil units of
     the *source* pixel's own CoC (`q = u R / CoC_src`). The far field uses `-u` (the source's disc
     must contain this pixel), the near field `+u` (it is mirrored through focus). The near-field coverage
     is renormalised by the masked kernel area, so a clipped foreground stays opaque.
  2. Brightness (`sota_lens_composite_shader`, `optical_vignette_from_pupil`): the image is multiplied by
     the transmitted area fraction `f(p) = A(pupil ∩ clips) / A(pupil)`, estimated with 64 fixed points.
     A defocused highlight keeps its surface brightness and loses area (its energy drops by `f`), while
     an in-focus point gets darker by `f`, as on a real lens.
Anamorphic lenses squeeze the pupil image horizontally (`squeeze` > 1: oval bokeh, taller than wide).

Sources: Kingslake, "Optics in Photography" (1992), ch. 6 on vignetting; Smith, "Modern Optical
Engineering" (4th ed.), sec. 6.3 on mechanical vignetting and the vignetted pupil; Abadie, "A Life of a
Bokeh" (SIGGRAPH 2018 Advances) for the scatter-as-gather mask on the gather side; Jimenez, "Next
Generation Post Processing in Call of Duty: Advanced Warfare" (SIGGRAPH 2014).
"""
from std.collections import List
from std.math import cos, max, min, sin, sqrt
from flint.render.shader_dsl import GpuF32, ShaderDsl, clamp_f, max_f, min_f, sqrt_f

comptime SOTA_PUPIL_SAMPLES: Int = 64


@fieldwise_init
struct SotaPupilClip(Copyable, ImplicitlyCopyable, Movable):
    """Per-frame pupil clip in *aperture* units (the stopped-down aperture has radius 1): clip radius and
    the front / rear clip centre shift per unit image radius."""
    var radius: Float32
    var front_shift: Float32
    var rear_shift: Float32

    def active(self) -> Bool:
        return self.front_shift > Float32(1.0e-5) or self.rear_shift > Float32(1.0e-5)


def sota_pupil_clip(cat_front: Float32, cat_rear: Float32, clip_radius: Float32, fstop: Float32, max_aperture_fstop: Float32) -> SotaPupilClip:
    """`cat_front` / `cat_rear` / `clip_radius` are lens properties in units of the wide-open pupil radius;
    the stopped-down aperture is `a = N_max / N` of it. Dividing by `a` puts the clips in aperture units."""
    var a = max(Float32(0.02), min(Float32(1.0), max_aperture_fstop / max(fstop, Float32(0.1))))
    return SotaPupilClip(max(Float32(0.05), clip_radius) / a, max(Float32(0.0), cat_front) / a, max(Float32(0.0), cat_rear) / a)


def pupil_image_pos_cpu(x: Float32, y: Float32, w: Int, h: Int) -> Tuple[Float32, Float32]:
    """Pixel centre `(x, y)` (pixel units, any resolution) -> image position, `|p| = 1` at the corner."""
    var half_diag = Float32(0.5) * sqrt(Float32(w * w + h * h))
    return ((x - Float32(w) * Float32(0.5)) / half_diag, (y - Float32(h) * Float32(0.5)) / half_diag)


def pupil_inside_cpu(qx: Float32, qy: Float32, px: Float32, py: Float32, clip: SotaPupilClip) -> Bool:
    """Is pupil point `q` (aperture units) transmitted at image position `p`?"""
    if qx * qx + qy * qy > Float32(1.0):
        return False
    var fx = qx + clip.front_shift * px
    var fy = qy + clip.front_shift * py
    if fx * fx + fy * fy > clip.radius * clip.radius:
        return False
    var rx = qx - clip.rear_shift * px
    var ry = qy - clip.rear_shift * py
    return rx * rx + ry * ry <= clip.radius * clip.radius


def pupil_area_fraction_cpu(px: Float32, py: Float32, clip: SotaPupilClip, n: Int = 256) -> Float32:
    """Exact-ish (n x n grid) transmitted fraction of the aperture disc at image position `p`."""
    var inside = 0
    var total = 0
    for j in range(n):
        for i in range(n):
            var qx = (Float32(i) + Float32(0.5)) / Float32(n) * Float32(2.0) - Float32(1.0)
            var qy = (Float32(j) + Float32(0.5)) / Float32(n) * Float32(2.0) - Float32(1.0)
            if qx * qx + qy * qy > Float32(1.0):
                continue
            total += 1
            if pupil_inside_cpu(qx, qy, px, py, clip):
                inside += 1
    return Float32(inside) / Float32(max(1, total))


def pupil_shape_moments_cpu(px: Float32, py: Float32, clip: SotaPupilClip, n: Int = 256) -> Tuple[Float32, Float32]:
    """Standard deviation of the transmitted pupil along the radial direction of `p` and across it
    (tangential), in aperture units. A full disc gives 0.5 / 0.5; a cat's eye is narrower radially."""
    var r = sqrt(px * px + py * py)
    var ux = px / r if r > Float32(1.0e-6) else Float32(1.0)
    var uy = py / r if r > Float32(1.0e-6) else Float32(0.0)
    var s_r = Float64(0.0)
    var s_rr = Float64(0.0)
    var s_t = Float64(0.0)
    var s_tt = Float64(0.0)
    var cnt = Float64(0.0)
    for j in range(n):
        for i in range(n):
            var qx = (Float32(i) + Float32(0.5)) / Float32(n) * Float32(2.0) - Float32(1.0)
            var qy = (Float32(j) + Float32(0.5)) / Float32(n) * Float32(2.0) - Float32(1.0)
            if not pupil_inside_cpu(qx, qy, px, py, clip):
                continue
            var a = Float64(qx * ux + qy * uy)
            var b = Float64(-qx * uy + qy * ux)
            s_r += a
            s_rr += a * a
            s_t += b
            s_tt += b * b
            cnt += 1.0
    if cnt < 1.0:
        return (Float32(0.0), Float32(0.0))
    var mr = s_r / cnt
    var mt = s_t / cnt
    return (Float32(sqrt(max(0.0, s_rr / cnt - mr * mr))), Float32(sqrt(max(0.0, s_tt / cnt - mt * mt))))


def pupil_sample_cpu(i: Int) -> Tuple[Float32, Float32]:
    """Fixed sunflower (Vogel) point `i` of `SOTA_PUPIL_SAMPLES` in the unit disc (area-uniform)."""
    var r = sqrt((Float32(i) + Float32(0.5)) / Float32(SOTA_PUPIL_SAMPLES))
    var th = Float32(i) * Float32(2.39996323)
    return (r * cos(th), r * sin(th))


def pupil_area_fraction_fast_cpu(px: Float32, py: Float32, clip: SotaPupilClip) -> Float32:
    """CPU twin of the GPU estimate: 64 sunflower points, each clip edge ramped over 0.12 aperture radii
    (the ramp keeps the estimate continuous in `p`, so smooth gradients do not band)."""
    var acc = Float32(0.0)
    for i in range(SOTA_PUPIL_SAMPLES):
        var s = pupil_sample_cpu(i)
        var fx = s[0] + clip.front_shift * px
        var fy = s[1] + clip.front_shift * py
        var rx = s[0] - clip.rear_shift * px
        var ry = s[1] - clip.rear_shift * py
        var mf = max(Float32(0.0), min(Float32(1.0), (clip.radius - sqrt(fx * fx + fy * fy)) * Float32(1.0 / 0.12) + Float32(0.5)))
        var mr = max(Float32(0.0), min(Float32(1.0), (clip.radius - sqrt(rx * rx + ry * ry)) * Float32(1.0 / 0.12) + Float32(0.5)))
        acc += mf * mr
    return acc / Float32(SOTA_PUPIL_SAMPLES)


# ----------------------------------------------------------------------------------------------
# Push-constant packing (the gather and lens composite have no spare words for three floats)
# ----------------------------------------------------------------------------------------------

def sota_pack_u16x2(lo: Float32, lo_scale: Float32, hi: Float32, hi_scale: Float32) -> UInt32:
    var a = UInt32(max(Float32(0.0), min(Float32(65535.0), lo * lo_scale + Float32(0.5))))
    var b = UInt32(max(Float32(0.0), min(Float32(65535.0), hi * hi_scale + Float32(0.5))))
    return a | (b << 16)


comptime SOTA_PUPIL_SHIFT_SCALE: Float32 = 4096.0   # shifts up to 16 aperture radii
comptime SOTA_PUPIL_RADIUS_SCALE: Float32 = 2048.0  # clip radius up to 32 aperture radii
comptime SOTA_SQUEEZE_SCALE: Float32 = 16384.0      # anamorphic squeeze up to 4


def sota_pack_pupil_shifts(clip: SotaPupilClip) -> UInt32:
    return sota_pack_u16x2(clip.front_shift, SOTA_PUPIL_SHIFT_SCALE, clip.rear_shift, SOTA_PUPIL_SHIFT_SCALE)


def sota_pack_pupil_radius_squeeze(clip: SotaPupilClip, squeeze: Float32) -> UInt32:
    return sota_pack_u16x2(clip.radius, SOTA_PUPIL_RADIUS_SCALE, max(Float32(0.25), min(Float32(3.99), squeeze)), SOTA_SQUEEZE_SCALE)


# ----------------------------------------------------------------------------------------------
# DSL helpers
# ----------------------------------------------------------------------------------------------

def pupil_clip_mask_dsl(mut dsl: ShaderDsl, qx: GpuF32, qy: GpuF32, cfx: GpuF32, cfy: GpuF32, crx: GpuF32, cry: GpuF32, radius: GpuF32, edge: GpuF32) -> GpuF32:
    """Barrel-clip transmission of pupil point `q`: product of the front and rear clip discs (centres `cf`,
    `cr`), each ramped over `1 / edge` (centred on the rim, like the CoC disc ramp)."""
    var fx = qx - cfx
    var fy = qy - cfy
    var rx = qx - crx
    var ry = qy - cry
    var mf = clamp_f((radius - sqrt_f(fx * fx + fy * fy)) * edge + Float32(0.5), dsl.f32(0.0), dsl.f32(1.0))
    var mr = clamp_f((radius - sqrt_f(rx * rx + ry * ry)) * edge + Float32(0.5), dsl.f32(0.0), dsl.f32(1.0))
    return mf * mr


def pupil_area_fraction_dsl(mut dsl: ShaderDsl, px: GpuF32, py: GpuF32, front: GpuF32, rear: GpuF32, radius: GpuF32) -> GpuF32:
    """GPU twin of `pupil_area_fraction_fast_cpu` (64 unrolled sunflower points)."""
    var cfx = -front * px
    var cfy = -front * py
    var crx = rear * px
    var cry = rear * py
    var acc = dsl.f32(0.0)
    for i in range(SOTA_PUPIL_SAMPLES):
        var s = pupil_sample_cpu(i)
        acc = acc + pupil_clip_mask_dsl(dsl, dsl.f32(s[0]), dsl.f32(s[1]), cfx, cfy, crx, cry, radius, dsl.f32(Float32(1.0 / 0.12)))
    return acc * Float32(1.0 / Float32(SOTA_PUPIL_SAMPLES))
