"""Area G (shadows) render SOTA, round 2: screen-space contact shadows (`sota_shadow_contact`).

The VSM resolves ~1.5-5 texels per cascade-0 page texel and needs a normal + depth bias, so thin contacts (a cup on
a table, feet on the ground, leaf on a branch, grass blades) lose their shadow over the first few centimetres.
Contact shadows march the sun direction through the depth buffer from the shaded point for a short world-space
distance and darken the sun term where the ray passes behind the depth surface by less than a thickness.

Design (after Bend Studio's "screen space shadows" talk, Days Gone, SIGGRAPH 2023, and the classic UE4 / Frostbite
contact shadows; no code was used):
- inline in the deferred shade pass (no extra pass, no extra buffer), only on the raster (VSM) sun path: the RT
  sun shadow already has exact contacts;
- `SOTA_SHADOW_CONTACT_STEPS` = 12 linear steps in world space, length `b_globals[241]` metres (default 0.35),
  per-pixel interleaved-gradient-noise start offset rotated by the frame index (TSR integrates it);
- a hit needs the ray to be behind the depth surface by more than a self-shadow epsilon (0.3 % of the view
  depth) and less than the thickness `b_globals[242]` (default 0.15 m, so thin occluders do not cast long
  shadows behind silhouettes);
- result `1 - fade(screen edge) * hit` multiplies the sun visibility (min with the VSM term).
`b_globals[243]` != 0 enables it (cvar `ren.sota_shadow.contact`, default off).
"""
from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_vsm import SOTA_SHADOW_GLOBALS_CONTACT_WORD

comptime SOTA_SHADOW_CONTACT_STEPS: Int = 12


def emit_sota_shadow_contact(
    mut s: AaaSpirvModule,
    b_globals: UInt32,
    depth_samp: Tuple[UInt32, UInt32],
    px: UInt32,
    py: UInt32,
    w_u: UInt32,
    h_u: UInt32,
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    gnx: UInt32,
    gny: UInt32,
    gnz: UInt32,
    sun_x: UInt32,
    sun_y: UInt32,
    sun_z: UInt32,
) -> UInt32:
    """Contact-shadow sun visibility in [0, 1] (1 when disabled). Must be called in uniform-free control flow
    (it only uses texel fetches)."""
    var vis = s.fvar(s.float(1.0))
    var enabled = s.ine(s.load_word(b_globals, s.uint(UInt32(SOTA_SHADOW_GLOBALS_CONTACT_WORD + 2))), s.uint(0))
    var ndl_g = s.fma(gnx, sun_x, s.fma(gny, sun_y, s.fmul(gnz, sun_z)))
    s.begin_if(s.logical_and(enabled, s.fgt(ndl_g, s.float(0.0))))
    var len_m = s.load_float(b_globals, s.uint(UInt32(SOTA_SHADOW_GLOBALS_CONTACT_WORD)))
    var thick = s.load_float(b_globals, s.uint(UInt32(SOTA_SHADOW_GLOBALS_CONTACT_WORD + 1)))
    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var near_z = s.load_float(b_globals, s.uint(35))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var tan_x = s.load_float(b_globals, s.uint(39))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var tan_y = s.load_float(b_globals, s.uint(43))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))
    var frame_idx = s.load_word(b_globals, s.uint(100))
    var jit_x = s.load_float(b_globals, s.uint(101))
    var jit_y = s.load_float(b_globals, s.uint(102))
    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)

    # interleaved gradient noise (Jimenez 2014), rotated per frame by the golden ratio
    var ign = s.fract(
        s.fmul(
            s.float(52.9829189),
            s.fract(s.fma(s.float_from_uint(px), s.float(0.06711056), s.fmul(s.float_from_uint(py), s.float(0.00583715)))),
        )
    )
    var jit = s.fract(s.fma(s.float_from_uint(s.bit_and(frame_idx, s.uint(63))), s.float(0.61803399), ign))

    # start slightly off the surface along the geometric normal (scaled with view distance)
    var rel_x = s.fsub(wx, cam_x)
    var rel_y = s.fsub(wy, cam_y)
    var rel_z = s.fsub(wz, cam_z)
    var vz0 = s.fma(rel_x, fwd_x, s.fma(rel_y, fwd_y, s.fmul(rel_z, fwd_z)))
    var nb = s.nmax(s.float(0.004), s.fmul(vz0, s.float(0.0015)))
    var ox = s.fma(gnx, nb, wx)
    var oy = s.fma(gny, nb, wy)
    var oz = s.fma(gnz, nb, wz)
    var step = s.fdiv(len_m, s.float(Float32(SOTA_SHADOW_CONTACT_STEPS)))

    var hit = s.fvar(s.float(0.0))
    for i in range(SOTA_SHADOW_CONTACT_STEPS):
        s.begin_if(s.flt(s.load_f(hit), s.float(0.5)))
        var t = s.fmul(step, s.fadd(s.float(Float32(i)), jit))
        var qx = s.fsub(s.fma(sun_x, t, ox), cam_x)
        var qy = s.fsub(s.fma(sun_y, t, oy), cam_y)
        var qz = s.fsub(s.fma(sun_z, t, oz), cam_z)
        var vz = s.fma(qx, fwd_x, s.fma(qy, fwd_y, s.fmul(qz, fwd_z)))
        var vx = s.fma(qx, rgt_x, s.fma(qy, rgt_y, s.fmul(qz, rgt_z)))
        var vy = s.fma(qx, up_x, s.fma(qy, up_y, s.fmul(qz, up_z)))
        var safe_z = s.nmax(vz, s.fmul(near_z, s.float(1.01)))
        var ndc_x = s.fdiv(vx, s.fmul(safe_z, tan_x))
        var ndc_y = s.fdiv(vy, s.fmul(safe_z, tan_y))
        # pixel = ndc -> [0, w) + TSR jitter (the depth buffer was rasterised jittered)
        var sx = s.fma(s.fmul(jit_x, s.float(0.5)), fw, s.fmul(s.fma(ndc_x, s.float(0.5), s.float(0.5)), fw))
        var sy = s.fsub(s.fmul(s.fma(s.fneg(ndc_y), s.float(0.5), s.float(0.5)), fh), s.fmul(s.fmul(jit_y, s.float(0.5)), fh))
        var on_screen = s.logical_and(
            s.logical_and(s.fge(sx, s.float(0.0)), s.flt(sx, fw)),
            s.logical_and(s.logical_and(s.fge(sy, s.float(0.0)), s.flt(sy, fh)), s.fgt(vz, near_z)),
        )
        s.begin_if(on_screen)
        var tex = s.uvec2(s.saturate_uint(sx), s.saturate_uint(sy))
        var rev = s.extract_f(s.texel_fetch_f2d(depth_samp, tex, s.uint(0)), 0)
        var z_s = s.fdiv(near_z, s.nmax(rev, s.float(1e-7)))
        var behind = s.fsub(vz, z_s)
        var eps = s.fmul(vz, s.float(0.003))
        var occluded = s.logical_and(s.fgt(behind, eps), s.flt(behind, thick))
        s.begin_if(occluded)
        # fade near the screen border, where the occluder may be off screen
        var ex = s.nmin(sx, s.fsub(fw, sx))
        var ey = s.nmin(sy, s.fsub(fh, sy))
        var edge = s.nclamp(s.fmul(s.nmin(ex, ey), s.float(0.05)), s.float(0.0), s.float(1.0))
        s.store(hit, s.nmax(edge, s.float(0.501)))
        s.store(vis, s.fsub(s.float(1.0), edge))
        s.end_if()
        s.end_if()
        s.end_if()
    s.end_if()
    return s.load_f(vis)
