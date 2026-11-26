"""Area B (render state of the art): shared core of the area-B GI modules.

Region layout of the area-B storage at the end of `BIND_AAA_SURFCACHE`, the control-block word
indices, and the small SPIR-V helpers every area-B shader and hook needs (hashing, cosine sampling,
SH9 basis, control-block loads and the world-radiance-cache irradiance lookup). This module only
depends on the SPIR-V builder and `aaa_visbuffer`, so shared shader code (`aaa_raytracing`,
`aaa_gi_svgf`) can import its hooks without an import cycle.
"""
from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_visbuffer import emit_vec3_cross, emit_vec3_dot, emit_vec3_normalize

# ---------------------------------------------------------------------------------------------
# Region layout (word offsets inside `BIND_AAA_SURFCACHE`)
# ---------------------------------------------------------------------------------------------
comptime SOTA_GI_SURFCACHE_TRIS: Int = 262144  # must equal `FlintAaaVulkanRenderer.max_surfcache_tris`
comptime SOTA_GI_BASE: Int = 4096 + SOTA_GI_SURFCACHE_TRIS * 4
comptime SOTA_GI_CTRL_WORDS: Int = 64
comptime SOTA_GI_PROBE_WORDS: Int = 48
comptime SOTA_GI_MAX_PROBES: Int = 32 * 16 * 32
comptime SOTA_GI_GBUF_WORDS: Int = 16
comptime SOTA_GI_ACCUM_WORDS: Int = 4
comptime SOTA_GI_PROBE_RAYS: Int = 64  # rays per probe per frame == update workgroup width
comptime SOTA_GI_CELL_BIAS: Int = 1 << 20  # cell coordinates are stored biased so they stay unsigned

# GI modes (`SotaGiSettings.mode`)
comptime SOTA_GI_MODE_LEGACY: Int = 0  # unchanged Flint v5 surface-cache GI (default)
comptime SOTA_GI_MODE_WRC: Int = 1  # probe-only diffuse GI, zero per-pixel GI rays
comptime SOTA_GI_MODE_HYBRID: Int = 2  # per-pixel surface-cache rays with the WRC as fallback + bounce source
comptime SOTA_GI_MODE_PATH: Int = 3  # per-pixel diffuse path tracer (+ WRC termination / progressive accumulation)

# Control-block words (relative to `SOTA_GI_BASE`)
comptime SOTA_CTRL_MODE: Int = 0
comptime SOTA_CTRL_FLAGS: Int = 1
comptime SOTA_CTRL_PT_BOUNCES: Int = 2
comptime SOTA_CTRL_PIXEL_RAYS: Int = 3
comptime SOTA_CTRL_ACCUM_N: Int = 4
comptime SOTA_CTRL_FRAME: Int = 5
comptime SOTA_CTRL_MAX_HISTORY: Int = 6
comptime SOTA_CTRL_CLAMP: Int = 7
comptime SOTA_CTRL_BASE_X: Int = 8
comptime SOTA_CTRL_SPACING: Int = 11
comptime SOTA_CTRL_DIM_X: Int = 12
comptime SOTA_CTRL_PROBE_COUNT: Int = 15
comptime SOTA_CTRL_PREV_BANK: Int = 16
comptime SOTA_CTRL_CUR_BANK: Int = 17
comptime SOTA_CTRL_GBUF: Int = 18
comptime SOTA_CTRL_ACCUM: Int = 19
comptime SOTA_CTRL_RENDER_W: Int = 20
comptime SOTA_CTRL_RENDER_H: Int = 21
# ReSTIR / contact / reflection words (`sota_gi_restir.mojo`)
comptime SOTA_CTRL_HIST_PREV: Int = 22  # per-pixel surface history (position + normal) written last frame
comptime SOTA_CTRL_HIST_CUR: Int = 23
comptime SOTA_CTRL_DI_PREV: Int = 24  # ReSTIR DI reservoir banks
comptime SOTA_CTRL_DI_CUR: Int = 25
comptime SOTA_CTRL_GI_PREV: Int = 26  # ReSTIR GI reservoir banks
comptime SOTA_CTRL_GI_CUR: Int = 27
comptime SOTA_CTRL_RESTIR_FLAGS: Int = 28
comptime SOTA_CTRL_DI_SPATIAL: Int = 29  # neighbours per pixel
comptime SOTA_CTRL_DI_RADIUS: Int = 30  # float, pixels
comptime SOTA_CTRL_DI_MCAP: Int = 31  # float, temporal confidence cap
comptime SOTA_CTRL_GI_SPATIAL: Int = 32
comptime SOTA_CTRL_GI_RADIUS: Int = 33
comptime SOTA_CTRL_GI_MCAP: Int = 34
comptime SOTA_CTRL_CONTACT_RAYS: Int = 35  # WRC-only mode: near-field rays per pixel
comptime SOTA_CTRL_CONTACT_RADIUS: Int = 36  # float, metres
comptime SOTA_CTRL_CAM_X: Int = 37  # float x3: camera position (history validation scale)
comptime SOTA_CTRL_LOWRES: Int = 40  # word offset of the reduced-resolution resolve buffer (round 3)

# Control flags
comptime SOTA_FLAG_WRC_VALID: UInt32 = 1  # the read bank holds at least one frame of probe data
comptime SOTA_FLAG_ACCUMULATE: UInt32 = 2  # progressive accumulation (photo mode)
comptime SOTA_FLAG_PT_WRC_TERMINATE: UInt32 = 4  # path tracer ends paths in the WRC instead of truncating
comptime SOTA_FLAG_WRC_FEEDS_SURFCACHE: UInt32 = 8  # surface-cache bounce comes from the WRC (hybrid)
comptime SOTA_FLAG_PROBE_USE_SURFCACHE: UInt32 = 16  # probe ray hits read the surface cache when a slot exists
comptime SOTA_FLAG_REFL_WRC: UInt32 = 32  # RT reflection hits take their indirect diffuse from the WRC

# `SOTA_CTRL_RESTIR_FLAGS` bits
comptime SOTA_RESTIR_DI: UInt32 = 1  # ReSTIR DI temporal + spatial reuse in the deferred shade
comptime SOTA_RESTIR_GI: UInt32 = 2  # ReSTIR GI in the resolve pass (hybrid / path tracer)
comptime SOTA_RESTIR_HIST_VALID: UInt32 = 4  # the previous history / reservoir banks were written last frame
comptime SOTA_RESTIR_HIST_WRITE: UInt32 = 8  # write the surface history this frame

# Probe words
comptime SOTA_PROBE_SH: Int = 0  # 9 SH coefficients x RGB: word = k * 3 + c
comptime SOTA_PROBE_VIS: Int = 27  # 6 axes x (E[t], E[t^2])
comptime SOTA_PROBE_KEY: Int = 39
comptime SOTA_PROBE_HIST: Int = 40
comptime SOTA_PROBE_BACKFACE: Int = 41

comptime SOTA_GI_HIST_WORDS: Int = 4  # x, y, z, packed octahedral normal
comptime SOTA_GI_DI_WORDS: Int = 4  # light id, W, M, unused
comptime SOTA_GI_GI_WORDS: Int = 8  # sample x, y, z, packed normal, half(Lr, Lg), half(Lb, M), W, unused
comptime SOTA_GI_LOWRES_WORDS: Int = 8  # reduced-res resolve sample: gi r, g, b, x, y, z, packed normal, stamp

comptime _PI: Float32 = 3.14159265


def sota_svgf_scratch_words(max_pix: Int) -> Int:
    """Words appended to `BIND_AAA_HISTORY` for the SVGF a-trous ping-pong (8 per pixel, the a-trous record).

    The a-trous steps used to ping-pong through `BIND_AAA_HISTORY[0, 8 * max_pix)`, which is SVGF history bank A;
    on even frames that overwrote the freshly written temporal history with filtered (rgb, z, var, ...) records,
    so the next frame read a depth as the history length and the effective history was 1-2 frames."""
    return max_pix * 8


def sota_gi_region_words(max_pix: Int) -> Int:
    """Words the area-B region needs at the end of the surface-cache buffer."""
    var per_pix = SOTA_GI_GBUF_WORDS + SOTA_GI_ACCUM_WORDS + 2 * (SOTA_GI_HIST_WORDS + SOTA_GI_DI_WORDS + SOTA_GI_GI_WORDS)
    return SOTA_GI_CTRL_WORDS + 2 * SOTA_GI_MAX_PROBES * SOTA_GI_PROBE_WORDS + max_pix * per_pix + sota_gi_lowres_words(max_pix)


def sota_gi_lowres_words(max_pix: Int) -> Int:
    """Reduced-resolution resolve buffer: one record per 2x2 block at most (half res), plus slack for the
    rounded-up edge blocks of odd sizes (`ceil(w/2) * ceil(h/2) <= max_pix / 4 + (w + h) / 2 + 1`)."""
    return SOTA_GI_LOWRES_WORDS * (max_pix // 4 + 16384)


def sota_gi_bank_offset(bank: Int) -> Int:
    return SOTA_GI_BASE + SOTA_GI_CTRL_WORDS + bank * SOTA_GI_MAX_PROBES * SOTA_GI_PROBE_WORDS


def sota_gi_gbuf_offset() -> Int:
    return SOTA_GI_BASE + SOTA_GI_CTRL_WORDS + 2 * SOTA_GI_MAX_PROBES * SOTA_GI_PROBE_WORDS


def sota_gi_accum_offset(max_pix: Int) -> Int:
    return sota_gi_gbuf_offset() + max_pix * SOTA_GI_GBUF_WORDS


def sota_gi_hist_offset(max_pix: Int, bank: Int) -> Int:
    return sota_gi_accum_offset(max_pix) + max_pix * SOTA_GI_ACCUM_WORDS + bank * max_pix * SOTA_GI_HIST_WORDS


def sota_gi_di_offset(max_pix: Int, bank: Int) -> Int:
    return sota_gi_hist_offset(max_pix, 2) + bank * max_pix * SOTA_GI_DI_WORDS


def sota_gi_gi_offset(max_pix: Int, bank: Int) -> Int:
    return sota_gi_di_offset(max_pix, 2) + bank * max_pix * SOTA_GI_GI_WORDS


def sota_gi_lowres_offset(max_pix: Int) -> Int:
    return sota_gi_gi_offset(max_pix, 2)


def sota_gi_bayer_offset(scale: Int, k: Int) -> Tuple[Int, Int]:
    """Sub-pixel `(x, y)` of the representative pixel of a `scale x scale` block on frame phase `k`:
    the inverse of the ordered-dither (Bayer) matrix, so `scale * scale` consecutive frames visit every
    pixel of the block once and consecutive phases are far apart (scale must be a power of two)."""
    var n = scale * scale
    var want = k % max(1, n)
    for y in range(scale):
        for x in range(scale):
            # Bayer value M_2n(x, y) = 4 M_n(x mod n, y mod n) + M_2(x / n, y / n), M_2 = [[0, 2], [3, 1]]:
            # the finest level is the most significant base-4 digit
            var v = 0
            var bit = 1
            while bit < scale:
                var bx = (x & bit) != 0
                var by = (y & bit) != 0
                var d = 0
                if bx and not by:
                    d = 2
                elif by and not bx:
                    d = 3
                elif bx and by:
                    d = 1
                v = v * 4 + d
                bit <<= 1
            if v == want:
                return (x, y)
    return (0, 0)


def sota_sh9_band_weight(k: Int) -> Float32:
    """`A_l / pi` of the clamped-cosine convolution: 1, 2/3, 1/4."""
    if k == 0:
        return Float32(1.0)
    if k < 4:
        return Float32(2.0 / 3.0)
    return Float32(0.25)



# ---------------------------------------------------------------------------------------------
# SPIR-V helpers
# ---------------------------------------------------------------------------------------------
def _emit_hash(mut s: AaaSpirvModule, seed: UInt32) -> Tuple[UInt32, UInt32]:
    """PCG hash: `(next_state, uniform [0, 1))`."""
    var st = s.iadd(s.imul(seed, s.uint(747796405)), s.uint(2891336453))
    var wd = s.imul(s.bit_xor(s.shr(st, s.iadd(s.shr(st, s.uint(28)), s.uint(4))), st), s.uint(277803737))
    var res = s.bit_xor(s.shr(wd, s.uint(22)), wd)
    return (res, s.fmul(s.float_from_uint(s.bit_and(res, s.uint(0x00FFFFFF))), s.float(1.0 / 16777216.0)))


def _emit_cosine_dir(mut s: AaaSpirvModule, nx: UInt32, ny: UInt32, nz: UInt32, u1: UInt32, u2: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
    var r = s.fsqrt(s.nclamp(u1, s.float(0.0), s.float(1.0)))
    var phi = s.fmul(u2, s.float(6.283185307))
    var lx = s.fmul(r, s.fcos(phi))
    var ly = s.fmul(r, s.fsin(phi))
    var lz = s.fsqrt(s.nmax(s.float(0.0001), s.fsub(s.float(1.0), u1)))
    var up_is_y = s.flt(s.fabs(ny), s.float(0.92))
    var t_raw = emit_vec3_cross(s, s.select_f(up_is_y, s.float(0.0), s.float(1.0)), s.select_f(up_is_y, s.float(1.0), s.float(0.0)), s.float(0.0), nx, ny, nz)
    var tang = emit_vec3_normalize(s, t_raw[0], t_raw[1], t_raw[2])
    var bitan = emit_vec3_cross(s, nx, ny, nz, tang[0], tang[1], tang[2])
    return emit_vec3_normalize(
        s,
        s.fma(tang[0], lx, s.fma(bitan[0], ly, s.fmul(nx, lz))),
        s.fma(tang[1], lx, s.fma(bitan[1], ly, s.fmul(ny, lz))),
        s.fma(tang[2], lx, s.fma(bitan[2], ly, s.fmul(nz, lz))),
    )


def _emit_sh9_basis(mut s: AaaSpirvModule, x: UInt32, y: UInt32, z: UInt32) -> List[UInt32]:
    var out = List[UInt32]()
    out.append(s.float(0.282095))
    out.append(s.fmul(s.float(0.488603), y))
    out.append(s.fmul(s.float(0.488603), z))
    out.append(s.fmul(s.float(0.488603), x))
    out.append(s.fmul(s.float(1.092548), s.fmul(x, y)))
    out.append(s.fmul(s.float(1.092548), s.fmul(y, z)))
    out.append(s.fmul(s.float(0.315392), s.fma(s.float(3.0), s.fmul(z, z), s.float(-1.0))))
    out.append(s.fmul(s.float(1.092548), s.fmul(x, z)))
    out.append(s.fmul(s.float(0.546274), s.fsub(s.fmul(x, x), s.fmul(y, y))))
    return out^


def _ctrl(mut s: AaaSpirvModule, b_sc: UInt32, word: Int) -> UInt32:
    return s.load_word(b_sc, s.uint(UInt32(SOTA_GI_BASE + word)))


def _ctrl_f(mut s: AaaSpirvModule, b_sc: UInt32, word: Int) -> UInt32:
    return s.load_float(b_sc, s.uint(UInt32(SOTA_GI_BASE + word)))


def emit_sota_gi_mode(mut s: AaaSpirvModule, b_surfcache: UInt32) -> UInt32:
    """Current area-B GI mode (uint) from the control block."""
    return _ctrl(s, b_surfcache, SOTA_CTRL_MODE)


def emit_sota_wrc_irradiance(
    mut s: AaaSpirvModule,
    b_sc: UInt32,
    bank_off: UInt32,
    px: UInt32,
    py: UInt32,
    pz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """Trilinear, backface- and Chebyshev-weighted WRC lookup at `P` with normal `N`.
    Returns `(E_r / pi, E_g / pi, E_b / pi, weight_sum)`; `weight_sum ~ 0` means no valid probe."""
    var spacing = _ctrl_f(s, b_sc, SOTA_CTRL_SPACING)
    var inv_sp = s.fdiv(s.float(1.0), spacing)
    var base_x = _ctrl(s, b_sc, SOTA_CTRL_BASE_X)
    var base_y = _ctrl(s, b_sc, SOTA_CTRL_BASE_X + 1)
    var base_z = _ctrl(s, b_sc, SOTA_CTRL_BASE_X + 2)
    var dim_x = _ctrl(s, b_sc, SOTA_CTRL_DIM_X)
    var dim_y = _ctrl(s, b_sc, SOTA_CTRL_DIM_X + 1)
    var dim_z = _ctrl(s, b_sc, SOTA_CTRL_DIM_X + 2)
    var bias_f = s.float(Float32(SOTA_GI_CELL_BIAS))

    # Normal bias keeps the lookup point off the surface (DDGI surface bias)
    var nb = s.fmul(spacing, s.float(0.25))
    var qx = s.fma(nx, nb, px)
    var qy = s.fma(ny, nb, py)
    var qz = s.fma(nz, nb, pz)
    var gx = s.fadd(s.fsub(s.fmul(qx, inv_sp), s.float(0.5)), bias_f)
    var gy = s.fadd(s.fsub(s.fmul(qy, inv_sp), s.float(0.5)), bias_f)
    var gz = s.fadd(s.fsub(s.fmul(qz, inv_sp), s.float(0.5)), bias_f)
    var flx = s.ffloor(gx)
    var fly = s.ffloor(gy)
    var flz = s.ffloor(gz)
    var fx = s.fsub(gx, flx)
    var fy = s.fsub(gy, fly)
    var fz = s.fsub(gz, flz)
    var c0x = s.uint_from_float(flx)
    var c0y = s.uint_from_float(fly)
    var c0z = s.uint_from_float(flz)

    var nyb = _emit_sh9_basis(s, nx, ny, nz)
    var acc_r = s.fvar(s.float(0.0))
    var acc_g = s.fvar(s.float(0.0))
    var acc_b = s.fvar(s.float(0.0))
    var acc_w = s.fvar(s.float(0.0))

    for corner in range(8):
        var ox = corner & 1
        var oy = (corner >> 1) & 1
        var oz = (corner >> 2) & 1
        var cx = s.iadd(c0x, s.uint(UInt32(ox)))
        var cy = s.iadd(c0y, s.uint(UInt32(oy)))
        var cz = s.iadd(c0z, s.uint(UInt32(oz)))
        # Inside the scrolled window [base, base + dim)?
        var rx = s.isub(cx, base_x)
        var ry = s.isub(cy, base_y)
        var rz = s.isub(cz, base_z)
        var inside = s.logical_and(s.ult(rx, dim_x), s.logical_and(s.ult(ry, dim_y), s.ult(rz, dim_z)))
        s.begin_if(inside)
        var slot = s.iadd(s.umod(cx, dim_x), s.imul(dim_x, s.iadd(s.umod(cy, dim_y), s.imul(dim_y, s.umod(cz, dim_z)))))
        var pb = s.iadd(bank_off, s.imul(slot, s.uint(UInt32(SOTA_GI_PROBE_WORDS))))
        var key_exp = s.bit_or(
            s.bit_or(s.bit_and(cx, s.uint(0x3FF)), s.shl(s.bit_and(cy, s.uint(0x3FF)), s.uint(10))),
            s.bit_or(s.shl(s.bit_and(cz, s.uint(0x3FF)), s.uint(20)), s.uint(0x40000000)),
        )
        var key = s.load_word(b_sc, s.iadd(pb, s.uint(UInt32(SOTA_PROBE_KEY))))
        var bf = s.load_float(b_sc, s.iadd(pb, s.uint(UInt32(SOTA_PROBE_BACKFACE))))
        s.begin_if(s.logical_and(s.ieq(key, key_exp), s.flt(bf, s.float(0.25))))
        # Probe world position
        var ppx = s.fmul(s.fadd(s.fsub(s.float_from_uint(cx), bias_f), s.float(0.5)), spacing)
        var ppy = s.fmul(s.fadd(s.fsub(s.float_from_uint(cy), bias_f), s.float(0.5)), spacing)
        var ppz = s.fmul(s.fadd(s.fsub(s.float_from_uint(cz), bias_f), s.float(0.5)), spacing)
        var tri = s.fmul(
            s.select_f(s.ieq(s.uint(UInt32(ox)), s.uint(1)), fx, s.fsub(s.float(1.0), fx)),
            s.fmul(
                s.select_f(s.ieq(s.uint(UInt32(oy)), s.uint(1)), fy, s.fsub(s.float(1.0), fy)),
                s.select_f(s.ieq(s.uint(UInt32(oz)), s.uint(1)), fz, s.fsub(s.float(1.0), fz)),
            ),
        )
        # Smooth backface weight from the true surface point
        var tpx = s.fsub(ppx, px)
        var tpy = s.fsub(ppy, py)
        var tpz = s.fsub(ppz, pz)
        var tp_n = emit_vec3_normalize(s, tpx, tpy, tpz)
        var wrap = s.fmul(s.fadd(emit_vec3_dot(s, tp_n[0], tp_n[1], tp_n[2], nx, ny, nz), s.float(1.0)), s.float(0.5))
        var w_bf = s.fadd(s.fmul(wrap, wrap), s.float(0.2))
        # Chebyshev visibility with six-axis moments, from the probe towards the biased point
        var dvx = s.fsub(qx, ppx)
        var dvy = s.fsub(qy, ppy)
        var dvz = s.fsub(qz, ppz)
        var dist = s.fsqrt(s.nmax(s.float(1e-8), emit_vec3_dot(s, dvx, dvy, dvz, dvx, dvy, dvz)))
        var inv_d = s.fdiv(s.float(1.0), dist)
        var ux = s.fmul(dvx, inv_d)
        var uy = s.fmul(dvy, inv_d)
        var uz = s.fmul(dvz, inv_d)
        var ax_x = s.select_u(s.fge(ux, s.float(0.0)), s.uint(0), s.uint(1))
        var ax_y = s.select_u(s.fge(uy, s.float(0.0)), s.uint(2), s.uint(3))
        var ax_z = s.select_u(s.fge(uz, s.float(0.0)), s.uint(4), s.uint(5))
        var vis_b = s.iadd(pb, s.uint(UInt32(SOTA_PROBE_VIS)))
        var wx2 = s.fmul(ux, ux)
        var wy2 = s.fmul(uy, uy)
        var wz2 = s.fmul(uz, uz)
        var m1 = s.fma(
            wx2, s.load_float(b_sc, s.iadd(vis_b, s.imul(ax_x, s.uint(2)))),
            s.fma(wy2, s.load_float(b_sc, s.iadd(vis_b, s.imul(ax_y, s.uint(2)))), s.fmul(wz2, s.load_float(b_sc, s.iadd(vis_b, s.imul(ax_z, s.uint(2)))))),
        )
        var m2 = s.fma(
            wx2, s.load_float(b_sc, s.iadd(vis_b, s.iadd(s.imul(ax_x, s.uint(2)), s.uint(1)))),
            s.fma(
                wy2, s.load_float(b_sc, s.iadd(vis_b, s.iadd(s.imul(ax_y, s.uint(2)), s.uint(1)))),
                s.fmul(wz2, s.load_float(b_sc, s.iadd(vis_b, s.iadd(s.imul(ax_z, s.uint(2)), s.uint(1))))),
            ),
        )
        var variance = s.nmax(s.fabs(s.fsub(m2, s.fmul(m1, m1))), s.float(1e-4))
        var dd = s.nmax(s.float(0.0), s.fsub(dist, m1))
        var cheb = s.fdiv(variance, s.fadd(variance, s.fmul(dd, dd)))
        var cheb3 = s.select_f(s.fle(dist, m1), s.float(1.0), s.fmul(cheb, s.fmul(cheb, cheb)))
        var w_raw = s.fmul(tri, s.fmul(w_bf, s.nmax(cheb3, s.float(0.0))))
        # Crush tiny weights (DDGI) so leaking probes do not dominate
        var w_cr = s.select_f(s.flt(w_raw, s.float(0.2)), s.fmul(w_raw, s.fmul(s.fmul(w_raw, w_raw), s.float(25.0))), w_raw)
        var w = s.nmax(w_cr, s.float(1e-6))
        # SH9 irradiance / pi
        var e_r = s.fvar(s.float(0.0))
        var e_g = s.fvar(s.float(0.0))
        var e_b = s.fvar(s.float(0.0))
        for k in range(9):
            var wk = s.fmul(nyb[k], s.float(sota_sh9_band_weight(k)))
            var cw = s.iadd(pb, s.uint(UInt32(SOTA_PROBE_SH + k * 3)))
            s.store(e_r, s.fma(s.load_float(b_sc, cw), wk, s.load_f(e_r)))
            s.store(e_g, s.fma(s.load_float(b_sc, s.iadd(cw, s.uint(1))), wk, s.load_f(e_g)))
            s.store(e_b, s.fma(s.load_float(b_sc, s.iadd(cw, s.uint(2))), wk, s.load_f(e_b)))
        s.store(acc_r, s.fma(s.nmax(s.load_f(e_r), s.float(0.0)), w, s.load_f(acc_r)))
        s.store(acc_g, s.fma(s.nmax(s.load_f(e_g), s.float(0.0)), w, s.load_f(acc_g)))
        s.store(acc_b, s.fma(s.nmax(s.load_f(e_b), s.float(0.0)), w, s.load_f(acc_b)))
        s.store(acc_w, s.fadd(s.load_f(acc_w), w))
        s.end_if()
        s.end_if()

    var wsum = s.load_f(acc_w)
    var inv_w = s.fdiv(s.float(1.0), s.nmax(wsum, s.float(1e-5)))
    return (s.fmul(s.load_f(acc_r), inv_w), s.fmul(s.load_f(acc_g), inv_w), s.fmul(s.load_f(acc_b), inv_w), wsum)




# ---------------------------------------------------------------------------------------------
# Hook emitted into `aaa_raytracing.emit_rt_textured_hit_shade` (RT reflections / glass / water)
# ---------------------------------------------------------------------------------------------
def emit_sota_reflection_hit_ambient(
    mut s: AaaSpirvModule,
    b_sc: UInt32,
    hx: UInt32,
    hy: UInt32,
    hz: UInt32,
    hnx: UInt32,
    hny: UInt32,
    hnz: UInt32,
    amb_r: UInt32,
    amb_g: UInt32,
    amb_b: UInt32,
    hit_ao: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """When an area-B mode keeps a valid world radiance cache, secondary (reflection) hits take their
    indirect diffuse from it instead of the unoccluded SH9 sky. Returns `(amb_r, amb_g, amb_b,
    keep_surfcache_bool)`: the surface-cache excess term stays on only where the surface cache is
    still updated (legacy and hybrid modes). Inert in mode 0."""
    var mode = _ctrl(s, b_sc, SOTA_CTRL_MODE)
    var flags = _ctrl(s, b_sc, SOTA_CTRL_FLAGS)
    var use = s.logical_and(
        s.ine(mode, s.uint(0)),
        s.ieq(s.bit_and(flags, s.uint(SOTA_FLAG_WRC_VALID | SOTA_FLAG_REFL_WRC)), s.uint(SOTA_FLAG_WRC_VALID | SOTA_FLAG_REFL_WRC)),
    )
    var o_r = s.fvar(amb_r)
    var o_g = s.fvar(amb_g)
    var o_b = s.fvar(amb_b)
    s.begin_if(use)
    var wi = emit_sota_wrc_irradiance(s, b_sc, _ctrl(s, b_sc, SOTA_CTRL_CUR_BANK), hx, hy, hz, hnx, hny, hnz)
    s.begin_if(s.fgt(wi[3], s.float(1e-4)))
    s.store(o_r, s.fmul(wi[0], hit_ao))
    s.store(o_g, s.fmul(wi[1], hit_ao))
    s.store(o_b, s.fmul(wi[2], hit_ao))
    s.end_if()
    s.end_if()
    var sc_dead = s.logical_or(s.ieq(mode, s.uint(UInt32(SOTA_GI_MODE_WRC))), s.ieq(mode, s.uint(UInt32(SOTA_GI_MODE_PATH))))
    return (s.load_f(o_r), s.load_f(o_g), s.load_f(o_b), s.logical_not(s.logical_and(use, sc_dead)))
