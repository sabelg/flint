"""Area B (render state of the art): ReSTIR DI reuse and ReSTIR GI.

- **ReSTIR DI** (Bitterli et al. 2020, "Spatiotemporal reservoir resampling for real-time ray tracing
  with dynamic direct lighting"). The v5 deferred shade already draws 4 RIS candidates per pixel from
  its froxel cluster. The hook below streams that reservoir together with the previous frame's
  reservoir at the motion-vector-reprojected pixel (temporal reuse, confidence capped) and with
  `k` previous-frame neighbours around it (fused spatiotemporal reuse, as RTXDI's fused mode does).
  The target function is the candidate pass's own unshadowed `luma * attenuation * N.L`. Surfaces are
  validated against a per-pixel position / normal history. After the final shadow ray an occluded
  sample is stored with `W = 0` (visibility reuse; the paper's biased variant).
- **ReSTIR GI** (Ouyang et al. 2021, "ReSTIR GI: Path Resampling for Real-Time Path Tracing"). Each
  hybrid / path-traced ray is a candidate `(x_s, n_s, L_o)`; reservoirs reuse temporally and
  spatially (previous-frame neighbours, solid-angle Jacobian, one visibility ray per spatial
  candidate). The estimate `f(y) W` goes to the demodulated GI channel, so SVGF / TSR still run.

Everything lives in the area-B region of `BIND_AAA_SURFCACHE` (see `sota_gi_core.mojo`); the hooks
are inert unless `SOTA_CTRL_RESTIR_FLAGS` enables them.
"""
from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_visbuffer import emit_vec3_dot, emit_vec3_normalize
from flint.render.aaa_clustered_lights import AAA_CLUSTER_LIGHT_WORDS, AAA_CLUSTER_LIGHTS_BASE
from flint.render.aaa_raytracing import emit_rt_shadow_query
from flint.render.sota_gi_core import (
    SOTA_CTRL_CAM_X,
    SOTA_CTRL_DI_CUR,
    SOTA_CTRL_DI_MCAP,
    SOTA_CTRL_DI_PREV,
    SOTA_CTRL_DI_RADIUS,
    SOTA_CTRL_DI_SPATIAL,
    SOTA_CTRL_FRAME,
    SOTA_CTRL_GI_CUR,
    SOTA_CTRL_GI_MCAP,
    SOTA_CTRL_GI_PREV,
    SOTA_CTRL_GI_RADIUS,
    SOTA_CTRL_GI_SPATIAL,
    SOTA_CTRL_HIST_CUR,
    SOTA_CTRL_HIST_PREV,
    SOTA_CTRL_RESTIR_FLAGS,
    SOTA_GI_DI_WORDS,
    SOTA_GI_GI_WORDS,
    SOTA_GI_HIST_WORDS,
    SOTA_RESTIR_DI,
    SOTA_RESTIR_HIST_VALID,
    SOTA_RESTIR_HIST_WRITE,
    _ctrl,
    _ctrl_f,
    _emit_hash,
)

comptime SOTA_RESTIR_NONE: UInt32 = 0xFFFFFFFF
comptime SOTA_RESTIR_MAX_SPATIAL: Int = 8


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def emit_sota_oct_pack(mut s: AaaSpirvModule, nx: UInt32, ny: UInt32, nz: UInt32) -> UInt32:
    """Octahedral normal in two halves, offset by +2 so a valid entry is never the zero word."""
    var inv_l1 = s.fdiv(s.float(1.0), s.nmax(s.float(1e-8), s.fadd(s.fabs(nx), s.fadd(s.fabs(ny), s.fabs(nz)))))
    var ox = s.fmul(nx, inv_l1)
    var oy = s.fmul(ny, inv_l1)
    var neg = s.flt(nz, s.float(0.0))
    var sx = s.select_f(s.fge(ox, s.float(0.0)), s.float(1.0), s.float(-1.0))
    var sy = s.select_f(s.fge(oy, s.float(0.0)), s.float(1.0), s.float(-1.0))
    var fx = s.select_f(neg, s.fmul(s.fsub(s.float(1.0), s.fabs(oy)), sx), ox)
    var fy = s.select_f(neg, s.fmul(s.fsub(s.float(1.0), s.fabs(ox)), sy), oy)
    return s.pack_f16x2(s.fadd(fx, s.float(2.0)), s.fadd(fy, s.float(2.0)))


def emit_sota_oct_unpack(mut s: AaaSpirvModule, w: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
    var uv = s.unpack_f16x2(w)
    var fx = s.fsub(uv[0], s.float(2.0))
    var fy = s.fsub(uv[1], s.float(2.0))
    var fz = s.fsub(s.fsub(s.float(1.0), s.fabs(fx)), s.fabs(fy))
    var t = s.nmax(s.fneg(fz), s.float(0.0))
    var ax = s.fsub(fx, s.select_f(s.fge(fx, s.float(0.0)), t, s.fneg(t)))
    var ay = s.fsub(fy, s.select_f(s.fge(fy, s.float(0.0)), t, s.fneg(t)))
    return emit_vec3_normalize(s, ax, ay, fz)


def _lum(mut s: AaaSpirvModule, r: UInt32, g: UInt32, b: UInt32) -> UInt32:
    return s.fma(r, s.float(0.2126), s.fma(g, s.float(0.7152), s.fmul(b, s.float(0.0722))))


def emit_sota_reproject(
    mut s: AaaSpirvModule, px: UInt32, py: UInt32, w_u: UInt32, h_u: UInt32, vel_u: UInt32, vel_v: UInt32
) -> Tuple[UInt32, UInt32, UInt32]:
    """Previous-frame pixel from the deferred shade's motion vector (`vel = uv_cur - uv_prev`).
    Returns `(on_screen_bool, prev_px, prev_py)`."""
    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var pu = s.fsub(s.fdiv(s.fadd(s.float_from_uint(px), s.float(0.5)), fw), vel_u)
    var pv = s.fsub(s.fdiv(s.fadd(s.float_from_uint(py), s.float(0.5)), fh), vel_v)
    var ok = s.logical_and(
        s.logical_and(s.fge(pu, s.float(0.0)), s.flt(pu, s.float(1.0))),
        s.logical_and(s.fge(pv, s.float(0.0)), s.flt(pv, s.float(1.0))),
    )
    var qx = s.uint_from_float(s.nclamp(s.fmul(pu, fw), s.float(0.0), s.fsub(fw, s.float(1.0))))
    var qy = s.uint_from_float(s.nclamp(s.fmul(pv, fh), s.float(0.0), s.fsub(fh, s.float(1.0))))
    return (ok, qx, qy)


def emit_sota_hist_similar(
    mut s: AaaSpirvModule,
    b_sc: UInt32,
    q_pix: UInt32,
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    min_ndot: Float32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """Is the previous-frame surface at pixel `q_pix` on the same surface as `(P, N)`? Plane distance
    relative to the camera distance plus a normal test. Returns `(similar_bool, prev_x, prev_y, prev_z)`."""
    var hb = s.iadd(_ctrl(s, b_sc, SOTA_CTRL_HIST_PREV), s.imul(q_pix, s.uint(UInt32(SOTA_GI_HIST_WORDS))))
    var hx = s.load_float(b_sc, hb)
    var hy = s.load_float(b_sc, s.iadd(hb, s.uint(1)))
    var hz = s.load_float(b_sc, s.iadd(hb, s.uint(2)))
    var hw = s.load_word(b_sc, s.iadd(hb, s.uint(3)))
    var hn = emit_sota_oct_unpack(s, hw)
    var cdx = s.fsub(wx, _ctrl_f(s, b_sc, SOTA_CTRL_CAM_X))
    var cdy = s.fsub(wy, _ctrl_f(s, b_sc, SOTA_CTRL_CAM_X + 1))
    var cdz = s.fsub(wz, _ctrl_f(s, b_sc, SOTA_CTRL_CAM_X + 2))
    var cam_d = s.fsqrt(s.nmax(s.float(1e-8), emit_vec3_dot(s, cdx, cdy, cdz, cdx, cdy, cdz)))
    var plane = s.fabs(emit_vec3_dot(s, nx, ny, nz, s.fsub(hx, wx), s.fsub(hy, wy), s.fsub(hz, wz)))
    var ndot = emit_vec3_dot(s, nx, ny, nz, hn[0], hn[1], hn[2])
    var ok = s.logical_and(
        s.ine(hw, s.uint(0)),
        s.logical_and(s.flt(plane, s.fmul(cam_d, s.float(0.015))), s.fgt(ndot, s.float(min_ndot))),
    )
    return (ok, hx, hy, hz)


def emit_sota_hist_write(
    mut s: AaaSpirvModule, b_sc: UInt32, pix: UInt32, wx: UInt32, wy: UInt32, wz: UInt32, nx: UInt32, ny: UInt32, nz: UInt32
):
    """Writes this frame's surface history entry (called from the deferred-shade G-buffer hook)."""
    var flags = _ctrl(s, b_sc, SOTA_CTRL_RESTIR_FLAGS)
    s.begin_if(s.ine(s.bit_and(flags, s.uint(SOTA_RESTIR_HIST_WRITE)), s.uint(0)))
    var hb = s.iadd(_ctrl(s, b_sc, SOTA_CTRL_HIST_CUR), s.imul(pix, s.uint(UInt32(SOTA_GI_HIST_WORDS))))
    s.store_float(b_sc, hb, wx)
    s.store_float(b_sc, s.iadd(hb, s.uint(1)), wy)
    s.store_float(b_sc, s.iadd(hb, s.uint(2)), wz)
    s.store_word(b_sc, s.iadd(hb, s.uint(3)), emit_sota_oct_pack(s, nx, ny, nz))
    s.end_if()


def _emit_disk_offset(
    mut s: AaaSpirvModule, cx: UInt32, cy: UInt32, w_u: UInt32, h_u: UInt32, radius: UInt32, u1: UInt32, u2: UInt32
) -> UInt32:
    """Neighbour pixel index in a disk of `radius` pixels around `(cx, cy)`, clamped to the screen."""
    var r = s.fmul(radius, s.fsqrt(u1))
    var phi = s.fmul(u2, s.float(6.283185307))
    var fx = s.fadd(s.fadd(s.float_from_uint(cx), s.float(0.5)), s.fmul(r, s.fcos(phi)))
    var fy = s.fadd(s.fadd(s.float_from_uint(cy), s.float(0.5)), s.fmul(r, s.fsin(phi)))
    var qx = s.uint_from_float(s.nclamp(fx, s.float(0.0), s.fsub(s.float_from_uint(w_u), s.float(1.0))))
    var qy = s.uint_from_float(s.nclamp(fy, s.float(0.0), s.fsub(s.float_from_uint(h_u), s.float(1.0))))
    return s.iadd(s.imul(qy, w_u), qx)


# ---------------------------------------------------------------------------------------------
# ReSTIR DI
# ---------------------------------------------------------------------------------------------
def emit_sota_light_phat(
    mut s: AaaSpirvModule, b_clusters: UInt32, light_id: UInt32, wx: UInt32, wy: UInt32, wz: UInt32, nx: UInt32, ny: UInt32, nz: UInt32
) -> UInt32:
    """Unshadowed target `luma * windowed 1/d^2 * N.L` (identical to the v5 candidate pass); 0 for an
    invalid or out-of-range light id."""
    var p = s.fvar(s.float(0.0))
    var total = s.load_word(b_clusters, s.uint(0))
    s.begin_if(s.ult(light_id, total))
    var l_off = s.iadd(s.uint(UInt32(AAA_CLUSTER_LIGHTS_BASE)), s.imul(light_id, s.uint(UInt32(AAA_CLUSTER_LIGHT_WORDS))))
    var tx = s.fsub(s.load_float(b_clusters, l_off), wx)
    var ty = s.fsub(s.load_float(b_clusters, s.iadd(l_off, s.uint(1))), wy)
    var tz = s.fsub(s.load_float(b_clusters, s.iadd(l_off, s.uint(2))), wz)
    var lrad = s.load_float(b_clusters, s.iadd(l_off, s.uint(3)))
    var luma = _lum(
        s,
        s.load_float(b_clusters, s.iadd(l_off, s.uint(4))),
        s.load_float(b_clusters, s.iadd(l_off, s.uint(5))),
        s.load_float(b_clusters, s.iadd(l_off, s.uint(6))),
    )
    var d2 = s.fma(tx, tx, s.fma(ty, ty, s.fmul(tz, tz)))
    var inv_d = s.inverse_sqrt(s.nmax(s.float(1e-6), d2))
    var ndl = s.nmax(s.float(0.0), s.fmul(s.fma(nx, tx, s.fma(ny, ty, s.fmul(nz, tz))), inv_d))
    var win = s.nclamp(s.fsub(s.float(1.0), s.fdiv(d2, s.fmul(lrad, lrad))), s.float(0.0), s.float(1.0))
    var att = s.fdiv(s.fmul(win, win), s.fadd(d2, s.float(0.25)))
    s.store(p, s.fmul(s.fmul(luma, att), ndl))
    s.end_if()
    return s.load_f(p)


def _emit_di_merge(
    mut s: AaaSpirvModule,
    b_sc: UInt32,
    b_clusters: UInt32,
    r_off: UInt32,
    m_cap: UInt32,
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    rnd: UInt32,
    y_v: UInt32,
    p_v: UInt32,
    wsum_v: UInt32,
    m_v: UInt32,
    count_empty: Bool = True,
):
    """Streams the stored reservoir at word `r_off` into the running reservoir `(y, p_hat(y), w_sum, M)`.

    `count_empty = False` (spatial neighbours): a neighbour whose sample contributes nothing here
    (`W = 0` after visibility reuse, or `p_hat = 0` at this pixel) does not add its M. Counting it
    under the 1/M weights darkened lit pixels next to shadows by ~8% (same fix as ReSTIR GI)."""
    var id_r = s.load_word(b_sc, r_off)
    var w_r = s.load_float(b_sc, s.iadd(r_off, s.uint(1)))
    var m_r = s.nmin(s.load_float(b_sc, s.iadd(r_off, s.uint(2))), s.float_from_uint(m_cap))
    var p_r = emit_sota_light_phat(s, b_clusters, id_r, wx, wy, wz, nx, ny, nz)
    var w = s.fmul(s.fmul(p_r, s.nmax(w_r, s.float(0.0))), m_r)
    var ws = s.fadd(s.load_f(wsum_v), w)
    s.store(wsum_v, ws)
    var m_add = s.uint_from_float(s.fadd(m_r, s.float(0.5)))
    if not count_empty:
        m_add = s.select_u(s.fgt(w, s.float(0.0)), m_add, s.uint(0))
    s.store(m_v, s.iadd(s.load_u(m_v), m_add))
    s.begin_if(s.logical_and(s.fgt(w, s.float(0.0)), s.flt(s.fmul(rnd, ws), w)))
    s.store(y_v, id_r)
    s.store(p_v, p_r)
    s.end_if()


def emit_sota_restir_di_reuse(
    mut s: AaaSpirvModule,
    b_sc: UInt32,
    b_clusters: UInt32,
    cand: Tuple[UInt32, UInt32, UInt32, UInt32],
    px: UInt32,
    py: UInt32,
    w_u: UInt32,
    h_u: UInt32,
    vel_u: UInt32,
    vel_v: UInt32,
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """Deferred-shade hook after `emit_restir_candidate_selection_spv`. Takes and returns the same
    `(light_id, w_sum, M, p_hat(light))` tuple, so the v5 shading code (`W = w_sum / (M p_hat)`) is
    unchanged; with reuse off the candidate reservoir passes through untouched."""
    var flags = _ctrl(s, b_sc, SOTA_CTRL_RESTIR_FLAGS)
    var on = s.ine(s.bit_and(flags, s.uint(SOTA_RESTIR_DI)), s.uint(0))
    var y_v = s.uvar(cand[0])
    var wsum_v = s.fvar(cand[1])
    var m_v = s.uvar(cand[2])
    var p_v = s.fvar(cand[3])
    s.begin_if(on)
    var pix = s.iadd(s.imul(py, w_u), px)
    var cur_ok = s.logical_and(s.ine(cand[0], s.uint(SOTA_RESTIR_NONE)), s.fgt(cand[3], s.float(1e-8)))
    s.store(wsum_v, s.select_f(cur_ok, cand[1], s.float(0.0)))
    s.store(p_v, s.select_f(cur_ok, cand[3], s.float(0.0)))
    var frame = _ctrl(s, b_sc, SOTA_CTRL_FRAME)
    var seed_v = s.uvar(s.iadd(s.imul(pix, s.uint(2654435761)), s.imul(frame, s.uint(40503))))
    var m_cap = s.uint_from_float(_ctrl_f(s, b_sc, SOTA_CTRL_DI_MCAP))
    var rp = emit_sota_reproject(s, px, py, w_u, h_u, vel_u, vel_v)
    var hist_ok = s.ine(s.bit_and(flags, s.uint(SOTA_RESTIR_HIST_VALID)), s.uint(0))
    var prev_base = _ctrl(s, b_sc, SOTA_CTRL_DI_PREV)
    s.begin_if(s.logical_and(hist_ok, rp[0]))
    var q_pix = s.iadd(s.imul(rp[2], w_u), rp[1])
    # Temporal
    var sim = emit_sota_hist_similar(s, b_sc, q_pix, wx, wy, wz, nx, ny, nz, Float32(0.9))
    s.begin_if(sim[0])
    var h0 = _emit_hash(s, s.load_u(seed_v))
    s.store(seed_v, h0[0])
    _emit_di_merge(
        s, b_sc, b_clusters, s.iadd(prev_base, s.imul(q_pix, s.uint(UInt32(SOTA_GI_DI_WORDS)))), m_cap,
        wx, wy, wz, nx, ny, nz, h0[1], y_v, p_v, wsum_v, m_v,
    )
    s.end_if()
    s.end_if()

    # Store the temporally resampled reservoir (W before visibility; the visibility hook may zero
    # it). Spatial results only feed this frame's shading, avoiding sample-copying feedback.
    var p_t = s.load_f(p_v)
    var m_t = s.load_u(m_v)
    var w_t = s.select_f(
        s.logical_and(s.fgt(p_t, s.float(1e-8)), s.ine(m_t, s.uint(0))),
        s.fdiv(s.load_f(wsum_v), s.fmul(s.float_from_uint(m_t), p_t)),
        s.float(0.0),
    )
    var r_cur = s.iadd(_ctrl(s, b_sc, SOTA_CTRL_DI_CUR), s.imul(pix, s.uint(UInt32(SOTA_GI_DI_WORDS))))
    s.store_word(b_sc, r_cur, s.select_u(s.fgt(p_t, s.float(1e-8)), s.load_u(y_v), s.uint(SOTA_RESTIR_NONE)))
    s.store_float(b_sc, s.iadd(r_cur, s.uint(1)), w_t)
    s.store_float(b_sc, s.iadd(r_cur, s.uint(2)), s.float_from_uint(m_t))

    # Fused spatial: neighbours of the reprojected pixel in the previous frame
    s.begin_if(s.logical_and(hist_ok, rp[0]))
    var n_sp = s.umin(_ctrl(s, b_sc, SOTA_CTRL_DI_SPATIAL), s.uint(UInt32(SOTA_RESTIR_MAX_SPATIAL)))
    var radius = _ctrl_f(s, b_sc, SOTA_CTRL_DI_RADIUS)
    var k_v = s.uvar(s.uint(0))
    s.begin_loop()
    var k = s.load_u(k_v)
    s.break_if(s.uge(k, n_sp))
    var ha = _emit_hash(s, s.load_u(seed_v))
    var hb = _emit_hash(s, ha[0])
    var hc = _emit_hash(s, hb[0])
    s.store(seed_v, hc[0])
    var nb = _emit_disk_offset(s, rp[1], rp[2], w_u, h_u, radius, ha[1], hb[1])
    var sim_n = emit_sota_hist_similar(s, b_sc, nb, wx, wy, wz, nx, ny, nz, Float32(0.9))
    s.begin_if(sim_n[0])
    _emit_di_merge(
        s, b_sc, b_clusters, s.iadd(prev_base, s.imul(nb, s.uint(UInt32(SOTA_GI_DI_WORDS)))), m_cap,
        wx, wy, wz, nx, ny, nz, hc[1], y_v, p_v, wsum_v, m_v, count_empty=False,
    )
    s.end_if()
    s.store(k_v, s.iadd(k, s.uint(1)))
    s.end_loop()
    s.end_if()

    var p_y = s.load_f(p_v)
    s.store(y_v, s.select_u(s.fgt(p_y, s.float(1e-8)), s.load_u(y_v), s.uint(SOTA_RESTIR_NONE)))
    s.end_if()
    return (s.load_u(y_v), s.load_f(wsum_v), s.load_u(m_v), s.load_f(p_v))


def emit_sota_restir_di_visibility(mut s: AaaSpirvModule, b_sc: UInt32, pix: UInt32, light_id: UInt32, vis: UInt32):
    """Deferred-shade hook after the ReSTIR shadow ray: when the shaded light is the one the stored
    (temporal) reservoir holds and it is occluded, the stored reservoir gets `W = 0` (visibility reuse)."""
    var flags = _ctrl(s, b_sc, SOTA_CTRL_RESTIR_FLAGS)
    s.begin_if(s.logical_and(s.ine(s.bit_and(flags, s.uint(SOTA_RESTIR_DI)), s.uint(0)), s.fle(vis, s.float(0.0))))
    var r_cur = s.iadd(_ctrl(s, b_sc, SOTA_CTRL_DI_CUR), s.imul(pix, s.uint(UInt32(SOTA_GI_DI_WORDS))))
    s.begin_if(s.ieq(s.load_word(b_sc, r_cur), light_id))
    s.store_float(b_sc, s.iadd(r_cur, s.uint(1)), s.float(0.0))
    s.end_if()
    s.end_if()


# ---------------------------------------------------------------------------------------------
# ReSTIR GI (used inside `sota_gi_resolve_shader`)
# ---------------------------------------------------------------------------------------------
struct SotaGiReservoir(Copyable, Movable):
    """SPIR-V variable ids of a running ReSTIR GI reservoir."""
    var sx: UInt32
    var sy: UInt32
    var sz: UInt32
    var snx: UInt32
    var sny: UInt32
    var snz: UInt32
    var lr: UInt32
    var lg: UInt32
    var lb: UInt32
    var p: UInt32  # target p_hat(y) at the current pixel
    var wsum: UInt32
    var m: UInt32  # float

    def __init__(out self, mut s: AaaSpirvModule):
        self.sx = s.fvar(s.float(0.0))
        self.sy = s.fvar(s.float(0.0))
        self.sz = s.fvar(s.float(0.0))
        self.snx = s.fvar(s.float(0.0))
        self.sny = s.fvar(s.float(1.0))
        self.snz = s.fvar(s.float(0.0))
        self.lr = s.fvar(s.float(0.0))
        self.lg = s.fvar(s.float(0.0))
        self.lb = s.fvar(s.float(0.0))
        self.p = s.fvar(s.float(0.0))
        self.wsum = s.fvar(s.float(0.0))
        self.m = s.fvar(s.float(0.0))


def emit_sota_gi_phat(
    mut s: AaaSpirvModule,
    vx: UInt32,
    vy: UInt32,
    vz: UInt32,
    vnx: UInt32,
    vny: UInt32,
    vnz: UInt32,
    sx: UInt32,
    sy: UInt32,
    sz: UInt32,
    lr: UInt32,
    lg: UInt32,
    lb: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Target `luma(L) * cos` of sample point `x_s` seen from visible point `x_v`.
    Returns `(p_hat, cos_v, dir_x, dir_y, dir_z, dist)`."""
    var dx = s.fsub(sx, vx)
    var dy = s.fsub(sy, vy)
    var dz = s.fsub(sz, vz)
    var dist = s.fsqrt(s.nmax(s.float(1e-10), emit_vec3_dot(s, dx, dy, dz, dx, dy, dz)))
    var inv = s.fdiv(s.float(1.0), dist)
    var ux = s.fmul(dx, inv)
    var uy = s.fmul(dy, inv)
    var uz = s.fmul(dz, inv)
    var cv = s.nmax(s.float(0.0), emit_vec3_dot(s, vnx, vny, vnz, ux, uy, uz))
    return (s.fmul(_lum(s, lr, lg, lb), cv), cv, ux, uy, uz, dist)


def emit_sota_gi_stream_candidate(
    mut s: AaaSpirvModule,
    res: SotaGiReservoir,
    rnd: UInt32,
    vx: UInt32,
    vy: UInt32,
    vz: UInt32,
    vnx: UInt32,
    vny: UInt32,
    vnz: UInt32,
    sx: UInt32,
    sy: UInt32,
    sz: UInt32,
    snx: UInt32,
    sny: UInt32,
    snz: UInt32,
    lr: UInt32,
    lg: UInt32,
    lb: UInt32,
):
    """Initial RIS: a cosine-sampled path is a candidate with source pdf `cos / pi`."""
    var ph = emit_sota_gi_phat(s, vx, vy, vz, vnx, vny, vnz, sx, sy, sz, lr, lg, lb)
    var w = s.select_f(s.fgt(ph[1], s.float(1e-4)), s.fdiv(s.fmul(ph[0], s.float(3.14159265)), ph[1]), s.float(0.0))
    _emit_gi_take(s, res, w, ph[0], rnd, sx, sy, sz, snx, sny, snz, lr, lg, lb, s.float(1.0))


def _emit_gi_take(
    mut s: AaaSpirvModule,
    res: SotaGiReservoir,
    w: UInt32,
    p: UInt32,
    rnd: UInt32,
    sx: UInt32,
    sy: UInt32,
    sz: UInt32,
    snx: UInt32,
    sny: UInt32,
    snz: UInt32,
    lr: UInt32,
    lg: UInt32,
    lb: UInt32,
    m: UInt32,
):
    var ws = s.fadd(s.load_f(res.wsum), w)
    s.store(res.wsum, ws)
    s.store(res.m, s.fadd(s.load_f(res.m), m))
    s.begin_if(s.logical_and(s.fgt(w, s.float(0.0)), s.flt(s.fmul(rnd, ws), w)))
    s.store(res.sx, sx)
    s.store(res.sy, sy)
    s.store(res.sz, sz)
    s.store(res.snx, snx)
    s.store(res.sny, sny)
    s.store(res.snz, snz)
    s.store(res.lr, lr)
    s.store(res.lg, lg)
    s.store(res.lb, lb)
    s.store(res.p, p)
    s.end_if()


def _emit_gi_merge_stored(
    mut s: AaaSpirvModule,
    b_sc: UInt32,
    res: SotaGiReservoir,
    r_off: UInt32,
    m_cap: UInt32,
    rnd: UInt32,
    vx: UInt32,
    vy: UInt32,
    vz: UInt32,
    vnx: UInt32,
    vny: UInt32,
    vnz: UInt32,
    ox: UInt32,
    oy: UInt32,
    oz: UInt32,
    trace_vis: Bool,
    tlas_var: UInt32,
    b_verts: UInt32,
    b_indices: UInt32,
    b_instances: UInt32,
    b_materials: UInt32,
    bindless_arr: Tuple[UInt32, UInt32],
    gnx: UInt32,
    gny: UInt32,
    gnz: UInt32,
):
    """Merges the stored reservoir at `r_off` (created at visible point `o`) into `res` at visible
    point `v`, with the solid-angle Jacobian of the shift and an optional visibility ray."""
    var sx = s.load_float(b_sc, r_off)
    var sy = s.load_float(b_sc, s.iadd(r_off, s.uint(1)))
    var sz = s.load_float(b_sc, s.iadd(r_off, s.uint(2)))
    var sn = emit_sota_oct_unpack(s, s.load_word(b_sc, s.iadd(r_off, s.uint(3))))
    var lrg = s.unpack_f16x2(s.load_word(b_sc, s.iadd(r_off, s.uint(4))))
    var lbm = s.unpack_f16x2(s.load_word(b_sc, s.iadd(r_off, s.uint(5))))
    var w_r = s.nmax(s.load_float(b_sc, s.iadd(r_off, s.uint(6))), s.float(0.0))
    var m_r = s.nmin(lbm[1], m_cap)
    var ph = emit_sota_gi_phat(s, vx, vy, vz, vnx, vny, vnz, sx, sy, sz, lrg[0], lrg[1], lbm[0])
    # Jacobian |cos phi_v| / |cos phi_o| * |x_o - x_s|^2 / |x_v - x_s|^2 (sample surface angles)
    var ox_s = s.fsub(ox, sx)
    var oy_s = s.fsub(oy, sy)
    var oz_s = s.fsub(oz, sz)
    var d_o2 = s.nmax(s.float(1e-10), emit_vec3_dot(s, ox_s, oy_s, oz_s, ox_s, oy_s, oz_s))
    var cos_o = s.fdiv(s.fabs(emit_vec3_dot(s, sn[0], sn[1], sn[2], ox_s, oy_s, oz_s)), s.fsqrt(d_o2))
    var cos_v = s.fabs(emit_vec3_dot(s, sn[0], sn[1], sn[2], s.fneg(ph[2]), s.fneg(ph[3]), s.fneg(ph[4])))
    var d_v2 = s.fmul(ph[5], ph[5])
    var jac = s.fdiv(s.fmul(cos_v, d_o2), s.nmax(s.float(1e-10), s.fmul(cos_o, d_v2)))
    var jac_ok = s.logical_and(s.fgt(jac, s.float(0.1)), s.flt(jac, s.float(10.0)))
    var p_eff = s.fvar(s.select_f(jac_ok, ph[0], s.float(0.0)))
    if trace_vis:
        s.begin_if(s.fgt(s.load_f(p_eff), s.float(0.0)))
        var vis = emit_rt_shadow_query(
            s, tlas_var, vx, vy, vz, gnx, gny, gnz, ph[2], ph[3], ph[4],
            s.nmax(s.float(0.001), s.fsub(s.fmul(ph[5], s.float(0.995)), s.float(0.01))),
            b_verts=b_verts, b_indices=b_indices, b_instances=b_instances, b_materials=b_materials, bindless_arr=bindless_arr,
        )
        s.store(p_eff, s.fmul(s.load_f(p_eff), vis))
        s.end_if()
    var w = s.fmul(s.fmul(s.load_f(p_eff), w_r), s.fmul(m_r, s.select_f(jac_ok, jac, s.float(0.0))))
    # A neighbour whose sample cannot reach this pixel (occluded, back-facing, extreme Jacobian)
    # does not count towards M: with the 1/M weights it would darken the result (biased ReSTIR's
    # well-known energy loss); a dark but valid sample still counts.
    var dark = s.fle(_lum(s, lrg[0], lrg[1], lbm[0]), s.float(0.0))
    var m_eff = s.select_f(s.logical_or(s.fgt(s.load_f(p_eff), s.float(0.0)), dark), m_r, s.float(0.0))
    _emit_gi_take(s, res, w, s.load_f(p_eff), rnd, sx, sy, sz, sn[0], sn[1], sn[2], lrg[0], lrg[1], lbm[0], m_eff)


def emit_sota_restir_gi_reuse(
    mut s: AaaSpirvModule,
    b_sc: UInt32,
    b_hdr: UInt32,
    res: SotaGiReservoir,
    px: UInt32,
    py: UInt32,
    w_u: UInt32,
    h_u: UInt32,
    vx: UInt32,
    vy: UInt32,
    vz: UInt32,
    vnx: UInt32,
    vny: UInt32,
    vnz: UInt32,
    gnx: UInt32,
    gny: UInt32,
    gnz: UInt32,
    tlas_var: UInt32,
    b_verts: UInt32,
    b_indices: UInt32,
    b_instances: UInt32,
    b_materials: UInt32,
    bindless_arr: Tuple[UInt32, UInt32],
) -> Tuple[UInt32, UInt32, UInt32]:
    """Temporal reuse (stored for the next frame), then fused spatial reuse of previous-frame
    neighbours for this frame's estimate. Returns the GI estimate `f(y) W` in E/pi units."""
    var pix = s.iadd(s.imul(py, w_u), px)
    var flags = _ctrl(s, b_sc, SOTA_CTRL_RESTIR_FLAGS)
    var frame = _ctrl(s, b_sc, SOTA_CTRL_FRAME)
    var seed_v = s.uvar(s.iadd(s.imul(pix, s.uint(2246822519)), s.imul(frame, s.uint(3266489917))))
    var ob = s.imul(pix, s.uint(8))
    var vel_u = s.load_float(b_hdr, s.iadd(ob, s.uint(4)))
    var vel_v = s.load_float(b_hdr, s.iadd(ob, s.uint(5)))
    var rp = emit_sota_reproject(s, px, py, w_u, h_u, vel_u, vel_v)
    var hist_ok = s.ine(s.bit_and(flags, s.uint(SOTA_RESTIR_HIST_VALID)), s.uint(0))
    var m_cap = _ctrl_f(s, b_sc, SOTA_CTRL_GI_MCAP)
    var prev_base = _ctrl(s, b_sc, SOTA_CTRL_GI_PREV)
    s.begin_if(s.logical_and(hist_ok, rp[0]))
    var q_pix = s.iadd(s.imul(rp[2], w_u), rp[1])
    var sim = emit_sota_hist_similar(s, b_sc, q_pix, vx, vy, vz, vnx, vny, vnz, Float32(0.9))
    s.begin_if(sim[0])
    var h0 = _emit_hash(s, s.load_u(seed_v))
    s.store(seed_v, h0[0])
    _emit_gi_merge_stored(
        s, b_sc, res, s.iadd(prev_base, s.imul(q_pix, s.uint(UInt32(SOTA_GI_GI_WORDS)))), m_cap, h0[1],
        vx, vy, vz, vnx, vny, vnz, sim[1], sim[2], sim[3],
        False, tlas_var, b_verts, b_indices, b_instances, b_materials, bindless_arr, gnx, gny, gnz,
    )
    s.end_if()
    s.end_if()

    # Store the temporally resampled reservoir: next frame reuses it temporally and as spatial
    # neighbour. Spatial results only feed this frame's estimate, which avoids the sample-copying
    # feedback (blotches) of storing post-spatial reservoirs.
    var p_t = s.load_f(res.p)
    var m_t = s.load_f(res.m)
    var w_t = s.select_f(
        s.logical_and(s.fgt(p_t, s.float(1e-8)), s.fgt(m_t, s.float(0.0))),
        s.fdiv(s.load_f(res.wsum), s.fmul(m_t, p_t)),
        s.float(0.0),
    )
    var r_cur = s.iadd(_ctrl(s, b_sc, SOTA_CTRL_GI_CUR), s.imul(pix, s.uint(UInt32(SOTA_GI_GI_WORDS))))
    s.store_float(b_sc, r_cur, s.load_f(res.sx))
    s.store_float(b_sc, s.iadd(r_cur, s.uint(1)), s.load_f(res.sy))
    s.store_float(b_sc, s.iadd(r_cur, s.uint(2)), s.load_f(res.sz))
    s.store_word(b_sc, s.iadd(r_cur, s.uint(3)), emit_sota_oct_pack(s, s.load_f(res.snx), s.load_f(res.sny), s.load_f(res.snz)))
    s.store_word(b_sc, s.iadd(r_cur, s.uint(4)), s.pack_f16x2(s.load_f(res.lr), s.load_f(res.lg)))
    s.store_word(b_sc, s.iadd(r_cur, s.uint(5)), s.pack_f16x2(s.load_f(res.lb), m_t))
    s.store_float(b_sc, s.iadd(r_cur, s.uint(6)), w_t)

    s.begin_if(s.logical_and(hist_ok, rp[0]))
    var n_sp = s.umin(_ctrl(s, b_sc, SOTA_CTRL_GI_SPATIAL), s.uint(UInt32(SOTA_RESTIR_MAX_SPATIAL)))
    var radius = _ctrl_f(s, b_sc, SOTA_CTRL_GI_RADIUS)
    var k_v = s.uvar(s.uint(0))
    s.begin_loop()
    var k = s.load_u(k_v)
    s.break_if(s.uge(k, n_sp))
    var ha = _emit_hash(s, s.load_u(seed_v))
    var hb = _emit_hash(s, ha[0])
    var hc = _emit_hash(s, hb[0])
    s.store(seed_v, hc[0])
    var nb = _emit_disk_offset(s, rp[1], rp[2], w_u, h_u, radius, ha[1], hb[1])
    var sim_n = emit_sota_hist_similar(s, b_sc, nb, vx, vy, vz, vnx, vny, vnz, Float32(0.8))
    s.begin_if(sim_n[0])
    _emit_gi_merge_stored(
        s, b_sc, res, s.iadd(prev_base, s.imul(nb, s.uint(UInt32(SOTA_GI_GI_WORDS)))), m_cap, hc[1],
        vx, vy, vz, vnx, vny, vnz, sim_n[1], sim_n[2], sim_n[3],
        True, tlas_var, b_verts, b_indices, b_instances, b_materials, bindless_arr, gnx, gny, gnz,
    )
    s.end_if()
    s.store(k_v, s.iadd(k, s.uint(1)))
    s.end_loop()
    s.end_if()

    var p_y = s.load_f(res.p)
    var m_tot = s.load_f(res.m)
    var w_out = s.select_f(
        s.logical_and(s.fgt(p_y, s.float(1e-8)), s.fgt(m_tot, s.float(0.0))),
        s.fdiv(s.load_f(res.wsum), s.fmul(m_tot, p_y)),
        s.float(0.0),
    )
    # f(y) W with f = L cos / pi
    var ph = emit_sota_gi_phat(s, vx, vy, vz, vnx, vny, vnz, s.load_f(res.sx), s.load_f(res.sy), s.load_f(res.sz), s.load_f(res.lr), s.load_f(res.lg), s.load_f(res.lb))
    var scale = s.fmul(w_out, s.fmul(ph[1], s.float(0.318309886)))
    return (s.fmul(s.load_f(res.lr), scale), s.fmul(s.load_f(res.lg), scale), s.fmul(s.load_f(res.lb), scale))
