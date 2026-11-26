"""Area E (`sota:E`) verification: characters (skin / eye / hair) and the production pipeline cache.

Zero external assets. Verifies:
  1. SPIR-V validation of the deferred shade pass with the character hooks (ray-query on and off).
  2. CPU references: closed-form pre-integrated Burley vs a converged 4096-tap quadrature, the flat-surface
     Lambert limit, ring energy conservation, red-over-blue terminator bleed; exact eye-frame recovery on a
     sphere through the engine's tangent convention; cornea refraction parallax; hair lobe placement and
     dual-scattering amplitude bounds; strand / card geometry and LOD pick.
  3. Live GPU: Burley skin vs standard vs legacy wrap at the terminator; eye iris / pupil / sclera and
     refraction parallax; strand hair highlights; deferred-pass cost budget with characters on vs off.
  4. Pipeline cache: header validation, SPIR-V keyed manifest, cold vs warm renderer start-up.
Set `FLINT_SOTA_E_PNG_DIR=<dir>` to write the before/after PNGs.
"""
from std.collections import List
from std.ffi import external_call
from std.memory import bitcast
from std.math import abs, cos, exp, max, min, sin, sqrt
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat, trs
from flint.platform.os import env_value, get_time, make_dirs
from flint.platform.vulkan import get32, get_float, put_float, ACCESS_HOST_READ, ACCESS_SHADER_WRITE, PIPELINE_COMPUTE, PIPELINE_HOST
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_npr_cel import AaaCelParams
from flint.render.aaa_gi_svgf import aaa_deferred_shade_shader
from flint.render.aaa_ibl import SkyDomeParams, bake_bc6h_ibl_cubemap_and_sh9
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.spirv_validate import validate_spirv_module
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_POST_SCRATCH
from flint.render.sota_e_characters import (
    SotaECharDecode,
    emit_sota_e_hair_direct,
    emit_sota_e_skin_direct,
    SOTA_E_MODEL_EYE,
    SOTA_E_MODEL_HAIR,
    SOTA_E_MODEL_SKIN,
    SOTA_E_MODEL_STANDARD,
    SotaEyeParams,
    SotaHairParams,
    SotaSkinParams,
    sota_e_burley_d,
    sota_e_burley_preintegrated_fast,
    sota_e_burley_preintegrated_ref,
    sota_e_eye_direct_cpu,
    sota_e_eye_frame_cpu,
    sota_e_eye_material,
    sota_e_eye_surface_cpu,
    sota_e_hair_backscatter_amp,
    sota_e_hair_direct_cpu,
    sota_e_hair_material,
    sota_e_material_model,
    sota_e_skin_direct_cpu,
    sota_e_skin_material,
)
from flint.render.sota_e_pipeline_cache import (
    SOTA_E_PSO_COLD_MISSING,
    SOTA_E_PSO_COLD_REJECTED,
    SOTA_E_PSO_WARM,
    SotaEDeviceIdentity,
    sota_e_pso_cache_path,
    sota_e_pso_decode,
    sota_e_pso_encode,
)
from flint.platform.os import read_bytes, remove_file, write_bytes
from sota_perf_timing import load_aware_ratio, min_of, quantile, timing_note, timing_rounds
from flint.render.sota_e_eye_uv import sota_e_eye_front_uv
from flint.render.sota_e_hair_shadow import (
    SOTA_E_DOM_FIXED,
    SOTA_E_DOM_LAYERS,
    sota_e_dom_build_cpu,
    sota_e_dom_frame,
    sota_e_dom_transmittance_cpu,
    sota_e_strands_to_points,
)
from flint.render.sota_e_hair_strands import (
    build_hair_strand_ribbons_aa,
    sota_e_groom_strands,
    sota_e_hair_coverage_lod,
    sota_e_px_per_m,
    sota_e_strand_kept,
    SOTA_E_HAIR_LOD_CARDS,
    SOTA_E_HAIR_LOD_STRANDS,
    SotaHairGroom,
    build_hair_cards,
    build_hair_strand_ribbons,
    sota_e_hair_card_texture,
    sota_e_pick_hair_lod,
    sota_e_strand_points,
)


def _fail(msg: String):
    print("  [FAIL]", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _png_dir() -> String:
    return env_value("FLINT_SOTA_E_PNG_DIR")


def _save(name: String, mut px: List[UInt32], w: Int, h: Int):
    var d = _png_dir()
    if d.byte_length() == 0:
        return
    _ = make_dirs(d)
    if write_png(d + "/" + name, px, w, h):
        print("    wrote", d + "/" + name)


def _rgb(argb: UInt32) -> Vec3:
    return Vec3(
        Float32((argb >> 16) & 0xFF) / Float32(255.0),
        Float32((argb >> 8) & 0xFF) / Float32(255.0),
        Float32(argb & 0xFF) / Float32(255.0),
    )


def _median(mut v: List[Float64]) -> Float64:
    if len(v) == 0:
        return 0.0
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if v[j] < v[i]:
                var t = v[i]
                v[i] = v[j]
                v[j] = t
    return v[len(v) // 2]


# ---------------------------------------------------------------------------------------------
# 1. SPIR-V
# ---------------------------------------------------------------------------------------------
def test_spirv():
    for rq in range(2):
        var words = aaa_deferred_shade_shader(rq == 1)
        var rep = validate_spirv_module(words)
        if not rep.valid:
            var err = rep.errors[0] if len(rep.errors) > 0 else String("unknown")
            _fail("deferred shade (ray_query=" + String(rq) + ") SPIR-V invalid: " + err)
    print("  [PASS] Suite 1: deferred shade pass with sota:E hooks validates (RT and raster variants)")


# ---------------------------------------------------------------------------------------------
# 2. CPU references
# ---------------------------------------------------------------------------------------------
def test_cpu_skin():
    var max_err = Float32(0.0)
    var radii = [Float32(0.004), Float32(0.01), Float32(0.03), Float32(0.1), Float32(0.5)]
    var ds = [Float32(0.0005), Float32(0.002), Float32(0.006)]
    for ri in range(len(radii)):
        for di in range(len(ds)):
            for k in range(41):
                var ct = Float32(-1.0) + Float32(k) * Float32(0.05)
                var a = sota_e_burley_preintegrated_fast(ct, radii[ri], ds[di])
                var b = sota_e_burley_preintegrated_ref(ct, radii[ri], ds[di])
                max_err = max(max_err, abs(a - b))
    print("    8-harmonic closed form vs 4096-tap quadrature pre-integrated Burley max |err| =", max_err)
    _assert(max_err < Float32(0.03), "closed-form Burley ring within 0.03 of the converged quadrature")

    # Flat limit: huge radius -> Lambert max(0, cos).
    var flat_err = Float32(0.0)
    for k in range(21):
        var ct = Float32(-1.0) + Float32(k) * Float32(0.1)
        flat_err = max(flat_err, abs(sota_e_burley_preintegrated_ref(ct, Float32(50.0), Float32(0.003)) - max(Float32(0.0), ct)))
    _assert(flat_err < Float32(0.01), "Pre-integrated Burley reduces to Lambert on flat skin")

    # Ring energy: the convolution preserves the circle integral of max(0, cos) (= 2).
    var e_sum = Float32(0.0)
    comptime NT = 720
    for k in range(NT):
        var th = (Float32(k) + Float32(0.5)) / Float32(NT) * Float32(6.2831853)
        e_sum += sota_e_burley_preintegrated_ref(cos(th), Float32(0.01), Float32(0.003))
    var e_int = e_sum * Float32(6.2831853) / Float32(NT)
    print("    ring energy integral =", e_int, "(expected 2.0)")
    _assert(abs(e_int - Float32(2.0)) < Float32(0.02), "Burley ring convolution conserves energy")

    # Terminator bleed: red scatters further than blue.
    var p = SotaSkinParams.human()
    var alb = Vec3(0.78, 0.52, 0.42)
    var n = Vec3(0.0, 0.0, -1.0)
    var v = Vec3(0.0, 0.0, -1.0)
    var l = Vec3(sin(Float32(1.68)), Float32(0.0), -cos(Float32(1.68)))  # 6 deg past the terminator
    var sk = sota_e_skin_direct_cpu(n, v, l, alb, Float32(0.028), Float32(0.45), Float32(0.02), p)
    print("    skin past terminator rgb =", sk.x, sk.y, sk.z)
    _assert(sk.x > Float32(0.004) and sk.x > sk.z * Float32(2.0), "Burley skin bleeds red light past the terminator")
    _assert(sota_e_burley_d(Float32(0.004), Float32(0.8)) > sota_e_burley_d(Float32(0.004), Float32(0.2)) * Float32(0.5), "Burley d fit sane")

    var m = sota_e_skin_material(alb, Float32(0.45), p)
    _assert(sota_e_material_model(m) == SOTA_E_MODEL_SKIN, "skin material tag round-trips")
    var words = List[UInt32]()
    m.append_words(words)
    _assert(len(words) == 24, "skin material stays 24 words")
    _assert(sota_e_material_model(AaaMaterial()) == SOTA_E_MODEL_STANDARD, "untagged material stays standard")
    print("  [PASS] Suite 2a: pre-integrated Burley skin + dual-lobe specular CPU reference")


def _engine_frame_on_sphere(u: Float32, v: Float32, upr: Float32, ex: Vec3, ey: Vec3, ax: Vec3) -> Tuple[Vec3, Vec3, Vec3]:
    """N, T, B at UV (u, v) of a planar front-projected unit sphere, built the way the engine cooks them:
    `T = dP/du` Gram-Schmidt'd against N, `B = sign * N x T` with sign toward `dP/dv`."""
    var x = (u - Float32(0.5)) / upr
    var y = (v - Float32(0.5)) / upr
    var z = sqrt(Float32(1.0) - x * x - y * y)
    var n = ex * x + ey * y + ax * z
    var dpdu = ex - ax * (x / z)
    var dpdv = ey - ax * (y / z)
    var t = (dpdu - n * n.dot(dpdu)).normalized()
    var b = n.cross(t)
    if b.dot(dpdv) < Float32(0.0):
        b = b * Float32(-1.0)
    return (n, t, b)


def test_cpu_eye():
    var ax = Vec3(0.0, 0.0, -1.0)
    var exv = Vec3(1.0, 0.0, 0.0)
    var eyv = Vec3(0.0, -1.0, 0.0)
    var worst = Float32(0.0)
    for hand in range(2):
        var eyh = eyv if hand == 0 else eyv * Float32(-1.0)
        for i in range(9):
            for j in range(9):
                var u = Float32(0.18) + Float32(i) * Float32(0.08)
                var v = Float32(0.18) + Float32(j) * Float32(0.08)
                var fr3 = _engine_frame_on_sphere(u, v, Float32(0.5), exv, eyh, ax)
                var fr = sota_e_eye_frame_cpu(fr3[0], fr3[1], fr3[2], u, v, Float32(0.5))
                worst = max(worst, (fr.axis - ax).length())
                worst = max(worst, (fr.ex - exv).length())
                worst = max(worst, (fr.ey - eyh).length())
    print("    eye frame recovery worst |err| =", worst)
    _assert(worst < Float32(1e-3), "Eye optical axis recovered exactly from the engine tangent frame (both UV handedness)")

    var p = SotaEyeParams.human()
    # Frontal view at the pupil: no refraction offset.
    var f0 = _engine_frame_on_sphere(Float32(0.5), Float32(0.5), p.uv_per_radius, exv, eyv, ax)
    var fr0 = sota_e_eye_frame_cpu(f0[0], f0[1], f0[2], Float32(0.5), Float32(0.5), p.uv_per_radius)
    var s0 = sota_e_eye_surface_cpu(f0[0], Vec3(0.0, 0.0, -1.0), fr0, Float32(0.5), Float32(0.5), p)
    _assert(abs(s0.u - Float32(0.5)) < Float32(1e-4) and abs(s0.v - Float32(0.5)) < Float32(1e-4), "frontal pupil ray is not displaced")
    _assert(s0.pupil > Float32(0.99) and s0.iris_mask > Float32(0.99), "pupil centre is inside pupil and iris")
    # Oblique view from +X: the refracted ray heads toward -X under the cornea, so the iris sample moves
    # toward -U (the viewer sees deeper-lying iris shifted away from them, as in a real eye).
    var view = Vec3(sin(Float32(0.6)), Float32(0.0), -cos(Float32(0.6)))
    var s1 = sota_e_eye_surface_cpu(f0[0], view, fr0, Float32(0.5), Float32(0.5), p)
    print("    oblique (34 deg) refracted iris uv offset =", s1.u - Float32(0.5), s1.v - Float32(0.5))
    _assert(s1.u < Float32(0.5) - Float32(0.02) and abs(s1.v - Float32(0.5)) < Float32(1e-3), "cornea refraction parallax shifts the iris sample")
    var s2 = sota_e_eye_surface_cpu(f0[0], Vec3(0.0, 0.0, -1.0), fr0, Float32(0.5) + Float32(0.3), Float32(0.5), p)
    _assert(s2.iris_mask < Float32(0.01), "sclera outside the limbus")

    # Caustic: light from +X focuses on the -X side of the iris.
    var l = Vec3(sin(Float32(1.1)), Float32(0.0), -cos(Float32(1.1)))
    var alb = Vec3(0.2, 0.4, 0.7)
    var fl = _engine_frame_on_sphere(Float32(0.32), Float32(0.5), p.uv_per_radius, exv, eyv, ax)
    var fr_l = sota_e_eye_frame_cpu(fl[0], fl[1], fl[2], Float32(0.32), Float32(0.5), p.uv_per_radius)
    var sl = sota_e_eye_surface_cpu(fl[0], Vec3(0.0, 0.0, -1.0), fr_l, Float32(0.32), Float32(0.5), p)
    var fr_r3 = _engine_frame_on_sphere(Float32(0.68), Float32(0.5), p.uv_per_radius, exv, eyv, ax)
    var fr_r = sota_e_eye_frame_cpu(fr_r3[0], fr_r3[1], fr_r3[2], Float32(0.68), Float32(0.5), p.uv_per_radius)
    var sr = sota_e_eye_surface_cpu(fr_r3[0], Vec3(0.0, 0.0, -1.0), fr_r, Float32(0.68), Float32(0.5), p)
    var cl = sota_e_eye_direct_cpu(fl[0], Vec3(0.0, 0.0, -1.0), l, fr_l, sl, alb, Float32(0.025), Float32(0.05), p)
    var cr = sota_e_eye_direct_cpu(fr_r3[0], Vec3(0.0, 0.0, -1.0), l, fr_r, sr, alb, Float32(0.025), Float32(0.05), p)
    var p0 = p.copy()
    p0.caustic_strength = Float32(0.0)
    var cl0 = sota_e_eye_direct_cpu(fl[0], Vec3(0.0, 0.0, -1.0), l, fr_l, sl, alb, Float32(0.025), Float32(0.05), p0)
    var cr0 = sota_e_eye_direct_cpu(fr_r3[0], Vec3(0.0, 0.0, -1.0), l, fr_r, sr, alb, Float32(0.025), Float32(0.05), p0)
    print("    iris b far side: caustic", cl.z, "vs none", cl0.z, "| near side: caustic", cr.z, "vs none", cr0.z)
    _assert(cl.z > cl0.z * Float32(1.3), "cornea caustic brightens the far side of the iris")
    _assert(abs(cr.z - cr0.z) < Float32(1e-4), "no caustic on the light-facing side of the iris")

    var m = sota_e_eye_material(Vec3(0.2, 0.4, 0.7), Vec3(0.85, 0.82, 0.8), Float32(0.05), p)
    _assert(sota_e_material_model(m) == SOTA_E_MODEL_EYE, "eye material tag round-trips")
    print("  [PASS] Suite 2b: eye frame recovery, cornea refraction parallax, limbus/pupil, caustic CPU reference")


def test_cpu_hair():
    var p = SotaHairParams.human()
    var col = Vec3(0.32, 0.17, 0.08)
    var amp = sota_e_hair_backscatter_amp(col, p)
    print("    dual-scattering back-scatter amplitude d_b*A_b =", amp.x, amp.y, amp.z)
    _assert(amp.x > amp.z and amp.x > Float32(0.0) and amp.x < Float32(1.0), "A_b positive, bounded and colour-ordered")
    var white = sota_e_hair_backscatter_amp(Vec3(0.95, 0.95, 0.95), p)
    _assert(white.x > amp.x, "lighter hair back-scatters more")

    # R-lobe placement: with the strand along +Y and light/view in the XY plane, sweep the view
    # elevation and find the specular peak; the cuticle tilt shifts it off the mirror direction.
    var t = Vec3(0.0, 1.0, 0.0)
    var l = Vec3(0.0, sin(Float32(0.3)), -cos(Float32(0.3)))
    var best = Float32(-1.0)
    var best_k = 0
    for k in range(161):
        var th = Float32(-0.8) + Float32(k) * Float32(0.01)
        var v = Vec3(0.0, sin(th), -cos(th))
        var dark = Vec3(0.02, 0.02, 0.02)
        var r = sota_e_hair_direct_cpu(t, v, l, dark, Vec3(0.0, 0.0, 0.0), p)
        if r.x > best:
            best = r.x
            best_k = k
    var th_peak = Float32(-0.8) + Float32(best_k) * Float32(0.01)
    print("    R-lobe peak view elevation =", th_peak, "rad (mirror = -0.30, shift alpha =", p.shift_rad, ")")
    # sin(theta_i) + sin(theta_o) = alpha at the peak  ->  theta_o ~ asin(alpha - sin(0.3)).
    _assert(abs(th_peak - Float32(-0.30)) > Float32(0.03) and th_peak < Float32(-0.30), "R lobe shifted by the cuticle tilt")

    # Energy: integrate the single-scattering response over the view sphere for white-ish hair.
    var tot = Float32(0.0)
    comptime NA = 64
    comptime NB = 128
    var li = Vec3(0.0, Float32(0.2), Float32(-0.98)).normalized()
    for a in range(NA):
        var ct = Float32(-1.0) + (Float32(a) + Float32(0.5)) * Float32(2.0) / Float32(NA)
        var st = sqrt(max(Float32(0.0), Float32(1.0) - ct * ct))
        for b in range(NB):
            var ph = (Float32(b) + Float32(0.5)) / Float32(NB) * Float32(6.2831853)
            var v = Vec3(st * cos(ph), ct, st * sin(ph))
            var r = sota_e_hair_direct_cpu(t, v, li, Vec3(0.99, 0.99, 0.99), Vec3(0.0, 0.0, 0.0), p)
            tot += r.x
    var e = tot * Float32(4.0 * 3.14159265) / Float32(NA * NB)
    print("    hair single-scatter albedo (white fibre) =", e)
    _assert(e > Float32(0.2) and e < Float32(1.1), "hair single scattering bounded (energy <= ~1)")

    var m = sota_e_hair_material(col, p)
    _assert(sota_e_material_model(m) == SOTA_E_MODEL_HAIR, "hair material tag round-trips")

    var g = SotaHairGroom.short_bob(Vec3(0.0, 0.0, 0.0), Float32(0.09))
    var pts = sota_e_strand_points(g, 7)
    _assert(len(pts) == g.segments + 1, "strand polyline has segments + 1 points")
    for k in range(len(pts)):
        _assert(pts[k].length() >= g.scalp_radius * Float32(0.999), "strands stay outside the scalp")
    var mesh = build_hair_strand_ribbons(g)
    _assert(len(mesh.indices) == g.strand_count * g.segments * 12, "ribbon index count (two-sided)")
    var cards = build_hair_cards(g, 160, Float32(0.012))
    _assert(len(cards.indices) == 160 * g.segments * 12, "card index count")
    var tex = sota_e_hair_card_texture(64, 10, UInt32(3))
    var cut = 0
    for i in range(len(tex)):
        if (tex[i] >> 24) < UInt32(128):
            cut += 1
    _assert(cut > 64 * 64 // 5 and cut < 64 * 64 * 4 // 5, "card texture has strand coverage gaps")
    _assert(sota_e_pick_hair_lod(Float32(600.0), 2400, 20000) == SOTA_E_HAIR_LOD_STRANDS, "close-up picks strands")
    _assert(sota_e_pick_hair_lod(Float32(60.0), 2400, 20000) == SOTA_E_HAIR_LOD_CARDS, "distant picks cards")
    _assert(sota_e_pick_hair_lod(Float32(600.0), 40000, 20000) == SOTA_E_HAIR_LOD_CARDS, "over budget picks cards")
    print("  [PASS] Suite 2c: Marschner + dual-scattering hair, strand/card geometry & LOD CPU reference")


def _slab_strands(levels: Int, dz: Float32, dy: Float32) -> List[List[Vec3]]:
    """`levels` horizontal sheets (y = -k * dy) of straight strands along X, `dz` apart in Z."""
    var out = List[List[Vec3]]()
    for k in range(levels):
        var nz = Int(Float32(0.1) / dz) + 1
        for zi in range(nz):
            var z = Float32(-0.05) + Float32(zi) * dz + (dz * Float32(0.5) if (k % 2) == 1 else Float32(0.0))
            var s = List[Vec3]()
            for j in range(9):
                s.append(Vec3(Float32(-0.05) + Float32(j) * Float32(0.0125), -Float32(k) * dy, z))
            out.append(s^)
    return out^


def test_cpu_hair_dom():
    # 6 sheets of strands (width 1.6 mm, 4 mm apart -> 0.4 coverage per sheet), sun straight down.
    var width = Float32(0.0016)
    var dz = Float32(0.004)
    var dy = Float32(0.008)
    var pts = sota_e_strands_to_points(_slab_strands(6, dz, dy))
    var fr = sota_e_dom_frame(pts, Vec3(0.0, 1.0, 0.0), 128, width, dy * Float32(1.2), Float32(1.0))
    var m = sota_e_dom_build_cpu(fr, pts)
    var nt = 128 * 128
    var cov_sum = Float64(0.0)
    for i in range(nt * SOTA_E_DOM_LAYERS):
        cov_sum += Float64(m[1][i]) / Float64(SOTA_E_DOM_FIXED)
    var expect_cov = Float64(len(pts) // 4 - len(pts) // 36) * Float64(width) * Float64(0.0125) / Float64(fr.texel * fr.texel)
    print("    DOM total coverage =", cov_sum, " expected", expect_cov, " texel =", fr.texel * 1000.0, "mm")
    _assert(abs(cov_sum - expect_cov) < expect_cov * 0.02, "DOM accumulates the fibre coverage of every segment")
    var tv = List[Float32]()
    for k in range(6):
        tv.append(sota_e_dom_transmittance_cpu(fr, m[0], m[1], Vec3(0.003, -Float32(k) * dy, 0.001)))
    print("    DOM transmittance per sheet =", tv[0], tv[1], tv[2], tv[3], tv[4], tv[5])
    _assert(tv[0] > Float32(0.85), "top sheet is lit")
    var mono = True
    for k in range(1, 6):
        if tv[k] > tv[k - 1] + Float32(1e-4):
            mono = False
    _assert(mono, "transmittance falls monotonically into the groom")
    # sheet k sits behind ~k sheets of 0.4 coverage: T ~ exp(-0.4 k)
    var t2 = Float32(exp(Float64(-0.8)))
    _assert(abs(tv[2] - t2) < Float32(0.15), "sheet 2 transmittance ~ exp(-k * coverage)")
    _assert(tv[4] > Float32(0.05) and tv[4] < Float32(0.5), "deep strands get partial light, not an opaque shadow")
    _assert(sota_e_dom_transmittance_cpu(fr, m[0], m[1], Vec3(0.5, 0.0, 0.0)) == Float32(1.0), "outside the map is unshadowed")
    var lod = sota_e_hair_coverage_lod(Float32(0.0005), Float32(200.0), Float32(1.0))
    _assert(abs(lod[0] * lod[1] - Float32(1.0)) < Float32(1e-5) and abs(lod[0] - Float32(10.0)) < Float32(1e-3), "coverage LOD keeps width x count")
    var g = SotaHairGroom.short_bob(Vec3(0.0, 0.0, 0.0), Float32(0.09))
    var kept = 0
    var nested = True
    for i in range(g.strand_count):
        if sota_e_strand_kept(g, i, Float32(0.25)):
            kept += 1
            if not sota_e_strand_kept(g, i, Float32(0.5)):
                nested = False
    _assert(kept > g.strand_count // 5 and kept < g.strand_count * 3 // 10, "kept strand fraction follows the LOD")
    _assert(nested, "kept strand subsets are nested across distances")
    print("  [PASS] Suite 2d: hair deep opacity map + coverage strand LOD CPU reference")


# ---------------------------------------------------------------------------------------------
# 3. GPU
# ---------------------------------------------------------------------------------------------
def _setup(mut ren: FlintAaaVulkanRenderer):
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.exposure = Float32(1.0)
    ren.bloom_intensity = Float32(0.0)
    ren.enable_vsm = False
    var sky_p = SkyDomeParams()
    sky_p.zenith_rgb = Vec3(0.30, 0.34, 0.42)
    sky_p.horizon_rgb = Vec3(0.28, 0.30, 0.34)
    sky_p.ground_rgb = Vec3(0.12, 0.11, 0.10)
    sky_p.sun_lobe_rgb = Vec3(0.0, 0.0, 0.0)
    ren.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(sky_p, base_dim=32, mip_levels=5))
    ren.ibl_intensity = Float32(0.25)
    ren.sun_radiance = Vec3(3.0, 2.9, 2.75)


def _sphere(r: Float32) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(r, r, r), rings=48, segments=72)
    return mb^.finish()


def _eye_mesh(r: Float32) -> NativeMesh:
    """Eye sphere with a planar front projection: UV = 0.5 + 0.5 * (x, -y) / r, pupil toward -Z.
    Built at unit radius and scaled afterwards (`MeshBuilder.add_tri` drops triangles under ~1e-6 m^2)."""
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), rings=64, segments=96)
    var m = mb^.finish()
    for i in range(m.vertex_count()):
        m.texcoords[i * 2] = Float32(0.5) + Float32(0.5) * m.positions[i * 3]
        m.texcoords[i * 2 + 1] = Float32(0.5) - Float32(0.5) * m.positions[i * 3 + 1]
    for k in range(len(m.positions)):
        m.positions[k] = m.positions[k] * r
    return m^


def _render(mut ren: FlintAaaVulkanRenderer) -> List[UInt32]:
    _ = ren.render_frame(capture_readback=True)
    return ren.readback_pixels()



# ---------------------------------------------------------------------------------------------
# GPU emitter == CPU reference harness (compute probe on the renderer's own layout)
# ---------------------------------------------------------------------------------------------
comptime PROBE_IN_WORDS: Int = 16
comptime PROBE_CASES: Int = 256


def _probe_shader(model: Int, params: List[Float32], in_off: Int, out_off: Int) -> List[UInt32]:
    """Case i reads n, v, l, albedo (RGB), f0, rough, curvature radius from post-scratch and writes the
    model's direct response; material params are baked as constants exactly as the decode would load them."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_POST_SCRATCH)
    var i = s.invocation_axis(0)
    s.return_if(s.uge(i, s.uint(UInt32(PROBE_CASES))))
    var base = s.iadd(s.uint(UInt32(in_off)), s.imul(i, s.uint(UInt32(PROBE_IN_WORDS))))
    var v = List[UInt32]()
    for k in range(15):
        v.append(s.load_float(b, s.iadd(base, s.uint(UInt32(k)))))
    var yes = s.ieq(s.uint(1), s.uint(1))
    var no = s.ieq(s.uint(1), s.uint(0))
    var dec = SotaECharDecode(
        s.float(Float32(model)), yes, yes,
        yes if model == SOTA_E_MODEL_SKIN else no, no, yes if model == SOTA_E_MODEL_HAIR else no,
        s.float(params[0]), s.float(params[1]), s.float(params[2]), s.float(params[3]), s.float(params[4]),
        s.float(params[5]), s.float(params[6]), s.float(params[7]), s.float(params[8]),
    )
    var r: Tuple[UInt32, UInt32, UInt32]
    if model == SOTA_E_MODEL_SKIN:
        r = emit_sota_e_skin_direct(s, dec, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14])
    else:
        # hair: v[0..2] = strand tangent
        r = emit_sota_e_hair_direct(s, dec, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11])
    var ob = s.iadd(s.uint(UInt32(out_off)), s.imul(i, s.uint(4)))
    s.store_float(b, ob, r[0])
    s.store_float(b, s.iadd(ob, s.uint(1)), r[1])
    s.store_float(b, s.iadd(ob, s.uint(2)), r[2])
    return s.finish()


def _material_params(m: AaaMaterial) -> List[Float32]:
    var words = List[UInt32]()
    m.append_words(words)
    var out = List[Float32]()
    for k in range(14, 23):
        out.append(bitcast[DType.float32](words[k]))
    return out^


def _rand_dir(seed: Int, hemi: Bool) -> Vec3:
    var a = Float32((seed * 7919 + 13) % 1000) / Float32(1000.0)
    var b = Float32((seed * 104729 + 71) % 1000) / Float32(1000.0)
    var z = Float32(1.0) - Float32(2.0) * a
    if hemi:
        z = -abs(z) - Float32(0.05)
    var r = sqrt(max(Float32(0.0), Float32(1.0) - z * z))
    var ph = b * Float32(6.2831853)
    return Vec3(r * cos(ph), r * sin(ph), z).normalized()


def run_emitter_probe(mut ren: FlintAaaVulkanRenderer, model: Int, m: AaaMaterial) -> Float32:
    """Max relative error between the GPU emitter and the CPU reference over PROBE_CASES random cases."""
    ren.flush_deferred_frame()
    ren.vk.wait_idle()
    # The probe owns the hair deep-opacity-map region (nothing else writes it, and the map is inactive
    # here). It used to sit at the end of post-scratch, inside the UI-prim region of the last frame in
    # flight, which the renderer's UI pass also writes: case 0 then intermittently read back 0.
    _assert(not ren.sota_e_hair.active(), "probe region is free (hair DOM inactive)")
    var in_off = ren.sota_e_hair.base_word
    var out_off = in_off + PROBE_CASES * PROBE_IN_WORDS
    var params = _material_params(m)
    var code = _probe_shader(model, params, in_off, out_off)
    var rep = validate_spirv_module(code)
    _assert(rep.valid, "probe shader validates")
    var pipe = ren.vk.create_aaa_compute_pipeline(ren.pipe_layout, code)
    _assert(pipe != 0, "probe pipeline created")
    var ins = List[List[Float32]]()
    var addr = ren.buf_post_scratch.cpu_address
    for c in range(PROBE_CASES):
        var n = _rand_dir(c * 3 + 1, False)
        if model == SOTA_E_MODEL_SKIN:
            n = Vec3(0.0, 0.0, -1.0)
        var vv = _rand_dir(c * 3 + 2, True)
        var l = _rand_dir(c * 3 + 3, False)
        var row = List[Float32]()
        row.append(n.x); row.append(n.y); row.append(n.z)
        row.append(vv.x); row.append(vv.y); row.append(vv.z)
        row.append(l.x); row.append(l.y); row.append(l.z)
        row.append(m.base_color.x); row.append(m.base_color.y); row.append(m.base_color.z)
        row.append(Float32(0.028)); row.append(Float32(0.3) + Float32(c % 7) * Float32(0.08))
        row.append(Float32(0.004) * Float32(1 + c % 9))
        row.append(Float32(0.0))
        for k in range(PROBE_IN_WORDS):
            put_float(addr + (in_off + c * PROBE_IN_WORDS + k) * 4, row[k])
        ins.append(row^)
    # Poison the outputs so a dispatch that did not land (or was not made host-visible) can never pass
    # on values left by an earlier probe.
    for c in range(PROBE_CASES * 4):
        put_float(addr + (out_off + c) * 4, Float32(-12345.0))
    ren.vk.wait_idle()   # no frame of the renderer may still own command buffer 0
    _ = ren.vk.begin_commands(0)
    var push = List[UInt32](length=AAA_PUSH_WORDS, fill=0)
    ren.vk.dispatch_aaa(0, pipe, ren.pipe_layout, ren.desc_sets[0], push, (PROBE_CASES + 63) // 64, 1)
    ren.vk.memory_barrier2(0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
    var fence = ren.vk.submit(0)
    _ = ren.vk.wait(fence)
    var worst = Float32(0.0)
    var sp = SotaSkinParams(params[0], Vec3(params[6], params[7], params[8]), params[1], params[2], params[3], params[4])
    var hp = SotaHairParams(params[0], params[1], params[2] > Float32(0.5), params[3], params[4])
    for c in range(PROBE_CASES):
        var row = ins[c].copy()
        var n = Vec3(row[0], row[1], row[2])
        var vv = Vec3(row[3], row[4], row[5])
        var l = Vec3(row[6], row[7], row[8])
        var alb = Vec3(row[9], row[10], row[11])
        var rf: Vec3
        if model == SOTA_E_MODEL_SKIN:
            rf = sota_e_skin_direct_cpu(n, vv, l, alb, row[12], row[13], row[14], sp)
        else:
            rf = sota_e_hair_direct_cpu(n, vv, l, alb, Vec3(params[6], params[7], params[8]), hp)
        var o = List[Float32]()
        for k in range(3):
            o.append(get_float(addr + (out_off + c * 4 + k) * 4))
        var refs = [rf.x, rf.y, rf.z]
        for k in range(3):
            var e = abs(o[k] - refs[k]) / max(Float32(0.02), abs(refs[k]))
            if e > worst:
                worst = e
            if c < 3 and k == 0:
                print("      case", c, "gpu", o[0], o[1], o[2], "cpu", rf.x, rf.y, rf.z)
    ren.vk.destroy_pipeline(pipe)
    return worst

def _skin_bands(px: List[UInt32], w: Int, h: Int, rad_px: Float32) -> Tuple[Vec3, Vec3]:
    """(terminator band just past the shadow line, lit band) of a sphere centred on screen, sun along +X."""
    var cx = w // 2
    var band = Vec3(0.0, 0.0, 0.0)
    var nb = 0
    var lit_c = Vec3(0.0, 0.0, 0.0)
    var nl = 0
    for y in range(h // 2 - 4, h // 2 + 5):
        for dx in range(Int(rad_px * Float32(0.08)), Int(rad_px * Float32(0.30))):
            band = band + _rgb(px[y * w + cx - dx])
            nb += 1
        for dx in range(Int(rad_px * Float32(0.45)), Int(rad_px * Float32(0.6))):
            lit_c = lit_c + _rgb(px[y * w + cx + dx])
            nl += 1
    return (band * (Float32(1.0) / Float32(max(1, nb))), lit_c * (Float32(1.0) / Float32(max(1, nl))))


def _render_one(mut ren: FlintAaaVulkanRenderer, mesh_id: Int, mat: Int, cam: Vec3) -> List[UInt32]:
    var px = List[UInt32]()
    for _ in range(4):  # let TSR history settle on the new material
        ren.begin_frame(cam_pos=cam, cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=40.0, near_z=0.01)
        ren.draw_instance(mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=mat)
        px = _render(ren)
    return px^


def test_gpu_skin(mut ren: FlintAaaVulkanRenderer, w: Int, h: Int):
    var alb = Vec3(0.78, 0.50, 0.40)
    var std_mat = AaaMaterial(base_color=alb, roughness=0.45, metallic=0.0)
    std_mat.specular_factor = Float32(0.7)
    var wrap_mat = std_mat.copy()
    wrap_mat.subsurface_color = Vec3(0.9, 0.3, 0.2)
    wrap_mat.subsurface_amount = Float32(0.5)
    var skin_mat = sota_e_skin_material(alb, Float32(0.45), SotaSkinParams.human())
    var id_std = ren.add_material(std_mat)
    var id_wrap = ren.add_material(wrap_mat)
    var id_skin = ren.add_material(skin_mat)
    var sph = ren.add_mesh(_sphere(Float32(0.02)), id_std)

    ren.sun_dir = Vec3(1.0, 0.25, 0.05).normalized()  # side light: terminator runs down the middle
    var cam = Vec3(0.0, 0.0, -0.2)
    # Same sphere, same pixels: one material at a time at the screen centre (no perspective bias).
    var rad_px = Float32(0.02) / (Float32(0.2) * Float32(0.36397)) * Float32(h) * Float32(0.5)
    var px_std = _render_one(ren, sph.mesh_id, id_std, cam)
    var px_skin = _render_one(ren, sph.mesh_id, id_skin, cam)
    var bs = _skin_bands(px_std, w, h, rad_px)
    var bk = _skin_bands(px_skin, w, h, rad_px)
    print("    terminator band rgb  std =", bs[0].x, bs[0].y, bs[0].z, " burley =", bk[0].x, bk[0].y, bk[0].z)
    print("    lit band rgb         std =", bs[1].x, bs[1].y, bs[1].z, " burley =", bk[1].x, bk[1].y, bk[1].z)
    _save("skin_std_centre.png", px_std, w, h)
    _save("skin_burley_centre.png", px_skin, w, h)
    _assert(bk[0].x > bs[0].x * Float32(1.15), "Burley skin carries red light past the terminator (GPU)")
    _assert((bk[0].x - bs[0].x) > (bk[0].z - bs[0].z) * Float32(1.5), "GPU terminator bleed is red-dominant")
    _assert(abs(bk[1].y - bs[1].y) < Float32(0.12), "lit skin stays close to Lambert away from the terminator")

    # Side by side (std | legacy wrap | Burley) and the cvar: off must only change the tagged sphere.
    var xs = [Float32(-0.05), Float32(0.0), Float32(0.05)]
    var ids = [id_std, id_wrap, id_skin]
    var px = List[UInt32]()
    var px_off = List[UInt32]()
    for pass_i in range(2):
        ren.sota_e_character_shading = pass_i == 0
        for _ in range(4):
            ren.begin_frame(cam_pos=cam, cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=40.0, near_z=0.01)
            for i in range(3):
                ren.draw_instance(sph.mesh_id, trs(Vec3(xs[i], 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=ids[i])
            if pass_i == 0:
                px = _render(ren)
            else:
                px_off = _render(ren)
    ren.sota_e_character_shading = True
    _save("skin_std_wrap_burley.png", px, w, h)
    _save("skin_std_wrap_burley_flag_off.png", px_off, w, h)
    var diff_left = 0
    var diff_right = 0
    for y in range(h):
        for x in range(w):
            if px_off[y * w + x] != px[y * w + x]:
                if x < w * 2 // 3:
                    diff_left += 1
                else:
                    diff_right += 1
    print("    flag off: changed pixels std/wrap =", diff_left, " skin =", diff_right)
    # A few left pixels may move through GI bounce / TSR from the neighbouring skin sphere.
    _assert(diff_right > 100 and diff_left * 20 < diff_right, "the cvar only changes tagged (skin) pixels")
    print("  [PASS] Suite 3a: GPU pre-integrated Burley skin vs standard / legacy wrap")


def _dark_centroid(px: List[UInt32], w: Int, h: Int, x0: Int, x1: Int, y0: Int, y1: Int) -> Tuple[Float32, Float32, Int]:
    var sx = Float32(0.0)
    var sy = Float32(0.0)
    var n = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            var c = _rgb(px[y * w + x])
            if c.x + c.y + c.z < Float32(0.3):
                sx += Float32(x)
                sy += Float32(y)
                n += 1
    if n == 0:
        return (Float32(-1.0), Float32(-1.0), 0)
    return (sx / Float32(n), sy / Float32(n), n)


def _render_scene(mut ren: FlintAaaVulkanRenderer, cam: Vec3, fov: Float32, near: Float32, meshes: List[Int], mats: List[Int]) -> List[UInt32]:
    """Renders a static scene (identity transforms) for a few frames so TSR history settles."""
    var px = List[UInt32]()
    for _ in range(4):
        ren.begin_frame(cam_pos=cam, cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=fov, near_z=near)
        for i in range(len(meshes)):
            ren.draw_instance(meshes[i], Mat4(), material_override=mats[i])
        px = _render(ren)
    return px^


def test_gpu_eye(mut ren: FlintAaaVulkanRenderer, w: Int, h: Int):
    var p = SotaEyeParams.human()
    var eye_mat = sota_e_eye_material(Vec3(0.16, 0.36, 0.62), Vec3(0.86, 0.83, 0.80), Float32(0.06), p)
    var p_flat = p.copy()
    p_flat.iris_depth = Float32(0.0)
    var eye_flat = sota_e_eye_material(Vec3(0.16, 0.36, 0.62), Vec3(0.86, 0.83, 0.80), Float32(0.06), p_flat)
    var id_eye = ren.add_material(eye_mat)
    var id_flat = ren.add_material(eye_flat)
    var r = Float32(0.012)
    var eye = ren.add_mesh(_eye_mesh(r), id_eye)
    _assert(eye.triangle_count > 1000, "eye mesh cooked")
    ren.sun_dir = Vec3(0.55, 0.45, -0.70).normalized()

    # Frontal close-up.
    var px = _render_scene(ren, Vec3(0.0, 0.0, -0.06), Float32(30.0), Float32(0.005), [eye.mesh_id], [id_eye])
    _save("eye_front.png", px, w, h)
    var px_ff = _render_scene(ren, Vec3(0.0, 0.0, -0.06), Float32(30.0), Float32(0.005), [eye.mesh_id], [id_flat])
    _save("eye_front_no_parallax.png", px_ff, w, h)
    var c_mid = _rgb(px[(h // 2) * w + w // 2])
    var rad_px = r / (Float32(0.06) * Float32(0.26795)) * Float32(h) * Float32(0.5)
    var iris_px = Int(rad_px * Float32(0.33))  # pupil edge ~0.16 r, limbus ~0.48 r
    var c_iris = _rgb(px[(h // 2) * w + w // 2 - iris_px])
    var c_scl = _rgb(px[(h // 2 + Int(rad_px * Float32(0.1))) * w + w // 2 - Int(rad_px * Float32(0.86))])
    print("    eye pupil =", c_mid.x, c_mid.y, c_mid.z, " iris =", c_iris.x, c_iris.y, c_iris.z, " sclera =", c_scl.x, c_scl.y, c_scl.z)
    _assert(c_mid.x + c_mid.y + c_mid.z < Float32(0.3), "pupil is dark")
    _assert(c_iris.z > c_iris.x * Float32(1.3), "iris shows the blue iris colour")
    _assert(c_scl.x > Float32(0.25) and abs(c_scl.x - c_scl.z) < Float32(0.25), "sclera is a light neutral")

    # Oblique view: refraction parallax moves the apparent pupil relative to a zero-depth iris.
    var cam_o = Vec3(0.06 * sin(Float32(0.6)), 0.0, -0.06 * cos(Float32(0.6)))
    var px_o = _render_scene(ren, cam_o, Float32(30.0), Float32(0.005), [eye.mesh_id], [id_eye])
    _save("eye_oblique_refracted.png", px_o, w, h)
    var px_f = _render_scene(ren, cam_o, Float32(30.0), Float32(0.005), [eye.mesh_id], [id_flat])
    _save("eye_oblique_no_parallax.png", px_f, w, h)
    var x0 = w // 2 - Int(rad_px)
    var x1 = w // 2 + Int(rad_px)
    var y0 = h // 2 - Int(rad_px)
    var y1 = h // 2 + Int(rad_px)
    var c_ref = _dark_centroid(px_o, w, h, x0, x1, y0, y1)
    var c_flat = _dark_centroid(px_f, w, h, x0, x1, y0, y1)
    print("    oblique pupil centroid refracted =", c_ref[0], c_ref[1], "(", c_ref[2], "px)  flat =", c_flat[0], c_flat[1], "(", c_flat[2], "px)")
    _assert(c_ref[2] > 20 and c_flat[2] > 20, "pupil visible in both oblique renders")
    # Under refraction the pupil appears magnified (more pupil pixels) and displaced on screen.
    _assert(abs(c_ref[0] - c_flat[0]) > Float32(1.0), "cornea refraction parallax displaces the apparent pupil")

    # Any eyeball UV layout: a lat-long UV sphere relaid with `sota_e_eye_front_uv` renders like the
    # hand-made planar-UV eye; re-laying with a rotated gaze moves the pupil.
    var c_front = _dark_centroid(px, w, h, x0, x1, y0, y1)
    var mb_ll = MeshBuilder()
    mb_ll.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), rings=64, segments=96)
    var m_ll = mb_ll^.finish()
    var m_gz = m_ll.copy()
    var n_back = sota_e_eye_front_uv(m_ll, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, -1.0), Vec3(0.0, 1.0, 0.0), Float32(1.0), p.uv_per_radius)
    var gz = Float32(0.35)
    _ = sota_e_eye_front_uv(m_gz, Vec3(0.0, 0.0, 0.0), Vec3(sin(gz), 0.0, -cos(gz)), Vec3(0.0, 1.0, 0.0), Float32(1.0), p.uv_per_radius)
    for k in range(len(m_ll.positions)):
        m_ll.positions[k] = m_ll.positions[k] * r
        m_gz.positions[k] = m_gz.positions[k] * r
    var eye_ll = ren.add_mesh(m_ll, id_eye)
    var eye_gz = ren.add_mesh(m_gz, id_eye)
    var px_ll = _render_scene(ren, Vec3(0.0, 0.0, -0.06), Float32(30.0), Float32(0.005), [eye_ll.mesh_id], [id_eye])
    var px_gz = _render_scene(ren, Vec3(0.0, 0.0, -0.06), Float32(30.0), Float32(0.005), [eye_gz.mesh_id], [id_eye])
    _save("eye_relaid_latlong_uv.png", px_ll, w, h)
    _save("eye_relaid_gaze_20deg.png", px_gz, w, h)
    var c_ll = _dark_centroid(px_ll, w, h, x0, x1, y0, y1)
    var c_gz = _dark_centroid(px_gz, w, h, x0, x1, y0, y1)
    print("    relaid lat-long eye pupil =", c_ll[0], c_ll[1], "(", c_ll[2], "px) vs planar-UV eye", c_front[0], c_front[1], "(", c_front[2], "px);", n_back, "back verts on rim; gaze 20 deg pupil x =", c_gz[0])
    _assert(c_ll[2] > 20 and abs(c_ll[0] - c_front[0]) < Float32(1.0) and abs(c_ll[1] - c_front[1]) < Float32(1.0), "relaid lat-long eye puts the pupil where the planar-UV eye has it")
    _assert(abs(Float32(c_ll[2] - c_front[2])) < Float32(c_front[2]) * Float32(0.1), "relaid eye pupil has the same size")
    _assert(abs(c_gz[0] - c_ll[0]) > Float32(4.0), "re-laying with a rotated gaze moves the pupil")
    print("  [PASS] Suite 3b: GPU eye (cornea refraction parallax, iris, limbus, pupil, sclera, UV relayout)")


def test_gpu_hair(mut ren: FlintAaaVulkanRenderer, w: Int, h: Int) -> Tuple[Float64, Float64]:
    var hair_p = SotaHairParams.human()
    var hair_mat = sota_e_hair_material(Vec3(0.30, 0.16, 0.08), hair_p)
    var id_hair = ren.add_material(hair_mat)
    var id_skin = ren.add_material(sota_e_skin_material(Vec3(0.78, 0.50, 0.40), Float32(0.45), SotaSkinParams.human()))
    var g = SotaHairGroom.short_bob(Vec3(0.0, 0.0, 0.0), Float32(0.09))
    var t0 = get_time()
    var strands = build_hair_strand_ribbons(g)
    var t_build = (get_time() - t0) * 1000.0
    print("    strand ribbons:", g.strand_count, "strands,", len(strands.indices) // 3, "tris, built in", t_build, "ms")
    var hair_e = ren.add_mesh(strands, id_hair)
    var head_e = ren.add_mesh(_sphere(Float32(0.09)), id_skin)
    ren.sun_dir = Vec3(0.3, 0.75, -0.6).normalized()
    var cam = Vec3(0.0, 0.05, -0.42)
    var px = _render_scene(ren, cam, Float32(45.0), Float32(0.01), [head_e.mesh_id, hair_e.mesh_id], [id_skin, id_hair])
    _save("hair_strands_marschner.png", px, w, h)
    ren.sota_e_character_shading = False
    var px_off = _render_scene(ren, cam, Float32(45.0), Float32(0.01), [head_e.mesh_id, hair_e.mesh_id], [id_skin, id_hair])
    _save("hair_strands_standard_bsdf.png", px_off, w, h)
    ren.sota_e_character_shading = True

    var hair_px = 0
    var bright = 0
    var nan_like = 0
    var lum_max = Float32(0.0)
    for y in range(h // 6, h // 2):
        for x in range(w // 4, w * 3 // 4):
            var c = _rgb(px[y * w + x])
            var l = c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)
            if c.x > c.z * Float32(1.2):
                hair_px += 1
            if l > Float32(0.35):
                bright += 1
            lum_max = max(lum_max, l)
            if c.x > Float32(0.99) and c.y < Float32(0.01) and c.z > Float32(0.99):
                nan_like += 1
    print("    hair region: warm px =", hair_px, " highlight px =", bright, " max luma =", lum_max)
    _assert(hair_px > 500, "strand hair covers the scalp")
    _assert(bright > 10, "Marschner R/TRT highlights present on the strands")
    _assert(nan_like == 0, "no invalid shading")

    # Card fallback (alpha-tested strand texture).
    var tex = sota_e_hair_card_texture(64, 10, UInt32(3))
    var slot = ren.upload_bc7_texture(tex, 64, 64, srgb=False, generate_mips=True)
    var card_mat = sota_e_hair_material(Vec3(0.30, 0.16, 0.08), hair_p)
    card_mat.albedo_tex = slot
    card_mat.alpha_cutoff = Float32(0.5)
    var id_card = ren.add_material(card_mat)
    var cards_e = ren.add_mesh(build_hair_cards(g, 220, Float32(0.014)), id_card)
    var px_c = _render_scene(ren, cam, Float32(45.0), Float32(0.01), [head_e.mesh_id, cards_e.mesh_id], [id_skin, id_card])
    _save("hair_cards_fallback.png", px_c, w, h)

    # Deferred-pass cost: same frame with the character models on vs off.
    var on_ms = List[Float64]()
    var off_ms = List[Float64]()
    for it in range(24):
        ren.sota_e_character_shading = (it % 2) == 0
        ren.begin_frame(cam_pos=cam, cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0, near_z=0.01)
        ren.draw_instance(head_e.mesh_id, Mat4(), material_override=id_skin)
        ren.draw_instance(hair_e.mesh_id, Mat4(), material_override=id_hair)
        _ = ren.render_frame(capture_readback=True)
        if it >= 4:
            var tim = ren.get_gpu_pass_timings()
            for k in range(len(tim)):
                if tim[k].pass_name == "DeferredMultiLobeRtShade":
                    # timings read back belong to the previous frame of this slot
                    if (it % 2) == 0:
                        on_ms.append(tim[k].gpu_ms)
                    else:
                        off_ms.append(tim[k].gpu_ms)
    ren.sota_e_character_shading = True
    var m_on = _median(on_ms)
    var m_off = _median(off_ms)
    print("    deferred shade GPU ms (median) flag-on frames =", m_on, " flag-off frames =", m_off)
    print("  [PASS] Suite 3c: GPU strand hair (Marschner + dual scattering) and card fallback")
    return (m_on, m_off)


def _hair_lum(px: List[UInt32], w: Int, x0: Int, x1: Int, y0: Int, y1: Int) -> Float64:
    var acc = Float64(0.0)
    for y in range(y0, y1):
        for x in range(x0, x1):
            var c = _rgb(px[y * w + x])
            acc += Float64(c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722))
    return acc / Float64(max(1, (x1 - x0) * (y1 - y0)))


def test_gpu_hair_dom_aa(mut ren: FlintAaaVulkanRenderer, w: Int, h: Int):
    var hair_p = SotaHairParams.human()
    var id_hair = ren.add_material(sota_e_hair_material(Vec3(0.55, 0.38, 0.22), hair_p))
    var id_skin = ren.add_material(sota_e_skin_material(Vec3(0.78, 0.50, 0.40), Float32(0.45), SotaSkinParams.human()))
    var g = SotaHairGroom.short_bob(Vec3(0.0, 0.0, 0.0), Float32(0.09))
    var hair_e = ren.add_mesh(build_hair_strand_ribbons(g), id_hair)
    var head_e = ren.add_mesh(_sphere(Float32(0.09)), id_skin)
    ren.sun_dir = Vec3(0.15, 0.95, -0.25).normalized()   # top light: the crown shadows the strands below it
    var cam = Vec3(0.0, 0.03, -0.42)
    var meshes = List[Int]()
    meshes.append(head_e.mesh_id)
    meshes.append(hair_e.mesh_id)
    var mats = List[Int]()
    mats.append(id_skin)
    mats.append(id_hair)
    # (a) DOM off vs on
    ren.sota_e_hair.clear_strands()
    var px_off = _render_scene(ren, cam, Float32(45.0), Float32(0.01), meshes, mats)
    var strands = sota_e_groom_strands(g)
    ren.sota_e_hair.set_strands(strands, g.width, ren.buf_post_scratch.cpu_address)
    if not ren.sota_e_hair.active():
        print("  [SKIP] Suite 3d: hair DOM pipelines unavailable")
        return
    var px_on = _render_scene(ren, cam, Float32(45.0), Float32(0.01), meshes, mats)
    _save("hair_dom_off.png", px_off, w, h)
    _save("hair_dom_on.png", px_on, w, h)
    # GPU map == CPU reference (integer atomics: order independent)
    var fr = ren.sota_e_hair.frame(ren.sun_dir)
    var pts = sota_e_strands_to_points(strands)
    var ref_m = sota_e_dom_build_cpu(fr, pts)
    var nt = fr.res * fr.res
    var base = ren.buf_post_scratch.cpu_address + ren.sota_e_hair.base_word * 4
    var depth_same = 0
    var filled = 0
    var gsum = Float64(0.0)
    var csum = Float64(0.0)
    var max_dq = 0
    for i in range(nt):
        var gd = Int(get32(base + i * 4))
        var cd = Int(ref_m[0][i])
        var dq = abs(gd - cd)
        if (gd == 0) == (cd == 0) and dq <= 256:   # 256 / 2^24 of the depth range: float rounding only
            depth_same += 1
        if cd != 0:
            filled += 1
            max_dq = max(max_dq, dq)
    for i in range(nt * SOTA_E_DOM_LAYERS):
        gsum += Float64(get32(base + (nt + i) * 4))
        csum += Float64(ref_m[1][i])
    print("    DOM 128^2: filled texels", filled, " depth words equal (<= 256 LSB)", depth_same, "/", nt, " max |dq|", max_dq, " coverage gpu/cpu =", gsum / max(1.0, csum))
    _assert(filled > 2000, "the groom fills the deep opacity map")
    _assert(depth_same > nt * 98 // 100 and abs(gsum / max(1.0, csum) - 1.0) < 0.01, "GPU deep opacity map matches the CPU reference")
    var x0 = w // 4
    var x1 = w * 3 // 4
    var y0 = h // 6
    var y1 = h * 2 // 3
    var l_off = _hair_lum(px_off, w, x0, x1, y0, y1)
    var l_on = _hair_lum(px_on, w, x0, x1, y0, y1)
    print("    hair region mean luma  DOM off =", l_off, " DOM on =", l_on)
    _assert(l_on < l_off * 0.97 and l_on > l_off * 0.25, "DOM self-shadows the groom with partial (not opaque) occlusion")
    # DOM pass cost on its own command buffer (wall clock incl. submit; shared GPU -> median)
    var dom_ms = List[Float64]()
    for _ in range(9):
        ren.vk.wait_idle()
        _ = ren.vk.begin_commands(0)
        ren.sota_e_hair.record(ren.vk, 0, ren.pipe_layout, ren.desc_sets[0])
        var t0 = get_time()
        var fence = ren.vk.submit(0)
        _ = ren.vk.wait(fence)
        dom_ms.append((get_time() - t0) * 1000.0)
    print("    DOM build (", len(pts) // 4, "points ) submit+wait median", _median(dom_ms), "ms")
    ren.sota_e_hair.clear_strands()

    # (b) Strand AA: thin strands (0.5 mm) far enough to be ~0.3 px wide. Ground truth = view-facing
    # ribbons at the true width (a fibre is a cylinder) averaged over 16 TSR-jittered frames (~16x
    # supersampling). Error = one frame vs that reference, for the plain fixed-orientation ribbons, the
    # view-facing true-width ribbons, and the view-facing coverage LOD.
    var g2 = SotaHairGroom.short_bob(Vec3(0.0, 0.0, 0.0), Float32(0.09))
    g2.width = Float32(0.0005)
    var fov = Float32(20.0)
    var dist = Float32(1.05)
    var cam2 = Vec3(0.0, 0.02, -dist)
    var ppm = sota_e_px_per_m(dist, fov, ren.render_h)
    var lod = sota_e_hair_coverage_lod(g2.width, ppm, Float32(1.0))
    var vmesh = List[Int]()
    vmesh.append(ren.add_mesh(build_hair_strand_ribbons(g2), id_hair).mesh_id)
    vmesh.append(ren.add_mesh(build_hair_strand_ribbons_aa(g2, cam2, ppm, Float32(1.0), False), id_hair).mesh_id)
    var t_b = get_time()
    var aa_mesh = build_hair_strand_ribbons_aa(g2, cam2, ppm, Float32(1.0), True)
    var aa_build_ms = (get_time() - t_b) * 1000.0
    vmesh.append(ren.add_mesh(aa_mesh, id_hair).mesh_id)
    print("    strand width", g2.width * ppm, "px -> coverage LOD width x", lod[0], " strands kept", lod[1], " rebuild", aa_build_ms, "ms")
    var keep_hist = ren.tsr_history_weight
    ren.tsr_history_weight = Float32(0.02)   # jitter on, (almost) no accumulation: one sample per frame
    var frames = List[List[List[UInt32]]]()
    for v in range(3):
        var fl = List[List[UInt32]]()
        for _ in range(16):
            ren.begin_frame(cam_pos=cam2, cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=fov, near_z=Float32(0.01))
            ren.draw_instance(head_e.mesh_id, Mat4(), material_override=id_skin)
            ren.draw_instance(vmesh[v], Mat4(), material_override=id_hair)
            fl.append(_render(ren))
        frames.append(fl^)
    ren.tsr_history_weight = keep_hist
    var acc = List[Vec3](length=w * h, fill=Vec3(0.0, 0.0, 0.0))
    for f in range(16):
        for i in range(w * h):
            acc[i] = acc[i] + _rgb(frames[1][f][i]) * Float32(1.0 / 16.0)
    var hx0 = w // 2 - 70
    var hx1 = w // 2 + 70
    var hy0 = h // 2 - 70
    var hy1 = h // 2 + 50
    var med = List[Float64]()
    for v in range(3):
        var ev = List[Float64]()
        for f in range(16):
            var se = Float64(0.0)
            for y in range(hy0, hy1):
                for x in range(hx0, hx1):
                    var i = y * w + x
                    var dlt = _rgb(frames[v][f][i]) - acc[i]
                    se += Float64(dlt.dot(dlt))
            ev.append(sqrt(se / Float64((hx1 - hx0) * (hy1 - hy0))))
        med.append(_median(ev))
    print("    single-frame RMSE vs 16x supersampled fibres  fixed ribbons =", med[0], " view-facing =", med[1], " view-facing + coverage LOD =", med[2])
    var f_plain = frames[0][15].copy()
    var f_face = frames[1][15].copy()
    var f_aa = frames[2][15].copy()
    _save("hair_aa_fixed_ribbons.png", f_plain, w, h)
    _save("hair_aa_view_facing.png", f_face, w, h)
    _save("hair_aa_coverage_lod.png", f_aa, w, h)
    var ref_px = List[UInt32](length=w * h, fill=0)
    for i in range(w * h):
        var c = acc[i]
        var r8 = UInt32(max(Float32(0.0), min(Float32(255.0), c.x * Float32(255.0) + Float32(0.5))))
        var g8 = UInt32(max(Float32(0.0), min(Float32(255.0), c.y * Float32(255.0) + Float32(0.5))))
        var b8 = UInt32(max(Float32(0.0), min(Float32(255.0), c.z * Float32(255.0) + Float32(0.5))))
        ref_px[i] = UInt32(0xFF000000) | (r8 << 16) | (g8 << 8) | b8
    _save("hair_aa_reference_16x.png", ref_px, w, h)
    _assert(med[1] < med[0] * 0.85, "view-facing ribbons cut single-frame error against the supersampled fibres")
    # With TSR accumulating (default history), sub-pixel strands shimmer: the history clamp keeps
    # rejecting the samples that hit / miss a 0.3 px strand. Shimmer = mean |frame - previous frame| of the
    # resolved output over the hair, static camera, after 12 frames of convergence.
    var shim = List[Float64]()
    var conv_err = List[Float64]()
    ren.tsr_history_weight = Float32(0.82)   # the renderer default (the suite's _setup turns TSR off)
    for v in range(3):
        var prev = List[UInt32]()
        var s_acc = Float64(0.0)
        var n_acc = 0
        var last = List[UInt32]()
        for f in range(28):
            ren.begin_frame(cam_pos=cam2, cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=fov, near_z=Float32(0.01))
            ren.draw_instance(head_e.mesh_id, Mat4(), material_override=id_skin)
            ren.draw_instance(vmesh[v], Mat4(), material_override=id_hair)
            var px = _render(ren)
            if f >= 12:
                for y in range(hy0, hy1):
                    for x in range(hx0, hx1):
                        var i = y * w + x
                        var dlt = _rgb(px[i]) - _rgb(prev[i])
                        s_acc += Float64(abs(dlt.x) + abs(dlt.y) + abs(dlt.z)) / 3.0
                        n_acc += 1
            prev = px.copy()
            last = px^
        shim.append(s_acc / Float64(max(1, n_acc)))
        var se = Float64(0.0)
        for y in range(hy0, hy1):
            for x in range(hx0, hx1):
                var i = y * w + x
                var dlt = _rgb(last[i]) - acc[i]
                se += Float64(dlt.dot(dlt))
        conv_err.append(sqrt(se / Float64((hx1 - hx0) * (hy1 - hy0))))
    ren.tsr_history_weight = keep_hist
    print("    TSR shimmer |dF| (hair region)  fixed =", shim[0], " view-facing =", shim[1], " coverage LOD =", shim[2])
    print("    TSR-resolved RMSE vs reference   fixed =", conv_err[0], " view-facing =", conv_err[1], " coverage LOD =", conv_err[2])
    _assert(conv_err[1] < conv_err[0] * 0.7, "view-facing ribbons converge much closer to the supersampled fibres under TSR")
    _assert(shim[2] < shim[1] * 0.9, "coverage strand LOD reduces TSR shimmer of sub-pixel strands")
    print("  [PASS] Suite 3d: hair deep opacity shadows + coverage-preserving strand AA")


def test_gpu():
    var w = 320
    var h = 240
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    _setup(ren)
    # Warm one frame so every ring buffer exists, then probe the emitters against the CPU references.
    ren.begin_frame()
    _ = ren.render_frame(capture_readback=True)
    var e_skin = run_emitter_probe(ren, SOTA_E_MODEL_SKIN, sota_e_skin_material(Vec3(0.78, 0.50, 0.40), Float32(0.45), SotaSkinParams.human()))
    var e_hair = run_emitter_probe(ren, SOTA_E_MODEL_HAIR, sota_e_hair_material(Vec3(0.30, 0.16, 0.08), SotaHairParams.human()))
    print("    GPU emitter vs CPU reference max rel err: skin =", e_skin, " hair =", e_hair)
    _assert(e_skin < Float32(2e-3) and e_hair < Float32(2e-3), "GPU skin / hair emitters match the CPU references")
    test_gpu_skin(ren, w, h)
    test_gpu_eye(ren, w, h)
    var t = test_gpu_hair(ren, w, h)
    test_gpu_hair_dom_aa(ren, w, h)
    _assert(abs(t[0] - t[1]) < 1.5, "character models cost < 1.5 ms in the deferred pass at 320x240")
    ren.shutdown()


# ---------------------------------------------------------------------------------------------
# Suite 4: validated on-disk pipeline cache, cold vs warm renderer start
# ---------------------------------------------------------------------------------------------
def _setenv(name: String, value: String):
    var n = name
    var v = value
    _ = external_call["setenv", Int32](n.as_c_string_span(), v.as_c_string_span(), Int32(1))


def _fake_payload(id: SotaEDeviceIdentity, n: Int) -> List[UInt8]:
    var p = List[UInt8](length=n, fill=0)
    p[0] = 32
    p[4] = 1
    for k in range(4):
        p[8 + k] = UInt8((id.vendor >> UInt32(8 * k)) & 0xFF)
        p[12 + k] = UInt8((id.device >> UInt32(8 * k)) & 0xFF)
    for i in range(16):
        p[16 + i] = id.uuid[i]
    for i in range(32, n):
        p[i] = UInt8((i * 7) & 0xFF)
    return p^


def test_cpu_pso_container():
    var uuid = List[UInt8]()
    for i in range(16):
        uuid.append(UInt8(i * 9 + 1))
    var id = SotaEDeviceIdentity(0x10DE, 0x2B85, 0x1234, uuid^)
    var payload = _fake_payload(id, 300)
    var manifest = List[UInt64]()
    manifest.append(UInt64(11)); manifest.append(UInt64(22)); manifest.append(UInt64(33))
    var file = sota_e_pso_encode(id, manifest, payload)
    var ok = sota_e_pso_decode(file, id)
    _assert(ok.ok and len(ok.manifest) == 3 and len(ok.payload) == 300, "container round-trips")
    var flipped = file.copy()
    flipped[len(flipped) - 5] ^= 0x40
    var r1 = sota_e_pso_decode(flipped, id)
    var cut = file.copy()
    cut.resize(len(cut) - 17, 0)
    var r2 = sota_e_pso_decode(cut, id)
    var other = id.copy()
    other.driver_version = 0x1235
    var r3 = sota_e_pso_decode(file, other)
    var bad_hdr = _fake_payload(id, 300)
    bad_hdr[8] ^= 0xFF  # Vulkan header names another vendor
    var r4 = sota_e_pso_decode(sota_e_pso_encode(id, manifest, bad_hdr), id)
    print("    rejects: bit flip ->", r1.reason, "| truncated ->", r2.reason, "| new driver ->", r3.reason, "| foreign header ->", r4.reason)
    _assert(not r1.ok and not r2.ok and not r3.ok and not r4.ok, "corrupt / foreign caches are rejected before the driver sees them")
    print("  [PASS] Suite 4a: pipeline-cache container validation (device identity, hash, size, Vulkan header)")


def _construct_ms() -> Tuple[Float64, Int, Int, Int, String, Float64]:
    """(construct ms, cache status, modules compiled, warm manifest hits, reason, compute-pipeline compile ms)."""
    var t0 = get_time()
    var ren = FlintAaaVulkanRenderer(width=320, height=240, tsr_scale=1.0)
    var ms = (get_time() - t0) * 1000.0
    if not ren.active:
        return (ms, -1, 0, 0, String("renderer unavailable"), Float64(0.0))
    var n = len(ren.vk.sota_e_spirv_hashes)
    var hits = ren.sota_e_pso.warm_hits(ren.vk.sota_e_spirv_hashes)
    var st = ren.sota_e_pso.status
    var reason = ren.sota_e_pso.reason
    var compile_ms = ren.vk.sota_perf_compute_compile_s * 1000.0
    ren.shutdown()
    return (ms, st, n, hits, reason, compile_ms)


def test_gpu_pso_cache():
    var dir = env_value("FLINT_SOTA_E_PSO_TMP")
    if dir.byte_length() == 0:
        dir = "build/v5/sota_e_pso_test"
    _ = make_dirs(dir)
    _setenv("FLINT_PSO_CACHE_DIR", dir)
    var path = sota_e_pso_cache_path()
    # Cold start, serial compile (the pre-round-2 behaviour) vs parallel JobPool compile.
    # --- sota:perf timing --- interleaved serial/parallel rounds, best of each, load-aware bound,
    # stops at the first passing round (tests/sota_perf_timing.mojo).
    var rounds = timing_rounds()
    var cold_serial = (Float64(1.0e300), 0, 0, 0, String(""), Float64(1.0e300))
    var cold = (Float64(1.0e300), 0, 0, 0, String(""), Float64(1.0e300))
    var speed_ok = False
    for r in range(rounds):
        _setenv("FLINT_PARALLEL_PSO", "0")
        _ = remove_file(path)
        var s_run = _construct_ms()
        if s_run[1] < 0:
            print("  [SKIP] Suite 4b: Vulkan renderer unavailable")
            return
        if s_run[5] < cold_serial[5]:
            cold_serial = s_run^
        _setenv("FLINT_PARALLEL_PSO", "1")
        _ = remove_file(path)
        var p_run = _construct_ms()
        if r == 0:
            _assert(p_run[1] == SOTA_E_PSO_COLD_MISSING, "parallel cold start compiles")
        if p_run[5] < cold[5]:
            cold = p_run^
        # Compare the compute-pipeline compile itself (what the JobPool changes), not the whole construct.
        # The parallel edge shrinks when the other cores are busy: 0.8x idle, 0.95x fully loaded.
        var need = load_aware_ratio(0.8, 0.95)
        speed_ok = cold[5] < cold_serial[5] * need
        timing_note("parallel vs serial compute-pipeline compile, cold", speed_ok, r, rounds,
                    String(Int(cold[5])) + " ms vs " + String(Int(cold_serial[5])) + " ms (bound x" + String(Float64(Int(need * 1000.0)) / 1000.0)
                    + "; construct " + String(Int(cold[0])) + " vs " + String(Int(cold_serial[0])) + " ms)")
        if speed_ok:
            break
    print("    cold construct serial", cold_serial[0], "ms -> parallel", cold[0], "ms (", cold[2], "modules )")
    _assert(cold[1] == SOTA_E_PSO_COLD_MISSING, "parallel cold start compiles")
    _assert(speed_ok, "parallel pipeline compile beats serial cold start")
    # The cache file on disk now comes from the last parallel cold start.
    var size = Int(len(read_bytes(path)))
    var warm = List[Float64]()
    var w_last = _construct_ms()
    warm.append(w_last[0])
    for _ in range(2):
        var w = _construct_ms()
        warm.append(w[0])
    var warm_med = min_of(warm)   # --- sota:perf timing --- best warm start vs best cold start
    print("    cold construct", cold[0], "ms (", cold[2], "modules,", cold[4], ")  warm best", warm_med, "ms  file", size, "B")
    print("    warm status", w_last[1], "manifest hits", w_last[3], "/", w_last[2])
    _assert(cold[1] == SOTA_E_PSO_COLD_MISSING and size > 1000, "cold start compiles and writes the cache atomically")
    _assert(w_last[1] == SOTA_E_PSO_WARM and w_last[3] == w_last[2], "warm start loads the validated cache; every module is a manifest hit")
    _assert(warm_med < cold[0] * 0.6, "warm start is well under the cold start")
    # A torn / corrupted file must be rejected (cold start, rewritten) instead of crashing the driver.
    var bytes = read_bytes(path)
    bytes.resize(len(bytes) // 2, 0)
    _ = write_bytes(path, bytes)
    var torn = _construct_ms()
    print("    torn file ->", torn[4])
    _assert(torn[1] == SOTA_E_PSO_COLD_REJECTED, "torn cache file is rejected")
    var healed = _construct_ms()
    _assert(healed[1] == SOTA_E_PSO_WARM, "cache is rewritten after a rejected load")
    print("  [PASS] Suite 4b: cold serial", Int(cold_serial[0]), "ms -> cold parallel", Int(cold[0]), "ms -> warm", Int(warm_med), "ms")


# ---------------------------------------------------------------------------------------------
# Suite 5: GPU-driven alpha-mask / outline packing on Area A's cluster culler
# ---------------------------------------------------------------------------------------------
def _small_sphere(r: Float32) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(r, r, r), rings=8, segments=12)
    return mb^.finish()


def _route_counts(ren: FlintAaaVulkanRenderer, slot: Int) -> Tuple[Int, Int]:
    return (
        Int(get32(ren.ring_alphamask_indirect[slot].cpu_address)),
        Int(get32(ren.ring_outline_indirect[slot].cpu_address)),
    )


def _grid_frame(mut ren: FlintAaaVulkanRenderer, cam: Vec3, mesh: Int, ids: List[Int], n_side: Int, spacing: Float32):
    ren.begin_frame(cam_pos=cam, cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=Float32(50.0), near_z=Float32(0.05))
    var half = Float32(n_side - 1) * Float32(0.5)
    for iy in range(n_side):
        for ix in range(n_side):
            var k = iy * n_side + ix
            var pos = Vec3((Float32(ix) - half) * spacing, (Float32(iy) - half) * spacing, Float32(0.0))
            ren.draw_instance(mesh, trs(pos, Quat(), Vec3(1.0, 1.0, 1.0)), material_override=ids[k % len(ids)])


def test_gpu_driven():
    var w = 320
    var h = 240
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    _setup(ren)
    if not ren.sota_e_gpu.active:
        print("  [SKIP] GPU-driven packing inactive (no drawIndirectCount or Area A culler)")
        ren.shutdown()
        return
    var am = AaaMaterial(base_color=Vec3(0.30, 0.62, 0.22), roughness=0.6, metallic=0.0)
    am.alpha_cutoff = Float32(0.5)
    var id_am = ren.add_material(am)
    var id_cel = ren.add_cel_material(AaaMaterial(base_color=Vec3(0.85, 0.55, 0.35), roughness=0.5, metallic=0.0), AaaCelParams(outline_width_px=3.0))
    var id_op = ren.add_material(AaaMaterial(base_color=Vec3(0.55, 0.55, 0.60), roughness=0.5, metallic=0.0))
    var sph = ren.add_mesh(_small_sphere(Float32(0.1)), id_op)
    var ids = List[Int]()
    ids.append(id_am); ids.append(id_cel); ids.append(id_am); ids.append(id_op)
    ren.sun_dir = Vec3(0.4, 0.8, -0.45).normalized()

    # 5a. Correctness: a 6x6 grid fully on screen; GPU lists == CPU lists (count and image).
    var cam = Vec3(0.0, 0.0, -3.2)
    var px_mode = List[List[UInt32]]()
    var counts = List[Tuple[Int, Int]]()
    for mode in range(2):
        ren.sota_e_gpu.enabled = mode == 0
        var px = List[UInt32]()
        var c = (0, 0)
        for _ in range(4):
            var slot = ren.frame_index % 2
            _grid_frame(ren, cam, sph.mesh_id, ids, 6, Float32(0.34))
            px = _render(ren)
            c = _route_counts(ren, slot)
        px_mode.append(px^)
        counts.append(c)
    print("    alpha-mask / outline draws  gpu =", counts[0][0], "/", counts[0][1], "  cpu =", counts[1][0], "/", counts[1][1])
    _assert(counts[0][0] > 0 and counts[0][1] > 0, "GPU route culler emits alpha-mask and outline draws")
    _assert(counts[0][0] == counts[1][0] and counts[0][1] == counts[1][1], "GPU draw lists match the CPU packers on an all-visible grid")
    var diff = Float64(0.0)
    var n_bad = 0
    for i in range(w * h):
        var a = _rgb(px_mode[0][i])
        var b = _rgb(px_mode[1][i])
        var d = abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)
        diff += Float64(d)
        if d > Float32(0.06):
            n_bad += 1
    print("    image gpu vs cpu lists: mean |d| =", diff / Float64(w * h), " pixels > 0.06:", n_bad)
    _save("gpu_driven_grid_gpu.png", px_mode[0], w, h)
    _save("gpu_driven_grid_cpu.png", px_mode[1], w, h)
    _assert(n_bad < (w * h) // 500, "GPU-packed alpha-mask / outline passes render the same image")

    # 5b. CPU cost: 96x96 = 9216 instances (4608 alpha-masked, 2304 outlined), all in view.
    var n_side = 96
    var cam_far = Vec3(0.0, 3.0, -26.0)   # whole field in view: the CPU packers write every meshlet
    # Modes alternate frame by frame so host-load drift hits both equally.
    # --- sota:perf timing --- compare the lower quartile of each mode (noise only adds time) and measure in
    # rounds of 40 frames, stopping at the first passing round (tests/sota_perf_timing.mojo).
    var ts_gpu = List[Float64]()
    var ts_cpu = List[Float64]()
    var res = List[Float64]()
    res.append(0.0)
    res.append(0.0)
    var rounds = timing_rounds()
    var lists_ok = False
    for r in range(rounds):
        for f in range(48 if r == 0 else 40):
            var gpu_mode = (f % 2) == 0
            ren.sota_e_gpu.enabled = gpu_mode
            _grid_frame(ren, cam_far, sph.mesh_id, ids, n_side, Float32(0.22))
            var t0 = get_time()
            _ = ren.render_frame()
            var dt = (get_time() - t0) * 1000.0
            if r > 0 or f >= 8:
                if gpu_mode:
                    ts_gpu.append(dt)
                else:
                    ts_cpu.append(dt)
        res[0] = quantile(ts_gpu, 0.25)
        res[1] = quantile(ts_cpu, 0.25)
        lists_ok = res[0] < res[1] * 1.10
        timing_note("GPU vs CPU draw lists, render_frame q25", lists_ok, r, rounds,
                    String(res[0]) + " ms vs " + String(res[1]) + " ms (medians " + String(_median(ts_gpu)) + " / " + String(_median(ts_cpu)) + ")")
        if lists_ok:
            break
    # The CPU packers alone (what the GPU path removes), timed on the same frame's instance list.
    var tp = List[Float64]()
    var tan_y = Float32(0.46630765)  # tan(25 deg)
    var tan_x = tan_y * Float32(w) / Float32(h)
    for _ in range(12):
        _grid_frame(ren, cam_far, sph.mesh_id, ids, n_side, Float32(0.22))
        var t0 = get_time()
        _ = ren._pack_alphamask_visbuffer_commands(n_side * n_side, tan_x, tan_y)
        _ = ren._pack_outline_visbuffer_commands(n_side * n_side, tan_x, tan_y)
        tp.append((get_time() - t0) * 1000.0)
    var pack_ms = _median(tp)
    _ = ren.render_frame()
    ren.sota_e_gpu.enabled = True
    print("    9216 instances: render_frame CPU q25  gpu lists =", res[0], "ms  cpu lists =", res[1], "ms  (CPU packers alone", pack_ms, "ms)")
    _assert(pack_ms > 0.0 and lists_ok, "GPU-driven lists remove the CPU packers without slowing the frame")
    print("  [PASS] Suite 5: GPU-driven alpha-mask / outline lists (identical image; CPU packers", pack_ms, "ms removed at 9216 instances)")
    ren.shutdown()


def main():
    print("=== Flint v5 sota:E Characters & Production Suite ===")
    test_spirv()
    test_cpu_skin()
    test_cpu_eye()
    test_cpu_hair()
    test_cpu_hair_dom()
    test_cpu_pso_container()
    if env_value("FLINT_SOTA_E_KEEP_DRIVER_CACHE").byte_length() == 0:
        _setenv("__GL_SHADER_DISK_CACHE", "0")  # measure Flint's cache, not the NVIDIA driver's own
    test_gpu_pso_cache()
    test_gpu()
    test_gpu_driven()
    print("=== ALL FLINT v5 sota:E TESTS PASSED ===")
