"""Area D round 3 (sota:camera): HDR display output (HDR10 PQ and scRGB) as a display mapping after AgX.

Pipeline. The SDR path stays exactly as it was: scene-referred HDR -> exposure -> AgX (Rec.709, sRGB
encoded, then the display grade) -> 8-bit BGRA. With `SotaHdrOutput.enable` and an HDR swapchain the
tonemap runs the same AgX look and then maps the *display-referred* AgX image to absolute luminance:

  1. linearise the AgX output with the sRGB EOTF (the image the SDR display would show, 0..1),
  2. expand the highlights above a knee `k` on the max channel (hue preserving, ratio scale):
         E(m) = m                                  for m <= k
         E(m) = m + (P - 1) ((m - k) / (1 - k))^2  for m >  k,    P = peak_nits / paper_white_nits
     E is C1 at the knee, monotonic (E' >= 1), E(1) = P, and the identity when P = 1 (so the HDR image
     at peak = paper white is the SDR image shown at paper white),
  3. scale by paper white (default 203 nits, ITU-R BT.2408 graphics / diffuse white), Rec.709 -> Rec.2020,
     SMPTE ST.2084 PQ, packed into A2B10G10R10 (the existing HDR10 word layout).
  For an scRGB (R16G16B16A16_SFLOAT, extended linear sRGB) swapchain the same PQ word is produced (so the
  UI overlay composites into it at the same paper white) and a small pass converts it to linear Rec.709
  with 1.0 = 80 nits after the UI.

Why after AgX and not a separate HDR tone curve: the look (contrast, hue path of bright saturated colours,
the Purkinje shadow treatment and the grade) is authored once and is identical on SDR and HDR displays
below the knee; HDR only adds highlight headroom, which is what BT.2408 sec. 5 and BT.2390 sec. 5.4 suggest
for mixing SDR-graded content into HDR (SDR white mapped to the reference white, highlights expanded).

Round 4: luminance-aware expansion (`expansion_mode = SOTA_HDR_EXPAND_LUMINANCE`), after the structure of
ITU-R BT.2446-1 method C (inverse direction): a crosstalk matrix with parameter alpha is applied to the
display-linear AgX RGB to estimate the driving luminance Yc = luma(X rgb), X = (1 - 2a) I + a 1 1^T (a
little of the other channels, so saturated colours are not judged by their luminance alone); Yc is expanded
with a linear segment up to the knee and an exponential segment above it (the inverse of method C's
logarithmic segment), C1 at the knee and solved so E(1) = P:
       E(y) = y                                  for y <= k
       E(y) = k + (exp(b (y - k)) - 1) / b       for y >  k,   b > 0 from k + (exp(b (1 - k)) - 1) / b = P
and the colour is scaled by E(Yc) / Yc (chromaticity kept), limited so no channel exceeds the peak. Bright
saturated colours (a red tail light, a blue sky patch) keep their SDR luminance instead of being pushed to
the display peak on their max channel, while near-white highlights (sun, specular, clouds) get the full
range. The default stays the round-3 max-channel curve, so existing HDR output is unchanged.

Display metadata: with `VK_EXT_hdr_metadata` the swapchain gets ST.2086 mastering primaries (BT.2020,
D65), max luminance = peak, MaxCLL = peak (the mapping never exceeds it) and MaxFALL = paper white.

SDR fallback: when the swapchain negotiated SDR (no `VK_EXT_swapchain_colorspace`, or the surface offers no
HDR10 / scRGB format) the renderer's display mode is SDR and this module changes nothing; headless runs use
the display mode the test sets, so default SDR PNGs are bit-identical.

Sources: SMPTE ST 2084:2014 (PQ); ITU-R BT.2100-2 (PQ, BT.2020 primaries); ITU-R BT.2408-7 (203 nit
reference white, sec. 5 SDR <-> HDR mapping); ITU-R BT.2390-11 sec. 5.4 (display mapping); SMPTE ST 2086
(mastering metadata); CTA-861.3 (MaxCLL / MaxFALL); Khronos VK_EXT_swapchain_colorspace / VK_EXT_hdr_metadata
specifications; Sobotka, AgX (MIT/CC0 reference configs) for the SDR look the mapping starts from.
"""
from std.collections import List
from std.math import exp, max, min, pow
from std.memory import bitcast
from flint.platform.vulkan import (
    ACCESS_HOST_READ,
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    ACCESS_TRANSFER_READ,
    PIPELINE_COMPUTE,
    PIPELINE_HOST,
    PIPELINE_TRANSFER,
    VulkanGpu,
    call4,
    put32,
    put64,
)
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_HISTORY, BIND_AAA_POST_SCRATCH

comptime SOTA_DISPLAY_AGX_HDR: UInt32 = 4  # tonemap mode: AgX look + HDR display mapping -> PQ word
comptime SOTA_HDR_LEGACY_PQ: UInt32 = 1    # AAA_DISPLAY_HDR10_PQ (raw linear -> PQ, no look)
comptime SOTA_HDR_LEGACY_SCRGB: UInt32 = 3  # AAA_DISPLAY_SCRGB_FP16

# Tonemap push words used by the HDR mapping (0..9 are the tonemap's own).
comptime SOTA_HDR_PUSH_PAPER: Int = 10
comptime SOTA_HDR_PUSH_PEAK_RATIO: Int = 11
comptime SOTA_HDR_PUSH_KNEE: Int = 12
comptime SOTA_HDR_PUSH_MODE: Int = 13       # round 4: 0 max channel, 1 luminance (BT.2446-C style)
comptime SOTA_HDR_PUSH_CROSSTALK: Int = 14
comptime SOTA_HDR_PUSH_EXP_B: Int = 15
comptime SOTA_HDR_EXPAND_MAX_CHANNEL: Int = 0
comptime SOTA_HDR_EXPAND_LUMINANCE: Int = 1
# UI overlay push word carrying the paper white in nits (0 = legacy 203 nits).
comptime SOTA_HDR_UI_PUSH_PAPER: Int = 6


struct SotaHdrOutput(Copyable, ImplicitlyCopyable, Movable):
    """HDR output controls (cvars). `enable` is off by default: the legacy HDR10 path (raw linear -> PQ)
    and every SDR path stay unchanged."""
    var enable: Bool
    var paper_white_nits: Float32  # SDR / UI white on the HDR display (BT.2408: 203)
    var peak_nits: Float32         # display peak; AgX white maps here
    var knee: Float32              # display-linear AgX value where the highlight expansion starts
    var min_nits: Float32          # ST.2086 mastering min luminance
    var expansion_mode: Int        # SOTA_HDR_EXPAND_MAX_CHANNEL (round 3 default) | SOTA_HDR_EXPAND_LUMINANCE
    var crosstalk: Float32         # BT.2446-C crosstalk alpha (0 .. 0.33) for the luminance mode
    var meta_swapchain: Int        # swapchain the metadata was last sent for
    var meta_key: UInt64           # settings the metadata was last sent with
    var meta_sent: Int             # number of vkSetHdrMetadataEXT calls (tests / diagnostics)

    def __init__(out self):
        self.enable = False
        self.paper_white_nits = Float32(203.0)
        self.peak_nits = Float32(1000.0)
        self.knee = Float32(0.6)
        self.min_nits = Float32(0.005)
        self.expansion_mode = SOTA_HDR_EXPAND_MAX_CHANNEL
        self.crosstalk = Float32(0.05)
        self.meta_swapchain = 0
        self.meta_key = 0
        self.meta_sent = 0

    def paper(self) -> Float32:
        return max(Float32(1.0), min(Float32(10000.0), self.paper_white_nits))

    def peak_ratio(self) -> Float32:
        """P = peak / paper white, at least 1 (a peak below paper white would dim the SDR range)."""
        return max(Float32(1.0), min(Float32(10000.0), self.peak_nits) / self.paper())

    def knee_clamped(self) -> Float32:
        return max(Float32(0.0), min(Float32(0.95), self.knee))

    def crosstalk_clamped(self) -> Float32:
        return max(Float32(0.0), min(Float32(0.33), self.crosstalk))

    def exp_b(self) -> Float32:
        """Exponential-segment rate of the luminance mode, solved so E(1) = P (0 = identity)."""
        return sota_hdr_solve_exp_b(self.peak_ratio(), self.knee_clamped())


def sota_hdr_tonemap_mode(hdr: SotaHdrOutput, display_mode: UInt32, legacy_mode: UInt32) -> UInt32:
    """Tonemap display mode for this frame. `display_mode` is what the swapchain negotiated (or what a
    headless caller set); `legacy_mode` is what the renderer would use without this module."""
    if hdr.enable and (display_mode == SOTA_HDR_LEGACY_PQ or display_mode == SOTA_HDR_LEGACY_SCRGB):
        return SOTA_DISPLAY_AGX_HDR
    return legacy_mode


def sota_hdr_ui_mode(tonemap_mode: UInt32) -> UInt32:
    """The UI overlay composites into PQ words for the mapped HDR output too."""
    return SOTA_HDR_LEGACY_PQ if tonemap_mode == SOTA_DISPLAY_AGX_HDR else tonemap_mode


def sota_hdr_ui_paper_push(hdr: SotaHdrOutput, tonemap_mode: UInt32) -> UInt32:
    return bitcast[DType.uint32](hdr.paper()) if tonemap_mode == SOTA_DISPLAY_AGX_HDR else UInt32(0)


def sota_hdr_tonemap_push(hdr: SotaHdrOutput, mut push: List[UInt32]):
    push[SOTA_HDR_PUSH_PAPER] = bitcast[DType.uint32](hdr.paper())
    push[SOTA_HDR_PUSH_PEAK_RATIO] = bitcast[DType.uint32](hdr.peak_ratio())
    push[SOTA_HDR_PUSH_KNEE] = bitcast[DType.uint32](hdr.knee_clamped())
    if hdr.expansion_mode == SOTA_HDR_EXPAND_LUMINANCE:
        push[SOTA_HDR_PUSH_MODE] = UInt32(1)
        push[SOTA_HDR_PUSH_CROSSTALK] = bitcast[DType.uint32](hdr.crosstalk_clamped())
        push[SOTA_HDR_PUSH_EXP_B] = bitcast[DType.uint32](hdr.exp_b())


# ----------------------------------------------------------------------------------------------
# CPU references
# ----------------------------------------------------------------------------------------------

def srgb_eotf_cpu(v: Float32) -> Float32:
    var c = max(Float32(0.0), min(Float32(1.0), v))
    if c <= Float32(0.04045):
        return c / Float32(12.92)
    return pow((c + Float32(0.055)) / Float32(1.055), Float32(2.4))


def srgb_oetf_cpu(l: Float32) -> Float32:
    var c = max(Float32(0.0), min(Float32(1.0), l))
    if c < Float32(0.0031308):
        return c * Float32(12.92)
    return Float32(1.055) * pow(c, Float32(1.0 / 2.4)) - Float32(0.055)


comptime _PQ_M1: Float32 = 0.1593017578125
comptime _PQ_M2: Float32 = 78.84375
comptime _PQ_C1: Float32 = 0.8359375
comptime _PQ_C2: Float32 = 18.8515625
comptime _PQ_C3: Float32 = 18.6875


def pq_oetf_cpu(nits: Float32) -> Float32:
    """ST.2084 inverse EOTF: absolute luminance (cd/m^2) -> PQ signal 0..1."""
    var y = max(Float32(0.0), min(Float32(1.0), nits / Float32(10000.0)))
    var ym = pow(max(Float32(1.0e-9), y), _PQ_M1)
    return max(Float32(0.0), min(Float32(1.0), pow((_PQ_C1 + _PQ_C2 * ym) / (Float32(1.0) + _PQ_C3 * ym), _PQ_M2)))


def pq_eotf_cpu(v: Float32) -> Float32:
    """ST.2084 EOTF: PQ signal 0..1 -> absolute luminance (cd/m^2)."""
    var e = pow(max(Float32(0.0), min(Float32(1.0), v)), Float32(1.0) / _PQ_M2)
    var num = max(Float32(0.0), e - _PQ_C1)
    var den = max(Float32(1.0e-9), _PQ_C2 - _PQ_C3 * e)
    return Float32(10000.0) * pow(num / den, Float32(1.0) / _PQ_M1)


def sota_hdr_expand_cpu(m: Float32, peak_ratio: Float32, knee: Float32) -> Float32:
    if m <= knee:
        return m
    var t = (m - knee) / max(Float32(1.0e-6), Float32(1.0) - knee)
    return m + (peak_ratio - Float32(1.0)) * t * t


def _exp_seg(t: Float32, b: Float32) -> Float32:
    """(exp(b t) - 1) / b, with the b -> 0 limit t."""
    if b < Float32(1.0e-4):
        return t
    return (exp(b * t) - Float32(1.0)) / b


def sota_hdr_solve_exp_b(peak_ratio: Float32, knee: Float32) -> Float32:
    """b > 0 with knee + (exp(b (1 - knee)) - 1) / b = P (bisection; monotonic in b). 0 when P <= 1."""
    var k = max(Float32(0.0), min(Float32(0.95), knee))
    var target = peak_ratio - k
    var span = Float32(1.0) - k
    if target <= span * Float32(1.0001):
        return Float32(0.0)
    var lo = Float32(0.0)
    var hi = Float32(1.0)
    while _exp_seg(span, hi) < target and hi < Float32(1.0e4):
        hi *= Float32(2.0)
    for _ in range(60):
        var mid = (lo + hi) * Float32(0.5)
        if _exp_seg(span, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * Float32(0.5)


def sota_hdr_expand_lum_cpu(y: Float32, knee: Float32, b: Float32) -> Float32:
    if y <= knee:
        return y
    return knee + _exp_seg(y - knee, b)


def sota_hdr_lum_scale_cpu(r: Float32, g: Float32, b: Float32, hdr: SotaHdrOutput) -> Float32:
    """Scale factor of the luminance mode for display-linear AgX rgb (0..1)."""
    var a = hdr.crosstalk_clamped()
    var sum = r + g + b
    var cr = (Float32(1.0) - Float32(3.0) * a) * r + a * sum
    var cg = (Float32(1.0) - Float32(3.0) * a) * g + a * sum
    var cb = (Float32(1.0) - Float32(3.0) * a) * b + a * sum
    var yc = Float32(0.2126) * cr + Float32(0.7152) * cg + Float32(0.0722) * cb
    if yc <= Float32(1.0e-6):
        return Float32(1.0)
    var sc = sota_hdr_expand_lum_cpu(yc, hdr.knee_clamped(), hdr.exp_b()) / yc
    var m = max(r, max(g, b))
    return max(Float32(1.0), min(sc, hdr.peak_ratio() / max(m, Float32(1.0e-6))))


def sota_hdr_map_nits_cpu(r_srgb: Float32, g_srgb: Float32, b_srgb: Float32, hdr: SotaHdrOutput) -> Tuple[Float32, Float32, Float32]:
    """Display-referred AgX (sRGB encoded, 0..1) -> absolute Rec.709 luminance per channel (nits)."""
    var r = srgb_eotf_cpu(r_srgb)
    var g = srgb_eotf_cpu(g_srgb)
    var b = srgb_eotf_cpu(b_srgb)
    var m = max(r, max(g, b))
    var s = Float32(1.0)
    if hdr.expansion_mode == SOTA_HDR_EXPAND_LUMINANCE:
        s = sota_hdr_lum_scale_cpu(r, g, b, hdr)
    elif m > Float32(1.0e-6):
        s = sota_hdr_expand_cpu(m, hdr.peak_ratio(), hdr.knee_clamped()) / m
    var pw = hdr.paper()
    return (r * s * pw, g * s * pw, b * s * pw)


def rec709_to_rec2020_cpu(r: Float32, g: Float32, b: Float32) -> Tuple[Float32, Float32, Float32]:
    return (
        Float32(0.627404) * r + Float32(0.329283) * g + Float32(0.043313) * b,
        Float32(0.069097) * r + Float32(0.919540) * g + Float32(0.011362) * b,
        Float32(0.016391) * r + Float32(0.088013) * g + Float32(0.895595) * b,
    )


def rec2020_to_rec709_cpu(r: Float32, g: Float32, b: Float32) -> Tuple[Float32, Float32, Float32]:
    return (
        Float32(1.660491) * r - Float32(0.587641) * g - Float32(0.072850) * b,
        Float32(-0.124550) * r + Float32(1.132900) * g - Float32(0.008349) * b,
        Float32(-0.018151) * r - Float32(0.100579) * g + Float32(1.118730) * b,
    )


def pq_word_decode_nits_cpu(word: UInt32) -> Tuple[Float32, Float32, Float32]:
    """A2B10G10R10 PQ word (Rec.2020) -> absolute Rec.2020 luminance per channel (nits)."""
    var r = pq_eotf_cpu(Float32(Int(word & 0x3FF)) / Float32(1023.0))
    var g = pq_eotf_cpu(Float32(Int((word >> 10) & 0x3FF)) / Float32(1023.0))
    var b = pq_eotf_cpu(Float32(Int((word >> 20) & 0x3FF)) / Float32(1023.0))
    return (r, g, b)


def f16_to_f32_cpu(h: UInt32) -> Float32:
    var sign = Float32(-1.0) if (h & 0x8000) != 0 else Float32(1.0)
    var ex = Int((h >> 10) & 0x1F)
    var mant = Float32(Int(h & 0x3FF))
    if ex == 0:
        return sign * mant * Float32(5.960464477539063e-08)
    if ex == 31:
        return sign * Float32(65504.0)
    return sign * pow(Float32(2.0), Float32(ex - 15)) * (Float32(1.0) + mant / Float32(1024.0))


def sota_hdr_metadata_words(hdr: SotaHdrOutput) -> List[UInt32]:
    """`VkHdrMetadataEXT` payload after sType / pNext (12 floats): BT.2020 R, G, B primaries, D65 white,
    max / min mastering luminance, MaxCLL, MaxFALL."""
    var f = List[Float32]()
    f.append(0.708); f.append(0.292)
    f.append(0.170); f.append(0.797)
    f.append(0.131); f.append(0.046)
    f.append(0.3127); f.append(0.3290)
    var peak = hdr.paper() * hdr.peak_ratio()
    f.append(peak)
    f.append(max(Float32(0.0), hdr.min_nits))
    f.append(peak)
    f.append(hdr.paper())
    var out = List[UInt32]()
    for v in f:
        out.append(bitcast[DType.uint32](v))
    return out^


comptime _STRUCTURE_TYPE_HDR_METADATA_EXT: UInt32 = 1000105000


def sota_hdr_update_metadata(mut hdr: SotaHdrOutput, mut vk: VulkanGpu) -> Bool:
    """Sends ST.2086 / CTA-861.3 metadata for the live HDR swapchain once per swapchain and setting
    change (`vkSetHdrMetadataEXT`, `VK_EXT_hdr_metadata`). No-op for SDR, headless or without the extension."""
    if not hdr.enable or vk.swapchain == 0 or not vk.is_hdr or not vk.has_hdr_metadata:
        return False
    var key = UInt64(Int(bitcast[DType.uint32](hdr.paper()))) << 32 | UInt64(Int(bitcast[DType.uint32](hdr.peak_ratio())))
    if hdr.meta_swapchain == Int(vk.swapchain) and hdr.meta_key == key:
        return False
    var fn_set = vk.device_call("vkSetHdrMetadataEXT")
    if fn_set == 0:
        return False
    vk.rewind()
    var meta = vk.scratch(64)
    put32(meta, _STRUCTURE_TYPE_HDR_METADATA_EXT)
    put64(meta + 8, 0)
    var words = sota_hdr_metadata_words(hdr)
    for i in range(len(words)):
        put32(meta + 16 + i * 4, words[i])
    var sc = vk.scratch(8)
    put64(sc, Int(vk.swapchain))
    _ = call4(fn_set, vk.device, 1, sc, meta)
    hdr.meta_swapchain = Int(vk.swapchain)
    hdr.meta_key = key
    hdr.meta_sent += 1
    return True


# ----------------------------------------------------------------------------------------------
# GPU: emitted into the display tonemap (hook) and the scRGB pass
# ----------------------------------------------------------------------------------------------

def _emit_srgb_eotf(mut s: AaaSpirvModule, v: UInt32) -> UInt32:
    var c = s.nclamp(v, s.float(0.0), s.float(1.0))
    var lo = s.fmul(c, s.float(1.0 / 12.92))
    var hi = s.fpow(s.fmul(s.fadd(c, s.float(0.055)), s.float(1.0 / 1.055)), s.float(2.4))
    return s.select_f(s.fgt(c, s.float(0.04045)), hi, lo)


def _emit_pq_oetf_nits(mut s: AaaSpirvModule, nits: UInt32) -> UInt32:
    var y = s.nclamp(s.fmul(nits, s.float(1.0 / 10000.0)), s.float(0.0), s.float(1.0))
    var ym = s.fpow(s.nmax(s.float(1e-9), y), s.float(_PQ_M1))
    var num = s.fma(s.float(_PQ_C2), ym, s.float(_PQ_C1))
    var den = s.fma(s.float(_PQ_C3), ym, s.float(1.0))
    return s.nclamp(s.fpow(s.fdiv(num, den), s.float(_PQ_M2)), s.float(0.0), s.float(1.0))


def _emit_pq_eotf_nits(mut s: AaaSpirvModule, v: UInt32) -> UInt32:
    var e = s.fpow(s.nclamp(v, s.float(0.0), s.float(1.0)), s.float(Float32(1.0) / _PQ_M2))
    var num = s.nmax(s.float(0.0), s.fsub(e, s.float(_PQ_C1)))
    var den = s.nmax(s.float(1e-9), s.fsub(s.float(_PQ_C2), s.fmul(s.float(_PQ_C3), e)))
    return s.fmul(s.float(10000.0), s.fpow(s.fdiv(num, den), s.float(Float32(1.0) / _PQ_M1)))


def sota_hdr_emit_tonemap_hdr(mut s: AaaSpirvModule, disp_mode: UInt32, dr: UInt32, dg: UInt32, db: UInt32, packed_px: UInt32):
    """Tonemap hook: when `disp_mode == SOTA_DISPLAY_AGX_HDR`, overwrite the packed SDR word with the
    HDR-mapped PQ word. `dr/dg/db` are the graded, sRGB-encoded AgX values before 8-bit quantisation."""
    s.begin_if(s.ieq(disp_mode, s.uint(SOTA_DISPLAY_AGX_HDR)))
    var paper = s.push_float(SOTA_HDR_PUSH_PAPER)
    var peak_ratio = s.push_float(SOTA_HDR_PUSH_PEAK_RATIO)
    var knee = s.push_float(SOTA_HDR_PUSH_KNEE)
    var r = _emit_srgb_eotf(s, dr)
    var g = _emit_srgb_eotf(s, dg)
    var b = _emit_srgb_eotf(s, db)
    var m = s.nmax(r, s.nmax(g, b))
    var t = s.fdiv(s.nmax(s.float(0.0), s.fsub(m, knee)), s.nmax(s.float(1e-6), s.fsub(s.float(1.0), knee)))
    var e = s.fma(s.fsub(peak_ratio, s.float(1.0)), s.fmul(t, t), m)
    var scale_max = s.select_f(s.fgt(m, s.float(1e-6)), s.fdiv(e, s.nmax(m, s.float(1e-6))), s.float(1.0))
    # Round 4: luminance-aware (BT.2446-C style) expansion, selected by push word 13.
    var lum_on = s.ine(s.push_word(SOTA_HDR_PUSH_MODE), s.uint(0))
    var xa = s.push_float(SOTA_HDR_PUSH_CROSSTALK)
    var xb = s.push_float(SOTA_HDR_PUSH_EXP_B)
    var xsum = s.fmul(s.fadd(r, s.fadd(g, b)), xa)
    var xk = s.fsub(s.float(1.0), s.fmul(xa, s.float(3.0)))
    var yc = s.fma(s.fma(r, xk, xsum), s.float(0.2126), s.fma(s.fma(g, xk, xsum), s.float(0.7152), s.fmul(s.fma(b, xk, xsum), s.float(0.0722))))
    var yt = s.nmax(s.float(0.0), s.fsub(yc, knee))
    var seg_exp = s.fdiv(s.fsub(s.exp2(s.fmul(s.fmul(xb, yt), s.float(1.4426950408889634))), s.float(1.0)), s.nmax(xb, s.float(1e-4)))
    var seg = s.select_f(s.fgt(xb, s.float(1e-4)), seg_exp, yt)
    var ey = s.select_f(s.fgt(yc, knee), s.fadd(knee, seg), yc)
    var s_lum = s.select_f(s.fgt(yc, s.float(1e-6)), s.fdiv(ey, s.nmax(yc, s.float(1e-6))), s.float(1.0))
    s_lum = s.nmax(s.float(1.0), s.nmin(s_lum, s.fdiv(peak_ratio, s.nmax(m, s.float(1e-6)))))
    var scale = s.fmul(s.select_f(lum_on, s_lum, scale_max), paper)
    var nr = s.fmul(r, scale)
    var ng = s.fmul(g, scale)
    var nb = s.fmul(b, scale)
    var r2020 = s.fma(nr, s.float(0.627404), s.fma(ng, s.float(0.329283), s.fmul(nb, s.float(0.043313))))
    var g2020 = s.fma(nr, s.float(0.069097), s.fma(ng, s.float(0.919540), s.fmul(nb, s.float(0.011362))))
    var b2020 = s.fma(nr, s.float(0.016391), s.fma(ng, s.float(0.088013), s.fmul(nb, s.float(0.895595))))
    var qr = s.umin(s.uint(1023), s.saturate_uint(s.fma(_emit_pq_oetf_nits(s, r2020), s.float(1023.0), s.float(0.5))))
    var qg = s.umin(s.uint(1023), s.saturate_uint(s.fma(_emit_pq_oetf_nits(s, g2020), s.float(1023.0), s.float(0.5))))
    var qb = s.umin(s.uint(1023), s.saturate_uint(s.fma(_emit_pq_oetf_nits(s, b2020), s.float(1023.0), s.float(0.5))))
    s.store(packed_px, s.bit_or(s.uint(0xC0000000), s.bit_or(s.shl(qb, s.uint(20)), s.bit_or(s.shl(qg, s.uint(10)), qr))))
    s.end_if()


def sota_hdr_emit_ui_paper_white(mut s: AaaSpirvModule) -> UInt32:
    """UI overlay hook: paper white in 100-nit units from push word 6 (0 = the legacy 203 nits)."""
    var p = s.push_float(SOTA_HDR_UI_PUSH_PAPER)
    return s.select_f(s.fgt(p, s.float(1.0)), s.fmul(p, s.float(0.01)), s.float(2.03))


def sota_hdr_scrgb_shader() -> List[UInt32]:
    """PQ display word (after the UI) -> extended linear scRGB fp16 (Rec.709, 1.0 = 80 nits).
    Push[0]=w, Push[1]=h, Push[2]=ldr word offset (history), Push[3]=scRGB word offset (post scratch)."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var dx = s.invocation_axis(0)
    var dy = s.invocation_axis(1)
    var w = s.push_word(0)
    var h = s.push_word(1)
    s.return_if(s.logical_or(s.uge(dx, w), s.uge(dy, h)))
    var pix = s.iadd(s.imul(dy, w), dx)
    var word = s.load_word(b_hist, s.iadd(s.push_word(2), pix))
    var inv = s.float(1.0 / 1023.0)
    var r = _emit_pq_eotf_nits(s, s.fmul(s.float_from_uint(s.bit_and(word, s.uint(0x3FF))), inv))
    var g = _emit_pq_eotf_nits(s, s.fmul(s.float_from_uint(s.bit_and(s.shr(word, s.uint(10)), s.uint(0x3FF))), inv))
    var b = _emit_pq_eotf_nits(s, s.fmul(s.float_from_uint(s.bit_and(s.shr(word, s.uint(20)), s.uint(0x3FF))), inv))
    var k = Float32(1.0 / 80.0)
    var r709 = s.fmul(s.fma(r, s.float(1.660491), s.fma(g, s.float(-0.587641), s.fmul(b, s.float(-0.072850)))), s.float(k))
    var g709 = s.fmul(s.fma(r, s.float(-0.124550), s.fma(g, s.float(1.132900), s.fmul(b, s.float(-0.008349)))), s.float(k))
    var b709 = s.fmul(s.fma(r, s.float(-0.018151), s.fma(g, s.float(-0.100579), s.fmul(b, s.float(1.118730)))), s.float(k))
    var dst = s.iadd(s.push_word(3), s.imul(pix, s.uint(2)))
    s.store_word(b_post, dst, s.pack_f16x2(r709, g709))
    s.store_word(b_post, s.iadd(dst, s.uint(1)), s.pack_f16x2(b709, s.float(1.0)))
    return s.finish()


def sota_hdr_record_scrgb(mut vk: VulkanGpu, slot: Int, pipe: Int, pipe_layout: Int, desc_set: Int, out_w: Int, out_h: Int, ldr_off: Int, scrgb_off: Int):
    vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
    var pc = List[UInt32](length=16, fill=0)
    pc[0] = UInt32(out_w)
    pc[1] = UInt32(out_h)
    pc[2] = UInt32(ldr_off)
    pc[3] = UInt32(scrgb_off)
    vk.dispatch_aaa(slot, pipe, pipe_layout, desc_set, pc, (out_w + 7) // 8, (out_h + 7) // 8)
    vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER | PIPELINE_HOST, Int(ACCESS_TRANSFER_READ | ACCESS_HOST_READ))
