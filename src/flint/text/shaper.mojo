"""OpenType Text Shaper & Multi-Font Fallback Chain (`flint.text.shaper`).

Combines:
- UAX #9 BiDi level resolution & per-line visual reordering (`flint.text.bidi`)
- Unicode Arabic contextual joining (`isol`, `init`, `medi`, `fina`)
- OpenType `GSUB` (LookupTypes 1-7 + `GDEF` lookup flags) and `GPOS`/`kern` kerning
- Multi-font `FontFallbackChain` (`List[TrueTypeFont]`) with per-font `units_per_em` / ascent scaling
"""
from std.collections import Dict
from flint.text.utf8 import decode_utf8
from flint.text.ttf import (
    TrueTypeFont,
    GsubGlyph,
    FEAT_LOCL,
    FEAT_CCMP,
    FEAT_ISOL,
    FEAT_INIT,
    FEAT_MEDI,
    FEAT_FINA,
    FEAT_RLIG,
    FEAT_LIGA,
    FEAT_CLIG,
    FEAT_CALT,
    FEAT_DEFAULT,
)
from flint.text.bidi import (
    BIDI_AUTO,
    BIDI_LTR,
    BIDI_RTL,
    BidiResolved,
    BidiRun,
    FORM_ISOL,
    FORM_INIT,
    FORM_MEDI,
    FORM_FINA,
    resolve_bidi,
    reorder_line,
    bidi_mirror,
    compute_arabic_forms,
)


@fieldwise_init
struct ShapedGlyph(Copyable, ImplicitlyCopyable, Movable):
    var codepoint: Int
    var glyph_id: Int
    var font_index: Int
    var cluster: Int
    var x_advance: Float32
    var y_advance: Float32
    var x_offset: Float32
    var y_offset: Float32
    var rtl: Bool


struct ShapedRun(Copyable, Movable):
    var glyphs: List[ShapedGlyph]
    var level: Int
    var script: String
    var width: Float32

    def __init__(out self, level: Int = 0, script: String = "latn"):
        self.glyphs = List[ShapedGlyph]()
        self.level = level
        self.script = script
        self.width = 0.0


struct ShapedLine(Copyable, Movable):
    var start: Int
    var end: Int
    var width: Float32
    var glyphs: List[ShapedGlyph]

    def __init__(out self, start: Int = 0, end: Int = 0, width: Float32 = 0.0):
        self.start = start
        self.end = end
        self.width = width
        self.glyphs = List[ShapedGlyph]()


struct FontFallbackChain(Copyable, Movable):
    var fonts: List[TrueTypeFont]
    var metric_scales: List[Float32]
    var has_baked_primary: Bool
    var baked_bake_px: Float32
    var baked_cp_to_gid: Dict[Int, Int]
    var baked_advances: List[Float32]
    var baked_kerning: Dict[Int, Float32]

    def __init__(out self):
        self.fonts = List[TrueTypeFont]()
        self.metric_scales = List[Float32]()
        self.has_baked_primary = False
        self.baked_bake_px = 1.0
        self.baked_cp_to_gid = Dict[Int, Int]()
        self.baked_advances = List[Float32]()
        self.baked_kerning = Dict[Int, Float32]()

    def set_primary(mut self, var font: TrueTypeFont):
        if len(self.fonts) == 0:
            self.fonts.append(font^)
            self.metric_scales.append(1.0)
        else:
            self.fonts[0] = font^
            self.metric_scales[0] = 1.0

    def set_baked_primary(
        mut self,
        bake_px: Float32,
        var cp_to_gid: Dict[Int, Int],
        var advances: List[Float32],
        var kerning: Dict[Int, Float32],
    ):
        self.has_baked_primary = True
        self.baked_bake_px = bake_px if bake_px > Float32(0.0) else Float32(1.0)
        self.baked_cp_to_gid = cp_to_gid^
        self.baked_advances = advances^
        self.baked_kerning = kerning^

    def add_fallback_scaled(mut self, var font: TrueTypeFont, metric_scale: Float32):
        self.fonts.append(font^)
        self.metric_scales.append(metric_scale)

    def add_fallback(mut self, var font: TrueTypeFont, match_ascent: Bool = False):
        var mscale: Float32 = 1.0
        if match_ascent and len(self.fonts) > 0 and self.fonts[0].valid and font.valid:
            var p_asc = Float32(max(1, self.fonts[0].ascender)) / Float32(max(1, self.fonts[0].units_per_em))
            var f_asc = Float32(max(1, font.ascender)) / Float32(max(1, font.units_per_em))
            if f_asc > Float32(1.0e-5):
                mscale = p_asc / f_asc
        self.fonts.append(font^)
        self.metric_scales.append(mscale)

    def clear_fallbacks(mut self):
        if len(self.fonts) > 1:
            self.fonts.resize(1, TrueTypeFont())
            self.metric_scales.resize(1, Float32(1.0))

    def count(self) -> Int:
        return len(self.fonts)

    def resolve_codepoint(self, code: Int) -> Tuple[Int, Int]:
        """Returns `(font_index, glyph_id)`. Falls back across `fonts[0..N-1]`."""
        for i in range(len(self.fonts)):
            if i == 0 and not self.fonts[0].valid and self.has_baked_primary:
                var found = self.baked_cp_to_gid.get(code)
                if found:
                    return (0, found.value())
                continue
            if not self.fonts[i].valid:
                continue
            var gid = self.fonts[i].glyph_id(code)
            if gid != 0:
                return (i, gid)
        return (0, 0)

    def scale_for(self, font_idx: Int, px: Float32) -> Float32:
        if font_idx < 0 or font_idx >= len(self.fonts):
            return 0.0
        var upem = max(1, self.fonts[font_idx].units_per_em)
        return (px * self.metric_scales[font_idx]) / Float32(upem)


def is_bidi_format_code(code: Int) -> Bool:
    return (
        code == 0x200E
        or code == 0x200F
        or code == 0x061C
        or (code >= 0x202A and code <= 0x202E)
        or (code >= 0x2066 and code <= 0x2069)
        or code == 0xFEFF
    )


def _is_invisible_or_control(code: Int) -> Bool:
    return (
        is_bidi_format_code(code)
        or code == 0x09
        or code == 0x0A
        or code == 0x0D
        or code == 0xAD
        or (code >= 0x200B and code <= 0x200D)
        or code == 0x2060
    )


def _is_script_common_or_inherited(code: Int) -> Bool:
    if code <= 0x40 or (code >= 0x5B and code <= 0x60) or (code >= 0x7B and code <= 0xBF):
        return True
    if (code >= 0x0300 and code <= 0x036F) or (code >= 0x2000 and code <= 0x206F) or (code >= 0x20A0 and code <= 0x20CF):
        return True
    if (code >= 0x2190 and code <= 0x2BFF) or (code >= 0xFE00 and code <= 0xFE0F) or (code >= 0xFE20 and code <= 0xFE2F):
        return True
    return False


def script_of(code: Int) -> String:
    if (code >= 0x0600 and code <= 0x06FF) or (code >= 0x0750 and code <= 0x077F) or (code >= 0x08A0 and code <= 0x08FF) or (code >= 0xFB50 and code <= 0xFDFF) or (code >= 0xFE70 and code <= 0xFEFF):
        return "arab"
    if (code >= 0x0590 and code <= 0x05FF) or (code >= 0xFB1D and code <= 0xFB4F):
        return "hebr"
    if (code >= 0x3040 and code <= 0x30FF) or (code >= 0x31F0 and code <= 0x31FF):
        return "kana"
    if (code >= 0xAC00 and code <= 0xD7AF) or (code >= 0x1100 and code <= 0x11FF):
        return "hang"
    if (code >= 0x4E00 and code <= 0x9FFF) or (code >= 0x3400 and code <= 0x4DBF) or (code >= 0xF900 and code <= 0xFAFF):
        return "hani"
    if code >= 0x0400 and code <= 0x052F:
        return "cyrl"
    if (code >= 0x0370 and code <= 0x03FF) or (code >= 0x1F00 and code <= 0x1FFF):
        return "grek"
    if (code >= 0x0041 and code <= 0x024F) or (code >= 0x1E00 and code <= 0x1EFF):
        return "latn"
    return "DFLT"


struct TextShaper(Copyable, Movable):
    var chain: FontFallbackChain
    var lang_tag: String
    var alt_choice: Int

    def __init__(out self):
        self.chain = FontFallbackChain()
        self.lang_tag = ""
        self.alt_choice = 0

    def __init__(out self, var chain: FontFallbackChain, lang_tag: String = "", alt_choice: Int = 0):
        self.chain = chain^
        self.lang_tag = lang_tag
        self.alt_choice = alt_choice

    def shape_run_slice(
        self,
        codes: List[Int],
        arabic_forms: List[Int],
        start: Int,
        end: Int,
        level: Int,
        px: Float32,
        cluster_offset: Int = 0,
    ) -> List[ShapedRun]:
        """Shape a single BiDi level slice `[start, end)` in logical order, then
        reverse shaped glyphs if `level` is odd (RTL) to produce visual L-to-R runs."""
        var runs = List[ShapedRun]()
        if end <= start or self.chain.count() == 0:
            return runs^

        var rtl = (level & 1) == 1
        # Resolve effective codepoints, font_index, and script per logical position
        var n = end - start
        var eff_codes = List[Int](capacity=n)
        var f_idxs = List[Int](capacity=n)
        var gids = List[Int](capacity=n)
        var scripts = List[String](capacity=n)

        var last_strong_script = String("DFLT")
        for i in range(start, end):
            var c = codes[i]
            var ec = bidi_mirror(c) if rtl else c
            eff_codes.append(ec)
            if is_bidi_format_code(c) or c == 0x0A or c == 0x0D:
                f_idxs.append(-1)
                gids.append(0)
                scripts.append(last_strong_script)
                continue
            var res = self.chain.resolve_codepoint(ec)
            if res[1] == 0 and ec == 0xA0:
                res = self.chain.resolve_codepoint(32)
            f_idxs.append(res[0])
            gids.append(res[1])
            if _is_script_common_or_inherited(ec):
                scripts.append(last_strong_script)
            else:
                var sc = script_of(ec)
                if sc != "DFLT":
                    last_strong_script = sc
                scripts.append(sc)

        # Back-fill leading common scripts with the first strong script in the slice
        var first_strong = String("latn")
        for k in range(n):
            if scripts[k] != "DFLT":
                first_strong = scripts[k]
                break
        for k in range(n):
            if scripts[k] == "DFLT":
                scripts[k] = first_strong

        # Prevent ZWJ/ZWNJ from fragmenting fallback-font GSUB runs
        for k in range(n):
            var c = eff_codes[k]
            if c == 0x200C or c == 0x200D:
                if gids[k] == 0:
                    f_idxs[k] = -1
                else:
                    var nb_fi = -1
                    var p = k - 1
                    while p >= 0:
                        if f_idxs[p] >= 0:
                            nb_fi = f_idxs[p]
                            break
                        p -= 1
                    if nb_fi < 0:
                        var q = k + 1
                        while q < n:
                            if f_idxs[q] >= 0:
                                nb_fi = f_idxs[q]
                                break
                            q += 1
                    if nb_fi >= 0 and nb_fi != f_idxs[k]:
                        var nb_gid = 0
                        if nb_fi == 0 and not self.chain.fonts[0].valid and self.chain.has_baked_primary:
                            var found = self.chain.baked_cp_to_gid.get(c)
                            nb_gid = found.value() if found else 0
                        elif nb_fi < self.chain.count() and self.chain.fonts[nb_fi].valid:
                            nb_gid = self.chain.fonts[nb_fi].glyph_id(c)
                        if nb_gid != 0:
                            f_idxs[k] = nb_fi
                            gids[k] = nb_gid
                        else:
                            f_idxs[k] = -1
                            gids[k] = 0

        # Split into contiguous (font_index, script) runs in logical order
        var sub_runs = List[ShapedRun]()
        var pos = 0
        while pos < n:
            if f_idxs[pos] < 0:
                pos += 1
                continue
            var fi = f_idxs[pos]
            var sc = scripts[pos]
            var r_start = pos
            while pos < n and (f_idxs[pos] < 0 or (f_idxs[pos] == fi and scripts[pos] == sc)):
                pos += 1

            var scale = self.chain.scale_for(fi, px)
            var gsub_list = List[GsubGlyph]()
            var orig_codes_for_cluster = Dict[Int, Int]()
            for k in range(r_start, pos):
                var logical_i = start + k
                var c = eff_codes[k]
                if f_idxs[k] < 0:
                    # ZWNJ inside a run blocks ligatures across it without emitting a glyph
                    if c == 0x200C and len(gsub_list) > 0:
                        gsub_list.append(GsubGlyph(-1, cluster_offset + logical_i, FEAT_DEFAULT, 0))
                    continue
                if (c == 0x200C or c == 0x200D) and gids[k] == 0:
                    continue
                var mask = FEAT_DEFAULT
                if sc == "arab":
                    var form = arabic_forms[logical_i]
                    if form == FORM_INIT:
                        mask |= FEAT_INIT
                    elif form == FORM_MEDI:
                        mask |= FEAT_MEDI
                    elif form == FORM_FINA:
                        mask |= FEAT_FINA
                    else:
                        mask |= FEAT_ISOL
                var cl = cluster_offset + logical_i
                orig_codes_for_cluster[cl] = c
                gsub_list.append(GsubGlyph(gids[k], cl, mask, self.alt_choice))

            if len(gsub_list) == 0:
                continue

            if fi >= 0 and fi < self.chain.count() and self.chain.fonts[fi].valid:
                self.chain.fonts[fi].apply_gsub(gsub_list, sc, self.lang_tag)

            var final_gsub = List[GsubGlyph](capacity=len(gsub_list))
            for g in gsub_list:
                if g.glyph_id >= 0:
                    final_gsub.append(g)

            var srun = ShapedRun(level, sc)
            var m = len(final_gsub)
            var use_baked_primary = (fi == 0 and (fi >= self.chain.count() or not self.chain.fonts[0].valid) and self.chain.has_baked_primary)
            var b_scale = px / max(Float32(1.0), self.chain.baked_bake_px)
            for g_i in range(m):
                var gid = final_gsub[g_i].glyph_id
                var cl = final_gsub[g_i].cluster
                var cp_opt = orig_codes_for_cluster.get(cl)
                var cp = cp_opt.value() if cp_opt else 0
                var adv: Float32 = 0.0
                if use_baked_primary:
                    var ds_idx = gid - 1
                    if ds_idx >= 0 and ds_idx < len(self.chain.baked_advances):
                        adv = self.chain.baked_advances[ds_idx] * b_scale
                    elif gid == 0 and cp > 0 and not _is_invisible_or_control(cp):
                        var fb_opt = self.chain.baked_cp_to_gid.get(63)
                        var fb_gid = fb_opt.value() if fb_opt else 0
                        if fb_gid > 0 and fb_gid - 1 < len(self.chain.baked_advances):
                            adv = self.chain.baked_advances[fb_gid - 1] * b_scale
                        else:
                            var fb_res = self.chain.resolve_codepoint(63)
                            if fb_res[1] != 0 and fb_res[0] > 0 and fb_res[0] < self.chain.count() and self.chain.fonts[fb_res[0]].valid:
                                adv = self.chain.fonts[fb_res[0]].advance(fb_res[1]) * self.chain.scale_for(fb_res[0], px)
                    if g_i + 1 < m:
                        var nxt_cp_opt = orig_codes_for_cluster.get(final_gsub[g_i + 1].cluster)
                        if nxt_cp_opt:
                            var k_opt = self.chain.baked_kerning.get(cp * 65536 + nxt_cp_opt.value())
                            if k_opt:
                                adv += k_opt.value() * b_scale
                else:
                    adv = self.chain.fonts[fi].advance(gid) * scale if (fi >= 0 and fi < self.chain.count()) else Float32(0.0)
                    if g_i + 1 < m and fi >= 0 and fi < self.chain.count():
                        adv += self.chain.fonts[fi].kerning(gid, final_gsub[g_i + 1].glyph_id) * scale
                srun.glyphs.append(ShapedGlyph(cp, gid, fi, cl, adv, 0.0, 0.0, 0.0, rtl))
                srun.width += adv

            if rtl and len(srun.glyphs) > 1:
                var lo = 0
                var hi = len(srun.glyphs) - 1
                while lo < hi:
                    var tmp = srun.glyphs[lo]
                    srun.glyphs[lo] = srun.glyphs[hi]
                    srun.glyphs[hi] = tmp
                    lo += 1
                    hi -= 1
            sub_runs.append(srun^)

        if rtl and len(sub_runs) > 1:
            var lo = 0
            var hi = len(sub_runs) - 1
            while lo < hi:
                var tmp = sub_runs[lo].copy()
                sub_runs[lo] = sub_runs[hi].copy()
                sub_runs[hi] = tmp^
                lo += 1
                hi -= 1
        for sr in sub_runs:
            runs.append(sr.copy())
        return runs^

    def shape_runs(
        self,
        text: String,
        px: Float32,
        base_dir: Int = BIDI_AUTO,
    ) -> List[ShapedRun]:
        var codes = decode_utf8(text)
        return self.shape_codes_runs(codes, px, base_dir)

    def shape_codes_runs(
        self,
        codes: List[Int],
        px: Float32,
        base_dir: Int = BIDI_AUTO,
    ) -> List[ShapedRun]:
        var out = List[ShapedRun]()
        if len(codes) == 0 or self.chain.count() == 0:
            return out^
        var resolved = resolve_bidi(codes, base_dir)
        var arabic_forms = compute_arabic_forms(codes)
        var bidi_runs = reorder_line(resolved, 0, len(codes))
        for br in bidi_runs:
            var sr_list = self.shape_run_slice(codes, arabic_forms, br.start, br.end, br.level, px)
            for sr in sr_list:
                out.append(sr.copy())
        return out^

    def shape_text(
        self,
        text: String,
        px: Float32,
        base_dir: Int = BIDI_AUTO,
    ) -> List[ShapedGlyph]:
        var glyphs = List[ShapedGlyph]()
        var runs = self.shape_runs(text, px, base_dir)
        for r in runs:
            for g in r.glyphs:
                glyphs.append(g)
        return glyphs^


def shape_text(
    chain: FontFallbackChain,
    text: String,
    px: Float32,
    base_dir: Int = BIDI_AUTO,
    lang_tag: String = "",
    alt_choice: Int = 0,
) -> List[ShapedGlyph]:
    var shaper = TextShaper(chain.copy(), lang_tag, alt_choice)
    return shaper.shape_text(text, px, base_dir)
