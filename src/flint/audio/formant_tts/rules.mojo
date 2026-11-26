"""Synthesis by rule: misaki phonemes -> segments -> 5 ms Klatt parameter tracks
(`flint.audio.formant_tts.rules`).

Pipeline:
1. `parse_phonemes`: misaki string -> segments with stress, word and pause marks; nucleus merging
   (ɜɹ ɪɹ ɛɹ ɑɹ ɔɹ ʊɹ, ᵊl / ᵊn / ᵊm and word-final C+l / C+n -> syllabic EL / EN / EM), light / dark
   l, postvocalic RX, glottalised t (Klatt 1979 phonological rules; phonetics note §0, §5).
2. `_durations`: Klatt 1979 / MITalk rules 1-11 (`DUR = MIN' + (INH - MIN') PRCNT/100`) plus the
   prosody note's rate scaling and punctuation pauses.
3. `_stops`: release / unreleased, VOT and aspiration (Lisker & Abramson 1964, Klatt 1975 / 1987),
   voice bars, affricate closure + frication, stop + sonorant devoicing, tr / dr affrication.
4. Formant tracks: per-segment targets with Holmes-Mattingly-Shearme boundary values
   (`F_b = L + k (F_v - L)`, locus rows from the phonetics note §4.1), smoothstep transitions whose
   deviations add (so short vowels undershoot, Lindblom 1963), reduction and schwa colouring,
   Öhman V-C-V anticipation, fronted /u/, nasalisation, then a zero-phase binomial smoother.
5. F0: Pierrehumbert-style tone targets on a declining baseline (Pierrehumbert 1981; downstep after
   Liberman & Pierrehumbert 1984), sentence types from punctuation, zero-phase smoothing
   (tau 30 ms), microprosody (Lehiste & Peterson 1961; Hombert 1978; Whalen & Levitt 1995).
"""
from std.math import sqrt
from flint.text.utf8 import decode_utf8
from flint.audio.formant_tts.phonemes import *
from flint.audio.formant_tts.synth import *
from flint.audio.formant_tts.voices import FormantVoice

comptime R_NOM: Float32 = 178.0       # nominal rate the Klatt 1979 durations represent (wpm, calibrated)
comptime LEAD_MS: Float32 = 30.0
comptime TAIL_MS: Float32 = 90.0

# word flags (per codepoint of the phoneme string, from the G2P word list)
comptime WF_FUNCTION: Int = 1
comptime WF_WH: Int = 2
comptime WF_EMPH: Int = 4

# phrase types
comptime PT_DECL: Int = 0
comptime PT_YN: Int = 1
comptime PT_WH: Int = 2
comptime PT_EXCL: Int = 3
comptime PT_CONT: Int = 4
comptime PT_INTER: Int = 5
comptime PT_PAREN: Int = 6
comptime PT_SUSP: Int = 7
comptime PT_LIST: Int = 8


struct Seg(Copyable, Movable):
    var ph: Int
    var stress: Int        # 0 none, 1 primary, 2 secondary
    var word: Int
    var punct: Int         # pauses: punctuation codepoint
    var cp: Int            # codepoint index in the phoneme string
    var lengthen: Bool
    var nuc: Int           # nucleus segment of this segment's syllable (-1 none)
    var dur: Float32       # ms
    var start: Float32     # ms
    var released: Bool
    var voicebar: Bool
    var accent: Float32
    var phrase: Int

    def __init__(out self, ph: Int, stress: Int, word: Int, cp: Int):
        self.ph = ph
        self.stress = stress
        self.word = word
        self.punct = 0
        self.cp = cp
        self.lengthen = False
        self.nuc = -1
        self.dur = 0.0
        self.start = 0.0
        self.released = True
        self.voicebar = False
        self.accent = 0.0
        self.phrase = 0


@fieldwise_init
struct Interval(Copyable, Movable):
    """Aspiration / short-lag interval on the absolute timeline (ms)."""

    var t0: Float32
    var t1: Float32
    var ah: Float32       # aspiration dB (0 = silent gap)
    var fric: Bool        # tr / dr affrication: add ʃ-like frication


@fieldwise_init
struct Phrase(Copyable, Movable):
    var first: Int        # segment range [first, last]
    var last: Int
    var kind: Int
    var sentence_first: Bool


@always_inline
def _smooth(x: Float32) -> Float32:
    var t = max(Float32(0.0), min(Float32(1.0), x))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


def _is_punct(c: Int) -> Bool:
    return c == 44 or c == 46 or c == 33 or c == 63 or c == 59 or c == 58 or c == 0x2014 or c == 0x2026


# ============================================================================ 1. parsing
def parse_phonemes(ps: String, tab: List[PhonemeDef]) -> List[Seg]:
    var codes = decode_utf8(ps)
    var raw = List[Seg]()
    var word = 0
    var in_word = False
    var pending = 0
    for i in range(len(codes)):
        var c = codes[i]
        if c == 32 or c == 9 or c == 10:
            if in_word:
                word += 1
                in_word = False
            continue
        if c == 0x2C8:
            pending = 1
            continue
        if c == 0x2CC:
            pending = 2
            continue
        if c == 0x2D0:  # ː
            if len(raw) > 0:
                raw[len(raw) - 1].lengthen = True
            continue
        if _is_punct(c):
            if in_word:
                word += 1
                in_word = False
            # collapse runs of punctuation into one pause (strongest wins)
            if len(raw) > 0 and raw[len(raw) - 1].ph == PH_PAU:
                if c == 46 or c == 33 or c == 63:
                    raw[len(raw) - 1].punct = c
                continue
            var p = Seg(PH_PAU, 0, -1, i)
            p.punct = c
            raw.append(p^)
            continue
        var ph = phoneme_of_codepoint(c)
        if ph < 0:
            continue
        var st = 0
        if tab[ph].has(FL_VOWEL):
            st = pending
            pending = 0
        raw.append(Seg(ph, st, word, i))
        in_word = True
    # ---- merging and allophones
    var out = List[Seg]()
    var n = len(raw)
    var i = 0
    while i < n:
        var s = raw[i].copy()
        var ph = s.ph
        var nx = raw[i + 1].ph if i + 1 < n else PH_PAU
        var same_next = i + 1 < n and raw[i + 1].word == s.word and s.word >= 0
        var nx2_vowel = i + 2 < n and raw[i + 2].word == s.word and tab[raw[i + 2].ph].has(FL_VOWEL)
        # vowel + postvocalic ɹ -> r-coloured nucleus
        if tab[ph].has(FL_VOWEL) and same_next and nx == PH_R and not nx2_vowel:
            var m = -1
            if ph == PH_ERGB or ph == PH_ER:
                m = PH_ER
            elif ph == PH_AX or ph == PH_SCHWA or ph == PH_AXR or ph == PH_AXT:
                m = PH_AXR
            elif ph == PH_IH or ph == PH_I:
                m = PH_IR
            elif ph == PH_EH or ph == PH_E or ph == PH_AE:
                m = PH_EHR
            elif ph == PH_AA or ph == PH_A:
                m = PH_AR
            elif ph == PH_AO or ph == PH_OW or ph == PH_O or ph == PH_OH:
                m = PH_OR
            elif ph == PH_UH or ph == PH_U:
                m = PH_UR
            if m >= 0:
                s.ph = m
                out.append(s^)
                i += 2
                continue
        # ᵊ + l / n / m -> syllabic consonant
        if ph == PH_SCHWA and same_next and (nx == PH_L or nx == PH_N or nx == PH_M) and not nx2_vowel:
            var m2 = PH_EL if nx == PH_L else (PH_EN if nx == PH_N else PH_EM)
            var t = raw[i + 1].copy()
            t.ph = m2
            t.stress = 0
            out.append(t^)
            i += 2
            continue
        # word-final C + l / n (misaki "bˈʌtn", "lˈɪtl") -> syllabic
        if (ph == PH_L or ph == PH_N) and len(out) > 0 and out[len(out) - 1].word == s.word:
            var pv = out[len(out) - 1].ph
            var obstr = tab[pv].has(FL_STOP) or tab[pv].has(FL_FRIC) or pv == PH_DX
            if obstr and not (same_next and tab[nx].has(FL_VOWEL)):
                s.ph = PH_EL if ph == PH_L else PH_EN
                if pv == PH_DX:
                    out[len(out) - 1].ph = PH_T   # "button": t closure, nasal release
                out.append(s^)
                i += 1
                continue
        out.append(s^)
        i += 1
    # second pass: l -> LX, ɹ -> RX, t -> TQ
    var m_n = len(out)
    for j in range(m_n):
        var ph = out[j].ph
        var w = out[j].word
        var prev_v = j > 0 and out[j - 1].word == w and tab[out[j - 1].ph].has(FL_VOWEL)
        if ph == PH_L:
            # Klatt 1979 rule 1: dark after a vowel unless a stressed vowel follows in the word
            if prev_v:
                var stressed_follows = False
                var q = j + 1
                while q < m_n and out[q].word == w:
                    if tab[out[q].ph].has(FL_VOWEL):
                        stressed_follows = out[q].stress == 1
                        break
                    q += 1
                if not stressed_follows:
                    out[j].ph = PH_LX
            elif j + 1 >= m_n or out[j + 1].word != w or not tab[out[j + 1].ph].has(FL_VOWEL):
                if j > 0 and out[j - 1].word == w:
                    out[j].ph = PH_LX
        elif ph == PH_R:
            var next_v = j + 1 < m_n and out[j + 1].word == w and tab[out[j + 1].ph].has(FL_VOWEL)
            if prev_v and not next_v:
                out[j].ph = PH_RX
        elif ph == PH_T:
            # glottalised word-final t before a sonorant-initial word (K79 rule 3)
            var final_t = j + 1 < m_n and out[j + 1].word != w and out[j + 1].ph != PH_PAU
            if final_t and j > 0 and out[j - 1].word == w and tab[out[j - 1].ph].has(FL_SON):
                ref nf = tab[out[j + 1].ph]
                if nf.has(FL_NASAL) or nf.has(FL_GLIDE) or nf.has(FL_LIQ):
                    out[j].ph = PH_TQ
    return out^


# ============================================================================ helpers
@always_inline
def _syl(tab: List[PhonemeDef], ph: Int) -> Bool:
    return tab[ph].has(FL_VOWEL) or tab[ph].has(FL_SYLL)


def _assign_nuclei(mut segs: List[Seg], tab: List[PhonemeDef]):
    """Each consonant joins a syllable nucleus: word-initial / onset consonants the next nucleus,
    the rest the previous one (maximal onset for obstruent + liquid / glide and s + stop)."""
    var n = len(segs)
    var i = 0
    while i < n:
        if segs[i].ph == PH_PAU:
            i += 1
            continue
        var w = segs[i].word
        var e = i
        while e < n and segs[e].word == w and segs[e].ph != PH_PAU:
            e += 1
        # nuclei in [i, e)
        var nucs = List[Int]()
        for q in range(i, e):
            if _syl(tab, segs[q].ph):
                nucs.append(q)
        if len(nucs) == 0:
            for q in range(i, e):
                segs[q].nuc = -1
            i = e
            continue
        for q in range(i, e):
            if _syl(tab, segs[q].ph):
                segs[q].nuc = q
        # before first nucleus
        for q in range(i, nucs[0]):
            segs[q].nuc = nucs[0]
        # after last
        for q in range(nucs[len(nucs) - 1] + 1, e):
            segs[q].nuc = nucs[len(nucs) - 1]
        # between nuclei
        for k in range(len(nucs) - 1):
            var a = nucs[k]
            var b = nucs[k + 1]
            var split = b - 1   # consonants >= split go to b
            if b - a >= 3:
                var c1 = segs[b - 2].ph
                var c2 = segs[b - 1].ph
                var onset2 = ((tab[c1].has(FL_STOP) or tab[c1].has(FL_FRIC)) and (tab[c2].has(FL_LIQ) or tab[c2].has(FL_GLIDE))) or (c1 == PH_S and tab[c2].has(FL_STOP))
                if onset2:
                    split = b - 2
            for q in range(a + 1, b):
                segs[q].nuc = b if q >= split else a
        i = e


def _word_bounds(segs: List[Seg], i: Int) -> Tuple[Int, Int]:
    var w = segs[i].word
    var a = i
    while a > 0 and segs[a - 1].word == w and segs[a - 1].ph != PH_PAU:
        a -= 1
    var b = i
    while b + 1 < len(segs) and segs[b + 1].word == w and segs[b + 1].ph != PH_PAU:
        b += 1
    return (a, b)


def _phrases(segs: List[Seg], wflags: List[Int]) -> List[Phrase]:
    var out = List[Phrase]()
    var n = len(segs)
    var i = 0
    var sent_first = True
    while i < n:
        if segs[i].ph == PH_PAU:
            i += 1
            continue
        var a = i
        while i < n and segs[i].ph != PH_PAU:
            i += 1
        var b = i - 1
        var p = segs[i].punct if i < n else 46
        var kind = PT_DECL
        if p == 63:
            kind = PT_YN
            # wh-question: first word of the sentence is a wh-word
            var first_cp = -1
            # find sentence start: walk back over phrases until a sentence end
            var q = a
            while q > 0:
                var pp = segs[q - 1].punct if segs[q - 1].ph == PH_PAU else 0
                if pp == 46 or pp == 33 or pp == 63:
                    break
                q -= 1
            while q < n and segs[q].ph == PH_PAU:
                q += 1
            if q < n:
                first_cp = segs[q].cp
            if first_cp >= 0 and first_cp < len(wflags) and (wflags[first_cp] & WF_WH) != 0:
                kind = PT_WH
        elif p == 33:
            kind = PT_EXCL
        elif p == 44:
            kind = PT_CONT
        elif p == 59 or p == 58:
            kind = PT_INTER
        elif p == 0x2014:
            kind = PT_PAREN
        elif p == 0x2026:
            kind = PT_SUSP
        out.append(Phrase(a, b, kind, sent_first))
        sent_first = p == 46 or p == 33 or p == 63
    # lists: comma phrases of <= 3 words followed by more commas or an "and/or" phrase
    var np = len(out)
    for k in range(np):
        if out[k].kind == PT_CONT:
            var nw = segs[out[k].last].word - segs[out[k].first].word + 1
            var next_short = k + 1 < np and (segs[out[k + 1].last].word - segs[out[k + 1].first].word + 1) <= 3
            var prev_list = k > 0 and out[k - 1].kind == PT_LIST
            if nw <= 3 and (next_short or prev_list):
                out[k].kind = PT_LIST
    return out^


# ============================================================================ 2. durations
def _durations(mut segs: List[Seg], tab: List[PhonemeDef], phrases: List[Phrase], wflags: List[Int], voice: FormantVoice):
    var n = len(segs)
    var r = max(Float32(0.6), min(Float32(1.6), R_NOM / max(Float32(100.0), min(Float32(280.0), voice.rate_wpm))))
    # phrase-final nuclei and phrase kinds per segment
    var pf_nuc = List[Int](length=n, fill=-1)
    var kind_of = List[Int](length=n, fill=PT_DECL)
    var last_phrase = List[Bool](length=n, fill=False)
    for pi in range(len(phrases)):
        var a = phrases[pi].first
        var b = phrases[pi].last
        var last_nuc = -1
        for q in range(a, b + 1):
            if _syl(tab, segs[q].ph):
                last_nuc = q
        for q in range(a, b + 1):
            pf_nuc[q] = last_nuc
            kind_of[q] = phrases[pi].kind
            segs[q].phrase = pi
            last_phrase[q] = pi == len(phrases) - 1
    for i in range(n):
        ref d = tab[segs[i].ph]
        if segs[i].ph == PH_PAU:
            var pd = Float32(200.0)
            var pc = segs[i].punct
            if pc == 59 or pc == 58:
                pd = 300.0
            elif pc == 0x2014:
                pd = 220.0
            elif pc == 0x2026:
                pd = 450.0
            elif pc == 46 or pc == 33 or pc == 63:
                pd = 500.0
            var is_last = True
            for q in range(i + 1, n):
                if segs[q].ph != PH_PAU:
                    is_last = False
                    break
            if is_last or i == 0:
                pd = 0.0
            segs[i].dur = pd * r * sqrt(r)
            continue
        var inh = d.inh
        var mn = d.mn
        if segs[i].lengthen:
            inh *= 1.4
        var syl = _syl(tab, segs[i].ph)
        var nuc = segs[i].nuc
        var stressed = nuc >= 0 and segs[nuc].stress > 0
        if syl:
            stressed = segs[i].stress > 0
        var wb = _word_bounds(segs, i)
        var wa = wb[0]
        var wz = wb[1]
        var nsyl = 0
        var first_nuc = -1
        var last_nuc_w = -1
        for q in range(wa, wz + 1):
            if _syl(tab, segs[q].ph):
                nsyl += 1
                if first_nuc < 0:
                    first_nuc = q
                last_nuc_w = q
        var pfs = nuc >= 0 and nuc == pf_nuc[i]
        var wfs = nuc >= 0 and nuc == last_nuc_w
        var prc = Float32(1.0)
        # rule 2 / 3: phrase-final lengthening, non-final shortening
        if syl:
            prc *= Float32(1.4) if pfs else Float32(0.6)
        elif pfs and i > nuc and (d.has(FL_LIQ) or d.has(FL_NASAL)):
            prc *= 1.4
        # rule 4: non-word-final syllabics
        if syl and not wfs:
            prc *= 0.85
        # rule 5: polysyllabic shortening
        if syl and nsyl > 1:
            prc *= 0.8
        # rule 6: non-initial consonants
        if not syl and i != wa:
            prc *= 0.85
        # rule 7: unstressed shortening
        if not stressed:
            mn *= 0.5
            if syl and d.has(FL_VOWEL):
                var medial = nsyl > 2 and i != first_nuc and i != last_nuc_w
                prc *= Float32(0.5) if medial else Float32(0.7)
            elif (d.has(FL_LIQ) or d.has(FL_GLIDE)) and i + 1 < n and tab[segs[i + 1].ph].has(FL_VOWEL) and segs[i + 1].word == segs[i].word:
                prc *= 0.1
            else:
                prc *= 0.7
        # rule 8: emphasis (nuclear vowel of an exclamation, or flagged words)
        if syl and d.has(FL_VOWEL):
            var emph = segs[i].cp < len(wflags) and (wflags[segs[i].cp] & WF_EMPH) != 0
            if kind_of[i] == PT_EXCL and pfs:
                emph = True
            if emph:
                prc *= 1.4
        # rule 9: postvocalic context
        if syl:
            var f = Float32(1.0)
            if i == wz:
                f = 1.2
            else:
                ref c = tab[segs[i + 1].ph]
                if c.has(FL_FRIC) and c.has(FL_VOICED):
                    f = 1.6
                elif c.has(FL_STOP) and (c.has(FL_VOICED)):
                    f = 1.2
                elif segs[i + 1].ph == PH_DX:
                    f = 1.2
                elif c.has(FL_NASAL):
                    f = 0.85
                elif c.has(FL_STOP):
                    f = 0.7
            if not pfs:
                f = Float32(1.0) + Float32(0.5) * (f - Float32(1.0))
            prc *= f
        # rule 10: cluster shortening (across words, broken by pauses)
        var pv = i > 0 and segs[i - 1].ph != PH_PAU
        var nv = i + 1 < n and segs[i + 1].ph != PH_PAU
        var pv_cons = pv and not _syl(tab, segs[i - 1].ph)
        var nv_cons = nv and not _syl(tab, segs[i + 1].ph)
        if syl and d.has(FL_VOWEL):
            if nv and tab[segs[i + 1].ph].has(FL_VOWEL):
                prc *= 1.2
            if pv and tab[segs[i - 1].ph].has(FL_VOWEL):
                prc *= 0.7
        elif not syl:
            if pv_cons and nv_cons:
                prc *= 0.5
            elif nv_cons:
                prc *= 0.7
            elif pv_cons:
                prc *= 0.7
        # question / suspension: final syllable longer
        if pfs and (kind_of[i] == PT_YN) and last_phrase[i]:
            prc *= 1.15
        if pfs and kind_of[i] == PT_SUSP:
            prc *= 1.3
        # rules 2 x 9 compound to 2.2x on a phrase-final vowel before a voiced fricative ("five.");
        # beyond 1.5x the vowel only sounds drawled (tuning log, it. 8)
        if syl:
            prc = min(prc, Float32(1.5))
        var dur = mn + (inh - mn) * prc
        # rate scaling (prosody note §1.4): compressible part scales fully
        dur = mn * (Float32(0.6) + Float32(0.4) * r) + (dur - mn) * r
        # rule 11: post-aspiration lengthening (+25 ms) of stressed vowels / sonorants
        if stressed and (d.has(FL_SON)) and i > 0:
            var pp = segs[i - 1].ph
            var after_s = i > 1 and segs[i - 2].ph == PH_S and segs[i - 2].word == segs[i - 1].word
            if (pp == PH_P or pp == PH_T or pp == PH_K) and not after_s and segs[i - 1].word >= 0:
                dur += 25.0
        segs[i].dur = max(Float32(10.0), dur)


# ============================================================================ 3. timeline
def _timeline(mut segs: List[Seg]) -> Float32:
    var t = LEAD_MS
    for i in range(len(segs)):
        segs[i].start = t
        t += segs[i].dur
    return t


@always_inline
def _end(segs: List[Seg], i: Int) -> Float32:
    return segs[i].start + segs[i].dur


# ============================================================================ 4. targets
struct Targets(Copyable, Movable):
    """Per-segment formant targets after reduction / colouring; s = onset, e = end (diphthongs)."""

    var f1s: Float32
    var f2s: Float32
    var f3s: Float32
    var b1s: Float32
    var b2s: Float32
    var b3s: Float32
    var f1e: Float32
    var f2e: Float32
    var f3e: Float32
    var b1e: Float32
    var b2e: Float32
    var b3e: Float32

    def __init__(out self, d: PhonemeDef):
        self.f1s = d.f1
        self.f2s = d.f2
        self.f3s = d.f3
        self.b1s = d.b1
        self.b2s = d.b2
        self.b3s = d.b3
        self.f1e = d.f1e
        self.f2e = d.f2e
        self.f3e = d.f3e
        self.b1e = d.b1e
        self.b2e = d.b2e
        self.b3e = d.b3e

    def get_s(self, p: Int) -> Float32:
        if p == 0:
            return self.f1s
        if p == 1:
            return self.f2s
        if p == 2:
            return self.f3s
        if p == 3:
            return self.b1s
        if p == 4:
            return self.b2s
        return self.b3s

    def get_e(self, p: Int) -> Float32:
        if p == 0:
            return self.f1e
        if p == 1:
            return self.f2e
        if p == 2:
            return self.f3e
        if p == 3:
            return self.b1e
        if p == 4:
            return self.b2e
        return self.b3e


def _targets(segs: List[Seg], tab: List[PhonemeDef], loci: List[Locus]) -> List[Targets]:
    var n = len(segs)
    var out = List[Targets]()
    for i in range(n):
        out.append(Targets(tab[segs[i].ph]))
    ref schwa = tab[PH_AX]
    for i in range(n):
        var ph = segs[i].ph
        ref d = tab[ph]
        if d.has(FL_VOWEL):
            var full = not d.has(FL_REDUCED)
            # stress-driven centralisation (phonetics §4.3)
            if full and segs[i].stress != 1:
                var rr = Float32(0.25) if segs[i].stress == 0 else Float32(0.1)
                var re = Float32(0.35) if segs[i].stress == 0 else Float32(0.1)
                out[i].f1s += rr * (schwa.f1 - out[i].f1s)
                out[i].f2s += rr * (schwa.f2 - out[i].f2s)
                out[i].f3s += rr * (schwa.f3 - out[i].f3s)
                if d.has(FL_DIPH):
                    # unstressed diphthongs undershoot their offglide (end = start + 0.7 (end - start))
                    out[i].f1e = out[i].f1s + Float32(0.7) * (out[i].f1e - out[i].f1s)
                    out[i].f2e = out[i].f2s + Float32(0.7) * (out[i].f2e - out[i].f2s)
                    out[i].f3e = out[i].f3s + Float32(0.7) * (out[i].f3e - out[i].f3s)
                    out[i].f2e += re * (schwa.f2 - out[i].f2e) * Float32(0.5)
                else:
                    out[i].f1e = out[i].f1s
                    out[i].f2e = out[i].f2s
                    out[i].f3e = out[i].f3s
            # reduced vowels take colour from the neighbouring consonants (§4.4)
            if d.has(FL_REDUCED) and ph != PH_AXR:
                var sum2 = Float32(0.0)
                var sum3 = Float32(0.0)
                var cnt = 0
                if i > 0 and not tab[segs[i - 1].ph].has(FL_VOWEL) and segs[i - 1].ph != PH_PAU and segs[i - 1].ph != PH_HH:
                    var pl = tab[segs[i - 1].ph].place
                    sum2 += loci[pl].f2 if loci[pl].f2 > 0 else tab[segs[i - 1].ph].f2
                    sum3 += loci[pl].f3 if loci[pl].f3 > 0 else tab[segs[i - 1].ph].f3
                    cnt += 1
                if i + 1 < n and not tab[segs[i + 1].ph].has(FL_VOWEL) and segs[i + 1].ph != PH_PAU and segs[i + 1].ph != PH_HH:
                    var pl2 = tab[segs[i + 1].ph].place
                    sum2 += loci[pl2].f2 if loci[pl2].f2 > 0 else tab[segs[i + 1].ph].f2
                    sum3 += loci[pl2].f3 if loci[pl2].f3 > 0 else tab[segs[i + 1].ph].f3
                    cnt += 1
                if cnt > 0:
                    var m2 = sum2 / Float32(cnt)
                    var m3 = sum3 / Float32(cnt)
                    var f2 = max(Float32(1100.0), min(Float32(1800.0), Float32(0.4) * d.f2 + Float32(0.6) * m2))
                    var f3 = max(Float32(2200.0), min(Float32(2700.0), Float32(0.4) * d.f3 + Float32(0.6) * m3))
                    if ph == PH_IX:
                        f2 = min(Float32(1900.0), f2 + 100.0)
                    out[i].f2s = f2
                    out[i].f2e = f2
                    out[i].f3s = f3
                    out[i].f3e = f3
                if i + 1 < n and (segs[i + 1].ph == PH_LX or segs[i + 1].ph == PH_EL):
                    out[i].f2s = min(out[i].f2s, Float32(1100.0))
                    out[i].f2e = 1000.0
            # fronted /u/ after coronals (§5.7)
            if (ph == PH_U or ph == PH_UH or ph == PH_UR) and i > 0:
                var pl3 = tab[segs[i - 1].ph].place
                if pl3 == PL_ALVEOLAR or pl3 == PL_POSTALV or pl3 == PL_DENTAL or pl3 == PL_PALATAL:
                    out[i].f2s = 1600.0
                    if ph == PH_U:
                        out[i].f2e = 1150.0
    # h takes the formants of the following (else preceding) sonorant, with B1 = 300 (K80)
    for i in range(n):
        if segs[i].ph == PH_HH:
            var src = -1
            if i + 1 < n and tab[segs[i + 1].ph].has(FL_SON):
                src = i + 1
            elif i > 0 and tab[segs[i - 1].ph].has(FL_SON):
                src = i - 1
            if src >= 0:
                var f1 = out[src].f1s if src > i else out[src].f1e
                var f2 = out[src].f2s if src > i else out[src].f2e
                var f3 = out[src].f3s if src > i else out[src].f3e
                out[i].f1s = f1
                out[i].f1e = f1
                out[i].f2s = f2
                out[i].f2e = f2
                out[i].f3s = f3
                out[i].f3e = f3
                out[i].b2s = out[src].b2s
                out[i].b2e = out[src].b2s
                out[i].b3s = out[src].b3s
                out[i].b3e = out[src].b3s
            out[i].b1s = 300.0
            out[i].b1e = 300.0
        elif segs[i].ph == PH_Q or segs[i].ph == PH_TQ or segs[i].ph == PH_PAU:
            # glottal closures / pauses: hold the neighbouring vowel shape
            var src2 = i - 1 if i > 0 and segs[i - 1].ph != PH_PAU else (i + 1 if i + 1 < n else -1)
            if src2 >= 0 and segs[i].ph != PH_TQ:
                out[i].f1s = out[src2].f1e
                out[i].f1e = out[src2].f1e
                out[i].f2s = out[src2].f2e
                out[i].f2e = out[src2].f2e
                out[i].f3s = out[src2].f3e
                out[i].f3e = out[src2].f3e
    return out^


@always_inline
def _tgt_at(tg: Targets, d: PhonemeDef, p: Int, frac: Float32) -> Float32:
    var a = tg.get_s(p)
    var b = tg.get_e(p)
    if a == b:
        return a
    var t0 = d.glide_t0
    var t1 = d.glide_t1
    var x = (frac - t0) / max(Float32(1e-3), t1 - t0)
    if t0 <= Float32(0.0) and t1 >= Float32(1.0):
        x = frac
        return a + (b - a) * max(Float32(0.0), min(Float32(1.0), x))
    return a + (b - a) * _smooth(x)


struct Boundary(Copyable, Movable):
    var bv: List[Float32]     # F1 F2 F3 B1 B2 B3
    var tl: List[Float32]     # transition inside the left segment (ms) per parameter
    var tr: List[Float32]     # inside the right segment

    def __init__(out self):
        self.bv = List[Float32](length=6, fill=0.0)
        self.tl = List[Float32](length=6, fill=0.0)
        self.tr = List[Float32](length=6, fill=0.0)


def _cons_side(d: PhonemeDef, dur: Float32) -> Float32:
    """How long the consonant side of a boundary transition lasts inside the consonant."""
    if d.has(FL_STOP) and not d.has(FL_AFFR):
        return dur
    if d.has(FL_AFFR):
        return dur * Float32(0.5)
    if d.has(FL_NASAL):
        return min(dur, Float32(15.0))
    if d.has(FL_FRIC):
        return min(dur, Float32(25.0))
    if d.has(FL_FLAP):
        return dur * Float32(0.5)
    return min(dur, Float32(30.0))


def _velar_locus(f2v: Float32) -> Tuple[Float32, Float32]:
    """Velar F2 / F3 locus with the velar pinch (F2 and F3 converge at release, [KP82; SMM91]): front
    vowels 2300 / 2300, back vowels 1500 / 2100, linear between."""
    var x = max(Float32(0.0), min(Float32(1.0), (f2v - Float32(1400.0)) / Float32(400.0)))
    return (Float32(1500.0) + x * Float32(800.0), Float32(2100.0) + x * Float32(200.0))


def _boundaries(segs: List[Seg], tab: List[PhonemeDef], loci: List[Locus], tg: List[Targets]) -> List[Boundary]:
    """Boundary j sits between segment j - 1 and j (j = 0 is the utterance start)."""
    var n = len(segs)
    var out = List[Boundary]()
    out.append(Boundary())
    for j in range(1, n):
        var b = Boundary()
        var li = j - 1
        var ri = j
        ref dl = tab[segs[li].ph]
        ref dr = tab[segs[ri].ph]
        var lp = segs[li].ph == PH_PAU
        var rp = segs[ri].ph == PH_PAU
        var lv = dl.has(FL_VOWEL)
        var rv = dr.has(FL_VOWEL)
        if lp or rp:
            for p in range(6):
                b.bv[p] = tg[ri].get_s(p) if lp else tg[li].get_e(p)
            out.append(b^)
            continue
        if segs[li].ph == PH_HH or segs[ri].ph == PH_HH:
            for p in range(6):
                b.bv[p] = tg[ri].get_s(p) if segs[li].ph == PH_HH else tg[li].get_e(p)
                if p == 3:
                    b.bv[p] = 0.5 * (tg[li].get_e(p) + tg[ri].get_s(p))
                    b.tl[p] = min(segs[li].dur, Float32(20.0))
                    b.tr[p] = min(segs[ri].dur, Float32(20.0))
            out.append(b^)
            continue
        if lv and rv:
            for p in range(6):
                b.bv[p] = 0.5 * (tg[li].get_e(p) + tg[ri].get_s(p))
                b.tl[p] = min(segs[li].dur, Float32(40.0))
                b.tr[p] = min(segs[ri].dur, Float32(40.0))
            out.append(b^)
            continue
        if lv != rv:
            # vowel / consonant boundary: the consonant dominates (locus equation)
            var ci = ri if lv else li
            var vi = li if lv else ri
            ref dc = tab[segs[ci].ph]
            var cv = not lv
            var fv1 = tg[vi].get_s(0) if cv else tg[vi].get_e(0)
            var fv2 = tg[vi].get_s(1) if cv else tg[vi].get_e(1)
            var fv3 = tg[vi].get_s(2) if cv else tg[vi].get_e(2)
            ref lc = loci[dc.place]
            var l2 = lc.f2
            var l3 = lc.f3
            var k2 = lc.k2
            var k3 = lc.k3
            if dc.place == PL_VELAR:
                var vl = _velar_locus(fv2)
                l2 = vl[0]
                l3 = vl[1]
            if dc.place == PL_NONE or dc.place == PL_GLOTTAL:
                l2 = dc.f2
                l3 = dc.f3
                k2 = 0.5
                k3 = 0.5
            if not cv:
                k2 = max(Float32(0.0), k2 - Float32(0.1))
                k3 = max(Float32(0.0), k3 - Float32(0.1))
            var bf2 = l2 + k2 * (fv2 - l2)
            var bf3 = l3 + k3 * (fv3 - l3)
            # Öhman V1-C-V2: the V1 -> C transition anticipates V2's F2
            if not cv and ci + 1 < n and tab[segs[ci + 1].ph].has(FL_VOWEL):
                bf2 += Float32(0.15) * (tg[ci + 1].get_s(1) - fv2)
            # rounding anticipation before w u O ʊ
            var bf1: Float32
            if dc.has(FL_STOP) or dc.has(FL_FRIC):
                var f1l = Float32(250.0) if not dc.has(FL_VOICED) else Float32(200.0)
                bf1 = f1l + Float32(0.2) * (fv1 - f1l)
            elif dc.has(FL_NASAL):
                bf1 = dc.f1 + Float32(0.3) * (fv1 - dc.f1)
            else:
                bf1 = dc.f1
            b.bv[0] = bf1
            b.bv[1] = bf2
            b.bv[2] = bf3
            b.bv[3] = 0.5 * (dc.b1 + (tg[vi].b1s if cv else tg[vi].b1e))
            b.bv[4] = dc.b2
            b.bv[5] = dc.b3
            if dc.has(FL_STOP) or dc.has(FL_FRIC):
                b.bv[3] = tg[vi].b1s if cv else tg[vi].b1e
                b.bv[4] = 0.5 * (dc.b2 + (tg[vi].b2s if cv else tg[vi].b2e))
                b.bv[5] = 0.5 * (dc.b3 + (tg[vi].b3s if cv else tg[vi].b3e))
            var vdur = segs[vi].dur
            var cdur = segs[ci].dur
            var t1v = lc.cv1 if cv else lc.vc
            var t23v = lc.cv23 if cv else lc.vc
            if dc.place == PL_NONE or dc.place == PL_GLOTTAL:
                t1v = 30.0
                t23v = 40.0
            if dc.has(FL_FLAP):
                t1v = 15.0
                t23v = 15.0
            var tc = _cons_side(dc, cdur)
            for p in range(6):
                var tv = t1v if (p == 0 or p == 3) else t23v
                if cv:
                    b.tl[p] = tc
                    b.tr[p] = min(vdur, tv)
                else:
                    b.tl[p] = min(vdur, tv)
                    b.tr[p] = tc
            out.append(b^)
            continue
        # consonant / consonant: meet halfway over <= 30 ms each side
        for p in range(6):
            var a = tg[li].get_e(p)
            var c = tg[ri].get_s(p)
            b.bv[p] = 0.5 * (a + c)
            b.tl[p] = min(segs[li].dur, Float32(30.0))
            b.tr[p] = min(segs[ri].dur, Float32(30.0))
            # closures are silent: let the stop take the whole transition
            if dl.has(FL_STOP) and not dl.has(FL_AFFR):
                b.bv[p] = c
                b.tl[p] = segs[li].dur
                b.tr[p] = 0.0
            elif dr.has(FL_STOP) and not dr.has(FL_AFFR):
                b.bv[p] = a
                b.tl[p] = 0.0
                b.tr[p] = segs[ri].dur
        out.append(b^)
    out.append(Boundary())
    # the final boundary (after the last segment) holds the last target
    for p in range(6):
        out[n].bv[p] = tg[n - 1].get_e(p) if n > 0 else Float32(0.0)
    if n > 0:
        for p in range(6):
            out[0].bv[p] = tg[0].get_s(p)
    return out^


# ============================================================================ 5. F0
@always_inline
def _bl(key: Float32, dp: Float32, t0: Float32, t: Float32) -> Float32:
    """Declining phrase baseline (semitones)."""
    return key - dp * (t - t0) * Float32(0.001)


@fieldwise_init
struct F0Target(Copyable, Movable):
    var t: Float32
    var v: Float32


def _f0_contour(segs: List[Seg], tab: List[PhonemeDef], phrases: List[Phrase], wflags: List[Int],
                voice: FormantVoice, nfr: Int) -> List[Float32]:
    """Semitone contour re F_b per 5 ms frame (before microprosody)."""
    var tg = List[F0Target]()
    var R = voice.range_st
    var prev_end_b = Float32(0.0)
    for pi in range(len(phrases)):
        ref ph = phrases[pi]
        var t0 = segs[ph.first].start
        var t1 = _end(segs, ph.last)
        var T = max(Float32(0.3), (t1 - t0) * Float32(0.001))
        var kind = ph.kind
        var key = Float32(0.0)
        var rx = Float32(1.0)
        var delta = Float32(0.8)
        var dtype = Float32(1.0)
        if not ph.sentence_first and pi > 0:
            var pk = phrases[pi - 1].kind
            if pk == PT_CONT or pk == PT_LIST:
                key = prev_end_b + 0.5
            elif pk == PT_INTER:
                key = min(Float32(0.0), prev_end_b + 1.5)
        if kind == PT_YN:
            delta = 0.95
            dtype = 0.5
        elif kind == PT_WH:
            key += 0.5
        elif kind == PT_EXCL:
            key += 2.0
            rx = 1.4
            delta = 0.85
        elif kind == PT_LIST:
            rx = 0.9
            delta = 1.0
            dtype = 0.3
        elif kind == PT_PAREN:
            key -= 1.5
            rx = 0.7
        elif kind == PT_SUSP:
            rx = 0.9
        var dp = min(Float32(2.5), Float32(2.0) / T) * dtype * voice.decl
        var Rp = R * rx

        # accents: primary-stressed vowels of content words (wh-words accented in questions)
        var acc = List[Int]()
        for q in range(ph.first, ph.last + 1):
            if not tab[segs[q].ph].has(FL_VOWEL) or segs[q].stress != 1:
                continue
            var cp = segs[q].cp
            var wf = wflags[cp] if cp < len(wflags) else 0
            if (wf & WF_FUNCTION) != 0 and not (kind == PT_WH and (wf & WF_WH) != 0):
                continue
            acc.append(q)
        if len(acc) == 0:
            # fall back to the last stressed (or last) vowel as the nucleus
            var last_v = -1
            var last_s = -1
            for q in range(ph.first, ph.last + 1):
                if _syl(tab, segs[q].ph):
                    last_v = q
                    if segs[q].stress > 0:
                        last_s = q
            if last_s >= 0:
                acc.append(last_s)
            elif last_v >= 0:
                acc.append(last_v)
        # boundary tone time: end of last voiced segment
        var tb = t1
        var q2 = ph.last
        while q2 >= ph.first and not tab[segs[q2].ph].has(FL_VOICED):
            q2 -= 1
        if q2 >= ph.first:
            tb = _end(segs, q2)
        tg.append(F0Target(t0, _bl(key, dp, t0, t0) + Float32(0.2) * Rp))
        var a_k = Float32(1.0)
        var last_peak_t = Float32(-1.0)
        for ai in range(len(acc)):
            var q = acc[ai]
            var v0 = segs[q].start
            var v1 = _end(segs, q)
            var nuclear = ai == len(acc) - 1
            var s_k = Float32(1.0)
            var a_use = a_k
            var cp = segs[q].cp
            if kind == PT_WH and cp < len(wflags) and (wflags[cp] & WF_WH) != 0:
                a_use = 1.2
            if nuclear:
                if kind == PT_DECL:
                    s_k = 0.85
                elif kind == PT_EXCL:
                    s_k = 1.3
                elif kind == PT_CONT:
                    s_k = 0.9
            var tk = v0 + Float32(0.65) * (v1 - v0)
            # sag between widely spaced prenuclear accents
            if last_peak_t > 0 and tk - last_peak_t > 300.0:
                var tm = 0.5 * (tk - last_peak_t) + last_peak_t
                tg.append(F0Target(tm, _bl(key, dp, t0, tm) + Float32(0.45) * Rp))
            var tl_ = v0 - 40.0
            if tl_ > t0 + 10.0 and (last_peak_t < 0 or tl_ > last_peak_t + 60.0):
                tg.append(F0Target(tl_, _bl(key, dp, t0, tl_) + Float32(0.15) * Rp))
            if nuclear and kind == PT_YN:
                var tmid = 0.5 * (v0 + v1)
                tg.append(F0Target(tmid, _bl(key, dp, t0, tmid)))
                last_peak_t = tmid
            else:
                tg.append(F0Target(tk, _bl(key, dp, t0, tk) + Rp * a_use * s_k))
                last_peak_t = tk
            if not nuclear:
                a_k = max(Float32(0.45), delta * a_k)
        # phrase accent + boundary tone
        var tpa = max(last_peak_t + 120.0, last_peak_t + 0.5 * (tb - last_peak_t))
        tpa = min(tpa, tb - 10.0)
        if kind == PT_DECL or kind == PT_EXCL or kind == PT_WH:
            var lb = Float32(-1.5) if kind == PT_DECL else (Float32(-2.0) if kind == PT_EXCL else Float32(-1.0))
            if tpa > last_peak_t:
                tg.append(F0Target(tpa, _bl(key, dp, t0, tpa)))
            tg.append(F0Target(tb, _bl(key, dp, t0, tb) + lb))
            prev_end_b = _bl(key, dp, t0, tb)
        elif kind == PT_YN:
            if tpa > last_peak_t:
                tg.append(F0Target(tpa, _bl(key, dp, t0, tpa) + Float32(0.9) * Rp))
            tg.append(F0Target(tb, _bl(key, dp, t0, tb) + Float32(1.6) * Rp))
            prev_end_b = _bl(key, dp, t0, tb)
        elif kind == PT_CONT or kind == PT_LIST:
            var hb = Float32(0.35) if kind == PT_CONT else Float32(0.5)
            if tpa > last_peak_t:
                tg.append(F0Target(tpa, _bl(key, dp, t0, tpa) + Float32(0.1) * Rp))
            tg.append(F0Target(tb, _bl(key, dp, t0, tb) + hb * Rp))
            prev_end_b = _bl(key, dp, t0, tb)
        elif kind == PT_SUSP:
            if tpa > last_peak_t:
                tg.append(F0Target(tpa, _bl(key, dp, t0, tpa) + Float32(0.3) * Rp))
            tg.append(F0Target(tb, _bl(key, dp, t0, tb) + Float32(0.3) * Rp))
            prev_end_b = _bl(key, dp, t0, tb)
        else:
            if tpa > last_peak_t:
                tg.append(F0Target(tpa, _bl(key, dp, t0, tpa)))
            tg.append(F0Target(tb, _bl(key, dp, t0, tb)))
            prev_end_b = _bl(key, dp, t0, tb)
    # sort targets by time (insertion sort, stable)
    for i in range(1, len(tg)):
        var j = i
        while j > 0 and tg[j - 1].t > tg[j].t:
            var tmp = tg[j - 1].copy()
            tg[j - 1] = tg[j].copy()
            tg[j] = tmp^
            j -= 1
    var st = List[Float32](length=nfr, fill=0.0)
    if len(tg) == 0:
        return st^
    var ti = 0
    for k in range(nfr):
        var t = Float32(k) * Float32(5.0)
        while ti + 1 < len(tg) and tg[ti + 1].t <= t:
            ti += 1
        if t <= tg[0].t:
            st[k] = tg[0].v
        elif ti + 1 >= len(tg):
            st[k] = tg[len(tg) - 1].v
        else:
            ref a = tg[ti]
            ref b = tg[ti + 1]
            var x = (t - a.t) / max(Float32(1e-3), b.t - a.t)
            st[k] = a.v + (b.v - a.v) * max(Float32(0.0), min(Float32(1.0), x))
    # zero-phase one-pole (tau 30 ms) forward + backward
    var alpha = Float32(1.0) - det_exp(-Float32(5.0) / Float32(30.0))
    for k in range(1, nfr):
        st[k] = st[k - 1] + alpha * (st[k] - st[k - 1])
    var k2 = nfr - 2
    while k2 >= 0:
        st[k2] = st[k2 + 1] + alpha * (st[k2] - st[k2 + 1])
        k2 -= 1
    return st^


# ============================================================================ 6. track
def _hash_str(s: String, salt: UInt32) -> UInt32:
    var h = UInt32(2166136261) ^ salt
    for b in s.as_bytes():
        h = (h ^ UInt32(b)) * UInt32(16777619)
    return h


def _smooth_param(mut tr: KlattTrack, p: Int):
    """[1 4 6 4 1] / 16 binomial, one pass (zero phase; two passes cost ~2 % WER, tuning log it. 13)."""
    var n = tr.nframes
    if n < 5:
        return
    var tmp = List[Float32](length=n, fill=0.0)
    for _ in range(1):
        for k in range(n):
            var a = tr.get(max(0, k - 2), p)
            var b = tr.get(max(0, k - 1), p)
            var c = tr.get(k, p)
            var d = tr.get(min(n - 1, k + 1), p)
            var e = tr.get(min(n - 1, k + 2), p)
            tmp[k] = (a + Float32(4.0) * b + Float32(6.0) * c + Float32(4.0) * d + e) * Float32(0.0625)
        for k in range(n):
            tr.set(k, p, tmp[k])


def build_track(ps: String, wflags: List[Int], voice: FormantVoice, tab: List[PhonemeDef], loci: List[Locus]) -> KlattTrack:
    """misaki phoneme string (+ optional per-codepoint word flags) -> Klatt parameter track."""
    var segs = parse_phonemes(ps, tab)
    var seed = _hash_str(ps, UInt32(voice.id + 7) * UInt32(0x9E3779B9))
    if len(segs) == 0:
        return KlattTrack(Int((LEAD_MS + TAIL_MS) / 5.0), seed)
    _assign_nuclei(segs, tab)
    var phrases = _phrases(segs, wflags)
    _durations(segs, tab, phrases, wflags, voice)
    var total = _timeline(segs)
    var n = len(segs)
    var nfr = Int((total + TAIL_MS) / 5.0) + 1
    var tr = KlattTrack(nfr, seed)

    # ---- stops: release, VOT / aspiration intervals, voice bars, burst events
    var ivals = List[Interval]()
    for i in range(n):
        ref d = tab[segs[i].ph]
        if not d.has(FL_STOP) or segs[i].ph == PH_TQ:
            continue
        var prev_voiced = i > 0 and tab[segs[i - 1].ph].has(FL_VOICED) and segs[i - 1].ph != PH_PAU
        var nxt = segs[i + 1].ph if i + 1 < n else PH_PAU
        ref dn = tab[nxt]
        var next_voiced = nxt != PH_PAU and dn.has(FL_VOICED)
        var rel = True
        if nxt != PH_PAU and dn.has(FL_STOP) and not dn.has(FL_AFFR) and nxt != PH_TQ:
            rel = False   # K79 rule 4: unreleased before another plosive
        segs[i].released = rel
        # phrase-initial voiced stops are prevoiced (Lisker & Abramson 1964: lead VOT is common in
        # English initial /b d g/); without it initial b / d / g were heard as voiceless or lost
        var phrase_init = i == 0 or segs[i - 1].ph == PH_PAU
        segs[i].voicebar = d.has(FL_VOICED) and ((prev_voiced and next_voiced) or phrase_init)
        var t_rel = _end(segs, i)
        if d.has(FL_AFFR):
            t_rel = segs[i].start + segs[i].dur * (Float32(0.55) if segs[i].ph == PH_CH else Float32(0.7))
        if not rel:
            continue
        var ph = segs[i].ph
        var voiced = d.has(FL_VOICED)
        var amp = Float32(68.0) if not voiced else Float32(63.0)
        var dec = Float32(10.0)
        if ph == PH_P or ph == PH_B:
            dec = 5.0
        elif ph == PH_T:
            dec = 10.0
        elif ph == PH_D:
            dec = 8.0
        elif ph == PH_K:
            dec = 18.0
        elif ph == PH_G:
            dec = 12.0
        var a2 = d.a2
        var a3 = d.a3
        var a4 = d.a4
        var a5 = d.a5
        var a6 = d.a6
        var ab = d.ab
        if d.has(FL_AFFR):
            # release transient: t / d burst spectrum at frication AF + 3 (phonetics §2.2)
            ref tb = tab[PH_T]
            a2 = tb.a2
            a3 = tb.a3
            a4 = tb.a4
            a5 = tb.a5
            a6 = tb.a6
            ab = tb.ab
            amp = d.af + 3.0
            dec = 5.0
        elif (ph == PH_K or ph == PH_G) and nxt != PH_PAU:
            # velar before back / rounded vowels: compact mid-frequency peak (K80 §V.B)
            var vi = i + 1
            while vi < n and not tab[segs[vi].ph].has(FL_VOWEL) and vi < i + 3:
                vi += 1
            if vi < n and tab[segs[vi].ph].has(FL_VOWEL) and tab[segs[vi].ph].f2 < 1400.0:
                a2 = 60.0
                a3 = 40.0
                a4 = 35.0
                a5 = 30.0
                a6 = 30.0
        elif (ph == PH_T or ph == PH_D) and nxt == PH_R:
            ref sh = tab[PH_SH]
            a2 = sh.a2
            a3 = sh.a3
            a4 = sh.a4
            a5 = sh.a5
            a6 = sh.a6
            ab = 0.0
        tr.events.append(KlattEvent(Int(t_rel * Float32(24.0)), amp, dec, a2, a3, a4, a5, a6, ab))
        if d.has(FL_AFFR):
            if ph == PH_CH and nxt != PH_PAU and dn.has(FL_SON):
                ivals.append(Interval(_end(segs, i), _end(segs, i) + 12.0, 50.0, False))
            continue
        if nxt == PH_PAU:
            if not voiced:
                ivals.append(Interval(t_rel, t_rel + 30.0, 50.0, False))
            continue
        if not dn.has(FL_SON) and nxt != PH_HH:
            continue
        # aspiration devoices only the stop's own onset (/pl tr kw/ ...). Before a nasal the stop
        # is nasally released, and across a word boundary the next word's onset sonorant keeps
        # its voicing ("pup gnawed" had its /n/ fully devoiced): a short voiceless release only.
        var cross_word = segs[i + 1].word != segs[i].word
        if dn.has(FL_NASAL) and not cross_word:
            continue
        if cross_word and not dn.has(FL_VOWEL) and nxt != PH_HH:
            if not voiced:
                ivals.append(Interval(t_rel, t_rel + 12.0, 50.0, False))
            continue
        # VOT (Lisker & Abramson 1964 medians; synthesis values from the phonetics note §3.1)
        var nuc = segs[i + 1].nuc
        var stressed = nuc >= 0 and segs[nuc].stress > 0
        var word_init = i == 0 or segs[i - 1].word != segs[i].word or segs[i - 1].ph == PH_PAU
        var after_s = i > 0 and segs[i - 1].ph == PH_S and segs[i - 1].word == segs[i].word
        var vot: Float32
        var place = 0 if (ph == PH_P or ph == PH_B) else (1 if (ph == PH_T or ph == PH_D) else 2)
        if not voiced:
            if after_s:
                vot = Float32(10.0) + Float32(5.0) * Float32(place)
            elif word_init and stressed:
                vot = Float32(50.0) + Float32(10.0) * Float32(place)
            elif stressed:
                vot = Float32(40.0) + Float32(10.0) * Float32(place)
            else:
                vot = Float32(20.0) + (Float32(5.0) if place == 1 else (Float32(15.0) if place == 2 else Float32(0.0)))
            if dn.has(FL_LIQ) or dn.has(FL_GLIDE):
                vot += Float32(15.0) if place == 0 else (Float32(25.0) if place == 1 else Float32(20.0))
                # keep a voiced tail on the devoiced sonorant so it stays identifiable
                var cap = Float32(0.6) * segs[i + 1].dur
                if after_s:
                    cap = Float32(0.4) * segs[i + 1].dur
                vot = min(vot, max(cap, Float32(12.0)))
            var fric = (ph == PH_T) and nxt == PH_R
            # short-lag release after /s/ is weakly aspirated, not silent (a silent gap between the
            # burst and voicing made "Skull" -> "How", tuning log it. 9)
            var ah = Float32(58.0)
            if not stressed:
                ah = 54.0
            if after_s:
                ah = 48.0
            ivals.append(Interval(t_rel, t_rel + vot, ah, fric))
        else:
            if not prev_voiced and not phrase_init:
                vot = Float32(10.0) + Float32(5.0) * Float32(place)
                ivals.append(Interval(t_rel, t_rel + vot, 44.0, ph == PH_D and nxt == PH_R))
    # stable sort of events by sample (insertion; they are nearly sorted)
    for a in range(1, len(tr.events)):
        var j = a
        while j > 0 and tr.events[j - 1].sample > tr.events[j].sample:
            var tmp = tr.events[j - 1].copy()
            tr.events[j - 1] = tr.events[j].copy()
            tr.events[j] = tmp^
            j -= 1

    # ---- formant targets and boundaries
    var tg = _targets(segs, tab, loci)
    var bd = _boundaries(segs, tab, loci, tg)
    var st = _f0_contour(segs, tab, phrases, wflags, voice, nfr)

    # ---- frame fill
    var si = 0
    var fs1 = voice.fs1
    var fs2 = voice.fs2
    var fs3 = voice.fs3
    var fs4 = voice.fs4
    var fs5 = voice.fs5
    var sq1 = sqrt(fs1) * voice.bw1
    var sq2 = sqrt(fs2)
    var sq3 = sqrt(fs3)
    var pidx = List[Int](length=n, fill=0)
    for i in range(n):
        pidx[i] = segs[i].phrase
    var micro = List[Float32](length=nfr, fill=0.0)
    for k in range(nfr):
        var t = Float32(k) * Float32(5.0)
        while si + 1 < n and segs[si + 1].start <= t:
            si += 1
        var s = si
        if t < segs[0].start:
            s = 0
        var in_lead = t < segs[0].start
        var after_end = t >= _end(segs, n - 1)
        ref d = tab[segs[s].ph]
        var u = t - segs[s].start
        var D = segs[s].dur
        var frac = max(Float32(0.0), min(Float32(1.0), u / max(Float32(1.0), D)))
        # ---------- formants: target + deviations from both boundaries
        var fv = List[Float32](length=6, fill=0.0)
        for p in range(6):
            var tgt = _tgt_at(tg[s], d, p, frac)
            var dev = Float32(0.0)
            ref bl = bd[s]
            ref br = bd[s + 1]
            if bl.tr[p] > 0 and u < bl.tr[p]:
                var w = Float32(1.0) - _smooth(u / bl.tr[p])
                dev += w * (bl.bv[p] - _tgt_at(tg[s], d, p, 0.0))
            var ue = D - u
            if br.tl[p] > 0 and ue < br.tl[p]:
                var w2 = Float32(1.0) - _smooth(ue / br.tl[p])
                dev += w2 * (br.bv[p] - _tgt_at(tg[s], d, p, 1.0))
            fv[p] = tgt + dev
        if in_lead or after_end:
            var si2 = 0 if in_lead else n - 1
            for p in range(6):
                fv[p] = tg[si2].get_s(p) if in_lead else tg[si2].get_e(p)
        var f1 = fv[0]
        var b1 = fv[3]
        # ---------- nasal coupling (phonetics §4.2)
        var fnp = Float32(270.0)
        var fnz = Float32(270.0)
        if d.has(FL_NASAL) and not in_lead and not after_end:
            fnz = d.nasal_fz
        elif d.has(FL_VOWEL) and not in_lead and not after_end:
            var nas = Float32(0.0)
            if s + 1 < n and tab[segs[s + 1].ph].has(FL_NASAL):
                var ue2 = D - u
                if ue2 < 60.0:
                    nas = Float32(1.0) - ue2 / Float32(60.0)
            if s > 0 and tab[segs[s - 1].ph].has(FL_NASAL) and u < 30.0:
                nas = max(nas, Float32(1.0) - u / Float32(30.0))
            if nas > 0:
                f1 += Float32(100.0) * nas
                fnz = Float32(270.0) + nas * (Float32(0.5) * (f1 + Float32(270.0)) - Float32(270.0))
        # ---------- source / amplitudes
        var av = Float32(0.0)
        var avs = Float32(0.0)
        var ah = Float32(0.0)
        var af = Float32(0.0)
        var a2 = Float32(0.0)
        var a3 = Float32(0.0)
        var a4 = Float32(0.0)
        var a5 = Float32(0.0)
        var a6 = Float32(0.0)
        var a7 = Float32(0.0)
        var ab = Float32(0.0)
        var f6 = Float32(4900.0)
        var b6 = Float32(1000.0)
        var b6p = Float32(1000.0)
        var tl_extra = Float32(0.0)
        var di = Float32(0.0)
        var pi = pidx[s]
        var ph_kind = phrases[pi].kind if pi < len(phrases) else PT_DECL
        if not in_lead and not after_end and segs[s].ph != PH_PAU:
            if d.has(FL_SON):
                av = d.av
                if d.has(FL_VOWEL) and not d.has(FL_REDUCED) and d.av >= 60.0:
                    av = Float32(60.0) if segs[s].stress == 1 else (Float32(58.0) if segs[s].stress == 2 else Float32(55.0))
                if d.has(FL_FLAP):
                    # AV dips 12 dB in the middle of the flap (phonetics §5.1)
                    var pv_av = Float32(56.0)
                    av = pv_av - Float32(12.0) * (Float32(1.0) - abs(Float32(2.0) * frac - Float32(1.0)))
                if d.has(FL_NASAL) and (s == 0 or segs[s - 1].ph == PH_PAU) and u < 20.0:
                    av -= 3.0
                # intonation-linked amplitude: declination -3 dB over the phrase, final -6 dB
                if pi < len(phrases):
                    var pt0 = segs[phrases[pi].first].start
                    var pt1 = _end(segs, phrases[pi].last)
                    var pos = (t - pt0) / max(Float32(1.0), pt1 - pt0)
                    av -= Float32(3.0) * pos
                    if (ph_kind == PT_DECL or ph_kind == PT_EXCL) and pt1 - t < 80.0:
                        var x = Float32(1.0) - (pt1 - t) / Float32(80.0)
                        av -= Float32(6.0) * x
                        tl_extra = Float32(4.0) * x
                ah = av + voice.ah_rel if av > 0 else Float32(0.0)
            elif d.has(FL_FRIC):
                af = d.af
                if segs[s].ph == PH_S and (s == 0 or segs[s - 1].word != segs[s].word):
                    af += 3.0
                av = d.av
                avs = d.avs
                a2 = d.a2
                a3 = d.a3
                a4 = d.a4
                a5 = d.a5
                a6 = d.a6
                ab = d.ab
                if segs[s].ph == PH_S or segs[s].ph == PH_Z:
                    f6 = 5500.0
                    b6 = 1500.0
                    b6p = 1500.0
                    a7 = a6 - 6.0
                if segs[s].ph == PH_F or segs[s].ph == PH_V or segs[s].ph == PH_TH or segs[s].ph == PH_DH:
                    b6p = 1500.0
                # frication envelope: gradual rise (>= 30 ms for ʃ / s, Stevens 1998), 20 ms decay
                var rise = min(Float32(30.0), D * Float32(0.35))
                var fall = min(Float32(20.0), D * Float32(0.3))
                var env = Float32(0.0)
                if u < rise:
                    env = Float32(-18.0) * (Float32(1.0) - u / rise)
                if D - u < fall:
                    env = min(env, Float32(-18.0) * (Float32(1.0) - (D - u) / fall))
                af += env
            elif d.has(FL_STOP) and segs[s].ph != PH_TQ:
                var split = Float32(1.0)
                if d.has(FL_AFFR):
                    split = Float32(0.55) if segs[s].ph == PH_CH else Float32(0.7)
                if frac < split or not segs[s].released:
                    if segs[s].voicebar:
                        avs = Float32(50.0) - Float32(10.0) * (frac / split)
                else:
                    # affricate frication: abrupt (< 20 ms) rise for ʧ, decays over the last 10 ms
                    af = d.af
                    av = d.av
                    avs = d.avs
                    a2 = d.a2
                    a3 = d.a3
                    a4 = d.a4
                    a5 = d.a5
                    a6 = d.a6
                    if D - u < 10.0:
                        af -= Float32(12.0) * (Float32(1.0) - (D - u) / Float32(10.0))
            elif d.has(FL_ASPIR):
                ah = d.ah
                var prev_v = s > 0 and tab[segs[s - 1].ph].has(FL_VOWEL)
                var next_v = s + 1 < n and tab[segs[s + 1].ph].has(FL_VOWEL)
                if prev_v and next_v:
                    av = 38.0   # intervocalic breathy-voiced h
                else:
                    # [h] has no release transient: noise rises over ~25 ms, and voicing fades in
                    # under the noise over the last 20 ms before the vowel (Klatt & Klatt 1990)
                    var h_rise = min(Float32(25.0), D * Float32(0.35))
                    if not prev_v and u < h_rise:
                        ah -= Float32(18.0) * (Float32(1.0) - u / h_rise)
                    if next_v and D - u < 20.0:
                        av = Float32(30.0) + Float32(15.0) * (Float32(1.0) - (D - u) / Float32(20.0))
        # aspiration / short-lag intervals override the source
        var in_asp = False
        for iv in ivals:
            if t >= iv.t0 and t < iv.t1:
                in_asp = True
                av = 0.0
                avs = 0.0
                ah = iv.ah
                if iv.ah > 0:
                    b1 = max(b1, Float32(300.0))
                if iv.fric:
                    af = 50.0
                    a3 = 57.0
                    a4 = 48.0
                    a5 = 48.0
                    a6 = 46.0
                break
        _ = in_asp
        # glottal stop: creaky edges (phonetics §5.4)
        if not in_lead and not after_end:
            if s + 1 < n and segs[s + 1].ph == PH_Q and D - u < 30.0:
                di = 30.0
            if s > 0 and segs[s - 1].ph == PH_Q and u < 30.0:
                di = 30.0
        # ---------- voice quality
        var rd = voice.rd
        var nuc = segs[s].nuc
        if nuc >= 0 and segs[nuc].stress == 1 and d.has(FL_VOWEL):
            rd -= 0.15
        var jit = voice.jitter
        if pi < len(phrases) and not in_lead:
            var pt0b = segs[phrases[pi].first].start
            var pt1b = _end(segs, phrases[pi].last)
            var posb = max(Float32(0.0), min(Float32(1.0), (t - pt0b) / max(Float32(1.0), pt1b - pt0b)))
            rd += Float32(0.25) * posb * posb
            # phrase-final creak for low voices (1 phrase end in 2, keyed off the seed)
            if voice.creak and (ph_kind == PT_DECL) and pt1b - t < 120.0 and pt1b - t >= 0:
                var pick = ((seed >> UInt32(pi % 16)) & UInt32(1)) == UInt32(0)
                if pick:
                    var x2 = Float32(1.0) - (pt1b - t) / Float32(120.0)
                    di = max(di, Float32(40.0) * x2)
                    rd = rd + (Float32(0.6) - rd) * x2
                    jit = jit + (Float32(0.03) - jit) * x2
        # ---------- microprosody (semitones)
        var mp = Float32(0.0)
        if d.has(FL_VOWEL) and not in_lead and not after_end:
            var ip = Float32(0.0)
            var phv = segs[s].ph
            if phv == PH_I or phv == PH_U:
                ip = 0.6
            elif phv == PH_IH or phv == PH_UH or phv == PH_IX:
                ip = 0.3
            elif phv == PH_AE or phv == PH_AA or phv == PH_OH:
                ip = -0.3
            if segs[s].stress == 0:
                ip *= 0.5
            mp += ip
            if s > 0:
                ref dp = tab[segs[s - 1].ph]
                var obst = dp.has(FL_STOP) or dp.has(FL_FRIC) or dp.has(FL_ASPIR)
                if obst and not dp.has(FL_VOICED):
                    mp += Float32(1.0) * det_exp(-u / Float32(40.0)) * min(Float32(1.0), u / Float32(10.0) + Float32(0.5))
                elif obst:
                    mp -= Float32(0.6) * det_exp(-u / Float32(30.0))
        elif (d.has(FL_STOP) or d.has(FL_FRIC)) and d.has(FL_VOICED):
            mp -= 1.5
        micro[k] = mp
        # ---------- write the frame (voice scaling: per-formant F, sqrt for B)
        tr.set(k, P_AV, av)
        tr.set(k, P_AVS, avs)
        tr.set(k, P_AH, ah)
        tr.set(k, P_AF, af)
        tr.set(k, P_A2, a2)
        tr.set(k, P_A3, a3)
        tr.set(k, P_A4, a4)
        tr.set(k, P_A5, a5)
        tr.set(k, P_A6, a6)
        tr.set(k, P_A7, a7)
        tr.set(k, P_AB, ab)
        tr.set(k, P_F1, f1 * fs1)
        tr.set(k, P_B1, b1 * sq1)
        tr.set(k, P_F2, fv[1] * fs2)
        tr.set(k, P_B2, fv[4] * sq2)
        tr.set(k, P_F3, fv[2] * fs3)
        tr.set(k, P_B3, fv[5] * sq3)
        # F4 / F5 raised to adult-male means (~3.5 / 4.35 kHz, [F60]); K80 3.3 / 3.75 kHz targets
        # date from 10 kHz synthesis where F5 sat at the band edge (tuning log, it. 12)
        var f4 = (d.f4 + Float32(200.0)) if not in_lead and not after_end else Float32(3500.0)
        var f5 = (d.f5 + Float32(600.0)) if not in_lead and not after_end else Float32(4350.0)
        tr.set(k, P_F4, f4 * fs4)
        tr.set(k, P_B4, 300.0)
        tr.set(k, P_F5, f5 * fs5)
        tr.set(k, P_B5, 350.0)
        tr.set(k, P_F6, f6 * fs5)
        tr.set(k, P_B6, b6)
        tr.set(k, P_B6P, b6p)
        tr.set(k, P_F7, Float32(7500.0) * fs5)
        tr.set(k, P_FNP, fnp * fs1)
        tr.set(k, P_FNZ, fnz * fs1)
        if voice.tracheal:
            tr.set(k, P_FTP, Float32(1550.0) * fs2)
            tr.set(k, P_BTP, 200.0)
            tr.set(k, P_FTZ, Float32(1350.0) * fs2)
            tr.set(k, P_BTZ, 200.0)
        tr.set(k, P_RD, rd)
        tr.set(k, P_TL, voice.tl + tl_extra)
        tr.set(k, P_JIT, jit)
        tr.set(k, P_SHIM, voice.shimmer)
        tr.set(k, P_FL, voice.flutter)
        tr.set(k, P_DI, di)
        tr.set(k, P_DF1, voice.df1)
        tr.set(k, P_DB1, voice.db1)
    # ---- smoothing of the SMOOTH class (formants, bandwidths, voice quality)
    _smooth_param(tr, P_F1)
    _smooth_param(tr, P_F2)
    _smooth_param(tr, P_F3)
    _smooth_param(tr, P_B1)
    _smooth_param(tr, P_B2)
    _smooth_param(tr, P_B3)
    _smooth_param(tr, P_F4)
    _smooth_param(tr, P_F5)
    _smooth_param(tr, P_F6)
    _smooth_param(tr, P_B6)
    _smooth_param(tr, P_RD)
    _smooth_param(tr, P_TL)
    _smooth_param(tr, P_DI)
    # microprosody is smoothed a little so its onsets ramp in (>= 10 ms)
    var mtmp = List[Float32](length=nfr, fill=0.0)
    for k in range(nfr):
        var a = micro[max(0, k - 1)]
        var b = micro[k]
        var c = micro[min(nfr - 1, k + 1)]
        mtmp[k] = (a + Float32(2.0) * b + c) * Float32(0.25)
    # ---- F0 in Hz
    for k in range(nfr):
        var s_t = st[k] + mtmp[k]
        var hz = voice.f0_base * det_exp(s_t * Float32(0.057762265))   # 2^(st/12)
        var floor_hz = voice.f0_floor
        if tr.get(k, P_DI) > 5.0:
            floor_hz = min(floor_hz, Float32(55.0))
        tr.set(k, P_F0, max(floor_hz, hz))
    # ---- AV onset / offset ramps (voicing rises over ~10 ms, falls over ~5 ms)
    for k in range(1, nfr - 1):
        var a0 = tr.get(k - 1, P_AV)
        var a1 = tr.get(k, P_AV)
        if a0 <= 0.5 and a1 > 0.5:
            tr.set(k, P_AV, a1 - 10.0)
            var a2v = tr.get(k + 1, P_AV)
            if a2v > 0.5:
                tr.set(k + 1, P_AV, a2v - 3.0)
    return tr^
