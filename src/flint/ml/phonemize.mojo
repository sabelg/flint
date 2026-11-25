"""English grapheme-to-phoneme conversion in pure Mojo (`flint.ml.phonemize`).

Emits the phoneme alphabet of misaki (https://github.com/hexgrad/misaki, Apache-2.0), the G2P that
Kokoro-82M was trained on: IPA-like symbols plus single-letter diphthongs (A = eɪ, I = aɪ, O = oʊ,
W = aʊ, Y = ɔɪ), ᵊ (syllabic schwa), ᵻ, T (the US flap) and stress marks ˈ ˌ written right
before a vowel. The lexicon logic (casing, stress, -s / -ed / -ing morphology, the context rules
for "the / to / a / an / in / am / I / used", numbers, currency) is a port of misaki's English
lexicon (Apache-2.0); spaCy's tagger is replaced by a small context tagger for homographs.

Lexicon data: misaki's `us_gold.json` + `us_silver.json` (or `gb_*`, Apache-2.0), read at run
time from a directory outside the repository (`G2P.load(dir)`); `tools/kokoro/fetch.sh` fetches
them. Words the lexicon lacks are split into known compounds ("nightfall" -> night + fall), and
otherwise spelled by the letter-to-sound rules below (written for Flint) with stress heuristics,
so every word gets a pronunciation. `add_word` adds or overrides entries (game names), and
`[Zorblax](/zˈɔɹblæks/)` in the text pins a pronunciation inline.

No espeak-ng code or data is used (it is GPL).

    var g2p = G2P.load("/path/to/lexicon_dir")     # or G2P() for rules only
    var r = g2p.phonemize("Hello there, traveller!")
    r.phonemes                                       # "həlˈO ðˈɛɹ, tɹˈævəlɚ!"-style string
    r.words                                          # per word: text, phonemes, codepoint span
"""

from std.collections import Dict
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.assets.json import parse_json
from flint.platform.os import read_bytes

comptime NONE_PS = "~"          # a lexicon entry that is explicitly null (spell it out)
comptime NO_STRESS: Float64 = 99.0
comptime PRIMARY: Int = 0x2C8   # ˈ
comptime SECONDARY: Int = 0x2CC # ˌ

comptime VOWELS = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ"
comptime CONSONANTS = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ"
comptime DIPHTHONGS = "AIOQWYʤʧ"
comptime US_TAUS = "AIOWYiuæɑəɛɪɹʊʌ"
comptime PUNCTS = ";:,.!?—…\"“”"
comptime NON_QUOTE_PUNCTS = ";:,.!?—…"


# ------------------------------------------------------------------------------ small helpers

def _cp_in(c: Int, chars: String) -> Bool:
    for x in decode_utf8(chars):
        if x == c:
            return True
    return False


def _has_cp(s: String, c: Int) -> Bool:
    for x in decode_utf8(s):
        if x == c:
            return True
    return False


def _has_any(s: String, chars: String) -> Bool:
    var set_ = decode_utf8(chars)
    for x in decode_utf8(s):
        for y in set_:
            if x == y:
                return True
    return False


def _last_cp(s: String) -> Int:
    var c = decode_utf8(s)
    return c[len(c) - 1] if len(c) > 0 else 0


def _drop_last(s: String) -> String:
    var c = decode_utf8(s)
    return encode_utf8(c, 0, max(0, len(c) - 1))


def _strip_cp(s: String, c: Int) -> String:
    var out = List[Int]()
    for x in decode_utf8(s):
        if x != c:
            out.append(x)
    return encode_utf8(out)


def _replace_cp(s: String, a: Int, b: Int) -> String:
    var out = decode_utf8(s)
    for i in range(len(out)):
        if out[i] == a:
            out[i] = b
    return encode_utf8(out)


@always_inline
def _is_upper_b(b: Int) -> Bool:
    return b >= 65 and b <= 90


@always_inline
def _is_lower_b(b: Int) -> Bool:
    return b >= 97 and b <= 122


@always_inline
def _is_alpha_b(b: Int) -> Bool:
    return _is_upper_b(b) or _is_lower_b(b)


@always_inline
def _is_digit_b(b: Int) -> Bool:
    return b >= 48 and b <= 57


def _byte(s: String, i: Int) -> Int:
    return Int(s.as_bytes()[i])


def _sub(s: String, a: Int, b: Int) -> String:
    return String(s[byte=a:b])


def _lower(s: String) -> String:
    return s.lower()


def _upper(s: String) -> String:
    return s.upper()


def _capitalize(s: String) -> String:
    if s.byte_length() == 0:
        return s
    return _upper(_sub(s, 0, 1)) + _lower(_sub(s, 1, s.byte_length()))


def _is_alpha(s: String) -> Bool:
    if s.byte_length() == 0:
        return False
    for b in s.as_bytes():
        if not _is_alpha_b(Int(b)):
            return False
    return True


def _is_digits(s: String) -> Bool:
    if s.byte_length() == 0:
        return False
    for b in s.as_bytes():
        if not _is_digit_b(Int(b)):
            return False
    return True


def _lexicon_chars(s: String) -> Bool:
    """Only ASCII letters, apostrophe and hyphen (misaki's LEXICON_ORDS)."""
    for b in s.as_bytes():
        var c = Int(b)
        if not (_is_alpha_b(c) or c == 39 or c == 45):
            return False
    return True


def _no_apos(s: String) -> String:
    return s.replace("'", "")


def _ends(s: String, suf: String) -> Bool:
    return s.endswith(suf)


def _fold_accents(codes: List[Int]) -> List[Int]:
    """Latin-1 accented letters to their ASCII base letter; typographic apostrophes to '."""
    var out = List[Int]()
    for c in codes:
        var r = c
        if c == 0x2018 or c == 0x2019 or c == 0x2BC:
            r = 39
        elif c >= 0xC0 and c <= 0xC5:
            r = 65
        elif c >= 0xE0 and c <= 0xE5:
            r = 97
        elif c == 0xC7:
            r = 67
        elif c == 0xE7:
            r = 99
        elif c >= 0xC8 and c <= 0xCB:
            r = 69
        elif c >= 0xE8 and c <= 0xEB:
            r = 101
        elif c >= 0xCC and c <= 0xCF:
            r = 73
        elif c >= 0xEC and c <= 0xEF:
            r = 105
        elif c == 0xD1:
            r = 78
        elif c == 0xF1:
            r = 110
        elif (c >= 0xD2 and c <= 0xD6) or c == 0xD8:
            r = 79
        elif (c >= 0xF2 and c <= 0xF6) or c == 0xF8:
            r = 111
        elif c >= 0xD9 and c <= 0xDC:
            r = 85
        elif c >= 0xF9 and c <= 0xFC:
            r = 117
        elif c == 0xDD:
            r = 89
        elif c == 0xFD or c == 0xFF:
            r = 121
        elif c == 0x2013:
            r = 45
        out.append(r)
    return out^


# ------------------------------------------------------------------------------------- stress

def stress_weight(ps: String) -> Int:
    var w = 0
    for c in decode_utf8(ps):
        w += 2 if _cp_in(c, DIPHTHONGS) else 1
    return w


def _restress(ps: String) -> String:
    """Move every stress mark to just before the next vowel (misaki's `restress`)."""
    var codes = decode_utf8(ps)
    var n = len(codes)
    var pos = List[Float64](length=n, fill=0.0)
    for i in range(n):
        pos[i] = Float64(i)
        if codes[i] == PRIMARY or codes[i] == SECONDARY:
            for j in range(i, n):
                if _cp_in(codes[j], VOWELS):
                    pos[i] = Float64(j) - 0.5
                    break
    # stable insertion sort by position
    var order = List[Int]()
    for i in range(n):
        var k = len(order)
        order.append(i)
        while k > 0 and pos[order[k - 1]] > pos[i]:
            order[k] = order[k - 1]
            k -= 1
        order[k] = i
    var out = List[Int]()
    for i in order:
        out.append(codes[i])
    return encode_utf8(out)


def apply_stress(ps: String, stress: Float64) -> String:
    """misaki's `apply_stress`: < -1 removes all stress, -1 (or <= 0 with a primary) demotes
    primary to secondary, 0..1 adds a secondary to an unstressed word, >= 1 promotes a lone
    secondary, > 1 adds a primary. `NO_STRESS` leaves the word alone."""
    if stress == NO_STRESS:
        return ps
    var has_p = _has_cp(ps, PRIMARY)
    var has_s = _has_cp(ps, SECONDARY)
    if stress < -1:
        return _strip_cp(_strip_cp(ps, PRIMARY), SECONDARY)
    if stress == -1 or ((stress == 0 or stress == -0.5) and has_p):
        return _replace_cp(_strip_cp(ps, SECONDARY), PRIMARY, SECONDARY)
    if (stress == 0 or stress == 0.5 or stress == 1) and not has_p and not has_s:
        if not _has_any(ps, VOWELS):
            return ps
        return _restress(encode_utf8([SECONDARY]) + ps)
    if stress >= 1 and not has_p and has_s:
        return _replace_cp(ps, SECONDARY, PRIMARY)
    if stress > 1 and not has_p and not has_s:
        if not _has_any(ps, VOWELS):
            return ps
        return _restress(encode_utf8([PRIMARY]) + ps)
    return ps


# --------------------------------------------------------------------------------- numbers

def _ones_word(n: Int) -> String:
    var w: List[String] = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                         "eighteen", "nineteen"]
    return w[n]


def _tens_word(n: Int) -> String:
    var w: List[String] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    return w[n]


def _below_thousand(n: Int, mut out: List[String]):
    var h = n // 100
    var r = n % 100
    if h > 0:
        out.append(_ones_word(h))
        out.append("hundred")
    if r >= 20:
        out.append(_tens_word(r // 10))
        if r % 10 != 0:
            out.append(_ones_word(r % 10))
    elif r > 0 or h == 0:
        out.append(_ones_word(r))


def cardinal_words(n_in: Int) -> List[String]:
    """English cardinal words without "and" (num2words split on non-letters, as misaki uses it)."""
    var out = List[String]()
    var n = n_in
    if n < 0:
        out.append("minus")
        n = -n
    if n == 0:
        out.append("zero")
        return out^
    var scales: List[String] = ["", "thousand", "million", "billion", "trillion", "quadrillion"]
    var groups = List[Int]()
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    var g = len(groups) - 1
    while g >= 0:
        if groups[g] != 0:
            _below_thousand(groups[g], out)
            if g > 0 and g < len(scales):
                out.append(scales[g])
        g -= 1
    return out^


def _ordinal_of(word: String) -> String:
    if word == "one": return "first"
    if word == "two": return "second"
    if word == "three": return "third"
    if word == "five": return "fifth"
    if word == "eight": return "eighth"
    if word == "nine": return "ninth"
    if word == "twelve": return "twelfth"
    if word.endswith("y"):
        return _sub(word, 0, word.byte_length() - 1) + "ieth"
    return word + "th"


def ordinal_words(n: Int) -> List[String]:
    var w = cardinal_words(n)
    w[len(w) - 1] = _ordinal_of(w[len(w) - 1])
    return w^


def year_words(v: Int) -> List[String]:
    """num2words' `to_year`: 1987 -> nineteen eighty seven, 1905 -> nineteen oh five,
    2005 -> two thousand five, 1900 -> nineteen hundred."""
    var high = v // 100
    var low = v % 100
    if high == 0 or (high % 10 == 0 and low < 10) or high >= 100:
        return cardinal_words(v)
    var out = cardinal_words(high)
    if low == 0:
        out.append("hundred")
    elif low < 10:
        out.append("oh")
        for w in cardinal_words(low):
            out.append(w)
    else:
        for w in cardinal_words(low):
            out.append(w)
    return out^


# ------------------------------------------------------------------------------ result types

struct G2PWord(Copyable, Movable):
    var text: String
    var phonemes: String
    var start: Int       # codepoint span [start, end) of this word in `G2PResult.phonemes`
    var end: Int
    var known: Bool      # from the lexicon (or a number / symbol), not the letter-to-sound rules

    def __init__(out self, text: String, phonemes: String, start: Int, end: Int, known: Bool):
        self.text = text
        self.phonemes = phonemes
        self.start = start
        self.end = end
        self.known = known


struct G2PResult(Copyable, Movable):
    var phonemes: String
    var words: List[G2PWord]

    def __init__(out self):
        self.phonemes = String()
        self.words = List[G2PWord]()

    def word_at(self, cp_index: Int) -> Int:
        """Index of the word whose phonemes contain codepoint `cp_index`, or -1 (space / punct)."""
        for i in range(len(self.words)):
            if cp_index >= self.words[i].start and cp_index < self.words[i].end:
                return i
        return -1


# token kinds
comptime TK_WORD: Int = 0
comptime TK_NUM: Int = 1
comptime TK_PUNCT: Int = 2
comptime TK_FIXED: Int = 3


struct _Tok(Copyable, Movable):
    var text: String
    var kind: Int
    var ws: Bool          # whitespace after
    var joined: Bool      # glued to the previous token (same written word: hyphen / camelCase)
    var ps: String
    var has_ps: Bool
    var known: Bool
    var currency: Int     # 0 none, 1 $, 2 £, 3 €
    var tag: String

    def __init__(out self, text: String, kind: Int):
        self.text = text
        self.kind = kind
        self.ws = False
        self.joined = False
        self.ps = String()
        self.has_ps = False
        self.known = False
        self.currency = 0
        self.tag = String()


# ------------------------------------------------------------------------------- letter rules

struct _Rule(Copyable, Movable):
    var left: String
    var pat: String
    var right: String
    var out: String

    def __init__(out self, left: String, pat: String, right: String, out_: String):
        self.left = left
        self.pat = pat
        self.right = right
        self.out = out_


def _letter_rules() -> List[_Rule]:
    """Letter-to-sound rules (written for Flint, in the style of the 1976 NRL report): `match`
    letters between `left` and `right` contexts, tried in order per first letter. Contexts:
    ' ' word edge, '#' one or more vowels, ':' zero or more consonants, '^' one consonant,
    '+' E / I / Y, '%' a suffix (E ES ED ER ING ELY), '@' T S R D L Z N J (yod dropping),
    other letters literal. Outputs are misaki phonemes before stress is assigned."""
    var R = List[_Rule]()
    # vowel + R + silent E, and other whole-ending patterns
    R.append(_Rule("", "ARE", " ", "ɛɹ"))
    R.append(_Rule("", "IRE", " ", "Iəɹ"))
    R.append(_Rule("", "ORE", " ", "ɔɹ"))
    R.append(_Rule("", "URE", " ", "jʊɹ"))
    R.append(_Rule("", "ERE", " ", "ɪɹ"))
    # stress-neutral suffixes
    R.append(_Rule("", "FUL", " ", "fᵊl"))
    R.append(_Rule("", "ABLE", " ", "əbᵊl"))
    R.append(_Rule("", "IBLE", " ", "əbᵊl"))
    R.append(_Rule("", "NESS", " ", "nəs"))
    R.append(_Rule("", "LESS", " ", "ləs"))
    R.append(_Rule("", "MENT", " ", "mənt"))
    # A
    R.append(_Rule("", "AUGH", "", "ɔ"))
    R.append(_Rule("", "AU", "", "ɔ"))
    R.append(_Rule("", "AW", "", "ɔ"))
    R.append(_Rule("", "AY", "", "A"))
    R.append(_Rule("", "AI", "", "A"))
    R.append(_Rule("", "AE", "", "i"))
    R.append(_Rule("", "ARR", "", "æɹ"))
    R.append(_Rule("", "AR", "#", "ɛɹ"))
    R.append(_Rule("", "AR", "", "ɑɹ"))
    R.append(_Rule("", "ALL", "", "ɔl"))
    R.append(_Rule("", "ALK", "", "ɔk"))
    R.append(_Rule("", "ANGE", "", "Anʤ"))
    R.append(_Rule("", "A", "^%", "A"))
    R.append(_Rule("", "A", "^+#", "A"))
    R.append(_Rule(" ", "A", "^#", "ə"))
    R.append(_Rule("", "A", " ", "ə"))
    R.append(_Rule("", "A", "", "æ"))
    # B
    R.append(_Rule("", "BB", "", "b"))
    R.append(_Rule("M", "B", " ", ""))
    R.append(_Rule("", "B", "", "b"))
    # C
    R.append(_Rule("", "CH", "R", "k"))
    R.append(_Rule("", "CH", "", "ʧ"))
    R.append(_Rule("", "CK", "", "k"))
    R.append(_Rule("", "CIA", "", "ʃə"))
    R.append(_Rule("", "CIO", "", "ʃ"))
    R.append(_Rule("", "CC", "+", "ks"))
    R.append(_Rule("", "C", "+", "s"))
    R.append(_Rule("", "C", "", "k"))
    # D
    R.append(_Rule("", "DD", "", "d"))
    R.append(_Rule("", "DGE", "", "ʤ"))
    R.append(_Rule("", "DG", "", "ʤ"))
    R.append(_Rule("", "D", "", "d"))
    # E
    R.append(_Rule("", "EAU", "", "O"))
    R.append(_Rule("", "EAR", "", "ɪɹ"))
    R.append(_Rule("", "EER", "", "ɪɹ"))
    R.append(_Rule("", "EE", "", "i"))
    R.append(_Rule("", "EA", "", "i"))
    R.append(_Rule("", "EW", "", "u"))
    R.append(_Rule("", "EU", "", "u"))
    R.append(_Rule("", "EY", " ", "i"))
    R.append(_Rule("", "EY", "", "A"))
    R.append(_Rule("", "EI", "", "A"))
    R.append(_Rule("", "ER", "#", "ɛɹ"))
    R.append(_Rule("", "ER", "", "əɹ"))
    R.append(_Rule("#:", "ES", " ", "z"))
    for l in ["T", "D"]:
        R.append(_Rule(l, "ED", " ", "ᵻd"))
    for l in ["P", "K", "F", "S", "X", "C", "H"]:
        R.append(_Rule(l, "ED", " ", "t"))
    R.append(_Rule("#:", "ED", " ", "d"))
    R.append(_Rule("#:", "E", " ", ""))
    R.append(_Rule(" :", "E", " ", "i"))
    R.append(_Rule("", "E", "^%", "i"))
    R.append(_Rule("", "E", "", "ɛ"))
    # F
    R.append(_Rule("", "FF", "", "f"))
    R.append(_Rule("", "F", "", "f"))
    # G
    R.append(_Rule("", "GG", "", "ɡ"))
    R.append(_Rule(" ", "GN", "", "n"))
    R.append(_Rule("", "GN", " ", "n"))
    R.append(_Rule(" ", "GH", "", "ɡ"))
    R.append(_Rule("", "GH", "", ""))
    R.append(_Rule("", "GE", " ", "ʤ"))
    R.append(_Rule("", "G", "+", "ʤ"))
    R.append(_Rule("", "G", "", "ɡ"))
    # H
    R.append(_Rule("", "H", "#", "h"))
    R.append(_Rule("", "H", "", ""))
    # I
    R.append(_Rule("", "IGH", "", "I"))
    R.append(_Rule("", "IE", " ", "I"))
    R.append(_Rule("", "IE", "", "i"))
    R.append(_Rule("", "IR", "#", "Iɹ"))
    R.append(_Rule("", "IR", "", "ɜɹ"))
    R.append(_Rule("", "IND", " ", "Ind"))
    R.append(_Rule("", "ILD", "", "Ild"))
    R.append(_Rule("", "I", "^%", "I"))
    R.append(_Rule("", "I", "^+#", "I"))
    R.append(_Rule("", "I", " ", "i"))
    R.append(_Rule("", "I", "#", "i"))
    R.append(_Rule("", "I", "", "ɪ"))
    # J K L M N
    R.append(_Rule("", "J", "", "ʤ"))
    R.append(_Rule(" ", "KN", "", "n"))
    R.append(_Rule("", "K", "", "k"))
    R.append(_Rule("^", "LE", " ", "ᵊl"))
    R.append(_Rule("", "LL", "", "l"))
    R.append(_Rule("", "L", "", "l"))
    R.append(_Rule("", "MM", "", "m"))
    R.append(_Rule("", "M", "", "m"))
    R.append(_Rule("", "NG", "", "ŋ"))
    R.append(_Rule("", "NK", "", "ŋk"))
    R.append(_Rule("", "NN", "", "n"))
    R.append(_Rule("", "N", "", "n"))
    # O
    R.append(_Rule("", "OUGH", "T", "ɔ"))
    R.append(_Rule("", "OUGH", "", "O"))
    R.append(_Rule("", "OOR", "", "ɔɹ"))
    R.append(_Rule("", "OO", "K", "ʊ"))
    R.append(_Rule("", "OO", "", "u"))
    R.append(_Rule("", "OA", "", "O"))
    R.append(_Rule("", "OE", " ", "O"))
    R.append(_Rule("", "OI", "", "Y"))
    R.append(_Rule("", "OY", "", "Y"))
    R.append(_Rule("", "OU", "", "W"))
    R.append(_Rule("", "OW", " ", "O"))
    R.append(_Rule("", "OW", "", "W"))
    R.append(_Rule("", "OR", "", "ɔɹ"))
    R.append(_Rule("", "OLL", "", "Ol"))
    R.append(_Rule("", "OLD", "", "Old"))
    R.append(_Rule("", "O", "^%", "O"))
    R.append(_Rule("", "O", "^+#", "O"))
    R.append(_Rule("", "O", " ", "O"))
    R.append(_Rule("", "O", "#", "O"))
    R.append(_Rule("", "O", "", "ɑ"))
    # P Q R
    R.append(_Rule("", "PH", "", "f"))
    R.append(_Rule("", "PP", "", "p"))
    R.append(_Rule(" ", "PS", "", "s"))
    R.append(_Rule("", "P", "", "p"))
    R.append(_Rule("", "QU", "", "kw"))
    R.append(_Rule("", "Q", "", "k"))
    R.append(_Rule(" ", "RH", "", "ɹ"))
    R.append(_Rule("", "RR", "", "ɹ"))
    R.append(_Rule("", "R", "", "ɹ"))
    # S
    R.append(_Rule("", "SCH", "", "sk"))
    R.append(_Rule("", "SC", "+", "s"))
    R.append(_Rule("", "SH", "", "ʃ"))
    R.append(_Rule("", "SION", "", "ʒən"))
    R.append(_Rule("", "SS", "", "s"))
    R.append(_Rule("#", "S", "#", "z"))
    R.append(_Rule("", "S", "", "s"))
    # T
    R.append(_Rule("", "TION", "", "ʃən"))
    R.append(_Rule("", "TIA", "", "ʃə"))
    R.append(_Rule("", "TURE", "", "ʧəɹ"))
    R.append(_Rule("", "TCH", "", "ʧ"))
    R.append(_Rule("", "TH", "", "θ"))
    R.append(_Rule("", "TT", "", "t"))
    R.append(_Rule("", "T", "", "t"))
    # U
    R.append(_Rule("", "UR", "#", "jʊɹ"))
    R.append(_Rule("", "UR", "", "ɜɹ"))
    R.append(_Rule("", "UE", " ", "u"))
    R.append(_Rule("", "UI", "", "u"))
    R.append(_Rule("@", "U", "^%", "u"))
    R.append(_Rule("", "U", "^%", "ju"))
    R.append(_Rule("", "U", " ", "u"))
    R.append(_Rule("", "U", "", "ʌ"))
    # V W X Y Z
    R.append(_Rule("", "V", "", "v"))
    R.append(_Rule(" ", "WR", "", "ɹ"))
    R.append(_Rule("", "WH", "", "w"))
    R.append(_Rule("", "W", "", "w"))
    R.append(_Rule(" ", "X", "", "z"))
    R.append(_Rule("", "X", "", "ks"))
    R.append(_Rule(" ", "Y", "#", "j"))
    R.append(_Rule(" :", "Y", " ", "I"))
    R.append(_Rule("", "Y", " ", "i"))
    R.append(_Rule("", "Y", "^%", "I"))
    R.append(_Rule("", "Y", "#", "j"))
    R.append(_Rule("", "Y", "", "ɪ"))
    R.append(_Rule("", "ZZ", "", "z"))
    R.append(_Rule("", "Z", "", "z"))
    R.append(_Rule("", "'S", " ", "z"))
    R.append(_Rule("", "'", "", ""))
    return R^


@always_inline
def _is_vowel_letter(c: Int) -> Bool:
    return c == 65 or c == 69 or c == 73 or c == 79 or c == 85 or c == 89


@always_inline
def _is_cons_letter(c: Int) -> Bool:
    return _is_upper_b(c) and not _is_vowel_letter(c)


def _match_right(w: List[Int], start: Int, ctx: String) -> Bool:
    """Match context `ctx` against w[start:] (w is " " + WORD + " " in uppercase ASCII)."""
    var p = start
    var cb = ctx.as_bytes()
    var i = 0
    while i < len(cb):
        var c = Int(cb[i])
        if c == 32:
            if p >= len(w) or w[p] != 32:
                return False
            p += 1
        elif c == 35:      # '#': one or more vowels
            if p >= len(w) or not _is_vowel_letter(w[p]):
                return False
            while p < len(w) and _is_vowel_letter(w[p]):
                p += 1
        elif c == 58:      # ':': zero or more consonants
            while p < len(w) and _is_cons_letter(w[p]):
                p += 1
        elif c == 94:      # '^': one consonant
            if p >= len(w) or not _is_cons_letter(w[p]):
                return False
            p += 1
        elif c == 43:      # '+': E I Y
            if p >= len(w) or not (w[p] == 69 or w[p] == 73 or w[p] == 89):
                return False
            p += 1
        elif c == 37:      # '%': suffix
            var ok = False
            var sufs: List[String] = ["ING ", "ELY ", "ER", "ES ", "ED ", "E "]
            for s in sufs:
                var sb = s.as_bytes()
                var good = True
                for q in range(len(sb)):
                    if p + q >= len(w) or w[p + q] != Int(sb[q]):
                        good = False
                        break
                if good:
                    p += len(sb) - (1 if Int(sb[len(sb) - 1]) == 32 else 0)
                    ok = True
                    break
            if not ok:
                return False
        elif c == 64:      # '@': T S R D L Z N J
            if p >= len(w) or not (w[p] == 84 or w[p] == 83 or w[p] == 82 or w[p] == 68 or w[p] == 76
                                   or w[p] == 90 or w[p] == 78 or w[p] == 74):
                return False
            p += 1
        else:
            if p >= len(w) or w[p] != c:
                return False
            p += 1
        i += 1
    return True


def _match_left(w: List[Int], end: Int, ctx: String) -> Bool:
    """Match `ctx` so that it ends right before w[end] (scanning leftwards)."""
    var p = end - 1
    var cb = ctx.as_bytes()
    var i = len(cb) - 1
    while i >= 0:
        var c = Int(cb[i])
        if c == 32:
            if p < 0 or w[p] != 32:
                return False
            p -= 1
        elif c == 35:
            if p < 0 or not _is_vowel_letter(w[p]):
                return False
            while p >= 0 and _is_vowel_letter(w[p]):
                p -= 1
        elif c == 58:
            while p >= 0 and _is_cons_letter(w[p]):
                p -= 1
        elif c == 94:
            if p < 0 or not _is_cons_letter(w[p]):
                return False
            p -= 1
        elif c == 43:
            if p < 0 or not (w[p] == 69 or w[p] == 73 or w[p] == 89):
                return False
            p -= 1
        elif c == 64:
            if p < 0 or not (w[p] == 84 or w[p] == 83 or w[p] == 82 or w[p] == 68 or w[p] == 76
                             or w[p] == 90 or w[p] == 78 or w[p] == 74):
                return False
            p -= 1
        else:
            if p < 0 or w[p] != c:
                return False
            p -= 1
        i -= 1
    return True


def _syllable_vowel(c: Int) -> Bool:
    return _cp_in(c, VOWELS) or c == 0x1D4A


def _stress_syllable(word: String, nsyl: Int) -> Int:
    """Which syllable (0-based) takes primary stress, from the spelling."""
    if nsyl <= 1:
        return 0
    var w = _lower(word)
    if nsyl >= 3:
        var ante: List[String] = ["ity", "ical", "ogy", "graphy", "ify", "ometer", "ious", "eous", "ial", "ian", "ia"]
        for s in ante:
            if w.endswith(s):
                return max(0, nsyl - 3) if s == "ity" or s == "ical" or s == "ogy" or s == "graphy" or s == "ometer" \
                    or s == "ify" else max(0, nsyl - 2 - (1 if s == "ia" or s == "ian" else 0))
    var pen: List[String] = ["tion", "sion", "cian", "ic", "ics", "ade", "oon", "eer", "ese", "ette", "ique", "esque"]
    for s in pen:
        if w.endswith(s):
            if s == "ade" or s == "oon" or s == "eer" or s == "ese" or s == "ette" or s == "ique" or s == "esque":
                return nsyl - 1
            return max(0, nsyl - 2)
    if nsyl == 2:
        var pre: List[String] = ["a", "be", "con", "com", "ex", "un", "dis", "mis", "en", "em", "im", "in", "ob"]
        for p in pre:
            if w.startswith(p) and w.byte_length() > p.byte_length() + 2:
                if p == "a" or p == "in" or p == "en" or p == "em" or p == "im" or p == "un" or p == "ex":
                    # only when followed by a consonant ("ago", "unlit"); "apple" stays initial
                    var nb = _byte(w, p.byte_length())
                    if nb == 97 or nb == 101 or nb == 105 or nb == 111 or nb == 117:
                        continue
                    if p == "a" and _byte(w, 1) == _byte(w, 2):
                        continue
                return 1
    return 0


def _assign_stress(word: String, raw: List[Int]) -> String:
    """Insert a primary stress, reduce unstressed vowels, flap t (US)."""
    var nuc = List[Int]()
    for i in range(len(raw)):
        if _syllable_vowel(raw[i]) and raw[i] != 0x1D4A:
            nuc.append(i)
    if len(nuc) == 0:
        return encode_utf8(raw)
    var s = min(_stress_syllable(word, len(nuc)), len(nuc) - 1)
    var out = List[Int]()
    for i in range(len(raw)):
        var c = raw[i]
        if i == nuc[s]:
            out.append(PRIMARY)
        elif len(nuc) > 1 and (c == 0xE6 or c == 0x28C or c == 0x25B or c == 0x251) and i != nuc[s]:
            # unstressed æ ʌ ɛ ɑ -> ə (keep word-initial short vowels)
            if i > 0:
                c = 0x259
        out.append(c)
    # flap: t between a vowel (or ɹ) and an unstressed vowel
    for i in range(1, len(out) - 1):
        if out[i] == 116 and _cp_in(out[i - 1], US_TAUS) and (out[i + 1] == 0x259 or out[i + 1] == 0x26A
                                                              or out[i + 1] == 105 or out[i + 1] == 0x1D4A):
            out[i] = 84
    return encode_utf8(out)


# --------------------------------------------------------------------------------------- G2P

struct G2P(Movable):
    var gold: Dict[String, String]
    var silver: Dict[String, String]
    var tagged: Dict[String, String]      # word + "|" + TAG -> phonemes (NONE_PS = null)
    var british: Bool
    var rules: List[_Rule]
    var rule_start: List[Int]             # first rule index per letter A..Z, then ' (26)
    var flap_t: Bool                      # misaki v1 (Kokoro 1.0): ɾ -> T and ʔ -> t

    def __init__(out self, british: Bool = False):
        self.gold = Dict[String, String]()
        self.silver = Dict[String, String]()
        self.tagged = Dict[String, String]()
        self.british = british
        # bucket the rules by first letter (stable), so each letter's rules are contiguous
        var all = _letter_rules()
        self.rules = List[_Rule]()
        for slot in range(27):
            for r in all:
                var c0 = _byte(r.pat, 0)
                if (c0 - 65 if _is_upper_b(c0) else 26) == slot:
                    self.rules.append(r.copy())
        self.rule_start = List[Int](length=28, fill=-1)
        self.flap_t = True
        for i in range(len(self.rules)):
            var c = _byte(self.rules[i].pat, 0)
            var slot = c - 65 if _is_upper_b(c) else 26
            if self.rule_start[slot] < 0:
                self.rule_start[slot] = i

    @staticmethod
    def load(dir: String, british: Bool = False) raises -> G2P:
        """Read misaki's `us_gold.json` / `us_silver.json` (`gb_*` when british) from `dir`."""
        var g = G2P(british)
        var pre = "gb" if british else "us"
        g._load_json(dir + "/" + pre + "_gold.json", True)
        g._load_json(dir + "/" + pre + "_silver.json", False)
        return g^

    def _load_json(mut self, path: String, gold: Bool) raises:
        var bytes = read_bytes(path)
        if len(bytes) == 0:
            raise Error("G2P: cannot read lexicon " + path)
        var doc = parse_json(bytes)
        if not doc.valid or doc.root < 0 or doc.nodes[doc.root].kind != 5:
            raise Error("G2P: " + path + " is not a JSON object")
        ref root = doc.nodes[doc.root]
        for i in range(len(root.keys)):
            ref v = doc.nodes[root.children[i]]
            var key = root.keys[i]
            if v.kind == 3:
                if gold:
                    self.gold[key] = v.text
                else:
                    self.silver[key] = v.text
            elif v.kind == 5:
                var dflt = String(NONE_PS)
                for j in range(len(v.keys)):
                    ref t = doc.nodes[v.children[j]]
                    var ps = t.text if t.kind == 3 else String(NONE_PS)
                    if v.keys[j] == "DEFAULT":
                        dflt = ps
                    self.tagged[key + "|" + v.keys[j]] = ps
                if gold:
                    self.gold[key] = dflt
                else:
                    self.silver[key] = dflt

    def add_word(mut self, word: String, phonemes: String):
        """Add or override a gold entry (e.g. a character or place name)."""
        self.gold[word] = phonemes

    def lexicon_size(self) -> Int:
        return len(self.gold) + len(self.silver)

    # ---------------------------------------------------------------- dictionary access

    def _resolve(self, gold: Bool, w: String) -> String:
        """The key `w` is stored under, with misaki's `grow_dictionary` case variants, or ""."""
        var has = (w in self.gold) if gold else (w in self.silver)
        if has:
            return w
        if w.byte_length() < 2:
            return String()
        var lw = _lower(w)
        if w == _capitalize(lw) and lw != w:
            # "Hello" finds "hello" (a lowercase key that capitalizes differently)
            var found = (lw in self.gold) if gold else (lw in self.silver)
            if found:
                return lw
        if w == lw:
            var cap = _capitalize(w)
            if cap != w:
                var found = (cap in self.gold) if gold else (cap in self.silver)
                if found:
                    return cap
        return String()

    def _in_gold(self, w: String) -> Bool:
        return self._resolve(True, w).byte_length() > 0

    def _in_silver(self, w: String) -> Bool:
        return self._resolve(False, w).byte_length() > 0

    def _get(self, gold: Bool, w: String, tag: String, future_vowel: Int) raises -> String:
        """Entry text ("" when absent, NONE_PS when null), picking the tagged variant."""
        var key = self._resolve(gold, w)
        if key.byte_length() == 0:
            return String()
        var base = self.gold[key] if gold else self.silver[key]
        if gold and (key + "|DEFAULT") in self.tagged:
            var t = tag
            if future_vowel < 0 and (key + "|None") in self.tagged:
                t = "None"
            elif not ((key + "|" + t) in self.tagged):
                t = _parent_tag(t)
            var k2 = key + "|" + t
            if k2 in self.tagged:
                return self.tagged[k2]
            return self.tagged[key + "|DEFAULT"]
        return base

    def _gold_plain(self, w: String) raises -> String:
        var k = self._resolve(True, w)
        if k.byte_length() == 0:
            return String()
        return self.gold[k]

    # ----------------------------------------------------------------------- lexicon logic

    def get_nnp(self, word: String) raises -> String:
        """Spell a word letter by letter (acronyms), "" when a letter is unknown."""
        var ps = String()
        for b in word.as_bytes():
            var c = Int(b)
            if not _is_alpha_b(c):
                continue
            var l = self._gold_plain(_upper(chr(c)))
            if l.byte_length() == 0 or l == NONE_PS:
                return String()
            ps += l
        if ps.byte_length() == 0:
            return String()
        ps = apply_stress(ps, 0)
        # last secondary becomes the primary
        var codes = decode_utf8(ps)
        var last = -1
        for i in range(len(codes)):
            if codes[i] == SECONDARY:
                last = i
        if last >= 0:
            codes[last] = PRIMARY
        return encode_utf8(codes)

    def is_known(self, word: String) -> Bool:
        if self._in_gold(word) or self._in_silver(word) or _symbol_word(word).byte_length() > 0:
            return True
        if not _is_alpha(_no_apos(word)) or not _lexicon_chars(word):
            return False
        if word.byte_length() == 1:
            return True
        if word == _upper(word) and self._in_gold(_lower(word)):
            return True
        var rest = _sub(word, 1, word.byte_length())
        return rest == _upper(rest)

    def lookup(self, word_in: String, tag: String, stress: Float64, future_vowel: Int) raises -> String:
        var word = word_in
        var is_nnp = False
        if word == _upper(word) and not self._in_gold(word):
            word = _lower(word)
            is_nnp = tag == "NNP"
        var ps = self._get(True, word, tag, future_vowel)
        if ps.byte_length() == 0 and not is_nnp:
            ps = self._get(False, word, tag, future_vowel)
        if ps.byte_length() == 0 or ps == NONE_PS or (is_nnp and not _has_cp(ps, PRIMARY)):
            var n = self.get_nnp(word)
            if n.byte_length() > 0:
                return n
            if ps == NONE_PS:
                return String()
        if ps.byte_length() == 0:
            return String()
        return apply_stress(ps, stress)

    def _s(self, stem: String) -> String:
        if stem.byte_length() == 0:
            return String()
        var c = _last_cp(stem)
        if _cp_in(c, "ptkfθ"):
            return stem + "s"
        if _cp_in(c, "szʃʒʧʤ"):
            return stem + ("ɪ" if self.british else "ᵻ") + "z"
        return stem + "z"

    def stem_s(self, word: String, tag: String, stress: Float64, fv: Int) raises -> String:
        var n = word.byte_length()
        if n < 3 or not word.endswith("s"):
            return String()
        var stem: String
        if not word.endswith("ss") and self.is_known(_sub(word, 0, n - 1)):
            stem = _sub(word, 0, n - 1)
        elif (word.endswith("'s") or (n > 4 and word.endswith("es") and not word.endswith("ies"))) \
                and self.is_known(_sub(word, 0, n - 2)):
            stem = _sub(word, 0, n - 2)
        elif n > 4 and word.endswith("ies") and self.is_known(_sub(word, 0, n - 3) + "y"):
            stem = _sub(word, 0, n - 3) + "y"
        else:
            return String()
        return self._s(self.lookup(stem, tag, stress, fv))

    def _ed(self, stem: String) -> String:
        if stem.byte_length() == 0:
            return String()
        var c = _last_cp(stem)
        if _cp_in(c, "pkfθʃsʧ"):
            return stem + "t"
        if c == 100:
            return stem + ("ɪ" if self.british else "ᵻ") + "d"
        if c != 116:
            return stem + "d"
        var codes = decode_utf8(stem)
        if self.british or len(codes) < 2:
            return stem + "ɪd"
        if _cp_in(codes[len(codes) - 2], US_TAUS):
            return _drop_last(stem) + "ɾᵻd"
        return stem + "ᵻd"

    def stem_ed(self, word: String, tag: String, stress: Float64, fv: Int) raises -> String:
        var n = word.byte_length()
        if n < 4 or not word.endswith("d"):
            return String()
        var stem: String
        if not word.endswith("dd") and self.is_known(_sub(word, 0, n - 1)):
            stem = _sub(word, 0, n - 1)
        elif n > 4 and word.endswith("ed") and not word.endswith("eed") and self.is_known(_sub(word, 0, n - 2)):
            stem = _sub(word, 0, n - 2)
        else:
            return String()
        return self._ed(self.lookup(stem, tag, stress, fv))

    def _ing(self, stem: String) -> String:
        if stem.byte_length() == 0:
            return String()
        var codes = decode_utf8(stem)
        if self.british:
            var c = codes[len(codes) - 1]
            if c == 0x259 or c == 0x2D0:
                return String()
        elif len(codes) > 1 and codes[len(codes) - 1] == 116 and _cp_in(codes[len(codes) - 2], US_TAUS):
            return _drop_last(stem) + "ɾɪŋ"
        return stem + "ɪŋ"

    def stem_ing(self, word: String, tag: String, stress: Float64, fv: Int) raises -> String:
        var n = word.byte_length()
        if n < 5 or not word.endswith("ing"):
            return String()
        var stem: String
        if n > 5 and self.is_known(_sub(word, 0, n - 3)):
            stem = _sub(word, 0, n - 3)
        elif self.is_known(_sub(word, 0, n - 3) + "e"):
            stem = _sub(word, 0, n - 3) + "e"
        elif n > 5 and _doubled_ing(word) and self.is_known(_sub(word, 0, n - 4)):
            stem = _sub(word, 0, n - 4)
        else:
            return String()
        return self._ing(self.lookup(stem, tag, stress, fv))

    def special_case(self, word: String, tag: String, stress: Float64, fv: Int, future_to: Bool) raises -> String:
        var sym = _symbol_word(word)
        if sym.byte_length() > 0:
            return self.lookup(sym, String(), NO_STRESS, fv)
        if word == "a" or word == "A":
            return "ɐ" if tag == "DT" else "ˈA"
        if word == "am" or word == "Am" or word == "AM":
            if tag.startswith("NN"):
                return self.get_nnp(word)
            if fv < 0 or word != "am" or (stress != NO_STRESS and stress > 0):
                return self._gold_plain("am")
            return "ɐm"
        if word == "an" or word == "An" or word == "AN":
            if word == "AN" and tag.startswith("NN"):
                return self.get_nnp(word)
            return "ɐn"
        if word == "I" and tag == "PRP":
            return "ˌI"
        if word == "to" or word == "To" or (word == "TO" and (tag == "TO" or tag == "IN")):
            if fv < 0:
                return self._gold_plain("to")
            return "tʊ" if fv == 1 else "tə"
        if word == "in" or word == "In" or (word == "IN" and tag != "NNP"):
            return ("ˈ" if fv < 0 or tag != "IN" else "") + "ɪn"
        if word == "the" or word == "The" or (word == "THE" and tag == "DT"):
            return "ði" if fv == 1 else "ðə"
        if tag == "IN" and (_lower(word) == "vs" or _lower(word) == "vs."):
            return self.lookup("versus", String(), NO_STRESS, fv)
        if word == "used" or word == "Used" or word == "USED":
            if future_to and "used|VBD" in self.tagged:
                return self.tagged["used|VBD"]
            if "used|DEFAULT" in self.tagged:
                return self.tagged["used|DEFAULT"]
        return String()

    def get_word(self, word_in: String, tag: String, stress: Float64, fv: Int, future_to: Bool) raises -> String:
        var word = word_in
        var sp = self.special_case(word, tag, stress, fv, future_to)
        if sp.byte_length() > 0:
            return sp
        var wl = _lower(word)
        if word.byte_length() > 1 and _is_alpha(_no_apos(word)) and word != wl and (tag != "NNP" or word.byte_length() > 7) \
                and not (word in self.gold) and not (word in self.silver) \
                and (word == _upper(word) or _sub(word, 1, word.byte_length()) == _lower(_sub(word, 1, word.byte_length()))):
            if self._in_gold(wl) or self._in_silver(wl) or self.stem_s(wl, tag, stress, fv).byte_length() > 0 \
                    or self.stem_ed(wl, tag, stress, fv).byte_length() > 0 or self.stem_ing(wl, tag, stress, fv).byte_length() > 0:
                word = wl
        if self.is_known(word):
            return self.lookup(word, tag, stress, fv)
        var n = word.byte_length()
        if word.endswith("s'") and self.is_known(_sub(word, 0, n - 2) + "'s"):
            return self.lookup(_sub(word, 0, n - 2) + "'s", tag, stress, fv)
        if word.endswith("'") and self.is_known(_sub(word, 0, n - 1)):
            return self.lookup(_sub(word, 0, n - 1), tag, stress, fv)
        var r = self.stem_s(word, tag, stress, fv)
        if r.byte_length() > 0:
            return r
        r = self.stem_ed(word, tag, stress, fv)
        if r.byte_length() > 0:
            return r
        return self.stem_ing(word, tag, 0.5 if stress == NO_STRESS else stress, fv)

    def _words_ps(self, words: List[String]) raises -> String:
        var out = String()
        for i in range(len(words)):
            var p = self.lookup(words[i], String(), -2.0 if words[i] == "point" else NO_STRESS, -1)
            if p.byte_length() == 0:
                p = self.spell(words[i])
            if out.byte_length() > 0:
                out += " "
            out += p
        return out

    def number(self, text: String, currency: Int) raises -> String:
        """Pronounce a number token: 12, 1,200, 3.14, 1987, 21st, 1980s, -5, with $ / £ / €."""
        var word = text
        var suffix = String()
        var n = word.byte_length()
        var k = n
        while k > 0 and (_is_lower_b(_byte(word, k - 1)) or _byte(word, k - 1) == 39):
            k -= 1
        if k < n:
            suffix = _sub(word, k, n)
            word = _sub(word, 0, k)
        var neg = False
        if word.startswith("-"):
            neg = True
            word = _sub(word, 1, word.byte_length())
        var clean = word.replace(",", "")
        var words = List[String]()
        if neg:
            words.append("minus")
        var is_ord = suffix == "st" or suffix == "nd" or suffix == "rd" or suffix == "th"
        if _is_digits(clean) and is_ord:
            for w in ordinal_words(Int(clean)):
                words.append(w)
        elif not neg and word.byte_length() == 4 and currency == 0 and _is_digits(word):
            for w in year_words(Int(word)):
                words.append(w)
        elif currency != 0 and clean.count(".") <= 1 and _is_currency(clean):
            var unit = "dollar" if currency == 1 else ("pound" if currency == 2 else "euro")
            var sub_unit = "pence" if currency == 2 else "cent"
            var parts = clean.split(".")
            var major = Int(String(parts[0])) if String(parts[0]).byte_length() > 0 else 0
            var minor = 0
            if len(parts) > 1 and String(parts[1]).byte_length() > 0:
                var ms = String(parts[1])
                minor = Int(ms) * (10 if ms.byte_length() == 1 else 1)
            var ps = String()
            if major != 0 or minor == 0:
                ps = self._words_ps(cardinal_words(major))
                ps += " " + (self.lookup(unit, String(), NO_STRESS, -1) if major == 1 else self._s(self.lookup(unit, String(), NO_STRESS, -1)))
            if minor != 0:
                if ps.byte_length() > 0:
                    ps += " " + self.lookup("and", String(), NO_STRESS, -1) + " "
                ps += self._words_ps(cardinal_words(minor))
                var u = self.lookup(sub_unit, String(), NO_STRESS, -1)
                ps += " " + (u if minor == 1 or currency == 2 else self._s(u))
            return ps
        elif _is_digits(clean):
            for w in cardinal_words(Int(clean)):
                words.append(w)
        elif clean.count(".") == 1:
            var parts = clean.split(".")
            var ip = String(parts[0])
            if ip.byte_length() > 0:
                for w in cardinal_words(Int(ip)):
                    words.append(w)
            else:
                words.append("zero")
            words.append("point")
            var fp = String(parts[1])
            for b in fp.as_bytes():
                words.append(_ones_word(Int(b) - 48))
        else:
            # dotted groups (version numbers, IPs): digits one by one around "point"
            var first = True
            for part in clean.split("."):
                if not first:
                    words.append("point")
                first = False
                for b in String(part).as_bytes():
                    if _is_digit_b(Int(b)):
                        words.append(_ones_word(Int(b) - 48))
        var ps = self._words_ps(words)
        if currency != 0:
            var unit = "dollar" if currency == 1 else ("pound" if currency == 2 else "euro")
            ps += " " + self._s(self.lookup(unit, String(), NO_STRESS, -1))
        if suffix == "s" or suffix == "'s":
            return self._s(ps)
        if suffix == "ed" or suffix == "'d":
            return self._ed(ps)
        if suffix == "ing":
            return self._ing(ps)
        return ps

    # ------------------------------------------------------------------------ fallbacks

    def compound(self, word: String, depth: Int = 0) raises -> String:
        """Split an unknown word into lexicon words ("outofdictionary" -> out of dictionary);
        the first part keeps its primary stress, later ones get secondary. "" when no split."""
        var w = _lower(word)
        var n = w.byte_length()
        if n < 4 or depth > 2 or not _is_alpha(w):
            return String()
        var i = n - 2
        while i >= 2:
            var left = _sub(w, 0, i)
            var right = _sub(w, i, n)
            if (self._in_gold(left) or self._in_silver(left)) and left.byte_length() >= 2:
                var rp = String()
                if (self._in_gold(right) or self._in_silver(right)) and right.byte_length() >= 2:
                    rp = self.lookup(right, String(), NO_STRESS, -1)
                if rp.byte_length() == 0:
                    rp = self.compound(right, depth + 1)
                if rp.byte_length() > 0:
                    var lp = self.lookup(left, String(), NO_STRESS, -1)
                    if lp.byte_length() > 0:
                        if not _has_cp(lp, PRIMARY):
                            lp = apply_stress(lp, 2.0)
                        return lp + apply_stress(rp, -1)
            i -= 1
        return String()

    def partial_compound(self, word: String) raises -> String:
        """A known part of 4+ letters plus a rule-spelled rest ("Gorthmonger" -> gorth + monger),
        the longest known part winning; "" when there is none."""
        var w = _lower(word)
        var n = w.byte_length()
        if n < 6 or not _is_alpha(w):
            return String()
        var best = -1
        var best_left = False
        for i in range(2, n - 1):
            var right = _sub(w, i, n)
            if n - i >= 4 and n - i > best and (self._in_gold(right) or self._in_silver(right)):
                best = n - i
                best_left = False
                break
        for i in range(n - 2, 3, -1):
            var left = _sub(w, 0, i)
            if i > best and (self._in_gold(left) or self._in_silver(left)):
                best = i
                best_left = True
                break
        if best < 0:
            return String()
        if best_left:
            var lp = self.lookup(_sub(w, 0, best), String(), NO_STRESS, -1)
            if not _has_cp(lp, PRIMARY):
                lp = apply_stress(lp, 2.0)
            return lp + apply_stress(self.spell(_sub(w, best, n)), -1)
        var rp = self.lookup(_sub(w, n - best, n), String(), NO_STRESS, -1)
        return self.spell(_sub(w, 0, n - best)) + apply_stress(rp, -1)

    def spell(self, word: String) raises -> String:
        """Letter-to-sound rules + stress heuristics for a word the lexicon does not have."""
        var up = _upper(word)
        var w = List[Int]()
        w.append(32)
        for b in up.as_bytes():
            var c = Int(b)
            if _is_upper_b(c) or c == 39:
                w.append(c)
        w.append(32)
        var raw = List[Int]()
        var p = 1
        while p < len(w) - 1:
            var c = w[p]
            var slot = c - 65 if _is_upper_b(c) else 26
            var done = False
            var r = self.rule_start[slot]
            while r >= 0 and r < len(self.rules):
                ref rule = self.rules[r]
                var mb = rule.pat.as_bytes()
                if Int(mb[0]) != c:
                    break
                var ok = p + len(mb) <= len(w)
                if ok:
                    for q in range(len(mb)):
                        if w[p + q] != Int(mb[q]):
                            ok = False
                            break
                if ok and _match_left(w, p, rule.left) and _match_right(w, p + len(mb), rule.right):
                    for x in decode_utf8(rule.out):
                        raw.append(x)
                    p += len(mb)
                    done = True
                    break
                r += 1
            if not done:
                p += 1
        return _assign_stress(word, raw)

    # ------------------------------------------------------------------------ tokenizer

    def _tokenize(self, text: String) -> List[_Tok]:
        var codes = _fold_accents(decode_utf8(text))
        var toks = List[_Tok]()
        var i = 0
        var n = len(codes)
        var currency = 0
        while i < n:
            var c = codes[i]
            # [text](/phonemes/) pins a pronunciation
            if c == 91:
                var close = -1
                for j in range(i + 1, n):
                    if codes[j] == 93:
                        close = j
                        break
                if close > 0 and close + 2 < n and codes[close + 1] == 40 and codes[close + 2] == 47:
                    var end = -1
                    for j in range(close + 3, n):
                        if codes[j] == 41:
                            end = j
                            break
                    if end > 0:
                        var ps_end = end
                        if codes[ps_end - 1] == 47:
                            ps_end -= 1
                        var t = _Tok(encode_utf8(codes, i + 1, close), TK_FIXED)
                        t.ps = encode_utf8(codes, close + 3, ps_end)
                        t.has_ps = True
                        t.known = True
                        toks.append(t^)
                        i = end + 1
                        continue
            if c == 32 or c == 9 or c == 10 or c == 13:
                if len(toks) > 0:
                    toks[len(toks) - 1].ws = True
                i += 1
                continue
            if c == 36 or c == 0xA3 or c == 0x20AC:
                currency = 1 if c == 36 else (2 if c == 0xA3 else 3)
                i += 1
                continue
            var start = i
            if _is_digit_b(c) or (c == 45 and i + 1 < n and _is_digit_b(codes[i + 1]) and (i == 0 or codes[i - 1] == 32)):
                i += 1
                while i < n and (_is_digit_b(codes[i]) or ((codes[i] == 44 or codes[i] == 46) and i + 1 < n and _is_digit_b(codes[i + 1]))):
                    i += 1
                # ordinal / plural suffix: 21st, 1980s, 90's
                var j = i
                while j < n and (_is_lower_b(codes[j]) or (codes[j] == 39 and j + 1 < n and _is_lower_b(codes[j + 1]))):
                    j += 1
                var suf = encode_utf8(codes, i, j)
                if suf == "st" or suf == "nd" or suf == "rd" or suf == "th" or suf == "s" or suf == "'s":
                    i = j
                var t = _Tok(encode_utf8(codes, start, i), TK_NUM)
                t.currency = currency
                t.joined = len(toks) > 0 and not toks[len(toks) - 1].ws and toks[len(toks) - 1].kind != TK_PUNCT
                currency = 0
                toks.append(t^)
                continue
            if _is_alpha_b(c):
                i += 1
                while i < n and (_is_alpha_b(codes[i]) or (codes[i] == 39 and i + 1 < n and _is_alpha_b(codes[i + 1]))):
                    # camelCase split: lower followed by upper
                    if _is_upper_b(codes[i]) and _is_lower_b(codes[i - 1]):
                        break
                    i += 1
                # trailing possessive apostrophe (players')
                if i < n and codes[i] == 39 and codes[i - 1] == 115 and (i + 1 >= n or not _is_alpha_b(codes[i + 1])):
                    i += 1
                var word = encode_utf8(codes, start, i)
                # abbreviations stored with their period (Mr. Dr. etc.)
                if i < n and codes[i] == 46 and self._in_gold(word + "."):
                    word += "."
                    i += 1
                var t = _Tok(word, TK_WORD)
                t.joined = len(toks) > 0 and not toks[len(toks) - 1].ws and \
                    (toks[len(toks) - 1].kind == TK_WORD or toks[len(toks) - 1].kind == TK_NUM)
                toks.append(t^)
                continue
            i += 1
            if c == 45 or c == 0x2014 or c == 0x2013:
                # a hyphen inside a word glues the parts; a free-standing dash is a pause
                var inside = start > 0 and start + 1 < n and codes[start - 1] != 32 and codes[start + 1] != 32
                if inside:
                    continue
                var t = _Tok("—", TK_PUNCT)
                t.ps = "—"
                t.has_ps = True
                toks.append(t^)
                continue
            var sym = _symbol_word(chr(c)) if c < 128 else String()
            if sym.byte_length() > 0:
                var t = _Tok(chr(c), TK_WORD)
                toks.append(t^)
                continue
            if _cp_in(c, PUNCTS) or c == 40 or c == 41:
                var ps = encode_utf8([c])
                if c == 34:
                    # straight quote: opening after a space / start, closing otherwise
                    var opening = start == 0 or codes[start - 1] == 32 or codes[start - 1] == 40
                    ps = "“" if opening else "”"
                if c == 46 and i + 1 < n and codes[i] == 46 and codes[i + 1] == 46:
                    ps = "…"
                    i += 2
                var t = _Tok(ps, TK_PUNCT)
                t.ps = ps
                t.has_ps = True
                toks.append(t^)
                continue
            # anything else is dropped
        return toks^

    def _tag(mut self, mut toks: List[_Tok]):
        """Heuristic part-of-speech tags for the words where misaki's lexicon uses them."""
        for k in range(len(toks)):
            if toks[k].kind != TK_WORD:
                continue
            var w = toks[k].text
            var lw = _lower(w)
            var prev = String()
            var j = k - 1
            if j >= 0 and toks[j].kind == TK_WORD:
                prev = _lower(toks[j].text)
            var next = String()
            if k + 1 < len(toks) and toks[k + 1].kind == TK_WORD:
                next = _lower(toks[k + 1].text)
            var t = String()
            if lw == "a" or lw == "the" or lw == "an":
                t = "DT"
            elif w == "I":
                t = "PRP"
            elif lw == "to":
                t = "TO"
            elif lw == "in" or lw == "vs" or lw == "vs.":
                t = "IN"
            elif w.byte_length() > 1 and w == _upper(w) and _is_alpha(_no_apos(w)):
                t = "NNP"
            elif _is_one_of(prev, "to will would can could shall should may might must do does did don't doesn't didn't won't can't cannot let's please i'll you'll we'll they'll"):
                t = "VB"
            elif _is_one_of(prev, "have has had having i've you've we've they've"):
                t = "VBN"
            elif _is_one_of(prev, "i you we they"):
                t = "VBP"
            elif _is_one_of(prev, "he she it"):
                # a present form after he / she / it ends in -s; anything else is past ("he read")
                t = "VBZ" if lw.endswith("s") else "VBD"
            elif (lw == "that" or lw == "this") and next.byte_length() > 0 and not _is_one_of(next, "i you he she it we they is are was were will would can could should has have had does did do the a an my your his her our their this that there here who which what when where why how to of in on at for with by as and or but if so not"):
                t = "DT"
            elif _is_one_of(prev, "the a an this that these those my your his her its our their some any no every each one"):
                t = "NN"
            elif _is_one_of(prev, "is are was were be been very so too more most quite rather"):
                t = "JJ"
            elif prev.byte_length() == 0 and _is_one_of(next, "the a an this that these those my your his her our their it me us them"):
                t = "VB"
            toks[k].tag = t

    def phonemize(mut self, text: String) raises -> G2PResult:
        var toks = self._tokenize(text)
        self._tag(toks)
        # right to left: each word sees the next word's first sound
        var fv = -1          # future vowel: -1 unknown / pause, 0 consonant, 1 vowel
        var future_to = False
        var k = len(toks) - 1
        while k >= 0:
            if not toks[k].has_ps:
                var ps: String
                var known = True
                if toks[k].kind == TK_NUM:
                    ps = self.number(toks[k].text, toks[k].currency)
                else:
                    var w = toks[k].text
                    var stress = NO_STRESS
                    if w != _lower(w):
                        stress = 2.0 if w == _upper(w) else 0.5
                    ps = self.get_word(w, toks[k].tag, stress, fv, future_to)
                    if ps.byte_length() == 0:
                        ps = self.compound(w)
                    if ps.byte_length() == 0:
                        ps = self.partial_compound(w)
                        known = False
                    if ps.byte_length() == 0:
                        ps = self.spell(w)
                toks[k].ps = ps
                toks[k].has_ps = True
                toks[k].known = known
            # context for the word before this one
            if toks[k].ps.byte_length() > 0:
                for c in decode_utf8(toks[k].ps):
                    if _cp_in(c, NON_QUOTE_PUNCTS):
                        fv = -1
                        break
                    if _cp_in(c, VOWELS):
                        fv = 1
                        break
                    if _cp_in(c, CONSONANTS):
                        fv = 0
                        break
            var lt = toks[k].text
            future_to = lt == "to" or lt == "To" or lt == "TO"
            k -= 1
        # glued parts of one written word (hyphen, camelCase): demote the lighter half
        var g = 0
        while g < len(toks):
            var e = g + 1
            while e < len(toks) and toks[e].joined:
                e += 1
            if e - g >= 2:
                _resolve_group(toks, g, e)
            g = e
        # join
        var res = G2PResult()
        var out = List[Int]()
        var w0 = 0
        while w0 < len(toks):
            var w1 = w0 + 1
            while w1 < len(toks) and toks[w1].joined:
                w1 += 1
            var start = len(out)
            var text_ = String()
            var known = True
            for q in range(w0, w1):
                var ps = toks[q].ps
                if self.flap_t:
                    ps = _replace_cp(_replace_cp(ps, 0x27E, 84), 0x294, 116)
                for c in decode_utf8(ps):
                    out.append(c)
                text_ += toks[q].text
                known = known and toks[q].known
            if toks[w0].kind != TK_PUNCT:
                res.words.append(G2PWord(text_, encode_utf8(out, start, len(out)), start, len(out), known))
            if toks[w1 - 1].ws and w1 < len(toks):
                out.append(32)
            w0 = w1
        res.phonemes = encode_utf8(out)
        return res^


def _resolve_group(mut toks: List[_Tok], g: Int, e: Int):
    """misaki `resolve_tokens` for glued sub-words: demote the lighter half of the stressed parts."""
    var idx = List[Int]()
    for q in range(g, e):
        if toks[q].ps.byte_length() > 0:
            idx.append(q)
    if len(idx) < 2:
        return
    var nprim = 0
    for q in idx:
        if _has_cp(toks[q].ps, PRIMARY):
            nprim += 1
    if len(idx) == 2 and toks[idx[0]].text.byte_length() == 1:
        toks[idx[1]].ps = apply_stress(toks[idx[1]].ps, -0.5)
        return
    if nprim <= (len(idx) + 1) // 2:
        return
    # sort by (has primary, weight, index), demote the first half
    var keys = List[Int]()
    for q in idx:
        keys.append((1 if _has_cp(toks[q].ps, PRIMARY) else 0) * 1000000 + stress_weight(toks[q].ps) * 1000 + q)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            if keys[b] < keys[a]:
                var t = keys[a]
                keys[a] = keys[b]
                keys[b] = t
    for a in range(len(keys) // 2):
        var q = keys[a] % 1000
        toks[q].ps = apply_stress(toks[q].ps, -0.5)


def _parent_tag(tag: String) -> String:
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("NN"):
        return "NOUN"
    if tag.startswith("ADV") or tag.startswith("RB"):
        return "ADV"
    if tag.startswith("ADJ") or tag.startswith("JJ"):
        return "ADJ"
    return tag


def _symbol_word(w: String) -> String:
    if w == "%": return "percent"
    if w == "&": return "and"
    if w == "+": return "plus"
    if w == "@": return "at"
    return String()


def _is_one_of(w: String, words: String) -> Bool:
    if w.byte_length() == 0:
        return False
    for x in words.split(" "):
        if String(x) == w:
            return True
    return False


def _doubled_ing(word: String) -> Bool:
    """`([bcdgklmnprstvxz])\\1ing$|cking$`"""
    var n = word.byte_length()
    if n >= 5 and word.endswith("cking"):
        return True
    if n < 5:
        return False
    var a = _byte(word, n - 5)
    var b = _byte(word, n - 4)
    return a == b and _cp_in(a, "bcdgklmnprstvxz")


def _is_currency(word: String) -> Bool:
    if word.count(".") == 0:
        return True
    var parts = word.split(".")
    var cents = String(parts[1])
    if cents.byte_length() < 3:
        return True
    for b in cents.as_bytes():
        if Int(b) != 48:
            return False
    return True
