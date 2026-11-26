"""Public API of the Flint formant voice (`flint.audio.formant_tts.api`).

    var tts = FormantTTS()                         # rules-only G2P (+ a small built-in core lexicon)
    var tts = FormantTTS.load(lexicon_dir)         # misaki us_/gb_ gold + silver lexicon
    var pcm = tts.synthesize("Roger that.", formant_voice(FV_DEFAULT))       # 24 kHz mono
    var pcm2 = synthesize_phonemes("həlˈO wˈɜɹld.", formant_voice(FV_DEFAULT))  # raw misaki

Rendering is deterministic: the noise and voice RNGs are seeded from the phoneme string and the
preset id, and nothing reads the clock.
"""
from flint.ml.phonemize import G2P, G2PResult
from flint.text.utf8 import decode_utf8
from flint.audio.formant_tts.phonemes import PhonemeDef, Locus, build_phoneme_table, build_locus_table
from flint.audio.formant_tts.synth import KlattSynth, KlattTrack, KLATT_SR
from flint.audio.formant_tts.voices import FormantVoice, formant_voice, FV_DEFAULT, voice_eq
from flint.audio.formant_tts.rules import build_track, WF_FUNCTION, WF_WH

comptime FORMANT_TTS_SAMPLE_RATE: Int = KLATT_SR

comptime _FUNCTION_WORDS = " a an the to of in on at for by with and or but is are was were be been am do does did has have had i you he she it we they me him her us them my your his its our their this that these those not from as if so than then there here "
comptime _WH_WORDS = " what where when why who whom whose which how "


def _core_lexicon(mut g: G2P):
    """Function words and frequent irregular words for the rules-only G2P (Flint's own
    transcriptions in misaki notation; the letter-to-sound rules get these wrong)."""
    var w: List[String] = [
        "of", "ʌv", "to", "tə", "was", "wʌz", "were", "wɜɹ", "are", "ɑɹ", "is", "ɪz", "as", "æz",
        "has", "hæz", "his", "hɪz", "does", "dʌz", "do", "du", "done", "dˈʌn", "one", "wˈʌn",
        "once", "wˈʌns", "two", "tˈu", "four", "fˈɔɹ", "eight", "ˈAt", "through", "θɹu",
        "from", "fɹʌm", "for", "fɔɹ", "or", "ɔɹ", "on", "ɑn", "you", "ju", "your", "jʊɹ",
        "they", "ðA", "their", "ðɛɹ", "there", "ðɛɹ", "where", "wˈɛɹ", "what", "wˈʌt",
        "who", "hˈu", "whom", "hˈum", "whose", "hˈuz", "how", "hˈW", "why", "wˈI",
        "when", "wˈɛn", "which", "wˈɪʧ", "the", "ðə", "a", "ɐ", "an", "ɐn", "and", "ænd",
        "be", "bi", "been", "bɪn", "he", "hi", "she", "ʃi", "we", "wi", "me", "mi", "my", "mI",
        "by", "bI", "no", "nˈO", "not", "nˈɑt", "so", "sˈO", "go", "ɡˈO", "said", "sˈɛd",
        "says", "sˈɛz", "again", "əɡˈɛn", "any", "ˈɛni", "many", "mˈɛni", "some", "sʌm",
        "come", "kˈʌm", "other", "ˈʌðɚ", "mother", "mˈʌðɚ", "father", "fˈɑðɚ", "water", "wˈɔTɚ",
        "word", "wˈɜɹd", "work", "wˈɜɹk", "world", "wˈɜɹld", "give", "ɡˈɪv", "live", "lˈɪv",
        "have", "hæv", "move", "mˈuv", "love", "lˈʌv", "eye", "ˈI", "only", "ˈOnli",
        "people", "pˈipᵊl", "would", "wʊd", "could", "kʊd", "should", "ʃʊd", "all", "ˈɔl",
        "full", "fˈʊl", "pull", "pˈʊl", "put", "pˈʊt", "push", "pˈʊʃ", "busy", "bˈɪzi",
        "women", "wˈɪmᵊn", "woman", "wˈʊmᵊn", "sure", "ʃˈʊɹ", "zero", "zˈɪɹO", "seven", "sˈɛvən",
        "eleven", "əlˈɛvən", "heart", "hˈɑɹt", "fire", "fˈIɚ", "great", "ɡɹˈAt", "break", "bɹˈAk",
        "friend", "fɹˈɛnd", "head", "hˈɛd", "dead", "dˈɛd", "ready", "ɹˈɛdi", "move", "mˈuv",
    ]
    var i = 0
    while i + 1 < len(w):
        g.add_word(w[i], w[i + 1])
        i += 2


def _in_list(word: String, lst: String) -> Bool:
    return lst.find(" " + word + " ") >= 0


def word_flags(r: G2PResult) -> List[Int]:
    """Per-codepoint word flags of `r.phonemes` (function word / wh-word) for the F0 rules."""
    var n = len(decode_utf8(r.phonemes))
    var out = List[Int](length=n, fill=0)
    for w in r.words:
        var lw = w.text.lower()
        var f = 0
        if _in_list(lw, _FUNCTION_WORDS) or _in_list(lw, _WH_WORDS):
            f |= WF_FUNCTION
        if _in_list(lw, _WH_WORDS):
            f |= WF_WH
        for c in range(max(0, w.start), min(n, w.end)):
            out[c] = f
    return out^


struct FormantTTS(Movable):
    """Text or misaki phonemes -> 24 kHz mono PCM with the Klatt formant voice."""

    var g2p: G2P
    var tab: List[PhonemeDef]
    var loci: List[Locus]
    var synth: KlattSynth
    var has_lexicon: Bool

    def __init__(out self, british: Bool = False):
        """Rules-only G2P (no files) with a small built-in core lexicon."""
        self.g2p = G2P(british)
        _core_lexicon(self.g2p)
        self.tab = build_phoneme_table()
        self.loci = build_locus_table()
        self.synth = KlattSynth()
        self.has_lexicon = False

    @staticmethod
    def load(dir: String, british: Bool = False) raises -> FormantTTS:
        """With misaki's lexicon (`us_gold.json` + `us_silver.json`, or `gb_*`) from `dir`."""
        var t = FormantTTS(british)
        t.g2p = G2P.load(dir, british)
        t.has_lexicon = True
        return t^

    def phonemize(mut self, text: String) raises -> G2PResult:
        return self.g2p.phonemize(text)

    def track(self, phonemes: String, wflags: List[Int], voice: FormantVoice) -> KlattTrack:
        return build_track(phonemes, wflags, voice, self.tab, self.loci)

    def synthesize_phonemes(self, phonemes: String, voice: FormantVoice) -> List[Float32]:
        """Raw misaki phonemes (stress marks before vowels, punctuation kept)."""
        var wf = List[Int]()
        var tr = build_track(phonemes, wf, voice, self.tab, self.loci)
        return voice_eq(self.synth.render(tr), voice)

    def synthesize(mut self, text: String, voice: FormantVoice) raises -> List[Float32]:
        """English text through the G2P. Text wrapped in `/.../` is taken as raw phonemes."""
        var b = text.byte_length()
        if b >= 2 and text.startswith("/") and text.endswith("/"):
            return self.synthesize_phonemes(String(text[byte=1 : b - 1]), voice)
        var r = self.g2p.phonemize(text)
        var wf = word_flags(r)
        var tr = build_track(r.phonemes, wf, voice, self.tab, self.loci)
        return voice_eq(self.synth.render(tr), voice)


def synthesize(text: String, voice: FormantVoice) raises -> List[Float32]:
    """One-shot text synthesis with the rules-only G2P (build a `FormantTTS` to reuse tables)."""
    var tts = FormantTTS(voice.british)
    return tts.synthesize(text, voice)


def synthesize_phonemes(phonemes: String, voice: FormantVoice) -> List[Float32]:
    var tts = FormantTTS(voice.british)
    return tts.synthesize_phonemes(phonemes, voice)
