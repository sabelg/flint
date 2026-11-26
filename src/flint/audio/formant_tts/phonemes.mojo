"""Phoneme target table for the Flint formant voice (`flint.audio.formant_tts.phonemes`).

One row per synthesis phoneme (the misaki inventory plus the allophones the rules create:
r-coloured nuclei, dark / syllabic l, syllabic n / m, the flap, glottal stop and glottalised t).
Values are adult-male targets for the Klatt cascade/parallel synthesiser.

Sources (citation keys as in the design notes):
- [K80] Klatt 1980, JASA 67(3) 971-995: Table I defaults, Table II vowels (onset and offglide rows),
  Table III consonants (formants, bandwidths, source and parallel amplitudes).
- [K79] Klatt 1979 / [AHK87] MITalk Table 9-1: inherent / minimum durations (ms).
- [L62] Lehiste 1962 via [K87] Klatt 1987: dark l, postvocalic r.
- [S98] Stevens 1998 Acoustic Phonetics: neutral tube, fricative spectra at fs >= 16 kHz.
- [ENG]: engineering values derived in the notes (interpolations and 24 kHz adjustments).

Amplitudes use the Klatt dB scale: 60 is a strong stressed vowel, 0 is off.
Stops and affricates store their burst / frication spectrum in A2..A6 / AB.
"""

# ---------------------------------------------------------------------------- flags
comptime FL_VOWEL: Int = 1
comptime FL_VOICED: Int = 2
comptime FL_SON: Int = 4
comptime FL_NASAL: Int = 8
comptime FL_STOP: Int = 16
comptime FL_FRIC: Int = 32
comptime FL_AFFR: Int = 64
comptime FL_LIQ: Int = 128
comptime FL_GLIDE: Int = 256
comptime FL_DIPH: Int = 512
comptime FL_SYLL: Int = 1024      # syllabic consonant (EL EN EM)
comptime FL_REDUCED: Int = 2048   # ə ᵊ ᵻ ɐ ɚ: no fixed target, coloured by neighbours
comptime FL_PAUSE: Int = 4096
comptime FL_ASPIR: Int = 8192     # h
comptime FL_GLOTTAL: Int = 16384  # ʔ / TQ
comptime FL_FLAP: Int = 32768

# ---------------------------------------------------------------------------- places (locus rows)
comptime PL_NONE: Int = 0
comptime PL_LABIAL: Int = 1
comptime PL_LABIODENTAL: Int = 2
comptime PL_DENTAL: Int = 3
comptime PL_ALVEOLAR: Int = 4
comptime PL_POSTALV: Int = 5
comptime PL_PALATAL: Int = 6
comptime PL_VELAR: Int = 7
comptime PL_LABVEL: Int = 8
comptime PL_RHOTIC: Int = 9
comptime PL_LATERAL: Int = 10
comptime PL_DARKL: Int = 11
comptime PL_GLOTTAL: Int = 12

# ---------------------------------------------------------------------------- phoneme ids
comptime PH_I: Int = 0      # i
comptime PH_IH: Int = 1     # ɪ
comptime PH_IX: Int = 2     # ᵻ
comptime PH_E: Int = 3      # e
comptime PH_EH: Int = 4     # ɛ
comptime PH_AE: Int = 5     # æ
comptime PH_A: Int = 6      # a (GB TRAP)
comptime PH_AA: Int = 7     # ɑ
comptime PH_OH: Int = 8     # ɒ
comptime PH_AO: Int = 9     # ɔ
comptime PH_AH: Int = 10    # ʌ
comptime PH_O: Int = 11     # o
comptime PH_UH: Int = 12    # ʊ
comptime PH_U: Int = 13     # u
comptime PH_ER: Int = 14    # ɝ
comptime PH_AXR: Int = 15   # ɚ
comptime PH_ERGB: Int = 16  # ɜ (non-rhotic)
comptime PH_AX: Int = 17    # ə
comptime PH_AXT: Int = 18   # ɐ
comptime PH_SCHWA: Int = 19 # ᵊ
comptime PH_EY: Int = 20    # A  eɪ
comptime PH_AY: Int = 21    # I  aɪ
comptime PH_OW: Int = 22    # O  oʊ
comptime PH_AW: Int = 23    # W  aʊ
comptime PH_OY: Int = 24    # Y  ɔɪ
comptime PH_GOAT: Int = 25  # Q  əʊ
comptime PH_IR: Int = 26    # ɪɹ
comptime PH_EHR: Int = 27   # ɛɹ
comptime PH_AR: Int = 28    # ɑɹ
comptime PH_OR: Int = 29    # ɔɹ / Oɹ
comptime PH_UR: Int = 30    # ʊɹ
comptime PH_RX: Int = 31    # postvocalic ɹ
comptime PH_W: Int = 32
comptime PH_J: Int = 33
comptime PH_R: Int = 34     # prevocalic ɹ
comptime PH_L: Int = 35     # light l
comptime PH_LX: Int = 36    # dark l
comptime PH_EL: Int = 37    # syllabic l
comptime PH_HH: Int = 38
comptime PH_M: Int = 39
comptime PH_N: Int = 40
comptime PH_NG: Int = 41
comptime PH_EN: Int = 42
comptime PH_EM: Int = 43
comptime PH_F: Int = 44
comptime PH_V: Int = 45
comptime PH_TH: Int = 46
comptime PH_DH: Int = 47
comptime PH_S: Int = 48
comptime PH_Z: Int = 49
comptime PH_SH: Int = 50
comptime PH_ZH: Int = 51
comptime PH_CH: Int = 52
comptime PH_JH: Int = 53
comptime PH_P: Int = 54
comptime PH_B: Int = 55
comptime PH_T: Int = 56
comptime PH_D: Int = 57
comptime PH_K: Int = 58
comptime PH_G: Int = 59
comptime PH_DX: Int = 60    # flap T / ɾ
comptime PH_Q: Int = 61     # glottal stop ʔ
comptime PH_TQ: Int = 62    # glottalised t
comptime PH_PAU: Int = 63
comptime PH_COUNT: Int = 64


struct PhonemeDef(Copyable, Movable):
    """Steady-state targets of one phoneme. `*e` fields are the end targets of diphthongs and
    glided vowels (equal to the onset for monophthongs)."""

    var name: String
    var f1: Float32
    var f2: Float32
    var f3: Float32
    var f4: Float32
    var f5: Float32
    var b1: Float32
    var b2: Float32
    var b3: Float32
    var f1e: Float32
    var f2e: Float32
    var f3e: Float32
    var b1e: Float32
    var b2e: Float32
    var b3e: Float32
    var nasal_fz: Float32   # nasal zero during the murmur (0 = not nasal)
    var av: Float32
    var avs: Float32
    var ah: Float32
    var af: Float32
    var a2: Float32
    var a3: Float32
    var a4: Float32
    var a5: Float32
    var a6: Float32
    var ab: Float32
    var inh: Float32
    var mn: Float32
    var flags: Int
    var place: Int
    var glide_t0: Float32   # diphthong onset plateau end (fraction of the segment)
    var glide_t1: Float32   # diphthong end plateau start

    def __init__(
        out self,
        name: String,
        f1: Float32, f2: Float32, f3: Float32,
        b1: Float32, b2: Float32, b3: Float32,
        av: Float32, avs: Float32, ah: Float32, af: Float32,
        a2: Float32, a3: Float32, a4: Float32, a5: Float32, a6: Float32, ab: Float32,
        inh: Float32, mn: Float32, flags: Int, place: Int,
    ):
        self.name = name
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = 3300.0   # [K80] Table I
        self.f5 = 3750.0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.f1e = f1
        self.f2e = f2
        self.f3e = f3
        self.b1e = b1
        self.b2e = b2
        self.b3e = b3
        self.nasal_fz = 0.0
        self.av = av
        self.avs = avs
        self.ah = ah
        self.af = af
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.ab = ab
        self.inh = inh
        self.mn = mn
        self.flags = flags
        self.place = place
        self.glide_t0 = 0.25   # [G68] timing: onset steady 0-0.25, transition to 0.80
        self.glide_t1 = 0.80

    def has(self, f: Int) -> Bool:
        return (self.flags & f) != 0

    def set_end(mut self, f1: Float32, f2: Float32, f3: Float32, b1: Float32, b2: Float32, b3: Float32):
        self.f1e = f1
        self.f2e = f2
        self.f3e = f3
        self.b1e = b1
        self.b2e = b2
        self.b3e = b3


comptime _V = FL_VOWEL | FL_VOICED | FL_SON
comptime _VR = FL_VOWEL | FL_VOICED | FL_SON | FL_REDUCED
comptime _VD = FL_VOWEL | FL_VOICED | FL_SON | FL_DIPH


def _v(name: String, f1: Float32, f2: Float32, f3: Float32, b1: Float32, b2: Float32, b3: Float32,
       av: Float32, inh: Float32, mn: Float32, flags: Int) -> PhonemeDef:
    return PhonemeDef(name, f1, f2, f3, b1, b2, b3, av, 0, 0, 0, 0, 0, 0, 0, 0, 0, inh, mn, flags, PL_NONE)


def build_phoneme_table() -> List[PhonemeDef]:
    """The full table, indexed by `PH_*`. Durations follow the prosody note's §1.2 (K79 Table 1 =
    AHK87 Table 9-1); targets the phonetics note's §1 (K80 Tables II and III)."""
    var t = List[PhonemeDef]()
    # ---- vowels: F1 F2 F3 from [HGCW95] men's means blended with [PB52] where Hillenbrand's
    #      Michigan vowels are raised (ae) or fronted. K80 T2 values (Klatt's own voice) were central
    #      enough that coarticulatory undershoot merged ae / eh and weakened i (tuning log, it. 5).
    #      B1 B2 B3 [K80 T2], AV, INH/MIN [K79]
    t.append(_v("i", 300, 2300, 3000, 45, 200, 400, 60, 155, 55, _V))          # K80 [iʸ]; K79 IY
    t.append(_v("ɪ", 420, 2000, 2650, 50, 100, 140, 60, 135, 40, _V))          # K80 [Iᵊ]; K79 IH
    t.append(_v("ᵻ", 440, 1800, 2570, 55, 95, 170, 55, 110, 40, _VR))          # ENG 0.65 ɪ + 0.35 ə; K79 IX
    t.append(_v("e", 480, 1950, 2600, 70, 100, 200, 60, 150, 60, _V))          # = A onset (K80 T2)
    t.append(_v("ɛ", 570, 1800, 2550, 60, 90, 200, 60, 150, 70, _V))           # K80 [ɛᵊ]; K79 EH
    t.append(_v("æ", 690, 1730, 2450, 70, 150, 320, 60, 230, 80, _V))          # K80 [æᵊ]; K79 AE
    t.append(_v("a", 700, 1450, 2500, 90, 100, 200, 60, 230, 80, _V))          # ENG GB TRAP
    t.append(_v("ɑ", 750, 1200, 2500, 130, 70, 160, 60, 240, 100, _V))         # K80 [ɑ]; K79 AA
    t.append(_v("ɒ", 640, 1000, 2550, 100, 80, 120, 60, 240, 100, _V))         # ENG GB LOT
    t.append(_v("ɔ", 620, 950, 2500, 90, 100, 80, 60, 240, 100, _V))           # K80 [ɔᵊ]; K79 AO
    t.append(_v("ʌ", 630, 1200, 2500, 80, 50, 140, 60, 140, 60, _V))           # K80 [ʌ]; K79 AH
    t.append(_v("o", 510, 920, 2450, 80, 70, 70, 60, 150, 60, _V))            # = O onset
    t.append(_v("ʊ", 460, 1100, 2400, 80, 100, 80, 60, 160, 60, _V))           # K80 [ʊᵊ]; K79 UH
    t.append(_v("u", 340, 1100, 2300, 65, 110, 140, 60, 210, 70, _V))          # K80 [uʷ]; K79 UW
    t.append(_v("ɝ", 470, 1350, 1650, 100, 60, 110, 60, 180, 80, _V))          # K80 [ɝ]; K79 RR
    t.append(_v("ɚ", 480, 1380, 1700, 100, 60, 110, 55, 180, 80, _VR))         # K80 [ɝ] F3 +60 (ENG); K79 AXR
    t.append(_v("ɜ", 480, 1420, 2450, 80, 70, 150, 60, 180, 80, _V))           # ENG non-rhotic; [S98]
    t.append(_v("ə", 500, 1400, 2450, 80, 90, 150, 55, 120, 55, _VR))          # [S98, F60] tube; K79 AX
    t.append(_v("ɐ", 600, 1300, 2500, 80, 70, 150, 55, 120, 55, _VR))          # ENG midway ʌ/ə
    t.append(_v("ᵊ", 500, 1400, 2450, 80, 90, 150, 52, 80, 40, _VR))           # ENG; prosody note 80/40
    # ---- diphthongs [K80 T2 onset + end rows]
    t.append(_v("A", 480, 1950, 2600, 70, 100, 200, 60, 180, 100, _VD))       # K79 EY
    t[PH_EY].set_end(340, 2250, 2900, 55, 100, 200)
    t.append(_v("I", 740, 1250, 2550, 100, 70, 200, 60, 250, 150, _VD))       # K79 AY
    t[PH_AY].set_end(420, 2050, 2650, 70, 100, 200)
    t.append(_v("O", 520, 950, 2450, 80, 70, 70, 60, 220, 80, _VD))          # K79 OW
    t[PH_OW].set_end(420, 820, 2350, 80, 70, 70)
    t.append(_v("W", 740, 1300, 2550, 80, 70, 140, 60, 260, 100, _VD))        # K79 AW
    t[PH_AW].set_end(430, 900, 2350, 80, 70, 80)
    t.append(_v("Y", 560, 900, 2450, 80, 50, 130, 60, 280, 150, _VD))         # K79 OY
    t[PH_OY].set_end(400, 2000, 2600, 60, 50, 160)
    t.append(_v("Q", 500, 1400, 2450, 80, 90, 150, 60, 220, 80, _VD))         # ENG GB GOAT
    t[PH_GOAT].set_end(450, 1000, 2350, 80, 70, 80)
    # ---- r-coloured nuclei (K79 IR ER AR OR UR): vowel row -> RX row, F3 falls 30 % -> 85 %
    t.append(_v("IR", 420, 2000, 2650, 50, 100, 140, 60, 230, 100, _VD))
    t.append(_v("ER", 570, 1800, 2550, 60, 90, 200, 60, 270, 100, _VD))
    t.append(_v("AR", 750, 1200, 2500, 130, 70, 160, 60, 260, 100, _VD))
    t.append(_v("OR", 620, 950, 2500, 90, 100, 80, 60, 240, 100, _VD))
    t.append(_v("UR", 460, 1100, 2400, 80, 100, 80, 60, 230, 100, _VD))
    for k in range(PH_IR, PH_UR + 1):
        t[k].set_end(450, 1250, 1600, 90, 70, 110)
        t[k].glide_t0 = 0.30
        t[k].glide_t1 = 0.85
    # ---- sonorant consonants [K80 T3]; AV = vowel - 6 (K80 text says -10; at -10 word-initial
    #      /r l w/ after a stop were lost, tuning log it. 7)
    t.append(PhonemeDef("RX", 450, 1250, 1600, 90, 70, 110, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 70,
                        FL_VOICED | FL_SON | FL_LIQ, PL_RHOTIC))                 # [L62] via K87; ENG
    t.append(PhonemeDef("w", 290, 610, 2150, 50, 80, 60, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 60,
                        FL_VOICED | FL_SON | FL_GLIDE, PL_LABVEL))               # K80 T3; K79 W
    t.append(PhonemeDef("j", 260, 2070, 3020, 40, 250, 500, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 40,
                        FL_VOICED | FL_SON | FL_GLIDE, PL_PALATAL))              # K80 T3 [y]; K79 Y
    t[PH_J].f4 = 3500.0   # ENG: raised to clear F3
    t[PH_J].f5 = 3900.0
    t.append(PhonemeDef("ɹ", 310, 1060, 1380, 70, 100, 120, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 30,
                        FL_VOICED | FL_SON | FL_LIQ, PL_RHOTIC))                 # K80 T3 [r]; K79 R
    t.append(PhonemeDef("l", 310, 1050, 2880, 50, 100, 280, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 40,
                        FL_VOICED | FL_SON | FL_LIQ, PL_LATERAL))                # K80 T3 [l]; K79 L
    t.append(PhonemeDef("LX", 450, 950, 2650, 70, 80, 150, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 70,
                        FL_VOICED | FL_SON | FL_LIQ, PL_DARKL))                  # [L62] via K87; ENG
    t.append(PhonemeDef("EL", 450, 950, 2650, 70, 80, 150, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 260, 110,
                        FL_VOICED | FL_SON | FL_LIQ | FL_SYLL, PL_DARKL))        # = LX; K79 EL
    t.append(PhonemeDef("h", 500, 1400, 2450, 300, 90, 150, 0, 0, 58, 0, 0, 0, 0, 0, 0, 0, 80, 20,
                        FL_ASPIR, PL_GLOTTAL))                                   # K80: next vowel formants, B1 300
    # ---- nasals [K80 T3]: murmur F1 480 / B1 40, FNP 270, FNZ 450
    t.append(PhonemeDef("m", 480, 1270, 2130, 40, 200, 200, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 60,
                        FL_VOICED | FL_SON | FL_NASAL, PL_LABIAL))
    t.append(PhonemeDef("n", 480, 1340, 2470, 40, 300, 300, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 50,
                        FL_VOICED | FL_SON | FL_NASAL, PL_ALVEOLAR))
    t.append(PhonemeDef("ŋ", 480, 1800, 2600, 40, 300, 300, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 60,
                        FL_VOICED | FL_SON | FL_NASAL, PL_VELAR))                # ENG velar locus
    t.append(PhonemeDef("EN", 480, 1340, 2470, 40, 300, 300, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 70,
                        FL_VOICED | FL_SON | FL_NASAL | FL_SYLL, PL_ALVEOLAR))
    t.append(PhonemeDef("EM", 480, 1270, 2130, 40, 200, 200, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 110,
                        FL_VOICED | FL_SON | FL_NASAL | FL_SYLL, PL_LABIAL))
    t[PH_M].nasal_fz = 450.0
    t[PH_N].nasal_fz = 450.0
    t[PH_NG].nasal_fz = 500.0
    t[PH_EN].nasal_fz = 450.0
    t[PH_EM].nasal_fz = 450.0
    # ---- fricatives [K80 T3]: AF 60 voiceless, AV 47 / AVS 47 / AF 50 voiced; parallel A2..A6, AB
    t.append(PhonemeDef("f", 340, 1100, 2080, 200, 120, 150, 0, 0, 0, 60, 0, 0, 0, 0, 0, 57, 100, 80,
                        FL_FRIC, PL_LABIODENTAL))
    t.append(PhonemeDef("v", 220, 1100, 2080, 60, 90, 120, 47, 47, 0, 50, 0, 0, 0, 0, 0, 57, 60, 40,
                        FL_FRIC | FL_VOICED, PL_LABIODENTAL))
    t.append(PhonemeDef("θ", 320, 1290, 2540, 200, 90, 200, 0, 0, 0, 60, 0, 0, 0, 0, 28, 48, 90, 60,
                        FL_FRIC, PL_DENTAL))
    t.append(PhonemeDef("ð", 270, 1290, 2540, 60, 80, 170, 47, 47, 0, 50, 0, 0, 0, 0, 28, 48, 50, 30,
                        FL_FRIC | FL_VOICED, PL_DENTAL))
    t.append(PhonemeDef("s", 320, 1390, 2530, 200, 80, 200, 0, 0, 0, 60, 0, 0, 0, 0, 52, 0, 105, 60,
                        FL_FRIC, PL_ALVEOLAR))
    t.append(PhonemeDef("z", 240, 1390, 2530, 70, 60, 180, 47, 47, 0, 50, 0, 0, 0, 0, 52, 0, 75, 40,
                        FL_FRIC | FL_VOICED, PL_ALVEOLAR))
    t.append(PhonemeDef("ʃ", 300, 1840, 2750, 200, 100, 300, 0, 0, 0, 60, 0, 57, 48, 48, 46, 0, 105, 80,
                        FL_FRIC, PL_POSTALV))
    t.append(PhonemeDef("ʒ", 220, 1840, 2750, 60, 100, 270, 47, 47, 0, 50, 0, 57, 48, 48, 46, 0, 70, 40,
                        FL_FRIC | FL_VOICED, PL_POSTALV))
    # ---- affricates: closure + frication [K80 T3 č ǰ]; durations closure + fric (phonetics §2.2)
    t.append(PhonemeDef("ʧ", 350, 1800, 2820, 200, 90, 300, 0, 0, 0, 60, 0, 44, 60, 53, 53, 0, 130, 90,
                        FL_STOP | FL_AFFR, PL_POSTALV))
    t.append(PhonemeDef("ʤ", 260, 1800, 2820, 60, 80, 270, 47, 47, 0, 50, 0, 44, 60, 53, 53, 0, 100, 70,
                        FL_STOP | FL_AFFR | FL_VOICED, PL_POSTALV))
    # ---- stops: F/B are release loci, A2..AB the burst [K80 T3]; INH/MIN = closure [K79]
    t.append(PhonemeDef("p", 400, 1100, 2150, 300, 150, 220, 0, 0, 0, 60, 0, 0, 0, 0, 0, 63, 90, 50,
                        FL_STOP, PL_LABIAL))
    t.append(PhonemeDef("b", 200, 1100, 2150, 60, 110, 130, 0, 50, 0, 55, 0, 0, 0, 0, 0, 63, 85, 60,
                        FL_STOP | FL_VOICED, PL_LABIAL))
    t.append(PhonemeDef("t", 400, 1600, 2600, 300, 120, 250, 0, 0, 0, 60, 0, 30, 45, 57, 63, 0, 75, 50,
                        FL_STOP, PL_ALVEOLAR))
    t.append(PhonemeDef("d", 200, 1600, 2600, 60, 100, 170, 0, 50, 0, 55, 0, 47, 60, 62, 60, 0, 75, 50,
                        FL_STOP | FL_VOICED, PL_ALVEOLAR))
    t.append(PhonemeDef("k", 300, 1990, 2850, 250, 160, 330, 0, 0, 0, 60, 50, 60, 45, 35, 30, 0, 80, 60,
                        FL_STOP, PL_VELAR))
    t.append(PhonemeDef("ɡ", 200, 1990, 2850, 60, 150, 280, 0, 50, 0, 55, 50, 60, 45, 35, 30, 0, 80, 60,
                        FL_STOP | FL_VOICED, PL_VELAR))
    # ---- flap, glottal stop, glottalised t, pause
    t.append(PhonemeDef("T", 300, 1650, 2600, 80, 100, 170, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 20,
                        FL_VOICED | FL_SON | FL_FLAP, PL_ALVEOLAR))              # K79 DX; [ZL79]
    t.append(PhonemeDef("ʔ", 500, 1400, 2450, 80, 90, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 20,
                        FL_GLOTTAL, PL_GLOTTAL))                                 # K79 Q; Flint 40/20
    t.append(PhonemeDef("TQ", 400, 1600, 2600, 60, 100, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 50,
                        FL_GLOTTAL | FL_STOP, PL_ALVEOLAR))                      # K79 TQ
    t.append(PhonemeDef("_", 500, 1400, 2450, 80, 90, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 200,
                        FL_PAUSE, PL_NONE))
    # glided vowels i / u and ɝ end rows [K80 T2]
    t[PH_I].set_end(280, 2350, 3050, 60, 200, 400)
    t[PH_U].set_end(320, 950, 2250, 65, 110, 140)
    t[PH_ER].set_end(430, 1380, 1650, 100, 60, 110)
    t[PH_I].glide_t0 = 0.0
    t[PH_I].glide_t1 = 1.0
    t[PH_U].glide_t0 = 0.0
    t[PH_U].glide_t1 = 1.0
    t[PH_ER].glide_t0 = 0.0
    t[PH_ER].glide_t1 = 1.0
    return t^


struct Locus(Copyable, Movable):
    """Consonant-place locus row (phonetics note §4.1): F_onset = L + k (F_vowel - L)."""

    var f2: Float32
    var k2: Float32
    var f3: Float32
    var k3: Float32
    var cv1: Float32    # CV transition of F1 (ms)
    var cv23: Float32   # CV transition of F2 / F3 (ms)
    var vc: Float32     # VC transition (ms)

    def __init__(out self, f2: Float32, k2: Float32, f3: Float32, k3: Float32, cv1: Float32, cv23: Float32, vc: Float32):
        self.f2 = f2
        self.k2 = k2
        self.f3 = f3
        self.k3 = k3
        self.cv1 = cv1
        self.cv23 = cv23
        self.vc = vc


def build_locus_table() -> List[Locus]:
    """Indexed by `PL_*`. L from [K80] T3 / [DLC55], k from the [SMM91] slope ranges (ENG)."""
    var t = List[Locus]()
    t.append(Locus(0, 1.0, 0, 1.0, 30, 40, 40))          # none (vowels)
    t.append(Locus(1000, 0.75, 2150, 0.6, 20, 40, 50))   # labial p b m
    t.append(Locus(1100, 0.75, 2080, 0.6, 20, 40, 50))   # labiodental f v
    t.append(Locus(1400, 0.5, 2540, 0.5, 25, 45, 50))    # dental θ ð
    t.append(Locus(1800, 0.4, 2650, 0.5, 25, 45, 55))    # alveolar t d n s z l ɾ
    t.append(Locus(1900, 0.35, 2750, 0.5, 25, 50, 55))   # postalveolar ʃ ʒ ʧ ʤ
    t.append(Locus(2070, 0.0, 3020, 0.0, 40, 80, 80))    # palatal glide j
    t.append(Locus(2300, 0.5, 2750, 0.4, 25, 50, 45))    # velar (front-vowel row; back row in rules)
    t.append(Locus(610, 0.0, 2150, 0.0, 40, 90, 80))     # w
    t.append(Locus(1060, 0.0, 1380, 0.0, 40, 80, 80))    # ɹ
    t.append(Locus(1050, 0.3, 2880, 0.0, 30, 60, 60))    # light l
    t.append(Locus(950, 0.0, 2650, 0.0, 40, 60, 55))     # dark l [L62]; VC 55 ms keeps the vowel
    t.append(Locus(0, 1.0, 0, 1.0, 0, 0, 0))             # h: takes the vowel's formants
    return t^


def phoneme_of_codepoint(c: Int) -> Int:
    """misaki symbol -> PH_* (-1 if not a phoneme). Merging (ɹ nuclei, EL / EN, LX) is done by the
    rule engine afterwards."""
    if c == 0x69:  # i
        return PH_I
    if c == 0x26A:
        return PH_IH
    if c == 0x1D7B:
        return PH_IX
    if c == 0x65:  # e
        return PH_E
    if c == 0x25B:
        return PH_EH
    if c == 0xE6:
        return PH_AE
    if c == 0x61:  # a
        return PH_A
    if c == 0x251:
        return PH_AA
    if c == 0x252:
        return PH_OH
    if c == 0x254:
        return PH_AO
    if c == 0x28C:
        return PH_AH
    if c == 0x6F:  # o
        return PH_O
    if c == 0x28A:
        return PH_UH
    if c == 0x75:  # u
        return PH_U
    if c == 0x25D:
        return PH_ER
    if c == 0x25A:
        return PH_AXR
    if c == 0x25C:
        return PH_ERGB
    if c == 0x259:
        return PH_AX
    if c == 0x250:
        return PH_AXT
    if c == 0x1D4A:
        return PH_SCHWA
    if c == 0x41:  # A
        return PH_EY
    if c == 0x49:  # I
        return PH_AY
    if c == 0x4F:  # O
        return PH_OW
    if c == 0x57:  # W
        return PH_AW
    if c == 0x59:  # Y
        return PH_OY
    if c == 0x51:  # Q
        return PH_GOAT
    if c == 0x77:  # w
        return PH_W
    if c == 0x6A:  # j
        return PH_J
    if c == 0x279 or c == 0x72:  # ɹ / r
        return PH_R
    if c == 0x6C:  # l
        return PH_L
    if c == 0x68:  # h
        return PH_HH
    if c == 0x6D:
        return PH_M
    if c == 0x6E:
        return PH_N
    if c == 0x14B:
        return PH_NG
    if c == 0x66:
        return PH_F
    if c == 0x76:
        return PH_V
    if c == 0x3B8:
        return PH_TH
    if c == 0xF0:
        return PH_DH
    if c == 0x73:
        return PH_S
    if c == 0x7A:
        return PH_Z
    if c == 0x283:
        return PH_SH
    if c == 0x292:
        return PH_ZH
    if c == 0x2A7:
        return PH_CH
    if c == 0x2A4:
        return PH_JH
    if c == 0x70:
        return PH_P
    if c == 0x62:
        return PH_B
    if c == 0x74:
        return PH_T
    if c == 0x64:
        return PH_D
    if c == 0x6B:
        return PH_K
    if c == 0x261 or c == 0x67:  # ɡ / g
        return PH_G
    if c == 0x54 or c == 0x27E:  # T / ɾ
        return PH_DX
    if c == 0x294:
        return PH_Q
    return -1
