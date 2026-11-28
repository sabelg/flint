"""Pure-Mojo XKB (`xkb_keymap` text v1) & X11 keysym-to-UTF-8 translator (`flint.platform.xkb`).

Parses `wl_keyboard.keymap` (`XKB_KEYMAP_FORMAT_TEXT_V1`) strings received via memfd
with zero `libxkbcommon` C dependency, maintains modifier/group state from
`wl_keyboard.modifiers`, and provides a complete US QWERTY fallback plus X11 keysym
codepoint resolution for `X11Window`.
"""
from std.collections import Dict, List
from flint.text.utf8 import encode_codepoint

comptime MAX_EVDEV_KEYS: Int = 256
comptime MAX_XKB_GROUPS: Int = 4
comptime MAX_XKB_LEVELS: Int = 8

comptime XKB_MOD_SHIFT: Int = 1
comptime XKB_MOD_LOCK: Int = 2
comptime XKB_MOD_CONTROL: Int = 4
comptime XKB_MOD_MOD1: Int = 8
comptime XKB_MOD_MOD2: Int = 16
comptime XKB_MOD_MOD5: Int = 128

comptime XKB_TYPE_ONE_LEVEL: Int = 1
comptime XKB_TYPE_TWO_LEVEL: Int = 2
comptime XKB_TYPE_ALPHABETIC: Int = 3
comptime XKB_TYPE_FOUR_LEVEL: Int = 4
comptime XKB_TYPE_FOUR_LEVEL_ALPHABETIC: Int = 5
comptime XKB_TYPE_FOUR_LEVEL_SEMIALPHABETIC: Int = 6
comptime XKB_TYPE_KEYPAD: Int = 7
comptime XKB_TYPE_FOUR_LEVEL_PLUS_LOCK: Int = 8


def _hex_val(b: UInt8) -> Int:
    if b >= 48 and b <= 57:
        return Int(b - 48)
    if b >= 65 and b <= 70:
        return Int(b - 55)
    if b >= 97 and b <= 102:
        return Int(b - 87)
    return -1


def _parse_hex_str(s: String) -> Int:
    var bytes = s.as_bytes()
    if len(bytes) == 0:
        return -1
    var acc = 0
    for i in range(len(bytes)):
        var d = _hex_val(bytes[i])
        if d < 0:
            return -1
        acc = (acc << 4) | d
    return acc


def _parse_dec_or_hex(s: String) -> Int:
    var bytes = s.as_bytes()
    if len(bytes) == 0:
        return -1
    if len(bytes) > 2 and bytes[0] == 48 and (bytes[1] == 120 or bytes[1] == 88):
        return _parse_hex_str(String(s[byte=2:]))
    var acc = 0
    for i in range(len(bytes)):
        var b = bytes[i]
        if b < 48 or b > 57:
            return -1
        acc = acc * 10 + Int(b - 48)
    return acc


def _parse_group_index(g_tok: String) -> Int:
    """Parses case-insensitive `GroupN` / `groupN` (1..MAX_XKB_GROUPS) -> 0-based index, or -1."""
    if g_tok.byte_length() < 6:
        return -1
    var b = g_tok.as_bytes()
    # "group" ASCII case-insensitive check
    if (
        (b[0] != 103 and b[0] != 71)
        or (b[1] != 114 and b[1] != 82)
        or (b[2] != 111 and b[2] != 79)
        or (b[3] != 117 and b[3] != 85)
        or (b[4] != 112 and b[4] != 80)
    ):
        return -1
    var gn = _parse_dec_or_hex(String(g_tok[byte=5:]))
    if gn >= 1 and gn <= MAX_XKB_GROUPS:
        return gn - 1
    return -1


def _is_alpha_cp(cp: Int) -> Bool:
    if cp >= 65 and cp <= 90:
        return True
    if cp >= 97 and cp <= 122:
        return True
    if (cp >= 0xC0 and cp <= 0xD6) or (cp >= 0xD8 and cp <= 0xDE):
        return True
    if (cp >= 0xE0 and cp <= 0xF6) or (cp >= 0xF8 and cp <= 0xFE):
        return True
    return False


def x11_keysym_to_codepoint(keysym: Int) -> Int:
    """Converts a numeric X11/XKB keysym to a printable Unicode codepoint (0 if non-printable)."""
    if keysym <= 0:
        return 0
    # Standard ASCII & Latin-1 direct mapping (0x0020..0x00FF excluding DEL/C1 control)
    if (keysym >= 0x0020 and keysym <= 0x007E) or (keysym >= 0x00A0 and keysym <= 0x00FF):
        return keysym
    # Modern Unicode keysym encoding: 0x01000000 | codepoint
    if keysym >= 0x01000020 and keysym <= 0x0110FFFF:
        var cp = keysym - 0x01000000
        if cp >= 0x20 and cp != 0x7F and not (cp >= 0xD800 and cp <= 0xDFFF):
            return cp
        return 0
    # Currency & common legacy keysyms
    if keysym == 0x20AC:
        return 0x20AC  # EuroSign
    if keysym == 0x0AA2:
        return 0x2013  # endash (approx legacy)
    if keysym == 0x0AAE:
        return 0x2026  # ellipsis
    # Keypad printable keysyms (0xFF80..0xFFB9)
    if keysym == 0xFF80:  # KP_Space
        return 32
    if keysym == 0xFFAA:  # KP_Multiply
        return 42
    if keysym == 0xFFAB:  # KP_Add
        return 43
    if keysym == 0xFFAC:  # KP_Separator
        return 44
    if keysym == 0xFFAD:  # KP_Subtract
        return 45
    if keysym == 0xFFAE:  # KP_Decimal
        return 46
    if keysym == 0xFFAF:  # KP_Divide
        return 47
    if keysym >= 0xFFB0 and keysym <= 0xFFB9:  # KP_0..KP_9
        return 48 + (keysym - 0xFFB0)
    if keysym == 0xFFBD:  # KP_Equal
        return 61
    # Extended Latin-2/3/4/9 common letters
    if keysym == 0x13BC:
        return 0x0152  # OE
    if keysym == 0x13BD:
        return 0x0153  # oe
    if keysym == 0x13BE:
        return 0x0178  # Ydiaeresis
    if keysym == 0x01A9:
        return 0x0160  # Scaron
    if keysym == 0x01B9:
        return 0x0161  # scaron
    if keysym == 0x01AE:
        return 0x017D  # Zcaron
    if keysym == 0x01BE:
        return 0x017E  # zcaron
    return 0


def xkb_keysym_name_to_codepoint(name: String) -> Int:
    """Resolves an XKB keysym name (`a`, `exclam`, `ssharp`, `EuroSign`, `U1E9E`, `0x...`) to Unicode."""
    var n = name.byte_length()
    if n == 0:
        return 0
    var bytes = name.as_bytes()
    if n == 1:
        var b = Int(bytes[0])
        if b >= 32 and b <= 126:
            return b
        return 0
    # Uxxxx Unicode keysym syntax
    if bytes[0] == 85 and n >= 5 and n <= 7:  # 'U'
        var cp = _parse_hex_str(String(name[byte=1:]))
        if cp >= 0x20 and cp <= 0x10FFFF and cp != 0x7F and not (cp >= 0xD800 and cp <= 0xDFFF):
            return cp
    # Hex keysym 0x...
    if n > 2 and bytes[0] == 48 and (bytes[1] == 120 or bytes[1] == 88):
        var ks = _parse_hex_str(String(name[byte=2:]))
        if ks > 0:
            return x11_keysym_to_codepoint(ks)
    # Non-printable / dead / control / modifier keysyms
    if (
        name == "NoSymbol"
        or name == "VoidSymbol"
        or name == "Return"
        or name == "KP_Enter"
        or name == "Tab"
        or name == "ISO_Left_Tab"
        or name == "BackSpace"
        or name == "Escape"
        or name == "Delete"
        or name.startswith("dead_")
        or name.startswith("Shift_")
        or name.startswith("Control_")
        or name.startswith("Alt_")
        or name.startswith("Meta_")
        or name.startswith("Super_")
        or name.startswith("Hyper_")
        or name.startswith("ISO_")
        or name.startswith("XF86")
    ):
        return 0

    # ASCII named symbols
    if name == "space":
        return 32
    if name == "exclam":
        return 33
    if name == "quotedbl":
        return 34
    if name == "numbersign":
        return 35
    if name == "dollar":
        return 36
    if name == "percent":
        return 37
    if name == "ampersand":
        return 38
    if name == "apostrophe" or name == "quoteright":
        return 39
    if name == "parenleft":
        return 40
    if name == "parenright":
        return 41
    if name == "asterisk":
        return 42
    if name == "plus":
        return 43
    if name == "comma":
        return 44
    if name == "minus":
        return 45
    if name == "period":
        return 46
    if name == "slash":
        return 47
    if name == "colon":
        return 58
    if name == "semicolon":
        return 59
    if name == "less":
        return 60
    if name == "equal":
        return 61
    if name == "greater":
        return 62
    if name == "question":
        return 63
    if name == "at":
        return 64
    if name == "bracketleft":
        return 91
    if name == "backslash":
        return 92
    if name == "bracketright":
        return 93
    if name == "asciicircum":
        return 94
    if name == "underscore":
        return 95
    if name == "grave" or name == "quoteleft":
        return 96
    if name == "braceleft":
        return 123
    if name == "bar":
        return 124
    if name == "braceright":
        return 125
    if name == "asciitilde":
        return 126

    # Keypad printable keysyms
    if name.startswith("KP_"):
        if n == 4 and bytes[3] >= 48 and bytes[3] <= 57:
            return Int(bytes[3])
        if name == "KP_Decimal":
            return 46
        if name == "KP_Divide":
            return 47
        if name == "KP_Multiply":
            return 42
        if name == "KP_Subtract":
            return 45
        if name == "KP_Add":
            return 43
        if name == "KP_Equal":
            return 61
        if name == "KP_Space":
            return 32
        if name == "KP_Separator":
            return 44
        return 0

    # Common international / Latin-1 / Euro / typography keysyms
    if name == "nobreakspace":
        return 0x00A0
    if name == "exclamdown":
        return 0x00A1
    if name == "cent":
        return 0x00A2
    if name == "sterling":
        return 0x00A3
    if name == "currency":
        return 0x00A4
    if name == "yen":
        return 0x00A5
    if name == "brokenbar":
        return 0x00A6
    if name == "section":
        return 0x00A7
    if name == "diaeresis":
        return 0x00A8
    if name == "copyright":
        return 0x00A9
    if name == "ordfeminine":
        return 0x00AA
    if name == "guillemotleft":
        return 0x00AB
    if name == "notsign":
        return 0x00AC
    if name == "hyphen":
        return 0x00AD
    if name == "registered":
        return 0x00AE
    if name == "macron":
        return 0x00AF
    if name == "degree":
        return 0x00B0
    if name == "plusminus":
        return 0x00B1
    if name == "twosuperior":
        return 0x00B2
    if name == "threesuperior":
        return 0x00B3
    if name == "acute":
        return 0x00B4
    if name == "mu":
        return 0x00B5
    if name == "paragraph":
        return 0x00B6
    if name == "periodcentered":
        return 0x00B7
    if name == "cedilla":
        return 0x00B8
    if name == "onesuperior":
        return 0x00B9
    if name == "masculine":
        return 0x00BA
    if name == "guillemotright":
        return 0x00BB
    if name == "onequarter":
        return 0x00BC
    if name == "onehalf":
        return 0x00BD
    if name == "threequarters":
        return 0x00BE
    if name == "questiondown":
        return 0x00BF
    if name == "Agrave":
        return 0x00C0
    if name == "Aacute":
        return 0x00C1
    if name == "Acircumflex":
        return 0x00C2
    if name == "Atilde":
        return 0x00C3
    if name == "Adiaeresis":
        return 0x00C4
    if name == "Aring":
        return 0x00C5
    if name == "AE":
        return 0x00C6
    if name == "Ccedilla":
        return 0x00C7
    if name == "Egrave":
        return 0x00C8
    if name == "Eacute":
        return 0x00C9
    if name == "Ecircumflex":
        return 0x00CA
    if name == "Ediaeresis":
        return 0x00CB
    if name == "Igrave":
        return 0x00CC
    if name == "Iacute":
        return 0x00CD
    if name == "Icircumflex":
        return 0x00CE
    if name == "Idiaeresis":
        return 0x00CF
    if name == "ETH":
        return 0x00D0
    if name == "Ntilde":
        return 0x00D1
    if name == "Ograve":
        return 0x00D2
    if name == "Oacute":
        return 0x00D3
    if name == "Ocircumflex":
        return 0x00D4
    if name == "Otilde":
        return 0x00D5
    if name == "Odiaeresis":
        return 0x00D6
    if name == "multiply":
        return 0x00D7
    if name == "Oslash" or name == "Ooblique":
        return 0x00D8
    if name == "Ugrave":
        return 0x00D9
    if name == "Uacute":
        return 0x00DA
    if name == "Ucircumflex":
        return 0x00DB
    if name == "Udiaeresis":
        return 0x00DC
    if name == "Yacute":
        return 0x00DD
    if name == "THORN":
        return 0x00DE
    if name == "ssharp":
        return 0x00DF
    if name == "agrave":
        return 0x00E0
    if name == "aacute":
        return 0x00E1
    if name == "acircumflex":
        return 0x00E2
    if name == "atilde":
        return 0x00E3
    if name == "adiaeresis":
        return 0x00E4
    if name == "aring":
        return 0x00E5
    if name == "ae":
        return 0x00E6
    if name == "ccedilla":
        return 0x00E7
    if name == "egrave":
        return 0x00E8
    if name == "eacute":
        return 0x00E9
    if name == "ecircumflex":
        return 0x00EA
    if name == "ediaeresis":
        return 0x00EB
    if name == "igrave":
        return 0x00EC
    if name == "iacute":
        return 0x00ED
    if name == "icircumflex":
        return 0x00EE
    if name == "idiaeresis":
        return 0x00EF
    if name == "eth":
        return 0x00F0
    if name == "ntilde":
        return 0x00F1
    if name == "ograve":
        return 0x00F2
    if name == "oacute":
        return 0x00F3
    if name == "ocircumflex":
        return 0x00F4
    if name == "otilde":
        return 0x00F5
    if name == "odiaeresis":
        return 0x00F6
    if name == "division":
        return 0x00F7
    if name == "oslash" or name == "ooblique":
        return 0x00F8
    if name == "ugrave":
        return 0x00F9
    if name == "uacute":
        return 0x00FA
    if name == "ucircumflex":
        return 0x00FB
    if name == "udiaeresis":
        return 0x00FC
    if name == "yacute":
        return 0x00FD
    if name == "thorn":
        return 0x00FE
    if name == "ydiaeresis":
        return 0x00FF
    if name == "EuroSign":
        return 0x20AC
    if name == "ellipsis":
        return 0x2026
    if name == "endash":
        return 0x2013
    if name == "emdash":
        return 0x2014
    if name == "leftsinglequotemark":
        return 0x2018
    if name == "rightsinglequotemark":
        return 0x2019
    if name == "leftdoublequotemark":
        return 0x201C
    if name == "rightdoublequotemark":
        return 0x201D
    if name == "bullet":
        return 0x2022
    if name == "OE":
        return 0x0152
    if name == "oe":
        return 0x0153
    if name == "Ydiaeresis":
        return 0x0178
    if name == "Scaron":
        return 0x0160
    if name == "scaron":
        return 0x0161
    if name == "Zcaron":
        return 0x017D
    if name == "zcaron":
        return 0x017E
    return 0


def _type_name_to_id(type_name: String) -> Int:
    if type_name == "ONE_LEVEL":
        return XKB_TYPE_ONE_LEVEL
    if type_name == "TWO_LEVEL":
        return XKB_TYPE_TWO_LEVEL
    if type_name == "ALPHABETIC":
        return XKB_TYPE_ALPHABETIC
    if type_name == "FOUR_LEVEL":
        return XKB_TYPE_FOUR_LEVEL
    if type_name == "FOUR_LEVEL_ALPHABETIC":
        return XKB_TYPE_FOUR_LEVEL_ALPHABETIC
    if type_name == "FOUR_LEVEL_SEMIALPHABETIC":
        return XKB_TYPE_FOUR_LEVEL_SEMIALPHABETIC
    if type_name == "FOUR_LEVEL_PLUS_LOCK":
        return XKB_TYPE_FOUR_LEVEL_PLUS_LOCK
    if type_name == "KEYPAD" or type_name == "FOUR_LEVEL_MIXED_KEYPAD":
        return XKB_TYPE_KEYPAD
    return 0


def _infer_key_type(levels: List[Int]) -> Int:
    var n = len(levels)
    if n <= 1:
        return XKB_TYPE_ONE_LEVEL
    var is_alpha = _is_alpha_cp(levels[0]) or _is_alpha_cp(levels[1])
    if n == 2:
        return XKB_TYPE_ALPHABETIC if is_alpha else XKB_TYPE_TWO_LEVEL
    return XKB_TYPE_FOUR_LEVEL_ALPHABETIC if is_alpha else XKB_TYPE_FOUR_LEVEL


def _tokenize_xkb(text: String) -> List[String]:
    var tokens = List[String]()
    var bytes = text.as_bytes()
    var n = len(bytes)
    var i = 0
    while i < n:
        var b = bytes[i]
        if b == 32 or b == 9 or b == 10 or b == 13:
            i += 1
            continue
        # // line comment
        if b == 47 and i + 1 < n and bytes[i + 1] == 47:
            i += 2
            while i < n and bytes[i] != 10:
                i += 1
            continue
        # /* block comment */
        if b == 47 and i + 1 < n and bytes[i + 1] == 42:
            i += 2
            while i + 1 < n and not (bytes[i] == 42 and bytes[i + 1] == 47):
                i += 1
            i += 2
            continue
        # String literal "..."
        if b == 34:
            i += 1
            var start = i
            while i < n and bytes[i] != 34:
                if bytes[i] == 92 and i + 1 < n:
                    i += 2
                else:
                    i += 1
            tokens.append('"' + String(text[byte=start:i]) + '"')
            if i < n:
                i += 1
            continue
        # Key name <NAME>
        if b == 60:
            var start = i
            i += 1
            while i < n and bytes[i] != 62:
                i += 1
            if i < n:
                i += 1
            tokens.append(String(text[byte=start:i]))
            continue
        # Single-char punctuation: { } [ ] ( ) = , ; .
        if (
            b == 123 or b == 125 or b == 91 or b == 93
            or b == 40 or b == 41 or b == 61 or b == 44 or b == 59 or b == 46
        ):
            tokens.append(String(text[byte=i : i + 1]))
            i += 1
            continue
        # Identifier or number
        var start = i
        while i < n:
            var c = bytes[i]
            if (
                c == 32 or c == 9 or c == 10 or c == 13
                or c == 123 or c == 125 or c == 91 or c == 93
                or c == 40 or c == 41 or c == 61 or c == 44 or c == 59 or c == 46
                or c == 60 or c == 34 or c == 47
            ):
                break
            i += 1
        tokens.append(String(text[byte=start:i]))
    return tokens^


struct XkbKeymap(Movable):
    """Pure-Mojo XKB keymap table (`xkb_keymap` text v1) with US QWERTY fallback."""
    var syms: List[Int]          # [evdev * 32 + group * 8 + level]
    var key_types: List[Int]     # [evdev * 4 + group]
    var level_counts: List[Int]  # [evdev * 4 + group]
    var group_counts: List[Int]  # [evdev]
    var mods_depressed: Int
    var mods_latched: Int
    var mods_locked: Int
    var active_group: Int
    var has_parsed_keymap: Bool

    def __init__(out self):
        self.syms = List[Int](length=MAX_EVDEV_KEYS * MAX_XKB_GROUPS * MAX_XKB_LEVELS, fill=0)
        self.key_types = List[Int](length=MAX_EVDEV_KEYS * MAX_XKB_GROUPS, fill=XKB_TYPE_TWO_LEVEL)
        self.level_counts = List[Int](length=MAX_EVDEV_KEYS * MAX_XKB_GROUPS, fill=0)
        self.group_counts = List[Int](length=MAX_EVDEV_KEYS, fill=0)
        self.mods_depressed = 0
        self.mods_latched = 0
        self.mods_locked = 0
        self.active_group = 0
        self.has_parsed_keymap = False
        self.reset_us_fallback()

    def __moveinit__(out self, deinit existing: Self):
        self.syms = existing.syms^
        self.key_types = existing.key_types^
        self.level_counts = existing.level_counts^
        self.group_counts = existing.group_counts^
        self.mods_depressed = existing.mods_depressed
        self.mods_latched = existing.mods_latched
        self.mods_locked = existing.mods_locked
        self.active_group = existing.active_group
        self.has_parsed_keymap = existing.has_parsed_keymap

    def _clear_tables(mut self):
        for i in range(len(self.syms)):
            self.syms[i] = 0
        for i in range(len(self.key_types)):
            self.key_types[i] = XKB_TYPE_TWO_LEVEL
            self.level_counts[i] = 0
        for i in range(len(self.group_counts)):
            self.group_counts[i] = 0

    def _clear_key(mut self, evdev: Int):
        if evdev < 0 or evdev >= MAX_EVDEV_KEYS:
            return
        self.group_counts[evdev] = 0
        for g in range(MAX_XKB_GROUPS):
            self.key_types[evdev * MAX_XKB_GROUPS + g] = 0
            self.level_counts[evdev * MAX_XKB_GROUPS + g] = 0
            var base = (evdev * MAX_XKB_GROUPS + g) * MAX_XKB_LEVELS
            for l in range(MAX_XKB_LEVELS):
                self.syms[base + l] = 0

    def _set_key_pair(mut self, evdev: Int, lower: Int, upper: Int, ktype: Int, group: Int = 0):
        if evdev < 0 or evdev >= MAX_EVDEV_KEYS or group < 0 or group >= MAX_XKB_GROUPS:
            return
        var base = (evdev * MAX_XKB_GROUPS + group) * MAX_XKB_LEVELS
        self.syms[base] = lower
        self.syms[base + 1] = upper
        self.level_counts[evdev * MAX_XKB_GROUPS + group] = 2
        self.key_types[evdev * MAX_XKB_GROUPS + group] = ktype
        if group + 1 > self.group_counts[evdev]:
            self.group_counts[evdev] = group + 1

    def _set_key_single(mut self, evdev: Int, cp: Int, group: Int = 0):
        if evdev < 0 or evdev >= MAX_EVDEV_KEYS or group < 0 or group >= MAX_XKB_GROUPS:
            return
        var base = (evdev * MAX_XKB_GROUPS + group) * MAX_XKB_LEVELS
        self.syms[base] = cp
        self.level_counts[evdev * MAX_XKB_GROUPS + group] = 1
        self.key_types[evdev * MAX_XKB_GROUPS + group] = XKB_TYPE_ONE_LEVEL
        if group + 1 > self.group_counts[evdev]:
            self.group_counts[evdev] = group + 1

    def reset_us_fallback(mut self):
        self._clear_tables()
        self.has_parsed_keymap = False
        # Letters (evdev codes)
        var letter_codes = [30, 48, 46, 32, 18, 33, 34, 35, 23, 36, 37, 38, 50, 49, 24, 25, 16, 19, 31, 20, 22, 47, 17, 45, 21, 44]
        for i in range(26):
            self._set_key_pair(letter_codes[i], 97 + i, 65 + i, XKB_TYPE_ALPHABETIC)
        # Number row
        self._set_key_pair(2, 49, 33, XKB_TYPE_TWO_LEVEL)   # 1 !
        self._set_key_pair(3, 50, 64, XKB_TYPE_TWO_LEVEL)   # 2 @
        self._set_key_pair(4, 51, 35, XKB_TYPE_TWO_LEVEL)   # 3 #
        self._set_key_pair(5, 52, 36, XKB_TYPE_TWO_LEVEL)   # 4 $
        self._set_key_pair(6, 53, 37, XKB_TYPE_TWO_LEVEL)   # 5 %
        self._set_key_pair(7, 54, 94, XKB_TYPE_TWO_LEVEL)   # 6 ^
        self._set_key_pair(8, 55, 38, XKB_TYPE_TWO_LEVEL)   # 7 &
        self._set_key_pair(9, 56, 42, XKB_TYPE_TWO_LEVEL)   # 8 *
        self._set_key_pair(10, 57, 40, XKB_TYPE_TWO_LEVEL)  # 9 (
        self._set_key_pair(11, 48, 41, XKB_TYPE_TWO_LEVEL)  # 0 )
        self._set_key_pair(12, 45, 95, XKB_TYPE_TWO_LEVEL)  # - _
        self._set_key_pair(13, 61, 43, XKB_TYPE_TWO_LEVEL)  # = +
        self._set_key_pair(26, 91, 123, XKB_TYPE_TWO_LEVEL) # [ {
        self._set_key_pair(27, 93, 125, XKB_TYPE_TWO_LEVEL) # ] }
        self._set_key_pair(39, 59, 58, XKB_TYPE_TWO_LEVEL)  # ; :
        self._set_key_pair(40, 39, 34, XKB_TYPE_TWO_LEVEL)  # ' "
        self._set_key_pair(41, 96, 126, XKB_TYPE_TWO_LEVEL) # ` ~
        self._set_key_pair(43, 92, 124, XKB_TYPE_TWO_LEVEL) # \ |
        self._set_key_pair(51, 44, 60, XKB_TYPE_TWO_LEVEL)  # , <
        self._set_key_pair(52, 46, 62, XKB_TYPE_TWO_LEVEL)  # . >
        self._set_key_pair(53, 47, 63, XKB_TYPE_TWO_LEVEL)  # / ?
        self._set_key_pair(86, 60, 62, XKB_TYPE_TWO_LEVEL)  # < > (102nd key)
        self._set_key_single(57, 32)                        # Space
        # Keypad
        self._set_key_single(55, 42)                        # KP_*
        self._set_key_single(74, 45)                        # KP_-
        self._set_key_single(78, 43)                        # KP_+
        self._set_key_single(98, 47)                        # KP_/
        self._set_key_pair(71, 0, 55, XKB_TYPE_KEYPAD)      # KP_7
        self._set_key_pair(72, 0, 56, XKB_TYPE_KEYPAD)      # KP_8
        self._set_key_pair(73, 0, 57, XKB_TYPE_KEYPAD)      # KP_9
        self._set_key_pair(75, 0, 52, XKB_TYPE_KEYPAD)      # KP_4
        self._set_key_pair(76, 0, 53, XKB_TYPE_KEYPAD)      # KP_5
        self._set_key_pair(77, 0, 54, XKB_TYPE_KEYPAD)      # KP_6
        self._set_key_pair(79, 0, 49, XKB_TYPE_KEYPAD)      # KP_1
        self._set_key_pair(80, 0, 50, XKB_TYPE_KEYPAD)      # KP_2
        self._set_key_pair(81, 0, 51, XKB_TYPE_KEYPAD)      # KP_3
        self._set_key_pair(82, 0, 48, XKB_TYPE_KEYPAD)      # KP_0
        self._set_key_pair(83, 0, 46, XKB_TYPE_KEYPAD)      # KP_.

    def set_modifiers(mut self, depressed: Int, latched: Int, locked: Int, group: Int):
        self.mods_depressed = depressed
        self.mods_latched = latched
        self.mods_locked = locked
        self.active_group = max(0, group)

    def parse(mut self, text: String) -> Bool:
        """Parses an XKB text v1 keymap (`xkb_keycodes` + `xkb_symbols`). Returns True if keys were parsed."""
        self.reset_us_fallback()
        var tokens = _tokenize_xkb(text)
        var n = len(tokens)
        if n == 0:
            return False

        var keyname_to_evdev = Dict[String, Int]()
        var alias_src = List[String]()
        var alias_dst = List[String]()
        var section = String("")
        var brace_depth = 0
        var parsed_symbols_count = 0

        var i = 0
        while i < n:
            var tok = tokens[i]
            if tok == "xkb_keycodes" or tok == "xkb_types" or tok == "xkb_compat" or tok == "xkb_symbols" or tok == "xkb_geometry":
                section = tok
                i += 1
                continue
            if tok == "{":
                brace_depth += 1
                i += 1
                continue
            if tok == "}":
                if brace_depth > 0:
                    brace_depth -= 1
                i += 1
                continue

            if section == "xkb_keycodes":
                if tok == "alias" and i + 3 < n and tokens[i + 2] == "=":
                    alias_src.append(tokens[i + 1])
                    alias_dst.append(tokens[i + 3])
                    i += 4
                    continue
                if tok.startswith("<") and i + 2 < n and tokens[i + 1] == "=":
                    var code = _parse_dec_or_hex(tokens[i + 2])
                    if code >= 8 and code - 8 < MAX_EVDEV_KEYS:
                        keyname_to_evdev[tok] = code - 8
                    i += 3
                    continue
            elif section == "xkb_symbols":
                # Resolve aliases (including chained forward aliases) once we enter xkb_symbols
                if len(alias_src) > 0:
                    var max_passes = len(alias_src)
                    for _ in range(max_passes):
                        var changed = False
                        for a_idx in range(len(alias_src)):
                            var src = alias_src[a_idx]
                            var dst = alias_dst[a_idx]
                            if (src not in keyname_to_evdev) and (dst in keyname_to_evdev):
                                try:
                                    keyname_to_evdev[src] = keyname_to_evdev[dst]
                                    changed = True
                                except:
                                    pass
                        if not changed:
                            break
                    alias_src.clear()
                    alias_dst.clear()

                if tok == "key" and i + 2 < n and tokens[i + 1].startswith("<") and tokens[i + 2] == "{":
                    var kname = tokens[i + 1]
                    var evdev = -1
                    if kname in keyname_to_evdev:
                        try:
                            evdev = keyname_to_evdev[kname]
                        except:
                            evdev = -1
                    var j = i + 3
                    var explicit_types = List[Int](length=MAX_XKB_GROUPS, fill=0)
                    var cleared_key = False
                    var at_stmt_start = True
                    while j < n and tokens[j] != "}":
                        var sub = tokens[j]
                        if sub == "," or sub == ";":
                            at_stmt_start = True
                            j += 1
                            continue
                        if sub == "type":
                            var grp = 0
                            var k = j + 1
                            if k + 2 < n and tokens[k] == "[" and tokens[k + 2] == "]":
                                var parsed_grp = _parse_group_index(tokens[k + 1])
                                if parsed_grp >= 0:
                                    grp = parsed_grp
                                k += 3
                            if k + 1 < n and tokens[k] == "=":
                                var t_str = tokens[k + 1]
                                if t_str.byte_length() >= 2 and t_str.startswith('"'):
                                    var unquoted = String(t_str[byte=1 : t_str.byte_length() - 1])
                                    t_str = unquoted
                                var tid = _type_name_to_id(t_str)
                                if tid != 0:
                                    explicit_types[grp] = tid
                                at_stmt_start = False
                                j = k + 2
                                continue
                        elif sub == "symbols" or (sub == "[" and at_stmt_start):
                            var grp = 0
                            var k = j
                            var was_bare = (sub == "[")
                            if sub == "symbols":
                                k += 1
                                if k + 2 < n and tokens[k] == "[" and tokens[k + 2] == "]":
                                    var parsed_grp = _parse_group_index(tokens[k + 1])
                                    if parsed_grp >= 0:
                                        grp = parsed_grp
                                    k += 3
                                if k < n and tokens[k] == "=":
                                    k += 1
                            if k < n and tokens[k] == "[":
                                k += 1
                                var lvls = List[Int]()
                                while k < n and tokens[k] != "]" and tokens[k] != "}":
                                    if tokens[k] != ",":
                                        lvls.append(xkb_keysym_name_to_codepoint(tokens[k]))
                                    k += 1
                                if k < n and tokens[k] == "]":
                                    k += 1
                                if evdev >= 0 and evdev < MAX_EVDEV_KEYS and len(lvls) > 0:
                                    if not cleared_key:
                                        self._clear_key(evdev)
                                        cleared_key = True
                                    var cnt = min(MAX_XKB_LEVELS, len(lvls))
                                    var base = (evdev * MAX_XKB_GROUPS + grp) * MAX_XKB_LEVELS
                                    for l in range(cnt):
                                        self.syms[base + l] = lvls[l]
                                    self.level_counts[evdev * MAX_XKB_GROUPS + grp] = cnt
                                    var tid = explicit_types[grp]
                                    if tid == 0:
                                        tid = _infer_key_type(lvls)
                                    self.key_types[evdev * MAX_XKB_GROUPS + grp] = tid
                                    if grp + 1 > self.group_counts[evdev]:
                                        self.group_counts[evdev] = grp + 1
                                    parsed_symbols_count += 1
                                at_stmt_start = was_bare
                                j = k
                                continue
                        # Skip unknown attribute statements (e.g. actions[Group1]=[...], vmods=..., overlay...)
                        at_stmt_start = False
                        var bracket_depth = 0
                        var paren_depth = 0
                        var inner_brace_depth = 0
                        while j < n:
                            var t = tokens[j]
                            if bracket_depth == 0 and paren_depth == 0 and inner_brace_depth == 0:
                                if t == "}":
                                    break
                                if t == "," or t == ";":
                                    at_stmt_start = True
                                    j += 1
                                    break
                            if t == "[":
                                bracket_depth += 1
                            elif t == "]":
                                if bracket_depth > 0:
                                    bracket_depth -= 1
                            elif t == "(":
                                paren_depth += 1
                            elif t == ")":
                                if paren_depth > 0:
                                    paren_depth -= 1
                            elif t == "{":
                                inner_brace_depth += 1
                            elif t == "}":
                                if inner_brace_depth > 0:
                                    inner_brace_depth -= 1
                                else:
                                    break
                            j += 1
                    # Apply explicit types even if `type[GroupN]` followed `symbols[GroupN]`
                    if evdev >= 0 and evdev < MAX_EVDEV_KEYS and cleared_key:
                        for g in range(MAX_XKB_GROUPS):
                            if explicit_types[g] != 0 and self.level_counts[evdev * MAX_XKB_GROUPS + g] > 0:
                                self.key_types[evdev * MAX_XKB_GROUPS + g] = explicit_types[g]
                    if j < n and tokens[j] == "}":
                        j += 1
                    i = j
                    continue
            i += 1

        if parsed_symbols_count > 0:
            self.has_parsed_keymap = True
        return self.has_parsed_keymap

    def translate_key_to_codepoint(self, evdev_code: Int) -> Int:
        if evdev_code < 0 or evdev_code >= MAX_EVDEV_KEYS:
            return 0
        var mods_active = self.mods_depressed | self.mods_latched
        # Suppress printable UTF-8 text while Control or Alt (Mod1) is held
        if (mods_active & (XKB_MOD_CONTROL | XKB_MOD_MOD1)) != 0:
            return 0
        var n_groups = self.group_counts[evdev_code]
        if n_groups <= 0:
            return 0
        var grp = self.active_group % n_groups
        var n_levels = self.level_counts[evdev_code * MAX_XKB_GROUPS + grp]
        if n_levels <= 0:
            grp = 0
            n_levels = self.level_counts[evdev_code * MAX_XKB_GROUPS]
            if n_levels <= 0:
                return 0
        var ktype = self.key_types[evdev_code * MAX_XKB_GROUPS + grp]
        if ktype == 0:
            ktype = XKB_TYPE_TWO_LEVEL

        var mods_eff = mods_active | self.mods_locked
        var shift = (mods_eff & XKB_MOD_SHIFT) != 0
        var caps = (self.mods_locked & XKB_MOD_LOCK) != 0
        var numlock = (mods_eff & XKB_MOD_MOD2) != 0
        var altgr = (mods_eff & XKB_MOD_MOD5) != 0

        var level = 0
        if ktype == XKB_TYPE_ONE_LEVEL:
            level = 0
        elif ktype == XKB_TYPE_TWO_LEVEL:
            level = 1 if shift else 0
        elif ktype == XKB_TYPE_ALPHABETIC:
            level = 1 if (shift != caps) else 0
        elif ktype == XKB_TYPE_KEYPAD:
            level = 1 if (numlock != shift) else 0
        elif ktype == XKB_TYPE_FOUR_LEVEL:
            level = (2 if altgr else 0) + (1 if shift else 0)
        elif ktype == XKB_TYPE_FOUR_LEVEL_ALPHABETIC:
            level = (2 if altgr else 0) + (1 if (shift != caps) else 0)
        elif ktype == XKB_TYPE_FOUR_LEVEL_SEMIALPHABETIC:
            if altgr:
                level = 2 + (1 if shift else 0)
            else:
                level = 1 if (shift != caps) else 0
        elif ktype == XKB_TYPE_FOUR_LEVEL_PLUS_LOCK:
            if caps and not altgr and not shift and n_levels > 4:
                level = 4
            elif altgr:
                level = 2 + (1 if shift else 0)
            else:
                level = 1 if shift else 0

        if level >= n_levels:
            if altgr and n_levels < 3:
                # Key has no level-3/4 altgr symbols; fall back to shifted/unshifted base level
                level = 1 if (shift and n_levels >= 2) else 0
            else:
                level = 1 if (shift and n_levels >= 2) else 0

        var base = (evdev_code * MAX_XKB_GROUPS + grp) * MAX_XKB_LEVELS
        return self.syms[base + level]

    def base_codepoint(self, evdev_code: Int) -> Int:
        """The unmodified (level 1) character of `evdev_code` in the active group, ignoring every
        modifier: what the key is labelled on this layout, for shortcut matching. 0 if none."""
        if evdev_code < 0 or evdev_code >= MAX_EVDEV_KEYS:
            return 0
        var n_groups = self.group_counts[evdev_code]
        if n_groups <= 0:
            return 0
        var grp = self.active_group % n_groups
        if self.level_counts[evdev_code * MAX_XKB_GROUPS + grp] <= 0:
            grp = 0
            if self.level_counts[evdev_code * MAX_XKB_GROUPS] <= 0:
                return 0
        return self.syms[(evdev_code * MAX_XKB_GROUPS + grp) * MAX_XKB_LEVELS]

    def translate_key_to_utf8(self, evdev_code: Int) -> String:
        var cp = self.translate_key_to_codepoint(evdev_code)
        if cp <= 0:
            return ""
        var bytes = List[UInt8](capacity=4)
        encode_codepoint(bytes, cp)
        return String(unsafe_from_utf8=bytes)
