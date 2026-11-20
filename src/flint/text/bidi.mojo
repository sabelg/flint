"""Unicode Bidirectional Algorithm (UAX #9) & Arabic Contextual Joining (`flint.text.bidi`).

Provides:
- `bidi_class(code)` & `bidi_mirror(code)`
- `resolve_bidi(codes, base_dir=BIDI_AUTO)` implementing P2/P3, X1-X10, W1-W7, N0 paired brackets, N1-N2, I1-I2
- `reorder_line(resolved, line_start, line_end)` implementing L1 trailing whitespace reset & L2 level reversal
- `arabic_joining_type(code)` & `compute_arabic_forms(codes)` implementing Unicode Arabic joining R1-R7
"""

# Paragraph base direction hints
comptime BIDI_AUTO: Int = -1
comptime BIDI_LTR: Int = 0
comptime BIDI_RTL: Int = 1

# UAX #9 Bidirectional character classes
comptime BIDI_L: Int = 0
comptime BIDI_R: Int = 1
comptime BIDI_AL: Int = 2
comptime BIDI_EN: Int = 3
comptime BIDI_ES: Int = 4
comptime BIDI_ET: Int = 5
comptime BIDI_AN: Int = 6
comptime BIDI_CS: Int = 7
comptime BIDI_NSM: Int = 8
comptime BIDI_BN: Int = 9
comptime BIDI_B: Int = 10
comptime BIDI_S: Int = 11
comptime BIDI_WS: Int = 12
comptime BIDI_ON: Int = 13
comptime BIDI_LRE: Int = 14
comptime BIDI_LRO: Int = 15
comptime BIDI_RLE: Int = 16
comptime BIDI_RLO: Int = 17
comptime BIDI_PDF: Int = 18
comptime BIDI_LRI: Int = 19
comptime BIDI_RLI: Int = 20
comptime BIDI_FSI: Int = 21
comptime BIDI_PDI: Int = 22

# Arabic contextual joining types & forms
comptime ARABIC_JOIN_U: Int = 0
comptime ARABIC_JOIN_R: Int = 1
comptime ARABIC_JOIN_D: Int = 2
comptime ARABIC_JOIN_L: Int = 3
comptime ARABIC_JOIN_C: Int = 4
comptime ARABIC_JOIN_T: Int = 5

comptime FORM_ISOL: Int = 0
comptime FORM_INIT: Int = 1
comptime FORM_MEDI: Int = 2
comptime FORM_FINA: Int = 3


@fieldwise_init
struct BidiRun(Copyable, ImplicitlyCopyable, Movable):
    var start: Int
    var end: Int
    var level: Int

    def is_rtl(self) -> Bool:
        return (self.level & 1) == 1


struct BidiResolved(Copyable, Movable):
    var base_level: Int
    var types: List[Int]
    var levels: List[Int]

    def __init__(out self, base_level: Int = 0):
        self.base_level = base_level
        self.types = List[Int]()
        self.levels = List[Int]()


def _is_nsm(code: Int) -> Bool:
    if code >= 0x0300 and code <= 0x036F:
        return True
    if code >= 0x0483 and code <= 0x0489:
        return True
    if (code >= 0x0591 and code <= 0x05BD) or code == 0x05BF or (code >= 0x05C1 and code <= 0x05C2) or (code >= 0x05C4 and code <= 0x05C5) or code == 0x05C7:
        return True
    if (code >= 0x0610 and code <= 0x061A) or (code >= 0x064B and code <= 0x065F) or code == 0x0670:
        return True
    if (code >= 0x06D6 and code <= 0x06DC) or (code >= 0x06DF and code <= 0x06E4) or (code >= 0x06E7 and code <= 0x06E8) or (code >= 0x06EA and code <= 0x06ED):
        return True
    if (code >= 0x20D0 and code <= 0x20FF) or (code >= 0xFE00 and code <= 0xFE0F) or (code >= 0xFE20 and code <= 0xFE2F):
        return True
    return False


def bidi_class(code: Int) -> Int:
    if code == 0x202A: return BIDI_LRE
    if code == 0x202B: return BIDI_RLE
    if code == 0x202C: return BIDI_PDF
    if code == 0x202D: return BIDI_LRO
    if code == 0x202E: return BIDI_RLO
    if code == 0x2066: return BIDI_LRI
    if code == 0x2067: return BIDI_RLI
    if code == 0x2068: return BIDI_FSI
    if code == 0x2069: return BIDI_PDI
    if code == 0x200E: return BIDI_L
    if code == 0x200F: return BIDI_R
    if code == 0x061C: return BIDI_AL
    if code == 0x0A or code == 0x0D or (code >= 0x1C and code <= 0x1E) or code == 0x85 or code == 0x2029:
        return BIDI_B
    if code == 0x09 or code == 0x0B or code == 0x1F:
        return BIDI_S
    if code == 0x20 or code == 0x0C or code == 0x1680 or (code >= 0x2000 and code <= 0x200A) or code == 0x2028 or code == 0x205F or code == 0x3000:
        return BIDI_WS
    if code == 0xAD or (code >= 0x200B and code <= 0x200D) or (code >= 0x2060 and code <= 0x2064) or code == 0xFEFF:
        return BIDI_BN
    if code >= 0 and code < 0x20:
        return BIDI_BN
    if _is_nsm(code):
        return BIDI_NSM
    if code == 0x2C or code == 0x2E or code == 0x2F or code == 0x3A or code == 0xA0 or code == 0x060C or code == 0x2044 or code == 0xFF0C or code == 0xFF0E or code == 0xFF0F or code == 0xFF1A:
        return BIDI_CS
    if code == 0x2B or code == 0x2D or code == 0x2212 or code == 0xFF0B or code == 0xFF0D:
        return BIDI_ES
    if (code >= 0x23 and code <= 0x25) or (code >= 0xA2 and code <= 0xA5) or code == 0xB0 or code == 0xB1 or code == 0x0609 or code == 0x060A or code == 0x066A or (code >= 0x2030 and code <= 0x2034) or (code >= 0x20A0 and code <= 0x20CF) or code == 0xFF03 or code == 0xFF04 or code == 0xFF05:
        return BIDI_ET
    if (code >= 0x30 and code <= 0x39) or code == 0xB2 or code == 0xB3 or code == 0xB9 or (code >= 0x06F0 and code <= 0x06F9) or (code >= 0xFF10 and code <= 0xFF19):
        return BIDI_EN
    if (code >= 0x0660 and code <= 0x0669) or code == 0x066B or code == 0x066C:
        return BIDI_AN
    if (code >= 0x0600 and code <= 0x06FF) or (code >= 0x0750 and code <= 0x077F) or (code >= 0x0870 and code <= 0x08FF) or (code >= 0xFB50 and code <= 0xFDFF) or (code >= 0xFE70 and code <= 0xFEFF) or (code >= 0x1EE00 and code <= 0x1EEFF):
        if code == 0x060E or code == 0x060F or code == 0x06DE or code == 0x06E9 or code == 0xFD3E or code == 0xFD3F:
            return BIDI_ON
        return BIDI_AL
    if (code >= 0x0590 and code <= 0x05FF) or (code >= 0x0700 and code <= 0x074F) or (code >= 0x0780 and code <= 0x085F) or (code >= 0xFB1D and code <= 0xFB4F) or (code >= 0x10800 and code <= 0x10FFF) or (code >= 0x1E800 and code <= 0x1EFFF):
        return BIDI_R
    if (code >= 0x21 and code <= 0x2F) or (code >= 0x3A and code <= 0x40) or (code >= 0x5B and code <= 0x60) or (code >= 0x7B and code <= 0x7E) or (code >= 0xA1 and code <= 0xBF) or (code >= 0x2010 and code <= 0x2027) or (code >= 0x2035 and code <= 0x205E) or (code >= 0x2190 and code <= 0x2BFF) or (code >= 0x3001 and code <= 0x303F) or (code >= 0xFE10 and code <= 0xFE1F) or (code >= 0xFF01 and code <= 0xFF0F) or (code >= 0xFF1A and code <= 0xFF20) or (code >= 0xFF3B and code <= 0xFF40) or (code >= 0xFF5B and code <= 0xFF65) or (code >= 0x1F000 and code <= 0x1FAFF):
        return BIDI_ON
    return BIDI_L


def bidi_mirror(code: Int) -> Int:
    match code:
        case 0x0028: return 0x0029  # ( -> )
        case 0x0029: return 0x0028  # ) -> (
        case 0x003C: return 0x003E  # < -> >
        case 0x003E: return 0x003C  # > -> <
        case 0x005B: return 0x005D  # [ -> ]
        case 0x005D: return 0x005B  # ] -> [
        case 0x007B: return 0x007D  # { -> }
        case 0x007D: return 0x007B  # } -> {
        case 0x00AB: return 0x00BB  # « -> »
        case 0x00BB: return 0x00AB  # » -> «
        case 0x2039: return 0x203A  # ‹ -> ›
        case 0x203A: return 0x2039  # › -> ‹
        case 0x27E8: return 0x27E9  # ⟨ -> ⟩
        case 0x27E9: return 0x27E8  # ⟩ -> ⟨
        case 0x27EA: return 0x27EB  # ⟪ -> ⟫
        case 0x27EB: return 0x27EA  # ⟫ -> ⟪
        case 0x300C: return 0x300D  # 「 -> 」
        case 0x300D: return 0x300C  # 」 -> 「
        case 0x300E: return 0x300F  # 『 -> 』
        case 0x300F: return 0x300E  # 』 -> 『
        case 0xFF08: return 0xFF09  # （ -> ）
        case 0xFF09: return 0xFF08  # ） -> （
        case 0xFF3B: return 0xFF3D  # ［ -> ］
        case 0xFF3D: return 0xFF3B  # ］ -> ［
        case 0xFF5B: return 0xFF5D  # ｛ -> ｝
        case 0xFF5D: return 0xFF5B  # ｝ -> ｛
        case _: return code


def _bracket_info(code: Int) -> Tuple[Int, Int]:
    """Returns (canonical_key, role) where role 1 = open, 2 = close, 0 = none."""
    match code:
        case 0x0028: return (0x0028, 1)
        case 0x0029: return (0x0028, 2)
        case 0x005B: return (0x005B, 1)
        case 0x005D: return (0x005B, 2)
        case 0x007B: return (0x007B, 1)
        case 0x007D: return (0x007B, 2)
        case 0x00AB: return (0x00AB, 1)
        case 0x00BB: return (0x00AB, 2)
        case 0x2039: return (0x2039, 1)
        case 0x203A: return (0x2039, 2)
        case 0x27E8: return (0x27E8, 1)
        case 0x27E9: return (0x27E8, 2)
        case 0x27EA: return (0x27EA, 1)
        case 0x27EB: return (0x27EA, 2)
        case 0x300C: return (0x300C, 1)
        case 0x300D: return (0x300C, 2)
        case 0x300E: return (0x300E, 1)
        case 0x300F: return (0x300E, 2)
        case 0xFF08: return (0xFF08, 1)
        case 0xFF09: return (0xFF08, 2)
        case 0xFF3B: return (0xFF3B, 1)
        case 0xFF3D: return (0xFF3B, 2)
        case 0xFF5B: return (0xFF5B, 1)
        case 0xFF5D: return (0xFF5B, 2)
        case _: return (0, 0)


def _detect_paragraph_level(types: List[Int], start: Int, end: Int) -> Int:
    var isolate_depth = 0
    for i in range(start, end):
        var t = types[i]
        if t == BIDI_LRI or t == BIDI_RLI or t == BIDI_FSI:
            isolate_depth += 1
        elif t == BIDI_PDI:
            if isolate_depth > 0:
                isolate_depth -= 1
        elif isolate_depth == 0:
            if t == BIDI_L:
                return 0
            if t == BIDI_R or t == BIDI_AL:
                return 1
    return 0


@always_inline
def _is_neutral_or_isolate(t: Int) -> Bool:
    return (
        t == BIDI_B or t == BIDI_S or t == BIDI_WS or t == BIDI_ON
        or t == BIDI_FSI or t == BIDI_LRI or t == BIDI_RLI or t == BIDI_PDI
    )


@always_inline
def _is_explicit_removed(t: Int) -> Bool:
    return (
        t == BIDI_LRE or t == BIDI_RLE or t == BIDI_LRO
        or t == BIDI_RLO or t == BIDI_PDF or t == BIDI_BN
    )


def _resolve_sequence(
    codes: List[Int],
    indices: List[Int],
    seq_level: Int,
    sos: Int,
    eos: Int,
    mut work_types: List[Int],
    mut levels: List[Int],
):
    var n = len(indices)
    if n == 0:
        return
    var emb_dir = BIDI_RTL if (seq_level & 1) == 1 else BIDI_LTR
    var emb_strong = BIDI_R if emb_dir == BIDI_RTL else BIDI_L

    # W1: NSM takes previous non-NSM type (or sos), or ON after LRI/RLI/FSI/PDI
    var prev_w1 = sos
    for k in range(n):
        var idx = indices[k]
        var t = work_types[idx]
        if t == BIDI_NSM:
            work_types[idx] = prev_w1
        elif t == BIDI_LRI or t == BIDI_RLI or t == BIDI_FSI or t == BIDI_PDI:
            prev_w1 = BIDI_ON
        else:
            prev_w1 = t

    # W2: EN -> AN when preceded by AL
    var last_strong_w2 = sos
    for k in range(n):
        var idx = indices[k]
        var t = work_types[idx]
        if t == BIDI_L or t == BIDI_R or t == BIDI_AL:
            last_strong_w2 = t
        elif t == BIDI_EN and last_strong_w2 == BIDI_AL:
            work_types[idx] = BIDI_AN

    # W3: AL -> R
    for k in range(n):
        var idx = indices[k]
        if work_types[idx] == BIDI_AL:
            work_types[idx] = BIDI_R

    # W4: single ES between EN, single CS between EN or AN
    for k in range(1, n - 1):
        var idx = indices[k]
        var t = work_types[idx]
        var prev_t = work_types[indices[k - 1]]
        var next_t = work_types[indices[k + 1]]
        if t == BIDI_ES and prev_t == BIDI_EN and next_t == BIDI_EN:
            work_types[idx] = BIDI_EN
        elif t == BIDI_CS and prev_t == next_t and (prev_t == BIDI_EN or prev_t == BIDI_AN):
            work_types[idx] = prev_t

    # W5: ET adjacent to EN -> EN
    for k in range(n):
        if work_types[indices[k]] == BIDI_ET:
            var adj_en = False
            var j = k - 1
            while j >= 0 and work_types[indices[j]] == BIDI_ET:
                j -= 1
            if j >= 0 and work_types[indices[j]] == BIDI_EN:
                adj_en = True
            if not adj_en:
                j = k + 1
                while j < n and work_types[indices[j]] == BIDI_ET:
                    j += 1
                if j < n and work_types[indices[j]] == BIDI_EN:
                    adj_en = True
            if adj_en:
                work_types[indices[k]] = BIDI_EN

    # W6: remaining ES, ET, CS -> ON
    for k in range(n):
        var idx = indices[k]
        var t = work_types[idx]
        if t == BIDI_ES or t == BIDI_ET or t == BIDI_CS:
            work_types[idx] = BIDI_ON

    # W7: EN -> L if last strong was L
    var last_strong_w7 = sos
    for k in range(n):
        var idx = indices[k]
        var t = work_types[idx]
        if t == BIDI_L or t == BIDI_R:
            last_strong_w7 = t
        elif t == BIDI_EN and last_strong_w7 == BIDI_L:
            work_types[idx] = BIDI_L

    # N0: Paired brackets
    var open_pos = List[Int]()
    var open_key = List[Int]()
    var pair_a = List[Int]()
    var pair_b = List[Int]()
    for k in range(n):
        var idx = indices[k]
        if work_types[idx] != BIDI_ON:
            continue
        var bi = _bracket_info(codes[idx])
        if bi[1] == 1:
            if len(open_pos) < 63:
                open_pos.append(k)
                open_key.append(bi[0])
        elif bi[1] == 2:
            var m = len(open_pos) - 1
            while m >= 0:
                if open_key[m] == bi[0]:
                    pair_a.append(open_pos[m])
                    pair_b.append(k)
                    while len(open_pos) > m:
                        _ = open_pos.pop()
                        _ = open_key.pop()
                    break
                m -= 1
    # Sort pairs by opening position ascending
    for p in range(1, len(pair_a)):
        var ka = pair_a[p]
        var kb = pair_b[p]
        var q = p - 1
        while q >= 0 and pair_a[q] > ka:
            pair_a[q + 1] = pair_a[q]
            pair_b[q + 1] = pair_b[q]
            q -= 1
        pair_a[q + 1] = ka
        pair_b[q + 1] = kb
    for p in range(len(pair_a)):
        var ka = pair_a[p]
        var kb = pair_b[p]
        var has_emb = False
        var has_opp = False
        for m in range(ka + 1, kb):
            var mt = work_types[indices[m]]
            var st = mt
            if mt == BIDI_EN or mt == BIDI_AN:
                st = BIDI_R
            if st == emb_strong:
                has_emb = True
                break
            elif st == BIDI_L or st == BIDI_R:
                has_opp = True
        if has_emb:
            work_types[indices[ka]] = emb_strong
            work_types[indices[kb]] = emb_strong
        elif has_opp:
            var prev_strong = sos
            var m = ka - 1
            while m >= 0:
                var mt = work_types[indices[m]]
                if mt == BIDI_L:
                    prev_strong = BIDI_L
                    break
                elif mt == BIDI_R or mt == BIDI_EN or mt == BIDI_AN:
                    prev_strong = BIDI_R
                    break
                m -= 1
            var target = prev_strong if (prev_strong == BIDI_L or prev_strong == BIDI_R) else emb_strong
            work_types[indices[ka]] = target
            work_types[indices[kb]] = target

    # N1 & N2: Neutrals and isolate initiators/terminators
    var k = 0
    while k < n:
        if not _is_neutral_or_isolate(work_types[indices[k]]):
            k += 1
            continue
        var run_start = k
        while k < n and _is_neutral_or_isolate(work_types[indices[k]]):
            k += 1
        var run_end = k
        var prev_s = sos
        if run_start > 0:
            var pt = work_types[indices[run_start - 1]]
            prev_s = BIDI_R if (pt == BIDI_EN or pt == BIDI_AN) else pt
        var next_s = eos
        if run_end < n:
            var nt = work_types[indices[run_end]]
            next_s = BIDI_R if (nt == BIDI_EN or nt == BIDI_AN) else nt
        var resolved_n = prev_s if prev_s == next_s else emb_strong
        for m in range(run_start, run_end):
            work_types[indices[m]] = resolved_n

    # I1 & I2: Implicit levels
    for idx_k in range(n):
        var idx = indices[idx_k]
        var t = work_types[idx]
        if (seq_level & 1) == 0:
            if t == BIDI_R:
                levels[idx] = seq_level + 1
            elif t == BIDI_AN or t == BIDI_EN:
                levels[idx] = seq_level + 2
            else:
                levels[idx] = seq_level
        else:
            if t == BIDI_L or t == BIDI_EN or t == BIDI_AN:
                levels[idx] = seq_level + 1
            else:
                levels[idx] = seq_level


def resolve_bidi(codes: List[Int], base_dir: Int = BIDI_AUTO) -> BidiResolved:
    """Resolve per-codepoint UAX #9 embedding levels."""
    var n = len(codes)
    var out = BidiResolved(0)
    if n == 0:
        out.base_level = 1 if base_dir == BIDI_RTL else 0
        return out^

    var orig_types = List[Int](capacity=n)
    for c in codes:
        orig_types.append(bidi_class(c))

    var base_level: Int
    if base_dir == BIDI_LTR:
        base_level = 0
    elif base_dir == BIDI_RTL:
        base_level = 1
    else:
        base_level = _detect_paragraph_level(orig_types, 0, n)

    out.base_level = base_level
    out.types = orig_types.copy()
    var work_types = orig_types.copy()
    var levels = List[Int](length=n, fill=base_level)

    # X1-X8: Directional stack
    var stack_lvl = List[Int]()
    var stack_ovr = List[Int]()  # 0=none, 1=L, 2=R
    var stack_iso = List[Bool]()
    stack_lvl.append(base_level)
    stack_ovr.append(0)
    stack_iso.append(False)
    var overflow_iso = 0
    var overflow_emb = 0
    var valid_iso = 0
    comptime MAX_DEPTH: Int = 125

    for i in range(n):
        var t = orig_types[i]
        var cur_lvl = stack_lvl[len(stack_lvl) - 1]
        var cur_ovr = stack_ovr[len(stack_ovr) - 1]
        if t == BIDI_B:
            levels[i] = base_level
            while len(stack_lvl) > 1:
                _ = stack_lvl.pop(); _ = stack_ovr.pop(); _ = stack_iso.pop()
            overflow_iso = 0
            overflow_emb = 0
            valid_iso = 0
            continue
        if t == BIDI_RLE or t == BIDI_LRE or t == BIDI_RLO or t == BIDI_LRO:
            var is_r = (t == BIDI_RLE or t == BIDI_RLO)
            var is_ovr = (t == BIDI_RLO or t == BIDI_LRO)
            var new_lvl = (cur_lvl + 1) | 1 if is_r else (cur_lvl + 2) & ~1
            if new_lvl <= MAX_DEPTH and overflow_iso == 0 and overflow_emb == 0:
                stack_lvl.append(new_lvl)
                stack_ovr.append((2 if is_r else 1) if is_ovr else 0)
                stack_iso.append(False)
            elif overflow_iso == 0:
                overflow_emb += 1
            levels[i] = cur_lvl
        elif t == BIDI_RLI or t == BIDI_LRI or t == BIDI_FSI:
            levels[i] = cur_lvl
            if cur_ovr == 1:
                work_types[i] = BIDI_L
            elif cur_ovr == 2:
                work_types[i] = BIDI_R
            var is_r = (t == BIDI_RLI)
            if t == BIDI_FSI:
                var iso_d = 0
                var sub_r = False
                for j in range(i + 1, n):
                    var tj = orig_types[j]
                    if tj == BIDI_LRI or tj == BIDI_RLI or tj == BIDI_FSI:
                        iso_d += 1
                    elif tj == BIDI_PDI:
                        if iso_d == 0:
                            break
                        iso_d -= 1
                    elif iso_d == 0:
                        if tj == BIDI_L:
                            break
                        if tj == BIDI_R or tj == BIDI_AL:
                            sub_r = True
                            break
                is_r = sub_r
            var new_lvl = (cur_lvl + 1) | 1 if is_r else (cur_lvl + 2) & ~1
            if new_lvl <= MAX_DEPTH and overflow_iso == 0 and overflow_emb == 0:
                valid_iso += 1
                stack_lvl.append(new_lvl)
                stack_ovr.append(0)
                stack_iso.append(True)
            else:
                overflow_iso += 1
        elif t == BIDI_PDI:
            if overflow_iso > 0:
                overflow_iso -= 1
            elif valid_iso > 0:
                overflow_emb = 0
                while len(stack_lvl) > 1 and not stack_iso[len(stack_iso) - 1]:
                    _ = stack_lvl.pop(); _ = stack_ovr.pop(); _ = stack_iso.pop()
                if len(stack_lvl) > 1:
                    _ = stack_lvl.pop(); _ = stack_ovr.pop(); _ = stack_iso.pop()
                    valid_iso -= 1
            levels[i] = stack_lvl[len(stack_lvl) - 1]
            var p_ovr = stack_ovr[len(stack_ovr) - 1]
            if p_ovr == 1:
                work_types[i] = BIDI_L
            elif p_ovr == 2:
                work_types[i] = BIDI_R
        elif t == BIDI_PDF:
            levels[i] = cur_lvl
            if overflow_iso == 0:
                if overflow_emb > 0:
                    overflow_emb -= 1
                elif len(stack_lvl) > 1 and not stack_iso[len(stack_iso) - 1]:
                    _ = stack_lvl.pop(); _ = stack_ovr.pop(); _ = stack_iso.pop()
        elif t == BIDI_BN:
            levels[i] = cur_lvl
        else:
            levels[i] = cur_lvl
            if cur_ovr == 1:
                work_types[i] = BIDI_L
            elif cur_ovr == 2:
                work_types[i] = BIDI_R

    # X10: Partition non-BN/explicit characters into level runs & resolve each run
    var i = 0
    var prev_non_removed_level = base_level
    while i < n:
        if _is_explicit_removed(orig_types[i]):
            i += 1
            continue
        var run_lvl = levels[i]
        var seq_indices = List[Int]()
        var j = i
        while j < n:
            if _is_explicit_removed(orig_types[j]):
                j += 1
                continue
            if levels[j] != run_lvl:
                break
            seq_indices.append(j)
            j += 1
        var next_lvl = base_level
        var k = j
        while k < n:
            if not _is_explicit_removed(orig_types[k]):
                next_lvl = levels[k]
                break
            k += 1
        var sos = BIDI_R if (max(prev_non_removed_level, run_lvl) & 1) == 1 else BIDI_L
        var eos = BIDI_R if (max(run_lvl, next_lvl) & 1) == 1 else BIDI_L
        _resolve_sequence(codes, seq_indices, run_lvl, sos, eos, work_types, levels)
        prev_non_removed_level = run_lvl
        i = j

    # Assign BN/explicit characters level of neighbor for clean run grouping
    for idx in range(n):
        if _is_explicit_removed(orig_types[idx]):
            levels[idx] = levels[idx - 1] if idx > 0 else base_level

    out.levels = levels^
    return out^


def reorder_line(resolved: BidiResolved, line_start: Int, line_end: Int) -> List[BidiRun]:
    """UAX #9 L1 + L2 per-line visual reordering. Returns runs in visual L-to-R order."""
    var runs = List[BidiRun]()
    var n = len(resolved.levels)
    var s = max(0, line_start)
    var e = min(n, line_end)
    if e <= s:
        return runs^

    var count = e - s
    var line_levels = List[Int](capacity=count)
    for i in range(s, e):
        line_levels.append(resolved.levels[i])

    # L1: Reset B, S, and trailing whitespace/isolate chars to base_level
    for i in range(count):
        var t = resolved.types[s + i]
        if t == BIDI_B or t == BIDI_S:
            line_levels[i] = resolved.base_level
            var b = i - 1
            while b >= 0:
                var bt = resolved.types[s + b]
                if bt == BIDI_WS or bt == BIDI_FSI or bt == BIDI_LRI or bt == BIDI_RLI or bt == BIDI_PDI or _is_explicit_removed(bt):
                    line_levels[b] = resolved.base_level
                    b -= 1
                else:
                    break
    var tail = count - 1
    while tail >= 0:
        var tt = resolved.types[s + tail]
        if tt == BIDI_WS or tt == BIDI_FSI or tt == BIDI_LRI or tt == BIDI_RLI or tt == BIDI_PDI or _is_explicit_removed(tt):
            line_levels[tail] = resolved.base_level
            tail -= 1
        else:
            break

    # Group contiguous equal-level segments into runs
    var idx = 0
    var max_lvl = 0
    var min_odd = 999
    while idx < count:
        var lvl = line_levels[idx]
        max_lvl = max(max_lvl, lvl)
        if (lvl & 1) == 1:
            min_odd = min(min_odd, lvl)
        var r_start = s + idx
        while idx < count and line_levels[idx] == lvl:
            idx += 1
        runs.append(BidiRun(r_start, s + idx, lvl))

    # L2: Reverse contiguous runs with level >= lvl from max_lvl down to min_odd
    if min_odd <= max_lvl:
        var lvl = max_lvl
        while lvl >= min_odd:
            var r = 0
            while r < len(runs):
                if runs[r].level < lvl:
                    r += 1
                    continue
                var r_end = r + 1
                while r_end < len(runs) and runs[r_end].level >= lvl:
                    r_end += 1
                var lo = r
                var hi = r_end - 1
                while lo < hi:
                    var tmp = runs[lo]
                    runs[lo] = runs[hi]
                    runs[hi] = tmp
                    lo += 1
                    hi -= 1
                r = r_end
            lvl -= 1
    return runs^


def arabic_joining_type(code: Int) -> Int:
    if code == 0x0640 or code == 0x200D:
        return ARABIC_JOIN_C
    if code == 0x200C or code == 0x0621:
        return ARABIC_JOIN_U
    if _is_nsm(code):
        return ARABIC_JOIN_T
    # Right-joining Arabic letters (do not join to the following letter)
    if (
        (code >= 0x0622 and code <= 0x0625)
        or code == 0x0627
        or code == 0x0629
        or (code >= 0x062F and code <= 0x0632)
        or code == 0x0648
        or (code >= 0x0671 and code <= 0x0677)
        or (code >= 0x0688 and code <= 0x0699)
        or code == 0x06C0
        or (code >= 0x06C3 and code <= 0x06CB)
        or code == 0x06CD
        or code == 0x06CF
        or (code >= 0x06D2 and code <= 0x06D3)
        or code == 0x06D5
    ):
        return ARABIC_JOIN_R
    # Dual-joining Arabic letters (join both sides)
    if (
        code == 0x0620
        or code == 0x0626
        or code == 0x0628
        or (code >= 0x062A and code <= 0x062E)
        or (code >= 0x0633 and code <= 0x063F)
        or (code >= 0x0641 and code <= 0x0647)
        or (code >= 0x0649 and code <= 0x064A)
        or (code >= 0x0678 and code <= 0x0687)
        or (code >= 0x069A and code <= 0x06BF)
        or code == 0x06C1
        or code == 0x06C2
        or code == 0x06CC
        or code == 0x06CE
        or (code >= 0x06D0 and code <= 0x06D1)
        or (code >= 0x06FA and code <= 0x06FC)
        or code == 0x06FF
    ):
        return ARABIC_JOIN_D
    return ARABIC_JOIN_U


def compute_arabic_forms(codes: List[Int]) -> List[Int]:
    """Compute contextual joining forms (FORM_ISOL..FORM_FINA) per Unicode rules R1-R7."""
    var n = len(codes)
    var forms = List[Int](length=n, fill=FORM_ISOL)
    var jtypes = List[Int](capacity=n)
    for c in codes:
        jtypes.append(arabic_joining_type(c))
    for i in range(n):
        var cur = jtypes[i]
        if cur != ARABIC_JOIN_R and cur != ARABIC_JOIN_D and cur != ARABIC_JOIN_L:
            forms[i] = FORM_ISOL
            continue
        var prev_j = ARABIC_JOIN_U
        var p = i - 1
        while p >= 0:
            if jtypes[p] != ARABIC_JOIN_T:
                prev_j = jtypes[p]
                break
            p -= 1
        var next_j = ARABIC_JOIN_U
        var q = i + 1
        while q < n:
            if jtypes[q] != ARABIC_JOIN_T:
                next_j = jtypes[q]
                break
            q += 1
        var joins_prev = (cur == ARABIC_JOIN_R or cur == ARABIC_JOIN_D) and (
            prev_j == ARABIC_JOIN_D or prev_j == ARABIC_JOIN_L or prev_j == ARABIC_JOIN_C
        )
        var joins_next = (cur == ARABIC_JOIN_L or cur == ARABIC_JOIN_D) and (
            next_j == ARABIC_JOIN_R or next_j == ARABIC_JOIN_D or next_j == ARABIC_JOIN_C
        )
        if joins_prev and joins_next:
            forms[i] = FORM_MEDI
        elif joins_prev:
            forms[i] = FORM_FINA
        elif joins_next:
            forms[i] = FORM_INIT
        else:
            forms[i] = FORM_ISOL
    return forms^
