"""The build stamp: which commit, whether the tree had changes, which Mojo.

`tools/build.sh` writes `core/build_stamp.mojo` before compiling anything, so
every binary knows the source it was built from. The stamp goes into
screenshots (a PNG text chunk), replay headers, saves, crash reports and the
performance history, so any of those can be traced back to its build.
"""
from flint.core.build_stamp import BUILD_COMMIT, BUILD_DIRTY, BUILD_MOJO

comptime STAMP_MAX_BYTES: Int = 96  # what saves and replay headers reserve for it
comptime FLINT_VERSION_MAJOR: Int = 5
comptime FLINT_VERSION_MINOR: Int = 0
comptime FLINT_VERSION_PATCH: Int = 0
comptime FLINT_ENGINE_VERSION: String = "5.0.0"


def flint_version_pin() -> String:
    """Canonical engine version + commit pin (`5.0.0+<commit>`)."""
    return String(FLINT_ENGINE_VERSION) + "+" + String(BUILD_COMMIT)


def _trim_toml_span(bytes: Span[UInt8, _], start: Int, end: Int) -> String:
    var s = start
    var e = end
    while s < e and (bytes[s] == 32 or bytes[s] == 9 or bytes[s] == 13):
        s += 1
    while e > s and (bytes[e - 1] == 32 or bytes[e - 1] == 9 or bytes[e - 1] == 13):
        e -= 1
    var out = List[UInt8](capacity=e - s if e > s else 0)
    for i in range(s, e):
        out.append(bytes[i])
    return String(unsafe_from_utf8=out)


def _unquote_toml_val(text: String) -> String:
    var b = text.as_bytes()
    var n = len(b)
    if n >= 2 and ((b[0] == 34 and b[n - 1] == 34) or (b[0] == 39 and b[n - 1] == 39)):
        var out = List[UInt8](capacity=n - 2)
        for i in range(1, n - 1):
            out.append(b[i])
        return String(unsafe_from_utf8=out)
    return text


def parse_flint_version_from_toml_bytes(raw: Span[UInt8, _]) -> String:
    """Extracts `[engine] flint_version` from `flint.toml` bytes (empty if omitted)."""
    var section = String("")
    var line_start = 0
    var n = len(raw)
    for idx in range(n + 1):
        if idx < n and raw[idx] != 10:
            continue
        var line_end = idx
        var quote_char: UInt8 = 0
        var escaped = False
        for c in range(line_start, line_end):
            var ch = raw[c]
            if quote_char == 0:
                if ch == 34 or ch == 39:
                    quote_char = ch
                    escaped = False
                elif ch == 35:
                    line_end = c
                    break
            else:
                if quote_char == 34 and escaped:
                    escaped = False
                elif quote_char == 34 and ch == 92:
                    escaped = True
                elif ch == quote_char:
                    quote_char = 0
        var trimmed = _trim_toml_span(raw, line_start, line_end)
        line_start = idx + 1
        var tb = trimmed.as_bytes()
        if len(tb) == 0:
            continue
        if len(tb) >= 2 and tb[0] == 91 and tb[len(tb) - 1] == 93:
            section = _trim_toml_span(tb, 1, len(tb) - 1)
            continue
        if section != "engine":
            continue
        var eq_pos = -1
        for c in range(len(tb)):
            if tb[c] == 61:
                eq_pos = c
                break
        if eq_pos <= 0:
            continue
        var key = _trim_toml_span(tb, 0, eq_pos)
        if key == "flint_version":
            return _unquote_toml_val(_trim_toml_span(tb, eq_pos + 1, len(tb)))
    return String()


def _parse_uint_digits(text: String) -> Tuple[Bool, Int]:
    var b = text.as_bytes()
    if len(b) == 0:
        return (False, 0)
    var val = 0
    for i in range(len(b)):
        if b[i] < 48 or b[i] > 57:
            return (False, 0)
        val = val * 10 + Int(b[i] - 48)
    return (True, val)


def _cmp_semver_tuple(a0: Int, a1: Int, a2: Int, b0: Int, b1: Int, b2: Int) -> Int:
    if a0 != b0:
        return 1 if a0 > b0 else -1
    if a1 != b1:
        return 1 if a1 > b1 else -1
    if a2 != b2:
        return 1 if a2 > b2 else -1
    return 0


def check_flint_version_pin(pin: String) -> Tuple[Bool, String]:
    """Validates an optional `[engine] flint_version` pin against this engine build."""
    var pb = pin.as_bytes()
    var trimmed = _trim_toml_span(pb, 0, len(pb))
    if trimmed.byte_length() == 0:
        return (True, String())

    var op = String("")
    var rest = trimmed
    if rest.startswith(">=") or rest.startswith("<=") or rest.startswith("==") or rest.startswith("~="):
        op = String(rest[byte=0:2])
        var r = String(rest[byte=2:])
        rest = r^
    elif rest.startswith(">") or rest.startswith("<") or rest.startswith("=") or rest.startswith("~") or rest.startswith("^"):
        op = String(rest[byte=0:1])
        var r = String(rest[byte=1:])
        rest = r^

    var rb = rest.as_bytes()
    var trimmed_rest = _trim_toml_span(rb, 0, len(rb))
    rest = trimmed_rest^
    if rest.startswith("v") or rest.startswith("V"):
        var r = String(rest[byte=1:])
        rest = r^

    var ver_part = rest
    var commit_part = String("")
    var plus_pos = rest.find("+")
    if plus_pos >= 0:
        ver_part = String(rest[byte=0:plus_pos])
        commit_part = String(rest[byte=plus_pos + 1 :])

    var dash_pos = ver_part.find("-")
    if dash_pos >= 0:
        var v = String(ver_part[byte=0:dash_pos])
        ver_part = v^

    var vb = ver_part.as_bytes()
    var parts = List[String]()
    var seg_start = 0
    for i in range(len(vb) + 1):
        if i == len(vb) or vb[i] == 46:
            var sb = List[UInt8](capacity=i - seg_start if i > seg_start else 0)
            for j in range(seg_start, i):
                sb.append(vb[j])
            parts.append(String(unsafe_from_utf8=sb))
            seg_start = i + 1

    if len(parts) < 1 or len(parts) > 3:
        return (False, "flint_version invalid: '" + pin + "' (expected semver like '5.0.0')")

    var p0 = _parse_uint_digits(parts[0])
    var p1 = _parse_uint_digits(parts[1]) if len(parts) >= 2 else (True, 0)
    var p2 = _parse_uint_digits(parts[2]) if len(parts) >= 3 else (True, 0)
    if not p0[0] or not p1[0] or not p2[0]:
        return (False, "flint_version invalid: '" + pin + "' (non-numeric version component)")

    var req_maj = p0[1]
    var req_min = p1[1]
    var req_pat = p2[1]
    var cmp = _cmp_semver_tuple(
        FLINT_VERSION_MAJOR,
        FLINT_VERSION_MINOR,
        FLINT_VERSION_PATCH,
        req_maj,
        req_min,
        req_pat,
    )

    var ver_ok: Bool
    if op == ">=":
        ver_ok = cmp >= 0
    elif op == ">":
        ver_ok = cmp > 0
    elif op == "<=":
        ver_ok = cmp <= 0
    elif op == "<":
        ver_ok = cmp < 0
    elif op == "^" or op == "~" or op == "~=":
        ver_ok = (req_maj == FLINT_VERSION_MAJOR) and (cmp >= 0)
    else:
        ver_ok = cmp == 0

    if not ver_ok:
        return (
            False,
            "flint_version mismatch: project requires '"
            + pin
            + "', incompatible with Flint Engine "
            + flint_version_pin(),
        )

    var cb = commit_part.as_bytes()
    var clean_commit = _trim_toml_span(cb, 0, len(cb))
    if (
        clean_commit.byte_length() > 0
        and clean_commit != "dirty"
        and String(BUILD_COMMIT) != "dev"
        and clean_commit != String(BUILD_COMMIT)
    ):
        return (
            False,
            "flint_version commit mismatch: project pins '"
            + pin
            + "', engine build is '"
            + flint_version_pin()
            + "'",
        )

    return (True, String())


def build_commit() -> String:
    """The short commit, with "+dirty" when the tree had uncommitted changes."""
    comptime if BUILD_DIRTY:
        return String(BUILD_COMMIT) + "+dirty"
    return String(BUILD_COMMIT)


def build_stamp() -> String:
    """One line: "<commit>[+dirty] mojo <version>". No commas, so it fits a CSV cell."""
    var text = build_commit() + " mojo " + String(BUILD_MOJO)
    if text.byte_length() > STAMP_MAX_BYTES:
        return String(text[byte=0:STAMP_MAX_BYTES])
    return text^


def stamp_words(stamp: String) -> List[Int32]:
    """The stamp packed four bytes to a word, little-endian, length first, for saves."""
    var bytes = stamp.as_bytes()
    var count = min(len(bytes), STAMP_MAX_BYTES)
    var words = List[Int32]()
    words.append(Int32(count))
    var index = 0
    while index < count:
        var word = UInt32(0)
        for shift in range(4):
            if index + shift < count:
                word |= UInt32(bytes[index + shift]) << UInt32(shift * 8)
        words.append(Int32(Int(word)))
        index += 4
    return words^


def stamp_from_words(words: List[Int32], start: Int) -> String:
    """Reverses `stamp_words` from `start`. Empty if the words are not a stamp."""
    if start < 0 or start >= len(words):
        return String()
    var count = Int(words[start])
    if count < 0 or count > STAMP_MAX_BYTES or start + 1 + (count + 3) // 4 > len(words):
        return String()
    var bytes = List[UInt8]()
    for index in range(count):
        var word = UInt32(Int(words[start + 1 + index // 4]) & 0xFFFFFFFF)
        var byte = UInt8((word >> UInt32((index % 4) * 8)) & 0xFF)
        if byte < 32 or byte > 126:
            return String()
        bytes.append(byte)
    return String(unsafe_from_utf8=bytes)
