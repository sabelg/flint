"""Pure-Mojo 1.2 compiler error parser, unified diagnostic store & Mojo symbol outline/index (`flint.devtools.problems`).

Provides:
- `parse_compiler_output()`: parses Mojo `path:line:col: error|warning|note: message` diagnostics
  and attaches compiler caret fix-it hints.
- `ProblemStore` & `ProblemFilter`: merges diagnostics from LSP, background builds, tests, and agents
  with current-file prioritization, stable line/col sorting, and deduplication.
- `index_mojo_source()` & `SymbolIndex`: zero-LSP indentation-aware Mojo symbol outline (`def`, `struct`,
  `__extension`, `trait`, top-level `comptime`), breadcrumbs, and project symbol lookup.
"""
from std.collections import Dict, List
from std.math import max
from flint.platform.os import read_bytes

comptime DIAG_SRC_LSP: Int = 0
comptime DIAG_SRC_BUILD: Int = 1
comptime DIAG_SRC_TEST: Int = 2
comptime DIAG_SRC_AGENT: Int = 3

comptime DIAG_ERROR: Int = 1
comptime DIAG_WARNING: Int = 2
comptime DIAG_INFO: Int = 3
comptime DIAG_HINT: Int = 4

comptime SYM_MODULE: Int = 2
comptime SYM_CLASS: Int = 5
comptime SYM_METHOD: Int = 6
comptime SYM_PROPERTY: Int = 7
comptime SYM_FIELD: Int = 8
comptime SYM_TRAIT: Int = 11
comptime SYM_FUNCTION: Int = 12
comptime SYM_VARIABLE: Int = 13
comptime SYM_CONSTANT: Int = 14
comptime SYM_STRUCT: Int = 23


def _lower_ascii(s: String) -> String:
    var b = s.as_bytes()
    var out = List[UInt8](capacity=len(b))
    for i in range(len(b)):
        var c = b[i]
        if c >= 65 and c <= 90:
            out.append(c + 32)
        else:
            out.append(c)
    return String(unsafe_from_utf8=out)


@always_inline
def _is_ident_byte(b: UInt8) -> Bool:
    return (
        (b >= 65 and b <= 90)
        or (b >= 97 and b <= 122)
        or (b >= 48 and b <= 57)
        or b == 95
        or b >= 128
    )


def severity_label(sev: Int) -> String:
    if sev == DIAG_ERROR:
        return "error"
    if sev == DIAG_WARNING:
        return "warning"
    if sev == DIAG_HINT:
        return "hint"
    return "note"


def source_label(src: Int) -> String:
    if src == DIAG_SRC_BUILD:
        return "build"
    if src == DIAG_SRC_TEST:
        return "test"
    if src == DIAG_SRC_AGENT:
        return "agent"
    return "lsp"


struct Diagnostic(Copyable, Movable):
    """Single compiler, LSP, test, or agent diagnostic."""
    var source: Int
    var severity: Int
    var path: String
    var line: Int        # 1-based (0: file-level)
    var col: Int         # 1-based
    var end_line: Int    # 1-based (0: unspecified)
    var end_col: Int     # 1-based (0: unspecified)
    var message: String
    var notes: List[String]

    def __init__(
        out self,
        source: Int = DIAG_SRC_BUILD,
        severity: Int = DIAG_ERROR,
        path: String = "",
        line: Int = 0,
        col: Int = 0,
        message: String = "",
        end_line: Int = 0,
        end_col: Int = 0,
    ):
        self.source = source
        self.severity = severity
        self.path = path
        self.line = line
        self.col = col
        self.end_line = end_line
        self.end_col = end_col
        self.message = message
        self.notes = List[String]()

    def format_row(self) -> String:
        var tag = ("  [" + source_label(self.source) + "]") if self.source != DIAG_SRC_LSP else String("")
        return (
            self.path
            + ":"
            + String(self.line)
            + ":"
            + String(self.col)
            + ": "
            + severity_label(self.severity)
            + ": "
            + self.message
            + tag
        )


def parse_compiler_output(
    output: String, base_dir: String = "", source: Int = DIAG_SRC_BUILD
) -> List[Diagnostic]:
    """Parses Mojo/Clang compiler output lines (`path:line:col: error|warning|note: msg`)."""
    var out = List[Diagnostic]()
    var saw_caret = False
    for raw in output.split("\n"):
        var l = String(String(raw).rstrip("\r"))
        var sev = 0
        var at = -1
        var word_len = 0
        var k_err = l.find(": error: ")
        if k_err > 0:
            at = k_err
            sev = DIAG_ERROR
            word_len = 9
        var k_warn = l.find(": warning: ")
        if k_warn > 0 and (at < 0 or k_warn < at):
            at = k_warn
            sev = DIAG_WARNING
            word_len = 11
        var k_note = l.find(": note: ")
        if k_note > 0 and (at < 0 or k_note < at):
            at = k_note
            sev = DIAG_INFO
            word_len = 8

        if at < 0:
            var trimmed = String(l.strip())
            if len(out) > 0 and trimmed.startswith("^"):
                saw_caret = True
            elif saw_caret and len(out) > 0:
                saw_caret = False
                if l.startswith(" ") and trimmed.byte_length() > 0:
                    out[len(out) - 1].notes.append("fix-it: " + trimmed)
            else:
                saw_caret = False
            continue

        saw_caret = False
        var loc = String(l[byte=0:at])
        var parts = loc.split(":")
        var n = len(parts)
        if n < 3:
            continue
        var line: Int
        var col: Int
        try:
            line = atol(String(parts[n - 2]))
            col = atol(String(parts[n - 1]))
        except:
            continue
        var path = String("")
        for k in range(n - 2):
            path += (":" if k > 0 else "") + String(parts[k])
        if not path.startswith("/") and base_dir.byte_length() > 0:
            if base_dir.endswith("/"):
                path = base_dir + path
            else:
                path = base_dir + "/" + path
        var msg = String(l[byte=at + word_len : l.byte_length()])
        out.append(Diagnostic(source, sev, path, line, col, msg))
    return out^


struct ProblemFilter(Copyable, Movable):
    var errors: Bool
    var warnings: Bool
    var infos: Bool
    var this_file_only: Bool
    var query: String

    def __init__(out self):
        self.errors = True
        self.warnings = True
        self.infos = True
        self.this_file_only = False
        self.query = ""

    def matches(self, d: Diagnostic, current_file: String) -> Bool:
        if d.severity == DIAG_ERROR and not self.errors:
            return False
        if d.severity == DIAG_WARNING and not self.warnings:
            return False
        if d.severity > DIAG_WARNING and not self.infos:
            return False
        if self.this_file_only and current_file.byte_length() > 0 and d.path != current_file:
            return False
        if self.query.byte_length() > 0:
            if _lower_ascii(d.format_row()).find(_lower_ascii(self.query)) < 0:
                return False
        return True


def order_diagnostics(diags: List[Diagnostic], current_file: String = "") -> List[Int]:
    """Orders diagnostics: current file first, then file discovery order, sorted by (line, col),
    keeping build `: note:` entries attached to their parent diagnostic and deduplicating."""
    var n = len(diags)
    var parent_of = List[Int](length=n, fill=-1)
    for k in range(n):
        if diags[k].source == DIAG_SRC_BUILD and diags[k].severity == DIAG_INFO:
            var j = k - 1
            while j >= 0 and diags[j].source == DIAG_SRC_BUILD:
                if diags[j].severity != DIAG_INFO:
                    parent_of[k] = j
                    break
                j -= 1

    var files = List[String]()
    if current_file.byte_length() > 0:
        files.append(current_file)
    for k in range(n):
        if parent_of[k] < 0 and diags[k].path not in files:
            files.append(diags[k].path)
    for k in range(n):
        if diags[k].path not in files:
            files.append(diags[k].path)

    var out = List[Int]()
    var seen = Dict[String, Bool]()
    for fi in range(len(files)):
        ref f = files[fi]
        var idx = List[Int]()
        for k in range(n):
            if parent_of[k] < 0 and diags[k].path == f:
                idx.append(k)
        for a in range(1, len(idx)):
            var v = idx[a]
            var b = a
            while b > 0 and (
                diags[idx[b - 1]].line > diags[v].line
                or (diags[idx[b - 1]].line == diags[v].line and diags[idx[b - 1]].col > diags[v].col)
            ):
                idx[b] = idx[b - 1]
                b -= 1
            idx[b] = v
        for ii in range(len(idx)):
            var k = idx[ii]
            var key = (
                diags[k].path
                + ":"
                + String(diags[k].line)
                + ":"
                + String(diags[k].col)
                + ":"
                + String(diags[k].severity)
                + ":"
                + diags[k].message
            )
            if key in seen:
                continue
            seen[key] = True
            out.append(k)
            for c in range(n):
                if parent_of[c] == k:
                    var ckey = (
                        diags[c].path
                        + ":"
                        + String(diags[c].line)
                        + ":"
                        + String(diags[c].col)
                        + ":"
                        + String(diags[c].severity)
                        + ":"
                        + diags[c].message
                    )
                    if ckey not in seen:
                        seen[ckey] = True
                        out.append(c)
    return out^


struct ProblemStore(Copyable, Movable):
    """Unified multi-source diagnostics container for Code Drawer, Status Bar & Agents."""
    var items: List[Diagnostic]
    var version: UInt64

    def __init__(out self):
        self.items = List[Diagnostic]()
        self.version = 0

    def clear(mut self):
        self.items.clear()
        self.version += 1

    def clear_source(mut self, source: Int):
        var kept = List[Diagnostic]()
        for i in range(len(self.items)):
            if self.items[i].source != source:
                kept.append(self.items[i].copy())
        self.items = kept^
        self.version += 1

    def replace_source(mut self, source: Int, diags: List[Diagnostic]):
        var kept = List[Diagnostic]()
        for i in range(len(self.items)):
            if self.items[i].source != source:
                kept.append(self.items[i].copy())
        for i in range(len(diags)):
            var d = diags[i].copy()
            d.source = source
            kept.append(d^)
        self.items = kept^
        self.version += 1

    def replace_file_source(mut self, source: Int, path: String, diags: List[Diagnostic]):
        var kept = List[Diagnostic]()
        for i in range(len(self.items)):
            if not (self.items[i].source == source and self.items[i].path == path):
                kept.append(self.items[i].copy())
        for i in range(len(diags)):
            var d = diags[i].copy()
            d.source = source
            if d.path.byte_length() == 0:
                d.path = path
            kept.append(d^)
        self.items = kept^
        self.version += 1

    def add(mut self, diag: Diagnostic):
        self.items.append(diag.copy())
        self.version += 1

    def error_count(self) -> Int:
        var n = 0
        for i in range(len(self.items)):
            if self.items[i].severity == DIAG_ERROR:
                n += 1
        return n

    def warning_count(self) -> Int:
        var n = 0
        for i in range(len(self.items)):
            if self.items[i].severity == DIAG_WARNING:
                n += 1
        return n

    def for_file(self, path: String) -> List[Diagnostic]:
        var out = List[Diagnostic]()
        var order = order_diagnostics(self.items, path)
        for i in range(len(order)):
            ref d = self.items[order[i]]
            if d.path == path:
                out.append(d.copy())
        return out^

    def query(self, filter: ProblemFilter, current_file: String = "") -> List[Diagnostic]:
        var out = List[Diagnostic]()
        var order = order_diagnostics(self.items, current_file)
        for i in range(len(order)):
            ref d = self.items[order[i]]
            if filter.matches(d, current_file):
                out.append(d.copy())
        return out^


# ============================================================================
# Lightweight Mojo Symbol Outline & Workspace Index
# ============================================================================

@fieldwise_init
struct SymbolEntry(Copyable, Movable):
    var name: String
    var kind: Int       # SYM_* (LSP SymbolKind compatible)
    var path: String
    var line: Int       # 0-based line index
    var col: Int        # 0-based byte column of symbol name
    var depth: Int      # nesting depth (0 = top-level)
    var owner: String   # enclosing struct/trait name ("" at top-level)


def symbol_tag(kind: Int) -> String:
    if kind == SYM_FUNCTION or kind == SYM_METHOD:
        return "fn"
    if kind == SYM_STRUCT or kind == SYM_CLASS:
        return "struct"
    if kind == SYM_TRAIT:
        return "trait"
    if kind == SYM_CONSTANT or kind == SYM_VARIABLE:
        return "const"
    return "sym"


def _parent_kind_at(entries: List[SymbolEntry], depth: Int) -> Int:
    var k = len(entries) - 1
    while k >= 0:
        if entries[k].depth == depth:
            return entries[k].kind
        k -= 1
    return 0


def index_mojo_source(text: String, path: String = "") -> List[SymbolEntry]:
    """Scans Mojo source into a nested outline of `def`, `struct`, `trait`, `__extension`, and `comptime`."""
    var entries = List[SymbolEntry]()
    var indents = List[Int]()
    var open_names = List[String]()
    var in_doc: UInt8 = 0
    var row = 0

    for raw in text.split("\n"):
        var line = String(String(raw).rstrip("\r"))
        var b = line.as_bytes()
        var n = len(b)
        var ind = 0
        while ind < n and (b[ind] == 32 or b[ind] == 9):
            ind += 1

        var started_inside = in_doc != 0
        var q = 0
        while q + 2 < n:
            if (b[q] == 34 or b[q] == 39) and b[q + 1] == b[q] and b[q + 2] == b[q]:
                if in_doc == 0:
                    in_doc = b[q]
                    q += 3
                    continue
                elif in_doc == b[q]:
                    in_doc = 0
                    q += 3
                    continue
            q += 1

        if not started_inside and ind < n and b[ind] != 35:
            var rest = String(line[byte=ind:n])
            var kind = 0
            var skip = 0
            if rest.startswith("def "):
                kind = SYM_FUNCTION
                skip = 4
            elif rest.startswith("struct "):
                kind = SYM_STRUCT
                skip = 7
            elif rest.startswith("__extension "):
                kind = SYM_STRUCT
                skip = 12
            elif rest.startswith("trait "):
                kind = SYM_TRAIT
                skip = 6
            elif rest.startswith("comptime ") and ind == 0:
                kind = SYM_CONSTANT
                skip = 9

            if kind != 0:
                var s0 = ind + skip
                while s0 < n and b[s0] == 32:
                    s0 += 1
                var e = s0
                while e < n and _is_ident_byte(b[e]):
                    e += 1
                if e > s0:
                    while len(indents) > 0 and indents[len(indents) - 1] >= ind:
                        _ = indents.pop()
                        if len(open_names) > 0:
                            _ = open_names.pop()
                    var depth = len(indents)
                    if kind == SYM_FUNCTION and depth > 0:
                        var pk = _parent_kind_at(entries, depth - 1)
                        if pk == SYM_STRUCT or pk == SYM_TRAIT:
                            kind = SYM_METHOD
                    var owner = open_names[len(open_names) - 1] if len(open_names) > 0 else String("")
                    var name = String(line[byte=s0:e])
                    entries.append(SymbolEntry(name, kind, path, row, s0, depth, owner))
                    indents.append(ind)
                    open_names.append(name)
        row += 1

    return entries^


def scope_breadcrumbs(entries: List[SymbolEntry], line: Int) -> String:
    """Returns `'Struct > method'` breadcrumb trail enclosing 0-based `line`."""
    var path_idx = List[Int]()
    var n = len(entries)
    for k in range(n):
        if entries[k].line > line:
            break
        var end_line = 1 << 30
        for j in range(k + 1, n):
            if entries[j].depth <= entries[k].depth:
                end_line = entries[j].line
                break
        if line < end_line:
            while len(path_idx) > 0 and entries[path_idx[len(path_idx) - 1]].depth >= entries[k].depth:
                _ = path_idx.pop()
            path_idx.append(k)
    var out = String("")
    for i in range(len(path_idx)):
        if i > 0:
            out += " > "
        out += entries[path_idx[i]].name
    return out^


struct SymbolIndex(Movable):
    """Lightweight in-memory symbol index across project Mojo files."""
    var entries: List[SymbolEntry]

    def __init__(out self):
        self.entries = List[SymbolEntry]()

    def clear(mut self):
        self.entries.clear()

    def remove_file(mut self, path: String):
        var kept = List[SymbolEntry]()
        for i in range(len(self.entries)):
            if self.entries[i].path != path:
                kept.append(self.entries[i].copy())
        self.entries = kept^

    def index_source(mut self, path: String, text: String):
        self.remove_file(path)
        var found = index_mojo_source(text, path)
        for i in range(len(found)):
            self.entries.append(found[i].copy())

    def index_file(mut self, path: String) -> Bool:
        var raw = read_bytes(path)
        if len(raw) == 0:
            self.remove_file(path)
            return False
        var txt = String(unsafe_from_utf8=raw)
        self.index_source(path, txt)
        return True

    def symbols_in_file(self, path: String) -> List[SymbolEntry]:
        var out = List[SymbolEntry]()
        for i in range(len(self.entries)):
            if self.entries[i].path == path:
                out.append(self.entries[i].copy())
        return out^

    def find_exact(self, name: String) -> List[SymbolEntry]:
        var out = List[SymbolEntry]()
        for i in range(len(self.entries)):
            if self.entries[i].name == name:
                out.append(self.entries[i].copy())
        return out^

    def search(self, query: String, limit: Int = 64) -> List[SymbolEntry]:
        var out = List[SymbolEntry]()
        var q = _lower_ascii(query)
        var cap = max(1, limit)
        for i in range(len(self.entries)):
            if len(out) >= cap:
                break
            if q.byte_length() == 0 or _lower_ascii(self.entries[i].name).find(q) >= 0:
                out.append(self.entries[i].copy())
        return out^
