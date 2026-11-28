"""Pure-Mojo 1.2 polyglot syntax highlighter (`flint.devtools.mojo_highlight`).

Provides a fast, zero-dependency, line-based incremental lexer with multi-line
state tracking for Mojo 1.2, Python, C/C++/CUDA, Rust, JavaScript/TypeScript,
TOML, JSON, Shell/YAML, and Markdown (fenced code blocks are highlighted in the
fence's language and keep that language's multi-line state, packed into the
Markdown state with `md_fence_state`). Produces coalesced `HighlightSpan` byte ranges suitable for GPU SDF
UI rendering (`UiDrawList`) and Tier-0 CPU SIMD rasterizer parity.
"""
from std.collections import Dict, List

comptime TOK_TEXT: Int = 0
comptime TOK_KEYWORD: Int = 1
comptime TOK_TYPE: Int = 2
comptime TOK_COMMENT: Int = 3
comptime TOK_STRING: Int = 4
comptime TOK_NUMBER: Int = 5
comptime TOK_DECORATOR: Int = 6
comptime TOK_SYMBOL: Int = 7
comptime TOK_BUILTIN: Int = 8
comptime TOK_SELF: Int = 9
comptime TOK_DEF: Int = 10
comptime TOK_PARAM: Int = 11
comptime TOK_PROP: Int = 12
comptime TOK_NAMESPACE: Int = 13
comptime TOK_COUNT: Int = 14

comptime LANG_MOJO: Int = 0
comptime LANG_PYTHON: Int = 1
comptime LANG_C_CUDA: Int = 2
comptime LANG_TOML: Int = 3
comptime LANG_JSON: Int = 4
comptime LANG_SHELL_YAML: Int = 5
comptime LANG_MARKDOWN: Int = 6
comptime LANG_RUST: Int = 7
comptime LANG_JS: Int = 8

comptime STATE_NORMAL: Int = 0
comptime STATE_TRIPLE_DOUBLE: Int = 1  # """..."""
comptime STATE_TRIPLE_SINGLE: Int = 2  # '''...'''
comptime STATE_BLOCK_COMMENT: Int = 3  # /* ... */
comptime STATE_MD_FENCE: Int = 4       # ``` ... ``` (low 3 bits; see `md_fence_state`)
comptime STATE_JS_TEMPLATE: Int = 5    # JS/TS `...` template literal (low 3 bits; see `js_template_frames`)
comptime _JS_MAX_FRAMES: Int = 7       # template / `${` interpolation nesting carried across lines
comptime _JS_MAX_BRACES: Int = 15      # `{` depth inside one interpolation (4 bits)
comptime _JS_CODE: Int = -1            # scanner marker: byte is JS code, classified afterwards


def md_fence_state(inner_lang: Int = LANG_MOJO, inner_state: Int = STATE_NORMAL) -> Int:
    """Markdown state inside a fenced block: `STATE_MD_FENCE` in the low 3 bits, the fence's
    language in bits 3-6 and that language's own multi-line state above. A plain fence with
    no info string is exactly `STATE_MD_FENCE`."""
    return STATE_MD_FENCE | (inner_lang << 3) | (inner_state << 7)


@always_inline
def md_in_fence(state: Int) -> Bool:
    return state > 0 and (state & 7) == STATE_MD_FENCE


@always_inline
def md_fence_lang(state: Int) -> Int:
    return (state >> 3) & 15


@always_inline
def md_fence_inner(state: Int) -> Int:
    return state >> 7


@always_inline
def js_in_template(state: Int) -> Bool:
    """True when a JS/TS line state continues inside a template literal (text or `${}`)."""
    return state > 0 and (state & 7) == STATE_JS_TEMPLATE


def js_template_frames(state: Int) -> List[Int]:
    """Unpacks a JS/TS template state into its nesting stack, outermost first: `-1` is a
    template-literal text frame, `d >= 0` a `${` interpolation with `d` open `{` braces.
    Frames alternate text / interpolation starting with text; packed as `STATE_JS_TEMPLATE`
    in bits 0-2, the frame count in bits 3-5 and 4 bits of brace depth per interpolation."""
    var frames = List[Int]()
    if not js_in_template(state):
        return frames^
    var count = (state >> 3) & 7
    for f in range(count):
        if f % 2 == 0:
            frames.append(-1)
        else:
            frames.append((state >> (6 + 4 * (f // 2))) & 15)
    return frames^


def _js_pack(frames: List[Int], block_comment: Bool) -> Int:
    if len(frames) == 0:
        return STATE_BLOCK_COMMENT if block_comment else STATE_NORMAL
    var st = STATE_JS_TEMPLATE | (len(frames) << 3)
    for f in range(1, len(frames), 2):
        st |= frames[f] << (6 + 4 * (f // 2))
    return st


def _js_scan(b: Span[UInt8, _], state: Int, mut cls: List[Int], mark: Bool) -> Int:
    """Structural JS/TS lexer shared by highlighting (`mark`) and `next_line_state`: comments,
    quoted strings and template literals, with `${ ... }` interpolations (balanced braces,
    nested templates) carried across lines. Marks code bytes `_JS_CODE` for word classing."""
    var n = len(b)
    var frames = js_template_frames(state)
    var in_block = state == STATE_BLOCK_COMMENT
    var i = 0
    while i < n:
        if in_block:
            var closed = -1
            var j = i
            while j + 1 < n:
                if b[j] == 42 and b[j + 1] == 47:
                    closed = j + 2
                    break
                j += 1
            var stop = closed if closed >= 0 else n
            if mark:
                for k in range(i, stop):
                    cls[k] = TOK_COMMENT
            if closed < 0:
                break
            in_block = False
            i = closed
            continue
        var c = b[i]
        var top = frames[len(frames) - 1] if len(frames) > 0 else -2
        if top == -1:
            # Template literal text.
            if c == 36 and i + 1 < n and b[i + 1] == 123:
                if mark:
                    cls[i] = TOK_SYMBOL
                    cls[i + 1] = TOK_SYMBOL
                frames.append(0)
                i += 2
                continue
            var w = 2 if (c == 92 and i + 1 < n) else 1
            if mark:
                for k in range(i, i + w):
                    cls[k] = TOK_STRING
            if c == 96:
                _ = frames.pop()
            i += w
            continue
        # Code: top level (`top == -2`) or inside a `${}` interpolation.
        if c == 47 and i + 1 < n and (b[i + 1] == 47 or b[i + 1] == 42):
            if b[i + 1] == 47:
                if mark:
                    for k in range(i, n):
                        cls[k] = TOK_COMMENT
                break
            if len(frames) == 0:
                if mark:
                    cls[i] = TOK_COMMENT
                    cls[i + 1] = TOK_COMMENT
                in_block = True
                i += 2
                continue
            # `/* */` inside an interpolation: closed on this line or runs to its end.
            var j = i + 2
            while j + 1 < n and not (b[j] == 42 and b[j + 1] == 47):
                j += 1
            var stop = min(j + 2, n)
            if mark:
                for k in range(i, stop):
                    cls[k] = TOK_COMMENT
            i = stop
            continue
        if c == 96 and len(frames) < _JS_MAX_FRAMES:
            if mark:
                cls[i] = TOK_STRING
            frames.append(-1)
            i += 1
            continue
        if c == 34 or c == 39 or c == 96:
            var j = i + 1
            while j < n:
                if b[j] == 92:
                    j += 2
                    continue
                if b[j] == c:
                    j += 1
                    break
                j += 1
            if mark:
                for k in range(i, min(j, n)):
                    cls[k] = TOK_STRING
            i = j
            continue
        if top >= 0 and c == 125:
            if top == 0:
                if mark:
                    cls[i] = TOK_SYMBOL
                _ = frames.pop()
                i += 1
                continue
            frames[len(frames) - 1] = top - 1
        elif top >= 0 and c == 123 and top < _JS_MAX_BRACES:
            frames[len(frames) - 1] = top + 1
        if mark:
            cls[i] = _JS_CODE
        i += 1
    return _js_pack(frames, in_block)


@fieldwise_init
struct SyntaxToken(Copyable, Movable, Equatable):
    """Single lexed syntax token span in UTF-8 byte offsets within a line."""
    var start_byte: Int
    var end_byte: Int
    var kind: Int

    def __eq__(self, other: Self) -> Bool:
        return (
            self.start_byte == other.start_byte
            and self.end_byte == other.end_byte
            and self.kind == other.kind
        )


@fieldwise_init
struct HighlightSpan(Copyable, Movable, Equatable):
    """Coalesced run of bytes sharing the same syntax token class."""
    var start_byte: Int
    var end_byte: Int
    var token_kind: Int

    def __eq__(self, other: Self) -> Bool:
        return (
            self.start_byte == other.start_byte
            and self.end_byte == other.end_byte
            and self.token_kind == other.token_kind
        )


@always_inline
def _is_digit(b: UInt8) -> Bool:
    return b >= 48 and b <= 57


@always_inline
def _is_alpha(b: UInt8) -> Bool:
    return (b >= 65 and b <= 90) or (b >= 97 and b <= 122) or b == 95 or b >= 128


@always_inline
def _is_ident(b: UInt8) -> Bool:
    return _is_alpha(b) or _is_digit(b)


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


def _basename(path: String) -> String:
    var b = path.as_bytes()
    var last = -1
    for i in range(len(b)):
        if b[i] == 47:
            last = i
    if last < 0:
        return path
    return String(path[byte=last + 1 : len(b)])


def lang_for_path(path: String, title: String = "") -> Int:
    """Infers highlighter language mode from file path (falling back to title)."""
    var target = path if path.byte_length() > 0 else title
    if target.byte_length() == 0:
        return LANG_MOJO
    var low = _lower_ascii(_basename(target))
    if low.endswith(".mojo") or target.endswith(".🔥"):
        return LANG_MOJO
    if low.endswith(".py") or low.endswith(".pyi") or low.endswith(".pyw"):
        return LANG_PYTHON
    for ext in [
        ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hxx",
        ".cu", ".cuh", ".go", ".java",
    ]:
        if low.endswith(String(ext)):
            return LANG_C_CUDA
    if low.endswith(".rs"):
        return LANG_RUST
    for ext in [".js", ".mjs", ".cjs", ".ts", ".mts", ".jsx", ".tsx"]:
        if low.endswith(String(ext)):
            return LANG_JS
    if low.endswith(".toml") or low.endswith(".ini") or low.endswith(".cfg") or low == ".editorconfig":
        return LANG_TOML
    if low.endswith(".json") or low.endswith(".jsonl") or low.endswith(".jsonc"):
        return LANG_JSON
    for ext in [".sh", ".bash", ".zsh", ".yaml", ".yml", ".env", ".conf"]:
        if low.endswith(String(ext)):
            return LANG_SHELL_YAML
    if low == "dockerfile" or low == "makefile" or low.startswith("dockerfile."):
        return LANG_SHELL_YAML
    if low.endswith(".md") or low.endswith(".mdx") or low.endswith(".markdown"):
        return LANG_MARKDOWN
    return LANG_MOJO


def _is_md_fence(b: Span[UInt8, _]) -> Bool:
    var n = len(b)
    var i = 0
    while i < n and (b[i] == 32 or b[i] == 9):
        i += 1
    return i + 2 < n and b[i] == 96 and b[i + 1] == 96 and b[i + 2] == 96


def _fence_info_lang(b: Span[UInt8, _]) -> Int:
    """Language of a fence's info string (```python, ```rust, ...); Mojo when absent/unknown."""
    var n = len(b)
    var i = 0
    while i < n and (b[i] == 32 or b[i] == 9):
        i += 1
    while i < n and b[i] == 96:
        i += 1
    while i < n and (b[i] == 32 or b[i] == 9):
        i += 1
    var tag = List[UInt8]()
    while i < n and b[i] != 32 and b[i] != 9 and b[i] != 123 and b[i] != 44:
        var c = b[i]
        tag.append(c + 32 if (c >= 65 and c <= 90) else c)
        i += 1
    var t = String(unsafe_from_utf8=tag)
    if t == "python" or t == "py" or t == "python3":
        return LANG_PYTHON
    if t == "rust" or t == "rs":
        return LANG_RUST
    if t == "js" or t == "javascript" or t == "ts" or t == "typescript" or t == "jsx" or t == "tsx":
        return LANG_JS
    if t == "c" or t == "cpp" or t == "c++" or t == "cuda" or t == "cu" or t == "h" or t == "go" or t == "java":
        return LANG_C_CUDA
    if t == "toml" or t == "ini":
        return LANG_TOML
    if t == "json" or t == "jsonc":
        return LANG_JSON
    if t == "sh" or t == "bash" or t == "zsh" or t == "shell" or t == "console" or t == "yaml" or t == "yml":
        return LANG_SHELL_YAML
    return LANG_MOJO


@always_inline
def _is_c_family(lang: Int) -> Bool:
    return lang == LANG_C_CUDA or lang == LANG_RUST or lang == LANG_JS


def _rust_char_end(b: Span[UInt8, _], i: Int) -> Int:
    """End of a Rust char literal starting at `'` (`'a'`, `'\\n'`, `'\\u{1F600}'`), or -1 when
    it is a lifetime / label (`'a`, `'static`)."""
    var n = len(b)
    if i + 1 >= n:
        return -1
    if b[i + 1] == 92:
        var j = i + 2
        while j < n and j < i + 14:
            if b[j] == 39:
                return j + 1
            j += 1
        return -1
    var j = i + 2
    while j < n and (b[j] & 0xC0) == 0x80:
        j += 1
    if j < n and b[j] == 39:
        return j + 1
    return -1


def _split_words(s: String) -> List[String]:
    var out = List[String]()
    for w in s.split(" "):
        var tok = String(w)
        if tok.byte_length() > 0:
            out.append(tok^)
    return out^


def _coalesce_spans(cls: List[Int]) -> List[HighlightSpan]:
    var out = List[HighlightSpan]()
    var n = len(cls)
    if n == 0:
        return out^
    var start = 0
    var cur = cls[0]
    for i in range(1, n):
        if cls[i] != cur:
            out.append(HighlightSpan(start, i, cur))
            start = i
            cur = cls[i]
    out.append(HighlightSpan(start, n, cur))
    return out^


def next_line_state(line: String, state: Int = STATE_NORMAL, lang: Int = LANG_MOJO) -> Int:
    """Computes multi-line lexer state after `line` without allocating token spans."""
    var b = line.as_bytes()
    var n = len(b)
    if lang == LANG_JSON or lang == LANG_SHELL_YAML:
        return STATE_NORMAL
    if lang == LANG_MARKDOWN:
        if _is_md_fence(b):
            return STATE_NORMAL if md_in_fence(state) else md_fence_state(_fence_info_lang(b))
        if md_in_fence(state):
            var inner_lang = md_fence_lang(state)
            return md_fence_state(inner_lang, next_line_state(line, md_fence_inner(state), inner_lang))
        return STATE_NORMAL
    if lang == LANG_JS:
        var no_cls = List[Int]()
        return _js_scan(b, state, no_cls, False)
    if _is_c_family(lang):
        var i = 0
        var st_c = STATE_BLOCK_COMMENT if state == STATE_BLOCK_COMMENT else STATE_NORMAL
        while i < n:
            if st_c == STATE_BLOCK_COMMENT:
                if i + 1 < n and b[i] == 42 and b[i + 1] == 47:
                    st_c = STATE_NORMAL
                    i += 2
                    continue
                i += 1
                continue
            var c = b[i]
            if c == 47 and i + 1 < n:
                if b[i + 1] == 47:
                    return STATE_NORMAL
                if b[i + 1] == 42:
                    st_c = STATE_BLOCK_COMMENT
                    i += 2
                    continue
            if c == 39 and lang == LANG_RUST:
                var ce = _rust_char_end(b, i)
                i = ce if ce > 0 else i + 1
                continue
            if c == 34 or c == 39:
                var j = i + 1
                while j < n:
                    if b[j] == 92:
                        j += 2
                        continue
                    if b[j] == c:
                        j += 1
                        break
                    j += 1
                i = j
                continue
            i += 1
        return st_c

    # Mojo, Python, TOML triple-quoted string state tracking
    var i = 0
    var st = state if (state == STATE_TRIPLE_DOUBLE or state == STATE_TRIPLE_SINGLE) else STATE_NORMAL
    if st != STATE_NORMAL:
        var q: UInt8 = 34 if st == STATE_TRIPLE_DOUBLE else 39
        var found = -1
        var j = 0
        while j + 2 < n:
            if b[j] == 92 and (lang != LANG_TOML or st == STATE_TRIPLE_DOUBLE):
                j += 2
                continue
            if b[j] == q and b[j + 1] == q and b[j + 2] == q:
                found = j
                break
            j += 1
        if found < 0:
            return st
        i = found + 3
    while i < n:
        var c = b[i]
        if c == 35:
            return STATE_NORMAL
        if c == 34 or c == 39:
            if i + 2 < n and b[i + 1] == c and b[i + 2] == c:
                var j = i + 3
                var found = -1
                while j + 2 < n:
                    if b[j] == 92 and (lang != LANG_TOML or c == 34):
                        j += 2
                        continue
                    if b[j] == c and b[j + 1] == c and b[j + 2] == c:
                        found = j
                        break
                    j += 1
                if found < 0:
                    return STATE_TRIPLE_DOUBLE if c == 34 else STATE_TRIPLE_SINGLE
                i = found + 3
                continue
            var j = i + 1
            while j < n:
                if b[j] == 92 and (lang != LANG_TOML or c == 34):
                    j += 2
                    continue
                if b[j] == c:
                    j += 1
                    break
                j += 1
            i = j
            continue
        i += 1
    return STATE_NORMAL


struct MojoHighlighter(Movable):
    """Reusable polyglot syntax highlighter with prebuilt keyword/type lookup tables."""
    var mojo_map: Dict[String, Int]
    var py_map: Dict[String, Int]
    var c_map: Dict[String, Int]
    var sh_map: Dict[String, Int]
    var rust_map: Dict[String, Int]
    var js_map: Dict[String, Int]

    def __init__(out self):
        self.mojo_map = Dict[String, Int]()
        for w in _split_words(
            "print len range min max abs round String Int str ord chr isinstance input open rebind"
            " constrained debug_assert external_call parallelize vectorize size_of align_of bitcast"
            " atol atof hash iter next enumerate zip reversed sorted sum any all swap"
        ):
            self.mojo_map[w] = TOK_BUILTIN
        for w in _split_words(
            "Int Int8 Int16 Int32 Int64 Int128 Int256 UInt UInt8 UInt16 UInt32 UInt64 UInt128 UInt256"
            " Float16 Float32 Float64 BFloat16 Bool String StringLiteral StringSlice List Dict Set"
            " Optional Variant Tuple SIMD DType Scalar Pointer UnsafePointer OpaquePointer Span Error"
            " Copyable Movable ImplicitlyCopyable Writable Stringable Representable Hashable"
            " Equatable Comparable Sized Boolable Intable AnyType NoneType Self Codepoint InlineArray"
            " Origin MutOrigin ImmutOrigin Iterator ArcPointer OwnedPointer DeviceContext"
            " TrivialRegisterPassable RegisterPassable MutAnyOrigin ImmutAnyOrigin Some Array"
        ):
            self.mojo_map[w] = TOK_TYPE
        for w in _split_words(
            "def struct trait var comptime if elif else for while in return raises raise"
            " try except finally with as import from pass break continue and or not is True False None"
            " out mut owned read ref deinit lambda yield async await del global nonlocal"
            " assert class match case where capturing escaping unsafe __extension abi thin"
        ):
            self.mojo_map[w] = TOK_KEYWORD
        self.mojo_map["self"] = TOK_SELF

        self.py_map = Dict[String, Int]()
        for w in _split_words(
            "print len range min max abs round str int float bool bytes list dict set tuple ord chr"
            " isinstance issubclass getattr setattr hasattr open iter next enumerate zip reversed"
            " sorted sum any all super repr id type callable vars dir"
        ):
            self.py_map[w] = TOK_BUILTIN
        for w in _split_words(
            "Any Optional Union Callable Sequence Mapping Iterable Iterator Tuple List Dict Set Type"
            " Final ClassVar Protocol TypedDict Path Exception ValueError TypeError RuntimeError"
            " KeyError IndexError OSError"
        ):
            self.py_map[w] = TOK_TYPE
        for w in _split_words(
            "def class if elif else for while in return raise try except finally with as import from"
            " pass break continue and or not is True False None lambda yield async await del global"
            " nonlocal assert match case"
        ):
            self.py_map[w] = TOK_KEYWORD
        self.py_map["self"] = TOK_SELF
        self.py_map["cls"] = TOK_SELF

        self.c_map = Dict[String, Int]()
        for w in _split_words(
            "printf fprintf snprintf malloc calloc realloc free memcpy memset memcmp strlen strcmp"
            " sizeof alignof static_assert"
        ):
            self.c_map[w] = TOK_BUILTIN
        for w in _split_words(
            "void bool char short int long float double signed unsigned size_t ssize_t ptrdiff_t"
            " int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t uintptr_t FILE"
            " vec2 vec3 vec4 mat3 mat4 u32 i32 f32 f64 usize isize"
        ):
            self.c_map[w] = TOK_TYPE
        for w in _split_words(
            "if else for while do switch case default break continue return goto struct class enum"
            " union typedef namespace template typename using const constexpr static extern inline"
            " virtual override final public private protected new delete try catch throw noexcept"
            " auto volatile restrict true false nullptr NULL fn let mut pub impl trait match mod"
            " use where async await func var export import from function"
        ):
            self.c_map[w] = TOK_KEYWORD
        for w in _split_words("this self Self"):
            self.c_map[w] = TOK_SELF

        self.rust_map = Dict[String, Int]()
        for w in _split_words(
            "i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64 bool char str String Vec"
            " Option Result Box Rc Arc RefCell Cell HashMap HashSet BTreeMap Some None Ok Err Self"
        ):
            self.rust_map[w] = TOK_TYPE
        for w in _split_words(
            "as async await break const continue crate dyn else enum extern false fn for if impl in"
            " let loop match mod move mut pub ref return static struct super trait true type unsafe"
            " use where while union macro_rules"
        ):
            self.rust_map[w] = TOK_KEYWORD
        self.rust_map["self"] = TOK_SELF
        self.rust_map["Self"] = TOK_SELF

        self.js_map = Dict[String, Int]()
        for w in _split_words(
            "console Math JSON Object Array Promise Map Set Symbol Number String Boolean Date RegExp"
            " Error window document globalThis require module exports"
        ):
            self.js_map[w] = TOK_BUILTIN
        for w in _split_words(
            "string number boolean any unknown never void object bigint undefined"
        ):
            self.js_map[w] = TOK_TYPE
        for w in _split_words(
            "break case catch class const continue debugger default delete do else export extends"
            " finally for function if import in instanceof let new return super switch throw try"
            " typeof var while with yield async await of static get set true false null from as"
            " interface type enum implements private public protected readonly declare namespace"
            " abstract keyof satisfies"
        ):
            self.js_map[w] = TOK_KEYWORD
        self.js_map["this"] = TOK_SELF

        self.sh_map = Dict[String, Int]()
        for w in _split_words("echo printf cd pwd ls rm cp mv mkdir cat grep sed awk find xargs test"):
            self.sh_map[w] = TOK_BUILTIN
        for w in _split_words(
            "if then else elif fi for in do done while until case esac function return exit"
            " export local readonly declare set unset shift trap source true false null yes no"
        ):
            self.sh_map[w] = TOK_KEYWORD

    def classify_word(self, word: String, lang: Int = LANG_MOJO) -> Int:
        if lang == LANG_PYTHON:
            return self.py_map.get(word, TOK_TEXT)
        if lang == LANG_C_CUDA:
            return self.c_map.get(word, TOK_TEXT)
        if lang == LANG_RUST:
            return self.rust_map.get(word, TOK_TEXT)
        if lang == LANG_JS:
            return self.js_map.get(word, TOK_TEXT)
        if lang == LANG_SHELL_YAML:
            return self.sh_map.get(word, TOK_TEXT)
        return self.mojo_map.get(word, TOK_TEXT)

    def _lex_c_cuda(self, line: String, state: Int, mut cls: List[Int], lang: Int = LANG_C_CUDA) -> Int:
        """C/C++/CUDA lexer; `lang` selects the Rust (`#[attr]`, lifetimes, `name!` macros)
        dialect. JavaScript/TypeScript is delegated to `_lex_js`."""
        if lang == LANG_JS:
            return self._lex_js(line, state, cls)
        var b = line.as_bytes()
        var n = len(b)
        var i = 0
        var st = STATE_BLOCK_COMMENT if state == STATE_BLOCK_COMMENT else STATE_NORMAL
        if st == STATE_NORMAL and lang == LANG_RUST:
            var p = 0
            while p < n and (b[p] == 32 or b[p] == 9):
                p += 1
            if p + 1 < n and b[p] == 35 and (b[p + 1] == 91 or b[p + 1] == 33):
                var depth = 0
                var q = p
                while q < n:
                    if b[q] == 91:
                        depth += 1
                    elif b[q] == 93:
                        depth -= 1
                        if depth == 0:
                            q += 1
                            break
                    q += 1
                for k in range(p, min(q, n)):
                    cls[k] = TOK_DECORATOR
                i = min(q, n)
        elif st == STATE_NORMAL and lang == LANG_C_CUDA:
            var p = 0
            while p < n and (b[p] == 32 or b[p] == 9):
                p += 1
            if p < n and b[p] == 35:
                var q = p + 1
                while q < n and (b[q] == 32 or b[q] == 9):
                    q += 1
                while q < n and _is_ident(b[q]):
                    q += 1
                for k in range(p, q):
                    cls[k] = TOK_DECORATOR
                i = q
        var prev_word = String("")
        while i < n:
            if st == STATE_BLOCK_COMMENT:
                var j = i
                var closed = -1
                while j + 1 < n:
                    if b[j] == 42 and b[j + 1] == 47:
                        closed = j + 2
                        break
                    j += 1
                if closed < 0:
                    for k in range(i, n):
                        cls[k] = TOK_COMMENT
                    return STATE_BLOCK_COMMENT
                for k in range(i, closed):
                    cls[k] = TOK_COMMENT
                st = STATE_NORMAL
                i = closed
                continue
            var c = b[i]
            if c == 47 and i + 1 < n:
                if b[i + 1] == 47:
                    for k in range(i, n):
                        cls[k] = TOK_COMMENT
                    return STATE_NORMAL
                if b[i + 1] == 42:
                    cls[i] = TOK_COMMENT
                    cls[i + 1] = TOK_COMMENT
                    st = STATE_BLOCK_COMMENT
                    i += 2
                    continue
            if c == 39 and lang == LANG_RUST and _rust_char_end(b, i) < 0:
                # Lifetime or loop label: `'a`, `'static`.
                var j = i + 1
                while j < n and _is_ident(b[j]):
                    j += 1
                for k in range(i, j):
                    cls[k] = TOK_TYPE
                i = j
                continue
            if c == 34 or c == 39:
                var j = i + 1
                while j < n:
                    if b[j] == 92:
                        j += 2
                        continue
                    if b[j] == c:
                        j += 1
                        break
                    j += 1
                for k in range(i, min(j, n)):
                    cls[k] = TOK_STRING
                i = j
                continue
            if _is_digit(c) and (i == 0 or not _is_ident(b[i - 1])):
                var j = i
                while j < n and (_is_ident(b[j]) or b[j] == 46):
                    j += 1
                for k in range(i, j):
                    cls[k] = TOK_NUMBER
                i = j
                continue
            if _is_alpha(c):
                var j = i + 1
                while j < n and _is_ident(b[j]):
                    j += 1
                var w = String(line[byte=i:j])
                var kind = self.classify_word(w, lang)
                if lang == LANG_RUST and kind == TOK_TEXT and j < n and b[j] == 33:
                    kind = TOK_BUILTIN  # macro invocation `println!`
                    j += 1
                if kind == TOK_TEXT and (
                    prev_word == "struct"
                    or prev_word == "class"
                    or prev_word == "namespace"
                    or prev_word == "fn"
                    or prev_word == "func"
                    or prev_word == "function"
                    or prev_word == "enum"
                    or prev_word == "trait"
                    or prev_word == "interface"
                ):
                    kind = TOK_DEF
                elif kind == TOK_TEXT and j < n and b[j] == 40:
                    kind = TOK_DEF
                for k in range(i, j):
                    cls[k] = kind
                prev_word = w
                i = j
                continue
            if c != 32 and c != 9:
                cls[i] = TOK_SYMBOL
                prev_word = ""
            i += 1
        return st

    def _lex_js(self, line: String, state: Int, mut cls: List[Int]) -> Int:
        """JavaScript/TypeScript: `_js_scan` structure (multi-line template literals and their
        `${}` interpolations), then keyword / number / call classing of the code bytes."""
        var b = line.as_bytes()
        var n = len(b)
        var next = _js_scan(b, state, cls, True)
        var prev_word = String("")
        var i = 0
        while i < n:
            if cls[i] != _JS_CODE:
                i += 1
                continue
            var c = b[i]
            if _is_digit(c) and (i == 0 or not _is_ident(b[i - 1])):
                var j = i
                while j < n and cls[j] == _JS_CODE and (_is_ident(b[j]) or b[j] == 46):
                    j += 1
                for k in range(i, j):
                    cls[k] = TOK_NUMBER
                i = j
                continue
            if _is_alpha(c) or c == 36:
                var j = i + 1
                while j < n and cls[j] == _JS_CODE and (_is_ident(b[j]) or b[j] == 36):
                    j += 1
                var w = String(line[byte=i:j])
                var kind = self.classify_word(w, LANG_JS)
                if kind == TOK_TEXT and (
                    prev_word == "class"
                    or prev_word == "function"
                    or prev_word == "enum"
                    or prev_word == "interface"
                    or prev_word == "namespace"
                ):
                    kind = TOK_DEF
                elif kind == TOK_TEXT and j < n and b[j] == 40:
                    kind = TOK_DEF
                for k in range(i, j):
                    cls[k] = kind
                prev_word = w
                i = j
                continue
            if c == 32 or c == 9:
                cls[i] = TOK_TEXT
            else:
                cls[i] = TOK_SYMBOL
                prev_word = ""
            i += 1
        return next

    def _lex_toml(self, line: String, state: Int, mut cls: List[Int]) -> Int:
        var b = line.as_bytes()
        var n = len(b)
        var i = 0
        var st = state if (state == STATE_TRIPLE_DOUBLE or state == STATE_TRIPLE_SINGLE) else STATE_NORMAL
        if st != STATE_NORMAL:
            var q: UInt8 = 34 if st == STATE_TRIPLE_DOUBLE else 39
            var j = 0
            var found = -1
            while j + 2 < n:
                if b[j] == 92 and st == STATE_TRIPLE_DOUBLE:
                    j += 2
                    continue
                if b[j] == q and b[j + 1] == q and b[j + 2] == q:
                    found = j
                    break
                j += 1
            if found < 0:
                for k in range(n):
                    cls[k] = TOK_STRING
                return st
            for k in range(found + 3):
                cls[k] = TOK_STRING
            i = found + 3
        if i == 0:
            var p = 0
            while p < n and (b[p] == 32 or b[p] == 9):
                p += 1
            if p < n and b[p] == 91:
                var rb = -1
                for k in range(p, n):
                    if b[k] == 35:
                        break
                    if b[k] == 93:
                        rb = k
                if rb > p:
                    for k in range(p, rb + 1):
                        cls[k] = TOK_DECORATOR
                    i = rb + 1
            elif p < n and b[p] != 35:
                var eq = -1
                for k in range(p, n):
                    if b[k] == 35:
                        break
                    if b[k] == 61:
                        eq = k
                        break
                if eq > p:
                    for k in range(p, eq):
                        if b[k] != 32 and b[k] != 9:
                            cls[k] = TOK_DEF
                    cls[eq] = TOK_SYMBOL
                    i = eq + 1
        while i < n:
            var c = b[i]
            if c == 35:
                for k in range(i, n):
                    cls[k] = TOK_COMMENT
                return STATE_NORMAL
            if c == 34 or c == 39:
                if i + 2 < n and b[i + 1] == c and b[i + 2] == c:
                    var j = i + 3
                    var found = -1
                    while j + 2 < n:
                        if b[j] == 92 and c == 34:
                            j += 2
                            continue
                        if b[j] == c and b[j + 1] == c and b[j + 2] == c:
                            found = j
                            break
                        j += 1
                    if found < 0:
                        for k in range(i, n):
                            cls[k] = TOK_STRING
                        return STATE_TRIPLE_DOUBLE if c == 34 else STATE_TRIPLE_SINGLE
                    for k in range(i, found + 3):
                        cls[k] = TOK_STRING
                    i = found + 3
                    continue
                var j = i + 1
                while j < n:
                    if b[j] == 92 and c == 34:
                        j += 2
                        continue
                    if b[j] == c:
                        j += 1
                        break
                    j += 1
                for k in range(i, min(j, n)):
                    cls[k] = TOK_STRING
                i = j
                continue
            if _is_digit(c) or (c == 45 and i + 1 < n and _is_digit(b[i + 1])):
                var j = i + 1
                while j < n and (_is_ident(b[j]) or b[j] == 46 or b[j] == 45 or b[j] == 58 or b[j] == 43):
                    j += 1
                for k in range(i, j):
                    cls[k] = TOK_NUMBER
                i = j
                continue
            if _is_alpha(c):
                var j = i
                while j < n and _is_ident(b[j]):
                    j += 1
                var w = String(line[byte=i:j])
                var kind = TOK_KEYWORD if (w == "true" or w == "false") else TOK_TEXT
                for k in range(i, j):
                    cls[k] = kind
                i = j
                continue
            if c != 32 and c != 9:
                cls[i] = TOK_SYMBOL
            i += 1
        return STATE_NORMAL

    def _lex_json(self, line: String, mut cls: List[Int]) -> Int:
        var b = line.as_bytes()
        var n = len(b)
        var i = 0
        while i < n:
            var c = b[i]
            if c == 47 and i + 1 < n and b[i + 1] == 47:
                for k in range(i, n):
                    cls[k] = TOK_COMMENT
                break
            if c == 34:
                var j = i + 1
                while j < n:
                    if b[j] == 92:
                        j += 2
                        continue
                    if b[j] == 34:
                        j += 1
                        break
                    j += 1
                var p = j
                while p < n and (b[p] == 32 or b[p] == 9):
                    p += 1
                var kind = TOK_DEF if (p < n and b[p] == 58) else TOK_STRING
                for k in range(i, min(j, n)):
                    cls[k] = kind
                i = j
                continue
            if _is_digit(c) or (c == 45 and i + 1 < n and _is_digit(b[i + 1])):
                var j = i + 1
                while j < n and (
                    _is_digit(b[j]) or b[j] == 46 or b[j] == 101 or b[j] == 69 or b[j] == 43 or b[j] == 45
                ):
                    j += 1
                for k in range(i, j):
                    cls[k] = TOK_NUMBER
                i = j
                continue
            if _is_alpha(c):
                var j = i
                while j < n and _is_ident(b[j]):
                    j += 1
                var w = String(line[byte=i:j])
                var kind = TOK_KEYWORD if (w == "true" or w == "false" or w == "null") else TOK_TEXT
                for k in range(i, j):
                    cls[k] = kind
                i = j
                continue
            if c != 32 and c != 9:
                cls[i] = TOK_SYMBOL
            i += 1
        return STATE_NORMAL

    def _lex_shell_yaml(self, line: String, mut cls: List[Int]) -> Int:
        var b = line.as_bytes()
        var n = len(b)
        var i = 0
        var p = 0
        while p < n and (b[p] == 32 or b[p] == 9 or b[p] == 45):
            if b[p] == 45:
                cls[p] = TOK_SYMBOL
            p += 1
        if p < n and (_is_alpha(b[p]) or b[p] == 46):
            var q = p
            while q < n and (_is_ident(b[q]) or b[q] == 45 or b[q] == 46):
                q += 1
            if q < n and b[q] == 58 and (q + 1 >= n or b[q + 1] == 32 or b[q + 1] == 9):
                for k in range(p, q):
                    cls[k] = TOK_DEF
                cls[q] = TOK_SYMBOL
                i = q + 1
        while i < n:
            var c = b[i]
            if c == 35 and (i == 0 or b[i - 1] == 32 or b[i - 1] == 9):
                for k in range(i, n):
                    cls[k] = TOK_COMMENT
                break
            if c == 34 or c == 39:
                var j = i + 1
                while j < n:
                    if b[j] == 92 and c == 34:
                        j += 2
                        continue
                    if b[j] == c:
                        j += 1
                        break
                    j += 1
                for k in range(i, min(j, n)):
                    cls[k] = TOK_STRING
                i = j
                continue
            if c == 36 and i + 1 < n:
                var j = i + 1
                if b[j] == 123:
                    while j < n and b[j] != 125:
                        j += 1
                    if j < n:
                        j += 1
                else:
                    while j < n and (_is_ident(b[j]) or b[j] == 63 or b[j] == 64 or b[j] == 35):
                        j += 1
                for k in range(i, min(j, n)):
                    cls[k] = TOK_SELF
                i = j
                continue
            if _is_digit(c) and (i == 0 or not _is_ident(b[i - 1])):
                var j = i
                while j < n and (_is_digit(b[j]) or b[j] == 46):
                    j += 1
                for k in range(i, j):
                    cls[k] = TOK_NUMBER
                i = j
                continue
            if _is_alpha(c):
                var j = i
                while j < n and (_is_ident(b[j]) or b[j] == 45):
                    j += 1
                var w = String(line[byte=i:j])
                var kind = self.classify_word(w, LANG_SHELL_YAML)
                for k in range(i, j):
                    cls[k] = kind
                i = j
                continue
            if c != 32 and c != 9:
                cls[i] = TOK_SYMBOL
            i += 1
        return STATE_NORMAL

    def _lex_markdown(self, line: String, state: Int, mut cls: List[Int]) -> Int:
        var b = line.as_bytes()
        var n = len(b)
        if _is_md_fence(b):
            for k in range(n):
                cls[k] = TOK_DECORATOR
            return STATE_NORMAL if md_in_fence(state) else md_fence_state(_fence_info_lang(b))
        if md_in_fence(state):
            # Fenced code: the fence's language, carrying its own multi-line state.
            var inner_lang = md_fence_lang(state)
            var inner_next = self.lex_bytes(line, md_fence_inner(state), cls, inner_lang)
            return md_fence_state(inner_lang, inner_next)
        var p = 0
        while p < n and (b[p] == 32 or b[p] == 9):
            p += 1
        if p < n and b[p] == 35:
            for k in range(p, n):
                cls[k] = TOK_DEF
            return STATE_NORMAL
        if p < n and b[p] == 62:
            for k in range(p, n):
                cls[k] = TOK_COMMENT
            return STATE_NORMAL
        if p + 1 < n and (b[p] == 45 or b[p] == 42 or b[p] == 43) and b[p + 1] == 32:
            cls[p] = TOK_KEYWORD
        var i = 0
        while i < n:
            var c = b[i]
            if c == 96:
                var j = i + 1
                while j < n and b[j] != 96:
                    j += 1
                if j < n:
                    j += 1
                for k in range(i, min(j, n)):
                    cls[k] = TOK_STRING
                i = j
                continue
            if c == 42 and i + 1 < n and b[i + 1] == 42:
                var end_b = -1
                for k in range(i + 2, n - 1):
                    if b[k] == 42 and b[k + 1] == 42:
                        end_b = k + 2
                        break
                if end_b > i:
                    for k in range(i, end_b):
                        cls[k] = TOK_KEYWORD
                    i = end_b
                    continue
            i += 1
        return STATE_NORMAL

    def lex_bytes(self, line: String, state: Int, mut cls: List[Int], lang: Int = LANG_MOJO) -> Int:
        """Populates `cls` with one `TOK_*` class per UTF-8 byte of `line`. Returns next state."""
        var b = line.as_bytes()
        var n = len(b)
        cls.clear()
        cls.resize(n, TOK_TEXT)
        if _is_c_family(lang):
            return self._lex_c_cuda(line, state, cls, lang)
        if lang == LANG_TOML:
            return self._lex_toml(line, state, cls)
        if lang == LANG_JSON:
            return self._lex_json(line, cls)
        if lang == LANG_SHELL_YAML:
            return self._lex_shell_yaml(line, cls)
        if lang == LANG_MARKDOWN:
            return self._lex_markdown(line, state, cls)

        var i = 0
        var st = state if (state == STATE_TRIPLE_DOUBLE or state == STATE_TRIPLE_SINGLE) else STATE_NORMAL
        if st != STATE_NORMAL:
            var q: UInt8 = 34 if st == STATE_TRIPLE_DOUBLE else 39
            var j = 0
            var found = -1
            while j + 2 < n:
                if b[j] == 92:
                    j += 2
                    continue
                if b[j] == q and b[j + 1] == q and b[j + 2] == q:
                    found = j
                    break
                j += 1
            if found < 0:
                for k in range(n):
                    cls[k] = TOK_STRING
                return st
            for k in range(found + 3):
                cls[k] = TOK_STRING
            i = found + 3
            st = STATE_NORMAL

        var prev_word = String("")
        while i < n:
            var c = b[i]
            if c == 35:
                for k in range(i, n):
                    cls[k] = TOK_COMMENT
                break
            if c == 34 or c == 39:
                if i + 2 < n and b[i + 1] == c and b[i + 2] == c:
                    var j = i + 3
                    var found = -1
                    while j + 2 < n:
                        if b[j] == 92:
                            j += 2
                            continue
                        if b[j] == c and b[j + 1] == c and b[j + 2] == c:
                            found = j
                            break
                        j += 1
                    if found < 0:
                        for k in range(i, n):
                            cls[k] = TOK_STRING
                        return STATE_TRIPLE_DOUBLE if c == 34 else STATE_TRIPLE_SINGLE
                    for k in range(i, found + 3):
                        cls[k] = TOK_STRING
                    i = found + 3
                    continue
                var j = i + 1
                while j < n:
                    if b[j] == 92:
                        j += 2
                        continue
                    if b[j] == c:
                        j += 1
                        break
                    j += 1
                for k in range(i, min(j, n)):
                    cls[k] = TOK_STRING
                i = j
                continue
            if _is_digit(c) and (i == 0 or not _is_ident(b[i - 1])):
                var j = i
                while j < n and (_is_ident(b[j]) or b[j] == 46):
                    j += 1
                for k in range(i, j):
                    cls[k] = TOK_NUMBER
                i = j
                continue
            if c == 64:
                var j = i + 1
                while j < n and (_is_ident(b[j]) or b[j] == 46):
                    j += 1
                for k in range(i, j):
                    cls[k] = TOK_DECORATOR
                i = j
                continue
            if _is_alpha(c):
                var j = i
                while j < n and _is_ident(b[j]):
                    j += 1
                var w = String(line[byte=i:j])
                var kind = self.classify_word(w, lang)
                if kind == TOK_TEXT and (
                    prev_word == "def"
                    or prev_word == "class"
                    or (
                        lang == LANG_MOJO
                        and (
                            prev_word == "struct"
                            or prev_word == "trait"
                            or prev_word == "comptime"
                            or prev_word == "__extension"
                        )
                    )
                ):
                    kind = TOK_DEF
                if j < n and (b[j] == 34 or b[j] == 39) and j - i == 1:
                    kind = TOK_STRING
                for k in range(i, j):
                    cls[k] = kind
                prev_word = w
                i = j
                continue
            if c != 32 and c != 9:
                cls[i] = TOK_SYMBOL
                prev_word = ""
            i += 1
        return st

    def highlight_line(
        self, line: String, state: Int = STATE_NORMAL, lang: Int = LANG_MOJO
    ) -> Tuple[List[HighlightSpan], Int]:
        var cls = List[Int]()
        var nxt = self.lex_bytes(line, state, cls, lang)
        return (_coalesce_spans(cls), nxt)

    def highlight_lines(
        self, lines: List[String], lang: Int = LANG_MOJO
    ) -> List[List[HighlightSpan]]:
        var out = List[List[HighlightSpan]](capacity=len(lines))
        var st = STATE_NORMAL
        var cls = List[Int]()
        for i in range(len(lines)):
            st = self.lex_bytes(lines[i], st, cls, lang)
            out.append(_coalesce_spans(cls))
        return out^


def highlight_mojo_line(
    line: String, state: Int = STATE_NORMAL, lang: Int = LANG_MOJO
) -> Tuple[List[HighlightSpan], Int]:
    """Convenience one-shot line highlighter (prefer `MojoHighlighter` in render loops)."""
    var hl = MojoHighlighter()
    return hl.highlight_line(line, state, lang)


def highlight_mojo_lines(
    lines: List[String], lang: Int = LANG_MOJO
) -> List[List[HighlightSpan]]:
    """Highlights a list of lines while tracking multi-line string/comment states."""
    var hl = MojoHighlighter()
    return hl.highlight_lines(lines, lang)
