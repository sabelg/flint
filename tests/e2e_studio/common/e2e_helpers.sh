#!/usr/bin/env bash
# Shared shell helpers for /home/user/flint/tests/e2e_studio/

if [[ -z "${ROOT:-}" ]]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
export ROOT

PASS=0
FAIL=0
TOTAL=0
RESOLVED_BIN=""
MOJO_BIN=""

resolve_mojo_env() {
    if [[ -n "${MOJO:-}" && -x "${MOJO}" ]]; then
        MOJO_BIN="${MOJO}"
        local maybe_prefix
        maybe_prefix="$(cd "$(dirname "${MOJO_BIN}")/.." && pwd)"
        if [[ -d "${maybe_prefix}/lib" ]]; then
            export MOJO_PREFIX="${MOJO_PREFIX:-$maybe_prefix}"
            if [[ -d "${maybe_prefix}/share/max" ]]; then
                export MODULAR_HOME="${MODULAR_HOME:-${maybe_prefix}/share/max}"
            fi
            case ":${LD_LIBRARY_PATH:-}:" in
                *":${maybe_prefix}/lib:"*) ;;
                *) export LD_LIBRARY_PATH="${maybe_prefix}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
            esac
        fi
        export MOJO="$MOJO_BIN"
        export MOJO_BIN
        return 0
    fi

    local candidate
    for candidate in \
        "${ROOT}/.pixi/envs/default" \
        "${ROOT}/../.pixi/envs/default" \
        "${HOME}/.pixi/envs/turbomojo"; do
        if [[ -x "${candidate}/bin/mojo" ]]; then
            export MOJO_PREFIX="$candidate"
            export MODULAR_HOME="${candidate}/share/max"
            export LD_LIBRARY_PATH="${candidate}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            MOJO_BIN="${candidate}/bin/mojo"
            export MOJO="$MOJO_BIN"
            export MOJO_BIN
            return 0
        fi
    done

    if command -v mojo >/dev/null 2>&1; then
        MOJO_BIN="$(command -v mojo)"
        export MOJO="$MOJO_BIN"
        export MOJO_BIN
        return 0
    fi

    echo "[e2e_studio] ERROR: Mojo 1.2 compiler not found (checked MOJO, .pixi/envs/default, turbomojo, PATH)." >&2
    return 1
}

_is_bin_fresh() {
    local bin_path="$1"
    local entry_src="$2"
    [[ -x "$bin_path" ]] || return 1
    [[ ! "$entry_src" -nt "$bin_path" ]] || return 1
    local newer_src
    newer_src="$(find "${ROOT}/src/flint" -type f -name '*.mojo' ! -name 'build_stamp.mojo' -newer "$bin_path" -print -quit 2>/dev/null || true)"
    [[ -z "$newer_src" ]]
}

resolve_bin() {
    local bin_name="$1"
    local example_subdir="$2"
    RESOLVED_BIN=""

    local entry_src="${ROOT}/examples/${example_subdir}/main.mojo"
    local prebuilt="${ROOT}/build/${bin_name}"
    if [[ ! -f "$entry_src" ]]; then
        echo "[e2e_studio] ERROR: Missing entrypoint $entry_src" >&2
        return 1
    fi

    if _is_bin_fresh "$prebuilt" "$entry_src"; then
        RESOLVED_BIN="$prebuilt"
        return 0
    fi

    local cache_base="${FLINT_STUDIO_E2E_TMP:-${TMPD:-/tmp/flint_e2e_studio_bins}}"
    local cache_bin_dir="${cache_base}/bin"
    mkdir -p "$cache_bin_dir"
    local cached="${cache_bin_dir}/${bin_name}"
    local fail_marker="${cached}.failed"
    local use_fail_marker=0
    if [[ -n "${TMPD:-}" || -n "${FLINT_STUDIO_E2E_TMP:-}" ]]; then
        use_fail_marker=1
    fi

    if [[ "$use_fail_marker" -eq 1 && -f "$fail_marker" ]]; then
        return 1
    fi

    if _is_bin_fresh "$cached" "$entry_src"; then
        RESOLVED_BIN="$cached"
        return 0
    fi

    resolve_mojo_env || return 1

    local tmp_out="${cached}.tmp.$$"
    local compile_log="${cache_bin_dir}/${bin_name}.compile.log"
    if ! "$MOJO_BIN" build -I "${ROOT}/src" -I "${ROOT}/tests" -O3 --fp-mode contract=off --Werror "$entry_src" -o "$tmp_out" >"$compile_log" 2>&1; then
        rm -f "$tmp_out"
        if [[ "$use_fail_marker" -eq 1 ]]; then
            touch "$fail_marker"
        fi
        echo "[e2e_studio] ERROR: Failed to compile $entry_src:" >&2
        tail -n 20 "$compile_log" >&2 || true
        return 1
    fi
    mv -f "$tmp_out" "$cached"
    RESOLVED_BIN="$cached"
    return 0
}

make_isolated_tmp() {
    local prefix="${1:-flint_studio_e2e}"
    TMPD="$(mktemp -d "${TMPDIR:-/tmp}/${prefix}_XXXXXX")"
    mkdir -p "${TMPD}/bin"
    export TMPD
    PASS=0
    FAIL=0
    TOTAL=0

    _e2e_cleanup_tmp() {
        local rc=$?
        local tmp_root="${TMPDIR:-/tmp}"
        tmp_root="${tmp_root%/}"
        if [[ "${FLINT_STUDIO_E2E_KEEP_TMP:-0}" != "1" && -n "${TMPD:-}" && "$TMPD" != "/" && ( "$TMPD" == /tmp/* || "$TMPD" == "${tmp_root}/"* ) ]]; then
            rm -rf "$TMPD"
        fi
        return "$rc"
    }
    trap _e2e_cleanup_tmp EXIT
}

pass_test() {
    local name="$1"
    PASS=$((PASS + 1))
    TOTAL=$((TOTAL + 1))
    echo "[PASS] ${name}"
}

fail_test() {
    local name="$1"
    local reason="${2:-}"
    FAIL=$((FAIL + 1))
    TOTAL=$((TOTAL + 1))
    if [[ -n "$reason" ]]; then
        echo "[FAIL] ${name}: ${reason}"
    else
        echo "[FAIL] ${name}"
    fi
}

print_summary_and_exit() {
    echo "[SUMMARY] ${PASS}/${TOTAL} passed"
    if [[ "$TOTAL" -gt 0 && "$FAIL" -eq 0 ]]; then
        exit 0
    fi
    exit 1
}

validate_png_dims() {
    local png_path="$1"
    local exp_w="$2"
    local exp_h="$3"
    python3 - "$png_path" "$exp_w" "$exp_h" <<'PY'
import struct, sys, zlib

path, exp_w, exp_h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "rb") as f:
    data = f.read()

if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
    raise SystemExit(f"Invalid PNG signature in {path}")

pos = 8
ihdr = None
idat_chunks = []
saw_iend = False
idx = 0
while pos + 12 <= len(data):
    length = struct.unpack(">I", data[pos:pos + 4])[0]
    ctype = data[pos + 4:pos + 8]
    end = pos + 8 + length
    if end + 4 > len(data):
        raise SystemExit(f"Truncated PNG chunk {ctype!r} in {path}")
    payload = data[pos + 8:end]
    crc_stored = struct.unpack(">I", data[end:end + 4])[0]
    crc_calc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
    if crc_stored != crc_calc:
        raise SystemExit(f"CRC mismatch in PNG chunk {ctype!r} in {path}")
    if idx == 0:
        if ctype != b"IHDR" or len(payload) != 13:
            raise SystemExit(f"First PNG chunk is not 13-byte IHDR in {path}")
        ihdr = struct.unpack(">IIBBBBB", payload)
    elif ctype == b"IDAT":
        idat_chunks.append(payload)
    elif ctype == b"IEND":
        if len(payload) != 0:
            raise SystemExit(f"Non-empty IEND chunk in {path}")
        saw_iend = True
        pos = end + 4
        break
    pos = end + 4
    idx += 1

if ihdr is None or not idat_chunks or not saw_iend or pos != len(data):
    raise SystemExit(f"Incomplete or trailing PNG structure in {path}")

w, h, bit_depth, color_type, comp, filt, interlace = ihdr
if w != exp_w or h != exp_h:
    raise SystemExit(f"PNG dimension mismatch in {path}: got {w}x{h}, expected {exp_w}x{exp_h}")
legal_depths = {
    0: {1, 2, 4, 8, 16},
    2: {8, 16},
    3: {1, 2, 4, 8},
    4: {8, 16},
    6: {8, 16},
}
if color_type not in legal_depths or bit_depth not in legal_depths[color_type]:
    raise SystemExit(f"Illegal PNG (color_type={color_type}, bit_depth={bit_depth}) in {path}")
if comp != 0 or filt != 0 or interlace not in (0, 1):
    raise SystemExit(f"Invalid PNG header flags comp={comp} filt={filt} interlace={interlace} in {path}")
channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
raw = zlib.decompress(b"".join(idat_chunks))
if interlace == 0:
    row_bytes = (w * channels * bit_depth + 7) // 8
    expected_len = h * (1 + row_bytes)
    if len(raw) != expected_len:
        raise SystemExit(f"PNG decompressed scanline length {len(raw)} != {expected_len} in {path}")
else:
    if len(raw) == 0:
        raise SystemExit(f"Empty decompressed interlaced PNG data in {path}")
PY
}

validate_flpk_v3() {
    local flpk_path="$1"
    shift || true
    python3 - "$flpk_path" "$@" <<'PY'
import struct, sys

path = sys.argv[1]
expected = sys.argv[2:]
with open(path, "rb") as f:
    data = f.read()

if len(data) < 64:
    raise SystemExit(f"FLPK file too small ({len(data)} bytes): {path}")

magic, ver, flags, count, toc_off, uuid_off, path_off, hcap, str_bytes, total_bytes, toc_crc64 = struct.unpack(
    "<4sIIIQQQIIQQ", data[:64]
)
if magic != b"FLPK" or ver != 3:
    raise SystemExit(f"Bad FLPK header magic={magic!r} ver={ver} in {path}")
if total_bytes != len(data):
    raise SystemExit(f"FLPK total_bytes={total_bytes} != file_len={len(data)} in {path}")

if count == 0:
    if hcap != 0:
        raise SystemExit(f"Empty FLPK must have hcap=0, got {hcap} in {path}")
else:
    if hcap < count or (hcap & (hcap - 1)) != 0:
        raise SystemExit(f"Invalid FLPK hash capacity {hcap} for count={count} in {path}")

toc_bytes = count * 64
uuid_bytes = hcap * 32
path_bytes = hcap * 16
str_off = path_off + path_bytes
meta_end = (str_off + str_bytes + 63) & ~63

if meta_end < 64 or meta_end > total_bytes:
    raise SystemExit(f"FLPK meta_end {meta_end} out of bounds ({total_bytes}) in {path}")

for label, off, sz in [
    ("toc", toc_off, toc_bytes),
    ("uuid_table", uuid_off, uuid_bytes),
    ("path_table", path_off, path_bytes),
    ("string_pool", str_off, str_bytes),
]:
    if off < 64 or off + sz > meta_end:
        raise SystemExit(f"FLPK {label} range [{off}, {off + sz}) outside [64, {meta_end}) in {path}")

def _crc64_ecma(buf: bytes) -> int:
    poly = 0x42F0E1EBA9EA3693
    mask = 0xFFFFFFFFFFFFFFFF
    tbl = []
    for n in range(16):
        c = (n << 60) & mask
        for _ in range(4):
            c = (((c << 1) ^ poly) & mask) if (c & 0x8000000000000000) else ((c << 1) & mask)
        tbl.append(c)
    crc = mask
    for b in buf:
        hi = ((crc >> 60) ^ (b >> 4)) & 0xF
        crc = ((crc << 4) & mask) ^ tbl[hi]
        lo = ((crc >> 60) ^ (b & 0xF)) & 0xF
        crc = ((crc << 4) & mask) ^ tbl[lo]
    return crc ^ mask

calc_toc_crc = _crc64_ecma(data[:56] + data[64:meta_end])
if calc_toc_crc != toc_crc64:
    raise SystemExit(f"FLPK toc_crc64 mismatch: got 0x{toc_crc64:016x}, expected 0x{calc_toc_crc:016x} in {path}")

entry_paths = []
for i in range(count):
    ebase = toc_off + i * 64
    u_hi, u_lo, pfnv, p_off, c_bytes, u_bytes, pcrc, sub_id, kind, comp, p_str = struct.unpack(
        "<QQQQQQQHBBI", data[ebase:ebase + 64]
    )
    if p_off < meta_end or p_off + c_bytes > total_bytes:
        raise SystemExit(f"FLPK TOC[{i}] payload range [{p_off}, {p_off + c_bytes}) overlaps metadata (<{meta_end}) or exceeds {total_bytes} in {path}")
    if comp not in (0, 1):
        raise SystemExit(f"FLPK TOC[{i}] unknown compression {comp} in {path}")
    if comp == 0 and c_bytes != u_bytes:
        raise SystemExit(f"FLPK TOC[{i}] uncompressed size mismatch {c_bytes} != {u_bytes} in {path}")
    s_off = p_str & 0xFFFFF
    s_len = (p_str >> 20) & 0xFFF
    if s_off + s_len > str_bytes:
        raise SystemExit(f"FLPK TOC[{i}] string slice out of bounds in {path}")
    ep = data[str_off + s_off:str_off + s_off + s_len].decode("utf-8", errors="replace")
    entry_paths.append(ep)

for exp in expected:
    if exp not in entry_paths:
        raise SystemExit(f"FLPK missing expected entry {exp!r}; found={entry_paths!r}")
PY
}

validate_wav_pcm() {
    local wav_path="$1"
    local exp_channels="${2:-any}"
    local exp_rate="${3:-48000}"
    local exp_bps="${4:-16}"
    python3 - "$wav_path" "$exp_channels" "$exp_rate" "$exp_bps" <<'PY'
import struct, sys

path, exp_ch, exp_rate, exp_bps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(path, "rb") as f:
    data = f.read()

if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
    raise SystemExit(f"Invalid RIFF/WAVE header in {path}")
riff_size = struct.unpack("<I", data[4:8])[0]
if riff_size + 8 != len(data):
    raise SystemExit(f"WAV RIFF chunk size mismatch: {riff_size} + 8 != {len(data)} in {path}")

pos = 12
fmt_info = None
data_len = 0
while pos + 8 <= len(data):
    cid = data[pos:pos + 4]
    clen = struct.unpack("<I", data[pos + 4:pos + 8])[0]
    cstart = pos + 8
    cend = cstart + clen
    if cend > len(data):
        raise SystemExit(f"Truncated WAV chunk {cid!r} in {path}")
    if cid == b"fmt ":
        if clen < 16:
            raise SystemExit(f"Short fmt chunk ({clen}B) in {path}")
        fmt_info = struct.unpack("<HHIIHH", data[cstart:cstart + 16])
    elif cid == b"data":
        data_len = clen
    pos = cend + (clen & 1)

if pos != len(data):
    raise SystemExit(f"Trailing bytes after WAV chunks ({pos} != {len(data)}) in {path}")

if fmt_info is None:
    raise SystemExit(f"Missing fmt chunk in {path}")
audio_fmt, channels, rate, _byte_rate, _align, bps = fmt_info
if audio_fmt != 1:
    raise SystemExit(f"Expected PCM (1), got {audio_fmt} in {path}")
if exp_ch == "any":
    if channels not in (1, 2):
        raise SystemExit(f"Expected 1 or 2 channels, got {channels} in {path}")
elif channels != int(exp_ch):
    raise SystemExit(f"Channel mismatch: got {channels}, expected {exp_ch} in {path}")
if exp_rate > 0 and rate != exp_rate:
    raise SystemExit(f"Sample rate mismatch: got {rate}, expected {exp_rate} in {path}")
if exp_bps > 0 and bps != exp_bps:
    raise SystemExit(f"Bits per sample mismatch: got {bps}, expected {exp_bps} in {path}")
if data_len <= 0:
    raise SystemExit(f"Empty WAV data chunk in {path}")
PY
}

run_in_clean_env() {
    local workdir="$1"
    shift
    (
        cd "$workdir" && \
        env -u FLINT_ROOT -u LD_LIBRARY_PATH -u MOJO -u MOJO_PREFIX -u MODULAR_HOME \
            PATH="/usr/local/bin:/usr/bin:/bin" "$@"
    )
}
