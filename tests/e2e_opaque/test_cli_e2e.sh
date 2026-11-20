#!/usr/bin/env bash
# Opaque-Box CLI & Binary E2E Suite (FA1 — Tiers 1 & 2).
# Covers `tools/flint.sh` subcommands, standalone binaries (`flint-pack`,
# `flint-cycles`, `flint-kms`, `flint-spark`, `flint-world`, `flint-studio`),
# binary artifact validation (`.flpk`, `.wav`, PNG IHDR, JSON-RPC 2.0),
# and boundary/error-rejection paths in isolated temporary directories.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MOJO_BIN="${MOJO:-${ROOT}/.pixi/envs/default/bin/mojo}"

if [[ -d "${ROOT}/.pixi/envs/default" ]]; then
    export MOJO_PREFIX="${ROOT}/.pixi/envs/default"
    export MODULAR_HOME="${MOJO_PREFIX}/share/max"
    export LD_LIBRARY_PATH="${MOJO_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT
mkdir -p "${TMPD}/bin"

PASS=0
FAIL=0
TOTAL=0
RESOLVED_BIN=""

pass_test() {
    local name="$1"
    PASS=$((PASS + 1))
    TOTAL=$((TOTAL + 1))
    echo "[PASS] ${name}"
}

fail_test() {
    local name="$1"
    local reason="${2:-assertion failed}"
    FAIL=$((FAIL + 1))
    TOTAL=$((TOTAL + 1))
    echo "[FAIL] ${name}: ${reason}" >&2
}

# Resolve pre-built binary in build/<bin_name> (only if not older than examples/<subdir>/main.mojo)
# or compile examples/<subdir>/main.mojo directly into $TMPD/bin/<bin_name> without invoking tools/flint.sh build or gate.
resolve_bin() {
    local bin_name="$1"
    local example_subdir="$2"
    RESOLVED_BIN=""
    local src="${ROOT}/examples/${example_subdir}/main.mojo"
    if [[ -x "${ROOT}/build/${bin_name}" && -f "$src" && ! "$src" -nt "${ROOT}/build/${bin_name}" ]]; then
        RESOLVED_BIN="${ROOT}/build/${bin_name}"
        return 0
    fi
    local cached="${TMPD}/bin/${bin_name}"
    if [[ -x "$cached" ]]; then
        RESOLVED_BIN="$cached"
        return 0
    fi
    if [[ -f "${cached}.failed" ]]; then
        return 1
    fi
    local src="${ROOT}/examples/${example_subdir}/main.mojo"
    if [[ ! -f "$src" ]]; then
        echo "Missing example source: ${src}" >"${TMPD}/build_${bin_name}.log"
        touch "${cached}.failed"
        return 1
    fi
    if [[ ! -x "$MOJO_BIN" ]]; then
        echo "Missing Mojo compiler: ${MOJO_BIN}" >"${TMPD}/build_${bin_name}.log"
        touch "${cached}.failed"
        return 1
    fi
    if "$MOJO_BIN" build -I "${ROOT}/src" -I "${ROOT}/tests" -O3 --fp-mode contract=off --Werror \
        "$src" -o "$cached" >"${TMPD}/build_${bin_name}.log" 2>&1; then
        RESOLVED_BIN="$cached"
        return 0
    fi
    touch "${cached}.failed"
    return 1
}

# ============================================================================
# TIER 1: HAPPY-PATH CLI & BINARY TESTS (11 tests)
# ============================================================================

# 1. tools/flint.sh help
t1_cli_help() {
    local rc=0
    bash "${ROOT}/tools/flint.sh" help >"${TMPD}/t1_help.out" 2>"${TMPD}/t1_help.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t1_cli_help" "exit code ${rc}"
        return
    fi
    if grep -q "Flint Engine v5 CLI" "${TMPD}/t1_help.out" && grep -q "Usage: bash tools/flint.sh" "${TMPD}/t1_help.out"; then
        pass_test "t1_cli_help"
    else
        fail_test "t1_cli_help" "missing CLI usage banner in stdout"
    fi
}

# 2. tools/flint.sh doctor
t1_cli_doctor() {
    local rc=0
    bash "${ROOT}/tools/flint.sh" doctor >"${TMPD}/t1_doctor.out" 2>"${TMPD}/t1_doctor.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t1_cli_doctor" "exit code ${rc}"
        return
    fi
    if grep -q "FLINT ENGINE ENVIRONMENT DOCTOR" "${TMPD}/t1_doctor.out" && grep -q "Mojo 1\.2" "${TMPD}/t1_doctor.out"; then
        pass_test "t1_cli_doctor"
    else
        fail_test "t1_cli_doctor" "missing doctor report or Mojo 1.2 status"
    fi
}

# 3. tools/flint.sh sync --verify
t1_cli_sync_verify() {
    local rc=0
    bash "${ROOT}/tools/flint.sh" sync --verify >"${TMPD}/t1_sync.out" 2>"${TMPD}/t1_sync.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t1_cli_sync_verify" "exit code ${rc}"
        return
    fi
    if grep -qE "\[PASS\] Zero-Game-Leak Gate" "${TMPD}/t1_sync.out"; then
        pass_test "t1_cli_sync_verify"
    else
        fail_test "t1_cli_sync_verify" "missing Zero-Game-Leak Gate pass banner"
    fi
}

# 4. tools/flint.sh lab --help
t1_cli_lab_help() {
    local rc=0
    bash "${ROOT}/tools/flint.sh" lab --help >"${TMPD}/t1_lab_help.out" 2>"${TMPD}/t1_lab_help.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t1_cli_lab_help" "exit code ${rc}"
        return
    fi
    if grep -q "Usage: bash tools/flint.sh lab" "${TMPD}/t1_lab_help.out"; then
        pass_test "t1_cli_lab_help"
    else
        fail_test "t1_cli_lab_help" "missing lab usage message"
    fi
}

# 5. flint-pack .flpk output validation
t1_bin_pack_flpk() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t1_bin_pack_flpk" "failed to resolve/compile flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"
    printf 'hello flint v5 asset payload\n' >"${TMPD}/asset_sample.txt"
    local out_flpk="${TMPD}/packed_sample.flpk"
    local rc=0
    "$pack_bin" "$out_flpk" "${TMPD}/asset_sample.txt" >"${TMPD}/t1_pack.out" 2>"${TMPD}/t1_pack.err" || rc=$?
    if [[ "$rc" -ne 0 || ! -s "$out_flpk" ]]; then
        fail_test "t1_bin_pack_flpk" "exit=${rc}, output missing or empty"
        return
    fi
    local py_err=""
    if py_err="$(python3 - "$out_flpk" "asset_sample.txt" "${TMPD}/asset_sample.txt" <<'PY' 2>&1
import struct, sys
pack_path, expected_name, expected_payload_path = sys.argv[1], sys.argv[2], sys.argv[3]
data = open(pack_path, "rb").read()
expected_payload = open(expected_payload_path, "rb").read()
assert len(data) >= 24, f"file too short: {len(data)}"
assert data[:4] == b"FLPK", f"bad magic: {data[:4]!r}"
ver = struct.unpack_from("<I", data, 4)[0]
assert ver in (2, 3), f"unexpected version: {ver}"
if ver == 2:
    count = struct.unpack_from("<I", data, 8)[0]
    dir_bytes = struct.unpack_from("<I", data, 12)[0]
    assert count == 1, f"expected v2 count=1, got {count}"
    assert 16 + dir_bytes <= len(data), f"v2 directory out of bounds: {dir_bytes}"
    assert expected_name.encode("utf-8") in data[16:16 + dir_bytes], "packed entry name missing in v2 directory"
    assert expected_payload in data[16 + dir_bytes:], "packed payload missing in v2 archive"
else:
    assert len(data) >= 64, f"v3 header shorter than 64 bytes: {len(data)}"
    flags, count, toc_off, uuid_off, path_off, hcap, str_bytes, total_bytes, toc_crc64 = struct.unpack_from(
        "<IIQQQIIQQ", data, 8
    )
    assert count == 1, f"expected v3 toc_count=1, got {count}"
    assert 64 <= total_bytes <= len(data), f"invalid v3 total_file_bytes={total_bytes} vs len={len(data)}"
    assert toc_off >= 64 and toc_off + count * 64 <= total_bytes, f"invalid toc_off={toc_off}"
    assert hcap >= count and (hcap & (hcap - 1)) == 0, f"invalid hash_table_capacity={hcap}"
    str_off = path_off + hcap * 16
    assert path_off >= uuid_off >= toc_off + count * 64, "invalid v3 table ordering"
    assert str_off + str_bytes <= total_bytes, "v3 string pool out of bounds"
    slab_off, comp_bytes, uncomp_bytes = struct.unpack_from("<QQQ", data, toc_off + 24)
    comp_mode = data[toc_off + 59]
    assert slab_off >= 64 and comp_bytes > 0 and uncomp_bytes > 0, (
        f"invalid v3 TOC entry 0: slab_off={slab_off}, comp={comp_bytes}, uncomp={uncomp_bytes}"
    )
    assert slab_off + comp_bytes <= total_bytes, "v3 payload slab out of bounds"
    assert expected_name.encode("utf-8") in data[str_off:str_off + str_bytes], "packed entry name missing in v3 string pool"
    if comp_mode == 0:
        assert data[slab_off:slab_off + len(expected_payload)] == expected_payload, "v3 payload mismatch"
PY
)"; then
        pass_test "t1_bin_pack_flpk"
    else
        fail_test "t1_bin_pack_flpk" "flpk validation failed: ${py_err}"
    fi
}

# 6. flint-cycles query 's "bd sn"' 1
t1_bin_cycles_query() {
    if ! resolve_bin "flint-cycles" "cycles"; then
        fail_test "t1_bin_cycles_query" "failed to resolve/compile flint-cycles"
        return
    fi
    local cycles_bin="$RESOLVED_BIN"
    local rc=0
    "$cycles_bin" query 's "bd sn"' 1 >"${TMPD}/t1_cq.out" 2>"${TMPD}/t1_cq.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t1_bin_cycles_query" "exit code ${rc}"
        return
    fi
    if grep -qF "(0>1/2) s=bd" "${TMPD}/t1_cq.out" && grep -qF "(1/2>1) s=sn" "${TMPD}/t1_cq.out"; then
        pass_test "t1_bin_cycles_query"
    else
        fail_test "t1_bin_cycles_query" "unexpected query events: $(tr '\n' ' ' <"${TMPD}/t1_cq.out")"
    fi
}

# 7. flint-cycles render demo.tidal -> WAV validation
t1_bin_cycles_render_wav() {
    if ! resolve_bin "flint-cycles" "cycles"; then
        fail_test "t1_bin_cycles_render_wav" "failed to resolve/compile flint-cycles"
        return
    fi
    local cycles_bin="$RESOLVED_BIN"
    local wav_out="${TMPD}/demo.wav"
    local rc=0
    "$cycles_bin" render "${ROOT}/examples/cycles/demo.tidal" "$wav_out" 1.0 >"${TMPD}/t1_cr.out" 2>"${TMPD}/t1_cr.err" || rc=$?
    if [[ "$rc" -ne 0 || ! -f "$wav_out" ]]; then
        fail_test "t1_bin_cycles_render_wav" "exit=${rc}, wav missing"
        return
    fi
    local py_err=""
    if py_err="$(python3 - "$wav_out" <<'PY' 2>&1
import struct, sys
data = open(sys.argv[1], "rb").read()
assert len(data) >= 12, f"wav too small: {len(data)}"
assert data[0:4] == b"RIFF", f"missing RIFF: {data[0:4]!r}"
riff_size = struct.unpack_from("<I", data, 4)[0]
assert data[8:12] == b"WAVE", f"missing WAVE: {data[8:12]!r}"
assert riff_size >= 4 and 8 + riff_size <= len(data), f"invalid RIFF size {riff_size} vs len {len(data)}"
pos = 12
fmt_chunk = None
data_len = None
data_off = None
while pos + 8 <= len(data):
    cid = data[pos:pos + 4]
    clen = struct.unpack_from("<I", data, pos + 4)[0]
    cend = pos + 8 + clen
    assert cend <= len(data), f"truncated WAV chunk {cid!r}: {cend} > {len(data)}"
    if cid == b"fmt ":
        fmt_chunk = data[pos + 8:cend]
    elif cid == b"data":
        data_off = pos + 8
        data_len = clen
    pos = cend + (clen & 1)
assert fmt_chunk is not None and len(fmt_chunk) >= 16, (
    f"missing or short 'fmt ' chunk: {None if fmt_chunk is None else len(fmt_chunk)}"
)
audio_fmt, channels, sample_rate, _byte_rate, _block_align, bits_per_sample = struct.unpack_from("<HHIIHH", fmt_chunk, 0)
assert audio_fmt == 1, f"expected PCM audio_format=1, got {audio_fmt}"
assert channels == 2, f"expected 2 channels, got {channels}"
assert sample_rate == 48000, f"expected 48000 Hz, got {sample_rate}"
assert bits_per_sample == 16, f"expected 16-bit PCM, got {bits_per_sample}"
assert data_len is not None and data_len > 0, f"missing or empty 'data' chunk: {data_len}"
assert data_off + data_len <= len(data), "WAV 'data' chunk exceeds file length"
PY
)"; then
        pass_test "t1_bin_cycles_render_wav"
    else
        fail_test "t1_bin_cycles_render_wav" "wav validation failed: ${py_err}"
    fi
}

# 8. flint-kms --probe
t1_bin_kms_probe() {
    if ! resolve_bin "flint-kms" "kms_console"; then
        fail_test "t1_bin_kms_probe" "failed to resolve/compile flint-kms"
        return
    fi
    local kms_bin="$RESOLVED_BIN"
    local rc=0
    "$kms_bin" --probe >"${TMPD}/t1_kms.out" 2>"${TMPD}/t1_kms.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t1_bin_kms_probe" "exit code ${rc}"
        return
    fi
    if grep -qE "/dev/dri/card|no /dev/dri/card" "${TMPD}/t1_kms.out"; then
        pass_test "t1_bin_kms_probe"
    else
        fail_test "t1_bin_kms_probe" "unexpected kms --probe output"
    fi
}

# Helper: validate PNG magic, full chunk structure (IHDR/IDAT/IEND + CRCs), dimensions, and zlib scanlines
validate_png_dims() {
    local png_path="$1"
    local exp_w="$2"
    local exp_h="$3"
    python3 - "$png_path" "$exp_w" "$exp_h" <<'PY' 2>&1
import struct, sys, zlib
path, exp_w, exp_h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "rb") as f:
    data = f.read()
assert len(data) >= 8, f"{path}: shorter than PNG signature ({len(data)})"
assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: invalid PNG magic {data[:8]!r}"
pos = 8
chunks = []
idat_chunks = []
saw_iend = False
while pos < len(data):
    assert pos + 12 <= len(data), f"{path}: truncated PNG chunk header at {pos}"
    clen = struct.unpack_from(">I", data, pos)[0]
    ctype = data[pos + 4:pos + 8]
    cstart = pos + 8
    cend = cstart + clen
    assert cend + 4 <= len(data), f"{path}: truncated PNG chunk {ctype!r} (len={clen}) at {pos}"
    cdata = data[cstart:cend]
    ccrc = struct.unpack_from(">I", data, cend)[0]
    calc_crc = zlib.crc32(ctype + cdata) & 0xFFFFFFFF
    assert ccrc == calc_crc, f"{path}: CRC mismatch in chunk {ctype!r}: {ccrc:#x} != {calc_crc:#x}"
    chunks.append((ctype, cdata))
    if ctype == b"IDAT":
        assert clen > 0, f"{path}: empty IDAT chunk"
        idat_chunks.append(cdata)
    elif ctype == b"IEND":
        assert clen == 0, f"{path}: non-empty IEND chunk (len={clen})"
        saw_iend = True
        pos = cend + 4
        break
    pos = cend + 4
assert len(chunks) >= 3, f"{path}: expected at least IHDR+IDAT+IEND, got {len(chunks)} chunks"
assert chunks[0][0] == b"IHDR", f"{path}: first chunk is {chunks[0][0]!r}, expected IHDR"
ihdr = chunks[0][1]
assert len(ihdr) == 13, f"{path}: invalid IHDR length {len(ihdr)} != 13"
w, h, bit_depth, color_type, _comp_method, _filter_method, interlace = struct.unpack(">IIBBBBB", ihdr)
assert (w, h) == (exp_w, exp_h), f"{path}: expected {exp_w}x{exp_h}, got {w}x{h}"
assert len(idat_chunks) >= 1, f"{path}: missing IDAT chunk"
assert saw_iend and pos == len(data), f"{path}: missing IEND or trailing bytes after IEND ({pos} != {len(data)})"
raw = zlib.decompress(b"".join(idat_chunks))
assert len(raw) > 0, f"{path}: decompressed IDAT is empty"
if interlace == 0 and bit_depth == 8 and color_type in (0, 2, 3, 4, 6):
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    expected_raw_len = h * (1 + w * channels)
    assert len(raw) == expected_raw_len, (
        f"{path}: decoded scanline bytes {len(raw)} != expected {expected_raw_len} ({w}x{h}x{channels})"
    )
PY
}

# 9. flint-spark --shots isolated inside $TMPD/spark
t1_bin_spark_shots() {
    if ! resolve_bin "flint-spark" "minimal_spark"; then
        fail_test "t1_bin_spark_shots" "failed to resolve/compile flint-spark"
        return
    fi
    local spark_bin="$RESOLVED_BIN"
    mkdir -p "${TMPD}/spark/build/shots"
    local rc=0
    (cd "${TMPD}/spark" && "$spark_bin" --shots) >"${TMPD}/t1_spark.out" 2>"${TMPD}/t1_spark.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t1_bin_spark_shots" "exit code ${rc}"
        return
    fi
    local png="${TMPD}/spark/build/shots/flint_spark.png"
    if [[ ! -f "$png" ]]; then
        fail_test "t1_bin_spark_shots" "missing output PNG ${png}"
        return
    fi
    local py_err=""
    if py_err="$(validate_png_dims "$png" 1280 720)"; then
        pass_test "t1_bin_spark_shots"
    else
        fail_test "t1_bin_spark_shots" "${py_err}"
    fi
}

# 10. flint-world --shots isolated inside $TMPD/world
t1_bin_world_shots() {
    if ! resolve_bin "flint-world" "open_world"; then
        fail_test "t1_bin_world_shots" "failed to resolve/compile flint-world"
        return
    fi
    local world_bin="$RESOLVED_BIN"
    mkdir -p "${TMPD}/world/build/shots"
    local rc=0
    (cd "${TMPD}/world" && "$world_bin" --shots) >"${TMPD}/t1_world.out" 2>"${TMPD}/t1_world.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_test "t1_bin_world_shots" "exit code ${rc}"
        return
    fi
    local shots=(
        "${TMPD}/world/build/shots/open_world_noon_vista.png"
        "${TMPD}/world/build/shots/open_world_golden_citadel.png"
        "${TMPD}/world/build/shots/open_world_night_ltc_bokeh.png"
    )
    local s
    for s in "${shots[@]}"; do
        if [[ ! -f "$s" ]]; then
            local hint=""
            if grep -qiE "Vulkan.*unavailable|ERROR" "${TMPD}/t1_world.out" "${TMPD}/t1_world.err" 2>/dev/null; then
                hint=" ($(grep -iE 'Vulkan.*unavailable|ERROR' "${TMPD}/t1_world.out" "${TMPD}/t1_world.err" | head -n1))"
            fi
            fail_test "t1_bin_world_shots" "missing shot $(basename "$s")${hint}"
            return
        fi
        local py_err=""
        if ! py_err="$(validate_png_dims "$s" 1280 720)"; then
            fail_test "t1_bin_world_shots" "${py_err}"
            return
        fi
    done
    pass_test "t1_bin_world_shots"
}

# 11. flint-studio --headless-smoke AND --mcp-stdio (both Content-Length and newline framing)
t1_bin_studio_smoke_and_mcp() {
    if ! resolve_bin "flint-studio" "studio"; then
        local err_line=""
        err_line="$(tail -n 5 "${TMPD}/build_flint-studio.log" 2>/dev/null | tr '\n' ' ' || true)"
        fail_test "t1_bin_studio_smoke_and_mcp" "failed to resolve/compile flint-studio (${err_line})"
        return
    fi
    local studio_bin="$RESOLVED_BIN"
    local studio_run_dir="${TMPD}/studio_run"
    mkdir -p "$studio_run_dir"
    local smoke_png="${TMPD}/studio_smoke.png"
    local rc=0
    (cd "$studio_run_dir" && "$studio_bin" --headless-smoke --out "$smoke_png") >"${TMPD}/t1_studio.out" 2>"${TMPD}/t1_studio.err" || rc=$?
    if [[ "$rc" -ne 0 || ! -f "$smoke_png" ]]; then
        local hint=""
        if grep -qiE "Vulkan.*unavailable|ERROR" "${TMPD}/t1_studio.out" "${TMPD}/t1_studio.err" 2>/dev/null; then
            hint=" ($(grep -iE 'Vulkan.*unavailable|ERROR' "${TMPD}/t1_studio.out" "${TMPD}/t1_studio.err" | head -n1))"
        fi
        fail_test "t1_bin_studio_smoke_and_mcp" "headless smoke failed (rc=${rc}, png_exists=$(test -f "$smoke_png" && echo yes || echo no))${hint}"
        return
    fi
    local py_err=""
    if ! py_err="$(validate_png_dims "$smoke_png" 1280 720)"; then
        fail_test "t1_bin_studio_smoke_and_mcp" "${py_err}"
        return
    fi

    if py_err="$(python3 - "$studio_bin" "$studio_run_dir" <<'PY' 2>&1
import json, subprocess, sys

studio_bin = sys.argv[1]
studio_cwd = sys.argv[2]

def cl_frame(obj):
    raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw

def parse_cl_frames(buf):
    msgs = []
    pos = 0
    while pos < len(buf):
        sep = buf.find(b"\r\n\r\n", pos)
        sep_len = 4
        if sep < 0:
            sep = buf.find(b"\n\n", pos)
            sep_len = 2
        assert sep >= 0, f"missing header terminator at {pos}: {buf[pos:pos+80]!r}"
        hdr = buf[pos:sep].decode("ascii")
        cl = None
        for line in hdr.splitlines():
            if line.lower().startswith("content-length:"):
                cl = int(line.split(":", 1)[1].strip())
        assert cl is not None, f"missing Content-Length in {hdr!r}"
        body_start = sep + sep_len
        body_end = body_start + cl
        assert body_end <= len(buf), f"truncated Content-Length body ({len(buf) - body_start} < {cl})"
        msgs.append(json.loads(buf[body_start:body_end].decode("utf-8")))
        pos = body_end
    return msgs

init_req = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "e2e", "version": "1.0"}},
}
notif_req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

# Session A: Content-Length framing
cl_in = cl_frame(init_req) + cl_frame(notif_req) + cl_frame(list_req)
p_cl = subprocess.run([studio_bin, "--mcp-stdio"], input=cl_in, capture_output=True, timeout=15, cwd=studio_cwd)
assert p_cl.returncode == 0, f"CL mcp rc={p_cl.returncode}, stderr={p_cl.stderr!r}"
assert len(p_cl.stderr) == 0, f"CL mcp unexpected stderr: {p_cl.stderr!r}"
cl_msgs = parse_cl_frames(p_cl.stdout)
assert len(cl_msgs) == 2, f"expected 2 CL responses (silent notification), got {len(cl_msgs)}: {cl_msgs}"
assert cl_msgs[0]["id"] == 1 and cl_msgs[0]["result"]["protocolVersion"] == "2024-11-05", f"bad CL init: {cl_msgs[0]}"
assert cl_msgs[1]["id"] == 2 and len(cl_msgs[1]["result"]["tools"]) == 14, f"expected 14 tools in CL, got {len(cl_msgs[1]['result']['tools'])}"

# Session B: Newline-delimited JSON-RPC framing
nl_in = (
    json.dumps(init_req, separators=(",", ":")) + "\n"
    + json.dumps(notif_req, separators=(",", ":")) + "\n"
    + json.dumps(list_req, separators=(",", ":")) + "\n"
).encode("utf-8")
p_nl = subprocess.run([studio_bin, "--mcp-stdio"], input=nl_in, capture_output=True, timeout=15, cwd=studio_cwd)
assert p_nl.returncode == 0, f"NL mcp rc={p_nl.returncode}, stderr={p_nl.stderr!r}"
assert len(p_nl.stderr) == 0, f"NL mcp unexpected stderr: {p_nl.stderr!r}"
nl_lines = [ln for ln in p_nl.stdout.decode("utf-8").splitlines() if ln.strip()]
assert len(nl_lines) == 2, f"expected 2 NL lines (silent notification), got {len(nl_lines)}: {nl_lines}"
nl_msgs = [json.loads(ln) for ln in nl_lines]
assert nl_msgs[0]["id"] == 1 and nl_msgs[0]["result"]["protocolVersion"] == "2024-11-05", f"bad NL init: {nl_msgs[0]}"
assert nl_msgs[1]["id"] == 2 and len(nl_msgs[1]["result"]["tools"]) == 14, f"expected 14 tools in NL, got {len(nl_msgs[1]['result']['tools'])}"
PY
)"; then
        pass_test "t1_bin_studio_smoke_and_mcp"
    else
        fail_test "t1_bin_studio_smoke_and_mcp" "MCP stdio failed: ${py_err}"
    fi
}

# ============================================================================
# TIER 2: BOUNDARY & ERROR-REJECTION TESTS (7 tests)
# ============================================================================

# 1. flint.sh run in empty dir (missing flint.toml)
t2_cli_run_missing_toml() {
    mkdir -p "${TMPD}/empty_proj"
    local rc=0
    (cd "${TMPD}/empty_proj" && bash "${ROOT}/tools/flint.sh" run) >"${TMPD}/t2_run.out" 2>"${TMPD}/t2_run.err" || rc=$?
    if [[ "$rc" -eq 1 ]] && grep -qE "No flint\.toml found" "${TMPD}/t2_run.err"; then
        pass_test "t2_cli_run_missing_toml"
    else
        fail_test "t2_cli_run_missing_toml" "expected exit=1 and 'No flint.toml found', got rc=${rc}"
    fi
}

# 2. flint.sh new (missing arg)
t2_cli_new_missing_arg() {
    local rc=0
    bash "${ROOT}/tools/flint.sh" new >"${TMPD}/t2_new.out" 2>"${TMPD}/t2_new.err" || rc=$?
    if [[ "$rc" -eq 1 ]] && grep -q "Usage: tools/flint.sh new" "${TMPD}/t2_new.err"; then
        pass_test "t2_cli_new_missing_arg"
    else
        fail_test "t2_cli_new_missing_arg" "expected exit=1 and 'Usage: tools/flint.sh new', got rc=${rc}"
    fi
}

# 3. flint.sh lab invalid
t2_cli_lab_invalid() {
    local rc=0
    bash "${ROOT}/tools/flint.sh" lab invalid >"${TMPD}/t2_lab.out" 2>"${TMPD}/t2_lab.err" || rc=$?
    if [[ "$rc" -eq 1 ]] && grep -q "Unknown lab 'invalid'" "${TMPD}/t2_lab.err"; then
        pass_test "t2_cli_lab_invalid"
    else
        fail_test "t2_cli_lab_invalid" "expected exit=1 and Unknown lab 'invalid', got rc=${rc}"
    fi
}

# 4. flint.sh preview --help
t2_cli_preview_help() {
    local rc=0
    bash "${ROOT}/tools/flint.sh" preview --help >"${TMPD}/t2_prev.out" 2>"${TMPD}/t2_prev.err" || rc=$?
    if [[ "$rc" -eq 1 ]] && grep -q "Usage: bash tools/flint.sh preview" "${TMPD}/t2_prev.out"; then
        pass_test "t2_cli_preview_help"
    else
        fail_test "t2_cli_preview_help" "expected exit=1 and preview usage on stdout, got rc=${rc}"
    fi
}

# 5. flint-pack <3 args usage AND missing input error + no output file created
t2_bin_pack_errors() {
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_test "t2_bin_pack_errors" "failed to resolve/compile flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"
    local rc1=0
    "$pack_bin" >"${TMPD}/t2_pack_args.out" 2>"${TMPD}/t2_pack_args.err" || rc1=$?
    if ! grep -q "Usage: flint-pack" "${TMPD}/t2_pack_args.out"; then
        fail_test "t2_bin_pack_errors" "missing usage message on <3 args"
        return
    fi
    local bad_out="${TMPD}/should_not_exist.flpk"
    rm -f "$bad_out"
    local rc2=0
    "$pack_bin" "$bad_out" "${TMPD}/nonexistent_input_asset.bin" >"${TMPD}/t2_pack_miss.out" 2>"${TMPD}/t2_pack_miss.err" || rc2=$?
    if grep -q "\[Flint Pack\] ERROR" "${TMPD}/t2_pack_miss.out" && [[ ! -e "$bad_out" ]]; then
        pass_test "t2_bin_pack_errors"
    else
        fail_test "t2_bin_pack_errors" "missing '[Flint Pack] ERROR' or created ${bad_out}"
    fi
}

# 6. flint-cycles <3 args usage, bad .tidal render exits non-zero, bad query exits non-zero
t2_bin_cycles_errors() {
    if ! resolve_bin "flint-cycles" "cycles"; then
        fail_test "t2_bin_cycles_errors" "failed to resolve/compile flint-cycles"
        return
    fi
    local cycles_bin="$RESOLVED_BIN"
    local rc1=0
    "$cycles_bin" >"${TMPD}/t2_cyc_args.out" 2>"${TMPD}/t2_cyc_args.err" || rc1=$?
    if ! grep -q "usage:" "${TMPD}/t2_cyc_args.out"; then
        fail_test "t2_bin_cycles_errors" "missing usage block on <3 args"
        return
    fi

    printf 'invalid @@@ syntax\n' >"${TMPD}/bad_syntax.tidal"
    local rc2=0
    "$cycles_bin" render "${TMPD}/bad_syntax.tidal" "${TMPD}/bad_syntax.wav" 1.0 >"${TMPD}/t2_cyc_r.out" 2>"${TMPD}/t2_cyc_r.err" || rc2=$?
    if [[ "$rc2" -eq 0 ]]; then
        fail_test "t2_bin_cycles_errors" "bad .tidal syntax in render unexpectedly exited 0"
        return
    fi

    local rc3=0
    "$cycles_bin" query "invalid(((" 1 >"${TMPD}/t2_cyc_q.out" 2>"${TMPD}/t2_cyc_q.err" || rc3=$?
    if [[ "$rc3" -eq 0 ]]; then
        fail_test "t2_bin_cycles_errors" "bad query expression unexpectedly exited 0"
        return
    fi
    pass_test "t2_bin_cycles_errors"
}

# 7. flint-studio --mcp-stdio invalid JSON-RPC & malformed headers
t2_bin_studio_mcp_invalid_jsonrpc() {
    if ! resolve_bin "flint-studio" "studio"; then
        local err_line=""
        err_line="$(tail -n 5 "${TMPD}/build_flint-studio.log" 2>/dev/null | tr '\n' ' ' || true)"
        fail_test "t2_bin_studio_mcp_invalid_jsonrpc" "failed to resolve/compile flint-studio (${err_line})"
        return
    fi
    local studio_bin="$RESOLVED_BIN"
    local studio_run_dir="${TMPD}/studio_run"
    mkdir -p "$studio_run_dir"
    local py_err=""
    if py_err="$(python3 - "$studio_bin" "$studio_run_dir" <<'PY' 2>&1
import json, subprocess, sys

studio_bin = sys.argv[1]
studio_cwd = sys.argv[2]

def run_nl(payload_str):
    p = subprocess.run([studio_bin, "--mcp-stdio"], input=payload_str.encode("utf-8"), capture_output=True, timeout=15, cwd=studio_cwd)
    lines = [ln for ln in p.stdout.decode("utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 1, f"expected error response, got stdout={p.stdout!r}, stderr={p.stderr!r}"
    return json.loads(lines[0])

# 1) Malformed JSON -> -32700 Parse error
r_parse = run_nl("{bad json\n")
assert r_parse["error"]["code"] == -32700, f"expected -32700, got {r_parse}"

# 2) Wrong jsonrpc version -> -32600 Invalid Request
r_ver = run_nl('{"jsonrpc":"1.0","id":1,"method":"initialize"}\n')
assert r_ver["error"]["code"] == -32600, f"expected -32600, got {r_ver}"

# 3) Unknown method -> -32601 Method not found
r_meth = run_nl('{"jsonrpc":"2.0","id":2,"method":"nonexistent/method"}\n')
assert r_meth["error"]["code"] == -32601, f"expected -32601, got {r_meth}"

# 4) Bad frame header -> -32700 Parse error
r_hdr = run_nl("BadHeader: 5\r\n\r\nhello")
assert r_hdr["error"]["code"] == -32700, f"expected -32700 for bad header, got {r_hdr}"
PY
)"; then
        pass_test "t2_bin_studio_mcp_invalid_jsonrpc"
    else
        fail_test "t2_bin_studio_mcp_invalid_jsonrpc" "${py_err}"
    fi
}

# Execute Tier 1
t1_cli_help
t1_cli_doctor
t1_cli_sync_verify
t1_cli_lab_help
t1_bin_pack_flpk
t1_bin_cycles_query
t1_bin_cycles_render_wav
t1_bin_kms_probe
t1_bin_spark_shots
t1_bin_world_shots
t1_bin_studio_smoke_and_mcp

# Execute Tier 2
t2_cli_run_missing_toml
t2_cli_new_missing_arg
t2_cli_lab_invalid
t2_cli_preview_help
t2_bin_pack_errors
t2_bin_cycles_errors
t2_bin_studio_mcp_invalid_jsonrpc

echo "[SUMMARY] ${PASS}/${TOTAL} passed"
if [[ "$FAIL" -ne 0 ]]; then
    exit 1
fi
exit 0
