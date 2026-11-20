#!/usr/bin/env bash
# Tier 4 Opaque-Box E2E Suite: End-User CLI Production Pipeline (Scenario 6).
# Covers:
#   1. tools/flint.sh doctor
#   2. tools/flint.sh new + tools/run.sh --headless-smoke
#   3. Headless showcase renders (flint-spark --shots, flint-world --shots, flint-studio --headless-smoke)
#   4. Audio synthesis (flint-cycles render -> RIFF/WAVE validation)
#   5. Asset bundling (flint-pack -> .flpk v3 binary + --inspect validation)
#   6. Multi-turn flint-studio --mcp-stdio JSON-RPC 2.0 session & artifact validation
#
# Runs entirely inside an isolated temp dir (mktemp -d) cleaned on exit.
# Never invokes tools/flint.sh build, gate, or mcp so concurrent milestones are unaffected.
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

pass_step() {
    local name="$1"
    PASS=$((PASS + 1))
    TOTAL=$((TOTAL + 1))
    echo "[PASS] ${name}"
}

fail_step() {
    local name="$1"
    local reason="${2:-assertion failed}"
    FAIL=$((FAIL + 1))
    TOTAL=$((TOTAL + 1))
    echo "[FAIL] ${name}: ${reason}" >&2
}

# Resolve build/<bin_name> only if executable AND newer than both examples/<subdir>/main.mojo
# and all src/flint/**/*.mojo; otherwise compile into $TMPD/bin/<bin_name>.
resolve_bin() {
    local bin_name="$1"
    local example_subdir="$2"
    RESOLVED_BIN=""
    local src="${ROOT}/examples/${example_subdir}/main.mojo"
    if [[ -x "${ROOT}/build/${bin_name}" && -f "$src" && ! "$src" -nt "${ROOT}/build/${bin_name}" ]]; then
        local newer_src=""
        newer_src="$(find "${ROOT}/src/flint" -type f -name '*.mojo' -newer "${ROOT}/build/${bin_name}" -print -quit 2>/dev/null || true)"
        if [[ -z "$newer_src" ]]; then
            RESOLVED_BIN="${ROOT}/build/${bin_name}"
            return 0
        fi
    fi

    local cached="${TMPD}/bin/${bin_name}"
    if [[ -x "$cached" ]]; then
        RESOLVED_BIN="$cached"
        return 0
    fi
    if [[ -f "${cached}.failed" ]]; then
        return 1
    fi
    if [[ ! -f "$src" || ! -x "$MOJO_BIN" ]]; then
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
assert len(chunks) >= 3, f"{path}: expected at least IHDR+IDAT+IEND, got {len(chunks)}"
assert chunks[0][0] == b"IHDR", f"{path}: first chunk is {chunks[0][0]!r}, expected IHDR"
ihdr = chunks[0][1]
assert len(ihdr) == 13, f"{path}: invalid IHDR length {len(ihdr)}"
w, h, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(">IIBBBBB", ihdr)
assert (w, h) == (exp_w, exp_h), f"{path}: expected {exp_w}x{exp_h}, got {w}x{h}"
assert len(idat_chunks) >= 1, f"{path}: missing IDAT chunk"
assert saw_iend and pos == len(data), f"{path}: missing IEND or trailing bytes ({pos} != {len(data)})"
raw = zlib.decompress(b"".join(idat_chunks))
assert len(raw) > 0, f"{path}: decompressed IDAT is empty"
if interlace == 0 and bit_depth == 8 and color_type in (0, 2, 3, 4, 6):
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    expected_raw_len = h * (1 + w * channels)
    assert len(raw) == expected_raw_len, (
        f"{path}: decoded scanlines {len(raw)} != expected {expected_raw_len}"
    )
PY
}

# ============================================================================
# Step 1: Environment check (tools/flint.sh doctor)
# ============================================================================
step_1_doctor() {
    local rc=0
    bash "${ROOT}/tools/flint.sh" doctor >"${TMPD}/doctor.out" 2>"${TMPD}/doctor.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_step "t4_cli_s1_doctor" "exit=${rc}"
        return
    fi
    if grep -qF "[OK]   Mojo 1.2 toolchain" "${TMPD}/doctor.out"; then
        pass_step "t4_cli_s1_doctor"
    else
        fail_step "t4_cli_s1_doctor" "missing '[OK]   Mojo 1.2 toolchain' in doctor output"
    fi
}

# ============================================================================
# Step 2: Scaffold standalone game & headless smoke run
# ============================================================================
step_2_scaffold_and_smoke() {
    local game_dir="${TMPD}/my_game"
    local rc=0
    bash "${ROOT}/tools/flint.sh" new "$game_dir" >"${TMPD}/new.out" 2>"${TMPD}/new.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_step "t4_cli_s2_new_and_smoke" "flint.sh new failed (rc=${rc})"
        return
    fi
    local req_file
    for req_file in flint.toml pixi.toml tools/run.sh src/main.mojo; do
        if [[ ! -f "${game_dir}/${req_file}" ]]; then
            fail_step "t4_cli_s2_new_and_smoke" "missing scaffolded file ${req_file}"
            return
        fi
    done
    local run_rc=0
    (cd "$game_dir" && FLINT_ROOT="$ROOT" bash tools/run.sh --headless-smoke) >"${TMPD}/game_smoke.out" 2>"${TMPD}/game_smoke.err" || run_rc=$?
    if [[ "$run_rc" -ne 0 || ! -s "${TMPD}/game_smoke.out" ]]; then
        fail_step "t4_cli_s2_new_and_smoke" "tools/run.sh --headless-smoke failed (rc=${run_rc})"
        return
    fi
    pass_step "t4_cli_s2_new_and_smoke"
}

# ============================================================================
# Step 3: Headless showcase renders (spark, open_world, studio)
# ============================================================================
SPARK_SHOT=""
STUDIO_SMOKE_PNG=""

step_3_showcase_renders() {
    if ! resolve_bin "flint-spark" "minimal_spark"; then
        fail_step "t4_cli_s3_showcase_renders" "failed to resolve/compile flint-spark"
        return
    fi
    local spark_bin="$RESOLVED_BIN"

    if ! resolve_bin "flint-world" "open_world"; then
        fail_step "t4_cli_s3_showcase_renders" "failed to resolve/compile flint-world"
        return
    fi
    local world_bin="$RESOLVED_BIN"

    if ! resolve_bin "flint-studio" "studio"; then
        fail_step "t4_cli_s3_showcase_renders" "failed to resolve/compile flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"

    mkdir -p "${TMPD}/spark_run/build/shots" "${TMPD}/world_run/build/shots" "${TMPD}/studio_run"
    local rc=0
    (cd "${TMPD}/spark_run" && "$spark_bin" --shots) >"${TMPD}/spark.out" 2>"${TMPD}/spark.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_step "t4_cli_s3_showcase_renders" "flint-spark --shots rc=${rc}"
        return
    fi
    SPARK_SHOT="${TMPD}/spark_run/build/shots/flint_spark.png"
    if [[ ! -f "$SPARK_SHOT" ]]; then
        fail_step "t4_cli_s3_showcase_renders" "missing ${SPARK_SHOT}"
        return
    fi
    local py_err=""
    if ! py_err="$(validate_png_dims "$SPARK_SHOT" 1280 720)"; then
        fail_step "t4_cli_s3_showcase_renders" "spark PNG invalid: ${py_err}"
        return
    fi

    rc=0
    (cd "${TMPD}/world_run" && "$world_bin" --shots) >"${TMPD}/world.out" 2>"${TMPD}/world.err" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail_step "t4_cli_s3_showcase_renders" "flint-world --shots rc=${rc}"
        return
    fi
    local shot
    for shot in open_world_noon_vista.png open_world_golden_citadel.png open_world_night_ltc_bokeh.png; do
        local shot_path="${TMPD}/world_run/build/shots/${shot}"
        if [[ ! -f "$shot_path" ]]; then
            fail_step "t4_cli_s3_showcase_renders" "missing ${shot}"
            return
        fi
        if ! py_err="$(validate_png_dims "$shot_path" 1280 720)"; then
            fail_step "t4_cli_s3_showcase_renders" "world PNG ${shot} invalid: ${py_err}"
            return
        fi
    done

    STUDIO_SMOKE_PNG="${TMPD}/studio_smoke.png"
    rc=0
    (cd "${TMPD}/studio_run" && "$studio_bin" --headless-smoke --out "$STUDIO_SMOKE_PNG") >"${TMPD}/studio.out" 2>"${TMPD}/studio.err" || rc=$?
    if [[ "$rc" -ne 0 || ! -f "$STUDIO_SMOKE_PNG" ]]; then
        fail_step "t4_cli_s3_showcase_renders" "flint-studio --headless-smoke failed (rc=${rc})"
        return
    fi
    if ! py_err="$(validate_png_dims "$STUDIO_SMOKE_PNG" 1280 720)"; then
        fail_step "t4_cli_s3_showcase_renders" "studio PNG invalid: ${py_err}"
        return
    fi

    pass_step "t4_cli_s3_showcase_renders"
}

# ============================================================================
# Step 4: Audio synthesis (flint-cycles render)
# ============================================================================
SOUNDTRACK_WAV=""

step_4_cycles_wav() {
    if ! resolve_bin "flint-cycles" "cycles"; then
        fail_step "t4_cli_s4_cycles_wav" "failed to resolve/compile flint-cycles"
        return
    fi
    local cycles_bin="$RESOLVED_BIN"
    SOUNDTRACK_WAV="${TMPD}/soundtrack.wav"
    local rc=0
    "$cycles_bin" render "${ROOT}/examples/cycles/demo.tidal" "$SOUNDTRACK_WAV" 1.5 >"${TMPD}/cycles.out" 2>"${TMPD}/cycles.err" || rc=$?
    if [[ "$rc" -ne 0 || ! -f "$SOUNDTRACK_WAV" ]]; then
        fail_step "t4_cli_s4_cycles_wav" "cycles render failed (rc=${rc})"
        return
    fi
    local py_err=""
    if py_err="$(python3 - "$SOUNDTRACK_WAV" <<'PY' 2>&1
import struct, sys
data = open(sys.argv[1], "rb").read()
assert len(data) >= 44, f"wav too short: {len(data)}"
assert data[0:4] == b"RIFF" and data[8:12] == b"WAVE", f"bad RIFF/WAVE header"
riff_size = struct.unpack_from("<I", data, 4)[0]
assert riff_size >= 36 and 8 + riff_size <= len(data), f"bad RIFF size {riff_size}"
pos = 12
fmt_chunk = None
data_len = 0
data_off = 0
while pos + 8 <= len(data):
    cid = data[pos:pos + 4]
    clen = struct.unpack_from("<I", data, pos + 4)[0]
    cend = pos + 8 + clen
    assert cend <= len(data), f"truncated chunk {cid!r}"
    if cid == b"fmt ":
        fmt_chunk = data[pos + 8:cend]
    elif cid == b"data":
        data_off = pos + 8
        data_len = clen
    pos = cend + (clen & 1)
assert fmt_chunk is not None and len(fmt_chunk) >= 16, "missing fmt chunk"
audio_fmt, channels, sample_rate, _br, _ba, bps = struct.unpack_from("<HHIIHH", fmt_chunk, 0)
assert audio_fmt == 1, f"expected PCM=1, got {audio_fmt}"
assert channels == 2, f"expected 2ch, got {channels}"
assert sample_rate == 48000, f"expected 48000Hz, got {sample_rate}"
assert bps == 16, f"expected 16-bit, got {bps}"
assert data_len > 0 and data_off + data_len <= len(data), f"invalid data chunk len={data_len}"
PY
)"; then
        pass_step "t4_cli_s4_cycles_wav"
    else
        fail_step "t4_cli_s4_cycles_wav" "WAV validation failed: ${py_err}"
    fi
}

# ============================================================================
# Step 5: Asset bundling & inspection (flint-pack)
# ============================================================================
step_5_pack_bundle() {
    if [[ ! -f "$SPARK_SHOT" || ! -f "$STUDIO_SMOKE_PNG" || ! -f "$SOUNDTRACK_WAV" ]]; then
        fail_step "t4_cli_s5_pack_bundle" "missing upstream generated PNG/WAV inputs"
        return
    fi
    if ! resolve_bin "flint-pack" "pack_cli"; then
        fail_step "t4_cli_s5_pack_bundle" "failed to resolve/compile flint-pack"
        return
    fi
    local pack_bin="$RESOLVED_BIN"
    local bundle="${TMPD}/game_bundle.flpk"
    "$pack_bin" "$bundle" "$SPARK_SHOT" "$STUDIO_SMOKE_PNG" "$SOUNDTRACK_WAV" >"${TMPD}/pack.out" 2>"${TMPD}/pack.err"
    if ! grep -qF "[Flint Pack] Packed" "${TMPD}/pack.out" || [[ ! -s "$bundle" ]]; then
        fail_step "t4_cli_s5_pack_bundle" "missing '[Flint Pack] Packed' or empty ${bundle}"
        return
    fi

    "$pack_bin" --inspect "$bundle" >"${TMPD}/pack_inspect.out" 2>"${TMPD}/pack_inspect.err"
    if ! grep -qF "flint_spark.png" "${TMPD}/pack_inspect.out" \
        || ! grep -qF "studio_smoke.png" "${TMPD}/pack_inspect.out" \
        || ! grep -qF "soundtrack.wav" "${TMPD}/pack_inspect.out"; then
        fail_step "t4_cli_s5_pack_bundle" "--inspect output missing expected entries"
        return
    fi

    local py_err=""
    if py_err="$(python3 - "$bundle" <<'PY' 2>&1
import struct, sys
data = open(sys.argv[1], "rb").read()
assert len(data) >= 64, f"archive too short: {len(data)}"
assert data[:4] == b"FLPK", f"bad magic: {data[:4]!r}"
ver = struct.unpack_from("<I", data, 4)[0]
assert ver == 3, f"expected FLPK v3, got {ver}"
flags, count, toc_off, uuid_off, path_off, hcap, str_bytes, total_bytes, _toc_crc64 = struct.unpack_from(
    "<IIQQQIIQQ", data, 8
)
assert count >= 3, f"expected >= 3 entries in bundle, got {count}"
assert 64 <= total_bytes <= len(data), f"bad total_file_bytes={total_bytes} vs len={len(data)}"
assert toc_off >= 64 and toc_off + count * 64 <= total_bytes, "bad toc_off"
assert hcap >= count and (hcap & (hcap - 1)) == 0, f"bad hcap={hcap}"
str_off = path_off + hcap * 16
assert path_off >= uuid_off >= toc_off + count * 64, "bad v3 table ordering"
assert str_off + str_bytes <= total_bytes, "string pool out of bounds"
str_pool = data[str_off:str_off + str_bytes]
for name in (b"flint_spark.png", b"studio_smoke.png", b"soundtrack.wav"):
    assert name in str_pool, f"missing {name!r} in v3 string pool"
for i in range(count):
    slab_off, comp_bytes, uncomp_bytes = struct.unpack_from("<QQQ", data, toc_off + i * 64 + 24)
    assert slab_off >= 64 and comp_bytes > 0 and uncomp_bytes > 0, f"bad TOC entry {i}"
    assert slab_off + comp_bytes <= total_bytes, f"entry {i} slab out of bounds"
PY
)"; then
        pass_step "t4_cli_s5_pack_bundle"
    else
        fail_step "t4_cli_s5_pack_bundle" "FLPK v3 binary validation failed: ${py_err}"
    fi
}

# ============================================================================
# Step 6: Multi-turn flint-studio --mcp-stdio authoring session
# ============================================================================
step_6_mcp_stdio_session() {
    if ! resolve_bin "flint-studio" "studio"; then
        fail_step "t4_cli_s6_mcp_stdio_session" "failed to resolve/compile flint-studio"
        return
    fi
    local studio_bin="$RESOLVED_BIN"
    local mcp_ws="${TMPD}/mcp_ws"
    mkdir -p "${mcp_ws}/assets"

    local py_err=""
    if py_err="$(python3 - "$studio_bin" "$mcp_ws" <<'PY' 2>&1
import json, os, struct, subprocess, sys, zlib

studio_bin = sys.argv[1]
mcp_ws = sys.argv[2]

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
        assert sep >= 0, f"missing Content-Length separator at {pos}: {buf[pos:pos+80]!r}"
        hdr = buf[pos:sep].decode("ascii")
        cl = None
        for line in hdr.splitlines():
            if line.lower().startswith("content-length:"):
                cl = int(line.split(":", 1)[1].strip())
        assert cl is not None, f"missing Content-Length header in {hdr!r}"
        body_start = sep + sep_len
        body_end = body_start + cl
        assert body_end <= len(buf), f"truncated body at {pos}"
        msgs.append(json.loads(buf[body_start:body_end].decode("utf-8")))
        pos = body_end
    return msgs

reqs = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t4-cli", "version": "1.0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "procedural.generate_asset", "arguments": {"preset": "material_orb", "output_glb_path": "assets/orb.glb", "spawn_in_scene": True, "position": [0.0, 2.0, 0.0]}}},
    {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "ecs.query_entities", "arguments": {"name_contains": "material_orb"}}},
    {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "ecs.patch_components", "arguments": {"entity_id": 0, "base_color": [0.95, 0.2, 0.15]}}},
    {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "physics.step_deterministic", "arguments": {"ticks": 10}}},
    {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "studio.pie_control", "arguments": {"action": "play"}}},
    {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "studio.pie_control", "arguments": {"action": "step", "ticks": 12}}},
    {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "studio.pie_control", "arguments": {"action": "stop"}}},
    {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "render.capture_viewport", "arguments": {"output_path": "cap_a.png", "width": 128, "height": 96}}},
    {"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "ecs.patch_components", "arguments": {"entity_id": 0, "position": [0.5, 2.8, -0.4], "base_color": [0.1, 0.9, 0.25]}}},
    {"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {"name": "render.capture_viewport", "arguments": {"output_path": "cap_b.png", "reference_path": "cap_a.png", "width": 128, "height": 96}}},
    {"jsonrpc": "2.0", "id": 13, "method": "tools/call", "params": {"name": "render.inspect_gpu_frame", "arguments": {}}},
    {"jsonrpc": "2.0", "id": 14, "method": "tools/call", "params": {"name": "assets.list_and_cook", "arguments": {"source_dir": "assets", "output_flpk": "assets.flpk"}}},
]

payload = b"".join(cl_frame(r) for r in reqs)
proc = subprocess.run([studio_bin, "--mcp-stdio"], input=payload, capture_output=True, timeout=30, cwd=mcp_ws)
assert proc.returncode == 0, f"mcp-stdio rc={proc.returncode}, stderr={proc.stderr!r}"
assert proc.stderr == b"", f"unexpected stderr from mcp-stdio: {proc.stderr!r}"

msgs = parse_cl_frames(proc.stdout)
assert len(msgs) == 14, f"expected 14 responses (notification is silent), got {len(msgs)}"
by_id = {m["id"]: m for m in msgs}

assert by_id[1]["result"]["protocolVersion"] == "2024-11-05"
assert len(by_id[2]["result"]["tools"]) == 14

for rid in range(3, 15):
    res = by_id[rid]["result"]
    assert not res.get("isError", False), f"tool call id={rid} returned error: {res}"
    sc = res["structuredContent"]
    assert sc.get("ok", False) is True, f"tool call id={rid} sc.ok != True: {sc}"

assert by_id[4]["result"]["structuredContent"]["count"] >= 1
assert by_id[6]["result"]["structuredContent"]["ticks_advanced"] == 10
assert by_id[9]["result"]["structuredContent"]["restored_exact"] is True
assert by_id[12]["result"]["structuredContent"]["has_flip_diff"] is True
assert by_id[12]["result"]["structuredContent"]["flip_mean"] > 0.0

for fname in ("assets/orb.glb", "cap_a.png", "cap_b.png", "assets.flpk"):
    fpath = os.path.join(mcp_ws, fname)
    assert os.path.isfile(fpath) and os.path.getsize(fpath) > 0, f"missing or empty {fpath}"

for png_name in ("cap_a.png", "cap_b.png"):
    pdata = open(os.path.join(mcp_ws, png_name), "rb").read()
    assert pdata[:8] == b"\x89PNG\r\n\x1a\n", f"bad PNG magic in {png_name}"
    w, h = struct.unpack(">II", pdata[16:24])
    assert (w, h) == (128, 96), f"unexpected dims {w}x{h} in {png_name}"

flpk_data = open(os.path.join(mcp_ws, "assets.flpk"), "rb").read()
assert flpk_data[:4] == b"FLPK", f"bad FLPK magic in assets.flpk: {flpk_data[:4]!r}"
PY
)"; then
        pass_step "t4_cli_s6_mcp_stdio_session"
    else
        fail_step "t4_cli_s6_mcp_stdio_session" "${py_err}"
    fi
}

step_1_doctor
step_2_scaffold_and_smoke
step_3_showcase_renders
step_4_cycles_wav
step_5_pack_bundle
step_6_mcp_stdio_session

echo "[SUMMARY] ${PASS}/${TOTAL} passed"
if [[ "$FAIL" -ne 0 ]]; then
    exit 1
fi
exit 0
