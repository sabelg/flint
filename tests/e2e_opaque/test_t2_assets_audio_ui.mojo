"""Tier 2 (Boundary, Corner & Invalid Input Rejection) Opaque-Box E2E Suite for FA5 & FA6.

Uses ONLY public Flint Engine v5 APIs (zero private `_` symbols from `flint.*`).
All filesystem writes are isolated to `/tmp/flint-e2e-opaque-t2-<pid>/` and cleaned up on exit.
"""
from std.collections import List
from std.ffi import external_call
from std.math import isinf, isnan
from std.testing import assert_equal, assert_false, assert_true

from flint.assets.cooker_v2 import (
    cook_native_model_bundle,
    cook_single_source_bytes,
    cook_workspace,
)
from flint.assets.flpk_v3 import (
    ASSET_KIND_RAW,
    AssetMeta,
    AssetUuid,
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    encode_flpk_v3,
    parse_flmeta_bytes,
)
from flint.assets.glb_native import parse_native_glb
from flint.assets.hdr_image import (
    decode_exr_fp16,
    decode_hdr_or_exr,
    decode_radiance_hdr,
    encode_exr_fp16,
)
from flint.assets.jpeg import decode_image_auto, decode_jpeg, encode_jpeg_baseline
from flint.assets.model import NativeModel
from flint.assets.server import AssetServer, RawBytesAsset
from flint.audio import (
    AudioServer,
    QoaAsset,
    QoaStreamCursor,
    decode_qoa,
    encode_qoa,
)
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.input import ACTION_BOOL, InputContext, InputMap
from flint.platform.input import InputState
from flint.platform.os import make_dirs, write_bytes
from flint.render.font import SdfFont
from flint.text.line_break import wrap
from flint.text.rich_text import RichSpan, layout_rich, parse_rich, plain_text
from flint.text.sdf_bake import bake_font
from flint.text.ttf import TrueTypeFont
from flint.text.utf8 import decode_utf8
from flint.ui.flexbox_dock import (
    FLEX_ROW,
    DockSpace,
    FlexStyle,
    FlexTree,
    RectF,
)

comptime SYSTEM_TTF_PATH: String = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def _tmp_dir(tag: String) -> String:
    var pid = Int(external_call["getpid", Int32]())
    return "/tmp/flint-e2e-opaque-t2-" + String(pid) + "-" + tag


def _cleanup_tmp(dir_path: String):
    if not dir_path.startswith("/tmp/flint-e2e-opaque-t2-"):
        return
    var cmd = "rm -rf " + dir_path
    _ = external_call["system", Int32](cmd.as_c_string_span())


def _append_u32_le(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


def _str_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _make_glb_with_json(json_text: String, bin_bytes: List[UInt8]) -> List[UInt8]:
    var jb = _str_to_bytes(json_text)
    while len(jb) % 4 != 0:
        jb.append(0x20)
    var bb = bin_bytes.copy()
    while len(bb) % 4 != 0:
        bb.append(0x00)
    var total_len = 12 + 8 + len(jb) + 8 + len(bb)
    var out = List[UInt8](capacity=total_len)
    _append_u32_le(out, 0x46546C67)
    _append_u32_le(out, 2)
    _append_u32_le(out, UInt32(total_len))
    _append_u32_le(out, UInt32(len(jb)))
    _append_u32_le(out, 0x4E4F534A)
    for b in jb:
        out.append(b)
    _append_u32_le(out, UInt32(len(bb)))
    _append_u32_le(out, 0x004E4942)
    for b in bb:
        out.append(b)
    return out^


# ============================================================================
# FA5 Tier 2: Boundary, Corner & Invalid Input Rejection (7 Tests)
# ============================================================================


def t2_fa5_cook_workspace_empty_and_bad_sources() raises:
    var root = _tmp_dir("empty_cook")
    _cleanup_tmp(root)
    var pool = JobPool()
    _ = pool.start(2)
    try:
        var empty_ws = root + "/empty_ws"
        var cache_dir = root + "/cache"
        var out_flpk = root + "/empty.flpk"
        _ = make_dirs(empty_ws)

        var rep = cook_workspace(empty_ws, out_flpk, pool, cache_dir=cache_dir)
        assert_true(rep.ok)
        assert_equal(rep.total_assets, 0)
        assert_equal(rep.archive_bytes, 64)

        var bad_png: List[UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x00, 0x01]
        var meta_tex = AssetMeta()
        meta_tex.uuid = AssetUuid.from_path("bad.png")
        var b_tex = cook_single_source_bytes(bad_png, meta_tex, "bad.png")
        assert_false(b_tex.valid)
        assert_true(b_tex.error.byte_length() > 0)

        var bad_glb: List[UInt8] = [0x67, 0x6C, 0x54, 0x46, 0x01, 0x00]
        var meta_mesh = AssetMeta()
        meta_mesh.uuid = AssetUuid.from_path("bad.glb")
        var b_mesh = cook_single_source_bytes(bad_glb, meta_mesh, "bad.glb")
        assert_false(b_mesh.valid)
        assert_true(b_mesh.error.byte_length() > 0)

        var empty_model = NativeModel()
        var b_empty_mdl = cook_native_model_bundle(empty_model, meta_mesh, "empty.glb")
        assert_false(b_empty_mdl.valid)
        assert_true(b_empty_mdl.error.byte_length() > 0)
    except e:
        pool.stop()
        _cleanup_tmp(root)
        raise e^
    pool.stop()
    _cleanup_tmp(root)


def t2_fa5_flpk_v3_corrupt_crc64_and_bad_files() raises:
    var root = _tmp_dir("bad_flpk")
    _cleanup_tmp(root)
    _ = make_dirs(root)
    try:
        var missing_arch = FlpkV3MappedArchive.open(root + "/does_not_exist.flpk")
        assert_false(missing_arch.valid)
        missing_arch.close()

        var tiny_path = root + "/tiny.flpk"
        var tiny_bytes = List[UInt8](length=16, fill=0x46)
        assert_true(write_bytes(tiny_path, tiny_bytes))
        var tiny_arch = FlpkV3MappedArchive.open(tiny_path)
        assert_false(tiny_arch.valid)
        tiny_arch.close()

        var payload = List[UInt8](length=128, fill=0x5A)
        var u0 = AssetUuid.from_path("data/item.bin")
        var chunks = List[FlpkBuildChunk]()
        chunks.append(FlpkBuildChunk(u0, "data/item.bin", 0, ASSET_KIND_RAW, payload^))
        var good_pack = encode_flpk_v3(chunks, page_align=False)
        var good_path = root + "/good.flpk"
        assert_true(write_bytes(good_path, good_pack))

        var good_arch = FlpkV3MappedArchive.open(good_path)
        assert_true(good_arch.valid)
        assert_equal(good_arch.find_by_path("nonexistent/item.bin", 0), -1)
        assert_equal(good_arch.find_by_uuid(AssetUuid.from_path("missing"), 0), -1)
        var pay_off = Int(good_arch.toc_entry(0).payload_offset)
        good_arch.close()

        # Flip 1 byte in TOC -> open must fail TOC CRC64 verification
        var bad_toc = good_pack.copy()
        bad_toc[72] ^= UInt8(0x01)
        var bad_toc_path = root + "/bad_toc.flpk"
        assert_true(write_bytes(bad_toc_path, bad_toc))
        var toc_arch = FlpkV3MappedArchive.open(bad_toc_path)
        assert_false(toc_arch.valid)
        toc_arch.close()

        # Flip 1 byte in payload -> open succeeds, entry CRC64 / mmap / read fail
        var bad_pay = good_pack.copy()
        bad_pay[pay_off + 4] ^= UInt8(0x08)
        var bad_pay_path = root + "/bad_pay.flpk"
        assert_true(write_bytes(bad_pay_path, bad_pay))
        var pay_arch = FlpkV3MappedArchive.open(bad_pay_path)
        assert_true(pay_arch.valid)
        assert_false(pay_arch.verify_entry_crc64(0))
        assert_false(pay_arch.mmap_slice_by_toc(0).valid)
        assert_false(pay_arch.read_payload_by_toc(0)[0])
        pay_arch.close()
    except e:
        _cleanup_tmp(root)
        raise e^
    _cleanup_tmp(root)


def t2_fa5_asset_server_missing_and_zero_lru_vs_pinned() raises:
    var root = _tmp_dir("srv_neg")
    _cleanup_tmp(root)
    _ = make_dirs(root)
    try:
        var server = AssetServer(stream_budget_bytes_per_tick=1024, enable_watcher=False)
        assert_false(server.mount_flpk(root + "/missing.flpk"))
        assert_false(server.load_mesh("missing.glb").is_valid())
        assert_false(server.load_texture("missing.png").is_valid())
        assert_false(server.load_raw("missing.bin").is_valid())

        var raw_data = List[UInt8](length=4096, fill=0x77)
        var h = server.raws.insert(RawBytesAsset(raw_data^), AssetUuid.from_path("pinned.bin"), "pinned.bin", 1)
        assert_true(h.is_valid())
        assert_true(server.raws.ref_count(h) >= 1)

        # Even with lru_max_bytes = 0, pinned (ref_count >= 1) must NOT be evicted
        server.lru_max_bytes = 0
        var ev_pinned = server.enforce_lru_budget()
        assert_equal(ev_pinned, 0)
        assert_true(server.raws.is_alive(h))
        assert_equal(server.total_resident_bytes(), 4096)

        # Once released (ref_count == 0), enforce_lru_budget evicts it cleanly
        assert_true(server.raws.release(h))
        var ev_released = server.enforce_lru_budget()
        assert_equal(ev_released, 1)
        assert_false(server.raws.is_alive(h))
        assert_equal(server.total_resident_bytes(), 0)
        server.close()
    except e:
        _cleanup_tmp(root)
        raise e^
    _cleanup_tmp(root)


def t2_fa5_glb_malformed_and_truncated() raises:
    assert_false(parse_native_glb(List[UInt8]()).loaded())

    # Bad version (= 1 instead of 2) and truncated header
    var bad_ver = List[UInt8](length=32, fill=0)
    bad_ver[0] = 0x67; bad_ver[1] = 0x6C; bad_ver[2] = 0x54; bad_ver[3] = 0x46
    bad_ver[4] = 1
    assert_false(parse_native_glb(bad_ver).loaded())

    var short_hdr: List[UInt8] = [0x67, 0x6C, 0x54, 0x46, 2, 0, 0, 0, 12, 0, 0, 0]
    assert_false(parse_native_glb(short_hdr).loaded())

    # Corrupt JSON chunk
    var bad_json_glb = _make_glb_with_json("{not valid json", List[UInt8](length=16, fill=0))
    assert_false(parse_native_glb(bad_json_glb).loaded())

    # Valid JSON referencing out-of-bounds accessor bufferView
    var oob_json = (
        '{"asset":{"version":"2.0"},'
        + '"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],'
        + '"nodes":[{"mesh":0}],'
        + '"accessors":[{"bufferView":0,"componentType":5126,"count":100,"type":"VEC3"}],'
        + '"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":12}]}'
    )
    var oob_glb = _make_glb_with_json(oob_json, List[UInt8](length=16, fill=0))
    assert_false(parse_native_glb(oob_glb).loaded())


def t2_fa5_jpeg_malformed_and_invalid_encode() raises:
    var px = List[UInt32](length=4, fill=0xFFFFFFFF)
    assert_equal(len(encode_jpeg_baseline(px, 0, 16)), 0)
    assert_equal(len(encode_jpeg_baseline(px, 16, 16)), 0)

    assert_false(decode_jpeg(List[UInt8]()).valid)

    var png_sig: List[UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert_false(decode_jpeg(png_sig).valid)

    # Truncated SOI + SOF0 without SOS
    var trunc_sof0: List[UInt8] = [
        0xFF, 0xD8,
        0xFF, 0xC0, 0x00, 0x0B, 8, 0, 8, 0, 8, 1, 1, 0x11, 0,
        0xFF, 0xD9,
    ]
    assert_false(decode_jpeg(trunc_sof0).valid)

    var garbage = List[UInt8](length=32, fill=0xAA)
    assert_false(decode_jpeg(garbage).valid)
    assert_false(decode_image_auto(garbage).valid)


def t2_fa5_hdr_exr_malformed_and_tiled_flag() raises:
    assert_false(decode_radiance_hdr(List[UInt8]()).valid)
    assert_false(decode_radiance_hdr(_str_to_bytes("BADMAGIC\n\n-Y 1 +X 1\n\x00\x00\x00\x00")).valid)
    assert_false(decode_radiance_hdr(_str_to_bytes("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n")).valid)
    assert_false(decode_radiance_hdr(_str_to_bytes("#?RADIANCE\n\n-Y 4 +X 4\n\x00\x00")).valid)

    assert_false(decode_exr_fp16(List[UInt8]()).valid)
    var bad_magic = List[UInt8](length=40, fill=0x11)
    assert_false(decode_exr_fp16(bad_magic).valid)

    # Start from valid EXR, then mutate tiled bit (0x200), unsupported compression (99), and truncate
    var valid_exr = encode_exr_fp16(List[Vec3](length=4 * 4, fill=Vec3(1.0, 1.0, 1.0)), 4, 4)
    assert_true(len(valid_exr) > 64)

    var tiled_exr = valid_exr.copy()
    tiled_exr[5] |= UInt8(0x02)  # sets tiled bit 0x200 in ver_word
    assert_false(decode_exr_fp16(tiled_exr).valid)

    # Locate compression attribute payload (byte after "compression\0compression\0" + u32(1))
    var bad_comp_exr = valid_exr.copy()
    var comp_tag = _str_to_bytes("compression")
    var found_comp = -1
    for i in range(8, len(bad_comp_exr) - 28):
        var is_match = True
        for k in range(len(comp_tag)):
            if bad_comp_exr[i + k] != comp_tag[k]:
                is_match = False
                break
        if is_match and bad_comp_exr[i + len(comp_tag)] == 0:
            found_comp = i + len(comp_tag) + 1 + len(comp_tag) + 1 + 4
            break
    assert_true(found_comp > 0)
    bad_comp_exr[found_comp] = UInt8(99)
    assert_false(decode_exr_fp16(bad_comp_exr).valid)

    var trunc_exr = List[UInt8](capacity=48)
    for i in range(48):
        trunc_exr.append(valid_exr[i])
    assert_false(decode_exr_fp16(trunc_exr).valid)

    assert_false(decode_hdr_or_exr(List[UInt8](length=32, fill=0x5A)).valid)


def t2_fa5_uuid_and_flmeta_invalid_inputs() raises:
    assert_false(AssetUuid.parse("")[0])
    assert_false(AssetUuid.parse("too-short")[0])
    assert_false(AssetUuid.parse("zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz")[0])
    assert_false(AssetUuid.parse("00000000_0000_0000_0000_000000000000")[0])

    assert_false(parse_flmeta_bytes(List[UInt8]())[0])
    assert_false(parse_flmeta_bytes(_str_to_bytes("{bad json"))[0])
    assert_false(parse_flmeta_bytes(_str_to_bytes('{"asset_kind": 1}'))[0])


# ============================================================================
# FA6 Tier 2: Boundary, Corner & Invalid Input Rejection (7 Tests)
# ============================================================================


def t2_fa6_audio_voice_stealing_and_invalid_assets() raises:
    var srv = AudioServer(48000, 256)
    var pcm = List[Int16](capacity=1024)
    for i in range(1024):
        pcm.append(Int16((i % 64) * 200))
    var aid = srv.register_pcm16_asset(pcm, 1, 48000)
    assert_true(aid >= 0)

    # First saturate all 64 physical voices with distant low-priority emitters, then steal >= 16 with closer high-priority emitters
    for i in range(64):
        _ = srv.play_3d(aid, Vec3(Float32(40 + (i % 8)), 0.0, -25.0), volume=0.25, looping=True, priority=1)

    var buf = List[Float32](length=512, fill=0.0)
    srv.render_offline_quantum(buf, 256)
    srv.render_offline_quantum(buf, 256)
    assert_equal(srv.active_physical_count(), 64)

    for i in range(20):
        _ = srv.play_3d(aid, Vec3(Float32((i % 8) - 4), 0.0, -4.0), volume=0.5, looping=True, priority=4)
    srv.render_offline_quantum(buf, 256)
    assert_equal(srv.active_physical_count(), 64)
    assert_true(srv.voices_stolen_count() >= UInt64(16))

    # Empty PCM & invalid QOA bytes must play silence cleanly without NaN or crash
    var bad_pcm_aid = srv.register_pcm16_asset(List[Int16](), 1, 48000)
    var bad_qoa_bytes: List[UInt8] = [1, 2, 3]
    var bad_qoa_aid = srv.register_qoa_asset(bad_qoa_bytes^)
    _ = srv.play_2d(bad_pcm_aid)
    _ = srv.play_2d(bad_qoa_aid)
    srv.render_offline_quantum(buf, 256)
    for s in buf:
        assert_false(isnan(s))
        assert_false(isinf(s))


def t2_fa6_qoa_invalid_bytes_and_oob_seek() raises:
    var pcm = List[Int16](length=128, fill=1000)
    assert_equal(len(encode_qoa(pcm, 0, 48000)), 0)
    assert_equal(len(encode_qoa(pcm, 9, 48000)), 0)
    assert_equal(len(encode_qoa(List[Int16](), 2, 48000)), 0)

    assert_false(decode_qoa(List[UInt8]()).valid)
    assert_false(decode_qoa(List[UInt8](length=32, fill=0x11)).valid)

    var encoded = encode_qoa(pcm, 2, 48000)
    assert_true(len(encoded) > 16)
    var trunc = List[UInt8](capacity=20)
    for i in range(20):
        trunc.append(encoded[i])
    assert_false(decode_qoa(trunc).valid)

    assert_false(QoaAsset.from_bytes(List[UInt8]()).valid)
    assert_false(QoaAsset.from_bytes(List[UInt8](length=20, fill=0x22)).valid)

    var cursor = QoaStreamCursor()
    var bad_cur_bytes: List[UInt8] = [1, 2, 3, 4]
    assert_false(cursor.open(bad_cur_bytes))

    assert_true(cursor.open(encoded))
    assert_true(cursor.seek(encoded, 9999999))
    var out_i16 = List[Int16](length=32, fill=0)
    assert_equal(cursor.decode_frames_into(encoded, out_i16, 0, 16), 0)
    assert_true(cursor.seek(encoded, -100))
    assert_true(cursor.decode_frames_into(encoded, out_i16, 0, 16) > 0)


def t2_fa6_input_map_unmapped_empty_pop_and_expired_timers() raises:
    var imap = InputMap()
    assert_false(imap.is_down("NoSuchAction"))
    assert_false(imap.just_pressed("NoSuchAction"))
    assert_false(imap.just_released("NoSuchAction"))
    assert_equal(imap.axis_1d("NoSuchAction"), Float32(0.0))
    var a2 = imap.axis_2d("NoSuchAction")
    assert_equal(a2[0], Float32(0.0))
    assert_equal(a2[1], Float32(0.0))
    assert_false(imap.consume_buffered("NoSuchAction"))
    assert_equal(imap.prompt_glyph_for("NoSuchAction").label, "?")

    assert_false(imap.pop_context())
    assert_false(imap.pop_context("UnknownContext"))
    assert_false(imap.load_json("{bad json"))

    _ = imap.register_action("Jump", ACTION_BOOL)
    var ctx = InputContext("Gameplay", priority=0)
    ctx.bind_key("Jump", 32)
    imap.add_context(ctx^)
    imap.push_context("Gameplay")

    var inp = InputState()
    imap.set_grounded(False)
    inp.begin_frame()
    inp.set_logical_key(32, True)
    imap.update(inp, 0.020)
    inp.set_logical_key(32, False)
    for _ in range(8):
        inp.begin_frame()
        imap.update(inp, 0.020)
    assert_false(imap.consume_buffered("Jump", 0.120))

    imap.set_grounded(True)
    inp.begin_frame()
    imap.update(inp, 0.020)
    imap.set_grounded(False)
    for _ in range(7):
        inp.begin_frame()
        imap.update(inp, 0.020)
    assert_false(imap.coyote_valid(0.100))


def t2_fa6_text_ttf_empty_invalid_and_otto() raises:
    var f_missing = TrueTypeFont()
    assert_false(f_missing.load("/tmp/flint_no_such_font_999.ttf"))
    assert_false(f_missing.valid)

    var f_empty = TrueTypeFont()
    assert_false(f_empty.load_bytes(List[UInt8]()))
    assert_false(f_empty.valid)
    assert_true(f_empty.error.byte_length() > 0)

    var f_rand = TrueTypeFont()
    assert_false(f_rand.load_bytes(List[UInt8](length=16, fill=0x7F)))
    assert_false(f_rand.valid)
    assert_true(f_rand.error.byte_length() > 0)

    # Truncated SFNT header claiming 10 tables in 12 bytes
    var trunc_hdr: List[UInt8] = [0x00, 0x01, 0x00, 0x00, 0x00, 10, 0, 0, 0, 0, 0, 0]
    var f_trunc = TrueTypeFont()
    assert_false(f_trunc.load_bytes(trunc_hdr^))
    assert_false(f_trunc.valid)
    assert_true(f_trunc.error.byte_length() > 0)

    # OTTO (CFF OpenType) header
    var otto = List[UInt8](length=32, fill=0)
    otto[0] = 0x4F; otto[1] = 0x54; otto[2] = 0x54; otto[3] = 0x4F
    var f_otto = TrueTypeFont()
    assert_false(f_otto.load_bytes(otto^))
    assert_false(f_otto.valid)
    assert_true(f_otto.error.byte_length() > 0)

    # Tiny atlas_width triggers explicit "atlas too small" error
    var valid_font = TrueTypeFont()
    assert_true(valid_font.load(SYSTEM_TTF_PATH))
    var threw = False
    try:
        var codes: List[Int] = [65]
        _ = bake_font(valid_font, codes, 32, 4, atlas_width=4)
    except:
        threw = True
    assert_true(threw)


def t2_fa6_text_line_break_zero_width_empty_and_tiny() raises:
    var sdf = SdfFont()
    assert_true(sdf.load_ttf(SYSTEM_TTF_PATH, 32, 4))

    var empty_lines = wrap(sdf, List[Int](), 20.0, 100.0)
    assert_equal(len(empty_lines), 0)

    var unlim = wrap(sdf, decode_utf8("Hello world"), 20.0, 0.0)
    assert_equal(len(unlim), 1)
    assert_equal(unlim[0].start, 0)
    assert_equal(unlim[0].end, 11)

    var tiny = wrap(sdf, decode_utf8("Word"), 20.0, 1.0)
    assert_equal(len(tiny), 4)

    var invalid_lines = wrap(SdfFont(), decode_utf8("Abc"), 20.0, 50.0)
    assert_equal(len(invalid_lines), 1)
    assert_equal(invalid_lines[0].width, Float32(0.0))


def t2_fa6_text_rich_text_malformed_tags() raises:
    var raw = "[c=ZZZ]bad[/c] [unknown] [/d] [s=999]x[/s] [[lit [unclosed"
    var spans = parse_rich(raw)
    var txt = plain_text(spans)
    assert_true(txt.find("[c=ZZZ]") >= 0)
    assert_true(txt.find("[unknown]") >= 0)
    assert_true(txt.find("[/d]") >= 0)
    assert_true(txt.find("[s=999]") >= 0)
    assert_true(txt.find("[lit") >= 0)
    assert_false(txt.find("[[lit") >= 0)

    var sdf = SdfFont()
    assert_true(sdf.load_ttf(SYSTEM_TTF_PATH, 32, 4))
    var empty_layout = layout_rich(List[RichSpan](), sdf, sdf, 20.0, 0, 100.0, 0, False, False)
    assert_equal(empty_layout.lines, 0)
    assert_equal(empty_layout.width, Float32(0.0))
    assert_equal(empty_layout.height, Float32(0.0))
    assert_equal(len(empty_layout.runs), 0)


def t2_fa6_dockspace_overflow_zero_width_and_bad_json() raises:
    var dock = DockSpace.create_studio_5panel()
    dock.solve_layout(RectF(0.0, 0.0, 0.0, 0.0))
    var z_leaves = dock.leaf_rects()
    assert_equal(len(z_leaves), 5)
    for r in z_leaves:
        assert_false(isnan(r.w) or isnan(r.h))
        assert_equal(r.w, Float32(0.0))
        assert_equal(r.h, Float32(0.0))
        assert_equal(r.area(), Float32(0.0))

    var empty_dock = DockSpace()
    empty_dock.solve_layout(RectF(0.0, 0.0, 800.0, 600.0))
    assert_equal(len(empty_dock.leaf_rects()), 0)
    assert_false(empty_dock.load_json("{invalid"))

    # FlexTree overconstrained container (sum_min = 150.0 > container width 10.0)
    var tree = FlexTree()
    var root_st = FlexStyle(FLEX_ROW)
    _ = tree.add_node(-1, root_st, 10.0, 40.0)
    for _ in range(3):
        var cst = FlexStyle(FLEX_ROW)
        cst.min_w = 50.0
        cst.min_h = 20.0
        _ = tree.add_node(0, cst, 50.0, 20.0)
    tree.solve(RectF(0.0, 0.0, 10.0, 40.0), 0)
    for i in range(1, 4):
        var cr = tree.nodes[i].rect
        assert_true(cr.w >= Float32(0.0) and cr.h >= Float32(0.0))
        assert_true(cr.x >= Float32(-0.01) and cr.right() <= Float32(10.01))
        for j in range(i + 1, 4):
            assert_false(cr.overlaps_interior(tree.nodes[j].rect, Float32(0.01)))


def main() raises:
    var passed = 0
    var total = 14

    t2_fa5_cook_workspace_empty_and_bad_sources()
    print("[PASS] t2_fa5_cook_workspace_empty_and_bad_sources")
    passed += 1

    t2_fa5_flpk_v3_corrupt_crc64_and_bad_files()
    print("[PASS] t2_fa5_flpk_v3_corrupt_crc64_and_bad_files")
    passed += 1

    t2_fa5_asset_server_missing_and_zero_lru_vs_pinned()
    print("[PASS] t2_fa5_asset_server_missing_and_zero_lru_vs_pinned")
    passed += 1

    t2_fa5_glb_malformed_and_truncated()
    print("[PASS] t2_fa5_glb_malformed_and_truncated")
    passed += 1

    t2_fa5_jpeg_malformed_and_invalid_encode()
    print("[PASS] t2_fa5_jpeg_malformed_and_invalid_encode")
    passed += 1

    t2_fa5_hdr_exr_malformed_and_tiled_flag()
    print("[PASS] t2_fa5_hdr_exr_malformed_and_tiled_flag")
    passed += 1

    t2_fa5_uuid_and_flmeta_invalid_inputs()
    print("[PASS] t2_fa5_uuid_and_flmeta_invalid_inputs")
    passed += 1

    t2_fa6_audio_voice_stealing_and_invalid_assets()
    print("[PASS] t2_fa6_audio_voice_stealing_and_invalid_assets")
    passed += 1

    t2_fa6_qoa_invalid_bytes_and_oob_seek()
    print("[PASS] t2_fa6_qoa_invalid_bytes_and_oob_seek")
    passed += 1

    t2_fa6_input_map_unmapped_empty_pop_and_expired_timers()
    print("[PASS] t2_fa6_input_map_unmapped_empty_pop_and_expired_timers")
    passed += 1

    t2_fa6_text_ttf_empty_invalid_and_otto()
    print("[PASS] t2_fa6_text_ttf_empty_invalid_and_otto")
    passed += 1

    t2_fa6_text_line_break_zero_width_empty_and_tiny()
    print("[PASS] t2_fa6_text_line_break_zero_width_empty_and_tiny")
    passed += 1

    t2_fa6_text_rich_text_malformed_tags()
    print("[PASS] t2_fa6_text_rich_text_malformed_tags")
    passed += 1

    t2_fa6_dockspace_overflow_zero_width_and_bad_json()
    print("[PASS] t2_fa6_dockspace_overflow_zero_width_and_bad_json")
    passed += 1

    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")
