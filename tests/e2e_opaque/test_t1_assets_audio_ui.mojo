"""Tier 1 (Happy Path) Opaque-Box E2E Suite for FA5 (Assets & Codecs) & FA6 (Audio, Input, Text & UI).

Uses ONLY public Flint Engine v5 APIs (zero private `_` symbols from `flint.*`).
All filesystem writes are isolated to `/tmp/flint-e2e-opaque-t1-<pid>/` and cleaned up on exit.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, isinf, isnan, log10, sin
from std.testing import assert_equal, assert_false, assert_true

from flint.assets.cooker_v2 import cook_workspace
from flint.assets.flpk_v3 import (
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    SUB_ID_BVH8_COLLIDER,
    AssetMeta,
    AssetUuid,
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    read_flmeta,
    write_flmeta,
)
from flint.assets.glb_native import load_native_glb, parse_native_glb
from flint.assets.glb_writer import encode_native_glb, write_native_glb
from flint.assets.hdr_image import (
    decode_exr_fp16,
    decode_hdr_or_exr,
    decode_radiance_hdr,
    encode_exr_fp16,
    encode_radiance_hdr,
)
from flint.assets.image import encode_png, write_png
from flint.assets.jpeg import decode_image_auto, decode_jpeg, encode_jpeg_baseline
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes
from flint.assets.model import NativeMesh, NativeModel, NativeTexture
from flint.assets.server import (
    ASSET_STATE_LOADING,
    ASSET_STATE_READY,
    AssetServer,
    RawBytesAsset,
)
from flint.audio import (
    BUS_MUSIC,
    BUS_SFX,
    AudioServer,
    QoaAsset,
    QoaStreamCursor,
    decode_qoa,
    encode_qoa,
)
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.input import (
    ACTION_AXIS2D,
    ACTION_BOOL,
    MOD_CTRL,
    InputContext,
    InputMap,
)
from flint.platform.input import InputState, PAD_A, PAD_DOWN
from flint.platform.os import make_dirs, write_bytes
from flint.render.font import SdfFont
from flint.text.line_break import wrap
from flint.text.rich_text import layout_rich, parse_rich, plain_text
from flint.text.sdf_bake import DSF_MAGIC, bake_font, bake_glyph, dsf_bytes
from flint.text.ttf import TrueTypeFont
from flint.text.utf8 import REPLACEMENT, decode_utf8
from flint.ui.context import (
    UI_KEY_BACKSPACE,
    UI_KEY_LEFT,
    UI_KEY_TAB,
    UiContext,
)
from flint.ui.flexbox_dock import DockSpace, RectF

comptime TAU64: Float64 = 6.283185307179586
comptime SYSTEM_TTF_PATH: String = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def _tmp_dir(tag: String) -> String:
    var pid = Int(external_call["getpid", Int32]())
    return "/tmp/flint-e2e-opaque-t1-" + String(pid) + "-" + tag


def _cleanup_tmp(dir_path: String):
    if not dir_path.startswith("/tmp/flint-e2e-opaque-t1-"):
        return
    var cmd = "rm -rf " + dir_path
    _ = external_call["system", Int32](cmd.as_c_string_span())


def _make_sample_glb_model() -> NativeModel:
    var model = NativeModel()
    var tex = NativeTexture()
    tex.width = 16
    tex.height = 16
    tex.pixels = List[UInt32](length=256, fill=0xFFD0E0F0)
    for y in range(16):
        for x in range(16):
            if ((x ^ y) & 3) == 0:
                tex.pixels[y * 16 + x] = 0xFF336699
    model.textures.append(tex^)

    var mb0 = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.30, metallic=0.75, texture=0)
    mb0.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), 10, 14)
    var prims0 = List[NativeMesh]()
    prims0.append(mb0^.finish())
    var root_idx = append_node_meshes(model, "body_root", prims0)

    var mb1 = MeshBuilder(
        base_color=0xC0FFE080,
        roughness=0.15,
        metallic=0.10,
        emissive_color=Vec3(0.9, 0.6, 0.2),
        emissive_strength=0.5,
        alpha_mode=2,
    )
    mb1.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.4, 0.8), 8, 10)
    var prims1 = List[NativeMesh]()
    prims1.append(mb1^.finish())
    _ = append_node_meshes(model, "visor", prims1, translation=Vec3(0.0, 0.9, 0.4), parent=root_idx)
    return model^


def _make_minimal_wav_16bit() -> List[UInt8]:
    var n_samples = 64
    var data_bytes = n_samples * 2
    var riff_size = 36 + data_bytes
    var out = List[UInt8](capacity=44 + data_bytes)
    for b in "RIFF".as_bytes():
        out.append(b)
    for s in range(4):
        out.append(UInt8((riff_size >> (s * 8)) & 0xFF))
    for b in "WAVEfmt ".as_bytes():
        out.append(b)
    for b in [UInt8(16), 0, 0, 0, 1, 0, 1, 0]:
        out.append(b)
    var sr = 48000
    for s in range(4):
        out.append(UInt8((sr >> (s * 8)) & 0xFF))
    var br = sr * 2
    for s in range(4):
        out.append(UInt8((br >> (s * 8)) & 0xFF))
    for b in [UInt8(2), 0, 16, 0]:
        out.append(b)
    for b in "data".as_bytes():
        out.append(b)
    for s in range(4):
        out.append(UInt8((data_bytes >> (s * 8)) & 0xFF))
    for i in range(n_samples):
        var v = Int16(Int(sin(Float32(i) * Float32(0.25)) * Float32(12000.0)))
        var u = UInt16(v)
        out.append(UInt8(u & 0xFF))
        out.append(UInt8((u >> 8) & 0xFF))
    return out^


def _make_sine_pcm16(
    freq_hz: Float64,
    frames: Int,
    channels: Int = 1,
    amp: Float64 = 0.8,
    sample_rate: Int = 48000,
) -> List[Int16]:
    var out = List[Int16](capacity=frames * channels)
    for f in range(frames):
        var s = amp * sin(TAU64 * freq_hz * Float64(f) / Float64(sample_rate))
        var q = Int16(max(-32768, min(32767, Int(s * 32767.0))))
        for _ in range(channels):
            out.append(q)
    return out^


def _populate_sample_workspace(ws: String) raises:
    _ = make_dirs(ws + "/meshes")
    _ = make_dirs(ws + "/textures")
    _ = make_dirs(ws + "/env")
    _ = make_dirs(ws + "/audio")

    var mdl = _make_sample_glb_model()
    var glb_path = ws + "/meshes/box.glb"
    assert_true(write_native_glb(glb_path, mdl))
    var meta = AssetMeta()
    meta.uuid = AssetUuid.from_path("meshes/box.glb")
    meta.asset_kind = ASSET_KIND_MESH
    meta.build_lods = True
    meta.build_collider = True
    meta.source_path = "meshes/box.glb"
    assert_true(write_flmeta(glb_path + ".flmeta", meta))
    var meta_back = read_flmeta(glb_path + ".flmeta")
    assert_true(meta_back[0] and meta_back[1].uuid == meta.uuid)

    var px = List[UInt32](length=256 * 256, fill=0)
    for y in range(256):
        for x in range(256):
            px[y * 256 + x] = 0xFF000000 | (UInt32(x) << 16) | (UInt32(y) << 8) | UInt32((x ^ y) & 0xFF)
    assert_true(write_png(ws + "/textures/albedo_256.png", px, 256, 256))

    var hdr_px = List[Vec3](length=16 * 8, fill=Vec3(1.5, 0.75, 0.35))
    var hdr_bytes = encode_radiance_hdr(hdr_px, 16, 8)
    assert_true(write_bytes(ws + "/env/probe.hdr", hdr_bytes))

    var wav_bytes = _make_minimal_wav_16bit()
    assert_true(write_bytes(ws + "/audio/tone.wav", wav_bytes))


# ============================================================================
# FA5 Tier 1: Asset Pipeline, Codecs & VFS (7 Happy-Path Tests)
# ============================================================================


def t1_fa5_cook_workspace_and_cache_hit() raises:
    var root = _tmp_dir("cook")
    _cleanup_tmp(root)
    var pool = JobPool()
    _ = pool.start(2)
    try:
        var ws = root + "/ws"
        var cache_dir = root + "/cache"
        var pack_path = root + "/bundle.flpk"
        _populate_sample_workspace(ws)

        var rep1 = cook_workspace(ws, pack_path, pool, cache_dir=cache_dir)
        assert_true(rep1.ok)
        assert_equal(rep1.total_assets, 4)
        assert_equal(rep1.cooked_count, 4)
        assert_equal(rep1.cache_hits, 0)
        assert_equal(rep1.failed_count, 0)
        assert_true(rep1.archive_bytes > 64)

        var rep2 = cook_workspace(ws, pack_path, pool, cache_dir=cache_dir)
        assert_true(rep2.ok)
        assert_equal(rep2.total_assets, 4)
        assert_equal(rep2.cache_hits, 4)
        assert_equal(rep2.cooked_count, 0)
    except e:
        pool.stop()
        _cleanup_tmp(root)
        raise e^
    pool.stop()
    _cleanup_tmp(root)


def t1_fa5_flpk_v3_mmap_and_lookups() raises:
    var root = _tmp_dir("flpk")
    _cleanup_tmp(root)
    var pool = JobPool()
    _ = pool.start(2)
    try:
        var ws = root + "/ws"
        var cache_dir = root + "/cache"
        var pack_path = root + "/bundle.flpk"
        _populate_sample_workspace(ws)
        var rep = cook_workspace(ws, pack_path, pool, cache_dir=cache_dir)
        assert_true(rep.ok)

        var arch = FlpkV3MappedArchive.open(pack_path)
        assert_true(arch.valid)
        assert_true(arch.toc_count() >= 5)

        var u_mesh = AssetUuid.from_path("meshes/box.glb")
        var idx_uuid = arch.find_by_uuid(u_mesh, 0)
        var idx_path = arch.find_by_path("meshes/box.glb", 0)
        assert_true(idx_uuid >= 0)
        assert_equal(idx_uuid, idx_path)
        assert_true(arch.verify_entry_crc64(idx_uuid))
        assert_true(arch.find_by_uuid(u_mesh, SUB_ID_BVH8_COLLIDER) >= 0)

        var slice = arch.mmap_slice_by_uuid(u_mesh, 0)
        assert_true(slice.valid and slice.addr != 0 and slice.size > 0)

        var by_path = arch.read_by_path("meshes/box.glb", 0)
        var by_uuid = arch.read_by_uuid(u_mesh, 0)
        assert_true(by_path[0] and len(by_path[1]) == slice.size)
        assert_true(by_uuid[0] and len(by_uuid[1]) == slice.size)
        arch.close()
    except e:
        pool.stop()
        _cleanup_tmp(root)
        raise e^
    pool.stop()
    _cleanup_tmp(root)


def t1_fa5_asset_server_sync_and_progressive_mips() raises:
    var root = _tmp_dir("srv")
    _cleanup_tmp(root)
    var pool = JobPool()
    _ = pool.start(2)
    try:
        var ws = root + "/ws"
        var cache_dir = root + "/cache"
        var pack_path = root + "/bundle.flpk"
        var loose_dir = root + "/loose"
        _populate_sample_workspace(ws)
        _ = make_dirs(loose_dir)
        var raw_bytes = List[UInt8](length=128, fill=0x5C)
        assert_true(write_bytes(loose_dir + "/blob.bin", raw_bytes))

        var rep = cook_workspace(ws, pack_path, pool, cache_dir=cache_dir)
        assert_true(rep.ok)

        var server = AssetServer(stream_budget_bytes_per_tick=1024, enable_watcher=False)
        assert_true(server.mount_flpk(pack_path))
        _ = server.watch_loose_directory(loose_dir)

        var h_mesh = server.load_mesh("meshes/box.glb")
        assert_true(h_mesh.is_valid() and server.meshes.is_alive(h_mesh))
        var mp = server.meshes.borrow(h_mesh)
        assert_true(mp[unsafe_offset=0].has_collider)
        assert_true(mp[unsafe_offset=0].geom.meshes[0].lod_count >= 1)

        var h_tex = server.load_texture("textures/albedo_256.png", progressive=True)
        assert_true(h_tex.is_valid() and server.textures.is_alive(h_tex))
        assert_equal(server.textures.borrow(h_tex)[unsafe_offset=0].resident_min_mip, 4)
        for expected_mip in [3, 2, 1, 0]:
            _ = server.tick(0.0)
            var tp = server.textures.borrow(h_tex)
            assert_equal(tp[unsafe_offset=0].resident_min_mip, expected_mip)
            assert_true(len(tp[unsafe_offset=0].mips[expected_mip]) > 0)

        var h_hdr = server.load_hdr("env/probe.hdr")
        assert_true(h_hdr.is_valid() and server.hdrs.is_alive(h_hdr))

        var h_aud = server.load_audio("audio/tone.wav")
        assert_true(h_aud.is_valid() and server.audios.is_alive(h_aud))

        var h_raw = server.load_raw("blob.bin")
        assert_true(h_raw.is_valid() and server.raws.is_alive(h_raw))
        assert_equal(len(server.raws.borrow(h_raw)[unsafe_offset=0].bytes), 128)

        # Async load transition: ASSET_STATE_LOADING -> poll_async_loads -> ASSET_STATE_READY
        _ = server.meshes.unload(h_mesh)
        var h_async = server.load_mesh_async("meshes/box.glb")
        assert_equal(server.mesh_state(h_async), ASSET_STATE_LOADING)
        assert_true(server.pending_async_count() >= 1)
        var done = server.poll_async_loads(pool)
        assert_true(done >= 1)
        assert_equal(server.mesh_state(h_async), ASSET_STATE_READY)

        server.close()
    except e:
        pool.stop()
        _cleanup_tmp(root)
        raise e^
    pool.stop()
    _cleanup_tmp(root)


def t1_fa5_asset_server_lru_eviction() raises:
    var server = AssetServer(stream_budget_bytes_per_tick=1024, lru_max_bytes=65536, enable_watcher=False)
    var b1 = List[UInt8](length=32768, fill=0x11)
    var b2 = List[UInt8](length=32768, fill=0x22)
    var b3 = List[UInt8](length=32768, fill=0x33)

    var h1 = server.raws.insert(RawBytesAsset(b1^), AssetUuid.from_path("r1.bin"), "r1.bin", 1)
    _ = server.tick(1.0)
    var h2 = server.raws.insert(RawBytesAsset(b2^), AssetUuid.from_path("r2.bin"), "r2.bin", 2)
    _ = server.tick(2.0)
    var h3 = server.raws.insert(RawBytesAsset(b3^), AssetUuid.from_path("r3.bin"), "r3.bin", 3)

    assert_true(server.total_resident_bytes() == 98304)
    assert_true(server.raws.release(h1))
    assert_true(server.raws.release(h2))
    # h3 remains pinned (ref_count == 1)

    var evicted = server.enforce_lru_budget()
    assert_true(evicted >= 1)
    assert_false(server.raws.is_alive(h1))
    assert_true(server.raws.is_alive(h3))
    assert_true(server.total_resident_bytes() <= server.lru_max_bytes)
    server.close()


def t1_fa5_glb_native_roundtrip() raises:
    var root = _tmp_dir("glb")
    _cleanup_tmp(root)
    _ = make_dirs(root)
    try:
        var model = _make_sample_glb_model()
        var glb_bytes = encode_native_glb(model)
        assert_true(len(glb_bytes) > 256)

        var parsed = parse_native_glb(glb_bytes)
        assert_true(parsed.loaded())
        assert_equal(len(parsed.meshes), 2)
        assert_equal(len(parsed.nodes), 2)
        assert_equal(len(parsed.textures), 1)
        assert_equal(parsed.textures[0].width, 16)
        assert_equal(parsed.textures[0].height, 16)
        assert_true(abs(parsed.meshes[0].roughness - Float32(0.30)) < Float32(0.02))
        assert_true(abs(parsed.meshes[0].metallic - Float32(0.75)) < Float32(0.02))
        assert_equal(parsed.meshes[1].alpha_mode, 2)

        var glb_file = root + "/roundtrip.glb"
        assert_true(write_native_glb(glb_file, model))
        var from_disk = load_native_glb(glb_file)
        assert_true(from_disk.loaded())
        assert_equal(len(from_disk.meshes), 2)
        assert_equal(len(from_disk.nodes), 2)
    except e:
        _cleanup_tmp(root)
        raise e^
    _cleanup_tmp(root)


def t1_fa5_jpeg_baseline_and_auto_decode() raises:
    var w = 16
    var h = 16
    var px = List[UInt32](length=w * h, fill=0)
    for y in range(h):
        for x in range(w):
            var r = UInt32((x * 16) & 0xFF)
            var g = UInt32((y * 16) & 0xFF)
            var b = UInt32(140)
            px[y * w + x] = 0xFF000000 | (r << 16) | (g << 8) | b

    var jpg = encode_jpeg_baseline(px, w, h, quality=92)
    assert_true(len(jpg) > 32)
    var dec_j = decode_jpeg(jpg)
    assert_true(dec_j.valid)
    assert_equal(dec_j.width, w)
    assert_equal(dec_j.height, h)
    assert_equal(len(dec_j.pixels), w * h)

    var auto_j = decode_image_auto(jpg)
    assert_true(auto_j.valid)
    assert_equal(auto_j.width, w)
    assert_equal(auto_j.height, h)

    var png = encode_png(px, w, h, "", True)
    assert_true(len(png) > 32)
    var auto_p = decode_image_auto(png)
    assert_true(auto_p.valid)
    assert_equal(auto_p.width, w)
    assert_equal(auto_p.height, h)


def t1_fa5_hdr_and_exr_roundtrip() raises:
    var hdr_px = List[Vec3](length=16 * 8, fill=Vec3(2.0, 0.5, 4.0))
    var hdr_bytes = encode_radiance_hdr(hdr_px, 16, 8)
    assert_true(len(hdr_bytes) > 32)
    var dec_hdr = decode_radiance_hdr(hdr_bytes)
    assert_true(dec_hdr.valid)
    assert_equal(dec_hdr.width, 16)
    assert_equal(dec_hdr.height, 8)
    assert_true(abs(dec_hdr.pixels[0].x - Float32(2.0)) < Float32(0.08))
    assert_true(abs(dec_hdr.pixels[0].y - Float32(0.5)) < Float32(0.05))
    assert_true(abs(dec_hdr.pixels[0].z - Float32(4.0)) < Float32(0.15))

    var auto_hdr = decode_hdr_or_exr(hdr_bytes)
    assert_true(auto_hdr.valid and auto_hdr.width == 16 and auto_hdr.height == 8)

    var exr_px = List[Vec3](length=8 * 8, fill=Vec3(1.5, 0.25, 3.25))
    var exr_bytes = encode_exr_fp16(exr_px, 8, 8)
    assert_true(len(exr_bytes) > 64)
    var dec_exr = decode_exr_fp16(exr_bytes)
    assert_true(dec_exr.valid)
    assert_equal(dec_exr.width, 8)
    assert_equal(dec_exr.height, 8)
    assert_true(abs(dec_exr.pixels[0].x - Float32(1.5)) < Float32(0.02))
    assert_true(abs(dec_exr.pixels[0].y - Float32(0.25)) < Float32(0.01))
    assert_true(abs(dec_exr.pixels[0].z - Float32(3.25)) < Float32(0.02))

    var auto_exr = decode_hdr_or_exr(exr_bytes)
    assert_true(auto_exr.valid and auto_exr.width == 8 and auto_exr.height == 8)


# ============================================================================
# FA6 Tier 1: Audio, Input, Text & UI (7 Happy-Path Tests)
# ============================================================================


def t1_fa6_audio_server_2d_3d_lufs_offline() raises:
    var srv = AudioServer(48000, 256)
    srv.set_listener(Vec3(0.0, 0.0, 0.0), Vec3(), Vec3(0.0, 0.0, -1.0), Vec3(1.0, 0.0, 0.0))
    var pcm = _make_sine_pcm16(440.0, 24000, 2, 0.7, 48000)
    var aid = srv.register_pcm16_asset(pcm, 2, 48000)
    assert_true(aid >= 0)

    var h2d = srv.play_2d(aid, volume=0.6, bus_id=BUS_MUSIC, looping=True, direct_stereo=True)
    var h3d = srv.play_3d(aid, Vec3(4.0, 0.0, -6.0), volume=0.8, bus_id=BUS_SFX, looping=True)
    assert_true(h2d.is_valid() and h3d.is_valid())

    var out_lr = List[Float32](length=512, fill=0.0)
    var peak = Float32(0.0)
    for _ in range(16):
        srv.render_offline_quantum(out_lr, 256)
        for i in range(512):
            assert_false(isnan(out_lr[i]))
            assert_false(isinf(out_lr[i]))
            var a = abs(out_lr[i])
            if a > peak:
                peak = a

    assert_true(peak > Float32(0.05) and peak <= Float32(0.99))
    assert_equal(srv.quantums_rendered(), UInt64(16))
    assert_true(srv.momentary_lufs() > Float32(-40.0))
    assert_true(srv.integrated_lufs() > Float32(-40.0))


def t1_fa6_qoa_roundtrip_seek_and_server_asset() raises:
    var n_frames = 12000
    var sr = 48000
    var pcm = List[Int16](capacity=n_frames * 2)
    for f in range(n_frames):
        var t = Float64(f) / Float64(sr)
        var wl = 0.7 * sin(TAU64 * 220.0 * t) + 0.2 * sin(TAU64 * 440.0 * t)
        var wr = 0.7 * sin(TAU64 * 330.0 * t) + 0.2 * sin(TAU64 * 660.0 * t)
        pcm.append(Int16(Int(wl * 32767.0)))
        pcm.append(Int16(Int(wr * 32767.0)))

    var encoded = encode_qoa(pcm, 2, sr)
    assert_true(len(encoded) > 16)

    var decoded = decode_qoa(encoded)
    assert_true(decoded.valid)
    assert_equal(decoded.channels, 2)
    assert_equal(decoded.sample_rate, sr)
    assert_equal(decoded.frame_count(), n_frames)

    var sig_pow = Float64(0.0)
    var err_pow = Float64(0.0)
    for i in range(n_frames * 2):
        var s = Float64(pcm[i])
        var d = Float64(decoded.samples[i])
        sig_pow += s * s
        err_pow += (s - d) * (s - d)
    var snr_db = 10.0 * log10(sig_pow / max(1.0, err_pow))
    assert_true(snr_db >= 28.0)

    var qasset = QoaAsset.from_bytes(encoded.copy())
    assert_true(qasset.valid)
    assert_equal(qasset.channels, 2)
    assert_equal(qasset.sample_rate, sr)

    var cursor = QoaStreamCursor()
    assert_true(cursor.open(encoded))
    assert_true(cursor.seek(encoded, 5121))
    var check_buf = List[Int16](length=64 * 2, fill=0)
    var got = cursor.decode_frames_into(encoded, check_buf, 0, 64)
    assert_equal(got, 64)
    for k in range(64 * 2):
        assert_equal(check_buf[k], decoded.samples[5121 * 2 + k])

    var srv = AudioServer(48000, 256)
    var qaid = srv.register_qoa_asset(encoded.copy())
    assert_true(qaid >= 0)
    var h = srv.play_2d(qaid, volume=0.9, direct_stereo=True)
    assert_true(h.is_valid())
    var out_lr = List[Float32](length=512, fill=0.0)
    srv.render_offline_quantum(out_lr, 256)
    srv.render_offline_quantum(out_lr, 256)
    var pk = Float32(0.0)
    for s in out_lr:
        pk = max(pk, abs(s))
    assert_true(pk > Float32(0.05))


def t1_fa6_input_map_contexts_coyote_buffer_json() raises:
    var imap = InputMap()
    _ = imap.register_action("Jump", ACTION_BOOL)
    _ = imap.register_action("Move", ACTION_AXIS2D)
    _ = imap.register_action("MenuConfirm", ACTION_BOOL)
    _ = imap.register_action("Save", ACTION_BOOL)

    var gameplay = InputContext("Gameplay", priority=0, blocks_lower_all=False)
    gameplay.bind_key("Jump", 32)
    gameplay.bind_pad_button("Jump", PAD_A)
    gameplay.bind_key("Save", 83, MOD_CTRL)
    gameplay.bind_pad_stick_2d("Move", 0)
    imap.add_context(gameplay^)

    var menu = InputContext("Menu", priority=10, blocks_lower_all=False)
    menu.bind_key("MenuConfirm", 32, consume=True)
    imap.add_context(menu^)

    imap.push_context("Gameplay")
    var inp = InputState()
    inp.begin_frame()
    inp.set_logical_key(32, True)
    imap.update(inp, 0.016)
    assert_true(imap.just_pressed("Jump"))
    assert_true(imap.is_down("Jump"))

    # Higher-priority Menu context consumes Space
    imap.push_context("Menu")
    inp.set_logical_key(32, False)
    inp.begin_frame()
    imap.update(inp, 0.016)
    inp.begin_frame()
    inp.set_logical_key(32, True)
    imap.update(inp, 0.016)
    assert_true(imap.just_pressed("MenuConfirm"))
    assert_false(imap.is_down("Jump"))
    assert_true(imap.pop_context("Menu"))
    inp.set_logical_key(32, False)

    # 120 ms jump buffer
    imap.set_grounded(False)
    inp.begin_frame()
    inp.set_logical_key(32, True)
    imap.update(inp, 0.016)
    inp.set_logical_key(32, False)
    for _ in range(5):
        inp.begin_frame()
        imap.update(inp, 0.016)
    assert_true(imap.consume_buffered("Jump", 0.120))
    assert_false(imap.consume_buffered("Jump", 0.120))

    # 100 ms coyote time
    imap.set_grounded(True)
    inp.begin_frame()
    imap.update(inp, 0.016)
    imap.set_grounded(False)
    for _ in range(4):
        inp.begin_frame()
        imap.update(inp, 0.016)
    assert_true(imap.coyote_valid(0.100))
    imap.consume_coyote()
    assert_false(imap.coyote_valid(0.100))

    # Rebind + JSON round-trip
    assert_true(imap.rebind_key("Gameplay", "Jump", 74))
    var json_str = imap.save_json()
    var imap2 = InputMap()
    assert_true(imap2.load_json(json_str))
    inp.begin_frame()
    inp.set_logical_key(74, True)
    imap2.update(inp, 0.016)
    assert_true(imap2.just_pressed("Jump"))


def t1_fa6_text_ttf_and_sdf_bake() raises:
    var font = TrueTypeFont()
    assert_true(font.load(SYSTEM_TTF_PATH))
    assert_true(font.valid)
    assert_true(font.has_glyph(65))
    var gid = font.glyph_id(65)
    assert_true(gid > 0)
    assert_true(font.advance(gid) > Float32(0.0))

    var baked_a = bake_glyph(font, 65, 32, 4)
    assert_true(baked_a.w > 0 and baked_a.h > 0)
    assert_equal(len(baked_a.pixels), baked_a.w * baked_a.h)

    var codes = List[Int]()
    for c in range(32, 127):
        codes.append(c)
    var baked_atlas = bake_font(font, codes, 32, 4, atlas_width=512)
    assert_true(baked_atlas.atlas_width == 512 and baked_atlas.atlas_height >= 64)
    var dsf = dsf_bytes(baked_atlas)
    assert_true(len(dsf) > 44)
    var magic = Int(dsf[0]) | (Int(dsf[1]) << 8) | (Int(dsf[2]) << 16) | (Int(dsf[3]) << 24)
    assert_equal(magic, DSF_MAGIC)

    var sdf = SdfFont()
    assert_true(sdf.load_bytes(dsf))
    assert_true(sdf.valid)
    assert_true(sdf.measure("ABC", 24.0) > Float32(0.0))


def t1_fa6_text_line_break_and_rich_text() raises:
    var sdf = SdfFont()
    assert_true(sdf.load_ttf(SYSTEM_TTF_PATH, 32, 4))

    var wrapped = wrap(sdf, decode_utf8("Hello world from Flint"), 20.0, 90.0)
    assert_true(len(wrapped) >= 2)
    for line in wrapped:
        assert_true(line.width <= Float32(90.01))

    var forced = wrap(sdf, decode_utf8("LineOne\nLineTwo"), 20.0, 500.0)
    assert_equal(len(forced), 2)

    var spans = parse_rich("[c=FF0000]Red[/c] [s=150]Big[/s] [btn=jump]")
    assert_equal(len(spans), 5)
    var plain = plain_text(spans)
    assert_true(plain.find("Red") >= 0 and plain.find("Big") >= 0)

    var layout = layout_rich(spans, sdf, sdf, 20.0, 0, 220.0, 0, False, False)
    assert_true(layout.lines >= 1)
    assert_true(len(layout.runs) >= 3)
    assert_true(layout.width > Float32(0.0) and layout.height > Float32(0.0))


def t1_fa6_ui_context_and_dockspace_5panel() raises:
    var dock = DockSpace.create_studio_5panel()
    var screen = RectF(0.0, 0.0, 1280.0, 720.0)
    dock.solve_layout(screen)
    var leaves = dock.leaf_rects()
    assert_equal(len(leaves), 5)

    var area_sum = Float32(0.0)
    for i in range(len(leaves)):
        area_sum += leaves[i].area()
        for j in range(i + 1, len(leaves)):
            assert_false(leaves[i].overlaps_interior(leaves[j], Float32(0.01)))
    assert_true(abs(area_sum - screen.area()) < Float32(0.5))

    var dock_json = dock.save_json()
    var dock2 = DockSpace()
    assert_true(dock2.load_json(dock_json))
    dock2.solve_layout(screen)
    assert_equal(len(dock2.leaf_rects()), 5)

    var ui = UiContext()
    var inp = InputState()
    ui.begin_frame(inp, 1280.0, 720.0)
    ui.update_dock_space(dock, screen)
    if ui.begin_dock_panel(dock, "Outliner"):
        ui.label("Scene Hierarchy")
        _ = ui.button("add_ent", "Add Entity")
        ui.end_dock_panel()
    ui.end_frame()
    assert_true(ui.draw_list.count() > 0)


def t1_fa6_ui_utf8_text_input_and_modal() raises:
    var ui = UiContext()
    var inp = InputState()
    var text = String("")

    inp.begin_frame()
    inp.move_mouse(50.0, 15.0)
    inp.set_mouse(0, True)
    inp.push_text_utf8("¡Hola! エンジン ★")
    ui.begin_frame(inp, 800.0, 600.0)
    var changed = ui.text_input("utf_box", text, 300.0)
    ui.end_frame()
    assert_true(changed)
    assert_equal(text, "¡Hola! エンジン ★")
    assert_equal(ui.text_cursor_codepoint("utf_box"), 13)

    inp.set_mouse(0, False)
    for _ in range(4):
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_LEFT, True)
        ui.begin_frame(inp, 800.0, 600.0)
        _ = ui.text_input("utf_box", text, 300.0)
        ui.end_frame()
        inp.set_logical_key(UI_KEY_LEFT, False)
        var boff = ui.text_cursor_byte_offset("utf_box", text)
        if boff < text.byte_length():
            assert_true((Int(text.as_bytes()[boff]) & 0xC0) != 0x80)

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_BACKSPACE, True)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_true(ui.text_input("utf_box", text, 300.0))
    ui.end_frame()
    inp.set_logical_key(UI_KEY_BACKSPACE, False)

    var codes = decode_utf8(text)
    assert_equal(len(codes), 12)
    for c in codes:
        assert_true(c != REPLACEMENT)
    assert_equal(text, "¡Hola! エジン ★")

    # Modal focus trap & background button click blocking
    var modal_open = True
    var bg_id: UInt64
    for step in range(4):
        inp.begin_frame()
        if step == 2:
            inp.move_mouse(50.0, 15.0)
            inp.set_mouse(0, True)
        else:
            inp.set_mouse(0, False)
            if step % 2 == 1:
                inp.set_logical_key(UI_KEY_TAB, True)
            else:
                inp.pads[0].set_control(PAD_DOWN, True)
        ui.begin_frame(inp, 1280.0, 720.0)
        bg_id = ui.widget_id("bg_btn")
        assert_false(ui.button("bg_btn", "Background"))
        if ui.begin_modal("confirm_dlg", "Confirm", 320.0, 180.0, modal_open):
            _ = ui.button("modal_ok", "OK")
            _ = ui.button("modal_cancel", "Cancel")
            ui.end_modal()
        ui.end_frame()
        inp.set_logical_key(UI_KEY_TAB, False)
        inp.pads[0].set_control(PAD_DOWN, False)
        assert_true(ui.focused_id != 0 and ui.focused_id != bg_id)


def main() raises:
    var passed = 0
    var total = 14

    t1_fa5_cook_workspace_and_cache_hit()
    print("[PASS] t1_fa5_cook_workspace_and_cache_hit")
    passed += 1

    t1_fa5_flpk_v3_mmap_and_lookups()
    print("[PASS] t1_fa5_flpk_v3_mmap_and_lookups")
    passed += 1

    t1_fa5_asset_server_sync_and_progressive_mips()
    print("[PASS] t1_fa5_asset_server_sync_and_progressive_mips")
    passed += 1

    t1_fa5_asset_server_lru_eviction()
    print("[PASS] t1_fa5_asset_server_lru_eviction")
    passed += 1

    t1_fa5_glb_native_roundtrip()
    print("[PASS] t1_fa5_glb_native_roundtrip")
    passed += 1

    t1_fa5_jpeg_baseline_and_auto_decode()
    print("[PASS] t1_fa5_jpeg_baseline_and_auto_decode")
    passed += 1

    t1_fa5_hdr_and_exr_roundtrip()
    print("[PASS] t1_fa5_hdr_and_exr_roundtrip")
    passed += 1

    t1_fa6_audio_server_2d_3d_lufs_offline()
    print("[PASS] t1_fa6_audio_server_2d_3d_lufs_offline")
    passed += 1

    t1_fa6_qoa_roundtrip_seek_and_server_asset()
    print("[PASS] t1_fa6_qoa_roundtrip_seek_and_server_asset")
    passed += 1

    t1_fa6_input_map_contexts_coyote_buffer_json()
    print("[PASS] t1_fa6_input_map_contexts_coyote_buffer_json")
    passed += 1

    t1_fa6_text_ttf_and_sdf_bake()
    print("[PASS] t1_fa6_text_ttf_and_sdf_bake")
    passed += 1

    t1_fa6_text_line_break_and_rich_text()
    print("[PASS] t1_fa6_text_line_break_and_rich_text")
    passed += 1

    t1_fa6_ui_context_and_dockspace_5panel()
    print("[PASS] t1_fa6_ui_context_and_dockspace_5panel")
    passed += 1

    t1_fa6_ui_utf8_text_input_and_modal()
    print("[PASS] t1_fa6_ui_utf8_text_input_and_modal")
    passed += 1

    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")
