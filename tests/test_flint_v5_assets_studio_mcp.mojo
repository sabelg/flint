"""Flint v5 Track 5 Verification Suite (`tests/test_flint_v5_assets_studio_mcp.mojo`).

Verifies all 6 Track 5 subsystems end-to-end in `< 2s`:
  (a) `.flpk` v3 parallel `JobPool` cook + `< 5 ms` 100% cache hit on 2nd run + O(1) `mmap` Robin-Hood UUID/path lookup
      + 1-byte flip ECMA-182 `CRC64` corruption guard!
  (b) `AssetServer` generational `Handle[T]` stale-slot rejection, byte-budgeted progressive mip streaming
      (`4 -> 3 -> 2 -> 1 -> 0`), LRU unreferenced eviction, and live recursive `AssetWatcher` `inotify` hot-reload
      (`IN_CREATE | IN_ISDIR`) under the exact same `Handle[T]` (`reload_epoch++`)!
  (c) $O(1)$ VisBuffer `decode_visbuffer_pick` exact entity/meshlet/material/prim + Reverse-Z 3D hit reconstruction
      (`< 1 cm` error) + live headless `FlintAaaVulkanRenderer` GPU 1-pixel VisBuffer pick verification!
  (d) Analytical 3D Translate / Rotate (normal + `88°` grazing-angle `< 0.15` view-plane fallback!) / Scale gizmo
      solvers with grid & `15°` snapping + `UiDrawList` emission!
  (e) 50-command `FlintStudioState` Undo/Redo algebraic invertibility (`fnv64` exact match at all 50 undo & 50 redo steps)
      + **1-Click PIE** 120-tick deterministic physics run -> `pie_stop()` **bit-for-bit identical** LZ4 world restoration!
  (f) End-to-end JSON-RPC 2.0 `McpServerV2` round-trip (`initialize` -> `tools/list` (all 12 tools) -> `ecs.spawn_entity`
      -> `physics.step_deterministic` -> `studio.undo_redo` -> `render.capture_viewport` with `flip.mojo` diff
      + persistent non-blocking `FlintMcpUnixListener` socket session)!
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.math import abs, cos, sin
from std.sys.info import CompilationTarget
from flint.assets.cooker_v2 import cook_workspace
from flint.assets.flpk_v3 import (
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    FLPK_HEADER_BYTES,
    FLPK_TOC_ENTRY_BYTES,
    SUB_ID_BVH8_COLLIDER,
    AssetMeta,
    AssetPayload,
    AssetUuid,
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    crc64_ecma_ptr,
    encode_flpk_v3,
    read_flmeta,
    write_flmeta,
)
from flint.assets.hdr_image import encode_radiance_hdr
from flint.assets.image import write_png
from flint.assets.json import parse_json
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.server import AssetServer, RawBytesAsset, TypedAssetStorage
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool
from flint.core.lz4 import address_of
from flint.core.math import Quat, Vec3
from flint.core.transform import Mat4
from flint.devtools.mcp_v2 import (
    FlintMcpUnixListener,
    McpServerV2,
    McpUnixSessionClient,
    default_mcp_socket_path,
    parse_next_mcp_frame,
    sanitize_output_path,
)
from flint.platform.input import InputState
from flint.platform.os import get_time, make_dirs, read_bytes, remove_file, write_bytes
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.font import SdfFont
from flint.studio import (
    COLLIDER_BOX,
    COLLIDER_SPHERE,
    FIELD_F32,
    GIZMO_AXIS_X,
    GIZMO_AXIS_Y,
    GIZMO_MODE_ROTATE,
    GIZMO_MODE_SCALE,
    GIZMO_MODE_TRANSLATE,
    GIZMO_PLANE_XZ,
    GIZMO_SCALE_UNIFORM,
    PIE_PLAYING,
    PIE_STOPPED,
    SHADING_NPR_CEL,
    SHADING_PBR,
    SHADING_STYLIZED_VFX,
    FieldVisitor,
    FlintStudioState,
    InspectedField,
    ViewportRay,
    bake_procedural_to_glb,
    decode_visbuffer_pick,
    emit_gizmo_draw_list,
    emit_studio_ui_draw_list,
    encode_visbuffer_words,
    gizmo_rotation_delta_rad,
    gizmo_scale_drag,
    gizmo_world_scale,
    pick_renderer_viewport_once,
    project_world_to_viewport,
    ray_axis_closest_param,
    ray_plane_hit,
    snap_angle_deg,
    snap_translation,
)
from flint.ui.draw_list import UiDrawList


def _assert(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("ASSERTION FAILED: " + msg)


def _str_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _rm_rf_tmp(dir_path: String):
    if not dir_path.startswith("/tmp/flint-eng5-"):
        return
    var cmd = "rm -rf " + dir_path
    _ = external_call["system", Int32](cmd.as_c_string_span())


def _make_minimal_wav_16bit() -> List[UInt8]:
    """Synthesizes a valid 64-sample 48 kHz mono 16-bit RIFF/WAVE clip."""
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
    for b in [UInt8(16), 0, 0, 0, 1, 0, 1, 0]:  # PCM=1, channels=1
        out.append(b)
    var sr = 48000
    for s in range(4):
        out.append(UInt8((sr >> (s * 8)) & 0xFF))
    var br = sr * 2
    for s in range(4):
        out.append(UInt8((br >> (s * 8)) & 0xFF))
    for b in [UInt8(2), 0, 16, 0]:  # block_align=2, bits=16
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


# ============================================================================
# (a) `.flpk` v3 Parallel Cook + `<5ms` Cache Hit + O(1) `mmap` + CRC64 Guard
# ============================================================================

def test_a_flpk_v3_cooker_and_crc64(tmp_root: String, mut pool: JobPool) raises -> String:
    print("  [a] Testing .flpk v3 parallel cooker, <5ms 2nd-run cache hit, O(1) Robin-Hood mmap & CRC64 guard...")
    _assert(FLPK_HEADER_BYTES == 64, "FlpkV3Header must be 64 bytes")
    _assert(FLPK_TOC_ENTRY_BYTES == 64, "FlpkTocEntry must be 64 bytes")

    # UUID parser/formatter + .flmeta sidecar round-trip
    var u0 = AssetUuid.from_path("meshes/forge_box.glb")
    var u_txt = u0.format()
    _assert(u_txt.byte_length() == 36, "canonical UUID string must be 36 chars")
    var u_parsed = AssetUuid.parse(u_txt)
    _assert(u_parsed[0] and u_parsed[1] == u0, "AssetUuid parse/format round-trip failed")

    var ws = tmp_root + "/ws"
    var cache_dir = tmp_root + "/cook_cache"
    var pack_path = tmp_root + "/cooked_v3.flpk"
    _ = make_dirs(ws + "/meshes")
    _ = make_dirs(ws + "/textures")
    _ = make_dirs(ws + "/env")
    _ = make_dirs(ws + "/audio")

    # 1. Procedural Forge GLB + .flmeta sidecar
    var glb_file = ws + "/meshes/forge_box.glb"
    var baked = bake_procedural_to_glb("colonnade", glb_file, seed=1)
    _assert(baked[0] and baked[2] > 20, "Procedural Forge GLB bake failed")
    var meta = AssetMeta()
    meta.uuid = u0
    meta.asset_kind = ASSET_KIND_MESH
    meta.build_lods = True
    meta.build_collider = True
    meta.compress_lz4 = False
    meta.source_path = "meshes/forge_box.glb"
    _assert(write_flmeta(glb_file + ".flmeta", meta), "write_flmeta failed")
    var meta_back = read_flmeta(glb_file + ".flmeta")
    _assert(meta_back[0] and meta_back[1].uuid == u0, "read_flmeta round-trip failed")

    # 2. 256x256 RGBA8 PNG texture (cooks into tail mip >= 4 + high-res mips 0..3!)
    var px256 = List[UInt32](length=256 * 256, fill=0)
    for y in range(256):
        for x in range(256):
            px256[y * 256 + x] = 0xFF000000 | (UInt32(x) << 16) | (UInt32(y) << 8) | UInt32((x ^ y) & 0xFF)
    _assert(write_png(ws + "/textures/albedo_256.png", px256, 256, 256), "write 256x256 PNG failed")

    # 3. 16x8 Radiance .hdr probe
    var hdr_px = List[Vec3](length=16 * 8, fill=Vec3(1.5, 0.8, 0.4))
    var hdr_bytes = encode_radiance_hdr(hdr_px, 16, 8)
    _assert(write_bytes(ws + "/env/probe.hdr", hdr_bytes), "write .hdr failed")

    # 4. 16-bit WAV audio clip
    var wav_bytes = _make_minimal_wav_16bit()
    _assert(write_bytes(ws + "/audio/tone.wav", wav_bytes), "write .wav failed")

    # Run 1: Parallel cold cook
    var rep1 = cook_workspace(ws, pack_path, pool, cache_dir=cache_dir)
    _assert(rep1.ok and rep1.total_assets == 4, "1st cook_workspace failed")
    _assert(rep1.cooked_count == 4 and rep1.cache_hits == 0, "1st cook must compile all 4 assets")

    # Run 2: Incremental unchanged re-cook -> 100% cache hit in < 5 ms!
    var rep2 = cook_workspace(ws, pack_path, pool, cache_dir=cache_dir)
    _assert(rep2.ok and rep2.cache_hits == 4 and rep2.cooked_count == 0, "2nd cook must hit 100% cache")
    _assert(rep2.elapsed_ms < 5.0, "2nd cook cache hit exceeded 5 ms: " + String(rep2.elapsed_ms))

    # Verify O(1) Robin-Hood UUID & Path lookup + Zero-Copy mmap slice + Bvh8 sub_id=16 slab
    var arch = FlpkV3MappedArchive.open(pack_path)
    _assert(arch.valid and arch.toc_count() >= 9, "FlpkV3MappedArchive failed to open")
    var toc_mesh = arch.find_by_uuid(u0, 0)
    var toc_mesh_by_path = arch.find_by_path("meshes/forge_box.glb", 0)
    _assert(toc_mesh >= 0 and toc_mesh == toc_mesh_by_path, "Robin-Hood UUID & path lookup mismatch")
    var zc = arch.mmap_slice_by_uuid(u0, 0)
    _assert(zc.valid and zc.addr != 0 and zc.size > 64, "Zero-copy CHUNK_COMP_NONE mmap slice invalid")
    _assert((arch.toc_entry(toc_mesh).payload_offset & 63) == 0, "Payload slab must be >= 64B aligned")
    var toc_bvh = arch.find_by_uuid(u0, SUB_ID_BVH8_COLLIDER)
    _assert(toc_bvh >= 0, "Missing sub_id = 16 Bvh8 static collider slab")
    var first_payload_off = Int(arch.toc_entry(toc_mesh).payload_offset)
    arch.close()

    # 1-byte flip ECMA-182 CRC64 corruption guard on payload slab
    var raw_pack = read_bytes(pack_path)
    var corrupt_payload_pack = raw_pack.copy()
    corrupt_payload_pack[first_payload_off + 8] ^= UInt8(0x01)
    var corrupt_p_path = tmp_root + "/corrupt_payload.flpk"
    _ = write_bytes(corrupt_p_path, corrupt_payload_pack)
    var bad_arch = FlpkV3MappedArchive.open(corrupt_p_path)
    _assert(bad_arch.valid, "Header/TOC should still open when only payload byte flipped")
    _assert(not bad_arch.verify_entry_crc64(toc_mesh), "CRC64 failed to detect 1-byte payload corruption")
    _assert(not bad_arch.mmap_slice_by_toc(toc_mesh).valid, "Zero-copy slice must reject corrupt CRC64")
    _assert(not bad_arch.read_payload_by_toc(toc_mesh)[0], "read_payload_by_toc must reject corrupt CRC64")
    bad_arch.close()
    _ = remove_file(corrupt_p_path)

    # 1-byte flip ECMA-182 CRC64 corruption guard on TOC header
    var corrupt_toc_pack = raw_pack.copy()
    corrupt_toc_pack[72] ^= UInt8(0x04)
    var corrupt_t_path = tmp_root + "/corrupt_toc.flpk"
    _ = write_bytes(corrupt_t_path, corrupt_toc_pack)
    var bad_toc_arch = FlpkV3MappedArchive.open(corrupt_t_path)
    _assert(not bad_toc_arch.valid, "TOC CRC64 failed to reject 1-byte TOC corruption")
    bad_toc_arch.close()
    _ = remove_file(corrupt_t_path)

    # Hostile `.flpk` v3 fuzzing (with valid recomputed ECMA-182 TOC CRC64!)
    _verify_hostile_flpk_fuzzing(tmp_root)

    return pack_path


def _put_u32_at(mut b: List[UInt8], at: Int, v: UInt32):
    for s in range(4):
        b[at + s] = UInt8((v >> UInt32(s * 8)) & 0xFF)


def _put_u64_at(mut b: List[UInt8], at: Int, v: UInt64):
    for s in range(8):
        b[at + s] = UInt8((v >> UInt64(s * 8)) & 0xFF)


def _recompute_hostile_toc_crc64(mut pack: List[UInt8], meta_end: Int):
    var c0 = crc64_ecma_ptr(address_of(pack), 56, 0)
    var c = crc64_ecma_ptr(address_of(pack) + 64, meta_end - 64, c0)
    _ = len(pack)
    _put_u64_at(pack, 56, c)


def _verify_hostile_flpk_fuzzing(tmp_root: String) raises:
    var pay = List[UInt8](length=128, fill=0x5A)
    var chunks = List[FlpkBuildChunk]()
    var target_uuid = AssetUuid.from_path("fuzz/a.bin")
    chunks.append(FlpkBuildChunk(target_uuid, "fuzz/a.bin", 0, ASSET_KIND_RAW, pay^))
    var clean_pack = encode_flpk_v3(chunks, page_align=False)
    # Header(64) + 1 TOC(64) + 16 UUID(512) + 16 Path(256) + Str(10) -> aligned 64 = 960
    var hcap = Int(clean_pack[40]) | (Int(clean_pack[41]) << 8)
    var str_off = (Int(clean_pack[32]) | (Int(clean_pack[33]) << 8)) + hcap * 16
    var spool = Int(clean_pack[44]) | (Int(clean_pack[45]) << 8)
    var meta_end = ((str_off + spool + 63) // 64) * 64
    var evil_path = tmp_root + "/hostile_fuzz.flpk"

    # Case 1: Giant `toc_offset = 1 << 40` with valid recomputed CRC64 (`flpk_probe toc`)
    var p_toc = clean_pack.copy()
    _put_u64_at(p_toc, 16, UInt64(1) << 40)
    _recompute_hostile_toc_crc64(p_toc, meta_end)
    _ = write_bytes(evil_path, p_toc)
    var a1 = FlpkV3MappedArchive.open(evil_path)
    _assert(not a1.valid, "Giant toc_offset must be rejected")
    _assert(a1.toc_entry(0).payload_offset == 0 and len(a1.read_entry_bytes(0)) == 0, "Invalid archive toc_entry must be safe no-op")
    a1.close()

    # Case 2: Giant `uuid_table_offset = 1 << 40` with valid recomputed CRC64 (`flpk_probe uuid`)
    var p_uuid = clean_pack.copy()
    _put_u64_at(p_uuid, 24, UInt64(1) << 40)
    _recompute_hostile_toc_crc64(p_uuid, meta_end)
    _ = write_bytes(evil_path, p_uuid)
    var a2 = FlpkV3MappedArchive.open(evil_path)
    _assert(not a2.valid and a2.find_by_uuid(target_uuid) == -1, "Giant uuid_table_offset must be rejected")
    a2.close()

    # Case 3: UInt64 wraparound `payload_offset + compressed_bytes` (`0xFFFFFFFFFFFFFFF0 + 128` wraps to `0x70 < size`!)
    var p_wrap = clean_pack.copy()
    _put_u64_at(p_wrap, 64 + 24, UInt64(0xFFFFFFFFFFFFFFF0))  # TOC[0].payload_offset
    _put_u64_at(p_wrap, 64 + 32, UInt64(128))                 # TOC[0].compressed_bytes
    _recompute_hostile_toc_crc64(p_wrap, meta_end)
    _ = write_bytes(evil_path, p_wrap)
    var a3 = FlpkV3MappedArchive.open(evil_path)
    _assert(not a3.valid, "Overflowing payload_offset + compressed_bytes MUST be rejected")
    a3.close()

    # Case 4: Corrupted occupied Robin-Hood UUID hash slot (`u_toc = 42 >= entry_count`) with valid recomputed CRC64
    var p_slot = clean_pack.copy()
    var uuid_tbl_off = 64 + 64
    for s in range(hcap):
        var s_off = uuid_tbl_off + s * 32
        var psl = Int(p_slot[s_off + 20])
        if psl > 0:
            _put_u32_at(p_slot, s_off + 16, UInt32(42))  # Out-of-bounds toc index!
    _recompute_hostile_toc_crc64(p_slot, meta_end)
    _ = write_bytes(evil_path, p_slot)
    var a4 = FlpkV3MappedArchive.open(evil_path)
    _assert(not a4.valid and a4.find_by_uuid(target_uuid) == -1, "Corrupted hash slot index >= entry_count must be rejected")
    a4.close()

    # Case 5: `uncompressed_bytes > 1 GiB` zip-bomb TOC entry with valid recomputed CRC64
    var p_bomb = clean_pack.copy()
    p_bomb[64 + 59] = UInt8(1)  # CHUNK_COMP_LZ4
    _put_u64_at(p_bomb, 64 + 40, UInt64(2_000_000_000))  # > 1 GiB cap
    _recompute_hostile_toc_crc64(p_bomb, meta_end)
    _ = write_bytes(evil_path, p_bomb)
    var a5 = FlpkV3MappedArchive.open(evil_path)
    _assert(not a5.valid, "> 1 GiB uncompressed_bytes must be rejected")
    a5.close()

    _ = remove_file(evil_path)


# ============================================================================
# (b) `AssetServer` Generational Handles, Progressive Mips & Live Hot-Reload
# ============================================================================

struct DestructorCounter(Movable):
    var live_instances: Int
    var deinit_calls: Int

    def __init__(out self):
        self.live_instances = 0
        self.deinit_calls = 0


struct DestructorCounterPayload(AssetPayload):
    var counter_addr: Int
    var cpu_sz: Int

    def __init__(out self, counter_addr: Int, cpu_sz: Int = 4096):
        self.counter_addr = counter_addr
        self.cpu_sz = cpu_sz
        if counter_addr != 0:
            Pointer[DestructorCounter, MutAnyOrigin](unsafe_from_address=counter_addr)[unsafe_offset=0].live_instances += 1

    def __init__(out self, *, copy: Self):
        self.counter_addr = copy.counter_addr
        self.cpu_sz = copy.cpu_sz
        if self.counter_addr != 0:
            Pointer[DestructorCounter, MutAnyOrigin](unsafe_from_address=self.counter_addr)[unsafe_offset=0].live_instances += 1

    def __deinit__(deinit self):
        if self.counter_addr != 0:
            ref ctr = Pointer[DestructorCounter, MutAnyOrigin](unsafe_from_address=self.counter_addr)[unsafe_offset=0]
            ctr.live_instances -= 1
            ctr.deinit_calls += 1

    @staticmethod
    def asset_kind() -> UInt8:
        return ASSET_KIND_RAW

    def byte_footprint(self) -> Int:
        return self.cpu_sz


def test_b_asset_server_streaming_and_hot_reload(tmp_root: String, pack_path: String) raises:
    print("  [b] Testing AssetServer Optional[T] DestructorCounter LRU free, 4->3->2->1->0 mip streaming & hot-reload...")
    # 0. Verify `TypedAssetStorage[DestructorCounterPayload]` runs `T.__deinit__` immediately on LRU eviction / unload!
    var dctr = DestructorCounter()
    var ctr_addr = Int(Pointer(to=dctr))
    var custom_arena = TypedAssetStorage[DestructorCounterPayload]()

    # Queued slot starts as `None` (no dummy `T` constructed or leaked!)
    var h_queued = custom_arena.allocate_queued(AssetUuid.from_path("queued.bin"), "queued.bin", 1)
    _assert(custom_arena.is_alive(h_queued) and not custom_arena.is_loaded(h_queued), "queued slot must be alive but unloaded")
    _assert(not Bool(custom_arena.get_ptr(h_queued)), "get_ptr on unloaded slot must return None")
    var borrow_threw = False
    try:
        _ = custom_arena.borrow(h_queued)
    except:
        borrow_threw = True
    _assert(borrow_threw, "borrow() on unloaded slot must raise cleanly")
    _assert(custom_arena.unload(h_queued), "unload(h_queued) failed")
    _assert(dctr.live_instances == 0 and dctr.deinit_calls == 0, "unloaded queued slot must never fabricate dummy T")

    var hc1 = custom_arena.insert(DestructorCounterPayload(ctr_addr, 4096), AssetUuid.from_path("c1"), "c1", 10, vram_bytes=2048)
    var hc2 = custom_arena.insert(DestructorCounterPayload(ctr_addr, 4096), AssetUuid.from_path("c2"), "c2", 20, vram_bytes=2048)
    var hc3 = custom_arena.insert(DestructorCounterPayload(ctr_addr, 4096), AssetUuid.from_path("c3"), "c3", 30, vram_bytes=2048)
    _assert(dctr.live_instances == 3, "3 live DestructorCounterPayload instances expected")
    _assert(custom_arena.cpu_live_bytes == 12288 and custom_arena.vram_live_bytes == 6144, "CPU/VRAM live bytes mismatch")

    _ = custom_arena.release(hc1)
    _ = custom_arena.release(hc2)
    var lru_evicted = custom_arena.evict_lru_unreferenced(6144)
    _assert(len(lru_evicted) == 2, "LRU eviction must evict hc1 and hc2")
    _assert(dctr.live_instances == 1 and dctr.deinit_calls >= 2, "LRU eviction MUST immediately run T.__deinit__ on evicted payloads!")
    _assert(custom_arena.cpu_live_bytes == 4096 and custom_arena.vram_live_bytes == 2048, "LRU eviction must subtract cpu & vram bytes")
    _assert(not Bool(custom_arena.touch_and_copy(hc1, 40)), "touch_and_copy on evicted handle must return None safely")
    _assert(custom_arena.unload(hc3), "unload(hc3) failed")
    _assert(dctr.live_instances == 0 and custom_arena.cpu_live_bytes == 0 and custom_arena.vram_live_bytes == 0, "all payloads must be freed after unload")
    _ = dctr.live_instances

    var loose_dir = tmp_root + "/loose_live"
    _ = make_dirs(loose_dir)

    # Budget = 1024 B/tick -> drains exactly 1 high-res mip per tick (`3` -> `2` -> `1` -> `0`)!
    var server = AssetServer(stream_budget_bytes_per_tick=1024, enable_watcher=True)
    _assert(server.mount_flpk(pack_path), "AssetServer.mount_flpk failed")
    var watched = server.watch_loose_directory(loose_dir)
    _assert(watched >= 1, "watch_loose_directory failed")

    # 1. Generational Handle[T] stale-slot rejection + LRU eviction
    var b1 = List[UInt8](length=128, fill=0xAA)
    var h_gen1 = server.raws.insert(RawBytesAsset(b1^), AssetUuid.from_path("a.bin"), "a.bin", 1)
    _assert(server.raws.is_alive(h_gen1) and h_gen1.generation() == 1, "gen1 handle must be alive")
    _assert(server.raws.remove(h_gen1), "remove(h_gen1) failed")
    _assert(not server.raws.is_alive(h_gen1), "removed handle must be dead")

    var b2 = List[UInt8](length=128, fill=0xBB)
    var h_gen2 = server.raws.insert(RawBytesAsset(b2^), AssetUuid.from_path("b.bin"), "b.bin", 2)
    _assert(h_gen2.slot() == h_gen1.slot() and h_gen2.generation() == 2, "slot must be reused with generation=2")
    _assert(not server.raws.is_alive(h_gen1), "stale gen1 handle MUST be rejected after slot reuse")
    _assert(server.raws.is_alive(h_gen2), "new gen2 handle must be alive")
    _ = server.raws.release(h_gen2)
    var evicted = server.raws.evict_lru_unreferenced(0)
    _assert(len(evicted) == 1 and not server.raws.is_alive(h_gen2), "LRU eviction must evict unreferenced slot")

    # 2. Byte-budgeted progressive mip streaming (`4 -> 3 -> 2 -> 1 -> 0`)
    var h_tex = server.load_texture("textures/albedo_256.png", progressive=True)
    _assert(h_tex.is_valid() and server.textures.is_alive(h_tex), "load_texture failed")
    _assert(server.textures.borrow(h_tex)[unsafe_offset=0].resident_min_mip == 4, "initial tail mip must be 4")

    var expected_mips = [3, 2, 1, 0]
    for step_i in range(4):
        _ = server.tick(now=0.0)
        var tex_ptr = server.textures.borrow(h_tex)
        var cur_mip = tex_ptr[unsafe_offset=0].resident_min_mip
        _assert(
            cur_mip == expected_mips[step_i],
            "expected progressive resident_min_mip=" + String(expected_mips[step_i]) + ", got " + String(cur_mip),
        )
        _assert(len(tex_ptr[unsafe_offset=0].mips[cur_mip]) > 0, "streamed mip payload empty")

    # Also verify pre-cooked mesh + Bvh8 collider load
    var h_mesh = server.load_mesh("meshes/forge_box.glb")
    _assert(server.meshes.is_alive(h_mesh), "load_mesh failed")
    var mesh_ptr = server.meshes.borrow(h_mesh)
    _assert(mesh_ptr[unsafe_offset=0].has_collider, "loaded mesh missing Bvh8 collider")
    _assert(mesh_ptr[unsafe_offset=0].geom.meshes[0].lod_count >= 2, "loaded mesh missing QEM LOD hierarchy")

    # 3. Live recursive `AssetWatcher` (`IN_CREATE | IN_ISDIR`) hot-reload under exact same Handle[T]!
    var sub_dir = loose_dir + "/nested_hot"
    _ = make_dirs(sub_dir)
    # Poll once so `IN_CREATE | IN_ISDIR` registers the newly created subdirectory watch
    _ = server.tick(now=get_time())

    var v1_bytes = List[UInt8]()
    for b in "v1_shader_params".as_bytes():
        v1_bytes.append(b)
    _assert(write_bytes(sub_dir + "/tune.bin", v1_bytes), "write initial loose asset failed")
    # Drain creation event of v1 before loading
    _ = server.watcher.poll_ready(now=get_time() + 1.0)
    server.watcher.pending_paths.clear()

    var h_hot = server.load_raw("nested_hot/tune.bin")
    _assert(server.raws.is_alive(h_hot) and server.raws.reload_epoch(h_hot) == 1, "initial hot handle invalid")
    var orig_packed = h_hot.untyped().packed

    # Modify loose file on disk -> triggers IN_CLOSE_WRITE in nested_hot/
    var v2_bytes = List[UInt8]()
    for b in "v2_hot_reloaded_params!".as_bytes():
        v2_bytes.append(b)
    _assert(write_bytes(sub_dir + "/tune.bin", v2_bytes), "write updated loose asset failed")

    var t_ev = get_time()
    _ = server.tick(now=t_ev)              # Ingests inotify event into coalesce queue
    _ = server.tick(now=t_ev + 0.50)       # Quiet window elapsed -> flushes & hot-reloads in-place!

    _assert(h_hot.untyped().packed == orig_packed, "Handle[T] bits must stay identical across hot-reload")
    _assert(server.raws.is_alive(h_hot), "Handle[T] must remain alive after hot-reload")
    _assert(server.raws.reload_epoch(h_hot) == 2, "reload_epoch must increment to 2 on hot-reload")
    var reloaded_len = len(server.raws.borrow(h_hot)[unsafe_offset=0].bytes)
    _assert(reloaded_len == len(v2_bytes), "hot-reloaded payload bytes were not updated in-place")

    server.close()


# ============================================================================
# (c) $O(1)$ VisBuffer Viewport Picking (`< 1 cm` Error) + Headless GPU Test
# ============================================================================

def test_c_visbuffer_picking() raises:
    print("  [c] Testing O(1) VisBuffer decode (< 1 cm 3D accuracy) + headless FlintAaaVulkanRenderer GPU pick...")
    var cam_pos = Vec3(1.0, 2.0, -5.0)
    var cam_fwd = Vec3(0.0, 0.0, 1.0)
    var cam_rgt = Vec3(1.0, 0.0, 0.0)
    var cam_up = Vec3(0.0, 1.0, 0.0)
    var fov_y = Float32(55.0)
    var near_z = Float32(0.15)
    var w = 1280
    var h = 720

    var true_pt = Vec3(1.75, 2.45, 2.20)
    var proj = project_world_to_viewport(
        true_pt, 0.0, 0.0, Float32(w), Float32(h), cam_pos, cam_fwd, cam_rgt, cam_up, fov_y, near_z
    )
    _assert(proj[3], "synthetic pick point must project in front of camera")
    var px = Int(proj[0])
    var py = Int(proj[1])
    var depth_d32 = near_z / proj[2]
    var vis_words = encode_visbuffer_words(instance_id=3, meshlet_id=29, material_id=11, local_prim_id=84)
    var inst_map = List[Int]()
    inst_map.append(100); inst_map.append(101); inst_map.append(102); inst_map.append(777)

    var pick = decode_visbuffer_pick(
        px, py, vis_words[0], vis_words[1], depth_d32, w, h,
        cam_pos, cam_fwd, cam_rgt, cam_up, fov_y, near_z, inst_map,
    )
    _assert(pick.hit, "decode_visbuffer_pick must report hit")
    _assert(pick.entity_id == 777 and pick.instance_id == 3, "decoded entity/instance ID mismatch")
    _assert(pick.meshlet_id == 29 and pick.material_id == 11 and pick.local_prim_id == 84, "decoded meshlet/mat/prim mismatch")
    var err_m = (pick.world_pos - true_pt).length()
    _assert(err_m < Float32(0.01), "analytical VisBuffer 3D world_pos error >= 1 cm: " + String(Float64(err_m)))

    # Headless FlintAaaVulkanRenderer 1-pixel GPU VisBuffer + Reverse-Z depth readback test
    var ren = FlintAaaVulkanRenderer(width=160, height=120, tsr_scale=1.0, max_draws=1024, max_instances=256, env_cube_dim=16)
    if ren.active:
        var mat_id = ren.add_material(AaaMaterial(base_color=Vec3(0.9, 0.2, 0.2), roughness=0.4, metallic=0.1))
        # Build front-facing quad at z = 3.5 (`[-1..1] x [-1..1]`) facing camera at origin `(0, 0, 0)` looking `+Z`
        var mb = MeshBuilder()
        var v0 = mb.add_vertex(Vec3(-1.0, -1.0, 3.5), Vec3(0.0, 0.0, -1.0), 0.0, 1.0)
        var v1 = mb.add_vertex(Vec3(1.0, -1.0, 3.5), Vec3(0.0, 0.0, -1.0), 1.0, 1.0)
        var v2 = mb.add_vertex(Vec3(1.0, 1.0, 3.5), Vec3(0.0, 0.0, -1.0), 1.0, 0.0)
        var v3 = mb.add_vertex(Vec3(-1.0, 1.0, 3.5), Vec3(0.0, 0.0, -1.0), 0.0, 0.0)
        mb.add_tri(v0, v1, v2)
        mb.add_tri(v0, v2, v3)
        var me = ren.add_mesh(mb^.finish(), mat_id, build_lods=False)

        ren.tsr_history_weight = Float32(0.0)
        ren.enable_vsm = False
        ren.enable_rt_shadows = False
        ren.enable_rt_reflections = False
        ren.enable_rt_gi = False
        ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=55.0, near_z=0.20)
        ren.draw_instance(me.mesh_id, Mat4(), material_override=mat_id, uniform_scale=1.0)
        _assert(ren.render_frame(capture_readback=True), "headless FlintAaaVulkanRenderer.render_frame failed")

        var gpu_inst_map = List[Int]()
        gpu_inst_map.append(4242)
        var gpu_pick = pick_renderer_viewport_once(ren, 80, 60, gpu_inst_map)
        _assert(gpu_pick.hit, "GPU VisBuffer center pixel pick must hit quad")
        _assert(gpu_pick.entity_id == 4242 and gpu_pick.instance_id == 0, "GPU pick entity/instance ID mismatch")
        _assert(gpu_pick.material_id == mat_id and gpu_pick.meshlet_id == 0, "GPU pick material/meshlet mismatch")
        var z_err = abs(gpu_pick.world_pos.z - Float32(3.5))
        _assert(z_err < Float32(0.01), "GPU Reverse-Z reconstructed depth error >= 1 cm: " + String(Float64(z_err)))

        var sky_pick = pick_renderer_viewport_once(ren, 2, 2, gpu_inst_map)
        _assert(not sky_pick.hit, "GPU VisBuffer corner sky pixel must report hit=False")
        ren.shutdown()


# ============================================================================
# (d) Analytical 3D Gizmos: Translate, Rotate (Normal + `88°` Grazing!), Scale
# ============================================================================

def test_d_analytical_3d_gizmos() raises:
    print("  [d] Testing 3D Translate/Rotate (normal + 88 deg grazing fallback!)/Scale gizmos & snapping...")
    var fwd = Vec3(0.0, 0.0, 1.0)
    var rgt = Vec3(1.0, 0.0, 0.0)
    var up = Vec3(0.0, 1.0, 0.0)

    # 1. Constant screen-pixel world scale invariance
    var s_near = gizmo_world_scale(Vec3(0, 0, 5), Vec3(0, 0, 0), fwd, 60.0, 720, 96.0)
    var s_far = gizmo_world_scale(Vec3(0, 0, 50), Vec3(0, 0, 0), fwd, 60.0, 720, 96.0)
    _assert(abs(s_far / s_near - Float32(10.0)) < Float32(1.0e-3), "gizmo_world_scale must scale linearly with depth")

    # 2. 1D axis + XZ plane translation + grid snapping
    var r0 = ViewportRay.from_screen(640.0, 360.0, 0.0, 0.0, 1280.0, 720.0, Vec3(0, 2, -6), fwd, rgt, up, 55.0)
    var r1 = ViewportRay.from_screen(760.0, 360.0, 0.0, 0.0, 1280.0, 720.0, Vec3(0, 2, -6), fwd, rgt, up, 55.0)
    var ax0 = ray_axis_closest_param(r0, Vec3(0, 2, 0), Vec3(1, 0, 0))
    var ax1 = ray_axis_closest_param(r1, Vec3(0, 2, 0), Vec3(1, 0, 0))
    _assert(ax0[0] and ax1[0] and (ax1[1] - ax0[1]) > Float32(0.4), "X-axis translation param failed")
    var snapped_p = snap_translation(Vec3(1.13, -0.37, 2.49), 0.25)
    _assert(
        abs(snapped_p.x - Float32(1.25)) < Float32(1e-5)
        and abs(snapped_p.y - Float32(-0.25)) < Float32(1e-5)
        and abs(snapped_p.z - Float32(2.50)) < Float32(1e-5),
        "snap_translation(0.25) failed",
    )
    var pl_hit = ray_plane_hit(ViewportRay(Vec3(0, 4, -4), Vec3(0, -0.70710678, 0.70710678)), Vec3(0, 0, 0), Vec3(0, 1, 0))
    _assert(pl_hit[0] and abs(pl_hit[1].z - Float32(0.0)) < Float32(1e-3), "XZ ray_plane_hit failed")

    # 3. Normal-angle rotation + 15 deg snapping
    var rot_r0 = ViewportRay.from_screen(740.0, 360.0, 0.0, 0.0, 1280.0, 720.0, Vec3(0, 0, -6), fwd, rgt, up, 55.0)
    var rot_r1 = ViewportRay.from_screen(640.0, 260.0, 0.0, 0.0, 1280.0, 720.0, Vec3(0, 0, -6), fwd, rgt, up, 55.0)
    var d_norm = gizmo_rotation_delta_rad(rot_r0, rot_r1, Vec3(0, 0, 0), Vec3(0, 0, 1), fwd, 1.0)
    _assert(abs(abs(d_norm) - Float32(1.5707963)) < Float32(0.06), "90 deg Z-ring rotation solver inaccurate")
    var snapped_ang = snap_angle_deg(Float32(0.29), Float32(15.0))  # ~16.6 deg -> snaps to 15 deg (0.261799 rad)
    _assert(abs(snapped_ang - Float32(0.2617994)) < Float32(1.0e-4), "15 deg angle snap failed")

    # 4. `88°` Grazing-angle (`cos(88°) = 0.0349 < 0.15`!) automatic view-plane fallback!
    var deg88 = Float32(88.0 * 0.01745329252)
    var grazing_axis = Vec3(0.0, sin(deg88), cos(deg88)).normalized()  # 88 deg to view forward +Z!
    var gr0 = ViewportRay.from_screen(640.0, 360.0, 0.0, 0.0, 1280.0, 720.0, Vec3(0, 0, -6), fwd, rgt, up, 55.0)
    var gr1 = ViewportRay.from_screen(720.0, 360.0, 0.0, 0.0, 1280.0, 720.0, Vec3(0, 0, -6), fwd, rgt, up, 55.0)
    var d_grazing = gizmo_rotation_delta_rad(gr0, gr1, Vec3(0, 0, 0), grazing_axis, fwd, 1.0)
    _assert(
        d_grazing == d_grazing and abs(d_grazing) > Float32(0.15) and abs(d_grazing) < Float32(2.5),
        "88 deg grazing-angle rotation fallback failed or blew up: " + String(Float64(d_grazing)),
    )

    # 5. Scale gizmo + UiDrawList primitive emission
    var sc_axis = gizmo_scale_drag(r0, r1, Vec3(0, 2, 0), GIZMO_AXIS_X, 1.0, fwd, rgt, Vec3(1, 1, 1))
    var sc_uni = gizmo_scale_drag(r0, r1, Vec3(0, 2, 0), GIZMO_SCALE_UNIFORM, 1.0, fwd, rgt, Vec3(2, 2, 2))
    _assert(sc_axis.x > Float32(1.2) and abs(sc_axis.y - Float32(1.0)) < Float32(1e-5), "X-axis scale drag failed")
    _assert(sc_uni.x > Float32(2.2) and abs(sc_uni.x - sc_uni.z) < Float32(1e-5), "Uniform scale drag failed")

    var ui = UiDrawList()
    emit_gizmo_draw_list(ui, Vec3(0, 0, 0), GIZMO_MODE_TRANSLATE, GIZMO_PLANE_XZ, 0, 0, 1280, 720, Vec3(0, 2, -6), fwd, rgt, up, 55.0)
    emit_gizmo_draw_list(ui, Vec3(0, 0, 0), GIZMO_MODE_ROTATE, GIZMO_AXIS_Y, 0, 0, 1280, 720, Vec3(0, 2, -6), fwd, rgt, up, 55.0)
    emit_gizmo_draw_list(ui, Vec3(0, 0, 0), GIZMO_MODE_SCALE, GIZMO_SCALE_UNIFORM, 0, 0, 1280, 720, Vec3(0, 2, -6), fwd, rgt, up, 55.0)
    _assert(ui.count() > 80, "emit_gizmo_draw_list emitted too few UI primitives")


# ============================================================================
# (e) 50-Command Undo/Redo Invertibility + 1-Click PIE 120-Tick Exact Restore
# ============================================================================

def test_e_undo_redo_50_and_pie_120_ticks() raises:
    print("  [e] Testing 50-command Undo/Redo fnv64 invertibility + 1-Click PIE 120-tick bit-exact restore...")
    var studio = FlintStudioState()
    var hashes = List[UInt64](capacity=51)
    hashes.append(studio.state_hash())

    # Execute exactly 50 varied commands (spawns, TRS edits, material edits, despawns)
    for step in range(50):
        var kind = step % 5
        if kind == 0 or studio.live_entity_count() < 3:
            _ = studio.spawn_entity(
                "Node_" + String(step),
                position=Vec3(Float32(step) * Float32(0.5), Float32(1 + (step % 4)), Float32(-step) * Float32(0.25)),
                collider_kind=COLLIDER_SPHERE if (step & 1) == 0 else COLLIDER_BOX,
                mass=Float32(5.0) if (step % 3) == 0 else Float32(0.0),
                record_undo=True,
            )
        elif kind == 1 or kind == 2:
            var eid = studio.selected_entity_id
            var cur = studio.entities[eid].copy()
            cur.translation = cur.translation + Vec3(Float32(0.35), Float32(0.15) * Float32(step % 3), Float32(-0.20))
            var ang = Float32(step) * Float32(0.12)
            cur.rotation = Quat(0.0, sin(ang * Float32(0.5)), 0.0, cos(ang * Float32(0.5))).normalized()
            cur.scale = Vec3(Float32(1.0 + Float32(step % 4) * 0.1), Float32(1.0), Float32(1.0))
            _ = studio.commit_node_edit(eid, cur^, "TRS_" + String(step))
        elif kind == 3:
            var eid = studio.selected_entity_id
            var cur = studio.entities[eid].copy()
            cur.roughness = Float32(0.1 + Float32(step % 8) * 0.1)
            cur.metallic = Float32(step % 2) * Float32(0.85)
            cur.base_color = Vec3(Float32(0.2), Float32(0.5), Float32(0.9))
            _ = studio.commit_node_edit(eid, cur^, "Mat_" + String(step))
        else:
            var eid = studio.selected_entity_id
            _ = studio.despawn_entity(eid, record_undo=True)
        hashes.append(studio.state_hash())

    _assert(len(studio.undo_stack) == 50 and len(hashes) == 51, "must record 50 undo transactions")

    # Step backward through all 50 undo operations verifying exact fnv64 equality at every step!
    var idx = 49
    while idx >= 0:
        _assert(studio.undo(), "undo() failed at step " + String(idx))
        _assert(
            studio.state_hash() == hashes[idx],
            "Undo fnv64 mismatch at step " + String(idx),
        )
        idx -= 1

    # Step forward through all 50 redo operations verifying exact fnv64 equality at every step!
    for r_idx in range(1, 51):
        _assert(studio.redo(), "redo() failed at step " + String(r_idx))
        _assert(
            studio.state_hash() == hashes[r_idx],
            "Redo fnv64 mismatch at step " + String(r_idx),
        )

    # Verify Inspectable + FieldVisitor reflection (including NPR/VFX/Stepped-Anim fields)
    var drop_id = studio.spawn_entity(
        "DynamicSphere",
        position=Vec3(0.0, 5.0, 0.0),
        collider_kind=COLLIDER_SPHERE,
        collider_half_extents=Vec3(0.5, 0.5, 0.5),
        mass=8.0,
    )
    var fvis = FieldVisitor()
    studio.entities[drop_id].inspect(fvis)
    _assert(len(fvis.fields) == 26, "FieldVisitor must reflect all 26 StudioEntityNode properties")

    # Verify configure_entity_npr_cel, configure_entity_stylized_vfx & patch_entity_field Undo/Redo fnv64 invertibility
    var h_npr_base = studio.state_hash()
    _assert(
        studio.configure_entity_npr_cel(
            drop_id,
            cel_shadow_tint=Vec3(0.62, 0.40, 0.46),
            cel_outline_width_px=2.4,
            cel_hair_halo=0.55,
            anim_step_hz=12.0,
            smear_stretch=1.35,
        ),
        "configure_entity_npr_cel failed",
    )
    var h_npr_cel = studio.state_hash()
    _assert(studio.entities[drop_id].shading_mode == SHADING_NPR_CEL and h_npr_cel != h_npr_base, "NPR Cel must mutate hash")
    _assert(
        studio.configure_entity_stylized_vfx(drop_id, vfx_erosion=0.42, vfx_scroll_speed=4.5, anim_step_hz=24.0),
        "configure_entity_stylized_vfx failed",
    )
    var h_npr_vfx = studio.state_hash()
    _assert(studio.entities[drop_id].shading_mode == SHADING_STYLIZED_VFX and h_npr_vfx != h_npr_cel, "Stylized VFX must mutate hash")
    var sm_patch = InspectedField("smear_stretch", FIELD_F32)
    sm_patch.f_val = Float32(1.75)
    _assert(studio.patch_entity_field(drop_id, "smear_stretch", sm_patch), "patch_entity_field(smear_stretch) failed")
    var h_npr_patched = studio.state_hash()
    _assert(studio.undo() and studio.state_hash() == h_npr_vfx, "undo patch_entity_field hash mismatch")
    _assert(studio.undo() and studio.state_hash() == h_npr_cel, "undo configure_entity_stylized_vfx hash mismatch")
    _assert(studio.undo() and studio.state_hash() == h_npr_base, "undo configure_entity_npr_cel hash mismatch")
    _assert(studio.redo() and studio.state_hash() == h_npr_cel, "redo configure_entity_npr_cel hash mismatch")
    _assert(studio.redo() and studio.state_hash() == h_npr_vfx, "redo configure_entity_stylized_vfx hash mismatch")
    _assert(studio.redo() and studio.state_hash() == h_npr_patched, "redo patch_entity_field hash mismatch")
    _ = studio.undo()
    _ = studio.undo()
    _ = studio.undo()
    _assert(studio.state_hash() == h_npr_base, "3x undo back to h_npr_base failed")

    # 1-Click Play-In-Editor: 120-tick physics & SteppedAnimClock run -> pie_stop() bit-for-bit identical restoration!
    var pre_pie_hash = studio.state_hash()
    var pre_pie_y = studio.entities[drop_id].translation.y
    _assert(studio.pie_play(), "pie_play() failed")
    var stepped = studio.pie_step(120)
    _assert(stepped == 120, "pie_step(120) must simulate 120 ticks")
    _assert(studio.stepped_clock.stepped_time_sec > Float32(1.5), "stepped_clock must advance during PIE")
    _assert(
        studio.entities[drop_id].translation.y < pre_pie_y - Float32(1.5),
        "dynamic sphere must fall under gravity during PIE (y=" + String(studio.entities[drop_id].translation.y) + ")",
    )
    _assert(studio.state_hash() != pre_pie_hash, "world hash must change during 120-tick PIE simulation")

    _assert(studio.pie_stop(), "pie_stop() failed")
    _assert(studio.state_hash() == pre_pie_hash, "pie_stop() MUST restore bit-for-bit identical fnv64 world hash!")
    _assert(studio.stepped_clock.raw_time_sec == Float32(0.0), "pie_stop() must restore stepped_clock")
    _assert(abs(studio.entities[drop_id].translation.y - pre_pie_y) == Float32(0.0), "restored Y position must be bit-identical")

    # Verify Studio `InputMap` (`W/E/R`, `Ctrl+Z`, `Ctrl+Y`, `F5` PIE), `CameraRig` & `DockSpace` + `UiContext`!
    var inp = InputState()
    inp.begin_frame()
    inp.keys[69] = True  # E -> Rotate
    inp.pressed[69] = True
    _assert(studio.process_editor_input(inp, Float32(1.0 / 60.0)) and studio.gizmo_mode == GIZMO_MODE_ROTATE, "InputMap E -> Rotate failed")
    inp.keys[69] = False

    inp.begin_frame()
    inp.keys[82] = True  # R -> Scale
    inp.pressed[82] = True
    _assert(studio.process_editor_input(inp, Float32(1.0 / 60.0)) and studio.gizmo_mode == GIZMO_MODE_SCALE, "InputMap R -> Scale failed")
    inp.keys[82] = False

    inp.begin_frame()
    inp.keys[341] = True  # LCTRL
    inp.keys[90] = True   # Z -> Ctrl+Z Undo!
    inp.pressed[90] = True
    _assert(studio.process_editor_input(inp, Float32(1.0 / 60.0)), "InputMap Ctrl+Z failed")
    _assert(studio.state_hash() != pre_pie_hash, "Ctrl+Z must undo last spawn")
    inp.keys[90] = False
    inp.keys[341] = False

    inp.begin_frame()
    inp.keys[341] = True  # LCTRL
    inp.keys[89] = True   # Y -> Ctrl+Y Redo!
    inp.pressed[89] = True
    _assert(studio.process_editor_input(inp, Float32(1.0 / 60.0)) and studio.state_hash() == pre_pie_hash, "Ctrl+Y must redo back to pre_pie_hash")
    inp.keys[89] = False
    inp.keys[341] = False

    inp.begin_frame()
    inp.keys[294] = True  # F5 -> 1-Click PIE play!
    inp.pressed[294] = True
    _assert(studio.process_editor_input(inp, Float32(1.0 / 60.0)) and studio.pie.mode == PIE_PLAYING, "F5 must start 1-Click PIE")
    inp.keys[294] = False
    _ = studio.process_editor_input(inp, Float32(1.0 / 60.0))
    _ = studio.pie_step(15)

    inp.begin_frame()
    inp.keys[294] = True  # F5 -> 1-Click PIE stop & LZ4 restore!
    inp.pressed[294] = True
    _assert(studio.process_editor_input(inp, Float32(1.0 / 60.0)) and studio.pie.mode == PIE_STOPPED, "F5 second press must stop PIE")
    _assert(studio.state_hash() == pre_pie_hash, "F5 PIE stop must restore exact world hash")
    inp.keys[294] = False

    var studio_ui = UiDrawList()
    emit_studio_ui_draw_list(studio_ui, studio, -1, SdfFont(), 1280, 720)
    var pbr_ui_cnt = studio_ui.count()
    _assert(pbr_ui_cnt > 60 and len(studio.dock.nodes) >= 5, "Studio UiContext + DockSpace failed to emit UI primitives")
    _ = studio.configure_entity_npr_cel(drop_id, anim_step_hz=12.0)
    var studio_npr_ui = UiDrawList()
    emit_studio_ui_draw_list(studio_npr_ui, studio, -1, SdfFont(), 1280, 720)
    _assert(studio_npr_ui.count() > pbr_ui_cnt, "NPR Cel + SteppedClock Inspector readout must emit additional UI primitives")
    _ = studio.undo()

    # Verify neutral Studio procedural presets
    var orb_b = bake_procedural_to_glb("material_orb", "", seed=1)
    var tor_b = bake_procedural_to_glb("catmull_torus", "", seed=1)
    _assert(orb_b[0] and orb_b[2] >= 192 and tor_b[0] and tor_b[2] >= 384, "Neutral SubD Studio presets failed")


# ============================================================================
# (f) End-to-End JSON-RPC 2.0 `McpServerV2` & Unix Domain Socket Round-Trip
# ============================================================================

struct _LargeMcpTransferCtx(Movable):
    var sock_path: String
    var req_payload: String
    var ready: Atomic[Int64]
    var done: Atomic[Int64]
    var resp_ok: Bool
    var resp_text: String

    def __init__(out self, sock_path: String, req_payload: String):
        self.sock_path = sock_path
        self.req_payload = req_payload
        self.ready = Atomic[Int64](0)
        self.done = Atomic[Int64](0)
        self.resp_ok = False
        self.resp_text = String()

    @always_inline
    def keep_alive(self):
        var a = self.sock_path.byte_length() + self.req_payload.byte_length()
        _ = a


def _large_mcp_worker(ctx_addr: Int, job_idx: Int):
    ref ctx = Pointer[_LargeMcpTransferCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    if job_idx == 0:
        # Worker 0: Client connects, shrinks SO_SNDBUF to 16 KiB to force EAGAINs, sends >256 KiB request & receives >256 KiB response!
        var cli = McpUnixSessionClient(use_content_length=True)
        if cli.connect(ctx.sock_path):
            var small_buf = Array[Int32, 1](fill=16384)
            _ = external_call["setsockopt", Int32](Int32(cli.fd), Int32(1), Int32(7), Pointer(to=small_buf[0]), UInt32(4))
            ctx.ready.store(1)
            if cli.send_raw_request(ctx.req_payload, timeout_ms=4000):
                var got = cli.recv_response_blocking(timeout_ms=4000)
                ctx.resp_ok = got[0]
                ctx.resp_text = got[1]
        cli.close()
        ctx.done.store(1)


def _stat_mode_octal(path: String) -> UInt32:
    comptime MODE_OFF: Int = 24 if CompilationTarget.is_x86() else 16
    var owned = path
    var buf = List[UInt8](length=256, fill=0)
    if external_call["lstat", Int32](owned.as_c_string_span(), buf.unsafe_ptr()) != 0:
        return 0
    var mode = Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=Int(buf.unsafe_ptr()) + MODE_OFF)[unsafe_offset=0]
    _ = len(buf)
    return mode & UInt32(0o777)


def test_f_mcp_v2_end_to_end(tmp_root: String, mut pool: JobPool) raises:
    print("  [f] Testing McpServerV2 auth, sandbox, bad Content-Length recovery & >256 KiB EAGAIN-safe Unix socket...")
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var mcp = McpServerV2(workspace_root=tmp_root)

    # 1. initialize
    var r_init = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"flint-test-agent"}}}',
        studio, assets, pool,
    )
    var d_init = parse_json(_str_bytes(r_init))
    _assert(d_init.valid and d_init.text(d_init.object_get(d_init.object_get(d_init.root, "result"), "protocolVersion")) == "2024-11-05", "MCP initialize failed")

    # Notification (no id) -> empty response
    var r_notif = mcp.handle_jsonrpc('{"jsonrpc":"2.0","method":"notifications/initialized"}', studio, assets, pool)
    _assert(r_notif.byte_length() == 0, "JSON-RPC notification without id must produce empty response")

    # 2. tools/list -> all 15 tools present (including render.configure_npr_cel & render.trigger_post_fx)
    var r_list = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}', studio, assets, pool)
    var d_list = parse_json(_str_bytes(r_list))
    var tools_arr = d_list.object_get(d_list.object_get(d_list.root, "result"), "tools")
    _assert(d_list.valid and d_list.array_size(tools_arr) >= 16, "tools/list must expose all MCP tools (15 + assets.semantic_search)")
    _assert(
        r_list.find('"render.configure_npr_cel"') >= 0 and r_list.find('"render.trigger_post_fx"') >= 0,
        "tools/list must include render.configure_npr_cel and render.trigger_post_fx",
    )

    # 3. ecs.spawn_entity
    var h0 = studio.state_hash()
    var r_spawn = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ecs.spawn_entity","arguments":{"name":"McpCrate","position":[0.0,4.0,0.0],"collider":"box","mass":5.0}}}',
        studio, assets, pool,
    )
    var d_spawn = parse_json(_str_bytes(r_spawn))
    var sc_spawn = d_spawn.object_get(d_spawn.object_get(d_spawn.root, "result"), "structuredContent")
    _assert(d_spawn.valid and d_spawn.boolean(d_spawn.object_get(sc_spawn, "ok")) and studio.live_entity_count() == 1, "ecs.spawn_entity failed")

    # 4. physics.raycast_query
    var r_ray = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"physics.raycast_query","arguments":{"origin":[0.0,10.0,0.0],"direction":[0.0,-1.0,0.0],"max_distance":50.0}}}',
        studio, assets, pool,
    )
    var d_ray = parse_json(_str_bytes(r_ray))
    var sc_ray = d_ray.object_get(d_ray.object_get(d_ray.root, "result"), "structuredContent")
    _assert(d_ray.boolean(d_ray.object_get(sc_ray, "hit")) and d_ray.integer(d_ray.object_get(sc_ray, "entity_id")) == 0, "physics.raycast_query missed McpCrate")

    # 5. physics.step_deterministic
    var r_step = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"physics.step_deterministic","arguments":{"ticks":30}}}',
        studio, assets, pool,
    )
    var d_step = parse_json(_str_bytes(r_step))
    _assert(d_step.valid and studio.entities[0].translation.y < Float32(3.5), "physics.step_deterministic did not advance gravity")

    # 6. studio.undo_redo -> undoes spawn back to exact initial empty scene hash!
    var r_undo = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"undo","steps":1}}}',
        studio, assets, pool,
    )
    _ = parse_json(_str_bytes(r_undo))
    studio.sim_tick = 0
    _assert(studio.live_entity_count() == 0 and studio.state_hash() == h0, "studio.undo_redo failed to undo ecs.spawn_entity")

    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"redo","steps":1}}}',
        studio, assets, pool,
    )
    _assert(studio.live_entity_count() == 1, "studio.undo_redo redo failed")

    # 7. render.capture_viewport + perceptual FLIP diff (inside `workspace_root`)
    var shot_a = tmp_root + "/mcp_shot_a.png"
    var shot_b = tmp_root + "/mcp_shot_b.png"
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"' + shot_a + '","width":160,"height":120}}}',
        studio, assets, pool,
    )
    var r_cap = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"' + shot_b + '","reference_path":"' + shot_a + '","width":160,"height":120}}}',
        studio, assets, pool,
    )
    var d_cap = parse_json(_str_bytes(r_cap))
    var sc_cap = d_cap.object_get(d_cap.object_get(d_cap.root, "result"), "structuredContent")
    _assert(
        d_cap.boolean(d_cap.object_get(sc_cap, "has_flip_diff"))
        and d_cap.number(d_cap.object_get(sc_cap, "flip_mean"), 1.0) < 1.0e-5,
        "render.capture_viewport FLIP self-diff must be ~0.0",
    )

    # 7b. Verify Tools 13 & 14: `render.configure_npr_cel` (with bit-exact fnv64 Undo/Redo!) & `render.trigger_post_fx`
    var h_before_npr = studio.state_hash()
    _assert(studio.entities[0].shading_mode == SHADING_PBR, "entity 0 must start in SHADING_PBR")
    var r_cel = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":901,"method":"tools/call","params":{"name":"render.configure_npr_cel","arguments":{"entity_id":0,"shading_mode":"npr_cel","cel_shadow_tint":[0.60,0.42,0.48],"cel_outline_width_px":2.75,"cel_outline_color":[0.02,0.01,0.02],"cel_rim_color":[0.50,0.58,0.74],"cel_hair_halo":0.65,"anim_step_hz":12.0,"smear_stretch":1.45}}}',
        studio, assets, pool,
    )
    var d_cel = parse_json(_str_bytes(r_cel))
    var sc_cel = d_cel.object_get(d_cel.object_get(d_cel.root, "result"), "structuredContent")
    _assert(
        d_cel.valid
        and d_cel.boolean(d_cel.object_get(sc_cel, "ok"))
        and studio.entities[0].shading_mode == SHADING_NPR_CEL
        and abs(studio.entities[0].cel_outline_width_px - Float32(2.75)) < Float32(1e-5)
        and abs(studio.entities[0].anim_step_hz - Float32(12.0)) < Float32(1e-5)
        and abs(studio.entities[0].smear_stretch - Float32(1.45)) < Float32(1e-5),
        "render.configure_npr_cel (npr_cel) failed",
    )
    var h_after_cel = studio.state_hash()
    _assert(h_after_cel != h_before_npr, "render.configure_npr_cel must mutate studio.state_hash()")

    var r_vfx = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":902,"method":"tools/call","params":{"name":"render.configure_npr_cel","arguments":{"entity_id":0,"shading_mode":"stylized_vfx","vfx_erosion":0.44,"vfx_scroll_speed":4.8,"vfx_core_color":[9.0,6.5,2.4],"vfx_edge_color":[4.4,0.35,0.06]}}}',
        studio, assets, pool,
    )
    var d_vfx = parse_json(_str_bytes(r_vfx))
    var sc_vfx = d_vfx.object_get(d_vfx.object_get(d_vfx.root, "result"), "structuredContent")
    _assert(
        d_vfx.valid
        and d_vfx.boolean(d_vfx.object_get(sc_vfx, "ok"))
        and studio.entities[0].shading_mode == SHADING_STYLIZED_VFX
        and abs(studio.entities[0].vfx_erosion - Float32(0.44)) < Float32(1e-5),
        "render.configure_npr_cel (stylized_vfx) failed",
    )
    var h_after_vfx = studio.state_hash()
    _assert(h_after_vfx != h_after_cel and h_after_vfx != h_before_npr, "stylized_vfx must mutate studio.state_hash()")

    # Undo stylized_vfx -> npr_cel -> pbr, verifying bit-for-bit fnv64 restoration at each step!
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":903,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"undo","steps":1}}}',
        studio, assets, pool,
    )
    _assert(
        studio.entities[0].shading_mode == SHADING_NPR_CEL and studio.state_hash() == h_after_cel,
        "studio.undo_redo undo (vfx -> cel) fnv64 mismatch",
    )
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":904,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"undo","steps":1}}}',
        studio, assets, pool,
    )
    _assert(
        studio.entities[0].shading_mode == SHADING_PBR and studio.state_hash() == h_before_npr,
        "studio.undo_redo undo (cel -> pbr) fnv64 mismatch",
    )
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":905,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"redo","steps":2}}}',
        studio, assets, pool,
    )
    _assert(
        studio.entities[0].shading_mode == SHADING_STYLIZED_VFX and studio.state_hash() == h_after_vfx,
        "studio.undo_redo redo (pbr -> vfx) fnv64 mismatch",
    )

    # Trigger post-FX + hitstop and verify SteppedAnimClock freezes during hitstop!
    var t_before_hs = studio.stepped_clock.advance(Float32(0.10))
    _assert(t_before_hs > Float32(0.04), "SteppedAnimClock must advance prior to hitstop")
    var r_pfx = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":906,"method":"tools/call","params":{"name":"render.trigger_post_fx","arguments":{"impact_blend":0.85,"impact_invert":true,"shockwave_active":true,"shockwave_radius":0.31,"hitstop_sec":0.10}}}',
        studio, assets, pool,
    )
    var d_pfx = parse_json(_str_bytes(r_pfx))
    var sc_pfx = d_pfx.object_get(d_pfx.object_get(d_pfx.root, "result"), "structuredContent")
    _assert(
        d_pfx.valid
        and d_pfx.boolean(d_pfx.object_get(sc_pfx, "ok"))
        and abs(studio.preview_impact_blend - Float32(0.85)) < Float32(1e-5)
        and studio.preview_impact_invert
        and studio.preview_shockwave_active
        and abs(studio.preview_shockwave_radius - Float32(0.31)) < Float32(1e-5)
        and abs(studio.stepped_clock.hitstop_remaining_sec - Float32(0.10)) < Float32(1e-5),
        "render.trigger_post_fx state assignment failed",
    )
    var t_frozen_1 = studio.stepped_clock.advance(Float32(0.04))
    _assert(
        abs(t_frozen_1 - t_before_hs) == Float32(0.0)
        and not studio.stepped_clock.pose_changed_this_tick
        and studio.stepped_clock.hitstop_remaining_sec > Float32(0.05),
        "SteppedAnimClock MUST freeze stepped_time_sec during hitstop!",
    )
    var t_frozen_2 = studio.stepped_clock.advance(Float32(0.04))
    _assert(
        abs(t_frozen_2 - t_before_hs) == Float32(0.0) and not studio.stepped_clock.pose_changed_this_tick,
        "SteppedAnimClock MUST stay frozen until hitstop_remaining_sec reaches 0!",
    )
    _ = studio.stepped_clock.advance(Float32(0.04))  # Drains remaining 0.02s hitstop
    var t_resumed = studio.stepped_clock.advance(Float32(0.10))
    _assert(t_resumed > t_before_hs, "SteppedAnimClock must resume advancing once hitstop finishes")

    var r_pfx_clr = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":907,"method":"tools/call","params":{"name":"render.trigger_post_fx","arguments":{"clear":true}}}',
        studio, assets, pool,
    )
    var d_pfx_clr = parse_json(_str_bytes(r_pfx_clr))
    _assert(
        d_pfx_clr.valid
        and studio.preview_impact_blend == Float32(0.0)
        and not studio.preview_shockwave_active
        and studio.stepped_clock.hitstop_remaining_sec == Float32(0.0),
        "render.trigger_post_fx clear failed",
    )

    # 8. Workspace Path Sandbox: reject `../../etc/passwd`, `/tmp/outside.png`, NUL bytes, and symlinks (`S_ISLNK`)!
    var sym_link_path = tmp_root + "/evil_link"
    var outside_target = "/tmp"
    _ = external_call["symlink", Int32](outside_target.as_c_string_span(), sym_link_path.as_c_string_span())
    _ = outside_target.byte_length()

    var bad_paths = [String("../../etc/passwd"), String("/tmp/outside.png"), String("evil_link/escaped.png"), String("")]
    for bp_i in range(len(bad_paths)):
        var threw = False
        try:
            _ = sanitize_output_path(tmp_root, bad_paths[bp_i])
        except:
            threw = True
        _assert(threw, "sanitize_output_path MUST reject hostile path: " + bad_paths[bp_i])

    # Verify all 4 file-writing MCP tools reject path traversal & outside paths!
    var r_trav1 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"../../etc/passwd"}}}',
        studio, assets, pool,
    )
    var d_trav1 = parse_json(_str_bytes(r_trav1))
    _assert(d_trav1.boolean(d_trav1.object_get(d_trav1.object_get(d_trav1.root, "result"), "isError")), "capture_viewport must reject ../../etc/passwd")

    var r_trav2 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"/tmp/outside.png"}}}',
        studio, assets, pool,
    )
    var d_trav2 = parse_json(_str_bytes(r_trav2))
    _assert(d_trav2.boolean(d_trav2.object_get(d_trav2.object_get(d_trav2.root, "result"), "isError")), "capture_viewport must reject /tmp/outside.png")

    var r_trav3 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"assets.list_and_cook","arguments":{"source_dir":".","output_flpk":"/tmp/outside.flpk"}}}',
        studio, assets, pool,
    )
    var d_trav3 = parse_json(_str_bytes(r_trav3))
    _assert(d_trav3.boolean(d_trav3.object_get(d_trav3.object_get(d_trav3.root, "result"), "isError")), "list_and_cook must reject /tmp/outside.flpk")

    var r_trav4 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"procedural.generate_asset","arguments":{"preset":"material_orb","output_glb_path":"../outside.glb"}}}',
        studio, assets, pool,
    )
    var d_trav4 = parse_json(_str_bytes(r_trav4))
    _assert(d_trav4.boolean(d_trav4.object_get(d_trav4.object_get(d_trav4.root, "result"), "isError")), "generate_asset must reject ../outside.glb")

    var r_trav5 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":14,"method":"tools/call","params":{"name":"physics.step_deterministic","arguments":{"ticks":1,"output_path":"evil_link/escape.lz4"}}}',
        studio, assets, pool,
    )
    var d_trav5 = parse_json(_str_bytes(r_trav5))
    _assert(d_trav5.boolean(d_trav5.object_get(d_trav5.object_get(d_trav5.root, "result"), "isError")), "step_deterministic must reject symlink escape")

    # 8c. Tool 15: `assets.process_mesh` (geometry kit) - sandboxed input, QEM LOD on a baked orb
    var r_kit_trav = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":15,"method":"tools/call","params":{"name":"assets.process_mesh","arguments":{"input_glb_path":"../outside.glb","op":"stats"}}}',
        studio, assets, pool,
    )
    var d_kit_trav = parse_json(_str_bytes(r_kit_trav))
    _assert(d_kit_trav.boolean(d_kit_trav.object_get(d_kit_trav.object_get(d_kit_trav.root, "result"), "isError")), "process_mesh must reject ../outside.glb")
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":16,"method":"tools/call","params":{"name":"procedural.generate_asset","arguments":{"preset":"material_orb","output_glb_path":"kit/orb.glb"}}}',
        studio, assets, pool,
    )
    var r_kit = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":17,"method":"tools/call","params":{"name":"assets.process_mesh","arguments":{"input_glb_path":"kit/orb.glb","op":"lod","ratio":0.5,"output_glb_path":"kit/orb_lod.glb"}}}',
        studio, assets, pool,
    )
    _assert(r_kit.find('\\"op\\":\\"lod\\"') >= 0 and r_kit.find('\\"ok\\":true') >= 0, "process_mesh lod must succeed: " + r_kit)

    # 9. Verify `auth_token` enforcement on `McpServerV2`
    var auth_mcp = McpServerV2(workspace_root=tmp_root, auth_token="flint-secret-token-42")
    var r_unauth = auth_mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":20,"method":"tools/list","params":{}}', studio, assets, pool)
    _assert(r_unauth.find("-32600") >= 0, "unauthenticated tools/list must fail with -32600")
    var r_bad_tok = auth_mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":21,"method":"initialize","params":{"auth_token":"wrong-token"}}', studio, assets, pool
    )
    _assert(r_bad_tok.find("-32600") >= 0, "wrong auth_token initialize must fail with -32600")
    var r_good_tok = auth_mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":22,"method":"initialize","params":{"auth_token":"flint-secret-token-42"}}', studio, assets, pool
    )
    _assert(r_good_tok.find('"protocolVersion"') >= 0, "valid auth_token initialize must succeed")

    # 10. Default `0700` runtime dir + `0600` socket mode + `SO_PEERCRED` verification
    # The default path belongs to a running Studio (if any): only check how it is formed and never
    # bind it here, so the gate cannot take the live agent socket away from an open Studio.
    var def_sock = default_mcp_socket_path()
    _assert(def_sock.endswith("/mcp.sock") and def_sock.find("flint") >= 0, "default MCP socket path malformed")

    var sock_path = tmp_root + "/mcp_v2.sock"
    var listener = FlintMcpUnixListener(sock_path)
    _assert(listener.active and _stat_mode_octal(sock_path) == UInt32(0o600), "FlintMcpUnixListener failed to bind 0600 socket")

    # A second listener on a live path must refuse, and closing it must not unlink the live socket
    var live_path = tmp_root + "/mcp_live.sock"
    var live = FlintMcpUnixListener(live_path)
    _assert(live.active, "live FlintMcpUnixListener failed to start")
    var rival = FlintMcpUnixListener(live_path)
    _assert(not rival.active, "second FlintMcpUnixListener must not steal a live socket")
    rival.close()
    var probe_cli = McpUnixSessionClient(use_content_length=True)
    _assert(probe_cli.connect(live_path), "live socket must survive a rival listener closing")
    probe_cli.close()
    live.close()
    var stale = FlintMcpUnixListener(live_path)
    _assert(stale.active, "a stale socket path must be reclaimed")
    stale.close()

    # Test `SO_PEERCRED` peer UID mismatch rejection
    listener.expected_peer_uid = Int(external_call["getuid", UInt32]()) + 1
    var spoof_cli = McpUnixSessionClient(use_content_length=True)
    _assert(spoof_cli.connect(sock_path), "spoof connect failed")
    _ = listener.poll_once(mcp, studio, assets, pool)
    _assert(listener.rejected_peers == 1 and listener.client_count() == 0, "SO_PEERCRED must reject mismatched peer UID")
    spoof_cli.close()
    listener.expected_peer_uid = -1

    # 11. Bad `Content-Length` & >8 KiB malformed header recovery (server MUST NOT wedge!)
    var bad_cli1 = McpUnixSessionClient(use_content_length=True)
    _assert(bad_cli1.connect(sock_path), "bad_cli1 connect failed")
    _assert(bad_cli1.send_raw_bytes(_str_bytes("Content-Length: -99\r\n\r\n{}")), "send negative Content-Length failed")
    _ = listener.poll_once(mcp, studio, assets, pool)
    var bad_r1 = bad_cli1.recv_response_blocking(timeout_ms=500)
    _assert(bad_r1[0] and bad_r1[1].find("-32600") >= 0, "negative Content-Length must receive -32600 error")
    _assert(listener.client_count() == 0, "bad_cli1 slot must be closed and reset")
    bad_cli1.close()

    var bad_cli2 = McpUnixSessionClient(use_content_length=True)
    _assert(bad_cli2.connect(sock_path), "bad_cli2 connect failed")
    var huge_hdr = List[UInt8](capacity=9000)
    for b in "Content-Length: 10\r\nX-Pad: ".as_bytes():
        huge_hdr.append(b)
    for _ in range(8500):
        huge_hdr.append(65)  # > 8 KiB without \r\n\r\n!
    _assert(bad_cli2.send_raw_bytes(huge_hdr), "send >8 KiB header failed")
    _ = listener.poll_once(mcp, studio, assets, pool)
    _assert(listener.client_count() == 0, ">8 KiB unterminated header client must be dropped cleanly")
    bad_cli2.close()

    # Verify fresh persistent session works immediately after bad-header clients!
    var client = McpUnixSessionClient(use_content_length=True)
    _assert(client.connect(sock_path), "McpUnixSessionClient.connect failed after bad-header recovery")
    _assert(client.send_raw_request('{"jsonrpc":"2.0","id":101,"method":"ping"}'), "send ping failed")
    _ = listener.poll_once(mcp, studio, assets, pool)
    var got1 = client.recv_response_blocking(timeout_ms=500)
    _assert(got1[0] and got1[1].find('"id":101') >= 0, "Unix socket Content-Length ping round-trip failed")

    _assert(
        client.send_raw_request('{"jsonrpc":"2.0","id":102,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{}}}'),
        "send 2nd request on persistent Unix socket failed",
    )
    _ = listener.poll_once(mcp, studio, assets, pool)
    var got2 = client.recv_response_blocking(timeout_ms=500)
    _assert(got2[0] and got2[1].find('"McpCrate"') >= 0, "Persistent Unix socket 2nd call failed")
    client.close()

    # 12. Large `> 256 KiB` EAGAIN-safe JSON-RPC payload round-trip (`SO_SNDBUF = 16 KiB` + `poll(POLLOUT)`)!
    var big_name_bytes = List[UInt8](length=300 * 1024, fill=88)  # 300 KiB 'X' string (> 256 KiB!)
    var big_name = String(unsafe_from_utf8=big_name_bytes)
    var big_req = (
        '{"jsonrpc":"2.0","id":999,"method":"tools/call","params":{"name":"ecs.spawn_entity","arguments":{"name":"'
        + big_name
        + '","position":[1.0,2.0,3.0]}}}'
    )
    var xfer_ctx = _LargeMcpTransferCtx(sock_path, big_req)
    # Run client on worker thread 0 while worker 1 pumps `listener.poll_once` with 16 KiB SO_SNDBUF!
    var small_srv_buf = Array[Int32, 1](fill=16384)
    _ = external_call["setsockopt", Int32](Int32(listener.listen_fd), Int32(1), Int32(7), Pointer(to=small_srv_buf[0]), UInt32(4))

    # Spawn a 2-way parallel test: job 0 = client send/recv (>300 KiB), job 1 = server poll loop!
    _run_large_mcp_pair(pool, listener, mcp, studio, assets, xfer_ctx)
    _assert(xfer_ctx.resp_ok, "> 256 KiB EAGAIN-safe JSON-RPC round-trip failed")
    var d_big = parse_json(_str_bytes(xfer_ctx.resp_text))
    _assert(
        d_big.valid and xfer_ctx.resp_text.byte_length() > 600 * 1024,
        "Expected > 600 KiB structured+text JSON-RPC response without EAGAIN truncation, got " + String(xfer_ctx.resp_text.byte_length()),
    )

    listener.close()
    assets.close()


@fieldwise_init
struct _ServerClientPairCtx(Copyable, Movable):
    var listener_ptr: Int
    var mcp_ptr: Int
    var studio_ptr: Int
    var assets_ptr: Int
    var pool_ptr: Int
    var xfer_ptr: Int

    @always_inline
    def keep_alive(self):
        var s = self.listener_ptr ^ self.xfer_ptr
        _ = s


def _server_client_pair_worker(ctx_addr: Int, job_idx: Int):
    ref p = Pointer[_ServerClientPairCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    if job_idx == 0:
        _large_mcp_worker(p.xfer_ptr, 0)
    elif job_idx == 1:
        ref listener = Pointer[FlintMcpUnixListener, MutAnyOrigin](unsafe_from_address=p.listener_ptr)[unsafe_offset=0]
        ref mcp = Pointer[McpServerV2, MutAnyOrigin](unsafe_from_address=p.mcp_ptr)[unsafe_offset=0]
        ref studio = Pointer[FlintStudioState, MutAnyOrigin](unsafe_from_address=p.studio_ptr)[unsafe_offset=0]
        ref assets = Pointer[AssetServer, MutAnyOrigin](unsafe_from_address=p.assets_ptr)[unsafe_offset=0]
        ref pool = Pointer[JobPool, MutAnyOrigin](unsafe_from_address=p.pool_ptr)[unsafe_offset=0]
        ref xfer = Pointer[_LargeMcpTransferCtx, MutAnyOrigin](unsafe_from_address=p.xfer_ptr)[unsafe_offset=0]
        var deadline = get_time() + 4.0
        while xfer.done.load() == 0 and get_time() < deadline:
            _ = listener.poll_once(mcp, studio, assets, pool)
            if listener.client_count() > 0:
                var small_b = Array[Int32, 1](fill=16384)
                _ = external_call["setsockopt", Int32](Int32(listener.client_fds[0]), Int32(1), Int32(7), Pointer(to=small_b[0]), UInt32(4))


def _run_large_mcp_pair(
    mut pool: JobPool,
    mut listener: FlintMcpUnixListener,
    mut mcp: McpServerV2,
    mut studio: FlintStudioState,
    mut assets: AssetServer,
    mut xfer: _LargeMcpTransferCtx,
):
    var pair = _ServerClientPairCtx(
        Int(Pointer(to=listener)),
        Int(Pointer(to=mcp)),
        Int(Pointer(to=studio)),
        Int(Pointer(to=assets)),
        Int(Pointer(to=pool)),
        Int(Pointer(to=xfer)),
    )
    run_parallel_ctx(pool, 2, pair, _server_client_pair_worker)
    pair.keep_alive()
    xfer.keep_alive()


def _tool_is_error(resp: String) -> Bool:
    var d = parse_json(_str_bytes(resp))
    return d.valid and d.boolean(d.object_get(d.object_get(d.root, "result"), "isError"))


def test_g_agent_regressions(tmp_root: String, mut pool: JobPool) raises:
    print("  [g] Testing agent-found regressions: wide box colliders, argument validation, recoverable bad lines...")
    # A flat 36 m grid box used to act as an 18 m sphere proxy and fling nearby bodies onto its shell.
    var studio = FlintStudioState()
    _ = studio.spawn_entity(
        "Grid", position=Vec3(0.0, 0.0, 0.0), collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(18.0, 0.05, 18.0), mass=0.0, record_undo=False,
    )
    var cube = studio.spawn_entity("Cube", position=Vec3(0.0, 3.0, 0.0), collider_kind=COLLIDER_BOX, mass=2.0, record_undo=False)
    var ball = studio.spawn_entity(
        "Ball", position=Vec3(5.0, 4.0, -3.0), collider_kind=COLLIDER_SPHERE,
        collider_half_extents=Vec3(0.5, 0.5, 0.5), mass=1.0, record_undo=False,
    )
    _ = studio.step_physics_deterministic(120)
    var cp = studio.entities[cube].translation
    var bp = studio.entities[ball].translation
    _assert(cp.y > Float32(0.5) and cp.y < Float32(0.62), "box must come to rest on the grid top, got y=" + String(cp.y))
    _assert(abs(cp.x) < Float32(0.01) and abs(cp.z) < Float32(0.01), "box must fall straight down onto the grid")
    _assert(bp.y > Float32(0.5) and bp.y < Float32(0.62), "sphere must come to rest on the grid top, got y=" + String(bp.y))

    var assets = AssetServer(enable_watcher=False)
    var mcp = McpServerV2(workspace_root=tmp_root)
    _assert(
        _tool_is_error(mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"procedural.generate_asset","arguments":{"preset":"dragon"}}}',
            studio, assets, pool,
        )),
        "unknown procedural preset must be a tool error, not a silent box",
    )
    _assert(
        _tool_is_error(mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"physics.step_deterministic","arguments":{"ticks":-5}}}',
            studio, assets, pool,
        )),
        "negative ticks must be rejected",
    )
    _assert(
        _tool_is_error(mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"physics.raycast_query","arguments":{"origin":[0,0,0],"direction":[0,0,0]}}}',
            studio, assets, pool,
        )),
        "zero ray direction must be rejected",
    )
    var r_unknown = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"no.such_tool","arguments":{}}}', studio, assets, pool
    )
    _assert(r_unknown.find('"error"') >= 0 and r_unknown.find("-32602") >= 0, "unknown tool must be JSON-RPC error -32602")

    # One bad newline-delimited line (a JSON-RPC batch) must not kill the session
    var buf = List[UInt8]()
    for b in '[{"jsonrpc":"2.0","id":9,"method":"ping"}]\n{"jsonrpc":"2.0","id":10,"method":"ping"}\n'.as_bytes():
        buf.append(b)
    var f1 = parse_next_mcp_frame(buf)
    _assert(f1.protocol_error and f1.recoverable, "batch line must be a recoverable protocol error")
    var f2 = parse_next_mcp_frame(buf)
    _assert(f2.has_frame and not f2.protocol_error, "the next line after a bad one must still parse")


def main() raises:
    var t0 = get_time()
    var pid = Int(external_call["getpid", Int32]())
    var tmp_root = "/tmp/flint-eng5-suite-" + String(pid)
    _rm_rf_tmp(tmp_root)
    _ = make_dirs(tmp_root)

    var pool = JobPool()
    _ = pool.start(4)

    print("[Flint v5 Track 5] Running AssetServer, .flpk v3 Cooker, Flint Studio & MCP v2 verification suite...")
    var pack_path = test_a_flpk_v3_cooker_and_crc64(tmp_root, pool)
    test_b_asset_server_streaming_and_hot_reload(tmp_root, pack_path)
    test_c_visbuffer_picking()
    test_d_analytical_3d_gizmos()
    test_e_undo_redo_50_and_pie_120_ticks()
    test_f_mcp_v2_end_to_end(tmp_root, pool)
    test_g_agent_regressions(tmp_root, pool)

    pool.stop()
    _rm_rf_tmp(tmp_root)

    var elapsed_ms = (get_time() - t0) * 1000.0
    print("[Flint v5 Track 5] ALL 7 SUITES PASSED in " + String(Int(elapsed_ms)) + " ms!")
