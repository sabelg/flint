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
from test_reference_fern import test_reference_fern
from std.atomic import Atomic
from std.ffi import external_call
from std.math import abs, cos, exp, log, max, min, pow, sin
from std.memory import bitcast
from std.sys.info import CompilationTarget
from std.time import monotonic
from flint.anim.flanim_ecs import BoneSocketAttachment
from flint.anim.graph import AnimGraphDef, Condition, OP_GT, OP_LE, StateDef
from flint.anim.plugin import AnimAssetCache, AnimatorComponent, BoneSocketComponent, RigComponent, SkeletonComponent
from flint.anim.pose import SkeletonInstance
from flint.anim.sync import ClipInfo
from flint.assets.asset_plugin import SceneAssetResolver
from flint.assets.asset_ref import AssetRef
from flint.assets.bc_compress import (
    bake_toksvig_normal_orm_pyramid,
    downsample_normals_and_orm_toksvig_2x,
    f32_to_uf16_bits,
)
from flint.assets.cooker_v2 import _is_dir, cook_workspace
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
from flint.assets.glb_native import parse_native_glb
from flint.assets.glb_writer import encode_native_glb, write_native_glb
from flint.assets.hdr_image import (
    EXR_DWAA_COMPRESSION,
    EXR_DWAB_COMPRESSION,
    EXR_NO_COMPRESSION,
    EXR_PIZ_COMPRESSION,
    EXR_RLE_COMPRESSION,
    decode_exr_fp16,
    decode_hdr_or_exr,
    encode_radiance_hdr,
)
from flint.assets.image import append_be32, append_chunk, decode_png, encode_png, write_png
from flint.assets.jpeg import decode_image_auto, decode_jpeg
from flint.assets.json import parse_json
from flint.assets.ktx2 import decode_ktx2, decompress_zstd
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.meshlet import AaaGeometryPool, MeshLodSlice, _SymQuadric, simplify_mesh_qem
from flint.assets.model import NativeAnimation, NativeChannel, NativeMesh, NativeModel, NativeNode, NativeTexture
from flint.audio.qoa import QoaAsset, encode_qoa
from flint.assets.server import (
    ASSET_STATE_FAILED,
    ASSET_STATE_LOADING,
    ASSET_STATE_READY,
    AssetServer,
    CookedHdrAsset,
    CookedTextureAsset,
    RawBytesAsset,
    TypedAssetStorage,
)
from flint.assets.watcher import AssetWatcher, IN_CLOSE_WRITE, IN_IGNORED, IN_Q_OVERFLOW
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool
from flint.core.lz4 import address_of
from flint.core.math import Quat, Vec2, Vec3
from flint.core.serial import BlobReader, BlobWriter
from flint.core.transform import Mat4
from flint.devtools.mcp_v2 import (
    MCP_GATE_AUTO,
    FlintMcpUnixListener,
    McpServerV2,
    McpUnixSessionClient,
    default_mcp_socket_path,
    encode_mcp_frame,
    parse_next_mcp_frame,
    sanitize_output_path,
)
from flint.physics.world import (
    BODY_MOTION_DYNAMIC,
    BODY_MOTION_KINEMATIC,
    BODY_MOTION_STATIC,
    PhysicsWorld,
)
from flint.physics.plugin import ColliderComponent, RigidBodyComponent
from flint.platform.input import InputState
from flint.platform.os import file_size, get_time, is_directory, make_dirs, read_bytes, remove_file, write_bytes
from flint.platform.vulkan import LAYOUT_SHADER_READ_ONLY, LAYOUT_UNDEFINED, VulkanGpu
from flint.procedural import (
    ENFORCER_BONE_COUNT,
    SubdCage,
    bake_conifer_needle_sprig_512,
    bake_cyber_billboard_512,
    bake_fissured_bark_256,
    bake_interior_windowbox_atlas_512,
    bake_oak_leaf_sprig_512,
    bake_palm_frond_512,
    build_articulated_crowd_pose_bank,
    build_botanical_oak,
    build_botanical_palm,
    build_botanical_pine,
    build_colonnade_arcade_bay,
    build_facade_mullion_spandrel_grid,
    build_flowering_bush,
    build_grass_wildflower_tuft,
    build_lathed_wheel_assembly,
    build_manicured_hedge_and_planter,
    build_rooftop_hvac_chiller,
    build_rooftop_vtol_helipad,
    build_rooftop_vtol_helipad_parts,
    build_street_bollard_and_grate,
    build_street_bollard_and_grate_parts,
    build_street_traffic_cantilever,
    build_street_traffic_cantilever_parts,
    build_subd_coaster_car_shell,
    build_subd_cyber_enforcer_rigged,
    build_subd_cyber_enforcer_rigged_v3,
    build_subd_spinner_hovercar_body_and_canopy,
    build_subd_supercar_body_and_glass,
    build_subd_traffic_vehicle,
    build_transit_shelter,
    build_transit_shelter_parts,
    build_urban_ginkgo_tree,
    creased_subdivide,
    extrude_profile_beveled,
    make_chamfered_girder_profile,
    make_curb_profile,
    subdivide_to_mesh,
    sweep_catenary_cable,
    sweep_profile_rmf,
)
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_renderer import FRAMES_IN_FLIGHT, FlintAaaVulkanRenderer
from flint.render.font import SdfFont
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import MaterialOverrideComponent, MeshRendererComponent
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, RENDER_TIER_2_AAA_VISBUFFER_RT, RenderWorld, ScalableRenderer
from flint.scene.scene import PrefabInstanceComponent
from flint.scene.transform_hierarchy import Children, GlobalTransform, HierarchyDepth, LocalTransform, Parent
from flint.studio import (
    COLLIDER_BOX,
    COLLIDER_SPHERE,
    FIELD_COLOR_RGB,
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
    StudioEntityMetaComponent,
    StudioStyleComponent,
    ViewportPickResult,
    ViewportRay,
    VisBufferGpuPicker,
    bake_procedural_to_glb,
    build_studio_reference_grid_mesh,
    build_subd_material_orb,
    decode_visbuffer_pick,
    emit_gizmo_draw_list,
    emit_studio_ui_draw_list,
    encode_visbuffer_words,
    fill_instance_entity_maps,
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
from flint.studio import scene_menu
from flint.studio.studio_app import _resolve_flint_launcher_path, populate_default_studio_scene
from flint.ui.context import UiContext, format_float2, pack_rgba8
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

    # 40-bit VisBuffer packing/decoding round-trip (instance_id > 1023, meshlet_id > 65535, 8-bit prim, 16-bit mat, sky=0)
    var w40 = encode_visbuffer_words(instance_id=131075, meshlet_id=200000, material_id=65535, local_prim_id=255)
    var p40 = decode_visbuffer_pick(
        px, py, w40[0], w40[1], depth_d32, w, h,
        cam_pos, cam_fwd, cam_rgt, cam_up, fov_y, near_z,
    )
    _assert(
        p40.hit
        and p40.instance_id == 131075
        and p40.meshlet_id == 200000
        and p40.material_id == 65535
        and p40.local_prim_id == 255,
        "40-bit VisBuffer encode/decode failed for large instance_id/meshlet_id",
    )
    var p_sky = decode_visbuffer_pick(
        px, py, UInt32(0), UInt32(0), depth_d32, w, h,
        cam_pos, cam_fwd, cam_rgt, cam_up, fov_y, near_z,
    )
    _assert(not p_sky.hit, "VisBuffer sky word0 == 0 must decode hit=False")

    # Bug 3 regression: NaN, -qNaN, +Inf, -Inf, sub-epsilon, huge depth, invalid camera/basis, and px/py clamping
    var nan_f32 = bitcast[DType.float32](UInt32(0x7FC00000))
    var qnan_neg = bitcast[DType.float32](UInt32(0xFFC00001))
    var pos_inf = bitcast[DType.float32](UInt32(0x7F800000))
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var bad_depths = List[Float32]()
    bad_depths.append(nan_f32)
    bad_depths.append(qnan_neg)
    bad_depths.append(pos_inf)
    bad_depths.append(neg_inf)
    bad_depths.append(Float32(1.0e-8))
    bad_depths.append(Float32(1.0e9))
    for di in range(len(bad_depths)):
        var p_bad = decode_visbuffer_pick(
            px, py, vis_words[0], vis_words[1], bad_depths[di], w, h,
            cam_pos, cam_fwd, cam_rgt, cam_up, fov_y, near_z, inst_map,
        )
        _assert(
            not p_bad.hit
            and p_bad.entity_id == -1
            and p_bad.linear_depth_m == Float32(0.0)
            and p_bad.world_pos.x == Float32(0.0)
            and p_bad.world_pos.y == Float32(0.0)
            and p_bad.world_pos.z == Float32(0.0),
            "non-finite or out-of-range depth_d32 must decode hit=False (idx=" + String(di) + ")",
        )

    var p_bad_near = decode_visbuffer_pick(
        px, py, vis_words[0], vis_words[1], depth_d32, w, h,
        cam_pos, cam_fwd, cam_rgt, cam_up, fov_y, Float32(0.0), inst_map,
    )
    var p_bad_fov = decode_visbuffer_pick(
        px, py, vis_words[0], vis_words[1], depth_d32, w, h,
        cam_pos, cam_fwd, cam_rgt, cam_up, Float32(180.0), near_z, inst_map,
    )
    var p_bad_jit = decode_visbuffer_pick(
        px, py, vis_words[0], vis_words[1], depth_d32, w, h,
        cam_pos, cam_fwd, cam_rgt, cam_up, fov_y, near_z, inst_map,
        jitter_clip_x=nan_f32,
    )
    var p_bad_pos = decode_visbuffer_pick(
        px, py, vis_words[0], vis_words[1], depth_d32, w, h,
        Vec3(nan_f32, 0.0, 0.0), cam_fwd, cam_rgt, cam_up, fov_y, near_z, inst_map,
    )
    var p_zero_fwd = decode_visbuffer_pick(
        px, py, vis_words[0], vis_words[1], depth_d32, w, h,
        cam_pos, Vec3(0.0, 0.0, 0.0), cam_rgt, cam_up, fov_y, near_z, inst_map,
    )
    var p_zero_rgt = decode_visbuffer_pick(
        px, py, vis_words[0], vis_words[1], depth_d32, w, h,
        cam_pos, cam_fwd, Vec3(0.0, 0.0, 0.0), cam_up, fov_y, near_z, inst_map,
    )
    _assert(
        not p_bad_near.hit
        and not p_bad_fov.hit
        and not p_bad_jit.hit
        and not p_bad_pos.hit
        and not p_zero_fwd.hit
        and not p_zero_rgt.hit,
        "invalid camera/projection/basis inputs must decode hit=False",
    )
    var p_clamp = decode_visbuffer_pick(
        -50, 99999, vis_words[0], vis_words[1], depth_d32, w, h,
        cam_pos, cam_fwd, cam_rgt, cam_up, fov_y, near_z, inst_map,
    )
    _assert(p_clamp.hit and p_clamp.world_pos.x == p_clamp.world_pos.x, "clamped out-of-bounds px/py must remain valid")

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
        var gpu_inst_bits = List[UInt64]()
        gpu_inst_bits.append(UInt64(0x123456789ABCDEF0))
        var gpu_pick = pick_renderer_viewport_once(ren, 80, 60, gpu_inst_map)
        _assert(gpu_pick.hit, "GPU VisBuffer center pixel pick must hit quad")
        _assert(gpu_pick.entity_id == 4242 and gpu_pick.instance_id == 0, "GPU pick entity/instance ID mismatch")
        _assert(gpu_pick.material_id == mat_id and gpu_pick.meshlet_id == 0, "GPU pick material/meshlet mismatch")
        var z_err = abs(gpu_pick.world_pos.z - Float32(3.5))
        _assert(z_err < Float32(0.01), "GPU Reverse-Z reconstructed depth error >= 1 cm: " + String(Float64(z_err)))

        var sky_pick = pick_renderer_viewport_once(ren, 2, 2, gpu_inst_map)
        _assert(not sky_pick.hit, "GPU VisBuffer corner sky pixel must report hit=False")

        # Async non-stalling FRAMES_IN_FLIGHT ring-buffered GPU picking + camera motion snapshot invariance
        ren.deferred_readback = True
        var async_picker = VisBufferGpuPicker()
        var harvested_picks = List[Tuple[UInt64, ViewportPickResult]]()

        # Frame 0: queue & dispatch center hit on slot 0
        var req_0 = async_picker.request_pick(80, 60, gpu_inst_map, gpu_inst_bits)
        ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=55.0, near_z=0.20)
        ren.draw_instance(me.mesh_id, Mat4(), material_override=mat_id, uniform_scale=1.0)
        _assert(ren.render_frame(capture_readback=False), "async frame 0 render failed")
        _assert(async_picker.dispatch_pending_pick(ren, gpu_inst_map, gpu_inst_bits), "async pick 0 dispatch failed")
        _assert(async_picker.total_submitted == 1, "expected total_submitted == 1 after frame 0")

        # Frame 1: queue & dispatch sky miss on slot 1 (overlapping FRAMES_IN_FLIGHT)
        var req_1 = async_picker.request_pick(2, 2, gpu_inst_map, gpu_inst_bits)
        ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=55.0, near_z=0.20)
        ren.draw_instance(me.mesh_id, Mat4(), material_override=mat_id, uniform_scale=1.0)
        _assert(ren.render_frame(capture_readback=False), "async frame 1 render failed")
        _assert(async_picker.dispatch_pending_pick(ren, gpu_inst_map, gpu_inst_bits), "async pick 1 dispatch failed")
        _assert(async_picker.total_submitted == 2, "expected total_submitted == 2 after frame 1")

        # Frame 2..16: move camera away while harvesting to verify camera snapshot invariance!
        for pf in range(16):
            while True:
                var polled = async_picker.poll_completed_pick_entry(ren)
                if not polled:
                    break
                harvested_picks.append(polled.value())
            if len(harvested_picks) >= 2:
                break
            var cam_x = Float32(10.0 + Float32(pf))
            ren.begin_frame(Vec3(cam_x, 3.0, -4.0), Vec3(cam_x, 3.0, 2.0), fov_y_deg=55.0, near_z=0.20)
            ren.draw_instance(me.mesh_id, Mat4(), material_override=mat_id, uniform_scale=1.0)
            _assert(ren.render_frame(capture_readback=False), "async pump frame failed")

        while len(harvested_picks) < 2 and async_picker.is_pick_inflight():
            ren.flush_deferred_frame()
            while True:
                var polled = async_picker.poll_completed_pick_entry(ren)
                if not polled:
                    break
                harvested_picks.append(polled.value())

        _assert(len(harvested_picks) == 2 and async_picker.total_harvested == 2, "async ring-buffer must harvest both picks")
        _assert(harvested_picks[0][0] == req_0 and harvested_picks[1][0] == req_1, "async pick request_id ordering mismatch")
        var hp0 = harvested_picks[0][1]
        var hp1 = harvested_picks[1][1]
        _assert(
            hp0.hit
            and hp0.entity_id == 4242
            and hp0.entity_bits == UInt64(0x123456789ABCDEF0)
            and hp0.material_id == mat_id
            and abs(hp0.world_pos.z - Float32(3.5)) < Float32(0.01),
            "async pick 0 snapshot-invariant decode failed",
        )
        _assert(not hp1.hit, "async pick 1 sky pixel must report hit=False")

        # Bug 2 regression: Option A preserves RenderGraph aliasing + keeps vis_img in LAYOUT_SHADER_READ_ONLY
        ren.flush_deferred_frame()
        var rg_m = ren.get_render_graph_aliasing_metrics()
        _assert(
            ren.vis_img.suballoc_block == -2
            and ren.post_alias_img.suballoc_block == -2
            and ren.render_graph.aliased_img_peak_bytes > 0
            and rg_m.compiled_barrier_count >= 7
            and ren.render_graph.aliasing_barriers_compiled > 0,
            "Bug 2 Option A must preserve all AaaRenderGraph aliasing allocations and metrics",
        )
        ren.deferred_readback = False
        ren.force_readback = False
        ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=55.0, near_z=0.20)
        ren.draw_instance(me.mesh_id, Mat4(), material_override=mat_id, uniform_scale=1.0)
        _assert(ren.render_frame(capture_readback=False, force_readback=False), "steady-state render_frame failed")
        _assert(
            ren.vis_img.layout == LAYOUT_SHADER_READ_ONLY
            and ren.depth_img.layout == LAYOUT_SHADER_READ_ONLY
            and ren.post_alias_img.layout == LAYOUT_UNDEFINED,
            "Bug 2 Option A: vis_img/depth_img must remain LAYOUT_SHADER_READ_ONLY and post_alias_img LAYOUT_UNDEFINED",
        )
        var steady_pick = pick_renderer_viewport_once(ren, 80, 60, gpu_inst_map)
        _assert(
            steady_pick.hit
            and steady_pick.entity_id == 4242
            and abs(steady_pick.world_pos.z - Float32(3.5)) < Float32(0.01),
            "post-frame pick after Bug 2 Option A fix must hit entity 4242 within 1 cm",
        )
        # Verify dispatch_pending_pick refuses LAYOUT_UNDEFINED vis_img and succeeds once restored
        _ = async_picker.request_pick(80, 60)
        ren.vis_img.layout = LAYOUT_UNDEFINED
        _assert(
            not async_picker.dispatch_pending_pick(ren, gpu_inst_map, gpu_inst_bits),
            "dispatch_pending_pick must return False when vis_img.layout != LAYOUT_SHADER_READ_ONLY",
        )
        ren.vis_img.layout = LAYOUT_SHADER_READ_ONLY
        _assert(
            async_picker.dispatch_pending_pick(ren, gpu_inst_map, gpu_inst_bits),
            "dispatch_pending_pick must succeed once vis_img.layout == LAYOUT_SHADER_READ_ONLY",
        )
        var b2_slot = (async_picker.write_cursor + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT
        if async_picker.slots[b2_slot].fence != 0:
            _ = ren.vk.wait(async_picker.slots[b2_slot].fence)
        var b2_got = async_picker.poll_completed_pick(ren)
        _assert(b2_got and b2_got.value().hit and b2_got.value().entity_id == 4242, "restored-layout pick failed")

        async_picker.shutdown(ren)
        ren.shutdown()

    # Bug 1 regression: Studio E2E instance-to-entity filtering (mesh_id=-1 / OOB), 1-arg vs explicit overload, and delete-all empty-map check
    var st_pick = FlintStudioState(headless=True, render_tier=RENDER_TIER_2_AAA_VISBUFFER_RT, width=320, height=240)
    var st_r_opt = st_pick.app.resources.get_ptr_mut[ScalableRenderer]()
    if st_r_opt and st_r_opt.value()[unsafe_offset=0].uses_vulkan():
        var quad_mb = MeshBuilder()
        var qv0 = quad_mb.add_vertex(Vec3(-0.75, -0.75, 0.0), Vec3(0.0, 0.0, -1.0), 0.0, 1.0)
        var qv1 = quad_mb.add_vertex(Vec3(0.75, -0.75, 0.0), Vec3(0.0, 0.0, -1.0), 1.0, 1.0)
        var qv2 = quad_mb.add_vertex(Vec3(0.75, 0.75, 0.0), Vec3(0.0, 0.0, -1.0), 1.0, 0.0)
        var qv3 = quad_mb.add_vertex(Vec3(-0.75, 0.75, 0.0), Vec3(0.0, 0.0, -1.0), 0.0, 0.0)
        quad_mb.add_tri(qv0, qv1, qv2)
        quad_mb.add_tri(qv0, qv2, qv3)
        var valid_mat_id = st_r_opt.value()[unsafe_offset=0].register_material(
            MaterialAsset(base_color=Vec3(0.2, 0.8, 0.3), roughness=0.4, metallic=0.1)
        )
        var valid_mesh_id = st_r_opt.value()[unsafe_offset=0].register_mesh(quad_mb^.finish(), valid_mat_id, build_lods=False)
        ref st_gpu = st_r_opt.value()[unsafe_offset=0].gpu_slot[0]
        st_gpu.tsr_history_weight = Float32(0.0)
        st_gpu.enable_vsm = False
        st_gpu.enable_rt_shadows = False
        st_gpu.enable_rt_reflections = False
        st_gpu.enable_rt_gi = False

        var bad_neg = st_pick.spawn_entity("BadNeg", position=Vec3(-4.0, 0.0, 4.0), mesh_id=-1, record_undo=False)
        var bad_oob = st_pick.spawn_entity("BadOob", position=Vec3(4.0, 0.0, 4.0), mesh_id=999, record_undo=False)
        var valid_first = st_pick.spawn_entity("ValidFirst", position=Vec3(-2.0, 0.0, 4.0), mesh_id=valid_mesh_id, record_undo=False)
        var valid_after_bad = st_pick.spawn_entity("ValidAfterBad", position=Vec3(0.0, 0.0, 4.0), mesh_id=valid_mesh_id, record_undo=False)
        var valid_after_bits = st_pick.resolve_entity(valid_after_bad).to_bits()

        st_pick.look_at(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 4.0))
        st_pick.render_editor_frame(320, 240)

        var inst_to_ent = List[Int]()
        var inst_to_bits = List[UInt64]()
        fill_instance_entity_maps(st_pick, inst_to_ent, inst_to_bits)
        _assert(
            len(inst_to_ent) == 2
            and len(inst_to_bits) == 2
            and inst_to_ent[0] == valid_first
            and inst_to_ent[1] == valid_after_bad
            and len(inst_to_ent) == len(st_gpu.frame_instances),
            "fill_instance_entity_maps must filter mesh_id=-1 and OOB mesh_id to match gpu.frame_instances",
        )

        var pt_proj = project_world_to_viewport(
            Vec3(0.0, 0.0, 4.0), 0.0, 0.0, 320.0, 240.0,
            st_pick.cam_pos, st_pick.cam_fwd, st_pick.cam_right, st_pick.cam_up, st_pick.fov_y_deg, Float32(0.15),
        )
        _assert(pt_proj[3], "ValidAfterBad center must project in front of camera")
        var px_v = Int(pt_proj[0])
        var py_v = Int(pt_proj[1])

        var st_picker = VisBufferGpuPicker()
        var req_explicit = st_picker.request_pick(px_v, py_v)
        _assert(
            st_picker.dispatch_pending_pick(st_gpu, inst_to_ent, inst_to_bits),
            "Studio E2E explicit-map dispatch_pending_pick failed",
        )
        _assert(
            len(st_picker.queued_inst_to_ent) == 0 and len(st_picker.queued_inst_to_bits) == 0,
            "dispatch_pending_pick must clear queued_inst_to_ent/bits upon successful dispatch",
        )
        var sl_exp = (st_picker.write_cursor + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT
        if st_picker.slots[sl_exp].fence != 0:
            _ = st_gpu.vk.wait(st_picker.slots[sl_exp].fence)
        var got_exp = st_picker.poll_completed_pick_entry(st_gpu)
        _assert(
            got_exp
            and got_exp.value()[0] == req_explicit
            and got_exp.value()[1].hit
            and got_exp.value()[1].entity_id == valid_after_bad
            and got_exp.value()[1].entity_bits == valid_after_bits,
            "ValidAfterBad explicit-map pick selected wrong entity or generation bits",
        )

        # 1-arg dispatch_pending_pick with maps passed at request_pick time
        var req_1arg = st_picker.request_pick(px_v, py_v, inst_to_ent, inst_to_bits)
        _assert(st_picker.dispatch_pending_pick(st_gpu), "1-arg dispatch_pending_pick failed")
        _assert(
            len(st_picker.queued_inst_to_ent) == 0 and len(st_picker.queued_inst_to_bits) == 0,
            "1-arg dispatch_pending_pick must clear queued_inst_to_ent/bits",
        )
        var sl_1arg = (st_picker.write_cursor + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT
        if st_picker.slots[sl_1arg].fence != 0:
            _ = st_gpu.vk.wait(st_picker.slots[sl_1arg].fence)
        var got_1arg = st_picker.poll_completed_pick_entry(st_gpu)
        _assert(
            got_1arg
            and got_1arg.value()[0] == req_1arg
            and got_1arg.value()[1].hit
            and got_1arg.value()[1].entity_id == valid_after_bad
            and got_1arg.value()[1].entity_bits == valid_after_bits,
            "1-arg dispatch_pending_pick failed to decode queued maps",
        )

        # Delete-all-entities empty-map check (explicit empty overload + 1-arg empty frame_instances)
        var req_del = st_picker.request_pick(160, 120, inst_to_ent, inst_to_bits)
        _ = st_pick.despawn_entity(bad_neg, record_undo=False)
        _ = st_pick.despawn_entity(bad_oob, record_undo=False)
        _ = st_pick.despawn_entity(valid_first, record_undo=False)
        _ = st_pick.despawn_entity(valid_after_bad, record_undo=False)
        st_pick.render_editor_frame(320, 240)
        fill_instance_entity_maps(st_pick, inst_to_ent, inst_to_bits)
        _assert(
            len(inst_to_ent) == 0 and len(inst_to_bits) == 0 and len(st_gpu.frame_instances) == 0,
            "delete-all-entities must yield empty inst_to_ent and frame_instances",
        )
        _assert(
            st_picker.dispatch_pending_pick(st_gpu, inst_to_ent, inst_to_bits),
            "empty-scene explicit dispatch_pending_pick failed",
        )
        var sl_del = (st_picker.write_cursor + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT
        _assert(
            len(st_picker.slots[sl_del].instance_to_entity) == 0
            and len(st_picker.slots[sl_del].instance_to_entity_bits) == 0
            and len(st_picker.queued_inst_to_ent) == 0
            and len(st_picker.queued_inst_to_bits) == 0,
            "empty-scene dispatch must NOT fall back to stale queued_inst_to_ent/bits",
        )
        if st_picker.slots[sl_del].fence != 0:
            _ = st_gpu.vk.wait(st_picker.slots[sl_del].fence)
        var got_del = st_picker.poll_completed_pick_entry(st_gpu)
        _assert(
            got_del and got_del.value()[0] == req_del and (not got_del.value()[1].hit) and got_del.value()[1].entity_id == -1,
            "empty-scene pick must report hit=False, entity_id=-1",
        )

        # Also verify 1-arg dispatch_pending_pick with 0 frame_instances ignores stale request_pick maps
        var stale_ents = List[Int]()
        stale_ents.append(valid_after_bad)
        var stale_bits = List[UInt64]()
        stale_bits.append(valid_after_bits)
        var req_del_1arg = st_picker.request_pick(160, 120, stale_ents, stale_bits)
        _assert(st_picker.dispatch_pending_pick(st_gpu), "empty-scene 1-arg dispatch_pending_pick failed")
        var sl_del_1arg = (st_picker.write_cursor + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT
        _assert(
            len(st_picker.slots[sl_del_1arg].instance_to_entity) == 0
            and len(st_picker.slots[sl_del_1arg].instance_to_entity_bits) == 0
            and len(st_picker.queued_inst_to_ent) == 0,
            "empty-scene 1-arg dispatch must drop stale queued_inst_to_ent when frame_instances is empty",
        )
        if st_picker.slots[sl_del_1arg].fence != 0:
            _ = st_gpu.vk.wait(st_picker.slots[sl_del_1arg].fence)
        var got_del_1arg = st_picker.poll_completed_pick_entry(st_gpu)
        _assert(
            got_del_1arg and got_del_1arg.value()[0] == req_del_1arg and (not got_del_1arg.value()[1].hit),
            "empty-scene 1-arg pick must report hit=False",
        )

        st_picker.shutdown(st_gpu)
    st_pick.shutdown()

    # Tier-0 / software fallback picking through VisBufferGpuPicker unified queue
    var sw_studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=320, height=240)
    sw_studio.look_at(Vec3(0.0, 1.0, -6.0), Vec3(0.0, 1.0, 0.0))
    var box_id = sw_studio.spawn_entity("SwPickBox", position=Vec3(0.0, 1.0, 0.0), collider_kind=COLLIDER_BOX, collider_half_extents=Vec3(0.75, 0.75, 0.75))
    var sw_picker = VisBufferGpuPicker()
    var sw_req = sw_picker.request_pick(160, 120)
    var sw_ray = ViewportRay.from_screen(
        160.0, 120.0, 0.0, 0.0, 320.0, 240.0,
        sw_studio.cam_pos, sw_studio.cam_fwd, sw_studio.cam_right, sw_studio.cam_up, sw_studio.fov_y_deg,
    )
    var sw_hit = sw_studio.raycast_scene(sw_ray.origin, sw_ray.dir, 100.0)
    _assert(sw_hit.hit and sw_hit.entity_id == box_id, "Tier-0 raycast_scene missed SwPickBox")
    var sw_res = ViewportPickResult()
    sw_res.hit = sw_hit.hit
    sw_res.entity_id = sw_hit.entity_id
    sw_res.entity_bits = sw_studio.resolve_entity(box_id).to_bits()
    sw_res.linear_depth_m = sw_hit.distance
    sw_res.world_pos = sw_hit.point
    sw_picker.push_software_pick_result(sw_req, sw_res)
    _assert(sw_picker.has_completed_pick() and sw_picker.total_harvested == 1, "software pick queue failed")
    sw_studio.shutdown()


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
        position=Vec3(-6.0, 5.0, 4.0),
        collider_kind=COLLIDER_SPHERE,
        collider_half_extents=Vec3(0.5, 0.5, 0.5),
        mass=8.0,
    )
    var fvis = FieldVisitor()
    studio.entities[drop_id].inspect(fvis)
    _assert(len(fvis.fields) == 36, "FieldVisitor must reflect all 36 StudioEntityView properties")

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
    _assert(studio.entities[drop_id].translation.y < Float32(1.5), "dynamic sphere must fall under gravity during PIE")
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
            # Socket clients must `initialize` before any tool call.
            var init_ok = cli.send_raw_request(
                '{"jsonrpc":"2.0","id":998,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"big-xfer"}}}',
                timeout_ms=4000,
            ) and cli.recv_response_blocking(timeout_ms=4000)[0]
            if init_ok and cli.send_raw_request(ctx.req_payload, timeout_ms=4000):
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

    # 2. tools/list -> all baseline 14 + M0 tools present (18 total)
    var r_list = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}', studio, assets, pool)
    var d_list = parse_json(_str_bytes(r_list))
    var tools_arr = d_list.object_get(d_list.object_get(d_list.root, "result"), "tools")
    _assert(d_list.valid and d_list.array_size(tools_arr) >= 18, "tools/list must expose all 14 baseline + 4 M0 MCP tools")
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

    # 9. Verify `auth_token` enforcement on `McpServerV2`
    var auth_mcp = McpServerV2(workspace_root=tmp_root, auth_token="flint-secret-token-42")
    var r_unauth = auth_mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":20,"method":"tools/list","params":{}}', studio, assets, pool)
    _assert(r_unauth.find("-32001") >= 0, "unauthenticated tools/list must fail with -32001 (unauthorized)")
    var r_bad_tok = auth_mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":21,"method":"initialize","params":{"auth_token":"wrong-token"}}', studio, assets, pool
    )
    _assert(r_bad_tok.find("-32001") >= 0, "wrong auth_token initialize must fail with -32001 (unauthorized)")
    var r_good_tok = auth_mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":22,"method":"initialize","params":{"auth_token":"flint-secret-token-42"}}', studio, assets, pool
    )
    _assert(r_good_tok.find('"protocolVersion"') >= 0, "valid auth_token initialize must succeed")

    # 10. Default `0700` runtime dir + `0600` socket mode + `SO_PEERCRED` verification
    var def_sock = default_mcp_socket_path()
    var def_listener = FlintMcpUnixListener(def_sock)
    _assert(def_listener.active, "default FlintMcpUnixListener failed to start")
    _assert(_stat_mode_octal(def_sock) == UInt32(0o600), "MCP Unix socket must have mode 0600")
    def_listener.close()

    var sock_path = tmp_root + "/mcp_v2.sock"
    var listener = FlintMcpUnixListener(sock_path)
    _assert(listener.active and _stat_mode_octal(sock_path) == UInt32(0o600), "FlintMcpUnixListener failed to bind 0600 socket")

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

    # Before `initialize` a socket client may only ping: tool calls are refused with -32002.
    _assert(
        client.send_raw_request('{"jsonrpc":"2.0","id":1011,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{}}}'),
        "send pre-init tools/call failed",
    )
    _ = listener.poll_once(mcp, studio, assets, pool)
    var got_pre = client.recv_response_blocking(timeout_ms=500)
    _assert(got_pre[0] and got_pre[1].find("-32002") >= 0, "pre-initialize tools/call must be refused with -32002: " + got_pre[1])
    _assert(
        client.send_raw_request('{"jsonrpc":"2.0","id":1012,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"sock-test"}}}'),
        "send socket initialize failed",
    )
    _ = listener.poll_once(mcp, studio, assets, pool)
    var got_init = client.recv_response_blocking(timeout_ms=500)
    _assert(
        got_init[0] and got_init[1].find('"protocolVersion":"2025-06-18"') >= 0 and got_init[1].find('"listChanged":false') >= 0,
        "socket initialize must negotiate 2025-06-18 and advertise listChanged:false: " + got_init[1],
    )

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

    # Socket clients start gated (`MCP_GATE_MUTATING`); this transfer test opts every client into AUTO.
    mcp.set_gate_mode(MCP_GATE_AUTO)
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


def _append_u32_le(mut buf: List[UInt8], v: UInt32):
    buf.append(UInt8(v & UInt32(0xFF)))
    buf.append(UInt8((v >> UInt32(8)) & UInt32(0xFF)))
    buf.append(UInt8((v >> UInt32(16)) & UInt32(0xFF)))
    buf.append(UInt8((v >> UInt32(24)) & UInt32(0xFF)))


def test_g_audit_regressions() raises:
    print("  [g] Testing drag_float3 formatting, UiContext GC, despawn reparenting, cross-pool LRU & watcher zero-len resilience...")

    # 1. drag_float3 / format_float2 formatting regressions
    _assert(format_float2(Float32(0.05)) == "0.05", "format_float2(0.05) must be '0.05', got: " + format_float2(Float32(0.05)))
    _assert(format_float2(Float32(-0.5)) == "-0.50", "format_float2(-0.5) must be '-0.50', got: " + format_float2(Float32(-0.5)))
    _assert(format_float2(Float32(-0.001)) == "0.00", "format_float2(-0.001) must be '0.00'")
    _assert(format_float2(Float32(-1.05)) == "-1.05", "format_float2(-1.05) must be '-1.05'")
    _assert(format_float2(Float32(12.34)) == "12.34", "format_float2(12.34) must be '12.34'")

    # 2. Unified 0xAARRGGBB color packing
    _assert(pack_rgba8(255, 32, 64, 255) == UInt32(0xFFFF2040), "pack_rgba8 must pack 0xAARRGGBB")

    # 3. UiContext frame-epoch GC evicts stale cached_rects & state_map entries
    var uctx = UiContext()
    uctx.gc_stale_frames = 2
    var inp = InputState()
    uctx.begin_frame(inp, 800.0, 600.0, 1.0 / 60.0)
    var s_tmp = String("temp")
    _ = uctx.text_input("ephemeral_input", s_tmp, 120.0)
    uctx.end_frame()
    _assert(len(uctx.cached_rects) >= 1 and len(uctx.state_map) >= 1, "ephemeral widget must populate caches")
    for _ in range(5):
        uctx.begin_frame(inp, 800.0, 600.0, 1.0 / 60.0)
        uctx.end_frame()
    _assert(len(uctx.cached_rects) == 0 and len(uctx.state_map) == 0, "UiContext frame-epoch GC must evict stale entries")

    # 4. Studio despawn_entity detaches child parent_ids & undo/redo restores cleanly
    var st = FlintStudioState()
    var pid = st.spawn_entity("ParentNode", record_undo=True)
    var cid = st.spawn_entity("ChildNode", parent_id=pid, record_undo=True)
    var h_before = st.state_hash()
    _assert(st.despawn_entity(pid, record_undo=True), "despawn parent failed")
    _assert(st.entities[cid].parent_id == -1, "despawn_entity must detach live children to parent_id == -1")
    _assert(st.undo() and st.entities[cid].parent_id == pid and st.state_hash() == h_before, "undo despawn must restore child parent_id")
    _assert(st.redo() and st.entities[cid].parent_id == -1, "redo despawn must re-detach child parent_id")

    # 5. Cross-pool LRU eviction (40% + 40% + 30% = 110% > 100%, each pool < 50%!)
    var srv = AssetServer(stream_budget_bytes_per_tick=4096, lru_max_bytes=100000, enable_watcher=False)
    var tex_a = CookedTextureAsset()
    tex_a.mips.append(List[UInt8](length=40000, fill=UInt8(1)))
    var h_tex = srv.textures.insert(tex_a^, AssetUuid.from_path("lru_tex.png"), "lru_tex.png", tick=1)
    _ = srv.textures.release(h_tex)

    var hdr_a = CookedHdrAsset()
    hdr_a.mips.append(List[UInt8](length=40000, fill=UInt8(2)))
    var h_hdr = srv.hdrs.insert(hdr_a^, AssetUuid.from_path("lru_sky.hdr"), "lru_sky.hdr", tick=2)
    _ = srv.hdrs.release(h_hdr)

    var h_raw = srv.raws.insert(RawBytesAsset(List[UInt8](length=30000, fill=UInt8(3))), AssetUuid.from_path("lru_blob.bin"), "lru_blob.bin", tick=3)
    _ = srv.raws.release(h_raw)

    _assert(srv.total_resident_bytes() == 110000, "pre-eviction resident bytes must be 110000")
    _ = srv.tick(now=1.0)
    _assert(
        srv.total_resident_bytes() <= srv.lru_max_bytes,
        "cross-pool LRU eviction must bring total_resident_bytes <= lru_max_bytes (got " + String(srv.total_resident_bytes()) + ")",
    )
    _assert(not srv.textures.is_alive(h_tex), "oldest unreferenced asset (tick=1 texture) must be evicted first")
    _assert(srv.hdrs.is_alive(h_hdr) and srv.raws.is_alive(h_raw), "newer unreferenced assets should remain within budget")

    # 6. AssetWatcher zero-length inotify event resilience (IN_Q_OVERFLOW + IN_IGNORED before valid file event)
    var w = AssetWatcher(enabled=False)
    w.watch_descriptors.append(Int32(7))
    w.watch_folders.append("/tmp/fake_root")
    w.watch_roots.append("/tmp/fake_root")
    var ev_buf = List[UInt8]()
    # Event 1: IN_Q_OVERFLOW (wd = -1, len = 0)
    _append_u32_le(ev_buf, UInt32(0xFFFFFFFF))
    _append_u32_le(ev_buf, IN_Q_OVERFLOW)
    _append_u32_le(ev_buf, UInt32(0))
    _append_u32_le(ev_buf, UInt32(0))
    # Event 2: IN_IGNORED (wd = 7, len = 0)
    _append_u32_le(ev_buf, UInt32(7))
    _append_u32_le(ev_buf, IN_IGNORED)
    _append_u32_le(ev_buf, UInt32(0))
    _append_u32_le(ev_buf, UInt32(0))
    # Event 3: Valid IN_CLOSE_WRITE for "hero.png" (wd = 7, len = 16 padded with NULs)
    _append_u32_le(ev_buf, UInt32(7))
    _append_u32_le(ev_buf, IN_CLOSE_WRITE)
    _append_u32_le(ev_buf, UInt32(0))
    _append_u32_le(ev_buf, UInt32(16))
    var fname = String("hero.png")
    for b in fname.as_bytes():
        ev_buf.append(b)
    for _ in range(16 - fname.byte_length()):
        ev_buf.append(UInt8(0))

    var parsed = w.process_inotify_buffer(ev_buf, len(ev_buf), 10.0)
    _assert(parsed == 3, "must parse all 3 inotify events without breaking on zero-length events")
    _assert(len(w.pending_paths) == 1 and w.pending_paths[0] == "hero.png", "hero.png must survive preceding IN_Q_OVERFLOW/IN_IGNORED events")
    print("    [OK] drag_float3 formatting, UiContext GC, despawn reparenting, cross-pool LRU & watcher zero-len resilience passed!")


def _build_uncompressed_png(
    width: Int,
    height: Int,
    bit_depth: Int,
    color_type: Int,
    raw_scanlines: List[UInt8],
    plte: List[UInt8] = List[UInt8](),
    trns: List[UInt8] = List[UInt8](),
) -> List[UInt8]:
    """Builds a spec-compliant uncompressed PNG byte stream for arbitrary color types & bit depths."""
    var out = List[UInt8]()
    var sig = [137, 80, 78, 71, 13, 10, 26, 10]
    for v in sig:
        out.append(UInt8(v))
    var ihdr = List[UInt8]()
    append_be32(ihdr, width)
    append_be32(ihdr, height)
    ihdr.append(UInt8(bit_depth))
    ihdr.append(UInt8(color_type))
    ihdr.append(UInt8(0))
    ihdr.append(UInt8(0))
    ihdr.append(UInt8(0))
    append_chunk(out, 0x49484452, ihdr)
    if len(plte) > 0:
        append_chunk(out, 0x504C5445, plte)
    if len(trns) > 0:
        append_chunk(out, 0x74524E53, trns)
    # Stored-block zlib stream
    var z = List[UInt8]()
    z.append(UInt8(0x78))
    z.append(UInt8(0x01))
    var n = len(raw_scanlines)
    z.append(UInt8(1))  # BFINAL=1, BTYPE=00
    z.append(UInt8(n & 0xFF))
    z.append(UInt8((n >> 8) & 0xFF))
    var inv = n ^ 0xFFFF
    z.append(UInt8(inv & 0xFF))
    z.append(UInt8((inv >> 8) & 0xFF))
    var a = 1
    var b = 0
    for byte in raw_scanlines:
        z.append(byte)
        a = (a + Int(byte)) % 65521
        b = (b + a) % 65521
    append_be32(z, (b << 16) | a)
    append_chunk(out, 0x49444154, z)
    append_chunk(out, 0x49454E44, List[UInt8]())
    return out^


def test_h_wave2_png_async_studio_scene(tmp_root: String, pack_path: String, mut pool: JobPool) raises:
    print("  [h] Testing hardened PNG decoder (gray/palette/gray+alpha/16-bit/box-filter), async AssetServer & Studio .flscene...")

    # 1a. Grayscale 8-bit PNG (color_type=0)
    var raw_g8 = List[UInt8]()
    raw_g8.append(UInt8(0)); raw_g8.append(UInt8(0)); raw_g8.append(UInt8(64))
    raw_g8.append(UInt8(0)); raw_g8.append(UInt8(128)); raw_g8.append(UInt8(255))
    var png_g8 = _build_uncompressed_png(2, 2, 8, 0, raw_g8)
    var img_g8 = decode_png(png_g8)
    _assert(img_g8.valid and img_g8.width == 2 and img_g8.height == 2, "grayscale 8-bit PNG decode failed")
    _assert(img_g8.pixels[0] == UInt32(0xFF000000) and img_g8.pixels[1] == UInt32(0xFF404040), "gray8 row0 mismatch")
    _assert(img_g8.pixels[2] == UInt32(0xFF808080) and img_g8.pixels[3] == UInt32(0xFFFFFFFF), "gray8 row1 mismatch")

    # 1b. Indexed/Palette PNG (color_type=3 + PLTE + tRNS)
    var plte = List[UInt8]()
    plte.append(UInt8(255)); plte.append(UInt8(0)); plte.append(UInt8(0))
    plte.append(UInt8(0)); plte.append(UInt8(255)); plte.append(UInt8(0))
    var trns = List[UInt8]()
    trns.append(UInt8(128))
    var raw_pal = List[UInt8]()
    raw_pal.append(UInt8(0)); raw_pal.append(UInt8(0)); raw_pal.append(UInt8(1))
    var png_pal = _build_uncompressed_png(2, 1, 8, 3, raw_pal, plte, trns)
    var img_pal = decode_png(png_pal)
    _assert(img_pal.valid and img_pal.width == 2 and img_pal.height == 1, "palette PNG decode failed")
    _assert(img_pal.pixels[0] == UInt32(0x80FF0000), "palette+tRNS pixel 0 mismatch")
    _assert(img_pal.pixels[1] == UInt32(0xFF00FF00), "palette pixel 1 mismatch")

    # 1c. Grayscale+Alpha 8-bit PNG (color_type=4)
    var raw_ga = List[UInt8]()
    raw_ga.append(UInt8(0)); raw_ga.append(UInt8(100)); raw_ga.append(UInt8(50)); raw_ga.append(UInt8(200)); raw_ga.append(UInt8(255))
    var png_ga = _build_uncompressed_png(2, 1, 8, 4, raw_ga)
    var img_ga = decode_png(png_ga)
    _assert(img_ga.valid and img_ga.pixels[0] == UInt32(0x32646464) and img_ga.pixels[1] == UInt32(0xFFC8C8C8), "gray+alpha PNG mismatch")

    # 1d. 16-bit Grayscale PNG (color_type=0, bit_depth=16)
    var raw_g16 = List[UInt8]()
    raw_g16.append(UInt8(0))
    raw_g16.append(UInt8(0x00)); raw_g16.append(UInt8(0x00))
    raw_g16.append(UInt8(0x80)); raw_g16.append(UInt8(0x00))
    var png_g16 = _build_uncompressed_png(2, 1, 16, 0, raw_g16)
    var img_g16 = decode_png(png_g16)
    _assert(img_g16.valid and img_g16.bit_depth == 16 and len(img_g16.pixels_u16) == 2, "16-bit PNG decode failed")
    _assert(img_g16.pixels_u16[0] == UInt16(0) and img_g16.pixels_u16[1] == UInt16(0x8000), "16-bit heightmap values mismatch")
    _assert((img_g16.pixels[1] & UInt32(0xFF)) == UInt32(128), "16-bit -> 8-bit downscale mismatch")

    # 1e. Box-filter downscaling (4x4 checkerboard 0/255 -> 2x2 averages to ~128, NOT 0!)
    var raw_chk = List[UInt8]()
    for y in range(4):
        raw_chk.append(UInt8(0))
        for x in range(4):
            raw_chk.append(UInt8(255) if ((x + y) & 1) == 1 else UInt8(0))
    var png_chk = _build_uncompressed_png(4, 4, 8, 0, raw_chk)
    var img_chk = decode_png(png_chk, maximum_size=2)
    _assert(img_chk.valid and img_chk.width == 2 and img_chk.height == 2, "box-filter downscale dimensions mismatch")
    var ch0 = Int(img_chk.pixels[0] & UInt32(0xFF))
    _assert(ch0 >= 126 and ch0 <= 129, "box-filter downscale must average checkerboard to ~128, got " + String(ch0))

    # 2. Async JobPool-backed AssetServer state transitions (LOADING -> READY / FAILED)
    var asrv = AssetServer(stream_budget_bytes_per_tick=4096, lru_max_bytes=16 * 1024 * 1024, enable_watcher=False)
    _assert(asrv.mount_flpk(pack_path), "mount_flpk failed for async test")
    var h_m = asrv.load_mesh_async("meshes/forge_box.glb")
    var h_t = asrv.load_texture_async("textures/albedo_256.png", progressive=True)
    var h_bad = asrv.load_mesh_async("meshes/nonexistent_asset.glb")
    _assert(asrv.mesh_state(h_m) == ASSET_STATE_LOADING, "async mesh handle must start in ASSET_STATE_LOADING")
    _assert(asrv.texture_state(h_t) == ASSET_STATE_LOADING, "async texture handle must start in ASSET_STATE_LOADING")
    _assert(asrv.mesh_state(h_bad) == ASSET_STATE_LOADING, "missing async handle must start in ASSET_STATE_LOADING")
    _assert(asrv.pending_async_count() == 3, "expected 3 pending async jobs")

    var done = asrv.poll_async_loads(pool)
    _assert(done == 3 and asrv.pending_async_count() == 0, "poll_async_loads must complete all 3 jobs")
    _assert(asrv.mesh_state(h_m) == ASSET_STATE_READY and asrv.meshes.is_loaded(h_m), "async mesh must reach ASSET_STATE_READY")
    _assert(asrv.texture_state(h_t) == ASSET_STATE_READY and asrv.textures.is_loaded(h_t), "async texture must reach ASSET_STATE_READY")
    _assert(asrv.mesh_state(h_bad) == ASSET_STATE_FAILED and not asrv.meshes.is_loaded(h_bad), "missing async mesh must reach ASSET_STATE_FAILED")
    asrv.close()

    # 3. Studio .flscene disk Save & Load round-trip + Duplicate Undo/Redo + Asset Browser scan
    var st = FlintStudioState()
    var e1 = st.spawn_entity("HeroMesh", position=Vec3(1.5, 2.0, -3.0), mass=4.5, mesh_id=1)
    _ = st.configure_entity_npr_cel(e1, cel_outline_width_px=2.5, anim_step_hz=12.0)
    var h_pre_dup = st.state_hash()
    var e2 = st.duplicate_entity(e1, record_undo=True)
    _assert(e2 >= 0 and st.entities[e2].name == "HeroMesh_Copy", "duplicate_entity failed")
    _assert(st.undo() and st.state_hash() == h_pre_dup, "undo duplicate_entity mismatch")
    _assert(st.redo() and st.entities[e2].alive, "redo duplicate_entity mismatch")

    var scene_path = tmp_root + "/roundtrip.flscene"
    var h_saved = st.state_hash()
    _assert(st.save_scene_file(scene_path), "save_scene_file failed")

    var st_loaded = FlintStudioState()
    _assert(st_loaded.load_scene_file(scene_path), "load_scene_file failed")
    _assert(st_loaded.state_hash() == h_saved, "loaded .flscene state_hash must match saved scene bit-for-bit")
    var scanned = st_loaded.scan_asset_browser(dir_path=tmp_root, pack_path=pack_path)
    _assert(scanned >= 2, "scan_asset_browser must discover pack & workspace entries")
    print("    [OK] Hardened PNG decoder, async AssetServer & Studio .flscene round-trip passed!")


def _test_put_u16_le(mut out: List[UInt8], v: Int):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))


def _test_put_u32_le(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


def _test_put_u64_le(mut out: List[UInt8], v: UInt64):
    _test_put_u32_le(out, UInt32(v & 0xFFFFFFFF))
    _test_put_u32_le(out, UInt32((v >> 32) & 0xFFFFFFFF))


def _test_put_u16_be(mut out: List[UInt8], v: Int):
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _wrap_zlib_stored(raw: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    out.append(0x78)
    out.append(0x01)
    var pos = 0
    while pos < len(raw) or len(raw) == 0:
        var chunk = min(65535, len(raw) - pos)
        var last = 1 if (pos + chunk >= len(raw)) else 0
        out.append(UInt8(last))
        _test_put_u16_le(out, chunk)
        _test_put_u16_le(out, chunk ^ 0xFFFF)
        for i in range(chunk):
            out.append(raw[pos + i])
        pos += chunk
        if len(raw) == 0:
            break
    # Adler32
    var s1: UInt32 = 1
    var s2: UInt32 = 0
    for b in raw:
        s1 = (s1 + UInt32(b)) % 65521
        s2 = (s2 + s1) % 65521
    var adler = (s2 << 16) | s1
    out.append(UInt8((adler >> 24) & 0xFF))
    out.append(UInt8((adler >> 16) & 0xFF))
    out.append(UInt8((adler >> 8) & 0xFF))
    out.append(UInt8(adler & 0xFF))
    return out^


def _wrap_zstd_frame(raw: List[UInt8], rle_byte: Int = -1, rle_len: Int = 0) -> List[UInt8]:
    var out = List[UInt8]()
    _test_put_u32_le(out, 0xFD2FB528)
    out.append(0x20)  # single segment, 1-byte FCS
    var total = rle_len if rle_byte >= 0 else len(raw)
    out.append(UInt8(total))
    var btype = 1 if rle_byte >= 0 else 0
    var bh = (total << 3) | (btype << 1) | 1
    out.append(UInt8(bh & 0xFF))
    out.append(UInt8((bh >> 8) & 0xFF))
    out.append(UInt8((bh >> 16) & 0xFF))
    if rle_byte >= 0:
        out.append(UInt8(rle_byte))
    else:
        for b in raw:
            out.append(b)
    return out^


def _build_ktx2_bytes(
    w: Int,
    h: Int,
    vk_fmt: Int,
    supercomp: Int,
    color_model: Int,
    level_payload: List[UInt8],
    uncomp_len: Int,
    sgd: List[UInt8] = List[UInt8](),
) -> List[UInt8]:
    var out = List[UInt8]()
    var magic: List[UInt8] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A]
    for b in magic:
        out.append(b)
    _test_put_u32_le(out, UInt32(vk_fmt))
    _test_put_u32_le(out, 1)  # typeSize
    _test_put_u32_le(out, UInt32(w))
    _test_put_u32_le(out, UInt32(h))
    _test_put_u32_le(out, 0)  # depth
    _test_put_u32_le(out, 0)  # layerCount
    _test_put_u32_le(out, 1)  # faceCount
    _test_put_u32_le(out, 1)  # levelCount
    _test_put_u32_le(out, UInt32(supercomp))

    var dfd_off = 80 + 24
    var dfd_len = 44
    var sgd_off = dfd_off + dfd_len
    var sgd_len = len(sgd)
    var data_off = sgd_off + sgd_len

    _test_put_u32_le(out, UInt32(dfd_off))
    _test_put_u32_le(out, UInt32(dfd_len))
    _test_put_u32_le(out, 0)
    _test_put_u32_le(out, 0)
    _test_put_u64_le(out, UInt64(sgd_off if sgd_len > 0 else 0))
    _test_put_u64_le(out, UInt64(sgd_len))

    # Level 0 index (24 bytes)
    _test_put_u64_le(out, UInt64(data_off))
    _test_put_u64_le(out, UInt64(len(level_payload)))
    _test_put_u64_le(out, UInt64(uncomp_len))

    # Standard 44-byte Khronos DFD v1.3 (+8 versionNumber=2, +10 blockSize=40, +12 colorModel)
    _test_put_u32_le(out, UInt32(dfd_len))
    _test_put_u32_le(out, 0)
    _test_put_u16_le(out, 2)
    _test_put_u16_le(out, dfd_len - 4)
    out.append(UInt8(color_model))
    out.append(1); out.append(1); out.append(0)
    for _ in range(28):
        out.append(0)

    for b in sgd:
        out.append(b)
    for b in level_payload:
        out.append(b)
    return out^


def _exr_rle_compress(raw: List[UInt8]) -> List[UInt8]:
    """Valid OpenEXR RLE compressor (literal runs)."""
    var out = List[UInt8]()
    var pos = 0
    while pos < len(raw):
        var run = min(127, len(raw) - pos)
        out.append(bitcast[DType.uint8](Int8(-run)))
        for i in range(run):
            out.append(raw[pos + i])
        pos += run
    return out^


def _exr_reorder_and_predict(raw: List[UInt8]) -> List[UInt8]:
    """Inverse of `_exr_undo_predictor_and_reorder` for building valid ZIP/RLE/DWA test fixtures."""
    var n = len(raw)
    if n <= 1:
        return raw.copy()
    var pred = List[UInt8](length=n, fill=0)
    var half = (n + 1) // 2
    var i1 = 0
    var i2 = half
    var j = 0
    while j < n:
        pred[i1] = raw[j]
        i1 += 1; j += 1
        if j < n:
            pred[i2] = raw[j]
            i2 += 1; j += 1
    var out = List[UInt8](length=n, fill=0)
    out[0] = pred[0]
    for i in range(1, n):
        out[i] = UInt8((Int(pred[i]) - Int(pred[i - 1]) + 128) & 0xFF)
    return out^


def _exr_wenc14(a: UInt16, b: UInt16) -> Tuple[UInt16, UInt16]:
    var ai = Int(a)
    var bi = Int(b)
    var hi = (ai - bi) & 0xFFFF
    var h_s = Int(bitcast[DType.int16](UInt16(hi)))
    var li = (ai - (h_s & 1) - (h_s >> 1)) & 0xFFFF
    return (UInt16(li), UInt16(hi))


def _exr_wav2_encode(mut buf: List[UInt16], base: Int, nx: Int, ox: Int, ny: Int, oy: Int):
    var n = nx if nx < ny else ny
    var p = 1
    var p2 = 2
    while p2 <= n:
        var y_end = oy * (ny - p2)
        var x_end = ox * (nx - p2)
        var iy = 0
        while iy <= y_end:
            var ix = 0
            while ix <= x_end:
                var i00 = base + iy + ix
                var i01 = i00 + ox * p
                var i10 = i00 + oy * p
                var i11 = i10 + ox * p
                var h0 = _exr_wenc14(buf[i00], buf[i01]); buf[i00] = h0[0]; buf[i01] = h0[1]
                var h1 = _exr_wenc14(buf[i10], buf[i11]); buf[i10] = h1[0]; buf[i11] = h1[1]
                var v0 = _exr_wenc14(buf[i00], buf[i10]); buf[i00] = v0[0]; buf[i10] = v0[1]
                var v1 = _exr_wenc14(buf[i01], buf[i11]); buf[i01] = v1[0]; buf[i11] = v1[1]
                ix += ox * p2
            if (nx & p) != 0:
                var i00 = base + iy + ix
                var i10 = i00 + oy * p
                var v0 = _exr_wenc14(buf[i00], buf[i10]); buf[i00] = v0[0]; buf[i10] = v0[1]
            iy += oy * p2
        if (ny & p) != 0:
            var ix = 0
            while ix <= x_end:
                var i00 = base + iy + ix
                var i01 = i00 + ox * p
                var h0 = _exr_wenc14(buf[i00], buf[i01]); buf[i00] = h0[0]; buf[i01] = h0[1]
                ix += ox * p2
        p = p2
        p2 <<= 1


struct _ExrMsbWriter(Movable):
    var out: List[UInt8]
    var buf: UInt64
    var cnt: Int
    var total_bits: Int

    def __init__(out self):
        self.out = List[UInt8]()
        self.buf = 0
        self.cnt = 0
        self.total_bits = 0

    def write_msb(mut self, val: UInt64, bits: Int):
        for i in range(bits):
            var b = (val >> UInt64(bits - 1 - i)) & 1
            self.buf = (self.buf << 1) | b
            self.cnt += 1
            self.total_bits += 1
            if self.cnt == 8:
                self.out.append(UInt8(self.buf & 0xFF))
                self.buf = 0
                self.cnt = 0

    def flush(mut self):
        if self.cnt > 0:
            self.out.append(UInt8((self.buf << UInt64(8 - self.cnt)) & 0xFF))
            self.buf = 0
            self.cnt = 0


def _build_exr_huf_blob(vals: List[UInt16]) -> List[UInt8]:
    """Minimal valid OpenEXR MSB-first canonical Huffman encoder for test fixtures."""
    var max_s = 0
    for v in vals:
        if Int(v) > max_s:
            max_s = Int(v)
    var sym_limit = max(1, max_s + 1)
    # Assign uniform 8-bit or 16-bit code length across 0..max_s
    var blen = 1
    while (1 << blen) <= sym_limit:
        blen += 1
    var im = 0
    var iM = max_s + 1  # rle_sym
    var tw = _ExrMsbWriter()
    for _ in range(im, iM + 1):
        tw.write_msb(UInt64(blen), 6)
    tw.flush()

    var first_code = UInt64(0)
    var dw = _ExrMsbWriter()
    for v in vals:
        dw.write_msb(first_code + UInt64(v), blen)
    var n_bits = dw.total_bits
    dw.flush()

    var out = List[UInt8]()
    _test_put_u32_le(out, UInt32(im))
    _test_put_u32_le(out, UInt32(iM))
    _test_put_u32_le(out, UInt32(len(tw.out)))
    _test_put_u32_le(out, UInt32(n_bits))
    _test_put_u32_le(out, 0)
    for b in tw.out:
        out.append(b)
    for b in dw.out:
        out.append(b)
    return out^


def _write_exr_header(mut out: List[UInt8], w: Int, h: Int, compression: Int, tiled: Bool, tile_x: Int = 4, tile_y: Int = 4, mipmap: Bool = False):
    _test_put_u32_le(out, 20000630)
    _test_put_u32_le(out, UInt32(2 | (0x200 if tiled else 0)))
    for attr in ["channels", "chlist"]:
        for b in attr.as_bytes():
            out.append(b)
        out.append(0)
    _test_put_u32_le(out, 3 * 18 + 1)
    for ch in ["B", "G", "R"]:
        for b in ch.as_bytes():
            out.append(b)
        out.append(0)
        _test_put_u32_le(out, 1)  # HALF
        out.append(1); out.append(0); out.append(0); out.append(0)
        _test_put_u32_le(out, 1); _test_put_u32_le(out, 1)
    out.append(0)
    for attr in ["compression", "compression"]:
        for b in attr.as_bytes():
            out.append(b)
        out.append(0)
    _test_put_u32_le(out, 1)
    out.append(UInt8(compression))
    for win in ["dataWindow", "displayWindow"]:
        for b in win.as_bytes():
            out.append(b)
        out.append(0)
        for b in "box2i".as_bytes():
            out.append(b)
        out.append(0)
        _test_put_u32_le(out, 16)
        _test_put_u32_le(out, 0); _test_put_u32_le(out, 0)
        _test_put_u32_le(out, UInt32(w - 1)); _test_put_u32_le(out, UInt32(h - 1))
    if tiled:
        for attr in ["tiles", "tiledesc"]:
            for b in attr.as_bytes():
                out.append(b)
            out.append(0)
        _test_put_u32_le(out, 9)
        _test_put_u32_le(out, UInt32(tile_x))
        _test_put_u32_le(out, UInt32(tile_y))
        out.append(UInt8(1 if mipmap else 0))
    out.append(0)


struct _Sof2BitWriter(Movable):
    var out: List[UInt8]
    var buf: UInt32
    var count: Int

    def __init__(out self):
        self.out = List[UInt8]()
        self.buf = 0
        self.count = 0

    def write_bits(mut self, code: Int, nbits: Int):
        if nbits <= 0:
            return
        self.buf = (self.buf << UInt32(nbits)) | UInt32(code & ((1 << nbits) - 1))
        self.count += nbits
        while self.count >= 8:
            self.count -= 8
            var b = UInt8((self.buf >> UInt32(self.count)) & 0xFF)
            self.out.append(b)
            if b == 0xFF:
                self.out.append(0x00)

    def flush_and_rst(mut self, rst_idx: Int):
        if self.count > 0:
            var pad = 8 - self.count
            self.write_bits((1 << pad) - 1, pad)
        self.count = 0
        self.buf = 0
        if rst_idx >= 0:
            self.out.append(0xFF)
            self.out.append(UInt8(0xD0 + (rst_idx & 7)))


def test_i_f20_residency_and_f21_codecs(tmp_root: String) raises:
    print("  [i] Testing F20 loose_root font/audio + Vulkan residency mip upload/eviction & F21 GLB/KTX2/SOF2/EXR codecs...")

    # ---- F20: AssetServer loose_root load_font / load_audio + hot reload ----
    var f20_ws = tmp_root + "/f20_loose"
    _ = make_dirs(f20_ws + "/audio")
    _ = make_dirs(f20_ws + "/fonts")
    var wav = _make_minimal_wav_16bit()
    _assert(write_bytes(f20_ws + "/audio/loose_tone.wav", wav), "write loose wav failed")
    var ttf_src = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    var ttf_bytes = read_bytes(ttf_src)
    var has_sys_ttf = len(ttf_bytes) > 0
    if has_sys_ttf:
        _assert(write_bytes(f20_ws + "/fonts/mono.ttf", ttf_bytes), "write loose ttf failed")

    var fsrv = AssetServer(stream_budget_bytes_per_tick=65536, enable_watcher=False)
    _ = fsrv.watch_loose_directory(f20_ws)
    var h_aud = fsrv.load_audio("audio/loose_tone.wav")
    _assert(h_aud.is_valid() and fsrv.audios.is_loaded(h_aud), "loose_root load_audio failed")
    var aud_p = fsrv.audios.borrow(h_aud)
    _assert(aud_p[].channels == 1 and aud_p[].sample_rate == 48000 and len(aud_p[].samples) == 64, "loose audio PCM mismatch")
    var ep_aud0 = fsrv.audios.reload_epoch(h_aud)
    _assert(fsrv._hot_reload_path("audio/loose_tone.wav"), "hot reload loose audio failed")
    _assert(fsrv.audios.reload_epoch(h_aud) > ep_aud0, "hot reload audio must bump reload_epoch")

    if has_sys_ttf:
        var h_fnt = fsrv.load_font("fonts/mono.ttf")
        _assert(h_fnt.is_valid() and fsrv.fonts.is_loaded(h_fnt), "loose_root load_font(.ttf) failed")
        var fnt_p = fsrv.fonts.borrow(h_fnt)
        _assert(fnt_p[].font.valid and len(fnt_p[].dsf_data) > 0, "loose TTF did not bake valid SdfFont")
        var dsf_cp = fnt_p[].dsf_data.copy()
        _assert(write_bytes(f20_ws + "/fonts/mono.dsf", dsf_cp), "write .dsf failed")
        var h_dsf = fsrv.load_font("fonts/mono.dsf")
        _assert(h_dsf.is_valid() and fsrv.fonts.borrow(h_dsf)[].font.valid, "loose_root load_font(.dsf) failed")
    fsrv.close()

    # ---- F20: VulkanGpu residency bridge + real cooked texture streaming + vk.close() before srv.close() ----
    var vsrv = AssetServer(stream_budget_bytes_per_tick=65536, enable_watcher=False)
    var vk = VulkanGpu()
    if vk.initialize(validate=False, need_discrete=False):
        vsrv.attach_vulkan_residency_bridge(vk)
        var qslot = vsrv.textures.allocate_queued(AssetUuid.from_path("residency/q.png"), "residency/q.png", 1)
        _assert(qslot.is_valid(), "allocate_queued failed")
        _assert(vsrv.upload_mip_to_gpu(qslot, mip_level=0, byte_size=16384), "upload_mip_to_gpu queued failed")
        _assert(vsrv.total_vram_live_bytes() > 0, "total_vram_live_bytes must be > 0 after GPU mip upload")
        _assert(vsrv.textures.slot_vram_bytes[Int(qslot.slot())] > 0, "slot_vram_bytes must be > 0")
        _assert(vsrv.streamer.vram_live_bytes > 0, "streamer.vram_live_bytes must be > 0")
        _ = vsrv.textures.release(qslot)
        _ = vsrv.textures.evict_lru_unreferenced(0)
        _assert(vsrv.total_vram_live_bytes() == 0, "VRAM must drop to 0 after eviction")
        _assert(vsrv.streamer.vram_live_bytes == 0, "streamer VRAM must drop to 0 after eviction")
        # Intentionally close vk before vsrv.close() to verify zero leak/crash
        vk.close()
    else:
        vk.close()
    vsrv.close()

    # ---- F21: KTX2 (None, Zlib, Zstd Raw/RLE, UASTC Mode 8, BasisLZ/ETC1S, BC7) + Corrupt Rejection ----
    var k_px = List[UInt8]()
    for _ in range(16):
        k_px.append(UInt8(200)); k_px.append(UInt8(100)); k_px.append(UInt8(50)); k_px.append(UInt8(255))
    var k_none = _build_ktx2_bytes(4, 4, 43, 0, 1, k_px, len(k_px))
    var d_none = decode_image_auto(k_none, 4096)
    _assert(d_none.valid and d_none.width == 4 and d_none.height == 4, "KTX2 RGBA8 None failed")
    _assert(d_none.pixels[0] == UInt32(0xFFC86432), "KTX2 RGBA8 pixel mismatch")

    var k_zlib = _build_ktx2_bytes(4, 4, 43, 3, 1, _wrap_zlib_stored(k_px), len(k_px))
    var d_zlib = decode_ktx2(k_zlib, 4096)
    _assert(d_zlib.valid and d_zlib.pixels[0] == UInt32(0xFFC86432), "KTX2 Zlib failed")

    var k_zstd_raw = _build_ktx2_bytes(4, 4, 43, 2, 1, _wrap_zstd_frame(k_px), len(k_px))
    var d_zstd_raw = decode_ktx2(k_zstd_raw, 4096)
    _assert(d_zstd_raw.valid and d_zstd_raw.pixels[0] == UInt32(0xFFC86432), "KTX2 Zstd raw failed")

    var k_zstd_rle = _build_ktx2_bytes(4, 4, 37, 2, 1, _wrap_zstd_frame(List[UInt8](), rle_byte=0x44, rle_len=64), 64)
    var d_zstd_rle = decode_ktx2(k_zstd_rle, 4096)
    _assert(d_zstd_rle.valid and d_zstd_rle.pixels[0] == UInt32(0x44444444), "KTX2 Zstd RLE failed")

    # UASTC Mode 8 constant RGBA block (Khronos DFS §23: mode=23 @0(5), R=0x12 @5, G=0x34 @13, B=0x56 @21, A=0xFE @29)
    var uastc_blk = List[UInt8](length=16, fill=0)
    var u_hdr = UInt64(23) | (UInt64(0x12) << 5) | (UInt64(0x34) << 13) | (UInt64(0x56) << 21) | (UInt64(0xFE) << 29)
    for bi in range(8):
        uastc_blk[bi] = UInt8((u_hdr >> UInt64(bi * 8)) & 0xFF)
    var k_uastc = _build_ktx2_bytes(4, 4, 0, 0, 166, uastc_blk, 16)
    var d_uastc = decode_ktx2(k_uastc, 4096)
    _assert(d_uastc.valid and d_uastc.pixels[0] == UInt32(0xFE123456), "KTX2 UASTC Mode 8 failed")

    # BasisLZ / ETC1S compact codebook fixture
    var sgd = List[UInt8]()
    _test_put_u16_le(sgd, 1); _test_put_u16_le(sgd, 1)  # 1 endpoint, 1 selector
    _test_put_u32_le(sgd, 3); _test_put_u32_le(sgd, 4); _test_put_u32_le(sgd, 0); _test_put_u32_le(sgd, 0)
    _test_put_u32_le(sgd, 0); _test_put_u32_le(sgd, 0); _test_put_u32_le(sgd, 1); _test_put_u32_le(sgd, 0); _test_put_u32_le(sgd, 0)
    var c5 = (31 << 10) | (0 << 5) | 0  # pure red
    _test_put_u16_le(sgd, c5); sgd.append(0)  # endpoint 0, inten table 0 (+/-2)
    _test_put_u32_le(sgd, 0)  # selector 0 (idx=0 -> +2)
    var etc_slice = List[UInt8]()
    etc_slice.append(0)
    var k_blz = _build_ktx2_bytes(4, 4, 0, 1, 163, etc_slice, 0, sgd)
    var d_blz = decode_ktx2(k_blz, 4096)
    _assert(d_blz.valid and ((d_blz.pixels[0] >> 16) & 0xFF) > 240, "KTX2 BasisLZ/ETC1S failed")

    # Corrupt KTX2 rejection
    var bad_ktx2 = k_none.copy()
    bad_ktx2.resize(40, UInt8(0))
    _assert(not decode_ktx2(bad_ktx2, 4096).valid, "Truncated KTX2 must be rejected")

    # ---- M3c Iter-2 KTX2 Regressions: multi-KB Zstd (of_val - 3), 4-stream regen_sz < 6, BC7 Modes 0/2/4, UASTC modes, BasisLZ Huffman ----
    # 1. Genuine `/usr/bin/zstd -3` 4096-byte (32x32 RGBA8) stream with 64-byte non-repeat match offset (of_code=6 -> of_val=67 -> offset=64)
    var zbig_frame: List[UInt8] = [
        40, 181, 47, 253, 4, 88, 85, 2, 0, 4, 4, 32, 39, 46, 255, 60, 67, 74, 255, 88, 95, 102, 255,
        116, 123, 130, 255, 144, 151, 158, 255, 172, 179, 186, 255, 200, 207, 214, 255, 228, 235, 242, 255,
        0, 7, 14, 255, 28, 35, 42, 255, 56, 63, 70, 255, 84, 91, 98, 255, 112, 119, 126, 255, 140, 147, 154, 255,
        168, 175, 182, 255, 196, 203, 210, 255, 1, 0, 64, 239, 7, 63, 68, 1, 86, 239, 134, 150,
    ]
    var k_zbig = _build_ktx2_bytes(32, 32, 37, 2, 1, zbig_frame, 4096)
    var d_zbig = decode_ktx2(k_zbig, 4096)
    _assert(d_zbig.valid and d_zbig.pixels[0] == UInt32(0xFF20272E) and d_zbig.pixels[16] == UInt32(0xFF20272E) and d_zbig.pixels[1023] == UInt32(0xFFC4CBD2), "KTX2 4KB Zstd non-repeat match offset failed")

    # Reject 4-stream Huffman Zstd literals with regen_sz < 6
    var z_bad_huf = List[UInt8]()
    _test_put_u32_le(z_bad_huf, 0xFD2FB528)
    z_bad_huf.append(UInt8(0x20)); z_bad_huf.append(UInt8(4))  # FCS=4
    var bh_bad = (10 << 3) | (2 << 1) | 1  # last=1, compressed, size=10
    z_bad_huf.append(UInt8(bh_bad & 0xFF)); z_bad_huf.append(UInt8((bh_bad >> 8) & 0xFF)); z_bad_huf.append(UInt8((bh_bad >> 16) & 0xFF))
    z_bad_huf.append(UInt8((4 << 4) | (1 << 2) | 2))  # Lit type=2 (Compressed/Huf), size_fmt=1 (4 streams!), regen_sz=4 (< 6!)
    for _ in range(9): z_bad_huf.append(UInt8(0))
    _assert(not decode_ktx2(_build_ktx2_bytes(1, 1, 37, 2, 1, z_bad_huf, 4), 4096).valid, "Zstd 4-stream Huffman with regen_sz < 6 must be rejected")

    # 2. BC7 Modes 0, 2 (3-subset Table.P3) & Mode 4 (idx_mode=1)
    # BC7 Mode 0: bit0=1, part=14 (4b: Table.P3[14] = 0xA9A59450 -> t0=sub0, t1=sub0? 0x50: t0=0,t1=0,t2=1,t3=1, t4=0,t5=1,t6=0,t7=1, t8=1,t9=1,t10=2,t11=2...)
    # Sub0=Red(R=15), Sub1=Green(G=15), Sub2=Blue(B=15), Alpha=255, all 3b indices=0 -> pick endpoint 0 of each subset!
    var bc7_m0 = List[UInt8](length=16, fill=0)
    # Header: bit0=1, part@1..4 = 14 -> bits 0..4 = 1 | (14<<1) = 29;
    # R0..R5 @5..28: R0=15(sub0),R1=15 @5..12 (0xFF<<5); G2=15,G3=15 @37..44; B4=15,B5=15 @69..76; pbits@77..82 = 0x3F
    var m0_lo = UInt64(29) | (UInt64(0xFF) << 5) | (UInt64(0xFF) << 37)
    var m0_hi = (UInt64(0xFF) << (69 - 64)) | (UInt64(0x3F) << (77 - 64))
    for bi in range(8):
        bc7_m0[bi] = UInt8((m0_lo >> UInt64(bi * 8)) & 0xFF)
        bc7_m0[8 + bi] = UInt8((m0_hi >> UInt64(bi * 8)) & 0xFF)
    var d_bc7_m0 = decode_ktx2(_build_ktx2_bytes(4, 4, 145, 0, 1, bc7_m0, 16), 4096)
    _assert(d_bc7_m0.valid and d_bc7_m0.pixels[0] == UInt32(0xFFFF0808) and d_bc7_m0.pixels[2] == UInt32(0xFF08FF08) and d_bc7_m0.pixels[15] == UInt32(0xFF0808FF), "BC7 Mode 0 Table.P3[14] failed")

    # BC7 Mode 2: bits0..2=0b100 (4), part=45 (6b @3..8: Table.P3[45] = 0x69286928 -> t0=0, t1=2, t2=2, t3=0, t4=1, t5=2...)
    # Sub0=Red(R0=31), Sub1=Green(G2=31), Sub2=Blue(B4=31), all 2b indices=0
    var bc7_m2 = List[UInt8](length=16, fill=0)
    var m2_lo = UInt64(4) | (UInt64(45) << 3) | (UInt64(31) << 9) | (UInt64(31) << 49)
    var m2_hi = UInt64(31) << (89 - 64)
    for bi in range(8):
        bc7_m2[bi] = UInt8((m2_lo >> UInt64(bi * 8)) & 0xFF)
        bc7_m2[8 + bi] = UInt8((m2_hi >> UInt64(bi * 8)) & 0xFF)
    var d_bc7_m2 = decode_ktx2(_build_ktx2_bytes(4, 4, 145, 0, 1, bc7_m2, 16), 4096)
    _assert(d_bc7_m2.valid and d_bc7_m2.pixels[0] == UInt32(0xFFFF0000) and d_bc7_m2.pixels[4] == UInt32(0xFF00FF00) and d_bc7_m2.pixels[1] == UInt32(0xFF0000FF), "BC7 Mode 2 Table.P3[45] failed")

    # BC7 Mode 4 with idx_mode=1: 0b10000(16) @0..4, rot=0 @5..6, idx_mode=1 @7, R0=0,R1=31 @8..17 -> R1=31 @13(5), G=0, B=0, A0=0,A1=63 @38..49 -> A1=63 @44(6)
    # 2b indices FIRST @50..80 (31b -> a_idx), 3b indices SECOND @81..127 (47b -> c_idx)
    # Set texel 1: a_idx[1] = 0 (@51..52 = 0), c_idx[1] = 7 (@83..85 = 7) -> expects R=255, A=0!
    var bc7_m4 = List[UInt8](length=16, fill=0)
    var m4_lo = UInt64(16) | (UInt64(1) << 7) | (UInt64(31) << 13) | (UInt64(63) << 44)
    var m4_hi = UInt64(7) << (83 - 64)
    for bi in range(8):
        bc7_m4[bi] = UInt8((m4_lo >> UInt64(bi * 8)) & 0xFF)
        bc7_m4[8 + bi] = UInt8((m4_hi >> UInt64(bi * 8)) & 0xFF)
    var d_bc7_m4 = decode_ktx2(_build_ktx2_bytes(4, 4, 145, 0, 1, bc7_m4, 16), 4096)
    _assert(d_bc7_m4.valid and d_bc7_m4.pixels[1] == UInt32(0x00FF0000), "BC7 Mode 4 idx_mode=1 block-order failed")

    # 3. UASTC Modes: Mode 0 (trits), Mode 7 (PAT=0), Mode 11 (dual-plane RGBA, ccs=3), Mode 15 (LA), Mode 18 (5b RGB+5b W) + Mode 19 rejection
    # UASTC Mode 0 (code 0x01 @0(4)): ETQ all 0 except RL=0, RH=(trit0? No, trit1=0, EBITS RH=1 -> ASTC val=1 -> 255!), W[1]=15 -> R=255 at px1, R=0 at px0!
    var u0_blk = List[UInt8](length=16, fill=0)
    var u0_lo = UInt64(0x01) | (UInt64(1) << 35)  # RH EBITS @29+6=35 = 1 -> 255; G=0,B=0
    var u0_hi = UInt64(15) << (68 - 64)  # W[0] is 3b @65..67 (=0), W[1] is 4b @68..71 (=15 -> w=64)
    for bi in range(8):
        u0_blk[bi] = UInt8((u0_lo >> UInt64(bi * 8)) & 0xFF)
        u0_blk[8 + bi] = UInt8((u0_hi >> UInt64(bi * 8)) & 0xFF)
    var d_u0 = decode_ktx2(_build_ktx2_bytes(4, 4, 0, 0, 166, u0_blk, 16), 4096)
    _assert(d_u0.valid and d_u0.pixels[0] == UInt32(0xFF000000) and d_u0.pixels[1] == UInt32(0xFFFF0000), "UASTC Mode 0 failed")

    # UASTC Mode 11 (code 0x00 @0(2), dual-plane RGBA, ccs=3 @19(2), EBITS R0=0,R1=1 @34..41 -> R1=1 -> 255, A0=0,A1=1 @58..65):
    # W[0]: p0=0, p1=0 @66..67 (1b each); W[1]: p0=3 @68..69 (R=255), p1=0 @70..71 (A=0) -> px1 = 0x00FF0000!
    var u11_blk = List[UInt8](length=16, fill=0)
    var u11_lo = (UInt64(3) << 19) | (UInt64(1) << 38) | (UInt64(1) << 62)
    var u11_hi = UInt64(3) << (68 - 64)
    for bi in range(8):
        u11_blk[bi] = UInt8((u11_lo >> UInt64(bi * 8)) & 0xFF)
        u11_blk[8 + bi] = UInt8((u11_hi >> UInt64(bi * 8)) & 0xFF)
    var d_u11 = decode_ktx2(_build_ktx2_bytes(4, 4, 0, 0, 166, u11_blk, 16), 4096)
    _assert(d_u11.valid and d_u11.pixels[1] == UInt32(0x00FF0000), "UASTC Mode 11 dual-plane failed")

    # UASTC Mode 15 (code 0x05 @0(7), LA direct): L0=0x80 @30, L1=0x80 @38, A0=0x40 @46, A1=0x40 @54 -> all pixels 0x40808080
    var u15_blk = List[UInt8](length=16, fill=0)
    var u15_lo = UInt64(0x05) | (UInt64(0x80) << 30) | (UInt64(0x80) << 38) | (UInt64(0x40) << 46) | (UInt64(0x40) << 54)
    for bi in range(8): u15_blk[bi] = UInt8((u15_lo >> UInt64(bi * 8)) & 0xFF)
    var d_u15 = decode_ktx2(_build_ktx2_bytes(4, 4, 0, 0, 166, u15_blk, 16), 4096)
    _assert(d_u15.valid and d_u15.pixels[0] == UInt32(0x40808080), "UASTC Mode 15 LA failed")

    # UASTC Mode 19 (reserved 0x45) must be rejected
    var u19_blk = List[UInt8](length=16, fill=0); u19_blk[0] = UInt8(0x45)
    _assert(not decode_ktx2(_build_ktx2_bytes(4, 4, 0, 0, 166, u19_blk, 16), 4096).valid, "UASTC reserved Mode 19 must be rejected")

    # 4. Nontrivial BasisLZ Huffman (`tbl_bytes > 0`, 3-symbol canonical Huffman table)
    # ep_huf: 32 syms (0..31), codelen_bits=2, len(0)=1 (code 0), len(1)=2 (code 0b10 -> rev 0b01), len(31)=2 (code 0b11 -> rev 0b11)
    # sel_huf: 2 syms (0..1), codelen_bits=2, len(0)=1 (code 0), len(1)=1 (code 1)
    var huf_tbl = List[UInt8](length=16, fill=0)
    var ep_hdr_val = UInt64(32) | (UInt64(2) << 14) | (UInt64(1) << 19) | (UInt64(2) << 21)
    for bi in range(8): huf_tbl[bi] = UInt8((ep_hdr_val >> UInt64(bi * 8)) & 0xFF)
    huf_tbl[10] = UInt8(0x04)  # sym 31 length=2 at bit 81..82
    var sel_hdr_u64 = UInt64(2) | (UInt64(2) << 14) | (UInt64(1) << 19) | (UInt64(1) << 21)
    for bit_i in range(23):
        if ((sel_hdr_u64 >> UInt64(bit_i)) & 1) != 0:
            var ab = 83 + bit_i
            huf_tbl[ab >> 3] |= UInt8(1 << (ab & 7))
    # ep stream: 1 endpoint -> dr=31 (code 0b11, 2b), dg=0 (0, 1b), db=0 (0, 1b), di=0 (0, 1b) = 0b00011 = 0x03 (1 byte)
    var sgd_h = List[UInt8]()
    _test_put_u16_le(sgd_h, 1); _test_put_u16_le(sgd_h, 1)
    _test_put_u32_le(sgd_h, 1); _test_put_u32_le(sgd_h, 1); _test_put_u32_le(sgd_h, 16); _test_put_u32_le(sgd_h, 0)
    _test_put_u32_le(sgd_h, 0); _test_put_u32_le(sgd_h, 0); _test_put_u32_le(sgd_h, 1); _test_put_u32_le(sgd_h, 0); _test_put_u32_le(sgd_h, 0)
    sgd_h.append(UInt8(0x03))  # ep_bytes (1)
    sgd_h.append(UInt8(0x00))  # sel_bytes (1)
    for b in huf_tbl: sgd_h.append(b)  # tbl_bytes (16)
    var etc_h_slice = List[UInt8](); etc_h_slice.append(UInt8(0x00))
    var d_blz_h = decode_ktx2(_build_ktx2_bytes(4, 4, 0, 1, 163, etc_h_slice, 0, sgd_h), 4096)
    _assert(d_blz_h.valid and ((d_blz_h.pixels[0] >> 16) & 0xFF) > 240, "KTX2 BasisLZ Huffman (tbl_bytes > 0) failed")

    # 5. Khronos 44-byte DFD colorModel @ +12 routing (vk_format=0, supercompression=0)
    var zero_16 = List[UInt8](length=16, fill=0)
    _assert(
        not decode_ktx2(_build_ktx2_bytes(4, 4, 0, 0, 163, zero_16, 16), 4096).valid,
        "KTX2 vk_format=0 supercomp=0 colorModel=163 (ETC1S) with invalid payload must fail (not route to UASTC)",
    )
    _assert(
        not decode_ktx2(_build_ktx2_bytes(4, 4, 0, 0, 1, zero_16, 16), 4096).valid,
        "KTX2 vk_format=0 supercomp=0 unknown/RGBSDA colorModel=1 must fail",
    )
    var d_uastc_std = decode_ktx2(_build_ktx2_bytes(4, 4, 0, 0, 166, uastc_blk, 16), 4096)
    _assert(d_uastc_std.valid and d_uastc_std.pixels[0] == UInt32(0xFE123456), "KTX2 standard 44-byte DFD UASTC (colorModel=166) failed")

    # ---- F21: GLB TANGENT (5126 & 5120 negative handedness), TEXCOORD_1, occlusion, emissive, KHR_texture_basisu, QEM/Cooked/Writer ----
    var mdl_w = NativeModel()
    var tex_w = NativeTexture()
    tex_w.width = 4; tex_w.height = 4
    tex_w.pixels = List[UInt32](length=16, fill=0xFF40C080)
    mdl_w.textures.append(tex_w^)
    var mesh_w = NativeMesh()
    mesh_w.positions = [Float32(0), Float32(0), Float32(0), Float32(1), Float32(0), Float32(0), Float32(0), Float32(1), Float32(0), Float32(1), Float32(1), Float32(0)]
    mesh_w.normals = [Float32(0), Float32(0), Float32(1), Float32(0), Float32(0), Float32(1), Float32(0), Float32(0), Float32(1), Float32(0), Float32(0), Float32(1)]
    mesh_w.tangents = [Float32(1), Float32(0), Float32(0), Float32(-1), Float32(1), Float32(0), Float32(0), Float32(-1), Float32(1), Float32(0), Float32(0), Float32(-1), Float32(1), Float32(0), Float32(0), Float32(-1)]
    mesh_w.texcoords = [Float32(0), Float32(0), Float32(1), Float32(0), Float32(0), Float32(1), Float32(1), Float32(1)]
    mesh_w.texcoords1 = [Float32(0.25), Float32(0.5), Float32(0.75), Float32(0.5), Float32(0.25), Float32(0.75), Float32(0.75), Float32(0.75)]
    mesh_w.indices = [0, 1, 2, 1, 3, 2]
    mesh_w.texture = 0
    mesh_w.occlusion_texture = 0
    mesh_w.occlusion_texcoord = 1
    mesh_w.occlusion_strength = Float32(0.75)
    mesh_w.emissive_texture = 0
    mesh_w.emissive_texcoord = 1
    mesh_w.emissive_color = Vec3(1.0, 0.5, 0.25)
    mesh_w.emissive_strength = Float32(2.0)
    mdl_w.meshes.append(mesh_w^)

    var glb_rt = encode_native_glb(mdl_w)
    var mdl_rt = parse_native_glb(glb_rt)
    _assert(mdl_rt.loaded() and len(mdl_rt.meshes) == 1, "GLB writer/reader round-trip failed")
    _assert(len(mdl_rt.meshes[0].tangents) == 16 and mdl_rt.meshes[0].tangents[3] < Float32(0.0), "GLB TANGENT round-trip lost handedness")
    _assert(len(mdl_rt.meshes[0].texcoords1) == 8 and abs(mdl_rt.meshes[0].texcoords1[0] - Float32(0.25)) < Float32(1e-4), "GLB TEXCOORD_1 round-trip failed")
    _assert(mdl_rt.meshes[0].occlusion_texture == 0 and mdl_rt.meshes[0].occlusion_texcoord == 1, "GLB occlusionTexture round-trip failed")
    _assert(abs(mdl_rt.meshes[0].occlusion_strength - Float32(0.75)) < Float32(1e-3), "GLB occlusion_strength mismatch")
    _assert(mdl_rt.meshes[0].emissive_texture == 0 and mdl_rt.meshes[0].emissive_texcoord == 1, "GLB emissiveTexture round-trip failed")

    # Verify authored negative-tw tangent reaches AaaGeometryPool vertex_words[11] bit 0
    var gpool = AaaGeometryPool()
    _ = gpool.add_mesh(mdl_rt.meshes[0], 0, build_lods=False)
    _assert((gpool.vertex_words[11] & UInt32(1)) == UInt32(1), "AaaGeometryPool must preserve authored TANGENT w=-1 sign bit")

    # Verify QEM decimation preserves tangents, texcoords1, occlusion & emissive
    var qem_m = simplify_mesh_qem(mdl_rt.meshes[0], 1)
    _assert(qem_m[0].occlusion_texture == 0 and qem_m[0].emissive_texture == 0, "simplify_mesh_qem must preserve occlusion/emissive")
    _assert(len(qem_m[0].tangents) > 0 and len(qem_m[0].texcoords1) > 0, "simplify_mesh_qem must preserve tangents/texcoords1")

    # GLB with KHR_texture_basisu pointing to KTX2 bufferView + INT8 TANGENT (5120) + omitted emissiveFactor
    var glb_bin = List[UInt8]()
    var pos_f = [Float32(0), Float32(0), Float32(0), Float32(1), Float32(0), Float32(0), Float32(0), Float32(1), Float32(0)]
    for fv in pos_f:
        _test_put_u32_le(glb_bin, bitcast[DType.uint32](fv))
    var t_off = len(glb_bin)
    for _ in range(3):
        glb_bin.append(127); glb_bin.append(0); glb_bin.append(0); glb_bin.append(bitcast[DType.uint8](Int8(-127)))
    var k_off = len(glb_bin)
    for b in k_uastc:
        glb_bin.append(b)
    while len(glb_bin) % 4 != 0:
        glb_bin.append(0)
    var gj = (
        '{"asset":{"version":"2.0"},"extensionsUsed":["KHR_texture_basisu","KHR_materials_emissive_strength"],'
        + '"buffers":[{"byteLength":' + String(len(glb_bin)) + '}],'
        + '"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},'
        + '{"buffer":0,"byteOffset":' + String(t_off) + ',"byteLength":12},'
        + '{"buffer":0,"byteOffset":' + String(k_off) + ',"byteLength":' + String(len(k_uastc)) + '}],'
        + '"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},'
        + '{"bufferView":1,"componentType":5120,"normalized":true,"count":3,"type":"VEC4"}],'
        + '"images":[{"bufferView":2,"mimeType":"image/ktx2"}],'
        + '"textures":[{"extensions":{"KHR_texture_basisu":{"source":0}}}],'
        + '"materials":[{"emissiveTexture":{"index":0},"extensions":{"KHR_materials_emissive_strength":{"emissiveStrength":3.0}}}],'
        + '"meshes":[{"primitives":[{"attributes":{"POSITION":0,"TANGENT":1},"material":0}]}],'
        + '"nodes":[{"mesh":0}],"scene":0,"scenes":[{"nodes":[0]}]}'
    )
    var gjb = _str_bytes(gj)
    while len(gjb) % 4 != 0:
        gjb.append(0x20)
    var kglb = List[UInt8]()
    var ktot = 12 + 8 + len(gjb) + 8 + len(glb_bin)
    _test_put_u32_le(kglb, 0x46546C67); _test_put_u32_le(kglb, 2); _test_put_u32_le(kglb, UInt32(ktot))
    _test_put_u32_le(kglb, UInt32(len(gjb))); _test_put_u32_le(kglb, 0x4E4F534A)
    for b in gjb: kglb.append(b)
    _test_put_u32_le(kglb, UInt32(len(glb_bin))); _test_put_u32_le(kglb, 0x004E4942)
    for b in glb_bin: kglb.append(b)
    var kmdl = parse_native_glb(kglb)
    _assert(kmdl.loaded() and len(kmdl.textures) == 1 and kmdl.textures[0].width == 4, "GLB KHR_texture_basisu KTX2 decode failed")
    _assert(kmdl.textures[0].pixels[0] == UInt32(0xFE123456), "GLB KHR_texture_basisu pixel mismatch")
    _assert(kmdl.meshes[0].tangents[3] < Float32(0.0) and abs(kmdl.meshes[0].emissive_strength - Float32(3.0)) < Float32(1e-3), "GLB INT8 TANGENT / emissiveStrength failed")

    # ---- F21: Progressive SOF2 JPEG (17x13 multi-scan 4:2:0 color + DRI/RST + DC refine + AC first EOBRUN + AC refine ZRL) ----
    var jout = List[UInt8]()
    jout.append(0xFF); jout.append(0xD8)
    jout.append(0xFF); jout.append(0xDB); _test_put_u16_be(jout, 67); jout.append(0)
    for _ in range(64): jout.append(1)
    # DRI = 2
    jout.append(0xFF); jout.append(0xDD); _test_put_u16_be(jout, 4); _test_put_u16_be(jout, 2)
    # SOF2 17x13, 3 components (Y 2x2, Cb 1x1, Cr 1x1 -> 2x1 MCUs)
    jout.append(0xFF); jout.append(0xC2); _test_put_u16_be(jout, 17)
    jout.append(8); _test_put_u16_be(jout, 13); _test_put_u16_be(jout, 17); jout.append(3)
    jout.append(1); jout.append(0x22); jout.append(0)
    jout.append(2); jout.append(0x11); jout.append(0)
    jout.append(3); jout.append(0x11); jout.append(0)
    # DHT DC0: sym 0 -> '0', sym 7 -> '10', sym 8 -> '11'
    jout.append(0xFF); jout.append(0xC4); _test_put_u16_be(jout, 22); jout.append(0x00)
    jout.append(1); jout.append(2)
    for _ in range(14): jout.append(0)
    jout.append(0); jout.append(7); jout.append(8)
    # SOS 1: Interleaved Y+Cb+Cr DC first (Ss=0, Se=0, Ah=0, Al=1)
    # Target DC: Y = +200 (shifted 1 => +100, cat 7, bits 100), Cb = -400 (shifted 1 => -200, cat 8, bits 55), Cr = +400 (shifted 1 => +200, cat 8, bits 200)
    jout.append(0xFF); jout.append(0xDA); _test_put_u16_be(jout, 12)
    jout.append(3); jout.append(1); jout.append(0x00); jout.append(2); jout.append(0x00); jout.append(3); jout.append(0x00)
    jout.append(0); jout.append(0); jout.append(0x01)
    var sb1 = _Sof2BitWriter()
    # MCU 0: 4 Y blocks (+100 then 0,0,0), 1 Cb (-200), 1 Cr (+200)
    sb1.write_bits(0b10, 2); sb1.write_bits(100, 7)  # Y0 diff +100
    sb1.write_bits(0b0, 1); sb1.write_bits(0b0, 1); sb1.write_bits(0b0, 1)  # Y1,Y2,Y3 diff 0
    sb1.write_bits(0b11, 2); sb1.write_bits(55, 8)   # Cb diff -200
    sb1.write_bits(0b11, 2); sb1.write_bits(200, 8)  # Cr diff +200
    # MCU 1 (all diff 0)
    for _ in range(6): sb1.write_bits(0b0, 1)
    sb1.flush_and_rst(-1)
    for b in sb1.out: jout.append(b)
    # SOS 2: Interleaved DC refine (Ss=0, Se=0, Ah=1, Al=0) -> 12 zero refinement bits
    jout.append(0xFF); jout.append(0xDA); _test_put_u16_be(jout, 12)
    jout.append(3); jout.append(1); jout.append(0x00); jout.append(2); jout.append(0x00); jout.append(3); jout.append(0x00)
    jout.append(0); jout.append(0); jout.append(0x10)
    var sb2 = _Sof2BitWriter()
    for _ in range(12): sb2.write_bits(0, 1)
    sb2.flush_and_rst(-1)
    for b in sb2.out: jout.append(b)
    # DHT AC0: sym 0x00 -> '00' (EOB), sym 0x10 -> '01' (EOBRUN=2..3), sym 0x01 -> '10' (s=1), sym 0xF0 -> '11' (ZRL)
    jout.append(0xFF); jout.append(0xC4); _test_put_u16_be(jout, 23); jout.append(0x10)
    jout.append(0); jout.append(4)
    for _ in range(14): jout.append(0)
    jout.append(0x00); jout.append(0x10); jout.append(0x01); jout.append(0xF0)
    # SOS 3: Y non-interleaved AC first (Ss=1, Se=63, Ah=0, Al=1), 4x2=8 blocks with DRI=2 -> RST0, RST1, RST2!
    # Block 0: k=1 -> sym 0x01 val +1 (coeffs[zz(1)] = +2), then EOB (0x00); Block 1: EOB; RST0; Blocks 2..7: EOBRUN=2 per pair!
    jout.append(0xFF); jout.append(0xDA); _test_put_u16_be(jout, 8)
    jout.append(1); jout.append(1); jout.append(0x00); jout.append(1); jout.append(63); jout.append(0x01)
    var sb3 = _Sof2BitWriter()
    sb3.write_bits(0b10, 2); sb3.write_bits(1, 1); sb3.write_bits(0b00, 2)  # blk 0
    sb3.write_bits(0b00, 2)  # blk 1
    sb3.flush_and_rst(0)
    sb3.write_bits(0b01, 2); sb3.write_bits(0, 1)  # EOBRUN=2 covers blk 2,3
    sb3.flush_and_rst(1)
    sb3.write_bits(0b01, 2); sb3.write_bits(0, 1)  # EOBRUN=2 covers blk 4,5
    sb3.flush_and_rst(2)
    sb3.write_bits(0b01, 2); sb3.write_bits(0, 1)  # EOBRUN=2 covers blk 6,7
    sb3.flush_and_rst(-1)
    for b in sb3.out: jout.append(b)
    # SOS 4: Y AC refine (Ss=1, Se=63, Ah=1, Al=0) with DRI=2: blk 0 ZRL(0xF0) skips 16 zeros (refines zz(1) with bit 1 -> +3), then s=1 (+1 at k=17), then EOB!
    jout.append(0xFF); jout.append(0xDA); _test_put_u16_be(jout, 8)
    jout.append(1); jout.append(1); jout.append(0x00); jout.append(1); jout.append(63); jout.append(0x10)
    var sb4 = _Sof2BitWriter()
    sb4.write_bits(0b11, 2); sb4.write_bits(1, 1)  # blk 0: ZRL (refines non-zero zz(1) with 1), k -> 17
    sb4.write_bits(0b10, 2); sb4.write_bits(1, 1); sb4.write_bits(0b00, 2)  # blk 0: s=1 sign=1 at k=17, then EOB
    sb4.write_bits(0b00, 2)  # blk 1: EOB
    sb4.flush_and_rst(0)
    sb4.write_bits(0b01, 2); sb4.write_bits(0, 1)
    sb4.flush_and_rst(1)
    sb4.write_bits(0b01, 2); sb4.write_bits(0, 1)
    sb4.flush_and_rst(2)
    sb4.write_bits(0b01, 2); sb4.write_bits(0, 1)
    sb4.flush_and_rst(-1)
    for b in sb4.out: jout.append(b)
    jout.append(0xFF); jout.append(0xD9)

    var j_prog = decode_jpeg(jout, 1024)
    _assert(j_prog.valid and j_prog.width == 17 and j_prog.height == 13, "Multi-scan SOF2 17x13 decode failed")
    var p0 = j_prog.pixels[0]
    var pr = Int((p0 >> 16) & 0xFF)
    var pb = Int(p0 & 0xFF)
    _assert(pr > 200 and pb < 80, "SOF2 YCbCr->RGB reddish pixel mismatch: r=" + String(pr) + " b=" + String(pb))

    # Corrupt SOF2 rejections (Ns > 1 on AC scan; invalid Ah != Al + 1)
    var bad_sof2 = jout.copy()
    var patched_sof2 = False
    for i in range(len(bad_sof2) - 10):
        if bad_sof2[i] == 0xFF and bad_sof2[i + 1] == 0xDA and bad_sof2[i + 4] == 1 and bad_sof2[i + 8] == 63:
            bad_sof2[i + 9] = 0x20  # Ah=2, Al=0 invalid!
            patched_sof2 = True
            break
    _assert(patched_sof2 and not decode_jpeg(bad_sof2, 1024).valid, "SOF2 with invalid Ah != Al + 1 must fail")

    # ---- F21: OpenEXR Tiled (5x5 edge clamping + MIPMAP), RLE, PIZ, DWAA & DWAB + Negative Checks ----
    # 1. Tiled 5x5 EXR (tile 4x4, MIPMAP_LEVELS -> level 0 tiles + level 1 ignored)
    var exr_tiled = List[UInt8]()
    _write_exr_header(exr_tiled, 5, 5, EXR_NO_COMPRESSION, tiled=True, tile_x=4, tile_y=4, mipmap=True)
    # Mipmap levels: L0 (5x5 -> 2x2=4 tiles), L1 (2x2 -> 1x1=1 tile), L2 (1x1 -> 1x1=1 tile) => 6 chunks
    var t_tbl_pos = len(exr_tiled)
    for _ in range(6):
        _test_put_u64_le(exr_tiled, 0)
    var t_offs = List[UInt64]()
    var t_dims = [(0, 0, 0, 4, 4), (1, 0, 0, 1, 4), (0, 1, 0, 4, 1), (1, 1, 0, 1, 1), (0, 0, 1, 2, 2), (0, 0, 2, 1, 1)]
    for td in t_dims:
        t_offs.append(UInt64(len(exr_tiled)))
        _test_put_u32_le(exr_tiled, UInt32(td[0]))
        _test_put_u32_le(exr_tiled, UInt32(td[1]))
        _test_put_u32_le(exr_tiled, UInt32(td[2]))
        _test_put_u32_le(exr_tiled, UInt32(td[2]))
        var pbytes = td[3] * td[4] * 3 * 2
        _test_put_u32_le(exr_tiled, UInt32(pbytes))
        var h_r = f32_to_uf16_bits(Float32(0.75)) if td[2] == 0 else f32_to_uf16_bits(Float32(9.0))
        var h_g = f32_to_uf16_bits(Float32(0.5))
        var h_b = f32_to_uf16_bits(Float32(0.25))
        for _ in range(td[4]):
            for _ in range(td[3]): _test_put_u16_le(exr_tiled, h_b)
            for _ in range(td[3]): _test_put_u16_le(exr_tiled, h_g)
            for _ in range(td[3]): _test_put_u16_le(exr_tiled, h_r)
    for ci in range(6):
        var v = t_offs[ci]
        for bi in range(8):
            exr_tiled[t_tbl_pos + ci * 8 + bi] = UInt8((v >> UInt64(bi * 8)) & 0xFF)
    var d_tiled = decode_exr_fp16(exr_tiled)
    _assert(d_tiled.valid and d_tiled.width == 5 and d_tiled.height == 5, "Tiled 5x5 EXR decode failed")
    _assert(abs(d_tiled.pixels[4 * 5 + 4].x - Float32(0.75)) < Float32(1e-2), "Tiled edge-tile (4,4) pixel mismatch")

    # 2. Scanline RLE EXR (4x2)
    var exr_rle = List[UInt8]()
    _write_exr_header(exr_rle, 4, 2, EXR_RLE_COMPRESSION, tiled=False)
    var rle_tbl = len(exr_rle)
    _test_put_u64_le(exr_rle, 0); _test_put_u64_le(exr_rle, 0)
    var rle_offs = List[UInt64]()
    for row in range(2):
        var raw_row = List[UInt8]()
        var hb = f32_to_uf16_bits(Float32(0.25))
        var hg = f32_to_uf16_bits(Float32(0.5))
        var hr = f32_to_uf16_bits(Float32(1.5))
        for _ in range(4): _test_put_u16_le(raw_row, hb)
        for _ in range(4): _test_put_u16_le(raw_row, hg)
        for _ in range(4): _test_put_u16_le(raw_row, hr)
        var comp_rle = _exr_rle_compress(_exr_reorder_and_predict(raw_row))
        rle_offs.append(UInt64(len(exr_rle)))
        _test_put_u32_le(exr_rle, UInt32(row))
        _test_put_u32_le(exr_rle, UInt32(len(comp_rle)))
        for b in comp_rle: exr_rle.append(b)
    for ri in range(2):
        for bi in range(8):
            exr_rle[rle_tbl + ri * 8 + bi] = UInt8((rle_offs[ri] >> UInt64(bi * 8)) & 0xFF)
    var d_rle = decode_exr_fp16(exr_rle)
    _assert(d_rle.valid and abs(d_rle.pixels[0].x - Float32(1.5)) < Float32(1e-2), "EXR RLE pixel mismatch")

    # 3. Scanline PIZ EXR (4x4, genuine bitmap + LUT + 2D Haar wavelet + canonical Huffman)
    var exr_piz = List[UInt8]()
    _write_exr_header(exr_piz, 4, 4, EXR_PIZ_COMPRESSION, tiled=False)
    var piz_tbl = len(exr_piz)
    _test_put_u64_le(exr_piz, 0)
    var piz_off = UInt64(len(exr_piz))
    for bi in range(8): exr_piz[piz_tbl + bi] = UInt8((piz_off >> UInt64(bi * 8)) & 0xFF)
    var piz_u16 = List[UInt16]()
    var h_b = UInt16(f32_to_uf16_bits(Float32(0.125)))
    var h_g = UInt16(f32_to_uf16_bits(Float32(0.5)))
    var h_r = UInt16(f32_to_uf16_bits(Float32(2.0)))
    for _ in range(16): piz_u16.append(h_b)
    for _ in range(16): piz_u16.append(h_g)
    for _ in range(16): piz_u16.append(h_r)
    # Build forward bitmap & reverse LUT (map raw u16 -> compact index)
    var bm = List[UInt8](length=8192, fill=0)
    for v in piz_u16:
        if v != 0:
            bm[Int(v) >> 3] |= UInt8(1 << (Int(v) & 7))
    var min_nz = 8191; var max_nz = 0
    for bi in range(8192):
        if bm[bi] != 0:
            if bi < min_nz: min_nz = bi
            if bi > max_nz: max_nz = bi
    var inv_lut = List[UInt16](length=65536, fill=0)
    var lk = 0
    for i in range(65536):
        if i == 0 or (bm[i >> 3] & UInt8(1 << (i & 7))) != 0:
            inv_lut[i] = UInt16(lk)
            lk += 1
    for i in range(len(piz_u16)):
        piz_u16[i] = inv_lut[Int(piz_u16[i])]
    for ci in range(3):
        _exr_wav2_encode(piz_u16, ci * 16, 4, 1, 4, 4)
    var huf_blob = _build_exr_huf_blob(piz_u16)
    var piz_chunk = List[UInt8]()
    _test_put_u16_le(piz_chunk, min_nz); _test_put_u16_le(piz_chunk, max_nz)
    for bi in range(min_nz, max_nz + 1): piz_chunk.append(bm[bi])
    _test_put_u32_le(piz_chunk, UInt32(len(huf_blob)))
    for b in huf_blob: piz_chunk.append(b)
    _test_put_u32_le(exr_piz, 0); _test_put_u32_le(exr_piz, UInt32(len(piz_chunk)))
    for b in piz_chunk: exr_piz.append(b)
    var d_piz = decode_exr_fp16(exr_piz)
    _assert(d_piz.valid and abs(d_piz.pixels[0].x - Float32(2.0)) < Float32(1e-2) and abs(d_piz.pixels[0].z - Float32(0.125)) < Float32(1e-2), "EXR PIZ pixel round-trip failed")

    # 4. Scanline DWAA (8) & DWAB (9) EXR (8x8 flat linear Rec.709 RGB -> FDCT DC-only -> zlib DC + AC EOB)
    for comp_dwa in [EXR_DWAA_COMPRESSION, EXR_DWAB_COMPRESSION]:
        var exr_dwa = List[UInt8]()
        _write_exr_header(exr_dwa, 8, 8, comp_dwa, tiled=False)
        var dwa_tbl = len(exr_dwa)
        _test_put_u64_le(exr_dwa, 0)
        var dwa_off = UInt64(len(exr_dwa))
        for bi in range(8): exr_dwa[dwa_tbl + bi] = UInt8((dwa_off >> UInt64(bi * 8)) & 0xFF)
        # Desired linear RGB = (0.5, 0.25, 0.05) -> nonlinear (0.5^(1/2.2), ...) -> Rec.709 Y, Cb, Cr -> * 8.0 for orthonormal 8x8 DC!
        var r_lin = Float32(0.5); var g_lin = Float32(0.25); var b_lin = Float32(0.05)
        var r_nl = exp(log(r_lin) / Float32(2.2))
        var g_nl = exp(log(g_lin) / Float32(2.2))
        var b_nl = exp(log(b_lin) / Float32(2.2))
        var y_nl = Float32(0.2126) * r_nl + Float32(0.7152) * g_nl + Float32(0.0722) * b_nl
        var cb_nl = (b_nl - y_nl) / Float32(1.8556)
        var cr_nl = (r_nl - y_nl) / Float32(1.5747)
        # OpenEXR chlist is sorted (B, G, R); DWA CSC planes are stored in [Y, Cb, Cr] order
        var dc_bytes = List[UInt8]()
        for v_dc in [y_nl * Float32(8.0), cb_nl * Float32(8.0), cr_nl * Float32(8.0)]:
            var hv = f32_to_uf16_bits(v_dc) if v_dc >= 0 else (0x8000 | f32_to_uf16_bits(-v_dc))
            _test_put_u16_le(dc_bytes, hv)
        var dc_comp = _wrap_zlib_stored(_exr_reorder_and_predict(dc_bytes))
        var ac_u16 = List[UInt16](length=3, fill=UInt16(0xFF00))  # 3 EOB markers
        var ac_bytes = List[UInt8]()
        for av in ac_u16: _test_put_u16_le(ac_bytes, Int(av))
        var ac_comp = _wrap_zlib_stored(ac_bytes)

        var dwa_chunk = List[UInt8]()
        _test_put_u64_le(dwa_chunk, 2)  # version 2
        _test_put_u64_le(dwa_chunk, 0); _test_put_u64_le(dwa_chunk, 0)  # unknown
        _test_put_u64_le(dwa_chunk, UInt64(len(ac_comp)))
        _test_put_u64_le(dwa_chunk, UInt64(len(dc_comp)))
        _test_put_u64_le(dwa_chunk, 0); _test_put_u64_le(dwa_chunk, 0); _test_put_u64_le(dwa_chunk, 0)  # rle
        _test_put_u64_le(dwa_chunk, 3)  # total_ac_cnt
        _test_put_u64_le(dwa_chunk, 3)  # total_dc_cnt
        _test_put_u64_le(dwa_chunk, 1)  # ac_compression = DEFLATE
        # Rule table: R -> LOSSY_DCT + csc 0 (val=(1<<2)|(1<<4)=0x14), G -> 0x24, B -> 0x34
        _test_put_u16_le(dwa_chunk, 11)
        dwa_chunk.append(UInt8(ord("R"))); dwa_chunk.append(0); dwa_chunk.append(0x14)
        dwa_chunk.append(UInt8(ord("G"))); dwa_chunk.append(0); dwa_chunk.append(0x24)
        dwa_chunk.append(UInt8(ord("B"))); dwa_chunk.append(0); dwa_chunk.append(0x34)
        for b in ac_comp: dwa_chunk.append(b)
        for b in dc_comp: dwa_chunk.append(b)

        _test_put_u32_le(exr_dwa, 0); _test_put_u32_le(exr_dwa, UInt32(len(dwa_chunk)))
        for b in dwa_chunk: exr_dwa.append(b)

        var d_dwa = decode_hdr_or_exr(exr_dwa)
        _assert(d_dwa.valid and d_dwa.width == 8 and d_dwa.height == 8, "EXR DWA decode failed for compression=" + String(comp_dwa))
        _assert(
            abs(d_dwa.pixels[0].x - r_lin) < Float32(0.02)
            and abs(d_dwa.pixels[0].y - g_lin) < Float32(0.02)
            and abs(d_dwa.pixels[0].z - b_lin) < Float32(0.02),
            "EXR DWA pixel mismatch",
        )

    # Strict EXR negative checks: tiled flag 0x200 without tiles attr; truncated PIZ/DWAA
    var bad_tiled = exr_rle.copy()
    bad_tiled[5] |= UInt8(0x02)
    _assert(not decode_exr_fp16(bad_tiled).valid, "EXR with 0x200 flag and no tiles attr must be rejected")
    var trunc_dwa = List[UInt8]()
    _write_exr_header(trunc_dwa, 4, 4, EXR_DWAA_COMPRESSION, tiled=False)
    _test_put_u64_le(trunc_dwa, UInt64(len(trunc_dwa) + 8))
    _test_put_u32_le(trunc_dwa, 0); _test_put_u32_le(trunc_dwa, 24)
    for _ in range(24): trunc_dwa.append(0)
    _assert(not decode_exr_fp16(trunc_dwa).valid, "Truncated 24-byte DWAA chunk must be rejected")

    # ---- M3c Iter-2 EXR Regressions: Oversubscribed Huffman, Truncated DWA sub-streams, OOB/missing tiles, 16x16 Multi-block DWAA/DWAB ----
    # 1. Oversubscribed EXR PIZ Huffman (im=0, iM=3, all 4 symbols have len=1 -> Kraft overflow)
    var bad_huf_exr = List[UInt8]()
    _write_exr_header(bad_huf_exr, 4, 4, EXR_PIZ_COMPRESSION, tiled=False)
    _test_put_u64_le(bad_huf_exr, UInt64(len(bad_huf_exr) + 8))
    var bad_huf_blob = List[UInt8]()
    _test_put_u32_le(bad_huf_blob, 0); _test_put_u32_le(bad_huf_blob, 3)  # im=0, iM=3
    _test_put_u32_le(bad_huf_blob, 3); _test_put_u32_le(bad_huf_blob, 4); _test_put_u32_le(bad_huf_blob, 0)
    # 4 x 6-bit lengths of 1 (= 0b000001): packed MSB-first = 0x04, 0x10, 0x41 + 1 dummy bit byte
    bad_huf_blob.append(UInt8(0x04)); bad_huf_blob.append(UInt8(0x10)); bad_huf_blob.append(UInt8(0x41)); bad_huf_blob.append(UInt8(0x00))
    var bhuf_chunk = List[UInt8]()
    _test_put_u16_le(bhuf_chunk, 0); _test_put_u16_le(bhuf_chunk, 0); bhuf_chunk.append(UInt8(0x0F))  # min_nz=0, max_nz=0, bitmap 4 syms
    _test_put_u32_le(bhuf_chunk, UInt32(len(bad_huf_blob)))
    for b in bad_huf_blob: bhuf_chunk.append(b)
    _test_put_u32_le(bad_huf_exr, 0); _test_put_u32_le(bad_huf_exr, UInt32(len(bhuf_chunk)))
    for b in bhuf_chunk: bad_huf_exr.append(b)
    _assert(not decode_exr_fp16(bad_huf_exr).valid, "Oversubscribed EXR PIZ Huffman must return valid=False without crashing")

    # 2. Truncated DWA sub-streams (`dc_comp_sz=0` with `total_dc_cnt=3` + truncated `ac_u16`)
    var bad_dwa_dc = List[UInt8]()
    _write_exr_header(bad_dwa_dc, 8, 8, EXR_DWAA_COMPRESSION, tiled=False)
    _test_put_u64_le(bad_dwa_dc, UInt64(len(bad_dwa_dc) + 8))
    var dc0_chunk = List[UInt8]()
    _test_put_u64_le(dc0_chunk, 2)
    _test_put_u64_le(dc0_chunk, 0); _test_put_u64_le(dc0_chunk, 0)
    _test_put_u64_le(dc0_chunk, 0); _test_put_u64_le(dc0_chunk, 0)  # ac_comp=0, dc_comp=0
    _test_put_u64_le(dc0_chunk, 0); _test_put_u64_le(dc0_chunk, 0); _test_put_u64_le(dc0_chunk, 0)
    _test_put_u64_le(dc0_chunk, 0); _test_put_u64_le(dc0_chunk, 3); _test_put_u64_le(dc0_chunk, 1)  # total_dc_cnt=3!
    _test_put_u16_le(dc0_chunk, 11)
    dc0_chunk.append(UInt8(ord("R"))); dc0_chunk.append(0); dc0_chunk.append(0x14)
    dc0_chunk.append(UInt8(ord("G"))); dc0_chunk.append(0); dc0_chunk.append(0x24)
    dc0_chunk.append(UInt8(ord("B"))); dc0_chunk.append(0); dc0_chunk.append(0x34)
    _test_put_u32_le(bad_dwa_dc, 0); _test_put_u32_le(bad_dwa_dc, UInt32(len(dc0_chunk)))
    for b in dc0_chunk: bad_dwa_dc.append(b)
    _assert(not decode_exr_fp16(bad_dwa_dc).valid, "DWA chunk with dc_comp_sz=0 and total_dc_cnt=3 must be rejected")

    # 3. OOB tile coordinates & missing level-0 tiles
    var oob_tile_exr = List[UInt8]()
    _write_exr_header(oob_tile_exr, 4, 4, EXR_NO_COMPRESSION, tiled=True)
    _test_put_u64_le(oob_tile_exr, UInt64(len(oob_tile_exr) + 8))
    _test_put_u32_le(oob_tile_exr, 1); _test_put_u32_le(oob_tile_exr, 0)  # tx=1 is OOB for 4x4 with tile_w=4!
    _test_put_u32_le(oob_tile_exr, 0); _test_put_u32_le(oob_tile_exr, 0)
    _test_put_u32_le(oob_tile_exr, 96)
    for _ in range(96): oob_tile_exr.append(0)
    _assert(not decode_exr_fp16(oob_tile_exr).valid, "EXR with OOB tile coordinate must be rejected")

    # 4. Multi-block 16x16 DWAA/DWAB (`[Y, Cb, Cr]` + `block_major` AC, `ac_compression=0` Huffman & `1` DEFLATE)
    for ac_mode in range(2):
        var c_type = EXR_DWAA_COMPRESSION if ac_mode == 1 else EXR_DWAB_COMPRESSION
        var exr16 = List[UInt8]()
        _write_exr_header(exr16, 16, 16, c_type, tiled=False)
        var t16 = len(exr16)
        _test_put_u64_le(exr16, 0)
        var off16 = UInt64(len(exr16))
        for bi in range(8): exr16[t16 + bi] = UInt8((off16 >> UInt64(bi * 8)) & 0xFF)
        # 4 blocks (8x8 each): b0=(0,0) gray(0.5), b1=(8,0) red-ish + non-zero Y AC[1], b2=(0,8) green-ish, b3=(8,8) blue-ish
        var b_y = List[Float32](); var b_cb = List[Float32](); var b_cr = List[Float32]()
        var br_vals: List[Float32] = [Float32(0.5), Float32(0.6), Float32(0.1), Float32(0.1)]
        var bg_vals: List[Float32] = [Float32(0.5), Float32(0.1), Float32(0.6), Float32(0.1)]
        var bb_vals: List[Float32] = [Float32(0.5), Float32(0.1), Float32(0.1), Float32(0.6)]
        for bi in range(4):
            var rn = exp(log(br_vals[bi]) / Float32(2.2))
            var gn = exp(log(bg_vals[bi]) / Float32(2.2))
            var bn = exp(log(bb_vals[bi]) / Float32(2.2))
            var yn = Float32(0.2126) * rn + Float32(0.7152) * gn + Float32(0.0722) * bn
            b_y.append(yn * Float32(8.0))
            b_cb.append(((bn - yn) / Float32(1.8556)) * Float32(8.0))
            b_cr.append(((rn - yn) / Float32(1.5747)) * Float32(8.0))
        # DC stream is channel-major: all 4 Y, all 4 Cb, all 4 Cr
        var dc16_bytes = List[UInt8]()
        for v_dc in b_y:
            var hv = f32_to_uf16_bits(v_dc) if v_dc >= 0 else (0x8000 | f32_to_uf16_bits(-v_dc))
            _test_put_u16_le(dc16_bytes, hv)
        for v_dc in b_cb:
            var hv = f32_to_uf16_bits(v_dc) if v_dc >= 0 else (0x8000 | f32_to_uf16_bits(-v_dc))
            _test_put_u16_le(dc16_bytes, hv)
        for v_dc in b_cr:
            var hv = f32_to_uf16_bits(v_dc) if v_dc >= 0 else (0x8000 | f32_to_uf16_bits(-v_dc))
            _test_put_u16_le(dc16_bytes, hv)
        var dc16_comp = _wrap_zlib_stored(_exr_reorder_and_predict(dc16_bytes))
        # AC stream is block-major: b0 (3 EOB), b1 Y has 1 non-zero AC coeff (0x3800 = 0.5) + EOB then Cb/Cr EOB, b2 (3 EOB), b3 (3 EOB) -> 13 u16s!
        var ac16_u16 = List[UInt16]()
        for _ in range(3): ac16_u16.append(UInt16(0xFF00))  # b0
        ac16_u16.append(UInt16(f32_to_uf16_bits(Float32(0.5)))); ac16_u16.append(UInt16(0xFF00))  # b1 Y
        ac16_u16.append(UInt16(0xFF00)); ac16_u16.append(UInt16(0xFF00))  # b1 Cb, Cr
        for _ in range(6): ac16_u16.append(UInt16(0xFF00))  # b2, b3
        var ac16_comp: List[UInt8]
        if ac_mode == 1:
            var ac16_bytes = List[UInt8]()
            for av in ac16_u16: _test_put_u16_le(ac16_bytes, Int(av))
            ac16_comp = _wrap_zlib_stored(ac16_bytes)
        else:
            ac16_comp = _build_exr_huf_blob(ac16_u16)

        var dw16_chunk = List[UInt8]()
        _test_put_u64_le(dw16_chunk, 2)
        _test_put_u64_le(dw16_chunk, 0); _test_put_u64_le(dw16_chunk, 0)
        _test_put_u64_le(dw16_chunk, UInt64(len(ac16_comp)))
        _test_put_u64_le(dw16_chunk, UInt64(len(dc16_comp)))
        _test_put_u64_le(dw16_chunk, 0); _test_put_u64_le(dw16_chunk, 0); _test_put_u64_le(dw16_chunk, 0)
        _test_put_u64_le(dw16_chunk, UInt64(len(ac16_u16)))
        _test_put_u64_le(dw16_chunk, 12)
        _test_put_u64_le(dw16_chunk, UInt64(ac_mode))
        _test_put_u16_le(dw16_chunk, 11)
        dw16_chunk.append(UInt8(ord("R"))); dw16_chunk.append(0); dw16_chunk.append(0x14)
        dw16_chunk.append(UInt8(ord("G"))); dw16_chunk.append(0); dw16_chunk.append(0x24)
        dw16_chunk.append(UInt8(ord("B"))); dw16_chunk.append(0); dw16_chunk.append(0x34)
        for b in ac16_comp: dw16_chunk.append(b)
        for b in dc16_comp: dw16_chunk.append(b)
        _test_put_u32_le(exr16, 0); _test_put_u32_le(exr16, UInt32(len(dw16_chunk)))
        for b in dw16_chunk: exr16.append(b)
        var d16 = decode_exr_fp16(exr16)
        _assert(d16.valid and d16.width == 16 and d16.height == 16, "16x16 multi-block DWA decode failed")
        # Verify block (8,0) vs (15,0) reflects block_major AC[1] (>0 at x=8, < at x=15), and (0,8) is green-dominant, (8,8) is blue-dominant
        _assert(d16.pixels[0 * 16 + 8].x > d16.pixels[0 * 16 + 15].x, "16x16 DWA block_major AC gradient failed")
        _assert(d16.pixels[8 * 16 + 0].y > d16.pixels[8 * 16 + 0].x and d16.pixels[8 * 16 + 8].z > d16.pixels[8 * 16 + 8].y, "16x16 DWA [Y,Cb,Cr] CSC failed")

    # 5. Scanline EXR OOB y_coord (y=99..102) & incomplete/duplicate scanline coverage rejection
    var oob_scan_exr = List[UInt8]()
    _write_exr_header(oob_scan_exr, 4, 4, EXR_NO_COMPRESSION, tiled=False)
    var oob_tbl = len(oob_scan_exr)
    for _ in range(4): _test_put_u64_le(oob_scan_exr, 0)
    for row in range(4):
        var r_off = UInt64(len(oob_scan_exr))
        for bi in range(8): oob_scan_exr[oob_tbl + row * 8 + bi] = UInt8((r_off >> UInt64(bi * 8)) & 0xFF)
        _test_put_u32_le(oob_scan_exr, UInt32(99 + row))
        _test_put_u32_le(oob_scan_exr, 24)
        for _ in range(24): oob_scan_exr.append(0)
    _assert(not decode_exr_fp16(oob_scan_exr).valid, "Scanline EXR with OOB y_coord (99..102) must return valid=False")

    var dup_scan_exr = List[UInt8]()
    _write_exr_header(dup_scan_exr, 4, 4, EXR_NO_COMPRESSION, tiled=False)
    var dup_tbl = len(dup_scan_exr)
    for _ in range(4): _test_put_u64_le(dup_scan_exr, 0)
    var dup_ys: List[Int] = [0, 0, 2, 3]
    for row in range(4):
        var r_off = UInt64(len(dup_scan_exr))
        for bi in range(8): dup_scan_exr[dup_tbl + row * 8 + bi] = UInt8((r_off >> UInt64(bi * 8)) & 0xFF)
        _test_put_u32_le(dup_scan_exr, UInt32(dup_ys[row]))
        _test_put_u32_le(dup_scan_exr, 24)
        for _ in range(24): dup_scan_exr.append(0)
    _assert(not decode_exr_fp16(dup_scan_exr).valid, "Scanline EXR with duplicate/incomplete scanline coverage must return valid=False")

    print("    [OK] F20 residency/loose_root & F21 GLB/KTX2/SOF2/EXR codecs passed!")


def test_j_studio_v5_unification_ux_and_headless_smoke(tmp_root: String) raises:
    print("  [j] Testing F30 Studio EngineApp+World+Scene+PhysicsWorld unification, .flscn/.flscnb, multi-select, hierarchy reparenting & 1280x720 smoke...")

    # 1. Empty Studio starts with 0 ECS entities & real EngineApp plugins/resources
    var st = FlintStudioState()
    _assert(st.live_entity_count() == 0 and st.app.world.entity_count() == 0, "empty Studio must have 0 entities")
    _assert(st.app.resources.has[PhysicsWorld]() and st.app.resources.has[ScalableRenderer](), "Studio must host PhysicsWorld and ScalableRenderer resources")

    # Spawn parent, child, and sibling dynamic sphere
    var p_id = st.spawn_entity("ParentRoot", position=Vec3(2.0, 0.0, 0.0), collider_kind=COLLIDER_BOX, mass=0.0)
    var c_id = st.spawn_entity("ChildBox", position=Vec3(0.0, 2.0, 0.0), collider_kind=COLLIDER_BOX, mass=0.0)
    var d_id = st.spawn_entity("DynBall", position=Vec3(-2.0, 5.0, 0.0), collider_kind=COLLIDER_SPHERE, collider_half_extents=Vec3(0.5, 0.5, 0.5), mass=4.0)
    var p_e = st.resolve_entity(p_id)
    var c_e = st.resolve_entity(c_id)
    var d_e = st.resolve_entity(d_id)
    _assert(
        st.app.world.has[LocalTransform](p_e)
        and st.app.world.has[GlobalTransform](p_e)
        and st.app.world.has[MeshRendererComponent](p_e)
        and st.app.world.has[MaterialOverrideComponent](p_e)
        and st.app.world.has[RigidBodyComponent](d_e)
        and st.app.world.has[ColliderComponent](d_e)
        and st.app.world.has[StudioEntityMetaComponent](p_e)
        and st.app.world.has[StudioStyleComponent](p_e),
        "Studio entities must carry v5 ECS components in app.world",
    )

    # 1b. M2 Inactive-Body & STATIC -> DYNAMIC wind_area_over_mass regression inside Studio PhysicsWorld
    var inact_id = st.spawn_entity("InactiveBody", position=Vec3(5.0, 3.0, 0.0), collider_kind=COLLIDER_BOX, mass=0.0)
    var inact_e = st.resolve_entity(inact_id)
    ref phys_p0 = st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    var body_cnt_0 = len(phys_p0.bodies)
    st.app.world.get_ptr_mut[RigidBodyComponent](inact_e).value()[unsafe_offset=0].active = False
    for _ in range(5):
        _ = st.step_simulation(1)
    var body_cnt_1 = len(st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies)
    _assert(body_cnt_1 == body_cnt_0, "inactive RigidBodyComponent must not leak PhysicsWorld bodies per frame")
    var f_mass = InspectedField("mass", FIELD_F32)
    f_mass.f_val = Float32(6.0)
    _assert(st.patch_entity_field(inact_id, "mass", f_mass), "STATIC->DYNAMIC mass patch failed")
    var inact_bid = st.app.world.get_ptr[RigidBodyComponent](inact_e).value()[unsafe_offset=0].body_id
    var inact_slot = st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].find_slot_by_id_any(inact_bid)
    _assert(
        inact_slot >= 0 and st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies[inact_slot].wind_area_over_mass > Float32(0.0),
        "STATIC->DYNAMIC transition must recompute wind_area_over_mass",
    )
    _ = st.despawn_entity(inact_id, record_undo=False)
    st.sim_tick = 0

    # 1c. Challenger 1 Regression: `_apply_collider_and_rb` must preserve custom RigidBodyComponent & ColliderComponent fields
    var cust_id = st.spawn_entity("CustomPhysics", position=Vec3(3.0, 4.0, 1.0), collider_kind=COLLIDER_BOX, mass=5.0, record_undo=False)
    var cust_e = st.resolve_entity(cust_id)
    ref cust_rb = st.app.world.get_ptr_mut[RigidBodyComponent](cust_e).value()[unsafe_offset=0]
    cust_rb.linear_damping = Float32(0.27)
    cust_rb.angular_damping = Float32(0.63)
    cust_rb.rolling_resistance = Float32(0.19)
    cust_rb.gravity_scale = Float32(0.4)
    cust_rb.material = 7
    cust_rb.enable_ccd = True
    ref cust_col = st.app.world.get_ptr_mut[ColliderComponent](cust_e).value()[unsafe_offset=0]
    cust_col.local_offset = Vec3(0.1, 0.2, 0.3)
    cust_col.is_trigger = True

    var col_patch = InspectedField("base_color", FIELD_COLOR_RGB)
    col_patch.v3_val = Vec3(0.2, 0.7, 0.4)
    _assert(st.patch_entity_field(cust_id, "base_color", col_patch), "patch base_color on custom physics entity failed")
    _assert(st.set_entity_transform(cust_id, Vec3(3.5, 4.5, 1.5), Quat(0.0, 0.0, 0.0, 1.0), Vec3(1.0, 1.0, 1.0)), "set_entity_transform on custom physics entity failed")

    ref post_rb = st.app.world.get_ptr[RigidBodyComponent](cust_e).value()[unsafe_offset=0]
    _assert(
        abs(post_rb.linear_damping - Float32(0.27)) < Float32(1e-5)
        and abs(post_rb.angular_damping - Float32(0.63)) < Float32(1e-5)
        and abs(post_rb.rolling_resistance - Float32(0.19)) < Float32(1e-5)
        and abs(post_rb.gravity_scale - Float32(0.4)) < Float32(1e-5)
        and post_rb.material == 7
        and post_rb.enable_ccd,
        "patch_entity_field / set_entity_transform must preserve custom RigidBodyComponent fields",
    )
    ref post_col = st.app.world.get_ptr[ColliderComponent](cust_e).value()[unsafe_offset=0]
    _assert(
        abs(post_col.local_offset.x - Float32(0.1)) < Float32(1e-5)
        and abs(post_col.local_offset.y - Float32(0.2)) < Float32(1e-5)
        and abs(post_col.local_offset.z - Float32(0.3)) < Float32(1e-5)
        and post_col.is_trigger,
        "patch_entity_field / set_entity_transform must preserve custom ColliderComponent fields",
    )
    _ = st.despawn_entity(cust_id, record_undo=False)

    # 1d. Challenger 1 Regression: BODY_MOTION_KINEMATIC must survive unrelated edits, while DYNAMIC <-> STATIC tracks mass
    var kin_id = st.spawn_entity("KinematicBody", position=Vec3(-3.0, 2.0, 0.0), collider_kind=COLLIDER_BOX, mass=0.0, record_undo=False)
    var kin_e = st.resolve_entity(kin_id)
    st.app.world.get_ptr_mut[RigidBodyComponent](kin_e).value()[unsafe_offset=0].motion_mode = BODY_MOTION_KINEMATIC
    _assert(st.patch_entity_field(kin_id, "base_color", col_patch), "patch base_color on kinematic entity failed")
    _assert(
        st.app.world.get_ptr[RigidBodyComponent](kin_e).value()[unsafe_offset=0].motion_mode == BODY_MOTION_KINEMATIC,
        "unrelated edit must preserve BODY_MOTION_KINEMATIC",
    )
    var dyn_sw_mass = InspectedField("mass", FIELD_F32)
    dyn_sw_mass.f_val = Float32(3.0)
    _assert(st.patch_entity_field(kin_id, "mass", dyn_sw_mass), "kinematic->dynamic mass patch failed")
    _assert(
        st.app.world.get_ptr[RigidBodyComponent](kin_e).value()[unsafe_offset=0].motion_mode == BODY_MOTION_DYNAMIC,
        "positive mass edit must switch motion_mode to BODY_MOTION_DYNAMIC",
    )
    dyn_sw_mass.f_val = Float32(0.0)
    _assert(st.patch_entity_field(kin_id, "mass", dyn_sw_mass), "dynamic->static mass patch failed")
    _assert(
        st.app.world.get_ptr[RigidBodyComponent](kin_e).value()[unsafe_offset=0].motion_mode == BODY_MOTION_STATIC,
        "zero mass edit from dynamic must switch motion_mode to BODY_MOTION_STATIC",
    )
    _ = st.despawn_entity(kin_id, record_undo=False)

    # Also verify _sync_compat_view projects mass=0.0 for BODY_MOTION_KINEMATIC even if rb.mass > 0
    var kin2_id = st.spawn_entity("KinematicNonZeroMass", position=Vec3(-4.0, 2.0, 0.0), collider_kind=COLLIDER_BOX, mass=2.5, record_undo=False)
    var kin2_e = st.resolve_entity(kin2_id)
    st.app.world.get_ptr_mut[RigidBodyComponent](kin2_e).value()[unsafe_offset=0].motion_mode = BODY_MOTION_KINEMATIC
    st._sync_compat_view()
    _assert(st.entities[kin2_id].mass == Float32(0.0), "_sync_compat_view must project 0.0 mass for BODY_MOTION_KINEMATIC")
    _assert(st.patch_entity_field(kin2_id, "base_color", col_patch), "patch base_color on kinematic non-zero mass entity failed")
    _assert(
        st.app.world.get_ptr[RigidBodyComponent](kin2_e).value()[unsafe_offset=0].motion_mode == BODY_MOTION_KINEMATIC,
        "unrelated edit must preserve BODY_MOTION_KINEMATIC when rb.mass was non-zero",
    )
    _ = st.despawn_entity(kin2_id, record_undo=False)

    # 1e. Challenger 1 Regression (Iter 3): moving active=False dynamic entity must not leak PhysicsWorld bodies or churn body_id
    var z_eid = st.spawn_entity("InactiveDynLeakCheck", position=Vec3(0.0, 3.0, 0.0), collider_kind=COLLIDER_BOX, mass=2.0, record_undo=False)
    var z_e = st.resolve_entity(z_eid)
    st.app.world.get_ptr_mut[RigidBodyComponent](z_e).value()[unsafe_offset=0].active = False
    st._sync_ecs_and_physics()
    var z_cnt_before = len(st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies)
    var orig_bid = st.app.world.get_ptr[RigidBodyComponent](z_e).value()[unsafe_offset=0].body_id
    _assert(orig_bid >= 0, "inactive dynamic entity must keep valid body_id after sync")
    for mi in range(5):
        _assert(
            st.set_entity_transform(z_eid, Vec3(Float32(mi + 1), 3.0, 0.0), Quat(0.0, 0.0, 0.0, 1.0), Vec3(1.0, 1.0, 1.0)),
            "set_entity_transform on inactive dynamic entity failed",
        )
        var cur_bid = st.app.world.get_ptr[RigidBodyComponent](z_e).value()[unsafe_offset=0].body_id
        _assert(cur_bid == orig_bid, "transforming active=False dynamic entity must preserve body_id")
        var cur_cnt = len(st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies)
        _assert(cur_cnt == z_cnt_before, "transforming active=False dynamic entity must not leak PhysicsWorld bodies")
    _assert(st.despawn_entity(z_eid, record_undo=False), "despawn inactive dynamic entity failed")
    var z_cnt_after = len(st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies)
    _assert(z_cnt_after == z_cnt_before - 1, "despawn_entity must remove inactive dynamic entity's PhysicsWorld body cleanly")
    st.sim_tick = 0

    # 2. Hierarchy Reparenting (`set_parent` / `clear_parent`), GlobalTransform propagation, Cycle Guard & Undo/Redo
    var h_pre_rep = st.state_hash()
    _assert(st.reparent_entity(c_id, p_id, preserve_world_transform=False, record_undo=True), "reparent c->p failed")
    _assert(st.parent_of(c_id) == p_id, "parent_of(c) must equal p_id")
    _assert(st.app.world.has[Parent](c_e), "world must have Parent component on child")
    _assert(st.app.world.has[Children](p_e) and st.app.world.get_ptr[Children](p_e).value()[unsafe_offset=0].contains(c_e), "parent Children must contain child")
    _assert(st.app.world.get_ptr[HierarchyDepth](c_e).value()[unsafe_offset=0].depth == 1, "child HierarchyDepth must be 1")
    var c_gt = st.app.world.get_ptr[GlobalTransform](c_e).value()[unsafe_offset=0].copy()
    _assert(abs(c_gt.translation.x - Float32(2.0)) < Float32(1e-4) and abs(c_gt.translation.y - Float32(2.0)) < Float32(1e-4), "GlobalTransform must compose parent+child")

    # Cycle guard: self-parent & descendant-parent must return False
    _assert(not st.reparent_entity(p_id, p_id), "self-parent must be rejected")
    _assert(not st.reparent_entity(p_id, c_id), "cyclic parent->child reparent must be rejected")

    # Undo/redo reparenting
    var h_post_rep = st.state_hash()
    _assert(h_post_rep != h_pre_rep, "reparenting must mutate state_hash")
    _assert(st.undo() and st.state_hash() == h_pre_rep and st.parent_of(c_id) == -1 and not st.app.world.has[Parent](c_e), "undo reparent failed")
    _assert(st.redo() and st.state_hash() == h_post_rep and st.parent_of(c_id) == p_id, "redo reparent failed")

    # preserve_world_transform=True test
    _assert(st.reparent_entity(c_id, -1, preserve_world_transform=True, record_undo=True), "unparent with preserve_world failed")
    var c_gt_unp = st.app.world.get_ptr[GlobalTransform](c_e).value()[unsafe_offset=0].copy()
    _assert(abs(c_gt_unp.translation.x - Float32(2.0)) < Float32(1e-4), "preserve_world_transform=True must keep world X=2.0")
    _ = st.undo()  # back to c under p

    # 3. `.flscn` JSON & `.flscnb` LZ4 Binary Save/Load Round-Trip
    var flscn_path = tmp_root + "/studio_v5_scene.flscn"
    var flscnb_path = tmp_root + "/studio_v5_scene.flscnb"
    var h_save = st.state_hash()
    _assert(st.save_scene_file(flscn_path) and st.save_scene_file(flscnb_path), "save .flscn/.flscnb failed")

    var st_json = FlintStudioState()
    _assert(st_json.load_scene_file(flscn_path), "load .flscn failed")
    _assert(st_json.live_entity_count() == 3 and st_json.state_hash() == h_save, ".flscn load state_hash mismatch")
    _assert(st_json.parent_of(1) == 0, ".flscn load did not reconstruct Parent/Children")
    st_json.shutdown()

    var st_bin = FlintStudioState()
    _assert(st_bin.load_scene_file(flscnb_path), "load .flscnb failed")
    _assert(st_bin.live_entity_count() == 3 and st_bin.state_hash() == h_save, ".flscnb load state_hash mismatch")
    _assert(st_bin.parent_of(1) == 0, ".flscnb load did not reconstruct Parent/Children")
    st_bin.shutdown()

    # 4. PIE 120-Tick Rollback (World checksum AND PhysicsWorld body state bit-for-bit!)
    var pre_pie_world_sum = st.app.world.state_checksum()
    var pre_pie_hash = st.state_hash()
    var d_bid = st.app.world.get_ptr[RigidBodyComponent](d_e).value()[unsafe_offset=0].body_id
    var d_slot = st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].find_slot_by_id(d_bid)
    var pre_pie_body_cnt = len(st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies)
    var pre_pie_body_y = st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies[d_slot].pos.y
    var pre_pie_body_vy = st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies[d_slot].linear_vel.y

    _assert(st.pie_play(), "PIE play failed")
    _assert(st.pie_step(120) == 120, "PIE step 120 failed")
    _assert(st.entities[d_id].translation.y < Float32(1.5), "DynBall must fall under PhysicsWorld gravity during PIE")
    _assert(st.app.world.state_checksum() != pre_pie_world_sum, "World state_checksum must mutate during PIE")
    _assert(st.pie_stop(), "PIE stop failed")
    _assert(not st.scene.has_pie_snapshot and len(st.scene.pie_snapshot) == 0, "PIE stop must clear Scene PIE snapshot")

    _assert(st.state_hash() == pre_pie_hash, "PIE stop state_hash mismatch")
    _assert(st.app.world.state_checksum() == pre_pie_world_sum, "PIE stop must restore World.state_checksum bit-for-bit")
    ref phys_after = st.app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    var d_slot_after = phys_after.find_slot_by_id(d_bid)
    _assert(
        len(phys_after.bodies) == pre_pie_body_cnt
        and d_slot_after >= 0
        and phys_after.bodies[d_slot_after].pos.y == pre_pie_body_y
        and phys_after.bodies[d_slot_after].linear_vel.y == pre_pie_body_vy,
        "PIE stop must restore PhysicsWorld body count, pos, and velocity bit-for-bit",
    )

    # 5. Multi-Select (Ctrl toggle, Shift range over DFS tree including collapsed subtrees), Bulk Ops & UI
    st.select_entity(p_id)
    _assert(st.selected_count() == 1 and st.is_selected(p_id), "single select failed")
    st.select_entity(d_id, multi_select=True)
    _assert(st.selected_count() == 2 and st.is_selected(p_id) and st.is_selected(d_id), "Ctrl multi-select add failed")
    st.select_entity(p_id, multi_select=True)
    _assert(st.selected_count() == 1 and not st.is_selected(p_id) and st.is_selected(d_id), "Ctrl multi-select toggle remove failed")

    # Shift range over expanded tree [p_id, c_id, d_id]
    _assert(len(st.hierarchy_visible_rows()) == 3, "expanded hierarchy must show 3 visible rows")
    st.select_entity(p_id)
    st.select_entity(d_id, range_select=True)
    _assert(st.selected_count() == 3 and st.is_selected(p_id) and st.is_selected(c_id) and st.is_selected(d_id), "Shift range select failed")

    # Collapse p_id -> visible order hides c_id -> Shift range p_id..d_id selects only [p_id, d_id]!
    st.set_hierarchy_expanded(p_id, False)
    var collapsed_rows = st.hierarchy_visible_rows()
    _assert(len(collapsed_rows) == 2 and collapsed_rows[0].entity_id == p_id and collapsed_rows[1].entity_id == d_id, "collapsed parent must hide child rows (expected 2 visible rows)")
    st.select_entity(p_id)
    st.select_entity(d_id, range_select=True)
    _assert(st.selected_count() == 2 and st.is_selected(p_id) and st.is_selected(d_id) and not st.is_selected(c_id), "collapsed subtree must be skipped in Shift range")
    st.set_hierarchy_expanded(p_id, True)
    _assert(len(st.hierarchy_visible_rows()) == 3, "re-expanded hierarchy must show 3 visible rows")

    # Bulk translate & bulk patch with Undo/Redo
    st.select_entity(p_id)
    st.select_entity(d_id, multi_select=True)
    var h_pre_bulk = st.state_hash()
    _assert(st.bulk_translate_selected(Vec3(1.0, 0.0, 0.0), coalesce_key=777), "bulk translate 1 failed")
    _assert(st.bulk_translate_selected(Vec3(0.5, 0.0, 0.0), coalesce_key=777), "bulk translate 2 coalesce failed")
    _assert(abs(st.entities[p_id].translation.x - Float32(3.5)) < Float32(1e-4), "bulk translate p_id mismatch")
    _assert(abs(st.entities[d_id].translation.x - Float32(-0.5)) < Float32(1e-4), "bulk translate d_id mismatch")
    var f_bulk_r = InspectedField("roughness", FIELD_F32)
    f_bulk_r.f_val = Float32(0.91)
    _assert(st.bulk_patch_field("roughness", f_bulk_r) == 2, "bulk_patch_field failed")
    var h_post_bulk = st.state_hash()
    _assert(st.undo() and st.undo() and st.state_hash() == h_pre_bulk, "bulk undo mismatch")
    _assert(st.redo() and st.redo() and st.state_hash() == h_post_bulk, "bulk redo mismatch")

    # Bulk duplicate & bulk delete + Undo
    var dup_ids = st.duplicate_selected(record_undo=True)
    _assert(len(dup_ids) == 2 and st.live_entity_count() == 5, "duplicate_selected failed")
    _assert(st.despawn_selected(record_undo=True) == 2 and st.live_entity_count() == 3, "despawn_selected failed")
    _assert(st.undo() and st.live_entity_count() == 5 and st.undo() and st.state_hash() == h_post_bulk, "undo bulk delete/dup mismatch")

    # Simulated UI Hierarchy & Multi-Select click interaction (`emit_studio_ui_draw_list`)
    var ui_dl = UiDrawList()
    var sim_in = InputState()
    emit_studio_ui_draw_list(ui_dl, st, -1, SdfFont(), 1280, 720, input=sim_in)
    _assert(ui_dl.count() > 50, "Studio UI draw list must emit Hierarchy + Inspector + Viewport primitives")
    # Click [Unparent] button in Hierarchy when c_id is selected
    st.select_entity(c_id)
    emit_studio_ui_draw_list(ui_dl, st, -1, SdfFont(), 1280, 720, input=sim_in)
    var h_info = st.dock.find_panel("Hierarchy")
    _assert(h_info[0] >= 0 and h_info[2], "Hierarchy dock panel must be active")
    var cx_btn = h_info[3].x + Float32(28.0)
    var cy_btn = h_info[3].y + Float32(16.0)
    sim_in.begin_frame()
    sim_in.move_mouse(cx_btn, cy_btn)
    sim_in.set_mouse(0, True)
    emit_studio_ui_draw_list(ui_dl, st, -1, SdfFont(), 1280, 720, input=sim_in)
    sim_in.begin_frame()
    sim_in.move_mouse(cx_btn, cy_btn)
    sim_in.set_mouse(0, False)
    emit_studio_ui_draw_list(ui_dl, st, -1, SdfFont(), 1280, 720, input=sim_in)
    _assert(st.parent_of(c_id) == -1, "clicking Hierarchy Unparent button must unparent selected child")

    # 6. `.otf` Font Discovery in `scan_asset_browser`
    var otf_dir = tmp_root + "/ab_otf"
    _ = make_dirs(otf_dir)
    var dummy_otf = List[UInt8](length=16, fill=UInt8(0x4F))
    _assert(write_bytes(otf_dir + "/studio_ui.otf", dummy_otf), "write dummy .otf failed")
    var ab_cnt = st.scan_asset_browser(dir_path=otf_dir)
    _assert(ab_cnt == 1 and st.asset_browser_entries[0] == "studio_ui.otf", "scan_asset_browser must discover .otf fonts")
    st.shutdown()

    # 7. 1280x720 Headless Smoke Render & PNG Dimension Verification
    var smoke_studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_2_AAA_VISBUFFER_RT, width=1280, height=720)
    var s_grid_mid = 0
    var s_orb_mid = 0
    var s_rend_opt = smoke_studio.app.resources.get_ptr_mut[ScalableRenderer]()
    if s_rend_opt:
        ref s_rend = s_rend_opt.value()[unsafe_offset=0]
        var m0 = s_rend.register_material(MaterialAsset(base_color=Vec3(0.30, 0.35, 0.42), roughness=0.6, metallic=0.05))
        var m1 = s_rend.register_material(MaterialAsset(base_color=Vec3(0.85, 0.25, 0.20), roughness=0.2, metallic=0.75))
        s_grid_mid = s_rend.register_mesh(build_studio_reference_grid_mesh(16.0, 8), m0, build_lods=False)
        s_orb_mid = s_rend.register_mesh(build_subd_material_orb(0.88, 2), m1, build_lods=False)
    _ = smoke_studio.spawn_entity("SmokeGrid", position=Vec3(0.0, 0.0, 0.0), mesh_id=s_grid_mid, record_undo=False)
    _ = smoke_studio.spawn_entity("SmokeOrb", position=Vec3(0.0, 1.0, 0.0), base_color=Vec3(0.85, 0.25, 0.20), mesh_id=s_orb_mid, record_undo=False)
    smoke_studio.look_at(Vec3(-5.0, 3.5, -6.5), Vec3(0.0, 0.8, 0.0))
    smoke_studio.render_editor_frame(1280, 720)
    var smoke_px = smoke_studio.app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    _assert(len(smoke_px) == 1280 * 720, "1280x720 headless Studio smoke pixel buffer size mismatch")
    var smoke_png = tmp_root + "/studio_smoke_1280x720.png"
    _assert(write_png(smoke_png, smoke_px, 1280, 720), "write headless smoke PNG failed")
    var smoke_dec = decode_png(read_bytes(smoke_png), maximum_size=2048)
    _assert(smoke_dec.valid and smoke_dec.width == 1280 and smoke_dec.height == 720, "decoded Studio smoke PNG must be 1280x720")
    var non_uniform = False
    var first_px = smoke_dec.pixels[0]
    for pi in range(0, len(smoke_dec.pixels), 97):
        if smoke_dec.pixels[pi] != first_px:
            non_uniform = True
            break
    _assert(non_uniform, "Studio smoke PNG must have non-uniform rendered pixels")
    smoke_studio.shutdown()

    print("    [OK] F30 Studio EngineApp+World+Scene+PhysicsWorld unification, UX & 1280x720 smoke passed!")


def _build_uv_seamed_torus_3072t() -> NativeMesh:
    """Builds a `32 x 48` (`3,072`-triangle) closed torus with an explicit UV seam along `u = 0 / 1` and `v = 0 / 1`."""
    var m = NativeMesh()
    comptime MAJ_SEGS: Int = 48
    comptime MIN_SEGS: Int = 32
    comptime R_MAJ: Float32 = 2.0
    comptime R_MIN: Float32 = 0.65
    comptime TAU: Float32 = 6.28318530

    for i in range(MAJ_SEGS + 1):
        var tu = Float32(i) / Float32(MAJ_SEGS)
        var phi = tu * TAU
        var cp = cos(phi)
        var sp = sin(phi)
        for j in range(MIN_SEGS + 1):
            var tv = Float32(j) / Float32(MIN_SEGS)
            var theta = tv * TAU
            var ct = cos(theta)
            var st = sin(theta)
            var ring_r = R_MAJ + R_MIN * ct
            var px = ring_r * cp
            var py = R_MIN * st
            var pz = ring_r * sp
            var nx = ct * cp
            var ny = st
            var nz = ct * sp
            m.positions.append(px); m.positions.append(py); m.positions.append(pz)
            m.normals.append(nx); m.normals.append(ny); m.normals.append(nz)
            m.texcoords.append(tu); m.texcoords.append(tv)
            m.ao.append(UInt8(220))
            var cr = UInt8(60 + Int(tu * Float32(160.0)))
            var cg = UInt8(90 + Int(tv * Float32(120.0)))
            m.colors.append(cr); m.colors.append(cg); m.colors.append(UInt8(210)); m.colors.append(UInt8(255))

    var stride = MIN_SEGS + 1
    for i in range(MAJ_SEGS):
        for j in range(MIN_SEGS):
            var i0 = i * stride + j
            var i1 = i0 + 1
            var i2 = i0 + stride + 1
            var i3 = i0 + stride
            m.indices.append(i0); m.indices.append(i1); m.indices.append(i2)
            m.indices.append(i0); m.indices.append(i2); m.indices.append(i3)
    return m^


def _count_open_spatial_boundary_edges(meshes: List[NativeMesh]) -> Int:
    """Welds spatial vertex positions within `1e-4` across `meshes` and counts undirected edges with `tri_count != 2`."""
    var rep_pos = List[Vec3]()
    var all_tris = List[Int]()

    for m_i in range(len(meshes)):
        ref m = meshes[m_i]
        var vc = m.vertex_count()
        var v_map = List[Int](length=vc, fill=0)
        for v in range(vc):
            var p = Vec3(m.positions[v * 3], m.positions[v * 3 + 1], m.positions[v * 3 + 2])
            var found = -1
            for r in range(len(rep_pos)):
                if (rep_pos[r] - p).length_squared() <= Float32(1e-8):
                    found = r
                    break
            if found >= 0:
                v_map[v] = found
            else:
                v_map[v] = len(rep_pos)
                rep_pos.append(p)
        var tc = len(m.indices) // 3
        for t in range(tc):
            var a = v_map[m.indices[t * 3]]
            var b = v_map[m.indices[t * 3 + 1]]
            var c = v_map[m.indices[t * 3 + 2]]
            if a != b and b != c and c != a:
                all_tris.append(a); all_tris.append(b); all_tris.append(c)

    var e_u = List[Int]()
    var e_v = List[Int]()
    var e_cnt = List[Int]()
    var n_t = len(all_tris) // 3
    for t in range(n_t):
        var vs = [all_tris[t * 3], all_tris[t * 3 + 1], all_tris[t * 3 + 2]]
        for k in range(3):
            var u0 = min(vs[k], vs[(k + 1) % 3])
            var u1 = max(vs[k], vs[(k + 1) % 3])
            var idx = -1
            for ei in range(len(e_u)):
                if e_u[ei] == u0 and e_v[ei] == u1:
                    idx = ei
                    break
            if idx >= 0:
                e_cnt[idx] += 1
            else:
                e_u.append(u0); e_v.append(u1); e_cnt.append(1)

    var non_manifold = 0
    for ei in range(len(e_cnt)):
        if e_cnt[ei] != 2:
            non_manifold += 1
    return non_manifold


def _assert_valid_mesh_attribs(m: NativeMesh, label: String, require_colors: Bool = False) raises:
    var vc = m.vertex_count()
    var tc = len(m.indices) // 3
    _assert(vc >= 3, label + ": vertex_count >= 3")
    _assert(tc >= 1, label + ": triangle_count >= 1")
    _assert(len(m.normals) == vc * 3, label + ": normals length == 3 * V")
    _assert(len(m.texcoords) == vc * 2, label + ": texcoords length == 2 * V")
    _assert(len(m.ao) == vc, label + ": ao length == V")
    if require_colors:
        _assert(len(m.colors) == vc * 4, label + ": colors length == 4 * V")
    for idx in m.indices:
        if idx < 0 or idx >= vc:
            raise Error(label + ": out-of-range triangle index")
    for v in range(vc):
        var n = Vec3(m.normals[v * 3], m.normals[v * 3 + 1], m.normals[v * 3 + 2])
        if abs(n.length() - Float32(1.0)) > Float32(0.05):
            raise Error(label + ": non-unit vertex normal")


def test_qem_minheap_speed_and_validity() raises:
    var torus = _build_uv_seamed_torus_3072t()
    _assert(torus.triangle_count() == 3072, "input torus has 3,072 triangles")

    var t_start_ns = monotonic()
    var lod1 = simplify_mesh_qem(torus, 1536)
    var lod2 = simplify_mesh_qem(lod1[0], 768)
    var lod3 = simplify_mesh_qem(lod2[0], 384)
    var elapsed_ms = Float64(monotonic() - t_start_ns) / 1_000_000.0
    print("    [Benchmark] 3,072-tri 3-LOD QEM chain simplified in", elapsed_ms, "ms")

    _assert(elapsed_ms < 15.0, "3,072-tri 3-step QEM simplification completes in < 15 ms")
    _assert(lod1[0].triangle_count() <= 1560 and lod1[0].triangle_count() >= 1400, "LOD 1 reaches ~1,536 tris")
    _assert(lod2[0].triangle_count() <= 800 and lod2[0].triangle_count() >= 700, "LOD 2 reaches ~768 tris")
    _assert(lod3[0].triangle_count() <= 420 and lod3[0].triangle_count() >= 340, "LOD 3 reaches ~384 tris")
    _assert(lod1[1] > Float32(0.0) and lod2[1] > Float32(0.0) and lod3[1] > Float32(0.0), "positive QEM geometric errors")

    _assert_valid_mesh_attribs(lod3[0], "QEM LOD3 torus", require_colors=True)
    _assert(Int(lod3[0].colors[2]) == 210, "QEM LOD3 preserves vertex color blue channel across collapses")

    var m_list = List[NativeMesh]()
    m_list.append(lod3[0].copy())
    _assert(_count_open_spatial_boundary_edges(m_list) == 0, "UV-seamed torus stays 100% watertight after 8x QEM decimation")


def test_vertex_words_11_ao_and_splat_packing() raises:
    var m_std = NativeMesh()
    m_std.positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    m_std.normals = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    m_std.texcoords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    m_std.ao = [UInt8(255), UInt8(128), UInt8(0)]
    m_std.indices = [0, 2, 1]

    var pool = AaaGeometryPool()
    var entry_std = pool.append_mesh(m_std, material_id=0, build_lods=False)
    _assert(entry_std.vertex_count == 3, "standard mesh stores 3 vertices")

    var w11_v0 = pool.vertex_words[0 * 12 + 11]
    var w11_v1 = pool.vertex_words[1 * 12 + 11]
    var w11_v2 = pool.vertex_words[2 * 12 + 11]

    _assert(Int((w11_v0 >> 1) & 1) == 0, "standard mesh bit 1 (is_terrain_splat) == 0")
    _assert(Int((w11_v0 >> 2) & 0x3F) == 63, "ao_byte=255 packs to ao_u6=63 in bits 2..7")
    _assert(Int(w11_v0 >> 8) == 0xFFFFFF, "uncolored standard mesh defaults bits 8..31 to white 0xFFFFFF")
    _assert(Int((w11_v1 >> 2) & 0x3F) == 32, "ao_byte=128 packs to ao_u6=32")
    _assert(Int((w11_v2 >> 2) & 0x3F) == 0, "ao_byte=0 packs to ao_u6=0")

    var m_col = m_std.copy()
    m_col.colors = [
        UInt8(0x12), UInt8(0x34), UInt8(0x56), UInt8(255),
        UInt8(0xFE), UInt8(0x80), UInt8(0x05), UInt8(255),
        UInt8(0x00), UInt8(0x00), UInt8(0x00), UInt8(255),
    ]
    var entry_col = pool.append_mesh(m_col, material_id=1, build_lods=False)
    var base_cv = entry_col.first_vertex
    var cw_v0 = pool.vertex_words[(base_cv + 0) * 12 + 11]
    var cw_v1 = pool.vertex_words[(base_cv + 1) * 12 + 11]
    var cw_v2 = pool.vertex_words[(base_cv + 2) * 12 + 11]

    _assert(Int((cw_v0 >> 1) & 1) == 0, "colored standard mesh bit 1 == 0")
    _assert(Int((cw_v0 >> 8) & 0xFF) == 0x12, "colored v0 r8 == 0x12 in bits 8..15")
    _assert(Int((cw_v0 >> 16) & 0xFF) == 0x34, "colored v0 g8 == 0x34 in bits 16..23")
    _assert(Int((cw_v0 >> 24) & 0xFF) == 0x56, "colored v0 b8 == 0x56 in bits 24..31")
    _assert(Int((cw_v1 >> 8) & 0xFF) == 0xFE, "colored v1 r8 == 0xFE")
    _assert(Int((cw_v1 >> 16) & 0xFF) == 0x80, "colored v1 g8 == 0x80")
    _assert(Int((cw_v1 >> 24) & 0xFF) == 0x05, "colored v1 b8 == 0x05")
    _assert(Int(cw_v2 >> 8) == 0x010000, "pure-black [0,0,0] clamps b8=1 so (meta >> 8) == 0x010000 != 0")

    var splat_w = List[UInt32]()
    splat_w.append(UInt32(0xC37F2A))
    splat_w.append(UInt32(0x112233))
    splat_w.append(UInt32(0xABCDEF))
    var splat_id = pool.add_splat_mesh(m_std, splat_w, build_lods=False)
    _assert(splat_id == 2, "add_splat_mesh returns mesh_id == 2")

    var base_sv = pool.meshes[splat_id].first_vertex
    var sw_v0 = pool.vertex_words[(base_sv + 0) * 12 + 11]
    _assert(Int((sw_v0 >> 1) & 1) == 1, "splat mesh bit 1 (is_terrain_splat) == 1")
    _assert(Int((sw_v0 >> 2) & 0x3F) == 63, "splat mesh preserves ao_u6=63 in bits 2..7")
    _assert(Int((sw_v0 >> 8) & 0xFF) == 0x2A, "bits 8..15 == w_sand (0x2A)")
    _assert(Int((sw_v0 >> 16) & 0xFF) == 0x7F, "bits 16..23 == w_rock (0x7F)")
    _assert(Int((sw_v0 >> 24) & 0xFF) == 0xC3, "bits 24..31 == w_cobble (0xC3)")


def test_toksvig_normal_variance_roughness() raises:
    comptime W: Int = 8
    comptime H: Int = 8
    var flat_n = List[Vec3](length=W * H, fill=Vec3(0.0, 0.0, 1.0))
    var orm_mirror = List[UInt32](length=W * H, fill=UInt32(0xFFFF1000))

    var ds_flat = downsample_normals_and_orm_toksvig_2x(flat_n, orm_mirror, W, H, toksvig_scale=1.0)
    var flat_g = Int((ds_flat[1][0] >> 8) & 255)
    _assert(flat_g == 16, "zero-variance flat normals preserve exact mirror roughness G=16")

    var bumpy_n = List[Vec3](capacity=W * H)
    for y in range(H):
        for x in range(W):
            var sx = Float32(-0.62) if ((x & 1) == 0) else Float32(0.62)
            var sy = Float32(-0.62) if ((y & 1) == 0) else Float32(0.62)
            bumpy_n.append(Vec3(sx, sy, Float32(0.48)).normalized())

    var ds_bumpy = downsample_normals_and_orm_toksvig_2x(bumpy_n, orm_mirror, W, H, toksvig_scale=1.0)
    var bumpy_g = Int((ds_bumpy[1][0] >> 8) & 255)
    _assert(bumpy_g > 160, "high-variance 2x2 normals dramatically boost Toksvig roughness (G > 160 vs 16)")
    _assert(abs(ds_bumpy[0][0].length() - Float32(1.0)) <= Float32(0.005), "downsampled Toksvig normal is normalized for BC5")

    var wave_n = List[Vec3](capacity=W * H)
    for _ in range(H):
        for x in range(W):
            var wx = Float32(0.35) if (((x >> 1) & 1) == 0) else Float32(-0.35)
            var nx = Float32(0.18) if ((x & 1) == 0) else Float32(-0.18)
            wave_n.append(Vec3(wx + nx, 0.0, Float32(0.90)).normalized())

    var pyr = bake_toksvig_normal_orm_pyramid(wave_n, orm_mirror, W, H, toksvig_scale=1.0)
    var g_mip0 = Int((pyr[1][0][0] >> 8) & 255)
    var g_mip1 = Int((pyr[1][1][0] >> 8) & 255)
    var g_mip2 = Int((pyr[1][2][0] >> 8) & 255)
    _assert(g_mip1 > g_mip0, "Toksvig mip 1 roughness > mip 0 roughness")
    _assert(g_mip2 > g_mip1, "Toksvig mip 2 roughness > mip 1 roughness across coarser normal wave")


def test_subd_creases_and_vehicles() raises:
    var c_smooth = SubdCage()
    var c_sharp = SubdCage()
    for sx in range(-1, 2, 2):
        for sy in range(-1, 2, 2):
            for sz in range(-1, 2, 2):
                var p = Vec3(Float32(sx), Float32(sy), Float32(sz))
                _ = c_smooth.add_vert(p)
                _ = c_sharp.add_vert(p)
    var faces = [
        [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
        [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3],
    ]
    for f_i in range(6):
        ref f = faces[f_i]
        c_smooth.add_quad(f[0], f[1], f[2], f[3])
        c_sharp.add_quad(f[0], f[1], f[2], f[3])
        for k in range(4):
            c_sharp.set_edge_sharpness(f[k], f[(k + 1) % 4], Float32(5.0))

    var grp_meshes = creased_subdivide(c_smooth, levels=2, split_sharp_threshold=1.4)
    _assert(len(grp_meshes) == 1, "single-group SubdCage emits 1 group mesh")
    var m_smooth = subdivide_to_mesh(c_smooth, levels=2, split_sharp_threshold=1.4)
    var m_sharp = subdivide_to_mesh(c_sharp, levels=2, split_sharp_threshold=1.4)
    _assert(m_smooth.triangle_count() == 192, "2-level Catmull-Clark cube yields 6 * 32 = 192 tris")
    _assert(m_sharp.triangle_count() == 192, "creased cube preserves exact 192-tri quad topology")
    _assert(m_sharp.vertex_count() > m_smooth.vertex_count(), "sharp creases split hard normal wedges cleanly")
    m_smooth.ensure_bounds()
    m_sharp.ensure_bounds()
    _assert(m_sharp.bounds_radius > m_smooth.bounds_radius + Float32(0.35), "sharp creases preserve crisp corner extent vs smooth sphere shrink")

    var sm_list = List[NativeMesh]()
    sm_list.append(m_smooth.copy())
    _assert(_count_open_spatial_boundary_edges(sm_list) == 0, "subdivided cube is 100% watertight")

    var sc = build_subd_supercar_body_and_glass()
    var sc_tris = sc[0].triangle_count() + sc[1].triangle_count()
    _assert(sc_tris >= 2200 and sc_tris <= 2850, "SubD supercar is within ~2,200-2,800 tri budget")
    _assert_valid_mesh_attribs(sc[0], "Supercar body")
    _assert_valid_mesh_attribs(sc[1], "Supercar glass")
    var sc_list = List[NativeMesh]()
    sc_list.append(sc[0].copy()); sc_list.append(sc[1].copy())
    _assert(_count_open_spatial_boundary_edges(sc_list) == 0, "mirror-welded SubD supercar body+glass is 100% watertight")

    var sp = build_subd_spinner_hovercar_body_and_canopy()
    var sp_tris = sp[0].triangle_count() + sp[1].triangle_count()
    _assert(sp_tris >= 1650 and sp_tris <= 2050, "SubD Spinner hovercar is ~1,856 tris")
    var sp_list = List[NativeMesh]()
    sp_list.append(sp[0].copy()); sp_list.append(sp[1].copy())
    _assert(_count_open_spatial_boundary_edges(sp_list) == 0, "twin-prong SubD Spinner hovercar is 100% watertight")

    for arch in range(3):
        var tv = build_subd_traffic_vehicle(arch)
        _assert(tv.triangle_count() >= 1200 and tv.triangle_count() <= 1600, "traffic vehicle archetype within ~1,200-1,600 tris")
        _assert_valid_mesh_attribs(tv, "Traffic vehicle " + String(arch))
        var tv_list = List[NativeMesh]()
        tv_list.append(tv^)
        _assert(_count_open_spatial_boundary_edges(tv_list) == 0, "traffic vehicle is watertight")

    for style in range(4):
        var cc = build_subd_coaster_car_shell(style)
        _assert(cc.triangle_count() >= 1100 and cc.triangle_count() <= 1550, "coaster car shell within ~1,100-1,500 tris")
        _assert_valid_mesh_attribs(cc, "Coaster car " + String(style))
        var cc_list = List[NativeMesh]()
        cc_list.append(cc^)
        _assert(_count_open_spatial_boundary_edges(cc_list) == 0, "coaster car shell is watertight")


def test_loft_bevel_and_lathed_wheels() raises:
    var wh = build_lathed_wheel_assembly(0.35, 0.28, spoke_count=5)
    var total_w_tris = wh[0].triangle_count() + wh[1].triangle_count() + wh[2].triangle_count()
    _assert(total_w_tris >= 360 and total_w_tris <= 480, "5-spoke lathed wheel+rotor+caliper is ~420 tris")
    _assert_valid_mesh_attribs(wh[0], "Lathed tire")
    _assert_valid_mesh_attribs(wh[1], "Alloy rim & rotor")
    _assert_valid_mesh_attribs(wh[2], "6-piston brake caliper")

    var girder_prof = make_chamfered_girder_profile()
    _assert(len(girder_prof) == 16, "chamfered box-girder spine profile has 16 vertices")

    var loop_pts = List[Vec3]()
    for i in range(24):
        var a = (Float32(i) / Float32(23.0)) * Float32(6.2831853)
        loop_pts.append(Vec3(Float32(i) * Float32(0.15), Float32(5.0) - cos(a) * Float32(4.0), sin(a) * Float32(4.0)))
    var loop_mesh = sweep_profile_rmf(loop_pts, girder_prof, closed_loop=False, closed_profile=True)
    _assert_valid_mesh_attribs(loop_mesh, "RMF vertical loop girder")

    var curb_prof = make_curb_profile()
    var curb_pts: List[Vec3] = [Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), Vec3(2.0, 0.0, 10.0)]
    var curb_mesh = sweep_profile_rmf(curb_pts, curb_prof)
    _assert_valid_mesh_attribs(curb_mesh, "RMF street curb")

    var cable = sweep_catenary_cable(Vec3(-4.0, 5.0, 0.0), Vec3(4.0, 5.0, 0.0), sag=0.65)
    _assert_valid_mesh_attribs(cable, "Catenary cable")

    var sq: List[Vec2] = [Vec2(-1.0, -1.0), Vec2(1.0, -1.0), Vec2(1.0, 1.0), Vec2(-1.0, 1.0)]
    var ext = extrude_profile_beveled(sq, height=1.2, corner_fillet=0.15, cap_chamfer=0.06)
    _assert_valid_mesh_attribs(ext, "Beveled extrusion")


def _assert_botanical_cutout_atlas_512(
    sprig: Tuple[List[UInt32], List[Vec3]],
    label: String,
    min_opaque: Int,
    min_cutout: Int,
) raises:
    _assert(len(sprig[0]) == 512 * 512, label + ": 512x512 ARGB texels")
    _assert(len(sprig[1]) == 512 * 512, label + ": 512x512 BC5 tangent normals")
    var opaque_px = 0
    var cutout_px = 0
    for i in range(len(sprig[0])):
        var a = Int((sprig[0][i] >> 24) & 255)
        if a >= 200:
            opaque_px += 1
        elif a == 0:
            cutout_px += 1
    _assert(opaque_px >= min_opaque, label + ": rich botanical leaf/needle coverage")
    _assert(cutout_px >= min_cutout, label + ": crisp alpha=0 sky cutout gaps between leaves/needles")
    _assert(abs(sprig[1][256 * 512 + 256].length() - Float32(1.0)) <= Float32(0.02), label + ": unit tangent normal")


def test_foliage_crowd_and_arch_kits() raises:
    test_reference_fern()
    var oak_sprig = bake_oak_leaf_sprig_512()
    _assert_botanical_cutout_atlas_512(oak_sprig, "Oak leaf sprig 512", min_opaque=25000, min_cutout=60000)

    var pine_sprig = bake_conifer_needle_sprig_512()
    _assert_botanical_cutout_atlas_512(pine_sprig, "Conifer needle sprig 512", min_opaque=18000, min_cutout=60000)

    var palm_sprig = bake_palm_frond_512()
    _assert_botanical_cutout_atlas_512(palm_sprig, "Palm frond 512", min_opaque=20000, min_cutout=60000)

    var bark = bake_fissured_bark_256()
    _assert(len(bark[0]) == 256 * 256, "fissured bark albedo is 256x256")
    _assert(len(bark[1]) == 256 * 256, "fissured bark normal map is 256x256")
    _assert(abs(bark[1][128 * 256 + 64].length() - Float32(1.0)) <= Float32(0.02), "fissured bark normal is normalized")

    var pine = build_botanical_pine(42, 5.5)
    var pine_tris = pine[0].triangle_count() + pine[1].triangle_count()
    _assert(pine_tris >= 500 and pine_tris <= 1300, "botanical pine within cross-foil sprig budget")
    _assert_valid_mesh_attribs(pine[0], "Pine trunk", require_colors=True)
    _assert_valid_mesh_attribs(pine[1], "Pine needles", require_colors=True)
    _assert(not pine[0].double_sided, "pine trunk is single-sided closed bark")
    _assert(pine[1].alpha_mode == 1, "pine canopy uses alpha_mode == 1 mask cutout")
    _assert(pine[1].double_sided, "pine canopy is double-sided")

    var oak0 = build_botanical_oak(0, 6.0)
    var oak1 = build_botanical_oak(1, 6.0)
    var oak2 = build_botanical_oak(2, 6.0)
    var oak_tris = oak0[0].triangle_count() + oak0[1].triangle_count()
    _assert(oak_tris >= 500 and oak_tris <= 1400, "botanical oak within cross-foil sprig budget")
    _assert_valid_mesh_attribs(oak0[0], "Oak trunk", require_colors=True)
    _assert_valid_mesh_attribs(oak0[1], "Oak canopy", require_colors=True)
    _assert(not oak0[0].double_sided, "oak trunk is single-sided")
    _assert(oak0[1].alpha_mode == 1, "oak canopy uses alpha_mode == 1")
    _assert(oak0[1].double_sided, "oak canopy is double-sided")

    var seed_diff_01 = Float32(0.0)
    var seed_diff_12 = Float32(0.0)
    var n_cmp = min(len(oak0[1].positions), min(len(oak1[1].positions), len(oak2[1].positions)))
    for i in range(n_cmp):
        seed_diff_01 += abs(oak0[1].positions[i] - oak1[1].positions[i])
        seed_diff_12 += abs(oak1[1].positions[i] - oak2[1].positions[i])
    _assert(seed_diff_01 > Float32(25.0) and seed_diff_12 > Float32(25.0), "seeds 0, 1, 2 produce distinct branch & crown silhouettes")

    var min_ao = 255
    var max_ao = 0
    for ao_b in oak0[1].ao:
        var v = Int(ao_b)
        if v < min_ao: min_ao = v
        if v > max_ao: max_ao = v
    _assert(min_ao <= 150 and max_ao >= 230, "oak canopy bakes radial crown AO gradient")

    var palm = build_botanical_palm(19, 5.8)
    _assert_valid_mesh_attribs(palm[0], "Palm trunk", require_colors=True)
    _assert_valid_mesh_attribs(palm[1], "Palm fronds", require_colors=True)
    _assert(palm[1].alpha_mode == 1, "palm fronds use alpha_mode == 1")

    var ginkgo = build_urban_ginkgo_tree(31, 5.2)
    _assert_valid_mesh_attribs(ginkgo[0], "Ginkgo trunk", require_colors=True)
    _assert_valid_mesh_attribs(ginkgo[1], "Ginkgo canopy", require_colors=True)
    _assert(ginkgo[1].alpha_mode == 1, "ginkgo canopy uses alpha_mode == 1")

    var tuft = build_grass_wildflower_tuft(7)
    _assert_valid_mesh_attribs(tuft, "Grass wildflower tuft", require_colors=True)
    _assert(tuft.triangle_count() >= 12 and tuft.triangle_count() <= 48, "grass/wildflower tuft is lightweight")

    var hedge_pair = build_manicured_hedge_and_planter(3.2, 1.1, 0.95)
    _assert_valid_mesh_attribs(hedge_pair[0], "Planter stone curb", require_colors=True)
    _assert_valid_mesh_attribs(hedge_pair[1], "Manicured hedge foliage", require_colors=True)

    var bush = build_flowering_bush(11, 0.85)
    _assert_valid_mesh_attribs(bush, "Flowering bush", require_colors=True)

    var mech_v3 = build_subd_cyber_enforcer_rigged_v3()
    _assert(mech_v3[0].bone_count == ENFORCER_BONE_COUNT, "Armor submesh has 16 bones")
    _assert(mech_v3[1].bone_count == ENFORCER_BONE_COUNT, "Chassis submesh has 16 bones")
    _assert(mech_v3[2].bone_count == ENFORCER_BONE_COUNT, "Emissive submesh has 16 bones")
    _assert(len(mech_v3[0].rig_model.nodes) == 16, "Armor rig_model has 16 NativeNodes")
    _assert_valid_mesh_attribs(mech_v3[0].mesh, "Cyber-Enforcer Armor", require_colors=True)
    _assert_valid_mesh_attribs(mech_v3[1].mesh, "Cyber-Enforcer Chassis", require_colors=True)
    _assert_valid_mesh_attribs(mech_v3[2].mesh, "Cyber-Enforcer HDR Emissive", require_colors=True)
    _assert(mech_v3[2].mesh.emissive_strength > Float32(1.0), "Cyber-Enforcer emissive submesh carries HDR emissive_strength")

    var mech = build_subd_cyber_enforcer_rigged()
    _assert(mech.bone_count == ENFORCER_BONE_COUNT, "Merged Cyber-Enforcer has 16 hierarchical bones")
    _assert(
        mech.mesh.triangle_count()
        == mech_v3[0].mesh.triangle_count() + mech_v3[1].mesh.triangle_count() + mech_v3[2].mesh.triangle_count(),
        "Merged Cyber-Enforcer equals sum of Armor + Chassis + Emissive triangles",
    )
    _assert_valid_mesh_attribs(mech.mesh, "Merged Cyber-Enforcer SubD mesh", require_colors=True)
    for v in range(mech.mesh.vertex_count()):
        var w_sum = mech.mesh.weights[v * 4] + mech.mesh.weights[v * 4 + 1] + mech.mesh.weights[v * 4 + 2] + mech.mesh.weights[v * 4 + 3]
        if abs(w_sum - Float32(1.0)) > Float32(1e-3):
            raise Error("unnormalized skin weight on Cyber-Enforcer")

    var crowd_dry = build_articulated_crowd_pose_bank(with_umbrella=False)
    var crowd_rain = build_articulated_crowd_pose_bank(with_umbrella=True)
    _assert(len(crowd_dry) == 12, "dry crowd bank has 12 poses")
    _assert(len(crowd_rain) == 12, "rain umbrella crowd bank has 12 poses")
    var dry_tris = crowd_dry[0][0].triangle_count() + crowd_dry[0][1].triangle_count()
    var rain_tris = crowd_rain[0][0].triangle_count() + crowd_rain[0][1].triangle_count()
    _assert(dry_tris >= 580 and dry_tris <= 780, "crowd pose is ~680 tris")
    _assert(rain_tris > dry_tris + 40, "12-ribbed domed umbrella adds canopy + shaft geometry")
    _assert_valid_mesh_attribs(crowd_rain[0][0], "Crowd upper garment", require_colors=True)
    _assert_valid_mesh_attribs(crowd_rain[0][1], "Crowd lower & umbrella", require_colors=True)

    var diff_accum = Float32(0.0)
    var pos0 = crowd_dry[0][1].positions.copy()
    var pos2 = crowd_dry[2][1].positions.copy()
    for i in range(len(pos0)):
        diff_accum += abs(pos0[i] - pos2[i])
    _assert(diff_accum > Float32(5.0), "walk cycle phases deform leg/shoe positions distinctly")

    var atlas = bake_interior_windowbox_atlas_512()
    _assert(len(atlas[0]) == 512 * 512, "interior albedo atlas is 512x512")
    _assert(len(atlas[1]) == 512 * 512, "mullion+warble normal atlas is 512x512")
    _assert(len(atlas[2]) == 512 * 512, "interior ORM atlas is 512x512")
    var border_px = atlas[0][64 * 512 + 1]
    var border_luma = Int((border_px >> 16) & 255) + Int((border_px >> 8) & 255) + Int(border_px & 255)
    _assert(border_luma > 45, "interior room tile edge has no black outer window frame")
    var glass_n = atlas[1][64 * 512 + 64]
    var mull_n = atlas[1][64 * 512 + 1]
    _assert(abs(glass_n.x) <= Float32(0.03) and abs(glass_n.y) <= Float32(0.03), "interior glass pane warble within +-1.2 deg")
    _assert(abs(mull_n.x) > Float32(0.30), "mullion border normal has crisp bevel tilt")

    var bb0 = bake_cyber_billboard_512(0)
    var bb3 = bake_cyber_billboard_512(3)
    for v_idx in range(8):
        var bb = bake_cyber_billboard_512(v_idx)
        _assert(len(bb) == 512 * 512, "cyber billboard variant " + String(v_idx) + " is 512x512")
        var bright_neon = 0
        for p_i in range(0, 512 * 512, 16):
            var px = bb[p_i]
            var mr = max(Int((px >> 16) & 255), max(Int((px >> 8) & 255), Int(px & 255)))
            if mr >= 180:
                bright_neon += 1
        _assert(bright_neon >= 300, "cyber billboard variant " + String(v_idx) + " has vibrant neon artwork/glyphs")
    var bb_diff = 0
    for p_i in range(0, 512 * 512, 32):
        if bb0[p_i] != bb3[p_i]:
            bb_diff += 1
    _assert(bb_diff > 1000, "cyber billboard variants depict distinct graphic layouts")

    _assert_valid_mesh_attribs(build_colonnade_arcade_bay(), "Colonnade arcade bay")
    _assert_valid_mesh_attribs(build_facade_mullion_spandrel_grid(), "Facade mullion grid")
    _assert_valid_mesh_attribs(build_rooftop_hvac_chiller(), "Rooftop HVAC chiller")

    var heli_parts = build_rooftop_vtol_helipad_parts()
    _assert_valid_mesh_attribs(heli_parts[0], "Helipad base", require_colors=True)
    _assert_valid_mesh_attribs(heli_parts[1], "Helipad HDR beacons", require_colors=True)
    _assert(heli_parts[1].emissive_strength > Float32(1.0), "helipad beacons have HDR emissive_strength")
    _assert_valid_mesh_attribs(build_rooftop_vtol_helipad(), "Rooftop VTOL helipad merged", require_colors=True)

    var cant_parts = build_street_traffic_cantilever_parts()
    _assert_valid_mesh_attribs(cant_parts[0], "Traffic cantilever mast", require_colors=True)
    _assert_valid_mesh_attribs(cant_parts[1], "Traffic signal lenses", require_colors=True)
    _assert(cant_parts[1].emissive_strength > Float32(1.0), "traffic lenses have HDR emissive_strength")
    _assert_valid_mesh_attribs(build_street_traffic_cantilever(), "Traffic cantilever merged", require_colors=True)

    var boll_parts = build_street_bollard_and_grate_parts()
    _assert_valid_mesh_attribs(boll_parts[0], "Street bollard & grate base", require_colors=True)
    _assert_valid_mesh_attribs(boll_parts[1], "Bollard cyan LED collar", require_colors=True)
    _assert_valid_mesh_attribs(build_street_bollard_and_grate(), "Street bollard & grate merged", require_colors=True)

    var shelt_parts = build_transit_shelter_parts()
    _assert_valid_mesh_attribs(shelt_parts[0], "Transit shelter frame", require_colors=True)
    _assert_valid_mesh_attribs(shelt_parts[1], "Transit shelter ad/LED strips", require_colors=True)
    _assert_valid_mesh_attribs(build_transit_shelter(), "Transit shelter merged", require_colors=True)


def test_f26_regression_and_lod_dag_cook() raises:
    var q = _SymQuadric.from_plane(0.0, 1.0, 0.0, -2.0)
    _assert(abs(Float32(q.eval(Vec3(0.0, 2.0, 0.0)))) <= Float32(0.001), "point on plane evaluates to zero quadric error")

    var mock_mb = MeshBuilder(base_color=0xFF00AAFF, roughness=0.3, metallic=0.1)
    mock_mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 2.0, 2.0), 12, 12)
    var mock_m = mock_mb^.finish()
    var red = simplify_mesh_qem(mock_m, len(mock_m.indices) // 6)
    _assert(len(red[0].indices) > 0 and len(red[0].indices) < len(mock_m.indices), "mock meshlet QEM reduction valid")

    var sc = build_subd_supercar_body_and_glass()
    var pool = AaaGeometryPool()
    var entry = pool.append_mesh(sc[0], material_id=3, build_lods=True)
    _assert(entry.lod_count == 4, "SubD supercar cooks full 4-level LOD 0..3 QEM Meshlet DAG")
    _assert(entry.lods[3].triangle_count < entry.lods[0].triangle_count, "LOD 3 has fewer triangles than LOD 0")
    _assert(entry.lods[1].object_error < entry.lods[2].object_error and entry.lods[2].object_error < entry.lods[3].object_error, "strictly monotonic LOD errors")

    var sl = MeshLodSlice(0, 4, 200, 0.025)
    _assert(sl.meshlet_count == 4, "MeshLodSlice preserved")


def test_qem_skin_lod_weights() raises:
    # 11c. QEM Skin-LOD Top-4 Joint/Weight Blending & Renormalization (`sum == 1.0`)
    var grid_m = NativeMesh()
    comptime GX: Int = 9
    comptime GY: Int = 9
    for gy in range(GY):
        for gx in range(GX):
            grid_m.positions.append(Float32(gx) * Float32(0.25))
            grid_m.positions.append(Float32(gy) * Float32(0.25))
            grid_m.positions.append(Float32(0.0))
            grid_m.normals.append(Float32(0.0)); grid_m.normals.append(Float32(0.0)); grid_m.normals.append(Float32(1.0))
            grid_m.texcoords.append(Float32(gx) / Float32(GX - 1))
            grid_m.texcoords.append(Float32(gy) / Float32(GY - 1))
            var b_base = UInt16((gx + gy) % 6)
            grid_m.joints.append(b_base)
            grid_m.joints.append((b_base + UInt16(1)) % UInt16(8))
            grid_m.joints.append((b_base + UInt16(2)) % UInt16(8))
            grid_m.joints.append((b_base + UInt16(3)) % UInt16(8))
            grid_m.weights.append(Float32(0.45))
            grid_m.weights.append(Float32(0.25))
            grid_m.weights.append(Float32(0.20))
            grid_m.weights.append(Float32(0.10))
    for gy in range(GY - 1):
        for gx in range(GX - 1):
            var i0 = gy * GX + gx
            var i1 = i0 + 1
            var i2 = i0 + GX + 1
            var i3 = i0 + GX
            grid_m.indices.append(i0); grid_m.indices.append(i1); grid_m.indices.append(i2)
            grid_m.indices.append(i0); grid_m.indices.append(i2); grid_m.indices.append(i3)
    var qem_sk = simplify_mesh_qem(grid_m, 32)
    var sv_cnt = qem_sk[0].vertex_count()
    _assert(qem_sk[0].triangle_count() <= 40 and sv_cnt > 0, "QEM skinned grid reduces cleanly")
    _assert(len(qem_sk[0].joints) == sv_cnt * 4, "QEM skinned grid preserves 4 joints per vertex")
    _assert(len(qem_sk[0].weights) == sv_cnt * 4, "QEM skinned grid preserves 4 weights per vertex")
    for sv in range(sv_cnt):
        var sw0 = qem_sk[0].weights[sv * 4 + 0]
        var sw1 = qem_sk[0].weights[sv * 4 + 1]
        var sw2 = qem_sk[0].weights[sv * 4 + 2]
        var sw3 = qem_sk[0].weights[sv * 4 + 3]
        _assert(
            sw0 >= sw1 - Float32(1e-5) and sw1 >= sw2 - Float32(1e-5) and sw2 >= sw3 - Float32(1e-5) and sw3 >= Float32(0.0),
            "QEM skinned weights remain descending and non-negative",
        )
        _assert(abs((sw0 + sw1 + sw2 + sw3) - Float32(1.0)) < Float32(1e-5), "QEM skinned weights renormalize to 1.0")


def test_k_procedural_qem_and_toksvig() raises:
    print("  [k] Testing O(T log T) QEM LODs, vertex_words[11] AO/splat, Toksvig, SubD, loft/RMF, foliage, crowd, arch kits & QEM skin LOD...")
    test_qem_minheap_speed_and_validity()
    test_vertex_words_11_ao_and_splat_packing()
    test_toksvig_normal_variance_roughness()
    test_subd_creases_and_vehicles()
    test_loft_bevel_and_lathed_wheels()
    test_foliage_crowd_and_arch_kits()
    test_f26_regression_and_lod_dag_cook()
    test_qem_skin_lod_weights()
    print("    [OK] Procedural SubD/Loft/Foliage/Crowd/Arch, QEM LODs, QEM skin LOD & Toksvig passed!")


def test_l_cook_workspace_symlink_loop_bomb(tmp_root: String, mut pool: JobPool) raises:
    # [HIGH-1] Dual self-symlinks (s1 -> ., s2 -> .) must be skipped immediately without hang or duplicate cooking
    var ws = tmp_root + "/ws_symlink_bomb"
    var cache = tmp_root + "/cache_symlink_bomb"
    var out_flpk = tmp_root + "/symlink_bomb.flpk"
    _ = make_dirs(ws)
    _ = make_dirs(cache)
    var px = List[UInt32](length=4, fill=UInt32(0xFF336699))
    _assert(write_png(ws + "/real_tex.png", px, 2, 2), "write real_tex.png for symlink bomb test")
    var dot = String(".")
    var s1 = ws + "/s1"
    var s2 = ws + "/s2"
    var f_link = ws + "/file_link.png"
    var f_target = String("real_tex.png")
    _ = external_call["symlink", Int32](dot.as_c_string_span(), s1.as_c_string_span())
    _ = external_call["symlink", Int32](dot.as_c_string_span(), s2.as_c_string_span())
    _ = external_call["symlink", Int32](f_target.as_c_string_span(), f_link.as_c_string_span())

    var t0 = get_time()
    var rep = cook_workspace(ws, out_flpk, pool, cache_dir=cache, force_recook=True)
    var dt_ms = (get_time() - t0) * 1000.0
    _assert(rep.ok, "cook_workspace with symlink bomb must succeed")
    _assert(rep.total_assets == 1, "symlink bomb must cook real_tex.png exactly once, got " + String(rep.total_assets))
    _assert(dt_ms < 2000.0, "symlink bomb cook must finish quickly, took " + String(dt_ms) + " ms")
    print("    [OK] Symlink-loop traversal bomb guard passed (" + String(Int(dt_ms)) + " ms, 1 asset cooked)!")


def test_m_mcp_socket_path_stdlib_open_coexistence() raises:
    # Verify flint.devtools.mcp_v2.default_mcp_socket_path() (mkdirat) compiles and runs cleanly
    # alongside Mojo stdlib `with open(...)` (no LLVM external_call["mkdir"] signature collision).
    with open("/dev/null", "r") as f:
        var devnull_bytes = f.read_bytes()
        _assert(len(devnull_bytes) == 0, "stdlib open('/dev/null', 'r') should return 0 bytes")
    var sock_path = default_mcp_socket_path()
    var uid_str = String(Int(external_call["getuid", UInt32]()))
    _assert(sock_path.startswith("/"), "default_mcp_socket_path must be absolute: " + sock_path)
    _assert(sock_path.endswith("/mcp.sock"), "default_mcp_socket_path must end with /mcp.sock: " + sock_path)
    _assert(sock_path.find(uid_str) >= 0, "default_mcp_socket_path must include uid: " + sock_path)
    print("    [OK] MCP default_mcp_socket_path + stdlib open() coexistence passed (" + sock_path + ")!")


def test_n_os_read_bytes_dir_and_cooker_trailing_slash(tmp_root: String, mut pool: JobPool) raises:
    var real_dir = tmp_root + "/os_guard_real_dir"
    var sym_dir = tmp_root + "/os_guard_sym_dir"
    var cache_dir = tmp_root + "/os_guard_cache"
    var out_flpk = tmp_root + "/os_guard_sym.flpk"
    _ = make_dirs(real_dir)
    _ = make_dirs(cache_dir)

    var px = List[UInt32](length=4, fill=UInt32(0xFF4488CC))
    _assert(write_png(real_dir + "/sample.png", px, 2, 2), "write sample.png")
    var empty_file = real_dir + "/empty.bin"
    var empty_bytes = List[UInt8]()
    _assert(write_bytes(empty_file, empty_bytes), "write empty.bin")

    var sym_target = String("os_guard_real_dir")
    _ = external_call["symlink", Int32](sym_target.as_c_string_span(), sym_dir.as_c_string_span())

    # Directory & symlink-to-directory reads must return empty without SIGTRAP / INT64_MAX alloc
    _assert(file_size(real_dir) == -1, "file_size on directory must return -1")
    _assert(file_size(sym_dir) == -1, "file_size on symlink-to-directory must return -1")
    _assert(len(read_bytes(real_dir)) == 0, "read_bytes on directory must return empty list")
    _assert(len(read_bytes(sym_dir)) == 0, "read_bytes on symlink-to-directory must return empty list")
    _assert(file_size(empty_file) == 0, "file_size on empty regular file must return 0")
    _assert(len(read_bytes(empty_file)) == 0, "read_bytes on empty regular file must return empty list")
    var png_size = file_size(real_dir + "/sample.png")
    var png_bytes = read_bytes(real_dir + "/sample.png")
    _assert(png_size > 0 and len(png_bytes) == png_size, "regular file size/read_bytes preserved")

    # cooker_v2 _is_dir & cook_workspace must reject symlinks even with trailing slash
    _assert(_is_dir(real_dir) and _is_dir(real_dir + "/"), "_is_dir accepts real directory with/without trailing slash")
    _assert(not _is_dir(sym_dir) and not _is_dir(sym_dir + "/"), "_is_dir rejects symlink directory with/without trailing slash")
    var rep = cook_workspace(sym_dir + "/", out_flpk, pool, cache_dir=cache_dir, force_recook=True)
    _assert(rep.total_assets == 0, "cook_workspace on symlink directory with trailing slash must cook 0 assets")
    print("    [OK] os.read_bytes/file_size dir guard & cooker_v2 trailing-slash symlink guard passed!")


def test_o_mcp_v2_m0_scene_and_transform_tools(tmp_root: String, mut pool: JobPool) raises:
    var proj = tmp_root + "/mcp_m0_proj"
    _ = make_dirs(proj + "/scenes")
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var assets = AssetServer(enable_watcher=False)
    var mcp = McpServerV2(workspace_root=proj)

    var r_list = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}', studio, assets, pool)
    _assert(r_list.find('"scene.save"') >= 0, "tools/list missing scene.save")
    _assert(r_list.find('"scene.load"') >= 0, "tools/list missing scene.load")
    _assert(r_list.find('"scene.set_entity_enabled"') >= 0, "tools/list missing scene.set_entity_enabled")
    _assert(r_list.find('"entity.set_transform"') >= 0, "tools/list missing entity.set_transform")

    var r_sp = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ecs.spawn_entity","arguments":{"name":"M0Box","position":[0.0,1.0,0.0]}}}',
        studio, assets, pool,
    )
    _assert(r_sp.find('"entity_id":0') >= 0, "spawn_entity failed: " + r_sp)

    var r_xf = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"entity.set_transform","arguments":{"entity_id":0,"position":[4.0,2.0,-1.0],"rotation":[0.0,0.0,0.0,1.0],"scale":[2.0,2.0,2.0]}}}',
        studio, assets, pool,
    )
    _assert(r_xf.find('\\"ok\\":true') >= 0, "entity.set_transform failed: " + r_xf)
    _assert(abs(studio.entities[0].translation.x - Float32(4.0)) < Float32(1.0e-4), "entity.set_transform did not move x")

    var r_dis = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"scene.set_entity_enabled","arguments":{"entity_id":0,"enabled":false}}}',
        studio, assets, pool,
    )
    _assert(r_dis.find('\\"ok\\":true') >= 0 and not studio.entities[0].visible, "scene.set_entity_enabled false failed")

    var r_qv = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{"visible_only":true}}}',
        studio, assets, pool,
    )
    _assert(r_qv.find('\\"count\\":0') >= 0, "visible_only query should exclude disabled entity: " + r_qv)

    _ = studio.undo()
    _assert(studio.entities[0].visible, "undo did not re-enable entity")
    _ = studio.redo()
    _assert(not studio.entities[0].visible, "redo did not re-disable entity")

    var r_sv = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"scene.save","arguments":{"path":"scenes/m0_unit.flscn"}}}',
        studio, assets, pool,
    )
    _assert(r_sv.find('\\"ok\\":true') >= 0 and file_size(proj + "/scenes/m0_unit.flscn") > 0, "scene.save failed")

    # Negative / traversal boundary checks
    var r_bad1 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"scene.save","arguments":{"path":"../escape.flscn"}}}',
        studio, assets, pool,
    )
    _assert(r_bad1.find('"isError":true') >= 0, "scene.save must reject traversal")
    var r_bad2 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"entity.set_transform","arguments":{"entity_id":0,"position":[1.0,2.0]}}}',
        studio, assets, pool,
    )
    _assert(r_bad2.find('"isError":true') >= 0, "entity.set_transform must reject 2-element position")
    var r_bad3 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"scene.set_entity_enabled","arguments":{"entity_id":0}}}',
        studio, assets, pool,
    )
    _assert(r_bad3.find('"isError":true') >= 0, "scene.set_entity_enabled must require enabled bool")

    # Reject fractional / non-finite entity_id and empty-stem .flscn paths
    var r_frac1 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":91,"method":"tools/call","params":{"name":"entity.set_transform","arguments":{"entity_id":0.5,"position":[9.0,9.0,9.0]}}}',
        studio, assets, pool,
    )
    _assert(r_frac1.find('"isError":true') >= 0, "entity.set_transform must reject fractional entity_id 0.5")
    var r_frac2 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":92,"method":"tools/call","params":{"name":"scene.set_entity_enabled","arguments":{"entity_id":0.5,"enabled":false}}}',
        studio, assets, pool,
    )
    _assert(r_frac2.find('"isError":true') >= 0, "scene.set_entity_enabled must reject fractional entity_id 0.5")
    var r_frac3 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":93,"method":"tools/call","params":{"name":"entity.set_transform","arguments":{"entity_id":1.5,"position":[9.0,9.0,9.0]}}}',
        studio, assets, pool,
    )
    _assert(r_frac3.find('"isError":true') >= 0, "entity.set_transform must reject fractional entity_id 1.5")
    var r_stem1 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":94,"method":"tools/call","params":{"name":"scene.save","arguments":{"path":".flscn"}}}',
        studio, assets, pool,
    )
    _assert(r_stem1.find('"isError":true') >= 0, "scene.save must reject empty stem .flscn")
    var r_stem2 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":95,"method":"tools/call","params":{"name":"scene.save","arguments":{"path":"scenes/.flscn"}}}',
        studio, assets, pool,
    )
    _assert(r_stem2.find('"isError":true') >= 0, "scene.save must reject empty stem scenes/.flscn")
    var r_stem3 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":96,"method":"tools/call","params":{"name":"scene.load","arguments":{"path":"scenes/.flscn"}}}',
        studio, assets, pool,
    )
    _assert(r_stem3.find('"isError":true') >= 0, "scene.load must reject empty stem scenes/.flscn")

    # Relative ../ workspace_root must work for sandboxed tools while still rejecting user_path ../escape.flscn
    var rel_caller = tmp_root + "/mcp_m0_caller"
    var rel_proj = tmp_root + "/mcp_m0_rel_proj"
    _ = make_dirs(rel_caller)
    _ = make_dirs(rel_proj + "/scenes")
    var cwd_buf = List[UInt8](length=4096, fill=0)
    var got_cwd = external_call["getcwd", Int](cwd_buf.unsafe_ptr(), Int64(4096))
    var orig_cwd_bytes = List[UInt8]()
    if got_cwd != 0:
        for ci in range(4096):
            if cwd_buf[ci] == 0:
                break
            orig_cwd_bytes.append(cwd_buf[ci])
    var orig_cwd = String(unsafe_from_utf8=orig_cwd_bytes)
    _ = len(cwd_buf)
    var ch_rc = external_call["chdir", Int32](rel_caller.as_c_string_span())
    _assert(ch_rc == 0, "chdir to rel_caller failed")
    var rel_mcp = McpServerV2(workspace_root="../mcp_m0_rel_proj")
    var r_rel_sv = rel_mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":97,"method":"tools/call","params":{"name":"scene.save","arguments":{"path":"scenes/rel.flscn"}}}',
        studio, assets, pool,
    )
    var r_rel_ld = rel_mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":98,"method":"tools/call","params":{"name":"scene.load","arguments":{"path":"scenes/rel.flscn"}}}',
        studio, assets, pool,
    )
    var r_rel_esc = rel_mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":99,"method":"tools/call","params":{"name":"scene.save","arguments":{"path":"../escape.flscn"}}}',
        studio, assets, pool,
    )
    if orig_cwd.byte_length() > 0:
        _ = external_call["chdir", Int32](orig_cwd.as_c_string_span())
    _assert(r_rel_sv.find('\\"ok\\":true') >= 0 and file_size(rel_proj + "/scenes/rel.flscn") > 0, "relative ../ workspace_root scene.save failed: " + r_rel_sv)
    _assert(r_rel_ld.find('\\"ok\\":true') >= 0, "relative ../ workspace_root scene.load failed: " + r_rel_ld)
    _assert(r_rel_esc.find('"isError":true') >= 0, "relative ../ workspace_root must still reject user_path ../escape.flscn")

    _ = studio.despawn_entity(0, record_undo=False)
    var r_ld = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"scene.load","arguments":{"path":"scenes/m0_unit.flscn"}}}',
        studio, assets, pool,
    )
    _assert(r_ld.find('\\"ok\\":true') >= 0 and studio.live_entity_count() == 1 and not studio.entities[0].visible, "scene.load failed to restore disabled entity")

    assets.close()
    studio.shutdown()
    print("    [OK] McpServerV2 M0 tools (scene.save/load, scene.set_entity_enabled, entity.set_transform) passed!")


def _mcp_write_text_p(path: String, text: String) -> Bool:
    var b = List[UInt8]()
    for c in text.as_bytes():
        b.append(c)
    return write_bytes(path, b)


def test_p_mcp_v2_m1_assets_prefab_material_and_upload(tmp_root: String, mut pool: JobPool) raises:
    var proj = tmp_root + "/mcp_m1_proj"
    var ext = tmp_root + "/mcp_m1_ext"
    _ = make_dirs(proj)
    _ = make_dirs(ext)

    # Prepare valid external source assets
    var mdl = NativeModel()
    var m = NativeMesh()
    m.positions.append(Float32(-0.5)); m.positions.append(Float32(-0.5)); m.positions.append(Float32(0.0))
    m.positions.append(Float32(0.5)); m.positions.append(Float32(-0.5)); m.positions.append(Float32(0.0))
    m.positions.append(Float32(0.0)); m.positions.append(Float32(0.5)); m.positions.append(Float32(0.0))
    m.normals.append(Float32(0.0)); m.normals.append(Float32(0.0)); m.normals.append(Float32(1.0))
    m.normals.append(Float32(0.0)); m.normals.append(Float32(0.0)); m.normals.append(Float32(1.0))
    m.normals.append(Float32(0.0)); m.normals.append(Float32(0.0)); m.normals.append(Float32(1.0))
    m.texcoords.append(Float32(0.0)); m.texcoords.append(Float32(0.0))
    m.texcoords.append(Float32(1.0)); m.texcoords.append(Float32(0.0))
    m.texcoords.append(Float32(0.5)); m.texcoords.append(Float32(1.0))
    m.indices.append(0); m.indices.append(1); m.indices.append(2)
    mdl.meshes.append(m^)
    _assert(write_native_glb(ext + "/ext_crate.glb", mdl), "write ext_crate.glb")
    _assert(
        _mcp_write_text_p(
            ext + "/crate.flmat",
            '{"format":"flmat","version":1,"name":"CrateMat","base_color":[0.8,0.4,0.2],"roughness":0.35,"metallic":0.2}',
        ),
        "write crate.flmat",
    )
    _assert(
        _mcp_write_text_p(
            ext + "/crate.flprefab",
            '{"format":"flprefab","version":1,"name":"CratePF","entities":['
            '{"name":"CrateRoot","position":[0.0,1.0,0.0],"mass":3.5,"mesh_ref":"assets/models/ext_crate.glb","material_ref":"assets/materials/crate.flmat"}]}',
        ),
        "write crate.flprefab",
    )
    _assert(_mcp_write_text_p(ext + "/corrupt.glb", "BAD_GLB_DATA"), "write corrupt.glb")
    _assert(_mcp_write_text_p(ext + "/bad.flmat", "{not_json"), "write bad.flmat")
    _assert(_mcp_write_text_p(ext + "/bad.flprefab", "{not_json"), "write bad.flprefab")

    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
        project_root=proj,
    )
    var mcp = McpServerV2(workspace_root=proj)

    def _call_mcp(
        mut s_mcp: McpServerV2,
        mut st: FlintStudioState,
        mut pl: JobPool,
        req: String,
    ) raises -> String:
        ref assets_srv = st.app.resources.get_ptr_mut[AssetServer]().value()[unsafe_offset=0]
        return s_mcp.handle_jsonrpc(req, st, assets_srv, pl)

    # 1. tools/list has all 22 tools including the 4 new M1 tools
    var r_tl = _call_mcp(mcp, studio, pool, '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}')
    _assert(r_tl.find('"asset.import"') >= 0, "tools/list missing asset.import")
    _assert(r_tl.find('"asset.list"') >= 0, "tools/list missing asset.list")
    _assert(r_tl.find('"material.assign"') >= 0, "tools/list missing material.assign")
    _assert(r_tl.find('"prefab.spawn"') >= 0, "tools/list missing prefab.spawn")

    # 2. Empty project asset.list returns count=0
    var r_empty = _call_mcp(
        mcp, studio, pool, '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"asset.list","arguments":{}}}'
    )
    _assert(r_empty.find('\\"count\\":0') >= 0, "empty project asset.list must return count=0: " + r_empty)

    # 2b. A source outside the workspace is refused until its directory is on the import allow-list.
    var r_not_allowed = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"asset.import","arguments":{"source_path":"'
        + ext
        + '/ext_crate.glb","dest_path":"assets/models/ext_crate.glb"}}}',
    )
    _assert(
        r_not_allowed.find('"isError":true') >= 0 and r_not_allowed.find("allow-list") >= 0
        and file_size(proj + "/assets/models/ext_crate.glb") < 0,
        "asset.import from a directory outside the workspace and allow-list must be refused: " + r_not_allowed,
    )
    mcp.allow_import_dir(ext)

    # 3. Boundary rejections: system file, traversal, corrupt payloads write ZERO bytes to disk
    var r_sys = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"asset.import","arguments":{"source_path":"/etc/passwd","dest_path":"assets/passwd"}}}',
    )
    _assert(r_sys.find('"isError":true') >= 0 and file_size(proj + "/assets/passwd") < 0, "must reject /etc/passwd with zero disk write")
    var r_trav = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"asset.import","arguments":{"source_path":"'
        + ext
        + '/ext_crate.glb","dest_path":"../escape.glb"}}}',
    )
    _assert(r_trav.find('"isError":true') >= 0, "must reject traversal dest_path")
    var r_cglb = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"asset.import","arguments":{"source_path":"'
        + ext
        + '/corrupt.glb","dest_path":"assets/models/corrupt.glb"}}}',
    )
    _assert(r_cglb.find('"isError":true') >= 0 and file_size(proj + "/assets/models/corrupt.glb") < 0, "must reject corrupt GLB before disk write")

    # 4. Valid asset.import + asset.list
    var r_i1 = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"asset.import","arguments":{"source_path":"'
        + ext
        + '/ext_crate.glb","dest_path":"assets/models/ext_crate.glb"}}}',
    )
    var r_i2 = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"asset.import","arguments":{"source_path":"'
        + ext
        + '/crate.flmat","dest_path":"assets/materials/crate.flmat"}}}',
    )
    var r_i3 = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"asset.import","arguments":{"source_path":"'
        + ext
        + '/crate.flprefab","dest_path":"assets/prefabs/crate.flprefab"}}}',
    )
    _assert(r_i1.find('\\"ok\\":true') >= 0 and r_i1.find('assets/models/ext_crate.glb') >= 0, "asset.import glb failed: " + r_i1)
    _assert(r_i2.find('\\"ok\\":true') >= 0 and r_i3.find('\\"ok\\":true') >= 0, "asset.import flmat/flprefab failed")

    var r_ls = _call_mcp(
        mcp, studio, pool, '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"asset.list","arguments":{}}}'
    )
    _assert(r_ls.find('\\"count\\":3') >= 0 and r_ls.find('assets/models/ext_crate.glb') >= 0, "asset.list failed: " + r_ls)

    # 5. procedural.generate_asset uploads real mesh_id >= 0 & binds relative MeshRendererComponent.mesh_ref
    var r_gen = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"procedural.generate_asset","arguments":{"preset":"crate","seed":7,"spawn_in_scene":true,"output_glb_path":"assets/models/proc_crate.glb"}}}',
    )
    _assert(r_gen.find('\\"ok\\":true') >= 0, "procedural.generate_asset failed: " + r_gen)
    _assert(r_gen.find('\\"mesh_id\\":-1') < 0, "procedural.generate_asset must produce mesh_id >= 0: " + r_gen)
    _assert(r_gen.find('assets/models/proc_crate.glb') >= 0, "procedural.generate_asset output_glb_path must be relative: " + r_gen)
    _assert(file_size(proj + "/assets/models/proc_crate.glb") > 0, "procedural.generate_asset did not write GLB")
    var gen_eid = -1
    for i in range(len(studio.entities)):
        if studio.entities[i].alive:
            gen_eid = i
            break
    _assert(gen_eid >= 0, "procedural.generate_asset did not spawn entity")
    var gen_e = studio.resolve_entity(gen_eid)
    _assert(
        studio.app.world.get_ptr[MeshRendererComponent](gen_e).value()[].mesh_ref.rel_path == "assets/models/proc_crate.glb",
        "procedural.generate_asset did not bind MeshRendererComponent.mesh_ref",
    )
    _assert(studio.app.world.get_ptr[MeshRendererComponent](gen_e).value()[].mesh_id >= 0, "generated entity mesh_id < 0")

    # 6. material.assign + invalid assign non-mutating check
    var r_ma = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"material.assign","arguments":{"entity_id":'
        + String(gen_eid)
        + ',"material_path":"assets/materials/crate.flmat"}}}',
    )
    _assert(r_ma.find('\\"ok\\":true') >= 0 and r_ma.find('assets/materials/crate.flmat') >= 0, "material.assign failed: " + r_ma)
    _assert(studio.app.world.has[MaterialOverrideComponent](gen_e), "material.assign missing MaterialOverrideComponent")
    var h_before_bad = studio.state_hash()
    var bad_flmat_b = read_bytes(ext + "/bad.flmat")
    _ = write_bytes(proj + "/assets/materials/bad.flmat", bad_flmat_b)
    var r_ma_bad = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"material.assign","arguments":{"entity_id":'
        + String(gen_eid)
        + ',"material_path":"assets/materials/bad.flmat"}}}',
    )
    _assert(r_ma_bad.find('"isError":true') >= 0 and studio.state_hash() == h_before_bad, "invalid material.assign must fail without mutating state")

    # 7. prefab.spawn materializes PrefabInstanceComponent + RigidBodyComponent from "mass"
    var r_pf = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"prefab.spawn","arguments":{"prefab_path":"assets/prefabs/crate.flprefab","position":[3.0,4.0,-2.0]}}}',
    )
    _assert(r_pf.find('\\"ok\\":true') >= 0, "prefab.spawn failed: " + r_pf)
    var found_pf = False
    for i in range(len(studio.entities)):
        if studio.entities[i].alive and studio.entities[i].name == "CrateRoot":
            var pe = studio.resolve_entity(i)
            _assert(studio.app.world.has[PrefabInstanceComponent](pe), "prefab.spawn missing PrefabInstanceComponent")
            _assert(studio.app.world.has[RigidBodyComponent](pe), "prefab.spawn missing RigidBodyComponent from mass")
            _assert(abs(studio.entities[i].translation.x - Float32(3.0)) < Float32(1.0e-4), "prefab.spawn position override failed")
            found_pf = True
    _assert(found_pf, "prefab.spawn CrateRoot not found")

    # 8. ecs.query_entities emits material_ref/mesh_ref/mesh_id & scene.save -> restart -> scene.load preserves state_hash()
    var r_q = _call_mcp(
        mcp, studio, pool, '{"jsonrpc":"2.0","id":14,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{}}}'
    )
    _assert(r_q.find('assets/models/proc_crate.glb') >= 0 and r_q.find('assets/materials/crate.flmat') >= 0, "ecs.query_entities missing refs: " + r_q)

    var r_sv = _call_mcp(
        mcp,
        studio,
        pool,
        '{"jsonrpc":"2.0","id":15,"method":"tools/call","params":{"name":"scene.save","arguments":{"path":"scenes/m1_unit.flscn"}}}',
    )
    _assert(r_sv.find('\\"ok\\":true') >= 0 and file_size(proj + "/scenes/m1_unit.flscn") > 0, "scene.save failed: " + r_sv)
    studio.shutdown()

    var studio2 = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
        project_root=proj,
    )
    var r_ld = _call_mcp(
        mcp,
        studio2,
        pool,
        '{"jsonrpc":"2.0","id":16,"method":"tools/call","params":{"name":"scene.load","arguments":{"path":"scenes/m1_unit.flscn"}}}',
    )
    _assert(r_ld.find('\\"ok\\":true') >= 0, "scene.load failed: " + r_ld)
    _assert(studio2.live_entity_count() == 2, "restarted scene.load entity count mismatch")
    var h2 = studio2.state_hash()
    var found_proc2 = False
    var found_pf2 = False
    for i in range(len(studio2.entities)):
        if not studio2.entities[i].alive:
            continue
        var e2 = studio2.resolve_entity(i)
        if studio2.entities[i].name == "Proc_crate":
            ref mr2 = studio2.app.world.get_ptr[MeshRendererComponent](e2).value()[unsafe_offset=0]
            _assert(mr2.mesh_ref.rel_path == "assets/models/proc_crate.glb" and mr2.mesh_id >= 0, "reloaded Proc_crate mesh missing")
            _assert(mr2.material_ref.rel_path == "assets/materials/crate.flmat" and mr2.material_id >= 0, "reloaded Proc_crate material missing")
            found_proc2 = True
        elif studio2.entities[i].name == "CrateRoot":
            _assert(studio2.app.world.has[PrefabInstanceComponent](e2), "reloaded CrateRoot missing PrefabInstanceComponent")
            _assert(studio2.app.world.has[RigidBodyComponent](e2), "reloaded CrateRoot missing RigidBodyComponent")
            _assert(abs(studio2.entities[i].translation.x - Float32(3.0)) < Float32(1.0e-4), "reloaded CrateRoot position mismatch")
            found_pf2 = True
    _assert(found_proc2 and found_pf2, "reloaded scene missing expected entities")
    _ = _call_mcp(
        mcp,
        studio2,
        pool,
        '{"jsonrpc":"2.0","id":17,"method":"tools/call","params":{"name":"scene.save","arguments":{"path":"scenes/m1_unit2.flscn"}}}',
    )
    studio2.shutdown()

    var studio3 = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
        project_root=proj,
    )
    var r_ld3 = _call_mcp(
        mcp,
        studio3,
        pool,
        '{"jsonrpc":"2.0","id":18,"method":"tools/call","params":{"name":"scene.load","arguments":{"path":"scenes/m1_unit2.flscn"}}}',
    )
    _assert(r_ld3.find('\\"ok\\":true') >= 0, "scene.load 2 failed: " + r_ld3)
    _assert(studio3.state_hash() == h2, "canonical asset-resolved scene.save/load state_hash mismatch across restart")
    studio3.shutdown()
    # 9. Gate 1 Remediation: Tier-0 mesh_id >= 6 (no collision with default 0..5), canonical material cache,
    # UInt64 SkeletonComponent.model_id round-trip, state_hash() invariance s1 == s2, and Tier-0 3D render parity
    var proj_g1 = tmp_root + "/mcp_g1_proj"
    _ = make_dirs(proj_g1 + "/assets/models")
    _ = make_dirs(proj_g1 + "/assets/materials")
    _ = make_dirs(proj_g1 + "/assets/prefabs")
    _ = make_dirs(proj_g1 + "/scenes")
    _ = write_native_glb(proj_g1 + "/assets/models/ext_crate.glb", mdl)
    _assert(
        _mcp_write_text_p(
            proj_g1 + "/assets/materials/canon.flmat",
            '{"format":"flmat","version":1,"name":"Canon","base_color":[0.25,0.65,0.85],"roughness":0.4,"metallic":0.1}',
        ),
        "write canon.flmat",
    )
    _assert(
        _mcp_write_text_p(
            proj_g1 + "/assets/materials/alt.flmat",
            '{"format":"flmat","version":1,"name":"Alt","base_color":[0.85,0.25,0.15],"roughness":0.7,"metallic":0.0}',
        ),
        "write alt.flmat",
    )
    # Canonical prefab omitting uuid in material_ref
    _assert(
        _mcp_write_text_p(
            proj_g1 + "/assets/prefabs/nil_uuid.flprefab",
            '{"format":"flprefab","version":1,"name":"NilUuidPF","nodes":['
            '{"name":"CanonBox","parent_slot":-1,"components":{'
            '"LocalTransform":{"translation":[2.0,0.5,0.0],"rotation":[0.0,0.0,0.0,1.0],"scale":[1.0,1.0,1.0]},'
            '"MeshRendererComponent":{"mesh_id":0,"material_id":-1,"cast_shadows":true,"visible":true,'
            '"mesh_ref":{"rel_path":"assets/models/ext_crate.glb","kind":1},'
            '"material_ref":{"rel_path":"assets/materials/canon.flmat","kind":0}},'
            '"MaterialOverrideComponent":{"material_id":-1,"enabled":true,"material_ref":{"rel_path":"assets/materials/canon.flmat","kind":0}}'
            '}}]}',
        ),
        "write nil_uuid.flprefab",
    )

    var s1 = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
        project_root=proj_g1,
    )
    var mcp_g1 = McpServerV2(workspace_root=proj_g1)
    populate_default_studio_scene(s1)
    _assert(
        s1.app.resources.get_ptr[ScalableRenderer]().value()[].cpu_meshes.__len__() >= 6,
        "populate_default_studio_scene must reserve 6 default CPU meshes in Tier 0",
    )
    var r_g1 = _call_mcp(
        mcp_g1,
        s1,
        pool,
        '{"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"procedural.generate_asset","arguments":{"preset":"material_orb","seed":3,"spawn_in_scene":true,"position":[1.5,1.0,-0.5],"output_glb_path":"assets/models/g1_orb.glb"}}}',
    )
    _assert(r_g1.find('\\"ok\\":true') >= 0, "g1 procedural.generate_asset failed: " + r_g1)
    _assert(r_g1.find('\\"mesh_id\\":0') < 0, "Tier-0 procedural.generate_asset collided with mesh_id=0: " + r_g1)
    var orb_eid = -1
    for i in range(len(s1.entities)):
        if s1.entities[i].alive and s1.entities[i].name == "Proc_material_orb":
            orb_eid = i
            break
    _assert(orb_eid >= 0, "missing Proc_material_orb")
    ref orb_mr = s1.app.world.get_ptr[MeshRendererComponent](s1.resolve_entity(orb_eid)).value()[unsafe_offset=0]
    _assert(orb_mr.mesh_id >= 6, "Proc_material_orb mesh_id must be >= 6 in Tier 0, got " + String(orb_mr.mesh_id))

    # Material reassignment history + nil-UUID prefab must hit same material cache key
    var r_ma1 = _call_mcp(
        mcp_g1,
        s1,
        pool,
        '{"jsonrpc":"2.0","id":21,"method":"tools/call","params":{"name":"material.assign","arguments":{"entity_id":'
        + String(orb_eid)
        + ',"material_path":"assets/materials/canon.flmat"}}}',
    )
    _assert(r_ma1.find('\\"ok\\":true') >= 0, "g1 material.assign 1 failed: " + r_ma1)
    var r_ma2 = _call_mcp(
        mcp_g1,
        s1,
        pool,
        '{"jsonrpc":"2.0","id":22,"method":"tools/call","params":{"name":"material.assign","arguments":{"entity_id":'
        + String(orb_eid)
        + ',"material_path":"assets/materials/alt.flmat"}}}',
    )
    _assert(r_ma2.find('\\"ok\\":true') >= 0, "g1 material.assign 2 failed: " + r_ma2)
    var r_ma3 = _call_mcp(
        mcp_g1,
        s1,
        pool,
        '{"jsonrpc":"2.0","id":23,"method":"tools/call","params":{"name":"material.assign","arguments":{"entity_id":'
        + String(orb_eid)
        + ',"material_path":"assets/materials/canon.flmat"}}}',
    )
    _assert(r_ma3.find('\\"ok\\":true') >= 0, "g1 material.assign 3 failed: " + r_ma3)
    var r_pf_nil = _call_mcp(
        mcp_g1,
        s1,
        pool,
        '{"jsonrpc":"2.0","id":24,"method":"tools/call","params":{"name":"prefab.spawn","arguments":{"prefab_path":"assets/prefabs/nil_uuid.flprefab"}}}',
    )
    _assert(r_pf_nil.find('\\"ok\\":true') >= 0, "g1 prefab.spawn nil_uuid failed: " + r_pf_nil)
    var cbox_eid = -1
    for i in range(len(s1.entities)):
        if s1.entities[i].alive and s1.entities[i].name == "CanonBox":
            cbox_eid = i
            break
    _assert(cbox_eid >= 0, "missing CanonBox")
    var orb_mat_id = s1.app.world.get_ptr[MaterialOverrideComponent](s1.resolve_entity(orb_eid)).value()[].material_id
    var cbox_mat_id = s1.app.world.get_ptr[MaterialOverrideComponent](s1.resolve_entity(cbox_eid)).value()[].material_id
    _assert(
        orb_mat_id >= 0 and orb_mat_id == cbox_mat_id,
        "nil-UUID and populated-UUID AssetRef for same .flmat must share canonical material_id",
    )

    # Rigged SkeletonComponent with UInt64 > Int64.MAX (high bit set) + BoneSocketComponent
    var rig_eid = s1.spawn_entity("RiggedHero", position=Vec3(-1.5, 0.5, 0.0), record_undo=False)
    var rig_e = s1.resolve_entity(rig_eid)
    var high_u64 = UInt64(15352672342455569117)
    var sk_inst = SkeletonInstance(2)
    sk_inst.parents[0] = -1
    sk_inst.parents[1] = 0
    sk_inst.pose.t[1] = Vec3(0.0, 0.8, 0.0)
    _ = s1.app.world.insert[SkeletonComponent](rig_e, SkeletonComponent(high_u64, sk_inst^, False, AssetRef()))
    var mr_rig = s1.app.world.get_ptr_mut[MeshRendererComponent](rig_e)
    if mr_rig:
        mr_rig.value()[].mesh_ref = AssetRef.from_path("assets/models/ext_crate.glb", ASSET_KIND_MESH)
        mr_rig.value()[].material_ref = AssetRef.from_path("assets/materials/canon.flmat", ASSET_KIND_RAW)
    _ = s1.app.world.insert[MaterialOverrideComponent](
        rig_e,
        MaterialOverrideComponent(
            -1,
            Vec3(0.25, 0.65, 0.85),
            Float32(0.4),
            Float32(0.1),
            Vec3(),
            enabled=True,
            material_ref=AssetRef.from_path("assets/materials/canon.flmat", ASSET_KIND_RAW),
        ),
    )
    var sock_eid = s1.spawn_entity("HeroSocket", position=Vec3(0.0, 0.0, 0.0), record_undo=False)
    var sock_e = s1.resolve_entity(sock_eid)
    _ = s1.app.world.insert[BoneSocketComponent](
        sock_e, BoneSocketComponent(rig_e, BoneSocketAttachment("Hand", 1), -1)
    )
    s1.look_at(Vec3(-5.6, 3.75, -7.4), Vec3(0.15, 0.95, 1.1))
    s1._sync_ecs_and_physics()
    s1.render_editor_frame(160, 120)
    var rw1_ptr = s1.app.resources.get_ptr_mut[RenderWorld]().value()
    var ren1_ptr = s1.app.resources.get_ptr_mut[ScalableRenderer]().value()
    var empty_ui1 = UiDrawList()
    _ = ren1_ptr[unsafe_offset=0].render_world(
        rw1_ptr[unsafe_offset=0], empty_ui1, capture_readback=True, force_readback=True
    )
    var px1 = ren1_ptr[unsafe_offset=0].readback_pixels()
    var h_s1 = s1.state_hash()

    var r_sv_g1 = _call_mcp(
        mcp_g1,
        s1,
        pool,
        '{"jsonrpc":"2.0","id":25,"method":"tools/call","params":{"name":"scene.save","arguments":{"path":"scenes/g1_roundtrip.flscn"}}}',
    )
    _assert(r_sv_g1.find('\\"ok\\":true') >= 0, "g1 scene.save failed: " + r_sv_g1)
    s1.shutdown()

    var s2 = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
        project_root=proj_g1,
    )
    var r_ld_g1 = _call_mcp(
        mcp_g1,
        s2,
        pool,
        '{"jsonrpc":"2.0","id":26,"method":"tools/call","params":{"name":"scene.load","arguments":{"path":"scenes/g1_roundtrip.flscn"}}}',
    )
    _assert(r_ld_g1.find('\\"ok\\":true') >= 0, "g1 scene.load failed: " + r_ld_g1)
    _assert(s2.state_hash() == h_s1, "s2.state_hash() must equal h1 directly across scene.save -> restart -> scene.load")
    var found_rig2 = False
    for i in range(len(s2.entities)):
        if s2.entities[i].alive and s2.entities[i].name == "RiggedHero":
            var re2 = s2.resolve_entity(i)
            _assert(
                s2.app.world.get_ptr[SkeletonComponent](re2).value()[].model_id == high_u64,
                "SkeletonComponent UInt64 model_id lost precision on .flscn JSON roundtrip",
            )
            found_rig2 = True
        elif s2.entities[i].alive and s2.entities[i].name == "Proc_material_orb":
            var pe2 = s2.resolve_entity(i)
            _assert(
                s2.app.world.get_ptr[MeshRendererComponent](pe2).value()[].mesh_id >= 6,
                "reloaded Proc_material_orb mesh_id collided with default showcase slots",
            )
    _assert(found_rig2, "RiggedHero missing after scene.load")

    s2.look_at(Vec3(-5.6, 3.75, -7.4), Vec3(0.15, 0.95, 1.1))
    s2.select_entity(1)
    s2.render_editor_frame(160, 120)
    var rw2_ptr = s2.app.resources.get_ptr_mut[RenderWorld]().value()
    var ren2_ptr = s2.app.resources.get_ptr_mut[ScalableRenderer]().value()
    var empty_ui2 = UiDrawList()
    _ = ren2_ptr[unsafe_offset=0].render_world(
        rw2_ptr[unsafe_offset=0], empty_ui2, capture_readback=True, force_readback=True
    )
    var px2 = ren2_ptr[unsafe_offset=0].readback_pixels()
    _assert(len(px1) == len(px2) and len(px1) > 0, "Tier-0 framebuffer size mismatch")
    var nonzero_px = 0
    var diff_px = 0
    for k in range(len(px2)):
        if px2[k] != UInt32(0):
            nonzero_px += 1
        if px1[k] != px2[k]:
            diff_px += 1
    _assert(nonzero_px > 100, "reloaded Tier-0 scene rendered empty framebuffer")
    _assert(diff_px == 0, "reloaded Tier-0 3D scene had pixel diffs: " + String(diff_px))
    s2.shutdown()
    print("    [OK] McpServerV2 M1 tools (asset.import/list, material.assign, prefab.spawn, procedural upload, scene roundtrip, Gate 1 hash/mesh_id/u64 parity) passed!")


def test_q_glb_unoverridden_material_reload_parity(tmp_root: String) raises:
    var proj = tmp_root + "/glb_default_mat_proj"
    _ = make_dirs(proj + "/assets/models")
    _ = make_dirs(proj + "/assets/materials")
    _ = make_dirs(proj + "/scenes")

    var mdl = NativeModel()
    var m = NativeMesh()
    m.base_color = 0xFFFFFFFF
    m.positions.append(Float32(-0.7)); m.positions.append(Float32(-0.7)); m.positions.append(Float32(0.0))
    m.positions.append(Float32(0.7)); m.positions.append(Float32(-0.7)); m.positions.append(Float32(0.0))
    m.positions.append(Float32(0.0)); m.positions.append(Float32(0.7)); m.positions.append(Float32(0.0))
    for _ in range(3):
        m.normals.append(Float32(0.0)); m.normals.append(Float32(0.0)); m.normals.append(Float32(1.0))
    m.texcoords.append(Float32(0.0)); m.texcoords.append(Float32(0.0))
    m.texcoords.append(Float32(1.0)); m.texcoords.append(Float32(0.0))
    m.texcoords.append(Float32(0.5)); m.texcoords.append(Float32(1.0))
    m.indices.append(0); m.indices.append(1); m.indices.append(2)
    mdl.meshes.append(m^)
    _assert(write_native_glb(proj + "/assets/models/prop.glb", mdl), "write prop.glb")
    _assert(write_native_glb(proj + "/assets/models/hero.glb", mdl), "write hero.glb")
    _assert(
        _mcp_write_text_p(
            proj + "/assets/materials/hero.flmat",
            '{"format":"flmat","version":1,"name":"Hero","base_color":[0.05,0.25,0.95],"roughness":0.3,"metallic":0.1}',
        ),
        "write hero.flmat",
    )

    var s1 = FlintStudioState(
        headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0, project_root=proj
    )
    var hero_id = s1.activate_asset_entry("assets/models/hero.glb", drop_pos=Vec3(-1.0, 0.0, 0.0), from_drop=True)
    _assert(hero_id >= 0, "spawn hero.glb failed")
    _assert(s1.activate_asset_entry("assets/materials/hero.flmat", target_entity_id=hero_id, from_drop=True) == hero_id, "assign hero.flmat failed")
    var prop_id = s1.activate_asset_entry("assets/models/prop.glb", drop_pos=Vec3(1.0, 0.0, 0.0), from_drop=True)
    _assert(prop_id >= 0, "spawn prop.glb failed")

    s1.look_at(Vec3(0.0, 1.0, 4.5), Vec3(0.0, 0.0, 0.0))
    s1.render_editor_frame(160, 120)
    var rw1 = s1.app.resources.get_ptr_mut[RenderWorld]().value()
    var ren1 = s1.app.resources.get_ptr_mut[ScalableRenderer]().value()
    var ui1 = UiDrawList()
    _ = ren1[unsafe_offset=0].render_world(rw1[unsafe_offset=0], ui1, capture_readback=True, force_readback=True)
    var px1 = ren1[unsafe_offset=0].readback_pixels()
    var h1 = s1.state_hash()
    _assert(s1.save_scene_file("scenes/mixed_assets.flscn"), "save mixed_assets.flscn")
    _assert(s1.save_scene_file("scenes/mixed_assets.flscnb"), "save mixed_assets.flscnb")
    s1.shutdown()

    for fmt in range(2):
        var s2 = FlintStudioState(
            headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0, project_root=proj
        )
        var spath = "scenes/mixed_assets.flscn" if fmt == 0 else "scenes/mixed_assets.flscnb"
        _assert(s2.load_scene_file(spath), "load " + spath)
        _assert(s2.state_hash() == h1, "state_hash mismatch on " + spath)
        s2.look_at(Vec3(0.0, 1.0, 4.5), Vec3(0.0, 0.0, 0.0))
        s2.render_editor_frame(160, 120)
        var rw2 = s2.app.resources.get_ptr_mut[RenderWorld]().value()
        var ren2 = s2.app.resources.get_ptr_mut[ScalableRenderer]().value()
        var ui2 = UiDrawList()
        _ = ren2[unsafe_offset=0].render_world(rw2[unsafe_offset=0], ui2, capture_readback=True, force_readback=True)
        var px2 = ren2[unsafe_offset=0].readback_pixels()
        var diffs = 0
        for k in range(len(px1)):
            if px1[k] != px2[k]:
                diffs += 1
        _assert(diffs == 0, "3D pixel diff on " + spath + ": " + String(diffs))
        ref h_mr = s2.app.world.get_ptr[MeshRendererComponent](s2.resolve_entity(hero_id)).value()[unsafe_offset=0]
        ref p_mr = s2.app.world.get_ptr[MeshRendererComponent](s2.resolve_entity(prop_id)).value()[unsafe_offset=0]
        _assert(p_mr.material_id >= 0 and p_mr.material_id != h_mr.material_id, "prop.glb aliased hero.flmat slot")
        var p_bc = ren2[unsafe_offset=0].materials[p_mr.material_id].base_color
        _assert(
            abs(p_bc.x - Float32(1.0)) < Float32(1.0e-4)
            and abs(p_bc.y - Float32(1.0)) < Float32(1.0e-4)
            and abs(p_bc.z - Float32(1.0)) < Float32(1.0e-4),
            "prop.glb GLB default base_color mismatch",
        )
        s2.shutdown()
    print("    [OK] GLB un-overridden default material non-aliasing & 0-diff reload parity passed!")


def test_r_mcp_v2_m2_runtime_tools(tmp_root: String, mut pool: JobPool) raises:
    print("  [r] Testing McpServerV2 M2 runtime tools (physics.trigger_events, anim.set_state, audio.play, nav.query_path)...")
    var ws = tmp_root + "/m2_mcp_ws"
    _rm_rf_tmp(ws)
    _ = make_dirs(ws + "/assets")
    _ = make_dirs(ws + "/scenes")
    var sfx_wav = _make_minimal_wav_16bit()
    _assert(write_bytes(ws + "/assets/sfx.wav", sfx_wav), "write sfx.wav")
    _assert(_mcp_write_text_p(ws + "/assets/not_audio.txt", "hello"), "write not_audio.txt")

    var studio = FlintStudioState(
        headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0, project_root=ws
    )
    var assets = AssetServer(enable_watcher=False)
    var mcp = McpServerV2(workspace_root=ws)

    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"m2-test"}}}',
        studio, assets, pool,
    )

    # 1. tools/list contains all 26 tools including the 4 M2 runtime tools
    var r_list = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}', studio, assets, pool)
    var d_list = parse_json(_str_bytes(r_list))
    var t_arr = d_list.object_get(d_list.object_get(d_list.root, "result"), "tools")
    _assert(d_list.valid and d_list.array_size(t_arr) >= 26, "tools/list must expose 26 tools")
    _assert(
        r_list.find('"physics.trigger_events"') >= 0
        and r_list.find('"anim.set_state"') >= 0
        and r_list.find('"audio.play"') >= 0
        and r_list.find('"nav.query_path"') >= 0,
        "tools/list missing M2 runtime tools",
    )

    # Spawn floor, trigger zone, falling crate, and anim actor
    var floor_id = studio.spawn_entity(
        "M2_Floor", position=Vec3(0.0, -0.5, 0.0), scale=Vec3(12.0, 1.0, 12.0), collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(6.0, 0.5, 6.0), mass=0.0,
    )
    var trig_id = studio.spawn_entity(
        "M2_Trigger", position=Vec3(0.0, 1.0, 0.0), collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(2.0, 2.0, 2.0), mass=0.0,
    )
    var trig_e = studio.resolve_entity(trig_id)
    var trig_col = studio.app.world.get_ptr_mut[ColliderComponent](trig_e)
    _assert(Bool(trig_col), "trigger collider missing")
    trig_col.value()[unsafe_offset=0].is_trigger = True
    studio._sync_ecs_and_physics()

    var mover_id = studio.spawn_entity(
        "M2_Mover", position=Vec3(0.0, 1.5, 0.0), collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(0.5, 0.5, 0.5), mass=1.0,
    )
    var actor_id = studio.spawn_entity(
        "M2_AnimActor", position=Vec3(3.0, 1.0, 0.0), collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(0.5, 0.5, 0.5), mass=0.0,
    )
    _assert(floor_id >= 0 and trig_id >= 0 and mover_id >= 0 and actor_id >= 0, "spawn entities failed")
    var pre_pie_hash = String(studio.state_hash())

    # 2. PIE_STOPPED negative guard on all 4 tools (no stub error strings)
    var stopped_reqs = [
        '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"physics.trigger_events","arguments":{}}}',
        '{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"anim.set_state","arguments":{"entity_id":' + String(actor_id) + ',"state":"idle"}}}',
        '{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"audio.play","arguments":{"asset_path":"assets/sfx.wav"}}}',
        '{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"nav.query_path","arguments":{"start":[0.0,0.0,0.0],"goal":[4.0,0.0,4.0]}}}',
    ]
    for si in range(len(stopped_reqs)):
        var rs = mcp.handle_jsonrpc(stopped_reqs[si], studio, assets, pool)
        _assert(rs.find('"isError":true') >= 0 and rs.find('\\"ok\\":false') >= 0, "PIE_STOPPED guard failed: " + rs)
        _assert(
            rs.find("unknown tool") < 0 and rs.find("not implemented") < 0 and rs.find("unimplemented") < 0 and rs.find("unsupported") < 0,
            "stub error string leaked in PIE_STOPPED guard: " + rs,
        )

    # 3. Enter PIE & step to generate trigger events
    var r_play = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"play"}}}',
        studio, assets, pool,
    )
    _assert(r_play.find('\\"pie_mode\\":\\"playing\\"') >= 0, "pie play failed: " + r_play)
    var r_step = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":21,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"step","ticks":1}}}',
        studio, assets, pool,
    )
    _assert(r_step.find('\\"ok\\":true') >= 0, "pie step failed: " + r_step)

    var r_trig = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":22,"method":"tools/call","params":{"name":"physics.trigger_events","arguments":{}}}',
        studio, assets, pool,
    )
    _assert(r_trig.find('\\"ok\\":true') >= 0 and r_trig.find('\\"events\\":[') >= 0 and r_trig.find('\\"enter\\"') >= 0, "trigger_events enter failed: " + r_trig)
    var r_step2 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":220,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"step","ticks":1}}}',
        studio, assets, pool,
    )
    _assert(r_step2.find('\\"ok\\":true') >= 0, "pie step 2 failed: " + r_step2)
    var r_trig2 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":221,"method":"tools/call","params":{"name":"physics.trigger_events","arguments":{"entity_id":' + String(trig_id) + '}}}',
        studio, assets, pool,
    )
    _assert(r_trig2.find('\\"ok\\":true') >= 0 and r_trig2.find('\\"stay\\"') >= 0, "trigger_events stay filter failed: " + r_trig2)
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":23,"method":"tools/call","params":{"name":"physics.trigger_events","arguments":{"entity_id":-1}}}',
            studio, assets, pool,
        ).find('"isError":true') >= 0,
        "trigger_events invalid entity_id must fail",
    )

    # 4. anim.set_state auto-attaches AnimatorComponent on live entity & updates existing
    var r_anim1 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":30,"method":"tools/call","params":{"name":"anim.set_state","arguments":{"entity_id":' + String(actor_id) + ',"state":"idle"}}}',
        studio, assets, pool,
    )
    _assert(r_anim1.find('\\"ok\\":true') >= 0 and r_anim1.find('\\"state\\":\\"idle\\"') >= 0, "anim.set_state idle failed: " + r_anim1)
    _assert(studio.app.world.has[AnimatorComponent](studio.resolve_entity(actor_id)), "AnimatorComponent must be auto-attached")
    var r_anim2 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":31,"method":"tools/call","params":{"name":"anim.set_state","arguments":{"entity_id":' + String(actor_id) + ',"state":"run","playing":false,"speed":1.5}}}',
        studio, assets, pool,
    )
    _assert(r_anim2.find('\\"ok\\":true') >= 0 and r_anim2.find('\\"playing\\":false') >= 0, "anim.set_state update failed: " + r_anim2)
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":32,"method":"tools/call","params":{"name":"anim.set_state","arguments":{"entity_id":999999,"state":"idle"}}}',
            studio, assets, pool,
        ).find('"isError":true') >= 0,
        "anim.set_state dead entity must fail",
    )
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":33,"method":"tools/call","params":{"name":"anim.set_state","arguments":{"entity_id":' + String(actor_id) + ',"state":""}}}',
            studio, assets, pool,
        ).find('"isError":true') >= 0,
        "anim.set_state empty state must fail",
    )

    # 5. audio.play positive + negative sandbox/missing/ext checks
    var r_aud = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":40,"method":"tools/call","params":{"name":"audio.play","arguments":{"asset_path":"assets/sfx.wav","volume":0.8}}}',
        studio, assets, pool,
    )
    _assert(r_aud.find('\\"ok\\":true') >= 0 and r_aud.find('\\"asset_id\\":') >= 0, "audio.play failed: " + r_aud)
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":41,"method":"tools/call","params":{"name":"audio.play","arguments":{"asset_path":"assets/does_not_exist.wav"}}}',
            studio, assets, pool,
        ).find('"isError":true') >= 0,
        "audio.play missing file must fail",
    )
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":42,"method":"tools/call","params":{"name":"audio.play","arguments":{"asset_path":"../escape.wav"}}}',
            studio, assets, pool,
        ).find('"isError":true') >= 0,
        "audio.play traversal must fail",
    )
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":43,"method":"tools/call","params":{"name":"audio.play","arguments":{"asset_path":"assets/not_audio.txt"}}}',
            studio, assets, pool,
        ).find('"isError":true') >= 0,
        "audio.play non-audio ext must fail",
    )

    # 6. nav.query_path positive (auto-bake) + negative validation
    var r_nav = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":50,"method":"tools/call","params":{"name":"nav.query_path","arguments":{"start":[0.0,0.0,0.0],"goal":[4.0,0.0,4.0]}}}',
        studio, assets, pool,
    )
    _assert(r_nav.find('\\"ok\\":true') >= 0 and r_nav.find('\\"found\\":true') >= 0 and r_nav.find('\\"waypoints\\":[[') >= 0, "nav.query_path failed: " + r_nav)
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":51,"method":"tools/call","params":{"name":"nav.query_path","arguments":{"start":[1.0,2.0],"goal":"bad"}}}',
            studio, assets, pool,
        ).find('"isError":true') >= 0,
        "nav.query_path malformed args must fail",
    )
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":52,"method":"tools/call","params":{"name":"nav.query_path","arguments":{"start":[0.0,0.0,0.0]}}}',
            studio, assets, pool,
        ).find('"isError":true') >= 0,
        "nav.query_path missing goal must fail",
    )

    # 7. CL & NDJSON framing parity via encode_mcp_frame / parse_next_mcp_frame while PIE is active
    var req_nd = '{"jsonrpc":"2.0","id":60,"method":"tools/call","params":{"name":"nav.query_path","arguments":{"start":[0.0,0.0,0.0],"goal":[2.0,0.0,2.0]}}}'
    for mode in range(2):
        var use_cl = mode == 0
        var wire = encode_mcp_frame(req_nd, use_content_length=use_cl)
        var parsed = parse_next_mcp_frame(wire)
        _assert(parsed.has_frame and parsed.used_content_length == use_cl, "frame parse mismatch")
        var resp_s = mcp._handle_request_bytes(parsed.payload, studio, assets, pool, False, 0)
        var resp_wire = encode_mcp_frame(resp_s, use_content_length=use_cl)
        var parsed_resp = parse_next_mcp_frame(resp_wire)
        _assert(parsed_resp.has_frame, "response frame parse mismatch")
        var resp_txt = String(unsafe_from_utf8=parsed_resp.payload)
        _assert(resp_txt.find('\\"found\\":true') >= 0, "CL/NDJSON parity nav response mismatch: " + resp_txt)

    # 8. Stop PIE restores exact pre_pie_hash and re-enables PIE_STOPPED guard
    var r_stop = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":70,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"stop"}}}',
        studio, assets, pool,
    )
    _assert(
        r_stop.find('\\"restored_exact\\":true') >= 0 and r_stop.find(pre_pie_hash) >= 0,
        "pie_stop hash parity failed: " + r_stop,
    )
    _assert(
        mcp.handle_jsonrpc(stopped_reqs[0], studio, assets, pool).find('"isError":true') >= 0,
        "post-stop PIE_STOPPED guard failed",
    )

    studio.shutdown()
    _rm_rf_tmp(ws)
    print("    [OK] McpServerV2 M2 runtime tools, PIE_STOPPED guard, and CL/NDJSON parity passed!")


def test_s_mcp_v2_m5b_r2_regressions(tmp_root: String, mut pool: JobPool) raises:
    print("  [s] Testing M5b r2 regressions (AnimGraphDef anim.set_state, trigger_events ring, QOA validation, asset.import JSON)...")
    var ws = tmp_root + "/m5b_r2_ws"
    _rm_rf_tmp(ws)
    _ = make_dirs(ws + "/assets")
    _ = make_dirs(ws + "/scenes")

    # 1. QoaAsset.from_bytes validation: 16B truncated header rejected, real encode_qoa accepted
    var trunc_qoa = List[UInt8](length=16, fill=0)
    trunc_qoa[0] = UInt8(0x71)  # 'q'
    trunc_qoa[1] = UInt8(0x6F)  # 'o'
    trunc_qoa[2] = UInt8(0x61)  # 'a'
    trunc_qoa[3] = UInt8(0x66)  # 'f'
    trunc_qoa[7] = UInt8(100)   # 100 samples
    trunc_qoa[8] = UInt8(1)     # 1 ch
    trunc_qoa[10] = UInt8(0xBB) # 48000 Hz (0x00BB80)
    trunc_qoa[11] = UInt8(0x80)
    _assert(not QoaAsset.from_bytes(trunc_qoa.copy()).valid, "16B truncated QOA must be invalid")
    _assert(write_bytes(ws + "/assets/corrupt.qoa", trunc_qoa), "write corrupt.qoa")

    var pcm = List[Int16](length=64, fill=Int16(1200))
    var good_qoa = encode_qoa(pcm, 1, 48000)
    _assert(QoaAsset.from_bytes(good_qoa.copy()).valid, "valid encode_qoa must pass QoaAsset.from_bytes")
    _assert(write_bytes(ws + "/assets/good.qoa", good_qoa), "write good.qoa")
    _assert(_mcp_write_text_p(ws + "/assets/src_bad.txt", "hello"), "write src_bad.txt")

    var studio = FlintStudioState(
        headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0, project_root=ws
    )
    var assets = AssetServer(enable_watcher=False)
    var mcp = McpServerV2(workspace_root=ws)
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"r2-test"}}}',
        studio, assets, pool,
    )

    # 2. asset.import invalid destination extension produces valid JSON with isError=true and no "unsupported"
    var r_imp = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"asset.import","arguments":{"source_path":"assets/src_bad.txt","dest_path":"assets/out.badext"}}}',
        studio, assets, pool,
    )
    _assert(r_imp.find('"isError":true') >= 0, "asset.import bad ext must fail: " + r_imp)
    _assert(r_imp.find("unsupported") < 0, "asset.import must not leak unsupported: " + r_imp)
    var d_imp = parse_json(_str_bytes(r_imp))
    _assert(d_imp.valid, "asset.import error response must be valid JSON: " + r_imp)
    var imp_txt = d_imp.text(
        d_imp.object_get(d_imp.at(d_imp.object_get(d_imp.object_get(d_imp.root, "result"), "content"), 0), "text"),
        "",
    )
    _assert(parse_json(_str_bytes(imp_txt)).valid, "asset.import content[0].text must be valid JSON: " + imp_txt)

    # 3. AnimatorComponent v1/v2 binary compat & legacy JSON compat (manual_state defaults to -1)
    var ac_compat = AnimatorComponent(graph_id=77, clip_name="Run")
    ac_compat.animator.manual_state = 2
    var bw = BlobWriter()
    ac_compat.write_bin(bw)
    var br_v3 = BlobReader(bw.data.copy())
    var rd_v3 = AnimatorComponent.read_bin(br_v3, 3)
    _assert(rd_v3.animator.manual_state == 2, "AnimatorComponent v3 must roundtrip manual_state")
    var br_v1 = BlobReader(bw.data.copy())
    var rd_v1 = AnimatorComponent.read_bin(br_v1, 1)
    _assert(rd_v1.animator.manual_state == -1, "AnimatorComponent v1 read_bin must default manual_state=-1")
    var br_v2 = BlobReader(bw.data.copy())
    var rd_v2 = AnimatorComponent.read_bin(br_v2, 2)
    _assert(rd_v2.animator.manual_state == -1, "AnimatorComponent v2 read_bin must default manual_state=-1")
    var leg_doc = parse_json(_str_bytes('{"graph_id":88,"clip_name":"Idle"}'))
    var rd_leg = AnimatorComponent.read_json(leg_doc, leg_doc.root)
    _assert(rd_leg.animator.manual_state == -1, "AnimatorComponent legacy JSON must default manual_state=-1")

    # 4. AnimGraphDef anim.set_state: pre-tick configure + playing=false + mid-PIE + pie_stop
    var mdl = NativeModel()
    var nd0 = NativeNode()
    nd0.name = "Root"
    nd0.base_scale = Vec3(1.0, 1.0, 1.0)
    nd0.scale = Vec3(1.0, 1.0, 1.0)
    var nd1 = NativeNode()
    nd1.name = "Spine"
    nd1.translation = Vec3(0.0, 1.0, 0.0)
    nd1.base_translation = Vec3(0.0, 1.0, 0.0)
    nd1.base_scale = Vec3(1.0, 1.0, 1.0)
    nd1.scale = Vec3(1.0, 1.0, 1.0)
    nd0.children.append(1)
    mdl.nodes.append(nd0^)
    mdl.nodes.append(nd1^)
    mdl.roots.append(0)

    var an_idle = NativeAnimation()
    an_idle.name = "Idle"
    an_idle.duration = Float32(1.0)
    var ch_i = NativeChannel()
    ch_i.node = 1
    ch_i.path = 0
    ch_i.times.append(Float32(0.0)); ch_i.times.append(Float32(1.0))
    ch_i.values.append(Float32(0.0)); ch_i.values.append(Float32(1.0)); ch_i.values.append(Float32(0.0))
    ch_i.values.append(Float32(0.0)); ch_i.values.append(Float32(1.0)); ch_i.values.append(Float32(0.0))
    an_idle.channels.append(ch_i^)
    mdl.animations.append(an_idle^)

    var an_run = NativeAnimation()
    an_run.name = "Run"
    an_run.duration = Float32(1.0)
    var ch_r = NativeChannel()
    ch_r.node = 1
    ch_r.path = 0
    ch_r.times.append(Float32(0.0)); ch_r.times.append(Float32(1.0))
    ch_r.values.append(Float32(0.0)); ch_r.values.append(Float32(2.5)); ch_r.values.append(Float32(0.0))
    ch_r.values.append(Float32(0.0)); ch_r.values.append(Float32(3.5)); ch_r.values.append(Float32(0.0))
    an_run.channels.append(ch_r^)
    mdl.animations.append(an_run^)

    var graph = AnimGraphDef()
    graph.name = "HeroGraph"
    graph.params.append("speed")
    var st_idle = StateDef()
    st_idle.name = "Idle"
    st_idle.clip = 0
    st_idle.when.append(Condition(0, OP_LE, Float32(0.5)))
    graph.states.append(st_idle^)
    var st_run = StateDef()
    st_run.name = "Run"
    st_run.clip = 1
    st_run.when.append(Condition(0, OP_GT, Float32(0.5)))
    graph.states.append(st_run^)
    var clips = List[ClipInfo]()
    clips.append(ClipInfo("Idle", Float32(1.0), True, Float32(0.0)))
    clips.append(ClipInfo("Run", Float32(1.0), True, Float32(0.0)))

    var ac_cache = studio.app.resources.get_ptr_mut[AnimAssetCache]()
    _assert(Bool(ac_cache), "AnimAssetCache missing")
    var mid = UInt64(9501)
    var gid = UInt64(9601)
    ac_cache.value()[unsafe_offset=0].register_model(mid, mdl^)
    ac_cache.value()[unsafe_offset=0].register_graph(gid, graph^, clips^)

    var g_eid = studio.spawn_entity("GraphActor")
    var g_ent = studio.resolve_entity(g_eid)
    _ = studio.app.world.insert[SkeletonComponent](g_ent, SkeletonComponent(mid))
    _ = studio.app.world.insert[AnimatorComponent](g_ent, AnimatorComponent(graph_id=gid))

    # 5. Trigger scene entities for multi-tick & render-interleaved buffering
    var floor_id = studio.spawn_entity(
        "R2_Floor", position=Vec3(0.0, -0.5, 0.0), scale=Vec3(12.0, 1.0, 12.0), collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(6.0, 0.5, 6.0), mass=0.0,
    )
    var trig_id = studio.spawn_entity(
        "R2_Trigger", position=Vec3(0.0, 1.0, 0.0), collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(2.0, 2.0, 2.0), mass=0.0,
    )
    studio.app.world.get_ptr_mut[ColliderComponent](studio.resolve_entity(trig_id)).value()[unsafe_offset=0].is_trigger = True
    studio._sync_ecs_and_physics()
    var mover_id = studio.spawn_entity(
        "R2_Mover", position=Vec3(0.0, 1.5, 0.0), collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(0.5, 0.5, 0.5), mass=1.0,
    )
    _assert(floor_id >= 0 and trig_id >= 0 and mover_id >= 0, "spawn r2 trigger scene failed")
    var pre_pie_hash = String(studio.state_hash())

    # Enter PIE -> pre-tick anim.set_state(Run, playing=false) must survive configure() + 3 steps
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"play"}}}',
        studio, assets, pool,
    )
    var r_pre = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"anim.set_state","arguments":{"entity_id":' + String(g_eid) + ',"state":"Run","playing":false}}}',
        studio, assets, pool,
    )
    _assert(r_pre.find('\\"ok\\":true') >= 0 and r_pre.find('\\"playing\\":false') >= 0, "pre-tick anim.set_state failed: " + r_pre)
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"step","ticks":3}}}',
        studio, assets, pool,
    )
    ref ac_post1 = studio.app.world.get_ptr[AnimatorComponent](g_ent).value()[unsafe_offset=0]
    ref sk_post1 = studio.app.world.get_ptr[SkeletonComponent](g_ent).value()[unsafe_offset=0]
    _assert(ac_post1.animator.state == 1 and ac_post1.animator.manual_state == 1, "AnimGraphDef reverted after pre-tick set_state")
    _assert(ac_post1.animator.clip_time == Float32(0.0), "clip_time must stay 0.0 while playing=false")
    _assert(abs(sk_post1.skeleton.pose.t[1].y - Float32(2.5)) < Float32(1.0e-3), "skeleton pose must reflect Run t=0 while paused")

    # audio.play rejects 16B corrupt.qoa and accepts valid good.qoa
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"audio.play","arguments":{"asset_path":"assets/corrupt.qoa"}}}',
            studio, assets, pool,
        ).find('"isError":true') >= 0,
        "audio.play must reject 16B corrupt.qoa",
    )
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":14,"method":"tools/call","params":{"name":"audio.play","arguments":{"asset_path":"assets/good.qoa"}}}',
            studio, assets, pool,
        ).find('\\"ok\\":true') >= 0,
        "audio.play must accept valid good.qoa",
    )

    # physics.trigger_events rejects arguments: []
    _assert(
        mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":15,"method":"tools/call","params":{"name":"physics.trigger_events","arguments":[]}}',
            studio, assets, pool,
        ).find('"isError":true') >= 0,
        "physics.trigger_events must reject array arguments",
    )

    # Query after 3 ticks (enter on tick 1 + stay on ticks 2,3) -> must preserve enter, zero duplicates on re-query
    var r_tev1 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":16,"method":"tools/call","params":{"name":"physics.trigger_events","arguments":{}}}',
        studio, assets, pool,
    )
    _assert(r_tev1.find('\\"enter\\"') >= 0 and r_tev1.find('\\"stay\\"') >= 0, "multi-tick trigger_events dropped enter/stay: " + r_tev1)
    var r_tev1_dup = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":17,"method":"tools/call","params":{"name":"physics.trigger_events","arguments":{}}}',
        studio, assets, pool,
    )
    _assert(r_tev1_dup.find('\\"count\\":0') >= 0, "immediate re-query must return 0 events: " + r_tev1_dup)

    # Move mover out of trigger -> exit on tick 1 + 4 more ticks via pie_control(step, ticks=5)
    studio.app.world.get_ptr_mut[LocalTransform](studio.resolve_entity(mover_id)).value()[unsafe_offset=0].translation = Vec3(50.0, 1.5, 0.0)
    studio._sync_ecs_and_physics()
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":18,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"step","ticks":5}}}',
        studio, assets, pool,
    )
    var r_tev_exit = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":19,"method":"tools/call","params":{"name":"physics.trigger_events","arguments":{}}}',
        studio, assets, pool,
    )
    _assert(r_tev_exit.find('\\"exit\\"') >= 0, "pie_step(5) dropped exit event: " + r_tev_exit)

    # Stop PIE -> exact hash & manual_state == -1, then fresh PIE session tests mid-PIE set_state & 3x render_editor_frame
    var r_stop1 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"stop"}}}',
        studio, assets, pool,
    )
    _assert(r_stop1.find('\\"restored_exact\\":true') >= 0 and r_stop1.find(pre_pie_hash) >= 0, "stop1 hash mismatch: " + r_stop1)
    _assert(studio.app.world.get_ptr[AnimatorComponent](g_ent).value()[].animator.manual_state == -1, "pie_stop must restore manual_state=-1")

    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":21,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"play"}}}',
        studio, assets, pool,
    )
    # Step 2 ticks so GraphActor starts in Idle (0), and Mover emits enter
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":22,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"step","ticks":2}}}',
        studio, assets, pool,
    )
    _assert(studio.app.world.get_ptr[AnimatorComponent](g_ent).value()[].animator.state == 0, "expected Idle before mid-PIE set_state")
    # 3x render_editor_frame swaps must NOT evict buffered trigger events
    studio.render_editor_frame(160, 120)
    studio.render_editor_frame(160, 120)
    studio.render_editor_frame(160, 120)
    var r_tev_ren = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":23,"method":"tools/call","params":{"name":"physics.trigger_events","arguments":{}}}',
        studio, assets, pool,
    )
    _assert(r_tev_ren.find('\\"enter\\"') >= 0, "3x render_editor_frame dropped enter event: " + r_tev_ren)

    # Mid-PIE anim.set_state(Run, playing=true) -> stays Run and advances clip_time
    var r_mid = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":24,"method":"tools/call","params":{"name":"anim.set_state","arguments":{"entity_id":' + String(g_eid) + ',"state":"Run","playing":true}}}',
        studio, assets, pool,
    )
    _assert(r_mid.find('\\"ok\\":true') >= 0, "mid-PIE set_state failed: " + r_mid)
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":25,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"step","ticks":3}}}',
        studio, assets, pool,
    )
    ref ac_mid = studio.app.world.get_ptr[AnimatorComponent](g_ent).value()[unsafe_offset=0]
    _assert(ac_mid.animator.state == 1 and ac_mid.animator.manual_state == 1, "mid-PIE AnimGraphDef reverted")
    _assert(ac_mid.animator.clip_time > Float32(0.0), "playing=true must advance clip_time")

    var r_stop2 = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":26,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"stop"}}}',
        studio, assets, pool,
    )
    _assert(r_stop2.find('\\"restored_exact\\":true') >= 0 and r_stop2.find(pre_pie_hash) >= 0, "stop2 hash mismatch: " + r_stop2)

    studio.shutdown()
    _rm_rf_tmp(ws)
    print("    [OK] M5b r2 regressions (AnimGraphDef anim.set_state, trigger_events ring, QOA validation, asset.import JSON) passed!")


def test_t_m3_studio_menu_and_mcp_project_package(tmp_root: String, mut pool: JobPool) raises:
    print("  [t] Testing F27b Studio Build->Package menu, SceneAssetResolver fallback & MCP project.package...")
    var proj = tmp_root + "/m3_pkg_proj"
    var no_toml = tmp_root + "/no_toml_dir"
    var outside = tmp_root + "/outside_pkg"
    _rm_rf_tmp(proj)
    _rm_rf_tmp(no_toml)
    _rm_rf_tmp(outside)
    _ = make_dirs(proj + "/src")
    _ = make_dirs(proj + "/assets")
    _ = make_dirs(no_toml)
    _ = make_dirs(outside)

    _assert(
        _mcp_write_text_p(
            proj + "/flint.toml",
            '[project]\nname = "m3_pkg_proj"\nversion = "0.1.0"\nentry = "src/main.mojo"\n\n[engine]\nflint_version = ">=5.0"\ndeterministic_fp = true\nwerror = true\n\n[assets]\npack_output = "build/assets.flpk"\n',
        ),
        "write flint.toml",
    )
    _assert(_mcp_write_text_p(proj + "/assets/hello.json", '{"hello":"m3"}\n'), "write hello.json")
    _assert(_mcp_write_text_p(proj + "/src/main.mojo", "def main():\n    pass\n"), "write minimal main.mojo")

    # 1. SceneMenu detects flint.toml without FLINT_STUDIO_IDE_LAUNCHER
    _assert(not scene_menu.SceneMenu(no_toml).has_project, "empty dir must have has_project=False")
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=1280,
        height=720,
        worker_threads=0,
        project_root=proj,
    )
    _assert(studio.menu.has_project, "SceneMenu(project_root) must detect flint.toml")
    _assert(len(studio.menu.bar.groups) >= 8, "SceneMenu must have Build group index 7")
    _assert(studio.menu.bar.groups[7].title == "Build", "group 7 must be Build")
    _assert(studio.menu.bar.groups[7].items[0].command == scene_menu.PACKAGE, "Build item 0 must be PACKAGE")

    var inp = InputState()
    inp.begin_frame()
    _ = studio.handle_command_menu(inp, 1280, 720)
    _assert(studio.menu.bar.groups[7].items[0].enabled, "Build->Package must be enabled in edit mode")
    _assert(studio.pie_play(), "pie_play must start")
    _ = studio.handle_command_menu(inp, 1280, 720)
    _assert(not studio.menu.bar.groups[7].items[0].enabled, "Build->Package must be disabled during PIE")
    _ = studio.pie_stop()

    # 2. _resolve_flint_launcher_path & package_project validation
    var launcher = _resolve_flint_launcher_path()
    _assert(launcher.endswith("/tools/flint.sh") and file_size(launcher) > 0, "launcher path not found: " + launcher)
    _assert(not studio.package_project(profile="bad;name")[0], "invalid profile name must fail")
    _assert(not studio.package_project(profile="release", target="spark")[0], "profile+target must fail")

    # 3. SceneAssetResolver._read_glb_bytes_for_ref with loose_root ending in /assets
    var mb_glb = MeshBuilder()
    mb_glb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.4, 0.4, 0.4))
    var mdl_glb = NativeModel()
    mdl_glb.meshes.append(mb_glb^.finish())
    var raw_glb = encode_native_glb(mdl_glb)
    _assert(write_bytes(proj + "/assets/prop.glb", raw_glb), "write prop.glb")

    var assets = AssetServer(enable_watcher=False)
    _ = assets.watch_loose_directory(proj + "/assets")
    var got_glb = SceneAssetResolver._read_glb_bytes_for_ref(assets, AssetRef.from_path("assets/prop.glb", ASSET_KIND_MESH))
    _assert(len(got_glb) == len(raw_glb), "SceneAssetResolver._read_glb_bytes_for_ref failed with /assets loose_root")

    # 4. MCP project.package tool registration, traversal/symlink sandboxing, and valid package
    var mcp = McpServerV2(workspace_root=proj)
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"m3-test"}}}',
        studio, assets, pool,
    )
    var r_list = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}', studio, assets, pool)
    _assert(r_list.find('"project.package"') >= 0, "tools/list missing project.package")
    _assert(r_list.find('"scene.save"') >= 0 and r_list.find('"scene.load"') >= 0, "tools/list missing scene.save/load")

    var sym_src = outside
    var sym_dst = proj + "/sym_escape"
    _assert(
        external_call["symlink", Int32](sym_src.as_c_string_span(), sym_dst.as_c_string_span()) == Int32(0),
        "symlink setup failed",
    )

    var bad_outs = List[String]()
    bad_outs.append('"../escape_pkg"')
    bad_outs.append('"' + outside + '/abs_escape"')
    bad_outs.append('"sym_escape"')
    bad_outs.append('"sym_escape/via_symlink"')
    bad_outs.append('"."')
    bad_outs.append('""')
    bad_outs.append('"assets/hello.json"')
    bad_outs.append("123")
    for bi in range(len(bad_outs)):
        var req = (
            '{"jsonrpc":"2.0","id":'
            + String(10 + bi)
            + ',"method":"tools/call","params":{"name":"project.package","arguments":{"output_dir":'
            + bad_outs[bi]
            + "}}}"
        )
        var r_bad = mcp.handle_jsonrpc(req, studio, assets, pool)
        _assert(
            r_bad.find('"isError":true') >= 0 and r_bad.find('\\"ok\\":false') >= 0,
            "bad output_dir must be rejected: " + bad_outs[bi] + " -> " + r_bad,
        )
    var r_bad_prof = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":30,"method":"tools/call","params":{"name":"project.package","arguments":{"output_dir":"dist/badp","profile":"bad$prof"}}}',
        studio, assets, pool,
    )
    _assert(r_bad_prof.find('"isError":true') >= 0, "bad profile must be rejected")

    var r_bad_tgt = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":31,"method":"tools/call","params":{"name":"project.package","arguments":{"output_dir":"dist/bad_tgt","target":"/etc/passwd"}}}',
        studio, assets, pool,
    )
    _assert(
        r_bad_tgt.find('"isError":true') >= 0 and r_bad_tgt.find('\\"ok\\":false') >= 0,
        "target /etc/passwd must be rejected: " + r_bad_tgt,
    )
    _assert(not is_directory(proj + "/dist/bad_tgt/bin"), "bad target must not create output_dir/bin")

    var r_no_args = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":32,"method":"tools/call","params":{"name":"project.package"}}',
        studio, assets, pool,
    )
    _assert(
        r_no_args.find('"isError":false') >= 0 and r_no_args.find('\\"ok\\":true') >= 0 and r_no_args.find("dist/package") >= 0,
        "omitted arguments project.package failed: " + r_no_args,
    )
    _assert(file_size(proj + "/dist/package/run.sh") > 0, "missing default dist/package/run.sh")

    var r_pkg = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":33,"method":"tools/call","params":{"name":"project.package","arguments":{"output_dir":"dist/unit_pkg"}}}',
        studio, assets, pool,
    )
    _assert(
        r_pkg.find('"isError":false') >= 0 and r_pkg.find('\\"ok\\":true') >= 0 and r_pkg.find("dist/unit_pkg") >= 0,
        "valid project.package failed: " + r_pkg,
    )
    _assert(file_size(proj + "/dist/unit_pkg/run.sh") > 0, "missing packaged run.sh")
    _assert(file_size(proj + "/dist/unit_pkg/assets.flpk") > 0, "missing packaged assets.flpk")
    _assert(file_size(proj + "/dist/unit_pkg/MANIFEST.txt") > 0, "missing packaged MANIFEST.txt")
    _assert(file_size(proj + "/build/package.log") > 0, "missing build/package.log")

    # 5. Custom [assets] pak_path reporting in MCP project.package
    var proj_custom = tmp_root + "/m3_custom_pak_proj"
    _rm_rf_tmp(proj_custom)
    _ = make_dirs(proj_custom + "/src")
    _ = make_dirs(proj_custom + "/assets")
    _assert(
        _mcp_write_text_p(
            proj_custom + "/flint.toml",
            '[project]\nname = "m3_custom_pak_proj"\nversion = "0.1.0"\nentry = "src/main.mojo"\n\n[engine]\nflint_version = ">=5.0"\ndeterministic_fp = true\nwerror = true\n\n[assets]\npak_path = "build/custom_mcp.flpk"\n',
        ),
        "write custom pak flint.toml",
    )
    _assert(_mcp_write_text_p(proj_custom + "/assets/hello.json", '{"hello":"custom"}\n'), "write custom hello.json")
    _assert(_mcp_write_text_p(proj_custom + "/src/main.mojo", "def main():\n    pass\n"), "write custom main.mojo")

    var studio_c = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=1280,
        height=720,
        worker_threads=0,
        project_root=proj_custom,
    )
    var assets_c = AssetServer(enable_watcher=False)
    var mcp_c = McpServerV2(workspace_root=proj_custom)
    _ = mcp_c.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"m3-custom-pak"}}}',
        studio_c, assets_c, pool,
    )
    var r_custom = mcp_c.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"project.package","arguments":{"output_dir":"dist/custom_pkg"}}}',
        studio_c, assets_c, pool,
    )
    _assert(
        r_custom.find('"isError":false') >= 0
        and r_custom.find('\\"ok\\":true') >= 0
        and r_custom.find('\\"has_assets_flpk\\":true') >= 0
        and r_custom.find("dist/custom_pkg/custom_mcp.flpk") >= 0,
        "custom pak_path project.package must report has_assets_flpk:true and custom_mcp.flpk: " + r_custom,
    )
    _assert(file_size(proj_custom + "/dist/custom_pkg/custom_mcp.flpk") > 0, "missing custom_mcp.flpk")
    _assert(file_size(proj_custom + "/dist/custom_pkg/run.sh") > 0, "missing custom run.sh")
    _assert(file_size(proj_custom + "/dist/custom_pkg/MANIFEST.txt") > 0, "missing custom MANIFEST.txt")
    assets_c.close()
    studio_c.shutdown()
    _rm_rf_tmp(proj_custom)

    assets.close()
    studio.shutdown()
    _rm_rf_tmp(proj)
    _rm_rf_tmp(no_toml)
    _rm_rf_tmp(outside)
    print("    [OK] F27b Studio Build->Package menu, SceneAssetResolver fallback & MCP project.package passed!")


def _t5_str_bytes(s: String) -> List[UInt8]:
    var src = s.as_bytes()
    var out = List[UInt8](capacity=len(src))
    for i in range(len(src)):
        out.append(src[i])
    return out^


def test_u_mcp_pie_isolation_and_material_assign_ext(
    tmp_root: String, mut pool: JobPool
) raises:
    var ws = tmp_root + "/mcp_pie_iso_ws"
    _rm_rf_tmp(ws)
    _ = make_dirs(ws + "/assets")
    _ = make_dirs(ws + "/scenes")

    var mat_bytes = _t5_str_bytes('{"format":"flmat","version":1,"base_color":[0.2,0.7,0.3],"roughness":0.4}')
    _assert(write_bytes(ws + "/assets/valid.flmat", mat_bytes), "write valid.flmat")
    var sneaky_bytes = _t5_str_bytes('{"name":"Sneaky"}')
    _assert(write_bytes(ws + "/assets/sneaky.json", sneaky_bytes), "write sneaky.json")
    var pf_bytes = _t5_str_bytes(
        '{"format":"flprefab","version":1,"name":"PieBox","nodes":[{"name":"PieBox","parent_slot":-1,"components":{"LocalTransform":{"translation":[0.0,0.0,0.0],"rotation":[0.0,0.0,0.0,1.0],"scale":[1.0,1.0,1.0]}}}]}'
    )
    _assert(write_bytes(ws + "/assets/box.flprefab", pf_bytes), "write box.flprefab")

    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
        project_root=ws,
    )
    var assets = AssetServer(enable_watcher=False)
    assets.loose_root = ws
    var mcp = McpServerV2(ws)

    var r_sp = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ecs.spawn_entity","arguments":{"name":"Hero","position":[0.0,10.0,0.0],"mass":1.0}}}',
        studio,
        assets,
        pool,
    )
    _assert(r_sp.find('\\"ok\\":true') >= 0, "spawn Hero failed: " + r_sp)
    var hero_id = 0
    _assert(len(studio.undo_stack) == 1, "expected undo_stack len == 1 after Hero spawn")

    # 1. material.assign rejects non-.flmat extension while accepting valid .flmat
    var r_bad_mat = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"material.assign","arguments":{"entity_id":0,"material_path":"assets/sneaky.json"}}}',
        studio,
        assets,
        pool,
    )
    _assert(
        r_bad_mat.find('"isError":true') >= 0 and r_bad_mat.find('\\"ok\\":false') >= 0,
        "material.assign must reject .json extension: " + r_bad_mat,
    )
    var r_good_mat = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"material.assign","arguments":{"entity_id":0,"material_path":"assets/valid.flmat"}}}',
        studio,
        assets,
        pool,
    )
    _assert(r_good_mat.find('\\"ok\\":true') >= 0, "material.assign valid .flmat failed: " + r_good_mat)
    var pre_pie_hash = String(studio.state_hash())
    var pre_undo_len = len(studio.undo_stack)

    # 2. Enter PIE & step gravity so Hero falls
    var r_play = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"play"}}}',
        studio,
        assets,
        pool,
    )
    _assert(r_play.find('\\"pie_mode\\":\\"playing\\"') >= 0, "pie play failed")
    var r_step = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"step","ticks":20}}}',
        studio,
        assets,
        pool,
    )
    _assert(r_step.find('\\"ok\\":true') >= 0, "pie step failed")
    _assert(studio.entities[hero_id].translation.y < Float32(9.8), "Hero did not fall during PIE")

    # 3. All Edit-world mutating/saving/packaging MCP tools must fail cleanly during PIE
    var pie_blocked_reqs: List[String] = [
        '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"ecs.spawn_entity","arguments":{"name":"PieLeak"}}}',
        '{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"ecs.patch_components","arguments":{"entity_id":0,"name":"PieMutated"}}}',
        '{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"ecs.despawn_entity","arguments":{"entity_id":0}}}',
        '{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"procedural.generate_asset","arguments":{"preset":"crate","spawn_in_scene":true}}}',
        '{"jsonrpc":"2.0","id":14,"method":"tools/call","params":{"name":"physics.step_deterministic","arguments":{"ticks":5}}}',
        '{"jsonrpc":"2.0","id":15,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"undo","steps":1}}}',
        '{"jsonrpc":"2.0","id":16,"method":"tools/call","params":{"name":"render.configure_npr_cel","arguments":{"entity_id":0,"shading_mode":"npr_cel"}}}',
        '{"jsonrpc":"2.0","id":17,"method":"tools/call","params":{"name":"scene.save","arguments":{"path":"scenes/pie_leaked.flscn"}}}',
        '{"jsonrpc":"2.0","id":18,"method":"tools/call","params":{"name":"scene.load","arguments":{"path":"scenes/pie_leaked.flscn"}}}',
        '{"jsonrpc":"2.0","id":19,"method":"tools/call","params":{"name":"scene.set_entity_enabled","arguments":{"entity_id":0,"enabled":false}}}',
        '{"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"entity.set_transform","arguments":{"entity_id":0,"position":[99.0,0.0,0.0]}}}',
        '{"jsonrpc":"2.0","id":21,"method":"tools/call","params":{"name":"material.assign","arguments":{"entity_id":0,"material_path":"assets/valid.flmat"}}}',
        '{"jsonrpc":"2.0","id":22,"method":"tools/call","params":{"name":"prefab.spawn","arguments":{"prefab_path":"assets/box.flprefab"}}}',
        '{"jsonrpc":"2.0","id":23,"method":"tools/call","params":{"name":"project.package","arguments":{"output_dir":"dist/pie_pkg"}}}',
    ]
    for ri in range(len(pie_blocked_reqs)):
        var resp = mcp.handle_jsonrpc(pie_blocked_reqs[ri], studio, assets, pool)
        _assert(
            resp.find('"isError":true') >= 0 and resp.find('\\"ok\\":false') >= 0,
            "PIE isolation failed for req " + String(ri) + ": " + resp,
        )

    _assert(file_size(ws + "/scenes/pie_leaked.flscn") < 0, "PIE scene.save leaked file to disk")
    _assert(len(studio.undo_stack) == pre_undo_len, "PIE leaked entries into undo_stack")

    # 4. Stop PIE -> exact state_hash restored & undo_stack uncorrupted
    var r_stop = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":30,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"stop"}}}',
        studio,
        assets,
        pool,
    )
    _assert(
        r_stop.find('\\"restored_exact\\":true') >= 0 and r_stop.find(pre_pie_hash) >= 0,
        "pie_stop hash mismatch: " + r_stop,
    )
    _assert(len(studio.undo_stack) == pre_undo_len, "undo_stack length changed across PIE")
    _assert(studio.undo(), "post-PIE undo() failed")
    _assert(studio.live_entity_count() == 1 and studio.entities[hero_id].translation.y > Float32(9.9), "post-PIE undo corrupted Hero position")

    assets.close()
    studio.shutdown()
    _rm_rf_tmp(ws)
    print("    [OK] McpServerV2 PIE isolation & material.assign .flmat extension check passed!")


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
    test_g_audit_regressions()
    test_h_wave2_png_async_studio_scene(tmp_root, pack_path, pool)
    test_i_f20_residency_and_f21_codecs(tmp_root)
    test_j_studio_v5_unification_ux_and_headless_smoke(tmp_root)
    test_k_procedural_qem_and_toksvig()
    test_l_cook_workspace_symlink_loop_bomb(tmp_root, pool)
    test_m_mcp_socket_path_stdlib_open_coexistence()
    test_n_os_read_bytes_dir_and_cooker_trailing_slash(tmp_root, pool)
    test_o_mcp_v2_m0_scene_and_transform_tools(tmp_root, pool)
    test_p_mcp_v2_m1_assets_prefab_material_and_upload(tmp_root, pool)
    test_q_glb_unoverridden_material_reload_parity(tmp_root)
    test_r_mcp_v2_m2_runtime_tools(tmp_root, pool)
    test_s_mcp_v2_m5b_r2_regressions(tmp_root, pool)
    test_t_m3_studio_menu_and_mcp_project_package(tmp_root, pool)
    test_u_mcp_pie_isolation_and_material_assign_ext(tmp_root, pool)

    pool.stop()
    _rm_rf_tmp(tmp_root)

    var elapsed_ms = (get_time() - t0) * 1000.0
    print("[Flint v5 Track 5] ALL 21 SUITES PASSED in " + String(Int(elapsed_ms)) + " ms!")
