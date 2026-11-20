"""Tier 5 Adversarial Coverage Hardening: Render & Asset Subsystems (11 tests).
Validates MaterialAsset GPU SSBO extensions, AaaGeometryPool + flpk_v3 cooked slabs,
unaligned GLB parsing, flpk_v3 alignment & corruption/overflow rejection, QEM + BC7/BC6H,
v5 SPIR-V shaders, F19 1025-material growth & 18-bit instance limits, HOST_CACHED
DEVICE_LOCAL staging readbacks for GPU skinning velocity, RenderGraph budgets, and ReSTIR/VSM.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs
from std.memory import bitcast
from flint.assets.bc_compress import (
    compute_argb_psnr_db,
    decode_bc6h_hdr,
    decode_bc7_rgba,
    encode_bc6h_hdr,
    encode_bc7_rgba,
)
from flint.assets.flpk_v3 import (
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    CHUNK_COMP_NONE,
    AssetUuid,
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    crc64_ecma_ptr,
    decode_cooked_mesh_slab_into,
    encode_cooked_mesh_slab,
    encode_flpk_v3,
)
from flint.assets.glb_native import parse_native_glb
from flint.assets.glb_writer import append_f32_le, append_u32_le, encode_native_glb
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.meshlet import AaaGeometryPool, simplify_mesh_qem
from flint.assets.model import NativeMesh, NativeModel, NativeNode
from flint.core.lz4 import address_of
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat, trs
from flint.platform.os import write_bytes
from flint.platform.vulkan import (
    ACCESS_HOST_READ,
    ACCESS_TRANSFER_WRITE,
    GpuBuffer,
    PIPELINE_HOST,
    PIPELINE_TRANSFER,
)
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_clustered_lights import RestirReservoir
from flint.render.aaa_froxel_volumetrics import (
    froxel_inject_and_reproject_shader,
    froxel_raymarch_integrate_shader,
)
from flint.render.aaa_gi_svgf import (
    AAA_SHADE_CLASS_STANDARD_PBR,
    aaa_deferred_shade_class_shader,
    aaa_shade_sky_shader,
    aaa_tile_classify_shader,
)
from flint.render.aaa_post_tsr import (
    AAA_DISPLAY_LINEAR_SRGB,
    aaa_display_tonemap_shader,
    aaa_tsr_v3_resolve_shader,
)
from flint.render.aaa_render_graph import (
    AaaRenderGraph,
    RG_QUEUE_ASYNC_COMPUTE,
    RG_QUEUE_GRAPHICS,
    RG_USAGE_COMPUTE_READ,
    RG_USAGE_COMPUTE_WRITE,
    RG_USAGE_FRAGMENT_READ,
    RgResourceAccess,
)
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_skinning_rt import AAA_SKIN_PREV_POS_BASE, DualQuat
from flint.render.aaa_visbuffer import (
    AAA_MAX_PACKED_INSTANCES,
    hiz_depth_downsample_shader,
    meshlet_cull_shader,
    visbuffer_fragment_shader,
    visbuffer_vertex_shader,
)
from flint.render.aaa_vsm import VSM_TABLE_WORDS, VSM_VIRTUAL_PAGE_COUNT, VsmPageManager
from flint.render.material_graph import (
    MaterialAsset,
    terrain_clipmap_eval_shader,
    vat_crowd_eval_shader,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.render.ui_overlay import aaa_ui_overlay_shader
from e2e.common.harness import TestContext


def _pid_str() -> String:
    return String(Int(external_call["getpid", Int32]()))


def _readback_dev_floats(mut ren: FlintAaaVulkanRenderer, buf_kind: Int, byte_off: Int, float_count: Int) -> List[Float32]:
    var out = List[Float32](length=float_count, fill=Float32(0.0))
    var src_vk = ren.buf_skinning.vk_buffer if buf_kind == 0 else ren.buf_hdr_out.vk_buffer
    if not ren.active or src_vk == 0 or float_count <= 0:
        return out^
    var byte_len = float_count * 4
    var rb = ren.vk.create_buffer(byte_len, local=False, cpu_reads=True)
    if rb.vk_buffer != 0 and rb.cpu_address != 0:
        ren.vk.wait_idle()
        if ren.vk.begin_commands(0):
            if buf_kind == 0:
                ren.vk.copy_buffer(0, ren.buf_skinning, byte_off, rb, 0, byte_len)
            else:
                ren.vk.copy_buffer(0, ren.buf_hdr_out, byte_off, rb, 0, byte_len)
            ren.vk.memory_barrier2(
                0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ)
            )
            var f = ren.vk.submit(0)
            if f != 0 and ren.vk.wait(f):
                _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), rb.cpu_address, byte_len)
    ren.vk.destroy_buffer(rb)
    return out^


def test_t5_cooked_model_preserves_extensions(mut ctx: TestContext) raises:
    var mat1 = MaterialAsset(base_color=Vec3(0.7, 0.75, 0.8), roughness=0.3, metallic=0.8)
    mat1.clearcoat = Float32(0.85)
    mat1.clearcoat_roughness = Float32(0.12)
    mat1.specular_factor = Float32(0.25)
    mat1.iridescence = Float32(0.90)
    mat1.iridescence_thickness_nm = Float32(420.0)
    mat1.iridescence_ior = Float32(1.45)
    mat1.sheen_color = Vec3(0.4, 0.6, 0.8)
    mat1.sheen_roughness = Float32(0.35)

    var mat2 = MaterialAsset(base_color=Vec3(0.7, 0.75, 0.8), roughness=0.3, metallic=0.8)
    var w1 = mat1.pack_words()
    var w2 = mat2.pack_words()
    var diff_words = 0
    for i in range(len(w1)):
        if w1[i] != w2[i]:
            diff_words += 1
    ctx.assert_true(diff_words >= 4, "MaterialAsset 24-word GPU packing detects PBR extension differences")

    var m1 = NativeModel()
    var mb = MeshBuilder(base_color=0xFFAABBCC, roughness=0.3, metallic=0.8)
    mb.add_beveled_box(Vec3(-1, -1, -1), Vec3(1, 1, 1), 0.1)
    var mesh = mb^.finish()
    mesh.clearcoat = 0.85
    mesh.clearcoat_roughness = 0.12
    mesh.specular_factor = 0.25
    mesh.iridescence = 0.90
    mesh.iridescence_thickness_nm = 420.0
    mesh.iridescence_ior = 1.45
    mesh.sheen_color = Vec3(0.4, 0.6, 0.8)
    mesh.sheen_roughness = 0.35
    m1.meshes.append(mesh.copy())

    var loaded = parse_native_glb(encode_native_glb(m1))
    ctx.assert_approx(loaded.meshes[0].clearcoat, 0.85, 0.01, "GLB preserves clearcoat")
    ctx.assert_approx(loaded.meshes[0].clearcoat_roughness, 0.12, 0.01, "GLB preserves clearcoat_roughness")
    ctx.assert_approx(loaded.meshes[0].specular_factor, 0.25, 0.01, "GLB preserves specular_factor")
    ctx.assert_approx(loaded.meshes[0].iridescence, 0.90, 0.01, "GLB preserves iridescence")
    ctx.assert_approx(loaded.meshes[0].iridescence_thickness_nm, 420.0, 2.5, "GLB preserves iridescence_thickness_nm")
    ctx.assert_approx(loaded.meshes[0].iridescence_ior, 1.45, 0.01, "GLB preserves iridescence_ior")
    ctx.assert_approx(loaded.meshes[0].sheen_color.x, 0.4, 0.02, "GLB preserves sheen_color")
    ctx.assert_approx(loaded.meshes[0].sheen_roughness, 0.35, 0.01, "GLB preserves sheen_roughness")

    var pool = AaaGeometryPool()
    var entry = pool.append_mesh(mesh, 0, build_lods=True)
    var slab = encode_cooked_mesh_slab(pool)
    var pool2 = AaaGeometryPool()
    var dec = decode_cooked_mesh_slab_into(slab, pool2)
    ctx.assert_true(dec[0], "flpk_v3 cooked mesh slab decodes cleanly")
    ctx.assert_equal(len(pool2.meshes), 1, "decoded AaaGeometryPool preserves mesh count")
    ctx.assert_equal(pool2.meshes[0].vertex_count, entry.vertex_count, "decoded slab preserves vertex_count")
    ctx.assert_equal(pool2.meshes[0].triangle_count, entry.triangle_count, "decoded slab preserves triangle_count")
    ctx.assert_equal(len(pool2.meshes[0].lods), len(entry.lods), "decoded slab preserves LOD slices")


def test_t5_glb_writer_roundtrips_extensions(mut ctx: TestContext) raises:
    var m = NativeModel()
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.3, metallic=0.8)
    mb.add_beveled_box(Vec3(-1, -1, -1), Vec3(1, 1, 1), 0.1)
    var mesh = mb^.finish()
    mesh.clearcoat = 0.75
    mesh.clearcoat_roughness = 0.22
    mesh.specular_factor = 0.3
    mesh.iridescence = 0.8
    mesh.iridescence_ior = 1.4
    mesh.iridescence_thickness_nm = 510.0
    mesh.sheen_color = Vec3(0.9, 0.2, 0.1)
    mesh.sheen_roughness = 0.45
    m.meshes.append(mesh^)

    var n = NativeNode()
    n.name = "mat_node"
    n.mesh = 0
    n.has_matrix = True
    n.matrix = Mat4()
    n.matrix.values[12] = 3.5
    m.nodes.append(n^)
    m.roots.append(0)

    var parsed = parse_native_glb(encode_native_glb(m))
    ctx.assert_approx(parsed.meshes[0].clearcoat, 0.75, 0.01, "GLB roundtrips KHR_materials_clearcoat")
    ctx.assert_approx(parsed.meshes[0].specular_factor, 0.3, 0.01, "GLB roundtrips KHR_materials_specular")
    ctx.assert_approx(parsed.meshes[0].iridescence, 0.8, 0.01, "GLB roundtrips KHR_materials_iridescence")
    ctx.assert_approx(parsed.meshes[0].sheen_color.x, 0.9, 0.02, "GLB roundtrips KHR_materials_sheen")
    ctx.assert_true(parsed.nodes[0].has_matrix, "GLB roundtrips node.matrix")
    ctx.assert_approx(parsed.nodes[0].matrix.values[12], 3.5, 0.01, "GLB preserves node.matrix translation")


def test_t5_glb_unaligned_accessor_and_extensions(mut ctx: TestContext) raises:
    var bin = List[UInt8]()
    bin.append(0xAA)  # Force byteOffset % 4 == 1
    var pts = [Float32(0.0), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    for p in pts:
        append_f32_le(bin, p)
    while len(bin) % 4 != 0:
        bin.append(0)
    var json_str = String(
        '{"asset":{"version":"2.0"},'
        '"buffers":[{"byteLength":' + String(len(bin)) + '}],'
        '"bufferViews":[{"buffer":0,"byteOffset":1,"byteLength":36}],'
        '"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],'
        '"materials":[{"pbrMetallicRoughness":{"roughnessFactor":0.4,"metallicFactor":0.5},'
        '"emissiveFactor":[0.5,0.25,0.0],'
        '"extensions":{'
        '"KHR_materials_emissive_strength":{"emissiveStrength":4.0},'
        '"KHR_materials_clearcoat":{"clearcoatFactor":0.9,"clearcoatRoughnessFactor":0.15},'
        '"KHR_materials_specular":{"specularFactor":0.35},'
        '"KHR_materials_iridescence":{"iridescenceFactor":0.85,"iridescenceIor":1.42,"iridescenceThicknessMaximum":520.0},'
        '"KHR_materials_sheen":{"sheenColorFactor":[0.2,0.4,0.8],"sheenRoughnessFactor":0.6}'
        '}}],'
        '"meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0,"mode":4}]}]}'
    )
    var jb = List[UInt8]()
    for b in json_str.as_bytes():
        jb.append(b)
    while len(jb) % 4 != 0:
        jb.append(0x20)
    var total = 12 + 8 + len(jb) + 8 + len(bin)
    var raw_glb = List[UInt8]()
    append_u32_le(raw_glb, 0x46546C67); append_u32_le(raw_glb, 2); append_u32_le(raw_glb, total)
    append_u32_le(raw_glb, len(jb)); append_u32_le(raw_glb, 0x4E4F534A)
    for b in jb: raw_glb.append(b)
    append_u32_le(raw_glb, len(bin)); append_u32_le(raw_glb, 0x004E4942)
    for b in bin: raw_glb.append(b)

    var p = parse_native_glb(raw_glb)
    ctx.assert_equal(len(p.meshes), 1, "unaligned GLB parsed")
    ctx.assert_approx(p.meshes[0].positions[3], 1.0, 0.0001, "unaligned byteOffset=1 Float32 loaded accurately")
    ctx.assert_approx(p.meshes[0].emissive_strength, 2.0, 0.001, "emissiveStrength normalized by max emissiveFactor")
    ctx.assert_approx(p.meshes[0].clearcoat, 0.9, 0.001, "clearcoat parsed from GLB")
    ctx.assert_approx(p.meshes[0].specular_factor, 0.35, 0.001, "specular parsed from GLB")
    ctx.assert_approx(p.meshes[0].iridescence, 0.85, 0.001, "iridescence parsed from GLB")
    ctx.assert_approx(p.meshes[0].sheen_color.z, 0.8, 0.001, "sheen parsed from GLB")


def test_t5_vfs_mmap_alignment_and_corrupt_reject(mut ctx: TestContext) raises:
    var pid_s = _pid_str()
    var p_path = "/tmp/flint_e2e_t5_v3_ok_" + pid_s + ".flpk"
    var bad_path = "/tmp/flint_e2e_t5_v3_bad_" + pid_s + ".flpk"
    var ovf_path = "/tmp/flint_e2e_t5_v3_ovf_" + pid_s + ".flpk"
    ctx.cleanup_artifact(p_path)
    ctx.cleanup_artifact(bad_path)
    ctx.cleanup_artifact(ovf_path)

    try:
        var chunks = List[FlpkBuildChunk]()
        chunks.append(FlpkBuildChunk(AssetUuid.from_path("empty.bin"), "empty.bin", 0, ASSET_KIND_RAW, List[UInt8](), CHUNK_COMP_NONE))
        var f1 = List[UInt8]()
        for i in range(13): f1.append(UInt8(65 + i))
        chunks.append(FlpkBuildChunk(AssetUuid.from_path("odd_name_123.txt"), "odd_name_123.txt", 0, ASSET_KIND_RAW, f1^, CHUNK_COMP_NONE))
        var f2 = List[UInt8]()
        for i in range(4096): f2.append(UInt8(i & 0xFF))
        chunks.append(FlpkBuildChunk(AssetUuid.from_path("tail.bin"), "tail.bin", 0, ASSET_KIND_MESH, f2^, CHUNK_COMP_NONE))

        var pack = encode_flpk_v3(chunks, page_align=True)
        ctx.assert_true(write_bytes(p_path, pack), "write valid flpk_v3")
        var arch = FlpkV3MappedArchive.open(p_path)
        ctx.assert_true(arch.valid, "FlpkV3MappedArchive mmap opens cleanly")
        ctx.assert_equal(arch.toc_count(), 3, "3 TOC entries indexed")
        ctx.assert_equal(Int(arch.toc_entry(0).payload_offset % UInt64(64)), 0, "small chunk 64-byte aligned")
        ctx.assert_equal(Int(arch.toc_entry(1).payload_offset % UInt64(64)), 0, "odd-length chunk 64-byte aligned")
        ctx.assert_equal(Int(arch.toc_entry(2).payload_offset % UInt64(4096)), 0, ">=4096B chunk 4096-byte page aligned")

        var r0 = arch.read_payload_by_toc(0, verify_crc=True)
        var r1 = arch.read_payload_by_toc(1, verify_crc=True)
        var r2 = arch.read_payload_by_toc(2, verify_crc=True)
        var t2_off = Int(arch.toc_entry(2).payload_offset)
        var str_end = Int(arch.header.path_table_offset) + Int(arch.header.hash_table_capacity) * 16 + Int(arch.header.string_pool_bytes)
        var meta_end = (str_end + 63) & ~63
        arch.close()

        ctx.assert_true(r0[0] and len(r0[1]) == 0, "0-byte payload reads cleanly with CRC64")
        ctx.assert_true(r1[0] and len(r1[1]) == 13, "13-byte payload reads cleanly with CRC64")
        ctx.assert_true(r2[0] and len(r2[1]) == 4096, "4096-byte payload reads cleanly with CRC64")

        # 1. Corrupt payload byte -> verify_entry_crc64 / read_payload_by_toc rejects cleanly
        var bad = pack.copy()
        bad[t2_off] = bad[t2_off] ^ 0xFF
        _ = write_bytes(bad_path, bad)
        var arch_bad = FlpkV3MappedArchive.open(bad_path)
        ctx.assert_true(arch_bad.valid, "metadata remains valid while payload is corrupted")
        var bad_crc_ok = arch_bad.verify_entry_crc64(2)
        var bad_read = arch_bad.read_payload_by_toc(2, verify_crc=True)
        arch_bad.close()
        ctx.assert_false(bad_crc_ok, "corrupted payload fails ECMA-182 CRC64 verification")
        ctx.assert_false(bad_read[0], "read_payload_by_toc rejects corrupted payload")

        # 2. Craft 64-bit signed offset+size overflow in TOC entry 0 with valid forged TOC CRC64 -> open() reaches _range_within_u64
        var ovf = pack.copy()
        var e0_off_at = 64 + 24
        var ovf_off = UInt64(0x7FFFFFFFFFFFF000)
        var ovf_sz = UInt64(0x2000)
        for b_i in range(8):
            ovf[e0_off_at + b_i] = UInt8((ovf_off >> UInt64(b_i * 8)) & 0xFF)
            ovf[e0_off_at + 8 + b_i] = UInt8((ovf_sz >> UInt64(b_i * 8)) & 0xFF)
        var hdr_crc = crc64_ecma_ptr(address_of(ovf), 56, 0)
        var meta_crc = crc64_ecma_ptr(address_of(ovf) + 64, meta_end - 64, hdr_crc)
        _ = len(ovf)
        for b_i in range(8):
            ovf[56 + b_i] = UInt8((meta_crc >> UInt64(b_i * 8)) & 0xFF)
        _ = write_bytes(ovf_path, ovf)
        var arch_ovf = FlpkV3MappedArchive.open(ovf_path)
        var ovf_valid = arch_ovf.valid
        var ovf_cnt = arch_ovf.toc_count()
        var ovf_note = arch_ovf.error_note
        arch_ovf.close()
        ctx.assert_false(ovf_valid, "FlpkV3MappedArchive rejects overflowing TOC entry even with valid CRC64")
        ctx.assert_equal(ovf_cnt, 0, "rejected flpk_v3 exposes 0 TOC entries")
        ctx.assert_equal(ovf_note, "TOC entry payload range out of bounds", "forged-CRC overflow reaches _range_within_u64 range guard")
    except e:
        ctx.cleanup_artifact(p_path)
        ctx.cleanup_artifact(bad_path)
        ctx.cleanup_artifact(ovf_path)
        raise e^

    ctx.cleanup_artifact(p_path)
    ctx.cleanup_artifact(bad_path)
    ctx.cleanup_artifact(ovf_path)


def test_t5_qem_preserves_extensions_and_bc_non4x4(mut ctx: TestContext) raises:
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0, 0, 0), Vec3(2, 2, 2), 12, 12)
    var m = mb^.finish()
    m.orm_texture = 7
    m.clearcoat = 0.9
    m.iridescence = 0.6
    m.skin = 2
    var vc = m.vertex_count()
    m.colors = List[UInt8](length=vc * 4, fill=200)
    m.joints = List[UInt16](length=vc * 4, fill=3)
    m.weights = List[Float32](length=vc * 4, fill=0.25)

    var qem = simplify_mesh_qem(m, len(m.indices) // 6)
    ctx.assert_equal(qem[0].orm_texture, 7, "QEM preserves orm_texture")
    ctx.assert_approx(qem[0].clearcoat, 0.9, 0.001, "QEM preserves clearcoat")
    ctx.assert_approx(qem[0].iridescence, 0.6, 0.001, "QEM preserves iridescence")
    ctx.assert_equal(qem[0].skin, 2, "QEM preserves skin index")
    ctx.assert_equal(len(qem[0].colors), qem[0].vertex_count() * 4, "QEM preserves vertex colors")
    ctx.assert_equal(len(qem[0].joints), qem[0].vertex_count() * 4, "QEM preserves vertex joints")
    ctx.assert_equal(len(qem[0].weights), qem[0].vertex_count() * 4, "QEM preserves vertex weights")

    # Non-4x4 BC7 + BC6H roundtrip
    var w = 7; var h = 5
    var px = List[UInt32](length=w * h, fill=0xFF00FF00)
    for y in range(h):
        for x in range(2):
            px[y * w + x] = 0xFFFF0000
    var bc7 = encode_bc7_rgba(px, w, h, multi_mode=True)
    var dec = decode_bc7_rgba(bc7, w, h)
    ctx.assert_true(compute_argb_psnr_db(px, dec) > 35.0, "BC7 7x5 sharp-edge PSNR > 35 dB")

    var hdr = List[Vec3](length=w * h, fill=Vec3(0.01, 0.02, 0.03))
    for i in range(w * h):
        if (i % 4) < 2: hdr[i] = Vec3(500.0, 1200.0, 50.0)
    var dec6 = decode_bc6h_hdr(encode_bc6h_hdr(hdr, w, h, partitioned=True), w, h)
    ctx.assert_equal(len(dec6), w * h, "BC6H 7x5 decoded count matches")


def test_t5_ui_overlay_shader_and_glyph_render(mut ctx: TestContext) raises:
    ctx.assert_true(validate_spirv_module(aaa_ui_overlay_shader()).valid, "aaa_ui_overlay_shader validates")
    ctx.assert_true(validate_spirv_module(visbuffer_vertex_shader()).valid, "visbuffer_vertex_shader validates")
    ctx.assert_true(validate_spirv_module(visbuffer_fragment_shader()).valid, "visbuffer_fragment_shader validates")
    ctx.assert_true(validate_spirv_module(hiz_depth_downsample_shader()).valid, "hiz_depth_downsample_shader validates")
    ctx.assert_true(validate_spirv_module(meshlet_cull_shader()).valid, "meshlet_cull_shader validates")
    ctx.assert_true(validate_spirv_module(aaa_tile_classify_shader()).valid, "aaa_tile_classify_shader validates")
    ctx.assert_true(
        validate_spirv_module(aaa_deferred_shade_class_shader(True, AAA_SHADE_CLASS_STANDARD_PBR, True)).valid,
        "aaa_deferred_shade_class_shader validates",
    )
    ctx.assert_true(validate_spirv_module(aaa_shade_sky_shader()).valid, "aaa_shade_sky_shader validates")
    ctx.assert_true(validate_spirv_module(aaa_tsr_v3_resolve_shader()).valid, "aaa_tsr_v3_resolve_shader validates")
    ctx.assert_true(validate_spirv_module(aaa_display_tonemap_shader()).valid, "aaa_display_tonemap_shader validates")
    ctx.assert_true(validate_spirv_module(froxel_inject_and_reproject_shader()).valid, "froxel_inject_and_reproject_shader validates")
    ctx.assert_true(validate_spirv_module(froxel_raymarch_integrate_shader()).valid, "froxel_raymarch_integrate_shader validates")
    ctx.assert_true(validate_spirv_module(vat_crowd_eval_shader()).valid, "vat_crowd_eval_shader validates")
    ctx.assert_true(validate_spirv_module(terrain_clipmap_eval_shader()).valid, "terrain_clipmap_eval_shader validates")

    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active for UI glyph render")
    ctx.assert_true(ren.ui_font.valid, "UI SDF font loaded in renderer")
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.begin_frame(Vec3(0, 0, -5), Vec3(0, 0, 0))
    ren.ui_draw_list.add_rect(0.0, 0.0, 160.0, 120.0, 0xFF000000, 0.0)
    ren.ui_draw_list.add_text(ren.ui_font_atlas_slot, ren.ui_font, "H", 20.0, 60.0, 32.0, 0xFFFFFFFF)
    ctx.assert_true(ren.render_frame(capture_readback=True), "UI glyph frame renders")
    var px = ren.readback_pixels()
    ren.shutdown()

    var lit = 0; var dark = 0
    for y in range(30, 62):
        for x in range(20, 44):
            if (px[y * 160 + x] & 0xFF) > 128: lit += 1
            elif (px[y * 160 + x] & 0xFF) < 20: dark += 1
    ctx.assert_true(lit > 20 and dark > 20, "UI text glyph 'H' renders distinct foreground and background pixels")


def test_t5_ui_scratch_overflow_guard(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=16, height=16)
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active for 16x16 scratch guard")
    ren.begin_frame()
    for _ in range(500):
        ren.ui_draw_list.add_rect(0.0, 0.0, 10.0, 10.0, 0xFF00FF00)
    var ok = ren.render_frame(capture_readback=True)
    ren.shutdown()
    ctx.assert_true(ok, "16x16 render_frame with 500 UI prims must not overflow buf_post_scratch")


def test_t5_renderer_1024_instances_and_materials(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=192, height=128, max_instances=2048)
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active")
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False
    var m_green = ren.add_material(AaaMaterial(base_color=Vec3(0.0, 1.0, 0.0), roughness=0.5))
    var mat_r = AaaMaterial(base_color=Vec3(1.0, 0.0, 0.0), roughness=0.5)
    mat_r.emissive = Vec3(5.0, 0.0, 0.0)
    var m_red = ren.add_material(mat_r)

    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0, 0, 0), Vec3(1.5, 1.5, 1.5), 16, 16)
    var me = ren.add_mesh(mb^.finish(), m_green)

    ren.begin_frame(Vec3(0.0, 0.0, -3.6), Vec3(0.0, 0.0, 0.0), fov_y_deg=50.0)
    var offscreen = trs(Vec3(-10000.0, 0.0, 0.0), Quat(), Vec3(1, 1, 1))
    for _ in range(1024):
        ren.draw_instance(me.mesh_id, offscreen, material_override=m_green)
    ren.draw_instance(me.mesh_id, trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1)), material_override=m_red)
    ctx.assert_true(ren.render_frame(capture_readback=True), "1025-instance frame renders")
    var center_px = ren.readback_pixels()[64 * 192 + 96]

    # F19: Dynamic material buffer growth beyond 1024 materials
    while len(ren.materials) < 1024:
        _ = ren.add_material(AaaMaterial())
    var mat_1024 = AaaMaterial(base_color=Vec3(0.0, 0.0, 1.0), roughness=0.5)
    mat_1024.emissive = Vec3(0.0, 0.0, 5.0)
    var id_1024 = ren.add_material(mat_1024)
    var mat_count = len(ren.materials)
    var mat_buf_size = ren.buf_materials.size
    ren.begin_frame(Vec3(0.0, 0.0, -3.6), Vec3(0.0, 0.0, 0.0), fov_y_deg=50.0)
    ren.draw_instance(me.mesh_id, trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1)), material_override=id_1024)
    ctx.assert_true(ren.render_frame(capture_readback=True), "1025th material draws cleanly after SSBO grow")
    var blue_center = ren.readback_pixels()[64 * 192 + 96]
    ren.shutdown()

    # F19: 18-bit VisBuffer packed instance limit (262143)
    var ren_64k = FlintAaaVulkanRenderer(width=64, height=64, max_instances=65536)
    ctx.assert_true(ren_64k.active, "ren_64k active")
    var max_64k = ren_64k.max_instances
    ren_64k.shutdown()

    var ren_cap = FlintAaaVulkanRenderer(width=64, height=64, max_instances=300000)
    ctx.assert_true(ren_cap.active, "ren_cap active")
    var capped_max_inst = ren_cap.max_instances
    ren_cap.shutdown()

    ctx.assert_equal(max_64k, 65536, "65536 instances supported without legacy 32768 cap")
    ctx.assert_equal(capped_max_inst, Int(AAA_MAX_PACKED_INSTANCES), "max_instances clamps to 18-bit VisBuffer limit 262143")
    ctx.assert_true(Int((center_px >> 16) & 255) > 240 and Int(center_px & 255) < 80, "instance 1024 renders red without VisBuffer wraparound")
    ctx.assert_equal(id_1024, 1024, "1025th material succeeds via F19 dynamic buffer growth")
    ctx.assert_equal(mat_count, 1025, "material list dynamically grows to 1025")
    ctx.assert_true(mat_buf_size >= 1025 * 24 * 4, "buf_materials grew to hold 1025 materials")
    ctx.assert_true(Int(blue_center & 255) > 220, "1025th material renders blue accurately")


def test_t5_skinned_prev_pos_offset(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=64, height=64)
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active for GPU skinning test")
    ctx.assert_equal(ren.buf_skinning.cpu_address, 0, "buf_skinning resides in DEVICE_LOCAL VRAM (cpu_address == 0)")
    ctx.assert_equal(ren.buf_hdr_out.cpu_address, 0, "buf_hdr_out resides in DEVICE_LOCAL VRAM (cpu_address == 0)")
    ctx.assert_equal(ren.buf_hdr_out.vk_suballoc_block, -2, "buf_hdr_out bound to aliased transient heap (-2)")

    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False
    var mb_s = MeshBuilder()
    mb_s.add_ellipsoid(Vec3(0, 0, 0), Vec3(1, 1, 1), 8, 8)
    _ = ren.add_mesh(mb_s^.finish(), 0)

    var mb_sk = MeshBuilder()
    mb_sk.add_beveled_box(Vec3(0, 0, 0), Vec3(1, 1, 1), 0.1)
    var sk_m = mb_sk^.finish()
    var j_pack = List[UInt32](length=sk_m.vertex_count(), fill=0)
    var j_w = List[Vec3](length=sk_m.vertex_count(), fill=Vec3(1.0, 0.0, 0.0))
    var sk_entry = ren.add_skinned_mesh(sk_m, j_pack, j_w, 0)

    # Read back DEVICE_LOCAL buf_skinning via HOST_CACHED staging copy (zero raw .cpu_address dereference!)
    var skin_off_bytes = (AAA_SKIN_PREV_POS_BASE + sk_entry.skin_bind_first_slot * 3) * 4
    var skin_floats = _readback_dev_floats(ren, 0, skin_off_bytes, 4)
    var at_bind0_x = skin_floats[0]

    # Frame 1: stationary skinned mesh added after static mesh must produce zero phantom motion vectors
    ren.begin_frame(Vec3(1.5, 0.5, -3.5), Vec3(1.5, 0.5, 0.0))
    ren.svgf_passes = 0
    ren.draw_instance(sk_entry.mesh_entry.mesh_id, trs(Vec3(1.5, 0.5, 0.0), Quat(), Vec3(1, 1, 1)))
    ctx.assert_true(ren.render_frame(capture_readback=True), "frame 1 stationary skinned render succeeds")
    var hdr1 = _readback_dev_floats(ren, 1, 0, 64 * 64 * 8)
    var pix_base = (32 * 64 + 32) * 8
    var linear_z1 = hdr1[pix_base + 3]
    var vel_u1 = hdr1[pix_base + 4]
    var vel_v1 = hdr1[pix_base + 5]
    var resolved1 = ren.readback_resolved_hdr()

    # Frame 2: real skeletal displacement via defer_to_frame=True must produce non-zero GPU motion vectors
    var id_bones = List[DualQuat]()
    id_bones.append(DualQuat())
    var moved_bones = List[DualQuat]()
    moved_bones.append(DualQuat.from_quat_trans(Quat(), Vec3(0.4, 0.0, 0.0)))
    _ = ren.update_skinned_mesh_bones(sk_entry, moved_bones, id_bones, invalidate_vsm=False, defer_to_frame=True)
    ren.begin_frame(Vec3(1.5, 0.5, -3.5), Vec3(1.5, 0.5, 0.0))
    ren.draw_instance(sk_entry.mesh_entry.mesh_id, trs(Vec3(1.5, 0.5, 0.0), Quat(), Vec3(1, 1, 1)))
    ctx.assert_true(ren.render_frame(capture_readback=True), "frame 2 displaced skinned render succeeds")
    var hdr2 = _readback_dev_floats(ren, 1, 0, 64 * 64 * 8)
    var max_vel = Float32(0.0)
    for pi in range(64 * 64):
        var pb = pi * 8
        if hdr2[pb + 3] < Float32(100.0):
            var vm = abs(hdr2[pb + 4]) + abs(hdr2[pb + 5])
            if vm > max_vel:
                max_vel = vm
    ren.shutdown()

    ctx.assert_approx(at_bind0_x, sk_m.positions[0], 0.001, "skinned prev_pos initialized at skin_bind_first_slot * 3")
    ctx.assert_true(linear_z1 < Float32(100.0), "center pixel hits stationary skinned mesh")
    ctx.assert_true(abs(vel_u1) < Float32(1e-3) and abs(vel_v1) < Float32(1e-3), "stationary skinned mesh after static mesh has zero phantom velocity")
    ctx.assert_equal(len(resolved1), 64 * 64, "readback_resolved_hdr returns full 64x64 buffer")
    ctx.assert_true(max_vel > Float32(1e-4), "multi-frame skeletal bone displacement emits non-zero GPU motion vectors")


def test_t5_render_graph_budget_and_multireader_barriers(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r0 = rg.add_resource("big_128k", 128 * 1024, transient=True)
    var acc = List[RgResourceAccess](); acc.append(RgResourceAccess(r0, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("p0", acc)
    ctx.assert_false(rg.compile(32 * 1024), "32 KiB heap budget must reject 128 KiB transient resource")

    var rg2 = AaaRenderGraph()
    var rs = rg2.add_resource("shared", 4096, transient=True)
    var a0 = List[RgResourceAccess](); a0.append(RgResourceAccess(rs, RG_USAGE_COMPUTE_WRITE))
    _ = rg2.add_pass("w", a0, RG_QUEUE_GRAPHICS)
    var a1 = List[RgResourceAccess](); a1.append(RgResourceAccess(rs, RG_USAGE_COMPUTE_READ))
    _ = rg2.add_pass("r_async", a1, RG_QUEUE_ASYNC_COMPUTE)
    var a2 = List[RgResourceAccess](); a2.append(RgResourceAccess(rs, RG_USAGE_FRAGMENT_READ))
    _ = rg2.add_pass("r_frag", a2, RG_QUEUE_GRAPHICS)
    _ = rg2.compile()
    var has_p2_barrier = False
    for b in rg2.barriers:
        if b.pass_index == 2: has_p2_barrier = True
    ctx.assert_true(has_p2_barrier, "multi-reader pass on different stage/queue requires barrier")


def test_t5_restir_rand1_and_vsm_oversubscription(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    _ = res.update(UInt32(42), Float32(10.0), Float32(5.0), Float32(1.0))
    ctx.assert_equal(Int(res.light_id), 42, "first valid sample with rand_01=1.0 must select candidate 42")

    var res_empty = RestirReservoir()
    var merged = res_empty.merge(res, Float32(10.0), Float32(1.0))
    ctx.assert_true(merged and Int(res_empty.light_id) == 42, "merge into empty reservoir with rand_01=1.0 must select candidate")

    var vsm = VsmPageManager()
    var tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
    for i in range(1100): tbl[VSM_VIRTUAL_PAGE_COUNT + i] = 1
    var allocs = vsm.process_requests(tbl)
    ctx.assert_equal(len(allocs), 1024, "VSM caps single-frame allocation at 1024 physical tiles")
    ctx.assert_equal(Int(tbl[1024]), 0, "unallocated 1025th virtual page remains unmapped")


def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T5_Render_Assets")
    try: test_t5_cooked_model_preserves_extensions(ctx); ctx.record_pass("test_t5_cooked_model_preserves_extensions") except e: ctx.record_fail("test_t5_cooked_model_preserves_extensions", String(e))
    try: test_t5_glb_writer_roundtrips_extensions(ctx); ctx.record_pass("test_t5_glb_writer_roundtrips_extensions") except e: ctx.record_fail("test_t5_glb_writer_roundtrips_extensions", String(e))
    try: test_t5_glb_unaligned_accessor_and_extensions(ctx); ctx.record_pass("test_t5_glb_unaligned_accessor_and_extensions") except e: ctx.record_fail("test_t5_glb_unaligned_accessor_and_extensions", String(e))
    try: test_t5_vfs_mmap_alignment_and_corrupt_reject(ctx); ctx.record_pass("test_t5_vfs_mmap_alignment_and_corrupt_reject") except e: ctx.record_fail("test_t5_vfs_mmap_alignment_and_corrupt_reject", String(e))
    try: test_t5_qem_preserves_extensions_and_bc_non4x4(ctx); ctx.record_pass("test_t5_qem_preserves_extensions_and_bc_non4x4") except e: ctx.record_fail("test_t5_qem_preserves_extensions_and_bc_non4x4", String(e))
    try: test_t5_ui_overlay_shader_and_glyph_render(ctx); ctx.record_pass("test_t5_ui_overlay_shader_and_glyph_render") except e: ctx.record_fail("test_t5_ui_overlay_shader_and_glyph_render", String(e))
    try: test_t5_ui_scratch_overflow_guard(ctx); ctx.record_pass("test_t5_ui_scratch_overflow_guard") except e: ctx.record_fail("test_t5_ui_scratch_overflow_guard", String(e))
    try: test_t5_renderer_1024_instances_and_materials(ctx); ctx.record_pass("test_t5_renderer_1024_instances_and_materials") except e: ctx.record_fail("test_t5_renderer_1024_instances_and_materials", String(e))
    try: test_t5_skinned_prev_pos_offset(ctx); ctx.record_pass("test_t5_skinned_prev_pos_offset") except e: ctx.record_fail("test_t5_skinned_prev_pos_offset", String(e))
    try: test_t5_render_graph_budget_and_multireader_barriers(ctx); ctx.record_pass("test_t5_render_graph_budget_and_multireader_barriers") except e: ctx.record_fail("test_t5_render_graph_budget_and_multireader_barriers", String(e))
    try: test_t5_restir_rand1_and_vsm_oversubscription(ctx); ctx.record_pass("test_t5_restir_rand1_and_vsm_oversubscription") except e: ctx.record_fail("test_t5_restir_rand1_and_vsm_oversubscription", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t5_render_assets failed")
