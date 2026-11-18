"""Tier 5 Adversarial Coverage Hardening: Render & Asset Subsystems."""
from std.collections import List
from std.math import abs
from std.memory import bitcast
from flint.core.math import Vec3
from flint.core.transform import Quat, Mat4, trs
from flint.platform.os import write_bytes, remove_file
from flint.assets.model import NativeModel, NativeMesh, NativeNode
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.cooked_model import model_bytes, read_model, model_difference
from flint.assets.glb_writer import encode_native_glb, append_u32_le, append_f32_le
from flint.assets.glb_native import parse_native_glb
from flint.assets.vfs import encode_pack, pack_directory, FlpkMappedArchive
from flint.assets.meshlet import simplify_mesh_qem
from flint.assets.bc_compress import encode_bc7_rgba, decode_bc7_rgba, encode_bc6h_hdr, decode_bc6h_hdr, compute_argb_psnr_db
from flint.render.spirv_validate import validate_spirv_module
from flint.render.ui_overlay import aaa_ui_overlay_shader
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_clustered_lights import RestirReservoir
from flint.render.aaa_render_graph import (
    AaaRenderGraph, RgResourceAccess,
    RG_QUEUE_GRAPHICS, RG_QUEUE_ASYNC_COMPUTE,
    RG_USAGE_COMPUTE_WRITE, RG_USAGE_COMPUTE_READ, RG_USAGE_FRAGMENT_READ,
)
from flint.render.aaa_vsm import VsmPageManager, VSM_TABLE_WORDS, VSM_VIRTUAL_PAGE_COUNT
from flint.render.aaa_skinning_rt import AAA_SKIN_PREV_POS_BASE
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.vulkan_shaders import (
    raster_shader, hiz_pyramid_shader, gpu_cull_shader,
    tsr_upscale_shader, froxel_integrate_shader, sky_lut_shader,
)
from e2e.common.harness import TestContext


def test_t5_cooked_model_preserves_extensions(mut ctx: TestContext) raises:
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
    m1.meshes.append(mesh^)

    var m2 = NativeModel()
    var mb2 = MeshBuilder(base_color=0xFFAABBCC, roughness=0.3, metallic=0.8)
    mb2.add_beveled_box(Vec3(-1, -1, -1), Vec3(1, 1, 1), 0.1)
    m2.meshes.append(mb2^.finish())

    ctx.assert_true(model_difference(m1, m2).byte_length() > 0, "model_difference must detect glTF extension differences")
    var err = String("")
    var loaded = read_model(model_bytes(m1), err)
    ctx.assert_equal(err.byte_length(), 0, "read_model error-free")
    ctx.assert_approx(loaded.meshes[0].clearcoat, 0.85, 0.001, "cooked model preserves clearcoat")
    ctx.assert_approx(loaded.meshes[0].clearcoat_roughness, 0.12, 0.001, "cooked model preserves clearcoat_roughness")
    ctx.assert_approx(loaded.meshes[0].specular_factor, 0.25, 0.001, "cooked model preserves specular_factor")
    ctx.assert_approx(loaded.meshes[0].iridescence, 0.90, 0.001, "cooked model preserves iridescence")
    ctx.assert_approx(loaded.meshes[0].iridescence_thickness_nm, 420.0, 0.01, "cooked model preserves iridescence_thickness_nm")
    ctx.assert_approx(loaded.meshes[0].iridescence_ior, 1.45, 0.001, "cooked model preserves iridescence_ior")
    ctx.assert_approx(loaded.meshes[0].sheen_color.x, 0.4, 0.001, "cooked model preserves sheen_color")
    ctx.assert_approx(loaded.meshes[0].sheen_roughness, 0.35, 0.001, "cooked model preserves sheen_roughness")


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
    var names = List[String]()
    var files = List[List[UInt8]]()
    names.append("empty.bin"); files.append(List[UInt8]())
    names.append("odd_name_123.txt")
    var f1 = List[UInt8]()
    for i in range(13): f1.append(UInt8(65 + i))
    files.append(f1^)
    names.append("tail.bin")
    var f2 = List[UInt8]()
    for i in range(70): f2.append(UInt8(i))
    files.append(f2^)

    var pack = encode_pack(names, files)
    var p_path = String("/tmp/flint_t5_vfs_test.flpk")
    var bad_path = String("/tmp/flint_t5_vfs_bad.flpk")
    _ = write_bytes(p_path, pack)
    var entries = pack_directory(p_path)
    ctx.assert_equal(len(entries), 3, "3 entries in directory")
    for i in range(len(entries)):
        ctx.assert_equal(entries[i].offset % 64, 0, "all flpk v2 entries 64-byte aligned")

    var arch = FlpkMappedArchive.open(p_path)
    ctx.assert_true(arch.addr != 0, "mmap succeeded")
    ctx.assert_equal(len(arch.read_entry("empty.bin")), 0, "0-byte payload reads cleanly")
    ctx.assert_equal(len(arch.read_entry("odd_name_123.txt")), 13, "13-byte payload reads cleanly")
    ctx.assert_equal(len(arch.read_entry("tail.bin")), 70, "70-byte payload reads cleanly")
    arch.close()

    # Corrupt negative size in entry 1: mmap parser must reject corrupt entries
    var bad = pack.copy()
    var e1_off_sz = 16 + (20 + 9) + (20 + 16) - 4
    bad[e1_off_sz + 0] = 0xFF; bad[e1_off_sz + 1] = 0xFF
    bad[e1_off_sz + 2] = 0xFF; bad[e1_off_sz + 3] = 0xFF
    _ = write_bytes(bad_path, bad)
    var arch_bad = FlpkMappedArchive.open(bad_path)
    var bad_read = arch_bad.read_entry("odd_name_123.txt")
    var bad_count = len(arch_bad.entries)
    arch_bad.close()

    # Craft 64-bit signed integer overflow in entry 0: offset = 0x7FFFFFFFFFFFF000, size = 0x2000
    var ovf = pack.copy()
    var e0_off_at = 16 + 4 + 9  # header(16) + name_len(4) + len("empty.bin")(9)
    var ovf_off = UInt64(0x7FFFFFFFFFFFF000)
    var ovf_sz = UInt64(0x2000)
    for b_i in range(8):
        ovf[e0_off_at + b_i] = UInt8((ovf_off >> UInt64(b_i * 8)) & 0xFF)
        ovf[e0_off_at + 8 + b_i] = UInt8((ovf_sz >> UInt64(b_i * 8)) & 0xFF)
    var ovf_path = String("/tmp/flint_t5_vfs_ovf.flpk")
    _ = write_bytes(ovf_path, ovf)
    var ovf_dir = pack_directory(ovf_path)
    var arch_ovf = FlpkMappedArchive.open(ovf_path)
    var ovf_mmap_cnt = len(arch_ovf.entries)
    arch_ovf.close()

    _ = remove_file(p_path)
    _ = remove_file(bad_path)
    _ = remove_file(ovf_path)
    ctx.assert_equal(bad_count, 0, "corrupt directory rejected entirely")
    ctx.assert_equal(len(bad_read), 0, "corrupt negative-size entry must be rejected safely")
    ctx.assert_equal(len(ovf_dir), 0, "pack_directory must reject 64-bit signed offset+size overflow")
    ctx.assert_equal(ovf_mmap_cnt, 0, "FlpkMappedArchive must reject 64-bit signed offset+size overflow")


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
    var rep = validate_spirv_module(aaa_ui_overlay_shader())
    ctx.assert_true(rep.valid, "aaa_ui_overlay_shader passes SPIR-V SSA validator")
    ctx.assert_true(validate_spirv_module(raster_shader()).valid, "raster_shader validates")
    ctx.assert_true(validate_spirv_module(hiz_pyramid_shader()).valid, "hiz_pyramid_shader validates")
    ctx.assert_true(validate_spirv_module(gpu_cull_shader()).valid, "gpu_cull_shader validates")
    ctx.assert_true(validate_spirv_module(tsr_upscale_shader()).valid, "tsr_upscale_shader validates")
    ctx.assert_true(validate_spirv_module(froxel_integrate_shader()).valid, "froxel_integrate_shader validates")
    ctx.assert_true(validate_spirv_module(sky_lut_shader()).valid, "sky_lut_shader validates")

    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.begin_frame(Vec3(0, 0, -5), Vec3(0, 0, 0))
    ren.ui_draw_list.add_rect(0.0, 0.0, 160.0, 120.0, 0xFF000000, 0.0)
    if ren.ui_font.valid:
        ren.ui_draw_list.add_text(ren.ui_font_atlas_slot, ren.ui_font, "H", 20.0, 60.0, 32.0, 0xFFFFFFFF)
    _ = ren.render_frame(capture_readback=True)
    var px = ren.readback_pixels()
    ren.shutdown()

    # Check that glyph bbox contains BOTH lit and unlit pixels (actual glyph silhouette, not solid block)
    var lit = 0; var dark = 0
    for y in range(30, 62):
        for x in range(20, 44):
            if (px[y * 160 + x] & 0xFF) > 128: lit += 1
            elif (px[y * 160 + x] & 0xFF) < 20: dark += 1
    ctx.assert_true(lit > 20 and dark > 20, "UI text glyph 'H' renders distinct foreground and background pixels")


def test_t5_ui_scratch_overflow_guard(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=16, height=16)
    ren.begin_frame()
    for _ in range(500):
        ren.ui_draw_list.add_rect(0.0, 0.0, 10.0, 10.0, 0xFF00FF00)
    var ok = ren.render_frame(capture_readback=True)
    ren.shutdown()
    ctx.assert_true(ok, "16x16 render_frame with 500 UI prims must not overflow buf_post_scratch")


def test_t5_renderer_1024_instances_and_materials(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=192, height=128, max_instances=2048)
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
    _ = ren.render_frame(capture_readback=True)
    var center_px = ren.readback_pixels()[64 * 192 + 96]

    # Verify 1024-material SSBO bounds check
    while len(ren.materials) < 1024:
        _ = ren.add_material(AaaMaterial())
    var overflow_id = ren.add_material(AaaMaterial())
    var mat_count = len(ren.materials)
    ren.shutdown()

    var ren_cap = FlintAaaVulkanRenderer(width=64, height=64, max_instances=65536)
    var capped_max_inst = ren_cap.max_instances
    ren_cap.shutdown()

    ctx.assert_equal(capped_max_inst, 32768, "max_instances must clamp to 15-bit VisBuffer limit 32768")
    ctx.assert_true(Int((center_px >> 16) & 255) > 240 and Int(center_px & 255) < 80, "instance 1024 must render red without VisBuffer wraparound")
    ctx.assert_equal(overflow_id, -1, "1025th material must be rejected cleanly")
    ctx.assert_equal(mat_count, 1024, "material list capped at 1024")


def test_t5_skinned_prev_pos_offset(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=64, height=64)
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
    var ptr_skin = Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=ren.buf_skinning.cpu_address)
    var at_bind0_x = bitcast[DType.float32](ptr_skin[unsafe_offset=AAA_SKIN_PREV_POS_BASE + sk_entry.skin_bind_first_slot * 3])

    # Stationary skinned mesh added after static mesh must produce zero phantom motion vectors
    ren.begin_frame(Vec3(1.5, 0.5, -3.5), Vec3(1.5, 0.5, 0.0))
    ren.svgf_passes = 0
    ren.draw_instance(sk_entry.mesh_entry.mesh_id, trs(Vec3(1.5, 0.5, 0.0), Quat(), Vec3(1, 1, 1)))
    _ = ren.render_frame(capture_readback=True)
    var hdr_ptr = Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=ren.buf_hdr_out.cpu_address)
    var pix_base = (32 * 64 + 32) * 8
    var linear_z = bitcast[DType.float32](hdr_ptr[unsafe_offset=pix_base + 3])
    var vel_u = bitcast[DType.float32](hdr_ptr[unsafe_offset=pix_base + 4])
    var vel_v = bitcast[DType.float32](hdr_ptr[unsafe_offset=pix_base + 5])
    ren.shutdown()
    ctx.assert_approx(at_bind0_x, sk_m.positions[0], 0.001, "skinned prev_pos initialized at skin_bind_first_slot * 3")
    ctx.assert_true(linear_z < Float32(100.0), "center pixel hits stationary skinned mesh")
    ctx.assert_true(abs(vel_u) < Float32(1e-3) and abs(vel_v) < Float32(1e-3), "stationary skinned mesh after static mesh must have zero phantom velocity")


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
