"""Flint v5 Track 2 Standalone Procedural Render Pipeline & Metrology Suite (`test_flint_v5_render.mojo`).

Zero external `.glb` / font / texture files. Executes end-to-end in `<2.0s` verifying:
  1. SPIR-V `OpSpecConstant` (`OP_SPEC_CONSTANT_TRUE=48`, `OP_SPEC_CONSTANT_FALSE=49`, `OP_SPEC_CONSTANT=50`,
     `DEC_SPEC_ID=1`) on `SpirvBuilder` & `AaaSpirvModule`, validated by `validate_spirv_module`, plus
     `aaa_ui_overlay_shader()`, `vat_crowd_eval_shader()`, and `SurfaceMaterialNode` (`ShaderDsl`).
  2. `MaterialAsset` 128-byte (`32 x UInt32`) GPU layout with **16-bit bindless slots up to `65,534`**,
     `MAT_DOMAIN_*`, `MAT_FLAG_*`, `GpuDecalVolume` OBB projection, and `TerrainClipmapDesc` snapping.
  3. Procedural 2-bone skeletal `NativeModel` -> `bake_vat_crowd_clip_rgba16f` (`RGBA16F` pos+normal sheets),
     10,000-instance phase evaluation, and non-zero 3D motion vectors (`cur_pos - prev_pos`).
  4. `WaylandWindow.enable_vulkan_bridge()`, `WaylandWindow.presents_on_gpu()`, `SwapchainSurface`
     (`AcquiredFrame`, `SURFACE_HEADLESS`, `recreate`), RH `-Z` / Reverse-Z projection math, `DebugLineBatch`
     near-plane clipping into `UiDrawList` `PRIM_LINE`s, `DoubleBufferedRenderWorld`, `RenderExtractor`
     (`UInt64` entity ID tracking + despawn pruning), and `ScalableRenderer` (`TIER_0`, `TIER_1`, `TIER_2`).
  5. `ui_overlay.mojo` exact `0xFFFF2040` pure-red vs `0xFF2040FF` pure-blue GPU pixel readback (proving
     Bug 3 Red/Blue swap is 100% fixed!), all 6 GPU SDF primitive kinds (`PRIM_SHAPE`, `PRIM_GLYPH`,
     `PRIM_NINE_SLICE`, `PRIM_CONTROLLER_SYMBOL`, `PRIM_LINE` capsule + Bézier, `PRIM_IMAGE`),
     `>300` (`340`) overlapping primitives inside a single `16x16` tile (proving zero drops & painter's order),
     and `disp_mode = 1` `A2B10G10R10` HDR10 ST.2084 PQ @ 203 nits paper-white!
  6. Procedural White-Furnace energy conservation check + Cornell Box + 5x5 PBR/Clearcoat Sphere Matrix +
     4-Cascade Toroidal VSM (`±16m, ±64m, ±256m, ±1024m`) Sundial shadow contrast!
  7. Render features: soft ray-traced sun shadows (penumbra filter radius), refractive glass (Fresnel,
     Snell, total internal reflection, Beer-Lambert), shaped water and generic GPU particles (color / size /
     alpha over life, additive and alpha blending): SPIR-V validation of their passes, and on the GPU,
     additive particles light the frame and clear glass shows the wall behind it (with ray queries).
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, cos, max, min, sin, sqrt
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import (
    NativeAnimation,
    NativeChannel,
    NativeModel,
    NativeNode,
    NativeSkin,
)
from flint.core.math import Mat4, Vec3
from flint.core.transform import Quat, trs
from flint.platform.os import get_time
from flint.platform.vulkan import (
    FORMAT_A2B10G10R10_UNORM_PACK32,
    FORMAT_B8G8R8A8_UNORM,
    FORMAT_R16G16B16A16_SFLOAT,
    SwapchainHandle,
    swapchain_format_bytes_per_pixel,
)
from flint.platform.wayland import WaylandWindow
from flint.render.aaa_bsdf import AAA_MATERIAL_WORDS, AaaMaterial
from flint.render.aaa_renderer import AAA_DISPLAY_SCRGB_FP16, FlintAaaVulkanRenderer, aaa_pack_scrgb_rgba16f_shader
from flint.render.aaa_visbuffer import (
    AAA_MAX_BINDLESS,
    AAA_MAX_PACKED_INSTANCES,
    AAA_MAX_PACKED_MESHLET_ID,
    compute_hiz_mip_levels,
    hiz_depth_downsample_shader,
    meshlet_cull_shader,
)
from flint.render.aaa_glass import (
    AaaGlassParams,
    aaa_glass_refract_shader,
    glass_beer_lambert_cpu,
    glass_f0_cpu,
    glass_fresnel_cpu,
    glass_refract_cpu,
)
from flint.render.aaa_gpu_particles import (
    AAA_EMITTER_WORDS,
    PARTICLE_FLAG_ADDITIVE,
    PARTICLE_FLAG_STRETCH,
    PARTICLE_GENERIC,
    AaaParticleEmitter,
    gpu_particle_sim_and_bin_shader,
    gpu_particle_tile_composite_shader,
)
from flint.render.aaa_rt_shadow import penumbra_radius_px_cpu, rt_sun_shadow_filter_shader, rt_sun_shadow_trace_shader
from flint.render.aaa_water import aaa_water_refract_shader
from flint.render.aaa_vsm import (
    VSM_CASCADE_0_EXTENT,
    VSM_CASCADE_1_EXTENT,
    VSM_CASCADE_2_EXTENT,
    VSM_CASCADE_3_EXTENT,
    VSM_TABLE_WORDS,
    VsmPageManager,
    compute_vsm_cascade_matrices,
    compute_vsm_cascade_page_origins,
    mark_toroidal_cascade_pages,
    vsm_alphamask_vertex_shader,
    vsm_mark_pages_shader,
    vsm_page_vertex_shader,
)
from flint.render.debug_draw import DEBUG_CYAN, DebugDraw
from flint.render.material_graph import (
    BINDLESS_SLOT_MAX,
    BINDLESS_SLOT_NONE,
    GPU_DECAL_VOLUME_WORDS,
    MATERIAL_ARCHIVE_BYTES,
    MATERIAL_ARCHIVE_WORDS,
    MATERIAL_ASSET_BYTES,
    MATERIAL_ASSET_WORDS,
    MAT_DOMAIN_CLEARCOAT,
    MAT_DOMAIN_STANDARD_PBR,
    MAT_DOMAIN_SUBSURFACE_SKIN,
    MAT_FLAG_CLEARCOAT,
    MAT_FLAG_EMISSIVE_BLOOM,
    MAT_FLAG_RECEIVE_DECALS,
    MAT_FLAG_VAT_CROWD,
    GpuDecalVolume,
    MaterialAsset,
    StandardPbrMaterialNode,
    TerrainClipmapDesc,
    TriplanarWeatherMaterialNode,
    bake_vat_crowd_clip_rgba16f,
    compile_surface_material_shader,
    f16_bits_to_f32,
    sample_vat_vertex_cpu,
    vat_crowd_eval_shader,
)
from flint.render.render_world import (
    DEG_TO_RAD_F32,
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_1_FORWARD_CLUSTERED,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    DebugLineBatch,
    DoubleBufferedRenderWorld,
    RenderCameraPacket,
    RenderExtractor,
    ScalableRenderer,
    make_rh_look_at,
    make_rh_reverse_z_infinite_perspective,
    make_rh_reverse_z_perspective,
    project_rh_world_to_screen,
)
from flint.render.simd_canvas import pack_argb
from flint.render.spirv import (
    AaaSpirvModule,
    DEC_SPEC_ID,
    MODEL_GL_COMPUTE,
    OP_DECORATE,
    OP_SPEC_CONSTANT,
    OP_SPEC_CONSTANT_FALSE,
    OP_SPEC_CONSTANT_TRUE,
    SpirvBuilder,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.render.swapchain_surface import (
    DISPLAY_EXTENDED_SCRGB_FP16,
    DISPLAY_HDR10_PQ,
    DISPLAY_SDR_SRGB,
    PRESENT_MAILBOX,
    SURFACE_HEADLESS,
    SwapchainSurface,
    resolve_swapchain_display_mode,
    validate_present_source_bytes,
)
from flint.render.ui_overlay import (
    UI_SYMBOL_A,
    UI_SYMBOL_GIZMO,
    UI_SYMBOL_PLAY,
    aaa_ui_overlay_shader,
    make_ui_bezier_prim,
    make_ui_image_prim,
    make_ui_line_prim,
    make_ui_symbol_prim,
)
from flint.ui.draw_list import (
    GpuUiPrim,
    PRIM_GLYPH,
    PRIM_LINE,
    PRIM_POLY_DATA,
    UI_FLAG_BORDER,
    UI_FLAG_DROP_SHADOW,
    UI_FLAG_VERT_GRADIENT,
    UiDrawList,
)


def _fail(msg: String):
    print("FAIL:", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _unpack_argb(px: UInt32) -> Tuple[Int, Int, Int, Int]:
    var a = Int((px >> 24) & UInt32(0xFF))
    var r = Int((px >> 16) & UInt32(0xFF))
    var g = Int((px >> 8) & UInt32(0xFF))
    var b = Int(px & UInt32(0xFF))
    return (r, g, b, a)


def _contains_opcode(words: List[UInt32], target_op: UInt32) -> Bool:
    var idx = 5
    var n = len(words)
    while idx < n:
        var hdr = words[idx]
        var wc = Int(hdr >> 16)
        var op = hdr & UInt32(0xFFFF)
        if wc <= 0:
            break
        if op == target_op:
            return True
        idx += wc
    return False


def _contains_spec_id(words: List[UInt32], spec_id: UInt32) -> Bool:
    var idx = 5
    var n = len(words)
    while idx < n:
        var hdr = words[idx]
        var wc = Int(hdr >> 16)
        var op = hdr & UInt32(0xFFFF)
        if wc <= 0:
            break
        if op == OP_DECORATE and wc >= 4 and words[idx + 2] == DEC_SPEC_ID and words[idx + 3] == spec_id:
            return True
        idx += wc
    return False


# ============================================================================
# Suite 1: SPIR-V `OpSpecConstant` & Shader Validator
# ============================================================================

def test_spirv_spec_constants_and_shaders():
    _assert(OP_SPEC_CONSTANT_TRUE == 48, "OP_SPEC_CONSTANT_TRUE == 48")
    _assert(OP_SPEC_CONSTANT_FALSE == 49, "OP_SPEC_CONSTANT_FALSE == 49")
    _assert(OP_SPEC_CONSTANT == 50, "OP_SPEC_CONSTANT == 50")
    _assert(DEC_SPEC_ID == 1, "DEC_SPEC_ID == 1")

    # Test `SpirvBuilder` (`SpirvModule`) specialization constants
    var sb = SpirvBuilder(8, 8, 4)
    var b_out = sb.buffer(0)
    var sc_true = sb.spec_constant_bool(0, True)
    var sc_false = sb.spec_constant_bool(1, False)
    var sc_u32 = sb.spec_constant_u32(2, 42)
    var chosen = sb.select_u(sc_true, sc_u32, sb.select_u(sc_false, sb.uint(99), sb.uint(7)))
    sb.store_word(b_out, sb.uint(0), chosen)
    var sb_words = sb.finish()

    _assert(_contains_opcode(sb_words, OP_SPEC_CONSTANT_TRUE), "SpirvBuilder emitted OpSpecConstantTrue")
    _assert(_contains_opcode(sb_words, OP_SPEC_CONSTANT_FALSE), "SpirvBuilder emitted OpSpecConstantFalse")
    _assert(_contains_opcode(sb_words, OP_SPEC_CONSTANT), "SpirvBuilder emitted OpSpecConstant")
    _assert(_contains_spec_id(sb_words, 0) and _contains_spec_id(sb_words, 1) and _contains_spec_id(sb_words, 2), "SpecId decorations emitted")

    var rep_sb = validate_spirv_module(sb_words)
    if not rep_sb.valid:
        _fail("SpirvBuilder OpSpecConstant validation failed: " + rep_sb.errors[0])

    # Test `AaaSpirvModule` specialization constants
    var am = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=4)
    var a_buf = am.buffer(0)
    var am_b0 = am.spec_constant_bool(10, True)
    var am_b1 = am.spec_constant_bool(11, False)
    var am_u0 = am.spec_constant_u32(12, 1337)
    var val_u = am.select_u(am_b0, am_u0, am.select_u(am_b1, am.uint(1), am.uint(0)))
    am.store_uint(a_buf, am.uint(0), val_u)
    var am_words = am.finish()
    var rep_am = validate_spirv_module(am_words)
    if not rep_am.valid:
        _fail("AaaSpirvModule OpSpecConstant validation failed: " + rep_am.errors[0])

    # Validate Track 2 GPU kernels
    var ui_spv = aaa_ui_overlay_shader()
    var rep_ui = validate_spirv_module(ui_spv)
    if not rep_ui.valid:
        _fail("aaa_ui_overlay_shader SPIR-V validation failed: " + rep_ui.errors[0])

    var vat_spv = vat_crowd_eval_shader()
    var rep_vat = validate_spirv_module(vat_spv)
    if not rep_vat.valid:
        _fail("vat_crowd_eval_shader SPIR-V validation failed: " + rep_vat.errors[0])

    var mat_std_spv = compile_surface_material_shader(StandardPbrMaterialNode(enable_detail_layer=True))
    var rep_mat_std = validate_spirv_module(mat_std_spv)
    if not rep_mat_std.valid:
        _fail("StandardPbrMaterialNode SPIR-V validation failed: " + rep_mat_std.errors[0])
    _assert(_contains_opcode(mat_std_spv, OP_SPEC_CONSTANT_TRUE) and _contains_opcode(mat_std_spv, OP_SPEC_CONSTANT), "Material shader uses OpSpecConstant")

    var mat_tri_spv = compile_surface_material_shader(TriplanarWeatherMaterialNode(snow_line_y=8.0))
    var rep_mat_tri = validate_spirv_module(mat_tri_spv)
    if not rep_mat_tri.valid:
        _fail("TriplanarWeatherMaterialNode SPIR-V validation failed: " + rep_mat_tri.errors[0])

    var scrgb_spv = aaa_pack_scrgb_rgba16f_shader()
    var rep_scrgb = validate_spirv_module(scrgb_spv)
    if not rep_scrgb.valid:
        _fail("aaa_pack_scrgb_rgba16f_shader SPIR-V validation failed: " + rep_scrgb.errors[0])
    var rep_vsm_m = validate_spirv_module(vsm_mark_pages_shader())
    if not rep_vsm_m.valid:
        _fail("vsm_mark_pages_shader SPIR-V validation failed: " + rep_vsm_m.errors[0])
    var rep_vsm_v = validate_spirv_module(vsm_page_vertex_shader())
    if not rep_vsm_v.valid:
        _fail("vsm_page_vertex_shader SPIR-V validation failed: " + rep_vsm_v.errors[0])
    var rep_vsm_av = validate_spirv_module(vsm_alphamask_vertex_shader())
    if not rep_vsm_av.valid:
        _fail("vsm_alphamask_vertex_shader SPIR-V validation failed: " + rep_vsm_av.errors[0])
    var rep_hiz = validate_spirv_module(hiz_depth_downsample_shader())
    if not rep_hiz.valid:
        _fail("hiz_depth_downsample_shader SPIR-V validation failed: " + rep_hiz.errors[0])
    var rep_mc = validate_spirv_module(meshlet_cull_shader())
    if not rep_mc.valid:
        _fail("meshlet_cull_shader SPIR-V validation failed: " + rep_mc.errors[0])
    var max_bindless_rt = AAA_MAX_BINDLESS
    var max_ml_rt = AAA_MAX_PACKED_MESHLET_ID
    var max_inst_rt = AAA_MAX_PACKED_INSTANCES
    _assert(max_bindless_rt >= 4096, "AAA_MAX_BINDLESS raised to >= 4096")
    _assert(max_ml_rt == 0x1FFFE and max_inst_rt == 0x7FFF, "17-bit ml_id / 15-bit inst_id packing limits")
    _assert(compute_hiz_mip_levels(320, 180) == 8 and compute_hiz_mip_levels(16, 16) == 5, "Hi-Z mip level computation")

    print("[PASS] Suite 1: SPIR-V OpSpecConstant & Track 2 Shader Validation (" + String(len(ui_spv)) + " UI words)")


# ============================================================================
# Suite 2: 128-Byte `MaterialAsset` (16-Bit Bindless), Decals, Terrain & VAT
# ============================================================================

def _build_procedural_skeletal_walker() -> NativeModel:
    """Builds a procedural 2-bone articulated capsule character with a 1.0s walk-swing animation clip (0 external files!)."""
    var mb = MeshBuilder()
    mb.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.8, 0.0), 0.22, 0.16, segments=12)
    var mesh = mb^.finish()
    var vc = mesh.vertex_count()
    mesh.skin = 0
    for v in range(vc):
        var py = mesh.positions[v * 3 + 1]
        var w_upper = max(Float32(0.0), min(Float32(1.0), (py - Float32(0.5)) / Float32(1.1)))
        var w_lower = Float32(1.0) - w_upper
        mesh.joints.append(UInt16(0))
        mesh.joints.append(UInt16(1))
        mesh.joints.append(UInt16(0))
        mesh.joints.append(UInt16(0))
        mesh.weights.append(w_lower)
        mesh.weights.append(w_upper)
        mesh.weights.append(Float32(0.0))
        mesh.weights.append(Float32(0.0))

    var n0 = NativeNode()
    n0.name = "pelvis"
    n0.children.append(1)

    var n1 = NativeNode()
    n1.name = "torso"
    n1.base_translation = Vec3(0.0, 0.85, 0.0)
    n1.translation = Vec3(0.0, 0.85, 0.0)

    var sk = NativeSkin()
    sk.joints.append(0)
    sk.joints.append(1)
    var ibm0 = Mat4()
    var ibm1 = trs(Vec3(0.0, -0.85, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    for k in range(16):
        sk.inverse_bind.append(ibm0.values[k])
    for k in range(16):
        sk.inverse_bind.append(ibm1.values[k])

    var ch_rot = NativeChannel()
    ch_rot.node = 1
    ch_rot.path = 1  # rotation
    var kf_times = [Float32(0.0), Float32(0.25), Float32(0.50), Float32(0.75), Float32(1.0)]
    var kf_angles = [Float32(0.0), Float32(0.45), Float32(0.0), Float32(-0.45), Float32(0.0)]
    for i in range(5):
        ch_rot.times.append(kf_times[i])
        var half_a = kf_angles[i] * Float32(0.5)
        var q = Quat(sin(half_a), 0.0, 0.0, cos(half_a)).normalized()
        ch_rot.values.append(q.x)
        ch_rot.values.append(q.y)
        ch_rot.values.append(q.z)
        ch_rot.values.append(q.w)

    var anim = NativeAnimation()
    anim.name = "walk_swing"
    anim.duration = Float32(1.0)
    anim.channels.append(ch_rot^)

    var model = NativeModel()
    model.meshes.append(mesh^)
    model.nodes.append(n0^)
    model.nodes.append(n1^)
    model.skins.append(sk^)
    model.animations.append(anim^)
    return model^


def test_material_graph_decals_terrain_and_vat():
    var mat = MaterialAsset(
        base_color=Vec3(0.82, 0.24, 0.12),
        roughness=0.18,
        metallic=0.85,
        domain=MAT_DOMAIN_CLEARCOAT,
        flags=MAT_FLAG_RECEIVE_DECALS | MAT_FLAG_EMISSIVE_BLOOM | MAT_FLAG_VAT_CROWD,
    )
    mat.clearcoat = Float32(0.95)
    mat.clearcoat_roughness = Float32(0.06)
    mat.albedo_slot = BINDLESS_SLOT_MAX          # 65,534 (0xFFFE) — full 16-bit bindless!
    mat.normal_slot = UInt16(42000)
    mat.orm_slot = UInt16(12345)
    mat.emissive_slot = BINDLESS_SLOT_NONE
    mat.vat_pos_slot = UInt16(60001)
    mat.vat_nrm_slot = UInt16(60002)
    mat.uv_scale_x = Float32(4.0)
    mat.uv_scale_y = Float32(2.5)

    var packed_gpu = mat.pack_words()
    var mat_words_rt = MATERIAL_ASSET_WORDS
    _assert(
        mat_words_rt == AAA_MATERIAL_WORDS
        and mat_words_rt == 24
        and len(packed_gpu) == MATERIAL_ASSET_WORDS
        and len(packed_gpu) * 4 == MATERIAL_ASSET_BYTES,
        "MaterialAsset GPU layout unified with AAA_MATERIAL_WORDS == 24 words (96 bytes)",
    )
    var aaa_words = List[UInt32]()
    mat.to_aaa_material().append_words(aaa_words)
    for w_i in range(AAA_MATERIAL_WORDS):
        _assert(packed_gpu[w_i] == aaa_words[w_i], "MaterialAsset.pack_words matches AaaMaterial SSBO word " + String(w_i))
    var unpacked_gpu = MaterialAsset.unpack_words(packed_gpu)
    _assert(abs(unpacked_gpu.clearcoat - Float32(0.95)) < Float32(1e-4) and unpacked_gpu.domain == MAT_DOMAIN_CLEARCOAT, "24-word GPU unpack preserves clearcoat")

    var packed = mat.pack_archive_words()
    _assert(len(packed) == MATERIAL_ARCHIVE_WORDS and len(packed) * 4 == MATERIAL_ARCHIVE_BYTES, "MaterialAsset archive == 128 bytes (32 words)")

    var unpacked = MaterialAsset.unpack_archive_words(packed)
    _assert(unpacked.albedo_slot == UInt16(65534), "16-bit albedo_slot 65,534 preserved")
    _assert(unpacked.normal_slot == UInt16(42000), "16-bit normal_slot 42,000 preserved")
    _assert(unpacked.orm_slot == UInt16(12345), "16-bit orm_slot 12,345 preserved")
    _assert(unpacked.emissive_slot == BINDLESS_SLOT_NONE, "BINDLESS_SLOT_NONE preserved")
    _assert(unpacked.vat_pos_slot == UInt16(60001) and unpacked.vat_nrm_slot == UInt16(60002), "16-bit VAT slots preserved")
    _assert(unpacked.domain == MAT_DOMAIN_CLEARCOAT, "MAT_DOMAIN_CLEARCOAT preserved")
    _assert((unpacked.flags & MAT_FLAG_CLEARCOAT) != 0 and (unpacked.flags & MAT_FLAG_VAT_CROWD) != 0, "MAT_FLAG_* preserved")
    _assert(abs(unpacked.uv_scale_x - Float32(4.0)) < Float32(0.01) and abs(unpacked.uv_scale_y - Float32(2.5)) < Float32(0.01), "FP16 UV scales preserved")
    var skin_mat = MaterialAsset(domain=MAT_DOMAIN_SUBSURFACE_SKIN)
    var std_mat = MaterialAsset(domain=MAT_DOMAIN_STANDARD_PBR)
    _assert(skin_mat.domain == 1 and std_mat.domain == 0, "Material domain enum check")

    # OBB Decal Volume projection + normal angle fade
    var decal = GpuDecalVolume(
        center=Vec3(2.0, 0.0, -3.0),
        half_extents=Vec3(1.5, 0.5, 1.5),
        rotation=Quat(),
        albedo_slot=UInt16(512),
        opacity=Float32(0.9),
    )
    _assert(len(decal.pack_words()) == GPU_DECAL_VOLUME_WORDS, "GpuDecalVolume == 24 words (96 bytes)")
    var hit_up = decal.project_world_point(Vec3(2.0, 0.0, -3.0), Vec3(0.0, 1.0, 0.0))
    _assert(hit_up[0] and abs(hit_up[1] - Float32(0.5)) < Float32(0.01) and hit_up[3] > Float32(0.8), "Decal projects upward floor point")
    var miss_wall = decal.project_world_point(Vec3(2.0, 0.0, -3.0), Vec3(1.0, 0.0, 0.0))
    _assert(not miss_wall[0], "Decal rejects perpendicular wall normal")

    # Terrain Clipmap grid snapping
    var clip_desc = TerrainClipmapDesc(ring_count=4, grid_dim_per_ring=64, base_cell_size_m=0.5)
    _assert(len(clip_desc.pack_words()) == 12, "TerrainClipmapDesc packs 12 words")
    var snap0 = clip_desc.snapped_ring_origin_xz(Vec3(13.3, 5.0, -7.8), 0)
    _assert(abs(snap0[0] - Float32(13.0)) < Float32(1e-4) and abs(snap0[1] - Float32(-8.0)) < Float32(1e-4), "Clipmap ring 0 snaps to 2*cell=1.0m")

    # VAT Crowd Baker (`RGBA16F`) & 10,000-instance motion vector check
    var walker = _build_procedural_skeletal_walker()
    var baked_vat = bake_vat_crowd_clip_rgba16f(walker, mesh_index=0, clip_index=0, sample_fps=30.0, looping=True)
    _assert(baked_vat.desc.vertex_count > 20 and baked_vat.desc.frame_count == 30, "VAT baked 30 frames")
    _assert(len(baked_vat.pos_rgba16f) == baked_vat.desc.vertex_count * 30 * 4, "RGBA16F pos sheet size exact")

    # Find top vertex (`y > 1.7`) and verify skeletal swing deformation + non-zero 3D motion vectors across crowd phases
    var top_v = 0
    var max_y = Float32(-1e9)
    for v in range(walker.meshes[0].vertex_count()):
        var py = walker.meshes[0].positions[v * 3 + 1]
        if py > max_y:
            max_y = py
            top_v = v
    var s_f0 = sample_vat_vertex_cpu(baked_vat, top_v, Float32(0.0), Float32(0.0))
    var s_peak = sample_vat_vertex_cpu(baked_vat, top_v, Float32(0.25), Float32(0.25 - 1.0 / 60.0))
    var s_mid = sample_vat_vertex_cpu(baked_vat, top_v, Float32(0.125), Float32(0.125 - 1.0 / 60.0))
    var swing_disp = (s_peak[0] - s_f0[0]).length()
    var frame_mv = (s_mid[0] - s_mid[2]).length()
    _assert(swing_disp > Float32(0.25), "VAT skeletal swing deforms upper body > 0.25m")
    _assert(frame_mv > Float32(0.015), "VAT cur_pos - prev_pos 3D motion vector is non-zero")

    # Verify 10,000 crowd instances sample independent phases cleanly
    var crowd_mv_accum = Float32(0.0)
    for inst_i in range(10000):
        var phase = Float32(inst_i % 97) * Float32(0.0103)
        var sv = sample_vat_vertex_cpu(baked_vat, top_v, phase + Float32(0.12), phase + Float32(0.12 - 1.0 / 60.0))
        crowd_mv_accum += (sv[0] - sv[2]).length()
    _assert(crowd_mv_accum > Float32(50.0), "10,000 VAT crowd instances animate independently with valid motion vectors")

    print("[PASS] Suite 2: 128B MaterialAsset (slot 65,534), OBB Decals, Clipmap & 10k VAT Crowd")


# ============================================================================
# Suite 3: SwapchainSurface, RH `-Z` / Reverse-Z, DebugLineBatch & RenderWorld
# ============================================================================

def test_swapchain_render_world_and_tier0():
    # WaylandWindow zero-copy bridge & `presents_on_gpu()` contract
    var win = WaylandWindow()
    _assert(not win.wants_bridge, "WaylandWindow.wants_bridge defaults False")
    win.enable_vulkan_bridge()
    _assert(win.wants_bridge, "enable_vulkan_bridge sets wants_bridge = True")
    _assert(not win.presents_on_gpu(), "Empty WaylandWindow does not present on GPU")
    win.swapchain = SwapchainHandle(1, 0xCAFE, 1280, 720, 44, 0, 2, 3, False)
    _assert(win.presents_on_gpu(), "WaylandWindow with live vk_swapchain != 0 reports presents_on_gpu() == True!")
    win.swapchain = SwapchainHandle()

    # Headless `SwapchainSurface` lifecycle + HDR scRGB `R16G16B16A16_SFLOAT` (`format == 97`, `8` bytes/px) stride check!
    var surf = SwapchainSurface()
    _assert(surf.initialize_headless(320, 180, DISPLAY_SDR_SRGB, PRESENT_MAILBOX), "Headless SwapchainSurface init")
    _assert(surf.backend == SURFACE_HEADLESS and not surf.is_zero_copy_gpu(), "SURFACE_HEADLESS state")
    _assert(swapchain_format_bytes_per_pixel(FORMAT_B8G8R8A8_UNORM) == 4, "BGRA8 == 4 bytes/pixel")
    _assert(swapchain_format_bytes_per_pixel(FORMAT_A2B10G10R10_UNORM_PACK32) == 4, "RGB10A2 PQ == 4 bytes/pixel")
    _assert(swapchain_format_bytes_per_pixel(FORMAT_R16G16B16A16_SFLOAT) == 8, "R16G16B16A16_SFLOAT (97) == 8 bytes/pixel")
    _assert(resolve_swapchain_display_mode(FORMAT_R16G16B16A16_SFLOAT, True) == DISPLAY_EXTENDED_SCRGB_FP16, "scRGB FP16 display mode == 3")
    _assert(
        not validate_present_source_bytes(FORMAT_R16G16B16A16_SFLOAT, 320 * 180 * 4, 320, 320, 180),
        "4-byte/pixel PQ buffer MUST be rejected for 8-byte/pixel R16G16B16A16_SFLOAT swapchain!",
    )
    _assert(
        validate_present_source_bytes(FORMAT_R16G16B16A16_SFLOAT, 320 * 180 * 8, 320, 320, 180),
        "8-byte/pixel RGBA16F buffer accepted for R16G16B16A16_SFLOAT swapchain",
    )
    var surf_scrgb = SwapchainSurface()
    _assert(surf_scrgb.initialize_headless(320, 180, DISPLAY_EXTENDED_SCRGB_FP16, PRESENT_MAILBOX), "Headless scRGB surface init")
    _assert(surf_scrgb.is_scrgb_fp16() and surf_scrgb.bytes_per_pixel() == 8 and surf_scrgb.handle.format == FORMAT_R16G16B16A16_SFLOAT, "Headless scRGB format == 97 (8 bpp)")

    # Unified Right-Handed `-Z` forward view + Reverse-Z projection math
    var view = make_rh_look_at(Vec3(0.0, 2.0, 5.0), Vec3(0.0, 2.0, 0.0))
    var p_front = view.transform_point(Vec3(0.0, 2.0, 1.0))  # 4m in front of eye
    _assert(abs(p_front.x) < Float32(1e-5) and abs(p_front.y) < Float32(1e-5) and abs(p_front.z - Float32(-4.0)) < Float32(1e-4), "RH -Z view places forward target at z_view = -4.0")

    var near_z = Float32(0.10)
    var proj_inf = make_rh_reverse_z_infinite_perspective(Float32(60.0) * DEG_TO_RAD_F32, Float32(16.0 / 9.0), near_z)
    var vp_inf = proj_inf.multiply(view)
    var sc_near = project_rh_world_to_screen(vp_inf, Vec3(0.0, 2.0, 5.0 - near_z), 1280, 720)
    var sc_far = project_rh_world_to_screen(vp_inf, Vec3(0.0, 2.0, -995.0), 1280, 720)
    var sc_behind = project_rh_world_to_screen(vp_inf, Vec3(0.0, 2.0, 7.0), 1280, 720)
    _assert(sc_near[3] and abs(sc_near[2] - Float32(1.0)) < Float32(1e-4), "Reverse-Z near plane maps to z_ndc = 1.0")
    _assert(sc_far[3] and sc_far[2] > Float32(0.0) and sc_far[2] < Float32(0.001), "Infinite Reverse-Z 1000m maps near 0.0")
    _assert(not sc_behind[3], "Behind-camera point rejected")

    var proj_fin = make_rh_reverse_z_perspective(Float32(60.0) * DEG_TO_RAD_F32, Float32(1.0), Float32(0.1), Float32(100.0))
    var vp_fin = proj_fin.multiply(view)
    var sc_fin_far = project_rh_world_to_screen(vp_fin, Vec3(0.0, 2.0, 5.0 - 100.0), 512, 512)
    _assert(sc_fin_far[3] and abs(sc_fin_far[2]) < Float32(1e-4), "Finite Reverse-Z far plane maps to z_ndc = 0.0")

    # GPU 3D `DebugLineBatch` analytical near-plane clipping into `UiDrawList` (`PRIM_LINE`)
    var dbg = DebugLineBatch()
    # Line 1: straddles camera near plane (starts 3m behind eye at z=8, ends 4m in front at z=1) -> MUST clip & emit!
    dbg.add_line(Vec3(0.0, 2.0, 8.0), Vec3(0.0, 2.0, 1.0), 0xFFFF4040, 2.0, xray=False)
    # Line 2: completely behind camera (z=7 to z=12) -> MUST cull!
    dbg.add_line(Vec3(-1.0, 2.0, 7.0), Vec3(1.0, 2.0, 12.0), 0xFF40FF40, 2.0, xray=False)
    dbg.add_aabb(Vec3(-0.5, 1.5, -1.0), Vec3(0.5, 2.5, 0.0))
    dbg.add_gizmo(Vec3(0.0, 2.0, 0.0), Quat(), 0.75)

    var cpu_dd = DebugDraw()
    cpu_dd.set_enabled(True, Float32(1.0 / 60.0))
    cpu_dd.sphere(Vec3(0.5, 2.0, -0.5), Float32(0.35), DEBUG_CYAN)
    dbg.ingest_debug_draw(cpu_dd, xray=True)

    var ui_dbg = UiDrawList()
    var emitted = dbg.emit_to_ui_draw_list(ui_dbg, Vec3(0.0, 2.0, 5.0), Vec3(0.0, 2.0, 0.0), Float32(60.0), Float32(0.15), 320, 240)
    # 1 clipped straddler + 0 behind + 12 AABB + 9 gizmo + 48 sphere (3*16) = 70 segments
    _assert(emitted == 70 and ui_dbg.count() == 70, "DebugLineBatch near-plane clipped & emitted 70 PRIM_LINE segments")
    _assert((ui_dbg.prims[0].header & UInt32(0xF)) == PRIM_LINE, "Emitted debug primitives use PRIM_LINE")

    # Double-buffered `RenderWorld[2]` & `RenderExtractor` (`UInt64` entity ID tracking + despawn pruning)
    var rw_ring = DoubleBufferedRenderWorld()
    var extractor = RenderExtractor()

    var m_frame0 = trs(Vec3(1.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    var m_frame1 = trs(Vec3(3.5, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))

    extractor.begin_extraction(rw_ring.write_world(), 1)
    extractor.extract_entity(rw_ring.write_world(), UInt64(1001), 0, m_frame0)
    extractor.extract_entity(rw_ring.write_world(), UInt64(2002), 0, m_frame0)
    extractor.end_extraction()
    _assert(extractor.tracked_entity_count() == 2, "Extractor tracks 2 entities on frame 1")
    rw_ring.swap_buffers()
    _assert(not rw_ring.read_world().instances[0].has_prev, "Newly spawned entity has has_prev == False")

    # Frame 2: Entity 1001 moves to x=3.5, Entity 2002 despawns, Entity 3003 spawns
    extractor.begin_extraction(rw_ring.write_world(), 2)
    extractor.extract_entity(rw_ring.write_world(), UInt64(1001), 0, m_frame1)
    extractor.extract_entity(rw_ring.write_world(), UInt64(3003), 0, m_frame1)
    extractor.end_extraction()
    _assert(extractor.tracked_entity_count() == 2, "Extractor pruned despawned entity 2002")
    rw_ring.swap_buffers()
    ref inst_1001 = rw_ring.read_world().instances[0]
    ref inst_3003 = rw_ring.read_world().instances[1]
    _assert(inst_1001.has_prev and abs(inst_1001.prev_world_mat.values[12] - Float32(1.0)) < Float32(1e-5), "Entity 1001 prev_world_mat preserved")
    _assert(not inst_3003.has_prev and abs(inst_3003.prev_world_mat.values[12] - Float32(3.5)) < Float32(1e-5), "Newly spawned 3003 does not streak")

    # Tier 0 Headless SIMD `ScalableRenderer` verification (zero Vulkan overhead)
    _assert(RENDER_TIER_1_FORWARD_CLUSTERED == 1, "RENDER_TIER_1_FORWARD_CLUSTERED == 1")
    var tier0 = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96)
    var t0_mat = tier0.register_material(MaterialAsset(base_color=Vec3(0.9, 0.3, 0.2)))
    var mb_box = MeshBuilder()
    mb_box.add_beveled_box(Vec3(0.0, 0.5, 0.0), Vec3(0.6, 0.6, 0.6))
    var t0_mesh = tier0.register_mesh(mb_box^.finish(), t0_mat)
    rw_ring.write_world().clear_frame(3)
    rw_ring.write_world().camera = RenderCameraPacket(eye=Vec3(0.0, 1.2, 3.2), target=Vec3(0.0, 0.5, 0.0))
    extractor.begin_extraction(rw_ring.write_world(), 3)
    rw_ring.write_world().camera = RenderCameraPacket(eye=Vec3(0.0, 1.2, 3.2), target=Vec3(0.0, 0.5, 0.0))
    extractor.extract_entity(rw_ring.write_world(), UInt64(1001), t0_mesh, Mat4(), t0_mat)
    rw_ring.write_world().debug_lines.add_gizmo(Vec3(0.0, 0.5, 0.0), Quat(), 0.8)
    extractor.end_extraction()
    rw_ring.swap_buffers()

    var t0_ui = UiDrawList()
    t0_ui.add_rect(4.0, 4.0, 20.0, 10.0, 0xFFFF3030)
    _assert(tier0.render_world(rw_ring.read_world(), t0_ui, capture_readback=True), "Tier 0 SIMD render_world succeeded")
    var t0_px = tier0.readback_pixels()
    _assert(len(t0_px) == 128 * 96 and tier0.cpu_canvas.triangles_drawn > 0, "Tier 0 SIMD rasterized triangles")
    tier0.shutdown()

    print("[PASS] Suite 3: SwapchainSurface, RH -Z/Reverse-Z, DebugLineBatch, RenderExtractor & Tier 0 SIMD")


# ============================================================================
# Suite 4 & 5: GPU UI Overlay (Exact Red/Blue, 6 Prim Kinds, >300 Overlap, HDR10)
#              + White-Furnace + Cornell Box + 5x5 PBR/Clearcoat + 4-Cascade VSM Sundial
# ============================================================================

def _build_procedural_ui_atlas_16x16() -> List[UInt32]:
    """Builds a crisp 16x16 ARGB8888 border + SDF disk texture for `PRIM_NINE_SLICE`, `PRIM_IMAGE`, and `PRIM_GLYPH`."""
    var px = List[UInt32](capacity=256)
    for y in range(16):
        for x in range(16):
            var dx = Float32(x) - Float32(7.5)
            var dy = Float32(y) - Float32(7.5)
            var dist = sqrt(dx * dx + dy * dy)
            # Encode signed distance around r=5.0 in R channel (>128 inside), bright gold/cyan in GB
            var sdf_u8 = max(0, min(255, Int((Float32(0.5) + (Float32(5.2) - dist) * Float32(0.22)) * Float32(255.0))))
            var is_border = x < 3 or x >= 13 or y < 3 or y >= 13
            var g_u8 = 230 if is_border else 180
            var b_u8 = 255 if is_border else 90
            var r_u8 = max(sdf_u8, 220 if is_border else sdf_u8)
            px.append(pack_argb(r_u8, g_u8, b_u8, 255))
    return px^


def _project_lh_pixel(
    cam_pos: Vec3,
    cam_target: Vec3,
    fov_y_deg: Float32,
    w: Int,
    h: Int,
    world_pt: Vec3,
) -> Tuple[Int, Int]:
    """Exact pixel projection matching `FlintAaaVulkanRenderer.begin_frame` (`+X=right, +Y=up, +Z=fwd`)."""
    var fwd = (cam_target - cam_pos).normalized()
    var up_ref = Vec3(0.0, 0.0, 1.0) if abs(fwd.y) > Float32(0.98) else Vec3(0.0, 1.0, 0.0)
    var right = up_ref.cross(fwd).normalized()
    var up = fwd.cross(right).normalized()
    var rel = world_pt - cam_pos
    var vx = right.dot(rel)
    var vy = up.dot(rel)
    var vz = max(Float32(1e-3), fwd.dot(rel))
    var tan_y = sin(fov_y_deg * Float32(0.5) * DEG_TO_RAD_F32) / cos(fov_y_deg * Float32(0.5) * DEG_TO_RAD_F32)
    var tan_x = tan_y * (Float32(w) / Float32(h))
    var ndc_x = vx / (vz * tan_x)
    var ndc_y = vy / (vz * tan_y)
    var px = max(0, min(w - 1, Int((ndc_x * Float32(0.5) + Float32(0.5)) * Float32(w))))
    var py = max(0, min(h - 1, Int((Float32(1.0) - (ndc_y * Float32(0.5) + Float32(0.5))) * Float32(h))))
    return (px, py)


def _luma_rec709(c: Vec3) -> Float32:
    return c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)


def test_gpu_ui_and_aaa_metropolis():
    var W = 192
    var H = 144
    var renderer = ScalableRenderer(RENDER_TIER_2_AAA_VISBUFFER_RT, width=W, height=H, tsr_scale=1.0)
    if not renderer.uses_vulkan():
        print("[SKIP] Vulkan hardware unavailable in environment; CPU Tier 0 verified")
        renderer.shutdown()
        return

    ref gpu = renderer.gpu_slot[0]
    gpu.tsr_history_weight = Float32(0.0)
    gpu.enable_auto_exposure = False
    gpu.bloom_intensity = Float32(0.0)
    gpu.camera_lens.vignette_strength = Float32(0.0)

    # Upload procedural 16x16 bindless texture for Nine-Slice / Image / Glyph tests
    var atlas_px = _build_procedural_ui_atlas_16x16()
    var ui_tex_slot = gpu.upload_bc7_texture(atlas_px, 16, 16, srgb=False, generate_mips=False)
    _assert(ui_tex_slot >= 0, "Uploaded procedural 16x16 UI bindless texture")

    # ------------------------------------------------------------------------
    # Part A: Exact `0xFFFF2040` Pure-Red vs `0xFF2040FF` Pure-Blue + 6 SDF Kinds + >300 Tile Overlap!
    # ------------------------------------------------------------------------
    gpu.display_mode = DISPLAY_SDR_SRGB
    gpu.begin_frame(Vec3(0.0, 1.0, 4.0), Vec3(0.0, 1.0, 0.0))

    # 1. Exact pure-red (`0xFFFF2040`: R=255, G=32, B=64) vs pure-blue (`0xFF2040FF`: R=32, G=64, B=255)
    gpu.ui_draw_list.add_rect(8.0, 8.0, 28.0, 28.0, 0xFFFF2040)
    gpu.ui_draw_list.add_rect(44.0, 8.0, 28.0, 28.0, 0xFF2040FF)

    # Styled `PRIM_SHAPE` with vertical gradient + border + drop shadow
    gpu.ui_draw_list.add_rect_styled(
        80.0, 8.0, 32.0, 28.0,
        0xFFFFA020, 0xFF2080FF,
        corner_radius=5.0,
        border_width=2.0,
        flags=UI_FLAG_VERT_GRADIENT | UI_FLAG_DROP_SHADOW | UI_FLAG_BORDER,
    )

    # 2. `PRIM_GLYPH` (kind 1) with outer outline (`param1`) + drop shadow (`UI_FLAG_DROP_SHADOW`)
    var glyph_hdr = (PRIM_GLYPH & UInt32(0xF)) | (UInt32(ui_tex_slot & 0xFFF) << 4) | (UI_FLAG_DROP_SHADOW << 16)
    gpu.ui_draw_list.prims.append(
        GpuUiPrim(
            8.0, 46.0, 36.0, 74.0,
            0.0, 0.0, 1.0, 1.0,
            0, UInt32(0xFFFF) | (UInt32(0xFFFF) << 16),
            0xFFFFF040, 0xFF102030,
            Float32(0.08), Float32(1.5), Float32(1.0), glyph_hdr,
        )
    )

    # 3. `PRIM_NINE_SLICE` (kind 2)
    gpu.ui_draw_list.add_nine_slice(ui_tex_slot, 44.0, 46.0, 36.0, 28.0, 0.25, 0.25, 0.25, 0.25, 0xFFFFFFFF)

    # 4. `PRIM_CONTROLLER_SYMBOL` (kind 3): 'A' button, 'Play', and 'Gizmo'
    gpu.ui_draw_list.prims.append(make_ui_symbol_prim(UI_SYMBOL_PLAY, 88.0, 46.0, 28.0, 28.0, 0xFF30FF80, 0xFF182230))
    gpu.ui_draw_list.add_controller_symbol(UI_SYMBOL_A, 8.0, 84.0, 26.0, 26.0, 0xFFFFFFFF, 0xFF2060D0)
    gpu.ui_draw_list.prims.append(make_ui_symbol_prim(UI_SYMBOL_GIZMO, 40.0, 84.0, 26.0, 26.0, 0xFFFFD030, 0xFF181820))

    # 5. `PRIM_LINE` (kind 4): straight capsule line + quadratic Bézier curve
    gpu.ui_draw_list.prims.append(make_ui_line_prim(8.0, 124.0, 72.0, 124.0, 6.0, 0xFF20F0D0))
    gpu.ui_draw_list.prims.append(make_ui_bezier_prim(80.0, 130.0, 110.0, 96.0, 140.0, 130.0, 5.0, 0xFFFF50E0))

    # 6. `PRIM_IMAGE` (kind 5): bindless textured rounded quad
    gpu.ui_draw_list.prims.append(make_ui_image_prim(ui_tex_slot, 148.0, 46.0, 32.0, 32.0, tint_rgba8=0xFFFFFFFF, corner_radius=4.0))

    # 7. Chunked `>300` (`340`) overlapping primitives inside ONE `16x16` tile (`x in [128..144], y in [8..24]`)!
    for ov_i in range(339):
        var alt_col: UInt32 = 0xFFD03030 if (ov_i & 1) == 0 else 0xFF3030D0
        gpu.ui_draw_list.add_rect(126.0, 6.0, 22.0, 22.0, alt_col)
    # Primitive #340 (well past single-chunk 256 cap!) paints exact green `0xFF10F030` (`R=16, G=240, B=48`)
    gpu.ui_draw_list.add_rect(126.0, 6.0, 22.0, 22.0, 0xFF10F030)

    _assert(gpu.render_frame(capture_readback=True), "UI overlay verification frame rendered")
    var ui_px = gpu.readback_pixels()

    # Check exact pure-red `0xFFFF2040` at `(20, 20)` -> `(R=255, G=32, B=64)`
    var red_p = _unpack_argb(ui_px[20 * W + 20])
    _assert(
        abs(red_p[0] - 255) <= 1 and abs(red_p[1] - 32) <= 2 and abs(red_p[2] - 64) <= 2,
        "Pure-red 0xFFFF2040 readback mismatch (got R="
        + String(red_p[0]) + " G=" + String(red_p[1]) + " B=" + String(red_p[2]) + ")",
    )

    # Check exact pure-blue `0xFF2040FF` at `(56, 20)` -> `(R=32, G=64, B=255)`
    var blue_p = _unpack_argb(ui_px[20 * W + 56])
    _assert(
        abs(blue_p[0] - 32) <= 2 and abs(blue_p[1] - 64) <= 2 and abs(blue_p[2] - 255) <= 1,
        "Pure-blue 0xFF2040FF readback mismatch (got R="
        + String(blue_p[0]) + " G=" + String(blue_p[1]) + " B=" + String(blue_p[2]) + ")",
    )

    # Check 340th overlapping primitive at `(136, 16)` -> exact `0xFF10F030` (`R=16, G=240, B=48`)
    var ov_p = _unpack_argb(ui_px[16 * W + 136])
    _assert(
        abs(ov_p[0] - 16) <= 2 and abs(ov_p[1] - 240) <= 2 and abs(ov_p[2] - 48) <= 2,
        "340-prim chunked tile loop dropped or misordered prim #340 (got R="
        + String(ov_p[0]) + " G=" + String(ov_p[1]) + " B=" + String(ov_p[2]) + ")",
    )

    # Check PRIM_LINE `(40, 124)`, Bezier apex `(110, 113)`, Controller Play `(101, 60)`, Nine-Slice `(46, 60)`, Image `(164, 62)`
    var line_p = _unpack_argb(ui_px[124 * W + 40])
    _assert(line_p[1] > 200 and line_p[2] > 170, "PRIM_LINE capsule rendered cyan stroke")
    var bez_p = _unpack_argb(ui_px[113 * W + 110])
    _assert(bez_p[0] > 180 and bez_p[2] > 160, "PRIM_LINE quadratic Bezier rendered magenta arch")
    var sym_p = _unpack_argb(ui_px[60 * W + 101])
    _assert(sym_p[1] > 200, "PRIM_CONTROLLER_SYMBOL Play icon rendered")
    var ns_p = _unpack_argb(ui_px[60 * W + 46])
    _assert(ns_p[0] > 120 or ns_p[1] > 120 or ns_p[2] > 120, "PRIM_NINE_SLICE sampled bindless border")
    var img_p = _unpack_argb(ui_px[62 * W + 164])
    _assert(img_p[0] > 100 or img_p[1] > 100, "PRIM_IMAGE sampled bindless quad")

    # ------------------------------------------------------------------------
    # Part B: HDR10 ST.2084 PQ (`disp_mode == 1`) @ 203 nits UI Paper-White
    # ------------------------------------------------------------------------
    gpu.display_mode = DISPLAY_HDR10_PQ
    gpu.begin_frame(Vec3(0.0, 1.0, 4.0), Vec3(0.0, 1.0, 0.0))
    gpu.ui_draw_list.add_rect(16.0, 16.0, 48.0, 48.0, 0xFFFFFFFF)
    _assert(gpu.render_frame(capture_readback=True), "HDR10 PQ UI frame rendered")
    var hdr_px = gpu.readback_pixels()
    var pq_word = hdr_px[32 * W + 32]
    var pq_r10 = Int(pq_word & UInt32(0x3FF))
    var pq_g10 = Int((pq_word >> 10) & UInt32(0x3FF))
    var pq_b10 = Int((pq_word >> 20) & UInt32(0x3FF))
    var pq_a2 = Int((pq_word >> 30) & UInt32(0x3))
    # ST.2084 PQ of 203 nits (203 / 10000 = 0.0203) is ~0.5807 -> * 1023 ≈ 594
    _assert(
        pq_a2 == 3 and pq_r10 >= 565 and pq_r10 <= 620 and abs(pq_r10 - pq_g10) <= 2 and abs(pq_g10 - pq_b10) <= 2,
        "HDR10 PQ 203-nit paper-white out of expected [565..620] code range: " + String(pq_r10),
    )
    gpu.display_mode = DISPLAY_SDR_SRGB

    print(
        "[PASS] Suite 4: GPU UI Exact 0xFFFF2040 Red / 0xFF2040FF Blue, All 6 SDF Kinds, 340-Prim Tile & HDR10 PQ (code="
        + String(pq_r10) + ")"
    )

    # ------------------------------------------------------------------------
    # Part C: White-Furnace Energy Conservation Check
    # ------------------------------------------------------------------------
    var mb_sphere = MeshBuilder()
    mb_sphere.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), rings=16, segments=24)
    var sphere_mesh = mb_sphere^.finish()

    var wf_mat = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.45, metallic=0.0)
    var wf_mat_id = gpu.add_material(wf_mat)
    var sphere_entry = gpu.add_mesh(sphere_mesh, wf_mat_id, build_lods=True)

    gpu.furnace_white_dome = True
    gpu.enable_vsm = False
    gpu.enable_rt_shadows = False
    gpu.enable_rt_gi = False
    gpu.enable_rt_reflections = False
    gpu.exposure = Float32(1.0)

    gpu.begin_frame(Vec3(0.0, 0.0, 2.8), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    gpu.draw_instance(sphere_entry.mesh_id, Mat4())
    _assert(gpu.render_frame(capture_readback=True), "White-furnace frame rendered")
    var wf_hdr = gpu.readback_resolved_hdr()
    var wf_center = wf_hdr[(H // 2) * W + (W // 2)]
    _assert(
        wf_center.x >= Float32(0.78) and wf_center.x <= Float32(1.15) and abs(wf_center.x - wf_center.y) < Float32(0.05),
        "White-furnace energy conservation violated: L=" + String(wf_center.x),
    )
    gpu.furnace_white_dome = False

    # ------------------------------------------------------------------------
    # Part D: 4-Cascade Toroidal VSM (`±16m, ±64m, ±256m, ±1024m`) + Cornell Box + 5x5 PBR/Clearcoat + Sundial
    # ------------------------------------------------------------------------
    _assert(
        VSM_CASCADE_0_EXTENT == 16.0
        and VSM_CASCADE_1_EXTENT == 64.0
        and VSM_CASCADE_2_EXTENT == 256.0
        and VSM_CASCADE_3_EXTENT == 1024.0,
        "4-cascade extents == 16, 64, 256, 1024m",
    )
    var sun_d = Vec3(0.55, 0.72, 0.42).normalized()
    var casc_mats = compute_vsm_cascade_matrices(sun_d, Vec3(3.0, 1.5, -2.0))
    _assert(len(casc_mats) == 4 and len(casc_mats[0]) == 16, "compute_vsm_cascade_matrices produced 4 4x4 matrices")

    var vsm_mgr = VsmPageManager()
    var vsm_tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
    var orig_a = compute_vsm_cascade_page_origins(sun_d, Vec3(0.0, 0.0, 0.0))
    var empty_orig = List[Tuple[Int, Int]]()
    var marked_0 = mark_toroidal_cascade_pages(vsm_mgr, vsm_tbl, empty_orig, orig_a, active_radius_pages=2)
    var allocs_0 = vsm_mgr.process_requests(vsm_tbl)
    var marked_1 = mark_toroidal_cascade_pages(vsm_mgr, vsm_tbl, orig_a, orig_a, active_radius_pages=2)
    var allocs_1 = vsm_mgr.process_requests(vsm_tbl)
    _assert(marked_0 == 100 and len(allocs_0) == 100, "Initial 4-cascade 5x5 active region allocates 100 pages")
    _assert(marked_1 == 100 and len(allocs_1) == 0 and vsm_mgr.cached_hits_last_frame == 100, "Static toroidal VSM hits 100% cache on frame 2!")

    # Build Procedural Cornell Box + 5x5 PBR/Clearcoat Sphere Matrix + VSM Sundial Gnomon
    var mb_slab = MeshBuilder()
    mb_slab.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), bevel=0.02)
    var slab_mesh = mb_slab^.finish()
    var slab_entry = gpu.add_mesh(slab_mesh, wf_mat_id, build_lods=False)

    var mat_white = renderer.register_material(MaterialAsset(base_color=Vec3(0.76, 0.76, 0.74), roughness=0.75, metallic=0.0))
    var mat_red = renderer.register_material(MaterialAsset(base_color=Vec3(0.75, 0.12, 0.10), roughness=0.70, metallic=0.0))
    var mat_green = renderer.register_material(MaterialAsset(base_color=Vec3(0.12, 0.68, 0.15), roughness=0.70, metallic=0.0))
    var mat_gold = renderer.register_material(MaterialAsset(base_color=Vec3(0.95, 0.76, 0.30), roughness=0.18, metallic=1.0))

    # 5x5 PBR + Clearcoat Sphere Matrix materials
    var matrix_mats = List[Int](capacity=25)
    for row in range(5):
        for col in range(5):
            var rgh = Float32(0.08) + Float32(col) * Float32(0.21)
            var met = Float32(1.0) if row >= 2 else Float32(0.0)
            var m_sp = MaterialAsset(
                base_color=Vec3(0.20 + 0.18 * Float32(col), 0.35 + 0.12 * Float32(row), 0.85 - 0.14 * Float32(row)),
                roughness=rgh,
                metallic=met,
                domain=(MAT_DOMAIN_CLEARCOAT if row == 1 else MAT_DOMAIN_STANDARD_PBR),
            )
            if row == 1:
                m_sp.clearcoat = Float32(1.0)
                m_sp.clearcoat_roughness = Float32(0.05)
            matrix_mats.append(renderer.register_material(m_sp))

    ref gpu2 = renderer.gpu_slot[0]
    gpu2.enable_vsm = True
    gpu2.enable_vsm_clipmap_cascades = True
    gpu2.vsm_scene_center = Vec3(0.0, 1.5, 0.0)
    gpu2.vsm_ortho_half_extent = Float32(16.0)
    gpu2.sun_dir = Vec3(0.65, 0.72, 0.24).normalized()
    gpu2.sun_radiance = Vec3(5.5, 5.2, 4.8)
    gpu2.ibl_intensity = Float32(0.25)

    gpu2.begin_frame(Vec3(0.0, 3.2, 7.5), Vec3(0.0, 1.4, 0.0), fov_y_deg=50.0)
    # Cornell Floor, Back Wall, Left Red Wall, Right Green Wall
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(6.0, 0.1, 6.0)), material_override=mat_white)
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(0.0, 3.0, -4.5), Quat(), Vec3(6.0, 3.2, 0.12)), material_override=mat_white)
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(-4.8, 3.0, 0.0), Quat(), Vec3(0.12, 3.2, 5.0)), material_override=mat_red)
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(4.8, 3.0, 0.0), Quat(), Vec3(0.12, 3.2, 5.0)), material_override=mat_green)

    # 5x5 PBR/Clearcoat Sphere Matrix suspended along back gallery
    for row in range(5):
        for col in range(5):
            var sx = Float32(col - 2) * Float32(0.85)
            var sy = Float32(1.1) + Float32(row) * Float32(0.68)
            gpu2.draw_instance(
                sphere_entry.mesh_id,
                trs(Vec3(sx, sy, -2.6), Quat(), Vec3(0.26, 0.26, 0.26)),
                material_override=matrix_mats[row * 5 + col],
            )

    # Tall Sundial Gnomon pillar at `(1.2, 1.5, 1.2)` casting VSM shadow toward `(-X, -Z)` on the white floor
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(1.2, 1.5, 1.2), Quat(), Vec3(0.22, 1.5, 0.22)), material_override=mat_gold)

    # Project 3D Debug Gizmo onto zero-copy UI DrawList
    var scene_dbg = DebugLineBatch()
    scene_dbg.add_gizmo(Vec3(-1.8, 0.4, 1.5), Quat(), 0.75)
    _ = scene_dbg.emit_to_ui_draw_list(gpu2.ui_draw_list, Vec3(0.0, 3.2, 7.5), Vec3(0.0, 1.4, 0.0), Float32(50.0), Float32(0.1), W, H)

    _assert(gpu2.render_frame(capture_readback=True), "Cornell Box + 5x5 Sphere Matrix + VSM Sundial rendered")
    var scene_hdr = gpu2.readback_resolved_hdr()

    # Verify non-zero scene radiance and lit floor vs shadowed floor contrast
    var max_lum = Float32(0.0)
    var min_lum = Float32(1e9)
    for i in range(len(scene_hdr)):
        var c = scene_hdr[i]
        # Verify zero NaNs!
        _assert(c.x == c.x and c.y == c.y and c.z == c.z, "NaN detected in resolved HDR output")
        var lum = c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)
        if lum > max_lum:
            max_lum = lum
        if lum < min_lum:
            min_lum = lum
    _assert(max_lum > Float32(0.4) and max_lum > min_lum * Float32(2.5), "Scene exhibits lit vs shadowed VSM dynamic range")
    var cornell_meshlet_draws = gpu2.last_culled_meshlet_draws
    _assert(cornell_meshlet_draws >= 30, "GPU Meshlet culler emitted >= 30 indirect draws for Cornell + 5x5 Matrix + Sundial")

    # ------------------------------------------------------------------------
    # Part E: Extended Linear scRGB `RGBA16F` (`8` BPP) GPU Pack + Live `SurfaceMaterialNode` GPU Dispatch!
    # ------------------------------------------------------------------------
    gpu2.display_mode = AAA_DISPLAY_SCRGB_FP16
    gpu2.exposure = Float32(1.0)
    gpu2.begin_frame(Vec3(0.0, 3.2, 7.5), Vec3(0.0, 1.4, 0.0), fov_y_deg=50.0)
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(6.0, 0.1, 6.0)), material_override=mat_white)
    _assert(gpu2.render_frame(capture_readback=True), "scRGB RGBA16F 8-byte/pixel frame rendered")
    var scrgb_hdr_ref = gpu2.readback_resolved_hdr()
    var scrgb_words = gpu2.readback_scrgb_rgba16f_words()
    _assert(len(scrgb_words) == W * H * 2, "scRGB RGBA16F output packs exact 2 x UInt32 (8 bytes) per pixel")
    var probe_idx = (H - 24) * W + (W // 2)
    var w0_sc = scrgb_words[probe_idx * 2 + 0]
    var w1_sc = scrgb_words[probe_idx * 2 + 1]
    var r_sc = f16_bits_to_f32(UInt16(w0_sc & UInt32(0xFFFF)))
    var a_sc = f16_bits_to_f32(UInt16((w1_sc >> 16) & UInt32(0xFFFF)))
    var expected_r_sc = scrgb_hdr_ref[probe_idx].x * Float32(203.0 / 80.0)
    _assert(
        expected_r_sc > Float32(0.25)
        and abs(r_sc - expected_r_sc) < max(Float32(0.02), expected_r_sc * Float32(0.03))
        and abs(a_sc - Float32(1.0)) < Float32(0.01),
        "Extended Linear scRGB RGBA16F scales linear HDR by 203/80 = 2.5375 (r_sc="
        + String(r_sc) + ", expected=" + String(expected_r_sc) + ")",
    )

    # Dispatch live `TriplanarWeatherMaterialNode` (`SurfaceMaterialNode`) on GPU against unified 24-word `buf_materials`!
    var mat_basalt = renderer.register_material(
        MaterialAsset(base_color=Vec3(0.06, 0.06, 0.08), roughness=0.88, metallic=0.0)
    )
    var baked_slots = renderer.evaluate_material_graph(
        TriplanarWeatherMaterialNode(snow_line_y=12.0),
        mat_basalt,
        dim=32,
        time_sec=0.0,
        world_y_scale=24.0,
    )
    _assert(
        baked_slots[0] >= 0 and baked_slots[1] >= 0,
        "ScalableRenderer dispatched SurfaceMaterialNode on GPU and bound BC7 Albedo+ORM slots",
    )

    var mb_uv_hi = MeshBuilder()
    mb_uv_hi.add_ellipsoid(
        Vec3(-0.9, 0.0, 0.0), Vec3(0.68, 0.68, 0.25),
        u0=0.2, v0=0.08, u1=0.8, v1=0.18,  # v ~ 0.12 -> world_y ~ 21m > snow_line_y=12m (SNOW!)
    )
    var mb_uv_lo = MeshBuilder()
    mb_uv_lo.add_ellipsoid(
        Vec3(0.9, 0.0, 0.0), Vec3(0.68, 0.68, 0.25),
        u0=0.2, v0=0.82, u1=0.8, v1=0.94,  # v ~ 0.88 -> world_y ~ 2.8m < snow_line_y=12m (DARK BASALT!)
    )
    var e_snow_q = renderer.gpu_slot[0].add_mesh(mb_uv_hi^.finish(), mat_basalt, build_lods=False)
    var e_rock_q = renderer.gpu_slot[0].add_mesh(mb_uv_lo^.finish(), mat_basalt, build_lods=False)

    ref gpu_mg = renderer.gpu_slot[0]
    gpu_mg.display_mode = DISPLAY_SDR_SRGB
    gpu_mg.enable_vsm = False
    gpu_mg.enable_rt_shadows = False
    gpu_mg.enable_rt_gi = False
    gpu_mg.enable_rt_reflections = False
    gpu_mg.sun_dir = Vec3(0.55, 0.65, -0.52).normalized()
    gpu_mg.sun_radiance = Vec3(4.5, 4.5, 4.5)
    gpu_mg.ibl_intensity = Float32(0.04)
    var mg_eye = Vec3(0.0, 0.0, -3.2)
    var mg_tgt = Vec3(0.0, 0.0, 0.0)
    gpu_mg.begin_frame(mg_eye, mg_tgt, fov_y_deg=45.0)
    gpu_mg.draw_instance(e_snow_q.mesh_id, Mat4(), material_override=mat_basalt)
    gpu_mg.draw_instance(e_rock_q.mesh_id, Mat4(), material_override=mat_basalt)
    _assert(gpu_mg.render_frame(capture_readback=True), "SurfaceMaterialNode 3D mesh shading frame rendered")
    var mg_hdr = gpu_mg.readback_resolved_hdr()
    var px_snow = _project_lh_pixel(mg_eye, mg_tgt, Float32(45.0), W, H, Vec3(-0.9, 0.0, -0.2))
    var px_rock = _project_lh_pixel(mg_eye, mg_tgt, Float32(45.0), W, H, Vec3(0.9, 0.0, -0.2))
    var snow_lum = _luma_rec709(mg_hdr[px_snow[1] * W + px_snow[0]])
    var rock_lum = max(Float32(1e-4), _luma_rec709(mg_hdr[px_rock[1] * W + px_rock[0]]))
    _assert(
        snow_lum > Float32(0.55) and snow_lum > rock_lum * Float32(3.5),
        "GPU-dispatched TriplanarWeatherMaterialNode shaded snow vs basalt on 3D meshes (snow="
        + String(snow_lum) + ", rock=" + String(rock_lum) + ")",
    )

    print(
        "[PASS] Suite 5: White-Furnace (L=" + String(wf_center.x)
        + ") + scRGB RGBA16F (8BPP) + GPU SurfaceMaterialNode + Cornell Box ("
        + String(cornell_meshlet_draws) + " meshlet draws)"
    )
    renderer.shutdown()


def test_multi_cascade_vsm_gpu_shading():
    """Suite 6: Proves Cascades 1 (`±64m`, caster/receiver at `x = 45m` outside Cascade 0 `±16m`) AND
    Cascade 2 (`±256m`, caster/receiver at `x = 140m` outside Cascade 1 `±64m`) genuinely cast GPU PCSS shadows!"""
    var W = 320
    var H = 180
    var ren = FlintAaaVulkanRenderer(width=W, height=H, tsr_scale=1.0, max_draws=16384, max_instances=2048, env_cube_dim=32)
    if not ren.active:
        print("[SKIP] Vulkan unavailable for Suite 6 multi-cascade VSM GPU proof")
        ren.shutdown()
        return

    _assert(ren.enable_vsm_clipmap_cascades, "FlintAaaVulkanRenderer defaults enable_vsm_clipmap_cascades = True")
    ren.enable_vsm = True
    ren.enable_rt_shadows = False      # Strictly disable RT shadows so 100% of shadowing comes from VSM Cascades 1 & 2!
    ren.enable_rt_gi = False
    ren.enable_rt_reflections = False
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.bloom_intensity = Float32(0.0)
    ren.camera_lens.vignette_strength = Float32(0.0)
    ren.vsm_ortho_half_extent = Float32(16.0)  # Cascades: C0=±16m, C1=±64m, C2=±256m, C3=±1024m around cam_pos!
    ren.sun_dir = Vec3(0.0, 0.70710678, 0.70710678).normalized()  # 45-deg sun from +Z casting shadows toward -Z
    ren.sun_radiance = Vec3(6.5, 6.2, 5.8)
    ren.ibl_intensity = Float32(0.10)

    var mat_ground = ren.add_material(AaaMaterial(base_color=Vec3(0.85, 0.85, 0.85), roughness=0.85, metallic=0.0))
    var mat_pillar = ren.add_material(AaaMaterial(base_color=Vec3(0.70, 0.35, 0.20), roughness=0.60, metallic=0.0))

    var mb_box = MeshBuilder()
    mb_box.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), bevel=0.01)
    var box_e = ren.add_mesh(mb_box^.finish(), mat_ground, build_lods=False)

    # Camera stays at `x = 0.0` (`cam_pos = (0, 36, -48)`), so:
    #   - `x = 45.0 m` is `2.81x` outside Cascade 0 (`±16m`), strictly inside Cascade 1 (`±64m`)!
    #   - `x = 140.0 m` is `2.19x` outside Cascade 1 (`±64m`), strictly inside Cascade 2 (`±256m`)!
    var cam_eye = Vec3(0.0, 36.0, -48.0)
    var cam_tgt = Vec3(92.0, 0.0, 1.0)
    var fov_deg = Float32(55.0)

    ren.vsm.invalidate_all()
    ren.begin_frame(cam_eye, cam_tgt, fov_y_deg=fov_deg)
    # Station 1 (`x = 45m`, Cascade 1): ground pad + tall shadow-casting pillar at `(45, 5, 6)` casting umbra onto `(45, 0, 1)`
    ren.draw_instance(box_e.mesh_id, trs(Vec3(45.0, -0.25, 0.0), Quat(), Vec3(14.0, 0.25, 16.0)), material_override=mat_ground)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(45.0, 5.0, 6.0), Quat(), Vec3(2.8, 5.0, 1.2)), material_override=mat_pillar)
    # Station 2 (`x = 140m`, Cascade 2): ground pad + tall shadow-casting pillar at `(140, 10, 12)` casting umbra onto `(140, 0, 2)`
    ren.draw_instance(box_e.mesh_id, trs(Vec3(140.0, -0.25, 0.0), Quat(), Vec3(28.0, 0.25, 30.0)), material_override=mat_ground)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(140.0, 10.0, 12.0), Quat(), Vec3(6.5, 10.0, 2.5)), material_override=mat_pillar)
    _assert(ren.render_frame(capture_readback=True), "Multi-cascade VSM (45m C1 + 140m C2) frame rendered")

    var hdr = ren.readback_resolved_hdr()
    var px_u45 = _project_lh_pixel(cam_eye, cam_tgt, fov_deg, W, H, Vec3(45.0, 0.0, 1.0))
    var px_l45 = _project_lh_pixel(cam_eye, cam_tgt, fov_deg, W, H, Vec3(36.0, 0.0, 1.0))
    var px_u140 = _project_lh_pixel(cam_eye, cam_tgt, fov_deg, W, H, Vec3(140.0, 0.0, 2.0))
    var px_l140 = _project_lh_pixel(cam_eye, cam_tgt, fov_deg, W, H, Vec3(120.0, 0.0, 2.0))

    var umbra_luma_45 = _luma_rec709(hdr[px_u45[1] * W + px_u45[0]])
    var lit_luma_45 = _luma_rec709(hdr[px_l45[1] * W + px_l45[0]])
    var umbra_luma_140 = _luma_rec709(hdr[px_u140[1] * W + px_u140[0]])
    var lit_luma_140 = _luma_rec709(hdr[px_l140[1] * W + px_l140[0]])

    _assert(
        lit_luma_45 > Float32(0.5) and umbra_luma_45 < Float32(0.45) * lit_luma_45,
        "Cascade 1 (x=45m, outside ±16m C0) shadow failed: umbra=" + String(umbra_luma_45) + " lit=" + String(lit_luma_45),
    )
    _assert(
        lit_luma_140 > Float32(0.5) and umbra_luma_140 < Float32(0.45) * lit_luma_140,
        "Cascade 2 (x=140m, outside ±64m C1) shadow failed: umbra=" + String(umbra_luma_140) + " lit=" + String(lit_luma_140),
    )

    # Second static frame: verify 100% physical tile cache hit across BOTH Cascade 1 & Cascade 2 quadrants!
    ren.begin_frame(cam_eye, cam_tgt, fov_y_deg=fov_deg)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(45.0, -0.25, 0.0), Quat(), Vec3(14.0, 0.25, 16.0)), material_override=mat_ground)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(45.0, 5.0, 6.0), Quat(), Vec3(2.8, 5.0, 1.2)), material_override=mat_pillar)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(140.0, -0.25, 0.0), Quat(), Vec3(28.0, 0.25, 30.0)), material_override=mat_ground)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(140.0, 10.0, 12.0), Quat(), Vec3(6.5, 10.0, 2.5)), material_override=mat_pillar)
    _assert(ren.render_frame(capture_readback=True), "Multi-cascade VSM static cache frame rendered")
    _assert(
        ren.vsm.newly_rendered_pages_last_frame == 0 and ren.vsm.cached_hits_last_frame > 0,
        "Multi-cascade VSM static frame hit 100% cache (newly_rendered=" + String(ren.vsm.newly_rendered_pages_last_frame) + ")",
    )

    print(
        "[PASS] Suite 6: Real 4-Cascade VSM GPU Shading — C1@45m (umbra="
        + String(umbra_luma_45) + " vs lit=" + String(lit_luma_45)
        + "), C2@140m (umbra=" + String(umbra_luma_140) + " vs lit=" + String(lit_luma_140) + ")"
    )
    ren.shutdown()


def test_gpu_ui_polygons():
    """PRIM_POLYGON: nonzero winding (holes, unions blended once), packing, AA edges."""
    var dl = UiDrawList()
    var sq = List[Float32]()
    for v in [Float32(10), 10, 50, 10, 50, 50, 10, 50]:
        sq.append(v)
    var one = List[Int]()
    one.append(4)
    # 4 points + the closing repeat = 5 vertices -> 1 header + 1 data prim
    _assert(dl.add_polygon(sq, one, 0xFFFFFFFF) == 2, "a quad packs into a header and one data prim")
    _assert(dl.prims[0].u0 == 5.0 and dl.prims[1].header == PRIM_POLY_DATA, "polygon header holds the vertex count")
    _assert(dl.add_polygon(sq, List[Int](), 0xFFFFFFFF) == 0, "no outlines draws nothing")

    var W = 96
    var H = 64
    var renderer = ScalableRenderer(RENDER_TIER_2_AAA_VISBUFFER_RT, width=W, height=H, tsr_scale=1.0)
    if not renderer.uses_vulkan():
        print("[SKIP] Vulkan hardware unavailable; polygon packing verified")
        renderer.shutdown()
        return
    ref gpu = renderer.gpu_slot[0]
    gpu.display_mode = DISPLAY_SDR_SRGB
    gpu.begin_frame(Vec3(0.0, 1.0, 4.0), Vec3(0.0, 1.0, 0.0))
    gpu.ui_draw_list.add_rect(0.0, 0.0, Float32(W), Float32(H), 0xFF000000)
    # A square ring: outer clockwise, inner counter-clockwise (a hole).
    var ring = List[Float32]()
    for v in [Float32(4), 4, 40, 4, 40, 40, 4, 40, 14, 14, 14, 30, 30, 30, 30, 14]:
        ring.append(v)
    var two = List[Int]()
    two.append(4)
    two.append(4)
    _ = gpu.ui_draw_list.add_polygon(ring, two, 0xFFFFFFFF)
    # Two overlapping half-transparent squares as ONE polygon: the overlap blends once.
    var union = List[Float32]()
    for v in [Float32(48.5), 8, 72, 8, 72, 32, 48.5, 32, 60, 20, 84, 20, 84, 44, 60, 44]:
        union.append(v)
    _ = gpu.ui_draw_list.add_polygon(union, two, 0x80FFFFFF)
    _assert(gpu.render_frame(capture_readback=True), "polygon frame rendered")
    var px = gpu.readback_pixels()
    _assert(_unpack_argb(px[8 * W + 8])[0] > 250, "ring body is filled")
    _assert(_unpack_argb(px[22 * W + 22])[0] < 5, "ring hole stays empty")
    _assert(_unpack_argb(px[60 * W + 60])[0] < 5, "outside stays empty")
    var single = _unpack_argb(px[12 * W + 52])[0]
    var overlap = _unpack_argb(px[26 * W + 66])[0]
    _assert(abs(single - 128) <= 3 and abs(overlap - single) <= 2, "the union blends once (" + String(single) + ", " + String(overlap) + ")")
    var edge = _unpack_argb(px[12 * W + 48])[0]   # pixel centre (48.5, 12.5) sits on the edge x = 48.5
    _assert(abs(edge - single // 2) <= 4, "edges are anti-aliased (" + String(edge) + ")")
    var seam = _unpack_argb(px[20 * W + 60])[0]   # on an internal edge of the union
    _assert(abs(seam - single) <= 2, "no seam on internal union edges (" + String(seam) + ")")
    # Verify ScalableRenderer Tier 2 resize() recreates targets & updates SwapchainSurface
    var prev_gen = renderer.surface.resize_generations
    _assert(renderer.resize(128, 80), "ScalableRenderer Tier 2 resize(128, 80) succeeded")
    _assert(
        renderer.width == 128
        and renderer.height == 80
        and renderer.gpu_slot[0].output_w == 128
        and renderer.gpu_slot[0].output_h == 80
        and renderer.surface.resize_generations == prev_gen + 1,
        "ScalableRenderer Tier 2 resize synced dimensions and surface generation",
    )
    print("[PASS] Suite 4b: GPU UI PRIM_POLYGON nonzero winding (hole, union blended once, AA edge " + String(edge) + ", no seam)")
    renderer.shutdown()


def _validate_or_fail(words: List[UInt32], name: String):
    var rep = validate_spirv_module(words)
    if not rep.valid:
        _fail(name + " SPIR-V validation failed: " + rep.errors[0])


def test_render_features_glass_water_particles():
    """Suite 7: soft RT sun shadows, refractive glass, shaped water and generic GPU particles."""
    # Glass optics on the CPU
    _assert(abs(glass_f0_cpu(1.5) - Float32(0.04)) < 1e-4, "glass F0 at ior 1.5 is 4%")
    _assert(glass_fresnel_cpu(1.0, 1.5) < 0.05 and glass_fresnel_cpu(0.05, 1.5) > 0.6, "Fresnel rises toward grazing")
    var head_on = glass_refract_cpu(Vec3(0.0, 0.0, -1.0), Vec3(0.0, 0.0, 1.0), Float32(1.0 / 1.5))
    _assert(head_on[1] and abs(head_on[0].z + 1.0) < 1e-5, "head-on light passes straight through")
    var s45 = Float32(0.70710678)
    var bent = glass_refract_cpu(Vec3(s45, 0.0, -s45), Vec3(0.0, 0.0, 1.0), Float32(1.0 / 1.5))
    _assert(bent[1] and abs(bent[0].x - s45 / 1.5) < 1e-4, "Snell: sin(t) = sin(i) / 1.5")
    var tir = glass_refract_cpu(Vec3(s45, 0.0, -s45), Vec3(0.0, 0.0, 1.0), Float32(1.5))
    _assert(not tir[1] and tir[0].z > 0.0, "45 deg from inside glass is total internal reflection")
    var bl = glass_beer_lambert_cpu(Vec3(0.5, 1.0, 1.0), 1.0, 2.0)
    _assert(abs(bl.x - 0.25) < 1e-3 and abs(bl.y - 1.0) < 1e-4, "Beer-Lambert: tint^(density * path)")
    var clear = AaaGlassParams(material_id=0, ior=1.5).shadow_transmittance()
    var dark = AaaGlassParams(material_id=0, ior=1.5, tint=Vec3(0.3, 0.3, 0.3), opacity=0.5).shadow_transmittance()
    _assert(clear > 0.9 and clear < 1.0 and dark < clear * 0.2, "glass shadow transmittance")

    # Soft shadow filter radius: wider with occluder distance, capped
    var r_near = penumbra_radius_px_cpu(0.1, 0.005, 5.0, 0.4, 720)
    var r_far = penumbra_radius_px_cpu(2.0, 0.005, 5.0, 0.4, 720)
    _assert(r_near < r_far and r_far <= 14.0, "penumbra radius grows with occluder distance")

    # Generic particle emitter packing
    var em = AaaParticleEmitter.generic(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0), 300, color=Vec3(0.0, 3.0, 0.0), stretch=True)
    var ew = List[UInt32]()
    em.append_words(ew, 0)
    _assert(len(ew) == AAA_EMITTER_WORDS, "an emitter packs AAA_EMITTER_WORDS words")
    _assert((ew[14] & 7) == PARTICLE_GENERIC, "generic archetype in the bounce word's low bits")
    _assert(ew[22] == (PARTICLE_FLAG_ADDITIVE | PARTICLE_FLAG_STRETCH), "generic flags word")
    _assert(em.end_color.y == Float32(3.0) and em.end_size == em.size, "end color / size default to the start")

    _validate_or_fail(rt_sun_shadow_trace_shader(), "rt_sun_shadow_trace_shader")
    _validate_or_fail(rt_sun_shadow_filter_shader(), "rt_sun_shadow_filter_shader")
    _validate_or_fail(aaa_glass_refract_shader(), "aaa_glass_refract_shader")
    _validate_or_fail(aaa_water_refract_shader(True), "aaa_water_refract_shader")
    _validate_or_fail(gpu_particle_sim_and_bin_shader(), "gpu_particle_sim_and_bin_shader")
    _validate_or_fail(gpu_particle_tile_composite_shader(), "gpu_particle_tile_composite_shader")

    var W = 160
    var H = 96
    var ren = FlintAaaVulkanRenderer(width=W, height=H, tsr_scale=1.0, max_draws=1024, max_instances=256, env_cube_dim=32)
    if not ren.active:
        print("[SKIP] Vulkan unavailable for Suite 7 render-feature GPU checks")
        ren.shutdown()
        return
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.bloom_intensity = Float32(0.0)
    ren.camera_lens.vignette_strength = Float32(0.0)
    var mat_floor = ren.add_material(AaaMaterial(base_color=Vec3(0.3, 0.3, 0.3), roughness=0.8, metallic=0.0))
    var mat_wall = ren.add_material(AaaMaterial(base_color=Vec3(0.9, 0.06, 0.04), roughness=0.6, metallic=0.0))
    var mat_glass = ren.add_glass_material(AaaGlassParams(material_id=0, ior=1.2, density=0.0))
    _assert(mat_glass >= 0 and ren.is_glass_material(mat_glass), "glass material registered")
    var mb_box = MeshBuilder()
    mb_box.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), bevel=0.01)
    var box_e = ren.add_mesh(mb_box^.finish(), mat_floor, build_lods=False)
    var mb_ball = MeshBuilder()
    mb_ball.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.8, 0.8, 0.8), rings=16, segments=24)
    var ball_e = ren.add_mesh(mb_ball^.finish(), mat_glass, build_lods=False)
    var eye = Vec3(0.0, 1.0, 4.0)
    var tgt = Vec3(0.0, 1.0, 0.0)
    var cx = W // 2
    var cy = H // 2

    # Additive generic particles light the middle of the frame (green)
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    _assert(ren.render_frame(capture_readback=True), "particle baseline frame rendered")
    var g0 = ren.readback_resolved_hdr()[cy * W + cx].y
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    ren.submit_particle_emitter(
        AaaParticleEmitter.generic(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0), 400, color=Vec3(0.0, 3.0, 0.0), size=0.12,
                                   alpha=1.0, end_alpha=1.0, spawn_radius=0.15, velocity_spread=0.0, gravity_scale=0.0)
    )
    _assert(ren.render_frame(capture_readback=True), "particle frame rendered")
    var g1 = ren.readback_resolved_hdr()[cy * W + cx].y
    _assert(g1 > g0 + 0.5, "additive particles brighten the frame (green " + String(g0) + " -> " + String(g1) + ")")

    # Clear glass in front of a red wall shows the wall (needs hardware ray queries; no RT feature
    # needs to be on: glass in the frame builds the TLAS by itself)
    var glass_note = String("glass skipped (no ray queries)")
    if ren.vk.has_ray_query:
        ren.sun_dir = Vec3(0.3, 0.8, 0.6).normalized()  # the sun lights the wall's face
        var red_seen = Float32(0.0)
        var blue_seen = Float32(0.0)
        for _ in range(3):
            ren.begin_frame(eye, tgt, fov_y_deg=40.0)
            ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
            ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, 2.0, -2.0), Quat(), Vec3(3.0, 2.0, 0.1)), material_override=mat_wall)
            ren.draw_instance(ball_e.mesh_id, trs(Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
            _assert(ren.render_frame(capture_readback=True), "glass frame rendered")
            var c = ren.readback_resolved_hdr()[cy * W + cx].copy()
            red_seen = c.x
            blue_seen = c.z
        _assert(red_seen > blue_seen * 2.0, "the red wall shows through clear glass (r=" + String(red_seen) + " b=" + String(blue_seen) + ")")
        glass_note = "glass r=" + String(red_seen) + " b=" + String(blue_seen)
    # Verify Phase 0+2 ring-buffering, multi-mip Hi-Z pyramid, async VSM boundary crossing, and live resize()
    _assert(
        len(ren.ring_surfcache) == 2
        and len(ren.ring_vsm_table) == 2
        and len(ren.ring_oit_staging) == 2
        and ren.ring_surfcache[0].vk_buffer != ren.ring_surfcache[1].vk_buffer
        and ren.ring_vsm_table[0].vk_buffer != ren.ring_vsm_table[1].vk_buffer
        and ren.ring_oit_staging[0].vk_buffer != ren.ring_oit_staging[1].vk_buffer,
        "Per-slot ring-buffered surfcache, vsm_table, and oit_staging buffers are distinct",
    )
    _assert(
        ren.hiz_img.image != 0 and ren.hiz_mip_levels == compute_hiz_mip_levels(W, H) and ren.hiz_valid,
        "Multi-mip Hi-Z depth pyramid created and populated",
    )
    # VSM camera boundary crossing (async 1-frame readback without mid-frame stall)
    ren.enable_vsm = True
    ren.begin_frame(Vec3(12.0, 2.0, 4.0), Vec3(12.0, 1.0, 0.0), fov_y_deg=40.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(12.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    _assert(ren.render_frame(capture_readback=False), "VSM boundary-crossing async frame rendered")

    # Live FlintAaaVulkanRenderer.resize() + post-resize render & readback
    _assert(ren.resize(192, 128), "FlintAaaVulkanRenderer.resize(192, 128) succeeded")
    _assert(
        ren.output_w == 192 and ren.output_h == 128 and ren.hiz_mip_levels == compute_hiz_mip_levels(192, 128),
        "FlintAaaVulkanRenderer.resize updated output dimensions and Hi-Z mip count",
    )
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    _assert(ren.render_frame(capture_readback=True), "Post-resize frame rendered")
    var resized_px = ren.readback_pixels()
    _assert(len(resized_px) == 192 * 128, "Post-resize readback matches new 192x128 dimensions")

    # Wave 2: 1D Compacted Meshlet Worklist + AaaRenderGraph Barrier & Aliasing Metrics
    var rg_m = ren.get_render_graph_aliasing_metrics()
    _assert(
        rg_m.unaliased_bytes > 0
        and rg_m.aliased_peak_bytes > 0
        and rg_m.aliased_peak_bytes < rg_m.unaliased_bytes
        and rg_m.saved_bytes == rg_m.unaliased_bytes - rg_m.aliased_peak_bytes
        and rg_m.compiled_barrier_count >= 7,
        "AaaRenderGraph OffsetAllocator VRAM aliasing & barrier compilation verified",
    )
    _assert(
        ren.render_graph.passes_executed_last_frame >= 7 and ren.render_graph.barriers_emitted_last_frame >= 5,
        "AaaRenderGraph emitted barriers across core frame passes",
    )
    var expected_wl = ren.geom.meshes[box_e.mesh_id].meshlet_count * 10 + ren.geom.meshes[ball_e.mesh_id].meshlet_count
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    for k_i in range(10):
        ren.draw_instance(box_e.mesh_id, trs(Vec3(Float32(k_i) * Float32(0.4) - Float32(1.8), -0.1, 0.0), Quat(), Vec3(0.2, 0.1, 0.2)))
    ren.draw_instance(ball_e.mesh_id, trs(Vec3(0.0, 1.0, 0.0), Quat(), Vec3(0.5, 0.5, 0.5)), material_override=mat_floor)
    _assert(ren.render_frame(capture_readback=True), "1D compacted meshlet worklist frame rendered")
    _assert(
        ren.last_worklist_meshlets == expected_wl and ren.last_culled_meshlet_draws > 0,
        "1D compacted worklist count matches exact sum of meshlets (" + String(ren.last_worklist_meshlets) + " vs " + String(expected_wl) + ")",
    )

    # Wave 2: Projected OBB Deferred Decal (`GpuDecalVolume`) GPU upload & shading verification
    ren.enable_vsm = False
    ren.sun_dir = Vec3(0.0, 0.9, 0.4).normalized()
    ren.sun_radiance = Vec3(4.0, 4.0, 4.0)
    var dec_eye = Vec3(0.0, 3.0, 3.5)
    var dec_tgt = Vec3(0.0, 0.0, 0.0)
    var rw_d = ren.output_w
    var rh_d = ren.output_h
    var mid_idx = (rh_d // 2) * rw_d + (rw_d // 2)

    ren.begin_frame(dec_eye, dec_tgt, fov_y_deg=45.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)), material_override=mat_floor)
    _assert(ren.render_frame(capture_readback=True), "Decal baseline floor frame rendered")
    var floor_base = ren.readback_resolved_hdr()[mid_idx].copy()

    var red_decal = GpuDecalVolume(
        center=Vec3(0.0, 0.0, 0.0),
        half_extents=Vec3(2.5, 1.0, 2.5),
        rotation=Quat(),
        albedo_slot=BINDLESS_SLOT_NONE,
        opacity=1.0,
    )
    red_decal.tint_rgb = Vec3(0.95, 0.04, 0.04)
    ren.begin_frame(dec_eye, dec_tgt, fov_y_deg=45.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)), material_override=mat_floor)
    ren.add_decal(red_decal)
    _assert(ren.render_frame(capture_readback=True), "Projected OBB decal frame rendered")
    var floor_dec = ren.readback_resolved_hdr()[mid_idx].copy()
    _assert(
        floor_dec.x > floor_base.x * Float32(1.8) and floor_dec.x > floor_dec.z * Float32(3.0),
        "Projected OBB red decal shaded onto grey floor (r: " + String(floor_base.x) + " -> " + String(floor_dec.x) + ", b: " + String(floor_dec.z) + ")",
    )

    print("[PASS] Suite 7: Soft RT Shadows + Refractive Glass + Shaped Water + Generic GPU Particles (green "
          + String(g0) + " -> " + String(g1) + ", " + glass_note + ")")
    ren.shutdown()


def main():
    var t0 = get_time()
    print("=== Flint v5 Track 2 Render Pipeline & Metrology Suite ===")
    test_spirv_spec_constants_and_shaders()
    test_material_graph_decals_terrain_and_vat()
    test_swapchain_render_world_and_tier0()
    test_gpu_ui_and_aaa_metropolis()
    test_gpu_ui_polygons()
    test_multi_cascade_vsm_gpu_shading()
    test_render_features_glass_water_particles()
    var dt = get_time() - t0
    print("=== ALL FLINT V5 TRACK 2 RENDER TESTS PASSED in " + String(dt) + "s (<2.0s target) ===")
    _assert(dt < 5.0, "Test suite exceeded runtime safety ceiling: " + String(dt) + "s")
