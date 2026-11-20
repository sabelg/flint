"""Tier 3 Cross-Feature Combinations: Render Pipeline Interactions (15 tests).
Genuine GPU/Vulkan & RenderWorld integration across `ScalableRenderer`, `FlintAaaVulkanRenderer`,
meshlet cache optimization & QEM LODs, `MaterialAsset`, `AaaClusterLight`, `VsmPageManager`,
`AaaRenderGraph` transient aliasing & cross-queue barriers, SVGF, SPIR-V kernels,
`AssetServer` residency bridge, `UiContext`/`DockSpace`/`UiDrawList`, and `DebugLineBatch`.
"""
from std.collections import List
from std.math import abs, max
from std.memory import bitcast
from flint.assets.flpk_v3 import AssetUuid
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.meshlet import (
    AaaGeometryPool,
    compute_acmr,
    optimize_vertex_cache_forsyth,
    simplify_mesh_qem,
)
from flint.assets.server import CookedTextureAsset
from flint.core.math import Mat4, Vec3
from flint.core.transform import Quat, trs
from flint.platform.input import InputState
from flint.render.aaa_bsdf import AaaMaterial, AAA_MATERIAL_WORDS, reoriented_normal_blend_cpu
from flint.render.aaa_clustered_lights import AaaClusterLight, aaa_cluster_cull_shader
from flint.render.aaa_gi_svgf import (
    AAA_SHADE_CLASS_STANDARD_PBR,
    aaa_deferred_shade_class_shader,
    aaa_tile_classify_shader,
    svgf_atrous_shader,
)
from flint.render.aaa_post_tsr import AaaPhysicalCamera
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_render_graph import (
    AaaRenderGraph,
    RG_QUEUE_ASYNC_COMPUTE,
    RG_QUEUE_GRAPHICS,
    RG_USAGE_COMPUTE_READ,
    RG_USAGE_COMPUTE_WRITE,
    RG_USAGE_FRAGMENT_READ,
    RgResourceAccess,
)
from flint.render.aaa_visbuffer import hiz_depth_downsample_shader
from flint.render.aaa_vsm import VsmPageManager, VSM_TABLE_WORDS, VSM_VIRTUAL_PAGE_COUNT
from flint.render.image_diff import compare_pixels
from flint.render.material_graph import MaterialAsset, vat_crowd_eval_shader
from flint.render.render_world import (
    DebugLineBatch,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    RenderCameraPacket,
    RenderInstancePacket,
    RenderWorld,
    ScalableRenderer,
    make_rh_look_at,
    make_rh_reverse_z_infinite_perspective,
    project_rh_world_to_screen,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.ui.context import UiContext
from flint.ui.draw_list import UiDrawList
from flint.ui.flexbox_dock import DockSpace, RectF
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh
from e2e.common.v5_fixtures import create_mock_asset_server


def _hdr_energy(hdr: List[Vec3]) -> Float32:
    var total = Float32(0.0)
    for i in range(len(hdr)):
        total += hdr[i].x + hdr[i].y + hdr[i].z
    return total


def _pixel_variance(px: List[UInt32]) -> Int:
    if len(px) == 0:
        return 0
    var p0 = px[0]
    var diff = 0
    for i in range(1, len(px)):
        if px[i] != p0:
            diff += 1
    return diff


# 1. Forsyth vertex cache optimization -> ScalableRenderer multi-frame LOD mesh render
def test_t3_meshlet_cache_interaction(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active")
    var m = create_mock_meshlet_mesh(24)
    var int_indices = List[Int]()
    for i in range(len(m.indices)):
        int_indices.append(Int(m.indices[i]))
    var opt = optimize_vertex_cache_forsyth(int_indices, len(m.positions) // 3, 32)
    var orig_acmr = compute_acmr(int_indices, 32)
    var opt_acmr = compute_acmr(opt, 32)
    ctx.assert_true(opt_acmr <= orig_acmr + 0.05, "Forsyth cache optimization preserves or improves ACMR")

    var mat_id = ren.add_material(AaaMaterial(base_color=Vec3(0.3, 0.7, 0.9), roughness=0.35))
    var me = ren.add_mesh(m, mat_id, build_lods=True)
    var id_m = Mat4()
    for _ in range(2):
        ren.begin_frame(Vec3(0.0, 1.0, -5.0), Vec3(0.0, 0.0, 0.0))
        ren.draw_instance(me.mesh_id, id_m, material_override=mat_id)
        ctx.assert_true(ren.render_frame(capture_readback=True), "meshlet LOD frame renders")

    var px = ren.readback_pixels()
    ctx.assert_equal(len(px), ren.output_w * ren.output_h, "readback returns full framebuffer")
    ctx.assert_true(_pixel_variance(px) > 50, "rendered Forsyth/meshlet sphere produces non-uniform pixels")


# 2. Clearcoat MaterialAsset + AaaClusterLight GPU HDR irradiance difference
def test_t3_material_clustered_light_interaction(mut ctx: TestContext, mut ren: ScalableRenderer) raises:
    ctx.assert_true(ren.uses_vulkan(), "ScalableRenderer uses Vulkan Tier 2")
    var mat = MaterialAsset(base_color=Vec3(0.6, 0.7, 0.9), roughness=0.25, metallic=0.1)
    mat.clearcoat = Float32(0.75)
    mat.clearcoat_roughness = Float32(0.05)
    var mat_id = ren.register_material(mat)
    var mesh_id = ren.register_mesh(create_mock_meshlet_mesh(12), mat_id)

    var rw = RenderWorld()
    rw.camera = RenderCameraPacket(eye=Vec3(0.0, 0.5, 5.0), target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
    rw.sun_radiance = Vec3(0.0, 0.0, 0.0)
    var id_m = Mat4()
    rw.instances.append(RenderInstancePacket(1, mesh_id, id_m, id_m, material_id=mat_id))

    var ui = UiDrawList()
    ctx.assert_true(ren.render_world(rw, ui, capture_readback=True), "unlit clearcoat frame renders")
    var e_unlit = _hdr_energy(ren.gpu_slot[0].readback_resolved_hdr())

    rw.cluster_lights.append(AaaClusterLight.point(Vec3(0.0, 2.0, 3.0), Vec3(120.0, 100.0, 90.0), 20.0))
    ctx.assert_true(ren.render_world(rw, ui, capture_readback=True), "lit clearcoat frame renders")
    var e_lit = _hdr_energy(ren.gpu_slot[0].readback_resolved_hdr())

    ctx.assert_true(e_lit > e_unlit + Float32(10.0), "clustered point light adds real GPU HDR energy onto clearcoat material")


# 3. VSM page manager + Hi-Z shader validation + live VSM dirty page tracking
def test_t3_vsm_hiz_interaction(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active")
    ctx.assert_true(validate_spirv_module(hiz_depth_downsample_shader()).valid, "hiz_depth_downsample_shader valid SPIR-V")
    var vsm = VsmPageManager()
    var tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
    var test_v = 40 * 128 + 40
    tbl[VSM_VIRTUAL_PAGE_COUNT + test_v] = 1
    var alloc0 = vsm.process_requests(tbl)
    ctx.assert_equal(len(alloc0), 1, "VsmPageManager allocates 1 page on initial request")
    tbl[VSM_VIRTUAL_PAGE_COUNT + test_v] = 1
    var alloc_cached = vsm.process_requests(tbl)
    ctx.assert_equal(len(alloc_cached), 0, "VsmPageManager caches valid virtual page on repeat request")
    ctx.assert_equal(vsm.cached_hits_last_frame, 1, "VsmPageManager records 1 cached hit")
    vsm.invalidate_region(0.2, 0.2, 0.6, 0.6)
    ctx.assert_false(vsm.virtual_valid[test_v], "invalidate_region marks page inside region dirty")
    tbl[VSM_VIRTUAL_PAGE_COUNT + test_v] = 1
    var alloc_re = vsm.process_requests(tbl)
    ctx.assert_equal(len(alloc_re), 1, "VsmPageManager re-allocates invalidated virtual page")

    var mat_id = ren.add_material(AaaMaterial(base_color=Vec3(0.8, 0.6, 0.4), roughness=0.4))
    var me = ren.add_mesh(create_mock_meshlet_mesh(10), mat_id)
    ren.enable_vsm = True
    var dirty_f0 = 0
    var dirty_f1 = 0
    for f in range(2):
        ren.begin_frame(Vec3(0.0, 3.0, -6.0), Vec3(0.0, 0.0, 0.0))
        ren.sun_dir = Vec3(0.4 + Float32(f) * 0.2, 0.8, 0.3).normalized()
        ren.draw_instance(me.mesh_id, trs(Vec3(Float32(f) * 0.5, 0.0, 0.0), Quat(), Vec3(1, 1, 1)), material_override=mat_id)
        ctx.assert_true(ren.render_frame(capture_readback=True), "VSM-enabled frame renders")
        if f == 0:
            dirty_f0 = ren.last_vsm_dirty_pages
        else:
            dirty_f1 = ren.last_vsm_dirty_pages

    ctx.assert_true(dirty_f0 > 0 and dirty_f1 > 0, "renderer marks non-zero VSM dirty pages on invalidation and sun-dir cascade shift")
    ctx.assert_true(_pixel_variance(ren.readback_pixels()) > 20, "VSM-lit scene produces non-uniform pixels")
    ren.enable_vsm = False


# 4. AaaRenderGraph transient heap aliasing + live renderer RenderGraph execution
def test_t3_rendergraph_aliasing_compilation(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active")
    var rg = AaaRenderGraph()
    var r_a = rg.add_resource("depth_pre", 8 * 1024 * 1024, transient=True)
    var r_b = rg.add_resource("hi_z", 8 * 1024 * 1024, transient=True)
    var r_c = rg.add_resource("hdr_tmp", 8 * 1024 * 1024, transient=True)
    var r_d = rg.add_resource("post_tmp", 8 * 1024 * 1024, transient=True)

    var p0 = List[RgResourceAccess](); p0.append(RgResourceAccess(r_a, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("p0", p0)
    var p1 = List[RgResourceAccess](); p1.append(RgResourceAccess(r_a, RG_USAGE_COMPUTE_READ)); p1.append(RgResourceAccess(r_b, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("p1", p1)
    var p2 = List[RgResourceAccess](); p2.append(RgResourceAccess(r_b, RG_USAGE_COMPUTE_READ)); p2.append(RgResourceAccess(r_c, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("p2", p2)
    var p3 = List[RgResourceAccess](); p3.append(RgResourceAccess(r_c, RG_USAGE_COMPUTE_READ)); p3.append(RgResourceAccess(r_d, RG_USAGE_FRAGMENT_READ))
    _ = rg.add_pass("p3", p3)

    ctx.assert_true(rg.compile(24 * 1024 * 1024), "4-pass graph compiles within 24 MiB aliased heap")
    var metrics = rg.aliasing_metrics()
    ctx.assert_true(metrics.saved_bytes > 0 and metrics.aliased_peak_bytes < metrics.unaliased_bytes, "transient aliasing saves heap bytes")

    ren.begin_frame(Vec3(0.0, 1.0, -4.0), Vec3(0.0, 0.0, 0.0))
    ctx.assert_true(ren.render_frame(), "live renderer RenderGraph executes")
    ctx.assert_true(ren.render_graph.passes_executed_last_frame > 0, "live renderer executed RenderGraph passes")


# 5. Reoriented normal blend + SVGF a-trous SPIR-V validation + 2-frame SVGF HDR accumulation
def test_t3_svgf_normal_blend_filtering(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active")
    var macro_n = Vec3(0.0, 0.0, 1.0)
    var detail_n = Vec3(0.05, 0.05, 0.99).normalized()
    var blended = reoriented_normal_blend_cpu(macro_n, detail_n, 1.0)
    ctx.assert_true(macro_n.dot(blended) > Float32(0.95), "reoriented normal blend preserves unit orientation")
    ctx.assert_true(validate_spirv_module(svgf_atrous_shader()).valid, "svgf_atrous_shader validates")

    var mat_id = ren.add_material(AaaMaterial(base_color=Vec3(0.7, 0.7, 0.7), roughness=0.4))
    var me = ren.add_mesh(create_mock_meshlet_mesh(10), mat_id)
    ren.svgf_passes = 3
    var hdr_prev = List[Vec3]()
    var hdr_curr = List[Vec3]()
    for f in range(2):
        ren.begin_frame(Vec3(0.0, 1.0, -4.5), Vec3(0.0, 0.0, 0.0))
        ren.draw_instance(me.mesh_id, Mat4(), material_override=mat_id)
        ctx.assert_true(ren.render_frame(), "SVGF frame renders")
        if f == 0:
            hdr_prev = ren.readback_resolved_hdr()
        else:
            hdr_curr = ren.readback_resolved_hdr()

    ctx.assert_equal(len(hdr_prev), ren.output_w * ren.output_h, "SVGF frame 1 HDR readback matches viewport")
    ctx.assert_equal(len(hdr_curr), ren.output_w * ren.output_h, "SVGF frame 2 HDR readback matches viewport")
    var e_prev = _hdr_energy(hdr_prev)
    var e_curr = _hdr_energy(hdr_curr)
    ctx.assert_true(e_prev > Float32(10.0) and e_curr > Float32(10.0), "SVGF-denoised frames retain positive resolved HDR radiance")
    ctx.assert_true(abs(e_curr - e_prev) < max(e_prev, e_curr) * Float32(0.25), "multi-frame SVGF HDR accumulation remains energy-consistent")
    ren.svgf_passes = 0


# 6. SPIR-V cluster cull + tile classify + split VisBuffer shade shader + GPU tile classify readback
def test_t3_spirv_clustered_binning_kernel(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active")
    ctx.assert_true(validate_spirv_module(aaa_cluster_cull_shader(0, 1)).valid, "aaa_cluster_cull_shader valid")
    ctx.assert_true(validate_spirv_module(aaa_tile_classify_shader()).valid, "aaa_tile_classify_shader valid")
    ctx.assert_true(
        validate_spirv_module(aaa_deferred_shade_class_shader(True, AAA_SHADE_CLASS_STANDARD_PBR, True)).valid,
        "aaa_deferred_shade_class_shader valid",
    )

    var mat_id = ren.add_material(AaaMaterial(base_color=Vec3(0.5, 0.8, 0.4), roughness=0.4))
    var me = ren.add_mesh(create_mock_meshlet_mesh(10), mat_id)
    ren.begin_frame(Vec3(0.0, 0.5, -4.5), Vec3(0.0, 0.0, 0.0))
    ren.draw_instance(me.mesh_id, Mat4(), material_override=mat_id)
    ren.add_cluster_light(AaaClusterLight.point(Vec3(0.0, 2.0, -2.0), Vec3(40.0, 40.0, 40.0), 15.0))
    ctx.assert_true(ren.render_frame(), "clustered binning + tile classify frame renders")

    var counts = ren.readback_tile_classify_counts()
    var sum_tiles = 0
    for i in range(len(counts)):
        sum_tiles += counts[i]
    ctx.assert_true(sum_tiles > 0, "GPU tile classifier bins active screen tiles")


# 7. QEM simplification -> AaaGeometryPool LOD build -> near vs far ScalableRenderer draw
def test_t3_qem_simplification_lod_error(mut ctx: TestContext, mut ren: ScalableRenderer) raises:
    ctx.assert_true(ren.uses_vulkan(), "ScalableRenderer uses Vulkan Tier 2")
    var m = create_mock_meshlet_mesh(16)
    var orig_tris = len(m.indices) // 3
    var qem = simplify_mesh_qem(m, orig_tris // 2)
    ctx.assert_true(len(qem[0].indices) // 3 < orig_tris and qem[1] >= 0.0, "QEM simplifies mesh with non-negative error")

    var gpool = AaaGeometryPool()
    var entry = gpool.append_mesh(m, 0, build_lods=True)
    ctx.assert_true(len(entry.lods) >= 2, "AaaGeometryPool builds multi-LOD slices")

    var mat_id = ren.register_material(MaterialAsset(base_color=Vec3(0.9, 0.5, 0.2), roughness=0.35))
    var mesh_id = ren.register_mesh(m, mat_id, build_lods=True)
    var ui = UiDrawList()

    var rw_near = RenderWorld()
    rw_near.camera = RenderCameraPacket(eye=Vec3(0.0, 0.0, 4.0), target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
    rw_near.instances.append(RenderInstancePacket(1, mesh_id, Mat4(), Mat4(), material_id=mat_id))
    ctx.assert_true(ren.render_world(rw_near, ui, capture_readback=True), "near LOD frame renders")
    var px_near = ren.readback_pixels()

    var rw_far = RenderWorld()
    rw_far.camera = RenderCameraPacket(eye=Vec3(0.0, 0.0, 32.0), target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
    rw_far.instances.append(RenderInstancePacket(1, mesh_id, Mat4(), Mat4(), material_id=mat_id))
    ctx.assert_true(ren.render_world(rw_far, ui, capture_readback=True), "far LOD frame renders")
    var px_far = ren.readback_pixels()

    var diff = compare_pixels(px_near, px_far, ren.width, ren.height, 4)
    ctx.assert_true(diff.differing > 50, "near vs far LOD renders produce distinct screen coverage")


# 8. UiContext + DockSpace layout -> UiDrawList composited inside ScalableRenderer
def test_t3_widget_canvas_clipping_interaction(mut ctx: TestContext, mut ren: ScalableRenderer) raises:
    ctx.assert_true(ren.uses_vulkan(), "ScalableRenderer uses Vulkan Tier 2")
    var dock = DockSpace.create_studio_5panel()
    var ui = UiContext()
    var inp = InputState()
    ui.begin_frame(inp, Float32(ren.width), Float32(ren.height))
    ui.update_dock_space(dock, RectF(0.0, 0.0, Float32(ren.width), Float32(ren.height)))
    if ui.begin_dock_panel(dock, "Inspector"):
        _ = ui.button("t3_btn", "Inspect", Float32(60.0), Float32(20.0))
        ui.end_dock_panel()
    ui.end_frame()
    ctx.assert_true(len(ui.draw_list.prims) > 0, "UiContext + DockSpace emits UI overlay primitives")

    var rw = RenderWorld()
    rw.camera = RenderCameraPacket(eye=Vec3(0.0, 1.0, 5.0), target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
    var ui_empty = UiDrawList()
    ctx.assert_true(ren.render_world(rw, ui_empty, capture_readback=True), "baseline render without UI overlay")
    var px_no_ui = ren.readback_pixels()

    ctx.assert_true(ren.render_world(rw, ui.draw_list, capture_readback=True), "composite render with UiDrawList overlay")
    var px_with_ui = ren.readback_pixels()
    var diff = compare_pixels(px_no_ui, px_with_ui, ren.width, ren.height, 2)
    ctx.assert_true(diff.differing > 20, "compositing UiContext DockSpace primitives changes output pixels")


# 9. Multi-light clustered radiance accumulation verified via GPU HDR readback
def test_t3_multi_light_radiance_accumulation(mut ctx: TestContext, mut ren: ScalableRenderer) raises:
    ctx.assert_true(ren.uses_vulkan(), "ScalableRenderer uses Vulkan Tier 2")
    var mat_id = ren.register_material(MaterialAsset(base_color=Vec3(0.85, 0.85, 0.85), roughness=0.5))
    var mesh_id = ren.register_mesh(create_mock_meshlet_mesh(12), mat_id)
    var ui = UiDrawList()

    var rw1 = RenderWorld()
    rw1.camera = RenderCameraPacket(eye=Vec3(0.0, 0.0, 5.5), target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
    rw1.sun_radiance = Vec3(0.0, 0.0, 0.0)
    rw1.instances.append(RenderInstancePacket(1, mesh_id, Mat4(), Mat4(), material_id=mat_id))
    rw1.cluster_lights.append(AaaClusterLight.point(Vec3(-2.0, 2.0, 3.0), Vec3(40.0, 20.0, 10.0), 18.0))
    ctx.assert_true(ren.render_world(rw1, ui, capture_readback=True), "1-light frame renders")
    var e1 = _hdr_energy(ren.gpu_slot[0].readback_resolved_hdr())

    rw1.cluster_lights.append(AaaClusterLight.point(Vec3(2.0, 2.0, 3.0), Vec3(10.0, 40.0, 20.0), 18.0))
    rw1.cluster_lights.append(AaaClusterLight.point(Vec3(0.0, -2.0, 3.0), Vec3(20.0, 10.0, 50.0), 18.0))
    ctx.assert_true(ren.render_world(rw1, ui, capture_readback=True), "3-light frame renders")
    var e3 = _hdr_energy(ren.gpu_slot[0].readback_resolved_hdr())

    ctx.assert_true(e3 > e1 + Float32(10.0), "3 clustered lights accumulate strictly more GPU HDR energy than 1 light")


# 10. AssetServer residency bridge -> mip upload -> MaterialAsset bindless slot -> rendered frame
def test_t3_streamer_budget_feedback_loop(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active")
    var srv = create_mock_asset_server()
    srv.attach_renderer_residency_bridge(ren)

    var tex = CookedTextureAsset()
    tex.width = 64
    tex.height = 64
    tex.mip_count = 5
    tex.tail_first_mip = 4
    tex.resident_min_mip = 4
    for m in range(5):
        var dim = 64 >> m
        var b = ((dim + 3) // 4) * ((dim + 3) // 4) * 16
        tex.mip_sizes[m] = UInt32(b)
        tex.mips.append(List[UInt8](length=b, fill=UInt8(0xAA)))

    var th = srv.textures.insert(tex^, AssetUuid.from_path("tex/albedo_loop.png"), "tex/albedo_loop.png", srv.current_tick)
    var slot = Int(th.slot())
    srv.streamer.enqueue_mip_upload(slot, 3, 256)
    var drained = srv.streamer.drain_uploads_within_budget(1024)
    ctx.assert_equal(len(drained), 1, "residency streamer drains mip within budget")
    ctx.assert_true(srv.upload_mip_to_gpu(th, 3, 256), "upload_mip_to_gpu updates Vulkan texture slot")
    ctx.assert_true(srv.textures.slot_vram_bytes[slot] > 0, "slot_vram_bytes tracks uploaded GPU mip")

    var mat = MaterialAsset(base_color=Vec3(0.8, 0.8, 0.8), roughness=0.4)
    mat.set_albedo_slot(slot)
    var mat_id = ren.add_material(mat.to_aaa_material())
    var me = ren.add_mesh(create_mock_meshlet_mesh(8), mat_id)
    ren.begin_frame(Vec3(0.0, 0.5, -4.5), Vec3(0.0, 0.0, 0.0))
    ren.draw_instance(me.mesh_id, Mat4(), material_override=mat_id)
    ctx.assert_true(ren.render_frame(capture_readback=True), "frame using streamed bindless texture slot renders")
    srv.close()


# 11. Reverse-Z projection + depth occlusion of near red box over far blue box
def test_t3_reverse_z_depth_bounds(mut ctx: TestContext, mut ren: ScalableRenderer) raises:
    ctx.assert_true(ren.uses_vulkan(), "ScalableRenderer uses Vulkan Tier 2")
    var cam = RenderCameraPacket(eye=Vec3(0.0, 0.0, 4.0), target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0, near_z=0.1)
    var view = make_rh_look_at(cam.eye, cam.target, cam.up)
    var proj = make_rh_reverse_z_infinite_perspective(Float32(55.0 * 0.017453292), Float32(ren.width) / Float32(ren.height), Float32(0.1))
    var vp = proj.multiply(view)
    var p_near = project_rh_world_to_screen(vp, Vec3(0.0, 0.0, 1.0), ren.width, ren.height)
    var p_far = project_rh_world_to_screen(vp, Vec3(0.0, 0.0, -2.0), ren.width, ren.height)
    ctx.assert_true(p_near[3] and p_far[3], "both near and far targets project in front of camera")
    ctx.assert_true(p_near[2] > p_far[2] and p_far[2] > 0.0, "Reverse-Z maps nearer z to larger positive NDC depth")

    var mat_red = MaterialAsset(base_color=Vec3(1.0, 0.02, 0.02), roughness=0.5)
    mat_red.emissive = Vec3(1.5, 0.0, 0.0)
    var m_red_id = ren.register_material(mat_red)
    var mat_blue = MaterialAsset(base_color=Vec3(0.02, 0.02, 1.0), roughness=0.5)
    mat_blue.emissive = Vec3(0.0, 0.0, 1.5)
    var m_blue_id = ren.register_material(mat_blue)

    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.2, 1.2, 0.3), 0.05)
    var mesh_id = ren.register_mesh(mb^.finish(), m_red_id)

    var rw = RenderWorld()
    rw.camera = cam
    rw.sun_radiance = Vec3(0.0, 0.0, 0.0)
    var m_far = trs(Vec3(0.0, 0.0, -2.0), Quat(), Vec3(1, 1, 1))
    var m_near = trs(Vec3(0.0, 0.0, 1.0), Quat(), Vec3(1, 1, 1))
    # Submit near red FIRST, far blue SECOND so Reverse-Z depth test must reject far blue
    rw.instances.append(RenderInstancePacket(1, mesh_id, m_near, m_near, material_id=m_red_id))
    rw.instances.append(RenderInstancePacket(2, mesh_id, m_far, m_far, material_id=m_blue_id))

    var ui = UiDrawList()
    ctx.assert_true(ren.render_world(rw, ui, capture_readback=True), "Reverse-Z depth occlusion frame renders")
    var c_idx = (ren.height // 2) * ren.width + (ren.width // 2)
    var px = ren.readback_pixels()
    var center = px[c_idx]
    var r = Int((center >> 16) & 255)
    var b = Int(center & 255)
    var center_hdr = ren.gpu_slot[0].readback_resolved_hdr()[c_idx]
    ctx.assert_true(center_hdr.x > center_hdr.z * Float32(2.0) and r > b + 40, "Reverse-Z depth test keeps nearer red mesh over farther blue mesh")

    # Swap materials (near blue FIRST, far red SECOND) to confirm depth order rather than draw/material order governs occlusion
    var rw_swap = RenderWorld()
    rw_swap.camera = cam
    rw_swap.sun_radiance = Vec3(0.0, 0.0, 0.0)
    rw_swap.instances.append(RenderInstancePacket(1, mesh_id, m_near, m_near, material_id=m_blue_id))
    rw_swap.instances.append(RenderInstancePacket(2, mesh_id, m_far, m_far, material_id=m_red_id))
    ctx.assert_true(ren.render_world(rw_swap, ui, capture_readback=True), "Reverse-Z swapped depth occlusion frame renders")
    var px2 = ren.readback_pixels()
    var c2 = px2[c_idx]
    var r2 = Int((c2 >> 16) & 255)
    var b2 = Int(c2 & 255)
    var hdr2 = ren.gpu_slot[0].readback_resolved_hdr()[c_idx]
    ctx.assert_true(hdr2.z > hdr2.x * Float32(2.0) and b2 > r2 + 40, "Reverse-Z depth test flips cleanly to nearer blue mesh over farther red mesh")


# 12. MaterialAsset normal/ORM slot SSBO packing + glossy vs matte GPU PBR pixel difference
def test_t3_normal_orm_roughness_derivation(mut ctx: TestContext, mut ren: ScalableRenderer) raises:
    ctx.assert_true(ren.uses_vulkan(), "ScalableRenderer uses Vulkan Tier 2")
    var glossy = MaterialAsset(base_color=Vec3(0.7, 0.7, 0.7), roughness=0.08, metallic=0.9)
    glossy.set_normal_slot(3)
    glossy.set_orm_slot(7)
    var words = glossy.pack_words()
    ctx.assert_equal(len(words), AAA_MATERIAL_WORDS, "MaterialAsset packs to 24 SSBO words")
    ctx.assert_equal(Int((words[23] >> 8) & 0xFF), 3, "normal_slot packed into bits 8..15")
    ctx.assert_equal(Int((words[23] >> 16) & 0xFF), 7, "orm_slot packed into bits 16..23")
    ctx.assert_approx(bitcast[DType.float32](words[4]), 0.08, 0.001, "roughness packed into word 4")

    var matte = MaterialAsset(base_color=Vec3(0.7, 0.7, 0.7), roughness=0.92, metallic=0.0)
    var m_glossy = ren.register_material(glossy)
    var m_matte = ren.register_material(matte)
    var mesh_id = ren.register_mesh(create_mock_meshlet_mesh(12), m_glossy)
    var ui = UiDrawList()

    var rw = RenderWorld()
    rw.camera = RenderCameraPacket(eye=Vec3(0.0, 1.0, 5.0), target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
    rw.cluster_lights.append(AaaClusterLight.point(Vec3(2.0, 2.5, 3.0), Vec3(60.0, 60.0, 60.0), 20.0))
    rw.instances.append(RenderInstancePacket(1, mesh_id, Mat4(), Mat4(), material_id=m_glossy))
    ctx.assert_true(ren.render_world(rw, ui, capture_readback=True), "glossy PBR frame renders")
    var px_g = ren.readback_pixels()

    rw.instances[0].material_id = m_matte
    ctx.assert_true(ren.render_world(rw, ui, capture_readback=True), "matte PBR frame renders")
    var px_m = ren.readback_pixels()

    var diff = compare_pixels(px_g, px_m, ren.width, ren.height, 4)
    ctx.assert_true(diff.differing > 50, "glossy metallic vs matte dielectric MaterialAsset renders distinct shading")


# 13. Dynamic resolution scaling (`ren.resize`) + AaaPhysicalCamera CoC aspect ratio scaling
def test_t3_drs_aspect_preservation(mut ctx: TestContext, mut ren: ScalableRenderer) raises:
    ctx.assert_true(ren.uses_vulkan(), "ScalableRenderer uses Vulkan Tier 2")
    var cam = AaaPhysicalCamera(aperture_fstop=2.8, shutter_seconds=0.01, focal_length_mm=50.0)
    var coc_hi = cam.compute_coc_scale_px(360)
    var coc_lo = cam.compute_coc_scale_px(180)
    ctx.assert_approx(coc_lo / coc_hi, 0.5, 0.01, "CoC scales proportionally with vertical resolution")

    var mat_id = ren.register_material(MaterialAsset(base_color=Vec3(0.4, 0.8, 0.5), roughness=0.4))
    var mesh_id = ren.register_mesh(create_mock_meshlet_mesh(8), mat_id)
    var rw = RenderWorld()
    rw.camera = RenderCameraPacket(eye=Vec3(0.0, 1.0, 5.0), target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
    rw.instances.append(RenderInstancePacket(1, mesh_id, Mat4(), Mat4(), material_id=mat_id))
    var ui = UiDrawList()

    ctx.assert_true(ren.resize(320, 180), "resize to 320x180 succeeds")
    ctx.assert_true(ren.render_world(rw, ui, capture_readback=True), "320x180 frame renders")
    ctx.assert_equal(len(ren.readback_pixels()), 320 * 180, "320x180 readback matches resized framebuffer")

    ctx.assert_true(ren.resize(160, 120), "resize back to 160x120 succeeds")
    ctx.assert_true(ren.render_world(rw, ui, capture_readback=True), "160x120 frame renders")
    ctx.assert_equal(len(ren.readback_pixels()), 160 * 120, "160x120 readback matches restored framebuffer")


# 14. AaaRenderGraph cross-queue barrier emission + vat_crowd_eval_shader + live GPU barriers
def test_t3_spirv_barrier_coherence(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active")
    ctx.assert_true(validate_spirv_module(vat_crowd_eval_shader()).valid, "vat_crowd_eval_shader validates SPIR-V")

    var rg = AaaRenderGraph()
    var r_shared = rg.add_resource("cross_q_buf", 4096, transient=True)
    var a0 = List[RgResourceAccess](); a0.append(RgResourceAccess(r_shared, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("async_compute_gen", a0, RG_QUEUE_ASYNC_COMPUTE)
    var a1 = List[RgResourceAccess](); a1.append(RgResourceAccess(r_shared, RG_USAGE_FRAGMENT_READ))
    _ = rg.add_pass("gfx_consume", a1, RG_QUEUE_GRAPHICS)
    ctx.assert_true(rg.compile(1024 * 1024), "cross-queue graph compiles")
    ctx.assert_true(rg.aliasing_metrics().compiled_barrier_count > 0, "cross-queue handoff emits compiled barriers")

    ren.begin_frame(Vec3(0.0, 1.0, -4.0), Vec3(0.0, 0.0, 0.0))
    ctx.assert_true(ren.render_frame(), "live GPU frame renders")
    ctx.assert_true(ren.render_graph.barriers_emitted_last_frame > 0, "live GPU RenderGraph emits synchronization barriers")


# 15. UiContext scroll view + DebugLineBatch AABB projection -> ScalableRenderer UI overlay
def test_t3_ui_tile_binned_scissor(mut ctx: TestContext, mut ren: ScalableRenderer) raises:
    ctx.assert_true(ren.uses_vulkan(), "ScalableRenderer uses Vulkan Tier 2")
    var ui = UiContext()
    var inp = InputState()
    ui.begin_frame(inp, Float32(ren.width), Float32(ren.height))
    ui.begin_scroll_view("t3_tile_ui", Float32(120.0), Float32(80.0), Float32(10 * 18))
    var vis = ui.scroll_visible_range(10, Float32(18.0))
    for i in range(vis[0], vis[1]):
        _ = ui.button("item_" + String(i), "Row", Float32(50.0), Float32(16.0))
    ui.end_scroll_view()
    ui.end_frame()

    var rw = RenderWorld()
    rw.camera = RenderCameraPacket(eye=Vec3(0.0, 1.0, 5.0), target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
    var dbg = DebugLineBatch()
    dbg.add_aabb(Vec3(-1.0, -1.0, -1.0), Vec3(1.0, 1.0, 1.0), 0xFF30FF60, Float32(2.0))
    var emitted = dbg.emit_to_ui_draw_list(ui.draw_list, rw.camera.eye, rw.camera.target, rw.camera.fov_y_deg, rw.camera.near_z, ren.width, ren.height)
    ctx.assert_true(emitted >= 8, "DebugLineBatch projects 3D AABB segments into UiDrawList")
    ctx.assert_true(len(ui.draw_list.prims) > 8, "UiDrawList holds both scroll widgets and debug lines")

    ctx.assert_true(ren.render_world(rw, ui.draw_list, capture_readback=True), "tile-binned UI + DebugLineBatch frame renders")
    var px = ren.readback_pixels()
    ctx.assert_equal(len(px), ren.width * ren.height, "readback captures full viewport")
    ctx.assert_true(_pixel_variance(px) > 30, "composited UI & 3D debug wireframe produce distinct framebuffer pixels")


# =========================================================================
# Aggregator for Tier 3 Render Pipeline
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T3_CrossFeature_RenderPipeline")
    var ren = ScalableRenderer(tier=RENDER_TIER_2_AAA_VISBUFFER_RT, width=160, height=120)

    try: test_t3_meshlet_cache_interaction(ctx, ren.gpu_slot[0]); ctx.record_pass("test_t3_meshlet_cache_interaction") except e: ctx.record_fail("test_t3_meshlet_cache_interaction", String(e))
    try: test_t3_material_clustered_light_interaction(ctx, ren); ctx.record_pass("test_t3_material_clustered_light_interaction") except e: ctx.record_fail("test_t3_material_clustered_light_interaction", String(e))
    try: test_t3_vsm_hiz_interaction(ctx, ren.gpu_slot[0]); ctx.record_pass("test_t3_vsm_hiz_interaction") except e: ctx.record_fail("test_t3_vsm_hiz_interaction", String(e))
    try: test_t3_rendergraph_aliasing_compilation(ctx, ren.gpu_slot[0]); ctx.record_pass("test_t3_rendergraph_aliasing_compilation") except e: ctx.record_fail("test_t3_rendergraph_aliasing_compilation", String(e))
    try: test_t3_svgf_normal_blend_filtering(ctx, ren.gpu_slot[0]); ctx.record_pass("test_t3_svgf_normal_blend_filtering") except e: ctx.record_fail("test_t3_svgf_normal_blend_filtering", String(e))
    try: test_t3_spirv_clustered_binning_kernel(ctx, ren.gpu_slot[0]); ctx.record_pass("test_t3_spirv_clustered_binning_kernel") except e: ctx.record_fail("test_t3_spirv_clustered_binning_kernel", String(e))
    try: test_t3_qem_simplification_lod_error(ctx, ren); ctx.record_pass("test_t3_qem_simplification_lod_error") except e: ctx.record_fail("test_t3_qem_simplification_lod_error", String(e))
    try: test_t3_widget_canvas_clipping_interaction(ctx, ren); ctx.record_pass("test_t3_widget_canvas_clipping_interaction") except e: ctx.record_fail("test_t3_widget_canvas_clipping_interaction", String(e))
    try: test_t3_multi_light_radiance_accumulation(ctx, ren); ctx.record_pass("test_t3_multi_light_radiance_accumulation") except e: ctx.record_fail("test_t3_multi_light_radiance_accumulation", String(e))
    try: test_t3_streamer_budget_feedback_loop(ctx, ren.gpu_slot[0]); ctx.record_pass("test_t3_streamer_budget_feedback_loop") except e: ctx.record_fail("test_t3_streamer_budget_feedback_loop", String(e))
    try: test_t3_reverse_z_depth_bounds(ctx, ren); ctx.record_pass("test_t3_reverse_z_depth_bounds") except e: ctx.record_fail("test_t3_reverse_z_depth_bounds", String(e))
    try: test_t3_normal_orm_roughness_derivation(ctx, ren); ctx.record_pass("test_t3_normal_orm_roughness_derivation") except e: ctx.record_fail("test_t3_normal_orm_roughness_derivation", String(e))
    try: test_t3_drs_aspect_preservation(ctx, ren); ctx.record_pass("test_t3_drs_aspect_preservation") except e: ctx.record_fail("test_t3_drs_aspect_preservation", String(e))
    try: test_t3_spirv_barrier_coherence(ctx, ren.gpu_slot[0]); ctx.record_pass("test_t3_spirv_barrier_coherence") except e: ctx.record_fail("test_t3_spirv_barrier_coherence", String(e))
    try: test_t3_ui_tile_binned_scissor(ctx, ren); ctx.record_pass("test_t3_ui_tile_binned_scissor") except e: ctx.record_fail("test_t3_ui_tile_binned_scissor", String(e))

    ren.shutdown()


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
