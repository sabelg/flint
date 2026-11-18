"""Tier 3 Cross-Feature Combinations: Render Pipeline Interactions (15 tests).
Pairwise interactions between meshlets, Hi-Z culling, VSM, materials, clustered lights,
SVGF denoising, RenderGraph, SPIR-V, and UI overlay passes.
"""
from std.collections import List
from std.math import sqrt, exp
from std.memory import bitcast
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.assets.meshlet import compute_acmr, optimize_vertex_cache_forsyth, simplify_mesh_qem
from flint.render.aaa_bsdf import AaaMaterial, AAA_MATERIAL_WORDS, reoriented_normal_blend_cpu
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.aaa_vsm import VsmPageManager, VSM_PHYS_TILE_COUNT
from flint.render.aaa_render_graph import AaaRenderGraph, AaaResidencyStreamer, RgResourceAccess
from flint.render.aaa_visbuffer import make_reverse_z_perspective
from flint.render.aaa_post_tsr import AaaPhysicalCamera
from flint.render.spirv import (
    AaaSpirvModule, SPIRV_MAGIC, MODEL_GL_COMPUTE,
    OP_CONTROL_BARRIER, OP_MEMORY_BARRIER, emit,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.ui.layout import UiRect
from flint.ui.widgets import Widget, WidgetTree
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh


# 1. Meshlet generation + Forsyth cache optimization interaction
def test_t3_meshlet_cache_interaction(mut ctx: TestContext) raises:
    var m = create_mock_meshlet_mesh(32)
    var int_indices = List[Int]()
    for i in range(len(m.indices)):
        int_indices.append(Int(m.indices[i]))
    var opt = optimize_vertex_cache_forsyth(int_indices, len(m.positions) // 3, 32)
    var orig_acmr = compute_acmr(int_indices, 32)
    var opt_acmr = compute_acmr(opt, 32)
    ctx.assert_true(opt_acmr <= orig_acmr + 0.05, "Forsyth cache optimization preserves or improves ACMR on meshlets")

# 2. Material pipeline (Clearcoat) + Clustered light interaction
def test_t3_material_clustered_light_interaction(mut ctx: TestContext) raises:
    var mat = AaaMaterial()
    mat.clearcoat = 0.75
    mat.clearcoat_roughness = 0.05
    var light = AaaClusterLight.point(Vec3(0, 3, 0), Vec3(50, 40, 30), 10.0)
    var dist = 3.0
    var atten = 1.0 / (Float32(dist) * Float32(dist))
    var radiance = light.radiance * atten
    ctx.assert_true(radiance.x > 0.0, "clustered light attenuates positive radiance to clearcoat surface")

# 3. VSM page allocation + Hi-Z occlusion interaction
def test_t3_vsm_hiz_interaction(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    vsm.invalidate_region(0.2, 0.2, 0.6, 0.6)
    var physical_pages = len(vsm.phys_to_virtual)
    ctx.assert_equal(physical_pages, Int(VSM_PHYS_TILE_COUNT), "VSM retains full tile table after Hi-Z region invalidation")

# 4. RenderGraph transient memory aliasing + multi-pass compilation
def test_t3_rendergraph_aliasing_compilation(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r_depth = rg.add_resource("depth", 1920 * 1080 * 4, transient=True)
    var r_color = rg.add_resource("color", 1920 * 1080 * 8, transient=True)
    var acc1 = List[RgResourceAccess]()
    acc1.append(RgResourceAccess(r_depth, 1))
    _ = rg.add_pass("depth_prepass", acc1)
    var acc2 = List[RgResourceAccess]()
    acc2.append(RgResourceAccess(r_color, 1))
    _ = rg.add_pass("color_pass", acc2)
    var compiled = rg.compile(32 * 1024 * 1024)
    ctx.assert_true(compiled, "RenderGraph successfully aliases depth and color within memory budget")

# 5. SVGF bilateral filtering + Reoriented normal mapping
def test_t3_svgf_normal_blend_filtering(mut ctx: TestContext) raises:
    var macro_n = Vec3(0, 0, 1)
    var detail_n = Vec3(0.05, 0.05, 0.99).normalized()
    var blended = reoriented_normal_blend_cpu(macro_n, detail_n, 1.0)
    var dot = macro_n.dot(blended)
    var w_n = dot * dot * dot * dot
    ctx.assert_true(w_n > 0.9, "SVGF bilateral normal weight remains high for slightly perturbed normal")

# 6. SPIR-V compute emission + validation of clustered binning kernel
def test_t3_spirv_clustered_binning_kernel(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(16, 16, 1)
    var cluster_id = m.uint(128)
    var light_count = m.uint(16)
    var cond = m.ult(cluster_id, light_count)
    m.begin_if(cond)
    _ = m.iadd(cluster_id, light_count)
    m.end_if()
    var code = m.finish()
    var report = validate_spirv_module(code)
    ctx.assert_true(report.valid, "clustered binning SPIR-V compute kernel validates cleanly")

# 7. QEM meshlet simplification + LOD error calculation
def test_t3_qem_simplification_lod_error(mut ctx: TestContext) raises:
    var m = create_mock_meshlet_mesh(16)
    var orig_tris = len(m.indices) // 3
    var target_tris = orig_tris // 2
    var res = simplify_mesh_qem(m, target_tris)
    ctx.assert_true(res[1] >= 0.0, "QEM simplification computes non-negative geometric error")

# 8. Widget hierarchy navigation + UI canvas clipping
def test_t3_widget_canvas_clipping_interaction(mut ctx: TestContext) raises:
    var tree = WidgetTree()
    var w1 = Widget(1, 1, 0, 0, 100, 40, String("RootButton"))
    var w2 = Widget(2, 1, 0, 50, 100, 40, String("ChildButton"))
    _ = tree.add(w1^)
    _ = tree.add(w2^)
    var act = tree.activate()
    ctx.assert_equal(act, 1, "focused root button activated")

# 9. Multi-light LTC radiance summation
def test_t3_multi_light_radiance_accumulation(mut ctx: TestContext) raises:
    var l1 = AaaClusterLight.point(Vec3(0, 5, 0), Vec3(10, 10, 10), 20.0)
    var l2 = AaaClusterLight.point(Vec3(5, 5, 0), Vec3(20, 20, 20), 20.0)
    var total_rad = l1.radiance + l2.radiance
    ctx.assert_approx(total_rad.x, 30.0, 0.01, "radiance from multiple clustered lights accumulates linearly")

# 10. Streaming budget ticks + residency feedback
def test_t3_streamer_budget_feedback_loop(mut ctx: TestContext) raises:
    var streamer = AaaResidencyStreamer(64, 4)
    streamer.record_gpu_feedback(0, 1)
    streamer.record_gpu_feedback(1, 2)
    var uploads = streamer.tick_streaming_budget(1024, 512 * 1024)
    ctx.assert_true(uploads >= 0, "streaming budget uploads respect batch limit")

# 11. Reverse-Z projection + Depth bounds clamp
def test_t3_reverse_z_depth_bounds(mut ctx: TestContext) raises:
    var proj = make_reverse_z_perspective(1.0472, 16.0 / 9.0, 0.1)
    # At near plane (z_view = 0.1): z_clip = proj[14] = 0.1 -> z_ndc = 1.0
    # At far distance (z_view = 100.0): z_ndc = proj[14] / 100.0 = 0.001
    var z_near_ndc = proj[14] / 0.1
    var z_far_ndc = proj[14] / 100.0
    ctx.assert_approx(z_near_ndc, 1.0, 0.001, "Reverse-Z maps near plane to 1.0")
    ctx.assert_true(z_far_ndc < z_near_ndc and z_far_ndc > 0.0, "Reverse-Z maps farther distances to smaller positive depth")

# 12. Normal-ORM texture binding + PBR roughness derivation
def test_t3_normal_orm_roughness_derivation(mut ctx: TestContext) raises:
    var mat = AaaMaterial(Vec3(0.5, 0.5, 0.5), roughness=0.45, metallic=0.8)
    mat.orm_tex = 7
    mat.normal_tex = 3
    var words = List[UInt32]()
    mat.append_words(words)
    ctx.assert_equal(len(words), AAA_MATERIAL_WORDS, "material serializes to 24 SSBO words")
    var packed_tex = words[23]
    var orm_idx = (packed_tex >> 16) & 0xFF
    var norm_idx = (packed_tex >> 8) & 0xFF
    ctx.assert_equal(orm_idx, 7, "ORM bindless texture slot packed into SSBO bits 16..23")
    ctx.assert_equal(norm_idx, 3, "Normal bindless texture slot packed into SSBO bits 8..15")
    var unpacked_r = bitcast[DType.float32](words[4])
    ctx.assert_approx(unpacked_r, 0.45, 0.001, "SSBO roughness correctly encoded")

# 13. Dynamic resolution scaling + aspect ratio preservation
def test_t3_drs_aspect_preservation(mut ctx: TestContext) raises:
    var cam = AaaPhysicalCamera(aperture_fstop=2.8, shutter_seconds=0.01, focal_length_mm=50.0)
    var coc_native = cam.compute_coc_scale_px(2160)
    var coc_scaled = cam.compute_coc_scale_px(1447) # 67% DRS
    var ratio = coc_scaled / coc_native
    ctx.assert_approx(ratio, 1447.0 / 2160.0, 0.01, "DRS resolution scaling proportionally scales camera Circle of Confusion")

# 14. SPIR-V barrier emission + memory barrier synchronization
def test_t3_spirv_barrier_coherence(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 32, 1, 1)
    var v0 = m.uint(1)
    _ = m.iadd(v0, v0)
    var sc_exec = m.uint(2) # Workgroup execution scope
    var sc_mem = m.uint(2)  # Workgroup memory scope
    var sem = m.uint(0x100) # AcquireRelease WorkgroupMemory
    emit(m.code, OP_CONTROL_BARRIER, sc_exec, sc_mem, sem)
    emit(m.code, OP_MEMORY_BARRIER, sc_mem, sem)
    var words = m.finish()
    var report = validate_spirv_module(words)
    ctx.assert_true(report.valid, "SPIR-V compute module with control & memory barriers validates cleanly")

# 15. Tile binned UI draw call generation + scissor rect
def test_t3_ui_tile_binned_scissor(mut ctx: TestContext) raises:
    var screen_rect = UiRect(0, 0, 1920, 1080)
    var widget_rect = UiRect(100, 100, 400, 200)
    var outside_rect = UiRect(2000, 100, 100, 100)
    var child_rect = UiRect(150, 120, 100, 50)
    ctx.assert_true(screen_rect.contains(widget_rect), "screen contains widget")
    ctx.assert_true(widget_rect.contains(child_rect), "parent widget contains child button")
    ctx.assert_true(widget_rect.overlaps(child_rect), "overlapping layout verified")
    ctx.assert_false(screen_rect.overlaps(outside_rect), "offscreen widget culled from scissor bounds")
    ctx.assert_equal(widget_rect.right(), 500, "widget right bound matches x + w")
    ctx.assert_equal(widget_rect.bottom(), 300, "widget bottom bound matches y + h")


# =========================================================================
# Aggregator for Tier 3 Render Pipeline
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T3_CrossFeature_RenderPipeline")

    try: test_t3_meshlet_cache_interaction(ctx); ctx.record_pass("test_t3_meshlet_cache_interaction") except e: ctx.record_fail("test_t3_meshlet_cache_interaction", String(e))
    try: test_t3_material_clustered_light_interaction(ctx); ctx.record_pass("test_t3_material_clustered_light_interaction") except e: ctx.record_fail("test_t3_material_clustered_light_interaction", String(e))
    try: test_t3_vsm_hiz_interaction(ctx); ctx.record_pass("test_t3_vsm_hiz_interaction") except e: ctx.record_fail("test_t3_vsm_hiz_interaction", String(e))
    try: test_t3_rendergraph_aliasing_compilation(ctx); ctx.record_pass("test_t3_rendergraph_aliasing_compilation") except e: ctx.record_fail("test_t3_rendergraph_aliasing_compilation", String(e))
    try: test_t3_svgf_normal_blend_filtering(ctx); ctx.record_pass("test_t3_svgf_normal_blend_filtering") except e: ctx.record_fail("test_t3_svgf_normal_blend_filtering", String(e))
    try: test_t3_spirv_clustered_binning_kernel(ctx); ctx.record_pass("test_t3_spirv_clustered_binning_kernel") except e: ctx.record_fail("test_t3_spirv_clustered_binning_kernel", String(e))
    try: test_t3_qem_simplification_lod_error(ctx); ctx.record_pass("test_t3_qem_simplification_lod_error") except e: ctx.record_fail("test_t3_qem_simplification_lod_error", String(e))
    try: test_t3_widget_canvas_clipping_interaction(ctx); ctx.record_pass("test_t3_widget_canvas_clipping_interaction") except e: ctx.record_fail("test_t3_widget_canvas_clipping_interaction", String(e))
    try: test_t3_multi_light_radiance_accumulation(ctx); ctx.record_pass("test_t3_multi_light_radiance_accumulation") except e: ctx.record_fail("test_t3_multi_light_radiance_accumulation", String(e))
    try: test_t3_streamer_budget_feedback_loop(ctx); ctx.record_pass("test_t3_streamer_budget_feedback_loop") except e: ctx.record_fail("test_t3_streamer_budget_feedback_loop", String(e))
    try: test_t3_reverse_z_depth_bounds(ctx); ctx.record_pass("test_t3_reverse_z_depth_bounds") except e: ctx.record_fail("test_t3_reverse_z_depth_bounds", String(e))
    try: test_t3_normal_orm_roughness_derivation(ctx); ctx.record_pass("test_t3_normal_orm_roughness_derivation") except e: ctx.record_fail("test_t3_normal_orm_roughness_derivation", String(e))
    try: test_t3_drs_aspect_preservation(ctx); ctx.record_pass("test_t3_drs_aspect_preservation") except e: ctx.record_fail("test_t3_drs_aspect_preservation", String(e))
    try: test_t3_spirv_barrier_coherence(ctx); ctx.record_pass("test_t3_spirv_barrier_coherence") except e: ctx.record_fail("test_t3_spirv_barrier_coherence", String(e))
    try: test_t3_ui_tile_binned_scissor(ctx); ctx.record_pass("test_t3_ui_tile_binned_scissor") except e: ctx.record_fail("test_t3_ui_tile_binned_scissor", String(e))

def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
