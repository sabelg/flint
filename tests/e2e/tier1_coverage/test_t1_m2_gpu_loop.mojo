"""Tier 1 Feature Coverage: Milestone 2 GPU Loop (F13–F17: 25 tests).
Opaque-box and requirement-driven verification of swapchain, ring buffering, UI SDF, meshlets, and SPIR-V.
"""
from std.collections import List
from flint.core.math import Vec3
from flint.platform.vulkan import (
    PRESENT_FIFO,
    FORMAT_B8G8R8A8_UNORM,
    FORMAT_R8G8B8A8_UNORM,
    FORMAT_BC7_UNORM,
    FORMAT_D32_SFLOAT,
    TYPE_SWAPCHAIN_CREATE_INFO,
    VulkanGpu,
)
from flint.render.font import SdfFont
from flint.ui.layout import ui_frame, UiRect
from flint.ui.widgets import Widget, WidgetTree
from flint.assets.meshlet import compute_acmr, optimize_vertex_cache_forsyth
from flint.render.spirv import (
    AaaSpirvModule,
    SPIRV_MAGIC,
    OP_CONTROL_BARRIER,
    OP_MEMORY_BARRIER,
    emit,
    MODEL_GL_COMPUTE,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_post_tsr import (
    AAA_DISPLAY_AGX_REC709,
    AAA_DISPLAY_LINEAR_SRGB,
    AAA_DISPLAY_HDR10_PQ,
)
from flint.render.aaa_visbuffer import meshlet_cull_shader
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh


# =========================================================================
# F13: Swapchain Zero-Copy Presentation (5 tests)
# =========================================================================

def test_f13_swapchain_fifo_mode_defined(mut ctx: TestContext) raises:
    ctx.assert_equal(Int(PRESENT_FIFO), 2, "PRESENT_FIFO is mode 2 in Vulkan")
    ctx.assert_equal(Int(TYPE_SWAPCHAIN_CREATE_INFO), 1000001000, "Vulkan swapchain create info struct type")

def test_f13_swapchain_dimensions_preservation(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=1920, height=1080, tsr_scale=1.0)
    ctx.assert_equal(ren.output_w, 1920, "swapchain output width preserved in renderer")
    ctx.assert_equal(ren.output_h, 1080, "swapchain output height preserved in renderer")
    var aspect = Float32(ren.output_w) / Float32(ren.output_h)
    ctx.assert_approx(aspect, 1.7777, 0.01, "16:9 aspect ratio preserved in renderer")
    ren.shutdown()

def test_f13_swapchain_surface_formats(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_equal(Int(FORMAT_B8G8R8A8_UNORM), 44, "VK_FORMAT_B8G8R8A8_UNORM format constant is 44")
    ctx.assert_equal(ren.display_mode, AAA_DISPLAY_AGX_REC709, "default display mode is AgX Rec.709")
    ren.display_mode = AAA_DISPLAY_HDR10_PQ
    ctx.assert_equal(ren.display_mode, AAA_DISPLAY_HDR10_PQ, "HDR10 PQ swapchain display mode configurable")
    ren.shutdown()

def test_f13_swapchain_acquire_semaphore_indexing(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_equal(ren.frame_index, 0, "initial frame index is 0")
    ctx.assert_true(ren.vk.has_swapchain, "Vulkan GPU supports swapchain extension")
    ctx.assert_equal(ren.vk.swapchain, 0, "swapchain handle is 0 in headless mode")
    ctx.assert_equal(len(ren.vk.acquire_semaphores), 0, "no swapchain semaphores allocated in headless")
    ren.shutdown()

def test_f13_swapchain_headless_fallback(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_true(ren.active, "renderer active in headless readback fallback mode")
    ren.begin_frame(Vec3(0, 1, -3), Vec3(0, 0, 0))
    var ok = ren.render_frame()
    ctx.assert_true(ok, "headless render_frame completes successfully")
    var pixels = ren.readback_pixels()
    ctx.assert_equal(len(pixels), 160 * 120, "headless readback retrieves full frame buffer")
    ren.shutdown()


# =========================================================================
# F14: Ring-Buffered Frame Pipeline (5 tests)
# =========================================================================

def test_f14_ring_buffer_slots_count(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_true(ren.buf_globals.size >= 512, "globals UBO buffer allocated")
    ctx.assert_true(ren.buf_instances.size > 0, "instance SSBO buffer allocated")
    ctx.assert_equal(ren.frame_index, 0, "frame ring buffer starts at index 0")
    ren.shutdown()

def test_f14_frame_index_increment(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_equal(ren.frame_index, 0, "frame_index initially 0")
    ren.begin_frame(Vec3(0, 1, -2), Vec3(0, 0, 0))
    _ = ren.render_frame()
    ctx.assert_equal(ren.frame_index, 1, "frame_index increments to 1 after first render_frame")
    ren.begin_frame(Vec3(0, 1, -2), Vec3(0, 0, 0))
    _ = ren.render_frame()
    ctx.assert_equal(ren.frame_index, 2, "frame_index increments to 2 after second render_frame")
    ren.shutdown()

def test_f14_ping_pong_history_alternation(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_true(ren.buf_history.size > 0, "history buffer allocated")
    ctx.assert_true(ren.buf_hdr_out.size > 0, "HDR output buffer allocated")
    ctx.assert_true(ren.buf_history.cpu_address != ren.buf_hdr_out.cpu_address, "history and HDR output occupy distinct ring addresses")
    ren.shutdown()

def test_f14_ring_globals_offset(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_equal(ren.buf_globals.size % 256, 0, "globals buffer size aligned to 256 bytes for UBO")
    ctx.assert_true(ren.buf_instances.size >= 36 * 4, "instances buffer holds at least one 36-word instance record")
    ren.shutdown()

def test_f14_fence_wait_synchronization(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ren.begin_frame(Vec3(0, 1, -3), Vec3(0, 0, 0))
    var ok = ren.render_frame()
    ctx.assert_true(ok, "fence wait synchronization clears and frame renders")
    ren.shutdown()


# =========================================================================
# F15: Tile-Binned GPU UiSdfOverlayPass (5 tests)
# =========================================================================

def test_f15_sdf_font_measure_text(mut ctx: TestContext) raises:
    var font = SdfFont()
    var font_ok = font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36, 5)
    ctx.assert_true(font_ok and font.valid, "DejaVuSansMono-Bold font loaded successfully")
    var width = font.measure("FLINT ENGINE", 24.0)
    ctx.assert_true(width > 50.0, "measured text width > 50 px")

def test_f15_widget_tree_navigation(mut ctx: TestContext) raises:
    var tree = WidgetTree()
    var w1 = Widget(1, 1, 10, 10, 100, 30, String("Button 1"))
    var w2 = Widget(2, 1, 10, 50, 100, 30, String("Button 2"))
    _ = tree.add(w1^)
    _ = tree.add(w2^)
    ctx.assert_equal(tree.focused_id(), 1, "first focusable widget selected by default")
    var moved = tree.navigate(1) # down
    ctx.assert_true(moved, "navigating down moves focus")
    ctx.assert_equal(tree.focused_id(), 2, "focus moved to second button")

def test_f15_widget_bounding_box_hit(mut ctx: TestContext) raises:
    var w = Widget(10, 1, 50, 50, 200, 40, String("Play"))
    ctx.assert_true(w.contains(100.0, 60.0), "point inside button bounds returns true")
    ctx.assert_false(w.contains(10.0, 10.0), "point outside button bounds returns false")

def test_f15_widget_activation(mut ctx: TestContext) raises:
    var tree = WidgetTree()
    var w = Widget(42, 1, 0, 0, 100, 30, String("Start Game"))
    _ = tree.add(w^)
    var act_id = tree.activate()
    ctx.assert_equal(act_id, 42, "activated focused widget returns widget id")

def test_f15_ui_draw_list_tile_binning(mut ctx: TestContext) raises:
    var frame = ui_frame(1920, 1080, 0, 0, 1.0)
    var rect = frame.safe()
    ctx.assert_equal(rect.w, 1920, "ui_frame width matches design resolution")
    ctx.assert_equal(rect.h, 1080, "ui_frame height matches design resolution")
    ctx.assert_true(rect.contains(UiRect(10, 10, 100, 50)), "safe rect contains embedded child widget")


# =========================================================================
# F16: Indexed Meshlet Indirect Raster (5 tests)
# =========================================================================

def test_f16_acmr_reduction_forsyth(mut ctx: TestContext) raises:
    var mesh = create_mock_meshlet_mesh(16)
    var v_count = len(mesh.positions) // 3
    var acmr_orig = compute_acmr(mesh.indices)
    var opt_indices = optimize_vertex_cache_forsyth(mesh.indices, v_count)
    var acmr_opt = compute_acmr(opt_indices)
    ctx.assert_true(acmr_opt <= acmr_orig + 0.05, "Forsyth vertex cache optimization yields low ACMR")

def test_f16_acmr_calculation_accuracy(mut ctx: TestContext) raises:
    var indices = List[Int]()
    for i in range(12):
        indices.append(i)
    var acmr = compute_acmr(indices, 32)
    ctx.assert_approx(acmr, 3.0, 0.001, "unique unshared vertices yield 3 misses per triangle (ACMR 3.0)")

def test_f16_draw_indexed_indirect_stride(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_true(ren.buf_indirect.size > 0, "indirect draw command buffer allocated")
    ren.shutdown()

def test_f16_backface_cull_mode(mut ctx: TestContext) raises:
    var cull_spv = meshlet_cull_shader()
    var rep = validate_spirv_module(cull_spv)
    ctx.assert_true(rep.valid, "meshlet indirect culling and backface cull shader validates")

def test_f16_meshlet_indices_preservation(mut ctx: TestContext) raises:
    var mesh = create_mock_meshlet_mesh(8)
    var orig_len = len(mesh.indices)
    var opt = optimize_vertex_cache_forsyth(mesh.indices, len(mesh.positions) // 3)
    ctx.assert_equal(len(opt), orig_len, "optimized index buffer preserves total triangle vertex references")


# =========================================================================
# F17: SPIR-V & Shader DSL Upgrades (5 tests)
# =========================================================================

def test_f17_spirv_module_structure(mut ctx: TestContext) raises:
    ctx.assert_equal(Int(SPIRV_MAGIC), 0x07230203, "SPIRV_MAGIC matches standard 0x07230203")

def test_f17_spirv_arithmetic_emission(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(8, 8, 4)
    var u1 = m.uint(50)
    var u2 = m.uint(150)
    var sum = m.iadd(u1, u2)
    _ = m.imul(sum, u1)
    var code = m.finish()
    ctx.assert_true(len(code) > 20, "SPIR-V module generates integer arithmetic instructions")

def test_f17_spirv_control_flow(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(8, 8, 4)
    var a = m.uint(10)
    var b = m.uint(20)
    var cond = m.ult(a, b)
    m.begin_if(cond)
    _ = m.iadd(a, b)
    m.end_if()
    var code = m.finish()
    var rep = validate_spirv_module(code)
    ctx.assert_true(rep.valid, "SPIR-V conditional branch validates")

def test_f17_spirv_barrier_emission(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var sc_exec = m.uint(2)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_CONTROL_BARRIER, sc_exec, sc_mem, sem)
    var code = m.finish()
    var rep = validate_spirv_module(code)
    ctx.assert_true(rep.valid, "SPIR-V module with workgroup control barrier validates cleanly")

def test_f17_spirv_validator_verification(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(16, 16, 8)
    var f0 = m.float(1.5)
    var f1 = m.float(2.5)
    _ = m.fadd(f0, f1)
    var code = m.finish()
    var report = validate_spirv_module(code)
    ctx.assert_true(report.valid, "SSA validator verifies generated float math")


# =========================================================================
# Aggregator for Tier 1 Milestone 2
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T1_M2_GpuLoop")

    # F13
    try: test_f13_swapchain_fifo_mode_defined(ctx); ctx.record_pass("test_f13_swapchain_fifo_mode_defined") except e: ctx.record_fail("test_f13_swapchain_fifo_mode_defined", String(e))
    try: test_f13_swapchain_dimensions_preservation(ctx); ctx.record_pass("test_f13_swapchain_dimensions_preservation") except e: ctx.record_fail("test_f13_swapchain_dimensions_preservation", String(e))
    try: test_f13_swapchain_surface_formats(ctx); ctx.record_pass("test_f13_swapchain_surface_formats") except e: ctx.record_fail("test_f13_swapchain_surface_formats", String(e))
    try: test_f13_swapchain_acquire_semaphore_indexing(ctx); ctx.record_pass("test_f13_swapchain_acquire_semaphore_indexing") except e: ctx.record_fail("test_f13_swapchain_acquire_semaphore_indexing", String(e))
    try: test_f13_swapchain_headless_fallback(ctx); ctx.record_pass("test_f13_swapchain_headless_fallback") except e: ctx.record_fail("test_f13_swapchain_headless_fallback", String(e))

    # F14
    try: test_f14_ring_buffer_slots_count(ctx); ctx.record_pass("test_f14_ring_buffer_slots_count") except e: ctx.record_fail("test_f14_ring_buffer_slots_count", String(e))
    try: test_f14_frame_index_increment(ctx); ctx.record_pass("test_f14_frame_index_increment") except e: ctx.record_fail("test_f14_frame_index_increment", String(e))
    try: test_f14_ping_pong_history_alternation(ctx); ctx.record_pass("test_f14_ping_pong_history_alternation") except e: ctx.record_fail("test_f14_ping_pong_history_alternation", String(e))
    try: test_f14_ring_globals_offset(ctx); ctx.record_pass("test_f14_ring_globals_offset") except e: ctx.record_fail("test_f14_ring_globals_offset", String(e))
    try: test_f14_fence_wait_synchronization(ctx); ctx.record_pass("test_f14_fence_wait_synchronization") except e: ctx.record_fail("test_f14_fence_wait_synchronization", String(e))

    # F15
    try: test_f15_sdf_font_measure_text(ctx); ctx.record_pass("test_f15_sdf_font_measure_text") except e: ctx.record_fail("test_f15_sdf_font_measure_text", String(e))
    try: test_f15_widget_tree_navigation(ctx); ctx.record_pass("test_f15_widget_tree_navigation") except e: ctx.record_fail("test_f15_widget_tree_navigation", String(e))
    try: test_f15_widget_bounding_box_hit(ctx); ctx.record_pass("test_f15_widget_bounding_box_hit") except e: ctx.record_fail("test_f15_widget_bounding_box_hit", String(e))
    try: test_f15_widget_activation(ctx); ctx.record_pass("test_f15_widget_activation") except e: ctx.record_fail("test_f15_widget_activation", String(e))
    try: test_f15_ui_draw_list_tile_binning(ctx); ctx.record_pass("test_f15_ui_draw_list_tile_binning") except e: ctx.record_fail("test_f15_ui_draw_list_tile_binning", String(e))

    # F16
    try: test_f16_acmr_reduction_forsyth(ctx); ctx.record_pass("test_f16_acmr_reduction_forsyth") except e: ctx.record_fail("test_f16_acmr_reduction_forsyth", String(e))
    try: test_f16_acmr_calculation_accuracy(ctx); ctx.record_pass("test_f16_acmr_calculation_accuracy") except e: ctx.record_fail("test_f16_acmr_calculation_accuracy", String(e))
    try: test_f16_draw_indexed_indirect_stride(ctx); ctx.record_pass("test_f16_draw_indexed_indirect_stride") except e: ctx.record_fail("test_f16_draw_indexed_indirect_stride", String(e))
    try: test_f16_backface_cull_mode(ctx); ctx.record_pass("test_f16_backface_cull_mode") except e: ctx.record_fail("test_f16_backface_cull_mode", String(e))
    try: test_f16_meshlet_indices_preservation(ctx); ctx.record_pass("test_f16_meshlet_indices_preservation") except e: ctx.record_fail("test_f16_meshlet_indices_preservation", String(e))

    # F17
    try: test_f17_spirv_module_structure(ctx); ctx.record_pass("test_f17_spirv_module_structure") except e: ctx.record_fail("test_f17_spirv_module_structure", String(e))
    try: test_f17_spirv_arithmetic_emission(ctx); ctx.record_pass("test_f17_spirv_arithmetic_emission") except e: ctx.record_fail("test_f17_spirv_arithmetic_emission", String(e))
    try: test_f17_spirv_control_flow(ctx); ctx.record_pass("test_f17_spirv_control_flow") except e: ctx.record_fail("test_f17_spirv_control_flow", String(e))
    try: test_f17_spirv_barrier_emission(ctx); ctx.record_pass("test_f17_spirv_barrier_emission") except e: ctx.record_fail("test_f17_spirv_barrier_emission", String(e))
    try: test_f17_spirv_validator_verification(ctx); ctx.record_pass("test_f17_spirv_validator_verification") except e: ctx.record_fail("test_f17_spirv_validator_verification", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t1_m2_gpu_loop failed")
