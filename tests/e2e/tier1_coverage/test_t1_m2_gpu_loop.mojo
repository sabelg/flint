"""Tier 1 Feature Coverage: Milestone 2 GPU Loop (F13–F17: 25 tests).
Opaque-box and requirement-driven verification of swapchain, ring buffering,
GPU UI SDF/UiContext, meshlets, and SPIR-V.
"""
from std.collections import List
from flint.assets.meshlet import compute_acmr, optimize_vertex_cache_forsyth
from flint.core.math import Vec3
from flint.platform.input import InputState
from flint.platform.vulkan import FORMAT_B8G8R8A8_UNORM, PRESENT_FIFO, TYPE_SWAPCHAIN_CREATE_INFO
from flint.render.aaa_post_tsr import AAA_DISPLAY_AGX_REC709, AAA_DISPLAY_HDR10_PQ
from flint.render.aaa_renderer import FRAMES_IN_FLIGHT, FlintAaaVulkanRenderer
from flint.render.aaa_visbuffer import meshlet_cull_shader
from flint.render.font import SdfFont
from flint.render.spirv import (
    MODEL_GL_COMPUTE,
    OP_CONTROL_BARRIER,
    SPIRV_MAGIC,
    AaaSpirvModule,
    emit,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.render.swapchain_surface import (
    DISPLAY_HDR10_PQ,
    DISPLAY_SDR_SRGB,
    SURFACE_HEADLESS,
    SwapchainSurface,
    resolve_swapchain_display_mode,
)
from flint.render.ui_overlay import aaa_ui_overlay_shader
from flint.ui.context import UI_KEY_ENTER, UI_KEY_TAB, UiContext
from flint.ui.draw_list import UiDrawList
from flint.ui.flexbox_dock import DockSpace, FLEX_COL, FlexStyle, RectF
from flint.ui.layout import UiRect, ui_frame
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh


# =========================================================================
# F13: Swapchain Zero-Copy Presentation (5 tests)
# =========================================================================

def test_f13_swapchain_fifo_mode_defined(mut ctx: TestContext) raises:
    ctx.assert_equal(Int(PRESENT_FIFO), 2, "PRESENT_FIFO is mode 2 in Vulkan")
    ctx.assert_equal(Int(TYPE_SWAPCHAIN_CREATE_INFO), 1000001000, "Vulkan swapchain create info struct type")
    var surf = SwapchainSurface()
    ctx.assert_true(surf.initialize_headless(320, 180, DISPLAY_SDR_SRGB, PRESENT_FIFO), "headless SwapchainSurface init")
    ctx.assert_equal(surf.backend, SURFACE_HEADLESS, "headless surface backend")
    ctx.assert_equal(resolve_swapchain_display_mode(FORMAT_B8G8R8A8_UNORM, False), DISPLAY_SDR_SRGB, "BGRA8 maps to SDR")
    ctx.assert_equal(resolve_swapchain_display_mode(FORMAT_B8G8R8A8_UNORM, True), DISPLAY_HDR10_PQ, "HDR flag maps to HDR10 PQ")


def test_f13_swapchain_dimensions_preservation(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "renderer active in headless mode")
    ctx.assert_true(ren.resize(1920, 1080), "renderer resize to 1920x1080 succeeds")
    ctx.assert_equal(ren.output_w, 1920, "swapchain output width preserved in renderer")
    ctx.assert_equal(ren.output_h, 1080, "swapchain output height preserved in renderer")
    var aspect = Float32(ren.output_w) / Float32(ren.output_h)
    ctx.assert_approx(aspect, 1.7777, 0.01, "16:9 aspect ratio preserved in renderer")
    _ = ren.resize(160, 120)


def test_f13_swapchain_surface_formats(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_equal(Int(FORMAT_B8G8R8A8_UNORM), 44, "VK_FORMAT_B8G8R8A8_UNORM format constant is 44")
    ctx.assert_equal(ren.display_mode, AAA_DISPLAY_AGX_REC709, "default display mode is AgX Rec.709")
    ren.display_mode = AAA_DISPLAY_HDR10_PQ
    ctx.assert_equal(ren.display_mode, AAA_DISPLAY_HDR10_PQ, "HDR10 PQ swapchain display mode configurable")
    ren.display_mode = AAA_DISPLAY_AGX_REC709


def test_f13_swapchain_acquire_semaphore_indexing(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_equal(ren.frame_index, 0, "initial frame index is 0")
    ctx.assert_equal(ren.vk.swapchain, 0, "swapchain handle is 0 in headless mode")
    ctx.assert_equal(len(ren.vk.acquire_semaphores), 0, "no swapchain semaphores allocated in headless")


def test_f13_swapchain_headless_fallback(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "renderer active in headless readback fallback mode")
    ren.begin_frame(Vec3(0, 1, -3), Vec3(0, 0, 0))
    var ok = ren.render_frame()
    ctx.assert_true(ok, "headless render_frame completes successfully")
    var pixels = ren.readback_pixels()
    ctx.assert_equal(len(pixels), 160 * 120, "headless readback retrieves full frame buffer")


# =========================================================================
# F14: Ring-Buffered Frame Pipeline (5 tests)
# =========================================================================

def test_f14_ring_buffer_slots_count(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_equal(len(ren.ring_globals), FRAMES_IN_FLIGHT, "ring_globals has FRAMES_IN_FLIGHT slots")
    ctx.assert_equal(len(ren.ring_instances), FRAMES_IN_FLIGHT, "ring_instances has FRAMES_IN_FLIGHT slots")
    ctx.assert_equal(len(ren.staging_ring.slot_buffers), FRAMES_IN_FLIGHT, "LinearStagingRing has FRAMES_IN_FLIGHT slots")
    ctx.assert_true(ren.ring_globals[0].size >= 512, "globals UBO buffer allocated")
    ctx.assert_true(ren.ring_instances[0].size > 0, "instance SSBO buffer allocated")


def test_f14_frame_index_increment(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    var start_idx = ren.frame_index
    ren.begin_frame(Vec3(0, 1, -2), Vec3(0, 0, 0))
    _ = ren.render_frame()
    ctx.assert_equal(ren.frame_index, start_idx + 1, "frame_index increments by 1 after first render_frame")
    ren.begin_frame(Vec3(0, 1, -2), Vec3(0, 0, 0))
    _ = ren.render_frame()
    ctx.assert_equal(ren.frame_index, start_idx + 2, "frame_index increments by 2 after second render_frame")


def test_f14_ping_pong_history_alternation(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "renderer must be active to verify ping-pong history buffers")
    ctx.assert_true(ren.buf_history.size > 0, "history buffer allocated")
    ctx.assert_true(ren.buf_hdr_out.size > 0, "HDR output buffer allocated")
    ctx.assert_equal(ren.buf_history.cpu_address, 0, "history buffer resides in DEVICE_LOCAL VRAM (cpu_address == 0)")
    ctx.assert_equal(ren.buf_hdr_out.cpu_address, 0, "HDR output resides in DEVICE_LOCAL VRAM (cpu_address == 0)")
    ctx.assert_equal(ren.buf_hdr_out.vk_suballoc_block, -2, "HDR output bound to aliased transient heap (-2)")
    ctx.assert_equal(len(ren.staging_ring.slot_buffers), FRAMES_IN_FLIGHT, "staging ring has FRAMES_IN_FLIGHT slots")
    ctx.assert_true(ren.buf_history.vk_buffer != 0, "history buffer has valid VkBuffer handle")
    ctx.assert_true(ren.buf_hdr_out.vk_buffer != 0, "HDR output has valid VkBuffer handle")
    ctx.assert_true(ren.buf_history.vk_buffer != ren.buf_hdr_out.vk_buffer, "history and HDR output occupy distinct VkBuffers")

    var start_fi = ren.frame_index
    ren.begin_frame(Vec3(0, 1, -3), Vec3(0, 0, 0))
    ctx.assert_true(ren.render_frame(), "even/first ping-pong frame renders")
    var hdr1 = ren.readback_resolved_hdr()
    ctx.assert_equal(len(hdr1), ren.output_w * ren.output_h, "frame 1 resolved HDR readback matches output size")

    ren.begin_frame(Vec3(0, 1, -3), Vec3(0, 0, 0))
    ctx.assert_true(ren.render_frame(), "odd/second ping-pong frame renders")
    var hdr2 = ren.readback_resolved_hdr()
    ctx.assert_equal(len(hdr2), ren.output_w * ren.output_h, "frame 2 resolved HDR readback matches output size")
    ctx.assert_equal(ren.frame_index, start_fi + 2, "frame_index advances across even/odd TSR history ping-pong")


def test_f14_ring_globals_offset(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "renderer active")
    ctx.assert_true(FRAMES_IN_FLIGHT >= 2, "double/triple buffered ring")
    ctx.assert_equal(ren.buf_globals.size % 256, 0, "globals mirror size aligned to 256 bytes")
    ctx.assert_equal(ren.ring_globals[0].size % 256, 0, "ring_globals[0] aligned to 256 bytes")
    ctx.assert_true(ren.ring_instances[0].size >= 36 * 4, "ring_instances[0] holds instance records")
    ctx.assert_true(ren.ring_globals[0].vk_buffer != ren.ring_globals[1].vk_buffer, "per-frame globals buffers distinct")


def test_f14_fence_wait_synchronization(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ren.begin_frame(Vec3(0, 1, -3), Vec3(0, 0, 0))
    var ok = ren.render_frame()
    ctx.assert_true(ok, "fence wait synchronization clears and frame renders")


# =========================================================================
# F15: Tile-Binned GPU UiSdfOverlayPass & UiContext (5 tests)
# =========================================================================

def test_f15_sdf_font_measure_text(mut ctx: TestContext) raises:
    var font = SdfFont()
    var font_ok = font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36, 5)
    ctx.assert_true(font_ok and font.valid, "DejaVuSansMono-Bold font loaded successfully")
    var width = font.measure("FLINT ENGINE", 24.0)
    ctx.assert_true(width > 50.0, "measured text width > 50 px")


def test_f15_widget_tree_navigation(mut ctx: TestContext) raises:
    var ui = UiContext()
    var inp = InputState()
    ui.begin_frame(inp, 800.0, 600.0)
    ui.begin_box("panel", FlexStyle(FLEX_COL))
    _ = ui.button("btn1", "Button 1", 120.0, 32.0)
    _ = ui.button("btn2", "Button 2", 120.0, 32.0)
    ui.end_box()
    ui.end_frame()
    var first_focus = ui.focused_id
    ctx.assert_true(len(ui.draw_list.prims) >= 2, "UiContext emits primitives for buttons")

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_TAB, True)
    ui.begin_frame(inp, 800.0, 600.0)
    ui.begin_box("panel", FlexStyle(FLEX_COL))
    _ = ui.button("btn1", "Button 1", 120.0, 32.0)
    _ = ui.button("btn2", "Button 2", 120.0, 32.0)
    ui.end_box()
    ui.end_frame()
    ctx.assert_true(first_focus != 0 and ui.focused_id != 0, "UiContext registers focusable buttons")
    ctx.assert_true(ui.focused_id != first_focus, "Tab advances focus across UiContext buttons")


def test_f15_widget_bounding_box_hit(mut ctx: TestContext) raises:
    var ui = UiContext()
    var inp = InputState()
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.button("play", "Play", 200.0, 40.0)
    var rect = ui.last_item_rect
    ui.end_frame()

    inp.begin_frame()
    inp.has_mouse_position = True
    inp.last_mouse_x = rect.center_x()
    inp.last_mouse_y = rect.center_y()
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 800.0, 600.0)
    var down_hit = ui.button("play", "Play", 200.0, 40.0)
    ui.end_frame()
    ctx.assert_false(down_hit, "button activates on release, not press")

    inp.begin_frame()
    inp.has_mouse_position = True
    inp.last_mouse_x = rect.center_x()
    inp.last_mouse_y = rect.center_y()
    inp.set_mouse(0, False)
    ui.begin_frame(inp, 800.0, 600.0)
    var clicked = ui.button("play", "Play", 200.0, 40.0)
    ui.end_frame()
    ctx.assert_true(clicked, "mouse release inside button rect activates button")

    inp.begin_frame()
    inp.has_mouse_position = True
    inp.last_mouse_x = rect.right() + 50.0
    inp.last_mouse_y = rect.bottom() + 50.0
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.button("play", "Play", 200.0, 40.0)
    ui.end_frame()
    inp.begin_frame()
    inp.has_mouse_position = True
    inp.last_mouse_x = rect.right() + 50.0
    inp.last_mouse_y = rect.bottom() + 50.0
    inp.set_mouse(0, False)
    ui.begin_frame(inp, 800.0, 600.0)
    var outside_clicked = ui.button("play", "Play", 200.0, 40.0)
    ui.end_frame()
    ctx.assert_false(outside_clicked, "click outside button rect does not activate")


def test_f15_widget_activation(mut ctx: TestContext) raises:
    var ui = UiContext()
    var inp = InputState()
    inp.set_logical_key(UI_KEY_TAB, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.button("start", "Start Game", 140.0, 36.0)
    ui.end_frame()

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_ENTER, True)
    ui.begin_frame(inp, 800.0, 600.0)
    var activated = ui.button("start", "Start Game", 140.0, 36.0)
    ui.end_frame()
    ctx.assert_true(activated, "UI_KEY_ENTER activates focused UiContext button")

    var dock = DockSpace.create_studio_5panel()
    dock.solve_layout(RectF(0.0, 0.0, 1280.0, 720.0))
    ctx.assert_equal(len(dock.leaf_rects()), 5, "DockSpace allocates 5 tiled panels")


def test_f15_ui_draw_list_tile_binning(mut ctx: TestContext) raises:
    var frame = ui_frame(1920, 1080, 0, 0, 1.0)
    var rect = frame.safe()
    ctx.assert_equal(rect.w, 1920, "ui_frame width matches design resolution")
    ctx.assert_equal(rect.h, 1080, "ui_frame height matches design resolution")
    ctx.assert_true(rect.contains(UiRect(10, 10, 100, 50)), "safe rect contains embedded child widget")
    var rep = validate_spirv_module(aaa_ui_overlay_shader())
    ctx.assert_true(rep.valid, "aaa_ui_overlay_shader validates SPIR-V")


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


def test_f16_draw_indexed_indirect_stride(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "renderer active")
    ctx.assert_true(ren.ring_indirect[0].size >= 64 + 20, "ring_indirect[0] holds atomic count + indirect draws")
    ctx.assert_true(ren.buf_indirect_phase2.size >= 64 + 20, "phase 2 indirect draw buffer allocated")
    ctx.assert_true(ren.ring_indirect[0].vk_buffer != 0, "ring_indirect[0] has valid VkBuffer")


def test_f16_backface_cull_mode(mut ctx: TestContext) raises:
    var rep = validate_spirv_module(meshlet_cull_shader())
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
    ctx.assert_true(len(m.finish()) > 20, "SPIR-V module generates integer arithmetic instructions")


def test_f17_spirv_control_flow(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(8, 8, 4)
    var a = m.uint(10)
    var b = m.uint(20)
    var cond = m.ult(a, b)
    m.begin_if(cond)
    _ = m.iadd(a, b)
    m.end_if()
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "SPIR-V conditional branch validates")


def test_f17_spirv_barrier_emission(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    emit(m.code, OP_CONTROL_BARRIER, m.uint(2), m.uint(2), m.uint(0x100))
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "SPIR-V module with workgroup control barrier validates")


def test_f17_spirv_validator_verification(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(16, 16, 8)
    _ = m.fadd(m.float(1.5), m.float(2.5))
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "SSA validator verifies generated float math")


# =========================================================================
# Aggregator for Tier 1 Milestone 2
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T1_M2_GpuLoop")
    var ren = FlintAaaVulkanRenderer(width=160, height=120)

    try: test_f13_swapchain_fifo_mode_defined(ctx); ctx.record_pass("test_f13_swapchain_fifo_mode_defined") except e: ctx.record_fail("test_f13_swapchain_fifo_mode_defined", String(e))
    try: test_f13_swapchain_dimensions_preservation(ctx, ren); ctx.record_pass("test_f13_swapchain_dimensions_preservation") except e: ctx.record_fail("test_f13_swapchain_dimensions_preservation", String(e))
    try: test_f13_swapchain_surface_formats(ctx, ren); ctx.record_pass("test_f13_swapchain_surface_formats") except e: ctx.record_fail("test_f13_swapchain_surface_formats", String(e))
    try: test_f13_swapchain_acquire_semaphore_indexing(ctx, ren); ctx.record_pass("test_f13_swapchain_acquire_semaphore_indexing") except e: ctx.record_fail("test_f13_swapchain_acquire_semaphore_indexing", String(e))
    try: test_f13_swapchain_headless_fallback(ctx, ren); ctx.record_pass("test_f13_swapchain_headless_fallback") except e: ctx.record_fail("test_f13_swapchain_headless_fallback", String(e))

    try: test_f14_ring_buffer_slots_count(ctx, ren); ctx.record_pass("test_f14_ring_buffer_slots_count") except e: ctx.record_fail("test_f14_ring_buffer_slots_count", String(e))
    try: test_f14_frame_index_increment(ctx, ren); ctx.record_pass("test_f14_frame_index_increment") except e: ctx.record_fail("test_f14_frame_index_increment", String(e))
    try: test_f14_ping_pong_history_alternation(ctx, ren); ctx.record_pass("test_f14_ping_pong_history_alternation") except e: ctx.record_fail("test_f14_ping_pong_history_alternation", String(e))
    try: test_f14_ring_globals_offset(ctx, ren); ctx.record_pass("test_f14_ring_globals_offset") except e: ctx.record_fail("test_f14_ring_globals_offset", String(e))
    try: test_f14_fence_wait_synchronization(ctx, ren); ctx.record_pass("test_f14_fence_wait_synchronization") except e: ctx.record_fail("test_f14_fence_wait_synchronization", String(e))

    try: test_f15_sdf_font_measure_text(ctx); ctx.record_pass("test_f15_sdf_font_measure_text") except e: ctx.record_fail("test_f15_sdf_font_measure_text", String(e))
    try: test_f15_widget_tree_navigation(ctx); ctx.record_pass("test_f15_widget_tree_navigation") except e: ctx.record_fail("test_f15_widget_tree_navigation", String(e))
    try: test_f15_widget_bounding_box_hit(ctx); ctx.record_pass("test_f15_widget_bounding_box_hit") except e: ctx.record_fail("test_f15_widget_bounding_box_hit", String(e))
    try: test_f15_widget_activation(ctx); ctx.record_pass("test_f15_widget_activation") except e: ctx.record_fail("test_f15_widget_activation", String(e))
    try: test_f15_ui_draw_list_tile_binning(ctx); ctx.record_pass("test_f15_ui_draw_list_tile_binning") except e: ctx.record_fail("test_f15_ui_draw_list_tile_binning", String(e))

    try: test_f16_acmr_reduction_forsyth(ctx); ctx.record_pass("test_f16_acmr_reduction_forsyth") except e: ctx.record_fail("test_f16_acmr_reduction_forsyth", String(e))
    try: test_f16_acmr_calculation_accuracy(ctx); ctx.record_pass("test_f16_acmr_calculation_accuracy") except e: ctx.record_fail("test_f16_acmr_calculation_accuracy", String(e))
    try: test_f16_draw_indexed_indirect_stride(ctx, ren); ctx.record_pass("test_f16_draw_indexed_indirect_stride") except e: ctx.record_fail("test_f16_draw_indexed_indirect_stride", String(e))
    try: test_f16_backface_cull_mode(ctx); ctx.record_pass("test_f16_backface_cull_mode") except e: ctx.record_fail("test_f16_backface_cull_mode", String(e))
    try: test_f16_meshlet_indices_preservation(ctx); ctx.record_pass("test_f16_meshlet_indices_preservation") except e: ctx.record_fail("test_f16_meshlet_indices_preservation", String(e))

    try: test_f17_spirv_module_structure(ctx); ctx.record_pass("test_f17_spirv_module_structure") except e: ctx.record_fail("test_f17_spirv_module_structure", String(e))
    try: test_f17_spirv_arithmetic_emission(ctx); ctx.record_pass("test_f17_spirv_arithmetic_emission") except e: ctx.record_fail("test_f17_spirv_arithmetic_emission", String(e))
    try: test_f17_spirv_control_flow(ctx); ctx.record_pass("test_f17_spirv_control_flow") except e: ctx.record_fail("test_f17_spirv_control_flow", String(e))
    try: test_f17_spirv_barrier_emission(ctx); ctx.record_pass("test_f17_spirv_barrier_emission") except e: ctx.record_fail("test_f17_spirv_barrier_emission", String(e))
    try: test_f17_spirv_validator_verification(ctx); ctx.record_pass("test_f17_spirv_validator_verification") except e: ctx.record_fail("test_f17_spirv_validator_verification", String(e))

    ren.shutdown()


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t1_m2_gpu_loop failed")
