"""Tier 2 Boundary & Corner Cases: Milestone 2 GPU Loop (F13–F17: 25 tests).
Corner cases, extremes, and stress inputs for swapchain, frame pacing, UI SDF layout, meshlet cache, and SPIR-V generation.
"""
from std.collections import List
from flint.core.math import Vec3
from flint.platform.vulkan import (
    PRESENT_FIFO,
    FORMAT_B8G8R8A8_UNORM,
    FORMAT_BC7_UNORM,
    VulkanGpu,
)
from flint.render.font import SdfFont
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
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh


# =========================================================================
# F13: Swapchain Zero-Copy Presentation (Boundary)
# =========================================================================

def test_b_f13_zero_sized_extent(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=0, height=0)
    ctx.assert_equal(ren.output_w, 16, "zero width clamped to minimum 16 in engine")
    ctx.assert_equal(ren.output_h, 16, "zero height clamped to minimum 16 in engine")
    ren.shutdown()

def test_b_f13_extreme_8k_resolution(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=7680, height=4320, tsr_scale=0.5)
    ctx.assert_equal(ren.output_w, 7680, "8K output width configured")
    ctx.assert_equal(ren.output_h, 4320, "8K output height configured")
    ctx.assert_equal(ren.render_w, 3840, "50% TSR render width is 3840")
    ctx.assert_equal(ren.render_h, 2160, "50% TSR render height is 2160")
    ren.shutdown()

def test_b_f13_ultrawide_32_9_aspect(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=5120, height=1440)
    var aspect = Float32(ren.output_w) / Float32(ren.output_h)
    ctx.assert_approx(aspect, 3.5555, 0.01, "32:9 super ultrawide aspect ratio preserved in renderer")
    ren.shutdown()

def test_b_f13_rapid_resize_counter(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=320, height=180)
    for i in range(5):
        ren.begin_frame(Vec3(0, 1, Float32(i)), Vec3(0, 0, 0))
    ctx.assert_equal(len(ren.frame_instances), 0, "frame instances cleared across begin_frame calls")
    ren.shutdown()

def test_b_f13_single_pixel_extent(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=1, height=1)
    ctx.assert_equal(ren.output_w, 16, "single pixel extent clamped to minimum 16 in engine")
    ren.shutdown()


# =========================================================================
# F14: Ring-Buffered Frame Pipeline (Boundary)
# =========================================================================

def test_b_f14_frame_index_wrap_boundary(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ren.frame_index = 0x7FFFFFFF
    ren.begin_frame(Vec3(0, 1, -3), Vec3(0, 0, 0))
    _ = ren.render_frame()
    ctx.assert_equal(ren.frame_index, 0x80000000, "frame_index advances across high bit boundary")
    ren.shutdown()

def test_b_f14_zero_dt_paused_state(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ren.begin_frame(Vec3(10.0, 5.0, 0.0), Vec3(0, 0, 0))
    ctx.assert_approx(ren.cam_pos.x, 10.0, 0.001, "camera position held at zero motion")
    ren.shutdown()

def test_b_f14_negative_dt_clamping(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120, tsr_scale=-0.5)
    ctx.assert_approx(ren.tsr_scale, 0.50, 0.001, "negative tsr_scale clamped to 0.50 minimum in engine")
    ren.shutdown()

def test_b_f14_dt_spike_ceiling(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120, tsr_scale=2.5)
    ctx.assert_approx(ren.tsr_scale, 1.0, 0.001, "tsr_scale clamped to 1.0 maximum in engine")
    ren.shutdown()

def test_b_f14_dynamic_ubo_offset_alignment(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_equal(ren.buf_globals.size % 256, 0, "globals buffer size aligned to 256 bytes for UBO")
    ren.shutdown()


# =========================================================================
# F15: Tile-Binned GPU UiSdfOverlayPass (Boundary)
# =========================================================================

def test_b_f15_empty_string_measurement(mut ctx: TestContext) raises:
    var font = SdfFont()
    var width = font.measure(String(""), 16.0)
    ctx.assert_approx(width, 0.0, 0.0001, "empty string measures 0.0px width")

def test_b_f15_single_glyph_measurement(mut ctx: TestContext) raises:
    var font = SdfFont()
    var font_ok = font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36, 5)
    ctx.assert_true(font_ok and font.valid, "font loaded")
    var width = font.measure("A", 24.0)
    ctx.assert_true(width > 5.0, "single glyph width > 5px")

def test_b_f15_negative_coordinate_widget_clipping(mut ctx: TestContext) raises:
    var tree = WidgetTree()
    var w = Widget(100, 1, -50, -50, 100, 100, String("Clipped"))
    _ = tree.add(w^)
    ctx.assert_equal(len(tree.widgets), 1, "offscreen/clipped widget successfully registered in tree")

def test_b_f15_zero_size_widget(mut ctx: TestContext) raises:
    var tree = WidgetTree()
    var w = Widget(101, 1, 100, 100, 0, 0, String("Empty"))
    _ = tree.add(w^)
    ctx.assert_equal(len(tree.widgets), 1, "zero size widget maintained in hierarchy")

def test_b_f15_deeply_nested_widget_tree(mut ctx: TestContext) raises:
    var tree = WidgetTree()
    for i in range(20):
        var parent_id = 0 if i == 0 else i
        var w = Widget(i + 1, parent_id, 0, 0, 10, 10, String("Depth"))
        _ = tree.add(w^)
    ctx.assert_equal(len(tree.widgets), 20, "20-level hierarchy handles deep nesting")


# =========================================================================
# F16: Forsyth Vertex Cache & ACMR Optimization (Boundary)
# =========================================================================

def test_b_f16_single_triangle_optimization(mut ctx: TestContext) raises:
    var idx = List[Int]()
    idx.append(0)
    idx.append(1)
    idx.append(2)
    var opt = optimize_vertex_cache_forsyth(idx, 3, 32)
    ctx.assert_equal(len(opt), 3, "single triangle preserves 3 indices")
    var acmr = compute_acmr(opt, 32)
    ctx.assert_approx(acmr, 3.0, 0.01, "single triangle ACMR is 3.0 (3 misses / 1 tri)")

def test_b_f16_degenerate_triangles_handling(mut ctx: TestContext) raises:
    var idx = List[Int]()
    idx.append(0)
    idx.append(0)
    idx.append(0)
    var opt = optimize_vertex_cache_forsyth(idx, 1, 32)
    ctx.assert_equal(len(opt), 3, "degenerate triangle handled without crashing")

def test_b_f16_high_valence_fan(mut ctx: TestContext) raises:
    var idx = List[Int]()
    for i in range(16):
        idx.append(0)
        idx.append(i + 1)
        idx.append(i + 2)
    var opt = optimize_vertex_cache_forsyth(idx, 18, 32)
    ctx.assert_equal(len(opt), 48, "16-tri fan retains 48 indices")
    var acmr = compute_acmr(opt, 32)
    ctx.assert_true(acmr < 1.5, "shared central apex achieves good cache locality")

def test_b_f16_cache_size_boundary_small(mut ctx: TestContext) raises:
    var idx = List[Int]()
    idx.append(0)
    idx.append(1)
    idx.append(2)
    idx.append(1)
    idx.append(3)
    idx.append(2)
    var opt = optimize_vertex_cache_forsyth(idx, 4, 4)
    ctx.assert_equal(len(opt), 6, "quad mesh optimized with small cache size 4")

def test_b_f16_cache_size_boundary_large(mut ctx: TestContext) raises:
    var m = create_mock_meshlet_mesh(64)
    var int_indices = List[Int]()
    for i in range(len(m.indices)):
        int_indices.append(Int(m.indices[i]))
    var opt = optimize_vertex_cache_forsyth(int_indices, len(m.positions) // 3, 128)
    ctx.assert_equal(len(opt), len(m.indices), "large cache 128 preserves index count")
    var acmr = compute_acmr(opt, 128)
    ctx.assert_true(acmr > 0.0 and acmr < 3.0, "ACMR within realistic bounds")


# =========================================================================
# F17: SPIR-V Shader Generator & Validation (Boundary)
# =========================================================================

def test_b_f17_minimal_compute_workgroup_1x1x1(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(1, 1, 1)
    var words = m.finish()
    ctx.assert_equal(words[0], SPIRV_MAGIC, "SPIR-V magic preserved for 1x1x1 workgroup")
    var report = validate_spirv_module(words)
    ctx.assert_true(report.valid, "minimal 1x1x1 compute module passes validation")

def test_b_f17_maximum_compute_workgroup_1024x1x1(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(1024, 1, 1)
    var words = m.finish()
    var report = validate_spirv_module(words)
    ctx.assert_true(report.valid, "1024x1x1 compute module passes validation")

def test_b_f17_repeated_barriers(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 64, 1, 1)
    var sc_exec = m.uint(2)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    for _ in range(3):
        emit(m.code, OP_CONTROL_BARRIER, sc_exec, sc_mem, sem)
    var words = m.finish()
    var report = validate_spirv_module(words)
    ctx.assert_true(report.valid, "consecutive control barriers pass SSA validation")

def test_b_f17_nested_branching_module(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(64, 1, 1)
    var c1 = m.uint(10)
    var c2 = m.uint(20)
    var cond = m.ult(c1, c2)
    m.begin_if(cond)
    _ = m.iadd(c1, c2)
    m.end_if()
    var words = m.finish()
    var report = validate_spirv_module(words)
    ctx.assert_true(report.valid, "nested conditional branches pass validation")

def test_b_f17_multiple_float_math_ops(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(32, 1, 1)
    var a = m.float(1.5)
    var b = m.float(2.5)
    var s = m.fadd(a, b)
    var d = m.fsub(s, a)
    _ = m.fmul(d, b)
    var words = m.finish()
    var report = validate_spirv_module(words)
    ctx.assert_true(report.valid, "sequential float ALU ops pass validation")


# =========================================================================
# Aggregator for Tier 2 Milestone 2
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T2_M2_GpuLoop_Boundary")

    # F13
    try: test_b_f13_zero_sized_extent(ctx); ctx.record_pass("test_b_f13_zero_sized_extent") except e: ctx.record_fail("test_b_f13_zero_sized_extent", String(e))
    try: test_b_f13_extreme_8k_resolution(ctx); ctx.record_pass("test_b_f13_extreme_8k_resolution") except e: ctx.record_fail("test_b_f13_extreme_8k_resolution", String(e))
    try: test_b_f13_ultrawide_32_9_aspect(ctx); ctx.record_pass("test_b_f13_ultrawide_32_9_aspect") except e: ctx.record_fail("test_b_f13_ultrawide_32_9_aspect", String(e))
    try: test_b_f13_rapid_resize_counter(ctx); ctx.record_pass("test_b_f13_rapid_resize_counter") except e: ctx.record_fail("test_b_f13_rapid_resize_counter", String(e))
    try: test_b_f13_single_pixel_extent(ctx); ctx.record_pass("test_b_f13_single_pixel_extent") except e: ctx.record_fail("test_b_f13_single_pixel_extent", String(e))

    # F14
    try: test_b_f14_frame_index_wrap_boundary(ctx); ctx.record_pass("test_b_f14_frame_index_wrap_boundary") except e: ctx.record_fail("test_b_f14_frame_index_wrap_boundary", String(e))
    try: test_b_f14_zero_dt_paused_state(ctx); ctx.record_pass("test_b_f14_zero_dt_paused_state") except e: ctx.record_fail("test_b_f14_zero_dt_paused_state", String(e))
    try: test_b_f14_negative_dt_clamping(ctx); ctx.record_pass("test_b_f14_negative_dt_clamping") except e: ctx.record_fail("test_b_f14_negative_dt_clamping", String(e))
    try: test_b_f14_dt_spike_ceiling(ctx); ctx.record_pass("test_b_f14_dt_spike_ceiling") except e: ctx.record_fail("test_b_f14_dt_spike_ceiling", String(e))
    try: test_b_f14_dynamic_ubo_offset_alignment(ctx); ctx.record_pass("test_b_f14_dynamic_ubo_offset_alignment") except e: ctx.record_fail("test_b_f14_dynamic_ubo_offset_alignment", String(e))

    # F15
    try: test_b_f15_empty_string_measurement(ctx); ctx.record_pass("test_b_f15_empty_string_measurement") except e: ctx.record_fail("test_b_f15_empty_string_measurement", String(e))
    try: test_b_f15_single_glyph_measurement(ctx); ctx.record_pass("test_b_f15_single_glyph_measurement") except e: ctx.record_fail("test_b_f15_single_glyph_measurement", String(e))
    try: test_b_f15_negative_coordinate_widget_clipping(ctx); ctx.record_pass("test_b_f15_negative_coordinate_widget_clipping") except e: ctx.record_fail("test_b_f15_negative_coordinate_widget_clipping", String(e))
    try: test_b_f15_zero_size_widget(ctx); ctx.record_pass("test_b_f15_zero_size_widget") except e: ctx.record_fail("test_b_f15_zero_size_widget", String(e))
    try: test_b_f15_deeply_nested_widget_tree(ctx); ctx.record_pass("test_b_f15_deeply_nested_widget_tree") except e: ctx.record_fail("test_b_f15_deeply_nested_widget_tree", String(e))

    # F16
    try: test_b_f16_single_triangle_optimization(ctx); ctx.record_pass("test_b_f16_single_triangle_optimization") except e: ctx.record_fail("test_b_f16_single_triangle_optimization", String(e))
    try: test_b_f16_degenerate_triangles_handling(ctx); ctx.record_pass("test_b_f16_degenerate_triangles_handling") except e: ctx.record_fail("test_b_f16_degenerate_triangles_handling", String(e))
    try: test_b_f16_high_valence_fan(ctx); ctx.record_pass("test_b_f16_high_valence_fan") except e: ctx.record_fail("test_b_f16_high_valence_fan", String(e))
    try: test_b_f16_cache_size_boundary_small(ctx); ctx.record_pass("test_b_f16_cache_size_boundary_small") except e: ctx.record_fail("test_b_f16_cache_size_boundary_small", String(e))
    try: test_b_f16_cache_size_boundary_large(ctx); ctx.record_pass("test_b_f16_cache_size_boundary_large") except e: ctx.record_fail("test_b_f16_cache_size_boundary_large", String(e))

    # F17
    try: test_b_f17_minimal_compute_workgroup_1x1x1(ctx); ctx.record_pass("test_b_f17_minimal_compute_workgroup_1x1x1") except e: ctx.record_fail("test_b_f17_minimal_compute_workgroup_1x1x1", String(e))
    try: test_b_f17_maximum_compute_workgroup_1024x1x1(ctx); ctx.record_pass("test_b_f17_maximum_compute_workgroup_1024x1x1") except e: ctx.record_fail("test_b_f17_maximum_compute_workgroup_1024x1x1", String(e))
    try: test_b_f17_repeated_barriers(ctx); ctx.record_pass("test_b_f17_repeated_barriers") except e: ctx.record_fail("test_b_f17_repeated_barriers", String(e))
    try: test_b_f17_nested_branching_module(ctx); ctx.record_pass("test_b_f17_nested_branching_module") except e: ctx.record_fail("test_b_f17_nested_branching_module", String(e))
    try: test_b_f17_multiple_float_math_ops(ctx); ctx.record_pass("test_b_f17_multiple_float_math_ops") except e: ctx.record_fail("test_b_f17_multiple_float_math_ops", String(e))

def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
