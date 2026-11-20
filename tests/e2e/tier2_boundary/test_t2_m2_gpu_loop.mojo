"""Tier 2 Boundary & Corner Cases: Milestone 2 GPU Loop (F13–F17: 25 tests).
Corner cases, extremes, and stress inputs for swapchain, FixedTime/ring buffering,
FlexTree/DockSpace UI layout, meshlet cache, and SPIR-V generation.
"""
from std.collections import List
from flint.app.engine_app import FixedTime
from flint.assets.meshlet import compute_acmr, optimize_vertex_cache_forsyth
from flint.core.math import Vec3
from flint.platform.input import InputState
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.font import SdfFont
from flint.render.spirv import (
    MODEL_GL_COMPUTE,
    OP_CONTROL_BARRIER,
    SPIRV_MAGIC,
    AaaSpirvModule,
    emit,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.render.swapchain_surface import SwapchainSurface
from flint.ui.context import UiContext
from flint.ui.flexbox_dock import DockSpace, FLEX_COL, FlexStyle, FlexTree, RectF
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh


# =========================================================================
# F13: Swapchain & Resize Extent Boundaries (reuses single renderer!)
# =========================================================================

def test_b_f13_zero_sized_extent(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    var surf = SwapchainSurface()
    ctx.assert_true(surf.initialize_headless(0, 0), "0x0 headless surface initializes safely")
    ctx.assert_equal(surf.width, 16, "0 width clamped to minimum 16")
    ctx.assert_equal(surf.height, 16, "0 height clamped to minimum 16")
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active in headless Vulkan mode")
    ctx.assert_true(ren.resize(0, 0), "resize(0, 0) clamps safely")
    ctx.assert_equal(ren.output_w, 16, "renderer output_w clamped to 16")
    ctx.assert_equal(ren.output_h, 16, "renderer output_h clamped to 16")


def test_b_f13_extreme_8k_resolution(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    var surf = SwapchainSurface()
    ctx.assert_true(surf.initialize_headless(7680, 4320), "8K headless surface initializes")
    ctx.assert_equal(surf.width, 7680, "8K surface width preserved")
    ctx.assert_equal(surf.height, 4320, "8K surface height preserved")
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active in headless Vulkan mode")
    var r2 = FlintAaaVulkanRenderer(width=3840, height=2160, tsr_scale=0.5)
    ctx.assert_true(r2.active, "4K renderer initializes active")
    ctx.assert_equal(r2.output_w, 3840, "4K output width configured")
    ctx.assert_equal(r2.render_w, 1920, "50% TSR render width is 1920")
    r2.shutdown()


def test_b_f13_ultrawide_32_9_aspect(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active in headless Vulkan mode")
    ctx.assert_true(ren.resize(5120, 1440), "ultrawide resize(5120, 1440) succeeds")
    var aspect = Float32(ren.output_w) / Float32(ren.output_h)
    ctx.assert_approx(aspect, 3.5555, 0.01, "32:9 super ultrawide aspect ratio preserved")


def test_b_f13_rapid_resize_counter(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active in headless Vulkan mode")
    ctx.assert_true(ren.resize(320, 180), "resize to 320x180")
    ctx.assert_true(ren.resize(640, 360), "resize to 640x360")
    ctx.assert_true(ren.resize(1920, 1080), "resize to 1920x1080")
    ctx.assert_true(ren.resize(160, 120), "resize back to 160x120")
    ctx.assert_equal(ren.output_w, 160, "final width matches 160")
    for i in range(5):
        ren.begin_frame(Vec3(0, 1, Float32(i)), Vec3(0, 0, 0))
    ctx.assert_equal(len(ren.frame_instances), 0, "frame instances cleared across begin_frame calls")


def test_b_f13_single_pixel_extent(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    var surf = SwapchainSurface()
    _ = surf.initialize_headless(1, 1)
    ctx.assert_equal(surf.width, 16, "single pixel surface clamped to 16")
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active in headless Vulkan mode")
    ctx.assert_true(ren.resize(1, 1), "resize(1, 1) clamps safely")
    ctx.assert_equal(ren.output_w, 16, "single pixel extent clamped to 16")
    _ = ren.resize(160, 120)


# =========================================================================
# F14: FixedTime Clamping & Ring-Buffered Frame Pipeline (Boundary)
# =========================================================================

def test_b_f14_frame_index_wrap_boundary(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ren.frame_index = 0x7FFFFFFF
    ren.begin_frame(Vec3(0, 1, -3), Vec3(0, 0, 0))
    _ = ren.render_frame()
    ctx.assert_equal(ren.frame_index, 0x80000000, "frame_index advances across high bit boundary")
    ren.frame_index = 0


def test_b_f14_zero_dt_paused_state(mut ctx: TestContext) raises:
    var ft = FixedTime(hz=60.0, max_steps_per_frame=8)
    var steps = ft.advance_frame(Float32(0.0))
    ctx.assert_equal(steps, 0, "zero dt produces 0 fixed simulation steps")
    ctx.assert_approx(ft.overstep_alpha, 0.0, 0.0001, "overstep_alpha stays 0.0 at dt=0")


def test_b_f14_negative_dt_clamping(mut ctx: TestContext) raises:
    var ft = FixedTime(hz=60.0, max_steps_per_frame=8)
    var steps = ft.advance_frame(Float32(-0.5))
    ctx.assert_equal(steps, 0, "negative dt clamped to 0.0 producing 0 steps")
    ctx.assert_true(ft.accumulator >= 0.0, "accumulator never negative")


def test_b_f14_dt_spike_ceiling(mut ctx: TestContext) raises:
    var ft = FixedTime(hz=60.0, max_steps_per_frame=8)
    var steps = ft.advance_frame(Float32(5.0))
    ctx.assert_equal(steps, 8, "5.0s dt spike clamped to max_steps_per_frame == 8")
    ctx.assert_true(ft.clamped_last_frame, "clamped_last_frame records spiral-of-death clamp")


def test_b_f14_dynamic_ubo_offset_alignment(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active in headless Vulkan mode")
    ctx.assert_equal(ren.buf_globals.size % 256, 0, "globals buffer size aligned to 256 bytes for UBO")
    var s1 = ren.staging_ring.alloc(ren.vk, 0, 13)
    var s2 = ren.staging_ring.alloc(ren.vk, 0, 17)
    ctx.assert_equal(s1.offset % 256, 0, "LinearStagingRing slice 1 aligned to 256 bytes")
    ctx.assert_equal(s2.offset % 256, 0, "LinearStagingRing slice 2 aligned to 256 bytes")
    ren.staging_ring.reset_slot(ren.vk, 0)


# =========================================================================
# F15: Flexbox, DockSpace & SdfFont UI Boundaries
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
    var outer = RectF(-50.0, -50.0, 100.0, 100.0)
    var viewport = RectF(0.0, 0.0, 800.0, 600.0)
    var clipped = outer.intersect(viewport)
    ctx.assert_approx(clipped.x, 0.0, 0.001, "clipped x clamped to 0")
    ctx.assert_approx(clipped.y, 0.0, 0.001, "clipped y clamped to 0")
    ctx.assert_approx(clipped.w, 50.0, 0.001, "clipped width is 50")
    ctx.assert_approx(clipped.h, 50.0, 0.001, "clipped height is 50")


def test_b_f15_zero_size_widget(mut ctx: TestContext) raises:
    var tree = FlexTree()
    var root_st = FlexStyle(FLEX_COL)
    var root = tree.add_node(-1, root_st, 200.0, 100.0)
    var child_st = FlexStyle(FLEX_COL)
    child_st.fixed_w = 0.0
    child_st.fixed_h = 0.0
    var c = tree.add_node(root, child_st, 0.0, 0.0)
    tree.solve(RectF(0.0, 0.0, 200.0, 100.0), root)
    ctx.assert_true(tree.nodes[c].rect.w >= 0.0 and tree.nodes[c].rect.h >= 0.0, "zero-size FlexTree child safe")

    var ui = UiContext()
    var inp = InputState()
    ui.begin_frame(inp, Float32(0.0), Float32(-10.0), Float32(0.0))
    ui.label("zero-vp")
    ui.end_frame()
    ctx.assert_approx(ui.screen_w, 1.0, 0.001, "UiContext clamps zero viewport width to 1.0")
    ctx.assert_approx(ui.screen_h, 1.0, 0.001, "UiContext clamps negative viewport height to 1.0")
    ctx.assert_true(ui.dt >= Float32(1.0e-4), "UiContext clamps zero dt to >= 1e-4")


def test_b_f15_deeply_nested_widget_tree(mut ctx: TestContext) raises:
    var tree = FlexTree()
    var parent = -1
    for _ in range(20):
        var st = FlexStyle(FLEX_COL)
        st.grow = 1.0
        st.pad_l = 1.0
        st.pad_t = 1.0
        parent = tree.add_node(parent, st, 200.0, 200.0)
    tree.solve(RectF(0.0, 0.0, 400.0, 400.0), 0)
    ctx.assert_equal(len(tree.nodes), 20, "20-level FlexTree hierarchy solved")
    ctx.assert_true(tree.nodes[19].rect.w > 0.0 and tree.nodes[19].rect.h > 0.0, "deepest node retains positive layout")

    var dock = DockSpace.create_studio_5panel()
    var json_s = dock.save_json()
    var dock2 = DockSpace()
    ctx.assert_true(dock2.load_json(json_s), "DockSpace JSON round-trip succeeds")


# =========================================================================
# F16: Forsyth Vertex Cache & ACMR Optimization (Boundary)
# =========================================================================

def test_b_f16_single_triangle_optimization(mut ctx: TestContext) raises:
    var idx = List[Int]()
    idx.append(0); idx.append(1); idx.append(2)
    var opt = optimize_vertex_cache_forsyth(idx, 3, 32)
    ctx.assert_equal(len(opt), 3, "single triangle preserves 3 indices")
    ctx.assert_approx(compute_acmr(opt, 32), 3.0, 0.01, "single triangle ACMR is 3.0")


def test_b_f16_degenerate_triangles_handling(mut ctx: TestContext) raises:
    var idx = List[Int]()
    idx.append(0); idx.append(0); idx.append(0)
    var opt = optimize_vertex_cache_forsyth(idx, 1, 32)
    ctx.assert_equal(len(opt), 3, "degenerate triangle handled without crashing")


def test_b_f16_high_valence_fan(mut ctx: TestContext) raises:
    var idx = List[Int]()
    for i in range(16):
        idx.append(0); idx.append(i + 1); idx.append(i + 2)
    var opt = optimize_vertex_cache_forsyth(idx, 18, 32)
    ctx.assert_equal(len(opt), 48, "16-tri fan retains 48 indices")
    ctx.assert_true(compute_acmr(opt, 32) < 1.5, "shared apex achieves low ACMR")


def test_b_f16_cache_size_boundary_small(mut ctx: TestContext) raises:
    var idx = List[Int]()
    idx.append(0); idx.append(1); idx.append(2)
    idx.append(1); idx.append(3); idx.append(2)
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
    ctx.assert_true(validate_spirv_module(words).valid, "minimal 1x1x1 compute module passes validation")


def test_b_f17_maximum_compute_workgroup_1024x1x1(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(1024, 1, 1)
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "1024x1x1 compute module passes validation")


def test_b_f17_repeated_barriers(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 64, 1, 1)
    var sc_exec = m.uint(2)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    for _ in range(3):
        emit(m.code, OP_CONTROL_BARRIER, sc_exec, sc_mem, sem)
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "consecutive control barriers pass SSA validation")


def test_b_f17_nested_branching_module(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(64, 1, 1)
    var c1 = m.uint(10)
    var c2 = m.uint(20)
    m.begin_if(m.ult(c1, c2))
    _ = m.iadd(c1, c2)
    m.end_if()
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "nested conditional branches pass validation")


def test_b_f17_multiple_float_math_ops(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(32, 1, 1)
    var a = m.float(1.5)
    var b = m.float(2.5)
    var s = m.fadd(a, b)
    var d = m.fsub(s, a)
    _ = m.fmul(d, b)
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "sequential float ALU ops pass validation")


# =========================================================================
# Aggregator for Tier 2 Milestone 2
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T2_M2_GpuLoop_Boundary")
    var ren = FlintAaaVulkanRenderer(width=160, height=120)

    try: test_b_f13_zero_sized_extent(ctx, ren); ctx.record_pass("test_b_f13_zero_sized_extent") except e: ctx.record_fail("test_b_f13_zero_sized_extent", String(e))
    try: test_b_f13_extreme_8k_resolution(ctx, ren); ctx.record_pass("test_b_f13_extreme_8k_resolution") except e: ctx.record_fail("test_b_f13_extreme_8k_resolution", String(e))
    try: test_b_f13_ultrawide_32_9_aspect(ctx, ren); ctx.record_pass("test_b_f13_ultrawide_32_9_aspect") except e: ctx.record_fail("test_b_f13_ultrawide_32_9_aspect", String(e))
    try: test_b_f13_rapid_resize_counter(ctx, ren); ctx.record_pass("test_b_f13_rapid_resize_counter") except e: ctx.record_fail("test_b_f13_rapid_resize_counter", String(e))
    try: test_b_f13_single_pixel_extent(ctx, ren); ctx.record_pass("test_b_f13_single_pixel_extent") except e: ctx.record_fail("test_b_f13_single_pixel_extent", String(e))

    try: test_b_f14_frame_index_wrap_boundary(ctx, ren); ctx.record_pass("test_b_f14_frame_index_wrap_boundary") except e: ctx.record_fail("test_b_f14_frame_index_wrap_boundary", String(e))
    try: test_b_f14_zero_dt_paused_state(ctx); ctx.record_pass("test_b_f14_zero_dt_paused_state") except e: ctx.record_fail("test_b_f14_zero_dt_paused_state", String(e))
    try: test_b_f14_negative_dt_clamping(ctx); ctx.record_pass("test_b_f14_negative_dt_clamping") except e: ctx.record_fail("test_b_f14_negative_dt_clamping", String(e))
    try: test_b_f14_dt_spike_ceiling(ctx); ctx.record_pass("test_b_f14_dt_spike_ceiling") except e: ctx.record_fail("test_b_f14_dt_spike_ceiling", String(e))
    try: test_b_f14_dynamic_ubo_offset_alignment(ctx, ren); ctx.record_pass("test_b_f14_dynamic_ubo_offset_alignment") except e: ctx.record_fail("test_b_f14_dynamic_ubo_offset_alignment", String(e))

    try: test_b_f15_empty_string_measurement(ctx); ctx.record_pass("test_b_f15_empty_string_measurement") except e: ctx.record_fail("test_b_f15_empty_string_measurement", String(e))
    try: test_b_f15_single_glyph_measurement(ctx); ctx.record_pass("test_b_f15_single_glyph_measurement") except e: ctx.record_fail("test_b_f15_single_glyph_measurement", String(e))
    try: test_b_f15_negative_coordinate_widget_clipping(ctx); ctx.record_pass("test_b_f15_negative_coordinate_widget_clipping") except e: ctx.record_fail("test_b_f15_negative_coordinate_widget_clipping", String(e))
    try: test_b_f15_zero_size_widget(ctx); ctx.record_pass("test_b_f15_zero_size_widget") except e: ctx.record_fail("test_b_f15_zero_size_widget", String(e))
    try: test_b_f15_deeply_nested_widget_tree(ctx); ctx.record_pass("test_b_f15_deeply_nested_widget_tree") except e: ctx.record_fail("test_b_f15_deeply_nested_widget_tree", String(e))

    try: test_b_f16_single_triangle_optimization(ctx); ctx.record_pass("test_b_f16_single_triangle_optimization") except e: ctx.record_fail("test_b_f16_single_triangle_optimization", String(e))
    try: test_b_f16_degenerate_triangles_handling(ctx); ctx.record_pass("test_b_f16_degenerate_triangles_handling") except e: ctx.record_fail("test_b_f16_degenerate_triangles_handling", String(e))
    try: test_b_f16_high_valence_fan(ctx); ctx.record_pass("test_b_f16_high_valence_fan") except e: ctx.record_fail("test_b_f16_high_valence_fan", String(e))
    try: test_b_f16_cache_size_boundary_small(ctx); ctx.record_pass("test_b_f16_cache_size_boundary_small") except e: ctx.record_fail("test_b_f16_cache_size_boundary_small", String(e))
    try: test_b_f16_cache_size_boundary_large(ctx); ctx.record_pass("test_b_f16_cache_size_boundary_large") except e: ctx.record_fail("test_b_f16_cache_size_boundary_large", String(e))

    try: test_b_f17_minimal_compute_workgroup_1x1x1(ctx); ctx.record_pass("test_b_f17_minimal_compute_workgroup_1x1x1") except e: ctx.record_fail("test_b_f17_minimal_compute_workgroup_1x1x1", String(e))
    try: test_b_f17_maximum_compute_workgroup_1024x1x1(ctx); ctx.record_pass("test_b_f17_maximum_compute_workgroup_1024x1x1") except e: ctx.record_fail("test_b_f17_maximum_compute_workgroup_1024x1x1", String(e))
    try: test_b_f17_repeated_barriers(ctx); ctx.record_pass("test_b_f17_repeated_barriers") except e: ctx.record_fail("test_b_f17_repeated_barriers", String(e))
    try: test_b_f17_nested_branching_module(ctx); ctx.record_pass("test_b_f17_nested_branching_module") except e: ctx.record_fail("test_b_f17_nested_branching_module", String(e))
    try: test_b_f17_multiple_float_math_ops(ctx); ctx.record_pass("test_b_f17_multiple_float_math_ops") except e: ctx.record_fail("test_b_f17_multiple_float_math_ops", String(e))

    ren.shutdown()


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
