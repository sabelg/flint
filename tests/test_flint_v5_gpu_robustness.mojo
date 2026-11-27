"""Flint v5 GPU robustness suite.

  1. `GeomFreeList` and `AaaGeometryPool.remove_mesh` / `append_mesh_reusing` free-range reuse.
  2. `FlintAaaSkinningArena` free + reuse of bind vertices, bone palettes and morph slots.
  3. `VkFaultHook` / `VkRobustState` result latching, `AaaDynamicResolution`, BC7 mip chains.
  4. Wayland `wl_output` monitor bookkeeping (no compositor needed).
  5. GPU (skipped without Vulkan): async `add_mesh` (no idle wait), deferred `remove_mesh` with range
     reuse, VRAM budget query, injected out-of-memory, runtime render scale, and device-loss recovery
     through the injectable fault hook (no real crash).
"""
from std.collections import List
from std.ffi import external_call
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.meshlet import AAA_MESHLET_WORDS, AAA_VERTEX_WORDS, AaaGeometryPool, GeomFreeList
from flint.assets.model import NativeMesh
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.platform.vk_robust import (
    VK_ERROR_DEVICE_LOST,
    VK_ERROR_OUT_OF_DEVICE_MEMORY,
    VkFaultHook,
    VkRobustState,
)
from flint.platform.wayland import WaylandOutputInfo, wayland_pick_output
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_gpu_robustness import AaaDynamicResolution, bc7_mip_count, encode_bc7_mip_chain
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_skinning_rt import AAA_MAX_SKIN_BONES, FlintAaaSkinningArena


def _fail(msg: String):
    print("FAIL:", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _box(size: Float32) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(size, size, size), bevel=0.01)
    return mb^.finish()


def _ball(rings: Int) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.8, 0.8, 0.8), rings=rings, segments=rings + 8)
    return mb^.finish()


def test_free_list_and_pool_reuse():
    var fl = GeomFreeList()
    fl.free(10, 5)
    fl.free(20, 5)
    fl.free(15, 5)  # bridges both neighbours
    _assert(fl.count() == 1 and fl.total() == 15, "free runs coalesce into one")
    _assert(fl.alloc(4) == 10 and fl.total() == 11, "first-fit allocation from the front")
    _assert(fl.trim_tail(25) == 14 and fl.total() == 0, "a run ending at the array end trims the tail")
    _assert(fl.alloc(1) == -1, "empty list allocates nothing")

    var pool = AaaGeometryPool()
    var a = pool.append_mesh_reusing(_box(1.0), 0)
    var b = pool.append_mesh_reusing(_ball(10), 1)
    var c = pool.append_mesh_reusing(_box(2.0), 2)
    var verts_full = pool.total_vertices()
    var idx_full = len(pool.index_words)
    var mls_full = len(pool.meshlets)
    _assert(pool.live_mesh_count() == 3, "three live meshes")
    _assert(pool.remove_mesh(b.mesh_id), "remove the middle mesh")
    _assert(not pool.remove_mesh(b.mesh_id), "a removed mesh cannot be removed twice")
    _assert(not pool.is_mesh_live(b.mesh_id) and pool.meshes[b.mesh_id].lod_count == 0, "removed entry is empty")
    _assert(pool.total_vertices() == verts_full, "a hole in the middle keeps the array length")
    _assert(pool.free_vertex_count() == b.vertex_count, "the hole is on the free list")

    # The same mesh again fits the hole exactly: arrays do not grow, data matches a fresh cook.
    var d = pool.append_mesh_reusing(_ball(10), 3)
    _assert(d.mesh_id == 3, "mesh ids are never reused")
    _assert(pool.total_vertices() == verts_full and len(pool.index_words) == idx_full and len(pool.meshlets) == mls_full,
            "the new mesh reused the freed ranges (no growth)")
    _assert(d.first_vertex == b.first_vertex and d.first_index == b.first_index and d.first_meshlet == b.first_meshlet,
            "relocated into the hole")
    _assert(pool.relocated_meshes >= 1 and pool.free_vertex_count() == 0, "free list consumed")
    var ref_pool = AaaGeometryPool()
    var r = ref_pool.append_mesh_reusing(_ball(10), 3)
    for k in range(r.vertex_count * AAA_VERTEX_WORDS):
        _assert(pool.vertex_words[d.first_vertex * AAA_VERTEX_WORDS + k] == ref_pool.vertex_words[k], "vertex words match")
    var n_idx = pool.mesh_spans[d.mesh_id].index_count
    for k in range(n_idx):
        _assert(
            Int(pool.index_words[d.first_index + k]) - d.first_vertex == Int(ref_pool.index_words[k]),
            "index words rebased onto the new vertex range",
        )
    for m in range(d.meshlet_count):
        var ml = pool.meshlets[d.first_meshlet + m].copy()
        var rl = ref_pool.meshlets[m].copy()
        _assert(ml.first_tri - d.first_index // 3 == rl.first_tri and ml.tri_count == rl.tri_count, "meshlet triangles rebased")
        _assert(Int(pool.meshlet_words[(d.first_meshlet + m) * AAA_MESHLET_WORDS + 8]) == ml.first_tri, "meshlet words follow")
    _assert(pool.meshes[d.mesh_id].lods[0].first_meshlet == d.first_meshlet, "LOD slices rebased")

    # Removing the last mesh shrinks the arrays instead of leaving a hole.
    _assert(pool.remove_mesh(c.mesh_id), "remove the tail mesh")
    _assert(pool.total_vertices() == verts_full - c.vertex_count and pool.free_vertex_count() == 0, "tail removal trims")
    _assert(pool.is_mesh_live(a.mesh_id) and pool.is_mesh_live(d.mesh_id), "other meshes stay live")
    print("[PASS] geometry pool: remove_mesh + free-range reuse (" + String(pool.relocated_meshes) + " relocations)")


def test_skin_arena_reuse():
    var arena = FlintAaaSkinningArena()
    var v0 = arena.alloc_vertices(1000)
    var v1 = arena.alloc_vertices(500)
    var b0 = arena.alloc_bone_palette(64)
    var b1 = arena.alloc_bone_palette(32)
    var m0 = arena.alloc_morph_targets(4, 1000)
    _assert(v0[0] == 0 and v1[0] == 1000 and b0 == 0 and b1 == 64 * 8, "bump allocation")
    arena.free_vertices_range(v0[0], v0[1])
    arena.free_bone_palette(b0, 64)
    arena.free_morph_targets(m0[0], m0[1], m0[2], 1000)
    _assert(arena.vertices_in_use() == 500 and arena.bones_in_use() == 32, "freed ranges leave the in-use totals")
    _assert(arena.morph_weight_cursor == 0 and arena.morph_delta_cursor == 0, "tail morph frees pull the cursors back")
    var v2 = arena.alloc_vertices(800)
    _assert(v2[0] == 0 and v2[1] == 800 and arena.vertex_cursor == 1500, "freed vertices reused without growth")
    var b2 = arena.alloc_bone_palette(48)
    _assert(b2 == 0 and arena.bone_cursor == 96, "freed bone slots reused")
    arena.free_vertices_range(v1[0], v1[1])
    _assert(arena.vertex_cursor == 800, "freeing the tail run coalesces and shrinks the cursor")
    var big = arena.alloc_bone_palette(AAA_MAX_SKIN_BONES)
    _assert(big == -1 and arena.failed_allocations >= 1, "over-capacity requests are counted")
    print("[PASS] skinning arena: free + reuse of vertices, bones and morph slots")


def test_fault_hook_and_state():
    var hook = VkFaultHook()
    _assert(not hook.tick_submit(), "hook off by default")
    hook.lose_device_after_submits = 2
    _assert(not hook.tick_submit() and not hook.tick_submit() and hook.tick_submit(), "third submit is lost")
    _assert(not hook.tick_submit(), "the hook fires once")
    hook.fail_next_allocations = 2
    _assert(hook.take_alloc_failure() and hook.take_alloc_failure() and not hook.take_alloc_failure(), "two OOMs")
    var st = VkRobustState()
    _ = st.note_result(VK_ERROR_OUT_OF_DEVICE_MEMORY, "alloc")
    _assert(st.oom_events == 1 and not st.device_lost, "OOM counted, not fatal")
    _assert(st.note_result(VK_ERROR_DEVICE_LOST, "submit") and st.device_lost, "device lost latched")
    _ = st.note_result(VK_ERROR_DEVICE_LOST, "wait")
    _assert(st.device_lost_events == 1 and st.device_lost_reason == "submit", "first loss reason kept")

    var dr = AaaDynamicResolution(target_ms=10.0, min_scale=0.5, max_scale=1.0)
    dr.settle_frames = 4
    for _ in range(40):
        _ = dr.update(20.0)
    _assert(dr.scale < Float32(0.8) and dr.scale >= Float32(0.5), "over budget lowers the scale")
    var low = dr.scale
    for _ in range(200):
        _ = dr.update(4.0)
    _assert(dr.scale > low, "under budget raises it again")

    var px = List[UInt32](length=64 * 32, fill=UInt32(0xFF336699))
    var chain = encode_bc7_mip_chain(px, 64, 32, True)
    _assert(chain.mip_count() == bc7_mip_count(64, 32, True) and chain.mip_count() == 4, "BC7 mip count")
    _assert(chain.offsets[1] == 64 * 32 and chain.widths[3] == 8 and chain.heights[3] == 4, "BC7 mip layout")
    for o in chain.offsets:
        _assert(o % 16 == 0, "BC7 mip offsets are 16-byte aligned")
    print("[PASS] fault hook, robust state latching, dynamic resolution controller, BC7 mip chain")


def test_wayland_output_bookkeeping():
    var outs = List[WaylandOutputInfo]()
    var a = WaylandOutputInfo()
    a.global_name = 31
    a.name = "DP-1"
    a.width = 2560
    a.height = 1440
    a.refresh_mhz = 144000
    a.scale = 1
    a.done = True
    var b = WaylandOutputInfo()
    b.global_name = 32
    b.name = "HDMI-A-1"
    b.width = 3840
    b.height = 2160
    b.refresh_mhz = 60000
    b.scale = 2
    b.done = True
    outs.append(a.copy())
    outs.append(b.copy())
    _assert(wayland_pick_output(outs, "HDMI-A-1", -1) == 1, "pick by connector name")
    _assert(wayland_pick_output(outs, "", 0) == 0, "pick by index")
    _assert(wayland_pick_output(outs, "", 5) == -1 and wayland_pick_output(outs, "eDP-1", -1) == -1, "unknown -> -1")
    _assert(outs[1].logical_width() == 1920 and outs[1].logical_height() == 1080, "logical size divides by scale")
    print("[PASS] Wayland wl_output monitor list and output selection")


def test_gpu_paths():
    var W = 160
    var H = 96
    var ren = FlintAaaVulkanRenderer(width=W, height=H, tsr_scale=1.0, max_draws=1024, max_instances=256, env_cube_dim=16)
    if not ren.active:
        print("[SKIP] Vulkan unavailable for GPU robustness checks")
        ren.shutdown()
        return
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.bloom_intensity = Float32(0.0)
    var mat = ren.add_material(AaaMaterial(base_color=Vec3(0.7, 0.3, 0.2), roughness=0.6, metallic=0.0))
    var floor_e = ren.add_mesh(_box(1.0), mat)
    var ball_e = ren.add_mesh(_ball(12), mat)
    var idle_waits_free = ren.gpu_robust.uploads.submitted >= 2
    _assert(idle_waits_free, "add_mesh went through the async upload queue")
    var eye = Vec3(0.0, 1.0, 4.0)
    var tgt = Vec3(0.0, 0.8, 0.0)

    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(floor_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    ren.draw_instance(ball_e.mesh_id, trs(Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "baseline frame")
    var base_px = ren.readback_pixels()
    var center = base_px[(H // 2) * W + W // 2]
    _assert((center & 0x00FFFFFF) != 0, "ball visible in the baseline frame")
    print("[PASS] GPU: async add_mesh (" + String(ren.gpu_robust.uploads.submitted) + " uploads, no device-wait-idle) renders")

    # Deferred removal: hidden at once, ranges freed only after the frames retire, then reused.
    var verts_before = ren.geom.total_vertices()
    _assert(ren.remove_mesh(ball_e.mesh_id), "remove_mesh accepted")
    _assert(not ren.remove_mesh(ball_e.mesh_id), "double remove refused")
    _assert(ren.geom.mesh_spans[ball_e.mesh_id].live, "ranges still reserved while frames are in flight")
    var n_before = len(ren.frame_instances)
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(ball_e.mesh_id, trs(Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(len(ren.frame_instances) == 0 or len(ren.frame_instances) == n_before - n_before, "removed mesh is not drawn")
    ren.draw_instance(floor_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    _assert(ren.render_frame(capture_readback=True), "frame after removal")
    ren.flush_gpu_uploads()
    _assert(not ren.geom.mesh_spans[ball_e.mesh_id].live, "ranges released after the frames retired")
    var ball2 = ren.add_mesh(_ball(12), mat)
    _assert(ren.geom.total_vertices() == verts_before and ball2.first_vertex == ball_e.first_vertex, "add_mesh reused the freed range")
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(floor_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    ren.draw_instance(ball2.mesh_id, trs(Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "frame with the reused range")
    var px2 = ren.readback_pixels()
    _assert(px2[(H // 2) * W + W // 2] == center, "reused-range mesh renders identically")
    print("[PASS] GPU: deferred remove_mesh + range reuse renders identically")

    # VRAM budget.
    var heaps = ren.vk.query_memory_budgets()
    var has_local = False
    for h in heaps:
        if h.device_local and h.budget > 0:
            has_local = True
    _assert(len(heaps) > 0 and has_local, "memory budget per heap")
    var headroom = ren.vk.device_local_headroom()
    _assert(headroom != 0, "device-local headroom known")
    ren.vk.robust.fault.budget_override_bytes = 1  # every soft allocation is now over budget
    var refused = ren.allocate_bindless_texture_slot(256, 256, 1, UInt32(37))
    ren.vk.robust.fault.budget_override_bytes = 0
    _assert(refused == -1 and ren.vk.robust.budget_refusals >= 1, "streamed texture refused over budget")
    print("[PASS] GPU: VRAM budget (" + String(len(heaps)) + " heaps, driver numbers=" + String(ren.vk.robust.has_memory_budget)
          + ", headroom " + String(headroom // (1024 * 1024)) + " MiB) and soft-allocation refusal")

    # Injected out-of-memory: uploads fail gracefully, rendering continues.
    var oom_before = ren.vk.robust.oom_events
    ren.vk.robust.fault.fail_next_allocations = 1000000
    var tex = ren.upload_bc7_texture(List[UInt32](length=64 * 64, fill=UInt32(0xFFFFFFFF)), 64, 64)
    var big_mesh = ren.add_mesh(_ball(40), mat)
    ren.vk.robust.fault.fail_next_allocations = 0
    _assert(tex == -1 and ren.vk.robust.oom_events > oom_before, "texture upload under OOM returns -1")
    _assert(ren.gpu_robust.oom_fallbacks > 0, "OOM fallbacks counted")
    _assert(big_mesh.lod_count == 0, "add_mesh under OOM reports failure (lod_count 0) instead of a broken mesh")
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(floor_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    _assert(ren.render_frame(capture_readback=True), "rendering continues after OOM")
    var tex_ok = ren.upload_bc7_texture(List[UInt32](length=64 * 64, fill=UInt32(0xFFFFFFFF)), 64, 64)
    _assert(tex_ok >= 0, "uploads work again once memory is back")
    print("[PASS] GPU: injected OOM degrades (" + String(ren.gpu_robust.oom_fallbacks) + " fallbacks), no crash")

    # Runtime render scale.
    _assert(ren.set_render_scale(0.75) and ren.render_w == 120 and ren.render_h == 72, "render scale 0.75")
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(floor_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    _assert(ren.render_frame(capture_readback=True), "frame at 0.75 scale")
    _assert(ren.set_render_scale(1.0) and ren.render_w == W, "render scale back to 1.0")
    print("[PASS] GPU: runtime render scale 1.0 -> 0.75 -> 1.0")

    # Device loss through the fault hook, then recovery.
    var live_before = ren.geom.live_mesh_count()
    ren.vk.robust.fault.lose_device_after_submits = 0
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(floor_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    _ = ren.render_frame(capture_readback=True)
    _assert(ren.is_device_lost(), "injected VK_ERROR_DEVICE_LOST latched")
    _assert(not ren.render_frame(capture_readback=True), "render_frame refuses on a lost device")
    _assert(ren.recover_from_device_loss(), "recovery: " + ren.gpu_robust.last_recovery_error)
    _assert(ren.active and not ren.is_device_lost() and ren.device_generation() == 1, "new device, generation 1")
    _assert(ren.geom.live_mesh_count() == live_before and len(ren.materials) == 1, "geometry and materials restored")
    ren.tsr_history_weight = Float32(0.0)
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(floor_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    ren.draw_instance(ball2.mesh_id, trs(Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "frame after recovery")
    var px3 = ren.readback_pixels()
    _assert(px3[(H // 2) * W + W // 2] == center, "recovered renderer draws the restored mesh identically")
    print("[PASS] GPU: injected device loss -> recover_from_device_loss (textures lost: "
          + String(ren.gpu_robust.textures_lost_on_recovery) + ") -> identical frame")
    ren.shutdown()


def main():
    print("=== Flint v5 GPU robustness suite ===")
    test_free_list_and_pool_reuse()
    test_skin_arena_reuse()
    test_fault_hook_and_state()
    test_wayland_output_bookkeeping()
    test_gpu_paths()
    print("=== [PASS] GPU robustness suite ===")
