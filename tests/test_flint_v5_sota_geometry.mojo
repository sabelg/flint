"""Sota:geometry (Area A) verification: cluster-LOD DAG (CPU reference), conservative Hi-Z two-pass occlusion and
the GPU DAG cut in the live AaaRenderer frame (image diffs + perf budgets)."""
from std.collections import List
from std.ffi import external_call
from std.math import abs, cos, max, min, sin, sqrt
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.core.math import Mat4, Vec3
from flint.core.transform import Quat, trs
from flint.platform.os import get_time
from flint.assets.image import write_png
from flint.assets.vfs import load_model
from std.os import getenv
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.image_diff import compare_pixels
from flint.render.render_world import RENDER_TIER_2_AAA_VISBUFFER_RT, ScalableRenderer
from flint.render.sota_geometry_cull import (
    hiz_build_cpu,
    sota_cluster_cull_shader,
    sota_hiz_copy_shader,
    sota_hiz_reduce_shader,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.core.jobs import JobPool, default_worker_count
from flint.render.sota_geometry_dag_cache import (
    DagCacheStats,
    build_cluster_dag_cached,
    mesh_content_hash,
    read_cluster_dag_flpk,
    write_cluster_dag_flpk,
)
from flint.render.sota_geometry_impostor import (
    OctImpostorBake,
    bake_octahedral_impostor,
    build_impostor_cards,
    hemi_oct_cell,
    hemi_oct_dir,
    impostor_lod_select,
    impostor_view_cell,
)
from flint.render.sota_geometry_vt import (
    SOTA_VT_FB_STRIDE,
    VtResidency,
    sota_vt_feedback_shader,
    vt_mip_offset,
    vt_total_pages,
)
from flint.render.sota_geometry_cluster_dag import (
    ClusterDag,
    SOTA_DAG_MAX_TRIS,
    SOTA_DAG_MAX_VERTS,
    build_cluster_dag,
    build_cluster_dag_mt,
    dag_check_monotonic,
    dag_cut_open_edges,
    dag_select_cut,
    weld_positions,
)


def _fail(msg: String):
    print("FAIL:", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _sphere(rings: Int, segs: Int) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), rings=rings, segments=segs)
    return mb^.finish()


def _terrain(n: Int, size: Float32) -> NativeMesh:
    """Open height-field grid (n x n quads) with smooth hills."""
    var mb = MeshBuilder()
    for z in range(n + 1):
        for x in range(n + 1):
            var fx = Float32(x) / Float32(n)
            var fz = Float32(z) / Float32(n)
            var h = Float32(0.08) * sin(fx * 9.0) * cos(fz * 7.0) + Float32(0.05) * sin((fx + fz) * 23.0)
            _ = mb.add_vertex(Vec3((fx - 0.5) * size, h * size, (fz - 0.5) * size), Vec3(0.0, 1.0, 0.0), fx, fz)
    for z in range(n):
        for x in range(n):
            var a = z * (n + 1) + x
            mb.add_quad(a, a + 1, a + n + 2, a + n + 1)
    return mb^.finish()


def _cluster_limits_ok(dag: ClusterDag) -> Bool:
    for c in dag.clusters:
        if c.tri_count() > SOTA_DAG_MAX_TRIS or c.tri_count() == 0:
            return False
        var seen = List[Int]()
        for v in c.tris:
            var f = False
            for s in seen:
                if s == v:
                    f = True
                    break
            if not f:
                seen.append(v)
        if len(seen) > SOTA_DAG_MAX_VERTS:
            return False
    return True


def test_dag_cpu():
    # --- closed sphere: watertight cut at every distance ---
    var sph = _sphere(96, 192)
    var t0 = get_time()
    var dag = build_cluster_dag(sph)
    var build_s = get_time() - t0
    var n_lod0 = dag.lod0_tris
    _assert(n_lod0 > 30000, "sphere LOD0 tris")
    _assert(dag.levels >= 4, "DAG has >= 4 levels, got " + String(dag.levels))
    _assert(_cluster_limits_ok(dag), "cluster 64v/124t limits")
    _assert(dag_check_monotonic(dag), "DAG errors/spheres monotonic")
    var open0 = dag_cut_open_edges(sph, dag, dag_select_cut(dag, Vec3(0.0, 0.0, 3.0), 0.1, 1080, 0.5, 0.0))
    # (the generator's UV seam is not bit-identical, so LOD0 has a few open seam edges; a crack ADDS open edges)
    var prev_tris = n_lod0 + 1
    var dists = List[Float32]()
    dists.append(1.5); dists.append(4.0); dists.append(12.0); dists.append(40.0); dists.append(150.0); dists.append(2000.0)
    var line = String("")
    for d in dists:
        # eye off-axis so the cut mixes levels across the surface
        var eye = Vec3(d * 0.6, d * 0.3, d * 0.74)
        var cut = dag_select_cut(dag, eye, 0.1, 1080, 0.41421356, 1.0)
        var tris = 0
        for ci in cut:
            tris += dag.clusters[ci].tri_count()
        var open = dag_cut_open_edges(sph, dag, cut)
        _assert(open == open0, "cut at d=" + String(d) + " has " + String(open) + " open edges vs LOD0 " + String(open0) + " (crack)")
        _assert(tris <= prev_tris, "cut tris monotonic with distance")
        prev_tris = tris
        line += " d" + String(Int(d)) + "=" + String(tris)
    _assert(prev_tris * 20 < n_lod0, "far cut reduces >20x (" + String(prev_tris) + " vs " + String(n_lod0) + ")")
    print("[PASS] sota DAG sphere: lod0=" + String(n_lod0) + " clusters=" + String(len(dag.clusters)) + " levels="
          + String(dag.levels) + " groups ok/fail=" + String(dag.groups_built) + "/" + String(dag.groups_failed)
          + " build=" + String(Int(build_s * 1000.0)) + "ms cut tris:" + line)

    # --- open terrain: border stays, cut is crack-free (open edges == LOD0 border) ---
    var ter = _terrain(128, 64.0)
    var dag_t = build_cluster_dag(ter)
    _assert(dag_check_monotonic(dag_t), "terrain DAG monotonic")
    var border = dag_cut_open_edges(ter, dag_t, dag_select_cut(dag_t, Vec3(0.0, 5.0, 0.0), 0.1, 1080, 0.41421356, 0.0))
    _assert(border == 128 * 4, "terrain LOD0 border edges = 512, got " + String(border))
    var ter_line = String("")
    for d in dists:
        var cut = dag_select_cut(dag_t, Vec3(d * 0.3, 3.0 + d * 0.2, d), 0.1, 1080, 0.41421356, 1.0)
        var open = dag_cut_open_edges(ter, dag_t, cut)
        var tris = 0
        for ci in cut:
            tris += dag_t.clusters[ci].tri_count()
        _assert(open <= border, "terrain cut at d=" + String(d) + " has cracks: open=" + String(open) + " border=" + String(border))
        ter_line += " " + String(tris)
    print("[PASS] sota DAG terrain: lod0=" + String(dag_t.lod0_tris) + " levels=" + String(dag_t.levels) + " cut tris:" + ter_line)


comptime EVAL_DIR = "/home/user/flint-models/_eval/render/geometry/"


def _median(v: List[Float64]) -> Float64:
    if len(v) == 0:
        return 0.0
    var c = v.copy()
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            if c[j] < c[i]:
                var t = c[i]
                c[i] = c[j]
                c[j] = t
    return c[len(c) // 2]


def _validate_or_fail(words: List[UInt32], name: String):
    var rep = validate_spirv_module(words)
    if not rep.valid:
        _fail(name + " SPIR-V validation failed: " + rep.errors[0])


def test_hiz_cpu_and_spirv():
    # Hi-Z reference: every level-L texel equals the min of the full-res pixels it covers (non-pow2 sizes)
    var w = 37
    var h = 23
    var depth = List[Float32]()
    for i in range(w * h):
        depth.append(Float32((i * 7919) % 1000) / 1000.0)
    var hz = hiz_build_cpu(depth, w, h)
    var levels = len(hz[1])
    _assert(hz[2][levels - 1] == 1 and hz[3][levels - 1] == 1, "Hi-Z ends at 1x1")
    for l in range(levels):
        var lw = hz[2][l]
        var lh = hz[3][l]
        var step = 1 << l
        for ty in range(lh):
            for tx in range(lw):
                var m = Float32(1e30)
                for y in range(ty * step, min(h, (ty + 1) * step)):
                    for x in range(tx * step, min(w, (tx + 1) * step)):
                        m = min(m, depth[y * w + x])
                _assert(hz[0][hz[1][l] + ty * lw + tx] == m, "Hi-Z level " + String(l) + " texel not the min of its footprint")
    _validate_or_fail(sota_cluster_cull_shader(), "sota_cluster_cull_shader")
    _validate_or_fail(sota_hiz_copy_shader(), "sota_hiz_copy_shader")
    _validate_or_fail(sota_hiz_reduce_shader(), "sota_hiz_reduce_shader")
    _validate_or_fail(sota_vt_feedback_shader(), "sota_vt_feedback_shader")
    print("[PASS] sota Hi-Z CPU reference (37x23, " + String(levels) + " levels, exact footprint min) + SPIR-V validation")


def _new_renderer(W: Int, H: Int) -> ScalableRenderer:
    var r = ScalableRenderer(RENDER_TIER_2_AAA_VISBUFFER_RT, width=W, height=H, tsr_scale=1.0)
    if r.uses_vulkan():
        ref g = r.gpu_slot[0]
        g.tsr_history_weight = Float32(0.0)
        g.enable_auto_exposure = False
        g.bloom_intensity = Float32(0.0)
        g.camera_lens.vignette_strength = Float32(0.0)
    return r^


def _occlusion_run(mode: Int, W: Int, H: Int, mut stats_out: List[UInt32], mut ms_out: List[Float64]) -> List[UInt32]:
    """Mode 0: reference (no occlusion culling); 1: sota conservative Hi-Z two-pass; 2: legacy 5-tap path."""
    var renderer = _new_renderer(W, H)
    var px = List[UInt32]()
    if not renderer.uses_vulkan():
        return px^
    ref gpu = renderer.gpu_slot[0]
    gpu.enable_hiz_cull = mode != 0
    gpu.sota_geom.enabled = mode != 2
    gpu.sota_geom.legacy_hiz_occlusion = mode == 2  # opt back into the v5 5-tap test to measure it
    var mat = gpu.add_material(AaaMaterial(base_color=Vec3(0.8, 0.8, 0.8), roughness=0.5, metallic=0.0))
    var mat_r = gpu.add_material(AaaMaterial(base_color=Vec3(0.9, 0.2, 0.15), roughness=0.4, metallic=0.0))
    var sph = _sphere(16, 32)
    var s_entry = gpu.add_mesh(sph, mat)
    var bb = MeshBuilder()
    bb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), 0.02)
    var b_entry = gpu.add_mesh(bb^.finish(), mat)
    for f in range(16):
        var cx = Float32(f) * Float32(0.07) - Float32(0.5)
        gpu.begin_frame(Vec3(cx, 2.0, 12.0), Vec3(cx * 0.5, 1.0, 0.0))
        # two wall halves with a 0.24 m slit, 400 spheres behind, 2 in front
        gpu.draw_instance(b_entry.mesh_id, trs(Vec3(-5.12, 3.0, 4.0), Quat(), Vec3(5.0, 4.0, 0.2)))
        gpu.draw_instance(b_entry.mesh_id, trs(Vec3(5.12, 3.0, 4.0), Quat(), Vec3(5.0, 4.0, 0.2)))
        for gz in range(20):
            for gx in range(20):
                var p = Vec3(Float32(gx - 10) * 0.9 + 0.3, 1.0 + Float32(gz % 3) * 0.4, Float32(-gz) * 1.1)
                gpu.draw_instance(s_entry.mesh_id, trs(p, Quat(), Vec3(0.45, 0.45, 0.45)), material_override=mat_r, uniform_scale=0.45)
        gpu.draw_instance(s_entry.mesh_id, trs(Vec3(-2.0, 1.0, 7.0), Quat(), Vec3(0.6, 0.6, 0.6)), uniform_scale=0.6)
        gpu.draw_instance(s_entry.mesh_id, trs(Vec3(2.5, 1.2, 7.0), Quat(), Vec3(0.8, 0.8, 0.8)), uniform_scale=0.8)
        _ = gpu.render_frame(capture_readback=True)
        if f >= 4:
            var t = Float64(0.0)
            for pt in gpu.get_gpu_pass_timings():
                if pt.pass_name == "MeshletCull" or pt.pass_name == "VisBufferHwRaster":
                    t += pt.gpu_ms
            ms_out.append(t)
        if f == 15:
            px = gpu.readback_pixels()
            gpu.sota_geom.read_stats(gpu.sota_geom.last_slot)
            stats_out = gpu.sota_geom.last_stats.copy()
    renderer.shutdown()
    return px^


def test_gpu_two_pass_occlusion():
    var W = 480
    var H = 270
    var st0 = List[UInt32]()
    var st1 = List[UInt32]()
    var st2 = List[UInt32]()
    var ms0 = List[Float64]()
    var ms1 = List[Float64]()
    var ms2 = List[Float64]()
    var ref_px = _occlusion_run(0, W, H, st0, ms0)
    if len(ref_px) == 0:
        print("[SKIP] Vulkan unavailable: sota GPU occlusion test skipped")
        return
    var sota_px = _occlusion_run(1, W, H, st1, ms1)
    var leg_px = _occlusion_run(2, W, H, st2, ms2)
    _ = write_png(EVAL_DIR + "occl_reference_nocull.png", ref_px, W, H)
    _ = write_png(EVAL_DIR + "occl_sota_hiz.png", sota_px, W, H)
    _ = write_png(EVAL_DIR + "occl_legacy_5tap.png", leg_px, W, H)
    var d_sota = compare_pixels(sota_px, ref_px, W, H, 8)
    var d_leg = compare_pixels(leg_px, ref_px, W, H, 8)
    var drawn = Int(st1[0] + st1[2])
    var occluded = Int(st1[7])
    print("  occlusion: ref drawn=" + String(Int(st0[0] + st0[2])) + " sota drawn p1/p2=" + String(Int(st1[0])) + "/" + String(Int(st1[2]))
          + " queued=" + String(Int(st1[6])) + " final-occluded=" + String(occluded)
          + " | diff vs ref: sota>" + String(d_sota.over_tolerance) + "px legacy>" + String(d_leg.over_tolerance) + "px"
          + " | cull+raster ms ref/sota/legacy=" + String(_median(ms0)) + "/" + String(_median(ms1)) + "/" + String(_median(ms2)))
    _assert(occluded > 3000, "sota Hi-Z rejects most clusters behind the wall (got " + String(occluded) + ")")
    _assert(drawn * 4 < Int(st0[0] + st0[2]), "sota draws < 25% of the unculled clusters")
    # conservative: the image matches the no-culling reference (no missing geometry through the slit)
    _assert(d_sota.over_tolerance <= W * H / 2000, "sota occlusion culling changed the image (false culls): " + d_sota.summary())
    _assert(_median(ms1) < 1.5, "sota cull + raster budget 1.5 ms, got " + String(_median(ms1)))
    print("[PASS] sota two-pass conservative Hi-Z occlusion: image == unculled reference")


def _dark_pixels(px: List[UInt32]) -> Int:
    var n = 0
    for p in px:
        if Int((p >> 16) & 255) < 50 and Int((p >> 8) & 255) < 50 and Int(p & 255) < 50:
            n += 1
    return n


def test_gpu_dag_lod():
    var W = 640
    var H = 360
    var renderer = _new_renderer(W, H)
    if not renderer.uses_vulkan():
        print("[SKIP] Vulkan unavailable: sota GPU DAG test skipped")
        return
    ref gpu = renderer.gpu_slot[0]
    var mat = gpu.add_material(AaaMaterial(base_color=Vec3(0.75, 0.7, 0.65), roughness=0.45, metallic=0.0))
    var sph = _sphere(160, 320)
    var t0 = get_time()
    var dag_entry = gpu.add_mesh_cluster_lod(sph, mat)
    var cook_ms = (get_time() - t0) * 1000.0
    var lod0_entry = gpu.add_mesh(sph, mat)
    var cpu_dag = build_cluster_dag(sph)
    _assert(dag_entry.mesh_id >= 0 and dag_entry.meshlet_count == len(cpu_dag.clusters), "DAG mesh cooked into pool")
    gpu.enable_cone_cull = False
    var dists = List[Float32]()
    dists.append(3.0); dists.append(8.0); dists.append(25.0); dists.append(80.0)
    var tan_y = Float32(0.4663077)   # tan(50 deg / 2)
    for di in range(len(dists)):
        var d = dists[di]
        var eye = Vec3(d * 0.35, d * 0.25, d * 0.9)
        var tris_dag = 0
        var tris_lod0 = 0
        var px_dag = List[UInt32]()
        var px_ref = List[UInt32]()
        var gpu_dag_clusters = 0
        var cull_ms = List[Float64]()
        for pass_i in range(2):
            gpu.enable_hiz_cull = False
            for f in range(6):
                gpu.begin_frame(eye, Vec3(0.0, 0.0, 0.0), fov_y_deg=50.0)
                gpu.draw_instance(dag_entry.mesh_id if pass_i == 0 else lod0_entry.mesh_id, Mat4())
                _ = gpu.render_frame(capture_readback=True)
                for pt in gpu.get_gpu_pass_timings():
                    if pt.pass_name == "MeshletCull" and pass_i == 0 and f >= 2:
                        cull_ms.append(pt.gpu_ms)
            gpu.sota_geom.read_stats(gpu.sota_geom.last_slot)
            var st = gpu.sota_geom.last_stats.copy()
            if pass_i == 0:
                tris_dag = Int(st[1] + st[3])
                gpu_dag_clusters = Int(st[8])
                px_dag = gpu.readback_pixels()
            else:
                tris_lod0 = Int(st[1] + st[3])
                px_ref = gpu.readback_pixels()
        var rtb = gpu.sota_geom.rt_bias_base if gpu.vk.has_ray_query else Float32(0.0)
        var cpu_cut = dag_select_cut(cpu_dag, eye, 0.1, H, tan_y, gpu.max_lod_error_px, rtb, gpu.sota_geom.rt_bias_slope)
        var dark_dag = _dark_pixels(px_dag)
        var dark_ref = _dark_pixels(px_ref)
        var diff = compare_pixels(px_dag, px_ref, W, H, 12)
        _ = write_png(EVAL_DIR + "dag_d" + String(Int(d)) + "_cluster_lod.png", px_dag, W, H)
        _ = write_png(EVAL_DIR + "dag_d" + String(Int(d)) + "_lod0_ref.png", px_ref, W, H)
        print("  DAG d=" + String(d) + ": gpu clusters=" + String(gpu_dag_clusters) + " cpu cut=" + String(len(cpu_cut))
              + " tris dag/lod0=" + String(tris_dag) + "/" + String(tris_lod0) + " diff>12: " + String(diff.over_tolerance)
              + "px mean_abs=" + String(diff.mean_abs) + " dark px=" + String(dark_dag) + "/" + String(dark_ref) + " cull ms=" + String(_median(cull_ms)))
        var lo = len(cpu_cut) - max(2, len(cpu_cut) // 100)
        var hi = len(cpu_cut) + max(2, len(cpu_cut) // 100)
        _assert(gpu_dag_clusters >= lo and gpu_dag_clusters <= hi, "GPU DAG cut matches the CPU reference cut")
        _assert(tris_dag <= tris_lod0, "DAG never draws more than LOD0")
        _assert(dark_dag <= dark_ref + 2, "no RT self-shadow specks on coarse clusters (dark px " + String(dark_dag) + " vs " + String(dark_ref) + ")")
        _assert(diff.mean_abs < 1.5, "1-px DAG cut is visually equivalent to LOD0 (mean_abs " + String(diff.mean_abs) + ")")
        if d >= 25.0:
            _assert(tris_dag * 8 < tris_lod0, "far DAG cut reduces triangles > 8x")
    print("[PASS] sota cluster-LOD DAG on GPU: cut == CPU reference, image == LOD0 at 1 px, cook " + String(Int(cook_ms)) + " ms")
    renderer.shutdown()


def _cornell_quad(mut mb: MeshBuilder, p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, n: Vec3):
    var i0 = mb.add_vertex(p0, n, 0.0, 0.0)
    var i1 = mb.add_vertex(p1, n, 1.0, 0.0)
    var i2 = mb.add_vertex(p2, n, 1.0, 1.0)
    var i3 = mb.add_vertex(p3, n, 0.0, 1.0)
    mb.add_quad(i0, i1, i2, i3)


def _cornell_run(mode: Int, W: Int, H: Int, mut draws: List[Int]) -> List[UInt32]:
    """Area B's Cornell box (camera inside, sun through the open front, RT shadows, no GI/VSM).
    Mode 0: no occlusion culling (reference); 1: sota two-pass Hi-Z; 2: legacy culler with the fixed default
    (5-tap occlusion off); 3: legacy culler with the v5 5-tap occlusion forced on (the reported bug)."""
    var renderer = _new_renderer(W, H)
    var px = List[UInt32]()
    if not renderer.uses_vulkan():
        return px^
    ref ren = renderer.gpu_slot[0]
    ren.enable_vsm = False
    ren.enable_rt_gi = False
    ren.enable_rt_reflections = False
    ren.sun_shadow_softness = Float32(0.0)
    ren.ibl_intensity = Float32(0.0)
    ren.sun_dir = Vec3(-0.32, 0.42, -0.849).normalized()
    ren.sun_radiance = Vec3(3.1, 3.0, 2.9)
    ren.enable_hiz_cull = mode != 0
    ren.sota_geom.enabled = mode <= 1
    ren.sota_geom.legacy_hiz_occlusion = mode == 3
    var m_w = ren.add_material(AaaMaterial(base_color=Vec3(0.76, 0.76, 0.76), roughness=1.0, metallic=0.0))
    var m_r = ren.add_material(AaaMaterial(base_color=Vec3(0.80, 0.08, 0.08), roughness=1.0, metallic=0.0))
    var m_g = ren.add_material(AaaMaterial(base_color=Vec3(0.08, 0.78, 0.12), roughness=1.0, metallic=0.0))
    var b_flr = MeshBuilder()
    var b_ceil = MeshBuilder()
    var b_back = MeshBuilder()
    var b_left = MeshBuilder()
    var b_rght = MeshBuilder()
    for gy in range(16):
        var a0 = Float32(-2.0) + Float32(gy) * Float32(0.25)
        var a1 = a0 + Float32(0.25)
        var z0 = Float32(gy) * Float32(0.25)
        var z1 = z0 + Float32(0.25)
        for gx in range(16):
            var x0 = Float32(-2.0) + Float32(gx) * Float32(0.25)
            var x1 = x0 + Float32(0.25)
            var zx0 = Float32(gx) * Float32(0.25)
            var zx1 = zx0 + Float32(0.25)
            _cornell_quad(b_flr, Vec3(x0, -2.0, z1), Vec3(x1, -2.0, z1), Vec3(x1, -2.0, z0), Vec3(x0, -2.0, z0), Vec3(0.0, 1.0, 0.0))
            _cornell_quad(b_ceil, Vec3(x0, 2.0, z0), Vec3(x1, 2.0, z0), Vec3(x1, 2.0, z1), Vec3(x0, 2.0, z1), Vec3(0.0, -1.0, 0.0))
            _cornell_quad(b_back, Vec3(x0, a0, 4.0), Vec3(x1, a0, 4.0), Vec3(x1, a1, 4.0), Vec3(x0, a1, 4.0), Vec3(0.0, 0.0, -1.0))
            _cornell_quad(b_left, Vec3(-2.0, a0, zx0), Vec3(-2.0, a0, zx1), Vec3(-2.0, a1, zx1), Vec3(-2.0, a1, zx0), Vec3(1.0, 0.0, 0.0))
            _cornell_quad(b_rght, Vec3(2.0, a0, zx1), Vec3(2.0, a0, zx0), Vec3(2.0, a1, zx0), Vec3(2.0, a1, zx1), Vec3(-1.0, 0.0, 0.0))
    var b_blk = MeshBuilder()
    for by in range(6):
        var py0 = Float32(-2.0) + Float32(by) / Float32(6.0) * Float32(1.80)
        var py1 = Float32(-2.0) + Float32(by + 1) / Float32(6.0) * Float32(1.80)
        for bx in range(6):
            var px0 = Float32(-0.75) + Float32(bx) / Float32(6.0) * Float32(1.50)
            var px1 = Float32(-0.75) + Float32(bx + 1) / Float32(6.0) * Float32(1.50)
            var pz0 = Float32(1.45) + Float32(bx) / Float32(6.0) * Float32(1.50)
            var pz1 = Float32(1.45) + Float32(bx + 1) / Float32(6.0) * Float32(1.50)
            _cornell_quad(b_blk, Vec3(px1, py0, 1.45), Vec3(px0, py0, 1.45), Vec3(px0, py1, 1.45), Vec3(px1, py1, 1.45), Vec3(0.0, 0.0, -1.0))
            _cornell_quad(b_blk, Vec3(px0, py0, 2.95), Vec3(px1, py0, 2.95), Vec3(px1, py1, 2.95), Vec3(px0, py1, 2.95), Vec3(0.0, 0.0, 1.0))
            _cornell_quad(b_blk, Vec3(-0.75, py0, pz0), Vec3(-0.75, py0, pz1), Vec3(-0.75, py1, pz1), Vec3(-0.75, py1, pz0), Vec3(-1.0, 0.0, 0.0))
            _cornell_quad(b_blk, Vec3(0.75, py0, pz1), Vec3(0.75, py0, pz0), Vec3(0.75, py1, pz0), Vec3(0.75, py1, pz1), Vec3(1.0, 0.0, 0.0))
            var pt0 = Float32(1.45) + Float32(by) / Float32(6.0) * Float32(1.50)
            var pt1 = Float32(1.45) + Float32(by + 1) / Float32(6.0) * Float32(1.50)
            _cornell_quad(b_blk, Vec3(px0, -0.20, pt1), Vec3(px1, -0.20, pt1), Vec3(px1, -0.20, pt0), Vec3(px0, -0.20, pt0), Vec3(0.0, 1.0, 0.0))
    var ids = List[Int]()
    ids.append(ren.add_mesh(b_flr^.finish(), m_w).mesh_id)
    ids.append(ren.add_mesh(b_ceil^.finish(), m_w).mesh_id)
    ids.append(ren.add_mesh(b_back^.finish(), m_w).mesh_id)
    ids.append(ren.add_mesh(b_left^.finish(), m_r).mesh_id)
    ids.append(ren.add_mesh(b_rght^.finish(), m_g).mesh_id)
    ids.append(ren.add_mesh(b_blk^.finish(), m_w).mesh_id)
    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    for _ in range(6):
        ren.begin_frame(Vec3(0.0, 0.0, -3.5), Vec3(0.0, 0.0, 2.0), fov_y_deg=50.0)
        for i in range(len(ids)):
            ren.draw_instance(ids[i], ident)
        _ = ren.render_frame(capture_readback=True)
        draws.append(ren.last_culled_meshlet_draws)
    px = ren.readback_pixels()
    renderer.shutdown()
    return px^


def test_gpu_cornell_hiz_regression():
    """Regression for area B's report: with v5 two-phase Hi-Z on, the Cornell back wall lost its sun shadow from
    frame 2 on. Cause: the v5 5-tap test samples single-mip depth at the corners of a 2 m wall-strip sphere, hits the
    nearer side walls and block, and false-culls visible back-wall meshlets."""
    var W = 128
    var H = 128
    var d0 = List[Int]()
    var d1 = List[Int]()
    var d2 = List[Int]()
    var d3 = List[Int]()
    var ref_px = _cornell_run(0, W, H, d0)
    if len(ref_px) == 0:
        print("[SKIP] Vulkan unavailable: Cornell Hi-Z regression skipped")
        return
    var sota_px = _cornell_run(1, W, H, d1)
    var safe_px = _cornell_run(2, W, H, d2)
    var bug_px = _cornell_run(3, W, H, d3)
    _ = write_png(EVAL_DIR + "cornell_nocull_ref.png", ref_px, W, H)
    _ = write_png(EVAL_DIR + "cornell_sota_hiz.png", sota_px, W, H)
    _ = write_png(EVAL_DIR + "cornell_legacy_fixed.png", safe_px, W, H)
    _ = write_png(EVAL_DIR + "cornell_legacy_5tap_bug.png", bug_px, W, H)
    var e_sota = compare_pixels(sota_px, ref_px, W, H, 12)
    var e_safe = compare_pixels(safe_px, ref_px, W, H, 12)
    var e_bug = compare_pixels(bug_px, ref_px, W, H, 12)
    var line = String("  cornell draws/frame ref=") + String(d0[5]) + " sota=" + String(d1[5]) + " legacy-fixed=" + String(d2[5])
    line += " legacy-5tap=" + String(d3[5]) + " | diff px sota=" + String(e_sota.over_tolerance) + " legacy-fixed="
    line += String(e_safe.over_tolerance) + " legacy-5tap=" + String(e_bug.over_tolerance)
    print(line)
    for f in range(1, 6):
        _assert(d1[f] == d0[f], "sota Hi-Z keeps every visible Cornell meshlet on frame " + String(f))
        _assert(d2[f] == d0[f], "legacy culler (fixed default) keeps every visible Cornell meshlet on frame " + String(f))
    _assert(e_sota.over_tolerance <= 2, "sota Hi-Z Cornell == unculled reference: " + e_sota.summary())
    _assert(e_safe.over_tolerance <= 2, "legacy fixed-default Cornell == unculled reference: " + e_safe.summary())
    print("[PASS] Cornell back-wall shadow stable with Hi-Z culling (area B regression)")


def test_vt_cpu():
    """Residency: coarse-first streaming, fallback page table, LRU eviction with a pinned root."""
    var r = VtResidency(n_slots=6)
    var vt = r.add_vt(3)                       # 8 x 8 pages at mip 0, 4 mips
    _assert(vt_total_pages(3) == 85, "VT page count 64 + 16 + 4 + 1")
    _assert(r.place(vt, 3, 0, 0, 0, pinned=True) >= 0, "root placed")
    var fb = List[UInt32](length=SOTA_VT_FB_STRIDE, fill=UInt32(0))
    # request mip-0 page (5, 6) and its mip-1 parent (2, 3)
    fb[vt_mip_offset(3, 0) + 6 * 8 + 5] = UInt32(7)
    fb[vt_mip_offset(3, 1) + 3 * 4 + 2] = UInt32(7)
    var tex0 = r.indirection_texels(vt)
    var t = tex0[6 * 16 + 5]
    _assert(Int(t & 255) == 3, "missing mip-0 page falls back to the root mip")
    var ups = r.process_feedback(fb, 7, 1)
    _assert(len(ups) == 1 and ups[0].mip == 1, "streamer drains the coarser page first")
    ups = r.process_feedback(fb, 8, 4)          # stale stamp: nothing requested
    _assert(len(ups) == 0, "only this frame's stamps count")
    ups = r.process_feedback(fb, 7, 4)
    _assert(len(ups) == 1 and ups[0].mip == 0 and ups[0].px == 5 and ups[0].py == 6, "then the mip-0 page")
    var tex1 = r.indirection_texels(vt)
    _assert(Int(tex1[6 * 16 + 5] & 255) == 0, "resident page maps to itself")
    var p1 = tex1[3 * 16 + 8 + 2]
    _assert(Int(p1 & 255) == 1, "mip-1 page maps to itself")
    # fill all 6 slots, then a new request must evict the least recently requested unpinned page
    var fb2 = List[UInt32](length=SOTA_VT_FB_STRIDE, fill=UInt32(0))
    for i in range(6):
        fb2[vt_mip_offset(3, 0) + i] = UInt32(20)
    _ = r.process_feedback(fb2, 20, 8)
    _assert(r.resident_pages() == 6, "atlas full")
    _assert(r.evictions_total >= 1, "eviction happened")
    var root_ok = r.page_slot[vt][vt_mip_offset(3, 3)] >= 0
    _assert(root_ok, "pinned root never evicted")
    print("[PASS] sota VT residency: coarse-first, fallback page table, LRU eviction, pinned root (evictions="
          + String(r.evictions_total) + ")")


def _vt_pattern(dim: Int) -> List[UInt32]:
    """Detail at every scale: 64-texel hashed colour cells, 1-texel lines every 16, a smooth gradient."""
    var px = List[UInt32](length=dim * dim, fill=UInt32(0))
    for y in range(dim):
        for x in range(dim):
            var cell = UInt32(((x >> 6) * 73856093) ^ ((y >> 6) * 19349663))
            cell = (cell ^ (cell >> 13)) * UInt32(0x5BD1E995)
            var r = 60 + Int((cell >> 8) & 127) + (x * 60) // dim
            var g = 60 + Int((cell >> 16) & 127)
            var b = 60 + Int((cell >> 24) & 127) + (y * 60) // dim
            if (x & 15) == 0 or (y & 15) == 0:
                r = r // 3
                g = g // 3
                b = b // 3
            px[y * dim + x] = UInt32(0xFF000000) | (UInt32(min(r, 255)) << 16) | (UInt32(min(g, 255)) << 8) | UInt32(min(b, 255))
    return px^


def _vt_plane(mut mb: MeshBuilder):
    _cornell_quad(mb, Vec3(-20.0, 0.0, 20.0), Vec3(20.0, 0.0, 20.0), Vec3(20.0, 0.0, -20.0), Vec3(-20.0, 0.0, -20.0), Vec3(0.0, 1.0, 0.0))


def _vt_run(use_vt: Bool, W: Int, H: Int, dim: Int, pattern: List[UInt32], mut log: List[String], mut first_px: List[UInt32]) -> List[UInt32]:
    var renderer = _new_renderer(W, H)
    var px = List[UInt32]()
    if not renderer.uses_vulkan():
        return px^
    ref gpu = renderer.gpu_slot[0]
    var base = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.8, metallic=0.0)
    var mat: Int
    var t0 = get_time()
    if use_vt:
        var vt = gpu.add_virtual_texture(pattern, dim)
        _assert(vt >= 0, "add_virtual_texture")
        mat = gpu.add_material_vt(base, vt)
    else:
        var slot = gpu.upload_bc7_texture(pattern, dim, dim, srgb=True, generate_mips=True)
        _assert(slot >= 0, "reference BC7 upload")
        var m = base.copy()
        m.albedo_tex = slot
        mat = gpu.add_material(m)
    var t_load = get_time() - t0
    var mb = MeshBuilder()
    _vt_plane(mb)
    var e = gpu.add_mesh(mb^.finish(), mat)
    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    var fb_ms = List[Float64]()
    var cpu_ms = List[Float64]()
    var converged_at = -1
    var uploads_total = 0
    for f in range(40):
        var cz = Float32(min(f, 30)) * Float32(0.5)   # fly low over the plane, then hold
        gpu.begin_frame(Vec3(0.0, 0.9, -18.0 + cz), Vec3(0.0, 0.0, -9.0 + cz), fov_y_deg=60.0)
        gpu.draw_instance(e.mesh_id, ident)
        _ = gpu.render_frame(capture_readback=True)
        if f == 0:
            first_px = gpu.readback_pixels()
        if use_vt:
            uploads_total += gpu.sota_vt.last_uploads
            if f >= 31 and converged_at < 0 and gpu.sota_vt.last_uploads == 0 and gpu.sota_vt.res.last_missing == 0:
                converged_at = f
            if f >= 4:
                cpu_ms.append(gpu.sota_vt.last_cpu_ms)
                for pt in gpu.get_gpu_pass_timings():
                    if pt.pass_name == "SotaVtFeedback":
                        fb_ms.append(pt.gpu_ms)
    px = gpu.readback_pixels()
    if use_vt:
        log.append("resident=" + String(gpu.sota_vt.res.resident_pages()) + " uploads=" + String(uploads_total)
                   + " converged@frame=" + String(converged_at) + " evictions=" + String(gpu.sota_vt.res.evictions_total)
                   + " feedback ms=" + String(_median(fb_ms)) + " cpu ms=" + String(_median(cpu_ms))
                   + " load s=" + String(t_load))
        _assert(converged_at >= 0, "VT streaming converges (no missing pages) within 9 frames of the camera stopping")
        _assert(_median(fb_ms) < 0.5, "VT feedback pass < 0.5 ms")
    else:
        log.append("reference BC7 load s=" + String(t_load))
    renderer.shutdown()
    return px^


def test_gpu_virtual_texture():
    var W = 640
    var H = 360
    var dim = 4096
    var pattern = _vt_pattern(dim)
    var log = List[String]()
    var first_ref = List[UInt32]()
    var first_vt = List[UInt32]()
    var ref_px = _vt_run(False, W, H, dim, pattern, log, first_ref)
    if len(ref_px) == 0:
        print("[SKIP] Vulkan unavailable: VT GPU test skipped")
        return
    var vt_px = _vt_run(True, W, H, dim, pattern, log, first_vt)
    _ = write_png(EVAL_DIR + "vt_reference_bc7.png", ref_px, W, H)
    _ = write_png(EVAL_DIR + "vt_streamed_converged.png", vt_px, W, H)
    _ = write_png(EVAL_DIR + "vt_frame0_root_only.png", first_vt, W, H)
    var d_final = compare_pixels(vt_px, ref_px, W, H, 24)
    var d_first = compare_pixels(first_vt, ref_px, W, H, 24)
    var full_mb = Float64(dim * dim * 4) * 4.0 / 3.0 / 1048576.0
    print("  VT " + String(dim) + "^2 (" + String(full_mb) + " MB as RGBA8 mips; atlas 18.1 MB fixed): " + log[1])
    print("  VT vs BC7 reference: frame0 (root page only) mean_abs=" + String(d_first.mean_abs) + " >24: "
          + String(d_first.over_tolerance) + " px | converged mean_abs=" + String(d_final.mean_abs) + " >24: "
          + String(d_final.over_tolerance) + " px | " + log[0])
    _assert(d_final.mean_abs * 2.0 < d_first.mean_abs, "streamed pages sharpen the root-only image")
    _assert(d_final.mean_abs < 4.0, "converged VT matches the fully resident BC7 texture (mean_abs)")
    print("[PASS] sota virtual texturing: GPU feedback -> residency streamer -> atlas pages, converged image == reference")


def test_impostor_cpu():
    """Hemi-octahedral mapping round-trips every cell; a sphere bake covers pi/4 of each cell, normals face out."""
    var frames = 8
    for fy in range(frames):
        for fx in range(frames):
            var c = hemi_oct_cell(frames, hemi_oct_dir(frames, fx, fy))
            _assert(c[0] == fx and c[1] == fy, "hemi-oct cell round trip (" + String(fx) + "," + String(fy) + ")")
    var top = hemi_oct_cell(frames, Vec3(0.0, 1.0, 0.0))
    _assert(top[0] >= 3 and top[0] <= 4 and top[1] >= 3 and top[1] <= 4, "straight down lands in the grid centre")
    var sph = _sphere(24, 48)
    var bake = bake_octahedral_impostor(sph, Vec3(0.5, 0.5, 0.5), frames, 32)
    _assert(abs(bake.coverage - Float32(0.7854)) < Float32(0.04), "sphere silhouette covers pi/4 of each cell (got " + String(bake.coverage) + ")")
    # centre texel of any cell faces the viewer: tangent-space normal ~ +z; texel right of centre tilts +x
    var cx = 3 * 32 + 16
    var cy = 5 * 32 + 16
    var n_c = bake.normals[cy * bake.dim + cx]
    var n_r = bake.normals[cy * bake.dim + cx + 10]
    var n_d = bake.normals[(cy + 10) * bake.dim + cx]
    _assert(n_c.z > Float32(0.98), "centre normal faces the camera")
    _assert(n_r.x > Float32(0.4), "right-of-centre normal tilts +T")
    _assert(n_d.y > Float32(0.4), "below-centre normal tilts +B (texture v runs down)")
    var cards = build_impostor_cards(bake)
    _assert(len(cards) == frames * frames and len(cards[0].indices) == 6, "one 2-triangle card per cell")
    print("[PASS] sota impostor CPU: hemi-oct round trip, sphere bake coverage=" + String(bake.coverage) + ", tangent normals")


def _tree(seed: Int) -> NativeMesh:
    """~5k-triangle stylised broadleaf tree: tapered trunk + clustered canopy blobs, vertex coloured."""
    var mb = MeshBuilder(roughness=0.8, metallic=0.0)
    mb.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 2.2, 0.0), 0.28, 0.16, segments=16, color=0xFF5A3E28)
    var h = UInt32(seed * 2654435761 + 12345)
    for i in range(7):
        h = h * UInt32(1664525) + UInt32(1013904223)
        var a = Float32(i) * Float32(0.8975979) + Float32(Int(h >> 24)) * Float32(0.004)
        var rr = Float32(0.55) if i > 0 else Float32(0.0)
        var cy = Float32(2.9) + Float32(Int((h >> 16) & 255)) * Float32(0.003) + (Float32(0.5) if i == 0 else Float32(0.0))
        var g = 110 + Int((h >> 8) & 63)
        var col = UInt32(0xFF000000) | (UInt32(40 + Int(h & 31)) << 16) | (UInt32(g) << 8) | UInt32(38)
        var rad = Float32(1.05) if i == 0 else Float32(0.75)
        mb.add_ellipsoid(Vec3(rr * cos(a), cy, rr * sin(a)), Vec3(rad, rad * 0.85, rad), rings=16, segments=24, color=col)
    return mb^.finish()


def _bg_mask(px: List[UInt32], bg: List[UInt32]) -> List[Bool]:
    var m = List[Bool](length=len(px), fill=False)
    for i in range(len(px)):
        var a = px[i]
        var b = bg[i]
        var d = abs(Int((a >> 16) & 255) - Int((b >> 16) & 255)) + abs(Int((a >> 8) & 255) - Int((b >> 8) & 255)) + abs(Int(a & 255) - Int(b & 255))
        m[i] = d > 18
    return m^


struct ImpostorAsset(Copyable, Movable):
    var first_card: Int
    var frames: Int
    var radius: Float32
    var center: Vec3

    def __init__(out self, first_card: Int, frames: Int, radius: Float32, center: Vec3):
        self.first_card = first_card
        self.frames = frames
        self.radius = radius
        self.center = center


def _upload_impostor(mut gpu: FlintAaaVulkanRenderer, bake: OctImpostorBake) -> ImpostorAsset:
    var alb = gpu.upload_bc7_texture(bake.albedo, bake.dim, bake.dim, srgb=True, generate_mips=True)
    var nrm = gpu.upload_bc5_normal_texture(bake.normals, bake.dim, bake.dim, generate_mips=True)
    _assert(alb >= 0 and nrm >= 0, "impostor atlas upload")
    var m = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.8, metallic=0.0)
    m.albedo_tex = alb
    m.normal_tex = nrm
    m.alpha_cutoff = Float32(0.5)
    var mat = gpu.add_material(m)
    var cards = build_impostor_cards(bake)
    var first = -1
    for i in range(len(cards)):
        var e = gpu.add_mesh(cards[i], mat)
        if i == 0:
            first = e.mesh_id
        _assert(e.mesh_id == first + i, "impostor cards get consecutive mesh ids")
    return ImpostorAsset(first, bake.frames, bake.radius, bake.center)


def _yaw_trs(p: Vec3, yaw: Float32) -> Mat4:
    return trs(p, Quat(0.0, sin(yaw * 0.5), 0.0, cos(yaw * 0.5)), Vec3(1.0, 1.0, 1.0))


def test_gpu_impostor_single():
    var W = 480
    var H = 480
    var renderer = _new_renderer(W, H)
    if not renderer.uses_vulkan():
        print("[SKIP] Vulkan unavailable: impostor GPU test skipped")
        return
    ref gpu = renderer.gpu_slot[0]
    var tree = _tree(1)
    var t0 = get_time()
    var bake = bake_octahedral_impostor(tree, Vec3(1.0, 1.0, 1.0), 8, 128)
    var bake_ms = (get_time() - t0) * 1000.0
    var mesh_mat = gpu.add_material(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.8, metallic=0.0))
    var mesh_e = gpu.add_mesh(tree, mesh_mat)
    var imp = _upload_impostor(gpu, bake)
    var a_png = List[UInt32](length=bake.dim * bake.dim, fill=UInt32(0))
    for i in range(len(a_png)):
        var p = bake.albedo[i]
        a_png[i] = p if (p >> 24) != 0 else UInt32(0xFF202020)
    _ = write_png(EVAL_DIR + "impostor_atlas_albedo.png", a_png, bake.dim, bake.dim)
    var worst_iou = Float64(1.0)
    var worst_mean = Float64(0.0)
    var views = List[Float32]()
    views.append(0.0); views.append(0.7); views.append(2.1)
    for vi in range(len(views)):
        var ang = views[vi]
        var dist = Float32(14.0)
        var eye = Vec3(dist * sin(ang), 4.0, dist * cos(ang))
        var target = Vec3(0.0, 2.4, 0.0)
        var imgs = List[List[UInt32]]()
        for mode in range(3):      # 0 background, 1 mesh, 2 impostor
            for _ in range(4):
                gpu.begin_frame(eye, target, fov_y_deg=30.0)
                if mode == 1:
                    gpu.draw_instance(mesh_e.mesh_id, Mat4())
                elif mode == 2:
                    var k = impostor_view_cell(imp.frames, Vec3(0.0, 0.0, 0.0), 0.0, eye - imp.center)
                    gpu.draw_instance(imp.first_card + k, Mat4())
                _ = gpu.render_frame(capture_readback=True)
            imgs.append(gpu.readback_pixels())
        var m_mesh = _bg_mask(imgs[1], imgs[0])
        var m_imp = _bg_mask(imgs[2], imgs[0])
        var inter = 0
        var uni = 0
        var sum_d = Float64(0.0)
        for i in range(len(m_mesh)):
            if m_mesh[i] and m_imp[i]:
                inter += 1
                var a = imgs[1][i]
                var b = imgs[2][i]
                sum_d += Float64(abs(Int((a >> 16) & 255) - Int((b >> 16) & 255)) + abs(Int((a >> 8) & 255) - Int((b >> 8) & 255)) + abs(Int(a & 255) - Int(b & 255))) / 3.0
            if m_mesh[i] or m_imp[i]:
                uni += 1
        var iou = Float64(inter) / Float64(max(1, uni))
        var mean_in = sum_d / Float64(max(1, inter))
        _ = write_png(EVAL_DIR + "impostor_v" + String(vi) + "_mesh.png", imgs[1], W, H)
        _ = write_png(EVAL_DIR + "impostor_v" + String(vi) + "_impostor.png", imgs[2], W, H)
        print("  impostor view " + String(vi) + " (yaw " + String(ang) + "): silhouette IoU=" + String(iou)
              + " mean_abs inside=" + String(mean_in) + " covered px mesh/imp=" + String(uni))
        worst_iou = min(worst_iou, iou)
        worst_mean = max(worst_mean, mean_in)
    _assert(worst_iou > 0.80, "impostor silhouette matches the mesh (IoU " + String(worst_iou) + ")")
    _assert(worst_mean < 28.0, "impostor shading matches the mesh (mean_abs " + String(worst_mean) + ")")
    print("[PASS] sota octahedral impostor: 8x8 x 128px bake in " + String(Int(bake_ms)) + " ms, " + String(len(tree.indices) // 3)
          + " tris -> 2, worst IoU=" + String(worst_iou) + " worst mean_abs=" + String(worst_mean))
    renderer.shutdown()


def _forest_run(use_imp: Bool, W: Int, H: Int, mut tris_out: List[Int], mut ms_out: List[Float64], mut n_imp: List[Int], mut geo_out: List[Float64]) -> List[UInt32]:
    var renderer = _new_renderer(W, H)
    var px = List[UInt32]()
    if not renderer.uses_vulkan():
        return px^
    ref gpu = renderer.gpu_slot[0]
    var tree = _tree(1)
    var mesh_mat = gpu.add_material(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.8, metallic=0.0))
    var mesh_e = gpu.add_mesh(tree, mesh_mat)
    var imp = _upload_impostor(gpu, bake_octahedral_impostor(tree, Vec3(1.0, 1.0, 1.0), 8, 64))
    var ground = MeshBuilder(roughness=0.9, metallic=0.0, base_color=0xFF4A5A30)
    _cornell_quad(ground, Vec3(-200.0, 0.0, 200.0), Vec3(200.0, 0.0, 200.0), Vec3(200.0, 0.0, -200.0), Vec3(-200.0, 0.0, -200.0), Vec3(0.0, 1.0, 0.0))
    var ground_e = gpu.add_mesh(ground^.finish(), mesh_mat)
    var eye = Vec3(0.0, 6.0, 0.0)
    var target = Vec3(0.0, 3.0, 60.0)
    var tan_y = Float32(0.5773503)  # tan(60/2)
    var n = 40                       # 40 x 40 = 1600 trees (~98k visible meshlets < max_draws 131072), 6 m spacing
    var ms = List[Float64]()
    var geo_ms = List[Float64]()
    var imp_count = 0
    for f in range(10):
        gpu.begin_frame(eye, target, fov_y_deg=60.0)
        gpu.draw_instance(ground_e.mesh_id, Mat4())
        imp_count = 0
        for gz in range(n):
            for gx in range(n):
                var h = UInt32((gx * 73856093) ^ (gz * 19349663))
                h = (h ^ (h >> 13)) * UInt32(0x5BD1E995)
                var p = Vec3(Float32(gx - n // 2) * 6.0 + Float32(Int(h & 255)) * 0.012, 0.0, Float32(gz) * 6.0 + 4.0 + Float32(Int((h >> 8) & 255)) * 0.012)
                var yaw = Float32(Int((h >> 16) & 1023)) * Float32(0.0061359)
                if use_imp and impostor_lod_select(imp.radius, p + imp.center, eye, H, tan_y, 90.0):
                    var k = impostor_view_cell(imp.frames, p, yaw, eye - imp.center)
                    gpu.draw_instance(imp.first_card + k, _yaw_trs(p, yaw))
                    imp_count += 1
                else:
                    gpu.draw_instance(mesh_e.mesh_id, _yaw_trs(p, yaw))
        _ = gpu.render_frame(capture_readback=(f == 9))
        if f >= 3:
            var tot = Float64(0.0)
            var geo = Float64(0.0)
            for pt in gpu.get_gpu_pass_timings():
                tot += pt.gpu_ms
                if pt.pass_name == "MeshletCull" or pt.pass_name == "VisBufferHwRaster" or pt.pass_name == "VsmTileRaster":
                    geo += pt.gpu_ms
            ms.append(tot)
            geo_ms.append(geo)
    gpu.sota_geom.read_stats(gpu.sota_geom.last_slot)
    var tris = Int(gpu.sota_geom.last_stats[1] + gpu.sota_geom.last_stats[3])
    px = gpu.readback_pixels()
    tris_out.append(tris)
    ms_out.append(_median(ms))
    geo_out.append(_median(geo_ms))
    n_imp.append(imp_count)
    renderer.shutdown()
    return px^


def test_gpu_impostor_forest():
    var W = 960
    var H = 540
    var tris = List[Int]()
    var ms = List[Float64]()
    var n_imp = List[Int]()
    var geo = List[Float64]()
    var px_mesh = _forest_run(False, W, H, tris, ms, n_imp, geo)
    if len(px_mesh) == 0:
        print("[SKIP] Vulkan unavailable: impostor forest test skipped")
        return
    var px_imp = _forest_run(True, W, H, tris, ms, n_imp, geo)
    _ = write_png(EVAL_DIR + "forest_mesh_only.png", px_mesh, W, H)
    _ = write_png(EVAL_DIR + "forest_impostor_lod.png", px_imp, W, H)
    var d = compare_pixels(px_imp, px_mesh, W, H, 24)
    print("  forest 1600 trees: mesh-only tris=" + String(tris[0]) + " gpu frame ms=" + String(ms[0]) + " geometry (cull+raster+VSM raster) ms=" + String(geo[0]) + " | impostor LOD ("
          + String(n_imp[1]) + " impostors < 90 px) tris=" + String(tris[1]) + " gpu frame ms=" + String(ms[1]) + " geometry ms=" + String(geo[1])
          + " | image mean_abs=" + String(d.mean_abs) + " >24: " + String(d.over_tolerance) + " px")
    _assert(n_imp[1] > 1300, "most of the forest is distant enough for impostors")
    _assert(tris[1] * 5 < tris[0], "impostor LOD cuts drawn triangles > 5x")
    # the shared GB10 makes whole-frame ms noisy (3 to 12 ms); assert on the geometry passes impostors replace
    _assert(geo[1] < geo[0], "impostor LOD makes the geometry passes faster on the GPU")
    _assert(d.mean_abs < 6.0, "impostor forest looks like the mesh forest (mean_abs " + String(d.mean_abs) + ")")
    print("[PASS] sota impostor forest: " + String(tris[0] // max(1, tris[1])) + "x fewer triangles, geometry GPU " + String(geo[0]) + " -> " + String(geo[1]) + " ms")


def _dags_equal(a: ClusterDag, b: ClusterDag) -> Bool:
    if len(a.clusters) != len(b.clusters) or a.levels != b.levels or a.groups_built != b.groups_built:
        return False
    for i in range(len(a.clusters)):
        ref x = a.clusters[i]
        ref y = b.clusters[i]
        if x.level != y.level or len(x.tris) != len(y.tris) or x.self_error != y.self_error or x.parent_error != y.parent_error:
            return False
        if x.parent_radius != y.parent_radius or x.self_center.x != y.self_center.x or x.parent_center.z != y.parent_center.z:
            return False
        for k in range(len(x.tris)):
            if x.tris[k] != y.tris[k]:
                return False
    return True


def test_dag_mt_and_cache():
    var mesh = _sphere(256, 512)
    var t0 = get_time()
    var ser = build_cluster_dag(mesh)
    var ser_ms = (get_time() - t0) * 1000.0
    var pool = JobPool()
    var workers = default_worker_count()
    _ = pool.start(workers)
    t0 = get_time()
    var par = build_cluster_dag_mt(mesh, pool)
    var par_ms = (get_time() - t0) * 1000.0
    _assert(_dags_equal(ser, par), "JobPool DAG build is bit-identical to the serial build")
    var dir = String("build/sota_dag_cache_test")
    var bdir = String("build")
    _ = external_call["mkdir", Int32](bdir.as_c_string_span(), Int32(0o755))
    _ = external_call["mkdir", Int32](dir.as_c_string_span(), Int32(0o755))
    var key = mesh_content_hash(mesh)
    var path = dir + "/dag_test.flpk"
    _ = external_call["unlink", Int32](path.as_c_string_span())
    t0 = get_time()
    _assert(write_cluster_dag_flpk(path, par, key), "write DAG .flpk")
    var write_ms = (get_time() - t0) * 1000.0
    var back = ClusterDag()
    t0 = get_time()
    _assert(read_cluster_dag_flpk(path, key, back), "read DAG .flpk")
    var read_ms = (get_time() - t0) * 1000.0
    _assert(_dags_equal(ser, back), "cached DAG == built DAG")
    var stale = ClusterDag()
    _assert(not read_cluster_dag_flpk(path, key ^ UInt64(1), stale), "stale key is a miss")
    # cache wrapper: miss -> write, then hit
    var stats = DagCacheStats()
    var cdir = dir + "/auto"
    _ = external_call["mkdir", Int32](cdir.as_c_string_span(), Int32(0o755))
    var d1 = build_cluster_dag_cached(mesh, cdir, pool, stats)
    var d2 = build_cluster_dag_cached(mesh, cdir, pool, stats)
    _assert(stats.hits >= 1 and _dags_equal(d1, d2), "build_cluster_dag_cached hits on the second load")
    var mesh2 = mesh.copy()
    mesh2.positions[0] += Float32(0.001)
    _assert(mesh_content_hash(mesh2) != key, "content hash changes with the mesh")
    pool.stop()
    print("  DAG build " + String(len(mesh.indices) // 3) + " tris, " + String(len(ser.clusters)) + " clusters: serial "
          + String(Int(ser_ms)) + " ms | JobPool x" + String(workers) + " " + String(Int(par_ms)) + " ms ("
          + String(ser_ms / max(par_ms, 0.001)) + "x) | .flpk write " + String(Int(write_ms)) + " ms, cached load "
          + String(read_ms) + " ms (" + String(ser_ms / max(read_ms, 0.001)) + "x)")
    _assert(read_ms * 4.0 < ser_ms, "cached load is > 4x faster than a build")
    if workers >= 2:
        _assert(par_ms < ser_ms, "JobPool build is faster than serial")
    print("[PASS] sota DAG build: JobPool bit-identical to serial, .flpk cache round trip + content-keyed invalidation")


def _scene_run(path: String, mode: Int, cache_dir: String, W: Int, H: Int, mut log: List[String]) -> List[UInt32]:
    """mode 0: legacy discrete QEM LODs, 1: cluster DAG (cold cache), 2: cluster DAG (warm .flpk cache)."""
    var renderer = _new_renderer(W, H)
    var px = List[UInt32]()
    if not renderer.uses_vulkan():
        return px^
    ref gpu = renderer.gpu_slot[0]
    gpu.sota_geom.cluster_lod_auto = mode > 0
    gpu.sota_geom.dag_cache_dir = cache_dir
    var t0 = get_time()
    var model = load_model(path)
    var load_s = get_time() - t0
    t0 = get_time()
    var up = gpu.upload_native_model(model, build_lods=True, max_tex_dim=512)
    var cook_s = get_time() - t0
    var c = model.bounds_center
    var r = max(model.bounds_radius, Float32(1.0))
    var eye = c + Vec3(r * 0.05, r * 0.12, -r * 0.75)
    var target = c + Vec3(0.0, -r * 0.05, 0.0)
    var ms = List[Float64]()
    var cull_ms = List[Float64]()
    for f in range(12):
        gpu.begin_frame(eye, target, fov_y_deg=55.0)
        gpu.draw_native_model(model, up)
        _ = gpu.render_frame(capture_readback=(f == 11))
        if f >= 4:
            var tot = Float64(0.0)
            for pt in gpu.get_gpu_pass_timings():
                tot += pt.gpu_ms
                if pt.pass_name == "MeshletCull":
                    cull_ms.append(pt.gpu_ms)
            ms.append(tot)
    gpu.sota_geom.read_stats(gpu.sota_geom.last_slot)
    var st = gpu.sota_geom.last_stats.copy()
    px = gpu.readback_pixels()
    log.append("mode=" + String(mode) + " meshes=" + String(len(model.meshes)) + " load s=" + String(load_s)
               + " cook s=" + String(cook_s) + " draws=" + String(Int(st[0] + st[2])) + " tris=" + String(Int(st[1] + st[3]))
               + " dag clusters=" + String(Int(st[8])) + " gpu ms=" + String(_median(ms)) + " cull ms=" + String(_median(cull_ms))
               + " cache hits/misses=" + String(gpu.sota_geom.dag_cache_stats.hits) + "/" + String(gpu.sota_geom.dag_cache_stats.misses))
    renderer.shutdown()
    return px^


def test_scene_bench():
    """Area F scenes (opt-in: FLINT_SOTA_SCENES=1; models live outside the repo): legacy LODs vs cluster DAG."""
    if getenv("FLINT_SOTA_SCENES") != "1":
        print("[SKIP] scene bench (set FLINT_SOTA_SCENES=1; needs ~/flint-models/_scenes)")
        return
    var W = 1280
    var H = 720
    var scenes = List[String]()
    scenes.append("forest/Forest.glb")
    scenes.append("bistro/BistroInterior_Wine.glb")
    var root = getenv("HOME") + "/flint-models/_scenes/"
    var cache = String("build/sota_dag_cache_scenes")
    var run_cache = cache + "/run" + String(Int(get_time() * 1000.0))
    var bdir = String("build")
    _ = external_call["mkdir", Int32](bdir.as_c_string_span(), Int32(0o755))
    _ = external_call["mkdir", Int32](cache.as_c_string_span(), Int32(0o755))
    for si in range(len(scenes)):
        var log = List[String]()
        var path = root + scenes[si]
        var px0 = _scene_run(path, 0, String(), W, H, log)
        if len(px0) == 0:
            print("[SKIP] Vulkan unavailable")
            return
        var cdir = run_cache + "_" + String(si)
        _ = external_call["mkdir", Int32](cdir.as_c_string_span(), Int32(0o755))
        var px1 = _scene_run(path, 1, cdir, W, H, log)
        var px2 = _scene_run(path, 2, cdir, W, H, log)
        var name = String("scene_forest") if si == 0 else String("scene_bistro")
        _ = write_png(EVAL_DIR + name + "_legacy_lod.png", px0, W, H)
        _ = write_png(EVAL_DIR + name + "_cluster_dag.png", px1, W, H)
        var d = compare_pixels(px1, px0, W, H, 16)
        var d2 = compare_pixels(px2, px1, W, H, 1)
        print("  " + scenes[si] + ":")
        for l in log:
            print("    " + l)
        print("    DAG vs legacy image mean_abs=" + String(d.mean_abs) + " >16: " + String(d.over_tolerance)
              + " px | warm-cache image == cold (>1: " + String(d2.over_tolerance) + " px)")
    print("[PASS] sota scene bench (numbers above)")


def main():
    test_dag_cpu()
    test_hiz_cpu_and_spirv()
    test_gpu_two_pass_occlusion()
    test_gpu_dag_lod()
    test_gpu_cornell_hiz_regression()
    test_vt_cpu()
    test_gpu_virtual_texture()
    test_dag_mt_and_cache()
    test_impostor_cpu()
    test_gpu_impostor_single()
    test_gpu_impostor_forest()
    test_scene_bench()
