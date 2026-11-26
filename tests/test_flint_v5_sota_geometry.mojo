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
from flint.render.sota_geometry_cluster_dag import (
    ClusterDag,
    SOTA_DAG_MAX_TRIS,
    SOTA_DAG_MAX_VERTS,
    build_cluster_dag,
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


def main():
    test_dag_cpu()
    test_hiz_cpu_and_spirv()
    test_gpu_two_pass_occlusion()
    test_gpu_dag_lod()
    test_gpu_cornell_hiz_regression()
