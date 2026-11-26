"""Area F round 2 (`sota:perf`): upload path, first-frame sunlight, frame pacing, alpha-mask route capacity.

CPU:
- Forsyth on a chunk's compact local vertex numbering == Forsyth on the global numbering (the upload fix).
- The parallel prepass (`sota_prepare_model`) cooks the same geometry as the serial path, bit for bit:
  the legacy LOD chain (`sota_append_mesh_prepared` vs `append_mesh(build_lods=True)`) and the SAH sort.
- `sota_decode_images` (parallel) == `decode_image_auto` (serial).
- Frame-pacing statistics, overlap fraction and route-capacity helpers.
GPU (skipped without Vulkan):
- A fresh renderer's frame 0 is sunlit: the floor is lit, and a sun-direction change shows on the very next
  frame (round 1's "no sun on the first frames" was an unloaded scene; `sota_bench_scene_drawable` guards it).
- Pipelined frames (`deferred_readback`) render; the alpha-mask / outline rings grow and rebind.
"""
from std.collections import List
from std.math import abs, max, min
from flint.assets.image import encode_png
from flint.assets.jpeg import decode_image_auto
from flint.assets.meshlet import AaaGeometryPool, _spatial_sort_triangles_sah, optimize_vertex_cache_forsyth
from flint.assets.model import NativeMesh, NativeModel
from flint.assets.sota_perf_decode import sota_decode_images
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.platform.os import env_now, get_time
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.sota_perf_async import (
    SotaFramePacing,
    sota_overlap_fraction,
    sota_pipelined_default,
    sota_route_capacity_for,
)
from flint.render.sota_perf_bench import (
    SOTA_BENCH_PRESET_DEFAULT,
    sota_bench_apply_preset,
    sota_bench_draw_scene,
    sota_bench_load_scene,
    sota_bench_scene_drawable,
    sota_bench_upload_scene,
    sota_fmt,
)
from flint.render.sota_perf_upload import SotaPreparedModel, sota_append_mesh_prepared, sota_prepare_model
from test_flint_v5_sota_perf import write_colonnade_gltf


def _assert(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("FAIL: " + msg)


struct _Lcg(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Int:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Int((self.s >> 33) & 0x7FFFFFFF)


def _wavy_grid(n: Int, seed: UInt64) -> NativeMesh:
    """(n+1)^2-vertex height-field grid with a shuffled vertex order (so chunks see scattered ids)."""
    var m = NativeMesh()
    var nv = (n + 1) * (n + 1)
    var perm = List[Int](capacity=nv)
    for i in range(nv):
        perm.append(i)
    var rng = _Lcg(seed)
    for i in range(nv - 1, 0, -1):
        var j = rng.next() % (i + 1)
        var t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
    m.positions = List[Float32](length=nv * 3, fill=0.0)
    m.normals = List[Float32](length=nv * 3, fill=0.0)
    m.texcoords = List[Float32](length=nv * 2, fill=0.0)
    for y in range(n + 1):
        for x in range(n + 1):
            var v = perm[y * (n + 1) + x]
            var fx = Float32(x) / Float32(n)
            var fy = Float32(y) / Float32(n)
            m.positions[v * 3] = fx * 10.0
            m.positions[v * 3 + 1] = Float32(((x * 7 + y * 13) % 11)) * 0.05
            m.positions[v * 3 + 2] = fy * 10.0
            m.normals[v * 3 + 1] = 1.0
            m.texcoords[v * 2] = fx
            m.texcoords[v * 2 + 1] = fy
    for y in range(n):
        for x in range(n):
            var a = perm[y * (n + 1) + x]
            var b = perm[y * (n + 1) + x + 1]
            var c = perm[(y + 1) * (n + 1) + x]
            var d = perm[(y + 1) * (n + 1) + x + 1]
            m.indices.append(a)
            m.indices.append(c)
            m.indices.append(b)
            m.indices.append(b)
            m.indices.append(c)
            m.indices.append(d)
    return m^


def test_forsyth_local_remap() raises:
    var mesh = _wavy_grid(40, 7)
    var vc = mesh.vertex_count()
    var start = 0
    var chunks = 0
    while start * 3 < len(mesh.indices):
        var end = min(len(mesh.indices) // 3, start + 248)
        var sub = List[Int]()
        for i in range(start * 3, end * 3):
            sub.append(mesh.indices[i])
        var ref_out = optimize_vertex_cache_forsyth(sub, vc)
        # compact, order-preserving local ids (what `_spatial_sort_triangles_sah` does now)
        var uniq = List[Int]()
        var seen = List[Bool](length=vc, fill=False)
        for v in sub:
            if not seen[v]:
                seen[v] = True
                uniq.append(v)
        sort(uniq)
        var local_of = List[Int](length=vc, fill=-1)
        for li in range(len(uniq)):
            local_of[uniq[li]] = li
        var loc = List[Int]()
        for v in sub:
            loc.append(local_of[v])
        var got = optimize_vertex_cache_forsyth(loc, len(uniq))
        _assert(len(got) == len(ref_out), "forsyth length")
        for i in range(len(got)):
            _assert(uniq[got[i]] == ref_out[i], "forsyth local remap differs at chunk " + String(chunks) + " index " + String(i))
        chunks += 1
        start = end
    print("  [PASS] Forsyth on local ids == global ids (" + String(chunks) + " chunks)")


def _same_pool(a: AaaGeometryPool, b: AaaGeometryPool, what: String) raises:
    _assert(len(a.vertex_words) == len(b.vertex_words), what + ": vertex words " + String(len(a.vertex_words)) + " vs " + String(len(b.vertex_words)))
    _assert(len(a.index_words) == len(b.index_words), what + ": index words")
    _assert(len(a.meshlets) == len(b.meshlets), what + ": meshlets " + String(len(a.meshlets)) + " vs " + String(len(b.meshlets)))
    for i in range(len(a.vertex_words)):
        _assert(a.vertex_words[i] == b.vertex_words[i], what + ": vertex word " + String(i))
    for i in range(len(a.index_words)):
        _assert(a.index_words[i] == b.index_words[i], what + ": index word " + String(i))
    for i in range(len(a.meshlets)):
        _assert(a.meshlets[i].first_tri == b.meshlets[i].first_tri and a.meshlets[i].tri_count == b.meshlets[i].tri_count
                and a.meshlets[i].lod_level == b.meshlets[i].lod_level and a.meshlets[i].self_error == b.meshlets[i].self_error,
                what + ": meshlet " + String(i))


def test_prepass_bit_identical() raises:
    var model = NativeModel()
    model.meshes.append(_wavy_grid(48, 11))   # 4608 tris: legacy LOD chain (>= 1024)
    model.meshes.append(_wavy_grid(30, 3))    # 1800 tris: DAG + SORT jobs
    var pool = JobPool()
    _assert(pool.start(4), "job pool")
    var prep = SotaPreparedModel()
    var no = List[Bool](length=0, fill=False)
    var want_dag: List[Bool] = [False, True]
    var want_lods: List[Bool] = [True, False]
    var t0 = get_time()
    _ = sota_prepare_model(model, prep, pool, no, no, no, want_dag, want_lods, 1024, "")
    var t_prep = get_time() - t0
    _assert(prep.has_lods(0), "LOD chain prepared")
    _assert(prep.has_sorted(1) and prep.has_dag(1), "DAG + sort prepared")
    # legacy LOD chain: prepared emission == serial append_mesh(build_lods=True)
    var serial = AaaGeometryPool()
    var e_s = serial.append_mesh(model.meshes[0], 0, build_lods=True)
    var par = AaaGeometryPool()
    var e_p = sota_append_mesh_prepared(par, model.meshes[0], 0, prep, 0)
    _assert(e_s.lod_count == e_p.lod_count and e_s.meshlet_count == e_p.meshlet_count, "LOD entry " + String(e_s.lod_count) + "/" + String(e_p.lod_count))
    _same_pool(serial, par, "LOD chain")
    # SAH sort job == inline sort
    var sorted = prep.take_sorted(1)
    var ref_sort = _spatial_sort_triangles_sah(model.meshes[1], model.meshes[1].indices)
    _assert(len(sorted) == len(ref_sort), "sorted length")
    for i in range(len(sorted)):
        _assert(sorted[i] == ref_sort[i], "sorted index " + String(i))
    pool.stop()
    print("  [PASS] parallel prepass == serial cook (LOD chain " + String(e_p.lod_count) + " levels, "
          + String(e_p.meshlet_count) + " meshlets; prepass " + sota_fmt(t_prep, 3) + " s)")


def test_parallel_decode() raises:
    var blobs = List[List[UInt8]]()
    for k in range(5):
        var w = 16 + k * 8
        var h = 12 + k * 4
        var px = List[UInt32](length=w * h, fill=0)
        for i in range(w * h):
            px[i] = UInt32(0xFF000000) | UInt32((i * 37 + k * 101) & 0xFFFFFF)
        blobs.append(encode_png(px, w, h))
    blobs.append(List[UInt8]())  # empty -> invalid image
    var par = sota_decode_images(blobs, 4096)
    _assert(len(par) == len(blobs), "decode count")
    for i in range(len(blobs) - 1):
        var ser = decode_image_auto(blobs[i], 4096)
        _assert(par[i].width == ser.width and par[i].height == ser.height and len(par[i].pixels) == len(ser.pixels), "decode dims " + String(i))
        for p in range(len(ser.pixels)):
            _assert(par[i].pixels[p] == ser.pixels[p], "decode pixel " + String(i))
    _assert(not par[len(blobs) - 1].valid, "empty blob")
    print("  [PASS] parallel image decode == serial")


def test_pacing_helpers() raises:
    var p = SotaFramePacing()
    var t = 0.0
    for i in range(100):
        p.mark(t)
        t += 0.016 if i != 50 else 0.048
    _assert(p.count() == 99, "pacing count")
    _assert(abs(p.median_ms() - 16.0) < 0.01, "pacing median " + String(p.median_ms()))
    _assert(p.p99_ms() >= 16.0 and p.hitch_ratio() >= 1.0, "pacing p99")
    _assert(abs(sota_overlap_fraction(30.0, 10.0, 40.0)) < 1e-9, "serial overlap")
    _assert(abs(sota_overlap_fraction(30.0, 10.0, 30.0) - 1.0) < 1e-9, "full overlap")
    _assert(abs(sota_overlap_fraction(30.0, 10.0, 37.0) - 0.3) < 1e-9, "partial overlap")
    _assert(sota_route_capacity_for(56788) == 73728, "route capacity " + String(sota_route_capacity_for(56788)))
    _assert(sota_route_capacity_for(16385) >= 16385 + 4096, "route headroom")
    if env_now("FLINT_SOTA_PIPELINED") != "1":
        _assert(not sota_pipelined_default(), "r.sota_perf.pipelined defaults off")
    print("  [PASS] pacing / overlap / route-capacity helpers")


def _floor_luma(px: List[UInt32], w: Int, h: Int) -> Float64:
    var s = 0.0
    var n = 0
    for y in range(h * 3 // 4, h):
        for x in range(w):
            var p = px[y * w + x]
            s += Float64((p >> 16) & 0xFF) * 0.2126 + Float64((p >> 8) & 0xFF) * 0.7152 + Float64(p & 0xFF) * 0.0722
            n += 1
    return s / Float64(max(1, n))


def _mean_abs_diff(a: List[UInt32], b: List[UInt32]) -> Float64:
    var s = 0.0
    for i in range(min(len(a), len(b))):
        s += Float64(abs(Int((a[i] >> 8) & 0xFF) - Int((b[i] >> 8) & 0xFF)))
    return s / Float64(max(1, min(len(a), len(b))))


def test_gpu_first_frame_sun_and_pipelining(dir: String) raises:
    var w = 320
    var h = 180
    var ren = FlintAaaVulkanRenderer(width=w, height=h)
    if not ren.active:
        ren.shutdown()
        print("  [SKIP] first-frame sun / pipelining (no Vulkan device)")
        return
    var scene = sota_bench_load_scene("colonnade", write_colonnade_gltf(dir))
    _assert(scene.loaded and sota_bench_upload_scene(ren, scene), "colonnade load + upload")
    _assert(sota_bench_scene_drawable(scene), "colonnade drawable")
    scene.cam_eye0 = Vec3(-14.0, 1.7, -14.0)
    scene.cam_tgt0 = Vec3(4.0, 1.5, 6.0)
    scene.sun_dir = Vec3(0.45, 0.62, 0.64).normalized()
    sota_bench_apply_preset(ren, SOTA_BENCH_PRESET_DEFAULT, scene)
    ren.begin_frame(scene.cam_eye0, scene.cam_tgt0, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
    sota_bench_draw_scene(ren, scene)
    _assert(ren.render_frame(), "frame 0")
    var f0 = ren.readback_pixels()
    var luma0 = _floor_luma(f0, w, h)
    ren.sun_dir = Vec3(-0.64, 0.62, 0.45).normalized()
    ren.begin_frame(scene.cam_eye0, scene.cam_tgt0, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
    sota_bench_draw_scene(ren, scene)
    _assert(ren.render_frame(), "frame 1")
    var f1 = ren.readback_pixels()
    var d01 = _mean_abs_diff(f0, f1)
    print("    frame 0 floor luma " + sota_fmt(luma0, 1) + ", |frame0 - frame1 (sun moved)| = " + sota_fmt(d01, 2))
    _assert(luma0 > 8.0, "frame 0 floor is dark (no sun on the first frame?) luma " + String(luma0))
    _assert(d01 > 1.0, "a sun-direction change does not show on the next frame: " + String(d01))
    # pipelined frames (r.sota_perf.pipelined): render, flush, read back
    ren.deferred_readback = True
    for _ in range(4):
        ren.begin_frame(scene.cam_eye0, scene.cam_tgt0, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
        sota_bench_draw_scene(ren, scene)
        _assert(ren.render_frame(), "pipelined frame")
    ren.flush_deferred_frame()
    ren.deferred_readback = False
    var fp = ren.readback_pixels()
    _assert(_floor_luma(fp, w, h) > 8.0, "pipelined frame is dark")
    # alpha-mask / outline route rings grow and rebind (Forest needs ~57k visible alpha-masked meshlets)
    var cap0 = ren.sota_route_draws
    _assert(ren._sota_grow_route_draws(cap0 * 2), "route ring growth")
    _assert(ren.sota_route_draws == cap0 * 2 and ren.buf_alphamask_indirect.size >= 64 + cap0 * 2 * 20, "route ring size")
    ren.begin_frame(scene.cam_eye0, scene.cam_tgt0, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
    sota_bench_draw_scene(ren, scene)
    _assert(ren.render_frame(), "frame after route growth")
    var fg = ren.readback_pixels()
    _assert(_mean_abs_diff(fg, f1) < 6.0, "route growth changed the image")
    ren.shutdown()
    print("  [PASS] fresh renderer is sunlit from frame 0; pipelined frames; route rings " + String(cap0) + " -> " + String(cap0 * 2))


def main() raises:
    var t0 = get_time()
    print("=== Flint v5 Area F round 2 (upload, first-frame sun, pacing) SOTA Suite ===")
    var dir = env_now("TMPDIR")
    if dir.byte_length() == 0:
        dir = "/tmp"
    dir += "/flint_sota_perf_r2_colonnade"
    test_forsyth_local_remap()
    test_prepass_bit_identical()
    test_parallel_decode()
    test_pacing_helpers()
    test_gpu_first_frame_sun_and_pipelining(dir)
    print("=== ALL FLINT V5 SOTA PERF R2 TESTS PASSED in " + sota_fmt(get_time() - t0, 3) + "s ===")
