"""Flint v5 render SOTA, Area F round 3: DAG build speed, the default-on DAG cache, async-compute plumbing.

CPU only (no GPU needed):
1. `sota_stable_order` (radix) returns exactly `stable_order`'s permutation, and is faster on 1M keys.
2. The sped-up cluster-DAG build emits the round-2 builder's DAG bit for bit (golden hashes taken from the
   round-2 builder), serial == JobPool, including a faceted mesh where every group is skipped early.
3. The DAG cache is on by default (XDG / HOME / FLINT_DAG_CACHE_DIR / FLINT_DAG_CACHE=0), writes atomically
   (no temporary file left), and stays inside its size budget.
GPU (skipped without Vulkan):
4. Cross-queue timeline waits + queue-family ownership transfers: a side-queue copy feeds a main-queue copy and
   the reverse, 64 MB each, bit-exact.
"""
from std.collections import List
from std.ffi import external_call
from std.math import cos, max, sin
from flint.assets.model import NativeMesh
from flint.core.detsort import stable_order
from flint.core.jobs import JobPool, default_worker_count
from flint.core.lz4 import hash_list
from flint.platform.os import file_stamp, get_time, list_directory, make_dirs, remove_file, write_bytes
from flint.render.sota_geometry_cluster_dag import build_cluster_dag, build_cluster_dag_mt, dag_check_monotonic
from flint.render.sota_geometry_cull import SotaGeometryGpu
from flint.render.sota_geometry_dag_cache import DagCacheStats, build_cluster_dag_cached, encode_cluster_dag
from flint.render.sota_perf_dag_cache import sota_dag_cache_default_dir, sota_dag_cache_prune
from flint.render.sota_perf_sort import sota_stable_order
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.platform.vulkan import ACCESS_HOST_READ, ACCESS_TRANSFER_READ, ACCESS_TRANSFER_WRITE, PIPELINE_HOST, PIPELINE_TRANSFER, get32, put32
from sota_perf_timing import min_of, timing_note, timing_rounds

comptime GOLDEN_SPHERE64: UInt64 = 4792193458194047704   # round-2 builder, encode_cluster_dag(dag, 1)
comptime GOLDEN_GRID160: UInt64 = 4442147612964721461


def _assert(ok: Bool, msg: String):
    if not ok:
        print("  [FAIL] " + msg)
        external_call["exit", NoneType](Int32(1))


def _setenv(name: String, value: String):
    var n = name
    var v = value
    _ = external_call["setenv", Int32](n.as_c_string_span(), v.as_c_string_span(), Int32(1))
    _ = n.byte_length() + v.byte_length()


def _unsetenv(name: String):
    var n = name
    _ = external_call["unsetenv", Int32](n.as_c_string_span())
    _ = n.byte_length()


def _sphere(rings: Int, segs: Int) -> NativeMesh:
    var m = NativeMesh()
    for r in range(rings + 1):
        var th = Float32(3.14159265) * Float32(r) / Float32(rings)
        for sg in range(segs + 1):
            var ph = Float32(6.2831853) * Float32(sg) / Float32(segs)
            var x = sin(th) * cos(ph); var y = cos(th); var z = sin(th) * sin(ph)
            m.positions.append(x); m.positions.append(y); m.positions.append(z)
            m.normals.append(x); m.normals.append(y); m.normals.append(z)
            m.texcoords.append(Float32(sg) / Float32(segs)); m.texcoords.append(Float32(r) / Float32(rings))
    for r in range(rings):
        for sg in range(segs):
            var a = r * (segs + 1) + sg
            var b = a + segs + 1
            m.indices.append(a); m.indices.append(b); m.indices.append(a + 1)
            m.indices.append(a + 1); m.indices.append(b); m.indices.append(b + 1)
    return m^


def _faceted_grid(n: Int) -> NativeMesh:
    """n x n quads, every quad with its own 4 wedges (flat UV islands): all pids are seams, like the Forest's
    2.4M-triangle mesh, so no group can simplify. Some quads are also lifted (open edges)."""
    var m = NativeMesh()
    for y in range(n):
        for x in range(n):
            var lift = Float32(0.05) if ((x * 7 + y * 3) % 5) == 0 else Float32(0.0)
            var base = m.vertex_count()
            for c in range(4):
                var dx = c & 1
                var dy = c >> 1
                m.positions.append(Float32(x + dx) * Float32(0.1)); m.positions.append(lift); m.positions.append(Float32(y + dy) * Float32(0.1))
                m.normals.append(0.0); m.normals.append(1.0); m.normals.append(0.0)
                m.texcoords.append(Float32(dx)); m.texcoords.append(Float32(dy))
            m.indices.append(base); m.indices.append(base + 2); m.indices.append(base + 1)
            m.indices.append(base + 1); m.indices.append(base + 2); m.indices.append(base + 3)
    return m^


def _same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def test_radix_order():
    var seed = UInt64(0x9E3779B97F4A7C15)
    var cases = List[List[Int]]()
    for c in range(5):
        var keys = List[Int]()
        var n = 1500 if c < 4 else 1000000
        for _ in range(n):
            seed = seed * 6364136223846793005 + 1442695040888963407
            var r = Int(seed >> 11)
            if c == 0:
                keys.append(r % 7)                      # heavy ties
            elif c == 1:
                keys.append(r & 0x3FFFFFFFFFFFFFFF)     # full 62-bit keys
            elif c == 2:
                keys.append(42)                         # one value: every pass skipped
            elif c == 3:
                keys.append(r % 1000 - 500)             # negative keys -> merge-sort fallback
            else:
                keys.append(r % 3000017)
        cases.append(keys^)
    for c in range(len(cases)):
        _assert(_same(sota_stable_order(cases[c]), stable_order(cases[c])), "radix order == stable_order, case " + String(c))
    # Robust timing: best of 3 per round, up to timing_rounds() rounds; the radix sort must stay well ahead.
    var rounds = timing_rounds()
    var tm = List[Float64]()
    var tr = List[Float64]()
    var ok = False
    for r in range(rounds):
        for _ in range(3):
            var t0 = get_time()
            _ = stable_order(cases[4])
            tm.append(get_time() - t0)
            t0 = get_time()
            _ = sota_stable_order(cases[4])
            tr.append(get_time() - t0)
        ok = min_of(tr) * 3.0 < min_of(tm)
        timing_note("radix vs merge stable order, 1M keys", ok, r, rounds,
                    String(Int(min_of(tr) * 1000.0)) + " ms vs " + String(Int(min_of(tm) * 1000.0)) + " ms")
        if ok:
            break
    _assert(ok, "radix stable order is > 3x faster than the merge sort on 1M keys")
    print("  [PASS] 1: radix stable order == merge-sort order (ties, 62-bit, constant, negative), "
          + String(min_of(tm) / max(min_of(tr), 1e-9)) + "x faster on 1M keys")


def test_dag_golden():
    var pool = JobPool()
    _ = pool.start(max(2, default_worker_count()))
    var a = _sphere(64, 128)
    var da = build_cluster_dag(a)
    var dam = build_cluster_dag_mt(a, pool)
    var ha = hash_list(encode_cluster_dag(da, 1))
    var ham = hash_list(encode_cluster_dag(dam, 1))
    print("    sphere 16k tris: levels", da.levels, "groups built", da.groups_built, "hash", ha)
    _assert(ha == GOLDEN_SPHERE64 and ham == GOLDEN_SPHERE64, "sphere DAG == round-2 builder (serial and JobPool)")
    _assert(dag_check_monotonic(da), "sphere DAG errors monotonic")
    var b = _faceted_grid(160)
    var db = build_cluster_dag(b)
    var dbm = build_cluster_dag_mt(b, pool)
    var hb = hash_list(encode_cluster_dag(db, 1))
    var hbm = hash_list(encode_cluster_dag(dbm, 1))
    print("    faceted grid 51k tris: levels", db.levels, "groups failed", db.groups_failed, "hash", hb)
    _assert(hb == GOLDEN_GRID160 and hbm == GOLDEN_GRID160, "faceted DAG == round-2 builder (every group skipped early)")
    _assert(db.groups_built == 0 and db.groups_failed > 0, "faceted mesh cannot simplify")
    pool.stop()
    print("  [PASS] 2: sped-up DAG build is bit-identical to the round-2 builder (golden hashes, serial == JobPool)")


def test_cache_default():
    var root = String("build/sota_perf_r3_dag_cache")
    _ = make_dirs(root)
    _unsetenv("FLINT_DAG_CACHE")
    _unsetenv("FLINT_DAG_CACHE_DIR")
    _setenv("XDG_CACHE_HOME", root + "/xdg")
    _assert(sota_dag_cache_default_dir() == root + "/xdg/flint/dag", "default dir under XDG_CACHE_HOME")
    var culler = SotaGeometryGpu()
    _assert(culler.dag_cache_dir == root + "/xdg/flint/dag", "the renderer's DAG cache is on by default")
    _setenv("FLINT_DAG_CACHE_DIR", root + "/explicit")
    _assert(sota_dag_cache_default_dir() == root + "/explicit", "FLINT_DAG_CACHE_DIR overrides")
    _setenv("FLINT_DAG_CACHE", "0")
    _assert(sota_dag_cache_default_dir() == "", "FLINT_DAG_CACHE=0 turns the cache off")
    _unsetenv("FLINT_DAG_CACHE")
    _unsetenv("FLINT_DAG_CACHE_DIR")
    # Miss -> atomic write (directory created on demand, no temporary left behind) -> hit.
    var dir = root + "/xdg/flint/dag"
    var old = list_directory(dir)
    for i in range(len(old)):
        _ = remove_file(dir + "/" + old[i])
    var pool = JobPool()
    _ = pool.start(2)
    var stats = DagCacheStats()
    var m = _sphere(32, 64)
    var d1 = build_cluster_dag_cached(m, culler.dag_cache_dir, pool, stats)
    var d2 = build_cluster_dag_cached(m, culler.dag_cache_dir, pool, stats)
    pool.stop()
    var names = list_directory(dir)
    print("    cache dir entries after miss + hit:", len(names), " hits", stats.hits, "misses", stats.misses, "writes", stats.writes)
    _assert(stats.misses == 1 and stats.writes == 1 and stats.hits == 1, "first load builds and writes, second hits")
    _assert(len(names) == 1 and hash_list(encode_cluster_dag(d1, 1)) == hash_list(encode_cluster_dag(d2, 1)), "one entry, no temporary file; hit == build")
    # Budget: 3 fake 1 MB entries + the real one with a 2 MB budget -> the oldest go first, the newest stays.
    var blob = List[UInt8](length=1048576, fill=7)
    for i in range(3):
        _ = write_bytes(dir + "/old" + String(i) + ".flpk", blob)
        _ = external_call["usleep", Int32](Int32(20000))
    var newest = dir + "/old2.flpk"
    var removed = sota_dag_cache_prune(dir, 2)
    var total = 0
    var left = list_directory(dir)
    for i in range(len(left)):
        total += file_stamp(dir + "/" + left[i])[0]
    print("    prune at 2 MB budget: removed", removed, "left", len(left), "files,", total, "B")
    _assert(removed >= 2 and total <= 2 * 1048576 and file_stamp(newest)[0] > 0, "prune keeps the cache inside its budget, newest first")
    _assert(sota_dag_cache_prune(dir, 2) == 0, "a cache inside its budget is left alone")
    print("  [PASS] 3: DAG cache on by default (XDG/HOME, override, off switch), atomic writes, bounded size")


def _check_words(addr: Int, n: Int, seed: UInt32) -> Int:
    var bad = 0
    for i in range(n):
        if get32(addr + i * 4) != (UInt32(i) * UInt32(2654435761)) ^ seed:
            bad += 1
    return bad


def test_gpu_async_plumbing():
    var ren = FlintAaaVulkanRenderer(width=160, height=120, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] 4: Vulkan renderer unavailable")
        return
    ref vk = ren.vk
    var side = vk.allocate_side_command_buffers(1)
    if side < 0 or not vk.allocate_command_buffers(1):
        print("  [SKIP] 4: no command buffers")
        ren.shutdown()
        return
    var main_cb = len(vk.command_buffers) - 1
    var nbytes = 64 * 1048576
    var n = nbytes // 4
    var a = vk.create_buffer(nbytes, local=False)
    var b = vk.create_buffer(nbytes, local=True)
    var c = vk.create_buffer(nbytes, local=False, cpu_reads=True)
    _assert(a.size > 0 and b.size > 0 and c.size > 0, "async test buffers")
    print("    side queue:", vk.has_side_queue(), " other family:", vk.sota_side_family_differs(), " family", vk.family, "side family", vk.side_family)
    for direction in range(2):
        var seed = UInt32(0x5EED0000 + direction)
        for i in range(n):
            put32(a.cpu_address + i * 4, (UInt32(i) * UInt32(2654435761)) ^ seed)
        _ = external_call["memset", Int](c.cpu_address, Int32(0), nbytes)
        var producer = side if direction == 0 else main_cb
        var consumer = main_cb if direction == 0 else side
        var to_side = direction == 1
        _ = vk.begin_commands(producer)
        vk.copy_buffer(producer, a, 0, b, 0, nbytes)
        vk.sota_buffer_ownership(producer, b, to_side, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), 0, 0)  # release
        var f_prod = vk.sota_submit_waiting(producer, direction == 0, 0, 0, PIPELINE_TRANSFER)
        _ = vk.begin_commands(consumer)
        vk.sota_buffer_ownership(consumer, b, to_side, 0, 0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ))     # acquire
        vk.copy_buffer(consumer, b, 0, c, 0, nbytes)
        vk.memory_barrier2(consumer, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
        var f_cons: UInt64
        if direction == 0:
            f_cons = vk.sota_submit_waiting(consumer, False, 0, f_prod, PIPELINE_TRANSFER)
            _ = vk.wait(f_cons)
        else:
            f_cons = vk.sota_submit_waiting(consumer, True, f_prod, 0, PIPELINE_TRANSFER)
            _ = vk.wait_side(f_cons)
        var bad = _check_words(c.cpu_address, n, seed)
        print("    " + (String("side -> main") if direction == 0 else String("main -> side")) + ": 64 MB through an ownership transfer, words wrong:", bad)
        _assert(f_prod > 0 and f_cons > 0 and bad == 0, "cross-queue wait + ownership transfer delivers every word")
    vk.wait_idle()
    vk.destroy_buffer(a)
    vk.destroy_buffer(b)
    vk.destroy_buffer(c)
    ren.shutdown()
    print("  [PASS] 4: cross-queue timeline waits + queue-family ownership transfers (both directions, bit-exact)")


def main():
    print("=== Flint v5 sota:perf round 3 (DAG build + default DAG cache) ===")
    test_radix_order()
    test_dag_golden()
    test_cache_default()
    test_gpu_async_plumbing()
    print("=== ALL FLINT v5 sota:perf r3 TESTS PASSED ===")
