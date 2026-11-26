"""Area F round 2: scene-upload instrumentation and the parallel upload prepass.

`SotaUploadStats` accumulates wall-clock time per upload phase (`upload_native_model` and `add_mesh`
hooks in `aaa_renderer.mojo`, marked `# --- sota:perf hook ---`), so the bench lab and tests can
report where a scene load goes.

`sota_prepare_model` is the parallel prepass: every texture's BC7/BC5 mip chain and every mesh's
cluster DAG is built on a `JobPool` before `upload_native_model` walks the meshes. The serial walk
then only does the order-dependent work (bindless slots, materials, pool append, BLAS) and consumes
the prepared results, so slots, materials and pool contents are bit-identical to the serial path.
cvar `r.sota_perf.parallel_upload` (`FLINT_SOTA_PARALLEL_UPLOAD=0` restores the serial walk).
"""
from std.collections import List
from std.ffi import external_call
from std.math import max, min
from flint.core.detsort import stable_order
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.assets.model import NativeMesh, NativeModel, NativeTexture
from flint.assets.meshlet import (
    AAA_MAX_MESH_LODS,
    AaaGeometryPool,
    CookedMeshEntry,
    MeshLodSlice,
    _spatial_sort_triangles_sah,
    simplify_mesh_qem,
)
from flint.assets.bc_compress import downsample_argb_2x, downsample_normals_2x, encode_bc5_normals, encode_bc7_rgba
from flint.platform.os import env_now, get_time
from flint.render.sota_geometry_cluster_dag import ClusterDag
from flint.render.sota_geometry_dag_cache import DagCacheStats, build_cluster_dag_cached


struct SotaUploadStats(Copyable, Movable):
    var prepare_s: Float64  # parallel prepass (texture encodes + DAG builds), wall time
    var tex_cpu_s: Float64  # downsample + BC7/BC5 encode (all mips) done in the serial walk
    var tex_gpu_s: Float64  # staging copy + command record + submit/wait + descriptor writes
    var textures: Int
    var mesh_cook_s: Float64  # meshlets / QEM LODs / cluster DAG into the CPU pool
    var dag_build_s: Float64  # part of mesh_cook_s: cluster DAG build in the serial walk
    var mesh_copy_s: Float64  # pool -> host-visible buffer memcpy (+ buffer growth)
    var blas_s: Float64  # BLAS record + build + submit/wait
    var meshes: Int
    var blas_builds: Int
    var submits: Int  # queue submissions made by the upload path
    var prepared_jobs: Int
    var prepared_workers: Int
    var dag_cache_hits: Int

    def __init__(out self):
        self.prepare_s = 0.0
        self.tex_cpu_s = 0.0
        self.tex_gpu_s = 0.0
        self.textures = 0
        self.mesh_cook_s = 0.0
        self.dag_build_s = 0.0
        self.mesh_copy_s = 0.0
        self.blas_s = 0.0
        self.meshes = 0
        self.blas_builds = 0
        self.submits = 0
        self.prepared_jobs = 0
        self.prepared_workers = 0
        self.dag_cache_hits = 0

    def summary(self) -> String:
        return (
            "prepass " + _f(self.prepare_s) + " s (" + String(self.prepared_jobs) + " jobs on "
            + String(self.prepared_workers) + " workers, " + String(self.dag_cache_hits) + " DAG cache hits), tex cpu "
            + _f(self.tex_cpu_s) + " s, tex gpu " + _f(self.tex_gpu_s) + " s (" + String(self.textures)
            + " tex), mesh cook " + _f(self.mesh_cook_s) + " s (dag " + _f(self.dag_build_s) + " s), copy "
            + _f(self.mesh_copy_s) + " s, blas " + _f(self.blas_s) + " s (" + String(self.meshes) + " meshes, "
            + String(self.blas_builds) + " BLAS, " + String(self.submits) + " submits)"
        )


def _f(x: Float64) -> String:
    var r = Float64(Int(x * 1000.0 + 0.5)) / 1000.0
    return String(r)


def sota_now() -> Float64:
    return get_time()


# ---------------------------------------------------------------------------------------------
# Parallel upload prepass
# ---------------------------------------------------------------------------------------------
comptime SOTA_PREP_ALBEDO = 0  # BC7 sRGB
comptime SOTA_PREP_NORMAL = 1  # BC5 tangent normals
comptime SOTA_PREP_ORM = 2  # BC7 UNORM, AO forced to 255 when 0
comptime SOTA_PREP_DAG = 3
comptime SOTA_PREP_SORT = 4  # `_spatial_sort_triangles_sah` of a DAG mesh (the vertex-pack order)
comptime SOTA_PREP_LODS = 5  # legacy QEM LOD 1..3 chain + every level's spatial sort


struct SotaPreparedTexture(Copyable, Movable):
    var ready: Bool
    var srgb: Bool
    var bc5: Bool
    var blobs: List[List[UInt8]]
    var widths: List[Int]
    var heights: List[Int]

    def __init__(out self):
        self.ready = False
        self.srgb = False
        self.bc5 = False
        self.blobs = List[List[UInt8]]()
        self.widths = List[Int]()
        self.heights = List[Int]()


def sota_mip_count(width: Int, height: Int, generate_mips: Bool) -> Int:
    """Same rule as `upload_bc7_texture` / `upload_bc5_normal_texture`."""
    var mips = 1
    if generate_mips:
        var d = min(width, height)
        while d > 4 and mips < 8:
            d >>= 1
            mips += 1
    return mips


def sota_encode_bc7_chain(pixels_argb: List[UInt32], width: Int, height: Int, srgb: Bool) -> SotaPreparedTexture:
    """The CPU half of `upload_bc7_texture(generate_mips=True)`: identical blobs."""
    var out = SotaPreparedTexture()
    out.srgb = srgb
    var mips = sota_mip_count(width, height, True)
    var cur_px = pixels_argb.copy()
    var cw = width
    var ch = height
    for m in range(mips):
        out.blobs.append(encode_bc7_rgba(cur_px, cw, ch))
        out.widths.append(cw)
        out.heights.append(ch)
        if m + 1 < mips:
            var ds = downsample_argb_2x(cur_px, cw, ch)
            cur_px = ds[0].copy()
            cw = ds[1]
            ch = ds[2]
    out.ready = True
    return out^


def sota_encode_bc5_chain(normals: List[Vec3], width: Int, height: Int) -> SotaPreparedTexture:
    """The CPU half of `upload_bc5_normal_texture(generate_mips=True)`: identical blobs."""
    var out = SotaPreparedTexture()
    out.bc5 = True
    var mips = sota_mip_count(width, height, True)
    var cur_n = normals.copy()
    var cw = width
    var ch = height
    for m in range(mips):
        out.blobs.append(encode_bc5_normals(cur_n, cw, ch))
        out.widths.append(cw)
        out.heights.append(ch)
        if m + 1 < mips:
            var ds = downsample_normals_2x(cur_n, cw, ch)
            cur_n = ds[0].copy()
            cw = ds[1]
            ch = ds[2]
    out.ready = True
    return out^


def _cap_argb(tx: NativeTexture, cap_dim: Int) -> Tuple[List[UInt32], Int, Int]:
    var cur_px = tx.pixels.copy()
    var cw = tx.width
    var ch = tx.height
    while (cw > cap_dim or ch > cap_dim) and cw >= 8 and ch >= 8:
        var ds = downsample_argb_2x(cur_px, cw, ch)
        cur_px = ds[0].copy()
        cw = ds[1]
        ch = ds[2]
    return (cur_px^, cw, ch)


def sota_prepare_texture(tx: NativeTexture, role: Int, cap_dim: Int) -> SotaPreparedTexture:
    """Exactly the per-role pre-processing of `upload_native_model`, then the encode chain."""
    if tx.width <= 0 or tx.height <= 0 or len(tx.pixels) < tx.width * tx.height:
        return SotaPreparedTexture()
    var c = _cap_argb(tx, cap_dim)
    var w = c[1]
    var h = c[2]
    if role == SOTA_PREP_ALBEDO:
        return sota_encode_bc7_chain(c[0], w, h, True)
    if role == SOTA_PREP_NORMAL:
        var n_vecs = List[Vec3](capacity=w * h)
        for p in c[0]:
            var nx = (Float32((p >> 16) & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
            var ny = (Float32((p >> 8) & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
            var nz = (Float32(p & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
            n_vecs.append(Vec3(nx, ny, max(Float32(0.05), nz)).normalized())
        return sota_encode_bc5_chain(n_vecs, w, h)
    var orm_px = List[UInt32](capacity=w * h)
    for p in c[0]:
        var r_ao = (p >> 16) & 0xFF
        var use_ao = UInt32(255) if r_ao == 0 else r_ao
        orm_px.append(UInt32(0xFF000000) | (use_ao << 16) | (p & 0x0000FFFF))
    return sota_encode_bc7_chain(orm_px, w, h, False)


struct SotaPreparedModel(Movable):
    """Prepass results for one `upload_native_model` call (cleared when it returns)."""
    var active: Bool
    var albedo: List[SotaPreparedTexture]
    var normal: List[SotaPreparedTexture]
    var orm: List[SotaPreparedTexture]
    var dags: List[ClusterDag]
    var dag_ready: List[Bool]
    var sorted: List[List[Int]]  # SOTA_PREP_SORT
    var sorted_ready: List[Bool]
    var lods: List[List[NativeMesh]]  # SOTA_PREP_LODS: levels 1.. (level 0 is the mesh itself)
    var lod_errs: List[List[Float32]]  # levels 0..
    var lod_sorts: List[List[List[Int]]]  # levels 0..
    var lods_ready: List[Bool]
    var current_mesh: Int  # mesh index `upload_native_model` is cooking (-1 outside it)
    var prepare_s: Float64
    var jobs: Int
    var workers: Int

    def __init__(out self):
        self.active = False
        self.albedo = List[SotaPreparedTexture]()
        self.normal = List[SotaPreparedTexture]()
        self.orm = List[SotaPreparedTexture]()
        self.dags = List[ClusterDag]()
        self.dag_ready = List[Bool]()
        self.sorted = List[List[Int]]()
        self.sorted_ready = List[Bool]()
        self.lods = List[List[NativeMesh]]()
        self.lod_errs = List[List[Float32]]()
        self.lod_sorts = List[List[List[Int]]]()
        self.lods_ready = List[Bool]()
        self.current_mesh = -1
        self.prepare_s = 0.0
        self.jobs = 0
        self.workers = 0

    def clear(mut self):
        self.active = False
        self.albedo = List[SotaPreparedTexture]()
        self.normal = List[SotaPreparedTexture]()
        self.orm = List[SotaPreparedTexture]()
        self.dags = List[ClusterDag]()
        self.dag_ready = List[Bool]()
        self.sorted = List[List[Int]]()
        self.sorted_ready = List[Bool]()
        self.lods = List[List[NativeMesh]]()
        self.lod_errs = List[List[Float32]]()
        self.lod_sorts = List[List[List[Int]]]()
        self.lods_ready = List[Bool]()
        self.current_mesh = -1

    def has_sorted(self, m: Int) -> Bool:
        return self.active and m >= 0 and m < len(self.sorted_ready) and self.sorted_ready[m]

    def take_sorted(mut self, m: Int) -> List[Int]:
        self.sorted_ready[m] = False
        var r = self.sorted[m].copy()
        self.sorted[m] = List[Int]()
        return r^

    def has_lods(self, m: Int) -> Bool:
        return self.active and m >= 0 and m < len(self.lods_ready) and self.lods_ready[m]

    def has_texture(self, role: Int, t: Int) -> Bool:
        if not self.active or t < 0:
            return False
        if role == SOTA_PREP_ALBEDO:
            return t < len(self.albedo) and self.albedo[t].ready
        if role == SOTA_PREP_NORMAL:
            return t < len(self.normal) and self.normal[t].ready
        return t < len(self.orm) and self.orm[t].ready

    def take_texture(mut self, role: Int, t: Int) -> SotaPreparedTexture:
        var r: SotaPreparedTexture
        if role == SOTA_PREP_ALBEDO:
            r = self.albedo[t].copy()
            self.albedo[t] = SotaPreparedTexture()
        elif role == SOTA_PREP_NORMAL:
            r = self.normal[t].copy()
            self.normal[t] = SotaPreparedTexture()
        else:
            r = self.orm[t].copy()
            self.orm[t] = SotaPreparedTexture()
        return r^

    def has_dag(self, m: Int) -> Bool:
        return self.active and m >= 0 and m < len(self.dag_ready) and self.dag_ready[m]

    def take_dag(mut self, m: Int) -> ClusterDag:
        self.dag_ready[m] = False
        var d = self.dags.pop(m)
        self.dags.insert(m, ClusterDag())
        return d^


struct _PrepCtx(Movable):
    var model: Int  # NativeModel
    var kinds: Int  # List[Int]
    var items: Int  # List[Int]
    var prep: Int  # SotaPreparedModel
    var hits: Int  # List[Int] per job: DAG .flpk cache hits
    var cap_dim: Int
    var cache_dir: Int  # String
    var mt_pool: Int  # JobPool for the single largest DAG build (0: none)
    var mt_mesh: Int

    def __init__(out self):
        self.model = 0
        self.kinds = 0
        self.items = 0
        self.prep = 0
        self.hits = 0
        self.cap_dim = 1024
        self.cache_dir = 0
        self.mt_pool = 0
        self.mt_mesh = -1


def _prep_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_PrepCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var model = Pointer[NativeModel, ImmutAnyOrigin](unsafe_from_address=c[].model)
    var kinds = Pointer[List[Int], ImmutAnyOrigin](unsafe_from_address=c[].kinds)
    var items = Pointer[List[Int], ImmutAnyOrigin](unsafe_from_address=c[].items)
    var prep = Pointer[SotaPreparedModel, MutAnyOrigin](unsafe_from_address=c[].prep)
    var hits = Pointer[List[Int], MutAnyOrigin](unsafe_from_address=c[].hits)
    var cache_dir = Pointer[String, ImmutAnyOrigin](unsafe_from_address=c[].cache_dir)
    var k = kinds[][job]
    var i = items[][job]
    if k == SOTA_PREP_DAG:
        var st = DagCacheStats()
        var dag: ClusterDag
        if i == c[].mt_mesh and c[].mt_pool != 0:
            # the critical path: the largest mesh also spreads its group simplification over a second pool
            dag = build_cluster_dag_cached(
                model[].meshes[i], cache_dir[], Pointer[JobPool, MutAnyOrigin](unsafe_from_address=c[].mt_pool)[], st
            )
        else:
            var serial = JobPool()  # no workers: the build runs on this worker thread (bit-identical to MT)
            dag = build_cluster_dag_cached(model[].meshes[i], cache_dir[], serial, st)
        prep[].dags[i] = dag^
        prep[].dag_ready[i] = True
        hits[][job] = st.hits
        return
    if k == SOTA_PREP_SORT:
        prep[].sorted[i] = _spatial_sort_triangles_sah(model[].meshes[i], model[].meshes[i].indices)
        prep[].sorted_ready[i] = True
        return
    if k == SOTA_PREP_LODS:
        _prep_lod_chain(model[].meshes[i], prep[], i)
        return
    var t = sota_prepare_texture(model[].textures[i], k, c[].cap_dim)
    if k == SOTA_PREP_ALBEDO:
        prep[].albedo[i] = t^
    elif k == SOTA_PREP_NORMAL:
        prep[].normal[i] = t^
    else:
        prep[].orm[i] = t^


def _prep_lod_chain(mesh: NativeMesh, mut prep: SotaPreparedModel, i: Int):
    """The LOD chain of `AaaGeometryPool._append_mesh_internal` (non-splat, build_lods, >= 32 tris), plus the
    spatial sort `_append_single_lod_geometry_ex` would do on every level."""
    var raw_tris = len(mesh.indices) // 3
    var levels = List[NativeMesh]()
    var errs = List[Float32]()
    var sorts = List[List[Int]]()
    errs.append(Float32(0.0))
    sorts.append(_spatial_sort_triangles_sah(mesh, mesh.indices))
    var cur_m = mesh.copy()
    var prev_err = Float32(0.0)
    for lvl in range(1, AAA_MAX_MESH_LODS):
        var tgt = max(8, raw_tris >> lvl)
        if len(cur_m.indices) // 3 <= tgt + 2:
            break
        var simp = simplify_mesh_qem(cur_m, tgt, lock_boundaries=False)
        var simp_tris = len(simp[0].indices) // 3
        if simp_tris <= 0 or simp_tris >= len(cur_m.indices) // 3:
            break
        var mono_err = max(prev_err * Float32(1.65) + Float32(0.002), prev_err + simp[1])
        cur_m = simp[0].copy()
        sorts.append(_spatial_sort_triangles_sah(simp[0], simp[0].indices))
        levels.append(simp[0].copy())
        errs.append(mono_err)
        prev_err = mono_err
    prep.lods[i] = levels^
    prep.lod_errs[i] = errs^
    prep.lod_sorts[i] = sorts^
    prep.lods_ready[i] = True


def sota_append_mesh_prepared(
    mut pool: AaaGeometryPool, mesh: NativeMesh, material_id: Int, mut prep: SotaPreparedModel, m: Int
) -> CookedMeshEntry:
    """`pool.append_mesh(mesh, material_id, build_lods=True)` from a prepared LOD chain: the same emission
    loop as `_append_mesh_internal`, with each level's sort handed over through `sota_next_sorted`."""
    prep.lods_ready[m] = False
    var entry = CookedMeshEntry()
    var mesh_id = len(pool.meshes)
    var slot_v = pool.total_vertices()
    var slot_i = len(pool.index_words)
    var slot_m = len(pool.meshlets)
    var n_lods = len(prep.lod_errs[m])
    var first_ml_all = len(pool.meshlets)
    var empty = List[UInt32]()
    for lvl in range(n_lods):
        var s_err = prep.lod_errs[m][lvl]
        var p_err = prep.lod_errs[m][lvl + 1] if (lvl + 1 < n_lods) else Float32(65504.0)
        pool.sota_next_sorted = prep.lod_sorts[m][lvl].copy()
        pool.sota_next_sorted_valid = True
        var sl: Tuple[Int, Int, Int, Int, Int, Int, Vec3, Float32]
        if lvl == 0:
            sl = pool._append_single_lod_geometry_ex(mesh, mesh_id, material_id, lvl, s_err, p_err, False, empty)
        else:
            sl = pool._append_single_lod_geometry_ex(prep.lods[m][lvl - 1], mesh_id, material_id, lvl, s_err, p_err, False, empty)
        entry.lods[lvl] = MeshLodSlice(sl[4], sl[5], sl[3], s_err)
        if lvl == 0:
            entry.first_vertex = sl[0]
            entry.vertex_count = sl[1]
            entry.first_index = sl[2]
            entry.triangle_count = sl[3]
            entry.lod0_meshlet_count = sl[5]
            entry.bounds_center = sl[6]
            entry.bounds_radius = sl[7]
    pool.sota_next_sorted_valid = False
    entry.mesh_id = mesh_id
    entry.material_id = material_id
    entry.first_meshlet = first_ml_all
    entry.meshlet_count = len(pool.meshlets) - first_ml_all
    entry.lod_count = n_lods
    pool.meshes.append(entry.copy())
    pool._record_slot(slot_v, slot_i, slot_m)
    prep.lods[m] = List[NativeMesh]()
    prep.lod_sorts[m] = List[List[Int]]()
    return entry^


def sota_upload_worker_count() -> Int:
    """`FLINT_SOTA_UPLOAD_THREADS` (0 = serial prepass), else every online core but one: the upload is a
    one-off burst, and `default_worker_count` assumes SMT siblings (it gives 9 workers on a 20-core GB10)."""
    var e = env_now("FLINT_SOTA_UPLOAD_THREADS")
    if e.byte_length() > 0:
        try:
            return max(0, min(31, Int(e)))
        except:
            pass
    var logical = Int(external_call["sysconf", Int64](Int32(84)))  # _SC_NPROCESSORS_ONLN
    return max(0, min(31, logical - 1))


def sota_upload_parallel_enabled() -> Bool:
    """cvar `r.sota_perf.parallel_upload` (`FLINT_SOTA_PARALLEL_UPLOAD=0` turns the prepass off)."""
    return env_now("FLINT_SOTA_PARALLEL_UPLOAD") != "0"


def sota_prepare_model(
    model: NativeModel,
    mut prep: SotaPreparedModel,
    mut pool: JobPool,
    want_albedo: List[Bool],
    want_normal: List[Bool],
    want_orm: List[Bool],
    want_dag: List[Bool],
    want_lods: List[Bool],
    cap_dim: Int,
    cache_dir: String,
    mt_pool_addr: Int = 0,
) -> Int:
    """Runs every wanted texture encode and DAG build on `pool`, largest first. Returns the DAG cache hits."""
    var t0 = get_time()
    var n_tex = len(model.textures)
    var n_mesh = len(model.meshes)
    prep.clear()
    prep.albedo = List[SotaPreparedTexture](length=n_tex, fill=SotaPreparedTexture())
    prep.normal = List[SotaPreparedTexture](length=n_tex, fill=SotaPreparedTexture())
    prep.orm = List[SotaPreparedTexture](length=n_tex, fill=SotaPreparedTexture())
    for _ in range(n_mesh):
        prep.dags.append(ClusterDag())
    prep.dag_ready = List[Bool](length=n_mesh, fill=False)
    prep.sorted = List[List[Int]](length=n_mesh, fill=List[Int]())
    prep.sorted_ready = List[Bool](length=n_mesh, fill=False)
    prep.lods = List[List[NativeMesh]](length=n_mesh, fill=List[NativeMesh]())
    prep.lod_errs = List[List[Float32]](length=n_mesh, fill=List[Float32]())
    prep.lod_sorts = List[List[List[Int]]](length=n_mesh, fill=List[List[Int]]())
    prep.lods_ready = List[Bool](length=n_mesh, fill=False)
    var kinds = List[Int]()
    var items = List[Int]()
    var cost = List[Int]()
    for t in range(n_tex):
        var px = model.textures[t].width * model.textures[t].height
        if t < len(want_albedo) and want_albedo[t]:
            kinds.append(SOTA_PREP_ALBEDO)
            items.append(t)
            cost.append(px)
        if t < len(want_normal) and want_normal[t]:
            kinds.append(SOTA_PREP_NORMAL)
            items.append(t)
            cost.append(px)
        if t < len(want_orm) and want_orm[t]:
            kinds.append(SOTA_PREP_ORM)
            items.append(t)
            cost.append(px)
    for m in range(n_mesh):
        if m < len(want_dag) and want_dag[m]:
            # DAG build ~1.8 us per triangle vs ~0.09 us per source texel (Forest): rough common unit
            kinds.append(SOTA_PREP_DAG)
            items.append(m)
            cost.append(len(model.meshes[m].indices) * 7)
            kinds.append(SOTA_PREP_SORT)
            items.append(m)
            cost.append(len(model.meshes[m].indices))
        elif m < len(want_lods) and want_lods[m]:
            kinds.append(SOTA_PREP_LODS)
            items.append(m)
            cost.append(len(model.meshes[m].indices) * 6)
    # largest first: the pool hands jobs out in index order, so the long DAG builds start at once
    var neg = List[Int](capacity=len(cost))
    for c in cost:
        neg.append(-c)
    var order = stable_order(neg)
    var k2 = List[Int](capacity=len(order))
    var i2 = List[Int](capacity=len(order))
    for o in order:
        k2.append(kinds[o])
        i2.append(items[o])
    var hits = List[Int](length=len(k2), fill=0)
    var cdir = cache_dir
    var ctx = _PrepCtx()
    ctx.model = Int(Pointer(to=model))
    ctx.kinds = Int(Pointer(to=k2))
    ctx.items = Int(Pointer(to=i2))
    ctx.prep = Int(Pointer(to=prep))
    ctx.hits = Int(Pointer(to=hits))
    ctx.cap_dim = cap_dim
    ctx.cache_dir = Int(Pointer(to=cdir))
    ctx.mt_pool = mt_pool_addr
    for j in range(len(k2)):
        if k2[j] == SOTA_PREP_DAG:  # largest DAG job (jobs are sorted by cost)
            ctx.mt_mesh = i2[j]
            break
    if len(k2) > 0:
        if pool.ready():
            run_parallel_ctx(pool, len(k2), ctx, _prep_worker)
        else:
            for j in range(len(k2)):
                _prep_worker(Int(Pointer(to=ctx)), j)
    # keep everything the workers read through raw addresses alive past the barrier (ASAP destructors)
    _ = len(k2)
    _ = len(i2)
    _ = cdir.byte_length()
    var n_hits = 0
    for h in hits:
        n_hits += h
    prep.active = True
    prep.jobs = len(k2)
    prep.workers = pool.workers
    prep.prepare_s = get_time() - t0
    return n_hits
