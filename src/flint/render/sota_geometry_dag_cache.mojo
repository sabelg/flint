"""Sota:geometry — cluster-DAG cache in `.flpk` v3 (versioned `BlobWriter` payload, LZ4 chunk, content-keyed).

A DAG build is deterministic in the mesh content, so the cache key is a 64-bit hash of the positions, normals, UVs
and indices; a stale or foreign entry (hash or version mismatch, CRC failure) is ignored and rebuilt. The payload
is one `FDAG` blob: header counts + per cluster (level, LOD spheres/errors, wedge-index triplets as raw int64).

`build_cluster_dag_cached` is what the renderer calls: `<cache_dir>/<hash>.flpk`, built on a `JobPool` on a miss.
"""
from std.collections import List
from std.ffi import external_call
from flint.assets.flpk_v3 import (
    CHUNK_COMP_LZ4,
    AssetUuid,
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    write_flpk_v3,
)
from flint.assets.model import NativeMesh
from flint.core.jobs import JobPool
from flint.core.lz4 import hash_list
from flint.core.serial import BlobReader, BlobWriter
from flint.render.sota_geometry_cluster_dag import (
    SOTA_DAG_GROUP_SIZE,
    ClusterDag,
    DagCluster,
    build_cluster_dag,
    build_cluster_dag_mt,
)

comptime SOTA_DAG_BLOB_TAG: UInt32 = 0x47414446      # "FDAG"
comptime SOTA_DAG_BLOB_VERSION: Int = 1
comptime SOTA_DAG_SUB_ID: UInt16 = 17                 # flpk sub-chunk id (16 = BVH8 collider)
comptime SOTA_DAG_ASSET_KIND: UInt8 = 40


def _append_raw(mut out: List[UInt8], addr: Int, n: Int):
    var o = len(out)
    out.resize(unsafe_uninit_length=o + n)
    if n > 0:
        _ = external_call["memcpy", Int](Int(out.unsafe_ptr()) + o, addr, n)


def mesh_content_hash(mesh: NativeMesh, group_size: Int = SOTA_DAG_GROUP_SIZE) -> UInt64:
    """Key of everything the DAG build reads (+ the group size and blob version)."""
    var b = List[UInt8](capacity=len(mesh.positions) * 8 + len(mesh.indices) * 8 + 16)
    _append_raw(b, Int(mesh.positions.unsafe_ptr()), len(mesh.positions) * 4)
    _append_raw(b, Int(mesh.normals.unsafe_ptr()), len(mesh.normals) * 4)
    _append_raw(b, Int(mesh.texcoords.unsafe_ptr()), len(mesh.texcoords) * 4)
    _append_raw(b, Int(mesh.indices.unsafe_ptr()), len(mesh.indices) * 8)
    b.append(UInt8(group_size & 255))
    b.append(UInt8(SOTA_DAG_BLOB_VERSION))
    return hash_list(b)


def encode_cluster_dag(dag: ClusterDag, key: UInt64) -> List[UInt8]:
    var w = BlobWriter()
    w.begin(SOTA_DAG_BLOB_TAG, SOTA_DAG_BLOB_VERSION)
    w.u64(key)
    w.integer(dag.levels)
    w.integer(dag.lod0_clusters)
    w.integer(dag.lod0_tris)
    w.integer(dag.groups_built)
    w.integer(dag.groups_failed)
    w.integer(len(dag.clusters))
    for c in dag.clusters:
        w.integer(c.level)
        w.vec3(c.self_center)
        w.f32(c.self_radius)
        w.f32(c.self_error)
        w.vec3(c.parent_center)
        w.f32(c.parent_radius)
        w.f32(c.parent_error)
        w.integer(len(c.tris))
        w.write_bytes_raw(Int(c.tris.unsafe_ptr()), len(c.tris) * 8)
    w.end()
    return w.data.copy()


def decode_cluster_dag(var data: List[UInt8], key: UInt64, mut dag: ClusterDag) -> Bool:
    """Fills `dag`; False for a truncated / foreign / stale (key mismatch) payload."""
    var r = BlobReader(data^)
    try:
        _ = r.enter(SOTA_DAG_BLOB_TAG, SOTA_DAG_BLOB_VERSION)
        if r.u64() != key:
            return False
        dag.levels = r.integer()
        dag.lod0_clusters = r.integer()
        dag.lod0_tris = r.integer()
        dag.groups_built = r.integer()
        dag.groups_failed = r.integer()
        var n = r.integer()
        if n < 0 or n > r.remaining() // 60:
            return False
        dag.clusters = List[DagCluster](capacity=n)
        for _ in range(n):
            var level = r.integer()
            var sc = r.vec3()
            var sr = r.f32()
            var se = r.f32()
            var pc = r.vec3()
            var pr = r.f32()
            var pe = r.f32()
            var nt = r.integer()
            if nt < 0 or nt * 8 > r.remaining():
                return False
            var tris = List[Int](length=nt, fill=0)
            if not r.read_bytes_raw(Int(tris.unsafe_ptr()), nt * 8):
                return False
            var c = DagCluster(tris^, level)
            c.self_center = sc
            c.self_radius = sr
            c.self_error = se
            c.parent_center = pc
            c.parent_radius = pr
            c.parent_error = pe
            dag.clusters.append(c^)
        r.leave()
    except:
        return False
    if r.failed:
        return False
    return True


def _key_hex(key: UInt64) -> String:
    var hexd = String("0123456789abcdef")
    var out = String()
    for i in range(16):
        var nib = Int((key >> UInt64((15 - i) * 4)) & 15)
        out += hexd[byte=nib]
    return out


def write_cluster_dag_flpk(path: String, dag: ClusterDag, key: UInt64) -> Bool:
    var chunks = List[FlpkBuildChunk]()
    var rel = "dag/" + _key_hex(key)
    chunks.append(FlpkBuildChunk(AssetUuid.from_path(rel), rel, SOTA_DAG_SUB_ID, SOTA_DAG_ASSET_KIND,
                                 encode_cluster_dag(dag, key), compression=CHUNK_COMP_LZ4))
    return write_flpk_v3(path, chunks, page_align=False)


def read_cluster_dag_flpk(path: String, key: UInt64, mut dag: ClusterDag) -> Bool:
    var arc = FlpkV3MappedArchive.open(path)
    if not arc.valid:
        return False
    var got = arc.read_by_path("dag/" + _key_hex(key), SOTA_DAG_SUB_ID, verify_crc=True)
    arc.close()
    if not got[0]:
        return False
    return decode_cluster_dag(got[1].copy(), key, dag)


struct DagCacheStats(Copyable, Movable):
    var hits: Int
    var misses: Int
    var writes: Int

    def __init__(out self):
        self.hits = 0
        self.misses = 0
        self.writes = 0


def build_cluster_dag_cached(
    mesh: NativeMesh, cache_dir: String, mut pool: JobPool, mut stats: DagCacheStats, group_size: Int = SOTA_DAG_GROUP_SIZE
) -> ClusterDag:
    """Loads `<cache_dir>/<content-hash>.flpk` when valid, else builds (on `pool` when it has workers) and
    writes it. An empty `cache_dir` disables the cache."""
    var key = mesh_content_hash(mesh, group_size)
    var path = cache_dir + "/" + _key_hex(key) + ".flpk"
    if cache_dir.byte_length() > 0:
        var cached = ClusterDag()
        if read_cluster_dag_flpk(path, key, cached):
            stats.hits += 1
            return cached^
    stats.misses += 1
    var dag: ClusterDag
    if pool.ready():
        dag = build_cluster_dag_mt(mesh, pool, group_size)
    else:
        dag = build_cluster_dag(mesh, group_size)
    if cache_dir.byte_length() > 0 and len(dag.clusters) > 0:
        if write_cluster_dag_flpk(path, dag, key):
            stats.writes += 1
    return dag^
