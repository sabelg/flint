"""AaaGeometryPool: freeing, replacing and reusing mesh room (streamed tiles, edited geometry)."""
from std.testing import assert_equal, assert_true, assert_false
from flint.core.math import Vec3
from flint.assets.model import NativeMesh
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.meshlet import AaaGeometryPool, AAA_MESHLET_WORDS, AAA_VERTEX_WORDS


def sphere(r: Float32, rings: Int) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(r, r, r), rings, rings * 2)
    return mb^.finish()


def check_mesh(pool: AaaGeometryPool, mesh_id: Int) raises:
    """Every meshlet of the mesh points at its own triangles, and those at its own vertices."""
    ref e = pool.meshes[mesh_id]
    var tris = 0
    for k in range(e.lod0_meshlet_count):
        var mw = (e.first_meshlet + k) * AAA_MESHLET_WORDS
        var first_tri = Int(pool.meshlet_words[mw + 8])
        var n = Int(pool.meshlet_words[mw + 9])
        assert_true(first_tri * 3 >= e.first_index)
        assert_true((first_tri + n) * 3 <= e.first_index + e.triangle_count * 3)
        assert_equal(pool.meshlets[e.first_meshlet + k].mesh_id, mesh_id)
        tris += n
        for t in range(first_tri, first_tri + n):
            for c in range(3):
                var v = Int(pool.index_words[t * 3 + c])
                assert_true(v >= e.first_vertex and v < e.first_vertex + e.vertex_count)
    assert_equal(tris, e.triangle_count)


def main() raises:
    var pool = AaaGeometryPool()
    var a = pool.append_mesh(sphere(1.0, 8), 0).mesh_id
    var b = pool.append_mesh(sphere(1.0, 16), 0).mesh_id
    var c = pool.append_mesh(sphere(1.0, 8), 0).mesh_id
    for m in [a, b, c]:
        check_mesh(pool, m)
    var words = len(pool.vertex_words)

    # in place: a smaller mesh fits in b's own room, so the pool does not grow
    var r1 = pool.replace_mesh(b, sphere(2.0, 8), 1)
    assert_equal(r1[0].mesh_id, b)
    assert_equal(len(pool.vertex_words), words)
    check_mesh(pool, b)
    check_mesh(pool, a)
    check_mesh(pool, c)

    # grows past its room: moves to the end, leaving a hole
    var r2 = pool.replace_mesh(a, sphere(1.0, 24), 0)
    assert_true(len(pool.vertex_words) > words)
    assert_equal(len(pool.holes), 1)
    check_mesh(pool, a)
    assert_equal(pool.slots[a].first_vertex, r2[1].first_vertex)

    # free + a new mesh reuses the freed room
    var freed = pool.free_mesh(c)
    assert_false(freed.empty())
    assert_equal(pool.meshes[c].meshlet_count, 0)
    for k in range(freed.meshlet_cap):
        assert_equal(Int(pool.meshlet_words[(freed.first_meshlet + k) * AAA_MESHLET_WORDS + 9]), 0)
    var before = len(pool.vertex_words)
    var d = pool.reserve_mesh_id()
    var r3 = pool.replace_mesh(d, sphere(0.5, 8), 0)
    assert_equal(len(pool.vertex_words), before)             # it went in a hole
    check_mesh(pool, d)
    _ = r3

    # many rounds of streaming do not grow the pool without bound
    var tile = pool.reserve_mesh_id()
    _ = pool.replace_mesh(tile, sphere(1.0, 12), 0)
    var high = len(pool.vertex_words)
    for i in range(40):
        _ = pool.replace_mesh(tile, sphere(1.0, 6 + (i % 7)), 0)
        check_mesh(pool, tile)
    assert_true(len(pool.vertex_words) <= high + 13 * 26 * AAA_VERTEX_WORDS * 4)
    print("geometry pool free/replace: OK")
