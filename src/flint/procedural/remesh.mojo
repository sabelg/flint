"""Isotropic and quad-dominant remeshing for Flint's geometry kit (`flint.procedural.remesh`).

`isotropic_remesh` is the incremental remesher of Botsch & Kobbelt 2004 ("A remeshing approach to
multiresolution modeling"), re-implemented on a corner table (triangles store their three corners;
half-edge `3f + k` runs from corner k to corner k+1):
  1. split edges longer than 4/3 L,
  2. collapse edges shorter than 4/5 L (link condition, no new edges longer than 4/3 L, no normal
     flips, boundary and sharp-feature curves only collapse along themselves),
  3. flip edges that bring both end valences closer to 6 (4 on boundaries),
  4. tangential area-weighted Laplacian relaxation,
  5. back-projection of every vertex onto the input surface (exact closest point).
Boundaries and sharp creases (dihedral above `feature_angle_deg`) are kept as constraints.
Attributes (normals, UV, colour, skin joints / weights) are transferred from the closest point of
the input with barycentric interpolation (skin weights re-ranked to the top 4 and renormalised).

`pair_quads` then turns the isotropic triangle mesh into a quad-dominant one: a greedy
maximum-quality matching of adjacent triangle pairs (quad quality = how close the four corners are
to right angles and how planar the pair is), as used by Blossom-Quad/Instant-Meshes style
pipelines for the final tri -> quad step.
"""
from std.math import abs, acos, cos, max, min, sqrt
from flint.core.math import Vec3
from flint.assets.model import NativeMesh
from flint.assets.mesh_kit import (
    EdgeMap, TriangleGrid, copy_mesh_material, pair_key, pos_at, set_pos, surface_area, unit, vpos,
    weld_positions, recompute_normals,
)


struct RemeshConfig(Copyable, Movable):
    var target_edge: Float32        # <= 0: derive from target_vertices
    var target_vertices: Int        # used when target_edge <= 0 (0: keep the input vertex count)
    var iterations: Int
    var feature_angle_deg: Float32  # dihedral above which an edge is a sharp feature (>= 180: none)
    var smoothing: Float32          # tangential relaxation step (0..1)
    var transfer_attributes: Bool

    def __init__(out self):
        self.target_edge = 0
        self.target_vertices = 0
        self.iterations = 6
        self.feature_angle_deg = 60
        self.smoothing = 0.5
        self.transfer_attributes = True


struct _CornerMesh(Movable):
    var p: List[Float32]      # 3 per vertex
    var fv: List[Int]         # 3 per face: corner vertex
    var tw: List[Int]         # 3 per face: twin half-edge or -1
    var feat: List[Bool]      # 3 per face: constrained half-edge (feature or boundary)
    var alive: List[Bool]     # per face
    var vc: List[Int]         # per vertex: one outgoing half-edge of a live face (-1: dead)
    var level: List[Int]      # per vertex: 0 free, 1 on a constraint curve, 2 corner

    def __init__(out self):
        self.p = List[Float32]()
        self.fv = List[Int]()
        self.tw = List[Int]()
        self.feat = List[Bool]()
        self.alive = List[Bool]()
        self.vc = List[Int]()
        self.level = List[Int]()

    @always_inline
    def nv(self) -> Int:
        return len(self.p) // 3

    @always_inline
    def nf(self) -> Int:
        return len(self.fv) // 3

    @always_inline
    def pos(self, v: Int) -> Vec3:
        return pos_at(self.p, v)

    @always_inline
    def org(self, h: Int) -> Int:
        return self.fv[h]

    @always_inline
    def dst(self, h: Int) -> Int:
        return self.fv[_nxt(h)]

    def face_normal(self, f: Int) -> Vec3:
        var a = self.pos(self.fv[f * 3])
        return (self.pos(self.fv[f * 3 + 1]) - a).cross(self.pos(self.fv[f * 3 + 2]) - a)

    def ring(self, v: Int) -> List[Int]:
        """Outgoing half-edges of v (both directions around a boundary fan)."""
        var out = List[Int]()
        var h0 = self.vc[v]
        if h0 < 0:
            return out^
        out.append(h0)
        var h = self.tw[_prv(h0)]
        var guard = 0
        while h >= 0 and h != h0 and guard < 4096:
            out.append(h)
            h = self.tw[_prv(h)]
            guard += 1
        if h < 0:
            # boundary: walk the other way
            var t = self.tw[h0]
            guard = 0
            while t >= 0 and guard < 4096:
                var g = _nxt(t)
                out.append(g)
                t = self.tw[g]
                guard += 1
        return out^

    def is_boundary_vertex(self, v: Int) -> Bool:
        var r = self.ring(v)
        for i in range(len(r)):
            if self.tw[r[i]] < 0 or self.tw[_prv(r[i])] < 0:
                return True
        return False

    def valence(self, v: Int) -> Int:
        var r = self.ring(v)
        var b = 0
        for i in range(len(r)):
            if self.tw[_prv(r[i])] < 0:
                b = 1
        return len(r) + b

    def link(mut self, a: Int, b: Int):
        if a >= 0:
            self.tw[a] = b
        if b >= 0:
            self.tw[b] = a

    def add_face(mut self, a: Int, b: Int, c: Int) -> Int:
        var f = self.nf()
        self.fv.append(a)
        self.fv.append(b)
        self.fv.append(c)
        for _ in range(3):
            self.tw.append(-1)
            self.feat.append(False)
        self.alive.append(True)
        return f


@always_inline
def _nxt(h: Int) -> Int:
    return h - h % 3 + (h % 3 + 1) % 3


@always_inline
def _prv(h: Int) -> Int:
    return h - h % 3 + (h % 3 + 2) % 3


def _build(mesh: NativeMesh, feature_cos: Float32) -> _CornerMesh:
    var m = _CornerMesh()
    var w = weld_positions(mesh, 0.0)
    var nr = w[1]
    m.p = List[Float32](length=nr * 3, fill=0.0)
    for v in range(mesh.vertex_count()):
        set_pos(m.p, w[0][v], vpos(mesh, v))
    for t in range(mesh.triangle_count()):
        var a = w[0][mesh.indices[t * 3]]
        var b = w[0][mesh.indices[t * 3 + 1]]
        var c = w[0][mesh.indices[t * 3 + 2]]
        if a == b or b == c or a == c:
            continue
        _ = m.add_face(a, b, c)
    # twins: pair a->b with b->a; anything else (non-manifold / inconsistent) stays a boundary
    var em = EdgeMap(m.nf() * 3)
    var bad = EdgeMap(16)
    for h in range(m.nf() * 3):
        var key = pair_key(m.org(h), m.dst(h))
        if em.find(key) >= 0:
            _ = bad.insert(key, 1)
        else:
            em.set(key, h)
    for h in range(m.nf() * 3):
        var a = m.org(h)
        var b = m.dst(h)
        if a < b:
            var k1 = pair_key(a, b)
            var k2 = pair_key(b, a)
            var o = em.find(k2)
            if o >= 0 and bad.find(k1) < 0 and bad.find(k2) < 0:
                m.link(h, o)
    # split non-manifold vertices (several fans) into one vertex per fan
    var nr0 = m.nv()
    m.vc = List[Int](length=nr0, fill=-1)
    var corner_of = List[List[Int]]()
    for _ in range(nr0):
        corner_of.append(List[Int]())
    for h in range(m.nf() * 3):
        corner_of[m.org(h)].append(h)
    var seen = List[Bool](length=m.nf() * 3, fill=False)
    for v in range(nr0):
        var first = True
        for ci in range(len(corner_of[v])):
            var h0 = corner_of[v][ci]
            if seen[h0]:
                continue
            # rewind to the fan start (boundary) then sweep it
            var start = h0
            var t = m.tw[start]
            var guard = 0
            while t >= 0 and _nxt(t) != h0 and guard < 4096:
                start = _nxt(t)
                t = m.tw[start]
                guard += 1
            var vid = v
            if not first:
                vid = m.nv()
                m.p.append(m.p[v * 3]); m.p.append(m.p[v * 3 + 1]); m.p.append(m.p[v * 3 + 2])
                m.vc.append(-1)
            first = False
            m.vc[vid] = start
            var h = start
            guard = 0
            while h >= 0 and not seen[h] and guard < 4096:
                seen[h] = True
                m.fv[h] = vid
                h = m.tw[_prv(h)]
                guard += 1
            # boundary fan: also sweep the other way from h0
            t = m.tw[h0]
            guard = 0
            while t >= 0 and guard < 4096:
                var g = _nxt(t)
                if seen[g]:
                    break
                seen[g] = True
                m.fv[g] = vid
                t = m.tw[g]
                guard += 1
    # constraints: boundary + sharp edges
    for h in range(m.nf() * 3):
        var o = m.tw[h]
        if o < 0:
            m.feat[h] = True
        elif feature_cos > -1.0:
            var n1 = unit(m.face_normal(h // 3))
            var n2 = unit(m.face_normal(o // 3))
            if n1.dot(n2) < feature_cos:
                m.feat[h] = True
    _update_levels(m)
    return m^


def _update_levels(mut m: _CornerMesh):
    m.level = List[Int](length=m.nv(), fill=0)
    var cnt = List[Int](length=m.nv(), fill=0)
    for h in range(m.nf() * 3):
        if not m.alive[h // 3] or not m.feat[h]:
            continue
        var o = m.tw[h]
        if o < 0 or m.org(h) < m.org(o):
            cnt[m.org(h)] += 1
            cnt[m.dst(h)] += 1
    for v in range(m.nv()):
        m.level[v] = 0 if cnt[v] == 0 else (1 if cnt[v] == 2 else 2)


def check_twins(m: _CornerMesh) -> Int:
    """Index of the first half-edge whose twin link is inconsistent, or -1 (debug aid)."""
    for h in range(m.nf() * 3):
        if not m.alive[h // 3]:
            continue
        var o = m.tw[h]
        if o < 0:
            continue
        if not m.alive[o // 3] or m.tw[o] != h or m.org(o) != m.dst(h) or m.dst(o) != m.org(h):
            return h
    return -1


def _fix_vc(mut m: _CornerMesh, v: Int, hint: Int):
    if hint >= 0 and m.alive[hint // 3] and m.org(hint) == v:
        m.vc[v] = hint


def _split(mut m: _CornerMesh, h: Int) -> Int:
    """Splits the edge of half-edge h at its midpoint; returns the new vertex."""
    var o = m.tw[h]
    var a = m.org(h)
    var b = m.dst(h)
    var mid = (m.pos(a) + m.pos(b)) * 0.5
    var mv = m.nv()
    m.p.append(mid.x); m.p.append(mid.y); m.p.append(mid.z)
    m.vc.append(-1)
    m.level.append(1 if m.feat[h] else 0)
    var is_feat = m.feat[h]
    # face f = (a, b, c) with h = a->b  ->  f = (a, m, c), f2 = (m, b, c)
    var hn = _nxt(h)
    var hp = _prv(h)
    var c = m.dst(hn)
    var tw_bc = m.tw[hn]
    var feat_bc = m.feat[hn]
    m.fv[hn] = mv                       # f: a -> m -> c
    var f2 = m.add_face(mv, b, c)
    m.link(f2 * 3 + 1, tw_bc)           # b->c keeps its twin
    m.feat[f2 * 3 + 1] = feat_bc
    m.link(hn, f2 * 3 + 2)              # m->c / c->m
    m.feat[hn] = False
    m.feat[f2 * 3 + 2] = False
    m.feat[f2 * 3] = is_feat            # m->b
    m.vc[b] = f2 * 3 + 1
    m.vc[mv] = hn
    m.vc[c] = f2 * 3 + 2
    _fix_vc(m, a, h)
    if o >= 0:
        # face g = (b, a, d) with o = b->a  ->  g = (b, m, d), g2 = (m, a, d)
        var on = _nxt(o)
        var d = m.dst(on)
        var tw_ad = m.tw[on]
        var feat_ad = m.feat[on]
        m.fv[on] = mv                   # g: b -> m -> d
        var g2 = m.add_face(mv, a, d)
        m.link(g2 * 3 + 1, tw_ad)
        m.feat[g2 * 3 + 1] = feat_ad
        m.link(on, g2 * 3 + 2)
        m.feat[on] = False
        m.feat[g2 * 3 + 2] = False
        m.feat[g2 * 3] = is_feat        # m->a
        # twins across the split edge: a->m (h) with m->a (g2*3); b->m (o) with m->b (f2*3)
        m.link(h, g2 * 3)
        m.link(o, f2 * 3)
        m.vc[a] = h
        m.vc[d] = g2 * 3 + 2
        _ = hp
    else:
        m.tw[h] = -1
        m.tw[f2 * 3] = -1
    return mv


def _flip(mut m: _CornerMesh, h: Int):
    """Flips interior edge h = a->b between f = (a, b, c) and g = (b, a, d) to c-d."""
    var o = m.tw[h]
    var f = h // 3
    var g = o // 3
    var a = m.org(h)
    var b = m.dst(h)
    var hn = _nxt(h)
    var hp = _prv(h)
    var on = _nxt(o)
    var op = _prv(o)
    var c = m.dst(hn)
    var d = m.dst(on)
    var t_bc = m.tw[hn]; var f_bc = m.feat[hn]
    var t_ca = m.tw[hp]; var f_ca = m.feat[hp]
    var t_ad = m.tw[on]; var f_ad = m.feat[on]
    var t_db = m.tw[op]; var f_db = m.feat[op]
    # f = (c, d, b): c->d, d->b, b->c ; g = (d, c, a): d->c, c->a, a->d
    m.fv[f * 3] = c; m.fv[f * 3 + 1] = d; m.fv[f * 3 + 2] = b
    m.fv[g * 3] = d; m.fv[g * 3 + 1] = c; m.fv[g * 3 + 2] = a
    m.link(f * 3, g * 3)
    m.feat[f * 3] = False; m.feat[g * 3] = False
    m.link(f * 3 + 1, t_db); m.feat[f * 3 + 1] = f_db
    m.link(f * 3 + 2, t_bc); m.feat[f * 3 + 2] = f_bc
    m.link(g * 3 + 1, t_ca); m.feat[g * 3 + 1] = f_ca
    m.link(g * 3 + 2, t_ad); m.feat[g * 3 + 2] = f_ad
    m.vc[a] = g * 3 + 2
    m.vc[b] = f * 3 + 2
    m.vc[c] = f * 3
    m.vc[d] = g * 3


def _collapse_ok(m: _CornerMesh, h: Int, keep_pos: Vec3, max_len2: Float32) -> Bool:
    var o = m.tw[h]
    var a = m.org(h)
    var b = m.dst(h)
    var la = m.level[a]
    var lb = m.level[b]
    if la > 0 and lb > 0 and not m.feat[h]:
        return False
    if lb == 2 and la == 2:
        return False
    if m.feat[h] and (la == 0 or lb == 0):
        return False
    var ra = m.ring(a)
    var rb = m.ring(b)
    # never collapse a valence-3 vertex fan away (tetrahedra / tiny components would vanish)
    var bnd_a = m.tw[_prv(ra[0])] < 0 if len(ra) > 0 else True
    var bnd_b = m.tw[_prv(rb[0])] < 0 if len(rb) > 0 else True
    for i in range(len(ra)):
        if m.tw[ra[i]] < 0 or m.tw[_prv(ra[i])] < 0:
            bnd_a = True
    for i in range(len(rb)):
        if m.tw[rb[i]] < 0 or m.tw[_prv(rb[i])] < 0:
            bnd_b = True
    if len(ra) <= (2 if bnd_a else 3) or len(rb) <= (2 if bnd_b else 3):
        return False
    if len(ra) + len(rb) <= 8 and not (bnd_a or bnd_b):
        return False
    # link condition: common neighbours must be the apexes of the faces on the edge
    var apex_c = m.dst(_nxt(h))
    var apex_d = m.dst(_nxt(o)) if o >= 0 else -1
    var common = 0
    var nbr_a = List[Int](capacity=len(ra) + 1)
    for i in range(len(ra)):
        nbr_a.append(m.dst(ra[i]))
        if m.tw[_prv(ra[i])] < 0:
            nbr_a.append(m.org(_prv(ra[i])))
    for i in range(len(rb)):
        var x = m.dst(rb[i])
        var extra = m.org(_prv(rb[i])) if m.tw[_prv(rb[i])] < 0 else -1
        for cand in [x, extra]:
            if cand < 0 or cand == a:
                continue
            var found = False
            for j in range(len(nbr_a)):
                if nbr_a[j] == cand:
                    found = True
            if found:
                if cand != apex_c and cand != apex_d:
                    return False
                common += 1
    var expect = 1 if o < 0 else 2
    if common > expect * 2:
        return False
    # a boundary-vertex pair joined by an interior edge would pinch the surface
    if o >= 0 and m.is_boundary_vertex(a) and m.is_boundary_vertex(b):
        return False
    # no long edges and no flipped / degenerate faces around a and b
    for side in range(2):
        var rr = ra.copy() if side == 0 else rb.copy()
        for i in range(len(rr)):
            var e = rr[i]
            var f = e // 3
            if f == h // 3 or (o >= 0 and f == o // 3):
                continue
            var x = m.dst(e)
            var y = m.dst(_nxt(e))
            if (m.pos(x) - keep_pos).length_squared() > max_len2:
                return False
            var n_old = m.face_normal(f)
            var n_new = (m.pos(x) - keep_pos).cross(m.pos(y) - keep_pos)
            var lo = n_old.length()
            var ln = n_new.length()
            if ln <= lo * 1e-3:
                return False
            if n_old.dot(n_new) < 0.2 * lo * ln:
                return False
    return True


def _collapse(mut m: _CornerMesh, h: Int, keep_pos: Vec3):
    """Collapses b into a for h = a->b (a moves to keep_pos)."""
    var o = m.tw[h]
    var a = m.org(h)
    var b = m.dst(h)
    var rb = m.ring(b)
    for i in range(len(rb)):
        m.fv[rb[i]] = a
    set_pos(m.p, a, keep_pos)
    m.level[a] = max(m.level[a], m.level[b])
    # remove f = (a, b, c): twins of b->c and c->a become twins of each other
    for side in range(2):
        var e = h if side == 0 else o
        if e < 0:
            continue
        var en = _nxt(e)
        var ep = _prv(e)
        var t1 = m.tw[en]
        var t2 = m.tw[ep]
        var fe = m.feat[en] or m.feat[ep]
        m.link(t1, t2)
        if t1 >= 0:
            m.feat[t1] = fe or m.feat[t1]
            m.vc[m.org(t1)] = t1
        if t2 >= 0:
            m.feat[t2] = fe or m.feat[t2]
            m.vc[m.org(t2)] = t2
        m.alive[e // 3] = False
        m.tw[e] = -1; m.tw[en] = -1; m.tw[ep] = -1
    m.vc[b] = -1
    # pick a live outgoing half-edge for a
    for i in range(len(rb)):
        if m.alive[rb[i] // 3]:
            m.vc[a] = rb[i]
            return
    var ra = m.ring(a)
    _ = ra


def _repair_vc(mut m: _CornerMesh):
    for v in range(m.nv()):
        m.vc[v] = -1
    for h in range(m.nf() * 3):
        if not m.alive[h // 3]:
            continue
        var v = m.org(h)
        # prefer the half-edge whose previous one is a boundary (fan start)
        if m.vc[v] < 0 or m.tw[_prv(h)] < 0:
            m.vc[v] = h


def isotropic_remesh(mesh: NativeMesh, cfg: RemeshConfig) -> NativeMesh:
    var feature_cos: Float32 = -2.0
    if cfg.feature_angle_deg < 180:
        feature_cos = Float32(cos_deg(cfg.feature_angle_deg))
    var m = _build(mesh, feature_cos)
    var grid = TriangleGrid(mesh)
    var L = cfg.target_edge
    if L <= 0:
        var tv = cfg.target_vertices if cfg.target_vertices > 0 else m.nv()
        var area = surface_area(mesh)
        L = Float32(sqrt(4.0 * area / (sqrt(3.0) * Float64(2 * max(4, tv)))))
    var hi2 = (L * 4.0 / 3.0) * (L * 4.0 / 3.0)
    var lo2 = (L * 4.0 / 5.0) * (L * 4.0 / 5.0)
    for _ in range(cfg.iterations):
        # 1. split long edges: each sweep splits the edges that exist when it starts, longest
        #    first (Rivara-style ordering; letting a sweep chase the edges it creates never ends)
        for _sweep in range(4):
            var cand = List[Int]()
            var clen = List[Float32]()
            for e in range(m.nf() * 3):
                if m.alive[e // 3]:
                    var o = m.tw[e]
                    if o < 0 or e < o:
                        var l2 = (m.pos(m.org(e)) - m.pos(m.dst(e))).length_squared()
                        if l2 > hi2:
                            cand.append(e)
                            clen.append(-l2)
            if len(cand) == 0:
                break
            var order = List[Int](capacity=len(cand))
            for i in range(len(cand)):
                order.append(i)
            _sort_desc(order, clen)
            var ca = List[Int](capacity=len(cand))
            var cb = List[Int](capacity=len(cand))
            for i in range(len(cand)):
                ca.append(m.org(cand[i]))
                cb.append(m.dst(cand[i]))
            for oi in range(len(order)):
                var i = order[oi]
                var e = cand[i]
                if m.alive[e // 3] and m.org(e) == ca[i] and m.dst(e) == cb[i]:
                    _ = _split(m, e)
        _repair_vc(m)
        # 2. collapse short edges
        var h = 0
        while h < m.nf() * 3:
            if m.alive[h // 3]:
                var o = m.tw[h]
                if o < 0 or h < o:
                    var a = m.org(h)
                    var b = m.dst(h)
                    if (m.pos(a) - m.pos(b)).length_squared() < lo2 and m.vc[a] >= 0 and m.vc[b] >= 0:
                        var la = m.level[a]
                        var lb = m.level[b]
                        var hh = h
                        if lb > la:
                            hh = o if o >= 0 else -1  # keep the more constrained vertex: collapse a into b
                        if hh >= 0:
                            var ka = m.org(hh)
                            var kb = m.dst(hh)
                            var keep = m.pos(ka) if m.level[ka] > m.level[kb] else (m.pos(ka) + m.pos(kb)) * 0.5
                            if _collapse_ok(m, hh, keep, hi2):
                                _collapse(m, hh, keep)
            h += 1
        _repair_vc(m)
        # 3. valence-improving flips
        var val = List[Int](length=m.nv(), fill=0)
        var bnd = List[Bool](length=m.nv(), fill=False)
        for e in range(m.nf() * 3):
            if m.alive[e // 3]:
                val[m.org(e)] += 1
                if m.tw[e] < 0:
                    val[m.dst(e)] += 1
                    bnd[m.org(e)] = True
                    bnd[m.dst(e)] = True
        h = 0
        while h < m.nf() * 3:
            var o = m.tw[h]
            if m.alive[h // 3] and o >= 0 and h < o and not m.feat[h]:
                var a = m.org(h)
                var b = m.dst(h)
                var c = m.dst(_nxt(h))
                var d = m.dst(_nxt(o))
                if c != d and not _edge_exists(m, c, d):
                    var ta = 4 if bnd[a] else 6
                    var tb = 4 if bnd[b] else 6
                    var tc = 4 if bnd[c] else 6
                    var td = 4 if bnd[d] else 6
                    var before = abs(val[a] - ta) + abs(val[b] - tb) + abs(val[c] - tc) + abs(val[d] - td)
                    var after = abs(val[a] - 1 - ta) + abs(val[b] - 1 - tb) + abs(val[c] + 1 - tc) + abs(val[d] + 1 - td)
                    if after < before and val[a] > 3 and val[b] > 3:
                        # geometric guard: the new faces must keep the orientation
                        var pa = m.pos(a); var pb = m.pos(b); var pc = m.pos(c); var pd = m.pos(d)
                        var n1 = (pd - pc).cross(pb - pc)
                        var n2 = (pc - pd).cross(pa - pd)
                        var nref = m.face_normal(h // 3) + m.face_normal(o // 3)
                        if n1.dot(nref) > 0 and n2.dot(nref) > 0 and n1.dot(n2) > 0.2 * n1.length() * n2.length():
                            _flip(m, h)
                            val[a] -= 1; val[b] -= 1; val[c] += 1; val[d] += 1
            h += 1
        _repair_vc(m)
        # 4. tangential relaxation + 5. projection
        var newp = m.p.copy()
        var nsum = List[Vec3](length=m.nv(), fill=Vec3(0, 0, 0))
        var csum = List[Vec3](length=m.nv(), fill=Vec3(0, 0, 0))
        var wsum = List[Float32](length=m.nv(), fill=0.0)
        for f in range(m.nf()):
            if not m.alive[f]:
                continue
            var fn_ = m.face_normal(f)
            var ar = fn_.length()
            var cen = (m.pos(m.fv[f * 3]) + m.pos(m.fv[f * 3 + 1]) + m.pos(m.fv[f * 3 + 2])) * (1.0 / 3.0)
            for k in range(3):
                var v = m.fv[f * 3 + k]
                nsum[v] = nsum[v] + fn_
                csum[v] = csum[v] + cen * ar
                wsum[v] += ar
        for v in range(m.nv()):
            if m.vc[v] < 0 or wsum[v] <= 0:
                continue
            var p = m.pos(v)
            if m.level[v] == 0:
                var n = unit(nsum[v])
                var q = csum[v] * (1.0 / wsum[v])
                var d = q - p
                d = d - n * d.dot(n)
                p = p + d * cfg.smoothing
            var hit = grid.closest(p)
            set_pos(newp, v, hit.point)
        m.p = newp^
    return _emit(m, mesh, grid, cfg)


def cos_deg(d: Float32) -> Float64:
    return cos(Float64(d) * 0.017453292519943295)


def _edge_exists(m: _CornerMesh, a: Int, b: Int) -> Bool:
    var r = m.ring(a)
    for i in range(len(r)):
        if m.dst(r[i]) == b:
            return True
        if m.tw[_prv(r[i])] < 0 and m.org(_prv(r[i])) == b:
            return True
    return False


def _emit(m: _CornerMesh, src: NativeMesh, mut grid: TriangleGrid, cfg: RemeshConfig) -> NativeMesh:
    var out = NativeMesh()
    copy_mesh_material(src, out)
    var remap = List[Int](length=m.nv(), fill=-1)
    for f in range(m.nf()):
        if not m.alive[f]:
            continue
        for k in range(3):
            var v = m.fv[f * 3 + k]
            if remap[v] < 0:
                remap[v] = out.vertex_count()
                var p = m.pos(v)
                out.positions.append(p.x); out.positions.append(p.y); out.positions.append(p.z)
            out.indices.append(remap[v])
    var nv = out.vertex_count()
    if cfg.transfer_attributes:
        var snv = src.vertex_count()
        var has_uv = len(src.texcoords) >= snv * 2
        var has_c = len(src.colors) >= snv * 4
        var has_s = len(src.joints) >= snv * 4 and len(src.weights) >= snv * 4
        var has_ao = len(src.ao) >= snv
        for v in range(nv):
            var p = vpos(out, v)
            var hit = grid.closest(p)
            var t = max(0, hit.tri)
            var i0 = src.indices[t * 3]
            var i1 = src.indices[t * 3 + 1]
            var i2 = src.indices[t * 3 + 2]
            var bc = _barycentric(hit.point, vpos(src, i0), vpos(src, i1), vpos(src, i2))
            if has_uv:
                for q in range(2):
                    out.texcoords.append(src.texcoords[i0 * 2 + q] * bc.x + src.texcoords[i1 * 2 + q] * bc.y + src.texcoords[i2 * 2 + q] * bc.z)
            if has_c:
                for q in range(4):
                    var c = Float32(Int(src.colors[i0 * 4 + q])) * bc.x + Float32(Int(src.colors[i1 * 4 + q])) * bc.y + Float32(Int(src.colors[i2 * 4 + q])) * bc.z
                    out.colors.append(UInt8(max(0, min(255, Int(c + 0.5)))))
            if has_ao:
                var c = Float32(Int(src.ao[i0])) * bc.x + Float32(Int(src.ao[i1])) * bc.y + Float32(Int(src.ao[i2])) * bc.z
                out.ao.append(UInt8(max(0, min(255, Int(c + 0.5)))))
            if has_s:
                _blend_skin(src, i0, i1, i2, bc, out)
    recompute_normals(out, True)
    return out^


def _barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    var v0 = b - a
    var v1 = c - a
    var v2 = p - a
    var d00 = v0.dot(v0)
    var d01 = v0.dot(v1)
    var d11 = v1.dot(v1)
    var d20 = v2.dot(v0)
    var d21 = v2.dot(v1)
    var den = d00 * d11 - d01 * d01
    if abs(den) < 1e-30:
        return Vec3(1, 0, 0)
    var v = (d11 * d20 - d01 * d21) / den
    var w = (d00 * d21 - d01 * d20) / den
    v = max(Float32(0), min(Float32(1), v))
    w = max(Float32(0), min(Float32(1) - v, w))
    return Vec3(1.0 - v - w, v, w)


def _blend_skin(src: NativeMesh, i0: Int, i1: Int, i2: Int, bc: Vec3, mut out: NativeMesh):
    var js = List[Int]()
    var ws = List[Float32]()
    for corner in range(3):
        var vi = i0 if corner == 0 else (i1 if corner == 1 else i2)
        var b = bc.x if corner == 0 else (bc.y if corner == 1 else bc.z)
        for k in range(4):
            var j = Int(src.joints[vi * 4 + k])
            var w = src.weights[vi * 4 + k] * b
            if w <= 0:
                continue
            var found = -1
            for q in range(len(js)):
                if js[q] == j:
                    found = q
            if found >= 0:
                ws[found] += w
            else:
                js.append(j)
                ws.append(w)
    var total: Float32 = 0
    var pick_j = List[Int](length=4, fill=0)
    var pick_w = List[Float32](length=4, fill=0.0)
    for k in range(4):
        var best = -1
        for q in range(len(ws)):
            if ws[q] > 0 and (best < 0 or ws[q] > ws[best]):
                best = q
        if best >= 0:
            pick_j[k] = js[best]
            pick_w[k] = ws[best]
            total += ws[best]
            ws[best] = -1
    for k in range(4):
        out.joints.append(UInt16(pick_j[k]))
        out.weights.append(pick_w[k] / total if total > 0 else (Float32(1.0) if k == 0 else Float32(0.0)))


# ------------------------------------------------------------------------------- quad pairing

@fieldwise_init
struct QuadDominantMesh(Copyable, Movable):
    var positions: List[Float32]
    var quads: List[Int]      # 4 per quad, CCW
    var triangles: List[Int]  # 3 per leftover triangle

    def quad_ratio(self) -> Float32:
        """Fraction of the surface's faces (counted as triangles) covered by quads."""
        var q = len(self.quads) // 4
        var t = len(self.triangles) // 3
        return Float32(2 * q) / max(Float32(1), Float32(2 * q + t))


def _quad_quality(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Float32:
    """1 for a planar square, falling with corner deviation from 90 degrees and with folding."""
    var pts = [a, b, c, d]
    var worst: Float32 = 1
    var nref = unit((c - a).cross(d - b))
    for k in range(4):
        var p = pts[k]
        var e1 = unit(pts[(k + 1) % 4] - p)
        var e0 = unit(pts[(k + 3) % 4] - p)
        var cosv = abs(e1.dot(e0))
        var cn = unit(e1.cross(e0))
        if cn.dot(nref) <= 0.3:
            return 0
        worst = min(worst, 1.0 - cosv)
    return worst


def pair_quads(mesh: NativeMesh, min_quality: Float32 = 0.5) -> QuadDominantMesh:
    """Greedy best-first matching of adjacent triangle pairs into quads (position-welded)."""
    var w = weld_positions(mesh, 0.0)
    var nt = mesh.triangle_count()
    var fr = List[Int](length=nt * 3, fill=0)
    for i in range(nt * 3):
        fr[i] = w[0][mesh.indices[i]]
    var pos = List[Float32](length=w[1] * 3, fill=0.0)
    for v in range(mesh.vertex_count()):
        set_pos(pos, w[0][v], vpos(mesh, v))
    var em = EdgeMap(nt * 3)
    for h in range(nt * 3):
        em.set(pair_key(fr[h], fr[_nxt(h)]), h)
    var cand_q = List[Float32]()
    var cand_h = List[Int]()
    for h in range(nt * 3):
        var o = em.find(pair_key(fr[_nxt(h)], fr[h]))
        if o < 0 or o < h:
            continue
        var a = fr[h]
        var c = fr[_nxt(_nxt(h))]
        var b = fr[_nxt(h)]
        var d = fr[_nxt(_nxt(o))]
        # quad a, d, b, c  (CCW: a -> d -> b -> c)
        var q = _quad_quality(pos_at(pos, a), pos_at(pos, d), pos_at(pos, b), pos_at(pos, c))
        if q >= min_quality:
            cand_q.append(-q)
            cand_h.append(h)
    # sort candidates by quality (descending) with a merge sort on keys
    var order = List[Int](capacity=len(cand_q))
    for i in range(len(cand_q)):
        order.append(i)
    _sort_desc(order, cand_q)
    var used = List[Bool](length=nt, fill=False)
    var out = QuadDominantMesh(pos^, List[Int](), List[Int]())
    for oi in range(len(order)):
        var h = cand_h[order[oi]]
        var o = em.find(pair_key(fr[_nxt(h)], fr[h]))
        var f = h // 3
        var g = o // 3
        if used[f] or used[g]:
            continue
        used[f] = True
        used[g] = True
        out.quads.append(fr[h]); out.quads.append(fr[_nxt(_nxt(o))]); out.quads.append(fr[_nxt(h)]); out.quads.append(fr[_nxt(_nxt(h))])
    for f in range(nt):
        if not used[f]:
            out.triangles.append(fr[f * 3]); out.triangles.append(fr[f * 3 + 1]); out.triangles.append(fr[f * 3 + 2])
    return out^


def _sort_desc(mut order: List[Int], key: List[Float32]):
    var n = len(order)
    var tmp = List[Int](length=n, fill=0)
    var width = 1
    while width < n:
        var i = 0
        while i < n:
            var mid = min(i + width, n)
            var hi = min(i + 2 * width, n)
            var a = i
            var b = mid
            var k = i
            while a < mid and b < hi:
                if key[order[a]] <= key[order[b]]:
                    tmp[k] = order[a]; a += 1
                else:
                    tmp[k] = order[b]; b += 1
                k += 1
            while a < mid:
                tmp[k] = order[a]; a += 1; k += 1
            while b < hi:
                tmp[k] = order[b]; b += 1; k += 1
            i += 2 * width
        for j in range(n):
            order[j] = tmp[j]
        width *= 2


# ------------------------------------------------------------------------------- quality metrics

@fieldwise_init
struct TriQuality(Copyable, ImplicitlyCopyable, Movable, Writable):
    var mean_edge: Float32
    var edge_cv: Float32          # coefficient of variation of edge lengths
    var min_angle_deg: Float32
    var mean_min_angle_deg: Float32
    var valence6_ratio: Float32   # interior vertices with valence 6

    def write_to[W: Writer](self, mut writer: W):
        writer.write("edge=", self.mean_edge, " edge_cv=", self.edge_cv, " min_angle=", self.min_angle_deg,
                     " mean_min_angle=", self.mean_min_angle_deg, " val6=", self.valence6_ratio)


def triangle_quality(mesh: NativeMesh) -> TriQuality:
    var w = weld_positions(mesh, 0.0)
    var nt = mesh.triangle_count()
    var s: Float64 = 0
    var s2: Float64 = 0
    var ne = 0
    var min_ang: Float64 = 180
    var sum_min: Float64 = 0
    var val = List[Int](length=w[1], fill=0)
    var em = EdgeMap(nt * 3)
    var cnt = List[Int]()
    for t in range(nt):
        var p = [vpos(mesh, mesh.indices[t * 3]), vpos(mesh, mesh.indices[t * 3 + 1]), vpos(mesh, mesh.indices[t * 3 + 2])]
        var tmin: Float64 = 180
        for k in range(3):
            var l = Float64((p[(k + 1) % 3] - p[k]).length())
            s += l
            s2 += l * l
            ne += 1
            var e1 = unit(p[(k + 1) % 3] - p[k])
            var e2 = unit(p[(k + 2) % 3] - p[k])
            var ang = acos(Float64(max(Float32(-1), min(Float32(1), e1.dot(e2))))) * 57.29577951308232
            tmin = min(tmin, ang)
            var a = w[0][mesh.indices[t * 3 + k]]
            var b = w[0][mesh.indices[t * 3 + (k + 1) % 3]]
            var slot = em.insert(pair_key(min(a, b), max(a, b)), len(cnt))
            if slot == len(cnt):
                cnt.append(0)
                val[a] += 1
                val[b] += 1
            cnt[slot] += 1
        min_ang = min(min_ang, tmin)
        sum_min += tmin
    # boundary vertices are excluded from the valence statistic
    var on_b = List[Bool](length=w[1], fill=False)
    for t in range(nt):
        for k in range(3):
            var a = w[0][mesh.indices[t * 3 + k]]
            var b = w[0][mesh.indices[t * 3 + (k + 1) % 3]]
            if cnt[em.find(pair_key(min(a, b), max(a, b)))] == 1:
                on_b[a] = True
                on_b[b] = True
    var interior = 0
    var six = 0
    for v in range(w[1]):
        if val[v] > 0 and not on_b[v]:
            interior += 1
            if val[v] == 6:
                six += 1
    var mean = s / Float64(max(1, ne))
    var var_ = max(0.0, s2 / Float64(max(1, ne)) - mean * mean)
    return TriQuality(Float32(mean), Float32(sqrt(var_) / max(mean, 1e-30)), Float32(min_ang),
                      Float32(sum_min / Float64(max(1, nt))), Float32(six) / max(Float32(1), Float32(interior)))
