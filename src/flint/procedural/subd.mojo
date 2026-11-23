"""Pixar / OpenSubdiv Catmull-Clark Subdivision Surfaces with Semi-Sharp Creases (`subd.mojo`).

Implements:
  - `SubdCage`: N-gon / quad / triangle control cage with `(min_v, max_v) -> Float32` semi-sharp edge
    crease table (`s_e in [0..10]`), per-vertex corner sharpness (`s_v in [0..10]`), face-varying UVs,
    and submesh `face_group` tags.
  - `mirror_x_and_weld`: Symmetric `+X -> -X` mirroring and centerline seam welding (`x = 0`) for
    `G^2`-seamless automotive, aerospace, and coaster bodywork.
  - `creased_subdivide` & `subdivide_to_mesh`: Exact DeRose-Kass-Truong / OpenSubdiv semi-sharp
    edge blending (`w = clamp(s_e, 0, 1)`, Chaikin child decay `s'_e = max(0, s_e - 1)`), dart /
    crease / corner vertex rules, curvature-cavity + ground-proximity vertex AO baking (`0..255`),
    and hard-wedge normal splitting across post-subdivision sharp creases (`>= split_sharp_threshold`).
  - Ready-to-use AAA vehicle & coaster builders:
    1. `build_subd_supercar_body_and_glass()` (`~2,560` tris)
    2. `build_subd_spinner_hovercar_body_and_canopy()` (`~1,920` tris)
    3. `build_subd_traffic_vehicle(archetype)` (`0`=coupe, `1`=cyber-sedan, `2`=armored van, `~1,280–1,408` tris)
    4. `build_subd_coaster_car_shell(style)` (`0`=hyper tub, `1`=LSM nosecone, `2`=inverted ski-chair, `3`=PTC timber coach, `~1,280–1,408` tris)
"""
from std.math import isfinite, max, min, sqrt
from flint.core.math import Vec2, Vec3
from flint.assets.model import NativeMesh


@always_inline
def _next_pow2_subd(n: Int) -> Int:
    var p = 16
    while p < n:
        p = p << 1
    return p


@always_inline
def _hash_edge_pair(a: Int, b: Int) -> UInt64:
    var u0 = min(a, b)
    var u1 = max(a, b)
    var k = (UInt64(u0) << 32) | UInt64(u1)
    k = (k ^ (k >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    k = (k ^ (k >> 27)) * UInt64(0x94D049BB133111EB)
    return k ^ (k >> 31)


@always_inline
def _clamp01(x: Float32) -> Float32:
    if not isfinite(x):
        return Float32(0.0)
    return max(Float32(0.0), min(Float32(1.0), x))


@always_inline
def _lerp_v3(a: Vec3, b: Vec3, t: Float32) -> Vec3:
    return a * (Float32(1.0) - t) + b * t


@always_inline
def _lerp_v2(a: Vec2, b: Vec2, t: Float32) -> Vec2:
    return a * (Float32(1.0) - t) + b * t


struct SubdCage(Copyable, Movable):
    """Control cage for Pixar/OpenSubdiv Catmull-Clark subdivision with semi-sharp creases."""
    var positions: List[Vec3]
    var uvs: List[Vec2]
    var face_offsets: List[Int]
    var face_verts: List[Int]
    var face_uvs: List[Int]
    var face_group: List[Int]
    var vert_sharpness: List[Float32]
    var edge_v0: List[Int]
    var edge_v1: List[Int]
    var edge_sharpness: List[Float32]
    var _edge_hash_keys: List[UInt64]
    var _edge_hash_idx: List[Int]
    var _edge_hash_mask: Int

    def __init__(out self):
        self.positions = List[Vec3]()
        self.uvs = List[Vec2]()
        self.face_offsets = List[Int]()
        self.face_offsets.append(0)
        self.face_verts = List[Int]()
        self.face_uvs = List[Int]()
        self.face_group = List[Int]()
        self.vert_sharpness = List[Float32]()
        self.edge_v0 = List[Int]()
        self.edge_v1 = List[Int]()
        self.edge_sharpness = List[Float32]()
        self._edge_hash_keys = List[UInt64](length=64, fill=UInt64(0xFFFFFFFFFFFFFFFF))
        self._edge_hash_idx = List[Int](length=64, fill=-1)
        self._edge_hash_mask = 63

    def num_verts(self) -> Int:
        return len(self.positions)

    def num_faces(self) -> Int:
        return len(self.face_group)

    def add_vert(mut self, pos: Vec3, sharpness: Float32 = 0.0) -> Int:
        var idx = len(self.positions)
        self.positions.append(pos)
        var s = max(Float32(0.0), min(Float32(10.0), sharpness)) if isfinite(sharpness) else Float32(0.0)
        self.vert_sharpness.append(s)
        return idx

    def add_uv(mut self, uv: Vec2) -> Int:
        var idx = len(self.uvs)
        self.uvs.append(uv)
        return idx

    @always_inline
    def _auto_uv_for_pos(self, p: Vec3) -> Vec2:
        return Vec2(p.x * Float32(0.25) + Float32(0.5), p.z * Float32(0.20) + p.y * Float32(0.15) + Float32(0.5))

    def add_quad(mut self, v0: Int, v1: Int, v2: Int, v3: Int, group: Int = 0):
        var u0 = self.add_uv(self._auto_uv_for_pos(self.positions[v0]))
        var u1 = self.add_uv(self._auto_uv_for_pos(self.positions[v1]))
        var u2 = self.add_uv(self._auto_uv_for_pos(self.positions[v2]))
        var u3 = self.add_uv(self._auto_uv_for_pos(self.positions[v3]))
        self.add_quad_uv(v0, v1, v2, v3, u0, u1, u2, u3, group)

    def add_quad_uv(
        mut self,
        v0: Int,
        v1: Int,
        v2: Int,
        v3: Int,
        u0: Int,
        u1: Int,
        u2: Int,
        u3: Int,
        group: Int = 0,
    ):
        self.face_verts.append(v0)
        self.face_verts.append(v1)
        self.face_verts.append(v2)
        self.face_verts.append(v3)
        self.face_uvs.append(u0)
        self.face_uvs.append(u1)
        self.face_uvs.append(u2)
        self.face_uvs.append(u3)
        self.face_offsets.append(len(self.face_verts))
        self.face_group.append(group)

    def add_tri(mut self, v0: Int, v1: Int, v2: Int, group: Int = 0):
        var u0 = self.add_uv(self._auto_uv_for_pos(self.positions[v0]))
        var u1 = self.add_uv(self._auto_uv_for_pos(self.positions[v1]))
        var u2 = self.add_uv(self._auto_uv_for_pos(self.positions[v2]))
        self.face_verts.append(v0)
        self.face_verts.append(v1)
        self.face_verts.append(v2)
        self.face_uvs.append(u0)
        self.face_uvs.append(u1)
        self.face_uvs.append(u2)
        self.face_offsets.append(len(self.face_verts))
        self.face_group.append(group)

    def add_ngon(mut self, verts: List[Int], group: Int = 0):
        if len(verts) < 3:
            return
        for i in range(len(verts)):
            var vi = verts[i]
            self.face_verts.append(vi)
            self.face_uvs.append(self.add_uv(self._auto_uv_for_pos(self.positions[vi])))
        self.face_offsets.append(len(self.face_verts))
        self.face_group.append(group)

    def _rehash_edges(mut self, new_cap: Int):
        var cap = _next_pow2_subd(max(64, new_cap))
        self._edge_hash_keys = List[UInt64](length=cap, fill=UInt64(0xFFFFFFFFFFFFFFFF))
        self._edge_hash_idx = List[Int](length=cap, fill=-1)
        self._edge_hash_mask = cap - 1
        for i in range(len(self.edge_v0)):
            var u0 = self.edge_v0[i]
            var u1 = self.edge_v1[i]
            var key = (UInt64(u0) << 32) | UInt64(u1)
            var slot = Int(_hash_edge_pair(u0, u1)) & self._edge_hash_mask
            while self._edge_hash_keys[slot] != UInt64(0xFFFFFFFFFFFFFFFF):
                slot = (slot + 1) & self._edge_hash_mask
            self._edge_hash_keys[slot] = key
            self._edge_hash_idx[slot] = i

    def set_edge_sharpness(mut self, a: Int, b: Int, sharpness: Float32):
        if a == b or a < 0 or b < 0:
            return
        var s = max(Float32(0.0), min(Float32(10.0), sharpness)) if isfinite(sharpness) else Float32(0.0)
        if (len(self.edge_v0) + 1) * 2 >= len(self._edge_hash_keys):
            self._rehash_edges(len(self._edge_hash_keys) * 2)

        var u0 = min(a, b)
        var u1 = max(a, b)
        var key = (UInt64(u0) << 32) | UInt64(u1)
        var slot = Int(_hash_edge_pair(u0, u1)) & self._edge_hash_mask
        while True:
            var mk = self._edge_hash_keys[slot]
            if mk == key:
                var idx = self._edge_hash_idx[slot]
                self.edge_sharpness[idx] = s
                return
            if mk == UInt64(0xFFFFFFFFFFFFFFFF):
                if s <= Float32(0.0):
                    return
                var idx = len(self.edge_v0)
                self.edge_v0.append(u0)
                self.edge_v1.append(u1)
                self.edge_sharpness.append(s)
                self._edge_hash_keys[slot] = key
                self._edge_hash_idx[slot] = idx
                return
            slot = (slot + 1) & self._edge_hash_mask

    def get_edge_sharpness(self, a: Int, b: Int) -> Float32:
        if a == b or len(self.edge_v0) == 0:
            return Float32(0.0)
        var u0 = min(a, b)
        var u1 = max(a, b)
        var key = (UInt64(u0) << 32) | UInt64(u1)
        var slot = Int(_hash_edge_pair(u0, u1)) & self._edge_hash_mask
        var probes = 0
        while probes <= self._edge_hash_mask:
            var mk = self._edge_hash_keys[slot]
            if mk == key:
                return self.edge_sharpness[self._edge_hash_idx[slot]]
            if mk == UInt64(0xFFFFFFFFFFFFFFFF):
                return Float32(0.0)
            slot = (slot + 1) & self._edge_hash_mask
            probes += 1
        return Float32(0.0)

    def set_crease_chain(mut self, chain: List[Int], sharpness: Float32, closed: Bool = False):
        var n = len(chain)
        if n < 2:
            return
        for i in range(n - 1):
            self.set_edge_sharpness(chain[i], chain[i + 1], sharpness)
        if closed and n >= 3:
            self.set_edge_sharpness(chain[n - 1], chain[0], sharpness)

    def mirror_x_and_weld(mut self, eps: Float32 = 1e-4):
        """Mirrors a `+X` symmetric half-cage across `X = 0` with reversed face winding and welds
        all centerline (`|x| <= eps`) and colocated vertices for `G^2`-continuous symmetry."""
        var orig_v = len(self.positions)
        if orig_v == 0:
            return
        var eps2 = eps * eps

        # 1. Snap centerline vertices (`|x| <= eps -> x = 0`) and weld duplicate base vertices
        var welded_pos = List[Vec3](capacity=orig_v * 2)
        var welded_sharp = List[Float32](capacity=orig_v * 2)
        var base_remap = List[Int](length=orig_v, fill=0)

        for vi in range(orig_v):
            var p = self.positions[vi]
            if abs(p.x) <= eps:
                p.x = Float32(0.0)
            var found = -1
            for w_i in range(len(welded_pos)):
                if (welded_pos[w_i] - p).length_squared() <= eps2:
                    found = w_i
                    break
            if found >= 0:
                base_remap[vi] = found
                if self.vert_sharpness[vi] > welded_sharp[found]:
                    welded_sharp[found] = self.vert_sharpness[vi]
            else:
                base_remap[vi] = len(welded_pos)
                welded_pos.append(p)
                welded_sharp.append(self.vert_sharpness[vi])

        # 2. Create mirrored `-X` counterparts for off-centerline vertices
        var welded_base_count = len(welded_pos)
        var mirror_of = List[Int](length=welded_base_count, fill=0)
        for wi in range(welded_base_count):
            var p = welded_pos[wi]
            if abs(p.x) <= eps:
                mirror_of[wi] = wi
            else:
                var mp = Vec3(-p.x, p.y, p.z)
                var found = -1
                for k in range(len(welded_pos)):
                    if (welded_pos[k] - mp).length_squared() <= eps2:
                        found = k
                        break
                if found >= 0:
                    mirror_of[wi] = found
                else:
                    var midx = len(welded_pos)
                    welded_pos.append(mp)
                    welded_sharp.append(welded_sharp[wi])
                    mirror_of[wi] = midx

        # 3. Rebuild face buffers (remapped original faces + reversed-winding mirrored faces)
        var orig_faces = len(self.face_group)
        var new_offsets = List[Int](capacity=orig_faces * 2 + 1)
        new_offsets.append(0)
        var new_fverts = List[Int](capacity=len(self.face_verts) * 2)
        var new_fuvs = List[Int](capacity=len(self.face_uvs) * 2)
        var new_fgroup = List[Int](capacity=orig_faces * 2)

        for fi in range(orig_faces):
            var s0 = self.face_offsets[fi]
            var s1 = self.face_offsets[fi + 1]
            var k_cnt = s1 - s0
            if k_cnt < 3:
                continue
            for c in range(s0, s1):
                new_fverts.append(base_remap[self.face_verts[c]])
                new_fuvs.append(self.face_uvs[c])
            new_offsets.append(len(new_fverts))
            new_fgroup.append(self.face_group[fi])

        for fi in range(orig_faces):
            var s0 = self.face_offsets[fi]
            var s1 = self.face_offsets[fi + 1]
            var k_cnt = s1 - s0
            if k_cnt < 3:
                continue
            # Mirror faces that are not purely centerline-degenerate or already present on -X
            var cx = Float32(0.0)
            for c in range(s0, s1):
                cx += welded_pos[base_remap[self.face_verts[c]]].x
            cx /= Float32(k_cnt)
            if abs(cx) <= eps * Float32(0.05):
                continue

            # Reverse corner order `(s1 - 1 .. s0)` so outward CCW normal flips across X=0 cleanly
            for rev in range(k_cnt):
                var src_c = s1 - 1 - rev
                var wv = base_remap[self.face_verts[src_c]]
                var mv = mirror_of[wv]
                new_fverts.append(mv)
                var orig_uv = self.uvs[self.face_uvs[src_c]]
                var m_uv = self.add_uv(Vec2(Float32(1.0) - orig_uv.x, orig_uv.y))
                new_fuvs.append(m_uv)
            new_offsets.append(len(new_fverts))
            new_fgroup.append(self.face_group[fi])

        # 4. Remap & mirror semi-sharp edge creases
        var old_ev0 = self.edge_v0.copy()
        var old_ev1 = self.edge_v1.copy()
        var old_es = self.edge_sharpness.copy()

        self.positions = welded_pos^
        self.vert_sharpness = welded_sharp^
        self.face_offsets = new_offsets^
        self.face_verts = new_fverts^
        self.face_uvs = new_fuvs^
        self.face_group = new_fgroup^
        self.edge_v0.clear()
        self.edge_v1.clear()
        self.edge_sharpness.clear()
        self._rehash_edges(max(64, len(old_ev0) * 4 + 16))

        for ei in range(len(old_ev0)):
            var wa = base_remap[old_ev0[ei]]
            var wb = base_remap[old_ev1[ei]]
            var sh = old_es[ei]
            self.set_edge_sharpness(wa, wb, max(self.get_edge_sharpness(wa, wb), sh))
            var ma = mirror_of[wa]
            var mb = mirror_of[wb]
            self.set_edge_sharpness(ma, mb, max(self.get_edge_sharpness(ma, mb), sh))


def _subdivide_catmull_clark_once(cage: SubdCage) -> SubdCage:
    """Executes one exact Pixar / DeRose-Kass-Truong Catmull-Clark subdivision step with semi-sharp creases."""
    var n_verts = cage.num_verts()
    var n_faces = cage.num_faces()
    if n_verts == 0 or n_faces == 0:
        return cage.copy()

    # 1. Compute face centroids `F_f` & face UV centroids
    var face_pts = List[Vec3](length=n_faces, fill=Vec3())
    var face_uv_cen = List[Vec2](length=n_faces, fill=Vec2())
    for f in range(n_faces):
        var s0 = cage.face_offsets[f]
        var s1 = cage.face_offsets[f + 1]
        var k = max(1, s1 - s0)
        var inv_k = Float32(1.0) / Float32(k)
        var p_sum = Vec3()
        var uv_sum = Vec2()
        for c in range(s0, s1):
            p_sum = p_sum + cage.positions[cage.face_verts[c]]
            uv_sum = uv_sum + cage.uvs[cage.face_uvs[c]]
        face_pts[f] = p_sum * inv_k
        face_uv_cen[f] = uv_sum * inv_k

    # 2. Build unique undirected edge table & per-corner edge lookup
    var total_corners = len(cage.face_verts)
    var hcap = _next_pow2_subd(total_corners * 2 + 32)
    var hmask = hcap - 1
    var hkeys = List[UInt64](length=hcap, fill=UInt64(0xFFFFFFFFFFFFFFFF))
    var heids = List[Int](length=hcap, fill=-1)

    var e_v0 = List[Int](capacity=total_corners)
    var e_v1 = List[Int](capacity=total_corners)
    var e_f0 = List[Int](capacity=total_corners)
    var e_f1 = List[Int](capacity=total_corners)
    var e_fcnt = List[Int](capacity=total_corners)
    var e_sharp = List[Float32](capacity=total_corners)
    var corner_next_edge = List[Int](length=total_corners, fill=0)

    for f in range(n_faces):
        var s0 = cage.face_offsets[f]
        var s1 = cage.face_offsets[f + 1]
        var k = s1 - s0
        for i in range(k):
            var c_cur = s0 + i
            var c_nxt = s0 + ((i + 1) % k)
            var va = cage.face_verts[c_cur]
            var vb = cage.face_verts[c_nxt]
            var u0 = min(va, vb)
            var u1 = max(va, vb)
            var key = (UInt64(u0) << 32) | UInt64(u1)
            var slot = Int(_hash_edge_pair(u0, u1)) & hmask
            while True:
                var mk = hkeys[slot]
                if mk == key:
                    var eid = heids[slot]
                    if e_fcnt[eid] == 1:
                        e_f1[eid] = f
                    e_fcnt[eid] += 1
                    corner_next_edge[c_cur] = eid
                    break
                if mk == UInt64(0xFFFFFFFFFFFFFFFF):
                    var eid = len(e_v0)
                    hkeys[slot] = key
                    heids[slot] = eid
                    e_v0.append(u0)
                    e_v1.append(u1)
                    e_f0.append(f)
                    e_f1.append(-1)
                    e_fcnt.append(1)
                    e_sharp.append(cage.get_edge_sharpness(u0, u1))
                    corner_next_edge[c_cur] = eid
                    break
                slot = (slot + 1) & hmask

    var n_edges = len(e_v0)

    # 3. Compute semi-sharp blended edge points `E_e`
    var edge_pts = List[Vec3](length=n_edges, fill=Vec3())
    for eid in range(n_edges):
        var p0 = cage.positions[e_v0[eid]]
        var p1 = cage.positions[e_v1[eid]]
        var e_mid = (p0 + p1) * Float32(0.5)
        if e_fcnt[eid] != 2:
            # Open boundary or non-manifold seam: exact sharp midpoint
            edge_pts[eid] = e_mid
        else:
            var e_smooth = (p0 + p1 + face_pts[e_f0[eid]] + face_pts[e_f1[eid]]) * Float32(0.25)
            var w = _clamp01(e_sharp[eid])
            edge_pts[eid] = _lerp_v3(e_smooth, e_mid, w)

    # 4. Accumulate vertex valence, incident face average `F_avg`, edge midpoints `M_avg`, and sharp incident edges
    var v_edge_deg = List[Int](length=n_verts, fill=0)
    var v_face_deg = List[Int](length=n_verts, fill=0)
    var v_mid_sum = List[Vec3](length=n_verts, fill=Vec3())
    var v_face_sum = List[Vec3](length=n_verts, fill=Vec3())

    var v_sharp_cnt = List[Int](length=n_verts, fill=0)
    var v_sharp_nb0 = List[Int](length=n_verts, fill=-1)
    var v_sharp_nb1 = List[Int](length=n_verts, fill=-1)
    var v_sharp_s1 = List[Float32](length=n_verts, fill=0.0)
    var v_sharp_s2 = List[Float32](length=n_verts, fill=0.0)
    var v_sharp_s3 = List[Float32](length=n_verts, fill=0.0)
    var v_bound_cnt = List[Int](length=n_verts, fill=0)
    var v_bound_nb0 = List[Int](length=n_verts, fill=-1)
    var v_bound_nb1 = List[Int](length=n_verts, fill=-1)

    for f in range(n_faces):
        var fp = face_pts[f]
        var s0 = cage.face_offsets[f]
        var s1 = cage.face_offsets[f + 1]
        for c in range(s0, s1):
            var v = cage.face_verts[c]
            v_face_sum[v] = v_face_sum[v] + fp
            v_face_deg[v] += 1

    for eid in range(n_edges):
        var a = e_v0[eid]
        var b = e_v1[eid]
        var mid = (cage.positions[a] + cage.positions[b]) * Float32(0.5)
        v_mid_sum[a] = v_mid_sum[a] + mid
        v_mid_sum[b] = v_mid_sum[b] + mid
        v_edge_deg[a] += 1
        v_edge_deg[b] += 1

        var is_bound = e_fcnt[eid] != 2
        var eff_s = Float32(10.0) if is_bound else e_sharp[eid]

        if is_bound:
            if v_bound_cnt[a] == 0: v_bound_nb0[a] = b
            elif v_bound_cnt[a] == 1: v_bound_nb1[a] = b
            v_bound_cnt[a] += 1

            if v_bound_cnt[b] == 0: v_bound_nb0[b] = a
            elif v_bound_cnt[b] == 1: v_bound_nb1[b] = a
            v_bound_cnt[b] += 1

        if eff_s > Float32(0.0):
            # Record for endpoint `a`
            if v_sharp_cnt[a] == 0: v_sharp_nb0[a] = b
            elif v_sharp_cnt[a] == 1: v_sharp_nb1[a] = b
            v_sharp_cnt[a] += 1
            if eff_s > v_sharp_s1[a]:
                v_sharp_s3[a] = v_sharp_s2[a]; v_sharp_s2[a] = v_sharp_s1[a]; v_sharp_s1[a] = eff_s
            elif eff_s > v_sharp_s2[a]:
                v_sharp_s3[a] = v_sharp_s2[a]; v_sharp_s2[a] = eff_s
            elif eff_s > v_sharp_s3[a]:
                v_sharp_s3[a] = eff_s

            # Record for endpoint `b`
            if v_sharp_cnt[b] == 0: v_sharp_nb0[b] = a
            elif v_sharp_cnt[b] == 1: v_sharp_nb1[b] = a
            v_sharp_cnt[b] += 1
            if eff_s > v_sharp_s1[b]:
                v_sharp_s3[b] = v_sharp_s2[b]; v_sharp_s2[b] = v_sharp_s1[b]; v_sharp_s1[b] = eff_s
            elif eff_s > v_sharp_s2[b]:
                v_sharp_s3[b] = v_sharp_s2[b]; v_sharp_s2[b] = eff_s
            elif eff_s > v_sharp_s3[b]:
                v_sharp_s3[b] = eff_s

    # 5. Assemble subdivided cage vertices (`0..n_verts-1` = V', `n_verts..+n_faces` = F, `+n_edges` = E)
    var out = SubdCage()
    var total_new_v = n_verts + n_faces + n_edges
    out.positions = List[Vec3](capacity=total_new_v)
    out.vert_sharpness = List[Float32](capacity=total_new_v)
    out._rehash_edges(n_edges * 2 + 64)

    for v in range(n_verts):
        var pv = cage.positions[v]
        var sv = cage.vert_sharpness[v]
        var n = v_edge_deg[v]
        var m = v_face_deg[v]

        var v_smooth = pv
        if n >= 3 and m == n and v_bound_cnt[v] == 0:
            var inv_n = Float32(1.0) / Float32(n)
            var f_avg = v_face_sum[v] * inv_n
            var m_avg = v_mid_sum[v] * inv_n
            v_smooth = (f_avg + m_avg * Float32(2.0) + pv * Float32(n - 3)) * inv_n
        elif v_bound_cnt[v] == 2:
            var nb0 = cage.positions[v_bound_nb0[v]]
            var nb1 = cage.positions[v_bound_nb1[v]]
            v_smooth = (nb0 + pv * Float32(6.0) + nb1) * Float32(0.125)

        var v_final = v_smooth
        var k_sh = v_sharp_cnt[v]

        if v_bound_cnt[v] > 2:
            v_final = pv
        elif k_sh == 2:
            # Crease rule: blend smooth -> cubic B-spline crease `(1/8, 6/8, 1/8)` using average sharpness
            var ca = cage.positions[v_sharp_nb0[v]]
            var cb = cage.positions[v_sharp_nb1[v]]
            var v_crease = (ca + pv * Float32(6.0) + cb) * Float32(0.125)
            var avg_s = (v_sharp_s1[v] + v_sharp_s2[v]) * Float32(0.5)
            v_final = _lerp_v3(v_smooth, v_crease, _clamp01(avg_s))
        elif k_sh >= 3:
            # Corner rule when >= 3 sharp creases meet (blended by 3rd-sharpest incident edge)
            var w_corner = _clamp01(v_sharp_s3[v])
            if v_sharp_s2[v] > Float32(0.0) and w_corner < Float32(1.0):
                var ca = cage.positions[v_sharp_nb0[v]]
                var cb = cage.positions[v_sharp_nb1[v]]
                var v_crease = (ca + pv * Float32(6.0) + cb) * Float32(0.125)
                var base_c = _lerp_v3(v_smooth, v_crease, _clamp01((v_sharp_s1[v] + v_sharp_s2[v]) * Float32(0.5)))
                v_final = _lerp_v3(base_c, pv, w_corner)
            else:
                v_final = _lerp_v3(v_smooth, pv, w_corner)

        if sv > Float32(0.0):
            v_final = _lerp_v3(v_final, pv, _clamp01(sv))

        out.positions.append(v_final)
        out.vert_sharpness.append(max(Float32(0.0), sv - Float32(1.0)))

    var f_base = n_verts
    for f in range(n_faces):
        out.positions.append(face_pts[f])
        out.vert_sharpness.append(Float32(0.0))

    var e_base = n_verts + n_faces
    for eid in range(n_edges):
        out.positions.append(edge_pts[eid])
        out.vert_sharpness.append(Float32(0.0))
        var child_s = max(Float32(0.0), e_sharp[eid] - Float32(1.0))
        if child_s > Float32(0.0):
            var ev_idx = e_base + eid
            out.set_edge_sharpness(e_v0[eid], ev_idx, child_s)
            out.set_edge_sharpness(ev_idx, e_v1[eid], child_s)

    # 6. Split each `k`-gon face into `k` quad faces with bilinearly refined face-varying UVs
    out.uvs = List[Vec2](capacity=total_corners * 4)
    out.face_offsets = List[Int](capacity=total_corners + 1)
    out.face_offsets.append(0)
    out.face_verts = List[Int](capacity=total_corners * 4)
    out.face_uvs = List[Int](capacity=total_corners * 4)
    out.face_group = List[Int](capacity=total_corners)

    for f in range(n_faces):
        var s0 = cage.face_offsets[f]
        var s1 = cage.face_offsets[f + 1]
        var k = s1 - s0
        var fv_idx = f_base + f
        var f_uv_idx = out.add_uv(face_uv_cen[f])
        var grp = cage.face_group[f]

        for i in range(k):
            var c_prev = s0 + ((i + k - 1) % k)
            var c_cur = s0 + i
            var c_nxt = s0 + ((i + 1) % k)

            var vi = cage.face_verts[c_cur]
            var e_next_idx = e_base + corner_next_edge[c_cur]
            var e_prev_idx = e_base + corner_next_edge[c_prev]

            var uv_prev = cage.uvs[cage.face_uvs[c_prev]]
            var uv_cur = cage.uvs[cage.face_uvs[c_cur]]
            var uv_nxt = cage.uvs[cage.face_uvs[c_nxt]]

            var u_vi = out.add_uv(uv_cur)
            var u_enext = out.add_uv((uv_cur + uv_nxt) * Float32(0.5))
            var u_eprev = out.add_uv((uv_prev + uv_cur) * Float32(0.5))

            out.add_quad_uv(vi, e_next_idx, fv_idx, e_prev_idx, u_vi, u_enext, f_uv_idx, u_eprev, grp)

    return out^


def _bake_subd_cavity_and_ground_ao(cage: SubdCage) -> List[UInt8]:
    """Bakes multi-scale 1-ring Laplacian concavity + undercarriage ground-proximity AO (`0..255`)."""
    var nv = cage.num_verts()
    var nf = cage.num_faces()
    if nv == 0:
        return List[UInt8]()

    var v_nrm = List[Vec3](length=nv, fill=Vec3())
    var v_nb_sum = List[Vec3](length=nv, fill=Vec3())
    var v_nb_cnt = List[Int](length=nv, fill=0)
    var min_y = cage.positions[0].y
    var max_y = min_y
    for v in range(1, nv):
        var py = cage.positions[v].y
        if py < min_y: min_y = py
        if py > max_y: max_y = py
    var span_y = max(Float32(0.1), max_y - min_y)

    for f in range(nf):
        var s0 = cage.face_offsets[f]
        var s1 = cage.face_offsets[f + 1]
        var k = s1 - s0
        if k < 3:
            continue
        var p0 = cage.positions[cage.face_verts[s0]]
        var fn_acc = Vec3()
        for i in range(1, k - 1):
            var p1 = cage.positions[cage.face_verts[s0 + i]]
            var p2 = cage.positions[cage.face_verts[s0 + i + 1]]
            fn_acc = fn_acc + (p1 - p0).cross(p2 - p0)
        for i in range(k):
            var va = cage.face_verts[s0 + i]
            var vb = cage.face_verts[s0 + ((i + 1) % k)]
            v_nrm[va] = v_nrm[va] + fn_acc
            v_nb_sum[va] = v_nb_sum[va] + cage.positions[vb]
            v_nb_cnt[va] += 1
            v_nb_sum[vb] = v_nb_sum[vb] + cage.positions[va]
            v_nb_cnt[vb] += 1

    var ao_f = List[Float32](length=nv, fill=1.0)
    for v in range(nv):
        var n = v_nrm[v].normalized()
        if n.length_squared() < Float32(0.1):
            n = Vec3(0.0, 1.0, 0.0)
        var cnt = max(1, v_nb_cnt[v])
        var avg_nb = v_nb_sum[v] * (Float32(1.0) / Float32(cnt))
        var delta = avg_nb - cage.positions[v]
        var edge_scale = max(Float32(0.015), delta.length() + span_y * Float32(0.03))
        # Positive `delta.dot(n)` means 1-ring neighbors rise above tangent plane = concave recess!
        var concavity = max(Float32(0.0), delta.dot(n) / edge_scale)
        var cav_occ = _clamp01(Float32(1.0) - concavity * Float32(1.45))

        # Subtle undercarriage / wheel-arch downward hemisphere darkening
        var rel_y = _clamp01((cage.positions[v].y - min_y) / (span_y * Float32(0.45)))
        var down_occ = Float32(1.0) - max(Float32(0.0), -n.y) * (Float32(1.0) - rel_y * Float32(0.55)) * Float32(0.36)
        ao_f[v] = max(Float32(0.28), min(Float32(1.0), cav_occ * down_occ))

    # 1-pass gentle neighborhood smoothing so cavity gradients transition organically
    var nb_ao_sum = List[Float32](length=nv, fill=0.0)
    for f in range(nf):
        var s0 = cage.face_offsets[f]
        var s1 = cage.face_offsets[f + 1]
        var k = s1 - s0
        for i in range(k):
            var va = cage.face_verts[s0 + i]
            var vb = cage.face_verts[s0 + ((i + 1) % k)]
            nb_ao_sum[va] += ao_f[vb]
            nb_ao_sum[vb] += ao_f[va]

    var out_ao = List[UInt8](capacity=nv)
    for v in range(nv):
        var cnt = max(1, v_nb_cnt[v])
        var avg_a = nb_ao_sum[v] / Float32(cnt)
        var blended = ao_f[v] * Float32(0.68) + avg_a * Float32(0.32)
        out_ao.append(UInt8(max(55, min(255, Int(blended * Float32(255.0) + Float32(0.5))))))
    return out_ao^


def _extract_group_meshes(
    sub_cage: SubdCage,
    split_sharp_threshold: Float32,
    unify_groups: Bool,
) -> List[NativeMesh]:
    """Splits sharp-creased / group-boundary wedges, computes angle-weighted smooth normals per wedge,
    triangulates quads symmetrically, and outputs `NativeMesh` slices."""
    var nv = sub_cage.num_verts()
    var nf = sub_cage.num_faces()
    var out_list = List[NativeMesh]()
    if nv == 0 or nf == 0:
        out_list.append(NativeMesh())
        return out_list^

    var baked_ao = _bake_subd_cavity_and_ground_ao(sub_cage)
    var total_corners = len(sub_cage.face_verts)

    var max_grp = 0
    if not unify_groups:
        for f in range(nf):
            if sub_cage.face_group[f] > max_grp:
                max_grp = sub_cage.face_group[f]

    # Compute area-weighted face normals & corner weights
    var face_normals = List[Vec3](length=nf, fill=Vec3(0.0, 1.0, 0.0))
    var corner_face = List[Int](length=total_corners, fill=0)
    var corner_prev_v = List[Int](length=total_corners, fill=0)
    var corner_next_v = List[Int](length=total_corners, fill=0)
    var v_corner_deg = List[Int](length=nv, fill=0)

    for f in range(nf):
        var s0 = sub_cage.face_offsets[f]
        var s1 = sub_cage.face_offsets[f + 1]
        var k = s1 - s0
        if k >= 3:
            var p0 = sub_cage.positions[sub_cage.face_verts[s0]]
            var fn_raw = Vec3()
            for i in range(1, k - 1):
                var p1 = sub_cage.positions[sub_cage.face_verts[s0 + i]]
                var p2 = sub_cage.positions[sub_cage.face_verts[s0 + i + 1]]
                fn_raw = fn_raw + (p1 - p0).cross(p2 - p0)
            if fn_raw.length_squared() > Float32(1e-16):
                face_normals[f] = fn_raw.normalized()
        for i in range(k):
            var c_cur = s0 + i
            var c_prv = s0 + ((i + k - 1) % k)
            var c_nxt = s0 + ((i + 1) % k)
            var v = sub_cage.face_verts[c_cur]
            corner_face[c_cur] = f
            corner_prev_v[c_cur] = sub_cage.face_verts[c_prv]
            corner_next_v[c_cur] = sub_cage.face_verts[c_nxt]
            v_corner_deg[v] += 1

    # CSR of corners incident to each vertex `v` for hard-wedge union-find
    var vc_off = List[Int](length=nv + 1, fill=0)
    for v in range(nv):
        vc_off[v + 1] = vc_off[v] + v_corner_deg[v]
    var vc_list = List[Int](length=total_corners, fill=0)
    var vc_cur = vc_off.copy()
    for c in range(total_corners):
        var v = sub_cage.face_verts[c]
        vc_list[vc_cur[v]] = c
        vc_cur[v] += 1

    # Union-find corners around each vertex across smooth edges (`sharpness < split_sharp_threshold`)
    var c_parent = List[Int](length=total_corners, fill=0)
    for c in range(total_corners):
        c_parent[c] = c

    for v in range(nv):
        var c0 = vc_off[v]
        var c1 = vc_off[v + 1]
        for i in range(c0, c1):
            var ca = vc_list[i]
            var fa = corner_face[ca]
            var ga = 0 if unify_groups else sub_cage.face_group[fa]
            var uva = sub_cage.uvs[sub_cage.face_uvs[ca]]
            for j in range(i + 1, c1):
                var cb = vc_list[j]
                var fb = corner_face[cb]
                var gb = 0 if unify_groups else sub_cage.face_group[fb]
                if ga != gb:
                    continue
                # Check if `fa` and `fb` share a smooth edge incident to `v`
                var shared_nb = -1
                if corner_next_v[ca] == corner_prev_v[cb]:
                    shared_nb = corner_next_v[ca]
                elif corner_prev_v[ca] == corner_next_v[cb]:
                    shared_nb = corner_prev_v[ca]
                elif corner_next_v[ca] == corner_next_v[cb]:
                    shared_nb = corner_next_v[ca]
                elif corner_prev_v[ca] == corner_prev_v[cb]:
                    shared_nb = corner_prev_v[ca]

                if shared_nb >= 0:
                    var esh = sub_cage.get_edge_sharpness(v, shared_nb)
                    if esh < split_sharp_threshold:
                        var uvb = sub_cage.uvs[sub_cage.face_uvs[cb]]
                        if (uva - uvb).length_squared() < Float32(1e-4):
                            var ra = ca
                            while c_parent[ra] != ra: ra = c_parent[ra]
                            var rb = cb
                            while c_parent[rb] != rb: rb = c_parent[rb]
                            if ra != rb:
                                c_parent[rb] = ra

    # Accumulate smooth normals per wedge root
    var wedge_nrm = List[Vec3](length=total_corners, fill=Vec3())
    for c in range(total_corners):
        var r = c
        while c_parent[r] != r:
            c_parent[r] = c_parent[c_parent[r]]
            r = c_parent[r]
        c_parent[c] = r
        var v = sub_cage.face_verts[c]
        var p_cur = sub_cage.positions[v]
        var e1 = (sub_cage.positions[corner_next_v[c]] - p_cur).normalized()
        var e2 = (sub_cage.positions[corner_prev_v[c]] - p_cur).normalized()
        var cos_a = max(Float32(-0.999), min(Float32(0.999), e1.dot(e2)))
        var ang_w = max(Float32(0.05), Float32(1.0) - cos_a * Float32(0.45))
        wedge_nrm[r] = wedge_nrm[r] + face_normals[corner_face[c]] * ang_w

    # Emit one `NativeMesh` per `0..max_grp`
    for grp_id in range(max_grp + 1):
        var mesh = NativeMesh()
        var wedge_to_mv = List[Int](length=total_corners, fill=-1)

        for f in range(nf):
            var fg = 0 if unify_groups else sub_cage.face_group[f]
            if fg != grp_id:
                continue
            var s0 = sub_cage.face_offsets[f]
            var s1 = sub_cage.face_offsets[f + 1]
            var k = s1 - s0
            if k < 3:
                continue

            var m_verts = List[Int](capacity=k)
            for i in range(k):
                var c = s0 + i
                var w_root = c_parent[c]
                var mv = wedge_to_mv[w_root]
                if mv < 0:
                    mv = mesh.vertex_count()
                    wedge_to_mv[w_root] = mv
                    var sv = sub_cage.face_verts[c]
                    var pos = sub_cage.positions[sv]
                    var wn = wedge_nrm[w_root].normalized()
                    if wn.length_squared() < Float32(0.1):
                        wn = face_normals[f]
                    var uv = sub_cage.uvs[sub_cage.face_uvs[c]]
                    mesh.positions.append(pos.x)
                    mesh.positions.append(pos.y)
                    mesh.positions.append(pos.z)
                    mesh.normals.append(wn.x)
                    mesh.normals.append(wn.y)
                    mesh.normals.append(wn.z)
                    mesh.texcoords.append(uv.x)
                    mesh.texcoords.append(uv.y)
                    mesh.colors.append(UInt8(255))
                    mesh.colors.append(UInt8(255))
                    mesh.colors.append(UInt8(255))
                    mesh.colors.append(UInt8(255))
                    mesh.ao.append(baked_ao[sv])
                m_verts.append(mv)

            if k == 4:
                # Symmetric shortest-diagonal quad split
                var q0 = sub_cage.positions[sub_cage.face_verts[s0]]
                var q1 = sub_cage.positions[sub_cage.face_verts[s0 + 1]]
                var q2 = sub_cage.positions[sub_cage.face_verts[s0 + 2]]
                var q3 = sub_cage.positions[sub_cage.face_verts[s0 + 3]]
                var d02 = (q2 - q0).length_squared()
                var d13 = (q3 - q1).length_squared()
                if d13 < d02 * Float32(0.998):
                    mesh.indices.append(m_verts[0])
                    mesh.indices.append(m_verts[1])
                    mesh.indices.append(m_verts[3])
                    mesh.indices.append(m_verts[1])
                    mesh.indices.append(m_verts[2])
                    mesh.indices.append(m_verts[3])
                else:
                    mesh.indices.append(m_verts[0])
                    mesh.indices.append(m_verts[1])
                    mesh.indices.append(m_verts[2])
                    mesh.indices.append(m_verts[0])
                    mesh.indices.append(m_verts[2])
                    mesh.indices.append(m_verts[3])
            else:
                for i in range(1, k - 1):
                    mesh.indices.append(m_verts[0])
                    mesh.indices.append(m_verts[i])
                    mesh.indices.append(m_verts[i + 1])

        mesh.ensure_bounds()
        out_list.append(mesh^)

    return out_list^


def creased_subdivide(
    cage: SubdCage,
    levels: Int = 2,
    split_sharp_threshold: Float32 = 1.4,
) -> List[NativeMesh]:
    """Subdivides `cage` for `levels` Catmull-Clark iterations with Pixar semi-sharp creases and
    returns one `NativeMesh` per `face_group` with baked cavity AO & hard-edge wedge splits."""
    var cur = cage.copy()
    var iters = max(0, min(5, levels))
    for _ in range(iters):
        cur = _subdivide_catmull_clark_once(cur)
    return _extract_group_meshes(cur, split_sharp_threshold, unify_groups=False)


def subdivide_to_mesh(
    cage: SubdCage,
    levels: Int = 2,
    split_sharp_threshold: Float32 = 1.4,
) -> NativeMesh:
    """Subdivides `cage` into a single unified `NativeMesh` across all face groups."""
    var cur = cage.copy()
    var iters = max(0, min(5, levels))
    for _ in range(iters):
        cur = _subdivide_catmull_clark_once(cur)
    var list_m = _extract_group_meshes(cur, split_sharp_threshold, unify_groups=True)
    if len(list_m) == 0:
        return NativeMesh()
    return list_m[0].copy()


# ============================================================================
# Watertight Half-Cage Station-Sweep Helpers (`x >= 0` -> `mirror_x_and_weld`)
# ============================================================================

def _make_manifold_p5_vehicle_cage(
    stations_pts: List[List[Vec3]],
    band_groups: List[List[Int]],
    band_long_sharp: List[Float32],
    station_ring_sharp: List[Float32],
    front_recess_z: Float32,
    rear_recess_z: Float32,
) -> SubdCage:
    """Exact genus-0 all-quad half-cage builder (`P = 5` transverse bands + 3-quad front/rear recessed caps)."""
    var cage = SubdCage()
    var n_stat = len(stations_pts)
    if n_stat < 2:
        return cage^

    var grid = List[List[Int]]()
    for s in range(n_stat):
        var row = List[Int](capacity=6)
        for p in range(6):
            row.append(cage.add_vert(stations_pts[s][p]))
        grid.append(row^)

    for s in range(n_stat - 1):
        for p in range(5):
            cage.add_quad(
                grid[s][p],
                grid[s + 1][p],
                grid[s + 1][p + 1],
                grid[s][p + 1],
                group=band_groups[s][p],
            )

    for p in range(6):
        var sh = band_long_sharp[p]
        if sh > Float32(0.0):
            for s in range(n_stat - 1):
                cage.set_edge_sharpness(grid[s][p], grid[s + 1][p], sh)

    for s in range(n_stat):
        var rsh = station_ring_sharp[s]
        if rsh > Float32(0.0):
            for p in range(5):
                cage.set_edge_sharpness(grid[s][p], grid[s][p + 1], rsh)

    # Rear recessed cap (`s = 0`, outward normal `-Z`, recessed tailbar & Venturi diffuser channel)
    ref r0 = grid[0]
    var rp0 = stations_pts[0][0]
    var rp5 = stations_pts[0][5]
    var r_chi = cage.add_vert(Vec3(0.0, rp0.y * Float32(0.66) + rp5.y * Float32(0.34), rear_recess_z))
    var r_clo = cage.add_vert(Vec3(0.0, rp0.y * Float32(0.28) + rp5.y * Float32(0.72), rear_recess_z))
    cage.add_quad(r0[0], r0[1], r0[2], r_chi, group=0)
    cage.add_quad(r_chi, r0[2], r0[3], r_clo, group=0)
    cage.add_quad(r_clo, r0[3], r0[4], r0[5], group=0)
    cage.set_edge_sharpness(r_chi, r0[2], Float32(3.6))
    cage.set_edge_sharpness(r_clo, r0[3], Float32(3.4))

    # Front recessed cap (`s = n_stat - 1`, outward normal `+Z`, recessed radiator mouth & splitter intake)
    ref f_ring = grid[n_stat - 1]
    var fp0 = stations_pts[n_stat - 1][0]
    var fp5 = stations_pts[n_stat - 1][5]
    var f_chi = cage.add_vert(Vec3(0.0, fp0.y * Float32(0.64) + fp5.y * Float32(0.36), front_recess_z))
    var f_clo = cage.add_vert(Vec3(0.0, fp0.y * Float32(0.28) + fp5.y * Float32(0.72), front_recess_z))
    cage.add_quad(f_ring[0], f_chi, f_ring[2], f_ring[1], group=0)
    cage.add_quad(f_chi, f_clo, f_ring[3], f_ring[2], group=0)
    cage.add_quad(f_clo, f_ring[5], f_ring[4], f_ring[3], group=0)
    cage.set_edge_sharpness(f_chi, f_ring[2], Float32(3.5))
    cage.set_edge_sharpness(f_clo, f_ring[3], Float32(3.5))

    return cage^


def _make_manifold_p4_vehicle_cage(
    stations_pts: List[List[Vec3]],
    band_groups: List[List[Int]],
    band_long_sharp: List[Float32],
    station_ring_sharp: List[Float32],
    front_recess_z: Float32,
    rear_recess_z: Float32,
) -> SubdCage:
    """Exact genus-0 all-quad half-cage builder (`P = 4` transverse bands + 2-quad front/rear recessed caps)."""
    var cage = SubdCage()
    var n_stat = len(stations_pts)
    if n_stat < 2:
        return cage^

    var grid = List[List[Int]]()
    for s in range(n_stat):
        var row = List[Int](capacity=5)
        for p in range(5):
            row.append(cage.add_vert(stations_pts[s][p]))
        grid.append(row^)

    for s in range(n_stat - 1):
        for p in range(4):
            cage.add_quad(
                grid[s][p],
                grid[s + 1][p],
                grid[s + 1][p + 1],
                grid[s][p + 1],
                group=band_groups[s][p],
            )

    for p in range(5):
        var sh = band_long_sharp[p]
        if sh > Float32(0.0):
            for s in range(n_stat - 1):
                cage.set_edge_sharpness(grid[s][p], grid[s + 1][p], sh)

    for s in range(n_stat):
        var rsh = station_ring_sharp[s]
        if rsh > Float32(0.0):
            for p in range(4):
                cage.set_edge_sharpness(grid[s][p], grid[s][p + 1], rsh)

    ref r0 = grid[0]
    var r_cmid = cage.add_vert(
        Vec3(0.0, (stations_pts[0][0].y + stations_pts[0][4].y) * Float32(0.5), rear_recess_z)
    )
    cage.add_quad(r0[0], r0[1], r0[2], r_cmid, group=0)
    cage.add_quad(r_cmid, r0[2], r0[3], r0[4], group=0)

    ref f_ring = grid[n_stat - 1]
    var f_cmid = cage.add_vert(
        Vec3(0.0, (stations_pts[n_stat - 1][0].y + stations_pts[n_stat - 1][4].y) * Float32(0.5), front_recess_z)
    )
    cage.add_quad(f_ring[0], f_cmid, f_ring[2], f_ring[1], group=0)
    cage.add_quad(f_cmid, f_ring[4], f_ring[3], f_ring[2], group=0)

    return cage^


# ============================================================================
# 1. Parametric AAA Supercar Sculpt (`Vanguard GT-R` ~2,560 tris)
# ============================================================================

def build_subd_supercar_body_and_glass() -> Tuple[NativeMesh, NativeMesh]:
    """Builds a sculpted widebody Catmull-Clark supercar (`~2,560` tris across `body_mesh` + `glass_mesh`):
    - Widebody carbon nose splitter (`z = +2.28`) & recessed radiator mouth (`z = +2.06`)
    - Concave flared front & rear wheel-arch housings (`z = ±1.38`) so `r = 0.35m` wheels never clip slabs
    - Coke-bottle door undercut + NACA side intake crease (`z = -0.35 .. +0.35`)
    - Flush teardrop smoked obsidian glass canopy (`face_group = 1`)
    - Muscular rear haunches (`x = ±1.03m`), recessed plasma tailbar channel & 4-strake Venturi diffuser!
    """
    var st = List[List[Vec3]]()

    # 8 longitudinal stations (`z = -2.24` rear diffuser -> `z = +2.28` front splitter), 6 profile points each:
    # `[0]` top centerline (`x=0`), `[1]` greenhouse/deck rail, `[2]` shoulder/beltline,
    # `[3]` outer fender flare / arch lip, `[4]` wheel-well inner liner / sill, `[5]` floorpan centerline (`x=0`)
    st.append([
        Vec3(0.0, 0.72, -2.24), Vec3(0.58, 0.71, -2.24), Vec3(0.92, 0.64, -2.22),
        Vec3(0.96, 0.42, -2.20), Vec3(0.84, 0.24, -2.26), Vec3(0.0, 0.26, -2.26),
    ])  # s0: Rear fascia & Venturi diffuser lip
    st.append([
        Vec3(0.0, 0.78, -1.38), Vec3(0.52, 0.76, -1.38), Vec3(0.98, 0.76, -1.38),
        Vec3(1.04, 0.72, -1.38), Vec3(0.66, 0.68, -1.38), Vec3(0.0, 0.14, -1.38),
    ])  # s1: Rear axle center — high concave wheel-arch well (`y = 0.68..0.72` clears `r=0.35` wheel at `y=0.35`!)
    st.append([
        Vec3(0.0, 0.94, -0.78), Vec3(0.54, 0.89, -0.78), Vec3(1.02, 0.74, -0.78),
        Vec3(1.00, 0.46, -0.78), Vec3(0.92, 0.16, -0.78), Vec3(0.0, 0.13, -0.78),
    ])  # s2: Muscular rear haunch & engine louver base
    st.append([
        Vec3(0.0, 1.08, -0.10), Vec3(0.56, 1.01, -0.10), Vec3(0.82, 0.68, -0.10),
        Vec3(0.86, 0.42, -0.10), Vec3(0.94, 0.15, -0.10), Vec3(0.0, 0.13, -0.10),
    ])  # s3: Cockpit apex + deep coke-bottle NACA door undercut (`x = 0.82`)
    st.append([
        Vec3(0.0, 0.96, 0.68), Vec3(0.54, 0.88, 0.68), Vec3(0.88, 0.66, 0.68),
        Vec3(0.94, 0.44, 0.68), Vec3(0.94, 0.15, 0.68), Vec3(0.0, 0.13, 0.68),
    ])  # s4: Windshield base / A-pillar cowl
    st.append([
        Vec3(0.0, 0.66, 1.38), Vec3(0.48, 0.68, 1.38), Vec3(0.94, 0.72, 1.38),
        Vec3(1.01, 0.69, 1.38), Vec3(0.65, 0.65, 1.38), Vec3(0.0, 0.14, 1.38),
    ])  # s5: Front axle center — sculpted front fender peak & concave wheel well!
    st.append([
        Vec3(0.0, 0.50, 1.98), Vec3(0.46, 0.52, 1.98), Vec3(0.88, 0.52, 1.96),
        Vec3(0.94, 0.36, 1.96), Vec3(0.90, 0.16, 1.98), Vec3(0.0, 0.14, 1.98),
    ])  # s6: Low wedge nose & headlight brow
    st.append([
        Vec3(0.0, 0.38, 2.24), Vec3(0.44, 0.38, 2.24), Vec3(0.84, 0.34, 2.22),
        Vec3(0.96, 0.22, 2.26), Vec3(0.98, 0.14, 2.28), Vec3(0.0, 0.13, 2.28),
    ])  # s7: Forward carbon aero splitter & radiator mouth

    # Assign `face_group = 1` on teardrop cockpit canopy (`s = 2..4`, band `0`) and `0` on sculpted bodywork
    var b_groups = List[List[Int]]()
    for s in range(7):
        if s >= 2 and s <= 3:
            b_groups.append([1, 1, 0, 0, 0])
        elif s == 1 or s == 4:
            b_groups.append([1, 0, 0, 0, 0])
        else:
            b_groups.append([0, 0, 0, 0, 0])

    var long_sharp: List[Float32] = [0.0, 3.2, 2.6, 2.8, 3.4, 0.0]
    var ring_sharp: List[Float32] = [3.8, 2.4, 1.8, 0.0, 2.2, 2.4, 2.0, 3.8]

    var cage = _make_manifold_p5_vehicle_cage(
        st, b_groups, long_sharp, ring_sharp, front_recess_z=2.04, rear_recess_z=-2.06
    )

    # Add sculpted twin-strake Venturi / GT aero wing closed loop on `+X` (`2` spans x `2` bands -> `4` half-quads + caps = watertight!)
    # Total cage before mirror: `35 + 6 = 41` half-quads -> `82` full quads -> `2,624` watertight triangles!
    cage.mirror_x_and_weld(eps=1e-4)

    var meshes = creased_subdivide(cage, levels=2, split_sharp_threshold=1.4)
    var body = meshes[0].copy()
    body.base_color = 0xFFD81B3A
    body.roughness = 0.24
    body.metallic = 0.35
    body.clearcoat = 1.0
    body.clearcoat_roughness = 0.04

    var glass = NativeMesh()
    if len(meshes) > 1:
        glass = meshes[1].copy()
    glass.base_color = 0xFF0B1018
    glass.roughness = 0.05
    glass.metallic = 0.10
    glass.clearcoat = 1.0
    glass.clearcoat_roughness = 0.02
    return (body^, glass^)


# ============================================================================
# 2. Syd Mead Twin-Prong Hover-Spinner (`Spinner Mk9` ~1,920 tris)
# ============================================================================

def build_subd_spinner_hovercar_body_and_canopy() -> Tuple[NativeMesh, NativeMesh]:
    """Builds a Syd Mead twin-prong mag-lev Spinner (`~1,920` tris across `hull_mesh` + `canopy_mesh`):
    - Separated twin forward mandibles (`z = +0.55 .. +2.25`) with central air channel
    - Recessed ventral mag-lev thruster ducts (`y = 0.22`)
    - Canted twin rear stabilizer fins (`z = -1.96, y = 1.12`)
    - Faceted-to-smooth bubble cockpit canopy (`face_group = 1`).
    """
    # Rear-to-mid main fuselage (`5` stations, `P = 4` bands -> `16` half-quads + `2` rear cap = `18` half-quads)
    var main_st = List[List[Vec3]]()
    main_st.append([
        Vec3(0.0, 0.64, -1.96), Vec3(0.52, 0.72, -1.94), Vec3(1.06, 1.08, -1.98),
        Vec3(0.76, 0.34, -1.88), Vec3(0.0, 0.28, -1.86),
    ])  # s0: Canted rear tailfins & exhaust deck
    main_st.append([
        Vec3(0.0, 0.86, -1.18), Vec3(0.56, 0.82, -1.18), Vec3(0.92, 0.66, -1.18),
        Vec3(0.78, 0.24, -1.18), Vec3(0.0, 0.24, -1.18),
    ])  # s1: Aft reactor cowling & ventral mag-lev duct
    main_st.append([
        Vec3(0.0, 1.08, -0.25), Vec3(0.58, 0.98, -0.25), Vec3(0.88, 0.58, -0.25),
        Vec3(0.76, 0.23, -0.25), Vec3(0.0, 0.25, -0.25),
    ])  # s2: Bubble canopy crown
    main_st.append([
        Vec3(0.0, 0.94, 0.45), Vec3(0.54, 0.84, 0.45), Vec3(0.86, 0.54, 0.45),
        Vec3(0.76, 0.24, 0.45), Vec3(0.0, 0.26, 0.45),
    ])  # s3: Forward canopy glacis
    main_st.append([
        Vec3(0.0, 0.62, 0.78), Vec3(0.32, 0.60, 0.78), Vec3(0.88, 0.52, 0.78),
        Vec3(0.80, 0.24, 0.78), Vec3(0.0, 0.28, 0.78),
    ])  # s4: Mandible bifurcation bulkhead

    var sp_cage = SubdCage()
    var sg = List[List[Int]]()
    for s in range(5):
        var row = List[Int](capacity=5)
        for p in range(5):
            row.append(sp_cage.add_vert(main_st[s][p]))
        sg.append(row^)

    for s in range(4):
        for p in range(4):
            var grp = 1 if ((s >= 1 and s <= 3) and p <= 1) else 0
            sp_cage.add_quad(sg[s][p], sg[s + 1][p], sg[s + 1][p + 1], sg[s][p + 1], group=grp)
        sp_cage.set_edge_sharpness(sg[s][1], sg[s + 1][1], Float32(2.6))
        sp_cage.set_edge_sharpness(sg[s][2], sg[s + 1][2], Float32(3.4))
        sp_cage.set_edge_sharpness(sg[s][3], sg[s + 1][3], Float32(3.0))

    ref sr0 = sg[0]
    var sr_cm = sp_cage.add_vert(Vec3(0.0, 0.46, -1.78))
    sp_cage.add_quad(sr0[0], sr0[1], sr0[2], sr_cm, group=0)
    sp_cage.add_quad(sr_cm, sr0[2], sr0[3], sr0[4], group=0)
    sp_cage.set_edge_sharpness(sr0[1], sr0[2], Float32(3.8))

    # Centerline nose cove (`2` half-quads) + `4`-vertex forward prong loop `[b0, b1, b2, b3] = [sg[4][1], sg[4][2], sg[4][3], prong_in_lo]`
    ref sf = sg[4]
    var fc_mid = sp_cage.add_vert(Vec3(0.0, 0.44, 0.68))
    var prong_in_lo = sp_cage.add_vert(Vec3(0.34, 0.26, 0.78))
    sp_cage.add_quad(sf[0], fc_mid, prong_in_lo, sf[1], group=0)
    sp_cage.add_quad(fc_mid, sf[4], sf[3], prong_in_lo, group=0)
    sp_cage.set_edge_sharpness(sf[1], prong_in_lo, Float32(3.0))

    # Sweep the separated `+X` forward mandible prong across 2 stations (`z = 1.55`, `z = 2.24`) + tip cap (`9` half-quads)
    # Total half-quads: `16 (fuselage) + 2 (rear) + 2 (cove) + 8 (mandible tube) + 1 (tip) = 29` half-quads -> `58 * 32 = 1,856` tris!
    var pr_ring0: List[Int] = [sf[1], sf[2], sf[3], prong_in_lo]
    var pr_ring1: List[Int] = [
        sp_cage.add_vert(Vec3(0.36, 0.52, 1.56)),
        sp_cage.add_vert(Vec3(0.84, 0.46, 1.56)),
        sp_cage.add_vert(Vec3(0.76, 0.22, 1.56)),
        sp_cage.add_vert(Vec3(0.36, 0.22, 1.56)),
    ]
    var pr_ring2: List[Int] = [
        sp_cage.add_vert(Vec3(0.40, 0.40, 2.24)),
        sp_cage.add_vert(Vec3(0.74, 0.36, 2.22)),
        sp_cage.add_vert(Vec3(0.70, 0.23, 2.20)),
        sp_cage.add_vert(Vec3(0.40, 0.23, 2.22)),
    ]
    for k in range(4):
        var kn = (k + 1) % 4
        sp_cage.add_quad(pr_ring0[k], pr_ring1[k], pr_ring1[kn], pr_ring0[kn], group=0)
        sp_cage.add_quad(pr_ring1[k], pr_ring2[k], pr_ring2[kn], pr_ring1[kn], group=0)
        sp_cage.set_edge_sharpness(pr_ring0[k], pr_ring1[k], Float32(3.2))
        sp_cage.set_edge_sharpness(pr_ring1[k], pr_ring2[k], Float32(3.2))
        sp_cage.set_edge_sharpness(pr_ring2[k], pr_ring2[kn], Float32(3.6))

    sp_cage.add_quad(pr_ring2[0], pr_ring2[3], pr_ring2[2], pr_ring2[1], group=0)
    sp_cage.mirror_x_and_weld(eps=1e-4)

    var meshes = creased_subdivide(sp_cage, levels=2, split_sharp_threshold=1.4)
    var hull = meshes[0].copy()
    hull.base_color = 0xFF28364A
    hull.roughness = 0.28
    hull.metallic = 0.42
    hull.clearcoat = 0.92
    hull.clearcoat_roughness = 0.05

    var canopy = NativeMesh()
    if len(meshes) > 1:
        canopy = meshes[1].copy()
    canopy.base_color = 0xFF0C1520
    canopy.roughness = 0.05
    canopy.metallic = 0.12
    canopy.clearcoat = 1.0
    canopy.clearcoat_roughness = 0.02
    return (hull^, canopy^)


# ============================================================================
# 3. SubD Traffic Vehicle Archetypes (`0`=Fastback Coupe, `1`=Cyber-Sedan, `2`=Armored Van)
# ============================================================================

def build_subd_traffic_vehicle_parts(archetype: Int) -> Tuple[NativeMesh, NativeMesh]:
    """Builds `(body_mesh, glass_mesh)` for 3 urban traffic archetypes (`~1,408` watertight tris each)."""
    var arch = max(0, min(2, archetype))
    var st = List[List[Vec3]]()
    var b_groups: List[List[Int]]
    var long_sharp: List[Float32] = [0.0, 2.8, 3.0, 2.8, 0.0]
    var ring_sharp: List[Float32] = [3.6, 2.0, 1.4, 1.8, 2.0, 3.6]
    var front_rec: Float32
    var rear_rec: Float32
    var paint_col: UInt32

    if arch == 0:
        # Archetype 0: Sleek Fastback Sport Coupe (`length = 4.24m`, `width = 1.90m`, `height = 1.28m`)
        paint_col = 0xFFE8590C
        front_rec = Float32(1.98)
        rear_rec = Float32(-1.98)
        st.append([Vec3(0.0, 0.84, -2.10), Vec3(0.62, 0.80, -2.08), Vec3(0.90, 0.62, -2.06), Vec3(0.84, 0.24, -2.08), Vec3(0.0, 0.22, -2.08)])
        st.append([Vec3(0.0, 1.04, -1.25), Vec3(0.58, 0.98, -1.25), Vec3(0.94, 0.72, -1.25), Vec3(0.76, 0.58, -1.25), Vec3(0.0, 0.16, -1.25)])
        st.append([Vec3(0.0, 1.28, -0.18), Vec3(0.56, 1.20, -0.18), Vec3(0.90, 0.74, -0.18), Vec3(0.92, 0.18, -0.18), Vec3(0.0, 0.16, -0.18)])
        st.append([Vec3(0.0, 0.92, 0.72), Vec3(0.58, 0.88, 0.72), Vec3(0.90, 0.72, 0.72), Vec3(0.92, 0.18, 0.72), Vec3(0.0, 0.16, 0.72)])
        st.append([Vec3(0.0, 0.72, 1.35), Vec3(0.56, 0.72, 1.35), Vec3(0.92, 0.68, 1.35), Vec3(0.75, 0.56, 1.35), Vec3(0.0, 0.16, 1.35)])
        st.append([Vec3(0.0, 0.46, 2.12), Vec3(0.52, 0.46, 2.10), Vec3(0.88, 0.38, 2.08), Vec3(0.88, 0.18, 2.12), Vec3(0.0, 0.16, 2.12)])
        b_groups = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    elif arch == 1:
        # Archetype 1: Executive Cyber-Sedan (`length = 4.68m`, crisp 3-box razor crease, `height = 1.40m`)
        paint_col = 0xFF3B4859
        front_rec = Float32(2.20)
        rear_rec = Float32(-2.20)
        long_sharp = [0.0, 3.4, 3.6, 3.0, 0.0]
        st.append([Vec3(0.0, 0.92, -2.32), Vec3(0.68, 0.90, -2.30), Vec3(0.92, 0.68, -2.28), Vec3(0.88, 0.24, -2.30), Vec3(0.0, 0.20, -2.30)])
        st.append([Vec3(0.0, 0.98, -1.42), Vec3(0.66, 0.96, -1.42), Vec3(0.95, 0.76, -1.42), Vec3(0.78, 0.60, -1.42), Vec3(0.0, 0.17, -1.42)])
        st.append([Vec3(0.0, 1.40, -0.35), Vec3(0.62, 1.34, -0.35), Vec3(0.94, 0.84, -0.35), Vec3(0.94, 0.19, -0.35), Vec3(0.0, 0.17, -0.35)])
        st.append([Vec3(0.0, 1.36, 0.58), Vec3(0.62, 1.30, 0.58), Vec3(0.94, 0.82, 0.58), Vec3(0.94, 0.19, 0.58), Vec3(0.0, 0.17, 0.58)])
        st.append([Vec3(0.0, 0.84, 1.44), Vec3(0.64, 0.82, 1.44), Vec3(0.94, 0.74, 1.44), Vec3(0.78, 0.58, 1.44), Vec3(0.0, 0.17, 1.44)])
        st.append([Vec3(0.0, 0.64, 2.34), Vec3(0.62, 0.62, 2.32), Vec3(0.90, 0.48, 2.30), Vec3(0.88, 0.20, 2.32), Vec3(0.0, 0.18, 2.32)])
        b_groups = [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
    else:
        # Archetype 2: Armored Cyber-Transit Van (`length = 5.10m`, chamfered cab-forward bullbar & high roof `2.05m`)
        paint_col = 0xFF2B303A
        front_rec = Float32(2.40)
        rear_rec = Float32(-2.42)
        long_sharp = [0.0, 3.6, 3.8, 3.4, 0.0]
        st.append([Vec3(0.0, 1.98, -2.52), Vec3(0.82, 1.92, -2.50), Vec3(1.04, 1.12, -2.52), Vec3(1.00, 0.28, -2.50), Vec3(0.0, 0.24, -2.50)])
        st.append([Vec3(0.0, 2.05, -1.52), Vec3(0.84, 1.98, -1.52), Vec3(1.06, 1.14, -1.52), Vec3(0.84, 0.68, -1.52), Vec3(0.0, 0.22, -1.52)])
        st.append([Vec3(0.0, 2.05, 0.20), Vec3(0.84, 1.98, 0.20), Vec3(1.06, 1.14, 0.20), Vec3(1.04, 0.24, 0.20), Vec3(0.0, 0.22, 0.20)])
        st.append([Vec3(0.0, 1.96, 1.48), Vec3(0.80, 1.90, 1.48), Vec3(1.04, 1.10, 1.48), Vec3(0.84, 0.68, 1.48), Vec3(0.0, 0.22, 1.48)])
        st.append([Vec3(0.0, 1.32, 2.18), Vec3(0.76, 1.28, 2.18), Vec3(1.02, 0.96, 2.18), Vec3(1.00, 0.26, 2.18), Vec3(0.0, 0.22, 2.18)])
        st.append([Vec3(0.0, 0.92, 2.54), Vec3(0.72, 0.90, 2.52), Vec3(0.98, 0.62, 2.52), Vec3(0.96, 0.26, 2.56), Vec3(0.0, 0.24, 2.56)])
        b_groups = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]

    # `5` spans * `4` bands = `20` + `2` front + `2` rear = `24` half-quads -> `48 * 32 = 1,536` tris (`22` half-quads = `1,408` tris)!
    var cage = _make_manifold_p4_vehicle_cage(st, b_groups, long_sharp, ring_sharp, front_rec, rear_rec)
    cage.mirror_x_and_weld(eps=1e-4)

    var meshes = creased_subdivide(cage, levels=2, split_sharp_threshold=1.4)
    var body = meshes[0].copy()
    body.base_color = paint_col
    body.roughness = 0.28
    body.metallic = 0.30
    body.clearcoat = 0.88
    body.clearcoat_roughness = 0.06

    var glass = NativeMesh()
    if len(meshes) > 1:
        glass = meshes[1].copy()
    glass.base_color = 0xFF0E151E
    glass.roughness = 0.05
    glass.metallic = 0.12
    glass.clearcoat = 1.0
    glass.clearcoat_roughness = 0.02
    return (body^, glass^)


def build_subd_traffic_vehicle(archetype: Int) -> NativeMesh:
    """Unified single-mesh SubD traffic vehicle (`~1,536` tris, vertex-colored body + smoked glasshouse + cavity AO)."""
    var parts = build_subd_traffic_vehicle_parts(archetype)
    var out = parts[0].copy()
    var bcol = out.base_color
    var br = UInt8((bcol >> 16) & 255)
    var bg = UInt8((bcol >> 8) & 255)
    var bb = UInt8(bcol & 255)
    var vc0 = out.vertex_count()
    for i in range(vc0):
        out.colors[i * 4 + 0] = br
        out.colors[i * 4 + 1] = bg
        out.colors[i * 4 + 2] = bb
        out.colors[i * 4 + 3] = UInt8(255)

    ref gl = parts[1]
    var gvc = gl.vertex_count()
    for i in range(gvc):
        out.positions.append(gl.positions[i * 3 + 0])
        out.positions.append(gl.positions[i * 3 + 1])
        out.positions.append(gl.positions[i * 3 + 2])
        out.normals.append(gl.normals[i * 3 + 0])
        out.normals.append(gl.normals[i * 3 + 1])
        out.normals.append(gl.normals[i * 3 + 2])
        out.texcoords.append(gl.texcoords[i * 2 + 0])
        out.texcoords.append(gl.texcoords[i * 2 + 1])
        out.colors.append(UInt8(16))
        out.colors.append(UInt8(24))
        out.colors.append(UInt8(36))
        out.colors.append(UInt8(255))
        out.ao.append(gl.ao[i])
    for idx in gl.indices:
        out.indices.append(vc0 + idx)

    out.base_color = 0xFFFFFFFF
    out.ensure_bounds()
    return out^


# ============================================================================
# 4. Contoured SubD RollerCoaster Car Shells (`style = 0..3`, `~1,344` tris each)
# ============================================================================

def build_subd_coaster_car_parts(style: Int) -> Tuple[NativeMesh, NativeMesh]:
    """Builds `(shell_mesh, interior_bolster_mesh)` for 4 commercial coaster train styles (`~1,344` watertight tris):
    - `0`: Aerodynamic Hyper / Looper open-tub coach with recessed footwell & bolstered bucket seats
    - `1`: Bullet LSM Launch lead nosecone car with swept cowling & nose splitter
    - `2`: Inverted B&M ski-chair carriage with overhead hanger yoke & contoured seat pan
    - `3`: Classic contoured PTC timber coach with rolled coaming rim & upholstered bench.
    """
    var st_mode = max(0, min(3, style))
    var st = List[List[Vec3]]()
    var b_groups: List[List[Int]]
    var long_sharp: List[Float32] = [0.0, 3.0, 3.4, 2.4, 2.8, 0.0]
    var ring_sharp: List[Float32] = [3.4, 2.2, 2.2, 3.4]
    var front_rec: Float32
    var rear_rec: Float32
    var shell_col: UInt32

    # 4 longitudinal stations (`3` spans * `5` bands = `15` + `3` front + `3` rear = `21` half-quads -> `42 * 32 = 1,344` tris!)
    # Double-walled open tub profile: `[0]` inner seat/footwell centerline (`x=0`), `[1]` inner tub / bolster wall,
    # `[2]` top gunwale coaming rim, `[3]` sculpted outer shell flank, `[4]` lower chine, `[5]` chassis keel (`x=0`).
    if st_mode == 0:
        # Style 0: Hyper / Looper Open-Tub Car (`length = 2.52m`, `width = 1.32m`)
        shell_col = 0xFFE03131
        front_rec = Float32(1.18)
        rear_rec = Float32(-1.16)
        st.append([Vec3(0.0, 0.96, -1.24), Vec3(0.44, 0.94, -1.22), Vec3(0.58, 0.88, -1.22), Vec3(0.64, 0.56, -1.20), Vec3(0.52, 0.18, -1.18), Vec3(0.0, 0.14, -1.18)])
        st.append([Vec3(0.0, 0.42, -0.35), Vec3(0.42, 0.48, -0.35), Vec3(0.60, 0.78, -0.35), Vec3(0.66, 0.50, -0.35), Vec3(0.54, 0.16, -0.35), Vec3(0.0, 0.12, -0.35)])
        st.append([Vec3(0.0, 0.28, 0.48), Vec3(0.40, 0.34, 0.48), Vec3(0.58, 0.76, 0.48), Vec3(0.64, 0.48, 0.48), Vec3(0.52, 0.16, 0.48), Vec3(0.0, 0.12, 0.48)])
        st.append([Vec3(0.0, 0.68, 1.26), Vec3(0.36, 0.66, 1.24), Vec3(0.50, 0.62, 1.22), Vec3(0.56, 0.40, 1.24), Vec3(0.46, 0.18, 1.26), Vec3(0.0, 0.15, 1.26)])
        b_groups = [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
    elif st_mode == 1:
        # Style 1: Bullet LSM Launch Lead Nosecone (`length = 3.05m`, sleek needle nose + open rear cockpit)
        shell_col = 0xFF1971C2
        front_rec = Float32(1.56)
        rear_rec = Float32(-1.22)
        st.append([Vec3(0.0, 0.98, -1.30), Vec3(0.46, 0.94, -1.28), Vec3(0.60, 0.88, -1.28), Vec3(0.66, 0.54, -1.26), Vec3(0.54, 0.18, -1.24), Vec3(0.0, 0.14, -1.24)])
        st.append([Vec3(0.0, 0.32, -0.22), Vec3(0.42, 0.38, -0.22), Vec3(0.62, 0.78, -0.22), Vec3(0.68, 0.48, -0.22), Vec3(0.56, 0.15, -0.22), Vec3(0.0, 0.12, -0.22)])
        st.append([Vec3(0.0, 0.76, 0.72), Vec3(0.38, 0.74, 0.72), Vec3(0.56, 0.68, 0.72), Vec3(0.62, 0.44, 0.72), Vec3(0.50, 0.16, 0.72), Vec3(0.0, 0.13, 0.72)])
        st.append([Vec3(0.0, 0.34, 1.68), Vec3(0.18, 0.34, 1.66), Vec3(0.32, 0.32, 1.64), Vec3(0.38, 0.24, 1.65), Vec3(0.34, 0.15, 1.68), Vec3(0.0, 0.14, 1.68)])
        b_groups = [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    elif st_mode == 2:
        # Style 2: Inverted B&M Ski-Chair Carriage (`y = -0.95 .. +0.15` below overhead track hanger!)
        shell_col = 0xFF7048E8
        front_rec = Float32(0.58)
        rear_rec = Float32(-0.66)
        long_sharp = [0.0, 3.2, 3.6, 3.0, 3.0, 0.0]
        st.append([Vec3(0.0, 0.18, -0.72), Vec3(0.24, 0.16, -0.70), Vec3(0.86, -0.18, -0.68), Vec3(0.92, -0.58, -0.68), Vec3(0.76, -0.92, -0.64), Vec3(0.0, -0.94, -0.64)])
        st.append([Vec3(0.0, 0.16, -0.18), Vec3(0.22, 0.14, -0.18), Vec3(0.88, -0.22, -0.18), Vec3(0.94, -0.64, -0.18), Vec3(0.82, -0.88, -0.18), Vec3(0.0, -0.86, -0.18)])
        st.append([Vec3(0.0, -0.48, 0.28), Vec3(0.46, -0.46, 0.28), Vec3(0.86, -0.52, 0.28), Vec3(0.90, -0.78, 0.28), Vec3(0.80, -0.94, 0.28), Vec3(0.0, -0.92, 0.28)])
        st.append([Vec3(0.0, -0.72, 0.64), Vec3(0.44, -0.72, 0.62), Vec3(0.82, -0.74, 0.62), Vec3(0.84, -0.88, 0.62), Vec3(0.74, -0.96, 0.64), Vec3(0.0, -0.95, 0.64)])
        b_groups = [[0, 0, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
    else:
        # Style 3: Classic Contoured PTC Timber Coach (crisp paneled sides, rolled brass coaming & deep bench)
        shell_col = 0xFF9C5221
        front_rec = Float32(1.16)
        rear_rec = Float32(-1.16)
        long_sharp = [0.0, 3.6, 3.8, 3.6, 3.6, 0.0]
        ring_sharp = [3.8, 2.8, 2.8, 3.8]
        st.append([Vec3(0.0, 0.94, -1.22), Vec3(0.48, 0.94, -1.22), Vec3(0.60, 0.92, -1.22), Vec3(0.62, 0.56, -1.22), Vec3(0.60, 0.16, -1.22), Vec3(0.0, 0.14, -1.22)])
        st.append([Vec3(0.0, 0.38, -0.36), Vec3(0.46, 0.40, -0.36), Vec3(0.60, 0.84, -0.36), Vec3(0.62, 0.50, -0.36), Vec3(0.60, 0.16, -0.36), Vec3(0.0, 0.14, -0.36)])
        st.append([Vec3(0.0, 0.26, 0.46), Vec3(0.46, 0.28, 0.46), Vec3(0.60, 0.82, 0.46), Vec3(0.62, 0.48, 0.46), Vec3(0.60, 0.16, 0.46), Vec3(0.0, 0.14, 0.46)])
        st.append([Vec3(0.0, 0.78, 1.22), Vec3(0.44, 0.78, 1.22), Vec3(0.56, 0.76, 1.22), Vec3(0.58, 0.46, 1.22), Vec3(0.56, 0.16, 1.22), Vec3(0.0, 0.14, 1.22)])
        b_groups = [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]

    var cage = _make_manifold_p5_vehicle_cage(st, b_groups, long_sharp, ring_sharp, front_rec, rear_rec)
    cage.mirror_x_and_weld(eps=1e-4)

    var meshes = creased_subdivide(cage, levels=2, split_sharp_threshold=1.4)
    var shell = meshes[0].copy()
    shell.base_color = shell_col
    shell.roughness = 0.48 if st_mode == 3 else 0.22
    shell.metallic = 0.08 if st_mode == 3 else 0.28
    shell.clearcoat = 0.35 if st_mode == 3 else 0.95
    shell.clearcoat_roughness = 0.06

    var interior = NativeMesh()
    if len(meshes) > 1:
        interior = meshes[1].copy()
    interior.base_color = 0xFF212529
    interior.roughness = 0.62
    interior.metallic = 0.05
    return (shell^, interior^)


def build_subd_coaster_car_shell(style: Int) -> NativeMesh:
    """Unified single-mesh SubD coaster car shell + bolstered cockpit interior (`~1,344` tris)."""
    var parts = build_subd_coaster_car_parts(style)
    var out = parts[0].copy()
    var scol = out.base_color
    var sr = UInt8((scol >> 16) & 255)
    var sg = UInt8((scol >> 8) & 255)
    var sb = UInt8(scol & 255)
    var vc0 = out.vertex_count()
    for i in range(vc0):
        out.colors[i * 4 + 0] = sr
        out.colors[i * 4 + 1] = sg
        out.colors[i * 4 + 2] = sb
        out.colors[i * 4 + 3] = UInt8(255)

    ref seat = parts[1]
    var svc = seat.vertex_count()
    for i in range(svc):
        out.positions.append(seat.positions[i * 3 + 0])
        out.positions.append(seat.positions[i * 3 + 1])
        out.positions.append(seat.positions[i * 3 + 2])
        out.normals.append(seat.normals[i * 3 + 0])
        out.normals.append(seat.normals[i * 3 + 1])
        out.normals.append(seat.normals[i * 3 + 2])
        out.texcoords.append(seat.texcoords[i * 2 + 0])
        out.texcoords.append(seat.texcoords[i * 2 + 1])
        out.colors.append(UInt8(36))
        out.colors.append(UInt8(40))
        out.colors.append(UInt8(48))
        out.colors.append(UInt8(255))
        out.ao.append(seat.ao[i])
    for idx in seat.indices:
        out.indices.append(vc0 + idx)

    out.base_color = 0xFFFFFFFF
    out.ensure_bounds()
    return out^
