"""Parametric curved 3D mesh builders for NativeMesh (lathes, lofts, airfoils, ellipsoids, capsules, beveled hulls)."""
from std.math import sin, cos, sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.assets.model import NativeMesh, NativeModel, NativeNode


comptime PI_F32: Float32 = 3.14159265
comptime TAU_F32: Float32 = 6.28318530


struct MeshBuilder(Copyable, Movable):
    var mesh: NativeMesh

    def __init__(
        out self,
        base_color: UInt32 = 0xFFFFFFFF,
        roughness: Float32 = 0.45,
        metallic: Float32 = 0.15,
        texture: Int = -1,
        emissive_color: Vec3 = Vec3(1, 1, 1),
        emissive_strength: Float32 = 0.0,
        alpha_mode: Int = 0,
        double_sided: Bool = False,
    ):
        self.mesh = NativeMesh()
        self.mesh.base_color = base_color
        self.mesh.roughness = roughness
        self.mesh.metallic = metallic
        self.mesh.texture = texture
        self.mesh.emissive_color = emissive_color
        self.mesh.emissive_strength = emissive_strength
        self.mesh.alpha_mode = alpha_mode
        self.mesh.double_sided = double_sided

    def finish(deinit self) -> NativeMesh:
        return self.mesh^

    def vertex_pos(self, idx: Int) -> Vec3:
        return Vec3(
            self.mesh.positions[idx * 3],
            self.mesh.positions[idx * 3 + 1],
            self.mesh.positions[idx * 3 + 2],
        )

    def vertex_norm(self, idx: Int) -> Vec3:
        return Vec3(
            self.mesh.normals[idx * 3],
            self.mesh.normals[idx * 3 + 1],
            self.mesh.normals[idx * 3 + 2],
        )

    def add_vertex(
        mut self,
        pos: Vec3,
        normal: Vec3,
        u: Float32,
        v: Float32,
        color: UInt32 = 0xFFFFFFFF,
        ao: UInt8 = 255,
    ) -> Int:
        var idx = self.mesh.vertex_count()
        var n = normal.normalized()
        if n.length_squared() < 0.001:
            n = Vec3(0, 1, 0)
        self.mesh.positions.append(pos.x)
        self.mesh.positions.append(pos.y)
        self.mesh.positions.append(pos.z)
        self.mesh.normals.append(n.x)
        self.mesh.normals.append(n.y)
        self.mesh.normals.append(n.z)
        self.mesh.texcoords.append(u)
        self.mesh.texcoords.append(v)
        self.mesh.colors.append(UInt8((color >> 16) & 255))
        self.mesh.colors.append(UInt8((color >> 8) & 255))
        self.mesh.colors.append(UInt8(color & 255))
        self.mesh.colors.append(UInt8((color >> 24) & 255))
        self.mesh.ao.append(ao)
        return idx

    def add_tri(mut self, a: Int, b: Int, c: Int):
        """Append triangle oriented so (b - a) x (c - a) aligns with the vertices' outward normal."""
        var pa = self.vertex_pos(a)
        var pb = self.vertex_pos(b)
        var pc = self.vertex_pos(c)
        var face_n = (pb - pa).cross(pc - pa)
        if face_n.length_squared() < 1.0e-12:
            return
        var avg_n = self.vertex_norm(a) + self.vertex_norm(b) + self.vertex_norm(c)
        if face_n.dot(avg_n) >= 0:
            self.mesh.indices.append(a)
            self.mesh.indices.append(b)
            self.mesh.indices.append(c)
        else:
            self.mesh.indices.append(a)
            self.mesh.indices.append(c)
            self.mesh.indices.append(b)

    def add_quad(mut self, a: Int, b: Int, c: Int, d: Int):
        self.add_tri(a, b, c)
        self.add_tri(a, c, d)

    def add_ellipsoid(
        mut self,
        center: Vec3,
        radii: Vec3,
        rings: Int = 12,
        segments: Int = 20,
        u0: Float32 = 0.0,
        v0: Float32 = 0.0,
        u1: Float32 = 1.0,
        v1: Float32 = 1.0,
        color: UInt32 = 0xFFFFFFFF,
        theta_start: Float32 = 0.0,
        theta_end: Float32 = PI_F32,
        phi_start: Float32 = 0.0,
        phi_end: Float32 = TAU_F32,
    ):
        """Smooth UV-mapped ellipsoid (or dome/canopy slice via theta/phi bounds)."""
        var base_v = self.mesh.vertex_count()
        var rx = max(Float32(0.001), radii.x)
        var ry = max(Float32(0.001), radii.y)
        var rz = max(Float32(0.001), radii.z)
        for r in range(rings + 1):
            var tr = Float32(r) / Float32(rings)
            var theta = theta_start + (theta_end - theta_start) * tr
            var st = sin(theta)
            var ct = cos(theta)
            for s in range(segments + 1):
                var ts = Float32(s) / Float32(segments)
                var phi = phi_start + (phi_end - phi_start) * ts
                var cp = cos(phi)
                var sp = sin(phi)
                var unit = Vec3(st * cp, ct, st * sp)
                var pos = center + Vec3(unit.x * rx, unit.y * ry, unit.z * rz)
                var n = Vec3(unit.x / rx, unit.y / ry, unit.z / rz).normalized()
                var u = u0 + (u1 - u0) * ts
                var v = v0 + (v1 - v0) * tr
                _ = self.add_vertex(pos, n, u, v, color)
        var stride = segments + 1
        for r in range(rings):
            for s in range(segments):
                var i0 = base_v + r * stride + s
                var i1 = i0 + 1
                var i2 = i0 + stride + 1
                var i3 = i0 + stride
                self.add_quad(i0, i1, i2, i3)

    def add_lathe_z(
        mut self,
        z_stations: List[Float32],
        rx_stations: List[Float32],
        ry_stations: List[Float32],
        y_offsets: List[Float32],
        segments: Int = 20,
        offset: Vec3 = Vec3(),
        u0: Float32 = 0.0,
        v0: Float32 = 0.0,
        u1: Float32 = 1.0,
        v1: Float32 = 1.0,
        color: UInt32 = 0xFFFFFFFF,
        cap_start: Bool = True,
        cap_end: Bool = True,
    ):
        """Smooth lofted body along +Z with independent elliptical (rx, ry) and camber y_offset per station."""
        var count = len(z_stations)
        if count < 2:
            return
        var base_v = self.mesh.vertex_count()
        for i in range(count):
            var tv = Float32(i) / Float32(count - 1)
            var z = z_stations[i]
            var rx = rx_stations[i]
            var ry = ry_stations[i]
            var cy = y_offsets[i]
            var i_prev = max(0, i - 1)
            var i_next = min(count - 1, i + 1)
            var dz = max(Float32(0.0001), abs(z_stations[i_next] - z_stations[i_prev]))
            var drx = (rx_stations[i_next] - rx_stations[i_prev]) / dz
            var dry = (ry_stations[i_next] - ry_stations[i_prev]) / dz
            for s in range(segments + 1):
                var tu = Float32(s) / Float32(segments)
                var phi = tu * TAU_F32
                var cp = cos(phi)
                var sp = sin(phi)
                var pos = offset + Vec3(cp * rx, cy + sp * ry, z)
                var nx = cp / max(Float32(0.01), rx)
                var ny = sp / max(Float32(0.01), ry)
                var nz = -(cp * drx + sp * dry)
                var n = Vec3(nx, ny, nz).normalized()
                if rx < 0.002 and ry < 0.002:
                    n = Vec3(0, 0, Float32(-1) if i == 0 else Float32(1))
                _ = self.add_vertex(pos, n, u0 + (u1 - u0) * tu, v0 + (v1 - v0) * tv, color)

        var stride = segments + 1
        for i in range(count - 1):
            for s in range(segments):
                var i0 = base_v + i * stride + s
                var i1 = i0 + 1
                var i2 = i0 + stride + 1
                var i3 = i0 + stride
                self.add_quad(i0, i1, i2, i3)

        if cap_start and (rx_stations[0] > 0.003 or ry_stations[0] > 0.003):
            var c0 = self.add_vertex(
                offset + Vec3(0, y_offsets[0], z_stations[0]),
                Vec3(0, 0, -1),
                (u0 + u1) * 0.5,
                v0,
                color,
            )
            for s in range(segments):
                var p0 = self.add_vertex(
                    self.vertex_pos(base_v + s),
                    Vec3(0, 0, -1),
                    u0 + (u1 - u0) * (Float32(s) / Float32(segments)),
                    v0,
                    color,
                )
                var p1 = self.add_vertex(
                    self.vertex_pos(base_v + s + 1),
                    Vec3(0, 0, -1),
                    u0 + (u1 - u0) * (Float32(s + 1) / Float32(segments)),
                    v0,
                    color,
                )
                self.add_tri(c0, p1, p0)

        if cap_end and (rx_stations[count - 1] > 0.003 or ry_stations[count - 1] > 0.003):
            var last_ring = base_v + (count - 1) * stride
            var c1 = self.add_vertex(
                offset + Vec3(0, y_offsets[count - 1], z_stations[count - 1]),
                Vec3(0, 0, 1),
                (u0 + u1) * 0.5,
                v1,
                color,
            )
            for s in range(segments):
                var p0 = self.add_vertex(
                    self.vertex_pos(last_ring + s),
                    Vec3(0, 0, 1),
                    u0 + (u1 - u0) * (Float32(s) / Float32(segments)),
                    v1,
                    color,
                )
                var p1 = self.add_vertex(
                    self.vertex_pos(last_ring + s + 1),
                    Vec3(0, 0, 1),
                    u0 + (u1 - u0) * (Float32(s + 1) / Float32(segments)),
                    v1,
                    color,
                )
                self.add_tri(c1, p0, p1)

    def add_cylinder(
        mut self,
        p0: Vec3,
        p1: Vec3,
        r0: Float32,
        r1: Float32,
        segments: Int = 16,
        u0: Float32 = 0.0,
        v0: Float32 = 0.0,
        u1: Float32 = 1.0,
        v1: Float32 = 1.0,
        color: UInt32 = 0xFFFFFFFF,
        caps: Bool = True,
    ):
        """Arbitrary-axis smooth cylinder / truncated cone between p0 and p1."""
        var axis = p1 - p0
        var length = axis.length()
        if length < 0.0001:
            return
        var fwd = axis * (1.0 / length)
        var up_ref = Vec3(0, 1, 0) if abs(fwd.y) < 0.92 else Vec3(1, 0, 0)
        var right = up_ref.cross(fwd).normalized()
        var up = fwd.cross(right).normalized()
        var slope = (r0 - r1) / length

        var base_v = self.mesh.vertex_count()
        for ring in range(2):
            var t = Float32(ring)
            var center = p0 + axis * t
            var rad = r0 if ring == 0 else r1
            for s in range(segments + 1):
                var ts = Float32(s) / Float32(segments)
                var phi = ts * TAU_F32
                var radial = right * cos(phi) + up * sin(phi)
                var n = (radial + fwd * slope).normalized()
                if rad < 0.001:
                    n = fwd * (Float32(-1) if ring == 0 else Float32(1))
                _ = self.add_vertex(
                    center + radial * rad,
                    n,
                    u0 + (u1 - u0) * ts,
                    v0 + (v1 - v0) * t,
                    color,
                )

        var stride = segments + 1
        for s in range(segments):
            self.add_quad(base_v + s, base_v + s + 1, base_v + stride + s + 1, base_v + stride + s)

        if caps and r0 > 0.002:
            var c0 = self.add_vertex(p0, fwd * -1.0, (u0 + u1) * 0.5, v0, color)
            for s in range(segments):
                var ts0 = Float32(s) / Float32(segments)
                var ts1 = Float32(s + 1) / Float32(segments)
                var q0 = self.add_vertex(p0 + (right * cos(ts0 * TAU_F32) + up * sin(ts0 * TAU_F32)) * r0, fwd * -1.0, u0 + (u1 - u0) * ts0, v0, color)
                var q1 = self.add_vertex(p0 + (right * cos(ts1 * TAU_F32) + up * sin(ts1 * TAU_F32)) * r0, fwd * -1.0, u0 + (u1 - u0) * ts1, v0, color)
                self.add_tri(c0, q1, q0)

        if caps and r1 > 0.002:
            var c1 = self.add_vertex(p1, fwd, (u0 + u1) * 0.5, v1, color)
            for s in range(segments):
                var ts0 = Float32(s) / Float32(segments)
                var ts1 = Float32(s + 1) / Float32(segments)
                var q0 = self.add_vertex(p1 + (right * cos(ts0 * TAU_F32) + up * sin(ts0 * TAU_F32)) * r1, fwd, u0 + (u1 - u0) * ts0, v1, color)
                var q1 = self.add_vertex(p1 + (right * cos(ts1 * TAU_F32) + up * sin(ts1 * TAU_F32)) * r1, fwd, u0 + (u1 - u0) * ts1, v1, color)
                self.add_tri(c1, q0, q1)

    def add_airfoil(
        mut self,
        root_le: Vec3,
        root_chord: Float32,
        root_thickness: Float32,
        tip_le: Vec3,
        tip_chord: Float32,
        tip_thickness: Float32,
        up_dir: Vec3 = Vec3(0, 1, 0),
        chord_dir: Vec3 = Vec3(0, 0, -1),
        chord_steps: Int = 8,
        span_steps: Int = 4,
        u0: Float32 = 0.0,
        v0: Float32 = 0.0,
        u1: Float32 = 1.0,
        v1: Float32 = 1.0,
        color: UInt32 = 0xFFFFFFFF,
    ):
        """NACA-style symmetric curved airfoil wing/stabilizer from root leading edge to tip leading edge."""
        var up_n = up_dir.normalized()
        var ch_n = chord_dir.normalized()
        var span_vec = tip_le - root_le
        var span_n = span_vec.normalized()
        var profile_pts = chord_steps * 2
        var base_v = self.mesh.vertex_count()

        for sp in range(span_steps + 1):
            var ts = Float32(sp) / Float32(span_steps)
            var le = root_le + span_vec * ts
            var chord = root_chord + (tip_chord - root_chord) * ts
            var thick = root_thickness + (tip_thickness - root_thickness) * ts
            for cp_i in range(profile_pts + 1):
                var angle = Float32(cp_i) / Float32(profile_pts) * TAU_F32
                # Cosine clustering around leading edge (angle = PI -> xc = 0, angle = 0/TAU -> xc = 1)
                var xc = 0.5 * (1.0 + cos(angle))
                var side = Float32(1.0) if sin(angle) >= 0 else Float32(-1.0)
                if cp_i == 0 or cp_i == profile_pts:
                    side = 0.0
                # NACA 00xx half-thickness distribution
                var yt = 5.0 * thick * (
                    0.2969 * sqrt(max(Float32(0.0001), xc))
                    - 0.1260 * xc
                    - 0.3516 * xc * xc
                    + 0.2843 * xc * xc * xc
                    - 0.1015 * xc * xc * xc * xc
                )
                var pos = le + ch_n * (xc * chord) + up_n * (yt * side * sin(angle) if abs(sin(angle)) < 0.25 else yt * side)
                var n = (up_n * sin(angle) + ch_n * (-0.35 * cos(angle))).normalized()
                if xc < 0.03:
                    n = (ch_n * -1.0 + up_n * (0.4 * sin(angle))).normalized()
                var u = u0 + (u1 - u0) * ts
                var v = v0 + (v1 - v0) * xc
                _ = self.add_vertex(pos, n, u, v, color)

        var stride = profile_pts + 1
        for sp in range(span_steps):
            for cp_i in range(profile_pts):
                var i0 = base_v + sp * stride + cp_i
                var i1 = i0 + 1
                var i2 = i0 + stride + 1
                var i3 = i0 + stride
                self.add_quad(i0, i1, i2, i3)

        # Tip cap
        var tip_ring = base_v + span_steps * stride
        var tip_center = tip_le + ch_n * (tip_chord * 0.45)
        var tc_idx = self.add_vertex(tip_center, span_n, u1, (v0 + v1) * 0.5, color)
        for cp_i in range(profile_pts):
            var p0 = self.add_vertex(self.vertex_pos(tip_ring + cp_i), span_n, u1, v0, color)
            var p1 = self.add_vertex(self.vertex_pos(tip_ring + cp_i + 1), span_n, u1, v1, color)
            self.add_tri(tc_idx, p0, p1)

    def add_beveled_box(
        mut self,
        center: Vec3,
        half_extents: Vec3,
        bevel: Float32 = 0.08,
        rot: Quat = Quat(),
        u0: Float32 = 0.0,
        v0: Float32 = 0.0,
        u1: Float32 = 1.0,
        v1: Float32 = 1.0,
        color: UInt32 = 0xFFFFFFFF,
    ):
        """Chamfered 3D armor block with smooth-angled edge normals that catch specular highlights."""
        var mat = trs(center, rot, Vec3(1, 1, 1))
        var hx = max(Float32(0.005), half_extents.x)
        var hy = max(Float32(0.005), half_extents.y)
        var hz = max(Float32(0.005), half_extents.z)
        var b = min(bevel, min(hx, min(hy, hz)) * 0.45)
        var ix = hx - b
        var iy = hy - b
        var iz = hz - b

        # 4x4 grid around Y for 4 vertical levels -> chamfered prism (all 6 faces + 12 bevels + 8 corners!)
        var y_levels = List[Float32]()
        y_levels.append(-hy)
        y_levels.append(-iy)
        y_levels.append(iy)
        y_levels.append(hy)

        var s_scales = List[Float32]()
        s_scales.append(Float32(0.72))
        s_scales.append(Float32(1.0))
        s_scales.append(Float32(1.0))
        s_scales.append(Float32(0.72))

        # 8 perimeter points per ring (chamfered rectangle in XZ)
        var px_ring = List[Float32]()
        var pz_ring = List[Float32]()
        px_ring.append(ix);  pz_ring.append(hz)
        px_ring.append(-ix); pz_ring.append(hz)
        px_ring.append(-hx); pz_ring.append(iz)
        px_ring.append(-hx); pz_ring.append(-iz)
        px_ring.append(-ix); pz_ring.append(-hz)
        px_ring.append(ix);  pz_ring.append(-hz)
        px_ring.append(hx);  pz_ring.append(-iz)
        px_ring.append(hx);  pz_ring.append(iz)

        var base_v = self.mesh.vertex_count()
        for lv in range(4):
            var y = y_levels[lv]
            var sc = s_scales[lv]
            var tv = Float32(lv) / 3.0
            for k in range(9):
                var idx8 = k % 8
                var lx = px_ring[idx8] * sc
                var lz = pz_ring[idx8] * sc
                var local_p = Vec3(lx, y, lz)
                var local_n = Vec3(lx / hx, y / hy * (Float32(0.35) if lv == 1 or lv == 2 else Float32(1.1)), lz / hz).normalized()
                var world_p = mat.transform_point(local_p)
                var world_n = mat.transform_vector(local_n).normalized()
                var tu = Float32(k) / 8.0
                _ = self.add_vertex(world_p, world_n, u0 + (u1 - u0) * tu, v0 + (v1 - v0) * tv, color)

        for lv in range(3):
            for k in range(8):
                var i0 = base_v + lv * 9 + k
                self.add_quad(i0, i0 + 1, i0 + 10, i0 + 9)

        # Bottom and top caps
        var bot_c = self.add_vertex(
            mat.transform_point(Vec3(0, -hy, 0)),
            mat.transform_vector(Vec3(0, -1, 0)).normalized(),
            (u0 + u1) * 0.5,
            (v0 + v1) * 0.5,
            color,
        )
        for k in range(8):
            self.add_tri(bot_c, base_v + k + 1, base_v + k)

        var top_ring = base_v + 3 * 9
        var top_c = self.add_vertex(
            mat.transform_point(Vec3(0, hy, 0)),
            mat.transform_vector(Vec3(0, 1, 0)).normalized(),
            (u0 + u1) * 0.5,
            (v0 + v1) * 0.5,
            color,
        )
        for k in range(8):
            self.add_tri(top_c, top_ring + k, top_ring + k + 1)


def append_node_meshes(
    mut model: NativeModel,
    name: String,
    mut meshes: List[NativeMesh],
    translation: Vec3 = Vec3(),
    rotation: Quat = Quat(),
    scale: Vec3 = Vec3(1, 1, 1),
    parent: Int = -1,
) -> Int:
    """Add a named articulated node with 1..N multi-material NativeMesh primitives to a NativeModel."""
    var node = NativeNode()
    node.name = name
    node.translation = translation
    node.rotation = rotation
    node.scale = scale
    node.base_translation = translation
    node.base_rotation = rotation
    node.base_scale = scale
    for i in range(len(meshes)):
        var m_idx = len(model.meshes)
        model.meshes.append(meshes[i].copy())
        if i == 0:
            node.mesh = m_idx
        node.mesh_indices.append(m_idx)
    var node_idx = len(model.nodes)
    model.nodes.append(node^)
    if parent >= 0 and parent < node_idx:
        model.nodes[parent].children.append(node_idx)
    else:
        model.roots.append(node_idx)
    return node_idx
