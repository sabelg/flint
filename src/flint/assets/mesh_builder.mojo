"""Parametric curved 3D mesh builders for NativeMesh (lathes, lofts, airfoils, ellipsoids, capsules, beveled hulls)."""
from std.math import sin, cos, sqrt, isfinite
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
        if rings < 1 or segments < 3:
            return
        if not (
            isfinite(center.x) and isfinite(center.y) and isfinite(center.z)
            and isfinite(theta_start) and isfinite(theta_end)
            and isfinite(phi_start) and isfinite(phi_end)
        ):
            return
        var base_v = self.mesh.vertex_count()
        var rx = max(Float32(0.001), radii.x if isfinite(radii.x) else Float32(0.001))
        var ry = max(Float32(0.001), radii.y if isfinite(radii.y) else Float32(0.001))
        var rz = max(Float32(0.001), radii.z if isfinite(radii.z) else Float32(0.001))
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
        var has_north_pole = theta_start <= 0.0005
        var has_south_pole = theta_end >= PI_F32 - 0.0005
        for r in range(rings):
            for s in range(segments):
                var i0 = base_v + r * stride + s
                var i1 = i0 + 1
                var i2 = i0 + stride + 1
                var i3 = i0 + stride
                if r == 0 and has_north_pole:
                    self.add_tri(i0, i2, i3)
                elif r == rings - 1 and has_south_pole:
                    self.add_tri(i0, i1, i3)
                else:
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
        if (
            count < 2
            or len(rx_stations) != count
            or len(ry_stations) != count
            or len(y_offsets) != count
            or segments < 3
        ):
            return
        if not (
            isfinite(offset.x) and isfinite(offset.y) and isfinite(offset.z)
            and isfinite(u0) and isfinite(v0) and isfinite(u1) and isfinite(v1)
        ):
            return
        for i in range(count):
            if not (
                isfinite(z_stations[i])
                and isfinite(rx_stations[i])
                and isfinite(ry_stations[i])
                and isfinite(y_offsets[i])
            ):
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
        if segments < 3:
            return
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
        if chord_steps < 1 or span_steps < 1:
            return
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

        # Root and tip caps with clean planar chordwise UVs
        var root_center = root_le + ch_n * (root_chord * 0.45)
        var rc_idx = self.add_vertex(root_center, span_n * -1.0, u0, (v0 + v1) * 0.5, color)
        for cp_i in range(profile_pts):
            var a0 = Float32(cp_i) / Float32(profile_pts) * TAU_F32
            var a1 = Float32(cp_i + 1) / Float32(profile_pts) * TAU_F32
            var vc0 = v0 + (v1 - v0) * (0.5 * (1.0 + cos(a0)))
            var vc1 = v0 + (v1 - v0) * (0.5 * (1.0 + cos(a1)))
            var p0 = self.add_vertex(self.vertex_pos(base_v + cp_i), span_n * -1.0, u0, vc0, color)
            var p1 = self.add_vertex(self.vertex_pos(base_v + cp_i + 1), span_n * -1.0, u0, vc1, color)
            self.add_tri(rc_idx, p1, p0)

        var tip_ring = base_v + span_steps * stride
        var tip_center = tip_le + ch_n * (tip_chord * 0.45)
        var tc_idx = self.add_vertex(tip_center, span_n, u1, (v0 + v1) * 0.5, color)
        for cp_i in range(profile_pts):
            var a0 = Float32(cp_i) / Float32(profile_pts) * TAU_F32
            var a1 = Float32(cp_i + 1) / Float32(profile_pts) * TAU_F32
            var vc0 = v0 + (v1 - v0) * (0.5 * (1.0 + cos(a0)))
            var vc1 = v0 + (v1 - v0) * (0.5 * (1.0 + cos(a1)))
            var p0 = self.add_vertex(self.vertex_pos(tip_ring + cp_i), span_n, u1, vc0, color)
            var p1 = self.add_vertex(self.vertex_pos(tip_ring + cp_i + 1), span_n, u1, vc1, color)
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
        """Watertight 24-vertex hard-surface chamfered prism: optically flat faces + Phong-rounded specular bevels."""
        var mat = trs(center, rot, Vec3(1, 1, 1))
        var hx = max(Float32(0.005), half_extents.x)
        var hy = max(Float32(0.005), half_extents.y)
        var hz = max(Float32(0.005), half_extents.z)
        var b = min(max(Float32(0.0005), bevel), min(hx, min(hy, hz)) * Float32(0.38))
        var ix = hx - b
        var iy = hy - b
        var iz = hz - b

        # Index map: octant (sx_i, sy_i, sz_i) in {0,1}^3 -> idx8 = sx_i*4 + sy_i*2 + sz_i
        var vx = List[Int](capacity=8)
        var vy = List[Int](capacity=8)
        var vz = List[Int](capacity=8)
        for sx_i in range(2):
            var sx = Float32(1.0) if sx_i == 1 else Float32(-1.0)
            for sy_i in range(2):
                var sy = Float32(1.0) if sy_i == 1 else Float32(-1.0)
                for sz_i in range(2):
                    var sz = Float32(1.0) if sz_i == 1 else Float32(-1.0)
                    var ux = u0 + (u1 - u0) * (Float32(1 - sz_i) if sx_i == 1 else Float32(sz_i))
                    var vx_uv = v0 + (v1 - v0) * Float32(1 - sy_i)
                    vx.append(
                        self.add_vertex(
                            mat.transform_point(Vec3(sx * hx, sy * iy, sz * iz)),
                            mat.transform_vector(Vec3(sx, 0.0, 0.0)).normalized(),
                            ux,
                            vx_uv,
                            color,
                        )
                    )
                    var uy = u0 + (u1 - u0) * Float32(sx_i)
                    var vy_uv = v0 + (v1 - v0) * (Float32(1 - sz_i) if sy_i == 1 else Float32(sz_i))
                    vy.append(
                        self.add_vertex(
                            mat.transform_point(Vec3(sx * ix, sy * hy, sz * iz)),
                            mat.transform_vector(Vec3(0.0, sy, 0.0)).normalized(),
                            uy,
                            vy_uv,
                            color,
                        )
                    )
                    var uz = u0 + (u1 - u0) * (Float32(sx_i) if sz_i == 1 else Float32(1 - sx_i))
                    var vz_uv = v0 + (v1 - v0) * Float32(1 - sy_i)
                    vz.append(
                        self.add_vertex(
                            mat.transform_point(Vec3(sx * ix, sy * iy, sz * hz)),
                            mat.transform_vector(Vec3(0.0, 0.0, sz)).normalized(),
                            uz,
                            vz_uv,
                            color,
                        )
                    )

        # 6 flat orthogonal armor faces (idx = sx_i*4 + sy_i*2 + sz_i)
        self.add_quad(vx[4], vx[5], vx[7], vx[6])  # +X
        self.add_quad(vx[1], vx[0], vx[2], vx[3])  # -X
        self.add_quad(vy[2], vy[3], vy[7], vy[6])  # +Y
        self.add_quad(vy[0], vy[4], vy[5], vy[1])  # -Y
        self.add_quad(vz[1], vz[5], vz[7], vz[3])  # +Z
        self.add_quad(vz[4], vz[0], vz[2], vz[6])  # -Z

        # 12 smooth-interpolated specular edge chamfers
        # 4 Y-parallel edges (sx_i, sz_i)
        for sx_i in range(2):
            for sz_i in range(2):
                var lo = sx_i * 4 + 0 * 2 + sz_i
                var hi = sx_i * 4 + 1 * 2 + sz_i
                self.add_quad(vx[lo], vz[lo], vz[hi], vx[hi])
        # 4 X-parallel edges (sy_i, sz_i)
        for sy_i in range(2):
            for sz_i in range(2):
                var lo = 0 * 4 + sy_i * 2 + sz_i
                var hi = 1 * 4 + sy_i * 2 + sz_i
                self.add_quad(vy[lo], vz[lo], vz[hi], vy[hi])
        # 4 Z-parallel edges (sx_i, sy_i)
        for sx_i in range(2):
            for sy_i in range(2):
                var lo = sx_i * 4 + sy_i * 2 + 0
                var hi = sx_i * 4 + sy_i * 2 + 1
                self.add_quad(vx[lo], vy[lo], vy[hi], vx[hi])

        # 8 smooth-interpolated corner bevel triangles
        for oct in range(8):
            self.add_tri(vx[oct], vy[oct], vz[oct])

    def add_tapered_hull(
        mut self,
        p0: Vec3,
        hx0: Float32,
        hy0: Float32,
        p1: Vec3,
        hx1: Float32,
        hy1: Float32,
        bevel: Float32 = 0.04,
        rot: Quat = Quat(),
        u0: Float32 = 0.0,
        v0: Float32 = 0.0,
        u1: Float32 = 1.0,
        v1: Float32 = 1.0,
        color: UInt32 = 0xFFFFFFFF,
        caps: Bool = True,
    ):
        """Chamfered 8-sided tapered prism from local p0 (half-extents hx0, hy0) to p1 (hx1, hy1)."""
        var mat = trs(Vec3(), rot, Vec3(1, 1, 1))
        var axis = p1 - p0
        var length = axis.length()
        if length < 0.0005:
            return
        var fwd = axis * (1.0 / length)
        var up_ref = Vec3(0, 1, 0) if abs(fwd.y) < 0.94 else Vec3(0, 0, 1)
        var right = up_ref.cross(fwd).normalized()
        var up = fwd.cross(right).normalized()

        var base_v = self.mesh.vertex_count()
        for end_i in range(2):
            var t = Float32(end_i)
            var ctr = p0 + axis * t
            var hx = max(Float32(0.004), hx0 if end_i == 0 else hx1)
            var hy = max(Float32(0.004), hy0 if end_i == 0 else hy1)
            var b = min(max(Float32(0.0005), bevel), min(hx, hy) * Float32(0.35))
            var ix = hx - b
            var iy = hy - b
            var rx = [ix, -ix, -hx, -hx, -ix, ix, hx, hx]
            var ry = [hy, hy, iy, -iy, -hy, -hy, -iy, iy]
            var nx = [Float32(0.0), Float32(0.0), Float32(-1.0), Float32(-1.0), Float32(0.0), Float32(0.0), Float32(1.0), Float32(1.0)]
            var ny = [Float32(1.0), Float32(1.0), Float32(0.0), Float32(0.0), Float32(-1.0), Float32(-1.0), Float32(0.0), Float32(0.0)]
            var perim = Float32(4.0) * (ix + iy + b * Float32(1.41421356))
            var acc = Float32(0.0)
            for k in range(9):
                var idx8 = k % 8
                if k > 0:
                    var prev8 = (k - 1) % 8
                    var dx = rx[idx8] - rx[prev8]
                    var dy = ry[idx8] - ry[prev8]
                    acc += sqrt(dx * dx + dy * dy)
                var lp = ctr + right * rx[idx8] + up * ry[idx8]
                var ln = (right * nx[idx8] + up * ny[idx8]).normalized()
                var tu = acc / max(Float32(0.001), perim)
                _ = self.add_vertex(
                    mat.transform_point(lp),
                    mat.transform_vector(ln).normalized(),
                    u0 + (u1 - u0) * tu,
                    v0 + (v1 - v0) * t,
                    color,
                )

        for k in range(8):
            var i0 = base_v + k
            self.add_quad(i0, i0 + 1, i0 + 10, i0 + 9)

        if caps:
            for end_i in range(2):
                var ctr = p0 if end_i == 0 else p1
                var cn = mat.transform_vector(fwd * (Float32(-1.0) if end_i == 0 else Float32(1.0))).normalized()
                var c_idx = self.add_vertex(mat.transform_point(ctr), cn, (u0 + u1) * 0.5, (v0 + v1) * 0.5, color)
                var ring_base = base_v + end_i * 9
                for k in range(8):
                    var p_a = self.vertex_pos(ring_base + k)
                    var p_b = self.vertex_pos(ring_base + k + 1)
                    var q_a = self.add_vertex(p_a, cn, (u0 + u1) * 0.5, v0 if end_i == 0 else v1, color)
                    var q_b = self.add_vertex(p_b, cn, (u0 + u1) * 0.5, v0 if end_i == 0 else v1, color)
                    if end_i == 0:
                        self.add_tri(c_idx, q_b, q_a)
                    else:
                        self.add_tri(c_idx, q_a, q_b)


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
