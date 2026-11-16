"""Pure-Mojo GPU Meshlet Partitioner & Tangent-Space Geometry Cooker (`meshlet.mojo`).

Partitions `NativeMesh` geometry into cache-optimal **64-vertex / 124-triangle**
`MeshletCluster`s with tight bounding spheres and backface normal cones for GPU
compute culling (`frustum + normal-cone + Reverse-Z Hi-Z` -> `vkCmdDrawIndirectCount`),
while sharing the exact same vertex (`48`-byte stride) and `UInt32` triangle index
buffers with `VK_KHR_acceleration_structure` hardware `BLAS` ray tracing!
"""
from std.math import sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.assets.model import NativeMesh

comptime MESHLET_MAX_VERTS: Int = 64
comptime MESHLET_MAX_TRIS: Int = 124

# Vertex SSBO layout: 12 x UInt32 (48 bytes, 16-byte aligned; words 0..2 = float3 position for BLAS!)
# [0..2]: px, py, pz (Float32)
# [3..5]: nx, ny, nz (Float32)
# [6..8]: tx, ty, tz (Float32 tangent)
# [9..10]: u0, v0 (Float32 UV)
# [11]: bit0 = tangent_sign (0 -> +1.0, 1 -> -1.0), bits 8..15 = AO (0..255), bits 16..31 = packed tint
comptime AAA_VERTEX_WORDS: Int = 12
comptime AAA_VERTEX_BYTES: Int = 48

# Meshlet descriptor SSBO layout: 12 x UInt32 (48 bytes)
# [0..2]: center_x, center_y, center_z (Float32, mesh-local)
# [3]: radius (Float32, mesh-local)
# [4..6]: cone_axis_x, cone_axis_y, cone_axis_z (Float32, mesh-local unit axis)
# [7]: cone_sin (Float32: sin(half_angle); >= 2.0 disables backface cone cull)
# [8]: first_tri (UInt32: index in global triangle triplet table, i.e. global_index_offset / 3)
# [9]: tri_count (UInt32: 1..124)
# [10]: mesh_id (UInt32)
# [11]: material_id (UInt32)
comptime AAA_MESHLET_WORDS: Int = 12


struct MeshletCluster(Copyable, Movable):
    var center: Vec3
    var radius: Float32
    var cone_axis: Vec3
    var cone_sin: Float32
    var first_tri: Int
    var tri_count: Int
    var mesh_id: Int
    var material_id: Int

    def __init__(out self):
        self.center = Vec3()
        self.radius = Float32(0.0)
        self.cone_axis = Vec3(0.0, 0.0, 1.0)
        self.cone_sin = Float32(2.0)
        self.first_tri = 0
        self.tri_count = 0
        self.mesh_id = 0
        self.material_id = 0


struct CookedMeshEntry(Copyable, Movable):
    var mesh_id: Int
    var material_id: Int
    var first_vertex: Int
    var vertex_count: Int
    var first_index: Int
    var triangle_count: Int
    var first_meshlet: Int
    var meshlet_count: Int
    var bounds_center: Vec3
    var bounds_radius: Float32

    def __init__(out self):
        self.mesh_id = 0
        self.material_id = 0
        self.first_vertex = 0
        self.vertex_count = 0
        self.first_index = 0
        self.triangle_count = 0
        self.first_meshlet = 0
        self.meshlet_count = 0
        self.bounds_center = Vec3()
        self.bounds_radius = Float32(0.0)


struct AaaGeometryPool(Movable):
    """Unified CPU-side staging pool for GPU Vertex, Triangle Index & Meshlet SSBOs + BLAS inputs."""
    var vertex_words: List[UInt32]
    var index_words: List[UInt32]
    var meshlet_words: List[UInt32]
    var meshlets: List[MeshletCluster]
    var meshes: List[CookedMeshEntry]

    def __init__(out self):
        self.vertex_words = List[UInt32]()
        self.index_words = List[UInt32]()
        self.meshlet_words = List[UInt32]()
        self.meshlets = List[MeshletCluster]()
        self.meshes = List[CookedMeshEntry]()

    def total_vertices(self) -> Int:
        return len(self.vertex_words) // AAA_VERTEX_WORDS

    def total_triangles(self) -> Int:
        return len(self.index_words) // 3

    def total_meshlets(self) -> Int:
        return len(self.meshlets)

    def append_mesh(mut self, mesh: NativeMesh, material_id: Int) -> CookedMeshEntry:
        var entry = CookedMeshEntry()
        var v_count = len(mesh.positions) // 3
        var raw_tris = len(mesh.indices) // 3
        if v_count <= 0 or raw_tris <= 0:
            return entry^

        var mesh_id = len(self.meshes)
        var base_v = self.total_vertices()
        var base_idx = len(self.index_words)
        var base_ml = len(self.meshlets)

        # 1. Compute Lengyel MikkTSpace-style smooth per-vertex tangent frame `(T, sign)`
        var tan_u = List[Vec3](length=v_count, fill=Vec3(0.0, 0.0, 0.0))
        var tan_v = List[Vec3](length=v_count, fill=Vec3(0.0, 0.0, 0.0))
        var has_uv = len(mesh.texcoords) >= v_count * 2
        var has_nrm = len(mesh.normals) >= v_count * 3

        for ti in range(raw_tris):
            var i0 = mesh.indices[ti * 3 + 0]
            var i1 = mesh.indices[ti * 3 + 1]
            var i2 = mesh.indices[ti * 3 + 2]
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= v_count or i1 >= v_count or i2 >= v_count:
                continue
            var p0 = Vec3(mesh.positions[i0 * 3], mesh.positions[i0 * 3 + 1], mesh.positions[i0 * 3 + 2])
            var p1 = Vec3(mesh.positions[i1 * 3], mesh.positions[i1 * 3 + 1], mesh.positions[i1 * 3 + 2])
            var p2 = Vec3(mesh.positions[i2 * 3], mesh.positions[i2 * 3 + 1], mesh.positions[i2 * 3 + 2])
            var e1 = p1 - p0
            var e2 = p2 - p0
            var du1 = (mesh.texcoords[i1 * 2] - mesh.texcoords[i0 * 2]) if has_uv else Float32(1.0)
            var dv1 = (mesh.texcoords[i1 * 2 + 1] - mesh.texcoords[i0 * 2 + 1]) if has_uv else Float32(0.0)
            var du2 = (mesh.texcoords[i2 * 2] - mesh.texcoords[i0 * 2]) if has_uv else Float32(0.0)
            var dv2 = (mesh.texcoords[i2 * 2 + 1] - mesh.texcoords[i0 * 2 + 1]) if has_uv else Float32(1.0)
            var det = du1 * dv2 - du2 * dv1
            if abs(det) > Float32(1e-8):
                var inv_det = Float32(1.0) / det
                var sdir = (e1 * dv2 - e2 * dv1) * inv_det
                var tdir = (e2 * du1 - e1 * du2) * inv_det
                tan_u[i0] = tan_u[i0] + sdir; tan_u[i1] = tan_u[i1] + sdir; tan_u[i2] = tan_u[i2] + sdir
                tan_v[i0] = tan_v[i0] + tdir; tan_v[i1] = tan_v[i1] + tdir; tan_v[i2] = tan_v[i2] + tdir

        var bmin = Vec3(1e9, 1e9, 1e9)
        var bmax = Vec3(-1e9, -1e9, -1e9)

        for vi in range(v_count):
            var px = mesh.positions[vi * 3 + 0]
            var py = mesh.positions[vi * 3 + 1]
            var pz = mesh.positions[vi * 3 + 2]
            if px < bmin.x: bmin.x = px
            if py < bmin.y: bmin.y = py
            if pz < bmin.z: bmin.z = pz
            if px > bmax.x: bmax.x = px
            if py > bmax.y: bmax.y = py
            if pz > bmax.z: bmax.z = pz

            var n = Vec3(0.0, 1.0, 0.0)
            if has_nrm:
                n = Vec3(mesh.normals[vi * 3], mesh.normals[vi * 3 + 1], mesh.normals[vi * 3 + 2]).normalized()

            var t_raw = tan_u[vi] - n * n.dot(tan_u[vi])
            var t_len2 = t_raw.length_squared()
            var tang: Vec3
            var sign_neg: UInt32 = 0
            if t_len2 > Float32(1e-8):
                tang = t_raw * (Float32(1.0) / sqrt(t_len2))
                if n.cross(tang).dot(tan_v[vi]) < Float32(0.0):
                    sign_neg = 1
            else:
                var helper = Vec3(1.0, 0.0, 0.0) if abs(n.y) > Float32(0.9) else Vec3(0.0, 1.0, 0.0)
                tang = helper.cross(n).normalized()

            var u0 = mesh.texcoords[vi * 2 + 0] if has_uv else (px * Float32(0.5))
            var v0 = mesh.texcoords[vi * 2 + 1] if has_uv else (pz * Float32(0.5))
            var ao_byte = UInt32(mesh.ao[vi]) if vi < len(mesh.ao) else UInt32(255)
            var packed_meta = sign_neg | (ao_byte << 8) | UInt32(0xFFFF0000)

            self.vertex_words.append(bitcast[DType.uint32](px))
            self.vertex_words.append(bitcast[DType.uint32](py))
            self.vertex_words.append(bitcast[DType.uint32](pz))
            self.vertex_words.append(bitcast[DType.uint32](n.x))
            self.vertex_words.append(bitcast[DType.uint32](n.y))
            self.vertex_words.append(bitcast[DType.uint32](n.z))
            self.vertex_words.append(bitcast[DType.uint32](tang.x))
            self.vertex_words.append(bitcast[DType.uint32](tang.y))
            self.vertex_words.append(bitcast[DType.uint32](tang.z))
            self.vertex_words.append(bitcast[DType.uint32](u0))
            self.vertex_words.append(bitcast[DType.uint32](v0))
            self.vertex_words.append(packed_meta)

        # 2. Partition triangles into cache-coherent 64-vertex / 124-triangle Meshlets
        var cur_tris = List[Int]()
        var cur_verts = List[Int]()

        for ti in range(raw_tris):
            var a = mesh.indices[ti * 3 + 0]
            var b = mesh.indices[ti * 3 + 1]
            var c = mesh.indices[ti * 3 + 2]
            if a < 0 or b < 0 or c < 0 or a >= v_count or b >= v_count or c >= v_count:
                continue

            var new_v = 0
            var seen_a = False; var seen_b = False; var seen_c = False
            for cv in cur_verts:
                if cv == a: seen_a = True
                if cv == b: seen_b = True
                if cv == c: seen_c = True
            if not seen_a: new_v += 1
            if not seen_b and b != a: new_v += 1
            if not seen_c and c != a and c != b: new_v += 1

            if len(cur_tris) >= MESHLET_MAX_TRIS or (len(cur_verts) + new_v > MESHLET_MAX_VERTS and len(cur_tris) > 0):
                self._emit_meshlet(mesh, base_v, mesh_id, material_id, mesh.double_sided, cur_tris)
                cur_tris.clear()
                cur_verts.clear()
                seen_a = False; seen_b = False; seen_c = False

            if not seen_a: cur_verts.append(a)
            if not seen_b and b != a: cur_verts.append(b)
            if not seen_c and c != a and c != b: cur_verts.append(c)
            cur_tris.append(ti)

        if len(cur_tris) > 0:
            self._emit_meshlet(mesh, base_v, mesh_id, material_id, mesh.double_sided, cur_tris)

        var mesh_c = (bmin + bmax) * Float32(0.5)
        var max_r2 = Float32(0.0001)
        for vi in range(v_count):
            var vp = Vec3(mesh.positions[vi * 3], mesh.positions[vi * 3 + 1], mesh.positions[vi * 3 + 2])
            var d2 = (vp - mesh_c).length_squared()
            if d2 > max_r2:
                max_r2 = d2

        entry.mesh_id = mesh_id
        entry.material_id = material_id
        entry.first_vertex = base_v
        entry.vertex_count = v_count
        entry.first_index = base_idx
        entry.triangle_count = (len(self.index_words) - base_idx) // 3
        entry.first_meshlet = base_ml
        entry.meshlet_count = len(self.meshlets) - base_ml
        entry.bounds_center = mesh_c
        entry.bounds_radius = sqrt(max_r2)
        self.meshes.append(entry.copy())
        return entry^

    def _emit_meshlet(
        mut self,
        mesh: NativeMesh,
        base_v: Int,
        mesh_id: Int,
        material_id: Int,
        double_sided: Bool,
        tri_list: List[Int],
    ):
        var first_tri_global = len(self.index_words) // 3
        var min_p = Vec3(1e9, 1e9, 1e9)
        var max_p = Vec3(-1e9, -1e9, -1e9)
        var tri_normals = List[Vec3](capacity=len(tri_list))
        var n_sum = Vec3(0.0, 0.0, 0.0)

        for ti in tri_list:
            var i0 = mesh.indices[ti * 3 + 0]
            var i1 = mesh.indices[ti * 3 + 1]
            var i2 = mesh.indices[ti * 3 + 2]
            self.index_words.append(UInt32(base_v + i0))
            self.index_words.append(UInt32(base_v + i1))
            self.index_words.append(UInt32(base_v + i2))

            var p0 = Vec3(mesh.positions[i0 * 3], mesh.positions[i0 * 3 + 1], mesh.positions[i0 * 3 + 2])
            var p1 = Vec3(mesh.positions[i1 * 3], mesh.positions[i1 * 3 + 1], mesh.positions[i1 * 3 + 2])
            var p2 = Vec3(mesh.positions[i2 * 3], mesh.positions[i2 * 3 + 1], mesh.positions[i2 * 3 + 2])
            for p in [p0, p1, p2]:
                if p.x < min_p.x: min_p.x = p.x
                if p.y < min_p.y: min_p.y = p.y
                if p.z < min_p.z: min_p.z = p.z
                if p.x > max_p.x: max_p.x = p.x
                if p.y > max_p.y: max_p.y = p.y
                if p.z > max_p.z: max_p.z = p.z

            var fn_raw = (p1 - p0).cross(p2 - p0)
            if fn_raw.length_squared() > Float32(1e-12):
                var fn_u = fn_raw.normalized()
                tri_normals.append(fn_u)
                n_sum = n_sum + fn_u

        var center = (min_p + max_p) * Float32(0.5)
        var max_r2 = Float32(1e-6)
        for ti in tri_list:
            for k in range(3):
                var iv = mesh.indices[ti * 3 + k]
                var p = Vec3(mesh.positions[iv * 3], mesh.positions[iv * 3 + 1], mesh.positions[iv * 3 + 2])
                var r2 = (p - center).length_squared()
                if r2 > max_r2:
                    max_r2 = r2
        var radius = sqrt(max_r2)

        var cone_axis = Vec3(0.0, 0.0, 1.0)
        var cone_sin = Float32(2.0)  # >= 2.0 means omnidirectional / double-sided (disable cone cull)
        if not double_sided and len(tri_normals) > 0 and n_sum.length_squared() > Float32(1e-6):
            cone_axis = n_sum.normalized()
            var min_dot = Float32(1.0)
            for tn in tri_normals:
                var d = cone_axis.dot(tn)
                if d < min_dot:
                    min_dot = d
            if min_dot > Float32(0.08):
                cone_sin = sqrt(max(Float32(0.0), Float32(1.0) - min_dot * min_dot))

        var ml = MeshletCluster()
        ml.center = center
        ml.radius = radius
        ml.cone_axis = cone_axis
        ml.cone_sin = cone_sin
        ml.first_tri = first_tri_global
        ml.tri_count = len(tri_list)
        ml.mesh_id = mesh_id
        ml.material_id = material_id
        self.meshlets.append(ml.copy())

        self.meshlet_words.append(bitcast[DType.uint32](center.x))
        self.meshlet_words.append(bitcast[DType.uint32](center.y))
        self.meshlet_words.append(bitcast[DType.uint32](center.z))
        self.meshlet_words.append(bitcast[DType.uint32](radius))
        self.meshlet_words.append(bitcast[DType.uint32](cone_axis.x))
        self.meshlet_words.append(bitcast[DType.uint32](cone_axis.y))
        self.meshlet_words.append(bitcast[DType.uint32](cone_axis.z))
        self.meshlet_words.append(bitcast[DType.uint32](cone_sin))
        self.meshlet_words.append(UInt32(first_tri_global))
        self.meshlet_words.append(UInt32(len(tri_list)))
        self.meshlet_words.append(UInt32(mesh_id))
        self.meshlet_words.append(UInt32(material_id))


def cull_meshlet_cpu_reference(
    ml: MeshletCluster,
    cam_pos: Vec3,
    cam_fwd: Vec3,
    cam_right: Vec3,
    cam_up: Vec3,
    tan_half_fov_x: Float32,
    tan_half_fov_y: Float32,
    near_z: Float32,
) -> Bool:
    """Exact CPU reference for GPU Meshlet Frustum + Normal-Cone Backface Culling."""
    var rel = ml.center - cam_pos
    var dist2 = rel.length_squared()
    var dist = sqrt(max(Float32(1e-8), dist2))

    # 1. Spherical backface cone cull: dot(C - E, axis) >= cone_sin * |C - E| + R
    if ml.cone_sin < Float32(1.5):
        if rel.dot(ml.cone_axis) >= ml.cone_sin * dist + ml.radius:
            return False

    # 2. View-frustum sphere test
    var vz = rel.dot(cam_fwd)
    if vz + ml.radius < near_z:
        return False
    var vx = abs(rel.dot(cam_right))
    var vy = abs(rel.dot(cam_up))
    var sec_x = sqrt(Float32(1.0) + tan_half_fov_x * tan_half_fov_x)
    var sec_y = sqrt(Float32(1.0) + tan_half_fov_y * tan_half_fov_y)
    if vx - ml.radius * sec_x > max(Float32(0.0), vz) * tan_half_fov_x:
        return False
    if vy - ml.radius * sec_y > max(Float32(0.0), vz) * tan_half_fov_y:
        return False
    return True
