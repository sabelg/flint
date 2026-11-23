"""Flint v5 Colliders & 6-DOF Motor Joint (`flint.physics.shapes_v5`).

Implements:
1. `CompoundChild` & `CompoundShape`: multi-part rigid compound colliders with exact
   parallel-axis theorem inertia (`compute_mass_properties`) and volumetric + plane-clip
   approximate convex decomposition (`decompose_vhacd_lite`).
2. `HeightfieldCollider`: quantized `UInt16` heightmap with 4-level hierarchical min/max
   chunk bounds, exact Amanatides-Woo 2D DDA `raycast_dda`, `sweep_sphere`, and smooth
   4-point contact manifold generation (`collide_sphere`, `collide_capsule`, `collide_box`).
3. `SixDofMotorJoint`: full 6-DOF constraint with 3 linear limits, 3 swing/twist angular
   limits, and soft PD linear/angular drives (`make_soft`), warm-starting, and serialization.
"""
from std.collections import List
from std.math import sqrt, abs, min, max, floor, isfinite
from flint.core.math import Vec3, Quat
from flint.core.detmath import PI, det_atan2
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.physics.solver_bodies import Sym3, SolverBodies, rotate_vec
from flint.physics.narrowphase import (
    ContactPoint,
    ContactManifold,
    ConvexHull,
    reduce_manifold_candidates,
    collide_sphere_sphere,
    collide_sphere_capsule,
    collide_sphere_box,
    collide_capsule_box,
    collide_convex_convex,
    get_box_axes,
    closest_points_segments,
)
from flint.physics.bvh import (
    MeshTriangle,
    RayHit,
    SweepHit,
    Bvh8MeshCollider,
    EDGE_ACTIVE_ALL,
    EDGE_ACTIVE_01,
    EDGE_ACTIVE_12,
    EDGE_ACTIVE_20,
)
from flint.physics.joints import (
    Softness,
    make_soft,
    quat_conj,
    quat_product,
    rotation_error,
    clamp_length,
    JOINT_WORLD,
)

comptime COMPOUND_CHILD_SPHERE: Int = 0
comptime COMPOUND_CHILD_CAPSULE: Int = 1
comptime COMPOUND_CHILD_BOX: Int = 2
comptime COMPOUND_CHILD_CONVEX: Int = 3

comptime COMPOUND_BLOB_TAG: UInt32 = fourcc("CMPD")
comptime SIXDOF_BLOB_TAG: UInt32 = fourcc("J6DF")
comptime SHAPES_V5_BLOB_VERSION: Int = 1


@always_inline
def _rotate_sym3(rot: Quat, m: Sym3) -> Sym3:
    """Computes `R * M * R^T` for symmetric 3x3 inertia tensor `M`."""
    var r0 = rotate_vec(rot, Vec3(1, 0, 0))
    var r1 = rotate_vec(rot, Vec3(0, 1, 0))
    var r2 = rotate_vec(rot, Vec3(0, 0, 1))
    # Rows of R in world space: row x = (r0.x, r1.x, r2.x), etc.
    var rx = Vec3(r0.x, r1.x, r2.x)
    var ry = Vec3(r0.y, r1.y, r2.y)
    var rz = Vec3(r0.z, r1.z, r2.z)
    var mr_x = m.apply(rx)
    var mr_y = m.apply(ry)
    var mr_z = m.apply(rz)
    return Sym3(
        rx.dot(mr_x),
        ry.dot(mr_y),
        rz.dot(mr_z),
        rx.dot(mr_y),
        rx.dot(mr_z),
        ry.dot(mr_z),
    )


# =========================================================================
# 1. CompoundChild & CompoundShape (with V-HACD-lite Decomposition)
# =========================================================================

struct CompoundChild(Copyable, Movable):
    var shape: Int
    var local_pos: Vec3
    var local_rot: Quat
    var radius: Float32
    var half_height: Float32
    var half_extents: Vec3
    var hull: ConvexHull
    var density_scale: Float32

    def __init__(out self):
        self.shape = COMPOUND_CHILD_BOX
        self.local_pos = Vec3()
        self.local_rot = Quat()
        self.radius = 0.5
        self.half_height = 0.5
        self.half_extents = Vec3(0.5, 0.5, 0.5)
        self.hull = ConvexHull()
        self.density_scale = 1.0

    @staticmethod
    def sphere(local_pos: Vec3, radius: Float32, density_scale: Float32 = 1.0) -> Self:
        var c = Self()
        c.shape = COMPOUND_CHILD_SPHERE
        c.local_pos = local_pos
        c.radius = max(Float32(0.005), radius)
        c.half_extents = Vec3(c.radius, c.radius, c.radius)
        c.density_scale = density_scale
        return c^

    @staticmethod
    def capsule(
        local_pos: Vec3,
        local_rot: Quat,
        radius: Float32,
        half_height: Float32,
        density_scale: Float32 = 1.0,
    ) -> Self:
        var c = Self()
        c.shape = COMPOUND_CHILD_CAPSULE
        c.local_pos = local_pos
        c.local_rot = local_rot.normalized()
        c.radius = max(Float32(0.005), radius)
        c.half_height = max(Float32(0.0), half_height)
        c.half_extents = Vec3(c.radius, c.half_height + c.radius, c.radius)
        c.density_scale = density_scale
        return c^

    @staticmethod
    def box(
        local_pos: Vec3,
        local_rot: Quat,
        half_extents: Vec3,
        density_scale: Float32 = 1.0,
    ) -> Self:
        var c = Self()
        c.shape = COMPOUND_CHILD_BOX
        c.local_pos = local_pos
        c.local_rot = local_rot.normalized()
        c.half_extents = Vec3(
            max(Float32(0.005), half_extents.x),
            max(Float32(0.005), half_extents.y),
            max(Float32(0.005), half_extents.z),
        )
        c.radius = c.half_extents.length()
        c.density_scale = density_scale
        return c^

    @staticmethod
    def convex(
        local_pos: Vec3,
        local_rot: Quat,
        hull: ConvexHull,
        density_scale: Float32 = 1.0,
    ) -> Self:
        var c = Self()
        c.shape = COMPOUND_CHILD_CONVEX
        c.local_pos = local_pos
        c.local_rot = local_rot.normalized()
        c.hull = hull.copy()
        var mx = Float32(0.01)
        var my = Float32(0.01)
        var mz = Float32(0.01)
        for i in range(len(hull.vertices)):
            var v = hull.vertices[i]
            mx = max(mx, abs(v.x))
            my = max(my, abs(v.y))
            mz = max(mz, abs(v.z))
        c.half_extents = Vec3(mx, my, mz)
        c.radius = c.half_extents.length()
        c.density_scale = density_scale
        return c^

    def world_pose(self, parent_pos: Vec3, parent_rot: Quat) -> Tuple[Vec3, Quat]:
        var wp = parent_pos + parent_rot.rotate(self.local_pos)
        var wr = (parent_rot * self.local_rot).normalized()
        return (wp, wr)

    def tight_aabb(self, parent_pos: Vec3, parent_rot: Quat) -> Tuple[Vec3, Vec3]:
        var wp_wr = self.world_pose(parent_pos, parent_rot)
        var wp = wp_wr[0]
        var wr = wp_wr[1]
        if self.shape == COMPOUND_CHILD_SPHERE:
            var rv = Vec3(self.radius, self.radius, self.radius)
            return (wp - rv, wp + rv)
        if self.shape == COMPOUND_CHILD_CAPSULE:
            var ax = wr.rotate(Vec3(0, self.half_height, 0))
            var a = wp - ax
            var b = wp + ax
            var rv = Vec3(self.radius, self.radius, self.radius)
            return (
                Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)) - rv,
                Vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)) + rv,
            )
        if self.shape == COMPOUND_CHILD_CONVEX and len(self.hull.vertices) > 0:
            var lo = Vec3(1.0e30, 1.0e30, 1.0e30)
            var hi = Vec3(-1.0e30, -1.0e30, -1.0e30)
            for i in range(len(self.hull.vertices)):
                var p = wp + wr.rotate(self.hull.vertices[i])
                lo.x = min(lo.x, p.x); lo.y = min(lo.y, p.y); lo.z = min(lo.z, p.z)
                hi.x = max(hi.x, p.x); hi.y = max(hi.y, p.y); hi.z = max(hi.z, p.z)
            return (lo, hi)
        var axes = get_box_axes(wr)
        var wx = abs(axes[0].x) * self.half_extents.x + abs(axes[1].x) * self.half_extents.y + abs(axes[2].x) * self.half_extents.z
        var wy = abs(axes[0].y) * self.half_extents.x + abs(axes[1].y) * self.half_extents.y + abs(axes[2].y) * self.half_extents.z
        var wz = abs(axes[0].z) * self.half_extents.x + abs(axes[1].z) * self.half_extents.y + abs(axes[2].z) * self.half_extents.z
        var ext = Vec3(wx, wy, wz)
        return (wp - ext, wp + ext)

    def compute_local_mass_properties(self, density: Float32) -> Tuple[Float32, Vec3, Sym3]:
        """Returns `(mass, com_in_compound_space, inertia_about_child_com_in_compound_space)`."""
        var rho = max(Float32(1.0e-6), density * self.density_scale)
        if self.shape == COMPOUND_CHILD_SPHERE:
            var r = self.radius
            var vol = Float32(4.0 / 3.0) * PI * r * r * r
            var m = rho * vol
            var i_s = Float32(0.4) * m * r * r
            return (m, self.local_pos, Sym3(i_s, i_s, i_s))
        if self.shape == COMPOUND_CHILD_CAPSULE:
            var r = self.radius
            var l = Float32(2.0) * self.half_height
            var r2 = r * r
            var v_cyl = PI * r2 * l
            var v_sph = Float32(4.0 / 3.0) * PI * r2 * r
            var m_cyl = rho * v_cyl
            var m_sph = rho * v_sph
            var m = m_cyl + m_sph
            var iy = Float32(0.5) * m_cyl * r2 + Float32(0.4) * m_sph * r2
            var i_perp = (
                m_cyl * (l * l * Float32(1.0 / 12.0) + Float32(0.25) * r2)
                + m_sph * (Float32(0.4) * r2 + Float32(0.25) * l * l + Float32(0.375) * r * l)
            )
            return (m, self.local_pos, _rotate_sym3(self.local_rot, Sym3(i_perp, iy, i_perp)))
        if self.shape == COMPOUND_CHILD_CONVEX and len(self.hull.vertices) >= 4:
            var mp = self.hull.compute_polyhedral_mass_properties(rho)
            var world_com = self.local_pos + self.local_rot.rotate(mp[1])
            return (mp[0], world_com, _rotate_sym3(self.local_rot, mp[2]))

        # Box
        var hx = self.half_extents.x
        var hy = self.half_extents.y
        var hz = self.half_extents.z
        var vol = Float32(8.0) * hx * hy * hz
        var m = rho * vol
        var sx2 = Float32(4.0) * hx * hx
        var sy2 = Float32(4.0) * hy * hy
        var sz2 = Float32(4.0) * hz * hz
        var k = m * Float32(1.0 / 12.0)
        var ibox = Sym3(k * (sy2 + sz2), k * (sx2 + sz2), k * (sx2 + sy2))
        return (m, self.local_pos, _rotate_sym3(self.local_rot, ibox))


struct _VhacdPart(Copyable, Movable):
    var tri_a: List[Vec3]
    var tri_b: List[Vec3]
    var tri_c: List[Vec3]
    var vox_x: List[Int]
    var vox_y: List[Int]
    var vox_z: List[Int]
    var hull: ConvexHull
    var hull_vol: Float32
    var voxel_vol: Float32

    def __init__(out self):
        self.tri_a = List[Vec3]()
        self.tri_b = List[Vec3]()
        self.tri_c = List[Vec3]()
        self.vox_x = List[Int]()
        self.vox_y = List[Int]()
        self.vox_z = List[Int]()
        self.hull = ConvexHull()
        self.hull_vol = 0.0
        self.voxel_vol = 0.0

    def concavity(self) -> Float32:
        return max(Float32(0.0), self.hull_vol - self.voxel_vol)


def _clip_tri_halfspace(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    axis: Int,
    split_coord: Float32,
    keep_upper: Bool,
    mut out_a: List[Vec3],
    mut out_b: List[Vec3],
    mut out_c: List[Vec3],
):
    """Clips triangle `(a, b, c)` against axis plane `p[axis] <= split_coord` (`keep_upper = False`)
    or `p[axis] >= split_coord` (`keep_upper = True`), emitting 0, 1, or 2 sub-triangles."""
    var poly = [a, b, c]
    var clipped = List[Vec3](capacity=4)
    for i in range(3):
        var p0 = poly[i]
        var p1 = poly[(i + 1) % 3]
        var c0 = p0.x if axis == 0 else (p0.y if axis == 1 else p0.z)
        var c1 = p1.x if axis == 0 else (p1.y if axis == 1 else p1.z)
        var d0 = (split_coord - c0) if keep_upper else (c0 - split_coord)
        var d1 = (split_coord - c1) if keep_upper else (c1 - split_coord)
        if d0 <= Float32(1.0e-5):
            clipped.append(p0)
            if d1 > Float32(1.0e-5):
                var t = d0 / (d0 - d1)
                clipped.append(p0 + (p1 - p0) * t)
        elif d1 <= Float32(1.0e-5):
            var t = d0 / (d0 - d1)
            clipped.append(p0 + (p1 - p0) * t)

    if len(clipped) >= 3:
        out_a.append(clipped[0])
        out_b.append(clipped[1])
        out_c.append(clipped[2])
        if len(clipped) >= 4:
            out_a.append(clipped[0])
            out_b.append(clipped[2])
            out_c.append(clipped[3])


def _finalize_vhacd_part(
    mut part: _VhacdPart,
    origin: Vec3,
    vox_size: Float32,
):
    var v3 = vox_size * vox_size * vox_size
    part.voxel_vol = Float32(len(part.vox_x)) * v3
    var pts = List[Vec3]()
    for i in range(len(part.tri_a)):
        pts.append(part.tri_a[i])
        pts.append(part.tri_b[i])
        pts.append(part.tri_c[i])

    # Also include interior/cut-face boundary voxel centers clamped to part triangle/voxel AABB
    if len(pts) < 4 and len(part.vox_x) > 0:
        var hs = vox_size * Float32(0.5)
        for vi in range(len(part.vox_x)):
            var vc = Vec3(
                origin.x + (Float32(part.vox_x[vi]) - Float32(0.5)) * vox_size,
                origin.y + (Float32(part.vox_y[vi]) - Float32(0.5)) * vox_size,
                origin.z + (Float32(part.vox_z[vi]) - Float32(0.5)) * vox_size,
            )
            pts.append(vc + Vec3(-hs, -hs, -hs))
            pts.append(vc + Vec3(hs, hs, hs))
    elif len(part.vox_x) > 0:
        # Ensure cut-plane interior points are represented so clipped open shells close cleanly
        var step_v = max(1, len(part.vox_x) // 48)
        for vi in range(0, len(part.vox_x), step_v):
            pts.append(Vec3(
                origin.x + (Float32(part.vox_x[vi]) - Float32(0.5)) * vox_size,
                origin.y + (Float32(part.vox_y[vi]) - Float32(0.5)) * vox_size,
                origin.z + (Float32(part.vox_z[vi]) - Float32(0.5)) * vox_size,
            ))

    part.hull = ConvexHull.build_quickhull(pts, max_verts=32, weld_eps=vox_size * Float32(0.02))
    var mp = part.hull.compute_polyhedral_mass_properties(Float32(1.0))
    part.hull_vol = mp[0]


struct CompoundShape(Copyable, Movable):
    var children: List[CompoundChild]

    def __init__(out self):
        self.children = List[CompoundChild]()

    def child_count(self) -> Int:
        return len(self.children)

    def add_child(mut self, var child: CompoundChild):
        self.children.append(child^)

    def tight_aabb(self, pos: Vec3, rot: Quat) -> Tuple[Vec3, Vec3]:
        if len(self.children) == 0:
            return (pos - Vec3(0.5, 0.5, 0.5), pos + Vec3(0.5, 0.5, 0.5))
        var lo = Vec3(1.0e30, 1.0e30, 1.0e30)
        var hi = Vec3(-1.0e30, -1.0e30, -1.0e30)
        for i in range(len(self.children)):
            var cb = self.children[i].tight_aabb(pos, rot)
            lo.x = min(lo.x, cb[0].x); lo.y = min(lo.y, cb[0].y); lo.z = min(lo.z, cb[0].z)
            hi.x = max(hi.x, cb[1].x); hi.y = max(hi.y, cb[1].y); hi.z = max(hi.z, cb[1].z)
        return (lo, hi)

    def compute_mass_properties(self, density: Float32 = 1000.0) -> Tuple[Float32, Vec3, Sym3]:
        """Exact parallel-axis compound mass, center of mass, and full `Sym3` inertia tensor."""
        if len(self.children) == 0:
            return (Float32(0.0), Vec3(), Sym3())
        var total_m = Float32(0.0)
        var weighted_com = Vec3()
        var child_m = List[Float32](capacity=len(self.children))
        var child_com = List[Vec3](capacity=len(self.children))
        var child_i = List[Sym3](capacity=len(self.children))

        for k in range(len(self.children)):
            var mp = self.children[k].compute_local_mass_properties(density)
            child_m.append(mp[0])
            child_com.append(mp[1])
            child_i.append(mp[2])
            total_m += mp[0]
            weighted_com = weighted_com + mp[1] * mp[0]

        if total_m <= Float32(1.0e-9):
            return (Float32(0.0), Vec3(), Sym3())

        var com = weighted_com * (Float32(1.0) / total_m)
        var total_i = Sym3()
        for k in range(len(self.children)):
            var mk = child_m[k]
            var d = child_com[k] - com
            var d2 = d.length_squared()
            var shift = Sym3(
                mk * (d2 - d.x * d.x),
                mk * (d2 - d.y * d.y),
                mk * (d2 - d.z * d.z),
                -mk * d.x * d.y,
                -mk * d.x * d.z,
                -mk * d.y * d.z,
            )
            total_i = total_i.plus(child_i[k]).plus(shift)
        return (total_m, com, total_i)

    def recenter_to_com(mut self, density: Float32 = 1000.0) -> Tuple[Float32, Vec3, Sym3]:
        """Shifts all `children[k].local_pos` by `-com` so compound origin coincides with CoM."""
        var mp = self.compute_mass_properties(density)
        var com = mp[1]
        for k in range(len(self.children)):
            self.children[k].local_pos = self.children[k].local_pos - com
        return mp

    def serialize(self, mut w: BlobWriter):
        w.begin(COMPOUND_BLOB_TAG, SHAPES_V5_BLOB_VERSION)
        w.integer(len(self.children))
        for i in range(len(self.children)):
            ref ch = self.children[i]
            w.integer(ch.shape)
            w.vec3(ch.local_pos)
            w.quat(ch.local_rot)
            w.f32(ch.radius)
            w.f32(ch.half_height)
            w.vec3(ch.half_extents)
            w.f32(ch.density_scale)
            w.integer(len(ch.hull.vertices))
            for vi in range(len(ch.hull.vertices)):
                w.vec3(ch.hull.vertices[vi])
            w.integer(len(ch.hull.indices))
            for ii in range(len(ch.hull.indices)):
                w.integer(Int(ch.hull.indices[ii]))
        w.end()

    @staticmethod
    def deserialize(mut r: BlobReader) raises -> CompoundShape:
        _ = r.enter(COMPOUND_BLOB_TAG, SHAPES_V5_BLOB_VERSION)
        var out = CompoundShape()
        var n_ch = r.integer()
        if r.failed or n_ch < 0 or n_ch > 4096 or n_ch > max(0, r.limit() - r.offset) // 76:
            raise Error("compound blob: invalid child count")
        for _ in range(n_ch):
            var ch = CompoundChild()
            ch.shape = r.integer()
            ch.local_pos = r.vec3()
            ch.local_rot = r.quat()
            ch.radius = r.f32()
            ch.half_height = r.f32()
            ch.half_extents = r.vec3()
            ch.density_scale = r.f32()
            var nv = r.integer()
            if r.failed or nv < 0 or nv > (1 << 20) or nv > max(0, r.limit() - r.offset) // 12:
                raise Error("compound blob: invalid vertex count")
            var verts = List[Vec3](capacity=nv)
            for _ in range(nv):
                verts.append(r.vec3())
            var ni = r.integer()
            if r.failed or ni < 0 or ni > (1 << 20) or ni > max(0, r.limit() - r.offset) // 8:
                raise Error("compound blob: invalid index count")
            var idxs = List[Int32](capacity=ni)
            for _ in range(ni):
                var idx = r.integer()
                if r.failed or idx < 0 or idx >= nv:
                    raise Error("compound blob: invalid hull vertex index")
                idxs.append(Int32(idx))
            ch.hull = ConvexHull(verts, idxs)
            out.children.append(ch^)
        r.leave()
        return out^

    @staticmethod
    def decompose_vhacd_lite(
        vertices: List[Vec3],
        indices: List[Int32],
        max_hulls: Int = 8,
        voxel_res: Int = 16,
        concavity_tol: Float32 = 0.02,
    ) -> CompoundShape:
        """Volumetric + plane-clip V-HACD-lite approximate convex decomposition."""
        var out = CompoundShape()
        if len(vertices) == 0:
            return out^

        var lo = vertices[0]
        var hi = vertices[0]
        for i in range(len(vertices)):
            var v = vertices[i]
            if not isfinite(v.x) or not isfinite(v.y) or not isfinite(v.z):
                return out^
            if i > 0:
                lo.x = min(lo.x, v.x); lo.y = min(lo.y, v.y); lo.z = min(lo.z, v.z)
                hi.x = max(hi.x, v.x); hi.y = max(hi.y, v.y); hi.z = max(hi.z, v.z)

        var n_tris = len(indices) // 3
        if len(vertices) < 4 or n_tris == 0:
            if len(vertices) >= 4:
                var h = ConvexHull.build_quickhull(vertices, max_verts=32)
                var mp = h.compute_polyhedral_mass_properties(Float32(1.0))
                var com = mp[1]
                for i in range(len(h.vertices)):
                    h.vertices[i] = h.vertices[i] - com
                out.add_child(CompoundChild.convex(com, Quat(), h))
            return out^

        var span = hi - lo
        var max_span = max(Float32(1.0e-3), max(span.x, max(span.y, span.z)))
        var res = max(6, min(32, voxel_res))
        var vox_size = max_span / Float32(res)
        var nx = max(1, Int(span.x / vox_size + Float32(0.999)))
        var ny = max(1, Int(span.y / vox_size + Float32(0.999)))
        var nz = max(1, Int(span.z / vox_size + Float32(0.999)))

        # Padded grid (0..nx+1, 0..ny+1, 0..nz+1): 0=unknown, 1=surface, 2=exterior air
        var gx = nx + 2
        var gy = ny + 2
        var gz = nz + 2
        var total_cells = gx * gy * gz
        var grid = List[UInt8](length=total_cells, fill=0)

        var root = _VhacdPart()
        var n_verts = len(vertices)
        for t in range(n_tris):
            var i0 = Int(indices[t * 3 + 0])
            var i1 = Int(indices[t * 3 + 1])
            var i2 = Int(indices[t * 3 + 2])
            if i0 < 0 or i0 >= n_verts or i1 < 0 or i1 >= n_verts or i2 < 0 or i2 >= n_verts:
                continue
            var a = vertices[i0]
            var b = vertices[i1]
            var c = vertices[i2]
            root.tri_a.append(a)
            root.tri_b.append(b)
            root.tri_c.append(c)

            var max_edge = max((b - a).length(), max((c - b).length(), (a - c).length()))
            var steps = max(1, min(64, Int(max_edge / (vox_size * Float32(0.45))) + 1))
            var inv_s = Float32(1.0) / Float32(steps)
            for su in range(steps + 1):
                var u = Float32(su) * inv_s
                for sv in range(steps + 1 - su):
                    var v = Float32(sv) * inv_s
                    var p = a * (Float32(1.0) - u - v) + b * u + c * v
                    var ix = max(1, min(nx, Int((p.x - lo.x) / vox_size) + 1))
                    var iy = max(1, min(ny, Int((p.y - lo.y) / vox_size) + 1))
                    var iz = max(1, min(nz, Int((p.z - lo.z) / vox_size) + 1))
                    grid[(iz * gy + iy) * gx + ix] = UInt8(1)

        if len(root.tri_a) == 0:
            return out^

        # Exterior air BFS from padded border (0, 0, 0)
        var queue = List[Int](capacity=total_cells)
        grid[0] = UInt8(2)
        queue.append(0)
        var q_head = 0
        while q_head < len(queue):
            var cur = queue[q_head]
            q_head += 1
            var cz = cur // (gx * gy)
            var rem = cur - cz * (gx * gy)
            var cy = rem // gx
            var cx = rem - cy * gx
            if cx > 0:
                var nb = cur - 1
                if grid[nb] == 0: grid[nb] = UInt8(2); queue.append(nb)
            if cx + 1 < gx:
                var nb = cur + 1
                if grid[nb] == 0: grid[nb] = UInt8(2); queue.append(nb)
            if cy > 0:
                var nb = cur - gx
                if grid[nb] == 0: grid[nb] = UInt8(2); queue.append(nb)
            if cy + 1 < gy:
                var nb = cur + gx
                if grid[nb] == 0: grid[nb] = UInt8(2); queue.append(nb)
            if cz > 0:
                var nb = cur - gx * gy
                if grid[nb] == 0: grid[nb] = UInt8(2); queue.append(nb)
            if cz + 1 < gz:
                var nb = cur + gx * gy
                if grid[nb] == 0: grid[nb] = UInt8(2); queue.append(nb)

        for iz in range(1, nz + 1):
            for iy in range(1, ny + 1):
                for ix in range(1, nx + 1):
                    if grid[(iz * gy + iy) * gx + ix] != UInt8(2):
                        root.vox_x.append(ix)
                        root.vox_y.append(iy)
                        root.vox_z.append(iz)

        _finalize_vhacd_part(root, lo, vox_size)
        var total_ref_vol = max(root.voxel_vol, root.hull_vol * Float32(0.1))
        var parts = List[_VhacdPart]()
        parts.append(root^)

        var target_hulls = max(1, max_hulls)
        while len(parts) < target_hulls:
            var worst_idx = -1
            var worst_conc = total_ref_vol * concavity_tol
            for pi in range(len(parts)):
                var c_val = parts[pi].concavity()
                if c_val > worst_conc and len(parts[pi].vox_x) >= 4:
                    worst_conc = c_val
                    worst_idx = pi
            if worst_idx < 0:
                break

            ref src = parts[worst_idx]
            var v_min = [nx + 1, ny + 1, nz + 1]
            var v_max = [0, 0, 0]
            for vi in range(len(src.vox_x)):
                var vx = src.vox_x[vi]; var vy = src.vox_y[vi]; var vz = src.vox_z[vi]
                if vx < v_min[0]: v_min[0] = vx
                if vx > v_max[0]: v_max[0] = vx
                if vy < v_min[1]: v_min[1] = vy
                if vy > v_max[1]: v_max[1] = vy
                if vz < v_min[2]: v_min[2] = vz
                if vz > v_max[2]: v_max[2] = vz

            # Find optimal axis-aligned split plane minimizing combined AABB waste + maximizing notch jump
            var best_ax = -1
            var best_split_i = -1
            var best_score = Float32(1.0e30)

            for ax in range(3):
                var lo_i = v_min[ax]
                var hi_i = v_max[ax]
                if hi_i - lo_i < 1:
                    continue
                var slice_cnt = List[Int](length=hi_i - lo_i + 1, fill=0)
                for vi in range(len(src.vox_x)):
                    var c_i = src.vox_x[vi] if ax == 0 else (src.vox_y[vi] if ax == 1 else src.vox_z[vi])
                    slice_cnt[c_i - lo_i] += 1

                for s_i in range(lo_i, hi_i):
                    # Split between voxel s_i (left: <= s_i) and s_i + 1 (right: > s_i)
                    var l_cnt = 0
                    var r_cnt = 0
                    var l_lo = [nx + 1, ny + 1, nz + 1]
                    var l_hi = [0, 0, 0]
                    var r_lo = [nx + 1, ny + 1, nz + 1]
                    var r_hi = [0, 0, 0]
                    for vi in range(len(src.vox_x)):
                        var vx = src.vox_x[vi]; var vy = src.vox_y[vi]; var vz = src.vox_z[vi]
                        var c_i = vx if ax == 0 else (vy if ax == 1 else vz)
                        if c_i <= s_i:
                            l_cnt += 1
                            if vx < l_lo[0]: l_lo[0] = vx
                            if vx > l_hi[0]: l_hi[0] = vx
                            if vy < l_lo[1]: l_lo[1] = vy
                            if vy > l_hi[1]: l_hi[1] = vy
                            if vz < l_lo[2]: l_lo[2] = vz
                            if vz > l_hi[2]: l_hi[2] = vz
                        else:
                            r_cnt += 1
                            if vx < r_lo[0]: r_lo[0] = vx
                            if vx > r_hi[0]: r_hi[0] = vx
                            if vy < r_lo[1]: r_lo[1] = vy
                            if vy > r_hi[1]: r_hi[1] = vy
                            if vz < r_lo[2]: r_lo[2] = vz
                            if vz > r_hi[2]: r_hi[2] = vz
                    if l_cnt == 0 or r_cnt == 0:
                        continue
                    var l_box = Float32((l_hi[0] - l_lo[0] + 1) * (l_hi[1] - l_lo[1] + 1) * (l_hi[2] - l_lo[2] + 1))
                    var r_box = Float32((r_hi[0] - r_lo[0] + 1) * (r_hi[1] - r_lo[1] + 1) * (r_hi[2] - r_lo[2] + 1))
                    var empty_waste = (l_box - Float32(l_cnt)) + (r_box - Float32(r_cnt))
                    var jump = abs(Float32(slice_cnt[s_i - lo_i] - slice_cnt[s_i + 1 - lo_i]))
                    var score = empty_waste - jump * Float32(0.75)
                    if score < best_score:
                        best_score = score
                        best_ax = ax
                        best_split_i = s_i

            if best_ax < 0:
                break

            var base_c = lo.x if best_ax == 0 else (lo.y if best_ax == 1 else lo.z)
            var split_coord = base_c + Float32(best_split_i) * vox_size

            # Snap split_coord to an exact nearby mesh vertex coordinate if within 0.6 * vox_size (crisp CAD/L-shape splits!)
            var best_snap = split_coord
            var best_snap_d = vox_size * Float32(0.6)
            for ti in range(len(src.tri_a)):
                var tv = [src.tri_a[ti], src.tri_b[ti], src.tri_c[ti]]
                for k in range(3):
                    var vc = tv[k].x if best_ax == 0 else (tv[k].y if best_ax == 1 else tv[k].z)
                    var d_s = abs(vc - split_coord)
                    if d_s < best_snap_d:
                        best_snap_d = d_s
                        best_snap = vc
            split_coord = best_snap

            var left_p = _VhacdPart()
            var right_p = _VhacdPart()
            for vi in range(len(src.vox_x)):
                var vx = src.vox_x[vi]; var vy = src.vox_y[vi]; var vz = src.vox_z[vi]
                var c_i = vx if best_ax == 0 else (vy if best_ax == 1 else vz)
                if c_i <= best_split_i:
                    left_p.vox_x.append(vx); left_p.vox_y.append(vy); left_p.vox_z.append(vz)
                else:
                    right_p.vox_x.append(vx); right_p.vox_y.append(vy); right_p.vox_z.append(vz)

            for ti in range(len(src.tri_a)):
                _clip_tri_halfspace(
                    src.tri_a[ti], src.tri_b[ti], src.tri_c[ti],
                    best_ax, split_coord, False,
                    left_p.tri_a, left_p.tri_b, left_p.tri_c,
                )
                _clip_tri_halfspace(
                    src.tri_a[ti], src.tri_b[ti], src.tri_c[ti],
                    best_ax, split_coord, True,
                    right_p.tri_a, right_p.tri_b, right_p.tri_c,
                )

            _finalize_vhacd_part(left_p, lo, vox_size)
            _finalize_vhacd_part(right_p, lo, vox_size)

            if len(left_p.hull.vertices) < 4 or len(right_p.hull.vertices) < 4:
                break

            parts[worst_idx] = left_p^
            parts.append(right_p^)

        for pi in range(len(parts)):
            var h = parts[pi].hull.copy()
            if len(h.vertices) < 4:
                continue
            var mp = h.compute_polyhedral_mass_properties(Float32(1.0))
            var part_com = mp[1]
            for vi in range(len(h.vertices)):
                h.vertices[vi] = h.vertices[vi] - part_com
            out.add_child(CompoundChild.convex(part_com, Quat(), h))

        return out^


def decompose_vhacd_lite(
    vertices: List[Vec3],
    indices: List[Int32],
    max_hulls: Int = 8,
    voxel_res: Int = 16,
    concavity_tol: Float32 = 0.02,
) -> CompoundShape:
    """Module-level alias for `CompoundShape.decompose_vhacd_lite`."""
    return CompoundShape.decompose_vhacd_lite(vertices, indices, max_hulls, voxel_res, concavity_tol)


# =========================================================================
# 2. HeightfieldCollider (Quantized UInt16 + 4-Level Mip Bounds + 2D DDA)
# =========================================================================

struct HeightfieldCollider(Copyable, Movable):
    """Quantized `UInt16` heightfield terrain collider with 4-level hierarchical min/max
    chunk bounds (`2x2, 4x4, 8x8, 16x16`), exact Amanatides-Woo 2D DDA raycast,
    and smooth-edge 4-point contact manifold generation."""
    var nx: Int
    var nz: Int
    var origin: Vec3
    var cell_size: Float32
    var min_y: Float32
    var max_y: Float32
    var y_scale: Float32
    var heights_u16: List[UInt16]
    var mip_min_u16: List[UInt16]
    var mip_max_u16: List[UInt16]
    var mip_offsets: Array[Int, 4]
    var mip_nx: Array[Int, 4]
    var mip_nz: Array[Int, 4]
    var material_id: UInt16

    def __init__(out self):
        self.nx = 0
        self.nz = 0
        self.origin = Vec3()
        self.cell_size = 1.0
        self.min_y = 0.0
        self.max_y = 0.0
        self.y_scale = 0.0
        self.heights_u16 = List[UInt16]()
        self.mip_min_u16 = List[UInt16]()
        self.mip_max_u16 = List[UInt16]()
        self.mip_offsets = Array[Int, 4](fill=0)
        self.mip_nx = Array[Int, 4](fill=0)
        self.mip_nz = Array[Int, 4](fill=0)
        self.material_id = 0

    @staticmethod
    def build(
        heights: List[Float32],
        nx: Int,
        nz: Int,
        origin: Vec3 = Vec3(),
        cell_size: Float32 = 1.0,
        material_id: UInt16 = 0,
    ) -> Self:
        comptime MAX_HF_DIM: Int = 16384
        var hf = Self()
        if nx < 2 or nz < 2 or nx > MAX_HF_DIM or nz > MAX_HF_DIM or nx > len(heights) // nz:
            return hf^
        if not isfinite(origin.x) or not isfinite(origin.y) or not isfinite(origin.z) or not isfinite(cell_size):
            return hf^
        var n_samples = nx * nz
        for i in range(n_samples):
            if not isfinite(heights[i]):
                return hf^
        var lo_y = heights[0]
        var hi_y = heights[0]
        for i in range(1, n_samples):
            var h = heights[i]
            if h < lo_y: lo_y = h
            if h > hi_y: hi_y = h
        var span_y_f64 = Float64(hi_y) - Float64(lo_y)
        var y_scale_f64 = span_y_f64 / 65535.0 if span_y_f64 > 1.0e-7 else 0.0
        if not isfinite(Float32(y_scale_f64)):
            return hf^

        hf.nx = nx
        hf.nz = nz
        hf.origin = origin
        hf.cell_size = max(Float32(1.0e-3), cell_size)
        hf.material_id = material_id
        hf.min_y = lo_y
        hf.max_y = hi_y
        hf.y_scale = Float32(y_scale_f64)
        var inv_scale_f64 = 65535.0 / span_y_f64 if span_y_f64 > 1.0e-7 else 0.0

        hf.heights_u16.reserve(n_samples)
        for i in range(n_samples):
            var q_f64 = (Float64(heights[i]) - Float64(lo_y)) * inv_scale_f64 + 0.5
            var q = Int(max(0.0, min(65535.0, q_f64)))
            hf.heights_u16.append(UInt16(q))

        # Build 4-level hierarchical cell min/max pyramid (2x2, 4x4, 8x8, 16x16 cells)
        var cells_x = nx - 1
        var cells_z = nz - 1

        # Level 0: 2x2 cell blocks directly over 4-corner cell bounds
        var m0_nx = (cells_x + 1) // 2
        var m0_nz = (cells_z + 1) // 2
        hf.mip_offsets[0] = 0
        hf.mip_nx[0] = m0_nx
        hf.mip_nz[0] = m0_nz
        for bz in range(m0_nz):
            for bx in range(m0_nx):
                var cmin = UInt16(65535)
                var cmax = UInt16(0)
                var cz_end = min(cells_z, bz * 2 + 2)
                var cx_end = min(cells_x, bx * 2 + 2)
                for cz in range(bz * 2, cz_end):
                    for cx in range(bx * 2, cx_end):
                        var h00 = hf.heights_u16[cz * nx + cx]
                        var h10 = hf.heights_u16[cz * nx + cx + 1]
                        var h01 = hf.heights_u16[(cz + 1) * nx + cx]
                        var h11 = hf.heights_u16[(cz + 1) * nx + cx + 1]
                        cmin = min(cmin, min(min(h00, h10), min(h01, h11)))
                        cmax = max(cmax, max(max(h00, h10), max(h01, h11)))
                hf.mip_min_u16.append(cmin)
                hf.mip_max_u16.append(cmax)

        # Levels 1..3: 2x2 reductions of previous level (4x4, 8x8, 16x16 cells)
        for lvl in range(1, 4):
            var prev_off = hf.mip_offsets[lvl - 1]
            var prev_nx = hf.mip_nx[lvl - 1]
            var prev_nz = hf.mip_nz[lvl - 1]
            var cur_nx = (prev_nx + 1) // 2
            var cur_nz = (prev_nz + 1) // 2
            hf.mip_offsets[lvl] = len(hf.mip_min_u16)
            hf.mip_nx[lvl] = cur_nx
            hf.mip_nz[lvl] = cur_nz
            for bz in range(cur_nz):
                for bx in range(cur_nx):
                    var cmin = UInt16(65535)
                    var cmax = UInt16(0)
                    var pz_end = min(prev_nz, bz * 2 + 2)
                    var px_end = min(prev_nx, bx * 2 + 2)
                    for pz in range(bz * 2, pz_end):
                        for px in range(bx * 2, px_end):
                            var idx = prev_off + pz * prev_nx + px
                            cmin = min(cmin, hf.mip_min_u16[idx])
                            cmax = max(cmax, hf.mip_max_u16[idx])
                    hf.mip_min_u16.append(cmin)
                    hf.mip_max_u16.append(cmax)

        return hf^

    @always_inline
    def _dequant_y_f64(self, q: UInt16) -> Float64:
        var span_f64 = Float64(self.max_y) - Float64(self.min_y)
        if span_f64 > 1.0e-7:
            var y = Float64(self.min_y) + (Float64(Int(q)) / 65535.0) * span_f64
            return min(Float64(self.max_y), max(Float64(self.min_y), y))
        return Float64(self.min_y) + Float64(Int(q)) * Float64(self.y_scale)

    @always_inline
    def dequant_y(self, q: UInt16) -> Float32:
        return Float32(self._dequant_y_f64(q))

    @always_inline
    def height_at_grid(self, ix: Int, iz: Int) -> Float32:
        if self.nx < 1 or self.nz < 1 or len(self.heights_u16) == 0:
            return self.origin.y
        var cx = max(0, min(self.nx - 1, ix))
        var cz = max(0, min(self.nz - 1, iz))
        return self.dequant_y(self.heights_u16[cz * self.nx + cx])

    def bounds(self) -> Tuple[Vec3, Vec3]:
        if self.nx < 2 or self.nz < 2:
            return (self.origin, self.origin)
        comptime F32_MAX_F64: Float64 = 3.4028234e38
        var hx = max(-F32_MAX_F64, min(F32_MAX_F64, Float64(self.origin.x) + Float64(self.nx - 1) * Float64(self.cell_size)))
        var hz = max(-F32_MAX_F64, min(F32_MAX_F64, Float64(self.origin.z) + Float64(self.nz - 1) * Float64(self.cell_size)))
        return (
            Vec3(self.origin.x, self.min_y, self.origin.z),
            Vec3(Float32(hx), self.max_y, Float32(hz)),
        )

    @always_inline
    def _cell_bounds_y(self, cx: Int, cz: Int) -> Tuple[Float32, Float32]:
        var r0 = cz * self.nx + cx
        var r1 = r0 + self.nx
        var h00 = self.heights_u16[r0]
        var h10 = self.heights_u16[r0 + 1]
        var h01 = self.heights_u16[r1]
        var h11 = self.heights_u16[r1 + 1]
        var qmin = min(min(h00, h10), min(h01, h11))
        var qmax = max(max(h00, h10), max(h01, h11))
        return (self.dequant_y(qmin), self.dequant_y(qmax))

    @always_inline
    def _chunk_bounds_y(self, level: Int, bx: Int, bz: Int) -> Tuple[Float32, Float32]:
        var idx = self.mip_offsets[level] + bz * self.mip_nx[level] + bx
        return (self.dequant_y(self.mip_min_u16[idx]), self.dequant_y(self.mip_max_u16[idx]))

    @always_inline
    def cell_triangles(self, cx: Int, cz: Int) -> Tuple[MeshTriangle, MeshTriangle]:
        """Returns the 2 upward-facing CCW triangles of cell `(cx, cz)` with interior diagonal smoothed."""
        var cs = self.cell_size
        var x0 = self.origin.x + Float32(cx) * cs
        var x1 = x0 + cs
        var z0 = self.origin.z + Float32(cz) * cs
        var z1 = z0 + cs
        var r0 = cz * self.nx + cx
        var r1 = r0 + self.nx
        var h00 = self._dequant_y_f64(self.heights_u16[r0])
        var h10 = self._dequant_y_f64(self.heights_u16[r0 + 1])
        var h01 = self._dequant_y_f64(self.heights_u16[r1])
        var h11 = self._dequant_y_f64(self.heights_u16[r1 + 1])
        var v00 = Vec3(x0, Float32(h00), z0)
        var v10 = Vec3(x1, Float32(h10), z0)
        var v01 = Vec3(x0, Float32(h01), z1)
        var v11 = Vec3(x1, Float32(h11), z1)
        # CCW upward: (v00, v01, v10) and (v10, v01, v11)
        var t0 = MeshTriangle(v00, v01, v10, self.material_id)
        var t1 = MeshTriangle(v10, v01, v11, self.material_id)
        var cs64 = Float64(cs)
        var n0x = -(h10 - h00)
        var n0z = -(h01 - h00)
        var l0 = sqrt(n0x * n0x + cs64 * cs64 + n0z * n0z)
        if l0 > 1.0e-12:
            t0.normal = Vec3(Float32(n0x / l0), Float32(cs64 / l0), Float32(n0z / l0))
        var n1x = -(h11 - h01)
        var n1z = -(h11 - h10)
        var l1 = sqrt(n1x * n1x + cs64 * cs64 + n1z * n1z)
        if l1 > 1.0e-12:
            t1.normal = Vec3(Float32(n1x / l1), Float32(cs64 / l1), Float32(n1z / l1))
        # Smooth shared diagonal edge (v01 <-> v10) when concave or nearly coplanar
        var dot_n = Float64(t0.normal.x) * Float64(t1.normal.x) + Float64(t0.normal.y) * Float64(t1.normal.y) + Float64(t0.normal.z) * Float64(t1.normal.z)
        var dv_dot = (Float64(x1) - Float64(x0)) * Float64(t0.normal.x) + (h11 - h01) * Float64(t0.normal.y)
        if dot_n > 0.996 or dv_dot >= -1.0e-4:
            t0.active_edges &= ~EDGE_ACTIVE_12
            t1.active_edges &= ~EDGE_ACTIVE_01
        # Smooth interior grid edges shared with neighbor cells so KCC/spheres never catch seams
        if cx > 0:
            t0.active_edges &= ~EDGE_ACTIVE_01
        if cz > 0:
            t0.active_edges &= ~EDGE_ACTIVE_20
        if cx + 1 < self.nx - 1:
            t1.active_edges &= ~EDGE_ACTIVE_20
        if cz + 1 < self.nz - 1:
            t1.active_edges &= ~EDGE_ACTIVE_12
        return (t0, t1)

    def sample_height_and_normal(self, world_x: Float32, world_z: Float32) -> Tuple[Float32, Vec3]:
        if self.nx < 2 or self.nz < 2 or not isfinite(world_x) or not isfinite(world_z):
            return (self.origin.y, Vec3(0, 1, 0))
        var fx = (Float64(world_x) - Float64(self.origin.x)) / Float64(self.cell_size)
        var fz = (Float64(world_z) - Float64(self.origin.z)) / Float64(self.cell_size)
        var cx = max(0, min(self.nx - 2, Int(floor(max(-1.0e9, min(1.0e9, fx))))))
        var cz = max(0, min(self.nz - 2, Int(floor(max(-1.0e9, min(1.0e9, fz))))))
        var u = max(0.0, min(1.0, fx - Float64(cx)))
        var v = max(0.0, min(1.0, fz - Float64(cz)))
        var r0 = cz * self.nx + cx
        var r1 = r0 + self.nx
        var h00 = self._dequant_y_f64(self.heights_u16[r0])
        var h10 = self._dequant_y_f64(self.heights_u16[r0 + 1])
        var h01 = self._dequant_y_f64(self.heights_u16[r1])
        var h11 = self._dequant_y_f64(self.heights_u16[r1 + 1])
        var cs = Float64(self.cell_size)
        var y64: Float64
        var nx64: Float64
        var nz64: Float64
        if u + v <= 1.0:
            y64 = h00 + v * (h01 - h00) + u * (h10 - h00)
            nx64 = -(h10 - h00)
            nz64 = -(h01 - h00)
        else:
            y64 = h11 + (1.0 - u) * (h01 - h11) + (1.0 - v) * (h10 - h11)
            nx64 = -(h11 - h01)
            nz64 = -(h11 - h10)
        if self.max_y >= self.min_y:
            y64 = min(Float64(self.max_y), max(Float64(self.min_y), y64))
        var l = sqrt(nx64 * nx64 + cs * cs + nz64 * nz64)
        var n = Vec3(Float32(nx64 / l), Float32(cs / l), Float32(nz64 / l)) if l > 1.0e-12 else Vec3(0, 1, 0)
        return (Float32(y64), n)

    def raycast_dda(self, origin: Vec3, dir: Vec3, max_t: Float32 = 1.0e6) -> RayHit:
        """Exact Amanatides-Woo 2D grid DDA traversal with hierarchical Y-slab culling."""
        var hit = RayHit()
        hit.t = max_t if isfinite(max_t) else Float32(1.0e6)
        if (
            self.nx < 2
            or self.nz < 2
            or not isfinite(max_t)
            or max_t <= Float32(0.0)
            or not (isfinite(origin.x) and isfinite(origin.y) and isfinite(origin.z))
            or not (isfinite(dir.x) and isfinite(dir.y) and isfinite(dir.z))
        ):
            return hit^

        var cells_x = self.nx - 1
        var cells_z = self.nz - 1
        var cs = self.cell_size
        var cs64 = Float64(cs)
        comptime F32_MAX_F64: Float64 = 3.4028234e38
        var b_lo = Vec3(self.origin.x, self.min_y - Float32(1.0e-3), self.origin.z)
        var b_hi = Vec3(
            Float32(max(-F32_MAX_F64, min(F32_MAX_F64, Float64(self.origin.x) + Float64(cells_x) * cs64))),
            self.max_y + Float32(1.0e-3),
            Float32(max(-F32_MAX_F64, min(F32_MAX_F64, Float64(self.origin.z) + Float64(cells_z) * cs64))),
        )

        # Clip ray [0, max_t] against heightfield 3D AABB
        var t_enter = Float32(0.0)
        var t_exit = max_t
        var orig_arr = [origin.x, origin.y, origin.z]
        var dir_arr = [dir.x, dir.y, dir.z]
        var lo_arr = [b_lo.x, b_lo.y, b_lo.z]
        var hi_arr = [b_hi.x, b_hi.y, b_hi.z]
        for ax in range(3):
            if abs(dir_arr[ax]) < Float32(1.0e-8):
                if orig_arr[ax] < lo_arr[ax] or orig_arr[ax] > hi_arr[ax]:
                    return hit^
            else:
                var inv_d = Float32(1.0) / dir_arr[ax]
                var t1 = (lo_arr[ax] - orig_arr[ax]) * inv_d
                var t2 = (hi_arr[ax] - orig_arr[ax]) * inv_d
                if t1 > t2:
                    var tmp = t1; t1 = t2; t2 = tmp
                t_enter = max(t_enter, t1)
                t_exit = min(t_exit, t2)
                if t_enter > t_exit:
                    return hit^

        # Pure vertical ray fast-path: single cell lookup!
        if abs(dir.x) < Float32(1.0e-7) and abs(dir.z) < Float32(1.0e-7):
            var fx_v = (Float64(origin.x) - Float64(self.origin.x)) / cs64
            var fz_v = (Float64(origin.z) - Float64(self.origin.z)) / cs64
            var cx = max(0, min(cells_x - 1, Int(floor(max(-1.0e9, min(1.0e9, fx_v))))))
            var cz = max(0, min(cells_z - 1, Int(floor(max(-1.0e9, min(1.0e9, fz_v))))))
            var tris = self.cell_triangles(cx, cz)
            var t_arr = [tris[0], tris[1]]
            for k in range(2):
                var res = t_arr[k].ray_intersect(origin, dir, hit.t)
                if res[0] and res[1] < hit.t:
                    hit.hit = True
                    hit.t = res[1]
                    hit.point = origin + dir * res[1]
                    hit.normal = res[2]
                    hit.triangle_index = (cz * cells_x + cx) * 2 + k
                    hit.material_id = self.material_id
            return hit^

        # Initialize 2D Amanatides-Woo DDA at ray entry point
        var p_start = origin + dir * t_enter
        if not (isfinite(p_start.x) and isfinite(p_start.y) and isfinite(p_start.z)):
            return hit^
        var fx = (Float64(p_start.x) - Float64(self.origin.x)) / cs64
        var fz = (Float64(p_start.z) - Float64(self.origin.z)) / cs64
        var ix = max(0, min(cells_x - 1, Int(floor(max(-1.0e9, min(1.0e9, fx))))))
        var iz = max(0, min(cells_z - 1, Int(floor(max(-1.0e9, min(1.0e9, fz))))))

        var step_x = 1 if dir.x > Float32(0.0) else (-1 if dir.x < Float32(0.0) else 0)
        var step_z = 1 if dir.z > Float32(0.0) else (-1 if dir.z < Float32(0.0) else 0)

        var t_max_x = Float32(1.0e30)
        var t_delta_x = Float32(1.0e30)
        if step_x != 0:
            var next_bx = self.origin.x + Float32(ix + (1 if step_x > 0 else 0)) * cs
            t_max_x = max(t_enter, (next_bx - origin.x) / dir.x)
            t_delta_x = cs / abs(dir.x)

        var t_max_z = Float32(1.0e30)
        var t_delta_z = Float32(1.0e30)
        if step_z != 0:
            var next_bz = self.origin.z + Float32(iz + (1 if step_z > 0 else 0)) * cs
            t_max_z = max(t_enter, (next_bz - origin.z) / dir.z)
            t_delta_z = cs / abs(dir.z)

        var t_cur = t_enter
        var max_steps = cells_x + cells_z + 4
        for _ in range(max_steps):
            if ix < 0 or ix >= cells_x or iz < 0 or iz >= cells_z or t_cur > min(t_exit, hit.t):
                break

            var t_next = min(t_exit, min(t_max_x, t_max_z))
            var y0 = origin.y + dir.y * t_cur
            var y1 = origin.y + dir.y * t_next
            var ray_ymin = min(y0, y1) - Float32(1.0e-4)
            var ray_ymax = max(y0, y1) + Float32(1.0e-4)

            # Level-0 (2x2) chunk Y-bounds early cull before unpacking 4 height vertices
            var c_chk = self._chunk_bounds_y(0, ix >> 1, iz >> 1)
            if ray_ymin <= c_chk[1] and ray_ymax >= c_chk[0]:
                var cb = self._cell_bounds_y(ix, iz)
                if ray_ymin <= cb[1] and ray_ymax >= cb[0]:
                    var tris = self.cell_triangles(ix, iz)
                    var t_arr = [tris[0], tris[1]]
                    for k in range(2):
                        var res = t_arr[k].ray_intersect(origin, dir, hit.t)
                        if res[0] and res[1] < hit.t:
                            hit.hit = True
                            hit.t = res[1]
                            hit.point = origin + dir * res[1]
                            hit.normal = res[2]
                            hit.triangle_index = (iz * cells_x + ix) * 2 + k
                            hit.material_id = self.material_id
                    if hit.hit and hit.t <= t_next + Float32(1.0e-5):
                        return hit^

            t_cur = t_next
            if t_max_x < t_max_z:
                t_max_x += t_delta_x
                ix += step_x
            else:
                t_max_z += t_delta_z
                iz += step_z

        return hit^

    def _query_aabb_triangles(self, q_lo: Vec3, q_hi: Vec3, mut out_tris: List[MeshTriangle], mut out_ids: List[Int]):
        if self.nx < 2 or self.nz < 2:
            return
        if not (
            isfinite(q_lo.x) and isfinite(q_lo.y) and isfinite(q_lo.z)
            and isfinite(q_hi.x) and isfinite(q_hi.y) and isfinite(q_hi.z)
        ):
            return
        if q_hi.y < self.min_y - Float32(1.0e-4) or q_lo.y > self.max_y + Float32(1.0e-4):
            return
        var cells_x = self.nx - 1
        var cells_z = self.nz - 1
        var cs64 = Float64(self.cell_size)
        var fx0 = (Float64(q_lo.x) - Float64(self.origin.x)) / cs64
        var fx1 = (Float64(q_hi.x) - Float64(self.origin.x)) / cs64
        var fz0 = (Float64(q_lo.z) - Float64(self.origin.z)) / cs64
        var fz1 = (Float64(q_hi.z) - Float64(self.origin.z)) / cs64
        var ix0 = max(0, Int(floor(max(-1.0e9, min(1.0e9, fx0)))))
        var ix1 = min(cells_x - 1, Int(floor(max(-1.0e9, min(1.0e9, fx1)))))
        var iz0 = max(0, Int(floor(max(-1.0e9, min(1.0e9, fz0)))))
        var iz1 = min(cells_z - 1, Int(floor(max(-1.0e9, min(1.0e9, fz1)))))
        if ix0 > ix1 or iz0 > iz1:
            return

        # Hierarchical 8x8 (level 2) chunk Y-cull for large queries
        var b2_x0 = ix0 >> 3
        var b2_x1 = ix1 >> 3
        var b2_z0 = iz0 >> 3
        var b2_z1 = iz1 >> 3
        for bz in range(b2_z0, b2_z1 + 1):
            for bx in range(b2_x0, b2_x1 + 1):
                var chk = self._chunk_bounds_y(2, bx, bz)
                if q_hi.y < chk[0] - Float32(1.0e-4) or q_lo.y > chk[1] + Float32(1.0e-4):
                    continue
                var sub_z0 = max(iz0, bz * 8)
                var sub_z1 = min(iz1, bz * 8 + 7)
                var sub_x0 = max(ix0, bx * 8)
                var sub_x1 = min(ix1, bx * 8 + 7)
                for cz in range(sub_z0, sub_z1 + 1):
                    for cx in range(sub_x0, sub_x1 + 1):
                        var cb = self._cell_bounds_y(cx, cz)
                        if q_hi.y >= cb[0] - Float32(1.0e-4) and q_lo.y <= cb[1] + Float32(1.0e-4):
                            var tris = self.cell_triangles(cx, cz)
                            var base_id = (cz * cells_x + cx) * 2
                            out_tris.append(tris[0])
                            out_ids.append(base_id)
                            out_tris.append(tris[1])
                            out_ids.append(base_id + 1)

    def sweep_sphere(
        self, start: Vec3, delta: Vec3, radius: Float32, max_toi: Float32 = 1.0,
    ) -> SweepHit:
        var out = SweepHit()
        out.toi = max_toi
        var end = start + delta * max_toi
        var rv = Vec3(radius, radius, radius)
        var q_lo = Vec3(min(start.x, end.x), min(start.y, end.y), min(start.z, end.z)) - rv
        var q_hi = Vec3(max(start.x, end.x), max(start.y, end.y), max(start.z, end.z)) + rv

        var tris = List[MeshTriangle]()
        var tids = List[Int]()
        self._query_aabb_triangles(q_lo, q_hi, tris, tids)
        if len(tris) == 0:
            return out^

        var dist = delta.length()
        if dist < Float32(1.0e-7):
            for i in range(len(tris)):
                var cp = tris[i].closest_point(start)
                var d = (start - cp[0]).length()
                if d <= radius:
                    out.hit = True
                    out.toi = Float32(0.0)
                    out.point = cp[0]
                    out.normal = cp[1]
                    out.triangle_index = tids[i]
                    return out^
            return out^

        var ndir = delta * (Float32(1.0) / dist)
        var max_d = dist * max_toi
        var best_d = max_d

        for i in range(len(tris)):
            ref tri = tris[i]
            # Initial overlap check
            var cp0 = tri.closest_point(start)
            var diff0 = start - cp0[0]
            if diff0.length_squared() <= radius * radius:
                out.hit = True
                out.toi = Float32(0.0)
                out.point = cp0[0]
                out.normal = cp0[1]
                out.triangle_index = tids[i]
                return out^

            # Sphere center hits plane (p - v0) . face_n = radius where face_n opposes ndir
            var face_n = tri.normal if (start - tri.v0).dot(tri.normal) >= Float32(0.0) else -tri.normal
            var denom = ndir.dot(face_n)
            if denom < Float32(-1.0e-7):
                var t_plane = (radius - (start - tri.v0).dot(face_n)) / denom
                if t_plane >= Float32(0.0) and t_plane < best_d:
                    var p_on_tri = start + ndir * t_plane - face_n * radius
                    var cp_t = tri.closest_point(p_on_tri)
                    if (cp_t[0] - p_on_tri).length_squared() <= Float32(1.0e-5):
                        best_d = t_plane
                        out.hit = True
                        out.toi = t_plane / dist
                        out.point = p_on_tri
                        out.normal = face_n
                        out.triangle_index = tids[i]

            # Swept sphere vs 3 triangle edges/vertices
            var edges = [(tri.v0, tri.v1), (tri.v1, tri.v2), (tri.v2, tri.v0)]
            for ei in range(3):
                var seg_res = Bvh8MeshCollider._sweep_sphere_segment(
                    start, ndir, best_d, radius, edges[ei][0], edges[ei][1]
                )
                if seg_res[0] and seg_res[1] < best_d:
                    best_d = seg_res[1]
                    out.hit = True
                    out.toi = seg_res[1] / dist
                    out.point = seg_res[2]
                    var cp_s = tri.closest_point(start + ndir * seg_res[1])
                    out.normal = cp_s[1]
                    out.triangle_index = tids[i]

        return out^

    def collide_sphere(self, s_pos: Vec3, s_radius: Float32) -> ContactManifold:
        var m = ContactManifold()
        var rv = Vec3(s_radius, s_radius, s_radius)
        var tris = List[MeshTriangle]()
        var tids = List[Int]()
        self._query_aabb_triangles(s_pos - rv, s_pos + rv, tris, tids)
        var cands = List[ContactPoint]()
        for i in range(len(tris)):
            var cp = tris[i].closest_point(s_pos)
            var diff = s_pos - cp[0]
            var d2 = diff.length_squared()
            if d2 <= s_radius * s_radius:
                var d = sqrt(d2)
                var depth = s_radius - d
                var n = cp[1]
                if d > Float32(1.0e-5) and (tris[i].active_edges == EDGE_ACTIVE_ALL):
                    n = diff * (Float32(1.0) / d)
                if n.y < Float32(0.0):
                    n = -n
                cands.append(ContactPoint(cp[0], n, depth, UInt32(tids[i])))
        reduce_manifold_candidates(m, cands)
        return m^

    def collide_capsule(self, cap_a: Vec3, cap_b: Vec3, cap_radius: Float32) -> ContactManifold:
        var m = ContactManifold()
        var rv = Vec3(cap_radius, cap_radius, cap_radius)
        var q_lo = Vec3(min(cap_a.x, cap_b.x), min(cap_a.y, cap_b.y), min(cap_a.z, cap_b.z)) - rv
        var q_hi = Vec3(max(cap_a.x, cap_b.x), max(cap_a.y, cap_b.y), max(cap_a.z, cap_b.z)) + rv
        var tris = List[MeshTriangle]()
        var tids = List[Int]()
        self._query_aabb_triangles(q_lo, q_hi, tris, tids)
        var cands = List[ContactPoint]()
        var seg = cap_b - cap_a
        var seg2 = seg.length_squared()
        for i in range(len(tris)):
            ref tri = tris[i]
            # Test endpoints + closest points to triangle edges
            var best_d2 = Float32(1.0e30)
            var best_cp = tri.v0
            var best_n = tri.normal

            var samples = [Float32(0.0), Float32(0.5), Float32(1.0)]
            if seg2 > Float32(1.0e-8):
                var e0 = closest_points_segments(cap_a, cap_b, tri.v0, tri.v1)
                var e1 = closest_points_segments(cap_a, cap_b, tri.v1, tri.v2)
                var e2 = closest_points_segments(cap_a, cap_b, tri.v2, tri.v0)
                var t_extra = [
                    max(Float32(0.0), min(Float32(1.0), (e0[0] - cap_a).dot(seg) / seg2)),
                    max(Float32(0.0), min(Float32(1.0), (e1[0] - cap_a).dot(seg) / seg2)),
                    max(Float32(0.0), min(Float32(1.0), (e2[0] - cap_a).dot(seg) / seg2)),
                ]
                for k in range(3):
                    var p_s = cap_a + seg * t_extra[k]
                    var cp = tri.closest_point(p_s)
                    var d2 = (p_s - cp[0]).length_squared()
                    if d2 < best_d2:
                        best_d2 = d2; best_cp = cp[0]; best_n = cp[1]
            for k in range(3):
                var p_s = cap_a + seg * samples[k]
                var cp = tri.closest_point(p_s)
                var d2 = (p_s - cp[0]).length_squared()
                if d2 < best_d2:
                    best_d2 = d2; best_cp = cp[0]; best_n = cp[1]

            if best_d2 <= cap_radius * cap_radius:
                var d = sqrt(best_d2)
                var n = best_n
                if n.y < Float32(0.0):
                    n = -n
                cands.append(ContactPoint(best_cp, n, cap_radius - d, UInt32(tids[i])))
        reduce_manifold_candidates(m, cands)
        return m^

    def collide_box(self, b_pos: Vec3, b_rot: Quat, b_half: Vec3) -> ContactManifold:
        var m = ContactManifold()
        var axes = get_box_axes(b_rot)
        var wx = abs(axes[0].x) * b_half.x + abs(axes[1].x) * b_half.y + abs(axes[2].x) * b_half.z
        var wy = abs(axes[0].y) * b_half.x + abs(axes[1].y) * b_half.y + abs(axes[2].y) * b_half.z
        var wz = abs(axes[0].z) * b_half.x + abs(axes[1].z) * b_half.y + abs(axes[2].z) * b_half.z
        var ext = Vec3(wx, wy, wz)
        var tris = List[MeshTriangle]()
        var tids = List[Int]()
        self._query_aabb_triangles(b_pos - ext, b_pos + ext, tris, tids)
        if len(tris) == 0:
            return m^

        var box_verts = List[Vec3](capacity=8)
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    box_verts.append(Vec3(Float32(sx) * b_half.x, Float32(sy) * b_half.y, Float32(sz) * b_half.z))
        var box_hull = ConvexHull(box_verts)
        var cands = List[ContactPoint]()

        for i in range(len(tris)):
            ref tri = tris[i]
            var tv = List[Vec3](capacity=3)
            tv.append(tri.v0); tv.append(tri.v1); tv.append(tri.v2)
            var thull = ConvexHull(tv)
            var tm = collide_convex_convex(box_hull, b_pos, b_rot, thull, Vec3(), Quat())
            if tm.has_collision:
                for pi in range(tm.count):
                    var cp = tm.get_point(pi)
                    var n_out = -cp.normal
                    if abs(n_out.dot(tri.normal)) > Float32(0.45):
                        n_out = tri.normal
                    elif n_out.y < Float32(0.0):
                        n_out = -n_out
                    cands.append(ContactPoint(cp.point, n_out, cp.depth, UInt32(tids[i] * 8 + pi)))
        reduce_manifold_candidates(m, cands)
        return m^


# =========================================================================
# 3. SixDofMotorJoint (3 Linear + 3 Swing/Twist Limits & Soft PD Drives)
# =========================================================================

struct SixDofMotorJoint(Copyable, ImplicitlyCopyable, Movable):
    """6-DOF constraint between `body_a` (`JOINT_WORLD` for world) and `body_b` supporting
    independent 3-axis linear limits, swing/twist angular limits, and soft PD motors."""
    var id: Int
    var body_a: Int
    var body_b: Int
    var local_anchor_a: Vec3
    var local_anchor_b: Vec3
    var local_frame_a: Quat
    var local_frame_b: Quat

    # Linear limits & soft PD drive in A's joint frame
    var enable_linear_limit: Bool
    var linear_lower: Vec3
    var linear_upper: Vec3
    var enable_linear_drive: Bool
    var linear_target: Vec3
    var linear_vel_target: Vec3
    var linear_hertz: Float32
    var linear_damping: Float32
    var max_linear_force: Float32

    # Angular swing-cone + twist limits & soft PD drive
    var enable_angular_limit: Bool
    var swing_y_limit: Float32
    var swing_z_limit: Float32
    var twist_lower: Float32
    var twist_upper: Float32
    var enable_angular_drive: Bool
    var angular_target: Quat
    var angular_vel_target: Vec3
    var angular_hertz: Float32
    var angular_damping: Float32
    var max_angular_torque: Float32

    var break_force: Float32
    var break_torque: Float32
    var broken: Bool

    # Persisted warm-start impulses across ticks/substeps
    var lin_lo_imp: Vec3
    var lin_hi_imp: Vec3
    var lin_drive_imp: Vec3
    var ang_limit_imp: Vec3
    var ang_drive_imp: Vec3

    def __init__(out self):
        self.id = 0
        self.body_a = JOINT_WORLD
        self.body_b = 0
        self.local_anchor_a = Vec3()
        self.local_anchor_b = Vec3()
        self.local_frame_a = Quat()
        self.local_frame_b = Quat()
        self.enable_linear_limit = True
        self.linear_lower = Vec3()
        self.linear_upper = Vec3()
        self.enable_linear_drive = False
        self.linear_target = Vec3()
        self.linear_vel_target = Vec3()
        self.linear_hertz = 8.0
        self.linear_damping = 1.0
        self.max_linear_force = 1000.0
        self.enable_angular_limit = False
        self.swing_y_limit = 0.785398
        self.swing_z_limit = 0.785398
        self.twist_lower = -0.523599
        self.twist_upper = 0.523599
        self.enable_angular_drive = False
        self.angular_target = Quat()
        self.angular_vel_target = Vec3()
        self.angular_hertz = 8.0
        self.angular_damping = 1.0
        self.max_angular_torque = 500.0
        self.break_force = 0.0
        self.break_torque = 0.0
        self.broken = False
        self.lin_lo_imp = Vec3()
        self.lin_hi_imp = Vec3()
        self.lin_drive_imp = Vec3()
        self.ang_limit_imp = Vec3()
        self.ang_drive_imp = Vec3()

    def reset_warm_impulses(mut self):
        self.lin_lo_imp = Vec3()
        self.lin_hi_imp = Vec3()
        self.lin_drive_imp = Vec3()
        self.ang_limit_imp = Vec3()
        self.ang_drive_imp = Vec3()

    def warm_start(self, ia_idx: Int, ib_idx: Int, mut bodies: SolverBodies):
        if self.broken:
            return
        var qa = Quat() if ia_idx >= bodies.count else bodies.orientation(ia_idx)
        var qb = Quat() if ib_idx >= bodies.count else bodies.orientation(ib_idx)
        var frame_w = quat_product(qa, self.local_frame_a) if self.body_a != JOINT_WORLD else self.local_frame_a

        var ra = rotate_vec(qa, self.local_anchor_a) if self.body_a != JOINT_WORLD else Vec3()
        var rb = rotate_vec(qb, self.local_anchor_b)

        var lin_local = (self.lin_lo_imp - self.lin_hi_imp) + self.lin_drive_imp
        var p_w = rotate_vec(frame_w, lin_local)
        var l_w = self.ang_limit_imp + self.ang_drive_imp
        var ma = bodies.inv_mass[ia_idx]
        var mb = bodies.inv_mass[ib_idx]
        var ina = bodies.inertia(ia_idx)
        var inb = bodies.inertia(ib_idx)
        bodies.set_velocity(ia_idx, bodies.velocity(ia_idx) - p_w * ma, bodies.angular(ia_idx) - ina.apply(ra.cross(p_w) + l_w))
        bodies.set_velocity(ib_idx, bodies.velocity(ib_idx) + p_w * mb, bodies.angular(ib_idx) + inb.apply(rb.cross(p_w) + l_w))

    def solve_step(mut self, ia_idx: Int, ib_idx: Int, mut bodies: SolverBodies, h: Float32, use_bias: Bool):
        if self.broken or h <= Float32(1.0e-7):
            return
        var inv_h = Float32(1.0) / h
        var nyquist = Float32(0.25) * inv_h
        var pa = Vec3() if ia_idx >= bodies.count else bodies.center(ia_idx)
        var qa = Quat() if ia_idx >= bodies.count else bodies.orientation(ia_idx)
        var pb = Vec3() if ib_idx >= bodies.count else bodies.center(ib_idx)
        var qb = Quat() if ib_idx >= bodies.count else bodies.orientation(ib_idx)

        var va = bodies.velocity(ia_idx)
        var wa = bodies.angular(ia_idx)
        var vb = bodies.velocity(ib_idx)
        var wb = bodies.angular(ib_idx)
        var ma = bodies.inv_mass[ia_idx]
        var mb = bodies.inv_mass[ib_idx]
        var ina = bodies.inertia(ia_idx)
        var inb = bodies.inertia(ib_idx)

        var world_a = self.body_a == JOINT_WORLD
        var ra = rotate_vec(qa, self.local_anchor_a) if not world_a else Vec3()
        var anchor_a = self.local_anchor_a if world_a else pa + ra
        var rb = rotate_vec(qb, self.local_anchor_b)
        var anchor_b = pb + rb

        var frame_a_w = self.local_frame_a if world_a else quat_product(qa, self.local_frame_a)
        var frame_b_w = quat_product(qb, self.local_frame_b)
        var ax_x = rotate_vec(frame_a_w, Vec3(1, 0, 0))
        var ax_y = rotate_vec(frame_a_w, Vec3(0, 1, 0))
        var ax_z = rotate_vec(frame_a_w, Vec3(0, 0, 1))
        var axes = [ax_x, ax_y, ax_z]

        # 1. Angular soft PD drive (evaluated in STAGE_SOLVE so STAGE_RELAX preserves motor/spring velocity)
        var k_ang = ina.plus(inb).inverse()
        if use_bias and self.enable_angular_drive and self.angular_hertz > Float32(0.0):
            var soft_a = make_soft(min(self.angular_hertz, nyquist), self.angular_damping, h)
            var target_w = quat_product(frame_a_w, self.angular_target)
            var c_ang = rotation_error(frame_b_w, target_w)
            var w_tgt_w = rotate_vec(frame_a_w, self.angular_vel_target)
            var cdot_a = (wb - wa) - w_tgt_w
            var imp = k_ang.apply(cdot_a + c_ang * soft_a.bias_rate) * (-soft_a.mass_scale) - self.ang_drive_imp * soft_a.impulse_scale
            var tot = clamp_length(self.ang_drive_imp + imp, self.max_angular_torque * h)
            imp = tot - self.ang_drive_imp
            self.ang_drive_imp = tot
            wa = wa - ina.apply(imp)
            wb = wb + inb.apply(imp)

        # 2. Angular swing/twist limits
        if self.enable_angular_limit:
            var q_rel = quat_product(quat_conj(frame_a_w), frame_b_w)
            if q_rel.w < Float32(0.0):
                q_rel = Quat(-q_rel.x, -q_rel.y, -q_rel.z, -q_rel.w)
            var twist_len = sqrt(q_rel.x * q_rel.x + q_rel.w * q_rel.w)
            var q_twist = Quat(q_rel.x / twist_len, 0, 0, q_rel.w / twist_len) if twist_len > Float32(1.0e-6) else Quat()
            var q_swing = quat_product(q_rel, quat_conj(q_twist))
            var twist_ang = Float32(2.0) * det_atan2(q_twist.x, q_twist.w)
            var swing_y = Float32(2.0) * det_atan2(q_swing.y, max(Float32(1.0e-6), q_swing.w))
            var swing_z = Float32(2.0) * det_atan2(q_swing.z, max(Float32(1.0e-6), q_swing.w))

            var clamped_tw = max(self.twist_lower, min(self.twist_upper, twist_ang))
            var clamped_sy = max(-self.swing_y_limit, min(self.swing_y_limit, swing_y))
            var clamped_sz = max(-self.swing_z_limit, min(self.swing_z_limit, swing_z))

            var err_local = Vec3(twist_ang - clamped_tw, swing_y - clamped_sy, swing_z - clamped_sz)
            if err_local.length_squared() > Float32(1.0e-8):
                var err_w = rotate_vec(frame_a_w, err_local)
                var lim_soft = make_soft(min(Float32(30.0), nyquist), Float32(1.0), h)
                var bias_a = err_w * (lim_soft.bias_rate if use_bias else Float32(0.0))
                var mscale_a = lim_soft.mass_scale if use_bias else Float32(1.0)
                var iscale_a = lim_soft.impulse_scale if use_bias else Float32(0.0)
                var imp = k_ang.apply((wb - wa) + bias_a) * (-mscale_a) - self.ang_limit_imp * iscale_a
                self.ang_limit_imp = self.ang_limit_imp + imp
                wa = wa - ina.apply(imp)
                wb = wb + inb.apply(imp)
            elif use_bias:
                self.ang_limit_imp = Vec3()

        # 3. Linear soft PD drive & 3-axis box limits in frame_a_w
        var d_w = anchor_b - anchor_a
        var lo_arr = [self.linear_lower.x, self.linear_lower.y, self.linear_lower.z]
        var hi_arr = [self.linear_upper.x, self.linear_upper.y, self.linear_upper.z]
        var tgt_arr = [self.linear_target.x, self.linear_target.y, self.linear_target.z]
        var vtgt_arr = [self.linear_vel_target.x, self.linear_vel_target.y, self.linear_vel_target.z]
        var lo_imp = [self.lin_lo_imp.x, self.lin_lo_imp.y, self.lin_lo_imp.z]
        var hi_imp = [self.lin_hi_imp.x, self.lin_hi_imp.y, self.lin_hi_imp.z]
        var drv_imp = [self.lin_drive_imp.x, self.lin_drive_imp.y, self.lin_drive_imp.z]
        var soft_l = make_soft(min(max(Float32(1.0e-3), self.linear_hertz), nyquist), self.linear_damping, h)
        var lim_soft_l = make_soft(min(Float32(30.0), nyquist), Float32(1.0), h)

        for k in range(3):
            var u = axes[k]
            var ca = ra.cross(u)
            var cb = rb.cross(u)
            var inv_k = ma + mb + ca.dot(ina.apply(ca)) + cb.dot(inb.apply(cb))
            if inv_k <= Float32(1.0e-12):
                continue
            var eff_m = Float32(1.0) / inv_k
            var coord = d_w.dot(u)

            if use_bias and self.enable_linear_drive and self.linear_hertz > Float32(0.0):
                var cdot = ((vb + wb.cross(rb)) - (va + wa.cross(ra))).dot(u) - vtgt_arr[k]
                var bias = (coord - tgt_arr[k]) * soft_l.bias_rate
                var dl = -eff_m * soft_l.mass_scale * (cdot + bias) - soft_l.impulse_scale * drv_imp[k]
                var cap = self.max_linear_force * h
                var tot = max(-cap, min(cap, drv_imp[k] + dl)) if cap > Float32(0.0) else (drv_imp[k] + dl)
                dl = tot - drv_imp[k]
                drv_imp[k] = tot
                var j_vec = u * dl
                va = va - j_vec * ma; wa = wa - ina.apply(ra.cross(j_vec))
                vb = vb + j_vec * mb; wb = wb + inb.apply(rb.cross(j_vec))

            if self.enable_linear_limit:
                # Lower limit: coord - lo >= 0
                var c_lo = coord - lo_arr[k]
                var cdot_lo = ((vb + wb.cross(rb)) - (va + wa.cross(ra))).dot(u)
                var bias_lo = c_lo * inv_h if c_lo > Float32(0.0) else (c_lo * lim_soft_l.bias_rate if use_bias else Float32(0.0))
                var mscale_lo = Float32(1.0) if (c_lo > Float32(0.0) or not use_bias) else lim_soft_l.mass_scale
                var iscale_lo = Float32(0.0) if (c_lo > Float32(0.0) or not use_bias) else lim_soft_l.impulse_scale
                var dl_lo = -eff_m * mscale_lo * (cdot_lo + bias_lo) - iscale_lo * lo_imp[k]
                var tot_lo = max(Float32(0.0), lo_imp[k] + dl_lo)
                dl_lo = tot_lo - lo_imp[k]
                lo_imp[k] = tot_lo
                if abs(dl_lo) > Float32(0.0):
                    var j_lo = u * dl_lo
                    va = va - j_lo * ma; wa = wa - ina.apply(ra.cross(j_lo))
                    vb = vb + j_lo * mb; wb = wb + inb.apply(rb.cross(j_lo))

                # Upper limit: hi - coord >= 0
                var c_hi = hi_arr[k] - coord
                var cdot_hi = -((vb + wb.cross(rb)) - (va + wa.cross(ra))).dot(u)
                var bias_hi = c_hi * inv_h if c_hi > Float32(0.0) else (c_hi * lim_soft_l.bias_rate if use_bias else Float32(0.0))
                var mscale_hi = Float32(1.0) if (c_hi > Float32(0.0) or not use_bias) else lim_soft_l.mass_scale
                var iscale_hi = Float32(0.0) if (c_hi > Float32(0.0) or not use_bias) else lim_soft_l.impulse_scale
                var dl_hi = -eff_m * mscale_hi * (cdot_hi + bias_hi) - iscale_hi * hi_imp[k]
                var tot_hi = max(Float32(0.0), hi_imp[k] + dl_hi)
                dl_hi = tot_hi - hi_imp[k]
                hi_imp[k] = tot_hi
                if abs(dl_hi) > Float32(0.0):
                    var j_hi = -u * dl_hi
                    va = va - j_hi * ma; wa = wa - ina.apply(ra.cross(j_hi))
                    vb = vb + j_hi * mb; wb = wb + inb.apply(rb.cross(j_hi))

        self.lin_lo_imp = Vec3(lo_imp[0], lo_imp[1], lo_imp[2])
        self.lin_hi_imp = Vec3(hi_imp[0], hi_imp[1], hi_imp[2])
        self.lin_drive_imp = Vec3(drv_imp[0], drv_imp[1], drv_imp[2])

        if use_bias:
            var net_lin = (self.lin_lo_imp - self.lin_hi_imp) + self.lin_drive_imp
            var net_ang = self.ang_limit_imp + self.ang_drive_imp
            if self.break_force > Float32(0.0) and net_lin.length_squared() * inv_h * inv_h > self.break_force * self.break_force:
                self.broken = True
                self.reset_warm_impulses()
            elif self.break_torque > Float32(0.0) and net_ang.length_squared() * inv_h * inv_h > self.break_torque * self.break_torque:
                self.broken = True
                self.reset_warm_impulses()

        bodies.set_velocity(ia_idx, va, wa)
        bodies.set_velocity(ib_idx, vb, wb)

    def serialize(self, mut w: BlobWriter):
        w.begin(SIXDOF_BLOB_TAG, SHAPES_V5_BLOB_VERSION)
        w.integer(self.id)
        w.integer(self.body_a)
        w.integer(self.body_b)
        w.vec3(self.local_anchor_a)
        w.vec3(self.local_anchor_b)
        w.quat(self.local_frame_a)
        w.quat(self.local_frame_b)
        w.boolean(self.enable_linear_limit)
        w.vec3(self.linear_lower)
        w.vec3(self.linear_upper)
        w.boolean(self.enable_linear_drive)
        w.vec3(self.linear_target)
        w.vec3(self.linear_vel_target)
        w.f32(self.linear_hertz)
        w.f32(self.linear_damping)
        w.f32(self.max_linear_force)
        w.boolean(self.enable_angular_limit)
        w.f32(self.swing_y_limit)
        w.f32(self.swing_z_limit)
        w.f32(self.twist_lower)
        w.f32(self.twist_upper)
        w.boolean(self.enable_angular_drive)
        w.quat(self.angular_target)
        w.vec3(self.angular_vel_target)
        w.f32(self.angular_hertz)
        w.f32(self.angular_damping)
        w.f32(self.max_angular_torque)
        w.f32(self.break_force)
        w.f32(self.break_torque)
        w.boolean(self.broken)
        w.vec3(self.lin_lo_imp)
        w.vec3(self.lin_hi_imp)
        w.vec3(self.lin_drive_imp)
        w.vec3(self.ang_limit_imp)
        w.vec3(self.ang_drive_imp)
        w.end()

    @staticmethod
    def deserialize(mut r: BlobReader) raises -> SixDofMotorJoint:
        _ = r.enter(SIXDOF_BLOB_TAG, SHAPES_V5_BLOB_VERSION)
        var j = SixDofMotorJoint()
        j.id = r.integer()
        j.body_a = r.integer()
        j.body_b = r.integer()
        j.local_anchor_a = r.vec3()
        j.local_anchor_b = r.vec3()
        j.local_frame_a = r.quat()
        j.local_frame_b = r.quat()
        j.enable_linear_limit = r.boolean()
        j.linear_lower = r.vec3()
        j.linear_upper = r.vec3()
        j.enable_linear_drive = r.boolean()
        j.linear_target = r.vec3()
        j.linear_vel_target = r.vec3()
        j.linear_hertz = r.f32()
        j.linear_damping = r.f32()
        j.max_linear_force = r.f32()
        j.enable_angular_limit = r.boolean()
        j.swing_y_limit = r.f32()
        j.swing_z_limit = r.f32()
        j.twist_lower = r.f32()
        j.twist_upper = r.f32()
        j.enable_angular_drive = r.boolean()
        j.angular_target = r.quat()
        j.angular_vel_target = r.vec3()
        j.angular_hertz = r.f32()
        j.angular_damping = r.f32()
        j.max_angular_torque = r.f32()
        j.break_force = r.f32()
        j.break_torque = r.f32()
        j.broken = r.boolean()
        j.lin_lo_imp = r.vec3()
        j.lin_hi_imp = r.vec3()
        j.lin_drive_imp = r.vec3()
        j.ang_limit_imp = r.vec3()
        j.ang_drive_imp = r.vec3()
        r.leave()
        return j^
