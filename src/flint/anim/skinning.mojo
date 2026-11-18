"""Skinning (package 3.4): how a vertex follows its joints.

Three methods, one definition, used by the CPU path (the reference, the CPU
backend, and whatever the GPU pass has not skinned yet) and mirrored step by
step by the GPU kernel in render/gpu_skinning.mojo:

    SKIN_LBS   linear blend: the weighted sum of the joints' matrices. Cheap,
               and what the engine did before 3.4; a twisted joint collapses
               (the candy wrapper) and a bent one loses volume.
    SKIN_DQS   dual quaternions (Kavan et al. 2008): the joints' rigid
               transforms are blended as unit dual quaternions, so the result
               is always a rotation and a translation. No collapse; a bent
               joint bulges instead.
    SKIN_COR   optimized centers of rotation (Le and Hodgins 2016): each
               vertex has a center, computed once from the weights of the
               whole mesh; the center is skinned linearly, and the vertex is
               turned about it by the blended rotation. No collapse and no
               bulge.

Which one a model gets is the asset's choice (the registry's "skinning", in
its manifest, `NativeModel.skin_method`): weights painted for linear blend
look worse under the other two, so a model that does not say is skinned
linearly. The `skin.method` console variable forces one on everything.

A draw's joints are a palette: per joint PALETTE_FLOATS floats, the 3x4 skin
matrix by rows (global transform x inverse bind), its rotation as a unit
quaternion, and the dual part (half the translation times the rotation).
Quaternion methods need rigid joints; a palette with a scaled or sheared
joint is skinned with SKIN_LBS (`effective_method`), which is exact for any
matrix.

The blended rotation is the average that does not care about quaternion
signs: the unit quaternion b that maximizes sum w (q . b)^2 (the dominant
eigenvector of sum w q q^T), found by SKIN_AVERAGE_STEPS power iterations
from the heaviest influence's quaternion. The usual shortcut, flipping each
quaternion to the side of a pivot and adding, jumps where neighbouring
vertices pick different pivots or a joint passes a half turn from the pivot;
on Maroon, whose arms go from all fours to straight overhead, that tore the
shoulders (edges stretched 28 times in the roll, against 6 for linear
blend). The average only becomes arbitrary where two rotations are equally
good, and there no blended rotation means anything, so the result fades to
linear blend: `agreement` = sum w (q . b)^2 / sum w is 1 when every joint
turned alike and at most 1/2 where the average is a toss-up; from
SKIN_AGREE_FULL down to SKIN_AGREE_NONE the vertex goes from the blended
rotation to linear blend. Two equal joints stay fully blended up to 143
degrees apart and are linear from 175. Dual quaternions are added on the
average's side, with a joint nearly a half turn from it counting for little
either way (SKIN_SIDE_SHARPNESS), so that sum has no jump either.

Weights are used as the file has them: a sum is divided by the vertex's total
once.

Morph targets move the bind position and normal before skinning.

Centers of rotation (`compute_cor`) are computed where a model is read
(assets/glb_native.mojo) and kept in cooked models: for vertex i,

    center_i = sum_t s(w_i, w_t) c_t a_t / sum_t s(w_i, w_t) a_t
    s(p, v)  = sum_{j<k} p_j p_k v_j v_k exp(-(p_j v_k - p_k v_j)^2 / sigma^2)

over triangles t (averaged weights w_t, centroid c_t, area a_t), sigma 0.1.
A vertex with one influence, or with no triangle like it, keeps its own
position as its center, which makes the method linear blend for it. Only
triangles that share a pair of joints with the vertex can contribute, so
they are listed per pair once and each vertex walks the lists of its pairs;
on Maroon's broad weights that is still 200 million terms for the 19,875
vertices of LOD0 (1.2 s on one core), so a big mesh's vertices are spread
over up to eight threads (0.14 s). Cutting small terms off was tried and
bought nothing: with weights this smooth almost every term matters.

Presentation only: nothing here is read by the simulation.
"""
from std.ffi import external_call
from std.math import sqrt, exp
from std.sys import size_of
from flint.core.math import Vec3
from flint.core.transform import Quat, Mat4
from flint.assets.model import NativeModel, NativeSkin

comptime SKIN_LBS: Int = 0
comptime SKIN_DQS: Int = 1
comptime SKIN_COR: Int = 2
comptime PALETTE_FLOATS: Int = 20
comptime PALETTE_QUAT: Int = 12
comptime PALETTE_DUAL: Int = 16
comptime COR_SIGMA: Float64 = 0.1
comptime COR_MAX_JOINTS: Int = 512
comptime COR_THREAD_WORK: Int = 4000000   # (vertex, triangle) terms before threads pay
comptime COR_MAX_THREADS: Int = 8
comptime RIGID_TOLERANCE: Float32 = 0.002
comptime SKIN_AVERAGE_STEPS: Int = 4
comptime SKIN_SIDE_SHARPNESS: Float32 = 4.0  # a quaternion within 1/4 of a right angle to the average counts in part
comptime SKIN_AGREE_FULL: Float32 = 0.66     # at least this share of the weight agrees: the blended rotation is used
comptime SKIN_AGREE_NONE: Float32 = 0.52     # this little: linear blend
comptime SKIN_MAX_MORPHS: Int = 8   # targets moving one vertex (assets/asset_check.mojo)


def skin_method_name(method: Int) -> String:
    if method == SKIN_LBS:
        return "lbs"
    if method == SKIN_DQS:
        return "dqs"
    return "cor"


def skin_method_from(value: Int) -> Int:
    """A method from a number someone typed or a manifest left out (linear blend)."""
    return max(SKIN_LBS, min(SKIN_COR, value))


def quat_from_matrix(m: Mat4) -> Quat:
    """The rotation of a matrix whose upper 3x3 is orthonormal."""
    var m00 = m.at(0, 0)
    var m11 = m.at(1, 1)
    var m22 = m.at(2, 2)
    var trace = m00 + m11 + m22
    if trace > 0:
        var s = sqrt(trace + 1.0) * 2.0
        return Quat((m.at(2, 1) - m.at(1, 2)) / s, (m.at(0, 2) - m.at(2, 0)) / s, (m.at(1, 0) - m.at(0, 1)) / s, 0.25 * s).normalized()
    if m00 > m11 and m00 > m22:
        var s = sqrt(max(Float32(0.000001), 1.0 + m00 - m11 - m22)) * 2.0
        return Quat(0.25 * s, (m.at(0, 1) + m.at(1, 0)) / s, (m.at(0, 2) + m.at(2, 0)) / s, (m.at(2, 1) - m.at(1, 2)) / s).normalized()
    if m11 > m22:
        var s = sqrt(max(Float32(0.000001), 1.0 + m11 - m00 - m22)) * 2.0
        return Quat((m.at(0, 1) + m.at(1, 0)) / s, 0.25 * s, (m.at(1, 2) + m.at(2, 1)) / s, (m.at(0, 2) - m.at(2, 0)) / s).normalized()
    var s = sqrt(max(Float32(0.000001), 1.0 + m22 - m00 - m11)) * 2.0
    return Quat((m.at(0, 2) + m.at(2, 0)) / s, (m.at(1, 2) + m.at(2, 1)) / s, 0.25 * s, (m.at(1, 0) - m.at(0, 1)) / s).normalized()


def matrix_rigid(m: Mat4) -> Bool:
    """A rotation and a translation, nothing else (within RIGID_TOLERANCE)."""
    var ax = Vec3(m.at(0, 0), m.at(1, 0), m.at(2, 0))
    var ay = Vec3(m.at(0, 1), m.at(1, 1), m.at(2, 1))
    var az = Vec3(m.at(0, 2), m.at(1, 2), m.at(2, 2))
    if abs(ax.length_squared() - 1.0) > RIGID_TOLERANCE or abs(ay.length_squared() - 1.0) > RIGID_TOLERANCE or abs(az.length_squared() - 1.0) > RIGID_TOLERANCE:
        return False
    var xy = ax.x * ay.x + ax.y * ay.y + ax.z * ay.z
    var xz = ax.x * az.x + ax.y * az.y + ax.z * az.z
    var yz = ay.x * az.x + ay.y * az.y + ay.z * az.z
    if abs(xy) > RIGID_TOLERANCE or abs(xz) > RIGID_TOLERANCE or abs(yz) > RIGID_TOLERANCE:
        return False
    # A mirrored joint is orthonormal too, and no rotation.
    var cx = ay.y * az.z - ay.z * az.y
    var cy = ay.z * az.x - ay.x * az.z
    var cz = ay.x * az.y - ay.y * az.x
    return ax.x * cx + ax.y * cy + ax.z * cz > 0


def write_palette_entry(address: Int, joint: Int, m: Mat4) -> Bool:
    """Joint `joint` of the palette at `address`. True if the matrix is rigid."""
    var out = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=address + joint * PALETTE_FLOATS * 4)
    for row in range(3):
        for column in range(4):
            out[unsafe_offset=row * 4 + column] = m.at(row, column)
    var q = quat_from_matrix(m)
    var tx = m.at(0, 3)
    var ty = m.at(1, 3)
    var tz = m.at(2, 3)
    out[unsafe_offset=PALETTE_QUAT] = q.x
    out[unsafe_offset=PALETTE_QUAT + 1] = q.y
    out[unsafe_offset=PALETTE_QUAT + 2] = q.z
    out[unsafe_offset=PALETTE_QUAT + 3] = q.w
    # The dual part: half of (translation, 0) x rotation.
    out[unsafe_offset=PALETTE_DUAL] = 0.5 * (tx * q.w + ty * q.z - tz * q.y)
    out[unsafe_offset=PALETTE_DUAL + 1] = 0.5 * (ty * q.w + tz * q.x - tx * q.z)
    out[unsafe_offset=PALETTE_DUAL + 2] = 0.5 * (tz * q.w + tx * q.y - ty * q.x)
    out[unsafe_offset=PALETTE_DUAL + 3] = -0.5 * (tx * q.x + ty * q.y + tz * q.z)
    return matrix_rigid(m)


def fill_palette(address: Int, skin: NativeSkin, transforms: List[Mat4]) -> Bool:
    """The palette of a skin in the pose of `transforms` (the model's global
    node transforms), at `address`: PALETTE_FLOATS floats a joint. True if
    every joint is rigid."""
    var rigid = True
    for joint_index in range(len(skin.joints)):
        var joint = skin.joints[joint_index]
        var matrix = Mat4()
        if joint >= 0 and joint < len(transforms):
            var inverse = Mat4()
            for component in range(16):
                var offset = joint_index * 16 + component
                if offset < len(skin.inverse_bind):
                    inverse.values[component] = skin.inverse_bind[offset]
            matrix = transforms[joint].multiply(inverse)
        rigid = write_palette_entry(address, joint_index, matrix) and rigid
    return rigid


def joint_pose(address: Int, skin: NativeSkin, transforms: List[Mat4]) -> List[Float32]:
    """Seven floats a joint: where it is in model space, and the rotation the
    palette at `address` holds for it. What a pose is compared by."""
    var pose = List[Float32](capacity=len(skin.joints) * 7)
    var palette = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=address)
    for joint_index in range(len(skin.joints)):
        var joint = skin.joints[joint_index]
        if joint >= 0 and joint < len(transforms):
            pose.append(transforms[joint].at(0, 3))
            pose.append(transforms[joint].at(1, 3))
            pose.append(transforms[joint].at(2, 3))
        else:
            pose.append(0)
            pose.append(0)
            pose.append(0)
        for component in range(4):
            pose.append(palette[unsafe_offset=joint_index * PALETTE_FLOATS + PALETTE_QUAT + component])
    return pose^


def effective_method(method: Int, rigid: Bool, has_centers: Bool) -> Int:
    """What a draw is skinned with: quaternions need rigid joints, and centers
    of rotation need the centers."""
    if not rigid:
        return SKIN_LBS
    if method == SKIN_COR and not has_centers:
        return SKIN_DQS
    return skin_method_from(method)


struct SkinDraw(Copyable, ImplicitlyCopyable, Movable):
    """One skinned draw, as the vertex jobs see it. Addresses, because the
    jobs read it from other threads while the draw waits."""
    var palette: Int        # joint_count x PALETTE_FLOATS floats
    var joint_count: Int
    var method: Int         # after `effective_method`
    var centers: Int        # three floats per vertex, or 0
    var morph_weights: Int  # morph_count floats, or 0
    var morph_count: Int
    var adopted: Int        # eight floats per vertex the GPU skinned (position, pad, normal, pad), or 0

    def __init__(out self):
        self.palette = 0
        self.joint_count = 0
        self.method = SKIN_LBS
        self.centers = 0
        self.morph_weights = 0
        self.morph_count = 0
        self.adopted = 0


@always_inline
def _cofactor_transform_normal(palette: Pointer[Float32, origin=MutAnyOrigin], m: Int, normal: Vec3) -> Vec3:
    """Transforms `normal` by the 3x3 cofactor matrix `cof(M_j) = det(M_j) * M_j^-T`
    whose rows are cross products of the 3x3 rows `(r1 x r2, r2 x r0, r0 x r1)`."""
    var r0 = Vec3(palette[unsafe_offset=m], palette[unsafe_offset=m + 1], palette[unsafe_offset=m + 2])
    var r1 = Vec3(palette[unsafe_offset=m + 4], palette[unsafe_offset=m + 5], palette[unsafe_offset=m + 6])
    var r2 = Vec3(palette[unsafe_offset=m + 8], palette[unsafe_offset=m + 9], palette[unsafe_offset=m + 10])
    var c0 = r1.cross(r2)
    var c1 = r2.cross(r0)
    var c2 = r0.cross(r1)
    return Vec3(c0.dot(normal), c1.dot(normal), c2.dot(normal))


def skin_vertex(draw: SkinDraw, position: Vec3, normal: Vec3, center: Vec3, joints_at: Int, weights_at: Int) -> Tuple[Vec3, Vec3]:
    """A bind-space position and normal in model space, in the draw's pose.
    Transforms normals via the 3x3 cofactor matrix `cof(M_j)` and normalizes."""
    var joints = Pointer[UInt16, origin=MutAnyOrigin](unsafe_from_address=joints_at)
    var weights = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=weights_at)
    var palette = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=draw.palette)
    var w = Array[Float32, 4](fill=0)
    var j = Array[Int, 4](fill=0)
    var total = Float32(0)
    for i in range(4):
        var weight = weights[unsafe_offset=i]
        var joint = Int(joints[unsafe_offset=i])
        if weight > 0 and joint < draw.joint_count:
            w[i] = weight
            j[i] = joint
            total += weight
    if total <= 0.00001:
        return (position, normal)
    var inverse = 1.0 / total
    if draw.method == SKIN_LBS:
        var sp = Vec3()
        var sn = Vec3()
        for i in range(4):
            if w[i] <= 0:
                continue
            var m = j[i] * PALETTE_FLOATS
            var p = Vec3(
                palette[unsafe_offset=m] * position.x + palette[unsafe_offset=m + 1] * position.y + palette[unsafe_offset=m + 2] * position.z + palette[unsafe_offset=m + 3],
                palette[unsafe_offset=m + 4] * position.x + palette[unsafe_offset=m + 5] * position.y + palette[unsafe_offset=m + 6] * position.z + palette[unsafe_offset=m + 7],
                palette[unsafe_offset=m + 8] * position.x + palette[unsafe_offset=m + 9] * position.y + palette[unsafe_offset=m + 10] * position.z + palette[unsafe_offset=m + 11],
            )
            var n = _cofactor_transform_normal(palette, m, normal)
            sp = sp + p * w[i]
            sn = sn + n * w[i]
        var sn_len = sn.length()
        var out_n = sn * (Float32(1.0) / sn_len) if sn_len > Float32(1.0e-6) else normal
        return (sp * inverse, out_n)
    # Linear blend of the vertex too: what the result fades to where the
    # joints' rotations have no sensible average.
    var linear = Vec3()
    var linear_normal = Vec3()
    var moved = Vec3()
    for i in range(4):
        if w[i] <= 0:
            continue
        var m = j[i] * PALETTE_FLOATS
        var p = Vec3(
            palette[unsafe_offset=m] * position.x + palette[unsafe_offset=m + 1] * position.y + palette[unsafe_offset=m + 2] * position.z + palette[unsafe_offset=m + 3],
            palette[unsafe_offset=m + 4] * position.x + palette[unsafe_offset=m + 5] * position.y + palette[unsafe_offset=m + 6] * position.z + palette[unsafe_offset=m + 7],
            palette[unsafe_offset=m + 8] * position.x + palette[unsafe_offset=m + 9] * position.y + palette[unsafe_offset=m + 10] * position.z + palette[unsafe_offset=m + 11],
        )
        var n = _cofactor_transform_normal(palette, m, normal)
        var c = Vec3(
            palette[unsafe_offset=m] * center.x + palette[unsafe_offset=m + 1] * center.y + palette[unsafe_offset=m + 2] * center.z + palette[unsafe_offset=m + 3],
            palette[unsafe_offset=m + 4] * center.x + palette[unsafe_offset=m + 5] * center.y + palette[unsafe_offset=m + 6] * center.z + palette[unsafe_offset=m + 7],
            palette[unsafe_offset=m + 8] * center.x + palette[unsafe_offset=m + 9] * center.y + palette[unsafe_offset=m + 10] * center.z + palette[unsafe_offset=m + 11],
        )
        linear = linear + p * w[i]
        linear_normal = linear_normal + n * w[i]
        moved = moved + c * w[i]
    linear = linear * inverse
    var ln_len = linear_normal.length()
    linear_normal = linear_normal * (Float32(1.0) / ln_len) if ln_len > Float32(1.0e-6) else normal
    moved = moved * inverse
    # The average rotation: the unit quaternion b that maximizes
    # sum w (q . b)^2, by power iteration from the heaviest influence's
    # quaternion. It does not depend on any quaternion's sign.
    var pivot = 0
    for i in range(1, 4):
        if w[i] > w[pivot]:
            pivot = i
    var pq = j[pivot] * PALETTE_FLOATS + PALETTE_QUAT
    var bx = palette[unsafe_offset=pq]
    var by = palette[unsafe_offset=pq + 1]
    var bz = palette[unsafe_offset=pq + 2]
    var bw = palette[unsafe_offset=pq + 3]
    var agreement = Float32(0)
    for _ in range(SKIN_AVERAGE_STEPS):
        var nx = Float32(0)
        var ny = Float32(0)
        var nz = Float32(0)
        var nw = Float32(0)
        for i in range(4):
            if w[i] <= 0:
                continue
            var q = j[i] * PALETTE_FLOATS + PALETTE_QUAT
            var qx = palette[unsafe_offset=q]
            var qy = palette[unsafe_offset=q + 1]
            var qz = palette[unsafe_offset=q + 2]
            var qw = palette[unsafe_offset=q + 3]
            var s = (bx * qx + by * qy + bz * qz + bw * qw) * w[i]
            nx = nx + qx * s
            ny = ny + qy * s
            nz = nz + qz * s
            nw = nw + qw * s
        var size = nx * nx + ny * ny + nz * nz + nw * nw
        if size < 1.0e-20:
            break
        var scale = 1.0 / sqrt(size)
        bx = nx * scale
        by = ny * scale
        bz = nz * scale
        bw = nw * scale
    # How much of the weight agrees with it: 1 when every joint turned alike,
    # one half or less when two rotations are as good as each other.
    var ex = Float32(0)
    var ey = Float32(0)
    var ez = Float32(0)
    var ew = Float32(0)
    var rx = Float32(0)
    var ry = Float32(0)
    var rz = Float32(0)
    var rw = Float32(0)
    for i in range(4):
        if w[i] <= 0:
            continue
        var q = j[i] * PALETTE_FLOATS + PALETTE_QUAT
        var qx = palette[unsafe_offset=q]
        var qy = palette[unsafe_offset=q + 1]
        var qz = palette[unsafe_offset=q + 2]
        var qw = palette[unsafe_offset=q + 3]
        var side = bx * qx + by * qy + bz * qz + bw * qw
        agreement = agreement + side * side * w[i]
        # Dual quaternions are added on the average's side; one nearly at right
        # angles to it (a half turn away) counts for little either way.
        var s = max(Float32(-1), min(Float32(1), side * SKIN_SIDE_SHARPNESS)) * w[i]
        rx = rx + qx * s
        ry = ry + qy * s
        rz = rz + qz * s
        rw = rw + qw * s
        ex = ex + palette[unsafe_offset=q + 4] * s
        ey = ey + palette[unsafe_offset=q + 5] * s
        ez = ez + palette[unsafe_offset=q + 6] * s
        ew = ew + palette[unsafe_offset=q + 7] * s
    agreement = agreement * inverse
    var fade = max(Float32(0), min(Float32(1), (SKIN_AGREE_FULL - agreement) * (1.0 / (SKIN_AGREE_FULL - SKIN_AGREE_NONE))))
    fade = fade * fade * (3.0 - 2.0 * fade)
    if draw.method == SKIN_DQS:
        var size = rx * rx + ry * ry + rz * rz + rw * rw
        if size < 0.00000001:
            return (linear, linear_normal)
        var scale = 1.0 / sqrt(size)
        bx = rx * scale
        by = ry * scale
        bz = rz * scale
        bw = rw * scale
        ex = ex * scale
        ey = ey * scale
        ez = ez * scale
        ew = ew * scale
    var xx = bx * bx
    var yy = by * by
    var zz = bz * bz
    var xy = bx * by
    var xz = bx * bz
    var yz = by * bz
    var wx = bw * bx
    var wy = bw * by
    var wz = bw * bz
    var r00 = 1.0 - 2.0 * (yy + zz)
    var r01 = 2.0 * (xy - wz)
    var r02 = 2.0 * (xz + wy)
    var r10 = 2.0 * (xy + wz)
    var r11 = 1.0 - 2.0 * (xx + zz)
    var r12 = 2.0 * (yz - wx)
    var r20 = 2.0 * (xz - wy)
    var r21 = 2.0 * (yz + wx)
    var r22 = 1.0 - 2.0 * (xx + yy)
    var turned_normal = Vec3(
        r00 * normal.x + r01 * normal.y + r02 * normal.z,
        r10 * normal.x + r11 * normal.y + r12 * normal.z,
        r20 * normal.x + r21 * normal.y + r22 * normal.z,
    )
    var out: Vec3
    if draw.method == SKIN_COR:
        var ox = position.x - center.x
        var oy = position.y - center.y
        var oz = position.z - center.z
        out = Vec3(
            r00 * ox + r01 * oy + r02 * oz + moved.x,
            r10 * ox + r11 * oy + r12 * oz + moved.y,
            r20 * ox + r21 * oy + r22 * oz + moved.z,
        )
    else:
        # The translation of a unit dual quaternion: twice dual x conjugate(real).
        var tx = 2.0 * (ex * bw - ew * bx - ey * bz + ez * by)
        var ty = 2.0 * (ey * bw - ew * by - ez * bx + ex * bz)
        var tz = 2.0 * (ez * bw - ew * bz - ex * by + ey * bx)
        out = Vec3(
            r00 * position.x + r01 * position.y + r02 * position.z + tx,
            r10 * position.x + r11 * position.y + r12 * position.z + ty,
            r20 * position.x + r21 * position.y + r22 * position.z + tz,
        )
    return (out + (linear - out) * fade, turned_normal + (linear_normal - turned_normal) * fade)


def morphed(position: Vec3, normal: Vec3, vertex: Int, vertex_count: Int, morph_positions: List[Float32], morph_normals: List[Float32], draw: SkinDraw) -> Tuple[Vec3, Vec3]:
    """The bind position and normal with the draw's morph weights applied."""
    var p = position
    var n = normal
    var weights = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=draw.morph_weights)
    for morph in range(draw.morph_count):
        var weight = weights[unsafe_offset=morph]
        if weight == 0:
            continue
        var at = (morph * vertex_count + vertex) * 3
        p = p + Vec3(morph_positions[at], morph_positions[at + 1], morph_positions[at + 2]) * weight
        n = n + Vec3(morph_normals[at], morph_normals[at + 1], morph_normals[at + 2]) * weight
    return (p, n)


struct CorWork(Copyable, Movable):
    """What the threads of `compute_cor` share: addresses of its lists."""
    var count: Int
    var joint_count: Int
    var step: Int
    var vj: Int
    var vw: Int
    var vn: Int
    var pair_start: Int
    var entry_triangle: Int
    var entry_low: Int
    var entry_high: Int
    var area: Int
    var centroid: Int
    var centers: Int

    def __init__(out self):
        self.count = 0
        self.joint_count = 0
        self.step = 1
        self.vj = 0
        self.vw = 0
        self.vn = 0
        self.pair_start = 0
        self.entry_triangle = 0
        self.entry_low = 0
        self.entry_high = 0
        self.area = 0
        self.centroid = 0
        self.centers = 0


@fieldwise_init
struct CorSlice(Copyable, ImplicitlyCopyable, Movable):
    var work: Int
    var first: Int


# Never inlined: this calls libm's exp, which must stay out of whatever
# simulation function reads a model (the machine-code audit, docs/DETERMINISM.md).
@no_inline
def cor_vertices(work: CorWork, first: Int):
    """The centers of vertices first, first + step, ..."""
    var vj = Pointer[Int32, origin=MutAnyOrigin](unsafe_from_address=work.vj)
    var vw = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=work.vw)
    var vn = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=work.vn)
    var pair_start = Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=work.pair_start)
    var entry_triangle = Pointer[Int32, origin=MutAnyOrigin](unsafe_from_address=work.entry_triangle)
    var entry_low = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=work.entry_low)
    var entry_high = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=work.entry_high)
    var area = Pointer[Float64, origin=MutAnyOrigin](unsafe_from_address=work.area)
    var centroid = Pointer[Float64, origin=MutAnyOrigin](unsafe_from_address=work.centroid)
    var centers = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=work.centers)
    var joint_count = work.joint_count
    var inverse_sigma = 1.0 / (COR_SIGMA * COR_SIGMA)
    for v in range(first, work.count, work.step):
        var n = Int(vn[unsafe_offset=v])
        if n < 2:
            continue
        var sum = Float64(0)
        var sx = Float64(0)
        var sy = Float64(0)
        var sz = Float64(0)
        for e in range(n):
            for f in range(e + 1, n):
                var je = Int(vj[unsafe_offset=v * 4 + e])
                var jf = Int(vj[unsafe_offset=v * 4 + f])
                var low = min(je, jf)
                var high = max(je, jf)
                var p_low = Float64(vw[unsafe_offset=v * 4 + e] if je < jf else vw[unsafe_offset=v * 4 + f])
                var p_high = Float64(vw[unsafe_offset=v * 4 + f] if je < jf else vw[unsafe_offset=v * 4 + e])
                var product = p_low * p_high
                var key = low * joint_count + high
                for at in range(pair_start[unsafe_offset=key], pair_start[unsafe_offset=key + 1]):
                    var v_low = Float64(entry_low[unsafe_offset=at])
                    var v_high = Float64(entry_high[unsafe_offset=at])
                    var difference = p_low * v_high - p_high * v_low
                    var x = difference * difference * inverse_sigma
                    if x > 30.0:
                        continue
                    var t = Int(entry_triangle[unsafe_offset=at])
                    var s = product * v_low * v_high * exp(-x) * area[unsafe_offset=t]
                    sum += s
                    sx += s * centroid[unsafe_offset=t * 3]
                    sy += s * centroid[unsafe_offset=t * 3 + 1]
                    sz += s * centroid[unsafe_offset=t * 3 + 2]
        if sum > 1.0e-30:
            centers[unsafe_offset=v * 3] = Float32(sx / sum)
            centers[unsafe_offset=v * 3 + 1] = Float32(sy / sum)
            centers[unsafe_offset=v * 3 + 2] = Float32(sz / sum)


def cor_thread(argument: Int) abi("C") -> Int:
    ref slice = Pointer[CorSlice, origin=MutAnyOrigin](unsafe_from_address=argument)[unsafe_offset=0]
    cor_vertices(Pointer[CorWork, origin=MutAnyOrigin](unsafe_from_address=slice.work)[unsafe_offset=0], slice.first)
    return 0


@no_inline
def compute_cor(positions: List[Float32], indices: List[Int], joints: List[UInt16], weights: List[Float32]) -> List[Float32]:
    """Every vertex's center of rotation, three floats each, in bind space."""
    var count = len(positions) // 3
    var centers = positions.copy()
    if count == 0 or len(weights) != count * 4 or len(joints) != count * 4:
        return centers^
    # Each vertex's influences: joints merged, weights summing to one.
    var vj = List[Int32](length=count * 4, fill=-1)
    var vw = List[Float32](length=count * 4, fill=0)
    var vn = List[UInt8](length=count, fill=0)
    var joint_count = 0
    for v in range(count):
        var total = Float32(0)
        var n = 0
        for i in range(4):
            var weight = weights[v * 4 + i]
            if weight <= 0:
                continue
            var joint = Int32(joints[v * 4 + i])
            total += weight
            var found = -1
            for k in range(n):
                if vj[v * 4 + k] == joint:
                    found = k
            if found < 0:
                found = n
                vj[v * 4 + n] = joint
                n += 1
            vw[v * 4 + found] += weight
            joint_count = max(joint_count, Int(joint) + 1)
        if total > 0.00001:
            for k in range(n):
                vw[v * 4 + k] = vw[v * 4 + k] / total
            vn[v] = UInt8(n)
    if joint_count < 2 or joint_count > COR_MAX_JOINTS:
        return centers^
    # Triangles: averaged weights (at most twelve joints), centroid, area.
    var triangles = len(indices) // 3
    var tj = List[Int32](length=triangles * 12, fill=-1)
    var tw = List[Float32](length=triangles * 12, fill=0)
    var tn = List[UInt8](length=triangles, fill=0)
    var area = List[Float64](length=triangles, fill=0)
    var centroid = List[Float64](length=triangles * 3, fill=0)
    var pair_start = List[Int](length=joint_count * joint_count + 1, fill=0)
    for t in range(triangles):
        var a = indices[t * 3]
        var b = indices[t * 3 + 1]
        var c = indices[t * 3 + 2]
        if a < 0 or b < 0 or c < 0 or a >= count or b >= count or c >= count:
            continue
        var ax = Float64(positions[a * 3])
        var ay = Float64(positions[a * 3 + 1])
        var az = Float64(positions[a * 3 + 2])
        var ux = Float64(positions[b * 3]) - ax
        var uy = Float64(positions[b * 3 + 1]) - ay
        var uz = Float64(positions[b * 3 + 2]) - az
        var wx = Float64(positions[c * 3]) - ax
        var wy = Float64(positions[c * 3 + 1]) - ay
        var wz = Float64(positions[c * 3 + 2]) - az
        var nx = uy * wz - uz * wy
        var ny = uz * wx - ux * wz
        var nz = ux * wy - uy * wx
        var size = 0.5 * sqrt(nx * nx + ny * ny + nz * nz)
        if size <= 0:
            continue
        area[t] = size
        centroid[t * 3] = ax + (ux + wx) / 3.0
        centroid[t * 3 + 1] = ay + (uy + wy) / 3.0
        centroid[t * 3 + 2] = az + (uz + wz) / 3.0
        var n = 0
        for corner in range(3):
            var v = indices[t * 3 + corner]
            for k in range(Int(vn[v])):
                var joint = vj[v * 4 + k]
                var found = -1
                for e in range(n):
                    if tj[t * 12 + e] == joint:
                        found = e
                if found < 0:
                    found = n
                    tj[t * 12 + n] = joint
                    n += 1
                tw[t * 12 + found] += vw[v * 4 + k] / 3.0
        tn[t] = UInt8(n)
        for e in range(n):
            for f in range(e + 1, n):
                var low = min(Int(tj[t * 12 + e]), Int(tj[t * 12 + f]))
                var high = max(Int(tj[t * 12 + e]), Int(tj[t * 12 + f]))
                pair_start[low * joint_count + high + 1] += 1
    for key in range(joint_count * joint_count):
        pair_start[key + 1] += pair_start[key]
    var entries = pair_start[joint_count * joint_count]
    var entry_triangle = List[Int32](length=entries, fill=0)
    var entry_low = List[Float32](length=entries, fill=0)
    var entry_high = List[Float32](length=entries, fill=0)
    var cursor = List[Int](length=joint_count * joint_count, fill=0)
    for t in range(triangles):
        var n = Int(tn[t])
        for e in range(n):
            for f in range(e + 1, n):
                var je = Int(tj[t * 12 + e])
                var jf = Int(tj[t * 12 + f])
                var key = min(je, jf) * joint_count + max(je, jf)
                var at = pair_start[key] + cursor[key]
                cursor[key] += 1
                entry_triangle[at] = Int32(t)
                entry_low[at] = tw[t * 12 + e] if je < jf else tw[t * 12 + f]
                entry_high[at] = tw[t * 12 + f] if je < jf else tw[t * 12 + e]
    var work = CorWork()
    work.count = count
    work.joint_count = joint_count
    work.vj = Int(vj.unsafe_ptr())
    work.vw = Int(vw.unsafe_ptr())
    work.vn = Int(vn.unsafe_ptr())
    work.pair_start = Int(pair_start.unsafe_ptr())
    work.entry_triangle = Int(entry_triangle.unsafe_ptr())
    work.entry_low = Int(entry_low.unsafe_ptr())
    work.entry_high = Int(entry_high.unsafe_ptr())
    work.area = Int(area.unsafe_ptr())
    work.centroid = Int(centroid.unsafe_ptr())
    work.centers = Int(centers.unsafe_ptr())
    # A big mesh is millions of (vertex, triangle) terms: spread the vertices
    # over a few threads. Each writes only its own vertices' centers, so the
    # result does not depend on how many there are.
    var threads = 1
    if count * (entries // max(joint_count, 1) + 1) > COR_THREAD_WORK:
        threads = max(1, min(COR_MAX_THREADS, Int(external_call["sysconf", Int64](Int32(84))) // 2))
    work.step = threads
    var slices = List[CorSlice]()
    for index in range(threads):
        slices.append(CorSlice(Int(Pointer(to=work)), index))
    var handles = List[UInt64]()
    for index in range(1, threads):
        var handle = UInt64(0)
        if external_call["pthread_create", Int32](Pointer(to=handle), Int(0), cor_thread, Int(slices.unsafe_ptr()) + index * size_of[CorSlice]()) == 0:
            handles.append(handle)
        else:
            cor_vertices(work, index)
    cor_vertices(work, 0)
    for handle in handles:
        _ = external_call["pthread_join", Int32](handle, Int(0))
    # The lists the threads read stay alive until here.
    _ = len(vj) + len(vw) + len(vn) + len(pair_start) + len(entry_triangle) + len(entry_low) + len(entry_high) + len(area) + len(centroid) + len(slices)
    return centers^


def skin_mesh(mut model: NativeModel, mesh_index: Int, method: Int) -> List[Float32]:
    """Every vertex of a skinned mesh in the model's pose now, on this thread:
    eight floats each (position, 0, normal, 0), the layout the GPU pass writes.
    For tests, tools and the benchmark; the game skins in vertex jobs."""
    var count = model.meshes[mesh_index].vertex_count()
    var out = List[Float32](length=count * 8, fill=0)
    var skin_index = model.meshes[mesh_index].skin
    if skin_index < 0 or skin_index >= len(model.skins) or len(model.meshes[mesh_index].weights) != count * 4:
        return out^
    var transforms = model.global_transforms()
    var palette = List[Float32](length=max(1, len(model.skins[skin_index].joints)) * PALETTE_FLOATS, fill=0)
    var draw = SkinDraw()
    draw.palette = Int(palette.unsafe_ptr())
    draw.joint_count = len(model.skins[skin_index].joints)
    var rigid = fill_palette(draw.palette, model.skins[skin_index], transforms)
    var has_centers = len(model.meshes[mesh_index].centers) == count * 3
    draw.method = effective_method(method, rigid, has_centers)
    var weights = model.morph_weights.copy()
    var morphs = model.meshes[mesh_index].morph_count
    if morphs > 0 and len(weights) >= morphs and len(model.meshes[mesh_index].morph_normals) == morphs * count * 3:
        draw.morph_count = morphs
        draw.morph_weights = Int(weights.unsafe_ptr())
    ref mesh = model.meshes[mesh_index]
    for vertex in range(count):
        var position = Vec3(mesh.positions[vertex * 3], mesh.positions[vertex * 3 + 1], mesh.positions[vertex * 3 + 2])
        var normal = Vec3(mesh.normals[vertex * 3], mesh.normals[vertex * 3 + 1], mesh.normals[vertex * 3 + 2])
        if draw.morph_count > 0:
            var moved = morphed(position, normal, vertex, count, mesh.morph_positions, mesh.morph_normals, draw)
            position = moved[0]
            normal = moved[1]
        var center = position
        if has_centers:
            center = Vec3(mesh.centers[vertex * 3], mesh.centers[vertex * 3 + 1], mesh.centers[vertex * 3 + 2])
        var result = skin_vertex(draw, position, normal, center, Int(mesh.joints.unsafe_ptr()) + vertex * 8, Int(mesh.weights.unsafe_ptr()) + vertex * 16)
        out[vertex * 8] = result[0].x
        out[vertex * 8 + 1] = result[0].y
        out[vertex * 8 + 2] = result[0].z
        out[vertex * 8 + 4] = result[1].x
        out[vertex * 8 + 5] = result[1].y
        out[vertex * 8 + 6] = result[1].z
    _ = len(palette) + len(weights)
    return out^


struct SkinStretch(Copyable, ImplicitlyCopyable, Movable):
    """How a skinned mesh's triangle edges compare with their bind lengths."""
    var worst: Float32      # the longest edge, in bind lengths
    var stretched: Int      # edges over twice their bind length
    var squashed: Int       # edges under half
    var edges: Int

    def __init__(out self):
        self.worst = 0
        self.stretched = 0
        self.squashed = 0
        self.edges = 0


def edge_stretch(model: NativeModel, mesh_index: Int, skinned: List[Float32]) -> SkinStretch:
    """Measures `skinned` (the layout `skin_mesh` gives) against the bind pose:
    what tools/blender/ds_stretch_check.py measures in Blender, here for any
    method. Edges under 2 mm at bind are left out."""
    var out = SkinStretch()
    var count = len(model.meshes[mesh_index].indices) // 3
    for triangle in range(count):
        for corner in range(3):
            var a = model.meshes[mesh_index].indices[triangle * 3 + corner]
            var b = model.meshes[mesh_index].indices[triangle * 3 + (corner + 1) % 3]
            var rx = model.meshes[mesh_index].positions[a * 3] - model.meshes[mesh_index].positions[b * 3]
            var ry = model.meshes[mesh_index].positions[a * 3 + 1] - model.meshes[mesh_index].positions[b * 3 + 1]
            var rz = model.meshes[mesh_index].positions[a * 3 + 2] - model.meshes[mesh_index].positions[b * 3 + 2]
            var rest = sqrt(rx * rx + ry * ry + rz * rz)
            if rest < 0.002:
                continue
            var dx = skinned[a * 8] - skinned[b * 8]
            var dy = skinned[a * 8 + 1] - skinned[b * 8 + 1]
            var dz = skinned[a * 8 + 2] - skinned[b * 8 + 2]
            var ratio = sqrt(dx * dx + dy * dy + dz * dz) / rest
            out.edges += 1
            out.worst = max(out.worst, ratio)
            if ratio > 2.0:
                out.stretched += 1
            if ratio < 0.5:
                out.squashed += 1
    return out
