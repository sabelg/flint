"""Phase 3 (`V18`): GPU Compute 4-Bone Dual-Quaternion Skinning, Exact 3D Skeletal
Motion Vectors & In-Place Hardware `BLAS` Refit (`aaa_skinning_rt.mojo`).

Implements:
  1. Volume-preserving Kavan **Dual-Quaternion (`DualQuat`)** bone conversion (`from_quat_trans`,
     `from_mat4`) & CPU reference evaluator (`skin_vertex_dq_cpu`) eliminating 180° candy-wrapper
     joint collapse via hemisphere antipodality sign alignment.
  2. GPU Compute Skinning Kernel (`aaa_dq_skinning_shader` authored in `ShaderDsl`) deforming
     `48`-byte bind-pose `(pos, normal, tangent)` directly into `BIND_AAA_VERTICES` (`BDA` buffer
     shared by `VisBuffer` AND `VK_KHR_acceleration_structure` `BLAS`!) while simultaneously writing
     previous-frame deformed positions for exact 3D `FlintTSR v2` skeletal motion vectors.
"""
from std.math import abs, max, min, sqrt
from std.memory import bitcast
from flint.assets.meshlet import GeomFreeList
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat
from flint.render.spirv import MODEL_GL_COMPUTE
from flint.render.shader_dsl import (
    GpuF32,
    GpuU32,
    GpuVec3,
    ShaderDsl,
    cross,
    dot,
    fma_f,
    max_f,
    rsqrt_f,
    select_f,
    select_u,
)

comptime AAA_MAX_SKIN_BONES: Int = 2048
comptime AAA_SKIN_CUR_BONES_BASE: Int = 0                                      # 2048 * 8 = 16,384 floats
comptime AAA_SKIN_PREV_BONES_BASE: Int = AAA_MAX_SKIN_BONES * 8                # 16,384 (2048 * 8 = 16,384 floats)
comptime AAA_SKIN_BIND_VERTS_BASE: Int = AAA_MAX_SKIN_BONES * 16               # 32,768: 16 words/vtx ([0..11]=bind, [12]=j4_u8, [13..15]=w0..w2)
comptime AAA_SKIN_BIND_STRIDE_WORDS: Int = 16
comptime AAA_SKIN_MAX_VERTS: Int = 98304
comptime AAA_SKIN_PREV_POS_BASE: Int = AAA_SKIN_BIND_VERTS_BASE + AAA_SKIN_MAX_VERTS * AAA_SKIN_BIND_STRIDE_WORDS  # 1,605,632
comptime AAA_SKIN_CUR_SMEAR_BASE: Int = AAA_SKIN_PREV_POS_BASE + AAA_SKIN_MAX_VERTS * 3                            # 1,900,544 (2048 * 8 floats)
comptime AAA_SKIN_PREV_SMEAR_BASE: Int = AAA_SKIN_CUR_SMEAR_BASE + AAA_MAX_SKIN_BONES * 8                          # 1,916,928 (2048 * 8 floats)
comptime AAA_SKIN_MAX_MORPH_SLOTS: Int = 256
comptime AAA_SKIN_MAX_MORPH_DELTAS: Int = 262144                                                                   # 6 floats (dx,dy,dz,dnx,dny,dnz) per (morph, vtx)
comptime AAA_SKIN_CUR_MORPH_WEIGHTS_BASE: Int = AAA_SKIN_PREV_SMEAR_BASE + AAA_MAX_SKIN_BONES * 8                  # 1,933,312
comptime AAA_SKIN_PREV_MORPH_WEIGHTS_BASE: Int = AAA_SKIN_CUR_MORPH_WEIGHTS_BASE + AAA_SKIN_MAX_MORPH_SLOTS        # 1,933,568
comptime AAA_SKIN_MORPH_DELTAS_BASE: Int = AAA_SKIN_PREV_MORPH_WEIGHTS_BASE + AAA_SKIN_MAX_MORPH_SLOTS             # 1,933,824
comptime AAA_SKIN_BUFFER_WORDS: Int = AAA_SKIN_MORPH_DELTAS_BASE + AAA_SKIN_MAX_MORPH_DELTAS * 6                   # 3,506,688 (~13.38 MiB)


struct FlintAaaSkinningArena(Copyable, Movable):
    """Tracks disjoint GPU bone-palette (`2,048` bones), bind-vertex (`98,304` verts) & morph-target sub-allocations.

    Cursors bump; `free_*` returns ranges to free lists that later allocations reuse first (first fit),
    and runs that end at a cursor pull the cursor back. The capacities themselves are fixed by the
    `BIND_AAA_SKINNING` layout constants, which the skinning and SVGF shaders bake in."""
    var bone_cursor: Int
    var vertex_cursor: Int
    var morph_weight_cursor: Int
    var morph_delta_cursor: Int
    var free_bones: GeomFreeList  # in bone slots
    var free_vertices: GeomFreeList  # in bind-vertex slots
    var free_morph_weights: GeomFreeList  # in morph slots
    var free_morph_deltas: GeomFreeList  # in (morph, vertex) delta slots
    var failed_allocations: Int  # requests that did not fit (clamped or refused)

    def __init__(out self):
        self.bone_cursor = 0
        self.vertex_cursor = 0
        self.morph_weight_cursor = 0
        self.morph_delta_cursor = 0
        self.free_bones = GeomFreeList()
        self.free_vertices = GeomFreeList()
        self.free_morph_weights = GeomFreeList()
        self.free_morph_deltas = GeomFreeList()
        self.failed_allocations = 0

    def reset(mut self):
        self.bone_cursor = 0
        self.vertex_cursor = 0
        self.morph_weight_cursor = 0
        self.morph_delta_cursor = 0
        self.free_bones = GeomFreeList()
        self.free_vertices = GeomFreeList()
        self.free_morph_weights = GeomFreeList()
        self.free_morph_deltas = GeomFreeList()

    def vertices_in_use(self) -> Int:
        return self.vertex_cursor - self.free_vertices.total()

    def bones_in_use(self) -> Int:
        return self.bone_cursor - self.free_bones.total()

    def free_vertices_range(mut self, first_slot: Int, count: Int):
        if count <= 0 or first_slot < 0:
            return
        self.free_vertices.free(first_slot, count)
        self.vertex_cursor = self.free_vertices.trim_tail(self.vertex_cursor)

    def free_bone_palette(mut self, bone_palette_offset: Int, bone_count: Int):
        """Frees the bone slots a palette (word offset `slot * 8`) used."""
        var eff = max(1, min(AAA_MAX_SKIN_BONES, bone_count))
        var slot = bone_palette_offset // 8
        if slot < 0:
            return
        eff = min(eff, max(0, AAA_MAX_SKIN_BONES - slot))
        self.free_bones.free(slot, eff)
        self.bone_cursor = self.free_bones.trim_tail(self.bone_cursor)

    def free_morph_targets(mut self, first_weight: Int, first_delta: Int, morph_count: Int, vertex_count: Int):
        if morph_count <= 0:
            return
        self.free_morph_weights.free(first_weight, morph_count)
        self.morph_weight_cursor = self.free_morph_weights.trim_tail(self.morph_weight_cursor)
        if vertex_count > 0:
            self.free_morph_deltas.free(first_delta, morph_count * vertex_count)
            self.morph_delta_cursor = self.free_morph_deltas.trim_tail(self.morph_delta_cursor)

    def alloc_bone_palette(mut self, bone_count: Int = 64, explicit_word_offset: Int = -1) -> Int:
        """Reserve a complete disjoint palette, or report exhaustion without overlapping a live one."""
        var count = max(1, bone_count)
        var slot = self.bone_cursor
        if explicit_word_offset >= 0:
            slot = explicit_word_offset // 8
            if explicit_word_offset % 8 != 0 or slot < self.bone_cursor:
                self.failed_allocations += 1
                return -1
        else:
            var reused = self.free_bones.alloc(count)
            if reused >= 0:
                return reused * 8
        if count > AAA_MAX_SKIN_BONES - slot:
            self.failed_allocations += 1
            return -1
        self.bone_cursor = slot + count
        return slot * 8

    def alloc_vertices(mut self, vertex_count: Int) -> Tuple[Int, Int]:
        """Reuse or reserve a complete mesh. Zero vertices signals failure."""
        if vertex_count <= 0:
            return (self.vertex_cursor, 0)
        var reused = self.free_vertices.alloc(vertex_count)
        if reused >= 0:
            return (reused, vertex_count)
        var slot = self.vertex_cursor
        if vertex_count > AAA_SKIN_MAX_VERTS - slot:
            self.failed_allocations += 1
            return (slot, 0)
        self.vertex_cursor += vertex_count
        return (slot, vertex_count)

    def alloc_morph_targets(mut self, morph_count: Int, vertex_count: Int) -> Tuple[Int, Int, Int]:
        """Allocates `(first_weight_slot, first_delta_slot, effective_morph_count)` clamped within
        `AAA_SKIN_MAX_MORPH_SLOTS` and `AAA_SKIN_MAX_MORPH_DELTAS`."""
        var first_w = self.morph_weight_cursor
        var first_d = self.morph_delta_cursor
        if morph_count <= 0 or vertex_count <= 0:
            return (first_w, first_d, 0)
        var rw = self.free_morph_weights.alloc(morph_count)
        if rw >= 0:
            var rd = self.free_morph_deltas.alloc(morph_count * vertex_count)
            if rd >= 0:
                return (rw, rd, morph_count)
            self.free_morph_weights.free(rw, morph_count)
        var avail_w = max(0, AAA_SKIN_MAX_MORPH_SLOTS - first_w)
        var avail_d_morphs = max(0, (AAA_SKIN_MAX_MORPH_DELTAS - first_d) // vertex_count)
        var eff_m = max(0, min(morph_count, min(avail_w, avail_d_morphs)))
        self.morph_weight_cursor += eff_m
        self.morph_delta_cursor += eff_m * vertex_count
        return (first_w, first_d, eff_m)


@fieldwise_init
struct DualQuat(Copyable, ImplicitlyCopyable, Movable):
    """Scale-aware Kavan Dual Quaternion `s * (q_real + eps * q_dual)` (`s = ||q_real||`, `1.0` for rigid unit rigs)."""
    var rx: Float32
    var ry: Float32
    var rz: Float32
    var rw: Float32
    var dx: Float32
    var dy: Float32
    var dz: Float32
    var dw: Float32

    def __init__(out self):
        self.rx = 0.0; self.ry = 0.0; self.rz = 0.0; self.rw = 1.0
        self.dx = 0.0; self.dy = 0.0; self.dz = 0.0; self.dw = 0.0

    @staticmethod
    def from_quat_trans(rot: Quat, trans: Vec3, scale: Float32 = 1.0) -> Self:
        var q = rot.normalized()
        var tx = trans.x; var ty = trans.y; var tz = trans.z
        var s_eff = Float32(1.0) if abs(scale - Float32(1.0)) < Float32(1e-6) else max(Float32(1e-6), scale)
        # q_dual = 0.5 * (tx, ty, tz, 0) * (qx, qy, qz, qw), scaled by s_eff so q_dual / ||q_real|| stays exact
        var dex = Float32(0.5) * (tx * q.w + ty * q.z - tz * q.y) * s_eff
        var dey = Float32(0.5) * (-tx * q.z + ty * q.w + tz * q.x) * s_eff
        var dez = Float32(0.5) * (tx * q.y - ty * q.x + tz * q.w) * s_eff
        var dew = Float32(0.5) * (-tx * q.x - ty * q.y - tz * q.z) * s_eff
        return Self(q.x * s_eff, q.y * s_eff, q.z * s_eff, q.w * s_eff, dex, dey, dez, dew)

    @staticmethod
    def from_quat_trans_scale(rot: Quat, trans: Vec3, scale: Float32) -> Self:
        return Self.from_quat_trans(rot, trans, scale)

    @staticmethod
    def from_mat4(m: Mat4) -> Self:
        """Extracts orthonormal rotation quaternion + uniform scale + translation from a column-major `Mat4`."""
        var raw_c0 = Vec3(m.values[0], m.values[1], m.values[2])
        var raw_c1 = Vec3(m.values[4], m.values[5], m.values[6])
        var raw_c2 = Vec3(m.values[8], m.values[9], m.values[10])
        var sx = raw_c0.length()
        var sy = raw_c1.length()
        var sz = raw_c2.length()
        var uni_s = max(Float32(1e-6), sx * sy * sz) ** Float32(1.0 / 3.0)
        if abs(uni_s - Float32(1.0)) < Float32(1e-5):
            uni_s = Float32(1.0)

        var c0 = raw_c0.normalized() if sx > Float32(1e-8) else Vec3(1.0, 0.0, 0.0)
        var c1 = raw_c1.normalized() if sy > Float32(1e-8) else Vec3(0.0, 1.0, 0.0)
        var c2 = c0.cross(c1).normalized()
        c1 = c2.cross(c0).normalized()

        var m00 = c0.x; var m10 = c0.y; var m20 = c0.z
        var m01 = c1.x; var m11 = c1.y; var m21 = c1.z
        var m02 = c2.x; var m12 = c2.y; var m22 = c2.z

        var tr = m00 + m11 + m22
        var q: Quat
        if tr > Float32(0.0):
            var s = sqrt(tr + Float32(1.0)) * Float32(2.0)
            q = Quat((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, Float32(0.25) * s)
        elif m00 > m11 and m00 > m22:
            var s = sqrt(Float32(1.0) + m00 - m11 - m22) * Float32(2.0)
            q = Quat(Float32(0.25) * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
        elif m11 > m22:
            var s = sqrt(Float32(1.0) + m11 - m00 - m22) * Float32(2.0)
            q = Quat((m01 + m10) / s, Float32(0.25) * s, (m12 + m21) / s, (m02 - m20) / s)
        else:
            var s = sqrt(Float32(1.0) + m22 - m00 - m11) * Float32(2.0)
            q = Quat((m02 + m20) / s, (m12 + m21) / s, Float32(0.25) * s, (m10 - m01) / s)

        return Self.from_quat_trans(q, Vec3(m.values[12], m.values[13], m.values[14]), uni_s)

    def uniform_scale(self) -> Float32:
        var r_vec = Vec3(self.rx, self.ry, self.rz)
        return sqrt(max(Float32(1e-12), r_vec.length_squared() + self.rw * self.rw))

    def transform_point(self, p: Vec3) -> Vec3:
        var r_vec = Vec3(self.rx, self.ry, self.rz)
        var d_vec = Vec3(self.dx, self.dy, self.dz)
        var len2 = max(Float32(1e-12), r_vec.length_squared() + self.rw * self.rw)
        var inv_len = Float32(1.0) / sqrt(len2)
        var s_len = len2 * inv_len
        var rv = r_vec * inv_len
        var rw = self.rw * inv_len
        var dv = d_vec * inv_len
        var dw = self.dw * inv_len
        var sp = p * s_len
        var rot_p = sp + rv.cross(rv.cross(sp) + sp * rw) * Float32(2.0)
        var trans = (dv * rw - rv * dw + rv.cross(dv)) * Float32(2.0)
        return rot_p + trans

    def transform_dir(self, dir_v: Vec3) -> Vec3:
        var r_vec = Vec3(self.rx, self.ry, self.rz)
        var inv_len = Float32(1.0) / sqrt(max(Float32(1e-12), r_vec.length_squared() + self.rw * self.rw))
        var rv = r_vec * inv_len
        var rw = self.rw * inv_len
        return (dir_v + rv.cross(rv.cross(dir_v) + dir_v * rw) * Float32(2.0)).normalized()

    def write_words(self, mut out: List[UInt32], offset_words: Int):
        out[offset_words + 0] = bitcast[DType.uint32](self.rx)
        out[offset_words + 1] = bitcast[DType.uint32](self.ry)
        out[offset_words + 2] = bitcast[DType.uint32](self.rz)
        out[offset_words + 3] = bitcast[DType.uint32](self.rw)
        out[offset_words + 4] = bitcast[DType.uint32](self.dx)
        out[offset_words + 5] = bitcast[DType.uint32](self.dy)
        out[offset_words + 6] = bitcast[DType.uint32](self.dz)
        out[offset_words + 7] = bitcast[DType.uint32](self.dw)


def normalize_and_sort_skin_weights_4(
    j0: Int,
    j1: Int,
    j2: Int,
    j3: Int,
    w0: Float32,
    w1: Float32,
    w2: Float32,
    w3: Float32,
) -> Tuple[UInt32, Float32, Float32, Float32, Float32]:
    """Coalesces duplicate joints, sorts `(joint, weight)` in descending weight order (`w0 >= w1 >= w2 >= w3`
    so `j0` is guaranteed to be the dominant hemisphere pivot), and normalizes weights to sum to `1.0`."""
    var in_j = [j0 & 0xFF, j1 & 0xFF, j2 & 0xFF, j3 & 0xFF]
    var in_w = [max(Float32(0.0), w0), max(Float32(0.0), w1), max(Float32(0.0), w2), max(Float32(0.0), w3)]

    var cj = [0, 0, 0, 0]
    var cw = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
    var n_act = 0
    for i in range(4):
        if in_w[i] <= Float32(0.0):
            continue
        var merged = False
        for m in range(n_act):
            if cj[m] == in_j[i]:
                cw[m] += in_w[i]
                merged = True
                break
        if not merged:
            cj[n_act] = in_j[i]
            cw[n_act] = in_w[i]
            n_act += 1

    # Insertion sort descending by weight
    for a in range(1, n_act):
        var kj = cj[a]
        var kw = cw[a]
        var b = a - 1
        while b >= 0 and cw[b] < kw:
            cj[b + 1] = cj[b]
            cw[b + 1] = cw[b]
            b -= 1
        cj[b + 1] = kj
        cw[b + 1] = kw

    var sum_w = cw[0] + cw[1] + cw[2] + cw[3]
    if sum_w <= Float32(1e-6):
        return (UInt32(0), Float32(1.0), Float32(0.0), Float32(0.0), Float32(0.0))

    var inv_s = Float32(1.0) / sum_w
    var nw0 = cw[0] * inv_s
    var nw1 = cw[1] * inv_s
    var nw2 = cw[2] * inv_s
    var nw3 = cw[3] * inv_s
    if n_act < 4:
        nw3 = Float32(0.0)
        if n_act == 1:
            nw0 = Float32(1.0); nw1 = Float32(0.0); nw2 = Float32(0.0)
        elif n_act == 2:
            nw2 = Float32(0.0)
            nw0 = max(Float32(0.0), Float32(1.0) - nw1)
        else:
            nw0 = max(Float32(0.0), Float32(1.0) - (nw1 + nw2))

    # Point zero-weight slots at dominant `cj[0]` so unused lanes never touch root bone `0`
    var dom_j = cj[0]
    for z in range(n_act, 4):
        cj[z] = dom_j

    var packed_j = (
        UInt32(cj[0] & 0xFF)
        | (UInt32(cj[1] & 0xFF) << 8)
        | (UInt32(cj[2] & 0xFF) << 16)
        | (UInt32(cj[3] & 0xFF) << 24)
    )
    return (packed_j, nw0, nw1, nw2, nw3)


def blend_dual_quats_cpu(
    bones: List[DualQuat],
    j0: Int,
    j1: Int,
    j2: Int,
    j3: Int,
    w0: Float32,
    w1: Float32,
    w2: Float32,
    w3: Float32,
) -> DualQuat:
    """CPU reference for scale-aware 4-bone antipodality-corrected Dual-Quaternion skinning (`DLB`)."""
    var js = [j0, j1, j2, j3]
    var ws = [max(Float32(0.0), w0), max(Float32(0.0), w1), max(Float32(0.0), w2), max(Float32(0.0), w3)]

    var piv_k = 0
    for k in range(1, 4):
        if ws[k] > ws[piv_k]:
            piv_k = k
    var bp = bones[js[piv_k]]
    var lp = sqrt(max(Float32(1e-12), bp.rx * bp.rx + bp.ry * bp.ry + bp.rz * bp.rz + bp.rw * bp.rw))
    var prx = bp.rx / lp; var pry = bp.ry / lp; var prz = bp.rz / lp; var prw = bp.rw / lp

    var arx = Float32(0.0); var ary = Float32(0.0); var arz = Float32(0.0); var arw = Float32(0.0)
    var adx = Float32(0.0); var ady = Float32(0.0); var adz = Float32(0.0); var adw = Float32(0.0)
    var s_acc = Float32(0.0)
    var w_sum = Float32(0.0)

    for k in range(4):
        var w = ws[k]
        if w <= Float32(0.0):
            continue
        var b = bones[js[k]]
        var lk = sqrt(max(Float32(1e-12), b.rx * b.rx + b.ry * b.ry + b.rz * b.rz + b.rw * b.rw))
        var inv_lk = Float32(1.0) / lk
        var urx = b.rx * inv_lk; var ury = b.ry * inv_lk; var urz = b.rz * inv_lk; var urw = b.rw * inv_lk
        var udx = b.dx * inv_lk; var udy = b.dy * inv_lk; var udz = b.dz * inv_lk; var udw = b.dw * inv_lk
        var dot_q = prx * urx + pry * ury + prz * urz + prw * urw
        var s_w = -w if dot_q < Float32(0.0) else w
        arx += s_w * urx; ary += s_w * ury; arz += s_w * urz; arw += s_w * urw
        adx += s_w * udx; ady += s_w * udy; adz += s_w * udz; adw += s_w * udw
        s_acc += w * lk
        w_sum += w

    if w_sum <= Float32(1e-6):
        return DualQuat()
    var s_blend = s_acc / w_sum
    var inv_al = Float32(1.0) / sqrt(max(Float32(1e-12), arx * arx + ary * ary + arz * arz + arw * arw))
    var k_out = inv_al * s_blend
    return DualQuat(
        arx * k_out, ary * k_out, arz * k_out, arw * k_out,
        adx * k_out, ady * k_out, adz * k_out, adw * k_out,
    )


def skin_vertex_dq_cpu(
    bones: List[DualQuat],
    j0: Int,
    j1: Int,
    j2: Int,
    j3: Int,
    w0: Float32,
    w1: Float32,
    w2: Float32,
    w3: Float32,
    bind_pos: Vec3,
    bind_nrm: Vec3 = Vec3(0.0, 1.0, 0.0),
) -> Tuple[Vec3, Vec3]:
    """CPU reference matching `add_skinned_mesh` normalization/sort + `aaa_dq_skinning_shader`."""
    var ns = normalize_and_sort_skin_weights_4(j0, j1, j2, j3, w0, w1, w2, w3)
    var jp = ns[0]
    var sj0 = Int(jp & UInt32(0xFF))
    var sj1 = Int((jp >> UInt32(8)) & UInt32(0xFF))
    var sj2 = Int((jp >> UInt32(16)) & UInt32(0xFF))
    var sj3 = Int((jp >> UInt32(24)) & UInt32(0xFF))
    var sw3 = max(Float32(0.0), Float32(1.0) - (ns[1] + ns[2] + ns[3]))
    var dq = blend_dual_quats_cpu(bones, sj0, sj1, sj2, sj3, ns[1], ns[2], ns[3], sw3)
    return (dq.transform_point(bind_pos), dq.transform_dir(bind_nrm))


struct BoneSmear(Copyable, ImplicitlyCopyable, Movable):
    """Non-uniform 3D affine smear-frame bone stretch, squash, and directional shear around `pivot_bind`.
    Stored in `BIND_AAA_SKINNING` as `(scale - 1.0, shear_xy, pivot_bind, shear_xz)` so zeroed memory is exact identity!
    """
    var scale: Vec3
    var shear_xy: Float32
    var pivot_bind: Vec3
    var shear_xz: Float32

    def __init__(
        out self,
        scale: Vec3 = Vec3(1.0, 1.0, 1.0),
        shear_xy: Float32 = 0.0,
        pivot_bind: Vec3 = Vec3(0.0, 0.0, 0.0),
        shear_xz: Float32 = 0.0,
    ):
        self.scale = Vec3(
            max(Float32(0.05), scale.x),
            max(Float32(0.05), scale.y),
            max(Float32(0.05), scale.z),
        )
        self.shear_xy = shear_xy
        self.pivot_bind = pivot_bind
        self.shear_xz = shear_xz

    def write_words(self, mut out: List[UInt32], offset_words: Int):
        out[offset_words + 0] = bitcast[DType.uint32](self.scale.x - Float32(1.0))
        out[offset_words + 1] = bitcast[DType.uint32](self.scale.y - Float32(1.0))
        out[offset_words + 2] = bitcast[DType.uint32](self.scale.z - Float32(1.0))
        out[offset_words + 3] = bitcast[DType.uint32](self.shear_xy)
        out[offset_words + 4] = bitcast[DType.uint32](self.pivot_bind.x)
        out[offset_words + 5] = bitcast[DType.uint32](self.pivot_bind.y)
        out[offset_words + 6] = bitcast[DType.uint32](self.pivot_bind.z)
        out[offset_words + 7] = bitcast[DType.uint32](self.shear_xz)


def skin_vertex_dq_smear_cpu(
    bones: List[DualQuat],
    smears: List[BoneSmear],
    j0: Int,
    j1: Int,
    j2: Int,
    j3: Int,
    w0: Float32,
    w1: Float32,
    w2: Float32,
    w3: Float32,
    bind_pos: Vec3,
    bind_nrm: Vec3 = Vec3(0.0, 1.0, 0.0),
) -> Tuple[Vec3, Vec3]:
    """CPU reference matching `aaa_dq_skinning_shader` with non-uniform `BoneSmear` deformation + `J^{-T}` normals."""
    var ns = normalize_and_sort_skin_weights_4(j0, j1, j2, j3, w0, w1, w2, w3)
    var jp = ns[0]
    var sj = [
        Int(jp & UInt32(0xFF)),
        Int((jp >> UInt32(8)) & UInt32(0xFF)),
        Int((jp >> UInt32(16)) & UInt32(0xFF)),
        Int((jp >> UInt32(24)) & UInt32(0xFF)),
    ]
    var sw3 = max(Float32(0.0), Float32(1.0) - (ns[1] + ns[2] + ns[3]))
    var sw = [ns[1], ns[2], ns[3], sw3]

    var ds = Vec3(0.0, 0.0, 0.0)
    var sh_xy = Float32(0.0)
    var sh_xz = Float32(0.0)
    var piv = Vec3(0.0, 0.0, 0.0)
    for k in range(4):
        var wk = sw[k]
        if wk > Float32(0.0) and sj[k] < len(smears):
            var sm = smears[sj[k]]
            ds = ds + (sm.scale - Vec3(1.0, 1.0, 1.0)) * wk
            sh_xy += sm.shear_xy * wk
            sh_xz += sm.shear_xz * wk
            piv = piv + sm.pivot_bind * wk

    var sx = max(Float32(0.05), Float32(1.0) + ds.x)
    var sy = max(Float32(0.05), Float32(1.0) + ds.y)
    var sz = max(Float32(0.05), Float32(1.0) + ds.z)
    var rel = bind_pos - piv
    var sm_pos = piv + Vec3(
        rel.x * sx + sh_xy * rel.y + sh_xz * rel.z,
        rel.y * sy,
        rel.z * sz,
    )

    var inv_sx = Float32(1.0) / sx
    var inv_sy = Float32(1.0) / sy
    var inv_sz = Float32(1.0) / sz
    var nx_s = bind_nrm.x * inv_sx
    var ny_s = (bind_nrm.y - sh_xy * nx_s) * inv_sy
    var nz_s = (bind_nrm.z - sh_xz * nx_s) * inv_sz
    var sm_nrm = Vec3(nx_s, ny_s, nz_s).normalized()

    var dq = blend_dual_quats_cpu(bones, sj[0], sj[1], sj[2], sj[3], sw[0], sw[1], sw[2], sw[3])
    return (dq.transform_point(sm_pos), dq.transform_dir(sm_nrm))


def skin_vertex_morph_dq_smear_cpu(
    bones: List[DualQuat],
    smears: List[BoneSmear],
    morph_weights: List[Float32],
    morph_pos_deltas: List[Vec3],
    morph_nrm_deltas: List[Vec3],
    j0: Int,
    j1: Int,
    j2: Int,
    j3: Int,
    w0: Float32,
    w1: Float32,
    w2: Float32,
    w3: Float32,
    bind_pos: Vec3,
    bind_nrm: Vec3 = Vec3(0.0, 1.0, 0.0),
) -> Tuple[Vec3, Vec3]:
    """CPU reference matching `aaa_dq_skinning_shader` with pre-skinning `BlendShape` morph targets + `BoneSmear` + `DualQuat`."""
    var m_cnt = min(len(morph_weights), len(morph_pos_deltas))
    var mp = bind_pos
    var mn = bind_nrm
    if m_cnt > 0:
        for m in range(m_cnt):
            var mw = morph_weights[m]
            mp = mp + morph_pos_deltas[m] * mw
            if m < len(morph_nrm_deltas):
                mn = mn + morph_nrm_deltas[m] * mw
        var mn_l2 = max(Float32(1e-12), mn.length_squared())
        mn = mn * (Float32(1.0) / sqrt(mn_l2))
    return skin_vertex_dq_smear_cpu(bones, smears, j0, j1, j2, j3, w0, w1, w2, w3, mp, mn)


def _emit_dq_blend_and_transform_dsl(
    mut dsl: ShaderDsl,
    b_skin_buf_id: UInt32,
    palette_base: GpuU32,
    smear_base: GpuU32,
    j0: GpuU32,
    j1: GpuU32,
    j2: GpuU32,
    j3: GpuU32,
    w0: GpuF32,
    w1: GpuF32,
    w2: GpuF32,
    w3: GpuF32,
    bind_pos: GpuVec3,
    bind_nrm: GpuVec3,
    bind_tan: GpuVec3,
) -> Tuple[GpuVec3, GpuVec3, GpuVec3]:
    """Blends 4 scale-aware bone Dual Quaternions + non-uniform `BoneSmear` affine deformations in `ShaderDsl`."""
    from flint.render.shader_dsl import BufferRef
    var b_skin = BufferRef[1](dsl.mod_addr, b_skin_buf_id)
    var pal = palette_base

    # Blend 4-bone non-uniform `BoneSmear` (`(scale - 1, shear_xy, pivot_bind, shear_xz)` -> 0 = exact identity)
    var sm0 = smear_base + j0 * UInt32(8)
    var acc_ds = b_skin.load_vec3(sm0) * w0
    var acc_sh_xy = b_skin.load_f32(sm0 + UInt32(3)) * w0
    var acc_piv = b_skin.load_vec3(sm0 + UInt32(4)) * w0
    var acc_sh_xz = b_skin.load_f32(sm0 + UInt32(7)) * w0

    var o0 = pal + j0 * UInt32(8)
    var r0_raw = b_skin.load_vec3(o0)
    var rw0_raw = b_skin.load_f32(o0 + UInt32(3))
    var d0_raw = b_skin.load_vec3(o0 + UInt32(4))
    var dw0_raw = b_skin.load_f32(o0 + UInt32(7))
    var l2_0 = max_f(fma_f(rw0_raw, rw0_raw, dot(r0_raw, r0_raw)), 1e-12)
    var inv_s0 = rsqrt_f(l2_0)
    var s0 = l2_0 * inv_s0
    var r0 = r0_raw * inv_s0
    var rw0 = rw0_raw * inv_s0
    var d0 = d0_raw * inv_s0
    var dw0 = dw0_raw * inv_s0

    var acc_r = r0 * w0
    var acc_rw = rw0 * w0
    var acc_d = d0 * w0
    var acc_dw = dw0 * w0
    var s_blend = s0 * w0

    var j_arr = [j1, j2, j3]
    var w_arr = [w1, w2, w3]
    for k in range(3):
        var wk = w_arr[k]
        var smk = smear_base + j_arr[k] * UInt32(8)
        acc_ds = acc_ds + b_skin.load_vec3(smk) * wk
        acc_sh_xy = fma_f(b_skin.load_f32(smk + UInt32(3)), wk, acc_sh_xy)
        acc_piv = acc_piv + b_skin.load_vec3(smk + UInt32(4)) * wk
        acc_sh_xz = fma_f(b_skin.load_f32(smk + UInt32(7)), wk, acc_sh_xz)

        var ok = pal + j_arr[k] * UInt32(8)
        var rk_raw = b_skin.load_vec3(ok)
        var rwk_raw = b_skin.load_f32(ok + UInt32(3))
        var dk_raw = b_skin.load_vec3(ok + UInt32(4))
        var dwk_raw = b_skin.load_f32(ok + UInt32(7))
        var l2_k = max_f(fma_f(rwk_raw, rwk_raw, dot(rk_raw, rk_raw)), 1e-12)
        var inv_sk = rsqrt_f(l2_k)
        var sk = l2_k * inv_sk
        var rk = rk_raw * inv_sk
        var rwk = rwk_raw * inv_sk
        var dk = dk_raw * inv_sk
        var dwk = dwk_raw * inv_sk
        var qdot = fma_f(rw0, rwk, dot(r0, rk))
        var sw = select_f(qdot.lt(0.0), -wk, wk)
        acc_r = acc_r + rk * sw
        acc_rw = acc_rw + rwk * sw
        acc_d = acc_d + dk * sw
        acc_dw = acc_dw + dwk * sw
        s_blend = fma_f(sk, wk, s_blend)

    var sx = max_f(acc_ds.x() + Float32(1.0), 0.05)
    var sy = max_f(acc_ds.y() + Float32(1.0), 0.05)
    var sz = max_f(acc_ds.z() + Float32(1.0), 0.05)
    var rel_p = bind_pos - acc_piv
    var sm_pos = acc_piv + GpuVec3(
        fma_f(acc_sh_xz, rel_p.z(), fma_f(acc_sh_xy, rel_p.y(), rel_p.x() * sx)),
        rel_p.y() * sy,
        rel_p.z() * sz,
    )

    var inv_sx = Float32(1.0) / sx
    var inv_sy = Float32(1.0) / sy
    var inv_sz = Float32(1.0) / sz
    var nx_s = bind_nrm.x() * inv_sx
    var ny_s = (bind_nrm.y() - acc_sh_xy * nx_s) * inv_sy
    var nz_s = (bind_nrm.z() - acc_sh_xz * nx_s) * inv_sz
    var sm_nrm = GpuVec3(nx_s, ny_s, nz_s).normalized()
    var sm_tan = GpuVec3(
        fma_f(acc_sh_xz, bind_tan.z(), fma_f(acc_sh_xy, bind_tan.y(), bind_tan.x() * sx)),
        bind_tan.y() * sy,
        bind_tan.z() * sz,
    ).normalized()

    var inv_l = rsqrt_f(max_f(fma_f(acc_rw, acc_rw, dot(acc_r, acc_r)), 1e-12))
    var rv = acc_r * inv_l
    var rw = acc_rw * inv_l
    var dv = acc_d * inv_l
    var dw = acc_dw * inv_l

    var trans = (dv * rw - rv * dw + cross(rv, dv)) * Float32(2.0)
    var scaled_pos = sm_pos * s_blend
    var out_pos = scaled_pos + cross(rv, cross(rv, scaled_pos) + scaled_pos * rw) * Float32(2.0) + trans
    var out_nrm = (sm_nrm + cross(rv, cross(rv, sm_nrm) + sm_nrm * rw) * Float32(2.0)).normalized()
    var out_tan = (sm_tan + cross(rv, cross(rv, sm_tan) + sm_tan * rw) * Float32(2.0)).normalized()
    return (out_pos, out_nrm, out_tan)


def aaa_dq_skinning_shader(bind_vertices: Int, bind_skinning: Int) -> List[UInt32]:
    """GPU Compute 4-Bone Dual-Quaternion + Morph-Target (`BlendShape`) + Smear-Frame Skinning Shader (`group = 64 x 1`):
    Push[0] = skinned_vertex_count, Push[1] = skin_bind_first_slot, Push[2] = dst_first_vertex_in_buf_vertices,
    Push[3] = cur_bones_base, Push[4] = prev_bones_base, Push[5] = cur_smear_base (0 -> default),
    Push[6] = prev_smear_base (0 -> default), Push[7] = morph_count, Push[8] = morph_weight_first_slot,
    Push[9] = morph_delta_first_slot.
    Deforms `(pos, nrm, tan)` in-place inside `buf_vertices` (ready for `VisBuffer`, outlines, AND `BLAS` `UPDATE_KHR` refit!)
    and writes previous-frame deformed vertex positions into `BIND_AAA_SKINNING` for exact 3D `FlintTSR` motion vectors!
    """
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=16)
    var b_verts = dsl.buffer[12](bind_vertices)
    var b_skin = dsl.buffer[1](bind_skinning)

    var local_v = dsl.global_id_x()
    var v_cnt = dsl.push_u32(0)
    var bind_first = dsl.push_u32(1)
    var dst_first = dsl.push_u32(2)
    var cur_bones_base = dsl.push_u32(3)
    var prev_bones_base = dsl.push_u32(4)
    var cur_smear_in = dsl.push_u32(5)
    var prev_smear_in = dsl.push_u32(6)
    var morph_cnt = dsl.push_u32(7)
    var morph_w_slot = dsl.push_u32(8)
    var morph_d_slot = dsl.push_u32(9)
    dsl.return_if(local_v.ge(v_cnt))

    var eff_prev_bones = select_u(prev_bones_base.eq(dsl.u32(0)), dsl.u32(UInt32(AAA_SKIN_PREV_BONES_BASE)), prev_bones_base)
    var eff_cur_smear = select_u(cur_smear_in.eq(dsl.u32(0)), dsl.u32(UInt32(AAA_SKIN_CUR_SMEAR_BASE)), cur_smear_in)
    var eff_prev_smear = select_u(prev_smear_in.eq(dsl.u32(0)), dsl.u32(UInt32(AAA_SKIN_PREV_SMEAR_BASE)), prev_smear_in)

    var src_base = dsl.u32(UInt32(AAA_SKIN_BIND_VERTS_BASE)) + (bind_first + local_v) * UInt32(AAA_SKIN_BIND_STRIDE_WORDS)
    var bind_pos = b_skin.load_vec3(src_base)
    var bind_nrm = b_skin.load_vec3(src_base + UInt32(3))
    var bind_tan = b_skin.load_vec3(src_base + UInt32(6))
    var uv_u = b_skin.load_f32(src_base + UInt32(9))
    var uv_v = b_skin.load_f32(src_base + UInt32(10))
    var meta = b_skin.load_u32(src_base + UInt32(11))

    var j_pack = b_skin.load_u32(src_base + UInt32(12))
    var j0 = j_pack & UInt32(0xFF)
    var j1 = (j_pack >> UInt32(8)) & UInt32(0xFF)
    var j2 = (j_pack >> UInt32(16)) & UInt32(0xFF)
    var j3 = (j_pack >> UInt32(24)) & UInt32(0xFF)

    var w0 = b_skin.load_f32(src_base + UInt32(13))
    var w1 = b_skin.load_f32(src_base + UInt32(14))
    var w2 = b_skin.load_f32(src_base + UInt32(15))
    var w3 = max_f(dsl.f32(1.0) - (w0 + w1 + w2), 0.0)

    var cur_bp = dsl.var_vec3(bind_pos)
    var cur_bn = dsl.var_vec3(bind_nrm)
    var prev_bp = dsl.var_vec3(bind_pos)
    var prev_bn = dsl.var_vec3(bind_nrm)
    var m_var = dsl.var_u32(dsl.u32(0))

    dsl.if_then(dsl.u32(0).lt(morph_cnt))
    dsl.begin_loop()
    var m_idx = m_var.get()
    dsl.break_if(m_idx.ge(morph_cnt))
    var w_off = morph_w_slot + m_idx
    var cw = b_skin.load_f32(dsl.u32(UInt32(AAA_SKIN_CUR_MORPH_WEIGHTS_BASE)) + w_off)
    var pw = b_skin.load_f32(dsl.u32(UInt32(AAA_SKIN_PREV_MORPH_WEIGHTS_BASE)) + w_off)
    var d_off = dsl.u32(UInt32(AAA_SKIN_MORPH_DELTAS_BASE)) + (morph_d_slot + m_idx * v_cnt + local_v) * UInt32(6)
    var dp = b_skin.load_vec3(d_off)
    var dn = b_skin.load_vec3(d_off + UInt32(3))
    cur_bp.set(cur_bp.get() + dp * cw)
    cur_bn.set(cur_bn.get() + dn * cw)
    prev_bp.set(prev_bp.get() + dp * pw)
    prev_bn.set(prev_bn.get() + dn * pw)
    m_var.set(m_idx + UInt32(1))
    dsl.end_loop()
    cur_bn.set(cur_bn.get().normalized())
    prev_bn.set(prev_bn.get().normalized())
    dsl.end_if()

    var cur_def = _emit_dq_blend_and_transform_dsl(
        dsl, b_skin.buf_id, cur_bones_base, eff_cur_smear,
        j0, j1, j2, j3, w0, w1, w2, w3,
        cur_bp.get(), cur_bn.get(), bind_tan,
    )
    var prev_def = _emit_dq_blend_and_transform_dsl(
        dsl, b_skin.buf_id, eff_prev_bones, eff_prev_smear,
        j0, j1, j2, j3, w0, w1, w2, w3,
        prev_bp.get(), prev_bn.get(), bind_tan,
    )

    var dst_v = dst_first + local_v
    var dst_base = b_verts.base_word(dst_v)
    b_verts.store_vec3(dst_base, cur_def[0])
    b_verts.store_vec3(dst_base + UInt32(3), cur_def[1])
    b_verts.store_vec3(dst_base + UInt32(6), cur_def[2])
    b_verts.store_f32(dst_base + UInt32(9), uv_u)
    b_verts.store_f32(dst_base + UInt32(10), uv_v)
    b_verts.store_u32(dst_base + UInt32(11), meta)

    # Index by per-mesh vertex offset (bind_first + local_v), not global scene vertex dst_v
    var prev_pos_off = dsl.u32(UInt32(AAA_SKIN_PREV_POS_BASE)) + (bind_first + local_v) * UInt32(3)
    b_skin.store_vec3(prev_pos_off, prev_def[0])

    return dsl.finish()
