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
comptime AAA_SKIN_BUFFER_WORDS: Int = AAA_SKIN_PREV_POS_BASE + AAA_SKIN_MAX_VERTS * 3                              # 1,900,544 (~7.25 MiB)


struct FlintAaaSkinningArena(Copyable, ImplicitlyCopyable, Movable):
    """Tracks disjoint GPU bone-palette (`2,048` bones) & bind-vertex (`98,304` verts) sub-allocations."""
    var bone_cursor: Int
    var vertex_cursor: Int

    def __init__(out self):
        self.bone_cursor = 0
        self.vertex_cursor = 0

    def reset(mut self):
        self.bone_cursor = 0
        self.vertex_cursor = 0

    def alloc_bone_palette(mut self, bone_count: Int = 64, explicit_word_offset: Int = -1) -> Int:
        """Returns `bone_palette_offset` in `UInt32` words (`bone_slot * 8`) without overwriting slot `0`."""
        var eff_bones = max(1, min(AAA_MAX_SKIN_BONES, bone_count))
        if explicit_word_offset >= 0:
            var end_bone = (explicit_word_offset + 7) // 8 + eff_bones
            if end_bone > self.bone_cursor:
                self.bone_cursor = min(AAA_MAX_SKIN_BONES, end_bone)
            return explicit_word_offset
        var slot = self.bone_cursor
        if slot + eff_bones > AAA_MAX_SKIN_BONES:
            slot = max(0, AAA_MAX_SKIN_BONES - eff_bones)
        self.bone_cursor = min(AAA_MAX_SKIN_BONES, slot + eff_bones)
        return slot * 8

    def alloc_vertices(mut self, vertex_count: Int) -> Tuple[Int, Int]:
        """Allocates `(first_slot, count)` in `[0 .. AAA_SKIN_MAX_VERTS]`."""
        var first_slot = self.vertex_cursor
        var v_cnt = max(0, min(vertex_count, AAA_SKIN_MAX_VERTS - first_slot))
        self.vertex_cursor += v_cnt
        return (first_slot, v_cnt)


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


def _emit_dq_blend_and_transform_dsl(
    mut dsl: ShaderDsl,
    b_skin_buf_id: UInt32,
    palette_base: GpuU32,
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
    """Blends 4 scale-aware bone Dual Quaternions around dominant pivot `j0` in `ShaderDsl` and transforms `(pos, nrm, tan)`."""
    from flint.render.shader_dsl import BufferRef
    var b_skin = BufferRef[1](dsl.mod_addr, b_skin_buf_id)
    var pal = palette_base

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
        var sw = select_f(qdot.lt(0.0), -w_arr[k], w_arr[k])
        acc_r = acc_r + rk * sw
        acc_rw = acc_rw + rwk * sw
        acc_d = acc_d + dk * sw
        acc_dw = acc_dw + dwk * sw
        s_blend = fma_f(sk, w_arr[k], s_blend)

    var inv_l = rsqrt_f(max_f(fma_f(acc_rw, acc_rw, dot(acc_r, acc_r)), 1e-12))
    var rv = acc_r * inv_l
    var rw = acc_rw * inv_l
    var dv = acc_d * inv_l
    var dw = acc_dw * inv_l

    var trans = (dv * rw - rv * dw + cross(rv, dv)) * Float32(2.0)
    var scaled_pos = bind_pos * s_blend
    var out_pos = scaled_pos + cross(rv, cross(rv, scaled_pos) + scaled_pos * rw) * Float32(2.0) + trans
    var out_nrm = (bind_nrm + cross(rv, cross(rv, bind_nrm) + bind_nrm * rw) * Float32(2.0)).normalized()
    var out_tan = (bind_tan + cross(rv, cross(rv, bind_tan) + bind_tan * rw) * Float32(2.0)).normalized()
    return (out_pos, out_nrm, out_tan)


def aaa_dq_skinning_shader(bind_vertices: Int, bind_skinning: Int) -> List[UInt32]:
    """GPU Compute 4-Bone Dual-Quaternion Skinning Shader (`group = 64 x 1`):
    Push[0] = skinned_vertex_count, Push[1] = skin_bind_first_slot, Push[2] = dst_first_vertex_in_buf_vertices.
    Deforms `(pos, nrm, tan)` in-place inside `buf_vertices` (ready for `VisBuffer` AND `BLAS` `UPDATE_KHR` refit!)
    and writes previous-frame deformed vertex positions into `BIND_AAA_SKINNING` for exact 3D `FlintTSR v2` motion vectors!
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
    dsl.return_if(local_v.ge(v_cnt))

    var eff_prev_bones = select_u(prev_bones_base.eq(dsl.u32(0)), dsl.u32(UInt32(AAA_SKIN_PREV_BONES_BASE)), prev_bones_base)

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

    var cur_def = _emit_dq_blend_and_transform_dsl(
        dsl, b_skin.buf_id, cur_bones_base,
        j0, j1, j2, j3, w0, w1, w2, w3,
        bind_pos, bind_nrm, bind_tan,
    )
    var prev_def = _emit_dq_blend_and_transform_dsl(
        dsl, b_skin.buf_id, eff_prev_bones,
        j0, j1, j2, j3, w0, w1, w2, w3,
        bind_pos, bind_nrm, bind_tan,
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
