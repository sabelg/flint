"""Area H (sota:atmo, round 3): image-based lighting baked from the physical sky on the GPU.

Round 2 re-baked the IBL on the CPU (64 x 36 CPU sky-view, SH9, a GGX-prefiltered cube, BC6H encode, a blocking
upload): about 17 ms whenever a light moved more than `ibl_update_deg`, a hitch in every time-lapse. Here the same
products come straight from the 192 x 108 sky-view LUT that the frame already computes, inside the frame's command
buffer, with no CPU work and no stall:

1. `sota_atmo_ibl_sh_shader` (one 256-thread workgroup): SH9 projection of the sky over 6 x 16 x 16 cube texels
   (same texel set, solid-angle weights, 4 pi normalisation and cosine-lobe band factors as the CPU bake), shared-
   memory tree reduction, then `vkCmdCopyBuffer` into `b_globals[72 .. 99)` before any pass reads the SH.
2. `sota_atmo_ibl_cube_shader` (one thread per 4 x 4 block, all faces, one dispatch per mip): mip 0 samples the sky,
   mip m > 0 is the 16-sample GGX prefilter of the CPU bake (roughness m / (mips - 1)); the 16 texels are then encoded
   to BC6H mode 11 (one subset, 10-bit endpoints, exhaustive 4-bit index search). That is the CPU encoder of
   `flint.assets.bc_compress` with `partitioned = False`, bit for bit (checked on a test pattern). The blocks are
   copied into the existing BC6H environment cube, so every consumer and the descriptor stay unchanged.

CPU twins: `ibl_sh9_from_sky_view_cpu`, `ibl_prefilter_texel_cpu`, `ibl_test_pattern_texel`.
"""
from std.collections import List
from std.math import cos, max, min, sin, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.platform.vulkan import (
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    ACCESS_TRANSFER_READ,
    ACCESS_TRANSFER_WRITE,
    GpuBuffer,
    LAYOUT_SHADER_READ_ONLY,
    LAYOUT_TRANSFER_TARGET,
    PIPELINE_COMPUTE,
    PIPELINE_TRANSFER,
    VulkanGpu,
    VulkanImage,
)
from flint.render.aaa_ibl import _importance_sample_ggx, _radical_inverse_vdc, cube_face_direction
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_GLOBALS, BIND_AAA_HISTORY
from flint.render.spirv import MODEL_GL_COMPUTE
from flint.render.shader_dsl import (
    BufferRef,
    GpuBool,
    GpuF32,
    GpuU32,
    GpuVarF32,
    GpuVec3,
    ShaderDsl,
    abs_f,
    cos_f,
    max_f,
    max_u,
    min_f,
    min_u,
    select_f,
    select_u,
    select_vec3,
    sin_f,
    sqrt_f,
)
from flint.render.sota_atmo_core import RGB, SOTA_ATMO_SKY_H, SOTA_ATMO_SKY_W, SotaAtmoParams, sample_sky_view
from flint.render.sota_atmo_dsl import _Atmo, _sky_view_radiance
from flint.render.sota_atmo_night import hash_u32

comptime SOTA_IBL_SH_DIM: Int = 16
comptime SOTA_IBL_SH_THREADS: Int = 256
comptime SOTA_IBL_GGX_SAMPLES: Int = 16
comptime _AAA_STAGE_ALL: Int = 0x10000


def ibl_block_words(base_dim: Int, mips: Int) -> Int:
    var n = 0
    for m in range(mips):
        var d = max(4, base_dim >> m)
        n += 6 * (d // 4) * (d // 4) * 4
    return n


def ibl_mip_block_offset(base_dim: Int, mip: Int) -> Int:
    """Word offset of `mip` inside the block region (faces follow each other, `nb * nb * 4` words each)."""
    var n = 0
    for m in range(mip):
        var d = max(4, base_dim >> m)
        n += 6 * (d // 4) * (d // 4) * 4
    return n


def ibl_region_words(base_dim: Int, mips: Int) -> Int:
    return 32 + ibl_block_words(base_dim, mips)


# ============================================================================
# CPU twins
# ============================================================================

def _sh_basis(d: Vec3) -> List[Float32]:
    return [
        Float32(0.282095),
        Float32(0.488603) * d.y,
        Float32(0.488603) * d.z,
        Float32(0.488603) * d.x,
        Float32(1.092548) * d.x * d.y,
        Float32(1.092548) * d.y * d.z,
        Float32(0.315392) * (Float32(3.0) * d.z * d.z - Float32(1.0)),
        Float32(1.092548) * d.x * d.z,
        Float32(0.546274) * (d.x * d.x - d.y * d.y),
    ]


def _sh_band() -> List[Float32]:
    return [Float32(0.28209479), 0.32573500, 0.32573500, 0.32573500, 0.27313711, 0.27313711, 0.07884789, 0.27313711, 0.13656855]


def ibl_sh9_from_sky_view_cpu(lut: List[RGB], w: Int, h: Int, alt_km: Float32, p: SotaAtmoParams) -> List[Vec3]:
    """The SH9 projection of the CPU bake (`SotaAtmosphere.bake_ibl`) over a sky-view LUT of any size."""
    var acc = List[Vec3](length=9, fill=Vec3(0.0, 0.0, 0.0))
    var wsum = Float32(0.0)
    var n = SOTA_IBL_SH_DIM
    for face in range(6):
        for y in range(n):
            var ty = (Float32(y) + Float32(0.5)) / Float32(n) * Float32(2.0) - Float32(1.0)
            for x in range(n):
                var tx = (Float32(x) + Float32(0.5)) / Float32(n) * Float32(2.0) - Float32(1.0)
                var tmp = Float32(1.0) + tx * tx + ty * ty
                var d_omega = Float32(4.0) / (sqrt(tmp) * tmp)
                var dir = cube_face_direction(face, (Float32(x) + Float32(0.5)) / Float32(n), (Float32(y) + Float32(0.5)) / Float32(n))
                var c = sample_sky_view(lut, dir, alt_km, p, w, h)
                var rad = Vec3(c[0], c[1], c[2])
                var bs = _sh_basis(dir)
                for k in range(9):
                    acc[k] = acc[k] + rad * (bs[k] * d_omega)
                wsum += d_omega
    var norm = Float32(12.5663706) / max(Float32(1e-6), wsum)
    var band = _sh_band()
    for k in range(9):
        acc[k] = acc[k] * (norm * band[k])
    return acc^


def ibl_prefilter_texel_cpu(
    lut: List[RGB], w: Int, h: Int, alt_km: Float32, p: SotaAtmoParams, face: Int, x: Int, y: Int, m_dim: Int, mip: Int, mips: Int
) -> Vec3:
    var u = (Float32(x) + Float32(0.5)) / Float32(m_dim)
    var v = (Float32(y) + Float32(0.5)) / Float32(m_dim)
    var r_dir = cube_face_direction(face, u, v)
    if mip == 0:
        var c = sample_sky_view(lut, r_dir, alt_km, p, w, h)
        return Vec3(c[0], c[1], c[2])
    var roughness = Float32(mip) / Float32(max(1, mips - 1))
    var acc = Vec3(0.0, 0.0, 0.0)
    var w_tot = Float32(0.0)
    for s_i in range(SOTA_IBL_GGX_SAMPLES):
        var hv = _importance_sample_ggx(Float32(s_i) / Float32(SOTA_IBL_GGX_SAMPLES), _radical_inverse_vdc(UInt32(s_i)), roughness, r_dir)
        var l = (hv * (Float32(2.0) * r_dir.dot(hv)) - r_dir).normalized()
        var ndl = max(Float32(0.0), r_dir.dot(l))
        if ndl > Float32(0.0):
            var c = sample_sky_view(lut, l, alt_km, p, w, h)
            acc = acc + Vec3(c[0], c[1], c[2]) * ndl
            w_tot += ndl
    return acc * (Float32(1.0) / max(Float32(1e-5), w_tot))


def ibl_test_pattern_texel(face: Int, x: Int, y: Int, m_dim: Int, mip: Int) -> Vec3:
    """Exactly representable HDR values (k / 1024, k < 65536) so the CPU and GPU encoders see the same f16 bits."""
    var k = UInt32(((mip * 6 + face) * m_dim + y) * m_dim + x)
    var h0 = hash_u32(k * UInt32(3) + UInt32(1))
    var h1 = hash_u32(k * UInt32(3) + UInt32(2))
    var h2 = hash_u32(k * UInt32(3) + UInt32(3))
    # Smooth-ish blocks plus noise: a shared per-block level (high bits of the block index) + small texel noise.
    var blk = hash_u32(UInt32(((mip * 6 + face) * m_dim + (y // 4) * 4) * m_dim + (x // 4) * 4) + UInt32(77))
    var base = Float32(blk & UInt32(0x3FFF))
    return Vec3(
        (base + Float32(h0 & UInt32(0x3FF))) / Float32(1024.0),
        (base * Float32(0.5) + Float32(h1 & UInt32(0x7FF))) / Float32(1024.0),
        (base * Float32(0.25) + Float32(h2 & UInt32(0x1FFF))) / Float32(1024.0),
    )


# ============================================================================
# DSL helpers
# ============================================================================

def _g_cube_dir(mut dsl: ShaderDsl, face: GpuU32, u: GpuF32, v: GpuF32) -> GpuVec3:
    """Vulkan cube-face direction (`cube_face_direction`) for a dynamic face."""
    var s = u * 2.0 - 1.0
    var t = v * 2.0 - 1.0
    var one = dsl.f32(1.0)
    var d0 = GpuVec3(one, -t, -s)
    var d1 = GpuVec3(-one, -t, s)
    var d2 = GpuVec3(s, one, t)
    var d3 = GpuVec3(s, -one, -t)
    var d4 = GpuVec3(s, -t, one)
    var d5 = GpuVec3(-s, -t, -one)
    var r = select_vec3(face.eq(UInt32(0)), d0, select_vec3(face.eq(UInt32(1)), d1, select_vec3(face.eq(UInt32(2)), d2,
        select_vec3(face.eq(UInt32(3)), d3, select_vec3(face.eq(UInt32(4)), d4, d5)))))
    return r.normalized()


def _g_f32_to_uf16(mut dsl: ShaderDsl, val: GpuF32) -> GpuU32:
    """`f32_to_uf16_bits` (bc_compress) in integer ops."""
    var v = min_f(max_f(val, 0.0), 65504.0)
    var bits = v.bitcast_u32()
    var e32 = (bits >> UInt32(23)) & UInt32(0xFF)
    var mant = bits & UInt32(0x7FFFFF)
    # Sub-normal half: exp16 = e32 - 112 <= 0 (e32 < 113); below 2^-24 (e32 < 102) it flushes to 0.
    var sh = select_u(e32.lt(UInt32(113)), dsl.u32(113) - e32, dsl.u32(0))
    var sub = (((mant | UInt32(0x800000)) >> min_u(sh, dsl.u32(31))) + UInt32(0x1000)) >> UInt32(13)
    var rounded = mant + UInt32(0x1000)
    var carry = (rounded & UInt32(0x800000)).ne(UInt32(0))
    var e16 = select_u(carry, e32 - UInt32(111), e32 - UInt32(112))
    var m16 = select_u(carry, dsl.u32(0), rounded >> UInt32(13))
    var normal = select_u(e16.ge(UInt32(31)), dsl.u32(0x7BFF), (e16 << UInt32(10)) | m16)
    var res = select_u(e32.lt(UInt32(113)), sub, normal)
    res = select_u(e32.lt(UInt32(102)), dsl.u32(0), res)
    res = select_u(e32.ge(UInt32(143)), dsl.u32(0x7BFF), res)
    return res


def _g_quant_u10(mut dsl: ShaderDsl, f: GpuU32) -> GpuU32:
    var interp = (f << UInt32(6)) // UInt32(31)
    var q = min_u((interp + UInt32(32)) >> UInt32(6), dsl.u32(1023))
    return select_u(f.eq(UInt32(0)), dsl.u32(0), select_u(f.ge(UInt32(0x7BFF)), dsl.u32(1023), q))


def _g_unquant_u10(mut dsl: ShaderDsl, q: GpuU32) -> GpuU32:
    var u = ((q << UInt32(16)) + UInt32(0x8000)) >> UInt32(10)
    return select_u(q.eq(UInt32(0)), dsl.u32(0), select_u(q.ge(UInt32(1023)), dsl.u32(0xFFFF), u))


def _g_interp(u0: GpuU32, u1: GpuU32, w: GpuU32) -> GpuU32:
    return (((u0 * (UInt32(64) - w) + u1 * w + UInt32(32)) >> UInt32(6)) * UInt32(31)) >> UInt32(6)


def _g_absdiff(a: GpuU32, b: GpuU32) -> GpuU32:
    return select_u(a.ge(b), a - b, b - a)


struct _Bits128(Movable):
    var w: List[GpuU32]

    def __init__(out self, mut dsl: ShaderDsl):
        self.w = List[GpuU32]()
        for _ in range(4):
            self.w.append(dsl.u32(0))

    def put(mut self, v: GpuU32, bits: Int, pos: Int):
        var wi = pos // 32
        var sh = pos % 32
        var masked = v & UInt32((1 << bits) - 1)
        self.w[wi] = self.w[wi] | (masked << UInt32(sh))
        if sh + bits > 32:
            self.w[wi + 1] = self.w[wi + 1] | (masked >> UInt32(32 - sh))


# ============================================================================
# 1. SH9 projection
# ============================================================================

def sota_atmo_ibl_sh_shader() -> List[UInt32]:
    """Push: 0 sky_off, 1 sh_out_off. One workgroup of 256 threads."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=SOTA_IBL_SH_THREADS, group_height=1, push_words=AAA_PUSH_WORDS)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var lid = dsl.global_id_x()
    var s_off = dsl.push_u32(0)
    var o_off = dsl.push_u32(1)
    var a = _Atmo(dsl, b_glob)
    var sh = dsl.shared_array_f32[SOTA_IBL_SH_THREADS * 28]()
    var acc = List[GpuVarF32Box]()
    for _ in range(28):
        acc.append(GpuVarF32Box(dsl))
    var n = SOTA_IBL_SH_DIM
    var total = 6 * n * n
    var i_v = dsl.var_u32(lid)
    dsl.begin_loop()
    var i = i_v.get()
    dsl.break_if(i.ge(UInt32(total)))
    var face = i // UInt32(n * n)
    var rem = i % UInt32(n * n)
    var y = rem // UInt32(n)
    var x = rem % UInt32(n)
    var fu = (x.to_f32() + 0.5) * (1.0 / Float32(n))
    var fv = (y.to_f32() + 0.5) * (1.0 / Float32(n))
    var tx = fu * 2.0 - 1.0
    var ty = fv * 2.0 - 1.0
    var tmp = tx * tx + ty * ty + 1.0
    var d_omega = 4.0 / (sqrt_f(tmp) * tmp)
    var d = _g_cube_dir(dsl, face, fu, fv)
    var c = _sky_view_radiance(dsl, b_hist, s_off, a, d)
    var basis = List[GpuF32]()
    basis.append(d_omega * 0.282095)
    basis.append(d.y() * (d_omega * 0.488603))
    basis.append(d.z() * (d_omega * 0.488603))
    basis.append(d.x() * (d_omega * 0.488603))
    basis.append(d.x() * d.y() * (d_omega * 1.092548))
    basis.append(d.y() * d.z() * (d_omega * 1.092548))
    basis.append((d.z() * d.z() * 3.0 - 1.0) * (d_omega * 0.315392))
    basis.append(d.x() * d.z() * (d_omega * 1.092548))
    basis.append((d.x() * d.x() - d.y() * d.y()) * (d_omega * 0.546274))
    for k in range(9):
        acc[k * 3 + 0].v.set(acc[k * 3 + 0].v.get() + c.x() * basis[k])
        acc[k * 3 + 1].v.set(acc[k * 3 + 1].v.get() + c.y() * basis[k])
        acc[k * 3 + 2].v.set(acc[k * 3 + 2].v.get() + c.z() * basis[k])
    acc[27].v.set(acc[27].v.get() + d_omega)
    i_v.set(i + UInt32(SOTA_IBL_SH_THREADS))
    dsl.end_loop()
    for k in range(28):
        sh.store(lid * UInt32(28) + UInt32(k), acc[k].v.get())
    var stride = SOTA_IBL_SH_THREADS // 2
    while stride >= 1:
        dsl.workgroup_barrier()
        dsl.if_then(lid.lt(UInt32(stride)))
        for k in range(28):
            var ia = lid * UInt32(28) + UInt32(k)
            var ib = (lid + UInt32(stride)) * UInt32(28) + UInt32(k)
            sh.store(ia, sh.load(ia) + sh.load(ib))
        dsl.end_if()
        stride //= 2
    dsl.workgroup_barrier()
    dsl.if_then(lid.eq(UInt32(0)))
    var norm = 12.5663706 / max_f(sh.load(dsl.u32(27)), 1e-6)
    var band = _sh_band()
    for k in range(9):
        for ch in range(3):
            var v = sh.load(dsl.u32(UInt32(k * 3 + ch))) * norm * band[k]
            b_hist.store_f32(o_off + UInt32(k * 3 + ch), v)
    b_hist.store_u32(o_off + UInt32(27), dsl.u32(1))
    dsl.end_if()
    return dsl.finish()


struct GpuVarF32Box(Movable):
    var v: GpuVarF32

    def __init__(out self, mut dsl: ShaderDsl):
        self.v = dsl.var_f32_const(0.0)


# ============================================================================
# 2. GGX-prefiltered cube + BC6H mode-11 encode
# ============================================================================

def sota_atmo_ibl_cube_shader() -> List[UInt32]:
    """One thread per 4 x 4 block (x, y) of face z. Push: 0 sky_off, 1 out_off (this mip's blocks), 2 m_dim,
    3 roughness (f32), 4 mip, 5 mode (0 sky, 1 test pattern)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var bx = dsl.global_id_x()
    var by = dsl.global_id_y()
    var face = dsl.global_id_z()
    var s_off = dsl.push_u32(0)
    var o_off = dsl.push_u32(1)
    var m_dim = dsl.push_u32(2)
    var rough = dsl.push_f32(3)
    var mip = dsl.push_u32(4)
    var mode = dsl.push_u32(5)
    var nb = m_dim >> UInt32(2)
    dsl.return_if(bx.ge(nb) | by.ge(nb) | face.ge(UInt32(6)))
    var a = _Atmo(dsl, b_glob)
    var lid = (bx & UInt32(7)) + (by & UInt32(7)) * UInt32(8)
    var texel = dsl.shared_array_u32[64 * 48]()
    var fm = m_dim.to_f32()
    var aa = max_f(rough * rough, 0.002)
    var a2m1 = aa * aa - 1.0
    var mn_r = dsl.var_u32(dsl.u32(0x7BFF))
    var mn_g = dsl.var_u32(dsl.u32(0x7BFF))
    var mn_b = dsl.var_u32(dsl.u32(0x7BFF))
    var mx_r = dsl.var_u32(dsl.u32(0))
    var mx_g = dsl.var_u32(dsl.u32(0))
    var mx_b = dsl.var_u32(dsl.u32(0))
    var t_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var t = t_v.get()
    dsl.break_if(t.ge(UInt32(16)))
    var x = bx * UInt32(4) + (t & UInt32(3))
    var y = by * UInt32(4) + (t >> UInt32(2))
    var col = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    dsl.if_then(mode.eq(UInt32(1)))
    # Test pattern (`ibl_test_pattern_texel`).
    var k = ((mip * UInt32(6) + face) * m_dim + y) * m_dim + x
    var h0 = _ghash_ibl(k * UInt32(3) + UInt32(1))
    var h1 = _ghash_ibl(k * UInt32(3) + UInt32(2))
    var h2 = _ghash_ibl(k * UInt32(3) + UInt32(3))
    var blk = _ghash_ibl(((mip * UInt32(6) + face) * m_dim + (y >> UInt32(2)) * UInt32(4)) * m_dim + (x >> UInt32(2)) * UInt32(4) + UInt32(77))
    var base = (blk & UInt32(0x3FFF)).to_f32()
    col.set(GpuVec3(
        (base + (h0 & UInt32(0x3FF)).to_f32()) * (1.0 / 1024.0),
        (base * 0.5 + (h1 & UInt32(0x7FF)).to_f32()) * (1.0 / 1024.0),
        (base * 0.25 + (h2 & UInt32(0x1FFF)).to_f32()) * (1.0 / 1024.0),
    ))
    dsl.else_branch()
    var fu = (x.to_f32() + 0.5) / fm
    var fv = (y.to_f32() + 0.5) / fm
    var n = _g_cube_dir(dsl, face, fu, fv)
    dsl.if_then(mip.eq(UInt32(0)))
    col.set(_sky_view_radiance(dsl, b_hist, s_off, a, n))
    dsl.else_branch()
    var up = select_vec3(abs_f(n.z()).gt(0.999), dsl.vec3_const(1.0, 0.0, 0.0), dsl.vec3_const(0.0, 0.0, 1.0))
    var tx = up.cross(n).normalized()
    var ty = n.cross(tx)
    var acc = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var wt = dsl.var_f32_const(0.0)
    for s_i in range(SOTA_IBL_GGX_SAMPLES):
        var u1 = Float32(s_i) / Float32(SOTA_IBL_GGX_SAMPLES)
        var u2 = _radical_inverse_vdc(UInt32(s_i))
        var phi = Float32(6.28318530718) * u1
        var cos_t = sqrt_f((1.0 - u2) / max_f(a2m1 * u2 + 1.0, 1e-6))
        var sin_t = sqrt_f(max_f(1.0 - cos_t * cos_t, 0.0))
        var hv = (tx * (sin_t * Float32(cos(phi))) + ty * (sin_t * Float32(sin(phi))) + n * cos_t).normalized()
        var l = (hv * (n.dot(hv) * 2.0) - n).normalized()
        var ndl = n.dot(l)
        dsl.if_then(ndl.gt(0.0))
        acc.set(acc.get() + _sky_view_radiance(dsl, b_hist, s_off, a, l) * ndl)
        wt.set(wt.get() + ndl)
        dsl.end_if()
    col.set(acc.get() * (1.0 / max_f(wt.get(), 1e-5)))
    dsl.end_if()
    dsl.end_if()
    var c = col.get()
    var hr = _g_f32_to_uf16(dsl, c.x())
    var hg = _g_f32_to_uf16(dsl, c.y())
    var hb = _g_f32_to_uf16(dsl, c.z())
    var sb = lid * UInt32(48) + t * UInt32(3)
    texel.store(sb, hr)
    texel.store(sb + UInt32(1), hg)
    texel.store(sb + UInt32(2), hb)
    mn_r.set(min_u(mn_r.get(), hr))
    mn_g.set(min_u(mn_g.get(), hg))
    mn_b.set(min_u(mn_b.get(), hb))
    mx_r.set(max_u(mx_r.get(), hr))
    mx_g.set(max_u(mx_g.get(), hg))
    mx_b.set(max_u(mx_b.get(), hb))
    t_v.set(t + UInt32(1))
    dsl.end_loop()

    # Mode 11 endpoints (min / max per channel), exhaustive 4-bit index search.
    var qr0 = _g_quant_u10(dsl, mn_r.get())
    var qg0 = _g_quant_u10(dsl, mn_g.get())
    var qb0 = _g_quant_u10(dsl, mn_b.get())
    var qr1 = _g_quant_u10(dsl, mx_r.get())
    var qg1 = _g_quant_u10(dsl, mx_g.get())
    var qb1 = _g_quant_u10(dsl, mx_b.get())
    var ur0 = _g_unquant_u10(dsl, qr0)
    var ug0 = _g_unquant_u10(dsl, qg0)
    var ub0 = _g_unquant_u10(dsl, qb0)
    var ur1 = _g_unquant_u10(dsl, qr1)
    var ug1 = _g_unquant_u10(dsl, qg1)
    var ub1 = _g_unquant_u10(dsl, qb1)
    var idx_lo = dsl.var_u32(dsl.u32(0))
    var idx_hi = dsl.var_u32(dsl.u32(0))
    var t2_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var t2 = t2_v.get()
    dsl.break_if(t2.ge(UInt32(16)))
    var sb2 = lid * UInt32(48) + t2 * UInt32(3)
    var pr = texel.load(sb2)
    var pg = texel.load(sb2 + UInt32(1))
    var pb = texel.load(sb2 + UInt32(2))
    var best_i = dsl.var_u32(dsl.u32(0))
    var best_d = dsl.var_u32(dsl.u32(0xFFFFFFFF))
    var wi_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var wi = wi_v.get()
    dsl.break_if(wi.ge(UInt32(16)))
    var w = (wi * UInt32(64) + UInt32(7)) // UInt32(15)
    var er = _g_absdiff(_g_interp(ur0, ur1, w), pr)
    var eg = _g_absdiff(_g_interp(ug0, ug1, w), pg)
    var eb = _g_absdiff(_g_interp(ub0, ub1, w), pb)
    var dd = er * er + eg * eg + eb * eb
    var better = dd.lt(best_d.get())
    best_i.set(select_u(better, wi, best_i.get()))
    best_d.set(select_u(better, dd, best_d.get()))
    wi_v.set(wi + UInt32(1))
    dsl.end_loop()
    var sh_amt = (t2 & UInt32(7)) * UInt32(4)
    var lo = t2.lt(UInt32(8))
    idx_lo.set(select_u(lo, idx_lo.get() | (best_i.get() << sh_amt), idx_lo.get()))
    idx_hi.set(select_u(lo, idx_hi.get(), idx_hi.get() | (best_i.get() << sh_amt)))
    t2_v.set(t2 + UInt32(1))
    dsl.end_loop()
    # The anchor index (texel 0) has an implicit top bit of 0: flip endpoints and indices when it is >= 8.
    var flip = (idx_lo.get() & UInt32(8)).ne(UInt32(0))
    var e0r = select_u(flip, qr1, qr0)
    var e0g = select_u(flip, qg1, qg0)
    var e0b = select_u(flip, qb1, qb0)
    var e1r = select_u(flip, qr0, qr1)
    var e1g = select_u(flip, qg0, qg1)
    var e1b = select_u(flip, qb0, qb1)
    var ilo = select_u(flip, idx_lo.get() ^ UInt32(0xFFFFFFFF), idx_lo.get())
    var ihi = select_u(flip, idx_hi.get() ^ UInt32(0xFFFFFFFF), idx_hi.get())
    var bits = _Bits128(dsl)
    bits.put(dsl.u32(0x03), 5, 0)
    bits.put(e0r, 10, 5)
    bits.put(e0g, 10, 15)
    bits.put(e0b, 10, 25)
    bits.put(e1r, 10, 35)
    bits.put(e1g, 10, 45)
    bits.put(e1b, 10, 55)
    bits.put(ilo, 3, 65)
    for tt in range(1, 16):
        var src = ilo if tt < 8 else ihi
        var v = (src >> UInt32((tt % 8) * 4)) & UInt32(0xF)
        bits.put(v, 4, 68 + (tt - 1) * 4)
    var o = o_off + ((face * nb + by) * nb + bx) * UInt32(4)
    for k in range(4):
        b_hist.store_u32(o + UInt32(k), bits.w[k])
    return dsl.finish()


def _ghash_ibl(x: GpuU32) -> GpuU32:
    var s = x * UInt32(747796405) + UInt32(2891336453)
    var w = ((s >> ((s >> UInt32(28)) + UInt32(4))) ^ s) * UInt32(277803737)
    return (w >> UInt32(22)) ^ w


# ============================================================================
# Recording
# ============================================================================

struct SotaAtmoIblGpu(Movable):
    var pipe_sh: Int
    var pipe_cube: Int
    var region_base: Int   # 4-word aligned (BC6H copies need 16-byte buffer offsets)
    var base_dim: Int
    var mips: Int
    var test_pattern: Bool
    var cube_updates: Int
    var sh_updates: Int

    def __init__(out self):
        self.pipe_sh = 0
        self.pipe_cube = 0
        self.region_base = 0
        self.base_dim = 64
        self.mips = 5
        self.test_pattern = False
        self.cube_updates = 0
        self.sh_updates = 0

    def create(mut self, mut vk: VulkanGpu, pipe_layout: Int):
        self.pipe_sh = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_ibl_sh_shader())
        self.pipe_cube = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_ibl_cube_shader())

    def destroy(mut self, mut vk: VulkanGpu):
        if self.pipe_sh != 0:
            vk.destroy_pipeline(self.pipe_sh)
        if self.pipe_cube != 0:
            vk.destroy_pipeline(self.pipe_cube)
        self.pipe_sh = 0
        self.pipe_cube = 0

    def ready(self) -> Bool:
        return self.pipe_sh != 0 and self.pipe_cube != 0

    def sh_off(self) -> Int:
        return self.region_base

    def blocks_off(self) -> Int:
        return self.region_base + 32

    def record(
        mut self,
        mut vk: VulkanGpu,
        slot: Int,
        pipe_layout: Int,
        desc_set: Int,
        sky_off: Int,
        globals_buf: GpuBuffer,
        history_buf: GpuBuffer,
        mut env_cube: VulkanImage,
        do_cube: Bool,
    ):
        var push = List[UInt32](length=16, fill=0)
        push[0] = UInt32(sky_off)
        push[1] = UInt32(self.sh_off())
        vk.dispatch_aaa(slot, self.pipe_sh, pipe_layout, desc_set, push, 1, 1)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER | PIPELINE_COMPUTE, Int(ACCESS_TRANSFER_READ | ACCESS_SHADER_READ))
        vk.copy_buffer(slot, history_buf, self.sh_off() * 4, globals_buf, 72 * 4, 27 * 4)
        vk.memory_barrier2(slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), _AAA_STAGE_ALL, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
        self.sh_updates += 1
        if not do_cube:
            return
        for m in range(self.mips):
            var d = max(4, self.base_dim >> m)
            var nb = d // 4
            var pc = List[UInt32](length=16, fill=0)
            pc[0] = UInt32(sky_off)
            pc[1] = UInt32(self.blocks_off() + ibl_mip_block_offset(self.base_dim, m))
            pc[2] = UInt32(d)
            pc[3] = bitcast_f32(Float32(m) / Float32(max(1, self.mips - 1)))
            pc[4] = UInt32(m)
            pc[5] = UInt32(1) if self.test_pattern else UInt32(0)
            vk.dispatch_aaa(slot, self.pipe_cube, pipe_layout, desc_set, pc, (nb + 7) // 8, (nb + 7) // 8, 6)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ))
        vk.transition_image(slot, env_cube, LAYOUT_TRANSFER_TARGET, _AAA_STAGE_ALL, ACCESS_SHADER_READ, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        for m in range(self.mips):
            var d = max(4, self.base_dim >> m)
            var nb = d // 4
            for f in range(6):
                var off_w = self.blocks_off() + ibl_mip_block_offset(self.base_dim, m) + f * nb * nb * 4
                vk.upload_image_mip(slot, env_cube, history_buf, off_w * 4, m, d, d, base_layer=f, layer_count=1)
        vk.transition_image(slot, env_cube, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, _AAA_STAGE_ALL, ACCESS_SHADER_READ)
        self.cube_updates += 1


@always_inline
def bitcast_f32(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)
