"""Ray-traced refractive glass (`aaa_glass.mojo`).

Glass materials are ordinary `AaaMaterial`s (shaded by the deferred pass as a smooth dielectric whose
base color is `tint * opacity`, which gives the Fresnel reflection, sun glints and any diffuse body),
plus an `AaaGlassParams` entry in a small table in `BIND_AAA_GLOBALS`. Instances drawn with a glass
material carry `AAA_INSTANCE_GLASS_BIT` in instance word 39 and TLAS mask `AAA_RT_MASK_GLASS`, so
shadow rays pass through them (`aaa_rt_shadow.mojo` dims the sun instead) and the VSM skips them.

After the opaque frame is shaded and denoised, `aaa_glass_refract_shader` runs on glass pixels:
  1. refract the camera ray into the surface (Snell, `IOR`),
  2. trace it against glass only (`SPV_KHR_ray_query`, mask `0x80`) to find where it leaves the body
     (no hit = a thin sheet: the ray goes straight through),
  3. refract out through the exit surface's interpolated normal (total internal reflection mirrors it),
  4. shade what it reaches with `emit_rt_textured_hit_shade` (or the sky cube on a miss),
  5. absorb along the path (Beer-Lambert from `tint` and `density`) and add
     `(1 - F) * (1 - opacity) * transmittance * background` to the pixel; `F` is Schlick's Fresnel.
The pixel's motion vector is replaced by that of a static world point, so TSR reprojects what is seen
through a spinning glass ball from the background, not from the ball's surface.

Without ray tracing (no hardware ray queries, or no RT feature enabled this frame) the pass does not run
and glass shows only its surface: reflections over a body of color `tint * opacity`.
Glass seen through glass shows only the nearer surface's refraction (the background ray skips glass).
"""
from std.math import exp, log, max, min, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_CLUSTERS,
    BIND_AAA_ENV_CUBE,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_MATERIALS,
    BIND_AAA_MESHLETS,
    BIND_AAA_SURFCACHE,
    BIND_AAA_TLAS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    BIND_AAA_VSM_TABLE,
    BIND_AAA_VSM_TEX,
    emit_mat4_mul_dir,
    emit_mat4_mul_point,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.aaa_raytracing import (
    AAA_RT_MASK_GLASS,
    AAA_RT_MASK_OPAQUE,
    RAY_FLAG_OPAQUE_KHR,
    emit_rt_textured_hit_shade,
)
from flint.render.aaa_rt_shadow import emit_pixel_camera_ray, emit_visbuffer_surface

comptime AAA_MAX_GLASS_MATERIALS: Int = 32
comptime AAA_INSTANCE_GLASS_BIT: UInt32 = 0x00040000  # instance word 39, bit 18
comptime AAA_GLOBALS_GLASS_COUNT_WORD: Int = 258
comptime AAA_GLOBALS_GLASS_TABLE_WORD: Int = 264  # AAA_MAX_GLASS_MATERIALS entries of AAA_GLASS_ENTRY_WORDS
comptime AAA_GLASS_ENTRY_WORDS: Int = 8


struct AaaGlassParams(Copyable, Movable):
    """A transparent material's optics (the surface itself is the `AaaMaterial` at `material_id`).
    - `ior`: index of refraction (1.0 = no bending, 1.33 water, 1.5 window glass, 2.4 diamond)
    - `tint`: color the light keeps after crossing `1 / density` metres of the body (1, 1, 1 = clear)
    - `opacity`: 0 = fully transmissive .. 1 = an opaque body of color `tint`
    - `density`: absorption scale per metre of path (0 = only the surface tints, once)
    - `roughness`: surface roughness (frosted glass blurs reflections and the sky seen through it)"""

    var material_id: Int
    var ior: Float32
    var tint: Vec3
    var opacity: Float32
    var density: Float32
    var roughness: Float32

    def __init__(
        out self,
        material_id: Int,
        ior: Float32 = 1.5,
        tint: Vec3 = Vec3(1.0, 1.0, 1.0),
        opacity: Float32 = 0.0,
        density: Float32 = 1.0,
        roughness: Float32 = 0.04,
    ):
        self.material_id = material_id
        self.ior = max(Float32(1.0), min(Float32(3.0), ior))
        self.tint = Vec3(
            max(Float32(0.0), min(Float32(1.0), tint.x)),
            max(Float32(0.0), min(Float32(1.0), tint.y)),
            max(Float32(0.0), min(Float32(1.0), tint.z)),
        )
        self.opacity = max(Float32(0.0), min(Float32(1.0), opacity))
        self.density = max(Float32(0.0), density)
        self.roughness = max(Float32(0.02), min(Float32(1.0), roughness))

    def append_words(self, mut out: List[UInt32]):
        out.append(UInt32(self.material_id & 0xFFFF))
        out.append(bitcast[DType.uint32](self.ior))
        out.append(bitcast[DType.uint32](self.tint.x))
        out.append(bitcast[DType.uint32](self.tint.y))
        out.append(bitcast[DType.uint32](self.tint.z))
        out.append(bitcast[DType.uint32](self.opacity))
        out.append(bitcast[DType.uint32](self.density))
        out.append(bitcast[DType.uint32](self.roughness))

    def shadow_transmittance(self) -> Float32:
        """How much sunlight a shadow ray keeps through this glass: the clear fraction, the mean tint
        and two Fresnel crossings at normal incidence."""
        var f0 = glass_f0_cpu(self.ior)
        var t = (self.tint.x + self.tint.y + self.tint.z) / Float32(3.0)
        return (Float32(1.0) - self.opacity) * t * (Float32(1.0) - f0) * (Float32(1.0) - f0)



@always_inline
def glass_f0_cpu(ior: Float32) -> Float32:
    """Normal-incidence reflectance of a dielectric in air: `((n - 1) / (n + 1))^2`."""
    var r = (ior - Float32(1.0)) / (ior + Float32(1.0))
    return r * r


def glass_fresnel_cpu(cos_i: Float32, ior: Float32) -> Float32:
    """Schlick's Fresnel reflectance for light arriving from air at angle `acos(cos_i)`."""
    var f0 = glass_f0_cpu(ior)
    var m = Float32(1.0) - max(Float32(0.0), min(Float32(1.0), cos_i))
    return f0 + (Float32(1.0) - f0) * m * m * m * m * m


def glass_refract_cpu(d: Vec3, n: Vec3, eta: Float32) -> Tuple[Vec3, Bool]:
    """Snell refraction of unit `d` through a surface with unit normal `n` facing against `d`, for
    `eta = n_from / n_to`. Returns `(direction, ok)`; `ok == False` is total internal reflection (the
    returned direction is then the mirror reflection)."""
    var cos_i = -(d.x * n.x + d.y * n.y + d.z * n.z)
    var k = Float32(1.0) - eta * eta * (Float32(1.0) - cos_i * cos_i)
    if k < Float32(0.0):
        var two_c = Float32(2.0) * cos_i
        return (Vec3(d.x + n.x * two_c, d.y + n.y * two_c, d.z + n.z * two_c), False)
    var a = eta * cos_i - sqrt(k)
    return (Vec3(d.x * eta + n.x * a, d.y * eta + n.y * a, d.z * eta + n.z * a), True)


def glass_beer_lambert_cpu(tint: Vec3, density: Float32, path_m: Float32) -> Vec3:
    """Transmittance after `path_m` metres: `tint ^ (density * path)`; a zero-length path (a thin sheet)
    still passes through the tint once."""
    if path_m <= Float32(0.0) or density <= Float32(0.0):
        return tint.copy()
    var k = density * path_m
    return Vec3(
        exp(log(max(Float32(1e-3), tint.x)) * k),
        exp(log(max(Float32(1e-3), tint.y)) * k),
        exp(log(max(Float32(1e-3), tint.z)) * k),
    )


def _emit_refract(
    mut s: AaaSpirvModule, dx: UInt32, dy: UInt32, dz: UInt32, nx: UInt32, ny: UInt32, nz: UInt32, eta: UInt32
) -> Tuple[UInt32, UInt32, UInt32]:
    """SPIR-V twin of `glass_refract_cpu` (falls back to the mirror direction on total internal reflection)."""
    var cos_i = s.fneg(emit_vec3_dot(s, dx, dy, dz, nx, ny, nz))
    var k = s.fsub(s.float(1.0), s.fmul(s.fmul(eta, eta), s.fsub(s.float(1.0), s.fmul(cos_i, cos_i))))
    var tir = s.flt(k, s.float(0.0))
    var a = s.fsub(s.fmul(eta, cos_i), s.fsqrt(s.nmax(k, s.float(0.0))))
    var two_c = s.fmul(s.float(2.0), cos_i)
    var tx = s.select_f(tir, s.fma(nx, two_c, dx), s.fma(dx, eta, s.fmul(nx, a)))
    var ty = s.select_f(tir, s.fma(ny, two_c, dy), s.fma(dy, eta, s.fmul(ny, a)))
    var tz = s.select_f(tir, s.fma(nz, two_c, dz), s.fma(dz, eta, s.fmul(nz, a)))
    return emit_vec3_normalize(s, tx, ty, tz)


def aaa_glass_refract_shader() -> List[UInt32]:
    """Post-denoise refraction pass over glass pixels (`group = 8 x 8`). Push[0] = render_w, [1] = render_h."""
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=8,
        group_height=8,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
        enable_ray_query=True,
    )
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_vsm = s.buffer(BIND_AAA_VSM_TABLE)
    var b_surfcache = s.buffer(BIND_AAA_SURFCACHE)
    var b_clusters = s.buffer(BIND_AAA_CLUSTERS)
    var vis_samp = s.sampled_image_2d(BIND_AAA_VIS_TEX, is_uint=True)
    var vsm_samp = s.sampled_image_2d(BIND_AAA_VSM_TEX)
    var env_cube = s.sampled_image_2d(BIND_AAA_ENV_CUBE, is_cube=True)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var tlas_var = s.acceleration_structure(BIND_AAA_TLAS)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var vis_uv4 = s.texel_fetch_u2d(vis_samp, s.uvec2(px, py), s.uint(0))
    var vis0 = s.extract_u(vis_uv4, 0)
    var vis1 = s.extract_u(vis_uv4, 1)
    s.return_if(s.ieq(vis0, s.uint(0)))
    var inst_id = s.shr(vis0, s.uint(17))
    var w39 = s.load_word(b_instances, s.iadd(s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS))), s.uint(39)))
    s.return_if(s.ieq(s.bit_and(w39, s.uint(AAA_INSTANCE_GLASS_BIT)), s.uint(0)))
    var mat_id = s.bit_and(w39, s.uint(0xFFFF))

    # Find this material's optics in the glass table
    var n_glass = s.umin(s.load_word(b_globals, s.uint(UInt32(AAA_GLOBALS_GLASS_COUNT_WORD))), s.uint(UInt32(AAA_MAX_GLASS_MATERIALS)))
    var entry = s.uvar(s.uint(0xFFFFFFFF))
    var gi_v = s.uvar(s.uint(0))
    s.begin_loop()
    var gi = s.load_u(gi_v)
    s.break_if(s.uge(gi, n_glass))
    var e_off = s.iadd(s.uint(UInt32(AAA_GLOBALS_GLASS_TABLE_WORD)), s.imul(gi, s.uint(UInt32(AAA_GLASS_ENTRY_WORDS))))
    s.begin_if(s.ieq(s.load_word(b_globals, e_off), mat_id))
    s.store(entry, e_off)
    s.end_if()
    s.store(gi_v, s.iadd(gi, s.uint(1)))
    s.end_loop()
    var e_base = s.load_u(entry)
    s.return_if(s.ieq(e_base, s.uint(0xFFFFFFFF)))
    var ior = s.nmax(s.float(1.0), s.load_float(b_globals, s.iadd(e_base, s.uint(1))))
    var tint_r = s.load_float(b_globals, s.iadd(e_base, s.uint(2)))
    var tint_g = s.load_float(b_globals, s.iadd(e_base, s.uint(3)))
    var tint_b = s.load_float(b_globals, s.iadd(e_base, s.uint(4)))
    var opacity = s.load_float(b_globals, s.iadd(e_base, s.uint(5)))
    var density = s.load_float(b_globals, s.iadd(e_base, s.uint(6)))
    var rough = s.load_float(b_globals, s.iadd(e_base, s.uint(7)))

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var cam = emit_pixel_camera_ray(s, b_globals, px, py, fw, fh)
    var dx = cam[3]
    var dy = cam[4]
    var dz = cam[5]
    var surf = emit_visbuffer_surface(s, b_verts, b_indices, b_meshlets, b_instances, vis0, vis1, cam[0], cam[1], cam[2], dx, dy, dz)
    var wx = surf[0]
    var wy = surf[1]
    var wz = surf[2]
    var nx = surf[3]
    var ny = surf[4]
    var nz = surf[5]

    # 1. Fresnel + refraction into the body
    var cos_i = s.nclamp(s.fneg(emit_vec3_dot(s, dx, dy, dz, nx, ny, nz)), s.float(0.0), s.float(1.0))
    var r0 = s.fdiv(s.fsub(ior, s.float(1.0)), s.fadd(ior, s.float(1.0)))
    var f0 = s.fmul(r0, r0)
    var om = s.fsub(s.float(1.0), cos_i)
    var om2 = s.fmul(om, om)
    var fres = s.fma(s.fsub(s.float(1.0), f0), s.fmul(s.fmul(om2, om2), om), f0)
    var inv_ior = s.fdiv(s.float(1.0), ior)
    var t1 = _emit_refract(s, dx, dy, dz, nx, ny, nz, inv_ior)

    var cam_d = s.fsqrt(s.fma(s.fsub(wx, cam[0]), s.fsub(wx, cam[0]), s.fma(s.fsub(wy, cam[1]), s.fsub(wy, cam[1]), s.fmul(s.fsub(wz, cam[2]), s.fsub(wz, cam[2])))))
    var eps = s.fma(cam_d, s.float(0.0004), s.float(0.0015))

    # 2. Where does the refracted ray leave the body? (glass-only trace from just inside the surface)
    var ex = s.fvar(s.fsub(wx, s.fmul(nx, eps)))
    var ey = s.fvar(s.fsub(wy, s.fmul(ny, eps)))
    var ez = s.fvar(s.fsub(wz, s.fmul(nz, eps)))
    var t2x = s.fvar(dx)
    var t2y = s.fvar(dy)
    var t2z = s.fvar(dz)
    var path_l = s.fvar(s.float(0.0))
    var rq = s.ray_query_var()
    s.ray_query_init(rq, tlas_var, s.uint(RAY_FLAG_OPAQUE_KHR), s.uint(AAA_RT_MASK_GLASS), s.vec3(s.load_f(ex), s.load_f(ey), s.load_f(ez)), s.float(0.0005), s.vec3(t1[0], t1[1], t1[2]), s.float(50.0))
    s.begin_loop()
    s.break_if(s.logical_not(s.ray_query_proceed(rq)))
    s.end_loop()
    s.begin_if(s.ine(s.ray_query_committed_type(rq), s.uint(0)))
    var t_hit = s.ray_query_committed_t(rq)
    var h_inst = s.ray_query_committed_instance_id(rq)
    var h_prim = s.ray_query_committed_prim_index(rq)
    var h_bary = s.ray_query_committed_barycentrics(rq)
    var hb1 = s.extract_f(h_bary, 0)
    var hb2 = s.extract_f(h_bary, 1)
    var hb0 = s.fsub(s.fsub(s.float(1.0), hb1), hb2)
    var h_ibase = s.imul(h_inst, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var h_ioff = s.iadd(s.load_word(b_instances, s.iadd(h_ibase, s.uint(38))), s.imul(h_prim, s.uint(3)))
    var hv0 = s.imul(s.load_word(b_indices, h_ioff), s.uint(12))
    var hv1 = s.imul(s.load_word(b_indices, s.iadd(h_ioff, s.uint(1))), s.uint(12))
    var hv2 = s.imul(s.load_word(b_indices, s.iadd(h_ioff, s.uint(2))), s.uint(12))
    var hlx = s.fma(s.load_float(b_verts, s.iadd(hv0, s.uint(3))), hb0, s.fma(s.load_float(b_verts, s.iadd(hv1, s.uint(3))), hb1, s.fmul(s.load_float(b_verts, s.iadd(hv2, s.uint(3))), hb2)))
    var hly = s.fma(s.load_float(b_verts, s.iadd(hv0, s.uint(4))), hb0, s.fma(s.load_float(b_verts, s.iadd(hv1, s.uint(4))), hb1, s.fmul(s.load_float(b_verts, s.iadd(hv2, s.uint(4))), hb2)))
    var hlz = s.fma(s.load_float(b_verts, s.iadd(hv0, s.uint(5))), hb0, s.fma(s.load_float(b_verts, s.iadd(hv1, s.uint(5))), hb1, s.fmul(s.load_float(b_verts, s.iadd(hv2, s.uint(5))), hb2)))
    var hn_raw = emit_mat4_mul_dir(s, b_instances, h_ibase, hlx, hly, hlz)
    var hn = emit_vec3_normalize(s, hn_raw[0], hn_raw[1], hn_raw[2])
    # Normal facing the ray (pointing back into the body)
    var hflip = s.select_f(s.fgt(emit_vec3_dot(s, hn[0], hn[1], hn[2], t1[0], t1[1], t1[2]), s.float(0.0)), s.float(-1.0), s.float(1.0))
    var inx = s.fmul(hn[0], hflip)
    var iny = s.fmul(hn[1], hflip)
    var inz = s.fmul(hn[2], hflip)
    var t2 = _emit_refract(s, t1[0], t1[1], t1[2], inx, iny, inz, ior)
    var hx = s.fma(t1[0], t_hit, s.load_f(ex))
    var hy = s.fma(t1[1], t_hit, s.load_f(ey))
    var hz = s.fma(t1[2], t_hit, s.load_f(ez))
    # Leave on the far side (opposite the inward-facing normal); a reflected ray stays inside, so step back in
    var out_side = s.select_f(s.flt(emit_vec3_dot(s, t2[0], t2[1], t2[2], inx, iny, inz), s.float(0.0)), s.float(-1.0), s.float(1.0))
    s.store(ex, s.fma(inx, s.fmul(eps, out_side), hx))
    s.store(ey, s.fma(iny, s.fmul(eps, out_side), hy))
    s.store(ez, s.fma(inz, s.fmul(eps, out_side), hz))
    s.store(t2x, t2[0])
    s.store(t2y, t2[1])
    s.store(t2z, t2[2])
    s.store(path_l, t_hit)
    s.end_if()

    # 3. What the light that leaves the glass sees: RT-shaded scene, or the sky
    var bx = s.load_f(t2x)
    var by = s.load_f(t2y)
    var bz = s.load_f(t2z)
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var max_env_mip = s.load_float(b_globals, s.uint(103))
    var env = s.sample_cube_lod(env_cube, s.vec3(bx, by, bz), s.fmul(s.nclamp(rough, s.float(0.0), s.float(1.0)), max_env_mip))
    var bg = emit_rt_textured_hit_shade(
        s,
        tlas_var,
        b_verts,
        b_indices,
        b_instances,
        b_materials,
        b_globals,
        b_vsm,
        b_surfcache,
        b_clusters,
        vsm_samp,
        env_cube,
        bindless_arr,
        s.load_f(ex), s.load_f(ey), s.load_f(ez),
        bx, by, bz,
        s.float(0.001), s.float(300.0),
        s.fadd(cam_d, s.load_f(path_l)), s.fmul(rough, rough),
        s.fmul(s.extract_f(env, 0), ibl_int), s.fmul(s.extract_f(env, 1), ibl_int), s.fmul(s.extract_f(env, 2), ibl_int),
        cull_mask=AAA_RT_MASK_OPAQUE,
        prefer_rt_sun_shadow=True,
    )

    # 4. Beer-Lambert absorption along the inside path (a thin sheet tints once)
    var thick = s.fmul(s.load_f(path_l), density)
    var thin = s.fle(thick, s.float(1e-5))
    var tr_r = s.select_f(thin, tint_r, s.exp2(s.fmul(s.log2(s.nmax(tint_r, s.float(1e-3))), thick)))
    var tr_g = s.select_f(thin, tint_g, s.exp2(s.fmul(s.log2(s.nmax(tint_g, s.float(1e-3))), thick)))
    var tr_b = s.select_f(thin, tint_b, s.exp2(s.fmul(s.log2(s.nmax(tint_b, s.float(1e-3))), thick)))
    var k_t = s.fmul(s.fsub(s.float(1.0), fres), s.fsub(s.float(1.0), s.nclamp(opacity, s.float(0.0), s.float(1.0))))

    var pix = s.iadd(s.imul(py, w_u), px)
    var o = s.imul(pix, s.uint(8))
    s.store_float(b_hdr, o, s.fma(s.fmul(bg[0], tr_r), k_t, s.load_float(b_hdr, o)))
    s.store_float(b_hdr, s.iadd(o, s.uint(1)), s.fma(s.fmul(bg[1], tr_g), k_t, s.load_float(b_hdr, s.iadd(o, s.uint(1)))))
    s.store_float(b_hdr, s.iadd(o, s.uint(2)), s.fma(s.fmul(bg[2], tr_b), k_t, s.load_float(b_hdr, s.iadd(o, s.uint(2)))))

    # 5. Static-world motion vector: TSR reprojects the refracted background, not the (spinning) surface
    var jit_x = s.load_float(b_globals, s.uint(101))
    var jit_y = s.load_float(b_globals, s.uint(102))
    var cur_u = s.fdiv(s.fsub(s.fadd(s.float_from_uint(px), s.float(0.5)), s.fmul(jit_x, s.fmul(fw, s.float(0.5)))), fw)
    var cur_v = s.fdiv(s.fadd(s.fadd(s.float_from_uint(py), s.float(0.5)), s.fmul(jit_y, s.fmul(fh, s.float(0.5)))), fh)
    var p_clip = emit_mat4_mul_point(s, b_globals, s.uint(16), wx, wy, wz)
    var inv_pw = s.fdiv(s.float(1.0), s.nmax(s.float(1e-4), p_clip[3]))
    s.store_float(b_hdr, s.iadd(o, s.uint(4)), s.fsub(cur_u, s.fma(s.fmul(p_clip[0], inv_pw), s.float(0.5), s.float(0.5))))
    s.store_float(b_hdr, s.iadd(o, s.uint(5)), s.fsub(cur_v, s.fma(s.fneg(s.fmul(p_clip[1], inv_pw)), s.float(0.5), s.float(0.5))))
    return s.finish()
