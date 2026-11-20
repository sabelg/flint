"""Anime / Stylized Non-Photorealistic Cel Shading & Hardware Clip-Space Outlines (`aaa_npr_cel.mojo`).

Implements first-class anime character & stylized prop shading inside the unified Vulkan AAA pipeline:
  1. **Hardware Clip-Space Variable-Width Outlines (`npr_outline_vertex_shader`, `npr_outline_fragment_shader`)**:
     Zero-duplicate-mesh `CULL_MODE_FRONT` hardware graphics pass inside `VisBuffer` dynamic rendering.
     Extrudes vertices along screen-space projected normals by `(2 * outline_width_px * vtx_w / render_h)`
     scaled by `min(clip_w, outline_fov_dist_cap)` so line thickness stays crisp during close-up zooms
     and naturally tapers at distance, with optional per-vertex width (`G` channel) and Z-bias (`B` channel).
  2. **ILM + SSS + Inner-Ink + Stepped Specular + Anisotropic Hair + Fresnel Rim (`aaa_npr_cel_shade_shader`)**:
     Companion `AaaCelParams` table in `BIND_AAA_GLOBALS` resolved at `render_w x render_h` after `SVGF`,
     combining real hardware RT / `16k` VSM sun shadows and stepped clustered lights with `ILM` maps
     (`R`=spec mask, `G`=shadow threshold bias, `B`=spec sharpness/metallic, `A`=inner ink lines),
     authored `SSS` shadow tint/textures, Kajiya-Kay hair halo bands, and light-gated Fresnel rim light —
     100% immune to stochastic GI grain while coexisting with ray-traced environments!
"""
from std.math import exp, log, max, min, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.render.spirv import (
    AaaSpirvModule,
    MODEL_FRAGMENT,
    MODEL_GL_COMPUTE,
    MODEL_VERTEX,
    OP_LOAD,
    OP_STORE,
    STORAGE_INPUT,
    STORAGE_OUTPUT,
    emit,
)
from flint.render.aaa_bsdf import AAA_MATERIAL_WORDS
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_CLUSTERS,
    BIND_AAA_DRAW_PAIRS,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_MATERIALS,
    BIND_AAA_MESHLETS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    BIND_AAA_VSM_TABLE,
    BIND_AAA_VSM_TEX,
    emit_mat4_mul_dir,
    emit_mat4_mul_point,
    emit_ray_barycentrics,
    emit_vec3_cross,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.aaa_vsm import emit_vsm_pcss_shadow
from flint.render.aaa_clustered_lights import (
    AAA_CLUSTER_COUNTS_BASE,
    AAA_CLUSTER_INDICES_BASE,
    AAA_CLUSTER_LIGHT_WORDS,
    AAA_CLUSTER_LIGHTS_BASE,
    AAA_CLUSTER_X,
    AAA_CLUSTER_Y,
    AAA_CLUSTER_Z,
    AAA_MAX_LIGHTS_PER_CLUSTER,
)
from flint.render.aaa_npr_vfx import (
    emit_eval_vfx_excess_and_color_spv,
    emit_find_vfx_entry_spv,
)

# `BIND_AAA_GLOBALS` Cel companion table (`640`, `704 .. 2495`)
comptime AAA_GLOBALS_CEL_COUNT_WORD: Int = 640
comptime AAA_GLOBALS_CEL_TABLE_WORD: Int = 704      # 64 * 28 = 1792 words: [704 .. 2495]
comptime AAA_MAX_CEL_MATERIALS: Int = 64
comptime AAA_CEL_ENTRY_WORDS: Int = 28
comptime AAA_CEL_NO_OUTLINE_MAT: Int = 0x7FFF

comptime AAA_CEL_FLAG_RECEIVE_SHADOWS: UInt32 = 1
comptime AAA_CEL_FLAG_HAIR_ANISO: UInt32 = 2
comptime AAA_CEL_FLAG_CUSTOM_LIGHT_DIR: UInt32 = 4


struct AaaCelParams(Copyable, Movable):
    """Stylized NPR Cel + Outline parameters paired with an `AaaMaterial` (`material_id`)."""
    var material_id: Int
    var outline_material_id: Int
    var sss_shadow_tint: Vec3           # Multiplied with `sss_tex` (or `albedo`) in the shadow band
    var sss_tex: Int                    # Optional bindless `BC7_SRGB` authored shadow/SSS texture (-1 = use albedo)
    var ilm_tex: Int                    # Optional bindless `BC7_UNORM` ILM map: R=spec, G=shadow bias, B=sharp/metal, A=inner ink
    var shadow_threshold: Float32       # Half-Lambert + shadow ramp threshold in [0..1] (default 0.50)
    var shadow_feather: Float32         # Anti-aliased ramp half-width (default 0.015 for crisp 1px cel edge)
    var rim_color: Vec3                 # Linear HDR rim-light tint
    var rim_power: Float32              # Fresnel exponent `(1 - N.V)^rim_power`
    var rim_threshold: Float32          # Stepped Fresnel rim cutoff in [0..1]
    var spec_intensity: Float32         # Stepped stylized highlight strength (modulated by `ILM.r`)
    var spec_step_threshold: Float32    # Hard step cutoff on `pow(N.H, exp)`
    var hair_halo_strength: Float32     # Kajiya-Kay shifted-tangent anisotropic hair band strength (`0.0` = off)
    var hair_shift: Float32             # Strand normal shift for hair highlight band
    var outline_width_px: Float32       # Screen-space outline width in pixels (`0.0` = no outline draw)
    var outline_color: Vec3             # Linear RGB ink color of hardware silhouette & `ILM.a` inner lines
    var outline_z_bias: Float32         # Reverse-Z depth pushback for back-facing outline hull
    var outline_fov_dist_cap: Float32   # Max camera distance (m) for zoom-invariant constant pixel width
    var inner_ink_strength: Float32     # Strength of `1 - ILM.a` inner hand-drawn ink lines (`0..1`)
    var ambient_floor: Float32          # Minimum clean ungrained shadow illumination floor
    var use_vertex_color_channels: Bool # True: R=shadow bias, G=outline width [0..1], B=outline Z-bias
    var receive_scene_shadows: Bool     # True: RT/VSM sun shadows drive the cel shadow band
    var use_custom_light_dir: Bool      # True: decouples character cel key-light direction from world sun (ArcSys hero light!)
    var custom_light_dir: Vec3          # World-space artistic key-light vector toward light when `use_custom_light_dir == True`
    var custom_light_intensity: Float32 # Multiplier on scene key radiance when `use_custom_light_dir == True`

    def __init__(
        out self,
        material_id: Int = 0,
        sss_shadow_tint: Vec3 = Vec3(0.58, 0.40, 0.44),
        sss_tex: Int = -1,
        ilm_tex: Int = -1,
        shadow_threshold: Float32 = 0.50,
        shadow_feather: Float32 = 0.015,
        rim_color: Vec3 = Vec3(0.42, 0.48, 0.62),
        rim_power: Float32 = 3.5,
        rim_threshold: Float32 = 0.65,
        spec_intensity: Float32 = 0.32,
        spec_step_threshold: Float32 = 0.86,
        hair_halo_strength: Float32 = 0.0,
        hair_shift: Float32 = 0.12,
        outline_width_px: Float32 = 2.0,
        outline_color: Vec3 = Vec3(0.018, 0.014, 0.020),
        outline_z_bias: Float32 = 0.0025,
        outline_fov_dist_cap: Float32 = 6.0,
        inner_ink_strength: Float32 = 1.0,
        ambient_floor: Float32 = 0.28,
        use_vertex_color_channels: Bool = False,
        receive_scene_shadows: Bool = True,
        use_custom_light_dir: Bool = False,
        custom_light_dir: Vec3 = Vec3(0.0, 0.55, -0.835),
        custom_light_intensity: Float32 = 1.0,
    ):
        self.material_id = material_id
        self.outline_material_id = AAA_CEL_NO_OUTLINE_MAT
        self.sss_shadow_tint = sss_shadow_tint
        self.sss_tex = sss_tex
        self.ilm_tex = ilm_tex
        self.shadow_threshold = max(Float32(0.02), min(Float32(0.98), shadow_threshold))
        self.shadow_feather = max(Float32(0.001), min(Float32(0.45), shadow_feather))
        self.rim_color = rim_color
        self.rim_power = max(Float32(0.5), min(Float32(16.0), rim_power))
        self.rim_threshold = max(Float32(0.05), min(Float32(0.98), rim_threshold))
        self.spec_intensity = max(Float32(0.0), spec_intensity)
        self.spec_step_threshold = max(Float32(0.05), min(Float32(0.995), spec_step_threshold))
        self.hair_halo_strength = max(Float32(0.0), hair_halo_strength)
        self.hair_shift = max(Float32(-1.0), min(Float32(1.0), hair_shift))
        self.outline_width_px = max(Float32(0.0), min(Float32(24.0), outline_width_px))
        self.outline_color = outline_color
        self.outline_z_bias = max(Float32(0.0), min(Float32(0.25), outline_z_bias))
        self.outline_fov_dist_cap = max(Float32(0.25), outline_fov_dist_cap)
        self.inner_ink_strength = max(Float32(0.0), min(Float32(1.0), inner_ink_strength))
        self.ambient_floor = max(Float32(0.02), min(Float32(2.0), ambient_floor))
        self.use_vertex_color_channels = use_vertex_color_channels
        self.receive_scene_shadows = receive_scene_shadows
        self.use_custom_light_dir = use_custom_light_dir
        self.custom_light_dir = custom_light_dir.normalized() if custom_light_dir.length_squared() > Float32(1e-8) else Vec3(0.0, 0.55, -0.835).normalized()
        self.custom_light_intensity = max(Float32(0.05), min(Float32(8.0), custom_light_intensity))

    @staticmethod
    def skin(
        sss_shadow_tint: Vec3 = Vec3(0.74, 0.44, 0.42),
        outline_width_px: Float32 = 1.75,
        outline_color: Vec3 = Vec3(0.032, 0.016, 0.018),
    ) -> Self:
        return Self(
            sss_shadow_tint=sss_shadow_tint,
            shadow_threshold=0.48,
            shadow_feather=0.018,
            rim_color=Vec3(0.35, 0.32, 0.38),
            rim_power=4.0,
            rim_threshold=0.72,
            spec_intensity=0.08,
            spec_step_threshold=0.92,
            outline_width_px=outline_width_px,
            outline_color=outline_color,
        )

    @staticmethod
    def hair(
        sss_shadow_tint: Vec3 = Vec3(0.52, 0.38, 0.48),
        halo_strength: Float32 = 0.65,
        outline_width_px: Float32 = 1.9,
        outline_color: Vec3 = Vec3(0.020, 0.015, 0.024),
    ) -> Self:
        return Self(
            sss_shadow_tint=sss_shadow_tint,
            shadow_threshold=0.50,
            shadow_feather=0.015,
            rim_color=Vec3(0.48, 0.52, 0.68),
            rim_power=3.5,
            rim_threshold=0.66,
            spec_intensity=0.22,
            spec_step_threshold=0.88,
            hair_halo_strength=halo_strength,
            hair_shift=0.14,
            outline_width_px=outline_width_px,
            outline_color=outline_color,
        )

    @staticmethod
    def cloth(
        sss_shadow_tint: Vec3 = Vec3(0.46, 0.38, 0.48),
        rim_color: Vec3 = Vec3(0.50, 0.55, 0.72),
        outline_width_px: Float32 = 2.2,
        outline_color: Vec3 = Vec3(0.015, 0.012, 0.018),
    ) -> Self:
        return Self(
            sss_shadow_tint=sss_shadow_tint,
            shadow_threshold=0.50,
            shadow_feather=0.014,
            rim_color=rim_color,
            rim_power=3.2,
            rim_threshold=0.64,
            spec_intensity=0.24,
            spec_step_threshold=0.86,
            outline_width_px=outline_width_px,
            outline_color=outline_color,
        )

    @staticmethod
    def metal(
        sss_shadow_tint: Vec3 = Vec3(0.38, 0.40, 0.50),
        spec_intensity: Float32 = 1.15,
        outline_width_px: Float32 = 2.2,
        outline_color: Vec3 = Vec3(0.012, 0.012, 0.016),
    ) -> Self:
        return Self(
            sss_shadow_tint=sss_shadow_tint,
            shadow_threshold=0.48,
            shadow_feather=0.012,
            rim_color=Vec3(0.75, 0.82, 0.98),
            rim_power=2.8,
            rim_threshold=0.58,
            spec_intensity=spec_intensity,
            spec_step_threshold=0.72,
            outline_width_px=outline_width_px,
            outline_color=outline_color,
        )

    def append_words(self, mut out: List[UInt32]):
        var out_id_u = UInt32(self.outline_material_id & 0x7FFF) if self.outline_material_id >= 0 else UInt32(AAA_CEL_NO_OUTLINE_MAT)
        var vc_bit = UInt32(0x80000000) if self.use_vertex_color_channels else UInt32(0)
        var w0 = UInt32(self.material_id & 0xFFFF) | (out_id_u << UInt32(16)) | vc_bit

        var sss_u = UInt32(self.sss_tex & 0xFF) if (self.sss_tex >= 0 and self.sss_tex < 255) else UInt32(0xFF)
        var ilm_u = UInt32(self.ilm_tex & 0xFF) if (self.ilm_tex >= 0 and self.ilm_tex < 255) else UInt32(0xFF)
        var flags: UInt32 = 0
        if self.receive_scene_shadows:
            flags |= AAA_CEL_FLAG_RECEIVE_SHADOWS
        if self.hair_halo_strength > Float32(1e-4):
            flags |= AAA_CEL_FLAG_HAIR_ANISO
        if self.use_custom_light_dir:
            flags |= AAA_CEL_FLAG_CUSTOM_LIGHT_DIR
        var w1 = sss_u | (ilm_u << UInt32(8)) | (flags << UInt32(16))

        out.append(w0)
        out.append(w1)
        out.append(bitcast[DType.uint32](self.sss_shadow_tint.x))
        out.append(bitcast[DType.uint32](self.sss_shadow_tint.y))
        out.append(bitcast[DType.uint32](self.sss_shadow_tint.z))
        out.append(bitcast[DType.uint32](self.shadow_threshold))
        out.append(bitcast[DType.uint32](self.shadow_feather))
        out.append(bitcast[DType.uint32](self.rim_color.x))
        out.append(bitcast[DType.uint32](self.rim_color.y))
        out.append(bitcast[DType.uint32](self.rim_color.z))
        out.append(bitcast[DType.uint32](self.rim_power))
        out.append(bitcast[DType.uint32](self.rim_threshold))
        out.append(bitcast[DType.uint32](self.spec_intensity))
        out.append(bitcast[DType.uint32](self.spec_step_threshold))
        out.append(bitcast[DType.uint32](self.hair_halo_strength))
        out.append(bitcast[DType.uint32](self.hair_shift))
        out.append(bitcast[DType.uint32](self.outline_width_px))
        out.append(bitcast[DType.uint32](self.outline_color.x))
        out.append(bitcast[DType.uint32](self.outline_color.y))
        out.append(bitcast[DType.uint32](self.outline_color.z))
        out.append(bitcast[DType.uint32](self.outline_z_bias))
        out.append(bitcast[DType.uint32](self.outline_fov_dist_cap))
        out.append(bitcast[DType.uint32](self.inner_ink_strength))
        out.append(bitcast[DType.uint32](self.ambient_floor))
        out.append(bitcast[DType.uint32](self.custom_light_dir.x))
        out.append(bitcast[DType.uint32](self.custom_light_dir.y))
        out.append(bitcast[DType.uint32](self.custom_light_dir.z))
        out.append(bitcast[DType.uint32](self.custom_light_intensity))


@always_inline
def _smoothstep_cpu(edge0: Float32, edge1: Float32, x: Float32) -> Float32:
    var span = max(Float32(1e-5), edge1 - edge0)
    var t = max(Float32(0.0), min(Float32(1.0), (x - edge0) / span))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


def eval_npr_cel_cpu(
    cel: AaaCelParams,
    albedo: Vec3,
    sss_tex_rgb: Vec3,
    has_sss_tex: Bool,
    ilm_rgba: Tuple[Float32, Float32, Float32, Float32],
    normal: Vec3,
    bitangent: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    sun_radiance: Vec3,
    sun_vis: Float32,
    sh0_ambient: Vec3,
    vc_shadow_r: Float32 = 0.5,
) -> Vec3:
    """CPU reference matching `aaa_npr_cel_shade_shader` for unit & metrology verification."""
    var n = normal.normalized()
    var v = view_dir.normalized()
    var l = cel.custom_light_dir.normalized() if cel.use_custom_light_dir else sun_dir.normalized()
    var eff_sun_rad = (sun_radiance * cel.custom_light_intensity) if cel.use_custom_light_dir else sun_radiance

    var ilm_r = ilm_rgba[0]
    var ilm_g = ilm_rgba[1]
    var ilm_b = ilm_rgba[2]
    var ilm_a = ilm_rgba[3]

    var hl = n.dot(l) * Float32(0.5) + Float32(0.5)
    var eff_vis = Float32(1.0) if cel.use_custom_light_dir else (sun_vis if cel.receive_scene_shadows else Float32(1.0))
    var sun_luma = eff_sun_rad.x * Float32(0.2126) + eff_sun_rad.y * Float32(0.7152) + eff_sun_rad.z * Float32(0.0722)
    var base_sig = min(hl, eff_vis) if sun_luma > Float32(1e-4) else Float32(0.0)
    var ilm_bias = (ilm_g - Float32(0.5)) * Float32(2.0)
    var vc_bias = (vc_shadow_r - Float32(0.5)) * Float32(1.5) if cel.use_vertex_color_channels else Float32(0.0)
    var cel_sig = base_sig + ilm_bias + vc_bias
    var lit_band = _smoothstep_cpu(
        cel.shadow_threshold - cel.shadow_feather,
        cel.shadow_threshold + cel.shadow_feather,
        cel_sig,
    )

    var sun_key = eff_sun_rad * Float32(0.34)
    var amb = Vec3(
        max(cel.ambient_floor, sh0_ambient.x * Float32(0.35)),
        max(cel.ambient_floor, sh0_ambient.y * Float32(0.35)),
        max(cel.ambient_floor, sh0_ambient.z * Float32(0.35)),
    )
    var sh_base = sss_tex_rgb if has_sss_tex else albedo
    var sh_alb = Vec3(
        sh_base.x * cel.sss_shadow_tint.x,
        sh_base.y * cel.sss_shadow_tint.y,
        sh_base.z * cel.sss_shadow_tint.z,
    )
    var lit_rig = sun_key + amb * Float32(0.45)
    var shd_rig = amb + sun_key * Float32(0.22)
    var lit_rgb = Vec3(albedo.x * lit_rig.x, albedo.y * lit_rig.y, albedo.z * lit_rig.z)
    var shd_rgb = Vec3(sh_alb.x * shd_rig.x, sh_alb.y * shd_rig.y, sh_alb.z * shd_rig.z)
    var out_rgb = shd_rgb * (Float32(1.0) - lit_band) + lit_rgb * lit_band

    # Stepped stylized specular
    var h_raw = l + v
    var h = h_raw.normalized() if h_raw.length_squared() > Float32(1e-8) else n
    var ndoth = max(Float32(1e-6), n.dot(h))
    var spec_exp = Float32(16.0) + ilm_b * Float32(96.0)
    var spec_pow = exp(log(ndoth) * spec_exp)
    var spec_step = _smoothstep_cpu(cel.spec_step_threshold - Float32(0.02), cel.spec_step_threshold + Float32(0.02), spec_pow)
    var spec_amt = spec_step * cel.spec_intensity * ilm_r * lit_band
    var spec_tint = Vec3(1.0, 1.0, 1.0) * (Float32(1.0) - ilm_b) + albedo * ilm_b
    out_rgb = out_rgb + Vec3(spec_tint.x * sun_key.x, spec_tint.y * sun_key.y, spec_tint.z * sun_key.z) * spec_amt

    # Kajiya-Kay shifted-strand hair halo band
    if cel.hair_halo_strength > Float32(1e-4):
        var t_sh = (bitangent + n * cel.hair_shift).normalized()
        var th = t_sh.dot(h)
        var sin_th = sqrt(max(Float32(1e-6), Float32(1.0) - th * th))
        var h_pow = exp(log(sin_th) * Float32(48.0))
        var h_step = _smoothstep_cpu(Float32(0.80), Float32(0.86), h_pow)
        out_rgb = out_rgb + sun_key * (h_step * cel.hair_halo_strength * ilm_r * lit_band)

    # Light-gated Fresnel rim light
    var ndotv = max(Float32(0.0), min(Float32(1.0), n.dot(v)))
    var fres = exp(log(max(Float32(1e-6), Float32(1.0) - ndotv)) * cel.rim_power)
    var rim_step = _smoothstep_cpu(cel.rim_threshold - Float32(0.04), cel.rim_threshold + Float32(0.04), fres)
    var rim_gate = Float32(0.20) + Float32(0.80) * lit_band
    out_rgb = out_rgb + cel.rim_color * (rim_step * rim_gate)

    # Inner ink lines from `ILM.a` (`1.0` = clean, `0.0` = solid outline ink)
    var line_keep = Float32(1.0) - cel.inner_ink_strength * (Float32(1.0) - _smoothstep_cpu(Float32(0.35), Float32(0.65), ilm_a))
    out_rgb = cel.outline_color * (Float32(1.0) - line_keep) + out_rgb * line_keep
    return out_rgb


# ============================================================================
# Hardware Clip-Space Variable-Width Outline Shaders (`CULL_MODE_FRONT`)
# ============================================================================

def _emit_smoothstep_spv(mut s: AaaSpirvModule, e0: UInt32, e1: UInt32, x: UInt32) -> UInt32:
    var span = s.nmax(s.float(1e-5), s.fsub(e1, e0))
    var t = s.nclamp(s.fdiv(s.fsub(x, e0), span), s.float(0.0), s.float(1.0))
    return s.fmul(t, s.fmul(t, s.fma(t, s.float(-2.0), s.float(3.0))))


def npr_outline_vertex_shader() -> List[UInt32]:
    """Hardware Clip-Space Outline Extrusion Vertex Shader (`CULL_MODE_FRONT`, Reverse-Z).
    Push[0] = vp_off (`0`), Push[1] = jx_clip, Push[2] = jy_clip, Push[3] = render_w, Push[4] = render_h.
    """
    var s = AaaSpirvModule(MODEL_VERTEX, push_words=AAA_PUSH_WORDS)
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_draw_pairs = s.buffer(BIND_AAA_DRAW_PAIRS)
    _ = b_indices
    var out_vis = s.add_stage_io(STORAGE_OUTPUT, s.type_uvec2, location=0, flat=True)

    var real_vtx = s.vertex_index()
    var draw_slot = s.instance_index()
    var dp_off = s.imul(draw_slot, s.uint(2))
    var inst_id = s.load_word(b_draw_pairs, dp_off)
    var ml_id = s.load_word(b_draw_pairs, s.iadd(dp_off, s.uint(1)))

    var ml_base = s.imul(ml_id, s.uint(12))
    var ml_mat = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(11)))

    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var mat_override = s.load_word(b_instances, s.iadd(inst_base, s.uint(34)))
    var mat_ov16 = s.bit_and(mat_override, s.uint(0xFFFF))
    var final_mat = s.select_u(s.ieq(mat_ov16, s.uint(0xFFFF)), ml_mat, mat_ov16)

    # Locate matching `AaaCelParams` entry in `b_globals`
    var n_cel = s.umin(
        s.load_word(b_globals, s.uint(UInt32(AAA_GLOBALS_CEL_COUNT_WORD))),
        s.uint(UInt32(AAA_MAX_CEL_MATERIALS)),
    )
    var cel_e_v = s.uvar(s.uint(0xFFFFFFFF))
    var ci_v = s.uvar(s.uint(0))
    s.begin_loop()
    var ci = s.load_u(ci_v)
    s.break_if(s.uge(ci, n_cel))
    var e_off = s.iadd(s.uint(UInt32(AAA_GLOBALS_CEL_TABLE_WORD)), s.imul(ci, s.uint(UInt32(AAA_CEL_ENTRY_WORDS))))
    var w0_e = s.load_word(b_globals, e_off)
    s.begin_if(s.ieq(s.bit_and(w0_e, s.uint(0xFFFF)), final_mat))
    s.store(cel_e_v, e_off)
    s.break_if(s.ieq(s.uint(1), s.uint(1)))
    s.end_if()
    s.store(ci_v, s.iadd(ci, s.uint(1)))
    s.end_loop()

    var cel_e = s.load_u(cel_e_v)
    var out_mat_v = s.uvar(s.uint(UInt32(AAA_CEL_NO_OUTLINE_MAT)))
    var use_vc_v = s.uvar(s.uint(0))
    var out_px_v = s.fvar(s.float(0.0))
    var z_bias_v = s.fvar(s.float(0.0025))
    var fov_cap_v = s.fvar(s.float(6.0))

    s.begin_if(s.ine(cel_e, s.uint(0xFFFFFFFF)))
    var w0_c = s.load_word(b_globals, cel_e)
    s.store(out_mat_v, s.bit_and(s.shr(w0_c, s.uint(16)), s.uint(0x7FFF)))
    s.store(use_vc_v, s.shr(w0_c, s.uint(31)))
    s.store(out_px_v, s.load_float(b_globals, s.iadd(cel_e, s.uint(16))))
    s.store(z_bias_v, s.load_float(b_globals, s.iadd(cel_e, s.uint(20))))
    s.store(fov_cap_v, s.load_float(b_globals, s.iadd(cel_e, s.uint(21))))
    s.end_if()

    var v_base = s.imul(real_vtx, s.uint(12))
    var lpx = s.load_float(b_verts, v_base)
    var lpy = s.load_float(b_verts, s.iadd(v_base, s.uint(1)))
    var lpz = s.load_float(b_verts, s.iadd(v_base, s.uint(2)))
    var lnx = s.load_float(b_verts, s.iadd(v_base, s.uint(3)))
    var lny = s.load_float(b_verts, s.iadd(v_base, s.uint(4)))
    var lnz = s.load_float(b_verts, s.iadd(v_base, s.uint(5)))
    var meta = s.load_word(b_verts, s.iadd(v_base, s.uint(11)))

    var use_vc = s.ine(s.load_u(use_vc_v), s.uint(0))
    var inv255 = s.float(1.0 / 255.0)
    var vc_g = s.fmul(s.float_from_uint(s.bit_and(s.shr(meta, s.uint(16)), s.uint(0xFF))), inv255)
    var vc_b = s.fmul(s.float_from_uint(s.shr(meta, s.uint(24))), inv255)
    var vtx_w = s.select_f(use_vc, vc_g, s.float(1.0))
    var z_w = s.select_f(use_vc, vc_b, s.float(1.0))

    # Cofactor world normal so non-uniformly scaled instances still extrude orthogonally
    var c0x = s.load_float(b_instances, inst_base)
    var c0y = s.load_float(b_instances, s.iadd(inst_base, s.uint(1)))
    var c0z = s.load_float(b_instances, s.iadd(inst_base, s.uint(2)))
    var c1x = s.load_float(b_instances, s.iadd(inst_base, s.uint(4)))
    var c1y = s.load_float(b_instances, s.iadd(inst_base, s.uint(5)))
    var c1z = s.load_float(b_instances, s.iadd(inst_base, s.uint(6)))
    var c2x = s.load_float(b_instances, s.iadd(inst_base, s.uint(8)))
    var c2y = s.load_float(b_instances, s.iadd(inst_base, s.uint(9)))
    var c2z = s.load_float(b_instances, s.iadd(inst_base, s.uint(10)))
    var cof0 = emit_vec3_cross(s, c1x, c1y, c1z, c2x, c2y, c2z)
    var cof1 = emit_vec3_cross(s, c2x, c2y, c2z, c0x, c0y, c0z)
    var cof2 = emit_vec3_cross(s, c0x, c0y, c0z, c1x, c1y, c1z)
    var det_m = emit_vec3_dot(s, c0x, c0y, c0z, cof0[0], cof0[1], cof0[2])
    var det_s = s.select_f(s.flt(det_m, s.float(0.0)), s.float(-1.0), s.float(1.0))
    var wnx_r = s.fmul(s.fma(cof0[0], lnx, s.fma(cof1[0], lny, s.fmul(cof2[0], lnz))), det_s)
    var wny_r = s.fmul(s.fma(cof0[1], lnx, s.fma(cof1[1], lny, s.fmul(cof2[1], lnz))), det_s)
    var wnz_r = s.fmul(s.fma(cof0[2], lnx, s.fma(cof1[2], lny, s.fmul(cof2[2], lnz))), det_s)
    var wn = emit_vec3_normalize(s, wnx_r, wny_r, wnz_r)

    var wp = emit_mat4_mul_point(s, b_instances, inst_base, lpx, lpy, lpz)
    var vp_off = s.push_word(0)
    var clip = emit_mat4_mul_point(s, b_globals, vp_off, wp[0], wp[1], wp[2])
    var n_clip = emit_mat4_mul_dir(s, b_globals, vp_off, wn[0], wn[1], wn[2])

    var fw = s.nmax(s.float(1.0), s.float_from_uint(s.push_word(3)))
    var fh = s.nmax(s.float(1.0), s.float_from_uint(s.push_word(4)))
    var near_z = s.nmax(s.float(0.01), s.load_float(b_globals, s.uint(35)))

    var npx = s.fmul(n_clip[0], fw)
    var npy = s.fmul(n_clip[1], fh)
    var nl2 = s.fma(npx, npx, s.fmul(npy, npy))
    var inv_nl = s.select_f(s.fgt(nl2, s.float(1e-12)), s.inverse_sqrt(s.nmax(s.float(1e-12), nl2)), s.float(0.0))
    var dir_x = s.fmul(npx, inv_nl)
    var dir_y = s.fmul(npy, inv_nl)

    var eff_px = s.fmul(s.load_f(out_px_v), vtx_w)
    var w_cap = s.nmin(s.nmax(clip[3], near_z), s.nmax(near_z, s.load_f(fov_cap_v)))
    var dx_clip = s.fmul(dir_x, s.fmul(s.fdiv(s.fmul(s.float(2.0), eff_px), fw), w_cap))
    var dy_clip = s.fmul(dir_y, s.fmul(s.fdiv(s.fmul(s.float(2.0), eff_px), fh), w_cap))

    var jx = s.push_float(1)
    var jy = s.push_float(2)
    var cx_jit = s.fadd(s.fma(jx, clip[3], clip[0]), dx_clip)
    var cy_jit = s.fadd(s.fma(jy, clip[3], clip[1]), dy_clip)
    var cz_out = s.nmax(s.float(0.0), s.fsub(clip[2], s.fmul(s.load_f(z_bias_v), z_w)))

    var out_mat = s.load_u(out_mat_v)
    var valid_out = s.logical_and(
        s.ult(out_mat, s.uint(UInt32(AAA_CEL_NO_OUTLINE_MAT))),
        s.fgt(eff_px, s.float(0.001)),
    )
    var final_cx = s.select_f(valid_out, cx_jit, s.float(0.0))
    var final_cy = s.select_f(valid_out, cy_jit, s.float(0.0))
    var final_cz = s.select_f(valid_out, cz_out, s.float(-2.0))
    var final_cw = s.select_f(valid_out, clip[3], s.float(1.0))
    s.write_clip_position(final_cx, final_cy, final_cz, final_cw)

    var pay0 = s.bit_or(
        s.shl(s.bit_and(inst_id, s.uint(0x3FF)), s.uint(22)),
        s.umin(s.iadd(ml_id, s.uint(1)), s.uint(0x3FFFFF)),
    )
    var pay1_base = s.bit_or(
        s.shl(out_mat, s.uint(16)),
        s.shl(s.bit_and(s.shr(inst_id, s.uint(10)), s.uint(0xFF)), s.uint(8)),
    )
    emit(s.code, OP_STORE, out_vis[1], s.uvec2(pay0, pay1_base))
    return s.finish()


def npr_outline_fragment_shader() -> List[UInt32]:
    """Hardware Outline Fragment Shader (`EarlyFragmentTests`) writing `outline_material_id` into `VisBuffer`."""
    var s = AaaSpirvModule(MODEL_FRAGMENT, push_words=AAA_PUSH_WORDS, early_depth=True)
    var prim_id = s.primitive_id()
    var in_vis = s.add_stage_io(STORAGE_INPUT, s.type_uvec2, location=0, flat=True)
    var out_col = s.add_stage_io(STORAGE_OUTPUT, s.type_uvec4, location=0, flat=False)

    var vis_val = s.fresh()
    emit(s.code, OP_LOAD, s.type_uvec2, vis_val, in_vis[1])
    var w0 = s.extract_u(vis_val, 0)
    var w1_base = s.extract_u(vis_val, 1)
    var w1 = s.bit_or(w1_base, s.bit_and(prim_id, s.uint(0xFF)))
    emit(s.code, OP_STORE, out_col[1], s.uvec4(w0, w1, s.uint(0), s.uint(1)))
    return s.finish()


# ============================================================================
# Compute NPR Cel, Outline & Stylized 3D Mesh VFX Resolve Pass (`8 x 8`)
# ============================================================================

def aaa_npr_cel_shade_shader() -> List[UInt32]:
    """Resolves `AaaCelParams`, hardware `outline_material_id` silhouettes, and `AaaStylizedVfxParams`
    meshes into `BIND_AAA_HDR_OUT` (`rgb` only, preserving `z_lin` & 3D motion vectors for `TSR v3`!).
    Push[0] = render_w, Push[1] = render_h, Push[2] = rt_sun_filtered_offset_words (`0` = VSM fallback).
    """
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=8,
        group_height=8,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
    )
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var b_vsm = s.buffer(BIND_AAA_VSM_TABLE)
    var b_clusters = s.buffer(BIND_AAA_CLUSTERS)
    var vis_samp = s.sampled_image_2d(BIND_AAA_VIS_TEX, is_uint=True)
    var vsm_samp = s.sampled_image_2d(BIND_AAA_VSM_TEX)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var rt_sun_off = s.push_word(2)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var vis_uv4 = s.texel_fetch_u2d(vis_samp, s.uvec2(px, py), s.uint(0))
    var vis0 = s.extract_u(vis_uv4, 0)
    var vis1 = s.extract_u(vis_uv4, 1)
    s.return_if(s.ieq(vis0, s.uint(0)))

    var mat_id = s.shr(vis1, s.uint(16))
    var pix_idx = s.iadd(s.imul(py, w_u), px)
    var hdr_base = s.imul(pix_idx, s.uint(8))

    # Scan Cel table for either `material_id == mat_id` OR `outline_material_id == mat_id`
    var n_cel = s.umin(
        s.load_word(b_globals, s.uint(UInt32(AAA_GLOBALS_CEL_COUNT_WORD))),
        s.uint(UInt32(AAA_MAX_CEL_MATERIALS)),
    )
    var cel_e_v = s.uvar(s.uint(0xFFFFFFFF))
    var out_e_v = s.uvar(s.uint(0xFFFFFFFF))
    s.begin_if(s.ult(s.uint(0), n_cel))
    var ci_v = s.uvar(s.uint(0))
    s.begin_loop()
    var ci = s.load_u(ci_v)
    s.break_if(s.uge(ci, n_cel))
    var e_off = s.iadd(s.uint(UInt32(AAA_GLOBALS_CEL_TABLE_WORD)), s.imul(ci, s.uint(UInt32(AAA_CEL_ENTRY_WORDS))))
    var w0_e = s.load_word(b_globals, e_off)
    var c_mat = s.bit_and(w0_e, s.uint(0xFFFF))
    var o_mat = s.bit_and(s.shr(w0_e, s.uint(16)), s.uint(0x7FFF))
    s.begin_if(s.ieq(c_mat, mat_id))
    s.store(cel_e_v, e_off)
    s.end_if()
    s.begin_if(s.logical_and(s.ult(o_mat, s.uint(UInt32(AAA_CEL_NO_OUTLINE_MAT))), s.ieq(o_mat, mat_id)))
    s.store(out_e_v, e_off)
    s.end_if()
    s.store(ci_v, s.iadd(ci, s.uint(1)))
    s.end_loop()
    s.end_if()

    # Case A: Hardware Outline Pixel -> write crisp unlit `outline_color` and return!
    var out_e = s.load_u(out_e_v)
    s.begin_if(s.ine(out_e, s.uint(0xFFFFFFFF)))
    var oc_r = s.load_float(b_globals, s.iadd(out_e, s.uint(17)))
    var oc_g = s.load_float(b_globals, s.iadd(out_e, s.uint(18)))
    var oc_b = s.load_float(b_globals, s.iadd(out_e, s.uint(19)))
    s.store_float(b_hdr, hdr_base, oc_r)
    s.store_float(b_hdr, s.iadd(hdr_base, s.uint(1)), oc_g)
    s.store_float(b_hdr, s.iadd(hdr_base, s.uint(2)), oc_b)
    s.end_if()
    s.return_if(s.ine(out_e, s.uint(0xFFFFFFFF)))

    var cel_e = s.load_u(cel_e_v)
    var vfx_e = emit_find_vfx_entry_spv(s, b_globals, mat_id)
    s.return_if(s.logical_and(s.ieq(cel_e, s.uint(0xFFFFFFFF)), s.ieq(vfx_e, s.uint(0xFFFFFFFF))))

    # Shared exact 3-Ray differential geometry decode for Cel & Stylized VFX pixels
    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var tan_x = s.load_float(b_globals, s.uint(39))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var tan_y = s.load_float(b_globals, s.uint(43))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))
    var jit_x = s.load_float(b_globals, s.uint(101))
    var jit_y = s.load_float(b_globals, s.uint(102))

    var fx0 = s.fsub(s.fadd(s.float_from_uint(px), s.float(0.5)), s.fmul(jit_x, s.fmul(fw, s.float(0.5))))
    var fy0 = s.fadd(s.fadd(s.float_from_uint(py), s.float(0.5)), s.fmul(jit_y, s.fmul(fh, s.float(0.5))))
    var ndc_x = s.fsub(s.fmul(s.fdiv(fx0, fw), s.float(2.0)), s.float(1.0))
    var ndc_y = s.fsub(s.float(1.0), s.fmul(s.fdiv(fy0, fh), s.float(2.0)))
    var ndc_x1 = s.fsub(s.fmul(s.fdiv(s.fadd(fx0, s.float(1.0)), fw), s.float(2.0)), s.float(1.0))
    var ndc_y1 = s.fsub(s.float(1.0), s.fmul(s.fdiv(s.fadd(fy0, s.float(1.0)), fh), s.float(2.0)))

    var ray_c = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x, tan_x), s.fma(up_x, s.fmul(ndc_y, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x, tan_x), s.fma(up_y, s.fmul(ndc_y, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x, tan_x), s.fma(up_z, s.fmul(ndc_y, tan_y), fwd_z)),
    )
    var ray_dx = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x1, tan_x), s.fma(up_x, s.fmul(ndc_y, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x1, tan_x), s.fma(up_y, s.fmul(ndc_y, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x1, tan_x), s.fma(up_z, s.fmul(ndc_y, tan_y), fwd_z)),
    )
    var ray_dy = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x, tan_x), s.fma(up_x, s.fmul(ndc_y1, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x, tan_x), s.fma(up_y, s.fmul(ndc_y1, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x, tan_x), s.fma(up_z, s.fmul(ndc_y1, tan_y), fwd_z)),
    )

    var inst_id = s.bit_or(
        s.shr(vis0, s.uint(22)),
        s.shl(s.bit_and(s.shr(vis1, s.uint(8)), s.uint(0xFF)), s.uint(10)),
    )
    var ml_id = s.isub(s.bit_and(vis0, s.uint(0x3FFFFF)), s.uint(1))
    var local_prim = s.bit_and(vis1, s.uint(0xFF))
    var first_tri = s.load_word(b_meshlets, s.iadd(s.imul(ml_id, s.uint(12)), s.uint(8)))
    var idx_off = s.imul(s.iadd(first_tri, local_prim), s.uint(3))
    var v0_off = s.imul(s.load_word(b_indices, idx_off), s.uint(12))
    var v1_off = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(1))), s.uint(12))
    var v2_off = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(2))), s.uint(12))
    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))

    var p0 = emit_mat4_mul_point(s, b_instances, inst_base, s.load_float(b_verts, v0_off), s.load_float(b_verts, s.iadd(v0_off, s.uint(1))), s.load_float(b_verts, s.iadd(v0_off, s.uint(2))))
    var p1 = emit_mat4_mul_point(s, b_instances, inst_base, s.load_float(b_verts, v1_off), s.load_float(b_verts, s.iadd(v1_off, s.uint(1))), s.load_float(b_verts, s.iadd(v1_off, s.uint(2))))
    var p2 = emit_mat4_mul_point(s, b_instances, inst_base, s.load_float(b_verts, v2_off), s.load_float(b_verts, s.iadd(v2_off, s.uint(1))), s.load_float(b_verts, s.iadd(v2_off, s.uint(2))))

    var bary = emit_ray_barycentrics(
        s,
        cam_x, cam_y, cam_z,
        ray_c[0], ray_c[1], ray_c[2],
        ray_dx[0], ray_dx[1], ray_dx[2],
        ray_dy[0], ray_dy[1], ray_dy[2],
        p0[0], p0[1], p0[2],
        p1[0], p1[1], p1[2],
        p2[0], p2[1], p2[2],
    )
    var b1 = bary[0]
    var b2 = bary[1]
    var b0 = s.fsub(s.fsub(s.float(1.0), b1), b2)
    var b0_dx = s.fsub(s.fsub(s.float(1.0), bary[2]), bary[3])
    var b0_dy = s.fsub(s.fsub(s.float(1.0), bary[4]), bary[5])

    var u0 = s.load_float(b_verts, s.iadd(v0_off, s.uint(9))); var v0_t = s.load_float(b_verts, s.iadd(v0_off, s.uint(10)))
    var u1 = s.load_float(b_verts, s.iadd(v1_off, s.uint(9))); var v1_t = s.load_float(b_verts, s.iadd(v1_off, s.uint(10)))
    var u2 = s.load_float(b_verts, s.iadd(v2_off, s.uint(9))); var v2_t = s.load_float(b_verts, s.iadd(v2_off, s.uint(10)))
    var uv_u = s.fma(u0, b0, s.fma(u1, b1, s.fmul(u2, b2)))
    var uv_v = s.fma(v0_t, b0, s.fma(v1_t, b1, s.fmul(v2_t, b2)))
    var mip_bias_g = s.load_float(b_globals, s.uint(127))
    var grad_sc = s.exp2(mip_bias_g)
    var ddx_u = s.fmul(s.fsub(s.fma(u0, b0_dx, s.fma(u1, bary[2], s.fmul(u2, bary[3]))), uv_u), grad_sc)
    var ddx_v = s.fmul(s.fsub(s.fma(v0_t, b0_dx, s.fma(v1_t, bary[2], s.fmul(v2_t, bary[3]))), uv_v), grad_sc)
    var ddy_u = s.fmul(s.fsub(s.fma(u0, b0_dy, s.fma(u1, bary[4], s.fmul(u2, bary[5]))), uv_u), grad_sc)
    var ddy_v = s.fmul(s.fsub(s.fma(v0_t, b0_dy, s.fma(v1_t, bary[4], s.fmul(v2_t, bary[5]))), uv_v), grad_sc)
    var uv_vec = s.vec2(uv_u, uv_v)
    var ddx_vec = s.vec2(ddx_u, ddx_v)
    var ddy_vec = s.vec2(ddy_u, ddy_v)

    var m_base = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var tex_pack = s.load_word(b_materials, s.iadd(m_base, s.uint(23)))
    var t_alb = s.bit_and(tex_pack, s.uint(0xFF))
    var t_nrm = s.bit_and(s.shr(tex_pack, s.uint(8)), s.uint(0xFF))
    var em_r = s.load_float(b_materials, s.iadd(m_base, s.uint(8)))
    var em_g = s.load_float(b_materials, s.iadd(m_base, s.uint(9)))
    var em_b = s.load_float(b_materials, s.iadd(m_base, s.uint(10)))

    # Case B: Stylized Eroding UV-Scroll 3D Mesh VFX -> write crisp ungrained 3-band HDR!
    s.begin_if(s.ine(vfx_e, s.uint(0xFFFFFFFF)))
    var base_mask_v = s.fvar(s.float(1.0))
    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var sa_vfx = s.sample_bindless_grad(bindless_arr, t_alb, uv_vec, ddx_vec, ddy_vec)
    s.store(base_mask_v, s.extract_f(sa_vfx, 3))
    s.end_if()
    var vfx_res = emit_eval_vfx_excess_and_color_spv(
        s, b_globals, bindless_arr, vfx_e, uv_u, uv_v, ddx_vec, ddy_vec, s.load_f(base_mask_v)
    )
    s.store_float(b_hdr, hdr_base, s.fadd(vfx_res[1], em_r))
    s.store_float(b_hdr, s.iadd(hdr_base, s.uint(1)), s.fadd(vfx_res[2], em_g))
    s.store_float(b_hdr, s.iadd(hdr_base, s.uint(2)), s.fadd(vfx_res[3], em_b))
    s.end_if()
    s.return_if(s.ine(vfx_e, s.uint(0xFFFFFFFF)))

    # Case C: Full Stylized NPR `ILM + SSS + Inner-Ink + Hair-Halo + Rim` Cel Shading
    var wx = s.fma(p0[0], b0, s.fma(p1[0], b1, s.fmul(p2[0], b2)))
    var wy = s.fma(p0[1], b0, s.fma(p1[1], b1, s.fmul(p2[1], b2)))
    var wz = s.fma(p0[2], b0, s.fma(p1[2], b1, s.fmul(p2[2], b2)))

    var c0x = s.load_float(b_instances, inst_base)
    var c0y = s.load_float(b_instances, s.iadd(inst_base, s.uint(1)))
    var c0z = s.load_float(b_instances, s.iadd(inst_base, s.uint(2)))
    var c1x = s.load_float(b_instances, s.iadd(inst_base, s.uint(4)))
    var c1y = s.load_float(b_instances, s.iadd(inst_base, s.uint(5)))
    var c1z = s.load_float(b_instances, s.iadd(inst_base, s.uint(6)))
    var c2x = s.load_float(b_instances, s.iadd(inst_base, s.uint(8)))
    var c2y = s.load_float(b_instances, s.iadd(inst_base, s.uint(9)))
    var c2z = s.load_float(b_instances, s.iadd(inst_base, s.uint(10)))
    var cof0 = emit_vec3_cross(s, c1x, c1y, c1z, c2x, c2y, c2z)
    var cof1 = emit_vec3_cross(s, c2x, c2y, c2z, c0x, c0y, c0z)
    var cof2 = emit_vec3_cross(s, c0x, c0y, c0z, c1x, c1y, c1z)
    var det_m = emit_vec3_dot(s, c0x, c0y, c0z, cof0[0], cof0[1], cof0[2])
    var det_s = s.select_f(s.flt(det_m, s.float(0.0)), s.float(-1.0), s.float(1.0))

    var lnx = s.fma(s.load_float(b_verts, s.iadd(v0_off, s.uint(3))), b0, s.fma(s.load_float(b_verts, s.iadd(v1_off, s.uint(3))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2_off, s.uint(3))), b2)))
    var lny = s.fma(s.load_float(b_verts, s.iadd(v0_off, s.uint(4))), b0, s.fma(s.load_float(b_verts, s.iadd(v1_off, s.uint(4))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2_off, s.uint(4))), b2)))
    var lnz = s.fma(s.load_float(b_verts, s.iadd(v0_off, s.uint(5))), b0, s.fma(s.load_float(b_verts, s.iadd(v1_off, s.uint(5))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2_off, s.uint(5))), b2)))
    var wn_geom = emit_vec3_normalize(
        s,
        s.fmul(s.fma(cof0[0], lnx, s.fma(cof1[0], lny, s.fmul(cof2[0], lnz))), det_s),
        s.fmul(s.fma(cof0[1], lnx, s.fma(cof1[1], lny, s.fmul(cof2[1], lnz))), det_s),
        s.fmul(s.fma(cof0[2], lnx, s.fma(cof1[2], lny, s.fmul(cof2[2], lnz))), det_s),
    )

    var ltx = s.fma(s.load_float(b_verts, s.iadd(v0_off, s.uint(6))), b0, s.fma(s.load_float(b_verts, s.iadd(v1_off, s.uint(6))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2_off, s.uint(6))), b2)))
    var lty = s.fma(s.load_float(b_verts, s.iadd(v0_off, s.uint(7))), b0, s.fma(s.load_float(b_verts, s.iadd(v1_off, s.uint(7))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2_off, s.uint(7))), b2)))
    var ltz = s.fma(s.load_float(b_verts, s.iadd(v0_off, s.uint(8))), b0, s.fma(s.load_float(b_verts, s.iadd(v1_off, s.uint(8))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2_off, s.uint(8))), b2)))
    var wtr = emit_mat4_mul_dir(s, b_instances, inst_base, ltx, lty, ltz)
    var td_n = emit_vec3_dot(s, wtr[0], wtr[1], wtr[2], wn_geom[0], wn_geom[1], wn_geom[2])
    var gstx = s.fsub(wtr[0], s.fmul(td_n, wn_geom[0]))
    var gsty = s.fsub(wtr[1], s.fmul(td_n, wn_geom[1]))
    var gstz = s.fsub(wtr[2], s.fmul(td_n, wn_geom[2]))
    var gsl2 = emit_vec3_dot(s, gstx, gsty, gstz, gstx, gsty, gstz)
    var fb_up = s.flt(s.fabs(wn_geom[1]), s.float(0.92))
    var fb_t = emit_vec3_cross(s, s.select_f(fb_up, s.float(0.0), s.float(1.0)), s.select_f(fb_up, s.float(1.0), s.float(0.0)), s.float(0.0), wn_geom[0], wn_geom[1], wn_geom[2])
    var gs_ok = s.fgt(gsl2, s.float(1e-8))
    var wt_u = emit_vec3_normalize(s, s.select_f(gs_ok, gstx, fb_t[0]), s.select_f(gs_ok, gsty, fb_t[1]), s.select_f(gs_ok, gstz, fb_t[2]))

    var meta0 = s.load_word(b_verts, s.iadd(v0_off, s.uint(11)))
    var meta1 = s.load_word(b_verts, s.iadd(v1_off, s.uint(11)))
    var meta2 = s.load_word(b_verts, s.iadd(v2_off, s.uint(11)))
    var t_sign = s.fmul(s.select_f(s.ieq(s.bit_and(meta0, s.uint(1)), s.uint(0)), s.float(1.0), s.float(-1.0)), det_s)
    var wb_raw = emit_vec3_cross(s, wn_geom[0], wn_geom[1], wn_geom[2], wt_u[0], wt_u[1], wt_u[2])
    var wb_u = (s.fmul(wb_raw[0], t_sign), s.fmul(wb_raw[1], t_sign), s.fmul(wb_raw[2], t_sign))

    # Optional BC5 tangent normal map perturbation
    var nx_v = s.fvar(wn_geom[0])
    var ny_v = s.fvar(wn_geom[1])
    var nz_v = s.fvar(wn_geom[2])
    s.begin_if(s.ult(t_nrm, s.uint(0xFF)))
    var sn = s.sample_bindless_grad(bindless_arr, t_nrm, uv_vec, ddx_vec, ddy_vec)
    var mx = s.fma(s.extract_f(sn, 0), s.float(2.0), s.float(-1.0))
    var my = s.fma(s.extract_f(sn, 1), s.float(2.0), s.float(-1.0))
    var mz = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(mx, mx, s.fmul(my, my)))))
    var mn = emit_vec3_normalize(
        s,
        s.fma(wt_u[0], mx, s.fma(wb_u[0], my, s.fmul(wn_geom[0], mz))),
        s.fma(wt_u[1], mx, s.fma(wb_u[1], my, s.fmul(wn_geom[1], mz))),
        s.fma(wt_u[2], mx, s.fma(wb_u[2], my, s.fmul(wn_geom[2], mz))),
    )
    s.store(nx_v, mn[0]); s.store(ny_v, mn[1]); s.store(nz_v, mn[2])
    s.end_if()
    var nx = s.load_f(nx_v)
    var ny = s.load_f(ny_v)
    var nz = s.load_f(nz_v)

    var vx = s.fneg(ray_c[0])
    var vy = s.fneg(ray_c[1])
    var vz = s.fneg(ray_c[2])

    # Load `AaaCelParams` words
    var w0_c = s.load_word(b_globals, cel_e)
    var w1_c = s.load_word(b_globals, s.iadd(cel_e, s.uint(1)))
    var use_vc = s.ine(s.shr(w0_c, s.uint(31)), s.uint(0))
    var sss_slot = s.bit_and(w1_c, s.uint(0xFF))
    var ilm_slot = s.bit_and(s.shr(w1_c, s.uint(8)), s.uint(0xFF))
    var cel_flags = s.bit_and(s.shr(w1_c, s.uint(16)), s.uint(0xFF))
    var recv_sh = s.ine(s.bit_and(cel_flags, s.uint(AAA_CEL_FLAG_RECEIVE_SHADOWS)), s.uint(0))
    var has_hair = s.ine(s.bit_and(cel_flags, s.uint(AAA_CEL_FLAG_HAIR_ANISO)), s.uint(0))
    var use_cust_l = s.ine(s.bit_and(cel_flags, s.uint(AAA_CEL_FLAG_CUSTOM_LIGHT_DIR)), s.uint(0))

    var sss_tr = s.load_float(b_globals, s.iadd(cel_e, s.uint(2)))
    var sss_tg = s.load_float(b_globals, s.iadd(cel_e, s.uint(3)))
    var sss_tb = s.load_float(b_globals, s.iadd(cel_e, s.uint(4)))
    var sh_thresh = s.load_float(b_globals, s.iadd(cel_e, s.uint(5)))
    var sh_feath = s.load_float(b_globals, s.iadd(cel_e, s.uint(6)))
    var rim_r = s.load_float(b_globals, s.iadd(cel_e, s.uint(7)))
    var rim_g = s.load_float(b_globals, s.iadd(cel_e, s.uint(8)))
    var rim_b = s.load_float(b_globals, s.iadd(cel_e, s.uint(9)))
    var rim_pow = s.load_float(b_globals, s.iadd(cel_e, s.uint(10)))
    var rim_th = s.load_float(b_globals, s.iadd(cel_e, s.uint(11)))
    var spec_int = s.load_float(b_globals, s.iadd(cel_e, s.uint(12)))
    var spec_th = s.load_float(b_globals, s.iadd(cel_e, s.uint(13)))
    var hair_str = s.load_float(b_globals, s.iadd(cel_e, s.uint(14)))
    var hair_shf = s.load_float(b_globals, s.iadd(cel_e, s.uint(15)))
    var out_cr = s.load_float(b_globals, s.iadd(cel_e, s.uint(17)))
    var out_cg = s.load_float(b_globals, s.iadd(cel_e, s.uint(18)))
    var out_cb = s.load_float(b_globals, s.iadd(cel_e, s.uint(19)))
    var ink_str = s.load_float(b_globals, s.iadd(cel_e, s.uint(22)))
    var amb_flr = s.load_float(b_globals, s.iadd(cel_e, s.uint(23)))
    var cust_lx = s.load_float(b_globals, s.iadd(cel_e, s.uint(24)))
    var cust_ly = s.load_float(b_globals, s.iadd(cel_e, s.uint(25)))
    var cust_lz = s.load_float(b_globals, s.iadd(cel_e, s.uint(26)))
    var cust_lint = s.load_float(b_globals, s.iadd(cel_e, s.uint(27)))

    # Base Albedo + Vertex Color (either RGB vertex tint or ArcSys `R` shadow bias channel)
    var alb_r = s.fvar(s.load_float(b_materials, m_base))
    var alb_g = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(1))))
    var alb_b = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(2))))
    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var s_alb = s.sample_bindless_grad(bindless_arr, t_alb, uv_vec, ddx_vec, ddy_vec)
    s.store(alb_r, s.fmul(s.load_f(alb_r), s.extract_f(s_alb, 0)))
    s.store(alb_g, s.fmul(s.load_f(alb_g), s.extract_f(s_alb, 1)))
    s.store(alb_b, s.fmul(s.load_f(alb_b), s.extract_f(s_alb, 2)))
    s.end_if()

    var inv255 = s.float(1.0 / 255.0)
    var r0_f = s.fmul(s.float_from_uint(s.bit_and(s.shr(meta0, s.uint(8)), s.uint(0xFF))), inv255)
    var r1_f = s.fmul(s.float_from_uint(s.bit_and(s.shr(meta1, s.uint(8)), s.uint(0xFF))), inv255)
    var r2_f = s.fmul(s.float_from_uint(s.bit_and(s.shr(meta2, s.uint(8)), s.uint(0xFF))), inv255)
    var vcr_lin = s.fma(r0_f, b0, s.fma(r1_f, b1, s.fmul(r2_f, b2)))

    s.begin_if(s.logical_not(use_vc))
    var g0_f = s.fmul(s.float_from_uint(s.bit_and(s.shr(meta0, s.uint(16)), s.uint(0xFF))), inv255)
    var g1_f = s.fmul(s.float_from_uint(s.bit_and(s.shr(meta1, s.uint(16)), s.uint(0xFF))), inv255)
    var g2_f = s.fmul(s.float_from_uint(s.bit_and(s.shr(meta2, s.uint(16)), s.uint(0xFF))), inv255)
    var b0_f = s.fmul(s.float_from_uint(s.shr(meta0, s.uint(24))), inv255)
    var b1_f = s.fmul(s.float_from_uint(s.shr(meta1, s.uint(24))), inv255)
    var b2_f = s.fmul(s.float_from_uint(s.shr(meta2, s.uint(24))), inv255)
    var vcg_lin = s.fma(g0_f, b0, s.fma(g1_f, b1, s.fmul(g2_f, b2)))
    var vcb_lin = s.fma(b0_f, b0, s.fma(b1_f, b1, s.fmul(b2_f, b2)))
    # Approximate sRGB^2.2 vertex color decode when `mesh.colors` was authored as RGB tint
    s.store(alb_r, s.fmul(s.load_f(alb_r), s.fpow(s.nmax(s.float(1e-5), vcr_lin), s.float(2.2))))
    s.store(alb_g, s.fmul(s.load_f(alb_g), s.fpow(s.nmax(s.float(1e-5), vcg_lin), s.float(2.2))))
    s.store(alb_b, s.fmul(s.load_f(alb_b), s.fpow(s.nmax(s.float(1e-5), vcb_lin), s.float(2.2))))
    s.end_if()

    var ar = s.load_f(alb_r)
    var ag = s.load_f(alb_g)
    var ab = s.load_f(alb_b)

    # Optional authored `SSS` shadow texture (`BC7_SRGB`)
    var sh_base_r = s.fvar(ar)
    var sh_base_g = s.fvar(ag)
    var sh_base_b = s.fvar(ab)
    s.begin_if(s.ult(sss_slot, s.uint(0xFF)))
    var s_sss = s.sample_bindless_grad(bindless_arr, sss_slot, uv_vec, ddx_vec, ddy_vec)
    s.store(sh_base_r, s.extract_f(s_sss, 0))
    s.store(sh_base_g, s.extract_f(s_sss, 1))
    s.store(sh_base_b, s.extract_f(s_sss, 2))
    s.end_if()

    # Optional `ILM` map (`R`=spec mask, `G`=shadow bias, `B`=spec sharpness/metallic, `A`=inner ink)
    var ilm_r_v = s.fvar(s.float(1.0))
    var ilm_g_v = s.fvar(s.float(0.5))
    var ilm_b_v = s.fvar(s.float(0.5))
    var ilm_a_v = s.fvar(s.float(1.0))
    s.begin_if(s.ult(ilm_slot, s.uint(0xFF)))
    var s_ilm = s.sample_bindless_grad(bindless_arr, ilm_slot, uv_vec, ddx_vec, ddy_vec)
    s.store(ilm_r_v, s.extract_f(s_ilm, 0))
    s.store(ilm_g_v, s.extract_f(s_ilm, 1))
    s.store(ilm_b_v, s.extract_f(s_ilm, 2))
    s.store(ilm_a_v, s.extract_f(s_ilm, 3))
    s.end_if()
    var ilm_r = s.load_f(ilm_r_v)
    var ilm_g = s.load_f(ilm_g_v)
    var ilm_b = s.load_f(ilm_b_v)
    var ilm_a = s.load_f(ilm_a_v)

    # Direct Sun Shadow (`RT` filtered sun shadow when active, else `16k VSM` fallback)
    var raw_sun_x = s.load_float(b_globals, s.uint(48))
    var raw_sun_y = s.load_float(b_globals, s.uint(49))
    var raw_sun_z = s.load_float(b_globals, s.uint(50))
    var sun_ang_r = s.load_float(b_globals, s.uint(51))
    var raw_srad_r = s.load_float(b_globals, s.uint(52))
    var raw_srad_g = s.load_float(b_globals, s.uint(53))
    var raw_srad_b = s.load_float(b_globals, s.uint(54))
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var r_flags = s.load_word(b_globals, s.uint(99))

    var sun_x = s.select_f(use_cust_l, cust_lx, raw_sun_x)
    var sun_y = s.select_f(use_cust_l, cust_ly, raw_sun_y)
    var sun_z = s.select_f(use_cust_l, cust_lz, raw_sun_z)
    var l_mul = s.select_f(use_cust_l, cust_lint, s.float(1.0))
    var sun_rad_r = s.fmul(raw_srad_r, l_mul)
    var sun_rad_g = s.fmul(raw_srad_g, l_mul)
    var sun_rad_b = s.fmul(raw_srad_b, l_mul)

    var sun_vis_v = s.fvar(s.float(1.0))
    s.begin_if(s.logical_and(recv_sh, s.logical_not(use_cust_l)))
    var has_rt_sun = s.logical_and(s.ine(s.bit_and(r_flags, s.uint(2)), s.uint(0)), s.ine(rt_sun_off, s.uint(0)))
    s.begin_if(has_rt_sun)
    s.store(sun_vis_v, s.load_float(b_hist, s.iadd(rt_sun_off, pix_idx)))
    s.begin_else()
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(1)), s.uint(0)))
    s.store(sun_vis_v, emit_vsm_pcss_shadow(s, b_globals, b_vsm, vsm_samp, wx, wy, wz, wn_geom[0], wn_geom[1], wn_geom[2], sun_ang_r))
    s.end_if()
    s.end_if()
    s.end_if()
    var sun_vis = s.load_f(sun_vis_v)

    # Half-Lambert + Shadow + `ILM.g` + `vc.r` 2-Tone Stepped Ramp
    var hl = s.fma(emit_vec3_dot(s, nx, ny, nz, sun_x, sun_y, sun_z), s.float(0.5), s.float(0.5))
    var sun_lum = s.fma(sun_rad_r, s.float(0.2126), s.fma(sun_rad_g, s.float(0.7152), s.fmul(sun_rad_b, s.float(0.0722))))
    var base_sig = s.select_f(s.fgt(sun_lum, s.float(1e-4)), s.nmin(hl, sun_vis), s.float(0.0))
    var ilm_bias = s.fmul(s.fsub(ilm_g, s.float(0.5)), s.float(2.0))
    var vc_bias = s.select_f(use_vc, s.fmul(s.fsub(vcr_lin, s.float(0.5)), s.float(1.5)), s.float(0.0))
    var cel_sig = s.fadd(base_sig, s.fadd(ilm_bias, vc_bias))
    var lit_band = _emit_smoothstep_spv(s, s.fsub(sh_thresh, sh_feath), s.fadd(sh_thresh, sh_feath), cel_sig)

    # Noise-free SH0 + Sun Cel Rig
    var sh0_r = s.fmul(s.fmul(s.load_float(b_globals, s.uint(72)), s.float(0.886227)), ibl_int)
    var sh0_g = s.fmul(s.fmul(s.load_float(b_globals, s.uint(73)), s.float(0.886227)), ibl_int)
    var sh0_b = s.fmul(s.fmul(s.load_float(b_globals, s.uint(74)), s.float(0.886227)), ibl_int)
    var amb_r = s.nmax(amb_flr, s.fmul(sh0_r, s.float(0.35)))
    var amb_g = s.nmax(amb_flr, s.fmul(sh0_g, s.float(0.35)))
    var amb_b = s.nmax(amb_flr, s.fmul(sh0_b, s.float(0.35)))

    var sk_r = s.fmul(sun_rad_r, s.float(0.34))
    var sk_g = s.fmul(sun_rad_g, s.float(0.34))
    var sk_b = s.fmul(sun_rad_b, s.float(0.34))

    var lit_rig_r = s.fma(amb_r, s.float(0.45), sk_r)
    var lit_rig_g = s.fma(amb_g, s.float(0.45), sk_g)
    var lit_rig_b = s.fma(amb_b, s.float(0.45), sk_b)
    var shd_rig_r = s.fma(sk_r, s.float(0.22), amb_r)
    var shd_rig_g = s.fma(sk_g, s.float(0.22), amb_g)
    var shd_rig_b = s.fma(sk_b, s.float(0.22), amb_b)

    var sha_r = s.fmul(s.load_f(sh_base_r), sss_tr)
    var sha_g = s.fmul(s.load_f(sh_base_g), sss_tg)
    var sha_b = s.fmul(s.load_f(sh_base_b), sss_tb)

    var lr_c = s.fmul(ar, lit_rig_r); var lg_c = s.fmul(ag, lit_rig_g); var lb_c = s.fmul(ab, lit_rig_b)
    var sr_c = s.fmul(sha_r, shd_rig_r); var sg_c = s.fmul(sha_g, shd_rig_g); var sb_c = s.fmul(sha_b, shd_rig_b)

    var out_r = s.fvar(s.fma(s.fsub(lr_c, sr_c), lit_band, sr_c))
    var out_g = s.fvar(s.fma(s.fsub(lg_c, sg_c), lit_band, sg_c))
    var out_b = s.fvar(s.fma(s.fsub(lb_c, sb_c), lit_band, sb_c))

    # Stepped Clustered Punctual/Area Lights (fireballs, sparks, stage neons)
    var total_cl = s.load_word(b_clusters, s.uint(0))
    s.begin_if(s.ine(total_cl, s.uint(0)))
    var z_lin = s.nmax(s.float(0.05), s.load_float(b_hdr, s.iadd(hdr_base, s.uint(3))))
    var cl_near = s.nmax(s.float(1e-4), s.load_float(b_clusters, s.uint(1)))
    var cl_log = s.nmax(s.float(1e-4), s.load_float(b_clusters, s.uint(3)))
    var cx_u = s.umin(s.uint(UInt32(AAA_CLUSTER_X - 1)), s.udiv(s.imul(px, s.uint(UInt32(AAA_CLUSTER_X))), w_u))
    var cy_u = s.umin(s.uint(UInt32(AAA_CLUSTER_Y - 1)), s.udiv(s.imul(py, s.uint(UInt32(AAA_CLUSTER_Y))), h_u))
    var zn_c = s.nclamp(s.fdiv(s.log2(s.nmax(s.float(1.0), s.fdiv(z_lin, cl_near))), cl_log), s.float(0.0), s.float(0.9999))
    var cz_u = s.umin(s.uint(UInt32(AAA_CLUSTER_Z - 1)), s.saturate_uint(s.fmul(zn_c, s.float(Float32(AAA_CLUSTER_Z)))))
    var cl_id = s.iadd(s.imul(cz_u, s.uint(UInt32(AAA_CLUSTER_X * AAA_CLUSTER_Y))), s.iadd(s.imul(cy_u, s.uint(UInt32(AAA_CLUSTER_X))), cx_u))
    var cl_cnt = s.umin(
        s.load_word(b_clusters, s.iadd(s.uint(UInt32(AAA_CLUSTER_COUNTS_BASE)), cl_id)),
        s.uint(UInt32(AAA_MAX_LIGHTS_PER_CLUSTER)),
    )
    var cl_list = s.iadd(s.uint(UInt32(AAA_CLUSTER_INDICES_BASE)), s.imul(cl_id, s.uint(UInt32(AAA_MAX_LIGHTS_PER_CLUSTER))))
    var lk_v = s.uvar(s.uint(0))
    s.begin_loop()
    var lk = s.load_u(lk_v)
    s.break_if(s.uge(lk, cl_cnt))
    var li = s.load_word(b_clusters, s.iadd(cl_list, lk))
    var lo = s.iadd(s.uint(UInt32(AAA_CLUSTER_LIGHTS_BASE)), s.imul(li, s.uint(UInt32(AAA_CLUSTER_LIGHT_WORDS))))
    var lpx_c = s.load_float(b_clusters, lo)
    var lpy_c = s.load_float(b_clusters, s.iadd(lo, s.uint(1)))
    var lpz_c = s.load_float(b_clusters, s.iadd(lo, s.uint(2)))
    var lrad_c = s.nmax(s.float(0.05), s.load_float(b_clusters, s.iadd(lo, s.uint(3))))
    var lrr = s.load_float(b_clusters, s.iadd(lo, s.uint(4)))
    var lrg = s.load_float(b_clusters, s.iadd(lo, s.uint(5)))
    var lrb = s.load_float(b_clusters, s.iadd(lo, s.uint(6)))
    var rlx = s.fsub(lpx_c, wx); var rly = s.fsub(lpy_c, wy); var rlz = s.fsub(lpz_c, wz)
    var ld2 = s.nmax(s.float(0.04), emit_vec3_dot(s, rlx, rly, rlz, rlx, rly, rlz))
    var ld = s.fsqrt(ld2)
    s.begin_if(s.flt(ld, lrad_c))
    var inv_ld = s.fdiv(s.float(1.0), ld)
    var cl_nl = emit_vec3_dot(s, nx, ny, nz, s.fmul(rlx, inv_ld), s.fmul(rly, inv_ld), s.fmul(rlz, inv_ld))
    var q_r = s.fdiv(ld, lrad_c)
    var q4 = s.fmul(s.fmul(q_r, q_r), s.fmul(q_r, q_r))
    var win = s.nmax(s.float(0.0), s.fsub(s.float(1.0), q4))
    var att = s.fdiv(s.fmul(win, win), s.fadd(ld2, s.float(0.35)))
    var step_nl = _emit_smoothstep_spv(s, s.float(0.12), s.float(0.20), cl_nl)
    var w_cl = s.fmul(att, s.fmul(step_nl, s.float(0.32)))
    s.store(out_r, s.fma(s.fmul(ar, lrr), w_cl, s.load_f(out_r)))
    s.store(out_g, s.fma(s.fmul(ag, lrg), w_cl, s.load_f(out_g)))
    s.store(out_b, s.fma(s.fmul(ab, lrb), w_cl, s.load_f(out_b)))
    s.end_if()
    s.store(lk_v, s.iadd(lk, s.uint(1)))
    s.end_loop()
    s.end_if()

    # Stylized Stepped Specular (`ILM.r` mask, `ILM.b` sharpness & metallic tint)
    var h_vec = emit_vec3_normalize(s, s.fadd(sun_x, vx), s.fadd(sun_y, vy), s.fadd(sun_z, vz))
    var ndoth = s.nmax(s.float(1e-6), emit_vec3_dot(s, nx, ny, nz, h_vec[0], h_vec[1], h_vec[2]))
    var sp_exp = s.fma(ilm_b, s.float(96.0), s.float(16.0))
    var sp_pow = s.fpow(ndoth, sp_exp)
    var sp_step = _emit_smoothstep_spv(s, s.fsub(spec_th, s.float(0.02)), s.fadd(spec_th, s.float(0.02)), sp_pow)
    var sp_amt = s.fmul(sp_step, s.fmul(spec_int, s.fmul(ilm_r, lit_band)))
    var st_r = s.fma(s.fsub(ar, s.float(1.0)), ilm_b, s.float(1.0))
    var st_g = s.fma(s.fsub(ag, s.float(1.0)), ilm_b, s.float(1.0))
    var st_b = s.fma(s.fsub(ab, s.float(1.0)), ilm_b, s.float(1.0))
    s.store(out_r, s.fma(s.fmul(st_r, sk_r), sp_amt, s.load_f(out_r)))
    s.store(out_g, s.fma(s.fmul(st_g, sk_g), sp_amt, s.load_f(out_g)))
    s.store(out_b, s.fma(s.fmul(st_b, sk_b), sp_amt, s.load_f(out_b)))

    # Kajiya-Kay Anisotropic Hair Halo Band
    s.begin_if(has_hair)
    var t_sh = emit_vec3_normalize(
        s,
        s.fma(nx, hair_shf, wb_u[0]),
        s.fma(ny, hair_shf, wb_u[1]),
        s.fma(nz, hair_shf, wb_u[2]),
    )
    var th_d = emit_vec3_dot(s, t_sh[0], t_sh[1], t_sh[2], h_vec[0], h_vec[1], h_vec[2])
    var sin_th = s.fsqrt(s.nmax(s.float(1e-6), s.fsub(s.float(1.0), s.fmul(th_d, th_d))))
    var h_pow = s.fpow(sin_th, s.float(48.0))
    var h_step = _emit_smoothstep_spv(s, s.float(0.80), s.float(0.86), h_pow)
    var h_amt = s.fmul(h_step, s.fmul(hair_str, s.fmul(ilm_r, lit_band)))
    s.store(out_r, s.fma(sk_r, h_amt, s.load_f(out_r)))
    s.store(out_g, s.fma(sk_g, h_amt, s.load_f(out_g)))
    s.store(out_b, s.fma(sk_b, h_amt, s.load_f(out_b)))
    s.end_if()

    # Light-Gated Fresnel Rim Light
    var ndotv = s.nclamp(emit_vec3_dot(s, nx, ny, nz, vx, vy, vz), s.float(0.0), s.float(1.0))
    var fres = s.fpow(s.nmax(s.float(1e-6), s.fsub(s.float(1.0), ndotv)), rim_pow)
    var rim_st = _emit_smoothstep_spv(s, s.fsub(rim_th, s.float(0.04)), s.fadd(rim_th, s.float(0.04)), fres)
    var rim_w = s.fmul(rim_st, s.fma(lit_band, s.float(0.80), s.float(0.20)))
    s.store(out_r, s.fma(rim_r, rim_w, s.load_f(out_r)))
    s.store(out_g, s.fma(rim_g, rim_w, s.load_f(out_g)))
    s.store(out_b, s.fma(rim_b, rim_w, s.load_f(out_b)))

    # Inner Ink Lines (`ILM.a`) + Material Emissive
    var ink_sm = _emit_smoothstep_spv(s, s.float(0.35), s.float(0.65), ilm_a)
    var line_keep = s.fsub(s.float(1.0), s.fmul(ink_str, s.fsub(s.float(1.0), ink_sm)))
    var fin_r = s.fadd(s.fma(s.fsub(s.load_f(out_r), out_cr), line_keep, out_cr), em_r)
    var fin_g = s.fadd(s.fma(s.fsub(s.load_f(out_g), out_cg), line_keep, out_cg), em_g)
    var fin_b = s.fadd(s.fma(s.fsub(s.load_f(out_b), out_cb), line_keep, out_cb), em_b)

    s.store_float(b_hdr, hdr_base, fin_r)
    s.store_float(b_hdr, s.iadd(hdr_base, s.uint(1)), fin_g)
    s.store_float(b_hdr, s.iadd(hdr_base, s.uint(2)), fin_b)
    return s.finish()
