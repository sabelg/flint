"""Flint v5 Material Graph, 128-Byte Bindless `MaterialAsset`, OBB Decals, Terrain Clipmaps & VAT Crowds (`material_graph.mojo`).

Architecture:
  1. `MaterialAsset` (`128` bytes = `32 x UInt32` cache-line-aligned GPU layout):
     - Full **16-bit bindless texture indices** (`0 .. 65,534`, `0xFFFF = BINDLESS_SLOT_NONE`) across 8 slots
       (albedo, normal, ORM, emissive, detail normal, subsurface, VAT position, VAT normal).
     - Explicit `MAT_DOMAIN_*` shading domains (`StandardPbr`, `SubsurfaceSkin`, `TwoSidedFoliage`,
       `Clearcoat`, `SingleLayerWater`, `DeferredDecal`, `UnlitUi`) and explicit `MAT_FLAG_*` bitflags.
  2. `SurfaceMaterialNode`: Composable `ShaderDsl` trait compiling procedural/textured surface graphs
     directly to SPIR-V 1.5 with `OpSpecConstant` permutation specialization.
  3. `GpuDecalVolume`: `96`-byte (`24 x UInt32`) OBB deferred decal volume with world-to-OBB projection
     and steep-angle normal fade.
  4. `TerrainClipmapDesc`: Concentric toroidal geometry clipmap terrain descriptor (`4`-layer height-modulated splat).
  5. `VatCrowdClipDesc` & `bake_vat_crowd_clip_rgba16f`: Vertex Animation Texture baker converting skeletal
     clips into `RGBA16F` position + normal sheets + GPU compute evaluator (`vat_crowd_eval_shader`) so
     `10,000+` crowd instances animate independently with exact 3D motion vectors!
"""
from std.collections import List
from std.math import abs, floor, max, min, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat, trs
from flint.assets.model import NativeChannel, NativeMesh, NativeModel
from flint.render.aaa_bsdf import (
    AAA_MATERIAL_WORDS,
    AAA_MAT_DET_NONE,
    AAA_MAT_INTERIOR_PARALLAX,
    AAA_MAT_SSS_BIT,
    AaaMaterial,
)
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.shader_dsl import (
    BindlessTex2D,
    BufferRef,
    GpuBool,
    GpuF32,
    GpuU32,
    GpuVec2,
    GpuVec3,
    ShaderDsl,
    abs_f,
    clamp_f,
    fma_f,
    max_f,
    mix_vec3,
    normalize,
    sin_f,
)
from flint.render.aaa_visbuffer import (
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_MATERIALS,
    BIND_AAA_POST_SCRATCH,
    BIND_AAA_SKINNING,
    BIND_AAA_VERTICES,
    emit_vec3_normalize,
)

# ============================================================================
# 1. `MaterialAsset` Constants, Domains & Flags (Unified 24-Word GPU SSBO + 32-Word Archive)
# ============================================================================

comptime MATERIAL_ASSET_WORDS: Int = AAA_MATERIAL_WORDS  # 24 words (96 bytes) — unified with `FlintAaaVulkanRenderer` Binding 4!
comptime MATERIAL_ASSET_BYTES: Int = AAA_MATERIAL_WORDS * 4
comptime MATERIAL_ARCHIVE_WORDS: Int = 32
comptime MATERIAL_ARCHIVE_BYTES: Int = 128
comptime BINDLESS_SLOT_NONE: UInt16 = 0xFFFF
comptime BINDLESS_SLOT_MAX: UInt16 = 65534  # 0xFFFE

# Material Shading Domains (`MAT_DOMAIN_*`)
comptime MAT_DOMAIN_STANDARD_PBR: UInt32 = 0
comptime MAT_DOMAIN_SUBSURFACE_SKIN: UInt32 = 1
comptime MAT_DOMAIN_TWO_SIDED_FOLIAGE: UInt32 = 2
comptime MAT_DOMAIN_CLEARCOAT: UInt32 = 3
comptime MAT_DOMAIN_SINGLE_LAYER_WATER: UInt32 = 4
comptime MAT_DOMAIN_DEFERRED_DECAL: UInt32 = 5
comptime MAT_DOMAIN_UNLIT_UI: UInt32 = 6

# Explicit Material Bitflags (`MAT_FLAG_*`)
comptime MAT_FLAG_NONE: UInt32 = 0
comptime MAT_FLAG_ALPHA_MASKED: UInt32 = 1 << 0
comptime MAT_FLAG_DOUBLE_SIDED: UInt32 = 1 << 1
comptime MAT_FLAG_SUBSURFACE: UInt32 = 1 << 2
comptime MAT_FLAG_CLEARCOAT: UInt32 = 1 << 3
comptime MAT_FLAG_IRIDESCENCE: UInt32 = 1 << 4
comptime MAT_FLAG_SHEEN: UInt32 = 1 << 5
comptime MAT_FLAG_INTERIOR_PARALLAX: UInt32 = 1 << 6
comptime MAT_FLAG_ROAD_WETNESS: UInt32 = 1 << 7
comptime MAT_FLAG_VAT_CROWD: UInt32 = 1 << 8
comptime MAT_FLAG_TERRAIN_SPLAT: UInt32 = 1 << 9
comptime MAT_FLAG_EMISSIVE_BLOOM: UInt32 = 1 << 10
comptime MAT_FLAG_RECEIVE_DECALS: UInt32 = 1 << 11


@always_inline
def f32_to_f16_bits(val: Float32) -> UInt16:
    """Deterministic IEEE-754 binary32 -> binary16 (`Float16`) bit encoder."""
    var bits = bitcast[DType.uint32](val)
    var sign = (bits >> 16) & UInt32(0x8000)
    var exp32 = Int((bits >> 23) & UInt32(0xFF)) - 127 + 15
    var mant = bits & UInt32(0x007FFFFF)
    if exp32 <= 0:
        if exp32 < -10:
            return UInt16(sign)
        var sub_m = (mant | UInt32(0x00800000)) >> UInt32(1 - exp32 + 13)
        return UInt16(sign | sub_m)
    if exp32 >= 31:
        return UInt16(sign | UInt32(0x7BFF))  # Clamp finite max +/-65504.0
    var rounded_m = (mant + UInt32(0x00001000)) >> 13
    if rounded_m >= UInt32(0x0400):
        rounded_m = 0
        exp32 += 1
        if exp32 >= 31:
            return UInt16(sign | UInt32(0x7BFF))
    return UInt16(sign | (UInt32(exp32) << 10) | (rounded_m & UInt32(0x03FF)))


@always_inline
def f16_bits_to_f32(h: UInt16) -> Float32:
    """Deterministic IEEE-754 binary16 (`Float16`) -> binary32 decoder."""
    var hu = UInt32(h)
    var sign = (hu & UInt32(0x8000)) << 16
    var exp16 = (hu >> 10) & UInt32(0x1F)
    var mant16 = hu & UInt32(0x03FF)
    if exp16 == 0:
        if mant16 == 0:
            return bitcast[DType.float32](sign)
        var m = mant16
        var e = Int32(-14)
        while (m & UInt32(0x0400)) == 0:
            m <<= 1
            e -= 1
        m &= UInt32(0x03FF)
        var exp32 = UInt32(Int(e) + 127)
        return bitcast[DType.float32](sign | (exp32 << 23) | (m << 13))
    if exp16 == 31:
        return bitcast[DType.float32](sign | UInt32(0x7F800000) | (mant16 << 13))
    var exp32 = (exp16 + UInt32(112)) << 23
    return bitcast[DType.float32](sign | exp32 | (mant16 << 13))


@always_inline
def _pack_u16_pair(lo: UInt16, hi: UInt16) -> UInt32:
    return UInt32(lo) | (UInt32(hi) << 16)


@always_inline
def _unpack_u16_pair(word: UInt32) -> Tuple[UInt16, UInt16]:
    return (UInt16(word & UInt32(0xFFFF)), UInt16((word >> 16) & UInt32(0xFFFF)))


struct MaterialAsset(Copyable, Movable):
    """128-byte (`32 x UInt32`) GPU material descriptor supporting 16-bit bindless slots (`0..65,534`)."""
    var base_color: Vec3
    var opacity: Float32
    var roughness: Float32
    var metallic: Float32
    var specular_factor: Float32
    var alpha_cutoff: Float32
    var emissive: Vec3
    var clearcoat: Float32
    var clearcoat_roughness: Float32
    var iridescence: Float32
    var iridescence_thickness_nm: Float32
    var iridescence_ior: Float32
    var sheen_color: Vec3
    var sheen_roughness: Float32
    var subsurface_color: Vec3
    var subsurface_amount: Float32
    var detail_uv_scale: Float32
    var detail_normal_strength: Float32
    var domain: UInt32
    var flags: UInt32
    var albedo_slot: UInt16
    var normal_slot: UInt16
    var orm_slot: UInt16
    var emissive_slot: UInt16
    var detail_normal_slot: UInt16
    var subsurface_slot: UInt16
    var vat_pos_slot: UInt16
    var vat_nrm_slot: UInt16
    var uv_scale_x: Float32
    var uv_scale_y: Float32

    def __init__(
        out self,
        base_color: Vec3 = Vec3(0.8, 0.8, 0.8),
        roughness: Float32 = 0.35,
        metallic: Float32 = 0.0,
        domain: UInt32 = MAT_DOMAIN_STANDARD_PBR,
        flags: UInt32 = MAT_FLAG_RECEIVE_DECALS,
    ):
        self.base_color = base_color
        self.opacity = Float32(1.0)
        self.roughness = max(Float32(0.04), min(Float32(1.0), roughness))
        self.metallic = max(Float32(0.0), min(Float32(1.0), metallic))
        self.specular_factor = Float32(1.0)
        self.alpha_cutoff = Float32(0.0)
        self.emissive = Vec3(0.0, 0.0, 0.0)
        self.clearcoat = Float32(0.0)
        self.clearcoat_roughness = Float32(0.08)
        self.iridescence = Float32(0.0)
        self.iridescence_thickness_nm = Float32(380.0)
        self.iridescence_ior = Float32(1.65)
        self.sheen_color = Vec3(0.0, 0.0, 0.0)
        self.sheen_roughness = Float32(0.5)
        self.subsurface_color = Vec3(0.9, 0.3, 0.2)
        self.subsurface_amount = Float32(0.0)
        self.detail_uv_scale = Float32(8.0)
        self.detail_normal_strength = Float32(0.0)
        self.domain = domain
        self.flags = flags
        self.albedo_slot = BINDLESS_SLOT_NONE
        self.normal_slot = BINDLESS_SLOT_NONE
        self.orm_slot = BINDLESS_SLOT_NONE
        self.emissive_slot = BINDLESS_SLOT_NONE
        self.detail_normal_slot = BINDLESS_SLOT_NONE
        self.subsurface_slot = BINDLESS_SLOT_NONE
        self.vat_pos_slot = BINDLESS_SLOT_NONE
        self.vat_nrm_slot = BINDLESS_SLOT_NONE
        self.uv_scale_x = Float32(1.0)
        self.uv_scale_y = Float32(1.0)

    @always_inline
    def set_albedo_slot(mut self, slot: Int):
        self.albedo_slot = UInt16(min(Int(BINDLESS_SLOT_MAX), slot)) if slot >= 0 else BINDLESS_SLOT_NONE

    @always_inline
    def set_normal_slot(mut self, slot: Int):
        self.normal_slot = UInt16(min(Int(BINDLESS_SLOT_MAX), slot)) if slot >= 0 else BINDLESS_SLOT_NONE

    @always_inline
    def set_orm_slot(mut self, slot: Int):
        self.orm_slot = UInt16(min(Int(BINDLESS_SLOT_MAX), slot)) if slot >= 0 else BINDLESS_SLOT_NONE

    def append_words(self, mut out: List[UInt32]):
        """Appends exact `24 x UInt32` (`96` bytes = `AAA_MATERIAL_WORDS`) unified GPU SSBO material representation."""
        self.to_aaa_material().append_words(out)

    def pack_words(self) -> List[UInt32]:
        """Packs unified `24`-word (`96`-byte) GPU `BIND_AAA_MATERIALS` SSBO representation."""
        var out = List[UInt32](capacity=MATERIAL_ASSET_WORDS)
        self.append_words(out)
        return out^

    @staticmethod
    def unpack_words(words: List[UInt32], offset: Int = 0) -> MaterialAsset:
        """Reconstructs a `MaterialAsset` from unified `24`-word (`AAA_MATERIAL_WORDS`) GPU SSBO words."""
        var m = MaterialAsset()
        if offset < 0 or offset + MATERIAL_ASSET_WORDS > len(words):
            return m^
        m.base_color = Vec3(
            bitcast[DType.float32](words[offset + 0]),
            bitcast[DType.float32](words[offset + 1]),
            bitcast[DType.float32](words[offset + 2]),
        )
        var enc_spec = bitcast[DType.float32](words[offset + 3])
        m.specular_factor = abs(enc_spec)
        if enc_spec < Float32(0.0):
            m.alpha_cutoff = Float32(0.5)
            m.flags |= MAT_FLAG_ALPHA_MASKED
        m.roughness = bitcast[DType.float32](words[offset + 4])
        m.metallic = bitcast[DType.float32](words[offset + 5])
        m.clearcoat = bitcast[DType.float32](words[offset + 6])
        var enc_cc_r = bitcast[DType.float32](words[offset + 7])
        m.clearcoat_roughness = abs(enc_cc_r)
        if enc_cc_r < Float32(0.0):
            m.flags |= MAT_FLAG_DOUBLE_SIDED
            m.domain = MAT_DOMAIN_TWO_SIDED_FOLIAGE
        if m.clearcoat > Float32(0.001):
            m.domain = MAT_DOMAIN_CLEARCOAT
            m.flags |= MAT_FLAG_CLEARCOAT
        m.emissive = Vec3(
            bitcast[DType.float32](words[offset + 8]),
            bitcast[DType.float32](words[offset + 9]),
            bitcast[DType.float32](words[offset + 10]),
        )
        var enc_det = bitcast[DType.float32](words[offset + 11])
        m.detail_uv_scale = abs(enc_det)
        if enc_det < Float32(0.0):
            m.flags |= MAT_FLAG_ROAD_WETNESS
        m.detail_normal_strength = bitcast[DType.float32](words[offset + 12])
        m.iridescence = bitcast[DType.float32](words[offset + 13])
        m.iridescence_thickness_nm = bitcast[DType.float32](words[offset + 14])
        m.iridescence_ior = bitcast[DType.float32](words[offset + 15])
        m.sheen_color = Vec3(
            bitcast[DType.float32](words[offset + 16]),
            bitcast[DType.float32](words[offset + 17]),
            bitcast[DType.float32](words[offset + 18]),
        )
        var enc_sh_r = bitcast[DType.float32](words[offset + 19])
        m.sheen_roughness = abs(enc_sh_r)
        if enc_sh_r < Float32(0.0):
            m.domain = MAT_DOMAIN_UNLIT_UI
        var sss_premult = Vec3(
            bitcast[DType.float32](words[offset + 20]),
            bitcast[DType.float32](words[offset + 21]),
            bitcast[DType.float32](words[offset + 22]),
        )
        var tex_pk = words[offset + 23]
        var t_alb = tex_pk & UInt32(0xFF)
        var t_nrm = (tex_pk >> UInt32(8)) & UInt32(0xFF)
        var t_orm = (tex_pk >> UInt32(16)) & UInt32(0xFF)
        var t_det = (tex_pk >> UInt32(24)) & UInt32(0x3F)
        m.albedo_slot = UInt16(t_alb) if t_alb < UInt32(0xFF) else BINDLESS_SLOT_NONE
        m.normal_slot = UInt16(t_nrm) if t_nrm < UInt32(0xFF) else BINDLESS_SLOT_NONE
        m.orm_slot = UInt16(t_orm) if t_orm < UInt32(0xFF) else BINDLESS_SLOT_NONE
        m.detail_normal_slot = UInt16(t_det) if t_det < AAA_MAT_DET_NONE else BINDLESS_SLOT_NONE
        if (tex_pk & AAA_MAT_INTERIOR_PARALLAX) != 0:
            m.flags |= MAT_FLAG_INTERIOR_PARALLAX
        if (tex_pk & AAA_MAT_SSS_BIT) != 0:
            var max_c = max(sss_premult.x, max(sss_premult.y, sss_premult.z))
            if max_c > Float32(1e-5):
                m.subsurface_amount = min(Float32(1.0), max_c)
                m.subsurface_color = sss_premult * (Float32(1.0) / m.subsurface_amount)
            m.flags |= MAT_FLAG_SUBSURFACE
            if m.domain == MAT_DOMAIN_STANDARD_PBR:
                m.domain = MAT_DOMAIN_SUBSURFACE_SKIN
        return m^

    def pack_archive_words(self) -> List[UInt32]:
        """Packs lossless `32 x UInt32` (`128` bytes) `.flpk` archive metadata including 16-bit bindless slots (`0..65,534`)."""
        var out = List[UInt32](capacity=MATERIAL_ARCHIVE_WORDS)
        var eff_flags = self.flags
        if self.alpha_cutoff > Float32(0.01):
            eff_flags |= MAT_FLAG_ALPHA_MASKED
        if self.domain == MAT_DOMAIN_TWO_SIDED_FOLIAGE:
            eff_flags |= MAT_FLAG_DOUBLE_SIDED
        if self.domain == MAT_DOMAIN_SUBSURFACE_SKIN or self.subsurface_amount > Float32(0.001):
            eff_flags |= MAT_FLAG_SUBSURFACE
        if self.domain == MAT_DOMAIN_CLEARCOAT or self.clearcoat > Float32(0.001):
            eff_flags |= MAT_FLAG_CLEARCOAT

        out.append(bitcast[DType.uint32](self.base_color.x))             # [0]
        out.append(bitcast[DType.uint32](self.base_color.y))             # [1]
        out.append(bitcast[DType.uint32](self.base_color.z))             # [2]
        out.append(bitcast[DType.uint32](self.opacity))                  # [3]
        out.append(bitcast[DType.uint32](self.roughness))                # [4]
        out.append(bitcast[DType.uint32](self.metallic))                 # [5]
        out.append(bitcast[DType.uint32](self.specular_factor))          # [6]
        out.append(bitcast[DType.uint32](self.alpha_cutoff))             # [7]
        out.append(bitcast[DType.uint32](self.emissive.x))               # [8]
        out.append(bitcast[DType.uint32](self.emissive.y))               # [9]
        out.append(bitcast[DType.uint32](self.emissive.z))               # [10]
        out.append(bitcast[DType.uint32](self.clearcoat))                # [11]
        out.append(bitcast[DType.uint32](self.clearcoat_roughness))      # [12]
        out.append(bitcast[DType.uint32](self.iridescence))              # [13]
        out.append(bitcast[DType.uint32](self.iridescence_thickness_nm)) # [14]
        out.append(bitcast[DType.uint32](self.iridescence_ior))          # [15]
        out.append(bitcast[DType.uint32](self.sheen_color.x))            # [16]
        out.append(bitcast[DType.uint32](self.sheen_color.y))            # [17]
        out.append(bitcast[DType.uint32](self.sheen_color.z))            # [18]
        out.append(bitcast[DType.uint32](self.sheen_roughness))          # [19]
        out.append(bitcast[DType.uint32](self.subsurface_color.x))       # [20]
        out.append(bitcast[DType.uint32](self.subsurface_color.y))       # [21]
        out.append(bitcast[DType.uint32](self.subsurface_color.z))       # [22]
        out.append(bitcast[DType.uint32](self.subsurface_amount))        # [23]
        out.append(bitcast[DType.uint32](self.detail_uv_scale))          # [24]
        out.append(bitcast[DType.uint32](self.detail_normal_strength))   # [25]
        out.append((self.domain & UInt32(0xFF)) | (eff_flags << 8))      # [26]
        out.append(_pack_u16_pair(self.albedo_slot, self.normal_slot))   # [27]
        out.append(_pack_u16_pair(self.orm_slot, self.emissive_slot))    # [28]
        out.append(_pack_u16_pair(self.detail_normal_slot, self.subsurface_slot))  # [29]
        out.append(_pack_u16_pair(self.vat_pos_slot, self.vat_nrm_slot))           # [30]
        out.append(_pack_u16_pair(f32_to_f16_bits(self.uv_scale_x), f32_to_f16_bits(self.uv_scale_y)))  # [31]
        return out^

    @staticmethod
    def unpack_archive_words(words: List[UInt32], offset: Int = 0) -> MaterialAsset:
        """Losslessly reconstructs a `MaterialAsset` (including 16-bit bindless slots `0..65,534`) from 32 archive words."""
        var m = MaterialAsset()
        if offset < 0 or offset + MATERIAL_ARCHIVE_WORDS > len(words):
            return m^
        m.base_color = Vec3(
            bitcast[DType.float32](words[offset + 0]),
            bitcast[DType.float32](words[offset + 1]),
            bitcast[DType.float32](words[offset + 2]),
        )
        m.opacity = bitcast[DType.float32](words[offset + 3])
        m.roughness = bitcast[DType.float32](words[offset + 4])
        m.metallic = bitcast[DType.float32](words[offset + 5])
        m.specular_factor = bitcast[DType.float32](words[offset + 6])
        m.alpha_cutoff = bitcast[DType.float32](words[offset + 7])
        m.emissive = Vec3(
            bitcast[DType.float32](words[offset + 8]),
            bitcast[DType.float32](words[offset + 9]),
            bitcast[DType.float32](words[offset + 10]),
        )
        m.clearcoat = bitcast[DType.float32](words[offset + 11])
        m.clearcoat_roughness = bitcast[DType.float32](words[offset + 12])
        m.iridescence = bitcast[DType.float32](words[offset + 13])
        m.iridescence_thickness_nm = bitcast[DType.float32](words[offset + 14])
        m.iridescence_ior = bitcast[DType.float32](words[offset + 15])
        m.sheen_color = Vec3(
            bitcast[DType.float32](words[offset + 16]),
            bitcast[DType.float32](words[offset + 17]),
            bitcast[DType.float32](words[offset + 18]),
        )
        m.sheen_roughness = bitcast[DType.float32](words[offset + 19])
        m.subsurface_color = Vec3(
            bitcast[DType.float32](words[offset + 20]),
            bitcast[DType.float32](words[offset + 21]),
            bitcast[DType.float32](words[offset + 22]),
        )
        m.subsurface_amount = bitcast[DType.float32](words[offset + 23])
        m.detail_uv_scale = bitcast[DType.float32](words[offset + 24])
        m.detail_normal_strength = bitcast[DType.float32](words[offset + 25])
        var df = words[offset + 26]
        m.domain = df & UInt32(0xFF)
        m.flags = df >> 8
        var p0 = _unpack_u16_pair(words[offset + 27])
        m.albedo_slot = p0[0]
        m.normal_slot = p0[1]
        var p1 = _unpack_u16_pair(words[offset + 28])
        m.orm_slot = p1[0]
        m.emissive_slot = p1[1]
        var p2 = _unpack_u16_pair(words[offset + 29])
        m.detail_normal_slot = p2[0]
        m.subsurface_slot = p2[1]
        var p3 = _unpack_u16_pair(words[offset + 30])
        m.vat_pos_slot = p3[0]
        m.vat_nrm_slot = p3[1]
        var uvs = _unpack_u16_pair(words[offset + 31])
        m.uv_scale_x = f16_bits_to_f32(uvs[0])
        m.uv_scale_y = f16_bits_to_f32(uvs[1])
        return m^

    def to_aaa_material(self) -> AaaMaterial:
        """Bridges `MaterialAsset` to `AaaMaterial` for direct `FlintAaaVulkanRenderer` VisBuffer+RT shading."""
        var out = AaaMaterial(
            base_color=self.base_color,
            roughness=self.roughness,
            metallic=self.metallic,
        )
        out.opacity = self.opacity
        out.specular_factor = max(Float32(0.0), min(Float32(1.0), self.specular_factor))
        out.clearcoat = self.clearcoat
        out.clearcoat_roughness = max(Float32(0.04), min(Float32(1.0), self.clearcoat_roughness))
        out.emissive = self.emissive
        out.detail_uv_scale = max(Float32(0.001), self.detail_uv_scale)
        out.detail_normal_strength = self.detail_normal_strength
        out.iridescence = self.iridescence
        out.iridescence_thickness_nm = self.iridescence_thickness_nm
        out.iridescence_ior = self.iridescence_ior
        out.sheen_color = self.sheen_color
        out.sheen_roughness = max(Float32(0.05), self.sheen_roughness)
        out.subsurface_color = self.subsurface_color
        out.subsurface_amount = self.subsurface_amount
        out.albedo_tex = Int(self.albedo_slot) if self.albedo_slot != BINDLESS_SLOT_NONE else -1
        out.normal_tex = Int(self.normal_slot) if self.normal_slot != BINDLESS_SLOT_NONE else -1
        out.orm_tex = Int(self.orm_slot) if self.orm_slot != BINDLESS_SLOT_NONE else -1
        out.detail_normal_tex = Int(self.detail_normal_slot) if self.detail_normal_slot != BINDLESS_SLOT_NONE else -1
        out.interior_parallax_enabled = (self.flags & MAT_FLAG_INTERIOR_PARALLAX) != 0
        out.road_wetness_mode = (self.flags & MAT_FLAG_ROAD_WETNESS) != 0
        out.unlit = (self.domain == MAT_DOMAIN_UNLIT_UI)
        out.two_sided_foliage = (self.domain == MAT_DOMAIN_TWO_SIDED_FOLIAGE) or ((self.flags & MAT_FLAG_DOUBLE_SIDED) != 0)
        if (self.flags & MAT_FLAG_ALPHA_MASKED) != 0 or self.alpha_cutoff > Float32(0.01):
            out.alpha_cutoff = max(Float32(0.05), self.alpha_cutoff)
        return out^


# ============================================================================
# 2. `SurfaceMaterialNode` `ShaderDsl` Trait & SPIR-V Material Graph Compiler
# ============================================================================

@fieldwise_init
struct SurfaceMaterialInputs(Copyable, ImplicitlyCopyable, Movable):
    var uv: GpuVec2
    var world_pos: GpuVec3
    var world_normal: GpuVec3
    var view_dir: GpuVec3
    var time_sec: GpuF32
    var mat_id: GpuU32


@fieldwise_init
struct SurfaceMaterialOutputs(Copyable, ImplicitlyCopyable, Movable):
    var base_color: GpuVec3
    var normal_ts: GpuVec3
    var roughness: GpuF32
    var metallic: GpuF32
    var emissive: GpuVec3
    var opacity: GpuF32
    var clearcoat: GpuF32
    var clearcoat_roughness: GpuF32
    var world_pos_offset: GpuVec3


trait SurfaceMaterialNode(Copyable, Movable):
    """Composable pure-Mojo `ShaderDsl` surface material graph node."""
    def evaluate_surface(
        self,
        mut dsl: ShaderDsl,
        inputs: SurfaceMaterialInputs,
        b_mats: BufferRef[MATERIAL_ASSET_WORDS],
        bindless: BindlessTex2D,
    ) -> SurfaceMaterialOutputs:
        ...


struct StandardPbrMaterialNode(SurfaceMaterialNode):
    """Evaluates the 128-byte `MaterialAsset` SSBO + 16-bit bindless Albedo/Normal/ORM textures on GPU."""
    var enable_detail_layer: Bool

    def __init__(out self, enable_detail_layer: Bool = True):
        self.enable_detail_layer = enable_detail_layer

    def evaluate_surface(
        self,
        mut dsl: ShaderDsl,
        inputs: SurfaceMaterialInputs,
        b_mats: BufferRef[MATERIAL_ASSET_WORDS],
        bindless: BindlessTex2D,
    ) -> SurfaceMaterialOutputs:
        var base = b_mats.base_word(inputs.mat_id)
        var base_col = b_mats.load_vec3(base)
        var rough = b_mats.load_f32(base + UInt32(4))
        var metal = b_mats.load_f32(base + UInt32(5))
        var cc = b_mats.load_f32(base + UInt32(6))
        var cc_r = b_mats.load_f32(base + UInt32(7))
        var emissive = b_mats.load_vec3(base + UInt32(8))

        var tex_pk = b_mats.load_u32(base + UInt32(23))
        var alb_slot = tex_pk & UInt32(0xFF)
        var nrm_slot = (tex_pk >> UInt32(8)) & UInt32(0xFF)
        var orm_slot = (tex_pk >> UInt32(16)) & UInt32(0xFF)

        var out_col = dsl.var_vec3(base_col)
        var out_op = dsl.var_f32(dsl.f32(1.0))
        var out_rgh = dsl.var_f32(rough)
        var out_met = dsl.var_f32(metal)
        var out_nrm = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 1.0))

        # Unified `AAA_MATERIAL_WORDS = 24` bindless slot check (`slot < 0xFF`, capped to `AAA_MAX_BINDLESS - 1`)
        var max_b = dsl.u32(UInt32(AAA_MAX_BINDLESS - 1))
        dsl.if_then(alb_slot.lt(UInt32(0xFF)))
        var s_alb = bindless.sample_lod(
            GpuU32(dsl.mod_addr, dsl.raw_module()[].umin(alb_slot.id, max_b.id)),
            inputs.uv,
            dsl.f32(0.0),
        )
        out_col.set(base_col * s_alb.xyz())
        out_op.set(s_alb.w())
        dsl.end_if()

        dsl.if_then(nrm_slot.lt(UInt32(0xFF)))
        var s_nrm = bindless.sample_lod(
            GpuU32(dsl.mod_addr, dsl.raw_module()[].umin(nrm_slot.id, max_b.id)),
            inputs.uv,
            dsl.f32(0.0),
        )
        var nx = fma_f(s_nrm.x(), dsl.f32(2.0), dsl.f32(-1.0))
        var ny = fma_f(s_nrm.y(), dsl.f32(2.0), dsl.f32(-1.0))
        var nz = dsl.f32(1.0) - clamp_f(nx * nx + ny * ny, 0.0, 0.98)
        out_nrm.set(normalize(dsl.vec3(nx, ny, max_f(nz, 0.1))))
        dsl.end_if()

        dsl.if_then(orm_slot.lt(UInt32(0xFF)))
        var s_orm = bindless.sample_lod(
            GpuU32(dsl.mod_addr, dsl.raw_module()[].umin(orm_slot.id, max_b.id)),
            inputs.uv,
            dsl.f32(0.0),
        )
        out_rgh.set(clamp_f(rough * s_orm.y(), 0.04, 1.0))
        out_met.set(clamp_f(metal * s_orm.z(), 0.0, 1.0))
        dsl.end_if()

        if self.enable_detail_layer:
            var det_scale = abs_f(b_mats.load_f32(base + UInt32(11)))
            var det_str = b_mats.load_f32(base + UInt32(12))
            dsl.if_then(det_str.gt(0.001))
            var ripple = sin_f(inputs.uv.x() * det_scale) * sin_f(inputs.uv.y() * det_scale) * det_str * Float32(0.15)
            var cur_n = out_nrm.get()
            out_nrm.set(normalize(dsl.vec3(cur_n.x() + ripple, cur_n.y() + ripple, cur_n.z())))
            dsl.end_if()

        return SurfaceMaterialOutputs(
            out_col.get(),
            out_nrm.get(),
            out_rgh.get(),
            out_met.get(),
            emissive,
            out_op.get(),
            cc,
            cc_r,
            dsl.vec3_const(0.0, 0.0, 0.0),
        )


struct TriplanarWeatherMaterialNode(SurfaceMaterialNode):
    """Procedural world-space triplanar + wetness surface graph node."""
    var snow_line_y: Float32

    def __init__(out self, snow_line_y: Float32 = 12.0):
        self.snow_line_y = snow_line_y

    def evaluate_surface(
        self,
        mut dsl: ShaderDsl,
        inputs: SurfaceMaterialInputs,
        b_mats: BufferRef[MATERIAL_ASSET_WORDS],
        bindless: BindlessTex2D,
    ) -> SurfaceMaterialOutputs:
        var std_node = StandardPbrMaterialNode(enable_detail_layer=False)
        var sub = std_node.evaluate_surface(dsl, inputs, b_mats, bindless)
        var snow_t = clamp_f(
            (inputs.world_pos.y() - dsl.f32(self.snow_line_y)) * Float32(0.25) * max_f(inputs.world_normal.y(), 0.0),
            0.0,
            1.0,
        )
        var snow_col = mix_vec3(sub.base_color, dsl.vec3_const(0.92, 0.95, 0.98), snow_t)
        var snow_rgh = fma_f(dsl.f32(0.65) - sub.roughness, snow_t, sub.roughness)
        return SurfaceMaterialOutputs(
            snow_col,
            sub.normal_ts,
            snow_rgh,
            sub.metallic * (dsl.f32(1.0) - snow_t),
            sub.emissive,
            sub.opacity,
            sub.clearcoat,
            sub.clearcoat_roughness,
            sub.world_pos_offset,
        )


def compile_surface_material_shader[NodeT: SurfaceMaterialNode](node: NodeT) -> List[UInt32]:
    """Compiles any `SurfaceMaterialNode` graph into a validated SPIR-V 1.5 compute evaluation kernel
    with `OpSpecConstant` (`SpecId 0 = enable_clearcoat`, `SpecId 1 = domain_override`)."""
    var dsl = ShaderDsl(
        MODEL_GL_COMPUTE,
        group_width=8,
        group_height=8,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
    )
    var spec_cc_id = dsl.raw_module()[].spec_constant_bool(0, True)
    var spec_mode_id = dsl.raw_module()[].spec_constant_u32(1, 0)
    var spec_cc = GpuBool(dsl.mod_addr, spec_cc_id)
    var spec_mode = GpuU32(dsl.mod_addr, spec_mode_id)

    var b_mats = dsl.buffer[MATERIAL_ASSET_WORDS](BIND_AAA_MATERIALS)
    var b_out = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var bindless = dsl.bindless_tex2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)

    var gx = dsl.global_id_x()
    var gy = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var mat_id = dsl.push_u32(2)
    var time_s = dsl.push_f32(3)
    var out_base = dsl.push_u32(4)
    var wy_scale = max_f(dsl.push_f32(5), 1.0)
    dsl.return_if(gx.ge(w_u) | gy.ge(h_u))

    var u = (gx.to_f32() + Float32(0.5)) / max_f(w_u.to_f32(), 1.0)
    var v = (gy.to_f32() + Float32(0.5)) / max_f(h_u.to_f32(), 1.0)
    var inputs = SurfaceMaterialInputs(
        dsl.vec2(u, v),
        dsl.vec3(u * wy_scale, (dsl.f32(1.0) - v) * wy_scale, dsl.f32(0.0)),
        dsl.vec3_const(0.0, 1.0, 0.0),
        dsl.vec3_const(0.0, 0.0, 1.0),
        time_s,
        mat_id,
    )
    var res = node.evaluate_surface(dsl, inputs, b_mats, bindless)
    var cc_out = dsl.var_f32(dsl.f32(0.0))
    var rough_out = dsl.var_f32(res.roughness)
    var metal_out = dsl.var_f32(res.metallic)
    var em_gain = dsl.var_f32(dsl.f32(1.0))

    dsl.if_then(spec_cc)
    cc_out.set(res.clearcoat)
    dsl.end_if()

    # SpecId 1 (`domain_override`): 0 = standard, 1 = clearcoat, 2 = subsurface skin, 4 = unlit emissive
    dsl.if_then(spec_mode.eq(dsl.u32(UInt32(MAT_DOMAIN_CLEARCOAT))))
    cc_out.set(max_f(cc_out.get(), dsl.f32(0.5)))
    rough_out.set(clamp_f(rough_out.get() * dsl.f32(0.85), dsl.f32(0.04), dsl.f32(1.0)))
    dsl.end_if()

    dsl.if_then(spec_mode.eq(dsl.u32(UInt32(MAT_DOMAIN_SUBSURFACE_SKIN))))
    metal_out.set(dsl.f32(0.0))
    rough_out.set(max_f(rough_out.get(), dsl.f32(0.35)))
    dsl.end_if()

    dsl.if_then(spec_mode.eq(dsl.u32(UInt32(MAT_DOMAIN_UNLIT_UI))))
    metal_out.set(dsl.f32(0.0))
    em_gain.set(dsl.f32(2.0))
    dsl.end_if()

    var out_idx = out_base + (gy * w_u + gx) * UInt32(8)
    b_out.store_vec3(out_idx, res.base_color + res.emissive * em_gain.get())
    b_out.store_f32(out_idx + UInt32(3), rough_out.get())
    b_out.store_f32(out_idx + UInt32(4), metal_out.get())
    b_out.store_f32(out_idx + UInt32(5), cc_out.get())
    b_out.store_f32(out_idx + UInt32(6), res.opacity)
    b_out.store_f32(out_idx + UInt32(7), abs_f(res.normal_ts.z()))
    return dsl.finish()


# ============================================================================
# 3. `GpuDecalVolume` (96-Byte / 24-Word OBB Deferred Decal Volume)
# ============================================================================

comptime GPU_DECAL_VOLUME_WORDS: Int = 24
comptime AAA_MAX_DECALS: Int = 32
comptime AAA_GLOBALS_DECAL_COUNT_WORD: Int = 3008
comptime AAA_GLOBALS_DECAL_TABLE_WORD: Int = 3016  # 32 * 24 = 768 words: [3016 .. 3783] in ring-buffered buf_globals

comptime DECAL_BLEND_ALBEDO: UInt16 = 1 << 0
comptime DECAL_BLEND_NORMAL: UInt16 = 1 << 1
comptime DECAL_BLEND_ROUGHNESS: UInt16 = 1 << 2
comptime DECAL_BLEND_STAIN_MULTIPLY: UInt16 = 1 << 3


struct GpuDecalVolume(Copyable, Movable):
    """Oriented Bounding Box (`OBB`) deferred decal projecting into unit local space `[-0.5, +0.5]^3`."""
    var world_to_obb: Mat4
    var obb_up_world: Vec3
    var albedo_slot: UInt16
    var normal_slot: UInt16
    var orm_slot: UInt16
    var blend_flags: UInt16
    var tint_rgb: Vec3
    var opacity: Float32
    var normal_fade_cos: Float32
    var sort_priority: UInt32

    def __init__(
        out self,
        center: Vec3 = Vec3(),
        half_extents: Vec3 = Vec3(1.0, 0.5, 1.0),
        rotation: Quat = Quat(),
        albedo_slot: UInt16 = BINDLESS_SLOT_NONE,
        opacity: Float32 = 1.0,
    ):
        var q = rotation.normalized()
        var rot_m = trs(Vec3(), q, Vec3(1.0, 1.0, 1.0))
        var rx = Vec3(rot_m.values[0], rot_m.values[1], rot_m.values[2]).normalized()
        var ry = Vec3(rot_m.values[4], rot_m.values[5], rot_m.values[6]).normalized()
        var rz = Vec3(rot_m.values[8], rot_m.values[9], rot_m.values[10]).normalized()
        var inv_sx = Float32(0.5) / max(Float32(1e-4), half_extents.x)
        var inv_sy = Float32(0.5) / max(Float32(1e-4), half_extents.y)
        var inv_sz = Float32(0.5) / max(Float32(1e-4), half_extents.z)

        var inv_m = Mat4()
        inv_m.set(0, 0, rx.x * inv_sx); inv_m.set(0, 1, rx.y * inv_sx); inv_m.set(0, 2, rx.z * inv_sx); inv_m.set(0, 3, -rx.dot(center) * inv_sx)
        inv_m.set(1, 0, ry.x * inv_sy); inv_m.set(1, 1, ry.y * inv_sy); inv_m.set(1, 2, ry.z * inv_sy); inv_m.set(1, 3, -ry.dot(center) * inv_sy)
        inv_m.set(2, 0, rz.x * inv_sz); inv_m.set(2, 1, rz.y * inv_sz); inv_m.set(2, 2, rz.z * inv_sz); inv_m.set(2, 3, -rz.dot(center) * inv_sz)

        self.world_to_obb = inv_m^
        self.obb_up_world = ry
        self.albedo_slot = albedo_slot
        self.normal_slot = BINDLESS_SLOT_NONE
        self.orm_slot = BINDLESS_SLOT_NONE
        self.blend_flags = DECAL_BLEND_ALBEDO | DECAL_BLEND_ROUGHNESS
        self.tint_rgb = Vec3(1.0, 1.0, 1.0)
        self.opacity = max(Float32(0.0), min(Float32(1.0), opacity))
        self.normal_fade_cos = Float32(0.25)
        self.sort_priority = 0

    def project_world_point(self, world_pt: Vec3, surface_normal: Vec3) -> Tuple[Bool, Float32, Float32, Float32]:
        """Projects `world_pt` into OBB decal UV `[0..1]^2`, returning `(inside, u, v, alpha_weight)`."""
        var loc = self.world_to_obb.transform_point(world_pt)
        if abs(loc.x) > Float32(0.5) or abs(loc.y) > Float32(0.5) or abs(loc.z) > Float32(0.5):
            return (False, 0.0, 0.0, 0.0)
        var cos_ang = surface_normal.normalized().dot(self.obb_up_world)
        if cos_ang <= self.normal_fade_cos:
            return (False, 0.0, 0.0, 0.0)
        var ang_fade = min(Float32(1.0), (cos_ang - self.normal_fade_cos) / max(Float32(0.05), Float32(1.0) - self.normal_fade_cos))
        var box_fade = min(Float32(1.0), (Float32(0.5) - abs(loc.y)) * Float32(6.0))
        return (True, loc.x + Float32(0.5), loc.z + Float32(0.5), self.opacity * ang_fade * box_fade)

    def pack_words(self) -> List[UInt32]:
        var out = List[UInt32](capacity=GPU_DECAL_VOLUME_WORDS)
        for k in range(16):
            out.append(bitcast[DType.uint32](self.world_to_obb.values[k]))
        out.append(_pack_u16_pair(self.albedo_slot, self.normal_slot))
        out.append(_pack_u16_pair(self.orm_slot, self.blend_flags))
        out.append(bitcast[DType.uint32](self.tint_rgb.x))
        out.append(bitcast[DType.uint32](self.tint_rgb.y))
        out.append(bitcast[DType.uint32](self.tint_rgb.z))
        out.append(bitcast[DType.uint32](self.opacity))
        out.append(bitcast[DType.uint32](self.normal_fade_cos))
        out.append(self.sort_priority)
        return out^


# ============================================================================
# 4. `TerrainClipmapDesc` (Concentric Toroidal Terrain Clipmap Descriptor)
# ============================================================================

comptime AAA_GLOBALS_TERRAIN_CLIPMAP_WORD: Int = 3784  # 12 words [3784..3795] from TerrainClipmapDesc.pack_words()
comptime AAA_GLOBALS_TERRAIN_META_WORD: Int = 3796     # [0]=enabled_u32, [1]=world_size_m, [2]=world_origin_x, [3]=world_origin_z
comptime AAA_GLOBALS_TERRAIN_ORIGINS_WORD: Int = 3800  # 8 x (ox_f32, oz_f32) = 16 words [3800..3815]

struct TerrainClipmapDesc(Copyable, Movable):
    """Configures concentric GPU geometry clipmap terrain rings with 16-bit bindless splat maps."""
    var ring_count: Int
    var grid_dim_per_ring: Int
    var base_cell_size_m: Float32
    var height_scale_m: Float32
    var height_offset_m: Float32
    var morph_start_ratio: Float32
    var heightmap_slot: UInt16
    var normalmap_slot: UInt16
    var splat_control_slot: UInt16
    var layer_albedo_slots: List[UInt16]
    var layer_normal_slots: List[UInt16]

    def __init__(
        out self,
        ring_count: Int = 5,
        grid_dim_per_ring: Int = 64,
        base_cell_size_m: Float32 = 0.5,
        height_scale_m: Float32 = 120.0,
    ):
        self.ring_count = max(1, min(8, ring_count))
        self.grid_dim_per_ring = max(16, grid_dim_per_ring)
        self.base_cell_size_m = max(Float32(0.05), base_cell_size_m)
        self.height_scale_m = height_scale_m
        self.height_offset_m = Float32(0.0)
        self.morph_start_ratio = Float32(0.70)
        self.heightmap_slot = BINDLESS_SLOT_NONE
        self.normalmap_slot = BINDLESS_SLOT_NONE
        self.splat_control_slot = BINDLESS_SLOT_NONE
        self.layer_albedo_slots = List[UInt16](length=4, fill=BINDLESS_SLOT_NONE)
        self.layer_normal_slots = List[UInt16](length=4, fill=BINDLESS_SLOT_NONE)

    def ring_cell_size(self, ring_idx: Int) -> Float32:
        var r = max(0, min(self.ring_count - 1, ring_idx))
        return self.base_cell_size_m * Float32(1 << r)

    def ring_half_extent(self, ring_idx: Int) -> Float32:
        return self.ring_cell_size(ring_idx) * Float32(self.grid_dim_per_ring) * Float32(0.5)

    def snapped_ring_origin_xz(self, cam_pos: Vec3, ring_idx: Int) -> Tuple[Float32, Float32]:
        """Snaps `cam_pos.xz` to `2 * ring_cell_size` (coarse parent grid step) to prevent T-junction swimming."""
        var snap_step = self.ring_cell_size(ring_idx) * Float32(2.0)
        var sx = floor(cam_pos.x / snap_step + Float32(0.5)) * snap_step
        var sz = floor(cam_pos.z / snap_step + Float32(0.5)) * snap_step
        return (sx, sz)

    def pack_words(self) -> List[UInt32]:
        var out = List[UInt32](capacity=12)
        out.append(UInt32(self.ring_count))
        out.append(UInt32(self.grid_dim_per_ring))
        out.append(bitcast[DType.uint32](self.base_cell_size_m))
        out.append(bitcast[DType.uint32](self.height_scale_m))
        out.append(bitcast[DType.uint32](self.height_offset_m))
        out.append(bitcast[DType.uint32](self.morph_start_ratio))
        out.append(_pack_u16_pair(self.heightmap_slot, self.normalmap_slot))
        out.append(_pack_u16_pair(self.splat_control_slot, 0))
        out.append(_pack_u16_pair(self.layer_albedo_slots[0], self.layer_albedo_slots[1]))
        out.append(_pack_u16_pair(self.layer_albedo_slots[2], self.layer_albedo_slots[3]))
        out.append(_pack_u16_pair(self.layer_normal_slots[0], self.layer_normal_slots[1]))
        out.append(_pack_u16_pair(self.layer_normal_slots[2], self.layer_normal_slots[3]))
        return out^


# ============================================================================
# 5. `VatCrowdClipDesc` & `RGBA16F` Skeletal -> Vertex Animation Texture Baker
# ============================================================================

struct VatCrowdClipDesc(Copyable, ImplicitlyCopyable, Movable):
    """Metadata describing a baked `RGBA16F` Vertex Animation Texture (`VAT`) crowd clip."""
    var clip_index: Int
    var vertex_count: Int
    var frame_count: Int
    var sample_fps: Float32
    var duration_sec: Float32
    var looping: Bool
    var pos_tex_slot: UInt16
    var nrm_tex_slot: UInt16

    def __init__(
        out self,
        clip_index: Int = 0,
        vertex_count: Int = 0,
        frame_count: Int = 1,
        sample_fps: Float32 = 30.0,
        duration_sec: Float32 = 1.0,
        looping: Bool = True,
    ):
        self.clip_index = clip_index
        self.vertex_count = max(0, vertex_count)
        self.frame_count = max(1, frame_count)
        self.sample_fps = max(Float32(1.0), sample_fps)
        self.duration_sec = max(Float32(1.0 / 60.0), duration_sec)
        self.looping = looping
        self.pos_tex_slot = BINDLESS_SLOT_NONE
        self.nrm_tex_slot = BINDLESS_SLOT_NONE


struct BakedVatCrowdClip(Copyable, Movable):
    """Baked `RGBA16F` (`width = vertex_count`, `height = frame_count`) position + normal VAT textures."""
    var desc: VatCrowdClipDesc
    var pos_rgba16f: List[UInt16]  # 4 x UInt16 (px, py, pz, 1.0) per (frame * vertex_count + v)
    var nrm_rgba16f: List[UInt16]  # 4 x UInt16 (nx, ny, nz, 0.0) per (frame * vertex_count + v)
    var bounds_min: Vec3
    var bounds_max: Vec3

    def __init__(out self, desc: VatCrowdClipDesc):
        self.desc = desc
        var texels = max(1, desc.vertex_count * desc.frame_count)
        self.pos_rgba16f = List[UInt16](length=texels * 4, fill=0)
        self.nrm_rgba16f = List[UInt16](length=texels * 4, fill=0)
        self.bounds_min = Vec3(1e9, 1e9, 1e9)
        self.bounds_max = Vec3(-1e9, -1e9, -1e9)

    def byte_size_per_sheet(self) -> Int:
        return len(self.pos_rgba16f) * 2

    def read_pos_frame(self, frame: Int, vertex: Int) -> Vec3:
        var f_cl = max(0, min(self.desc.frame_count - 1, frame))
        var v_cl = max(0, min(self.desc.vertex_count - 1, vertex))
        var idx = (f_cl * self.desc.vertex_count + v_cl) * 4
        return Vec3(
            f16_bits_to_f32(self.pos_rgba16f[idx + 0]),
            f16_bits_to_f32(self.pos_rgba16f[idx + 1]),
            f16_bits_to_f32(self.pos_rgba16f[idx + 2]),
        )

    def read_nrm_frame(self, frame: Int, vertex: Int) -> Vec3:
        var f_cl = max(0, min(self.desc.frame_count - 1, frame))
        var v_cl = max(0, min(self.desc.vertex_count - 1, vertex))
        var idx = (f_cl * self.desc.vertex_count + v_cl) * 4
        return Vec3(
            f16_bits_to_f32(self.nrm_rgba16f[idx + 0]),
            f16_bits_to_f32(self.nrm_rgba16f[idx + 1]),
            f16_bits_to_f32(self.nrm_rgba16f[idx + 2]),
        ).normalized()


def _sample_channel_vec3(ch: NativeChannel, t: Float32, fallback: Vec3) -> Vec3:
    var n = len(ch.times)
    if n == 0 or len(ch.values) < n * 3:
        return fallback
    if n == 1 or t <= ch.times[0]:
        return Vec3(ch.values[0], ch.values[1], ch.values[2])
    if t >= ch.times[n - 1]:
        var last = (n - 1) * 3
        return Vec3(ch.values[last], ch.values[last + 1], ch.values[last + 2])
    var k = 0
    while k + 1 < n and ch.times[k + 1] < t:
        k += 1
    if ch.step:
        return Vec3(ch.values[k * 3], ch.values[k * 3 + 1], ch.values[k * 3 + 2])
    var dt = max(Float32(1e-6), ch.times[k + 1] - ch.times[k])
    var alpha = max(Float32(0.0), min(Float32(1.0), (t - ch.times[k]) / dt))
    var a = Vec3(ch.values[k * 3], ch.values[k * 3 + 1], ch.values[k * 3 + 2])
    var b = Vec3(ch.values[(k + 1) * 3], ch.values[(k + 1) * 3 + 1], ch.values[(k + 1) * 3 + 2])
    return a + (b - a) * alpha


def _sample_channel_quat(ch: NativeChannel, t: Float32, fallback: Quat) -> Quat:
    var n = len(ch.times)
    if n == 0 or len(ch.values) < n * 4:
        return fallback
    if n == 1 or t <= ch.times[0]:
        return Quat(ch.values[0], ch.values[1], ch.values[2], ch.values[3]).normalized()
    if t >= ch.times[n - 1]:
        var last = (n - 1) * 4
        return Quat(ch.values[last], ch.values[last + 1], ch.values[last + 2], ch.values[last + 3]).normalized()
    var k = 0
    while k + 1 < n and ch.times[k + 1] < t:
        k += 1
    var qa = Quat(ch.values[k * 4], ch.values[k * 4 + 1], ch.values[k * 4 + 2], ch.values[k * 4 + 3]).normalized()
    if ch.step:
        return qa
    var qb = Quat(ch.values[(k + 1) * 4], ch.values[(k + 1) * 4 + 1], ch.values[(k + 1) * 4 + 2], ch.values[(k + 1) * 4 + 3]).normalized()
    if qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w < Float32(0.0):
        qb = Quat(-qb.x, -qb.y, -qb.z, -qb.w)
    var dt = max(Float32(1e-6), ch.times[k + 1] - ch.times[k])
    var alpha = max(Float32(0.0), min(Float32(1.0), (t - ch.times[k]) / dt))
    var om = Float32(1.0) - alpha
    return Quat(
        qa.x * om + qb.x * alpha,
        qa.y * om + qb.y * alpha,
        qa.z * om + qb.z * alpha,
        qa.w * om + qb.w * alpha,
    ).normalized()


def bake_vat_crowd_clip_rgba16f(
    model: NativeModel,
    mesh_index: Int = 0,
    clip_index: Int = 0,
    sample_fps: Float32 = 30.0,
    looping: Bool = True,
) -> BakedVatCrowdClip:
    """Bakes a skeletal `NativeModel` animation clip into `RGBA16F` position + normal sheets (`width = verts, height = frames`)."""
    if mesh_index < 0 or mesh_index >= len(model.meshes):
        return BakedVatCrowdClip(VatCrowdClipDesc())

    ref mesh = model.meshes[mesh_index]
    var v_cnt = mesh.vertex_count()
    if v_cnt <= 0:
        return BakedVatCrowdClip(VatCrowdClipDesc())

    var dur = Float32(1.0)
    var has_clip = clip_index >= 0 and clip_index < len(model.animations)
    if has_clip:
        dur = max(Float32(1.0 / 30.0), model.animations[clip_index].duration)
    var fps = max(Float32(1.0), sample_fps)
    var f_cnt = max(2, Int(dur * fps + Float32(0.5)) + (0 if looping else 1))

    var desc = VatCrowdClipDesc(clip_index, v_cnt, f_cnt, fps, dur, looping)
    var out = BakedVatCrowdClip(desc)

    var n_nodes = len(model.nodes)
    var parents = List[Int](length=n_nodes, fill=-1)
    for ni in range(n_nodes):
        for ch_idx in model.nodes[ni].children:
            if ch_idx >= 0 and ch_idx < n_nodes:
                parents[ch_idx] = ni

    var has_skin = (
        mesh.skin >= 0
        and mesh.skin < len(model.skins)
        and len(mesh.joints) >= v_cnt * 4
        and len(mesh.weights) >= v_cnt * 4
    )
    var f16_one = f32_to_f16_bits(Float32(1.0))
    var f16_zero = f32_to_f16_bits(Float32(0.0))

    for f in range(f_cnt):
        var t = (Float32(f) / Float32(max(1, f_cnt))) * dur if looping else (Float32(f) / Float32(max(1, f_cnt - 1))) * dur

        # Compute node local & global transforms at time `t`
        var locals = List[Mat4](capacity=max(1, n_nodes))
        for ni in range(n_nodes):
            var tr = model.nodes[ni].base_translation
            var rt = model.nodes[ni].base_rotation
            var sc = model.nodes[ni].base_scale
            if has_clip:
                for ch in model.animations[clip_index].channels:
                    if ch.node == ni:
                        if ch.path == 0:
                            tr = _sample_channel_vec3(ch, t, tr)
                        elif ch.path == 1:
                            rt = _sample_channel_quat(ch, t, rt)
                        elif ch.path == 2:
                            sc = _sample_channel_vec3(ch, t, sc)
            locals.append(trs(tr, rt, sc))

        var globals = List[Mat4](length=n_nodes, fill=Mat4())
        var resolved = List[Bool](length=n_nodes, fill=False)
        for ni in range(n_nodes):
            var chain = List[Int]()
            var cur = ni
            while cur >= 0 and cur < n_nodes and not resolved[cur] and len(chain) <= n_nodes:
                chain.append(cur)
                cur = parents[cur]
            for ci in range(len(chain) - 1, -1, -1):
                var idx_n = chain[ci]
                var p_idx = parents[idx_n]
                if p_idx >= 0 and p_idx < n_nodes:
                    globals[idx_n] = globals[p_idx].multiply(locals[idx_n])
                else:
                    globals[idx_n] = locals[idx_n].copy()
                resolved[idx_n] = True

        var skin_mats = List[Mat4]()
        if has_skin:
            ref sk = model.skins[mesh.skin]
            var n_j = len(sk.joints)
            for ji in range(n_j):
                var j_node = sk.joints[ji]
                var g_m = globals[j_node].copy() if (j_node >= 0 and j_node < n_nodes) else Mat4()
                var ibm = Mat4()
                if len(sk.inverse_bind) >= (ji + 1) * 16:
                    for k in range(16):
                        ibm.values[k] = sk.inverse_bind[ji * 16 + k]
                skin_mats.append(g_m.multiply(ibm))

        for v in range(v_cnt):
            var bp = Vec3(mesh.positions[v * 3 + 0], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2])
            var bn = (
                Vec3(mesh.normals[v * 3 + 0], mesh.normals[v * 3 + 1], mesh.normals[v * 3 + 2]).normalized()
                if len(mesh.normals) >= (v + 1) * 3
                else Vec3(0.0, 1.0, 0.0)
            )
            var out_p = bp
            var out_n = bn

            if has_skin and len(skin_mats) > 0:
                var acc_p = Vec3(0.0, 0.0, 0.0)
                var acc_n = Vec3(0.0, 0.0, 0.0)
                var w_sum = Float32(0.0)
                for inf in range(4):
                    var w = mesh.weights[v * 4 + inf]
                    if w > Float32(1e-5):
                        var j_idx = min(len(skin_mats) - 1, max(0, Int(mesh.joints[v * 4 + inf])))
                        ref sm = skin_mats[j_idx]
                        acc_p = acc_p + sm.transform_point(bp) * w
                        var rn = Vec3(
                            sm.values[0] * bn.x + sm.values[4] * bn.y + sm.values[8] * bn.z,
                            sm.values[1] * bn.x + sm.values[5] * bn.y + sm.values[9] * bn.z,
                            sm.values[2] * bn.x + sm.values[6] * bn.y + sm.values[10] * bn.z,
                        )
                        acc_n = acc_n + rn * w
                        w_sum += w
                if w_sum > Float32(1e-5):
                    out_p = acc_p * (Float32(1.0) / w_sum)
                    out_n = acc_n.normalized()

            out.bounds_min = Vec3(min(out.bounds_min.x, out_p.x), min(out.bounds_min.y, out_p.y), min(out.bounds_min.z, out_p.z))
            out.bounds_max = Vec3(max(out.bounds_max.x, out_p.x), max(out.bounds_max.y, out_p.y), max(out.bounds_max.z, out_p.z))

            var dst = (f * v_cnt + v) * 4
            out.pos_rgba16f[dst + 0] = f32_to_f16_bits(out_p.x)
            out.pos_rgba16f[dst + 1] = f32_to_f16_bits(out_p.y)
            out.pos_rgba16f[dst + 2] = f32_to_f16_bits(out_p.z)
            out.pos_rgba16f[dst + 3] = f16_one

            out.nrm_rgba16f[dst + 0] = f32_to_f16_bits(out_n.x)
            out.nrm_rgba16f[dst + 1] = f32_to_f16_bits(out_n.y)
            out.nrm_rgba16f[dst + 2] = f32_to_f16_bits(out_n.z)
            out.nrm_rgba16f[dst + 3] = f16_zero

    return out^


def sample_vat_vertex_cpu(
    baked: BakedVatCrowdClip,
    vertex_idx: Int,
    cur_time_sec: Float32,
    prev_time_sec: Float32,
) -> Tuple[Vec3, Vec3, Vec3]:
    """Interpolates `(cur_pos, cur_nrm, prev_pos)` from a `BakedVatCrowdClip` at arbitrary per-instance phases,
    delivering smooth sub-frame deformation and exact 3D `cur_pos - prev_pos` motion vectors for TSR v3!"""
    if baked.desc.vertex_count <= 0 or baked.desc.frame_count <= 0:
        return (Vec3(), Vec3(0.0, 1.0, 0.0), Vec3())

    var f_cnt = baked.desc.frame_count
    var dur = baked.desc.duration_sec

    var t_cur = cur_time_sec
    var t_prev = prev_time_sec
    if baked.desc.looping:
        t_cur = t_cur - floor(t_cur / dur) * dur
        t_prev = t_prev - floor(t_prev / dur) * dur
        var fc = (t_cur / dur) * Float32(f_cnt)
        var fp = (t_prev / dur) * Float32(f_cnt)
        var c0 = Int(floor(fc)) % f_cnt
        var c1 = (c0 + 1) % f_cnt
        var tc = fc - floor(fc)
        var p0 = Int(floor(fp)) % f_cnt
        var p1 = (p0 + 1) % f_cnt
        var tp = fp - floor(fp)

        var cp = baked.read_pos_frame(c0, vertex_idx) * (Float32(1.0) - tc) + baked.read_pos_frame(c1, vertex_idx) * tc
        var cn = (baked.read_nrm_frame(c0, vertex_idx) * (Float32(1.0) - tc) + baked.read_nrm_frame(c1, vertex_idx) * tc).normalized()
        var pp = baked.read_pos_frame(p0, vertex_idx) * (Float32(1.0) - tp) + baked.read_pos_frame(p1, vertex_idx) * tp
        return (cp, cn, pp)

    var fc_cl = max(Float32(0.0), min(Float32(1.0), t_cur / dur)) * Float32(f_cnt - 1)
    var fp_cl = max(Float32(0.0), min(Float32(1.0), t_prev / dur)) * Float32(f_cnt - 1)
    var c0 = min(f_cnt - 1, Int(floor(fc_cl)))
    var c1 = min(f_cnt - 1, c0 + 1)
    var tc = fc_cl - Float32(c0)
    var p0 = min(f_cnt - 1, Int(floor(fp_cl)))
    var p1 = min(f_cnt - 1, p0 + 1)
    var tp = fp_cl - Float32(p0)
    var cp = baked.read_pos_frame(c0, vertex_idx) * (Float32(1.0) - tc) + baked.read_pos_frame(c1, vertex_idx) * tc
    var cn = (baked.read_nrm_frame(c0, vertex_idx) * (Float32(1.0) - tc) + baked.read_nrm_frame(c1, vertex_idx) * tc).normalized()
    var pp = baked.read_pos_frame(p0, vertex_idx) * (Float32(1.0) - tp) + baked.read_pos_frame(p1, vertex_idx) * tp
    return (cp, cn, pp)


def vat_crowd_eval_shader() -> List[UInt32]:
    """GPU Compute Vertex Animation Texture (`VAT`) Crowd Evaluator (`group = 64 x 1`):
    Samples bindless `RGBA16F` VAT position & normal sheets at `cur_frame_uv` and `prev_frame_uv`
    for `10,000+` independent crowd instances, writing deformed vertices + previous-frame positions!
    Push[0] = vertex_count, Push[1] = frame_count, Push[2] = pos_tex_slot, Push[3] = nrm_tex_slot,
    Push[4] = dst_first_vertex, Push[5] = prev_pos_base_word, Push[6] = cur_norm_time, Push[7] = prev_norm_time.
    """
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=64,
        group_height=1,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
    )
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_skin = s.buffer(BIND_AAA_SKINNING)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)

    var v_id = s.invocation_axis(0)
    var v_cnt = s.push_word(0)
    var pos_slot = s.push_word(2)
    var nrm_slot = s.push_word(3)
    var dst_first_v = s.push_word(4)
    var prev_pos_base = s.push_word(5)
    var cur_v = s.nclamp(s.push_float(6), s.float(0.0), s.float(1.0))
    var prev_v = s.nclamp(s.push_float(7), s.float(0.0), s.float(1.0))
    s.return_if(s.uge(v_id, v_cnt))

    var u_coord = s.fdiv(s.fadd(s.float_from_uint(v_id), s.float(0.5)), s.nmax(s.float(1.0), s.float_from_uint(v_cnt)))
    var cur_pos4 = s.sample_bindless_lod(bindless_arr, pos_slot, s.vec2(u_coord, cur_v), s.float(0.0))
    var cur_nrm4 = s.sample_bindless_lod(bindless_arr, nrm_slot, s.vec2(u_coord, cur_v), s.float(0.0))
    var prev_pos4 = s.sample_bindless_lod(bindless_arr, pos_slot, s.vec2(u_coord, prev_v), s.float(0.0))

    var v_base = s.imul(s.iadd(dst_first_v, v_id), s.uint(12))
    s.store_float(b_verts, v_base, s.extract_f(cur_pos4, 0))
    s.store_float(b_verts, s.iadd(v_base, s.uint(1)), s.extract_f(cur_pos4, 1))
    s.store_float(b_verts, s.iadd(v_base, s.uint(2)), s.extract_f(cur_pos4, 2))
    s.store_float(b_verts, s.iadd(v_base, s.uint(3)), s.extract_f(cur_nrm4, 0))
    s.store_float(b_verts, s.iadd(v_base, s.uint(4)), s.extract_f(cur_nrm4, 1))
    s.store_float(b_verts, s.iadd(v_base, s.uint(5)), s.extract_f(cur_nrm4, 2))

    var p_base = s.iadd(prev_pos_base, s.imul(v_id, s.uint(3)))
    s.store_float(b_skin, p_base, s.extract_f(prev_pos4, 0))
    s.store_float(b_skin, s.iadd(p_base, s.uint(1)), s.extract_f(prev_pos4, 1))
    s.store_float(b_skin, s.iadd(p_base, s.uint(2)), s.extract_f(prev_pos4, 2))
    return s.finish()


def terrain_clipmap_eval_shader() -> List[UInt32]:
    """GPU Compute Toroidal Terrain Clipmap Evaluator (`group = 64 x 1`):
    Reads canonical ring-local `(lx, lz)` from `b_verts[v_base + 0, 2]` and skirt bit `(meta & 4) == 0`,
    samples bindless `RGBA16F` heightmap (or deterministic procedural FBM fallback), applies geomorphing
    near `morph_start_ratio * ring_half_extent_m`, drops outer skirts by `skirt_depth_m`, derives
    splat weights, and writes `(lx, wy, lz)`, normal, tangent, world UV, and splat `meta`!
    Push[0]=vertex_count, [1]=dst_first_vertex, [2]=heightmap_slot, [3]=splat_control_slot,
    [4]=ring_snapped_ox, [5]=ring_snapped_oz, [6]=ring_cell_size_m, [7]=ring_half_extent_m,
    [8]=height_scale_m, [9]=height_offset_m, [10]=morph_start_ratio,
    [11]=world_size_m, [12]=world_origin_x, [13]=world_origin_z, [14]=skirt_depth_m, [15]=grid_dim_per_ring.
    """
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=64,
        group_height=1,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
    )
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)

    var v_id = s.invocation_axis(0)
    var v_cnt = s.push_word(0)
    s.return_if(s.uge(v_id, v_cnt))

    var dst_first_v = s.push_word(1)
    var hm_slot = s.push_word(2)
    var splat_slot = s.push_word(3)
    var ring_ox = s.push_float(4)
    var ring_oz = s.push_float(5)
    var cell_sz = s.nmax(s.float(0.05), s.push_float(6))
    var half_ext = s.nmax(s.float(0.5), s.push_float(7))
    var h_scale = s.push_float(8)
    var h_offset = s.push_float(9)
    var morph_start = s.nclamp(s.push_float(10), s.float(0.1), s.float(0.95))
    var world_sz = s.nmax(s.float(1.0), s.push_float(11))
    var world_ox = s.push_float(12)
    var world_oz = s.push_float(13)
    var skirt_drop = s.push_float(14)

    var v_base = s.imul(s.iadd(dst_first_v, v_id), s.uint(12))
    var lx = s.load_float(b_verts, v_base)
    var lz = s.load_float(b_verts, s.iadd(v_base, s.uint(2)))
    var meta_in = s.load_word(b_verts, s.iadd(v_base, s.uint(11)))
    var is_skirt = s.ieq(s.bit_and(meta_in, s.uint(4)), s.uint(0))

    var wx = s.fadd(ring_ox, lx)
    var wz = s.fadd(ring_oz, lz)
    var inv_wsz = s.fdiv(s.float(1.0), world_sz)
    var hm_u = s.nclamp(s.fmul(s.fsub(wx, world_ox), inv_wsz), s.float(0.0), s.float(1.0))
    var hm_v = s.nclamp(s.fmul(s.fsub(wz, world_oz), inv_wsz), s.float(0.0), s.float(1.0))

    var h0_v = s.fvar(s.float(0.0))
    var dh_dx_v = s.fvar(s.float(0.0))
    var dh_dz_v = s.fvar(s.float(0.0))

    # Geomorphing weight near outer boundary of ring
    var r_cheb = s.nmax(s.fabs(lx), s.fabs(lz))
    var r_norm = s.fdiv(r_cheb, half_ext)
    var morph_w = s.nclamp(
        s.fdiv(s.fsub(r_norm, morph_start), s.nmax(s.float(0.05), s.fsub(s.float(1.0), morph_start))),
        s.float(0.0),
        s.float(1.0),
    )
    var gx_i = s.saturate_uint(s.ffloor(s.fadd(s.fdiv(s.fabs(lx), cell_sz), s.float(0.5))))
    var gz_i = s.saturate_uint(s.ffloor(s.fadd(s.fdiv(s.fabs(lz), cell_sz), s.float(0.5))))
    var odd_x = s.ine(s.bit_and(gx_i, s.uint(1)), s.uint(0))
    var odd_z = s.ine(s.bit_and(gz_i, s.uint(1)), s.uint(0))

    var step_uv = s.nmax(s.float(1.0 / 512.0), s.fmul(cell_sz, inv_wsz))
    var step_m = s.fmul(step_uv, world_sz)

    s.begin_if(s.ult(hm_slot, s.uint(UInt32(AAA_MAX_BINDLESS))))
    var hc = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(hm_u, hm_v), s.float(0.0)), 0)
    var hr = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(s.nmin(s.float(1.0), s.fadd(hm_u, step_uv)), hm_v), s.float(0.0)), 0)
    var hl = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(s.nmax(s.float(0.0), s.fsub(hm_u, step_uv)), hm_v), s.float(0.0)), 0)
    var hu = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(hm_u, s.nmin(s.float(1.0), s.fadd(hm_v, step_uv))), s.float(0.0)), 0)
    var hd = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(hm_u, s.nmax(s.float(0.0), s.fsub(hm_v, step_uv))), s.float(0.0)), 0)

    var du_c = s.fmul(cell_sz, inv_wsz)
    var h_coarse = s.fvar(hc)
    s.begin_if(s.logical_and(odd_x, odd_z))
    var h_c0 = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(s.nmax(s.float(0.0), s.fsub(hm_u, du_c)), s.nmax(s.float(0.0), s.fsub(hm_v, du_c))), s.float(0.0)), 0)
    var h_c1 = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(s.nmin(s.float(1.0), s.fadd(hm_u, du_c)), s.nmin(s.float(1.0), s.fadd(hm_v, du_c))), s.float(0.0)), 0)
    s.store(h_coarse, s.fmul(s.fadd(h_c0, h_c1), s.float(0.5)))
    s.begin_else()
    s.begin_if(odd_x)
    var h_x0 = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(s.nmax(s.float(0.0), s.fsub(hm_u, du_c)), hm_v), s.float(0.0)), 0)
    var h_x1 = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(s.nmin(s.float(1.0), s.fadd(hm_u, du_c)), hm_v), s.float(0.0)), 0)
    s.store(h_coarse, s.fmul(s.fadd(h_x0, h_x1), s.float(0.5)))
    s.begin_else()
    s.begin_if(odd_z)
    var h_z0 = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(hm_u, s.nmax(s.float(0.0), s.fsub(hm_v, du_c))), s.float(0.0)), 0)
    var h_z1 = s.extract_f(s.sample_bindless_lod(bindless_arr, hm_slot, s.vec2(hm_u, s.nmin(s.float(1.0), s.fadd(hm_v, du_c))), s.float(0.0)), 0)
    s.store(h_coarse, s.fmul(s.fadd(h_z0, h_z1), s.float(0.5)))
    s.end_if()
    s.end_if()
    s.end_if()

    var h_morphed = s.fma(s.fsub(s.load_f(h_coarse), hc), morph_w, hc)
    s.store(h0_v, h_morphed)
    var inv_2s = s.fdiv(h_scale, s.fmul(s.float(2.0), step_m))
    s.store(dh_dx_v, s.fmul(s.fsub(hr, hl), inv_2s))
    s.store(dh_dz_v, s.fmul(s.fsub(hu, hd), inv_2s))

    s.begin_else()
    # Deterministic procedural terrain height fallback
    var kx = s.float(0.045)
    var kz = s.float(0.038)
    var px_p = s.fmul(wx, kx)
    var pz_p = s.fmul(wz, kz)
    var s1 = s.fsin(px_p)
    var c1 = s.fcos(pz_p)
    var h_proc = s.fma(s.fmul(s1, c1), s.float(0.45), s.float(0.50))
    s.store(h0_v, h_proc)
    s.store(dh_dx_v, s.fmul(h_scale, s.fmul(s.float(0.45), s.fmul(kx, s.fmul(s.fcos(px_p), c1)))))
    s.store(dh_dz_v, s.fmul(h_scale, s.fmul(s.float(-0.45), s.fmul(kz, s.fmul(s1, s.fsin(pz_p))))))
    s.end_if()

    var h_norm = s.load_f(h0_v)
    var wy_base = s.fma(h_norm, h_scale, h_offset)
    var wy = s.select_f(is_skirt, s.fsub(wy_base, skirt_drop), wy_base)

    var n_u = emit_vec3_normalize(s, s.fneg(s.load_f(dh_dx_v)), s.float(1.0), s.fneg(s.load_f(dh_dz_v)))
    var nx = n_u[0]; var ny = n_u[1]; var nz = n_u[2]
    # Tangent orthogonal to normal: normalize(1 - nx*nx, -nx*ny, -nx*nz)
    var t_u = emit_vec3_normalize(s, s.fsub(s.float(1.0), s.fmul(nx, nx)), s.fneg(s.fmul(nx, ny)), s.fneg(s.fmul(nx, nz)))

    # Splat weights `(w_sand, w_rock, w_cobb)` packed into bits 8..31 of `meta`
    var w_sand_v = s.fvar(s.float(0.0))
    var w_rock_v = s.fvar(s.float(0.0))
    var w_cobb_v = s.fvar(s.float(0.0))
    s.begin_if(s.ult(splat_slot, s.uint(UInt32(AAA_MAX_BINDLESS))))
    var sp = s.sample_bindless_lod(bindless_arr, splat_slot, s.vec2(hm_u, hm_v), s.float(0.0))
    var sp0 = s.nmax(s.float(0.0), s.extract_f(sp, 0))
    var sp1 = s.nmax(s.float(0.0), s.extract_f(sp, 1))
    var sp2 = s.nmax(s.float(0.0), s.extract_f(sp, 2))
    var sp3 = s.nmax(s.float(0.0), s.extract_f(sp, 3))
    var sp_sum = s.fadd(s.fadd(sp0, sp1), s.fadd(sp2, sp3))
    var has_sp = s.fgt(sp_sum, s.float(1e-4))
    var inv_sp = s.select_f(has_sp, s.fdiv(s.float(1.0), s.nmax(s.float(1e-4), sp_sum)), s.float(0.0))
    s.store(w_sand_v, s.nclamp(s.fmul(sp1, inv_sp), s.float(0.0), s.float(1.0)))
    s.store(w_rock_v, s.nclamp(s.fmul(sp2, inv_sp), s.float(0.0), s.float(1.0)))
    s.store(w_cobb_v, s.nclamp(s.fmul(sp3, inv_sp), s.float(0.0), s.float(1.0)))
    s.begin_else()
    var slope = s.nclamp(s.fsub(s.float(1.0), ny), s.float(0.0), s.float(1.0))
    var w_rk = s.nclamp(s.fmul(slope, s.float(3.0)), s.float(0.0), s.float(1.0))
    var w_sd = s.fmul(s.nclamp(s.fsub(s.float(1.0), s.fmul(s.fabs(s.fsub(h_norm, s.float(0.15))), s.float(4.0))), s.float(0.0), s.float(1.0)), s.fsub(s.float(1.0), w_rk))
    var w_cb = s.fmul(s.nclamp(s.fmul(s.fsub(h_norm, s.float(0.75)), s.float(3.0)), s.float(0.0), s.float(1.0)), s.fsub(s.float(1.0), w_rk))
    s.store(w_sand_v, w_sd)
    s.store(w_rock_v, w_rk)
    s.store(w_cobb_v, w_cb)
    s.end_if()

    var w1_u = s.umin(s.uint(255), s.saturate_uint(s.fma(s.load_f(w_sand_v), s.float(255.0), s.float(0.5))))
    var w2_u = s.umin(s.uint(255), s.saturate_uint(s.fma(s.load_f(w_rock_v), s.float(255.0), s.float(0.5))))
    var w3_u = s.umin(s.uint(255), s.saturate_uint(s.fma(s.load_f(w_cobb_v), s.float(255.0), s.float(0.5))))
    var meta_out = s.bit_or(
        s.bit_and(meta_in, s.uint(0xFF)),
        s.bit_or(s.shl(w1_u, s.uint(8)), s.bit_or(s.shl(w2_u, s.uint(16)), s.shl(w3_u, s.uint(24)))),
    )

    s.store_float(b_verts, s.iadd(v_base, s.uint(1)), wy)
    s.store_float(b_verts, s.iadd(v_base, s.uint(3)), nx)
    s.store_float(b_verts, s.iadd(v_base, s.uint(4)), ny)
    s.store_float(b_verts, s.iadd(v_base, s.uint(5)), nz)
    s.store_float(b_verts, s.iadd(v_base, s.uint(6)), t_u[0])
    s.store_float(b_verts, s.iadd(v_base, s.uint(7)), t_u[1])
    s.store_float(b_verts, s.iadd(v_base, s.uint(8)), t_u[2])
    s.store_float(b_verts, s.iadd(v_base, s.uint(9)), s.fmul(wx, s.float(0.25)))
    s.store_float(b_verts, s.iadd(v_base, s.uint(10)), s.fmul(wz, s.float(0.25)))
    s.store_word(b_verts, s.iadd(v_base, s.uint(11)), meta_out)
    return s.finish()
