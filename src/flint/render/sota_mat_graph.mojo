"""Area I round 3: layer outputs for the material graph, and real `OpSpecConstant` specialisation.

`SotaLayeredMaterialNode` extends `SurfaceMaterialNode` with `evaluate_layers` (anisotropy, its rotation and the
transmission weight per texel). `compile_sota_layered_material_shader` emits the surface outputs plus the layer
outputs (12 words per texel) and three specialisation constants that the pipeline sets through
`VkSpecializationInfo` (`create_aaa_compute_pipeline_spec`):

  SpecId 0  enable_clearcoat (bool, default true)   false: coat output 0
  SpecId 1  domain override (uint, default 0)       MAT_DOMAIN_UNLIT_UI: rough 1, metal 0, coat 0, no layers;
                                                    MAT_DOMAIN_CLEARCOAT: coat >= 1;
                                                    MAT_DOMAIN_TWO_SIDED_FOLIAGE: transmission >= 0.35
  SpecId 2  enable_layers (bool, default true)      false: layer outputs 0 (the driver drops the layer code)

`sota_mat_bake_layered_graph` dispatches it on the renderer, bakes albedo + ORM (as the legacy bake does) and a
layer mask texture (R anisotropy, G rotation / pi, B transmission, A coat multiplier 1), and attaches the mask to the
material's extension words (word 9) with multiplier constants 1, so the deferred pass reads the graph's layers
per pixel when the layered model is on. The legacy material words are written exactly as the legacy bake does.
"""

from std.collections import List
from std.ffi import external_call
from std.math import max, min, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.platform.vulkan import ACCESS_HOST_READ, ACCESS_SHADER_WRITE, PIPELINE_COMPUTE, PIPELINE_HOST
from flint.render.spirv import MODEL_GL_COMPUTE
from flint.render.shader_dsl import (
    BindlessTex2D,
    BufferRef,
    GpuBool,
    GpuF32,
    GpuU32,
    ShaderDsl,
    abs_f,
    clamp_f,
    max_f,
    select_f,
)
from flint.render.aaa_visbuffer import AAA_MAX_BINDLESS, AAA_PUSH_WORDS, BIND_AAA_BINDLESS_TEX, BIND_AAA_MATERIALS, BIND_AAA_POST_SCRATCH
from flint.render.aaa_post_tsr import AAA_POST_BLOOM_BASE
from flint.render.material_graph import (
    MAT_DOMAIN_CLEARCOAT,
    MAT_DOMAIN_TWO_SIDED_FOLIAGE,
    MAT_DOMAIN_UNLIT_UI,
    MATERIAL_ASSET_WORDS,
    StandardPbrMaterialNode,
    SurfaceMaterialInputs,
    SurfaceMaterialNode,
    SurfaceMaterialOutputs,
)
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.sota_mat_core import SotaMatLayers

comptime SOTA_MAT_GRAPH_SPEC_CLEARCOAT: UInt32 = 0
comptime SOTA_MAT_GRAPH_SPEC_DOMAIN: UInt32 = 1
comptime SOTA_MAT_GRAPH_SPEC_LAYERS: UInt32 = 2
comptime SOTA_MAT_GRAPH_TEXEL_WORDS: Int = 12
comptime _PI: Float32 = 3.14159265


@fieldwise_init
struct SotaMatLayerOutputs(Copyable, ImplicitlyCopyable, Movable):
    var anisotropy: GpuF32           # 0..1
    var anisotropy_rotation: GpuF32  # radians, 0..pi (the stretch axis is a line, rotation mod pi)
    var transmission: GpuF32         # 0..1


trait SotaLayeredMaterialNode(SurfaceMaterialNode):
    """A surface node that also drives the layered model's per-texel layers."""
    def evaluate_layers(
        self,
        mut dsl: ShaderDsl,
        inputs: SurfaceMaterialInputs,
        surface: SurfaceMaterialOutputs,
        b_mats: BufferRef[MATERIAL_ASSET_WORDS],
        bindless: BindlessTex2D,
    ) -> SotaMatLayerOutputs:
        ...


struct SotaMatGraphSpec(Copyable, Movable):
    """Values for the three specialisation constants (see the module doc)."""
    var enable_clearcoat: Bool
    var domain: UInt32
    var enable_layers: Bool

    def __init__(out self, enable_clearcoat: Bool = True, domain: UInt32 = 0, enable_layers: Bool = True):
        self.enable_clearcoat = enable_clearcoat
        self.domain = domain
        self.enable_layers = enable_layers

    def ids(self) -> List[UInt32]:
        return [SOTA_MAT_GRAPH_SPEC_CLEARCOAT, SOTA_MAT_GRAPH_SPEC_DOMAIN, SOTA_MAT_GRAPH_SPEC_LAYERS]

    def values(self) -> List[UInt32]:
        return [UInt32(1) if self.enable_clearcoat else UInt32(0), self.domain, UInt32(1) if self.enable_layers else UInt32(0)]


struct BrushedSpunMetalNode(SotaLayeredMaterialNode):
    """Example graph: the standard PBR node for the surface, plus a spun-metal layer pattern (anisotropy stretched
    tangentially around the UV centre, as on a lathe-turned disc), a brushed band where |v - 0.5| > `band`, and
    an optional transmissive window (radius < `window_r`)."""
    var strength: Float32
    var band: Float32
    var window_r: Float32

    def __init__(out self, strength: Float32 = 0.85, band: Float32 = 0.42, window_r: Float32 = 0.0):
        self.strength = strength
        self.band = band
        self.window_r = window_r

    def evaluate_surface(
        self,
        mut dsl: ShaderDsl,
        inputs: SurfaceMaterialInputs,
        b_mats: BufferRef[MATERIAL_ASSET_WORDS],
        bindless: BindlessTex2D,
    ) -> SurfaceMaterialOutputs:
        var std_node = StandardPbrMaterialNode(enable_detail_layer=False)
        return std_node.evaluate_surface(dsl, inputs, b_mats, bindless)

    def evaluate_layers(
        self,
        mut dsl: ShaderDsl,
        inputs: SurfaceMaterialInputs,
        surface: SurfaceMaterialOutputs,
        b_mats: BufferRef[MATERIAL_ASSET_WORDS],
        bindless: BindlessTex2D,
    ) -> SotaMatLayerOutputs:
        var du = inputs.uv.x() - Float32(0.5)
        var dv = inputs.uv.y() - Float32(0.5)
        var r = dsl.vec3(du, dv, dsl.f32(0.0)).length()
        # Tangential direction around the centre: angle(du, dv) + pi/2, folded into 0..pi
        var m = dsl.raw_module()
        var ang = GpuF32(dsl.mod_addr, m[].fatan2(dv.id, du.id))
        var rot_t = ang + Float32(0.5 * _PI)
        rot_t = select_f(rot_t.lt(0.0), rot_t + _PI, rot_t)
        rot_t = select_f(rot_t.ge(_PI), rot_t - _PI, rot_t)
        var in_band = abs_f(dv).gt(self.band)
        var rot = select_f(in_band, dsl.f32(0.0), rot_t)
        var an = select_f(r.lt(0.02), dsl.f32(0.0), dsl.f32(self.strength))
        var tr = select_f(r.lt(self.window_r), dsl.f32(1.0), dsl.f32(0.0))
        return SotaMatLayerOutputs(an, rot, tr)


def compile_sota_layered_material_shader[NodeT: SotaLayeredMaterialNode](node: NodeT) -> List[UInt32]:
    """Surface + layer outputs, 12 words per texel: 0..2 colour + emissive, 3 roughness, 4 metallic, 5 coat,
    6 opacity, 7 |n_ts.z|, 8 anisotropy, 9 rotation / pi, 10 transmission, 11 coat multiplier (1)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS, enable_bindless=True)
    var spec_cc = GpuBool(dsl.mod_addr, dsl.raw_module()[].spec_constant_bool(Int(SOTA_MAT_GRAPH_SPEC_CLEARCOAT), True))
    var spec_dom = GpuU32(dsl.mod_addr, dsl.raw_module()[].spec_constant_u32(Int(SOTA_MAT_GRAPH_SPEC_DOMAIN), 0))
    var spec_lay = GpuBool(dsl.mod_addr, dsl.raw_module()[].spec_constant_bool(Int(SOTA_MAT_GRAPH_SPEC_LAYERS), True))

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
    var an = dsl.var_f32(dsl.f32(0.0))
    var rot = dsl.var_f32(dsl.f32(0.0))
    var tr = dsl.var_f32(dsl.f32(0.0))
    var unlit = spec_dom.eq(MAT_DOMAIN_UNLIT_UI)
    dsl.if_then(spec_lay & unlit.not_())
    var lay = node.evaluate_layers(dsl, inputs, res, b_mats, bindless)
    an.set(clamp_f(lay.anisotropy, 0.0, 1.0))
    rot.set(clamp_f(lay.anisotropy_rotation * Float32(1.0 / _PI), 0.0, 1.0))
    tr.set(clamp_f(lay.transmission, 0.0, 1.0))
    dsl.end_if()
    tr.set(select_f(spec_dom.eq(MAT_DOMAIN_TWO_SIDED_FOLIAGE), max_f(tr.get(), 0.35), tr.get()))
    var cc = dsl.var_f32(dsl.f32(0.0))
    dsl.if_then(spec_cc)
    cc.set(select_f(spec_dom.eq(MAT_DOMAIN_CLEARCOAT), max_f(res.clearcoat, 1.0), res.clearcoat))
    dsl.end_if()
    var rough = select_f(unlit, dsl.f32(1.0), res.roughness)
    var metal = select_f(unlit, dsl.f32(0.0), res.metallic)
    var cc_o = select_f(unlit, dsl.f32(0.0), cc.get())

    var o = out_base + (gy * w_u + gx) * UInt32(SOTA_MAT_GRAPH_TEXEL_WORDS)
    b_out.store_vec3(o, res.base_color + res.emissive)
    b_out.store_f32(o + UInt32(3), rough)
    b_out.store_f32(o + UInt32(4), metal)
    b_out.store_f32(o + UInt32(5), cc_o)
    b_out.store_f32(o + UInt32(6), res.opacity)
    b_out.store_f32(o + UInt32(7), abs_f(res.normal_ts.z()))
    b_out.store_f32(o + UInt32(8), an.get())
    b_out.store_f32(o + UInt32(9), rot.get())
    b_out.store_f32(o + UInt32(10), tr.get())
    b_out.store_f32(o + UInt32(11), dsl.f32(1.0))
    return dsl.finish()


struct SotaMatGraphBake(Copyable, Movable):
    var albedo_slot: Int
    var orm_slot: Int
    var layer_slot: Int
    var texels: List[Float32]   # the GPU read-back, 12 words per texel
    var dim: Int
    var max_aniso: Float32
    var max_trans: Float32

    def __init__(out self):
        self.albedo_slot = -1
        self.orm_slot = -1
        self.layer_slot = -1
        self.texels = List[Float32]()
        self.dim = 0
        self.max_aniso = Float32(0.0)
        self.max_trans = Float32(0.0)


def sota_mat_graph_eval[NodeT: SotaLayeredMaterialNode](
    mut r: FlintAaaVulkanRenderer,
    node: NodeT,
    mat_id: Int,
    dim: Int = 32,
    spec: SotaMatGraphSpec = SotaMatGraphSpec(),
    time_sec: Float32 = 0.0,
    world_y_scale: Float32 = 24.0,
) -> SotaMatGraphBake:
    """Compiles, specialises and dispatches the layered graph; returns the read-back texels (no upload)."""
    var out = SotaMatGraphBake()
    if not r.active or mat_id < 0 or mat_id >= len(r.materials):
        return out^
    var tex_dim = max(8, min(256, (dim + 3) & ~3))
    var spv = compile_sota_layered_material_shader(node)
    var pipe = r.vk.create_aaa_compute_pipeline_spec(r.pipe_layout, spv, spec.ids(), spec.values())
    if pipe == 0:
        return out^
    var off = AAA_POST_BLOOM_BASE
    var words = tex_dim * tex_dim * SOTA_MAT_GRAPH_TEXEL_WORDS
    if (off + words) * 4 > r.buf_post_scratch.size or not r.vk.begin_commands(0):
        r.vk.destroy_pipeline(pipe)
        return out^
    var push = List[UInt32](length=16, fill=0)
    push[0] = UInt32(tex_dim)
    push[1] = UInt32(tex_dim)
    push[2] = UInt32(mat_id)
    push[3] = bitcast[DType.uint32](time_sec)
    push[4] = UInt32(off)
    push[5] = bitcast[DType.uint32](world_y_scale)
    r.vk.dispatch_aaa(0, pipe, r.pipe_layout, r.desc_sets[0], push, (tex_dim + 7) // 8, (tex_dim + 7) // 8)
    r.vk.memory_barrier2(0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
    var f = r.vk.submit(0)
    if f != 0:
        _ = r.vk.wait(f)
    r.vk.destroy_pipeline(pipe)
    if f == 0:
        return out^
    out.texels = List[Float32](length=words, fill=0.0)
    _ = external_call["memcpy", Int](Int(out.texels.unsafe_ptr()), r.buf_post_scratch.cpu_address + off * 4, words * 4)
    out.dim = tex_dim
    for p in range(tex_dim * tex_dim):
        out.max_aniso = max(out.max_aniso, out.texels[p * SOTA_MAT_GRAPH_TEXEL_WORDS + 8])
        out.max_trans = max(out.max_trans, out.texels[p * SOTA_MAT_GRAPH_TEXEL_WORDS + 10])
    return out^


def _u8(x: Float32) -> UInt32:
    return UInt32(min(255, max(0, Int(max(Float32(0.0), min(Float32(1.0), x)) * Float32(255.0) + Float32(0.5)))))


def sota_mat_bake_layered_graph[NodeT: SotaLayeredMaterialNode](
    mut r: FlintAaaVulkanRenderer,
    node: NodeT,
    mat_id: Int,
    dim: Int = 32,
    spec: SotaMatGraphSpec = SotaMatGraphSpec(),
    time_sec: Float32 = 0.0,
    world_y_scale: Float32 = 24.0,
) -> SotaMatGraphBake:
    """Evaluates the graph, uploads albedo (sRGB) + ORM + the layer mask (linear), points `mat_id` at them and
    attaches the layer mask to its extension words (the legacy material words get what the legacy bake writes)."""
    var out = sota_mat_graph_eval(r, node, mat_id, dim, spec, time_sec, world_y_scale)
    if out.dim == 0:
        return out^
    var n = out.dim * out.dim
    var alb = List[UInt32](length=n, fill=0xFFFFFFFF)
    var orm = List[UInt32](length=n, fill=0xFFFFFFFF)
    var lay = List[UInt32](length=n, fill=0xFFFFFFFF)
    for p in range(n):
        var b = p * SOTA_MAT_GRAPH_TEXEL_WORDS
        var rs = _u8(sqrt(max(Float32(0.0), out.texels[b + 0])))
        var gs = _u8(sqrt(max(Float32(0.0), out.texels[b + 1])))
        var bs = _u8(sqrt(max(Float32(0.0), out.texels[b + 2])))
        alb[p] = (_u8(out.texels[b + 6]) << 24) | (rs << 16) | (gs << 8) | bs
        var rgh8 = max(UInt32(10), _u8(max(Float32(0.04), out.texels[b + 3])))
        orm[p] = UInt32(0xFF000000) | (UInt32(255) << 16) | (rgh8 << 8) | _u8(out.texels[b + 4])
        # ARGB: A = coat multiplier, R = anisotropy, G = rotation / pi, B = transmission
        lay[p] = (_u8(out.texels[b + 11]) << 24) | (_u8(out.texels[b + 8]) << 16) | (_u8(out.texels[b + 9]) << 8) | _u8(out.texels[b + 10])
    out.albedo_slot = r.upload_bc7_texture(alb, out.dim, out.dim, srgb=True, generate_mips=True)
    out.orm_slot = r.upload_bc7_texture(orm, out.dim, out.dim, srgb=False, generate_mips=True)
    out.layer_slot = r.upload_bc7_texture(lay, out.dim, out.dim, srgb=False, generate_mips=False)  # rotation wraps: no mips
    if out.albedo_slot >= 0 and out.orm_slot >= 0:
        var upd = r.materials[mat_id].copy()
        upd.base_color = Vec3(1.0, 1.0, 1.0)
        upd.roughness = Float32(1.0)
        upd.metallic = Float32(1.0)
        upd.albedo_tex = out.albedo_slot
        upd.orm_tex = out.orm_slot
        r.update_material(mat_id, upd)
    if out.layer_slot >= 0:
        var layers = SotaMatLayers()
        layers.anisotropy = Float32(1.0) if out.max_aniso > Float32(0.0) else Float32(0.0)
        layers.transmission = Float32(1.0) if out.max_trans > Float32(0.0) else Float32(0.0)
        layers.layer_tex = out.layer_slot
        r.sota_mat_set_layers(mat_id, layers)
    return out^
