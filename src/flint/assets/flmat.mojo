"""Standalone `.flmat` Shared Material Asset Codec (`flint.assets.flmat`)."""
from std.math import isfinite, max, min
from flint.assets.asset_ref import AssetRef
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote
from flint.core.math import Vec3
from flint.platform.os import read_bytes, write_bytes
from flint.render.aaa_npr_cel import AaaCelParams
from flint.render.aaa_npr_vfx import AaaStylizedVfxParams

comptime MAT_SHADING_PBR: Int = 0
comptime MAT_SHADING_NPR_CEL: Int = 1
comptime MAT_SHADING_STYLIZED_VFX: Int = 2


@always_inline
def _f32_s(v: Float32) -> String:
    return String(Float64(v)) if isfinite(v) else String("0.0")


def _vec3_json(v: Vec3) -> String:
    return "[" + _f32_s(v.x) + ", " + _f32_s(v.y) + ", " + _f32_s(v.z) + "]"


def _read_vec3_json(doc: JsonDocument, node: Int, fallback: Vec3 = Vec3()) -> Vec3:
    if doc.array_size(node) < 3:
        return fallback
    var x = Float32(doc.number(doc.at(node, 0), Float64(fallback.x)))
    var y = Float32(doc.number(doc.at(node, 1), Float64(fallback.y)))
    var z = Float32(doc.number(doc.at(node, 2), Float64(fallback.z)))
    return Vec3(
        x if isfinite(x) else fallback.x,
        y if isfinite(y) else fallback.y,
        z if isfinite(z) else fallback.z,
    )


def _read_f32(doc: JsonDocument, node: Int, key: String, fallback: Float32) -> Float32:
    var c = doc.object_get(node, key)
    if c < 0:
        return fallback
    var v = Float32(doc.number(c, Float64(fallback)))
    return v if isfinite(v) else fallback


struct FlmatAsset(Copyable, Movable):
    """Shared material asset serialized to/from `.flmat` JSON."""
    var name: String
    var shading_mode: Int
    var base_color: Vec3
    var roughness: Float32
    var metallic: Float32
    var clearcoat: Float32
    var clearcoat_roughness: Float32
    var opacity: Float32
    var alpha_cutoff: Float32
    var sheen: Float32
    var subsurface: Float32
    var emissive: Vec3
    var albedo_tex: AssetRef
    var normal_tex: AssetRef
    var orm_tex: AssetRef
    var emissive_tex: AssetRef
    var cel_params: AaaCelParams
    var vfx_params: AaaStylizedVfxParams

    def __init__(out self, name: String = ""):
        self.name = name
        self.shading_mode = MAT_SHADING_PBR
        self.base_color = Vec3(0.8, 0.8, 0.8)
        self.roughness = Float32(0.5)
        self.metallic = Float32(0.0)
        self.clearcoat = Float32(0.0)
        self.clearcoat_roughness = Float32(0.08)
        self.opacity = Float32(1.0)
        self.alpha_cutoff = Float32(0.0)
        self.sheen = Float32(0.0)
        self.subsurface = Float32(0.0)
        self.emissive = Vec3(0.0, 0.0, 0.0)
        self.albedo_tex = AssetRef()
        self.normal_tex = AssetRef()
        self.orm_tex = AssetRef()
        self.emissive_tex = AssetRef()
        self.cel_params = AaaCelParams()
        self.vfx_params = AaaStylizedVfxParams()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.shading_mode = copy.shading_mode
        self.base_color = copy.base_color
        self.roughness = copy.roughness
        self.metallic = copy.metallic
        self.clearcoat = copy.clearcoat
        self.clearcoat_roughness = copy.clearcoat_roughness
        self.opacity = copy.opacity
        self.alpha_cutoff = copy.alpha_cutoff
        self.sheen = copy.sheen
        self.subsurface = copy.subsurface
        self.emissive = copy.emissive
        self.albedo_tex = copy.albedo_tex.copy()
        self.normal_tex = copy.normal_tex.copy()
        self.orm_tex = copy.orm_tex.copy()
        self.emissive_tex = copy.emissive_tex.copy()
        self.cel_params = copy.cel_params.copy()
        self.vfx_params = copy.vfx_params.copy()

    def to_json(self) -> String:
        ref cp = self.cel_params
        ref vp = self.vfx_params
        var out = String("{\n")
        out += '  "format": "flmat",\n'
        out += '  "version": 1,\n'
        out += '  "name": ' + json_quote(self.name) + ",\n"
        out += '  "shading_mode": ' + String(self.shading_mode) + ",\n"
        out += '  "base_color": ' + _vec3_json(self.base_color) + ",\n"
        out += '  "roughness": ' + _f32_s(self.roughness) + ",\n"
        out += '  "metallic": ' + _f32_s(self.metallic) + ",\n"
        out += '  "clearcoat": ' + _f32_s(self.clearcoat) + ",\n"
        out += '  "clearcoat_roughness": ' + _f32_s(self.clearcoat_roughness) + ",\n"
        out += '  "opacity": ' + _f32_s(self.opacity) + ",\n"
        out += '  "alpha_cutoff": ' + _f32_s(self.alpha_cutoff) + ",\n"
        out += '  "sheen": ' + _f32_s(self.sheen) + ",\n"
        out += '  "subsurface": ' + _f32_s(self.subsurface) + ",\n"
        out += '  "emissive": ' + _vec3_json(self.emissive) + ",\n"
        out += '  "albedo_tex": ' + self.albedo_tex.write_json() + ",\n"
        out += '  "normal_tex": ' + self.normal_tex.write_json() + ",\n"
        out += '  "orm_tex": ' + self.orm_tex.write_json() + ",\n"
        out += '  "emissive_tex": ' + self.emissive_tex.write_json() + ",\n"
        out += '  "cel_params": {'
        out += '"sss_shadow_tint": ' + _vec3_json(cp.sss_shadow_tint)
        out += ', "shadow_threshold": ' + _f32_s(cp.shadow_threshold)
        out += ', "shadow_feather": ' + _f32_s(cp.shadow_feather)
        out += ', "rim_color": ' + _vec3_json(cp.rim_color)
        out += ', "rim_power": ' + _f32_s(cp.rim_power)
        out += ', "rim_threshold": ' + _f32_s(cp.rim_threshold)
        out += ', "spec_intensity": ' + _f32_s(cp.spec_intensity)
        out += ', "spec_step_threshold": ' + _f32_s(cp.spec_step_threshold)
        out += ', "hair_halo_strength": ' + _f32_s(cp.hair_halo_strength)
        out += ', "hair_shift": ' + _f32_s(cp.hair_shift)
        out += ', "outline_width_px": ' + _f32_s(cp.outline_width_px)
        out += ', "outline_color": ' + _vec3_json(cp.outline_color)
        out += ', "outline_z_bias": ' + _f32_s(cp.outline_z_bias)
        out += ', "outline_fov_dist_cap": ' + _f32_s(cp.outline_fov_dist_cap)
        out += ', "inner_ink_strength": ' + _f32_s(cp.inner_ink_strength)
        out += ', "ambient_floor": ' + _f32_s(cp.ambient_floor)
        out += ', "use_vertex_color_channels": ' + ("true" if cp.use_vertex_color_channels else "false")
        out += ', "receive_scene_shadows": ' + ("true" if cp.receive_scene_shadows else "false")
        out += ', "use_custom_light_dir": ' + ("true" if cp.use_custom_light_dir else "false")
        out += ', "custom_light_dir": ' + _vec3_json(cp.custom_light_dir)
        out += ', "custom_light_intensity": ' + _f32_s(cp.custom_light_intensity)
        out += "},\n"
        out += '  "vfx_params": {'
        out += '"uv_scroll_u": ' + _f32_s(vp.uv_scroll_u)
        out += ', "uv_scroll_v": ' + _f32_s(vp.uv_scroll_v)
        out += ', "warp_scale": ' + _f32_s(vp.warp_scale)
        out += ', "warp_strength": ' + _f32_s(vp.warp_strength)
        out += ', "erosion": ' + _f32_s(vp.erosion)
        out += ', "edge_band_width": ' + _f32_s(vp.edge_band_width)
        out += ', "core_color": ' + _vec3_json(vp.core_color)
        out += ', "edge_color": ' + _vec3_json(vp.edge_color)
        out += ', "ink_rim_width": ' + _f32_s(vp.ink_rim_width)
        out += ', "ink_rim_darkness": ' + _f32_s(vp.ink_rim_darkness)
        out += "}\n"
        out += "}\n"
        return out^


def save_flmat_file(path: String, mat: FlmatAsset) -> Bool:
    var txt = mat.to_json()
    var raw = List[UInt8](capacity=txt.byte_length())
    for b in txt.as_bytes():
        raw.append(b)
    return write_bytes(path, raw)


def load_flmat_bytes(data: List[UInt8]) raises -> FlmatAsset:
    if len(data) == 0:
        raise Error("empty .flmat payload")
    var doc = parse_json(data)
    if not doc.valid or doc.root < 0 or doc.nodes[doc.root].kind != 5:
        raise Error("invalid .flmat JSON document")

    var fmt_node = doc.object_get(doc.root, "format")
    if fmt_node >= 0:
        var fmt = doc.text(fmt_node, "")
        if fmt.byte_length() > 0 and fmt != "flmat":
            raise Error("unexpected .flmat format: " + fmt)

    var out = FlmatAsset(doc.text(doc.object_get(doc.root, "name"), ""))
    var sm_node = doc.object_get(doc.root, "shading_mode")
    if sm_node >= 0:
        if doc.nodes[sm_node].kind == 3:
            var sm_str = doc.text(sm_node, "")
            if sm_str == "npr_cel":
                out.shading_mode = MAT_SHADING_NPR_CEL
            elif sm_str == "stylized_vfx":
                out.shading_mode = MAT_SHADING_STYLIZED_VFX
            else:
                out.shading_mode = MAT_SHADING_PBR
        elif doc.nodes[sm_node].kind == 2:
            out.shading_mode = doc.integer(sm_node, MAT_SHADING_PBR)
        else:
            raise Error("invalid shading_mode in .flmat")

    out.base_color = _read_vec3_json(doc, doc.object_get(doc.root, "base_color"), out.base_color)
    out.roughness = _read_f32(doc, doc.root, "roughness", out.roughness)
    out.metallic = _read_f32(doc, doc.root, "metallic", out.metallic)
    out.clearcoat = _read_f32(doc, doc.root, "clearcoat", out.clearcoat)
    out.clearcoat_roughness = _read_f32(doc, doc.root, "clearcoat_roughness", out.clearcoat_roughness)
    out.opacity = _read_f32(doc, doc.root, "opacity", out.opacity)
    out.alpha_cutoff = _read_f32(doc, doc.root, "alpha_cutoff", out.alpha_cutoff)
    out.sheen = _read_f32(doc, doc.root, "sheen", out.sheen)
    out.subsurface = _read_f32(doc, doc.root, "subsurface", out.subsurface)
    out.emissive = _read_vec3_json(doc, doc.object_get(doc.root, "emissive"), out.emissive)

    var tex_node = doc.object_get(doc.root, "textures")
    if tex_node >= 0:
        if doc.nodes[tex_node].kind != 5:
            raise Error("invalid textures object in .flmat")
        var t_alb = doc.object_get(tex_node, "albedo")
        if t_alb >= 0:
            out.albedo_tex = AssetRef.read_json(doc, t_alb)
        var t_nrm = doc.object_get(tex_node, "normal")
        if t_nrm >= 0:
            out.normal_tex = AssetRef.read_json(doc, t_nrm)
        var t_orm = doc.object_get(tex_node, "orm")
        if t_orm >= 0:
            out.orm_tex = AssetRef.read_json(doc, t_orm)
        var t_emi = doc.object_get(tex_node, "emissive")
        if t_emi >= 0:
            out.emissive_tex = AssetRef.read_json(doc, t_emi)

    var alb_node = doc.object_get(doc.root, "albedo_tex")
    if alb_node >= 0:
        out.albedo_tex = AssetRef.read_json(doc, alb_node)
    var nrm_node = doc.object_get(doc.root, "normal_tex")
    if nrm_node >= 0:
        out.normal_tex = AssetRef.read_json(doc, nrm_node)
    var orm_node = doc.object_get(doc.root, "orm_tex")
    if orm_node >= 0:
        out.orm_tex = AssetRef.read_json(doc, orm_node)
    var emi_node = doc.object_get(doc.root, "emissive_tex")
    if emi_node >= 0:
        out.emissive_tex = AssetRef.read_json(doc, emi_node)

    var cel_node = doc.object_get(doc.root, "cel_params")
    if cel_node >= 0:
        if doc.nodes[cel_node].kind != 5:
            raise Error("invalid cel_params object in .flmat")
        out.cel_params = AaaCelParams(
            sss_shadow_tint=_read_vec3_json(doc, doc.object_get(cel_node, "sss_shadow_tint"), Vec3(0.58, 0.40, 0.44)),
            shadow_threshold=_read_f32(doc, cel_node, "shadow_threshold", Float32(0.50)),
            shadow_feather=_read_f32(doc, cel_node, "shadow_feather", Float32(0.015)),
            rim_color=_read_vec3_json(doc, doc.object_get(cel_node, "rim_color"), Vec3(0.42, 0.48, 0.62)),
            rim_power=_read_f32(doc, cel_node, "rim_power", Float32(3.5)),
            rim_threshold=_read_f32(doc, cel_node, "rim_threshold", Float32(0.65)),
            spec_intensity=_read_f32(doc, cel_node, "spec_intensity", Float32(0.32)),
            spec_step_threshold=_read_f32(doc, cel_node, "spec_step_threshold", Float32(0.86)),
            hair_halo_strength=_read_f32(doc, cel_node, "hair_halo_strength", Float32(0.0)),
            hair_shift=_read_f32(doc, cel_node, "hair_shift", Float32(0.12)),
            outline_width_px=_read_f32(doc, cel_node, "outline_width_px", Float32(2.0)),
            outline_color=_read_vec3_json(doc, doc.object_get(cel_node, "outline_color"), Vec3(0.018, 0.014, 0.020)),
            outline_z_bias=_read_f32(doc, cel_node, "outline_z_bias", Float32(0.0025)),
            outline_fov_dist_cap=_read_f32(doc, cel_node, "outline_fov_dist_cap", Float32(6.0)),
            inner_ink_strength=_read_f32(doc, cel_node, "inner_ink_strength", Float32(1.0)),
            ambient_floor=_read_f32(doc, cel_node, "ambient_floor", Float32(0.28)),
            use_vertex_color_channels=doc.boolean(doc.object_get(cel_node, "use_vertex_color_channels"), False),
            receive_scene_shadows=doc.boolean(doc.object_get(cel_node, "receive_scene_shadows"), True),
            use_custom_light_dir=doc.boolean(doc.object_get(cel_node, "use_custom_light_dir"), False),
            custom_light_dir=_read_vec3_json(doc, doc.object_get(cel_node, "custom_light_dir"), Vec3(0.0, 0.55, -0.835)),
            custom_light_intensity=_read_f32(doc, cel_node, "custom_light_intensity", Float32(1.0)),
        )

    var vfx_node = doc.object_get(doc.root, "vfx_params")
    if vfx_node >= 0:
        if doc.nodes[vfx_node].kind != 5:
            raise Error("invalid vfx_params object in .flmat")
        out.vfx_params = AaaStylizedVfxParams(
            uv_scroll_u=_read_f32(doc, vfx_node, "uv_scroll_u", Float32(0.0)),
            uv_scroll_v=_read_f32(doc, vfx_node, "uv_scroll_v", Float32(-2.4)),
            warp_scale=_read_f32(doc, vfx_node, "warp_scale", Float32(4.5)),
            warp_strength=_read_f32(doc, vfx_node, "warp_strength", Float32(0.28)),
            erosion=_read_f32(doc, vfx_node, "erosion", Float32(0.35)),
            edge_band_width=_read_f32(doc, vfx_node, "edge_band_width", Float32(0.20)),
            core_color=_read_vec3_json(doc, doc.object_get(vfx_node, "core_color"), Vec3(7.5, 5.8, 2.4)),
            edge_color=_read_vec3_json(doc, doc.object_get(vfx_node, "edge_color"), Vec3(3.8, 0.32, 0.05)),
            ink_rim_width=_read_f32(doc, vfx_node, "ink_rim_width", Float32(0.045)),
            ink_rim_darkness=_read_f32(doc, vfx_node, "ink_rim_darkness", Float32(0.88)),
        )

    return out^


def load_flmat_file(path: String) raises -> FlmatAsset:
    var raw = read_bytes(path)
    if len(raw) == 0:
        raise Error("failed to read .flmat file: " + path)
    return load_flmat_bytes(raw)
