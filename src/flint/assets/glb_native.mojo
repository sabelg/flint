"""Complete native GLB 2.0 runtime for Flint Engine (`flint.*`)."""
from std.ffi import external_call
from std.math import isfinite, isnan
from std.memory import bitcast
from flint.platform.os import read_bytes
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.assets.json import JsonDocument, parse_json
from flint.assets.jpeg import decode_image_auto
from flint.anim.skinning import compute_cor
from flint.assets.model import NativeModel, NativeNode, NativeMesh, NativeTexture, NativeSkin, NativeAnimation, NativeChannel


def little_u32(data: List[UInt8], offset: Int) -> Int:
    if offset < 0 or offset + 4 > len(data):
        return -1
    return Int(data[offset]) | (Int(data[offset + 1]) << 8) | (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)


@always_inline
def _float_to_u8(v: Float64) -> UInt8:
    if isnan(v) or v <= 0.0:
        return UInt8(0)
    if v >= 1.0:
        return UInt8(255)
    return UInt8(max(0, min(255, Int(v * 255.0))))


@always_inline
def _sanitize_emissive_comp(v: Float64) -> Float32:
    if isnan(v) or v <= 0.0:
        return Float32(0.0)
    if not isfinite(v) or v > 1.0e6:
        return Float32(1.0e6)
    return Float32(v)


@always_inline
def _clamp_f32_range(v: Float64, lo: Float32, hi: Float32, fallback: Float32) -> Float32:
    if isnan(v):
        return fallback
    if not isfinite(v):
        return hi if v > 0.0 else lo
    return max(lo, min(hi, Float32(v)))


struct NativeAccessor(Copyable, Movable):
    var valid: Bool
    var offset: Int
    var stride: Int
    var count: Int
    var component_type: Int
    var components: Int
    var normalized: Bool

    def __init__(out self):
        self.valid = False
        self.offset = 0
        self.stride = 0
        self.count = 0
        self.component_type = 0
        self.components = 0
        self.normalized = False

    def component_size(self) -> Int:
        if self.component_type == 5120 or self.component_type == 5121:
            return 1
        if self.component_type == 5122 or self.component_type == 5123:
            return 2
        return 4

    def unsigned_value(self, binary: List[UInt8], element: Int, component: Int = 0) -> Int:
        if not self.valid or element < 0 or element >= self.count or component < 0 or component >= self.components:
            return 0
        var offset = self.offset + element * self.stride + component * self.component_size()
        if offset < 0 or offset + self.component_size() > len(binary):
            return 0
        if self.component_type == 5121:
            return Int(binary[offset])
        if self.component_type == 5123:
            return Int(binary[offset]) | (Int(binary[offset + 1]) << 8)
        if self.component_type == 5125:
            return little_u32(binary, offset)
        return 0

    def float_value(self, binary: List[UInt8], element: Int, component: Int = 0) -> Float32:
        if not self.valid or element < 0 or element >= self.count or component < 0 or component >= self.components:
            return 0
        var offset = self.offset + element * self.stride + component * self.component_size()
        if offset < 0 or offset + self.component_size() > len(binary):
            return 0
        if self.component_type == 5126:
            var u = little_u32(binary, offset)
            return bitcast[DType.float32](UInt32(u))
        if self.component_type == 5121:
            var value = Float32(binary[offset])
            return value / 255.0 if self.normalized else value
        if self.component_type == 5123:
            var value = Float32(Int(binary[offset]) | (Int(binary[offset + 1]) << 8))
            return value / 65535.0 if self.normalized else value
        if self.component_type == 5120:
            var value = Float32(bitcast[DType.int8](binary[offset]))
            return max(Float32(-1), value / 127.0) if self.normalized else value
        if self.component_type == 5122:
            var raw_u16 = UInt16(binary[offset]) | (UInt16(binary[offset + 1]) << 8)
            var value = Float32(bitcast[DType.int16](raw_u16))
            return max(Float32(-1), value / 32767.0) if self.normalized else value
        return 0


def component_count(kind: String) -> Int:
    if kind == "SCALAR":
        return 1
    if kind == "VEC2":
        return 2
    if kind == "VEC3":
        return 3
    if kind == "VEC4":
        return 4
    if kind == "MAT2":
        return 4
    if kind == "MAT3":
        return 9
    if kind == "MAT4":
        return 16
    return 0


def load_accessor(document: JsonDocument, index: Int, binary_size: Int) -> NativeAccessor:
    """Read an accessor. It is valid only if all its elements lie inside its buffer view."""
    var accessor = NativeAccessor()
    var accessors = document.object_get(document.root, "accessors")
    var views = document.object_get(document.root, "bufferViews")
    var node = document.at(accessors, index)
    if node < 0:
        return accessor^
    accessor.count = document.integer(document.object_get(node, "count"), 0)
    accessor.component_type = document.integer(document.object_get(node, "componentType"), 0)
    accessor.components = component_count(document.text(document.object_get(node, "type")))
    accessor.normalized = document.boolean(document.object_get(node, "normalized"), False)
    var view_index = document.integer(document.object_get(node, "bufferView"), -1)
    var view = document.at(views, view_index)
    if view < 0 or accessor.count <= 0 or accessor.components <= 0 or accessor.count > binary_size:
        return accessor^
    var view_offset = document.integer(document.object_get(view, "byteOffset"), 0)
    var local_offset = document.integer(document.object_get(node, "byteOffset"), 0)
    accessor.offset = view_offset + local_offset
    accessor.stride = document.integer(document.object_get(view, "byteStride"), accessor.components * accessor.component_size())
    var element = accessor.components * accessor.component_size()
    if accessor.stride < element or accessor.stride > binary_size:
        return accessor^
    var view_length = document.integer(document.object_get(view, "byteLength"), 0)
    if view_offset < 0 or view_length < 0 or view_offset > binary_size or view_length > binary_size - view_offset:
        return accessor^
    if local_offset < 0 or local_offset > view_length:
        return accessor^
    if local_offset + (accessor.count - 1) * accessor.stride + element > view_length:
        return accessor^
    accessor.valid = True
    return accessor^


@always_inline
def _sanitize_json_f32(v: Float64, fallback: Float32 = Float32(0.0)) -> Float32:
    if not isfinite(v):
        return fallback
    var f = Float32(v)
    if not isfinite(f):
        return fallback
    return f


def read_vec3(document: JsonDocument, node: Int, fallback: Vec3) -> Vec3:
    if document.array_size(node) != 3:
        return fallback
    var v0 = document.number(document.at(node, 0), Float64(fallback.x))
    var v1 = document.number(document.at(node, 1), Float64(fallback.y))
    var v2 = document.number(document.at(node, 2), Float64(fallback.z))
    if not isfinite(v0) or not isfinite(v1) or not isfinite(v2):
        return fallback
    var f0 = Float32(v0)
    var f1 = Float32(v1)
    var f2 = Float32(v2)
    if not isfinite(f0) or not isfinite(f1) or not isfinite(f2):
        return fallback
    return Vec3(f0, f1, f2)


def read_quat(document: JsonDocument, node: Int) -> Quat:
    if document.array_size(node) != 4:
        return Quat()
    var v0 = document.number(document.at(node, 0), 0.0)
    var v1 = document.number(document.at(node, 1), 0.0)
    var v2 = document.number(document.at(node, 2), 0.0)
    var v3 = document.number(document.at(node, 3), 1.0)
    if not isfinite(v0) or not isfinite(v1) or not isfinite(v2) or not isfinite(v3):
        return Quat()
    var f0 = Float32(v0)
    var f1 = Float32(v1)
    var f2 = Float32(v2)
    var f3 = Float32(v3)
    if not isfinite(f0) or not isfinite(f1) or not isfinite(f2) or not isfinite(f3):
        return Quat()
    var q = Quat(f0, f1, f2, f3)
    var len2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
    if not isfinite(len2) or len2 <= Float32(1.0e-12):
        return Quat()
    return q.normalized()


def load_native_glb(path: String) -> NativeModel:
    return parse_native_glb(read_bytes(path))


def _decode_glb_image_node(
    document: JsonDocument,
    images: Int,
    views: Int,
    binary: List[UInt8],
    img_idx: Int,
) -> NativeTexture:
    var texture = NativeTexture()
    if img_idx < 0:
        return texture^
    var image_node = document.at(images, img_idx)
    if image_node < 0:
        return texture^
    var view = document.at(views, document.integer(document.object_get(image_node, "bufferView"), -1))
    if view >= 0:
        var offset = document.integer(document.object_get(view, "byteOffset"), 0)
        var length = document.integer(document.object_get(view, "byteLength"), 0)
        if offset >= 0 and length > 0 and length <= len(binary) - offset:
            var bytes = List[UInt8](length=length, fill=0)
            _ = external_call["memcpy", Int](Int(bytes.unsafe_ptr()), Int(binary.unsafe_ptr()) + offset, length)
            var decoded = decode_image_auto(bytes, 4096)
            if decoded.valid:
                texture.width = decoded.width
                texture.height = decoded.height
                swap(texture.pixels, decoded.pixels)
    return texture^


def parse_native_glb(file: List[UInt8]) -> NativeModel:
    var model = NativeModel()
    if len(file) < 28 or little_u32(file, 0) != 0x46546C67 or little_u32(file, 4) != 2 or little_u32(file, 8) != len(file):
        return model^
    var json_length = little_u32(file, 12)
    if json_length <= 0 or little_u32(file, 16) != 0x4E4F534A or 20 + json_length + 8 > len(file):
        return model^
    var json_bytes = List[UInt8](length=json_length, fill=0)
    _ = external_call["memcpy", Int](Int(json_bytes.unsafe_ptr()), Int(file.unsafe_ptr()) + 20, json_length)
    var document = parse_json(json_bytes)
    if not document.valid:
        return model^
    var binary_header = 20 + json_length
    var binary_length = little_u32(file, binary_header)
    if binary_length < 0 or little_u32(file, binary_header + 4) != 0x004E4942 or binary_header + 8 + binary_length > len(file):
        return model^
    var binary = List[UInt8](length=binary_length, fill=0)
    if binary_length > 0:
        _ = external_call["memcpy", Int](Int(binary.unsafe_ptr()), Int(file.unsafe_ptr()) + binary_header + 8, binary_length)

    var images = document.object_get(document.root, "images")
    var textures = document.object_get(document.root, "textures")
    var views = document.object_get(document.root, "bufferViews")
    for texture_index in range(document.array_size(textures)):
        var texture_node = document.at(textures, texture_index)
        var std_source = document.integer(document.object_get(texture_node, "source"), -1)
        var tex_ext = document.object_get(texture_node, "extensions")
        var basisu_source = -1
        if tex_ext != -1:
            var basisu_ext = document.object_get(tex_ext, "KHR_texture_basisu")
            if basisu_ext != -1:
                basisu_source = document.integer(document.object_get(basisu_ext, "source"), -1)
        var texture = NativeTexture()
        if basisu_source >= 0:
            texture = _decode_glb_image_node(document, images, views, binary, basisu_source)
        if texture.width <= 0 and std_source >= 0 and std_source != basisu_source:
            texture = _decode_glb_image_node(document, images, views, binary, std_source)
        model.textures.append(texture^)

    var materials = document.object_get(document.root, "materials")
    var meshes = document.object_get(document.root, "meshes")
    var primitive_counts = List[Int]()
    for mesh_index in range(document.array_size(meshes)):
        var primitives = document.object_get(document.at(meshes, mesh_index), "primitives")
        primitive_counts.append(document.array_size(primitives))
        var target_names = document.object_get(document.object_get(document.at(meshes, mesh_index), "extras"), "targetNames")
        for index in range(document.array_size(target_names)):
            model.morph_names.append(document.text(document.at(target_names, index)))
        for primitive_index in range(document.array_size(primitives)):
            var primitive = document.at(primitives, primitive_index)
            if document.integer(document.object_get(primitive, "mode"), 4) != 4:
                return NativeModel()
            var attributes = document.object_get(primitive, "attributes")
            var mesh = NativeMesh()
            var positions = load_accessor(document, document.integer(document.object_get(attributes, "POSITION"), -1), len(binary))
            if not positions.valid or positions.components != 3:
                return NativeModel()
            var normals = load_accessor(document, document.integer(document.object_get(attributes, "NORMAL"), -1), len(binary))
            var tangents = load_accessor(document, document.integer(document.object_get(attributes, "TANGENT"), -1), len(binary))
            var uv = load_accessor(document, document.integer(document.object_get(attributes, "TEXCOORD_0"), -1), len(binary))
            var uv1 = load_accessor(document, document.integer(document.object_get(attributes, "TEXCOORD_1"), -1), len(binary))
            var colors = load_accessor(document, document.integer(document.object_get(attributes, "COLOR_0"), -1), len(binary))
            var joints = load_accessor(document, document.integer(document.object_get(attributes, "JOINTS_0"), -1), len(binary))
            var weights = load_accessor(document, document.integer(document.object_get(attributes, "WEIGHTS_0"), -1), len(binary))
            var weights_extra = load_accessor(document, document.integer(document.object_get(attributes, "WEIGHTS_1"), -1), len(binary))
            var has_tangents = tangents.valid and tangents.components == 4 and tangents.count >= positions.count
            var has_uv1 = uv1.valid and uv1.components >= 2 and uv1.count >= positions.count
            mesh.positions.reserve(positions.count * 3)
            mesh.normals.reserve(positions.count * 3)
            if has_tangents:
                mesh.tangents.reserve(positions.count * 4)
            mesh.texcoords.reserve(positions.count * 2)
            if has_uv1:
                mesh.texcoords1.reserve(positions.count * 2)
            mesh.colors.reserve(positions.count * 4)
            for vertex in range(positions.count):
                for axis in range(3):
                    mesh.positions.append(positions.float_value(binary, vertex, axis))
                    mesh.normals.append(normals.float_value(binary, vertex, axis) if normals.valid else (Float32(1) if axis == 1 else Float32(0)))
                if has_tangents:
                    var tx = tangents.float_value(binary, vertex, 0)
                    var ty = tangents.float_value(binary, vertex, 1)
                    var tz = tangents.float_value(binary, vertex, 2)
                    var tw_raw = tangents.float_value(binary, vertex, 3)
                    var t_vec = Vec3(tx, ty, tz)
                    if t_vec.length_squared() > Float32(1e-12):
                        t_vec = t_vec.normalized()
                    else:
                        t_vec = Vec3(1.0, 0.0, 0.0)
                    mesh.tangents.append(t_vec.x)
                    mesh.tangents.append(t_vec.y)
                    mesh.tangents.append(t_vec.z)
                    mesh.tangents.append(Float32(-1.0) if tw_raw < Float32(0.0) else Float32(1.0))
                mesh.texcoords.append(uv.float_value(binary, vertex, 0) if uv.valid else 0.0)
                mesh.texcoords.append(uv.float_value(binary, vertex, 1) if uv.valid else 0.0)
                if has_uv1:
                    mesh.texcoords1.append(uv1.float_value(binary, vertex, 0))
                    mesh.texcoords1.append(uv1.float_value(binary, vertex, 1))
                if colors.valid:
                    mesh.colors.append(_float_to_u8(Float64(colors.float_value(binary, vertex, 0))))
                    mesh.colors.append(_float_to_u8(Float64(colors.float_value(binary, vertex, 1))))
                    mesh.colors.append(_float_to_u8(Float64(colors.float_value(binary, vertex, 2))))
                    mesh.colors.append(_float_to_u8(Float64(colors.float_value(binary, vertex, 3))) if colors.components > 3 else 255)
                else:
                    mesh.colors.append(255)
                    mesh.colors.append(255)
                    mesh.colors.append(255)
                    mesh.colors.append(255)
                if joints.valid and weights.valid:
                    for axis in range(4):
                        mesh.joints.append(UInt16(joints.unsigned_value(binary, vertex, axis)))
                        mesh.weights.append(weights.float_value(binary, vertex, axis))
                    if weights_extra.valid:
                        # A second joint set means more than 4 influences. The
                        # renderer has no slots for them; the import check refuses.
                        var influences = 0
                        for axis in range(4):
                            if weights.float_value(binary, vertex, axis) > 0:
                                influences += 1
                            if weights_extra.float_value(binary, vertex, axis) > 0:
                                influences += 1
                        model.max_influences = max(model.max_influences, influences)
            var targets = document.object_get(primitive, "targets")
            for target_index in range(document.array_size(targets)):
                var target = document.at(targets, target_index)
                var delta = load_accessor(document, document.integer(document.object_get(target, "POSITION"), -1), len(binary))
                var normal = load_accessor(document, document.integer(document.object_get(target, "NORMAL"), -1), len(binary))
                if not delta.valid or delta.count != positions.count:
                    return NativeModel()
                for index in range(delta.count * 3):
                    mesh.morph_positions.append(delta.float_value(binary, index // 3, index % 3))
                    mesh.morph_normals.append(normal.float_value(binary, index // 3, index % 3) if normal.valid else 0)
                mesh.morph_count += 1
            var indices_node = document.object_get(primitive, "indices")
            var indices = load_accessor(document, document.integer(indices_node, -1), len(binary))
            if indices.valid:
                if indices.component_type != 5121 and indices.component_type != 5123 and indices.component_type != 5125:
                    return NativeModel()
                mesh.indices.reserve(indices.count)
                for index in range(indices.count):
                    var vi = indices.unsigned_value(binary, index)
                    if vi < 0 or vi >= positions.count:
                        return NativeModel()
                    mesh.indices.append(vi)
            elif indices_node >= 0:
                return NativeModel()
            else:
                mesh.indices.reserve(positions.count)
                for index in range(positions.count):
                    mesh.indices.append(index)
            var material = document.at(materials, document.integer(document.object_get(primitive, "material"), -1))
            if material >= 0:
                var pbr = document.object_get(material, "pbrMetallicRoughness")
                var factor = document.object_get(pbr, "baseColorFactor")
                if document.array_size(factor) == 4:
                    var red = _float_to_u8(document.number(document.at(factor, 0), 1))
                    var green = _float_to_u8(document.number(document.at(factor, 1), 1))
                    var blue = _float_to_u8(document.number(document.at(factor, 2), 1))
                    var alpha = _float_to_u8(document.number(document.at(factor, 3), 1))
                    mesh.base_color = UInt32(alpha) << 24 | UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)
                var base_texture = document.object_get(pbr, "baseColorTexture")
                mesh.texture = document.integer(document.object_get(base_texture, "index"), -1)
                var mr_texture = document.object_get(pbr, "metallicRoughnessTexture")
                mesh.orm_texture = document.integer(document.object_get(mr_texture, "index"), -1)
                var norm_texture = document.object_get(material, "normalTexture")
                mesh.normal_orm_texture = document.integer(document.object_get(norm_texture, "index"), -1)
                var occ_texture = document.object_get(material, "occlusionTexture")
                if occ_texture != -1:
                    mesh.occlusion_texture = document.integer(document.object_get(occ_texture, "index"), -1)
                    mesh.occlusion_texcoord = document.integer(document.object_get(occ_texture, "texCoord"), 0)
                    mesh.occlusion_strength = _clamp_f32_range(document.number(document.object_get(occ_texture, "strength"), 1.0), Float32(0.0), Float32(1.0), Float32(1.0))
                var em_texture = document.object_get(material, "emissiveTexture")
                if em_texture != -1:
                    mesh.emissive_texture = document.integer(document.object_get(em_texture, "index"), -1)
                    mesh.emissive_texcoord = document.integer(document.object_get(em_texture, "texCoord"), 0)
                mesh.roughness = _clamp_f32_range(document.number(document.object_get(pbr, "roughnessFactor"), 0.55), Float32(0.05), Float32(1.0), Float32(0.55))
                mesh.metallic = _clamp_f32_range(document.number(document.object_get(pbr, "metallicFactor"), 0.0), Float32(0.0), Float32(1.0), Float32(0.0))
                var alpha_mode = document.text(document.object_get(material, "alphaMode"), "OPAQUE")
                mesh.alpha_mode = 2 if alpha_mode == "BLEND" else (1 if alpha_mode == "MASK" else 0)
                mesh.double_sided = document.boolean(document.object_get(material, "doubleSided"), False)
                var extensions = document.object_get(material, "extensions")
                var ext_em_strength: Float32 = 1.0
                if extensions != -1:
                    var emissive_extension = document.object_get(extensions, "KHR_materials_emissive_strength")
                    if emissive_extension == -1:
                        emissive_extension = document.object_get(extensions, "KHR_emissive_strength")
                    if emissive_extension != -1:
                        ext_em_strength = _sanitize_emissive_comp(document.number(document.object_get(emissive_extension, "emissiveStrength"), 1.0))
                var emissive = document.object_get(material, "emissiveFactor")
                if document.array_size(emissive) == 3:
                    mesh.emissive_color = Vec3(
                        _sanitize_emissive_comp(document.number(document.at(emissive, 0), 0)),
                        _sanitize_emissive_comp(document.number(document.at(emissive, 1), 0)),
                        _sanitize_emissive_comp(document.number(document.at(emissive, 2), 0)),
                    )
                    var em_max = max(mesh.emissive_color.x, max(mesh.emissive_color.y, mesh.emissive_color.z))
                    if em_max > 0.0001:
                        mesh.emissive_strength = em_max * ext_em_strength
                        mesh.emissive_color = mesh.emissive_color * (Float32(1.0) / em_max)
                    elif mesh.emissive_texture >= 0:
                        mesh.emissive_color = Vec3(1.0, 1.0, 1.0)
                        mesh.emissive_strength = ext_em_strength
                    else:
                        mesh.emissive_strength = 0.0
                elif mesh.emissive_texture >= 0:
                    mesh.emissive_color = Vec3(1.0, 1.0, 1.0)
                    mesh.emissive_strength = ext_em_strength

                if extensions != -1:
                    var cc_ext = document.object_get(extensions, "KHR_materials_clearcoat")
                    if cc_ext != -1:
                        mesh.clearcoat = _clamp_f32_range(document.number(document.object_get(cc_ext, "clearcoatFactor"), 0.0), Float32(0.0), Float32(1.0), Float32(0.0))
                        mesh.clearcoat_roughness = _clamp_f32_range(document.number(document.object_get(cc_ext, "clearcoatRoughnessFactor"), 0.0), Float32(0.0), Float32(1.0), Float32(0.0))
                    var spec_ext = document.object_get(extensions, "KHR_materials_specular")
                    if spec_ext != -1:
                        mesh.specular_factor = _clamp_f32_range(document.number(document.object_get(spec_ext, "specularFactor"), 1.0), Float32(0.0), Float32(1.0), Float32(1.0))
                    var irid_ext = document.object_get(extensions, "KHR_materials_iridescence")
                    if irid_ext != -1:
                        mesh.iridescence = _clamp_f32_range(document.number(document.object_get(irid_ext, "iridescenceFactor"), 0.0), Float32(0.0), Float32(1.0), Float32(0.0))
                        mesh.iridescence_ior = _clamp_f32_range(document.number(document.object_get(irid_ext, "iridescenceIor"), 1.3), Float32(1.0), Float32(5.0), Float32(1.3))
                        mesh.iridescence_thickness_nm = _clamp_f32_range(document.number(document.object_get(irid_ext, "iridescenceThicknessMaximum"), 100.0), Float32(0.0), Float32(10000.0), Float32(100.0))
                    var sheen_ext = document.object_get(extensions, "KHR_materials_sheen")
                    if sheen_ext != -1:
                        var sc = document.object_get(sheen_ext, "sheenColorFactor")
                        if document.array_size(sc) == 3:
                            mesh.sheen_color = Vec3(
                                _clamp_f32_range(document.number(document.at(sc, 0), 0.0), Float32(0.0), Float32(1.0), Float32(0.0)),
                                _clamp_f32_range(document.number(document.at(sc, 1), 0.0), Float32(0.0), Float32(1.0), Float32(0.0)),
                                _clamp_f32_range(document.number(document.at(sc, 2), 0.0), Float32(0.0), Float32(1.0), Float32(0.0)),
                            )
                        mesh.sheen_roughness = _clamp_f32_range(document.number(document.object_get(sheen_ext, "sheenRoughnessFactor"), 0.0), Float32(0.0), Float32(1.0), Float32(0.0))
            if len(mesh.weights) == mesh.vertex_count() * 4 and len(mesh.weights) > 0:
                # Package 3.4: the centers of rotation, once, where the model is read.
                mesh.centers = compute_cor(mesh.positions, mesh.indices, mesh.joints, mesh.weights)
            model.meshes.append(mesh^)

    var nodes = document.object_get(document.root, "nodes")
    for node_index in range(document.array_size(nodes)):
        var source = document.at(nodes, node_index)
        var node = NativeNode()
        node.name = document.text(document.object_get(source, "name"))
        node.translation = read_vec3(document, document.object_get(source, "translation"), Vec3())
        node.rotation = read_quat(document, document.object_get(source, "rotation"))
        node.scale = read_vec3(document, document.object_get(source, "scale"), Vec3(1, 1, 1))
        node.base_translation = node.translation
        node.base_rotation = node.rotation
        node.base_scale = node.scale
        var matrix = document.object_get(source, "matrix")
        if document.array_size(matrix) == 16:
            for index in range(16):
                var def_m = Float32(1.0) if (index % 5 == 0) else Float32(0.0)
                node.matrix.values[index] = _sanitize_json_f32(
                    document.number(document.at(matrix, index), Float64(def_m)),
                    def_m,
                )
            node.has_matrix = True
        var children = document.object_get(source, "children")
        for index in range(document.array_size(children)):
            node.children.append(document.integer(document.at(children, index), -1))
        var mesh_index = document.integer(document.object_get(source, "mesh"), -1)
        if mesh_index >= 0 and mesh_index < len(primitive_counts):
            var first = 0
            for index in range(mesh_index):
                first += primitive_counts[index]
            if primitive_counts[mesh_index] > 0:
                node.mesh = first
                for primitive in range(primitive_counts[mesh_index]):
                    node.mesh_indices.append(first + primitive)
            var skin = document.integer(document.object_get(source, "skin"), -1)
            if skin >= 0 and node.mesh >= 0:
                for primitive in range(primitive_counts[mesh_index]):
                    model.meshes[node.mesh + primitive].skin = skin
        model.nodes.append(node^)

    var scenes = document.object_get(document.root, "scenes")
    var scene_index = document.integer(document.object_get(document.root, "scene"), 0)
    var scene = document.at(scenes, scene_index)
    var roots = document.object_get(scene, "nodes")
    for index in range(document.array_size(roots)):
        model.roots.append(document.integer(document.at(roots, index), -1))
    if len(model.roots) == 0 and len(model.nodes) > 0:
        model.roots.append(0)

    var skins = document.object_get(document.root, "skins")
    for skin_index in range(document.array_size(skins)):
        var source = document.at(skins, skin_index)
        var skin = NativeSkin()
        var joints = document.object_get(source, "joints")
        for index in range(document.array_size(joints)):
            skin.joints.append(document.integer(document.at(joints, index), -1))
        var inverse_bind = load_accessor(document, document.integer(document.object_get(source, "inverseBindMatrices"), -1), len(binary))
        if inverse_bind.valid and inverse_bind.components == 16:
            for index in range(inverse_bind.count * 16):
                skin.inverse_bind.append(inverse_bind.float_value(binary, index // 16, index % 16))
        model.skins.append(skin^)

    var animations = document.object_get(document.root, "animations")
    for animation_index in range(document.array_size(animations)):
        var source = document.at(animations, animation_index)
        var animation = NativeAnimation()
        animation.name = document.text(document.object_get(source, "name"))
        var samplers = document.object_get(source, "samplers")
        var channels = document.object_get(source, "channels")
        for channel_index in range(document.array_size(channels)):
            var source_channel = document.at(channels, channel_index)
            var sampler = document.at(samplers, document.integer(document.object_get(source_channel, "sampler"), -1))
            var target = document.object_get(source_channel, "target")
            var channel = NativeChannel()
            channel.node = document.integer(document.object_get(target, "node"), -1)
            var path_name = document.text(document.object_get(target, "path"))
            channel.path = 0 if path_name == "translation" else (1 if path_name == "rotation" else 2)
            if path_name != "translation" and path_name != "rotation" and path_name != "scale":
                continue
            var interp_name = document.text(document.object_get(sampler, "interpolation"), "LINEAR")
            channel.step = interp_name == "STEP"
            if interp_name == "STEP":
                channel.interp_mode = 1
            elif interp_name == "CUBICSPLINE":
                channel.interp_mode = 2
            else:
                channel.interp_mode = 0
            var input = load_accessor(document, document.integer(document.object_get(sampler, "input"), -1), len(binary))
            var output = load_accessor(document, document.integer(document.object_get(sampler, "output"), -1), len(binary))
            if not input.valid or not output.valid:
                continue
            for index in range(input.count):
                channel.times.append(input.float_value(binary, index))
                animation.duration = max(animation.duration, input.float_value(binary, index))
            for index in range(output.count * output.components):
                channel.values.append(output.float_value(binary, index // output.components, index % output.components))
            animation.channels.append(channel^)
        model.animations.append(animation^)
    return model^
