"""Pure-Mojo binary glTF 2.0 (.glb) serializer for NativeModel."""
from flint.platform.os import write_bytes
from flint.core.math import Vec3
from flint.assets.image import encode_png
from flint.assets.jpeg import encode_jpeg_baseline
from flint.assets.json_write import json_quote
from flint.assets.model import NativeModel, NativeMesh, NativeNode


def append_u32_le(mut out: List[UInt8], value: Int):
    out.append(UInt8(value & 255))
    out.append(UInt8((value >> 8) & 255))
    out.append(UInt8((value >> 16) & 255))
    out.append(UInt8((value >> 24) & 255))


def append_f32_le(mut out: List[UInt8], value: Float32):
    var copy = value
    var bits = Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0]
    append_u32_le(out, Int(bits))


def pad4(mut out: List[UInt8], pad_byte: UInt8 = 0):
    while len(out) % 4 != 0:
        out.append(pad_byte)


def fmt_f32(value: Float32) -> String:
    return String(Float64(value))


@fieldwise_init
struct PrimExportIds(Copyable, ImplicitlyCopyable, Movable):
    var pos_acc: Int
    var norm_acc: Int
    var uv_acc: Int
    var col_acc: Int
    var idx_acc: Int
    var mat_idx: Int
    var joints_acc: Int
    var weights_acc: Int
    var skin: Int


def _prim_attrs(pe: PrimExportIds) -> String:
    var attrs = '"POSITION": ' + String(pe.pos_acc) + ', "NORMAL": ' + String(pe.norm_acc) + ', "TEXCOORD_0": ' + String(pe.uv_acc)
    if pe.col_acc >= 0:
        attrs += ', "COLOR_0": ' + String(pe.col_acc)
    if pe.joints_acc >= 0 and pe.weights_acc >= 0:
        attrs += ', "JOINTS_0": ' + String(pe.joints_acc) + ', "WEIGHTS_0": ' + String(pe.weights_acc)
    return attrs


def encode_native_glb(mut model: NativeModel, jpeg_texture_mask: UInt64 = 0) -> List[UInt8]:
    """Serialize a NativeModel (meshes, nodes, PBR materials, PNG/JPEG textures) into a standard GLB 2.0 binary."""
    var bin_data = List[UInt8]()
    var buffer_views = List[String]()
    var accessors = List[String]()
    var images_json = List[String]()
    var textures_json = List[String]()
    var materials_json = List[String]()

    # 1. Embed textures as lossless RGBA PNG or baseline YCbCr JPEG bufferViews
    for tex_idx in range(len(model.textures)):
        var w = model.textures[tex_idx].width
        var h = model.textures[tex_idx].height
        var pixels = model.textures[tex_idx].pixels.copy()
        if w <= 0 or h <= 0 or len(pixels) < w * h:
            w = 4
            h = 4
            pixels = List[UInt32](length=16, fill=0xFFFFFFFF)
        var as_jpeg = tex_idx < 64 and ((jpeg_texture_mask >> UInt64(tex_idx)) & 1) != 0
        var img_bytes = encode_jpeg_baseline(pixels, w, h, 94) if as_jpeg else encode_png(pixels, w, h, "", True)
        var mime_str = "image/jpeg" if as_jpeg else "image/png"
        pad4(bin_data, 0)
        var offset = len(bin_data)
        for b in img_bytes:
            bin_data.append(b)
        var length = len(img_bytes)
        pad4(bin_data, 0)
        var bv_idx = len(buffer_views)
        buffer_views.append(
            '{"buffer": 0, "byteOffset": ' + String(offset) + ', "byteLength": ' + String(length) + "}"
        )
        var img_idx = len(images_json)
        images_json.append('{"bufferView": ' + String(bv_idx) + ', "mimeType": "' + mime_str + '"}')
        textures_json.append('{"source": ' + String(img_idx) + "}")

    # 2. Serialize each NativeMesh's vertex attributes, indices, and PBR material
    var prim_exports = List[PrimExportIds]()
    for m_idx in range(len(model.meshes)):
        ref mesh = model.meshes[m_idx]
        var vcount = mesh.vertex_count()
        if vcount <= 0:
            prim_exports.append(PrimExportIds(-1, -1, -1, -1, -1, -1, -1, -1, -1))
            continue

        # POSITION (VEC3 float32 + min/max)
        pad4(bin_data, 0)
        var pos_off = len(bin_data)
        var min_x = mesh.positions[0]
        var min_y = mesh.positions[1]
        var min_z = mesh.positions[2]
        var max_x = min_x
        var max_y = min_y
        var max_z = min_z
        for v in range(vcount):
            var px = mesh.positions[v * 3]
            var py = mesh.positions[v * 3 + 1]
            var pz = mesh.positions[v * 3 + 2]
            min_x = min(min_x, px)
            min_y = min(min_y, py)
            min_z = min(min_z, pz)
            max_x = max(max_x, px)
            max_y = max(max_y, py)
            max_z = max(max_z, pz)
            append_f32_le(bin_data, px)
            append_f32_le(bin_data, py)
            append_f32_le(bin_data, pz)
        var pos_len = len(bin_data) - pos_off
        var pos_bv = len(buffer_views)
        buffer_views.append(
            '{"buffer": 0, "byteOffset": ' + String(pos_off) + ', "byteLength": ' + String(pos_len) + ', "target": 34962}'
        )
        var pos_acc = len(accessors)
        accessors.append(
            '{"bufferView": ' + String(pos_bv)
            + ', "componentType": 5126, "count": ' + String(vcount)
            + ', "type": "VEC3", "min": [' + fmt_f32(min_x) + ", " + fmt_f32(min_y) + ", " + fmt_f32(min_z)
            + '], "max": [' + fmt_f32(max_x) + ", " + fmt_f32(max_y) + ", " + fmt_f32(max_z) + "]}"
        )

        # NORMAL (VEC3 float32)
        pad4(bin_data, 0)
        var norm_off = len(bin_data)
        for v in range(vcount):
            var nx = mesh.normals[v * 3] if v * 3 + 2 < len(mesh.normals) else Float32(0)
            var ny = mesh.normals[v * 3 + 1] if v * 3 + 2 < len(mesh.normals) else Float32(1)
            var nz = mesh.normals[v * 3 + 2] if v * 3 + 2 < len(mesh.normals) else Float32(0)
            append_f32_le(bin_data, nx)
            append_f32_le(bin_data, ny)
            append_f32_le(bin_data, nz)
        var norm_len = len(bin_data) - norm_off
        var norm_bv = len(buffer_views)
        buffer_views.append(
            '{"buffer": 0, "byteOffset": ' + String(norm_off) + ', "byteLength": ' + String(norm_len) + ', "target": 34962}'
        )
        var norm_acc = len(accessors)
        accessors.append(
            '{"bufferView": ' + String(norm_bv) + ', "componentType": 5126, "count": ' + String(vcount) + ', "type": "VEC3"}'
        )

        # TEXCOORD_0 (VEC2 float32)
        pad4(bin_data, 0)
        var uv_off = len(bin_data)
        for v in range(vcount):
            var tu = mesh.texcoords[v * 2] if v * 2 + 1 < len(mesh.texcoords) else Float32(0)
            var tv = mesh.texcoords[v * 2 + 1] if v * 2 + 1 < len(mesh.texcoords) else Float32(0)
            append_f32_le(bin_data, tu)
            append_f32_le(bin_data, tv)
        var uv_len = len(bin_data) - uv_off
        var uv_bv = len(buffer_views)
        buffer_views.append(
            '{"buffer": 0, "byteOffset": ' + String(uv_off) + ', "byteLength": ' + String(uv_len) + ', "target": 34962}'
        )
        var uv_acc = len(accessors)
        accessors.append(
            '{"bufferView": ' + String(uv_bv) + ', "componentType": 5126, "count": ' + String(vcount) + ', "type": "VEC2"}'
        )

        # COLOR_0 (optional VEC4 normalized uint8 -> 5121 normalized)
        var col_acc = -1
        if len(mesh.colors) >= vcount * 4:
            var has_nonwhite = False
            for c_i in range(vcount * 4):
                if mesh.colors[c_i] != 255:
                    has_nonwhite = True
                    break
            if has_nonwhite:
                pad4(bin_data, 0)
                var col_off = len(bin_data)
                for c_i in range(vcount * 4):
                    bin_data.append(mesh.colors[c_i])
                var col_len = len(bin_data) - col_off
                pad4(bin_data, 0)
                var col_bv = len(buffer_views)
                buffer_views.append(
                    '{"buffer": 0, "byteOffset": ' + String(col_off) + ', "byteLength": ' + String(col_len) + ', "target": 34962}'
                )
                col_acc = len(accessors)
                accessors.append(
                    '{"bufferView": ' + String(col_bv) + ', "componentType": 5121, "normalized": true, "count": ' + String(vcount) + ', "type": "VEC4"}'
                )

        # JOINTS_0 (VEC4 uint16 = 5123) + WEIGHTS_0 (VEC4 float), when the mesh is skinned
        var joints_acc = -1
        var weights_acc = -1
        if len(model.nodes) > 0 and mesh.skin >= 0 and mesh.skin < len(model.skins) and len(mesh.joints) >= vcount * 4 and len(mesh.weights) >= vcount * 4:
            pad4(bin_data, 0)
            var j_off = len(bin_data)
            for j_i in range(vcount * 4):
                var jv = Int(mesh.joints[j_i])
                bin_data.append(UInt8(jv & 255))
                bin_data.append(UInt8((jv >> 8) & 255))
            var j_len = len(bin_data) - j_off
            var j_bv = len(buffer_views)
            buffer_views.append(
                '{"buffer": 0, "byteOffset": ' + String(j_off) + ', "byteLength": ' + String(j_len) + ', "target": 34962}'
            )
            joints_acc = len(accessors)
            accessors.append(
                '{"bufferView": ' + String(j_bv) + ', "componentType": 5123, "count": ' + String(vcount) + ', "type": "VEC4"}'
            )
            pad4(bin_data, 0)
            var w_off = len(bin_data)
            for w_i in range(vcount * 4):
                append_f32_le(bin_data, mesh.weights[w_i])
            var w_len = len(bin_data) - w_off
            var w_bv = len(buffer_views)
            buffer_views.append(
                '{"buffer": 0, "byteOffset": ' + String(w_off) + ', "byteLength": ' + String(w_len) + ', "target": 34962}'
            )
            weights_acc = len(accessors)
            accessors.append(
                '{"bufferView": ' + String(w_bv) + ', "componentType": 5126, "count": ' + String(vcount) + ', "type": "VEC4"}'
            )

        # INDICES (SCALAR uint32 = 5125)
        pad4(bin_data, 0)
        var idx_off = len(bin_data)
        var icount = len(mesh.indices) if len(mesh.indices) > 0 else vcount
        if len(mesh.indices) > 0:
            for idx_val in mesh.indices:
                append_u32_le(bin_data, idx_val)
        else:
            for idx_val in range(vcount):
                append_u32_le(bin_data, idx_val)
        var idx_len = len(bin_data) - idx_off
        var idx_bv = len(buffer_views)
        buffer_views.append(
            '{"buffer": 0, "byteOffset": ' + String(idx_off) + ', "byteLength": ' + String(idx_len) + ', "target": 34963}'
        )
        var idx_acc = len(accessors)
        accessors.append(
            '{"bufferView": ' + String(idx_bv) + ', "componentType": 5125, "count": ' + String(icount) + ', "type": "SCALAR"}'
        )

        # MATERIAL
        var bc = mesh.base_color
        var b_r = Float32((bc >> 16) & 255) / 255.0
        var b_g = Float32((bc >> 8) & 255) / 255.0
        var b_b = Float32(bc & 255) / 255.0
        var b_a = Float32((bc >> 24) & 255) / 255.0
        var pbr_str = (
            '{"baseColorFactor": [' + fmt_f32(b_r) + ", " + fmt_f32(b_g) + ", " + fmt_f32(b_b) + ", " + fmt_f32(b_a) + "]"
            + ', "metallicFactor": ' + fmt_f32(mesh.metallic)
            + ', "roughnessFactor": ' + fmt_f32(mesh.roughness)
        )
        if mesh.texture >= 0 and mesh.texture < len(textures_json):
            pbr_str += ', "baseColorTexture": {"index": ' + String(mesh.texture) + "}"
        if mesh.orm_texture >= 0 and mesh.orm_texture < len(textures_json):
            pbr_str += ', "metallicRoughnessTexture": {"index": ' + String(mesh.orm_texture) + "}"
        pbr_str += "}"

        var alpha_str = "BLEND" if mesh.alpha_mode == 2 else ("MASK" if mesh.alpha_mode == 1 else "OPAQUE")
        var mat_str = (
            '{"pbrMetallicRoughness": ' + pbr_str
            + ', "alphaMode": ' + json_quote(alpha_str)
            + ', "doubleSided": ' + ("true" if mesh.double_sided else "false")
        )
        if mesh.normal_orm_texture >= 0 and mesh.normal_orm_texture < len(textures_json):
            mat_str += ', "normalTexture": {"index": ' + String(mesh.normal_orm_texture) + "}"
        var ext_parts = List[String]()
        if mesh.emissive_strength > 0.0001:
            var ef = mesh.emissive_color
            var em_max = max(Float32(0.0001), max(ef.x, max(ef.y, ef.z)))
            var norm_ef = ef * (min(Float32(1.0), mesh.emissive_strength) / em_max)
            mat_str += ', "emissiveFactor": [' + fmt_f32(norm_ef.x) + ", " + fmt_f32(norm_ef.y) + ", " + fmt_f32(norm_ef.z) + "]"
            if mesh.emissive_strength > 1.0:
                ext_parts.append('"KHR_materials_emissive_strength": {"emissiveStrength": ' + fmt_f32(mesh.emissive_strength) + "}")
        if mesh.clearcoat > 0.0001 or abs(mesh.clearcoat_roughness - Float32(0.08)) > 0.0001:
            ext_parts.append(
                '"KHR_materials_clearcoat": {"clearcoatFactor": ' + fmt_f32(mesh.clearcoat)
                + ', "clearcoatRoughnessFactor": ' + fmt_f32(mesh.clearcoat_roughness) + "}"
            )
        if abs(mesh.specular_factor - Float32(1.0)) > 0.0001:
            ext_parts.append('"KHR_materials_specular": {"specularFactor": ' + fmt_f32(mesh.specular_factor) + "}")
        if mesh.iridescence > 0.0001:
            ext_parts.append(
                '"KHR_materials_iridescence": {"iridescenceFactor": ' + fmt_f32(mesh.iridescence)
                + ', "iridescenceIor": ' + fmt_f32(mesh.iridescence_ior)
                + ', "iridescenceThicknessMaximum": ' + fmt_f32(mesh.iridescence_thickness_nm) + "}"
            )
        if mesh.sheen_color.x > 0.0001 or mesh.sheen_color.y > 0.0001 or mesh.sheen_color.z > 0.0001:
            ext_parts.append(
                '"KHR_materials_sheen": {"sheenColorFactor": ['
                + fmt_f32(mesh.sheen_color.x) + ", " + fmt_f32(mesh.sheen_color.y) + ", " + fmt_f32(mesh.sheen_color.z)
                + '], "sheenRoughnessFactor": ' + fmt_f32(mesh.sheen_roughness) + "}"
            )
        if len(ext_parts) > 0:
            mat_str += ', "extensions": {'
            for e_i in range(len(ext_parts)):
                if e_i > 0:
                    mat_str += ", "
                mat_str += ext_parts[e_i]
            mat_str += "}"
        mat_str += "}"
        var mat_idx = len(materials_json)
        materials_json.append(mat_str)

        prim_exports.append(PrimExportIds(pos_acc, norm_acc, uv_acc, col_acc, idx_acc, mat_idx, joints_acc, weights_acc,
                                          mesh.skin if joints_acc >= 0 else -1))

    # Helper to format a primitive JSON string
    var gltf_meshes_json = List[String]()
    var nodes_json = List[String]()
    var roots_json = List[Int]()

    if len(model.nodes) == 0:
        var prims_str = String("[")
        var first_p = True
        for m_idx in range(len(prim_exports)):
            var pe = prim_exports[m_idx]
            if pe.pos_acc < 0:
                continue
            if not first_p:
                prims_str += ", "
            first_p = False
            var attrs = _prim_attrs(pe)
            prims_str += '{"attributes": {' + attrs + '}, "indices": ' + String(pe.idx_acc) + ', "material": ' + String(pe.mat_idx) + ', "mode": 4}'
        prims_str += "]"
        gltf_meshes_json.append('{"primitives": ' + prims_str + "}")
        nodes_json.append('{"name": "root", "mesh": 0}')
        roots_json.append(0)
    else:
        for n_idx in range(len(model.nodes)):
            ref node = model.nodes[n_idx]
            var gltf_mesh_id = -1
            var node_prims = List[Int]()
            if len(node.mesh_indices) > 0:
                for mi in node.mesh_indices:
                    if mi >= 0 and mi < len(prim_exports) and prim_exports[mi].pos_acc >= 0:
                        node_prims.append(mi)
            elif node.mesh >= 0 and node.mesh < len(prim_exports) and prim_exports[node.mesh].pos_acc >= 0:
                node_prims.append(node.mesh)

            var node_skin = -1
            if len(node_prims) > 0:
                node_skin = prim_exports[node_prims[0]].skin
                var prims_str = String("[")
                for p_i in range(len(node_prims)):
                    if p_i > 0:
                        prims_str += ", "
                    var pe = prim_exports[node_prims[p_i]]
                    var attrs = _prim_attrs(pe)
                    prims_str += '{"attributes": {' + attrs + '}, "indices": ' + String(pe.idx_acc) + ', "material": ' + String(pe.mat_idx) + ', "mode": 4}'
                prims_str += "]"
                gltf_mesh_id = len(gltf_meshes_json)
                gltf_meshes_json.append('{"name": ' + json_quote(node.name) + ', "primitives": ' + prims_str + "}")

            var n_str = '{"name": ' + json_quote(node.name)
            if node.has_matrix:
                n_str += ', "matrix": ['
                for m_i in range(16):
                    if m_i > 0:
                        n_str += ", "
                    n_str += fmt_f32(node.matrix.values[m_i])
                n_str += "]"
            else:
                n_str += ', "translation": [' + fmt_f32(node.translation.x) + ", " + fmt_f32(node.translation.y) + ", " + fmt_f32(node.translation.z) + "]"
                n_str += ', "rotation": [' + fmt_f32(node.rotation.x) + ", " + fmt_f32(node.rotation.y) + ", " + fmt_f32(node.rotation.z) + ", " + fmt_f32(node.rotation.w) + "]"
                n_str += ', "scale": [' + fmt_f32(node.scale.x) + ", " + fmt_f32(node.scale.y) + ", " + fmt_f32(node.scale.z) + "]"
            if gltf_mesh_id >= 0:
                n_str += ', "mesh": ' + String(gltf_mesh_id)
                if node_skin >= 0:
                    n_str += ', "skin": ' + String(node_skin)
            if len(node.children) > 0:
                n_str += ', "children": ['
                for c_i in range(len(node.children)):
                    if c_i > 0:
                        n_str += ", "
                    n_str += String(node.children[c_i])
                n_str += "]"
            n_str += "}"
            nodes_json.append(n_str)

        if len(model.roots) > 0:
            for r in model.roots:
                roots_json.append(r)
        else:
            roots_json.append(0)

    # SKINS: joint node lists + inverse bind matrices (MAT4 float, column-major)
    var skins_json = List[String]()
    if len(model.nodes) > 0:
        for s_idx in range(len(model.skins)):
            ref sk = model.skins[s_idx]
            var sk_str = String('{"joints": [')
            for j_i in range(len(sk.joints)):
                if j_i > 0:
                    sk_str += ", "
                sk_str += String(sk.joints[j_i])
            sk_str += "]"
            if len(sk.inverse_bind) >= len(sk.joints) * 16 and len(sk.joints) > 0:
                pad4(bin_data, 0)
                var ib_off = len(bin_data)
                for m_i in range(len(sk.joints) * 16):
                    append_f32_le(bin_data, sk.inverse_bind[m_i])
                var ib_len = len(bin_data) - ib_off
                var ib_bv = len(buffer_views)
                buffer_views.append(
                    '{"buffer": 0, "byteOffset": ' + String(ib_off) + ', "byteLength": ' + String(ib_len) + "}"
                )
                var ib_acc = len(accessors)
                accessors.append(
                    '{"bufferView": ' + String(ib_bv) + ', "componentType": 5126, "count": ' + String(len(sk.joints)) + ', "type": "MAT4"}'
                )
                sk_str += ', "inverseBindMatrices": ' + String(ib_acc)
            sk_str += "}"
            skins_json.append(sk_str)

    pad4(bin_data, 0)

    var json = String('{"asset": {"version": "2.0", "generator": "Flint Pure-Mojo GLB Writer"}, "scene": 0, "scenes": [{"nodes": [')
    for i in range(len(roots_json)):
        if i > 0:
            json += ", "
        json += String(roots_json[i])
    json += ']}], "buffers": [{"byteLength": ' + String(len(bin_data)) + '}], "bufferViews": ['
    for i in range(len(buffer_views)):
        if i > 0:
            json += ", "
        json += buffer_views[i]
    json += '], "accessors": ['
    for i in range(len(accessors)):
        if i > 0:
            json += ", "
        json += accessors[i]
    json += "]"
    if len(images_json) > 0:
        json += ', "images": ['
        for i in range(len(images_json)):
            if i > 0:
                json += ", "
            json += images_json[i]
        json += '], "textures": ['
        for i in range(len(textures_json)):
            if i > 0:
                json += ", "
            json += textures_json[i]
        json += "]"
    json += ', "materials": ['
    for i in range(len(materials_json)):
        if i > 0:
            json += ", "
        json += materials_json[i]
    json += '], "meshes": ['
    for i in range(len(gltf_meshes_json)):
        if i > 0:
            json += ", "
        json += gltf_meshes_json[i]
    json += '], "nodes": ['
    for i in range(len(nodes_json)):
        if i > 0:
            json += ", "
        json += nodes_json[i]
    json += "]"
    if len(skins_json) > 0:
        json += ', "skins": ['
        for i in range(len(skins_json)):
            if i > 0:
                json += ", "
            json += skins_json[i]
        json += "]"
    json += "}"

    var json_bytes = List[UInt8]()
    for b in json.as_bytes():
        json_bytes.append(b)
    pad4(json_bytes, 0x20)

    var total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    var glb = List[UInt8](capacity=total_len)
    append_u32_le(glb, 0x46546C67)  # "glTF"
    append_u32_le(glb, 2)           # version 2
    append_u32_le(glb, total_len)

    append_u32_le(glb, len(json_bytes))
    append_u32_le(glb, 0x4E4F534A)  # "JSON"
    for b in json_bytes:
        glb.append(b)

    append_u32_le(glb, len(bin_data))
    append_u32_le(glb, 0x004E4942)  # "BIN\0"
    for b in bin_data:
        glb.append(b)

    return glb^


def write_native_glb(path: String, mut model: NativeModel) -> Bool:
    var bytes = encode_native_glb(model)
    return write_bytes(path, bytes)
