"""Flint v5 Content-Addressed Incremental Parallel `JobPool` Asset Cooker (`cooker_v2.mojo`).

Compiles workspace source assets (`.glb` / procedural `NativeModel`, `.png`, `.hdr`/`.exr`, `.ttf`, `.wav`)
into content-addressed `.flcook` cache bundles (`SHA-256` + ECMA-182 `CRC64` over source bytes + `.flmeta`
import settings + cooker version) and packs them into zero-parse `.flpk` v3 archives:
  - Parallel `JobPool` worker fan-out across independent asset cook tasks
  - `< 5 ms` 100% cache hit on unchanged workspace re-runs
  - Pre-cooked `AaaGeometryPool` (`64v/124t` SAH+Forsyth meshlets + `QEM` `LOD0..3`) + `sub_id = 16` `Bvh8` collider
  - Streamable `BC7` / `BC5` / `BC6H` mip sub-chunks (`sub_id = 0` tail + `sub_id = 1..15` high-res mips)
  - SDF TrueType `.dsf` font atlases & seekable RIFF/WAVE PCM slabs with per-page CRCs.
"""
from std.ffi import external_call
from flint.assets.image import srgb_u8_to_linear as _srgb_u8_to_linear, linear_to_srgb_u8 as _linear_to_srgb_u8
from std.memory import bitcast
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool
from flint.core.lz4 import hash_list
from flint.core.math import Vec3
from flint.core.serial import BlobReader, BlobWriter
from flint.assets.bc_compress import (
    downsample_argb_2x,
    downsample_normals_2x,
    encode_bc5_normals,
    encode_bc6h_hdr,
    encode_bc7_rgba,
)
from flint.anim.flrig import FrigJointDef, FrigRigDef, load_flrig_bytes, save_flrig_bytes
from flint.assets.flpk_v3 import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_FONT,
    ASSET_KIND_HDR,
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    ASSET_KIND_TEXTURE,
    CHUNK_COMP_LZ4,
    CHUNK_COMP_NONE,
    SUB_ID_ANIMATED_GLB,
    SUB_ID_FONT_OUTLINES,
    SUB_ID_AUDIO_PAGE_HASHES,
    SUB_ID_BVH8_COLLIDER,
    SUB_ID_PRIMARY_OR_TAIL,
    SUB_ID_RIG,
    TEX_FMT_BC5_UNORM,
    TEX_FMT_BC6H_UFLOAT,
    TEX_FMT_BC7_SRGB,
    TEX_FMT_BC7_UNORM,
    AssetMeta,
    AssetUuid,
    FlpkBuildChunk,
    crc64_ecma,
    encode_audio_pcm_slab,
    encode_audio_page_hashes,
    encode_bvh8_collider_slab,
    encode_cooked_mesh_slab,
    encode_cooked_rig_slab,
    encode_flpk_v3,
    encode_texture_tail_slab,
    load_or_init_flmeta,
    mip_to_sub_id,
)
from flint.assets.glb_native import parse_native_glb
from flint.assets.hdr_image import decode_hdr_or_exr
from flint.assets.jpeg import decode_image_auto
from flint.assets.meshlet import AaaGeometryPool
from flint.assets.model import NativeModel
from flint.physics.bvh import Bvh8MeshCollider
from flint.platform.audio import decode_wav
from flint.platform.os import (
    close_fd,
    file_stamp,
    get_time,
    is_symlink,
    list_directory,
    make_dirs,
    open_read,
    parent_directory,
    read_bytes,
    write_file_durable,
)
from flint.text.sdf_bake import bake_font, default_charset, dsf_bytes
from flint.text.ttf import TrueTypeFont

comptime COOKER_V2_VERSION: UInt32 = 5006
comptime DEFAULT_COOK_CACHE_DIR = "build/cook_cache"
comptime _COOK_BUNDLE_TAG: UInt32 = 0x324B4346  # "FCK2"
comptime _COOK_INDEX_TAG: UInt32 = 0x32584943   # "CIX2"


# ============================================================================
# 1. Pure-Mojo FIPS 180-4 SHA-256 Content-Addressing Digest
# ============================================================================

@always_inline
def _rotr32(x: UInt32, n: UInt32) -> UInt32:
    return (x >> n) | (x << (32 - n))


def sha256_bytes(data: List[UInt8]) -> Array[UInt32, 8]:
    """Pure-Mojo FIPS 180-4 SHA-256 over `data`."""
    var k_tbl = [
        UInt32(0x428A2F98), UInt32(0x71374491), UInt32(0xB5C0FBCF), UInt32(0xE9B5DBA5),
        UInt32(0x3956C25B), UInt32(0x59F111F1), UInt32(0x923F82A4), UInt32(0xAB1C5ED5),
        UInt32(0xD807AA98), UInt32(0x12835B01), UInt32(0x243185BE), UInt32(0x550C7DC3),
        UInt32(0x72BE5D74), UInt32(0x80DEB1FE), UInt32(0x9BDC06A7), UInt32(0xC19BF174),
        UInt32(0xE49B69C1), UInt32(0xEFBE4786), UInt32(0x0FC19DC6), UInt32(0x240CA1CC),
        UInt32(0x2DE92C6F), UInt32(0x4A7484AA), UInt32(0x5CB0A9DC), UInt32(0x76F988DA),
        UInt32(0x983E5152), UInt32(0xA831C66D), UInt32(0xB00327C8), UInt32(0xBF597FC7),
        UInt32(0xC6E00BF3), UInt32(0xD5A79147), UInt32(0x06CA6351), UInt32(0x14292967),
        UInt32(0x27B70A85), UInt32(0x2E1B2138), UInt32(0x4D2C6DFC), UInt32(0x53380D13),
        UInt32(0x650A7354), UInt32(0x766A0ABB), UInt32(0x81C2C92E), UInt32(0x92722C85),
        UInt32(0xA2BFE8A1), UInt32(0xA81A664B), UInt32(0xC24B8B70), UInt32(0xC76C51A3),
        UInt32(0xD192E819), UInt32(0xD6990624), UInt32(0xF40E3585), UInt32(0x106AA070),
        UInt32(0x19A4C116), UInt32(0x1E376C08), UInt32(0x2748774C), UInt32(0x34B0BCB5),
        UInt32(0x391C0CB3), UInt32(0x4ED8AA4A), UInt32(0x5B9CCA4F), UInt32(0x682E6FF3),
        UInt32(0x748F82EE), UInt32(0x78A5636F), UInt32(0x84C87814), UInt32(0x8CC70208),
        UInt32(0x90BEFFFA), UInt32(0xA4506CEB), UInt32(0xBEF9A3F7), UInt32(0xC67178F2),
    ]
    var h = Array[UInt32, 8](fill=0)
    h[0] = 0x6A09E667; h[1] = 0xBB67AE85; h[2] = 0x3C6EF372; h[3] = 0xA54FF53A
    h[4] = 0x510E527F; h[5] = 0x9B05688C; h[6] = 0x1F83D9AB; h[7] = 0x5BE0CD19

    var msg_len = len(data)
    var bit_len = UInt64(msg_len) * 8
    var padded_len = ((msg_len + 9 + 63) // 64) * 64
    var msg = List[UInt8](length=padded_len, fill=0)
    for i in range(msg_len):
        msg[i] = data[i]
    msg[msg_len] = 0x80
    for s in range(8):
        msg[padded_len - 1 - s] = UInt8((bit_len >> UInt64(s * 8)) & 0xFF)

    var w = Array[UInt32, 64](fill=0)
    var blk = 0
    while blk < padded_len:
        for t in range(16):
            var bi = blk + t * 4
            w[t] = (
                (UInt32(msg[bi]) << 24)
                | (UInt32(msg[bi + 1]) << 16)
                | (UInt32(msg[bi + 2]) << 8)
                | UInt32(msg[bi + 3])
            )
        for t in range(16, 64):
            var s0 = _rotr32(w[t - 15], 7) ^ _rotr32(w[t - 15], 18) ^ (w[t - 15] >> 3)
            var s1 = _rotr32(w[t - 2], 17) ^ _rotr32(w[t - 2], 19) ^ (w[t - 2] >> 10)
            w[t] = w[t - 16] + s0 + w[t - 7] + s1

        var a = h[0]; var b = h[1]; var c = h[2]; var d = h[3]
        var e = h[4]; var f = h[5]; var g = h[6]; var hh = h[7]
        for t in range(64):
            var s1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
            var ch = (e & f) ^ ((~e) & g)
            var temp1 = hh + s1 + ch + k_tbl[t] + w[t]
            var s0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
            var maj = (a & b) ^ (a & c) ^ (b & c)
            var temp2 = s0 + maj
            hh = g; g = f; f = e; e = d + temp1
            d = c; c = b; b = a; a = temp1 + temp2

        h[0] += a; h[1] += b; h[2] += c; h[3] += d
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh
        blk += 64
    return h^


def compute_cook_cache_key(source_bytes: List[UInt8], meta: AssetMeta) -> String:
    """Computes content-addressed `SHA-256` + `CRC64` cache key over `(COOKER_V2_VERSION, meta, source_bytes)`."""
    var pre = List[UInt8](capacity=32 + len(source_bytes))
    var v = COOKER_V2_VERSION
    for s in range(4):
        pre.append(UInt8((v >> UInt32(s * 8)) & 0xFF))
    var sh = meta.settings_hash()
    for s in range(8):
        pre.append(UInt8((sh >> UInt64(s * 8)) & 0xFF))
    for b in source_bytes:
        pre.append(b)
    var crc = crc64_ecma(pre)
    var digest = sha256_bytes(pre)
    var hex = "0123456789abcdef".as_bytes()
    var out = List[UInt8](capacity=49)
    for wi in range(4):
        var word = digest[wi]
        for nib in range(8):
            var shift = UInt32((7 - nib) * 4)
            out.append(hex[Int((word >> shift) & 0xF)])
    out.append(95)  # '_'
    for nib in range(16):
        var shift = UInt64((15 - nib) * 4)
        out.append(hex[Int((crc >> shift) & 0xF)])
    return String(unsafe_from_utf8=out)


# ============================================================================
# 2. Per-Asset Cooked Bundle Serialization & Compilers
# ============================================================================

struct CookedAssetBundle(Copyable, Movable):
    var valid: Bool
    var uuid: AssetUuid
    var rel_path: String
    var asset_kind: UInt8
    var chunks: List[FlpkBuildChunk]
    var error: String

    def __init__(out self):
        self.valid = False
        self.uuid = AssetUuid()
        self.rel_path = String()
        self.asset_kind = ASSET_KIND_RAW
        self.chunks = List[FlpkBuildChunk]()
        self.error = String()


def _save_bundle_to_cache(cache_path: String, bundle: CookedAssetBundle) -> Bool:
    var w = BlobWriter()
    w.begin(_COOK_BUNDLE_TAG, 1)
    w.u64(bundle.uuid.hi)
    w.u64(bundle.uuid.lo)
    w.text(bundle.rel_path)
    w.u8(bundle.asset_kind)
    w.u32(UInt32(len(bundle.chunks)))
    for i in range(len(bundle.chunks)):
        ref ch = bundle.chunks[i]
        w.u32(UInt32(ch.sub_id))
        w.u8(ch.asset_kind)
        w.u8(ch.compression)
        w.u32(UInt32(len(ch.raw_payload)))
        for b in ch.raw_payload:
            w.u8(b)
    w.end()
    return write_file_durable(cache_path, w.data, sync=False)


def _is_normal_rel_path(rel_path: String) -> Bool:
    return (
        rel_path.endswith("_normal.png")
        or rel_path.endswith("_nrm.png")
        or rel_path.endswith("_normal.jpg")
        or rel_path.endswith("_nrm.jpg")
        or rel_path.endswith("_normal.jpeg")
        or rel_path.endswith("_nrm.jpeg")
    )


def _load_bundle_from_cache(cache_path: String, expected_rel_path: String = "") -> CookedAssetBundle:
    var b = CookedAssetBundle()
    var raw = read_bytes(cache_path)
    if len(raw) < 12:
        return b^
    try:
        var r = BlobReader(raw^)
        _ = r.enter(_COOK_BUNDLE_TAG, 1)
        var u_hi = r.u64()
        var u_lo = r.u64()
        b.uuid = AssetUuid(u_hi, u_lo)
        var stored_rel_path = r.text()
        b.rel_path = expected_rel_path if expected_rel_path.byte_length() > 0 else stored_rel_path
        if expected_rel_path.byte_length() > 0 and _is_normal_rel_path(stored_rel_path) != _is_normal_rel_path(b.rel_path):
            return CookedAssetBundle()
        b.asset_kind = r.u8()
        var n_ch = Int(r.u32())
        if r.failed or n_ch < 0 or n_ch > 64:
            return CookedAssetBundle()
        for _ in range(n_ch):
            var sub_id = UInt16(r.u32() & 0xFFFF)
            var ak = r.u8()
            var comp = r.u8()
            var plen = Int(r.u32())
            if r.failed or plen < 0 or plen > r.remaining():
                return CookedAssetBundle()
            var pay = List[UInt8](capacity=plen)
            for _ in range(plen):
                pay.append(r.u8())
            b.chunks.append(FlpkBuildChunk(b.uuid, b.rel_path, sub_id, ak, pay^, comp))
        r.leave()
        if not r.failed:
            b.valid = True
    except:
        return CookedAssetBundle()
    return b^


def cook_native_model_bundle(
    model: NativeModel,
    meta: AssetMeta,
    rel_path: String,
) -> CookedAssetBundle:
    """Cooks a `NativeModel` into `AaaGeometryPool` (`QEM LOD 0..3` + `64v/124t` meshlets) + `sub_id=16` `Bvh8` collider."""
    var bundle = CookedAssetBundle()
    bundle.uuid = meta.uuid
    bundle.rel_path = rel_path
    bundle.asset_kind = ASSET_KIND_MESH

    if len(model.meshes) == 0:
        bundle.error = "model has 0 meshes"
        return bundle^

    var geo = AaaGeometryPool()
    var all_verts = List[Vec3]()
    var all_indices = List[Int32]()

    for mi in range(len(model.meshes)):
        ref mesh = model.meshes[mi]
        var vc = mesh.vertex_count()
        if vc <= 0 or len(mesh.positions) < vc * 3:
            continue
        for idx_i in range(len(mesh.indices)):
            var vi = mesh.indices[idx_i]
            if vi < 0 or vi >= vc:
                bundle.error = "mesh vertex index out of range"
                return bundle^
        _ = geo.append_mesh(mesh, mi, build_lods=meta.build_lods)
        if meta.build_collider:
            var base_v = Int32(len(all_verts))
            for vi in range(vc):
                all_verts.append(Vec3(mesh.positions[vi * 3], mesh.positions[vi * 3 + 1], mesh.positions[vi * 3 + 2]))
            if len(mesh.indices) >= 3:
                var tc = len(mesh.indices) // 3
                for ti in range(tc):
                    all_indices.append(base_v + Int32(mesh.indices[ti * 3]))
                    all_indices.append(base_v + Int32(mesh.indices[ti * 3 + 1]))
                    all_indices.append(base_v + Int32(mesh.indices[ti * 3 + 2]))

    if len(geo.meshes) == 0:
        bundle.error = "no valid triangles in model"
        return bundle^

    var comp_mode = CHUNK_COMP_LZ4 if meta.compress_lz4 else CHUNK_COMP_NONE
    var mesh_slab = encode_cooked_mesh_slab(geo)
    bundle.chunks.append(
        FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_PRIMARY_OR_TAIL, ASSET_KIND_MESH, mesh_slab^, comp_mode)
    )

    if meta.build_collider and len(all_indices) >= 3:
        var bvh = Bvh8MeshCollider.build(all_verts, all_indices)
        var bvh_slab = encode_bvh8_collider_slab(bvh)
        bundle.chunks.append(
            FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_BVH8_COLLIDER, ASSET_KIND_MESH, bvh_slab^, comp_mode)
        )

    var has_morph = len(model.morph_names) > 0
    if not has_morph:
        for mi in range(len(model.meshes)):
            if model.meshes[mi].morph_count > 0 or len(model.meshes[mi].morph_positions) > 0:
                has_morph = True
                break
    if len(model.skins) > 0 or len(model.animations) > 0 or has_morph:
        var n_nodes = len(model.nodes)
        var parents = List[Int](length=n_nodes, fill=-1)
        for ni in range(n_nodes):
            for ci in range(len(model.nodes[ni].children)):
                var ch = model.nodes[ni].children[ci]
                if ch >= 0 and ch < n_nodes:
                    parents[ch] = ni
        var rig = FrigRigDef(rel_path)
        for ni in range(n_nodes):
            ref nd = model.nodes[ni]
            rig.joints.append(
                FrigJointDef(nd.name, parents[ni], nd.base_translation, nd.base_rotation, nd.base_scale)
            )
        var rig_slab = encode_cooked_rig_slab(save_flrig_bytes(rig))
        bundle.chunks.append(
            FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_RIG, ASSET_KIND_MESH, rig_slab^, comp_mode)
        )

    bundle.valid = True
    return bundle^


def _downsample_hdr_2x(hdr: List[Vec3], w: Int, h: Int) -> Tuple[List[Vec3], Int, Int]:
    var nw = max(1, w >> 1)
    var nh = max(1, h >> 1)
    var out = List[Vec3](length=nw * nh, fill=Vec3())
    for y in range(nh):
        var y0 = min(h - 1, y * 2)
        var y1 = min(h - 1, y * 2 + 1)
        for x in range(nw):
            var x0 = min(w - 1, x * 2)
            var x1 = min(w - 1, x * 2 + 1)
            out[y * nw + x] = (
                hdr[y0 * w + x0] + hdr[y0 * w + x1] + hdr[y1 * w + x0] + hdr[y1 * w + x1]
            ) * Float32(0.25)
    return (out^, nw, nh)


def _downsample_argb_2x_srgb(pixels: List[UInt32], w: Int, h: Int) -> Tuple[List[UInt32], Int, Int]:
    """2x box filter for sRGB `0xAARRGGBB` textures in physically-correct linear RGB space."""
    var nw = max(1, w >> 1)
    var nh = max(1, h >> 1)
    var out = List[UInt32](length=nw * nh, fill=UInt32(0))
    for y in range(nh):
        var y0 = min(h - 1, y * 2)
        var y1 = min(h - 1, y * 2 + 1)
        for x in range(nw):
            var x0 = min(w - 1, x * 2)
            var x1 = min(w - 1, x * 2 + 1)
            var p0 = pixels[y0 * w + x0]
            var p1 = pixels[y0 * w + x1]
            var p2 = pixels[y1 * w + x0]
            var p3 = pixels[y1 * w + x1]
            var a = (Int((p0 >> 24) & 0xFF) + Int((p1 >> 24) & 0xFF) + Int((p2 >> 24) & 0xFF) + Int((p3 >> 24) & 0xFF) + 2) >> 2
            var r_lin = (
                _srgb_u8_to_linear(Int((p0 >> 16) & 0xFF))
                + _srgb_u8_to_linear(Int((p1 >> 16) & 0xFF))
                + _srgb_u8_to_linear(Int((p2 >> 16) & 0xFF))
                + _srgb_u8_to_linear(Int((p3 >> 16) & 0xFF))
            ) * Float32(0.25)
            var g_lin = (
                _srgb_u8_to_linear(Int((p0 >> 8) & 0xFF))
                + _srgb_u8_to_linear(Int((p1 >> 8) & 0xFF))
                + _srgb_u8_to_linear(Int((p2 >> 8) & 0xFF))
                + _srgb_u8_to_linear(Int((p3 >> 8) & 0xFF))
            ) * Float32(0.25)
            var b_lin = (
                _srgb_u8_to_linear(Int(p0 & 0xFF))
                + _srgb_u8_to_linear(Int(p1 & 0xFF))
                + _srgb_u8_to_linear(Int(p2 & 0xFF))
                + _srgb_u8_to_linear(Int(p3 & 0xFF))
            ) * Float32(0.25)
            var r = _linear_to_srgb_u8(r_lin)
            var g = _linear_to_srgb_u8(g_lin)
            var b = _linear_to_srgb_u8(b_lin)
            out[y * nw + x] = (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
    return (out^, nw, nh)


def cook_png_texture_bundle(
    png_bytes: List[UInt8],
    meta: AssetMeta,
    rel_path: String,
    force_normal: Bool = False,
) -> CookedAssetBundle:
    """Cooks a `.png` / `.jpg` / `.jpeg` into streamable `BC7` (or `BC5` normal) mip sub-chunks (`sub_id = 0` tail + `1..15` high-res mips)."""
    var bundle = CookedAssetBundle()
    bundle.uuid = meta.uuid
    bundle.rel_path = rel_path
    bundle.asset_kind = ASSET_KIND_TEXTURE

    var img = decode_image_auto(png_bytes, max(256, meta.max_tex_dim))
    if not img.valid or img.width <= 0 or img.height <= 0:
        bundle.error = "failed to decode image: " + rel_path
        return bundle^

    var is_normal = (
        force_normal
        or rel_path.endswith("_normal.png")
        or rel_path.endswith("_nrm.png")
        or rel_path.endswith("_normal.jpg")
        or rel_path.endswith("_nrm.jpg")
        or rel_path.endswith("_normal.jpeg")
        or rel_path.endswith("_nrm.jpeg")
    )
    var mips = List[List[UInt8]]()
    var cw = img.width
    var ch = img.height
    var comp_mode = CHUNK_COMP_LZ4 if meta.compress_lz4 else CHUNK_COMP_NONE

    if is_normal:
        var cur_n = List[Vec3](capacity=cw * ch)
        for p in img.pixels:
            var nx = (Float32((p >> 16) & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
            var ny = (Float32((p >> 8) & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
            var nz = (Float32(p & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
            cur_n.append(Vec3(nx, ny, max(Float32(0.01), nz)).normalized())
        while len(mips) < 15:
            mips.append(encode_bc5_normals(cur_n, cw, ch))
            if (not meta.generate_mips) or (cw <= 4 and ch <= 4):
                break
            var ds = downsample_normals_2x(cur_n, cw, ch)
            cur_n = ds[0].copy()
            cw = ds[1]
            ch = ds[2]
        var tf = min(max(0, len(mips) - 1), meta.tail_first_mip)
        var tail_slab = encode_texture_tail_slab(img.width, img.height, TEX_FMT_BC5_UNORM, tf, mips)
        bundle.chunks.append(FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_PRIMARY_OR_TAIL, ASSET_KIND_TEXTURE, tail_slab^, comp_mode))
        for m in range(tf):
            var mcopy = mips[m].copy()
            bundle.chunks.append(FlpkBuildChunk(meta.uuid, rel_path, mip_to_sub_id(m), ASSET_KIND_TEXTURE, mcopy^, comp_mode))
    else:
        var cur_px = img.pixels.copy()
        while len(mips) < 15:
            mips.append(encode_bc7_rgba(cur_px, cw, ch, multi_mode=True))
            if (not meta.generate_mips) or (cw <= 4 and ch <= 4):
                break
            var ds = _downsample_argb_2x_srgb(cur_px, cw, ch) if meta.srgb else downsample_argb_2x(cur_px, cw, ch)
            cur_px = ds[0].copy()
            cw = ds[1]
            ch = ds[2]
        var fmt = TEX_FMT_BC7_SRGB if meta.srgb else TEX_FMT_BC7_UNORM
        var tf = min(max(0, len(mips) - 1), meta.tail_first_mip)
        var tail_slab = encode_texture_tail_slab(img.width, img.height, fmt, tf, mips)
        bundle.chunks.append(FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_PRIMARY_OR_TAIL, ASSET_KIND_TEXTURE, tail_slab^, comp_mode))
        for m in range(tf):
            var mcopy = mips[m].copy()
            bundle.chunks.append(FlpkBuildChunk(meta.uuid, rel_path, mip_to_sub_id(m), ASSET_KIND_TEXTURE, mcopy^, comp_mode))

    bundle.valid = True
    return bundle^


def cook_hdr_texture_bundle(
    hdr_bytes: List[UInt8],
    meta: AssetMeta,
    rel_path: String,
) -> CookedAssetBundle:
    """Cooks a `.hdr` / `.exr` into streamable `BC6H_UFLOAT` mip sub-chunks (`sub_id = 0` tail + `1..15` high-res mips)."""
    var bundle = CookedAssetBundle()
    bundle.uuid = meta.uuid
    bundle.rel_path = rel_path
    bundle.asset_kind = ASSET_KIND_HDR

    var hdr = decode_hdr_or_exr(hdr_bytes)
    if not hdr.valid or hdr.width <= 0 or hdr.height <= 0:
        bundle.error = "failed to decode HDR/EXR: " + rel_path
        return bundle^

    var mips = List[List[UInt8]]()
    var cur_px = hdr.pixels.copy()
    var cw = hdr.width
    var ch = hdr.height
    while len(mips) < 15:
        mips.append(encode_bc6h_hdr(cur_px, cw, ch, partitioned=False))
        if (not meta.generate_mips) or (cw <= 4 and ch <= 4):
            break
        var ds = _downsample_hdr_2x(cur_px, cw, ch)
        cur_px = ds[0].copy()
        cw = ds[1]
        ch = ds[2]

    var comp_mode = CHUNK_COMP_LZ4 if meta.compress_lz4 else CHUNK_COMP_NONE
    var tf = min(max(0, len(mips) - 1), meta.tail_first_mip)
    var tail_slab = encode_texture_tail_slab(hdr.width, hdr.height, TEX_FMT_BC6H_UFLOAT, tf, mips)
    bundle.chunks.append(FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_PRIMARY_OR_TAIL, ASSET_KIND_HDR, tail_slab^, comp_mode))
    for m in range(tf):
        var mcopy = mips[m].copy()
        bundle.chunks.append(FlpkBuildChunk(meta.uuid, rel_path, mip_to_sub_id(m), ASSET_KIND_HDR, mcopy^, comp_mode))
    bundle.valid = True
    return bundle^


def cook_ttf_font_bundle(
    ttf_bytes: List[UInt8],
    meta: AssetMeta,
    rel_path: String,
) -> CookedAssetBundle:
    var bundle = CookedAssetBundle()
    bundle.uuid = meta.uuid
    bundle.rel_path = rel_path
    bundle.asset_kind = ASSET_KIND_FONT

    var face = TrueTypeFont()
    var copy_b = ttf_bytes.copy()
    if not face.load_bytes(copy_b^):
        bundle.error = "invalid TrueType font: " + rel_path
        return bundle^
    try:
        var baked = bake_font(face, default_charset(), 32, 4, atlas_width=512)
        var dsf = dsf_bytes(baked)
        bundle.chunks.append(
            FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_PRIMARY_OR_TAIL, ASSET_KIND_FONT, dsf^, CHUNK_COMP_LZ4)
        )
        bundle.chunks.append(
            FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_FONT_OUTLINES, ASSET_KIND_FONT, ttf_bytes.copy(), CHUNK_COMP_LZ4)
        )
        bundle.valid = True
    except:
        bundle.error = "SDF bake failed: " + rel_path
    return bundle^


def cook_wav_audio_bundle(
    wav_bytes: List[UInt8],
    meta: AssetMeta,
    rel_path: String,
) -> CookedAssetBundle:
    var bundle = CookedAssetBundle()
    bundle.uuid = meta.uuid
    bundle.rel_path = rel_path
    bundle.asset_kind = ASSET_KIND_AUDIO

    var clip = decode_wav(wav_bytes)
    if not clip.valid or clip.channels <= 0 or clip.sample_rate <= 0:
        bundle.error = "invalid RIFF/WAVE file: " + rel_path
        return bundle^
    var slab = encode_audio_pcm_slab(clip.channels, clip.sample_rate, clip.samples)
    bundle.chunks.append(
        FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_PRIMARY_OR_TAIL, ASSET_KIND_AUDIO, slab^, CHUNK_COMP_NONE)
    )
    bundle.chunks.append(FlpkBuildChunk(
        meta.uuid, rel_path, SUB_ID_AUDIO_PAGE_HASHES, ASSET_KIND_AUDIO, encode_audio_page_hashes(bundle.chunks[0].raw_payload),
    ))
    bundle.valid = True
    return bundle^


def cook_single_source_bytes(
    source_bytes: List[UInt8],
    meta: AssetMeta,
    rel_path: String,
) -> CookedAssetBundle:
    if meta.asset_kind == ASSET_KIND_MESH or rel_path.endswith(".glb"):
        var mdl = parse_native_glb(source_bytes)
        if not mdl.loaded():
            var bad = CookedAssetBundle()
            bad.error = "invalid GLB: " + rel_path
            return bad^
        var bundle = cook_native_model_bundle(mdl, meta, rel_path)
        if bundle.valid and (len(mdl.skins) > 0 or len(mdl.animations) > 0):
            bundle.chunks.append(FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_ANIMATED_GLB,
                ASSET_KIND_MESH, source_bytes.copy(), CHUNK_COMP_LZ4))
        return bundle^
    if (
        meta.asset_kind == ASSET_KIND_TEXTURE
        or rel_path.endswith(".png")
        or rel_path.endswith(".jpg")
        or rel_path.endswith(".jpeg")
    ):
        return cook_png_texture_bundle(source_bytes, meta, rel_path)
    if meta.asset_kind == ASSET_KIND_HDR or rel_path.endswith(".hdr") or rel_path.endswith(".exr"):
        return cook_hdr_texture_bundle(source_bytes, meta, rel_path)
    if meta.asset_kind == ASSET_KIND_FONT or rel_path.endswith(".ttf"):
        return cook_ttf_font_bundle(source_bytes, meta, rel_path)
    if meta.asset_kind == ASSET_KIND_AUDIO or rel_path.endswith(".wav"):
        return cook_wav_audio_bundle(source_bytes, meta, rel_path)
    if rel_path.endswith(".flrig"):
        try:
            _ = load_flrig_bytes(source_bytes)
            var rig_bundle = CookedAssetBundle()
            rig_bundle.uuid = meta.uuid
            rig_bundle.rel_path = rel_path
            rig_bundle.asset_kind = ASSET_KIND_RAW
            var rig_comp = CHUNK_COMP_LZ4 if meta.compress_lz4 else CHUNK_COMP_NONE
            var raw_copy = source_bytes.copy()
            rig_bundle.chunks.append(
                FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_PRIMARY_OR_TAIL, ASSET_KIND_RAW, raw_copy^, rig_comp)
            )
            var rig_slab = encode_cooked_rig_slab(source_bytes)
            rig_bundle.chunks.append(
                FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_RIG, ASSET_KIND_RAW, rig_slab^, rig_comp)
            )
            rig_bundle.valid = True
            return rig_bundle^
        except:
            pass
    var raw_bundle = CookedAssetBundle()
    raw_bundle.uuid = meta.uuid
    raw_bundle.rel_path = rel_path
    raw_bundle.asset_kind = ASSET_KIND_RAW
    var copy_raw = source_bytes.copy()
    var comp_mode = CHUNK_COMP_LZ4 if (
        meta.compress_lz4
        or rel_path.endswith(".flscn")
        or rel_path.endswith(".flprefab")
        or rel_path.endswith(".flmat")
        or rel_path.endswith(".json")
    ) else CHUNK_COMP_NONE
    # QOA is frame-compressed already and supports random access in the pack.
    if rel_path.endswith(".qoa"):
        comp_mode = CHUNK_COMP_NONE
    raw_bundle.chunks.append(
        FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_PRIMARY_OR_TAIL, ASSET_KIND_RAW, copy_raw^, comp_mode)
    )
    if rel_path.endswith(".qoa"):
        raw_bundle.chunks.append(FlpkBuildChunk(
            meta.uuid, rel_path, SUB_ID_AUDIO_PAGE_HASHES, ASSET_KIND_AUDIO, encode_audio_page_hashes(source_bytes),
        ))
    raw_bundle.valid = True
    return raw_bundle^


# ============================================================================
# 3. Parallel `JobPool` Workspace Cooker (`cook_workspace`)
# ============================================================================

struct ProceduralCookItem(Copyable, Movable):
    """Optional in-memory procedural `NativeModel` input cooked alongside workspace files."""
    var rel_path: String
    var model: NativeModel
    var build_lods: Bool
    var build_collider: Bool

    def __init__(
        out self,
        rel_path: String,
        var model: NativeModel,
        build_lods: Bool = True,
        build_collider: Bool = True,
    ):
        self.rel_path = rel_path
        self.model = model^
        self.build_lods = build_lods
        self.build_collider = build_collider


struct _CookTaskSlot(Movable):
    var full_path: String
    var rel_path: String
    var default_kind: UInt8
    var cache_dir: String
    var force_recook: Bool
    var is_procedural: Bool
    var proc_model: NativeModel
    var proc_lods: Bool
    var proc_collider: Bool
    var source_hash: UInt64
    var settings_hash: UInt64
    var was_cache_hit: Bool
    var result: CookedAssetBundle

    def __init__(
        out self,
        full_path: String,
        rel_path: String,
        default_kind: UInt8,
        cache_dir: String,
        force_recook: Bool,
    ):
        self.full_path = full_path
        self.rel_path = rel_path
        self.default_kind = default_kind
        self.cache_dir = cache_dir
        self.force_recook = force_recook
        self.is_procedural = False
        self.proc_model = NativeModel()
        self.proc_lods = True
        self.proc_collider = True
        self.source_hash = 0
        self.settings_hash = 0
        self.was_cache_hit = False
        self.result = CookedAssetBundle()


struct _CookParallelCtx(Movable):
    var tasks: List[_CookTaskSlot]

    def __init__(out self, var tasks: List[_CookTaskSlot]):
        self.tasks = tasks^

    @always_inline
    def keep_alive(self):
        var anchor = Int(self.tasks.unsafe_ptr())
        _ = anchor


def _cook_worker_body(ctx_addr: Int, task_idx: Int):
    ref ctx = Pointer[_CookParallelCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    ref slot = ctx.tasks[task_idx]

    if slot.is_procedural:
        var meta = AssetMeta()
        meta.uuid = AssetUuid.from_path(slot.rel_path)
        meta.asset_kind = ASSET_KIND_MESH
        meta.build_lods = slot.proc_lods
        meta.build_collider = slot.proc_collider
        meta.source_path = slot.rel_path

        # Deterministic content key over FULL procedural mesh geometry (all positions, normals, UVs, indices)
        var geom_h = UInt64(0xCBF29CE484222325)
        var prime = UInt64(0x100000001B3)
        var total_pos = 0
        var total_idx = 0
        for mi in range(len(slot.proc_model.meshes)):
            ref m = slot.proc_model.meshes[mi]
            total_pos += len(m.positions)
            total_idx += len(m.indices)
            geom_h ^= UInt64(len(m.positions))
            geom_h *= prime
            for pi in range(len(m.positions)):
                var bits = UInt64(bitcast[DType.uint32, 1](m.positions[pi]))
                geom_h ^= bits
                geom_h *= prime
            geom_h ^= UInt64(len(m.normals))
            geom_h *= prime
            for ni in range(len(m.normals)):
                var bits = UInt64(bitcast[DType.uint32, 1](m.normals[ni]))
                geom_h ^= bits
                geom_h *= prime
            geom_h ^= UInt64(len(m.texcoords))
            geom_h *= prime
            for ui in range(len(m.texcoords)):
                var bits = UInt64(bitcast[DType.uint32, 1](m.texcoords[ui]))
                geom_h ^= bits
                geom_h *= prime
            geom_h ^= UInt64(len(m.indices))
            geom_h *= prime
            for ii in range(len(m.indices)):
                geom_h ^= UInt64(m.indices[ii])
                geom_h *= prime
        var sig_bytes = List[UInt8](capacity=20)
        for sh in range(8):
            sig_bytes.append(UInt8((geom_h >> UInt64(sh * 8)) & UInt64(0xFF)))
        for sh in range(4):
            sig_bytes.append(UInt8((total_pos >> (sh * 8)) & 0xFF))
        for sh in range(4):
            sig_bytes.append(UInt8((total_idx >> (sh * 8)) & 0xFF))
        for sh in range(4):
            sig_bytes.append(UInt8((len(slot.proc_model.meshes) >> (sh * 8)) & 0xFF))
        var key = compute_cook_cache_key(sig_bytes, meta)
        var cache_file = slot.cache_dir + "/" + key + ".flcook"
        if not slot.force_recook:
            var cached = _load_bundle_from_cache(cache_file, slot.rel_path)
            if cached.valid:
                slot.was_cache_hit = True
                slot.result = cached^
                return
        var cooked = cook_native_model_bundle(slot.proc_model, meta, slot.rel_path)
        if cooked.valid:
            _ = _save_bundle_to_cache(cache_file, cooked)
        slot.result = cooked^
        return

    var src_bytes = List[UInt8]()
    if not _read_cook_source(slot.full_path, slot.default_kind, src_bytes):
        slot.result.error = "missing, empty or unreadable source: " + slot.full_path
        return
    var meta = load_or_init_flmeta(slot.full_path, slot.rel_path, slot.default_kind, persist_if_missing=False)
    slot.source_hash = hash_list(src_bytes)
    slot.settings_hash = meta.settings_hash()
    var key = compute_cook_cache_key(src_bytes, meta)
    var cache_file = slot.cache_dir + "/" + key + ".flcook"
    if not slot.force_recook:
        var cached = _load_bundle_from_cache(cache_file, slot.rel_path)
        if cached.valid:
            slot.was_cache_hit = True
            slot.result = cached^
            return

    var cooked = cook_single_source_bytes(src_bytes, meta, slot.rel_path)
    if cooked.valid:
        _ = _save_bundle_to_cache(cache_file, cooked)
    slot.result = cooked^


def _classify_asset_ext(name: String) -> Tuple[Bool, UInt8]:
    if name.endswith(".flmeta"):
        return (False, ASSET_KIND_RAW)
    if name.endswith(".glb"):
        return (True, ASSET_KIND_MESH)
    if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg"):
        return (True, ASSET_KIND_TEXTURE)
    if name.endswith(".hdr") or name.endswith(".exr"):
        return (True, ASSET_KIND_HDR)
    if name.endswith(".ttf"):
        return (True, ASSET_KIND_FONT)
    if name.endswith(".wav"):
        return (True, ASSET_KIND_AUDIO)
    if (
        name.endswith(".flscn")
        or name.endswith(".flscnb")
        or name.endswith(".flprefab")
        or name.endswith(".flmat")
        or name.endswith(".flanim")
        or name.endswith(".flrig")
        or name.endswith(".json")
        or name.endswith(".qoa")
    ):
        return (True, ASSET_KIND_RAW)
    return (False, ASSET_KIND_RAW)


comptime MAX_COOK_DEPTH: Int = 64


def _strip_trailing_slashes(path: String) -> String:
    var out = path
    while out.byte_length() > 1 and out.as_bytes()[out.byte_length() - 1] == 47:
        var trimmed = String(out[byte=0 : out.byte_length() - 1])
        out = trimmed^
    return out^


def _is_dir(path: String) -> Bool:
    var clean = _strip_trailing_slashes(path)
    if is_symlink(clean):
        return False
    var d = external_call["opendir", Int](clean.as_c_string_span())
    if d == 0:
        return False
    _ = external_call["closedir", Int32](Pointer[UInt8, MutAnyOrigin](unsafe_from_address=d))
    return True


def _collect_workspace_sources(
    cur_dir: String,
    rel_prefix: String,
    mut out_full: List[String],
    mut out_rel: List[String],
    mut out_kinds: List[UInt8],
    depth: Int = 0,
):
    if depth >= MAX_COOK_DEPTH:
        return
    var names = list_directory(cur_dir)
    for name in names:
        var full = (cur_dir + "/" + name) if cur_dir != "/" else ("/" + name)
        if is_symlink(full):
            continue
        var rel = (rel_prefix + "/" + name) if rel_prefix.byte_length() > 0 else name
        if _is_dir(full):
            _collect_workspace_sources(full, rel, out_full, out_rel, out_kinds, depth + 1)
        else:
            var cls = _classify_asset_ext(name)
            if cls[0]:
                out_full.append(full)
                out_rel.append(rel)
                out_kinds.append(cls[1])


def _read_cook_source(path: String, kind: UInt8, mut data: List[UInt8]) -> Bool:
    var size = file_stamp(path)[0]
    data = read_bytes(path)
    if size == 0 and len(data) == 0 and kind == ASSET_KIND_RAW:
        var fd = open_read(path)
        if fd >= 0:
            close_fd(fd)
            return True
    return size > 0 and len(data) == size


def _workspace_signature(
    rel_paths: List[String], hashes: List[Tuple[UInt64, UInt64]], page_align: Bool
) -> UInt64:
    var w = BlobWriter()
    w.u32(COOKER_V2_VERSION)
    w.boolean(page_align)
    for i in range(len(rel_paths)):
        w.text(rel_paths[i])
        w.u64(hashes[i][0])
        w.u64(hashes[i][1])
    return hash_list(w.data)


def _matches_workspace(
    full_paths: List[String], rel_paths: List[String], kinds: List[UInt8], page_align: Bool, expected: UInt64
) -> Bool:
    var hashes = List[Tuple[UInt64, UInt64]]()
    for i in range(len(full_paths)):
        var data = List[UInt8]()
        if not _read_cook_source(full_paths[i], kinds[i], data):
            return False
        var meta = load_or_init_flmeta(full_paths[i], rel_paths[i], kinds[i], persist_if_missing=False)
        hashes.append((hash_list(data), meta.settings_hash()))
    return _workspace_signature(rel_paths, hashes, page_align) == expected


struct CookWorkspaceReport(Copyable, Movable):
    var ok: Bool
    var total_assets: Int
    var cooked_count: Int
    var cache_hits: Int
    var failed_count: Int
    var total_chunks: Int
    var archive_bytes: Int
    var elapsed_ms: Float64
    var uuids: List[AssetUuid]
    var rel_paths: List[String]

    def __init__(out self):
        self.ok = False
        self.total_assets = 0
        self.cooked_count = 0
        self.cache_hits = 0
        self.failed_count = 0
        self.total_chunks = 0
        self.archive_bytes = 0
        self.elapsed_ms = 0.0
        self.uuids = List[AssetUuid]()
        self.rel_paths = List[String]()


def cook_workspace(
    source_dir: String,
    output_flpk: String,
    mut pool: JobPool,
    cache_dir: String = DEFAULT_COOK_CACHE_DIR,
    procedural_items: List[ProceduralCookItem] = List[ProceduralCookItem](),
    force_recook: Bool = False,
    page_align: Bool = True,
) -> CookWorkspaceReport:
    """Incrementally cooks all supported assets in `source_dir` (+ optional `procedural_items`) across
    `JobPool` worker threads, caching per-asset `.flcook` bundles in `cache_dir` and writing `output_flpk`."""
    var t0 = get_time()
    var rep = CookWorkspaceReport()

    _ = make_dirs(cache_dir)
    var out_parent = parent_directory(output_flpk)
    if out_parent.byte_length() > 0 and out_parent != ".":
        _ = make_dirs(out_parent)

    var full_paths = List[String]()
    var rel_paths = List[String]()
    var kinds = List[UInt8]()
    if source_dir.byte_length() > 0:
        var clean_src = _strip_trailing_slashes(source_dir)
        if not is_symlink(clean_src):
            _collect_workspace_sources(clean_src, "", full_paths, rel_paths, kinds)

    var total_n = len(full_paths) + len(procedural_items)
    rep.total_assets = total_n
    if total_n == 0:
        var empty_img = encode_flpk_v3(List[FlpkBuildChunk](), page_align)
        rep.ok = write_file_durable(output_flpk, empty_img)
        rep.archive_bytes = len(empty_img)
        rep.elapsed_ms = (get_time() - t0) * 1000.0
        return rep^

    # Match the bytes actually cooked and published, including timestamp-preserving edits.
    var out_u = AssetUuid.from_path(output_flpk)
    var index_path = cache_dir + "/workspace_stamp_" + String(out_u.hi) + "_" + String(out_u.lo) + ".bin"
    var out_stamp = file_stamp(output_flpk)
    if (not force_recook) and len(procedural_items) == 0 and out_stamp[0] >= 64:
        var idx_raw = read_bytes(index_path)
        if len(idx_raw) >= 12:
            try:
                var ir = BlobReader(idx_raw^)
                if ir.enter(_COOK_INDEX_TAG, 2) != 2:
                    raise Error("obsolete workspace index")
                var saved_sig = ir.u64()
                var saved_pack_size = ir.integer()
                var saved_pack_hash = ir.u64()
                var saved_chunks = Int(ir.u32())
                var saved_cnt = Int(ir.u32())
                if (
                    not ir.failed
                    and saved_pack_size == out_stamp[0]
                    and saved_cnt == total_n
                    and _matches_workspace(full_paths, rel_paths, kinds, page_align, saved_sig)
                    and saved_pack_hash == hash_list(read_bytes(output_flpk))
                ):
                    var cached_uuids = List[AssetUuid](capacity=saved_cnt)
                    for i in range(saved_cnt):
                        var u_hi = ir.u64()
                        var u_lo = ir.u64()
                        if ir.text() != rel_paths[i]:
                            raise Error("workspace index paths differ")
                        cached_uuids.append(AssetUuid(u_hi, u_lo))
                    ir.leave()
                    if not ir.failed:
                        rep.uuids = cached_uuids^
                        rep.rel_paths = rel_paths.copy()
                        rep.ok = True
                        rep.cache_hits = total_n
                        rep.cooked_count = 0
                        rep.total_chunks = saved_chunks
                        rep.archive_bytes = out_stamp[0]
                        rep.elapsed_ms = (get_time() - t0) * 1000.0
                        return rep^
            except:
                pass

    var tasks = List[_CookTaskSlot](capacity=total_n)
    for i in range(len(full_paths)):
        tasks.append(_CookTaskSlot(full_paths[i], rel_paths[i], kinds[i], cache_dir, force_recook))
    for i in range(len(procedural_items)):
        ref pi = procedural_items[i]
        var ps = _CookTaskSlot("", pi.rel_path, ASSET_KIND_MESH, cache_dir, force_recook)
        ps.is_procedural = True
        ps.proc_model = pi.model.copy()
        ps.proc_lods = pi.build_lods
        ps.proc_collider = pi.build_collider
        tasks.append(ps^)

    var ctx = _CookParallelCtx(tasks^)
    var n_tasks = len(ctx.tasks)
    if pool.allocated() and pool.workers > 0 and n_tasks > 1:
        run_parallel_ctx(pool, n_tasks, ctx, _cook_worker_body)
        ctx.keep_alive()
    else:
        for ti in range(n_tasks):
            _cook_worker_body(Int(Pointer(to=ctx)), ti)
        ctx.keep_alive()

    var all_chunks = List[FlpkBuildChunk]()
    var input_hashes = List[Tuple[UInt64, UInt64]]()
    for ti in range(len(ctx.tasks)):
        ref slot = ctx.tasks[ti]
        if slot.result.valid:
            input_hashes.append((slot.source_hash, slot.settings_hash))
            if slot.was_cache_hit:
                rep.cache_hits += 1
            else:
                rep.cooked_count += 1
            rep.uuids.append(slot.result.uuid)
            rep.rel_paths.append(slot.result.rel_path)
            for ci in range(len(slot.result.chunks)):
                all_chunks.append(slot.result.chunks[ci].copy())
        else:
            rep.failed_count += 1

    if rep.failed_count > 0:
        rep.elapsed_ms = (get_time() - t0) * 1000.0
        return rep^

    var flpk_bytes = encode_flpk_v3(all_chunks, page_align=page_align)
    rep.ok = write_file_durable(output_flpk, flpk_bytes)
    rep.total_chunks = len(all_chunks)
    rep.archive_bytes = len(flpk_bytes)

    if rep.ok and len(procedural_items) == 0:
        var iw = BlobWriter()
        iw.begin(_COOK_INDEX_TAG, 2)
        iw.u64(_workspace_signature(rel_paths, input_hashes, page_align))
        iw.integer(len(flpk_bytes))
        iw.u64(hash_list(flpk_bytes))
        iw.u32(UInt32(rep.total_chunks))
        iw.u32(UInt32(len(rep.uuids)))
        for i in range(len(rep.uuids)):
            iw.u64(rep.uuids[i].hi)
            iw.u64(rep.uuids[i].lo)
            iw.text(rep.rel_paths[i])
        iw.end()
        _ = write_file_durable(index_path, iw.data, sync=False)

    rep.elapsed_ms = (get_time() - t0) * 1000.0
    return rep^
