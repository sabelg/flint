"""Flint v5 Content-Addressed Incremental Parallel `JobPool` Asset Cooker (`cooker_v2.mojo`).

Compiles workspace source assets (`.glb` / procedural `NativeModel`, `.png`, `.hdr`/`.exr`, `.ttf`, `.wav`)
into content-addressed `.flcook` cache bundles (`SHA-256` + ECMA-182 `CRC64` over source bytes + `.flmeta`
import settings + cooker version) and packs them into zero-parse `.flpk` v3 archives:
  - Parallel `JobPool` worker fan-out across independent asset cook tasks
  - `< 5 ms` 100% cache hit on unchanged workspace re-runs
  - Pre-cooked `AaaGeometryPool` (`64v/124t` SAH+Forsyth meshlets + `QEM` `LOD0..3`) + `sub_id = 16` `Bvh8` collider
  - Streamable `BC7` / `BC5` / `BC6H` mip sub-chunks (`sub_id = 0` tail + `sub_id = 1..15` high-res mips)
  - SDF TrueType `.dsf` font atlases & RIFF/WAVE PCM slabs (`CHUNK_COMP_LZ4`).
"""
from std.ffi import external_call
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.serial import BlobReader, BlobWriter, fnv64, fourcc
from flint.assets.bc_compress import (
    downsample_argb_2x,
    downsample_normals_2x,
    encode_bc5_normals,
    encode_bc6h_hdr,
    encode_bc7_rgba,
)
from flint.assets.flpk_v3 import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_FONT,
    ASSET_KIND_HDR,
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    ASSET_KIND_TEXTURE,
    CHUNK_COMP_LZ4,
    CHUNK_COMP_NONE,
    SUB_ID_BVH8_COLLIDER,
    SUB_ID_PRIMARY_OR_TAIL,
    TEX_FMT_BC5_UNORM,
    TEX_FMT_BC6H_UFLOAT,
    TEX_FMT_BC7_SRGB,
    TEX_FMT_BC7_UNORM,
    AssetMeta,
    AssetUuid,
    FlpkBuildChunk,
    crc64_ecma,
    encode_audio_pcm_slab,
    encode_bvh8_collider_slab,
    encode_cooked_mesh_slab,
    encode_flpk_v3,
    encode_texture_tail_slab,
    load_or_init_flmeta,
    mip_to_sub_id,
)
from flint.assets.glb_native import parse_native_glb
from flint.assets.hdr_image import decode_hdr_or_exr
from flint.assets.image import decode_png
from flint.assets.meshlet import AaaGeometryPool
from flint.assets.model import NativeModel
from flint.physics.bvh import Bvh8MeshCollider
from flint.platform.audio import decode_wav
from flint.platform.os import (
    file_stamp,
    get_time,
    list_directory,
    make_dirs,
    parent_directory,
    read_bytes,
    rename_file,
    write_bytes,
)
from flint.text.sdf_bake import bake_font, default_charset, dsf_bytes
from flint.text.ttf import TrueTypeFont

comptime COOKER_V2_VERSION: UInt32 = 5002
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
    var tmp = cache_path + "." + String(Int(external_call["getpid", Int32]())) + ".tmp"
    if not write_bytes(tmp, w.data):
        return False
    return rename_file(tmp, cache_path)


def _load_bundle_from_cache(cache_path: String) -> CookedAssetBundle:
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
        b.rel_path = r.text()
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
        if vc <= 0:
            continue
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


def cook_png_texture_bundle(
    png_bytes: List[UInt8],
    meta: AssetMeta,
    rel_path: String,
) -> CookedAssetBundle:
    """Cooks a `.png` into streamable `BC7` (or `BC5` normal) mip sub-chunks (`sub_id = 0` tail + `1..15` high-res mips)."""
    var bundle = CookedAssetBundle()
    bundle.uuid = meta.uuid
    bundle.rel_path = rel_path
    bundle.asset_kind = ASSET_KIND_TEXTURE

    var img = decode_png(png_bytes, max(256, meta.max_tex_dim))
    if not img.valid or img.width <= 0 or img.height <= 0:
        bundle.error = "failed to decode PNG: " + rel_path
        return bundle^

    var is_normal = rel_path.endswith("_normal.png") or rel_path.endswith("_nrm.png")
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
            var ds = downsample_argb_2x(cur_px, cw, ch)
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
        FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_PRIMARY_OR_TAIL, ASSET_KIND_AUDIO, slab^, CHUNK_COMP_LZ4)
    )
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
        return cook_native_model_bundle(mdl, meta, rel_path)
    if meta.asset_kind == ASSET_KIND_TEXTURE or rel_path.endswith(".png"):
        return cook_png_texture_bundle(source_bytes, meta, rel_path)
    if meta.asset_kind == ASSET_KIND_HDR or rel_path.endswith(".hdr") or rel_path.endswith(".exr"):
        return cook_hdr_texture_bundle(source_bytes, meta, rel_path)
    if meta.asset_kind == ASSET_KIND_FONT or rel_path.endswith(".ttf"):
        return cook_ttf_font_bundle(source_bytes, meta, rel_path)
    if meta.asset_kind == ASSET_KIND_AUDIO or rel_path.endswith(".wav"):
        return cook_wav_audio_bundle(source_bytes, meta, rel_path)
    var raw_bundle = CookedAssetBundle()
    raw_bundle.uuid = meta.uuid
    raw_bundle.rel_path = rel_path
    raw_bundle.asset_kind = ASSET_KIND_RAW
    var copy_raw = source_bytes.copy()
    raw_bundle.chunks.append(
        FlpkBuildChunk(meta.uuid, rel_path, SUB_ID_PRIMARY_OR_TAIL, ASSET_KIND_RAW, copy_raw^, CHUNK_COMP_NONE)
    )
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

        # Deterministic content key over procedural mesh vertex/index counts & positions
        var sig_bytes = List[UInt8]()
        for mi in range(len(slot.proc_model.meshes)):
            ref m = slot.proc_model.meshes[mi]
            var vc = len(m.positions)
            for pi in range(min(64, vc)):
                var bits = Int(m.positions[pi] * Float32(1024.0))
                sig_bytes.append(UInt8(bits & 0xFF))
                sig_bytes.append(UInt8((bits >> 8) & 0xFF))
            sig_bytes.append(UInt8(len(m.indices) & 0xFF))
        var key = compute_cook_cache_key(sig_bytes, meta)
        var cache_file = slot.cache_dir + "/" + key + ".flcook"
        if not slot.force_recook:
            var cached = _load_bundle_from_cache(cache_file)
            if cached.valid:
                slot.was_cache_hit = True
                slot.result = cached^
                return
        var cooked = cook_native_model_bundle(slot.proc_model, meta, slot.rel_path)
        if cooked.valid:
            _ = _save_bundle_to_cache(cache_file, cooked)
        slot.result = cooked^
        return

    var src_bytes = read_bytes(slot.full_path)
    if len(src_bytes) == 0:
        slot.result.error = "empty or unreadable file: " + slot.full_path
        return
    var meta = load_or_init_flmeta(slot.full_path, slot.rel_path, slot.default_kind, persist_if_missing=False)
    var key = compute_cook_cache_key(src_bytes, meta)
    var cache_file = slot.cache_dir + "/" + key + ".flcook"
    if not slot.force_recook:
        var cached = _load_bundle_from_cache(cache_file)
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
    if name.endswith(".png"):
        return (True, ASSET_KIND_TEXTURE)
    if name.endswith(".hdr") or name.endswith(".exr"):
        return (True, ASSET_KIND_HDR)
    if name.endswith(".ttf"):
        return (True, ASSET_KIND_FONT)
    if name.endswith(".wav"):
        return (True, ASSET_KIND_AUDIO)
    return (False, ASSET_KIND_RAW)


def _is_dir(path: String) -> Bool:
    var owned = path
    var d = external_call["opendir", Int](owned.as_c_string_span())
    if d == 0:
        return False
    _ = external_call["closedir", Int32](d)
    return True


def _collect_workspace_sources(
    cur_dir: String,
    rel_prefix: String,
    mut out_full: List[String],
    mut out_rel: List[String],
    mut out_kinds: List[UInt8],
):
    var names = list_directory(cur_dir)
    for name in names:
        var full = (cur_dir + "/" + name) if cur_dir != "/" else ("/" + name)
        var rel = (rel_prefix + "/" + name) if rel_prefix.byte_length() > 0 else name
        if _is_dir(full):
            _collect_workspace_sources(full, rel, out_full, out_rel, out_kinds)
        else:
            var cls = _classify_asset_ext(name)
            if cls[0]:
                out_full.append(full)
                out_rel.append(rel)
                out_kinds.append(cls[1])


def _compute_workspace_fast_signature(
    output_flpk: String,
    full_paths: List[String],
    procedural_count: Int,
) -> UInt64:
    var w = BlobWriter()
    w.u32(COOKER_V2_VERSION)
    w.text(output_flpk)
    w.u32(UInt32(len(full_paths)))
    w.u32(UInt32(procedural_count))
    for i in range(len(full_paths)):
        var st = file_stamp(full_paths[i])
        var mst = file_stamp(full_paths[i] + ".flmeta")
        w.text(full_paths[i])
        w.integer(st[0])
        w.integer(st[1])
        w.integer(mst[0])
        w.integer(mst[1])
    return fnv64(w.data)


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
        var clean_src = source_dir
        while clean_src.byte_length() > 1 and clean_src.as_bytes()[clean_src.byte_length() - 1] == 47:
            var trimmed = String(clean_src[byte=0 : clean_src.byte_length() - 1])
            clean_src = trimmed^
        _collect_workspace_sources(clean_src, "", full_paths, rel_paths, kinds)

    var total_n = len(full_paths) + len(procedural_items)
    rep.total_assets = total_n
    if total_n == 0:
        var empty_img = encode_flpk_v3(List[FlpkBuildChunk](), page_align)
        rep.ok = write_bytes(output_flpk, empty_img)
        rep.archive_bytes = len(empty_img)
        rep.elapsed_ms = (get_time() - t0) * 1000.0
        return rep^

    # Fast-path stamp index check (`< 1 ms` 100% cache hit when workspace & output archive are untouched!)
    var index_path = cache_dir + "/workspace_stamp.bin"
    var cur_sig = _compute_workspace_fast_signature(output_flpk, full_paths, len(procedural_items))
    var out_stamp = file_stamp(output_flpk)
    if (not force_recook) and len(procedural_items) == 0 and out_stamp[0] >= 64:
        var idx_raw = read_bytes(index_path)
        if len(idx_raw) >= 12:
            try:
                var ir = BlobReader(idx_raw^)
                _ = ir.enter(_COOK_INDEX_TAG, 1)
                var saved_sig = ir.u64()
                var saved_pack_size = ir.integer()
                var saved_pack_mtime = ir.integer()
                var saved_chunks = Int(ir.u32())
                var saved_cnt = Int(ir.u32())
                if (
                    not ir.failed
                    and saved_sig == cur_sig
                    and saved_pack_size == out_stamp[0]
                    and saved_pack_mtime == out_stamp[1]
                    and saved_cnt == total_n
                ):
                    for _ in range(saved_cnt):
                        var u_hi = ir.u64()
                        var u_lo = ir.u64()
                        var rp = ir.text()
                        rep.uuids.append(AssetUuid(u_hi, u_lo))
                        rep.rel_paths.append(rp)
                    ir.leave()
                    if not ir.failed:
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
    for ti in range(len(ctx.tasks)):
        ref slot = ctx.tasks[ti]
        if slot.result.valid:
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

    var flpk_bytes = encode_flpk_v3(all_chunks, page_align=page_align)
    var tmp_pack = output_flpk + "." + String(Int(external_call["getpid", Int32]())) + ".tmp"
    var wrote = write_bytes(tmp_pack, flpk_bytes) and rename_file(tmp_pack, output_flpk)
    rep.ok = wrote and rep.failed_count == 0
    rep.total_chunks = len(all_chunks)
    rep.archive_bytes = len(flpk_bytes)

    if rep.ok and len(procedural_items) == 0:
        var new_out_st = file_stamp(output_flpk)
        var iw = BlobWriter()
        iw.begin(_COOK_INDEX_TAG, 1)
        iw.u64(cur_sig)
        iw.integer(new_out_st[0])
        iw.integer(new_out_st[1])
        iw.u32(UInt32(rep.total_chunks))
        iw.u32(UInt32(len(rep.uuids)))
        for i in range(len(rep.uuids)):
            iw.u64(rep.uuids[i].hi)
            iw.u64(rep.uuids[i].lo)
            iw.text(rep.rel_paths[i])
        iw.end()
        _ = write_bytes(index_path, iw.data)

    rep.elapsed_ms = (get_time() - t0) * 1000.0
    return rep^
