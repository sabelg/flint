# Verifies M3 Features (F15–F22) & M4 Features (F23–F28)
"""Gated M3/M4 Contract Verification Suite (`tests/e2e_opaque/gated/test_gated_m3_m4_gpu_plat.mojo`).

Standalone M3/M4-gated contract tests (NOT imported by live suites; compiled and
executed in `--strict` mode once Milestones M3 and M4 land).

Verified Milestone M3 Features (`PROJECT.md § Feature Inventory` & `§ Interface Contracts`):
- **F15**: Non-ReBAR per-frame `LinearStagingRing` + `vkCmdCopyBuffer` to `DEVICE_LOCAL` VRAM in `aaa_renderer.mojo`.
- **F16**: `AaaRenderGraph` aliased `VkDeviceMemory` heaps (`create_aliased_heap`, `bind_buffer_aliased`, `bind_image_aliased`) + `submit_side` async compute + barriers.
- **F17**: Split deferred VisBuffer shading (Material-ID tile classification + per-material VisBuffer shade shaders).
- **F18**: Wired `vat_crowd_eval_shader` & `TerrainClipmapDesc` pipelines/descriptors in `FlintAaaVulkanRenderer`.
- **F19**: Growable geometry pools beyond `32/32/8 MiB` caps & direct swapchain image presentation.
- **F20**: `AaaResidencyStreamer` Vulkan image/sampler mip uploads & non-zero `TypedAssetStorage.slot_vram_bytes` / `total_vram_live_bytes()` tracking.
- **F21**: Codec parity — `glb_native.mojo` (`TANGENT`, `TEXCOORD_1`, `occlusionTexture`, `emissiveTexture`, KTX2), `jpeg.mojo` (`SOF2` progressive), `hdr_image.mojo` (tiled `0x200`, `PIZ`, `DWAA` EXR).
- **F22**: GPU/Vulkan RAII (`VulkanGpu`, `VulkanImage`, `VulkanAccelStruct`, `SwapchainHandle`, `VulkanMemBlock`, `GpuBuffer`, `AmdGpu`, `SwapchainSurface`) + Win32/macOS WSI `UInt64` attach helpers.

Verified Milestone M4 Features (`PROJECT.md § Feature Inventory` & `§ Interface Contracts`):
- **F23**: Wayland desktop parity (`wp_fractional_scale_v1`, `zxdg_decoration_manager_v1`, `wl_data_device_manager` clipboard/DnD, XKB keymap + `text_input_utf8`).
- **F24**: X11 desktop parity (`MIT-SHM`, `XI_RawMotion` pointer lock, XKB/`text_input_utf8`, `X11Window.attach_vulkan`).
- **F25**: Direct `EvdevHub` window integration (no `/dev/input/js*` reads in `platform/input.mojo`).
- **F26**: Text & UI parity (`TAG_OTTO` CFF/OTF outlines, MSDF/MTSDF baking, `GSUB`/RTL shaping, fallback fonts, `UiContext.text_input` clipboard/mouse-caret/IME).
- **F27**: Multi-OS pure-Mojo Win32 & macOS window, socket (`NetTransport` via `CompilationTarget`), and audio backends.
- **F28**: Platform handle RAII (`WaylandWindow`, `WaylandBridge`, `X11Window`, `KmsWindow`, `LogindSession`, `PulseClient`, `NativeAudio`, `LinuxGamepad`, `LockstepPeer`, `RollbackPeer`).
"""
from std.collections import List
from std.math import abs, exp, log
from std.memory import bitcast
from std.sys.info import CompilationTarget
from std.testing import assert_equal, assert_false, assert_true

from flint.assets.bc_compress import f32_to_uf16_bits
from flint.assets.flpk_v3 import AssetUuid
from flint.assets.glb_native import parse_native_glb
from flint.assets.hdr_image import (
    EXR_DWAA_COMPRESSION,
    EXR_NO_COMPRESSION,
    EXR_PIZ_COMPRESSION,
    decode_exr_fp16,
    decode_hdr_or_exr,
)
from flint.assets.jpeg import decode_jpeg
from flint.assets.ktx2 import VK_FORMAT_R8G8B8A8_UNORM
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeAnimation, NativeChannel, NativeModel
from flint.assets.server import AssetServer, CookedTextureAsset, TEX_FMT_BC7_SRGB
from flint.core.math import Mat4, Vec3
from flint.input import EvdevHub
from flint.input.evdev import ABS_Z
from flint.net import DatagramView, NetEndpoint, UdpSocket
from flint.platform.amdgpu import AmdGpu, GpuBuffer
from flint.platform.audio import NativeAudio
from flint.platform.input import InputState, PAD_AXIS_LT
from flint.platform.kms import KmsWindow
from flint.platform.logind import LogindSession
from flint.platform.macos_audio import MacOsAudio
from flint.platform.macos_window import CocoaWindow, cocoa_flip_y, cocoa_keycode_to_flint_key
from flint.platform.os import LinuxGamepad, LockstepPeer, RollbackPeer, read_bytes
from flint.platform.pulse import PulseClient
from flint.platform.traits import AudioBackend, WindowBackend
from flint.platform.vulkan import (
    BUFFER_USAGE,
    FORMAT_R8G8B8A8_UNORM,
    IMAGE_USAGE_SAMPLED,
    IMAGE_USAGE_TRANSFER_TARGET,
    SwapchainHandle,
    VkMemReq,
    VulkanAccelStruct,
    VulkanGpu,
    VulkanImage,
    VulkanMemBlock,
)
from flint.platform.wayland import WaylandWindow, parse_uri_list
from flint.platform.wayland_bridge import WaylandBridge
from flint.platform.win32_audio import Win32Audio
from flint.platform.win32_window import Win32Window, utf16le_to_utf8, utf8_to_utf16le, win32_vk_to_flint_key
from flint.platform.x11 import X11Window
from flint.platform.xkb import XKB_MOD_SHIFT, XkbKeymap, x11_keysym_to_codepoint
from flint.render.aaa_render_graph import (
    RG_QUEUE_ASYNC_COMPUTE,
    RG_QUEUE_GRAPHICS,
    RG_USAGE_COLOR_ATT_WRITE,
    RG_USAGE_COMPUTE_READ,
    RG_USAGE_COMPUTE_WRITE,
    RG_USAGE_FRAGMENT_READ,
    AaaRenderGraph,
    RgResourceAccess,
)
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_gi_svgf import (
    AAA_SHADE_CLASS_STANDARD_PBR,
    aaa_deferred_shade_class_shader,
    aaa_tile_classify_shader,
)
from flint.render.aaa_post_tsr import aaa_display_tonemap_direct_shader
from flint.render.aaa_renderer import FlintAaaVulkanRenderer, LinearStagingRing
from flint.render.material_graph import (
    TerrainClipmapDesc,
    bake_vat_crowd_clip_rgba16f,
    vat_crowd_eval_shader,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.render.swapchain_surface import SwapchainSurface
from flint.render.ui_overlay import aaa_ui_overlay_direct_shader
from flint.text.bidi import BIDI_AUTO, resolve_bidi
from flint.text.line_break import wrap_shaped
from flint.text.rich_text import parse_rich
from flint.text.sdf_bake import bake_glyph_msdf, bake_glyph_mtsdf
from flint.text.shaper import FontFallbackChain
from flint.text.ttf import GsubGlyph, TrueTypeFont
from flint.text.utf8 import decode_utf8
from flint.ui.context import (
    UI_KEY_A,
    UI_KEY_C,
    UI_KEY_LEFT_CTRL,
    UI_KEY_V,
    UI_KEY_X,
    UiContext,
)


def _str_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _put_u32_le(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


def _put_u16_be(mut out: List[UInt8], v: UInt16):
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _put_u32_be(mut out: List[UInt8], v: UInt32):
    out.append(UInt8((v >> 24) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _build_f21_extended_glb() -> List[UInt8]:
    """Minimal GLB 2.0 exercising TANGENT, TEXCOORD_1, occlusionTexture, emissiveTexture & KHR_texture_basisu."""
    var bin = List[UInt8]()
    # 3 vertices: pos(3*12) + tangent(3*16) + uv1(3*8) = 108 B
    var verts = [
        Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(1.0), Float32(0.0), Float32(0.0),
        Float32(0.0), Float32(1.0), Float32(0.0),
    ]
    for f in verts:
        _put_u32_le(bin, bitcast[DType.uint32](f))
    var pos_len = len(bin)
    for _ in range(3):
        for tf in [Float32(1.0), Float32(0.0), Float32(0.0), Float32(1.0)]:
            _put_u32_le(bin, bitcast[DType.uint32](tf))
    var tan_off = pos_len
    var tan_len = len(bin) - tan_off
    for uv in [Float32(0.0), Float32(0.0), Float32(1.0), Float32(0.0), Float32(0.0), Float32(1.0)]:
        _put_u32_le(bin, bitcast[DType.uint32](uv))
    var uv_off = tan_off + tan_len
    var uv_len = len(bin) - uv_off

    # Build a genuine 4x4 RGBA8 uncompressed KTX2 container (magic + 80B header + 24B level index + 64B RGBA8)
    var ktx2 = List[UInt8](capacity=168)
    for b in [
        UInt8(0xAB), UInt8(0x4B), UInt8(0x54), UInt8(0x58),
        UInt8(0x20), UInt8(0x32), UInt8(0x30), UInt8(0xBB),
        UInt8(0x0D), UInt8(0x0A), UInt8(0x1A), UInt8(0x0A),
    ]:
        ktx2.append(b)
    _put_u32_le(ktx2, UInt32(VK_FORMAT_R8G8B8A8_UNORM))  # vkFormat = 37
    _put_u32_le(ktx2, 1)    # typeSize
    _put_u32_le(ktx2, 4)    # pixelWidth
    _put_u32_le(ktx2, 4)    # pixelHeight
    _put_u32_le(ktx2, 0)    # pixelDepth
    _put_u32_le(ktx2, 0)    # layerCount
    _put_u32_le(ktx2, 1)    # faceCount
    _put_u32_le(ktx2, 1)    # levelCount
    _put_u32_le(ktx2, 0)    # supercompressionScheme = KTX2_SC_NONE
    _put_u32_le(ktx2, 0); _put_u32_le(ktx2, 0)  # dfdByteOffset, dfdByteLength
    _put_u32_le(ktx2, 0); _put_u32_le(ktx2, 0)  # kvdByteOffset, kvdByteLength
    _put_u64_le(ktx2, 0); _put_u64_le(ktx2, 0)  # sgdByteOffset, sgdByteLength
    _put_u64_le(ktx2, 104)  # level[0].byteOffset
    _put_u64_le(ktx2, 64)   # level[0].byteLength
    _put_u64_le(ktx2, 64)   # level[0].uncompressedByteLength
    for _ in range(16):
        ktx2.append(0x40)  # R
        ktx2.append(0x80)  # G
        ktx2.append(0xC0)  # B
        ktx2.append(0xFF)  # A

    var img_off = len(bin)
    for b in ktx2:
        bin.append(b)
    var img_len = len(ktx2)
    while len(bin) % 4 != 0:
        bin.append(0)

    var j = (
        '{"asset":{"version":"2.0"},'
        + '"extensionsUsed":["KHR_texture_basisu"],'
        + '"buffers":[{"byteLength":' + String(len(bin)) + '}],'
        + '"bufferViews":['
        + '{"buffer":0,"byteOffset":0,"byteLength":' + String(pos_len) + '},'
        + '{"buffer":0,"byteOffset":' + String(tan_off) + ',"byteLength":' + String(tan_len) + '},'
        + '{"buffer":0,"byteOffset":' + String(uv_off) + ',"byteLength":' + String(uv_len) + '},'
        + '{"buffer":0,"byteOffset":' + String(img_off) + ',"byteLength":' + String(img_len) + '}'
        + '],'
        + '"accessors":['
        + '{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},'
        + '{"bufferView":1,"componentType":5126,"count":3,"type":"VEC4"},'
        + '{"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"}'
        + '],'
        + '"images":[{"bufferView":3,"mimeType":"image/ktx2"}],'
        + '"textures":[{"extensions":{"KHR_texture_basisu":{"source":0}}}],'
        + '"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1]},'
        + '"occlusionTexture":{"index":0,"texCoord":1},'
        + '"emissiveTexture":{"index":0,"texCoord":1},'
        + '"emissiveFactor":[0.5,0.25,0.1]}],'
        + '"meshes":[{"primitives":[{"attributes":{"POSITION":0,"TANGENT":1,"TEXCOORD_1":2},"material":0}]}],'
        + '"nodes":[{"mesh":0}],"scene":0,"scenes":[{"nodes":[0]}]}'
    )
    var jb = _str_bytes(j)
    while len(jb) % 4 != 0:
        jb.append(0x20)
    var total = 12 + 8 + len(jb) + 8 + len(bin)
    var out = List[UInt8](capacity=total)
    _put_u32_le(out, 0x46546C67)
    _put_u32_le(out, 2)
    _put_u32_le(out, UInt32(total))
    _put_u32_le(out, UInt32(len(jb)))
    _put_u32_le(out, 0x4E4F534A)
    for b in jb:
        out.append(b)
    _put_u32_le(out, UInt32(len(bin)))
    _put_u32_le(out, 0x004E4942)
    for b in bin:
        out.append(b)
    return out^


def _build_f21_minimal_sof2_jpeg() -> List[UInt8]:
    """Minimal valid 8x8 grayscale Progressive SOF2 JPEG stream."""
    var out = List[UInt8]()
    # SOI
    out.append(0xFF); out.append(0xD8)
    # DQT (table 0, all 1s)
    out.append(0xFF); out.append(0xDB); _put_u16_be(out, 67); out.append(0)
    for _ in range(64):
        out.append(1)
    # SOF2 (8x8, 1 component)
    out.append(0xFF); out.append(0xC2); _put_u16_be(out, 11)
    out.append(8); _put_u16_be(out, 8); _put_u16_be(out, 8)
    out.append(1); out.append(1); out.append(0x11); out.append(0)
    # DHT DC0 (sym 0 -> 1-bit '0')
    out.append(0xFF); out.append(0xC4); _put_u16_be(out, 20); out.append(0x00)
    out.append(1)
    for _ in range(15):
        out.append(0)
    out.append(0)
    # DHT AC0 (sym 0x00 EOB -> 1-bit '0')
    out.append(0xFF); out.append(0xC4); _put_u16_be(out, 20); out.append(0x10)
    out.append(1)
    for _ in range(15):
        out.append(0)
    out.append(0x00)
    # SOS DC scan (Ss=0, Se=0, Ah=0, Al=0)
    out.append(0xFF); out.append(0xDA); _put_u16_be(out, 8)
    out.append(1); out.append(1); out.append(0x00)
    out.append(0); out.append(0); out.append(0x00)
    out.append(0x00)
    # SOS AC scan (Ss=1, Se=63, Ah=0, Al=0)
    out.append(0xFF); out.append(0xDA); _put_u16_be(out, 8)
    out.append(1); out.append(1); out.append(0x00)
    out.append(1); out.append(63); out.append(0x00)
    out.append(0x00)
    # EOI
    out.append(0xFF); out.append(0xD9)
    return out^


def _put_u16_le(mut out: List[UInt8], v: Int):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))


def _put_u64_le(mut out: List[UInt8], v: UInt64):
    for b in range(8):
        out.append(UInt8((v >> UInt64(b * 8)) & 0xFF))


def _wrap_zlib_stored(raw: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    out.append(0x78)
    out.append(0x01)
    var pos = 0
    while pos < len(raw) or len(raw) == 0:
        var chunk = min(65535, len(raw) - pos)
        var last = 1 if (pos + chunk >= len(raw)) else 0
        out.append(UInt8(last))
        _put_u16_le(out, chunk)
        _put_u16_le(out, chunk ^ 0xFFFF)
        for i in range(chunk):
            out.append(raw[pos + i])
        pos += chunk
        if len(raw) == 0:
            break
    var s1: UInt32 = 1
    var s2: UInt32 = 0
    for b in raw:
        s1 = (s1 + UInt32(b)) % 65521
        s2 = (s2 + s1) % 65521
    var adler = (s2 << 16) | s1
    _put_u32_be(out, adler)
    return out^


def _exr_reorder_and_predict(raw: List[UInt8]) -> List[UInt8]:
    var n = len(raw)
    if n <= 1:
        return raw.copy()
    var pred = List[UInt8](length=n, fill=0)
    var half = (n + 1) // 2
    var i1 = 0
    var i2 = half
    var j = 0
    while j < n:
        pred[i1] = raw[j]
        i1 += 1
        j += 1
        if j < n:
            pred[i2] = raw[j]
            i2 += 1
            j += 1
    var out = List[UInt8](length=n, fill=0)
    out[0] = pred[0]
    for i in range(1, n):
        out[i] = UInt8((Int(pred[i]) - Int(pred[i - 1]) + 128) & 0xFF)
    return out^


def _exr_wenc14(a: UInt16, b: UInt16) -> Tuple[UInt16, UInt16]:
    var ai = Int(a)
    var bi = Int(b)
    var hi = (ai - bi) & 0xFFFF
    var h_s = Int(bitcast[DType.int16](UInt16(hi)))
    var li = (ai - (h_s & 1) - (h_s >> 1)) & 0xFFFF
    return (UInt16(li), UInt16(hi))


def _exr_wav2_encode(mut buf: List[UInt16], base: Int, nx: Int, ox: Int, ny: Int, oy: Int):
    var n = nx if nx < ny else ny
    var p = 1
    var p2 = 2
    while p2 <= n:
        var y_end = oy * (ny - p2)
        var x_end = ox * (nx - p2)
        var iy = 0
        while iy <= y_end:
            var ix = 0
            while ix <= x_end:
                var i00 = base + iy + ix
                var i01 = i00 + ox * p
                var i10 = i00 + oy * p
                var i11 = i10 + ox * p
                var h0 = _exr_wenc14(buf[i00], buf[i01]); buf[i00] = h0[0]; buf[i01] = h0[1]
                var h1 = _exr_wenc14(buf[i10], buf[i11]); buf[i10] = h1[0]; buf[i11] = h1[1]
                var v0 = _exr_wenc14(buf[i00], buf[i10]); buf[i00] = v0[0]; buf[i10] = v0[1]
                var v1 = _exr_wenc14(buf[i01], buf[i11]); buf[i01] = v1[0]; buf[i11] = v1[1]
                ix += ox * p2
            if (nx & p) != 0:
                var i00 = base + iy + ix
                var i10 = i00 + oy * p
                var v0 = _exr_wenc14(buf[i00], buf[i10]); buf[i00] = v0[0]; buf[i10] = v0[1]
            iy += oy * p2
        if (ny & p) != 0:
            var ix = 0
            while ix <= x_end:
                var i00 = base + iy + ix
                var i01 = i00 + ox * p
                var h0 = _exr_wenc14(buf[i00], buf[i01]); buf[i00] = h0[0]; buf[i01] = h0[1]
                ix += ox * p2
        p = p2
        p2 <<= 1


struct _ExrTestMsbWriter(Movable):
    var out: List[UInt8]
    var buf: UInt64
    var cnt: Int
    var total_bits: Int

    def __init__(out self):
        self.out = List[UInt8]()
        self.buf = 0
        self.cnt = 0
        self.total_bits = 0

    def write_msb(mut self, val: UInt64, bits: Int):
        for i in range(bits):
            var b = (val >> UInt64(bits - 1 - i)) & 1
            self.buf = (self.buf << 1) | b
            self.cnt += 1
            self.total_bits += 1
            if self.cnt == 8:
                self.out.append(UInt8(self.buf & 0xFF))
                self.buf = 0
                self.cnt = 0

    def flush(mut self):
        if self.cnt > 0:
            self.out.append(UInt8((self.buf << UInt64(8 - self.cnt)) & 0xFF))
            self.buf = 0
            self.cnt = 0


def _build_exr_huf_blob(vals: List[UInt16]) -> List[UInt8]:
    var max_s = 0
    for v in vals:
        if Int(v) > max_s:
            max_s = Int(v)
    var sym_limit = max(1, max_s + 1)
    var blen = 1
    while (1 << blen) <= sym_limit:
        blen += 1
    var im = 0
    var iM = max_s + 1
    var tw = _ExrTestMsbWriter()
    for _ in range(im, iM + 1):
        tw.write_msb(UInt64(blen), 6)
    tw.flush()

    var dw = _ExrTestMsbWriter()
    for v in vals:
        dw.write_msb(UInt64(v), blen)
    var n_bits = dw.total_bits
    dw.flush()

    var out = List[UInt8]()
    _put_u32_le(out, UInt32(im))
    _put_u32_le(out, UInt32(iM))
    _put_u32_le(out, UInt32(len(tw.out)))
    _put_u32_le(out, UInt32(n_bits))
    _put_u32_le(out, 0)
    for b in tw.out:
        out.append(b)
    for b in dw.out:
        out.append(b)
    return out^


def _write_f21_exr_header(mut out: List[UInt8], w: Int, h: Int, compression: Int, tiled: Bool):
    _put_u32_le(out, 20000630)
    _put_u32_le(out, UInt32(2 | (0x200 if tiled else 0)))
    for attr in ["channels", "chlist"]:
        for b in attr.as_bytes():
            out.append(b)
        out.append(0)
    _put_u32_le(out, 3 * 18 + 1)
    for ch in ["B", "G", "R"]:
        for b in ch.as_bytes():
            out.append(b)
        out.append(0)
        _put_u32_le(out, 1)  # HALF
        out.append(1); out.append(0); out.append(0); out.append(0)
        _put_u32_le(out, 1); _put_u32_le(out, 1)
    out.append(0)
    for attr in ["compression", "compression"]:
        for b in attr.as_bytes():
            out.append(b)
        out.append(0)
    _put_u32_le(out, 1)
    out.append(UInt8(compression))
    for win in ["dataWindow", "displayWindow"]:
        for b in win.as_bytes():
            out.append(b)
        out.append(0)
        for b in "box2i".as_bytes():
            out.append(b)
        out.append(0)
        _put_u32_le(out, 16)
        _put_u32_le(out, 0); _put_u32_le(out, 0)
        _put_u32_le(out, UInt32(w - 1)); _put_u32_le(out, UInt32(h - 1))
    if tiled:
        for attr in ["tiles", "tiledesc"]:
            for b in attr.as_bytes():
                out.append(b)
            out.append(0)
        _put_u32_le(out, 9)
        _put_u32_le(out, UInt32(w)); _put_u32_le(out, UInt32(h)); out.append(0)
    out.append(0)


def _build_f21_tiled_exr() -> List[UInt8]:
    """Valid 4x4 tiled OpenEXR v2 image encoding RGB (0.75, 0.5, 0.25)."""
    var out = List[UInt8]()
    _write_f21_exr_header(out, 4, 4, EXR_NO_COMPRESSION, tiled=True)
    var tbl_off = len(out) + 8
    _put_u64_le(out, UInt64(tbl_off))
    _put_u32_le(out, 0); _put_u32_le(out, 0); _put_u32_le(out, 0); _put_u32_le(out, 0)
    var px_bytes = 4 * 4 * 3 * 2
    _put_u32_le(out, UInt32(px_bytes))
    var h_b = f32_to_uf16_bits(Float32(0.25))
    var h_g = f32_to_uf16_bits(Float32(0.5))
    var h_r = f32_to_uf16_bits(Float32(0.75))
    for _ in range(4):
        for _ in range(4):
            _put_u16_le(out, h_b)
        for _ in range(4):
            _put_u16_le(out, h_g)
        for _ in range(4):
            _put_u16_le(out, h_r)
    return out^


def _build_f21_piz_exr() -> List[UInt8]:
    """Genuine 4x4 PIZ OpenEXR v2 image (bitmap + LUT + 2D Haar wavelet + canonical Huffman) for RGB (2.0, 0.5, 0.125)."""
    var out = List[UInt8]()
    _write_f21_exr_header(out, 4, 4, EXR_PIZ_COMPRESSION, tiled=False)
    var tbl_pos = len(out)
    _put_u64_le(out, 0)
    var chunk_off = UInt64(len(out))
    for bi in range(8):
        out[tbl_pos + bi] = UInt8((chunk_off >> UInt64(bi * 8)) & 0xFF)

    var piz_u16 = List[UInt16]()
    var h_b = UInt16(f32_to_uf16_bits(Float32(0.125)))
    var h_g = UInt16(f32_to_uf16_bits(Float32(0.5)))
    var h_r = UInt16(f32_to_uf16_bits(Float32(2.0)))
    for _ in range(16):
        piz_u16.append(h_b)
    for _ in range(16):
        piz_u16.append(h_g)
    for _ in range(16):
        piz_u16.append(h_r)

    var bm = List[UInt8](length=8192, fill=0)
    for v in piz_u16:
        if v != 0:
            bm[Int(v) >> 3] |= UInt8(1 << (Int(v) & 7))
    var min_nz = 8191
    var max_nz = 0
    for bi in range(8192):
        if bm[bi] != 0:
            if bi < min_nz:
                min_nz = bi
            if bi > max_nz:
                max_nz = bi
    var inv_lut = List[UInt16](length=65536, fill=0)
    var lk = 0
    for i in range(65536):
        if i == 0 or (bm[i >> 3] & UInt8(1 << (i & 7))) != 0:
            inv_lut[i] = UInt16(lk)
            lk += 1
    for i in range(len(piz_u16)):
        piz_u16[i] = inv_lut[Int(piz_u16[i])]
    for ci in range(3):
        _exr_wav2_encode(piz_u16, ci * 16, 4, 1, 4, 4)

    var huf_blob = _build_exr_huf_blob(piz_u16)
    var piz_chunk = List[UInt8]()
    _put_u16_le(piz_chunk, min_nz)
    _put_u16_le(piz_chunk, max_nz)
    for bi in range(min_nz, max_nz + 1):
        piz_chunk.append(bm[bi])
    _put_u32_le(piz_chunk, UInt32(len(huf_blob)))
    for b in huf_blob:
        piz_chunk.append(b)

    _put_u32_le(out, 0)
    _put_u32_le(out, UInt32(len(piz_chunk)))
    for b in piz_chunk:
        out.append(b)
    return out^


def _build_f21_dwaa_exr() -> List[UInt8]:
    """Genuine 8x8 DWAA OpenEXR v2 image (DWA v2 header + CSC rules + zlib DC + AC EOB) for RGB (0.5, 0.25, 0.05)."""
    var out = List[UInt8]()
    _write_f21_exr_header(out, 8, 8, EXR_DWAA_COMPRESSION, tiled=False)
    var tbl_pos = len(out)
    _put_u64_le(out, 0)
    var chunk_off = UInt64(len(out))
    for bi in range(8):
        out[tbl_pos + bi] = UInt8((chunk_off >> UInt64(bi * 8)) & 0xFF)

    var r_nl = exp(log(Float32(0.5)) / Float32(2.2))
    var g_nl = exp(log(Float32(0.25)) / Float32(2.2))
    var b_nl = exp(log(Float32(0.05)) / Float32(2.2))
    var y_nl = Float32(0.2126) * r_nl + Float32(0.7152) * g_nl + Float32(0.0722) * b_nl
    var cb_nl = (b_nl - y_nl) / Float32(1.8556)
    var cr_nl = (r_nl - y_nl) / Float32(1.5747)

    var dc_bytes = List[UInt8]()
    for v_dc in [y_nl * Float32(8.0), cb_nl * Float32(8.0), cr_nl * Float32(8.0)]:
        var hv = f32_to_uf16_bits(v_dc) if v_dc >= 0 else (0x8000 | f32_to_uf16_bits(-v_dc))
        _put_u16_le(dc_bytes, hv)
    var dc_comp = _wrap_zlib_stored(_exr_reorder_and_predict(dc_bytes))

    var ac_bytes = List[UInt8]()
    for _ in range(3):
        _put_u16_le(ac_bytes, 0xFF00)
    var ac_comp = _wrap_zlib_stored(ac_bytes)

    var dwa_chunk = List[UInt8]()
    _put_u64_le(dwa_chunk, 2)  # version 2
    _put_u64_le(dwa_chunk, 0); _put_u64_le(dwa_chunk, 0)  # unknown uncomp/comp
    _put_u64_le(dwa_chunk, UInt64(len(ac_comp)))
    _put_u64_le(dwa_chunk, UInt64(len(dc_comp)))
    _put_u64_le(dwa_chunk, 0); _put_u64_le(dwa_chunk, 0); _put_u64_le(dwa_chunk, 0)  # rle
    _put_u64_le(dwa_chunk, 3)  # total_ac_cnt
    _put_u64_le(dwa_chunk, 3)  # total_dc_cnt
    _put_u64_le(dwa_chunk, 1)  # ac_compression = DEFLATE
    _put_u16_le(dwa_chunk, 11)
    dwa_chunk.append(UInt8(ord("R"))); dwa_chunk.append(0); dwa_chunk.append(0x14)
    dwa_chunk.append(UInt8(ord("G"))); dwa_chunk.append(0); dwa_chunk.append(0x24)
    dwa_chunk.append(UInt8(ord("B"))); dwa_chunk.append(0); dwa_chunk.append(0x34)
    for b in ac_comp:
        dwa_chunk.append(b)
    for b in dc_comp:
        dwa_chunk.append(b)

    _put_u32_le(out, 0)
    _put_u32_le(out, UInt32(len(dwa_chunk)))
    for b in dwa_chunk:
        out.append(b)
    return out^


def _build_f18_vat_model() -> NativeModel:
    """Minimal skinned/node NativeModel + clip for VAT crowd baking in F18."""
    var m = NativeModel()
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(0.0, 0.5, 0.0), Vec3(0.3, 0.5, 0.3))
    m.meshes.append(mb^.finish())
    var anim = NativeAnimation()
    anim.name = "Idle"
    anim.duration = Float32(0.5)
    var ch = NativeChannel()
    ch.node = 0
    ch.path = 0
    ch.times = [Float32(0.0), Float32(0.5)]
    ch.values = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.2), Float32(0.0)]
    anim.channels.append(ch^)
    m.animations.append(anim^)
    return m^


# ===========================================================================
# Contract Tests F15–F28
# ===========================================================================


def test_f15_f16_f17_f18_f19_aaa_renderer_and_graph() raises:
    # F16: AaaRenderGraph aliased offset computation & cross-queue barriers
    var rg = AaaRenderGraph()
    rg.begin_frame()
    var r_vis = rg.add_resource("visbuffer", 4 * 1024 * 1024, transient=True, is_image=True)
    var r_gi = rg.add_resource("gi_scratch", 4 * 1024 * 1024, transient=True, is_image=False)
    var r_post = rg.add_resource("post_scratch", 4 * 1024 * 1024, transient=True, is_image=True)

    var a0 = List[RgResourceAccess]()
    a0.append(RgResourceAccess(r_vis, RG_USAGE_COLOR_ATT_WRITE))
    _ = rg.add_pass("vis_pass", a0, RG_QUEUE_GRAPHICS)

    var a1 = List[RgResourceAccess]()
    a1.append(RgResourceAccess(r_vis, RG_USAGE_FRAGMENT_READ))
    a1.append(RgResourceAccess(r_gi, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("gi_async", a1, RG_QUEUE_ASYNC_COMPUTE)

    var a2 = List[RgResourceAccess]()
    a2.append(RgResourceAccess(r_gi, RG_USAGE_COMPUTE_READ))
    a2.append(RgResourceAccess(r_post, RG_USAGE_COLOR_ATT_WRITE))
    _ = rg.add_pass("post_pass", a2, RG_QUEUE_GRAPHICS)

    assert_true(rg.compile(32 * 1024 * 1024))
    assert_true(rg.aliasing_savings_ratio() > Float32(0.0))
    assert_true(len(rg.barriers) >= 1)

    # F17, F18 & F19: Split VisBuffer Material-ID tile classify + shade, VAT/clipmap & direct swapchain SPIR-V
    var cls_rep = validate_spirv_module(aaa_tile_classify_shader())
    assert_true(cls_rep.valid)
    var shd_rep = validate_spirv_module(aaa_deferred_shade_class_shader(False, AAA_SHADE_CLASS_STANDARD_PBR, use_tile_list=True))
    assert_true(shd_rep.valid)
    var vat_rep = validate_spirv_module(vat_crowd_eval_shader())
    assert_true(vat_rep.valid)
    var tonemap_direct_rep = validate_spirv_module(aaa_display_tonemap_direct_shader())
    assert_true(tonemap_direct_rep.valid)
    var ui_direct_rep = validate_spirv_module(aaa_ui_overlay_direct_shader())
    assert_true(ui_direct_rep.valid)
    var clip = TerrainClipmapDesc()
    assert_true(clip.ring_count > 0)
    assert_equal(len(clip.pack_words()), 12)

    # F15, F16, F18, F19, F22: VulkanGpu device-local / staging ring / aliased heaps / submit_side + Renderer
    var vk = VulkanGpu()
    if vk.initialize():
        assert_true(vk.allocate_command_buffers(1))

        # F15: DEVICE_LOCAL buffer has cpu_address == 0 + LinearStagingRing copy
        var dev_buf = vk.create_buffer_device_local(4096, BUFFER_USAGE, bda=False)
        assert_true(dev_buf.vk_buffer != 0)
        assert_equal(dev_buf.cpu_address, 0)

        var ring = LinearStagingRing()
        assert_true(ring.init(vk, 65536))
        var stage_payload = List[UInt8](length=256, fill=0x5A)
        var slice = ring.stage_bytes(vk, 0, Int(stage_payload.unsafe_ptr()), 256)
        assert_true(slice.cpu_address != 0)
        assert_equal(slice.size, 256)
        assert_true(vk.begin_commands(0))
        ring.record_copy(vk, 0, 0, slice, dev_buf, 0, 256)
        var copy_fence = vk.submit(0)
        assert_true(copy_fence != 0 and vk.wait(copy_fence))
        ring.reset_slot(vk, 0)
        ring.shutdown(vk)
        vk.destroy_buffer(dev_buf)

        # F16: Aliased buffer & image heap bindings + submit_side
        var req0 = VkMemReq()
        var req1 = VkMemReq()
        var b0 = vk.create_buffer_unbound(4096, req0)
        var b1 = vk.create_buffer_unbound(4096, req1)
        var heap = vk.create_aliased_heap(65536, req0.memory_type_bits & req1.memory_type_bits)
        assert_true(heap != 0)
        assert_true(vk.bind_buffer_aliased(b0, heap, 0))
        assert_true(vk.bind_buffer_aliased(b1, heap, 0))
        assert_true(b0.vk_buffer != 0 and b1.vk_buffer != 0)
        vk.destroy_buffer(b0)
        vk.destroy_buffer(b1)
        vk.destroy_aliased_heap(heap)

        var ireq0 = VkMemReq()
        var ireq1 = VkMemReq()
        var img0 = vk.create_image_2d_unbound(
            64, 64, FORMAT_R8G8B8A8_UNORM, IMAGE_USAGE_SAMPLED | IMAGE_USAGE_TRANSFER_TARGET, ireq0
        )
        var img1 = vk.create_image_2d_unbound(
            64, 64, FORMAT_R8G8B8A8_UNORM, IMAGE_USAGE_SAMPLED | IMAGE_USAGE_TRANSFER_TARGET, ireq1
        )
        var iheap_sz = max(65536, max(ireq0.size, ireq1.size))
        var iheap = vk.create_aliased_heap(iheap_sz, ireq0.memory_type_bits & ireq1.memory_type_bits)
        assert_true(iheap != 0)
        assert_true(vk.bind_image_aliased(img0, iheap, 0))
        assert_true(vk.bind_image_aliased(img1, iheap, 0))
        assert_true(img0.image != 0 and img0.view != 0 and img1.image != 0 and img1.view != 0)
        vk.destroy_image(img0)
        vk.destroy_image(img1)
        vk.destroy_aliased_heap(iheap)

        var side_slot = vk.allocate_side_command_buffers(1)
        assert_true(side_slot >= 0)
        assert_true(vk.begin_commands(side_slot))
        var side_fence = vk.submit_side(side_slot)
        assert_true(side_fence != 0)
        if vk.has_side_queue():
            assert_true(vk.wait_side(side_fence))
        else:
            assert_true(vk.wait(side_fence))

        var renderer = FlintAaaVulkanRenderer(
            width=64,
            height=64,
            tsr_scale=1.0,
            max_draws=256,
            max_instances=256,
            env_cube_dim=16,
        )
        if renderer.active:
            assert_equal(renderer.buf_vertices.cpu_address, 0)
            assert_equal(renderer.buf_indices.cpu_address, 0)
            var mat_cap0 = renderer.buf_materials.size
            var ib_cap0 = renderer.buf_indices.size
            var last_mat = 0
            for _ in range(260):
                last_mat = renderer.add_material(AaaMaterial(Vec3(0.3, 0.6, 0.8), roughness=0.4, metallic=0.1))
            assert_true(renderer.buf_materials.size > mat_cap0)

            # F19: Grow vertex geometry pool beyond legacy 32 MiB cap (520 x 256-seg cylinders)
            var mb_big = MeshBuilder()
            for _ in range(520):
                mb_big.add_cylinder(Vec3(0.0, -0.4, 0.0), Vec3(0.0, 0.4, 0.0), 0.5, 0.5, segments=256)
            var big_entry = renderer.add_mesh(mb_big^.finish(), last_mat)
            assert_true(renderer.buf_vertices.size > 32 * 1024 * 1024)
            assert_true(renderer.buf_indices.size > ib_cap0)

            # F18: Wire TerrainClipmapDesc and VAT crowd clip into FlintAaaVulkanRenderer
            renderer.set_terrain_clipmap(clip, last_mat)
            var vat_model = _build_f18_vat_model()
            var vat_clip = bake_vat_crowd_clip_rgba16f(vat_model, mesh_index=0, clip_index=0)
            var vat_cid = renderer.register_vat_crowd_clip(vat_clip, vat_model.meshes[0], last_mat)
            assert_true(vat_cid >= 0)

            renderer.begin_frame(cam_pos=Vec3(0.0, 0.6, 2.5), cam_target=Vec3(0.0, 0.0, 0.0))
            renderer.draw_instance(big_entry.mesh_id, Mat4())
            renderer.draw_vat_crowd_instance(vat_cid, Float32(0.15), Float32(0.0), Mat4())
            assert_true(renderer.render_frame(capture_readback=True))
        renderer.shutdown()
        vk.close()
    else:
        vk.close()


def test_f20_residency_streamer_vram_tracking() raises:
    var srv = AssetServer(stream_budget_bytes_per_tick=65536, enable_watcher=False)
    var ren = FlintAaaVulkanRenderer(
        width=64,
        height=64,
        tsr_scale=1.0,
        max_draws=64,
        max_instances=64,
        env_cube_dim=16,
    )
    if ren.active:
        srv.attach_renderer_residency_bridge(ren)
        var tex = CookedTextureAsset()
        tex.width = 4
        tex.height = 4
        tex.mip_count = 1
        tex.tex_format = TEX_FMT_BC7_SRGB
        tex.mip_sizes[0] = 16
        tex.mips.append(List[UInt8](length=16, fill=0xAB))
        var slot = srv.textures.insert(
            tex^, AssetUuid.from_path("residency/real_tex.png"), "residency/real_tex.png", 1
        )
        assert_true(slot.is_valid())
        assert_true(srv.upload_mip_to_gpu(slot, mip_level=0))
        assert_true(srv.total_vram_live_bytes() > 0)
        assert_true(srv.textures.slot_vram_bytes[Int(slot.slot())] > 0)
        _ = srv.textures.release(slot)
        _ = srv.textures.evict_lru_unreferenced(0)
        assert_equal(srv.total_vram_live_bytes(), 0)
    srv.close()
    ren.shutdown()


def test_f21_codecs_glb_jpeg_exr() raises:
    var glb = _build_f21_extended_glb()
    var mdl = parse_native_glb(glb)
    assert_true(mdl.loaded())
    assert_equal(len(mdl.meshes), 1)
    assert_true(len(mdl.meshes[0].tangents) > 0)
    assert_true(len(mdl.meshes[0].texcoords1) > 0)
    assert_true(mdl.meshes[0].occlusion_texture >= 0)
    assert_true(mdl.meshes[0].emissive_texture >= 0)
    assert_true(len(mdl.textures) >= 1)
    assert_equal(mdl.textures[0].width, 4)
    assert_equal(mdl.textures[0].height, 4)
    assert_equal(mdl.textures[0].pixels[0], UInt32(0xFF4080C0))

    var sof2 = _build_f21_minimal_sof2_jpeg()
    var j_img = decode_jpeg(sof2)
    assert_true(j_img.valid)
    assert_equal(j_img.width, 8)
    assert_equal(j_img.height, 8)

    var tiled_exr = _build_f21_tiled_exr()
    var piz_exr = _build_f21_piz_exr()
    var dwaa_exr = _build_f21_dwaa_exr()
    var d_tiled = decode_exr_fp16(tiled_exr)
    assert_true(d_tiled.valid)
    assert_equal(d_tiled.width, 4)
    assert_equal(d_tiled.height, 4)
    assert_true(abs(d_tiled.pixels[0].x - Float32(0.75)) < Float32(1e-2))
    assert_true(abs(d_tiled.pixels[0].y - Float32(0.5)) < Float32(1e-2))
    assert_true(abs(d_tiled.pixels[0].z - Float32(0.25)) < Float32(1e-2))

    var d_piz = decode_exr_fp16(piz_exr)
    assert_true(d_piz.valid)
    assert_equal(d_piz.width, 4)
    assert_equal(d_piz.height, 4)
    assert_true(abs(d_piz.pixels[0].x - Float32(2.0)) < Float32(1e-2))
    assert_true(abs(d_piz.pixels[0].y - Float32(0.5)) < Float32(1e-2))
    assert_true(abs(d_piz.pixels[0].z - Float32(0.125)) < Float32(1e-2))

    # Negative PIZ corruption check (flipped Huffman payload byte)
    var bad_piz = piz_exr.copy()
    bad_piz[len(bad_piz) - 1] ^= 0xFF
    assert_false(decode_exr_fp16(bad_piz).valid)

    var d_dwaa = decode_hdr_or_exr(dwaa_exr)
    assert_true(d_dwaa.valid)
    assert_equal(d_dwaa.width, 8)
    assert_equal(d_dwaa.height, 8)
    assert_true(abs(d_dwaa.pixels[0].x - Float32(0.5)) < Float32(0.02))
    assert_true(abs(d_dwaa.pixels[0].y - Float32(0.25)) < Float32(0.02))
    assert_true(abs(d_dwaa.pixels[0].z - Float32(0.05)) < Float32(0.02))

    # Negative DWAA corruption checks: zlib stored-block NLEN mismatch & corrupted DC payload
    var bad_dwaa_nlen = dwaa_exr.copy()
    bad_dwaa_nlen[len(bad_dwaa_nlen) - 13] ^= 0xFF
    assert_false(decode_hdr_or_exr(bad_dwaa_nlen).valid)

    var bad_dwaa_dc = dwaa_exr.copy()
    bad_dwaa_dc[len(bad_dwaa_dc) - 6] ^= 0xFF
    var d_dwaa_bad_dc = decode_hdr_or_exr(bad_dwaa_dc)
    assert_true(
        (not d_dwaa_bad_dc.valid) or abs(d_dwaa_bad_dc.pixels[0].x - Float32(0.5)) > Float32(0.05)
    )


def test_f22_f28_raii_ownership_checks() raises:
    # M3 F22: GPU/Vulkan RAII (non-Copyable on image/AS/memblock/surface; Movable + __deinit__ on all 8,
    # while VulkanGpu/GpuBuffer/AmdGpu/SwapchainHandle drop Copyable in M6 per PROJECT.md § F22)
    comptime assert conforms_to(VulkanImage, Movable)
    comptime assert not conforms_to(VulkanImage, Copyable)
    comptime assert conforms_to(VulkanAccelStruct, Movable)
    comptime assert not conforms_to(VulkanAccelStruct, Copyable)
    comptime assert conforms_to(VulkanMemBlock, Movable)
    comptime assert not conforms_to(VulkanMemBlock, Copyable)
    comptime assert conforms_to(SwapchainSurface, Movable)
    comptime assert not conforms_to(SwapchainSurface, Copyable)
    comptime assert conforms_to(VulkanGpu, Movable)
    comptime assert conforms_to(GpuBuffer, Movable)
    comptime assert conforms_to(AmdGpu, Movable)
    comptime assert conforms_to(SwapchainHandle, Movable)

    # M4 F28: Platform handle RAII (non-Copyable, Movable)
    comptime assert conforms_to(WaylandWindow, Movable)
    comptime assert not conforms_to(WaylandWindow, Copyable)
    comptime assert conforms_to(WaylandBridge, Movable)
    comptime assert not conforms_to(WaylandBridge, Copyable)
    comptime assert conforms_to(X11Window, Movable)
    comptime assert not conforms_to(X11Window, Copyable)
    comptime assert conforms_to(KmsWindow, Movable)
    comptime assert not conforms_to(KmsWindow, Copyable)
    comptime assert conforms_to(LogindSession, Movable)
    comptime assert not conforms_to(LogindSession, Copyable)
    comptime assert conforms_to(PulseClient, Movable)
    comptime assert not conforms_to(PulseClient, Copyable)
    comptime assert conforms_to(NativeAudio, Movable)
    comptime assert not conforms_to(NativeAudio, Copyable)
    comptime assert conforms_to(LinuxGamepad, Movable)
    comptime assert not conforms_to(LinuxGamepad, Copyable)
    comptime assert conforms_to(LockstepPeer, Movable)
    comptime assert not conforms_to(LockstepPeer, Copyable)
    comptime assert conforms_to(RollbackPeer, Movable)
    comptime assert not conforms_to(RollbackPeer, Copyable)

    var s1 = LogindSession()
    var s2 = s1^
    _ = s2^


def test_f23_f24_f25_window_and_evdev_parity() raises:
    comptime assert conforms_to(WaylandWindow, WindowBackend)
    comptime assert conforms_to(X11Window, WindowBackend)
    comptime assert conforms_to(KmsWindow, WindowBackend)

    # F23: Wayland DnD URI list parser, XKB keymap parsing & UTF-8 text input, framebuffer/scale
    var uris = parse_uri_list("file:///tmp/hello%20world.txt\r\n# comment\r\n")
    assert_equal(len(uris), 1)
    assert_equal(uris[0], "/tmp/hello world.txt")

    var km = XkbKeymap()
    var xkb_src = String(
        'xkb_keymap {\n'
        '  xkb_keycodes "evdev" { <AC01> = 38; <AD01> = 24; };\n'
        '  xkb_symbols "pc" {\n'
        '    key <AC01> { [ a, A ] };\n'
        '    key <AD01> { [ q, Q ] };\n'
        '  };\n'
        '};\n'
    )
    assert_true(km.parse(xkb_src))
    assert_equal(km.translate_key_to_utf8(30), "a")
    km.set_modifiers(XKB_MOD_SHIFT, 0, 0, 0)
    assert_equal(km.translate_key_to_utf8(30), "A")
    assert_equal(x11_keysym_to_codepoint(0x20AC), 0x20AC)

    var inp = InputState()
    inp.begin_frame()
    inp.push_text_utf8(km.translate_key_to_utf8(30))
    assert_equal(inp.text_input_utf8, "A")

    var wl = WaylandWindow()
    wl.clipboard_set("flint-wl-clip")
    assert_equal(wl.clipboard_get(), "flint-wl-clip")
    var wl_fb = wl.framebuffer_size()
    assert_true(wl_fb[0] > 0 and wl_fb[1] > 0)
    assert_true(wl.content_scale() >= Float32(1.0))

    # F24: X11 framebuffer/scale + null-safe attach_vulkan + clipboard
    var x11 = X11Window()
    x11.clipboard_set("flint-x11-clip")
    assert_equal(x11.clipboard_get(), "flint-x11-clip")
    var x11_fb = x11.framebuffer_size()
    assert_true(x11_fb[0] > 0 and x11_fb[1] > 0)
    assert_true(x11.content_scale() >= Float32(1.0))
    var vk_null = VulkanGpu()
    var sc_null = x11.attach_vulkan(vk_null, 640, 480)
    assert_equal(sc_null.vk_swapchain, 0)
    vk_null.close()

    # F25: Direct EvdevHub axis calibration (trigger rest at 0.0) & poll_into
    var hub = EvdevHub(enable_hotplug=False)
    hub.pads[0].connected = True
    hub.pads[0].configure_abs_axis(ABS_Z, 0, 1023, fuzz=0, flat=0, initial_raw=0)
    hub.poll_into(inp)
    assert_equal(inp.gamepad_axis(0, PAD_AXIS_LT), Float32(0.0))
    assert_false(inp.uses_legacy_jsdev())
    hub.close()


def test_f26_text_and_ui_parity() raises:
    # Real TAG_OTTO CFF OpenType font + non-empty MSDF/MTSDF baking
    var otf_bytes = read_bytes("/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Regular.otf")
    assert_true(len(otf_bytes) > 0)
    var otf = TrueTypeFont()
    assert_true(otf.load_bytes(otf_bytes^))
    assert_true(otf.valid)
    assert_true(otf.is_cff)
    assert_true(otf.has_glyph(65))
    assert_true(otf.glyph_id(65) > 0)

    var msdf = bake_glyph_msdf(otf, 65, 32, 4)
    assert_true(msdf.w > 0 and msdf.h > 0)
    assert_equal(msdf.channels, 3)
    assert_equal(len(msdf.pixels), msdf.w * msdf.h * 3)
    var min_p = 255
    var max_p = 0
    for b in msdf.pixels:
        var v = Int(b)
        if v < min_p:
            min_p = v
        if v > max_p:
            max_p = v
    assert_true(min_p < max_p)

    var mtsdf = bake_glyph_mtsdf(otf, 65, 32, 4)
    assert_true(mtsdf.w > 0 and mtsdf.h > 0)
    assert_equal(mtsdf.channels, 4)
    assert_equal(len(mtsdf.pixels), mtsdf.w * mtsdf.h * 4)

    # GSUB shaping call, FontFallbackChain, wrap_shaped & RTL BiDi level resolution
    var mono_bytes = read_bytes("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
    assert_true(len(mono_bytes) > 0)
    var mono = TrueTypeFont()
    assert_true(mono.load_bytes(mono_bytes^))
    assert_true(mono.valid)

    var gsub_glyphs = List[GsubGlyph]()
    gsub_glyphs.append(GsubGlyph(otf.glyph_id(102), 0))
    gsub_glyphs.append(GsubGlyph(otf.glyph_id(105), 1))
    otf.apply_gsub(gsub_glyphs, "latn")
    assert_true(len(gsub_glyphs) >= 1)

    var chain = FontFallbackChain()
    chain.set_primary(otf.copy())
    chain.add_fallback(mono^)
    assert_equal(chain.count(), 2)
    var shaped_lines = wrap_shaped(chain, "Hello World", Float32(16.0), Float32(200.0))
    assert_true(len(shaped_lines) >= 1)
    assert_true(len(shaped_lines[0].glyphs) >= 10)
    assert_true(shaped_lines[0].width > Float32(0.0))

    var rtl_spans = parse_rich("שלום")
    assert_true(len(rtl_spans) >= 1)
    var bidi_res = resolve_bidi(decode_utf8("שלום world"), BIDI_AUTO)
    assert_equal(bidi_res.base_level, 1)

    # UiContext.text_input: mouse-click caret placement, Ctrl+A/C/X/V clipboard & IME preedit
    var ui = UiContext()
    var inp = InputState()
    var txt = String("hello")

    # Frame 1: click inside box around 16 px into text -> caret inside [1..4]
    inp.begin_frame()
    inp.move_mouse(Float32(24.0), Float32(10.0))
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("ime_box", txt, 240.0)
    ui.end_frame()
    var mid_cp = ui.text_cursor_codepoint("ime_box")
    assert_true(mid_cp >= 1 and mid_cp <= 4)

    # Click at left padding -> caret moves to 0
    inp.begin_frame()
    inp.set_mouse(0, False)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("ime_box", txt, 240.0)
    ui.end_frame()

    inp.begin_frame()
    inp.move_mouse(Float32(8.0), Float32(10.0))
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("ime_box", txt, 240.0)
    ui.end_frame()
    assert_equal(ui.text_cursor_codepoint("ime_box"), 0)

    # Ctrl+A then Ctrl+C
    inp.begin_frame()
    inp.set_mouse(0, False)
    inp.set_key(UI_KEY_LEFT_CTRL, True)
    inp.set_key(UI_KEY_A, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("ime_box", txt, 240.0)
    ui.end_frame()
    assert_equal(ui.text_selection_range_cp("ime_box")[0], 0)
    assert_equal(ui.text_selection_range_cp("ime_box")[1], 5)

    inp.begin_frame()
    inp.set_key(UI_KEY_A, False)
    inp.set_key(UI_KEY_C, True)
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("ime_box", txt, 240.0)
    ui.end_frame()
    assert_equal(ui.get_clipboard(), "hello")

    # Ctrl+X cuts text, Ctrl+V pastes it back
    inp.begin_frame()
    inp.set_key(UI_KEY_C, False)
    inp.set_key(UI_KEY_X, True)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_true(ui.text_input("ime_box", txt, 240.0))
    ui.end_frame()
    assert_equal(txt, "")
    assert_equal(ui.get_clipboard(), "hello")

    inp.begin_frame()
    inp.set_key(UI_KEY_X, False)
    inp.set_key(UI_KEY_V, True)
    ui.begin_frame(inp, 800.0, 600.0)
    assert_true(ui.text_input("ime_box", txt, 240.0))
    ui.end_frame()
    assert_equal(txt, "hello")

    # IME preedit composition
    inp.begin_frame()
    inp.set_key(UI_KEY_V, False)
    inp.set_key(UI_KEY_LEFT_CTRL, False)
    inp.set_ime_preedit("nihon")
    ui.begin_frame(inp, 800.0, 600.0)
    _ = ui.text_input("ime_box", txt, 240.0)
    ui.end_frame()
    assert_equal(ui.active_ime_preedit("ime_box"), "nihon")


def test_f27_win32_macos_cross_platform_backends() raises:
    comptime assert conforms_to(Win32Window, WindowBackend)
    comptime assert conforms_to(CocoaWindow, WindowBackend)
    comptime assert conforms_to(Win32Audio, AudioBackend)
    comptime assert conforms_to(MacOsAudio, AudioBackend)

    # Win32 & macOS protocol/encoding/coordinate/keymap helpers
    var wide = utf8_to_utf16le("Hi★", True)
    assert_equal(utf16le_to_utf8(wide), "Hi★")
    assert_equal(cocoa_flip_y(100.0, 720), Float32(620.0))
    assert_equal(win32_vk_to_flint_key(0x41), 65)
    assert_equal(cocoa_keycode_to_flint_key(0x00), 65)

    var w32 = Win32Window()
    assert_true(w32.content_scale() >= Float32(1.0))
    var mac_w = CocoaWindow()
    assert_true(mac_w.content_scale() >= Float32(1.0))

    # Win32Audio & MacOsAudio lifecycle + offline mixer exercise on Linux
    var wa = Win32Audio()
    assert_false(wa.initialize())
    wa.trigger(1, Float32(1.0), Float32(0.8))
    wa.set_ambient_drone(Float32(0.25))
    wa.pump(Float32(1.0))
    assert_true(len(wa.last_mixed_slot()) > 0)
    wa.shutdown()

    var ma = MacOsAudio()
    assert_false(ma.initialize())
    ma.trigger(1, Float32(1.0), Float32(0.8))
    ma.set_ambient_drone(Float32(0.25))
    ma.mix_into_slot(0, Float32(1.0), 64)
    ma.shutdown()

    var vk = VulkanGpu()
    assert_false(vk.attach_win32_surface(UInt64(0), UInt64(0)).surface != 0)
    assert_false(vk.attach_macos_surface(UInt64(0)).surface != 0)
    vk.close()

    # UdpSocket non-blocking open/bind/recv/close/error checks
    var sock = UdpSocket()
    _ = CompilationTarget.is_linux()
    assert_false(sock.is_valid())
    assert_false(sock.send_to(NetEndpoint.ipv4_loopback(9999), List[UInt8](length=4, fill=1)))
    assert_true(sock.bind(NetEndpoint.ipv4_loopback(0)))
    assert_true(sock.is_valid() and sock.local_port() > 0)
    var rx = List[DatagramView]()
    assert_equal(sock.recv_batch(rx), 0)
    sock.close()
    assert_false(sock.is_valid())


def main() raises:
    var passed = 0
    var total = 7

    test_f15_f16_f17_f18_f19_aaa_renderer_and_graph()
    print("[PASS] test_f15_f16_f17_f18_f19_aaa_renderer_and_graph")
    passed += 1

    test_f20_residency_streamer_vram_tracking()
    print("[PASS] test_f20_residency_streamer_vram_tracking")
    passed += 1

    test_f21_codecs_glb_jpeg_exr()
    print("[PASS] test_f21_codecs_glb_jpeg_exr")
    passed += 1

    test_f22_f28_raii_ownership_checks()
    print("[PASS] test_f22_f28_raii_ownership_checks")
    passed += 1

    test_f23_f24_f25_window_and_evdev_parity()
    print("[PASS] test_f23_f24_f25_window_and_evdev_parity")
    passed += 1

    test_f26_text_and_ui_parity()
    print("[PASS] test_f26_text_and_ui_parity")
    passed += 1

    test_f27_win32_macos_cross_platform_backends()
    print("[PASS] test_f27_win32_macos_cross_platform_backends")
    passed += 1

    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")
