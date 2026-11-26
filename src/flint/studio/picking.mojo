"""Flint Studio $O(1)$ 64-Bit VisBuffer Viewport Entity Picking (`picking.mojo`).

Decodes `FlintAaaVulkanRenderer`'s hardware `VK_FORMAT_R32G32_UINT` VisBuffer (`vis_img`:
`word0 = (inst_id << 17) | (meshlet_id % 131071 + 1)`, `word1 = (mat_id << 16) | (meshlet_id / 131071) << 8 | prim_id`,
see `flint.render.sota_geometry_visid`) and
Reverse-Z `VK_FORMAT_D32_SFLOAT` depth (`depth_img`: `d = near_z / z_view`) into exact
`(entity_id, instance_id, meshlet_id, material_id, local_prim_id, world_pos)` in $O(1)$,
plus a zero-modification 1-pixel GPU compute readback helper for `FlintAaaVulkanRenderer`.
"""
from std.math import max, min, tan
from std.memory import bitcast
from flint.core.math import Vec3
from flint.render.sota_geometry_visid import sota_vis_decode, sota_vis_encode
from flint.platform.vulkan import (
    ACCESS_HOST_READ,
    ACCESS_SHADER_WRITE,
    ACCESS_TRANSFER_READ,
    ACCESS_TRANSFER_WRITE,
    PIPELINE_COMPUTE,
    PIPELINE_HOST,
    PIPELINE_TRANSFER,
    get32,
    get_float,
)
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_visbuffer import (
    AAA_PUSH_WORDS,
    BIND_AAA_DEPTH_TEX,
    BIND_AAA_POST_SCRATCH,
    BIND_AAA_VIS_TEX,
)
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE


@fieldwise_init
struct ViewportPickResult(Copyable, ImplicitlyCopyable, Movable):
    """Decoded $O(1)$ viewport pick result from a single 64-bit VisBuffer + Reverse-Z depth texel."""
    var hit: Bool
    var entity_id: Int
    var instance_id: Int
    var meshlet_id: Int
    var material_id: Int
    var local_prim_id: Int
    var depth_d32: Float32
    var linear_depth_m: Float32
    var world_pos: Vec3

    def __init__(out self):
        self.hit = False
        self.entity_id = -1
        self.instance_id = -1
        self.meshlet_id = -1
        self.material_id = -1
        self.local_prim_id = -1
        self.depth_d32 = Float32(0.0)
        self.linear_depth_m = Float32(0.0)
        self.world_pos = Vec3()


@always_inline
def encode_visbuffer_words(
    instance_id: Int,
    meshlet_id: Int,
    material_id: Int,
    local_prim_id: Int,
) -> Tuple[UInt32, UInt32]:
    """Packs `(instance_id, meshlet_id, material_id, local_prim_id)` using the exact `visbuffer_vertex_shader` layout."""
    return sota_vis_encode(max(0, instance_id), max(0, meshlet_id), material_id & 0xFFFF, local_prim_id)


def decode_visbuffer_pick(
    px: Int,
    py: Int,
    vis_word0: UInt32,
    vis_word1: UInt32,
    depth_d32: Float32,
    width: Int,
    height: Int,
    cam_pos: Vec3,
    cam_fwd: Vec3,
    cam_right: Vec3,
    cam_up: Vec3,
    fov_y_deg: Float32,
    near_z: Float32,
    instance_to_entity: List[Int] = List[Int](),
    jitter_clip_x: Float32 = 0.0,
    jitter_clip_y: Float32 = 0.0,
) -> ViewportPickResult:
    """Decodes `R32G32_UINT` VisBuffer words + Reverse-Z `D32_SFLOAT` depth (`near_z / z_view`)
    into exact entity/meshlet/material/primitive IDs and 3D world-space hit position."""
    var res = ViewportPickResult()
    if vis_word0 == 0 or depth_d32 <= Float32(1.0e-7) or width <= 0 or height <= 0:
        return res

    var raw_ml = Int(vis_word0 & UInt32(0x1FFFF))
    if raw_ml == 0:
        return res

    var ids = sota_vis_decode(vis_word0, vis_word1)   # 24-bit meshlet id (sota:geometry)
    var inst_id = ids[0]
    var ml_id = ids[1]
    var mat_id = ids[2]
    var prim_id = ids[3]
    var ent_id = instance_to_entity[inst_id] if (inst_id >= 0 and inst_id < len(instance_to_entity)) else inst_id

    var aspect = Float32(width) / Float32(max(1, height))
    var fov_rad = fov_y_deg * Float32(0.01745329252)
    var tan_y = tan(fov_rad * Float32(0.5))
    var tan_x = tan_y * aspect

    # Match Vulkan negative-height viewport (`flip_y = True`) + Halton TSR clip jitter in `aaa_gi_svgf.mojo`
    var ndc_x = ((Float32(px) + Float32(0.5)) / Float32(width)) * Float32(2.0) - Float32(1.0) - jitter_clip_x
    var ndc_y = Float32(1.0) - ((Float32(py) + Float32(0.5)) / Float32(height)) * Float32(2.0) - jitter_clip_y

    var z_view = near_z / depth_d32
    var x_view = ndc_x * tan_x * z_view
    var y_view = ndc_y * tan_y * z_view

    var fwd_u = cam_fwd.normalized()
    var rgt_u = cam_right.normalized()
    var up_u = cam_up.normalized()
    var wpos = cam_pos + rgt_u * x_view + up_u * y_view + fwd_u * z_view

    res.hit = True
    res.entity_id = ent_id
    res.instance_id = inst_id
    res.meshlet_id = ml_id
    res.material_id = mat_id
    res.local_prim_id = prim_id
    res.depth_d32 = depth_d32
    res.linear_depth_m = z_view
    res.world_pos = wpos
    return res


def visbuffer_pick_readback_shader() -> List[UInt32]:
    """1x1 Compute shader fetching `(vis_word0, vis_word1)` from `BIND_AAA_VIS_TEX` and `depth_d32`
    from `BIND_AAA_DEPTH_TEX` into `BIND_AAA_POST_SCRATCH[out_word_offset + 0..2]`."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=1, group_height=1, push_words=AAA_PUSH_WORDS)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var vis_samp = s.sampled_image_2d(BIND_AAA_VIS_TEX, is_uint=True)
    var depth_samp = s.sampled_image_2d(BIND_AAA_DEPTH_TEX)

    var px = s.push_word(0)
    var py = s.push_word(1)
    var out_off = s.push_word(2)

    var coord = s.uvec2(px, py)
    var vis_uv4 = s.texel_fetch_u2d(vis_samp, coord, s.uint(0))
    var d_fv4 = s.texel_fetch_f2d(depth_samp, coord, s.uint(0))

    s.store_word(b_post, out_off, s.extract_u(vis_uv4, 0))
    s.store_word(b_post, s.iadd(out_off, s.uint(1)), s.extract_u(vis_uv4, 1))
    s.store_word(b_post, s.iadd(out_off, s.uint(2)), s.bitcast_uint(s.extract_f(d_fv4, 0)))
    return s.finish()


struct VisBufferGpuPicker(Movable):
    """Reusable GPU 1-pixel VisBuffer + Reverse-Z depth readback helper for `FlintAaaVulkanRenderer`."""
    var pipeline: Int

    def __init__(out self):
        self.pipeline = 0

    def shutdown(mut self, mut ren: FlintAaaVulkanRenderer):
        if self.pipeline != 0 and ren.vk.device != 0:
            ren.vk.destroy_pipeline(self.pipeline)
            self.pipeline = 0

    def read_raw_pixel(
        mut self,
        mut ren: FlintAaaVulkanRenderer,
        out_px: Int,
        out_py: Int,
    ) -> Tuple[UInt32, UInt32, Float32, Int, Int]:
        """Reads 1 pixel from `ren.vis_img` + `ren.depth_img` corresponding to output pixel `(out_px, out_py)`."""
        if not ren.active or ren.render_w <= 0 or ren.render_h <= 0:
            return (UInt32(0), UInt32(0), Float32(0.0), 0, 0)

        ren.flush_deferred_frame()
        if self.pipeline == 0:
            self.pipeline = ren.vk.create_aaa_compute_pipeline(ren.pipe_layout, visbuffer_pick_readback_shader())
            if self.pipeline == 0:
                return (UInt32(0), UInt32(0), Float32(0.0), 0, 0)

        var rx = max(0, min(ren.render_w - 1, (out_px * ren.render_w) // max(1, ren.output_w)))
        var ry = max(0, min(ren.render_h - 1, (out_py * ren.render_h) // max(1, ren.output_h)))
        var scratch_word_off = 512  # safe idle word slot in buf_post_scratch after frame completion
        var rb_byte_off = 32        # unused header bytes [32..44] in buf_ldr_readback (pixels start at 64)

        if not ren.vk.begin_commands(0):
            return (UInt32(0), UInt32(0), Float32(0.0), rx, ry)

        var push = List[UInt32](length=AAA_PUSH_WORDS, fill=0)
        push[0] = UInt32(rx)
        push[1] = UInt32(ry)
        push[2] = UInt32(scratch_word_off)

        ren.vk.dispatch_aaa(0, self.pipeline, ren.pipe_layout, ren.desc_set, push, 1, 1)
        ren.vk.memory_barrier2(0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ))
        ren.vk.copy_buffer(0, ren.buf_post_scratch, scratch_word_off * 4, ren.buf_ldr_readback, rb_byte_off, 12)
        ren.vk.memory_barrier2(0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))

        var fence = ren.vk.submit(0)
        if fence != 0:
            _ = ren.vk.wait(fence)

        var base_addr = ren.buf_ldr_readback.cpu_address + rb_byte_off
        var w0 = get32(base_addr + 0)
        var w1 = get32(base_addr + 4)
        var d_bits = get32(base_addr + 8)
        var d_f32 = bitcast[DType.float32](d_bits)
        return (w0, w1, d_f32, rx, ry)

    def pick_viewport(
        mut self,
        mut ren: FlintAaaVulkanRenderer,
        out_px: Int,
        out_py: Int,
        instance_to_entity: List[Int] = List[Int](),
    ) -> ViewportPickResult:
        """Executes a 1-pixel GPU VisBuffer + Reverse-Z depth readback and decodes the exact 3D hit."""
        var raw = self.read_raw_pixel(ren, out_px, out_py)
        var jx_clip = Float32(0.0)
        var jy_clip = Float32(0.0)
        if ren.buf_globals.cpu_address != 0:
            jx_clip = get_float(ren.buf_globals.cpu_address + 101 * 4)
            jy_clip = get_float(ren.buf_globals.cpu_address + 102 * 4)
        return decode_visbuffer_pick(
            raw[3],
            raw[4],
            raw[0],
            raw[1],
            raw[2],
            ren.render_w,
            ren.render_h,
            ren.cam_pos,
            ren.cam_fwd,
            ren.cam_right,
            ren.cam_up,
            ren.fov_y_deg,
            ren.near_z,
            instance_to_entity,
            jx_clip,
            jy_clip,
        )


def pick_renderer_viewport_once(
    mut ren: FlintAaaVulkanRenderer,
    out_px: Int,
    out_py: Int,
    instance_to_entity: List[Int] = List[Int](),
) -> ViewportPickResult:
    """Convenience one-shot GPU viewport pick that cleans up its temporary compute pipeline."""
    var picker = VisBufferGpuPicker()
    var res = picker.pick_viewport(ren, out_px, out_py, instance_to_entity)
    picker.shutdown(ren)
    return res
