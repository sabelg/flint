"""Flint Studio $O(1)$ 40-Bit VisBuffer Viewport Entity Picking (`picking.mojo`).

Decodes `FlintAaaVulkanRenderer`'s hardware `VK_FORMAT_R32G32_UINT` VisBuffer (`vis_img`:
`word0 = ((inst_id & 0x3FF) << 22) | min(meshlet_id + 1, 0x3FFFFF)`,
`word1 = (mat_id << 16) | (((inst_id >> 10) & 0xFF) << 8) | (prim_id & 0xFF)`)
and Reverse-Z `VK_FORMAT_D32_SFLOAT` depth (`depth_img`: `d = near_z / z_view`) into exact
`(entity_id, entity_bits, instance_id, meshlet_id, material_id, local_prim_id, world_pos)` in $O(1)$,
with a non-stalling `FRAMES_IN_FLIGHT` ring-buffered 1-pixel GPU compute readback state machine
for `FlintAaaVulkanRenderer` plus Tier-0 software picking integration.
"""
from std.math import isinf, isnan, max, min, tan
from std.memory import bitcast
from flint.core.math import Vec3
from flint.platform.amdgpu import GpuBuffer
from flint.platform.vulkan import (
    ACCESS_HOST_READ,
    ACCESS_SHADER_WRITE,
    ACCESS_TRANSFER_READ,
    ACCESS_TRANSFER_WRITE,
    LAYOUT_SHADER_READ_ONLY,
    PIPELINE_COMPUTE,
    PIPELINE_HOST,
    PIPELINE_TRANSFER,
    SUCCESS,
    VulkanGpu,
    call3,
    get32,
    get64,
    get_float,
    put64,
)
from flint.render.aaa_renderer import FRAMES_IN_FLIGHT, FlintAaaVulkanRenderer
from flint.render.aaa_visbuffer import (
    AAA_PUSH_WORDS,
    BIND_AAA_DEPTH_TEX,
    BIND_AAA_POST_SCRATCH,
    BIND_AAA_VIS_TEX,
)
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE

comptime VISBUF_MESHLET_MASK: UInt32 = 0x3FFFFF
comptime VISBUF_INST_LOW_MASK: UInt32 = 0x3FF
comptime VISBUF_INST_HIGH_MASK: UInt32 = 0xFF
comptime VISBUF_PRIM_MASK: UInt32 = 0xFF
comptime VISBUF_MAT_MASK: UInt32 = 0xFFFF


@fieldwise_init
struct ViewportPickResult(Copyable, ImplicitlyCopyable, Movable):
    """Decoded $O(1)$ viewport pick result from a single 40-bit VisBuffer + Reverse-Z depth texel."""
    var hit: Bool
    var entity_id: Int
    var entity_bits: UInt64
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
        self.entity_bits = 0
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
    """Packs `(instance_id, meshlet_id, material_id, local_prim_id)` matching M3 `visbuffer_vertex_shader`."""
    var inst_u = UInt32(max(0, instance_id))
    var ml_u = min(UInt32(max(0, meshlet_id) + 1), VISBUF_MESHLET_MASK)
    var w0 = ((inst_u & VISBUF_INST_LOW_MASK) << 22) | ml_u
    var w1 = (
        (UInt32(material_id & 0xFFFF) << 16)
        | (((inst_u >> 10) & VISBUF_INST_HIGH_MASK) << 8)
        | UInt32(local_prim_id & 0xFF)
    )
    return (w0, w1)


@always_inline
def _is_finite_f32(v: Float32) -> Bool:
    return (not isnan(v)) and (not isinf(v))


@always_inline
def _vec3_is_finite(v: Vec3) -> Bool:
    return _is_finite_f32(v.x) and _is_finite_f32(v.y) and _is_finite_f32(v.z)


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
    instance_to_entity_bits: List[UInt64] = List[UInt64](),
) -> ViewportPickResult:
    """Decodes 40-bit `R32G32_UINT` VisBuffer words + Reverse-Z `D32_SFLOAT` depth (`near_z / z_view`)
    into exact entity/meshlet/material/primitive IDs and 3D world-space hit position."""
    var res = ViewportPickResult()
    if vis_word0 == 0 or width <= 0 or height <= 0:
        return res
    if (
        (not _is_finite_f32(depth_d32))
        or depth_d32 <= Float32(1.0e-7)
        or depth_d32 >= Float32(1.0e9)
    ):
        return res
    if (
        (not _is_finite_f32(near_z))
        or near_z <= Float32(0.0)
        or near_z >= Float32(1.0e6)
    ):
        return res
    if (
        (not _is_finite_f32(fov_y_deg))
        or fov_y_deg <= Float32(0.0)
        or fov_y_deg >= Float32(179.9)
    ):
        return res
    if (not _is_finite_f32(jitter_clip_x)) or (not _is_finite_f32(jitter_clip_y)):
        return res
    if (
        (not _vec3_is_finite(cam_pos))
        or (not _vec3_is_finite(cam_fwd))
        or (not _vec3_is_finite(cam_right))
        or (not _vec3_is_finite(cam_up))
    ):
        return res
    var fwd_len = cam_fwd.length()
    var rgt_len = cam_right.length()
    var up_len = cam_up.length()
    if (
        (not _is_finite_f32(fwd_len))
        or (not _is_finite_f32(rgt_len))
        or (not _is_finite_f32(up_len))
        or fwd_len <= Float32(1.0e-6)
        or rgt_len <= Float32(1.0e-6)
        or up_len <= Float32(1.0e-6)
    ):
        return res

    var raw_ml = Int(vis_word0 & VISBUF_MESHLET_MASK)
    if raw_ml == 0:
        return res

    var inst_id = Int(vis_word0 >> 22) | (Int((vis_word1 >> 8) & VISBUF_INST_HIGH_MASK) << 10)
    var ml_id = raw_ml - 1
    var mat_id = Int((vis_word1 >> 16) & VISBUF_MAT_MASK)
    var prim_id = Int(vis_word1 & VISBUF_PRIM_MASK)
    var ent_id = instance_to_entity[inst_id] if (inst_id >= 0 and inst_id < len(instance_to_entity)) else inst_id
    var ent_bits = (
        instance_to_entity_bits[inst_id]
        if (inst_id >= 0 and inst_id < len(instance_to_entity_bits))
        else UInt64(max(0, ent_id))
    )

    var cpx = max(0, min(width - 1, px))
    var cpy = max(0, min(height - 1, py))
    var aspect = Float32(width) / Float32(max(1, height))
    var fov_rad = fov_y_deg * Float32(0.01745329252)
    var tan_y = tan(fov_rad * Float32(0.5))
    var tan_x = tan_y * aspect

    # Match Vulkan negative-height viewport (`flip_y = True`) + Halton TSR clip jitter in `aaa_gi_svgf.mojo`
    var ndc_x = ((Float32(cpx) + Float32(0.5)) / Float32(width)) * Float32(2.0) - Float32(1.0) - jitter_clip_x
    var ndc_y = Float32(1.0) - ((Float32(cpy) + Float32(0.5)) / Float32(height)) * Float32(2.0) - jitter_clip_y

    var z_view = near_z / depth_d32
    var x_view = ndc_x * tan_x * z_view
    var y_view = ndc_y * tan_y * z_view

    var fwd_u = cam_fwd.normalized()
    var rgt_u = cam_right.normalized()
    var up_u = cam_up.normalized()
    var wpos = cam_pos + rgt_u * x_view + up_u * y_view + fwd_u * z_view
    if (not _is_finite_f32(z_view)) or (not _vec3_is_finite(wpos)):
        return res

    res.hit = True
    res.entity_id = ent_id
    res.entity_bits = ent_bits
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


def is_timeline_fence_complete(mut vk: VulkanGpu, fence: UInt64) -> Bool:
    """Non-blocking timeline semaphore completion query (`vkGetSemaphoreCounterValue`)."""
    if fence == 0:
        return True
    if vk.device == 0 or vk.timeline == 0:
        return False
    var fn_get = vk.device_call("vkGetSemaphoreCounterValue")
    if fn_get == 0:
        return False
    vk.rewind()
    var ptr = vk.scratch(8)
    put64(ptr, 0)
    if call3(fn_get, vk.device, vk.timeline, ptr) != SUCCESS:
        return False
    return UInt64(get64(ptr)) >= fence


struct PickReadbackSlot(Copyable, Movable):
    """Per-frame-in-flight async 1-pixel readback slot with snapshotted camera & instance map."""
    var pending: Bool
    var fence: UInt64
    var request_id: UInt64
    var rx: Int
    var ry: Int
    var render_w: Int
    var render_h: Int
    var cam_pos: Vec3
    var cam_fwd: Vec3
    var cam_right: Vec3
    var cam_up: Vec3
    var fov_y_deg: Float32
    var near_z: Float32
    var jitter_clip_x: Float32
    var jitter_clip_y: Float32
    var instance_to_entity: List[Int]
    var instance_to_entity_bits: List[UInt64]

    def __init__(out self):
        self.pending = False
        self.fence = 0
        self.request_id = 0
        self.rx = 0
        self.ry = 0
        self.render_w = 0
        self.render_h = 0
        self.cam_pos = Vec3()
        self.cam_fwd = Vec3(0.0, 0.0, 1.0)
        self.cam_right = Vec3(1.0, 0.0, 0.0)
        self.cam_up = Vec3(0.0, 1.0, 0.0)
        self.fov_y_deg = Float32(55.0)
        self.near_z = Float32(0.15)
        self.jitter_clip_x = Float32(0.0)
        self.jitter_clip_y = Float32(0.0)
        self.instance_to_entity = List[Int]()
        self.instance_to_entity_bits = List[UInt64]()

    def __init__(out self, *, copy: Self):
        self.pending = copy.pending
        self.fence = copy.fence
        self.request_id = copy.request_id
        self.rx = copy.rx
        self.ry = copy.ry
        self.render_w = copy.render_w
        self.render_h = copy.render_h
        self.cam_pos = copy.cam_pos
        self.cam_fwd = copy.cam_fwd
        self.cam_right = copy.cam_right
        self.cam_up = copy.cam_up
        self.fov_y_deg = copy.fov_y_deg
        self.near_z = copy.near_z
        self.jitter_clip_x = copy.jitter_clip_x
        self.jitter_clip_y = copy.jitter_clip_y
        self.instance_to_entity = copy.instance_to_entity.copy()
        self.instance_to_entity_bits = copy.instance_to_entity_bits.copy()


struct VisBufferGpuPicker(Movable):
    """Non-stalling `FRAMES_IN_FLIGHT` ring-buffered 1-pixel VisBuffer + Reverse-Z depth picker."""
    var pipeline: Int
    var cmd_base: Int
    var rb_buf: GpuBuffer
    var slots: List[PickReadbackSlot]
    var write_cursor: Int
    var has_pending_pick: Bool
    var queued_px: Int
    var queued_py: Int
    var queued_request_id: UInt64
    var next_request_id: UInt64
    var queued_inst_to_ent: List[Int]
    var queued_inst_to_bits: List[UInt64]
    var completed_results: List[Tuple[UInt64, ViewportPickResult]]
    var total_submitted: Int
    var total_harvested: Int

    def __init__(out self):
        self.pipeline = 0
        self.cmd_base = -1
        self.rb_buf = GpuBuffer()
        self.slots = List[PickReadbackSlot](capacity=FRAMES_IN_FLIGHT)
        for _ in range(FRAMES_IN_FLIGHT):
            self.slots.append(PickReadbackSlot())
        self.write_cursor = 0
        self.has_pending_pick = False
        self.queued_px = 0
        self.queued_py = 0
        self.queued_request_id = 0
        self.next_request_id = 1
        self.queued_inst_to_ent = List[Int]()
        self.queued_inst_to_bits = List[UInt64]()
        self.completed_results = List[Tuple[UInt64, ViewportPickResult]]()
        self.total_submitted = 0
        self.total_harvested = 0

    def shutdown(mut self, mut ren: FlintAaaVulkanRenderer):
        var any_pending = False
        for i in range(len(self.slots)):
            if self.slots[i].pending:
                any_pending = True
                self.slots[i].pending = False
                self.slots[i].fence = 0
        if ren.vk.device != 0:
            if any_pending:
                _ = ren.vk.wait_idle()
            if self.rb_buf.vk_buffer != 0 or self.rb_buf.size > 0:
                ren.vk.destroy_buffer(self.rb_buf)
                self.rb_buf = GpuBuffer()
            if self.pipeline != 0:
                ren.vk.destroy_pipeline(self.pipeline)
                self.pipeline = 0
        self.cmd_base = -1
        self.has_pending_pick = False

    def is_pick_inflight(self) -> Bool:
        if self.has_pending_pick:
            return True
        for i in range(len(self.slots)):
            if self.slots[i].pending:
                return True
        return False

    def has_completed_pick(self) -> Bool:
        return len(self.completed_results) > 0

    def request_pick(
        mut self,
        out_px: Int,
        out_py: Int,
        instance_to_entity: List[Int] = List[Int](),
        instance_to_entity_bits: List[UInt64] = List[UInt64](),
    ) -> UInt64:
        """Queues a 1-pixel pick request with zero GPU calls and zero CPU waits."""
        var req_id = self.next_request_id
        self.next_request_id += 1
        self.queued_px = out_px
        self.queued_py = out_py
        self.queued_request_id = req_id
        self.queued_inst_to_ent = instance_to_entity.copy()
        self.queued_inst_to_bits = instance_to_entity_bits.copy()
        self.has_pending_pick = True
        return req_id

    def push_software_pick_result(mut self, request_id: UInt64, result: ViewportPickResult):
        """Completes a Tier-0 / software pick through the same async harvest queue."""
        if self.has_pending_pick and self.queued_request_id == request_id:
            self.has_pending_pick = False
        self.completed_results.append((request_id, result))
        self.total_submitted += 1
        self.total_harvested += 1

    def _harvest_slot(mut self, slot_idx: Int) -> Bool:
        if slot_idx < 0 or slot_idx >= len(self.slots):
            return False
        if not self.slots[slot_idx].pending:
            return False
        if self.rb_buf.cpu_address == 0:
            self.slots[slot_idx].pending = False
            self.slots[slot_idx].fence = 0
            return False

        var base_addr = self.rb_buf.cpu_address + slot_idx * 16
        var w0 = get32(base_addr + 0)
        var w1 = get32(base_addr + 4)
        var d_bits = get32(base_addr + 8)
        var d_f32 = bitcast[DType.float32](d_bits)

        ref s = self.slots[slot_idx]
        var res = decode_visbuffer_pick(
            s.rx,
            s.ry,
            w0,
            w1,
            d_f32,
            s.render_w,
            s.render_h,
            s.cam_pos,
            s.cam_fwd,
            s.cam_right,
            s.cam_up,
            s.fov_y_deg,
            s.near_z,
            s.instance_to_entity,
            s.jitter_clip_x,
            s.jitter_clip_y,
            s.instance_to_entity_bits,
        )
        var req_id = s.request_id
        s.pending = False
        s.fence = 0
        self.completed_results.append((req_id, res))
        self.total_harvested += 1
        return True

    def _dispatch_pending_pick_impl(
        mut self,
        mut ren: FlintAaaVulkanRenderer,
        instance_to_entity: List[Int],
        instance_to_entity_bits: List[UInt64],
    ) -> Bool:
        """Dispatches a queued 1-pixel readback on the GPU without blocking the CPU."""
        _ = self.harvest_completed_picks(ren)
        if (
            not self.has_pending_pick
            or not ren.active
            or ren.render_w <= 0
            or ren.render_h <= 0
            or ren.vis_img.image == 0
            or ren.depth_img.image == 0
            or ren.vis_img.layout != LAYOUT_SHADER_READ_ONLY
            or ren.depth_img.layout != LAYOUT_SHADER_READ_ONLY
        ):
            return False

        var slot = self.write_cursor % FRAMES_IN_FLIGHT
        if self.slots[slot].pending:
            if is_timeline_fence_complete(ren.vk, self.slots[slot].fence):
                _ = self._harvest_slot(slot)
            else:
                return False

        if self.pipeline == 0:
            self.pipeline = ren.vk.create_aaa_compute_pipeline(ren.pipe_layout, visbuffer_pick_readback_shader())
            if self.pipeline == 0:
                return False

        if self.cmd_base < 0:
            var base = len(ren.vk.command_buffers)
            if not ren.vk.allocate_command_buffers(FRAMES_IN_FLIGHT):
                return False
            self.cmd_base = base

        if self.rb_buf.cpu_address == 0:
            self.rb_buf = ren.vk.create_buffer(FRAMES_IN_FLIGHT * 16, local=False, cpu_reads=True)
            if self.rb_buf.cpu_address == 0:
                return False

        var rx = max(0, min(ren.render_w - 1, (self.queued_px * ren.render_w) // max(1, ren.output_w)))
        var ry = max(0, min(ren.render_h - 1, (self.queued_py * ren.render_h) // max(1, ren.output_h)))
        var scratch_word_off = 512 + slot * 4
        var rb_byte_off = slot * 16
        var cmd_idx = self.cmd_base + slot

        if not ren.vk.begin_commands(cmd_idx):
            return False

        var push = List[UInt32](length=AAA_PUSH_WORDS, fill=0)
        push[0] = UInt32(rx)
        push[1] = UInt32(ry)
        push[2] = UInt32(scratch_word_off)

        ren.vk.dispatch_aaa(cmd_idx, self.pipeline, ren.pipe_layout, ren.desc_set, push, 1, 1)
        ren.vk.memory_barrier2(cmd_idx, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ))
        ren.vk.copy_buffer(cmd_idx, ren.buf_post_scratch, scratch_word_off * 4, self.rb_buf, rb_byte_off, 12)
        ren.vk.memory_barrier2(cmd_idx, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))

        var fence: UInt64
        if ren.prev_submitted_fence != 0:
            fence = ren.vk.submit_with_wait(cmd_idx, ren.vk.timeline, ren.prev_submitted_fence, PIPELINE_COMPUTE)
        else:
            fence = ren.vk.submit(cmd_idx)
        if fence == 0:
            return False
        ren.prev_submitted_fence = fence

        var jx_clip = Float32(0.0)
        var jy_clip = Float32(0.0)
        if ren.buf_globals.cpu_address != 0:
            jx_clip = get_float(ren.buf_globals.cpu_address + 101 * 4)
            jy_clip = get_float(ren.buf_globals.cpu_address + 102 * 4)

        ref s = self.slots[slot]
        s.pending = True
        s.fence = fence
        s.request_id = self.queued_request_id
        s.rx = rx
        s.ry = ry
        s.render_w = ren.render_w
        s.render_h = ren.render_h
        s.cam_pos = ren.cam_pos
        s.cam_fwd = ren.cam_fwd
        s.cam_right = ren.cam_right
        s.cam_up = ren.cam_up
        s.fov_y_deg = ren.fov_y_deg
        s.near_z = ren.near_z
        s.jitter_clip_x = jx_clip
        s.jitter_clip_y = jy_clip
        s.instance_to_entity = instance_to_entity.copy()
        s.instance_to_entity_bits = instance_to_entity_bits.copy()

        self.has_pending_pick = False
        self.queued_inst_to_ent.clear()
        self.queued_inst_to_bits.clear()
        self.write_cursor = (slot + 1) % FRAMES_IN_FLIGHT
        self.total_submitted += 1
        return True

    def dispatch_pending_pick(mut self, mut ren: FlintAaaVulkanRenderer) -> Bool:
        """Dispatches a queued 1-pixel readback using maps supplied at `request_pick()` time."""
        if len(ren.frame_instances) == 0:
            return self._dispatch_pending_pick_impl(ren, List[Int](), List[UInt64]())
        var q_ent = self.queued_inst_to_ent.copy()
        var q_bits = self.queued_inst_to_bits.copy()
        return self._dispatch_pending_pick_impl(ren, q_ent, q_bits)

    def dispatch_pending_pick(
        mut self,
        mut ren: FlintAaaVulkanRenderer,
        instance_to_entity: List[Int],
        instance_to_entity_bits: List[UInt64] = List[UInt64](),
    ) -> Bool:
        """Dispatches a queued 1-pixel readback using the explicitly provided current-frame instance maps."""
        return self._dispatch_pending_pick_impl(ren, instance_to_entity, instance_to_entity_bits)

    def harvest_completed_picks(mut self, mut ren: FlintAaaVulkanRenderer) -> Int:
        """Non-blocking check of in-flight readback slots via `vkGetSemaphoreCounterValue`."""
        var harvested = 0
        for offset in range(FRAMES_IN_FLIGHT):
            var slot_idx = (self.write_cursor + offset) % FRAMES_IN_FLIGHT
            if self.slots[slot_idx].pending:
                if is_timeline_fence_complete(ren.vk, self.slots[slot_idx].fence):
                    if self._harvest_slot(slot_idx):
                        harvested += 1
        return harvested

    def poll_completed_pick_entry(mut self, mut ren: FlintAaaVulkanRenderer) -> Optional[Tuple[UInt64, ViewportPickResult]]:
        """Harvests finished GPU slots non-blocking and pops the oldest `(request_id, ViewportPickResult)`."""
        _ = self.harvest_completed_picks(ren)
        if len(self.completed_results) == 0:
            return None
        var first = self.completed_results[0]
        var remaining = List[Tuple[UInt64, ViewportPickResult]](capacity=max(0, len(self.completed_results) - 1))
        for i in range(1, len(self.completed_results)):
            remaining.append(self.completed_results[i])
        self.completed_results = remaining^
        return Optional[Tuple[UInt64, ViewportPickResult]](first)

    def poll_completed_pick(mut self, mut ren: FlintAaaVulkanRenderer) -> Optional[ViewportPickResult]:
        """Harvests finished GPU slots non-blocking and pops the oldest completed pick result."""
        var entry = self.poll_completed_pick_entry(ren)
        if not entry:
            return None
        return Optional[ViewportPickResult](entry.value()[1])

    def pick_viewport(
        mut self,
        mut ren: FlintAaaVulkanRenderer,
        out_px: Int,
        out_py: Int,
        instance_to_entity: List[Int] = List[Int](),
    ) -> ViewportPickResult:
        """Explicit synchronous pick helper (for one-shot tests; waits only on its own pick slot fence)."""
        var my_id = self.request_pick(out_px, out_py, instance_to_entity)
        if not self.dispatch_pending_pick(ren, instance_to_entity):
            return ViewportPickResult()
        var last_slot = (self.write_cursor + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT
        var f = self.slots[last_slot].fence
        if f != 0 and ren.vk.device != 0:
            _ = ren.vk.wait(f)
        _ = self.harvest_completed_picks(ren)
        # Take *this* request's result; older async results stay queued for their own caller
        # (popping the oldest entry used to hand back the previous click's entity).
        var mine = self.take_pick_result(my_id)
        return mine.value() if mine else ViewportPickResult()

    def take_pick_result(mut self, request_id: UInt64) -> Optional[ViewportPickResult]:
        """Removes and returns the completed result for `request_id`, leaving other results queued."""
        for i in range(len(self.completed_results)):
            if self.completed_results[i][0] == request_id:
                var res = self.completed_results[i][1]
                _ = self.completed_results.pop(i)
                return Optional[ViewportPickResult](res)
        return None


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


# ============================================================================
# Hover-Pick Preview & 180ms Click Selection Confirmation Pulse
# ============================================================================

comptime PICK_HOVER_POLL_INTERVAL_SEC: Float32 = 0.06
comptime PICK_HOVER_DWELL_SEC: Float32 = 0.12
comptime PICK_SELECT_PULSE_SEC: Float32 = 0.18


struct ViewportPickFeedback(Copyable, Movable):
    """Non-intrusive viewport hover-pick preview & 180ms click selection confirmation pulse.

    Strictly deterministic: when `instant_mode=True` (`UI_ANIM_INSTANT`), transient pulses
    suppress themselves so golden screenshots stay pixel-exact.
    """
    var hovered_entity_id: Int
    var hovered_entity_name: String
    var hovered_screen_x: Float32
    var hovered_screen_y: Float32
    var hovered_world_pos: Vec3
    var _last_cursor_x: Float32
    var _last_cursor_y: Float32
    var hover_dwell_sec: Float32
    var hover_poll_cooldown_sec: Float32
    var pulse_active: Bool
    var pulse_entity_id: Int
    var pulse_x: Float32
    var pulse_y: Float32
    var pulse_age_sec: Float32
    var pulse_duration_sec: Float32

    def __init__(out self):
        self.hovered_entity_id = -1
        self.hovered_entity_name = ""
        self.hovered_screen_x = Float32(-10000.0)
        self.hovered_screen_y = Float32(-10000.0)
        self.hovered_world_pos = Vec3()
        self._last_cursor_x = Float32(-10000.0)
        self._last_cursor_y = Float32(-10000.0)
        self.hover_dwell_sec = Float32(0.0)
        self.hover_poll_cooldown_sec = Float32(0.0)
        self.pulse_active = False
        self.pulse_entity_id = -1
        self.pulse_x = Float32(0.0)
        self.pulse_y = Float32(0.0)
        self.pulse_age_sec = Float32(0.0)
        self.pulse_duration_sec = PICK_SELECT_PULSE_SEC

    def __init__(out self, *, copy: Self):
        self.hovered_entity_id = copy.hovered_entity_id
        self.hovered_entity_name = copy.hovered_entity_name
        self.hovered_screen_x = copy.hovered_screen_x
        self.hovered_screen_y = copy.hovered_screen_y
        self.hovered_world_pos = copy.hovered_world_pos
        self._last_cursor_x = copy._last_cursor_x
        self._last_cursor_y = copy._last_cursor_y
        self.hover_dwell_sec = copy.hover_dwell_sec
        self.hover_poll_cooldown_sec = copy.hover_poll_cooldown_sec
        self.pulse_active = copy.pulse_active
        self.pulse_entity_id = copy.pulse_entity_id
        self.pulse_x = copy.pulse_x
        self.pulse_y = copy.pulse_y
        self.pulse_age_sec = copy.pulse_age_sec
        self.pulse_duration_sec = copy.pulse_duration_sec

    def clear_hover(mut self):
        self.hovered_entity_id = -1
        self.hovered_entity_name = ""
        self.hover_dwell_sec = Float32(0.0)

    def should_request_hover_pick(
        mut self,
        pointer_in_viewport: Bool,
        cursor_x: Float32,
        cursor_y: Float32,
        dt: Float32,
        dragging: Bool = False,
    ) -> Bool:
        """Returns `True` at most ~16 Hz when cursor settles inside the 3D viewport."""
        var safe_dt = dt if (_is_finite_f32(dt) and dt > Float32(0.0)) else Float32(0.0)
        if self.hover_poll_cooldown_sec > Float32(0.0):
            self.hover_poll_cooldown_sec = max(Float32(0.0), self.hover_poll_cooldown_sec - safe_dt)
        if (not pointer_in_viewport) or dragging or (not _is_finite_f32(cursor_x)) or (not _is_finite_f32(cursor_y)):
            self.clear_hover()
            return False
        var dx = cursor_x - self._last_cursor_x
        var dy = cursor_y - self._last_cursor_y
        var moved_sq = dx * dx + dy * dy
        self._last_cursor_x = cursor_x
        self._last_cursor_y = cursor_y
        if moved_sq > Float32(4.0):
            self.hover_dwell_sec = Float32(0.0)
        else:
            self.hover_dwell_sec += safe_dt
        if self.hover_poll_cooldown_sec <= Float32(0.0):
            self.hover_poll_cooldown_sec = PICK_HOVER_POLL_INTERVAL_SEC
            return True
        return False

    def record_hover_result(
        mut self,
        res: ViewportPickResult,
        entity_name: String,
        screen_x: Float32,
        screen_y: Float32,
    ):
        """Updates cached hover preview target from a harvested `ViewportPickResult`."""
        if (not res.hit) or res.entity_id < 0:
            self.hovered_entity_id = -1
            self.hovered_entity_name = ""
            return
        self.hovered_entity_id = res.entity_id
        self.hovered_entity_name = entity_name
        self.hovered_screen_x = screen_x if _is_finite_f32(screen_x) else Float32(0.0)
        self.hovered_screen_y = screen_y if _is_finite_f32(screen_y) else Float32(0.0)
        self.hovered_world_pos = res.world_pos

    def show_hover_preview(self) -> Bool:
        """Returns `True` when a hovered entity preview chip should be shown."""
        return (
            self.hovered_entity_id >= 0
            and self.hovered_entity_name.byte_length() > 0
            and self.hover_dwell_sec >= PICK_HOVER_DWELL_SEC
        )

    def trigger_select_pulse(
        mut self,
        entity_id: Int,
        screen_x: Float32,
        screen_y: Float32,
        instant_mode: Bool = False,
        duration_sec: Float32 = PICK_SELECT_PULSE_SEC,
    ):
        """Starts a subtle 180ms expanding confirmation ring at `(screen_x, screen_y)`."""
        if instant_mode or entity_id < 0 or (not _is_finite_f32(screen_x)) or (not _is_finite_f32(screen_y)):
            self.pulse_active = False
            return
        self.pulse_active = True
        self.pulse_entity_id = entity_id
        self.pulse_x = screen_x
        self.pulse_y = screen_y
        self.pulse_age_sec = Float32(0.0)
        self.pulse_duration_sec = max(Float32(0.05), duration_sec if _is_finite_f32(duration_sec) else PICK_SELECT_PULSE_SEC)

    def tick(mut self, dt: Float32, instant_mode: Bool = False):
        """Advances active click selection pulse (`instant_mode=True` clears immediately)."""
        if instant_mode:
            self.pulse_active = False
            self.pulse_age_sec = Float32(0.0)
            return
        if not self.pulse_active:
            return
        if (not _is_finite_f32(dt)) or dt <= Float32(0.0):
            return
        self.pulse_age_sec += min(Float32(0.25), dt)
        if self.pulse_age_sec >= self.pulse_duration_sec:
            self.pulse_active = False
            self.pulse_age_sec = self.pulse_duration_sec

    def pulse_ring_metrics(self) -> Tuple[Bool, Float32, Float32, Float32]:
        """Returns `(active, radius_px, alpha, stroke_px)` for GPU SDF ring emission."""
        if not self.pulse_active:
            return (False, Float32(0.0), Float32(0.0), Float32(0.0))
        var t = max(Float32(0.0), min(Float32(1.0), self.pulse_age_sec / max(Float32(0.05), self.pulse_duration_sec)))
        # Fast-out ease: 1 - (1 - t)^3
        var inv = Float32(1.0) - t
        var ease = Float32(1.0) - inv * inv * inv
        var radius_px = Float32(8.0) + ease * Float32(20.0)
        var alpha = (Float32(1.0) - ease) * Float32(0.85)
        var stroke_px = Float32(2.2) - ease * Float32(1.0)
        return (True, radius_px, alpha, stroke_px)
