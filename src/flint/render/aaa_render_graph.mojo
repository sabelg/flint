"""Milestone 4 (`V15`): Declarative `VK_KHR_synchronization2` `AaaRenderGraph`,
`OffsetAllocator` Transient Memory Aliasing, Async Compute Queue Scheduling &
Zero-Hitch Mip/Meshlet Residency Streaming (`aaa_render_graph.mojo`).
"""
from flint.core.offset_allocator import OffsetAllocation, OffsetAllocator
from flint.platform.vulkan import (
    ACCESS_COLOR_ATTACHMENT_WRITE,
    ACCESS_DEPTH_ATTACHMENT_READ,
    ACCESS_DEPTH_ATTACHMENT_WRITE,
    ACCESS_INDIRECT_READ,
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    LAYOUT_COLOR_ATTACHMENT,
    LAYOUT_DEPTH_ATTACHMENT,
    LAYOUT_GENERAL,
    LAYOUT_SHADER_READ_ONLY,
    LAYOUT_UNDEFINED,
    PIPELINE_COLOR_ATTACHMENT_OUTPUT,
    PIPELINE_COMPUTE,
    PIPELINE_DRAW_INDIRECT,
    PIPELINE_EARLY_FRAGMENT_TESTS,
    PIPELINE_FRAGMENT_SHADER,
    PIPELINE_LATE_FRAGMENT_TESTS,
    VulkanGpu,
    VulkanImage,
)

comptime RG_QUEUE_GRAPHICS: Int = 0
comptime RG_QUEUE_ASYNC_COMPUTE: Int = 1

comptime RG_USAGE_COMPUTE_READ: Int = 0
comptime RG_USAGE_COMPUTE_WRITE: Int = 1
comptime RG_USAGE_INDIRECT_ARGS: Int = 2
comptime RG_USAGE_COLOR_ATT_WRITE: Int = 3
comptime RG_USAGE_DEPTH_ATT_WRITE: Int = 4
comptime RG_USAGE_FRAGMENT_READ: Int = 5

comptime RG_UNIT_BYTES: Int = 256


struct RgResourceDesc(Copyable, Movable):
    var name: String
    var size_bytes: Int
    var transient: Bool
    var is_image: Bool
    var first_pass: Int
    var last_pass: Int
    var aliased_offset_bytes: Int

    def __init__(out self, name: String, size_bytes: Int, transient: Bool = True, is_image: Bool = False):
        self.name = name
        self.size_bytes = max(RG_UNIT_BYTES, size_bytes)
        self.transient = transient
        self.is_image = is_image
        self.first_pass = 999999
        self.last_pass = -1
        self.aliased_offset_bytes = 0


@fieldwise_init
struct RgAliasingMetrics(Copyable, ImplicitlyCopyable, Movable):
    """Snapshot of `OffsetAllocator` transient VRAM aliasing and compiled barrier counts."""
    var unaliased_bytes: Int
    var aliased_peak_bytes: Int
    var saved_bytes: Int
    var savings_ratio: Float32
    var transient_resource_count: Int
    var compiled_barrier_count: Int


struct RgResourceAccess(Copyable, Movable):
    var resource_id: Int
    var usage: Int

    def __init__(out self, resource_id: Int, usage: Int):
        self.resource_id = resource_id
        self.usage = usage


struct RgPassNode(Copyable, Movable):
    var name: String
    var queue_kind: Int
    var accesses: List[RgResourceAccess]

    def __init__(out self, name: String, queue_kind: Int = RG_QUEUE_GRAPHICS):
        self.name = name
        self.queue_kind = queue_kind
        self.accesses = List[RgResourceAccess]()


struct RgBarrierRecord(Copyable, Movable):
    var pass_index: Int
    var resource_id: Int
    var src_stage: Int
    var src_access: Int
    var dst_stage: Int
    var dst_access: Int
    var cross_queue_handoff: Bool
    var is_image_barrier: Bool
    var old_layout: UInt32
    var new_layout: UInt32

    def __init__(
        out self,
        pass_index: Int,
        resource_id: Int,
        src_stage: Int,
        src_access: Int,
        dst_stage: Int,
        dst_access: Int,
        cross_queue_handoff: Bool = False,
        is_image_barrier: Bool = False,
        old_layout: UInt32 = LAYOUT_UNDEFINED,
        new_layout: UInt32 = LAYOUT_UNDEFINED,
    ):
        self.pass_index = pass_index
        self.resource_id = resource_id
        self.src_stage = src_stage
        self.src_access = src_access
        self.dst_stage = dst_stage
        self.dst_access = dst_access
        self.cross_queue_handoff = cross_queue_handoff
        self.is_image_barrier = is_image_barrier
        self.old_layout = old_layout
        self.new_layout = new_layout


struct AaaGpuPassTiming(Copyable, Movable):
    """Hardware `VkQueryPool` GPU execution duration for a single `AaaRenderGraph` pass (`V19`)."""
    var pass_name: String
    var gpu_ns: UInt64
    var gpu_ms: Float64

    def __init__(out self, pass_name: String, gpu_ns: UInt64, gpu_ms: Float64):
        self.pass_name = pass_name
        self.gpu_ns = gpu_ns
        self.gpu_ms = gpu_ms


@always_inline
def _usage_to_vk_stage_access(usage: Int) -> Tuple[Int, Int, Bool]:
    """Maps logical `RG_USAGE_*` to `(VkPipelineStageFlags2, VkAccessFlags2, is_write)`."""
    if usage == RG_USAGE_COMPUTE_WRITE:
        return (PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), True)
    if usage == RG_USAGE_COMPUTE_READ:
        return (PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ), False)
    if usage == RG_USAGE_INDIRECT_ARGS:
        return (PIPELINE_DRAW_INDIRECT, Int(ACCESS_INDIRECT_READ), False)
    if usage == RG_USAGE_COLOR_ATT_WRITE:
        return (PIPELINE_COLOR_ATTACHMENT_OUTPUT, Int(ACCESS_COLOR_ATTACHMENT_WRITE), True)
    if usage == RG_USAGE_DEPTH_ATT_WRITE:
        return (
            PIPELINE_EARLY_FRAGMENT_TESTS | PIPELINE_LATE_FRAGMENT_TESTS,
            Int(ACCESS_DEPTH_ATTACHMENT_READ | ACCESS_DEPTH_ATTACHMENT_WRITE),
            True,
        )
    return (PIPELINE_FRAGMENT_SHADER, Int(ACCESS_SHADER_READ), False)


@always_inline
def _usage_to_vk_layout(usage: Int) -> UInt32:
    """Maps logical `RG_USAGE_*` to target `VkImageLayout` for image resources."""
    if usage == RG_USAGE_COLOR_ATT_WRITE:
        return LAYOUT_COLOR_ATTACHMENT
    if usage == RG_USAGE_DEPTH_ATT_WRITE:
        return LAYOUT_DEPTH_ATTACHMENT
    if usage == RG_USAGE_COMPUTE_WRITE:
        return LAYOUT_GENERAL
    return LAYOUT_SHADER_READ_ONLY


struct AaaRenderGraph(Movable):
    """Compiled declarative frame graph with `VK_KHR_synchronization2` barrier synthesis,
    `OffsetAllocator` transient VRAM aliasing, and hardware `VkQueryPool` pass timings."""
    var resources: List[RgResourceDesc]
    var passes: List[RgPassNode]
    var barriers: List[RgBarrierRecord]
    var pass_timings: List[AaaGpuPassTiming]
    var total_gpu_ms: Float64
    var unaliased_transient_bytes: Int
    var aliased_peak_bytes: Int
    var async_compute_pass_count: Int
    var barriers_emitted_last_frame: Int
    var passes_executed_last_frame: Int

    def __init__(out self):
        self.resources = List[RgResourceDesc]()
        self.passes = List[RgPassNode]()
        self.barriers = List[RgBarrierRecord]()
        self.pass_timings = List[AaaGpuPassTiming]()
        self.total_gpu_ms = 0.0
        self.unaliased_transient_bytes = 0
        self.aliased_peak_bytes = 0
        self.async_compute_pass_count = 0
        self.barriers_emitted_last_frame = 0
        self.passes_executed_last_frame = 0

    def begin_frame(mut self):
        self.barriers_emitted_last_frame = 0
        self.passes_executed_last_frame = 0

    def add_resource(mut self, name: String, size_bytes: Int, transient: Bool = True, is_image: Bool = False) -> Int:
        var id = len(self.resources)
        self.resources.append(RgResourceDesc(name, size_bytes, transient, is_image))
        return id

    def add_pass(mut self, name: String, accesses: List[RgResourceAccess], queue_kind: Int = RG_QUEUE_GRAPHICS) -> Int:
        var pid = len(self.passes)
        var node = RgPassNode(name, queue_kind)
        node.accesses = accesses.copy()
        self.passes.append(node^)
        return pid

    def compile(mut self, heap_capacity_bytes: Int = 256 * 1024 * 1024) -> Bool:
        """Computes live intervals, aliases transient memory in `O(1)` via `OffsetAllocator`,
        and synthesizes minimal `VK_KHR_synchronization2` hazard barriers + async queue syncs!"""
        self.barriers.clear()
        self.unaliased_transient_bytes = 0
        self.aliased_peak_bytes = 0
        self.async_compute_pass_count = 0

        for r in range(len(self.resources)):
            self.resources[r].first_pass = 999999
            self.resources[r].last_pass = -1
            self.resources[r].aliased_offset_bytes = 0

        for p_idx in range(len(self.passes)):
            if self.passes[p_idx].queue_kind == RG_QUEUE_ASYNC_COMPUTE:
                self.async_compute_pass_count += 1
            for acc in self.passes[p_idx].accesses:
                var rid = acc.resource_id
                if p_idx < self.resources[rid].first_pass:
                    self.resources[rid].first_pass = p_idx
                if p_idx > self.resources[rid].last_pass:
                    self.resources[rid].last_pass = p_idx

        # 1. Transient VRAM aliasing via OffsetAllocator
        var has_active_transient = False
        for r in range(len(self.resources)):
            if self.resources[r].transient and self.resources[r].last_pass >= 0:
                var aligned = ((self.resources[r].size_bytes + RG_UNIT_BYTES - 1) // RG_UNIT_BYTES) * RG_UNIT_BYTES
                self.unaliased_transient_bytes += aligned
                has_active_transient = True

        if heap_capacity_bytes <= 0:
            if has_active_transient:
                return False
        var heap_units = UInt32(max(1, (heap_capacity_bytes + RG_UNIT_BYTES - 1) // RG_UNIT_BYTES))
        var oa = OffsetAllocator(heap_units, max_allocations=1024)
        var live_allocs = List[OffsetAllocation](length=len(self.resources), fill=OffsetAllocation())

        for p_idx in range(len(self.passes)):
            # Allocate resources whose lifetime starts on `p_idx`
            for r in range(len(self.resources)):
                if self.resources[r].transient and self.resources[r].first_pass == p_idx:
                    var units = UInt32((self.resources[r].size_bytes + RG_UNIT_BYTES - 1) // RG_UNIT_BYTES)
                    var al = oa.allocate(units)
                    if not al.ok():
                        return False
                    live_allocs[r] = al
                    self.resources[r].aliased_offset_bytes = Int(al.offset) * RG_UNIT_BYTES

            # Free resources whose lifetime ends on `p_idx` so downstream passes alias the exact same VRAM!
            for r in range(len(self.resources)):
                if self.resources[r].transient and self.resources[r].last_pass == p_idx:
                    oa.free(live_allocs[r])
                    live_allocs[r] = OffsetAllocation()

        self.aliased_peak_bytes = Int(oa.peak_used) * RG_UNIT_BYTES

        # 2. Automatic `VK_KHR_synchronization2` RAW / WAR / WAW hazard barrier synthesis
        var last_write_stage = List[Int](length=len(self.resources), fill=0)
        var last_write_access = List[Int](length=len(self.resources), fill=0)
        var last_write_queue = List[Int](length=len(self.resources), fill=RG_QUEUE_GRAPHICS)
        var synced_stages_q0 = List[Int](length=len(self.resources), fill=0)
        var synced_stages_q1 = List[Int](length=len(self.resources), fill=0)
        var read_stages_q0 = List[Int](length=len(self.resources), fill=0)
        var read_stages_q1 = List[Int](length=len(self.resources), fill=0)
        var last_layout = List[UInt32](length=len(self.resources), fill=LAYOUT_SHADER_READ_ONLY)

        for p_idx in range(len(self.passes)):
            var cur_q = self.passes[p_idx].queue_kind
            for acc in self.passes[p_idx].accesses:
                var rid = acc.resource_id
                var sa = _usage_to_vk_stage_access(acc.usage)
                var dst_st = sa[0]
                var dst_ac = sa[1]
                var is_wr = sa[2]
                var is_img = self.resources[rid].is_image
                var tgt_layout = _usage_to_vk_layout(acc.usage) if is_img else LAYOUT_UNDEFINED
                var prev_layout = last_layout[rid] if is_img else LAYOUT_UNDEFINED

                if is_wr:
                    if read_stages_q0[rid] != 0 or read_stages_q1[rid] != 0:
                        if read_stages_q0[rid] != 0:
                            self.barriers.append(
                                RgBarrierRecord(
                                    p_idx,
                                    rid,
                                    read_stages_q0[rid],
                                    0,
                                    dst_st,
                                    dst_ac,
                                    RG_QUEUE_GRAPHICS != cur_q,
                                    is_img,
                                    prev_layout,
                                    tgt_layout,
                                )
                            )
                        if read_stages_q1[rid] != 0:
                            self.barriers.append(
                                RgBarrierRecord(
                                    p_idx,
                                    rid,
                                    read_stages_q1[rid],
                                    0,
                                    dst_st,
                                    dst_ac,
                                    RG_QUEUE_ASYNC_COMPUTE != cur_q,
                                    is_img,
                                    prev_layout,
                                    tgt_layout,
                                )
                            )
                    elif last_write_stage[rid] != 0:
                        self.barriers.append(
                            RgBarrierRecord(
                                p_idx,
                                rid,
                                last_write_stage[rid],
                                last_write_access[rid],
                                dst_st,
                                dst_ac,
                                last_write_queue[rid] != cur_q,
                                is_img,
                                prev_layout,
                                tgt_layout,
                            )
                        )
                    elif is_img and prev_layout != tgt_layout:
                        self.barriers.append(
                            RgBarrierRecord(
                                p_idx,
                                rid,
                                PIPELINE_COMPUTE,
                                Int(ACCESS_SHADER_READ),
                                dst_st,
                                dst_ac,
                                False,
                                True,
                                prev_layout,
                                tgt_layout,
                            )
                        )
                    last_write_stage[rid] = dst_st
                    last_write_access[rid] = dst_ac
                    last_write_queue[rid] = cur_q
                    if is_img:
                        last_layout[rid] = tgt_layout
                    synced_stages_q0[rid] = 0
                    synced_stages_q1[rid] = 0
                    read_stages_q0[rid] = 0
                    read_stages_q1[rid] = 0
                else:
                    if last_write_stage[rid] != 0:
                        var need_barrier = False
                        if cur_q == RG_QUEUE_GRAPHICS:
                            if (synced_stages_q0[rid] & dst_st) != dst_st:
                                need_barrier = True
                                synced_stages_q0[rid] |= dst_st
                        else:
                            if (synced_stages_q1[rid] & dst_st) != dst_st:
                                need_barrier = True
                                synced_stages_q1[rid] |= dst_st
                        if need_barrier:
                            self.barriers.append(
                                RgBarrierRecord(
                                    p_idx,
                                    rid,
                                    last_write_stage[rid],
                                    last_write_access[rid],
                                    dst_st,
                                    dst_ac,
                                    last_write_queue[rid] != cur_q,
                                    is_img,
                                    prev_layout,
                                    tgt_layout,
                                )
                            )
                    if is_img:
                        last_layout[rid] = tgt_layout
                    if cur_q == RG_QUEUE_GRAPHICS:
                        read_stages_q0[rid] |= dst_st
                    else:
                        read_stages_q1[rid] |= dst_st

        return True

    def emit_pass_barriers(mut self, mut vk: VulkanGpu, cmd_slot: Int, pass_index: Int):
        """Emits per-resource `vkCmdPipelineBarrier2` buffer/memory barriers scheduled before `pass_index`."""
        self.passes_executed_last_frame += 1
        var emitted = 0
        for b in self.barriers:
            if b.pass_index == pass_index and not b.is_image_barrier:
                vk.memory_barrier2(cmd_slot, b.src_stage, b.src_access, b.dst_stage, b.dst_access)
                emitted += 1
        self.barriers_emitted_last_frame += emitted

    def emit_pass_image_transition(
        mut self,
        mut vk: VulkanGpu,
        cmd_slot: Int,
        pass_index: Int,
        resource_id: Int,
        mut img: VulkanImage,
    ) -> Bool:
        """Emits a compiled image layout transition barrier for `resource_id` scheduled before `pass_index`."""
        for b in self.barriers:
            if b.pass_index == pass_index and b.resource_id == resource_id and b.is_image_barrier:
                if img.layout != b.new_layout:
                    vk.transition_image(
                        cmd_slot,
                        img,
                        b.new_layout,
                        b.src_stage,
                        UInt32(b.src_access),
                        b.dst_stage,
                        UInt32(b.dst_access),
                    )
                else:
                    vk.memory_barrier2(cmd_slot, b.src_stage, b.src_access, b.dst_stage, b.dst_access)
                self.barriers_emitted_last_frame += 1
                return True
        return False

    def aliasing_metrics(self) -> RgAliasingMetrics:
        """Returns a structured snapshot of transient VRAM aliasing savings and compiled barrier counts."""
        var transient_cnt = 0
        for r in range(len(self.resources)):
            if self.resources[r].transient and self.resources[r].last_pass >= 0:
                transient_cnt += 1
        var saved = max(0, self.unaliased_transient_bytes - self.aliased_peak_bytes)
        return RgAliasingMetrics(
            self.unaliased_transient_bytes,
            self.aliased_peak_bytes,
            saved,
            self.aliasing_savings_ratio(),
            transient_cnt,
            len(self.barriers),
        )

    def aliasing_savings_ratio(self) -> Float32:
        if self.unaliased_transient_bytes <= 0:
            return Float32(0.0)
        return Float32(1.0) - Float32(self.aliased_peak_bytes) / Float32(self.unaliased_transient_bytes)

    def format_gpu_profile_summary(self) -> String:
        """Formats per-pass `VkQueryPool` GPU milliseconds & `OffsetAllocator` VRAM watermarks."""
        var out = String("AaaRenderGraph GPU Profile (total=") + String(Int(self.total_gpu_ms * 1000.0)) + " us, aliased=" + String(self.aliased_peak_bytes // 1024) + " KiB / " + String(self.unaliased_transient_bytes // 1024) + " KiB): "
        for i in range(len(self.pass_timings)):
            if i > 0:
                out += ", "
            out += self.pass_timings[i].pass_name + "=" + String(Int(self.pass_timings[i].gpu_ns // 1000)) + "us"
        return out


@fieldwise_init
struct ResidencyUploadItem(Copyable, ImplicitlyCopyable, Movable):
    """Queued byte-budgeted GPU mip or meshlet LOD upload request."""
    var target_id: Int
    var sub_level: Int
    var byte_size: Int
    var priority: Int
    var is_mesh_lod: Bool


struct AaaResidencyStreamer(Movable):
    """Zero-hitch GPU Mip-Tail & Meshlet Residency Feedback Streamer."""
    var resident_min_mip: List[Int]     # Current resident finest mip per bindless texture
    var requested_min_mip: List[Int]    # GPU feedback requested finest mip
    var streamed_bytes_total: Int
    var streamed_chunks_last_tick: Int
    var upload_queue: List[ResidencyUploadItem]
    var drained_bytes_last_tick: Int

    def __init__(out self, max_textures: Int = 256, initial_tail_mip: Int = 4):
        self.resident_min_mip = List[Int](length=max_textures, fill=initial_tail_mip)
        self.requested_min_mip = List[Int](length=max_textures, fill=initial_tail_mip)
        self.streamed_bytes_total = 0
        self.streamed_chunks_last_tick = 0
        self.upload_queue = List[ResidencyUploadItem]()
        self.drained_bytes_last_tick = 0

    def record_gpu_feedback(mut self, tex_id: Int, needed_mip: Int):
        if tex_id >= 0 and tex_id < len(self.requested_min_mip):
            var clamped = max(0, needed_mip)
            if clamped < self.requested_min_mip[tex_id]:
                self.requested_min_mip[tex_id] = clamped

    def record_feedback(mut self, tex_id: Int, needed_mip: Int):
        self.record_gpu_feedback(tex_id, needed_mip)

    def enqueue_mip_upload(
        mut self,
        tex_id: Int,
        mip_level: Int,
        byte_size: Int,
        priority: Int = 0,
        is_mesh_lod: Bool = False,
    ):
        """Enqueues a byte-budgeted mip or LOD upload ordered by `(priority DESC, coarse mip_level DESC)`."""
        var clamped_bytes = max(1, byte_size)
        var lvl = max(0, mip_level)
        if (not is_mesh_lod) and tex_id >= 0 and tex_id < len(self.requested_min_mip):
            if lvl < self.requested_min_mip[tex_id]:
                self.requested_min_mip[tex_id] = lvl

        for i in range(len(self.upload_queue)):
            ref q = self.upload_queue[i]
            if q.target_id == tex_id and q.sub_level == lvl and q.is_mesh_lod == is_mesh_lod:
                if priority > q.priority:
                    q.priority = priority
                return

        var item = ResidencyUploadItem(tex_id, lvl, clamped_bytes, priority, is_mesh_lod)
        var ins = len(self.upload_queue)
        for i in range(len(self.upload_queue)):
            ref cur = self.upload_queue[i]
            if item.priority > cur.priority or (item.priority == cur.priority and item.sub_level > cur.sub_level):
                ins = i
                break
        self.upload_queue.insert(ins, item)

    def drain_uploads_within_budget(mut self, max_bytes_per_tick: Int) -> List[ResidencyUploadItem]:
        """Drains queued mip/LOD uploads in coarse-to-fine priority order within `max_bytes_per_tick`."""
        var drained = List[ResidencyUploadItem]()
        if max_bytes_per_tick <= 0 or len(self.upload_queue) == 0:
            self.drained_bytes_last_tick = 0
            return drained^

        var used_bytes = 0
        var idx = 0
        while idx < len(self.upload_queue):
            var cand = self.upload_queue[idx]
            if used_bytes + cand.byte_size <= max_bytes_per_tick or (used_bytes == 0 and len(drained) == 0):
                _ = self.upload_queue.pop(idx)
                used_bytes += cand.byte_size
                if (not cand.is_mesh_lod) and cand.target_id >= 0 and cand.target_id < len(self.resident_min_mip):
                    if cand.sub_level < self.resident_min_mip[cand.target_id]:
                        self.resident_min_mip[cand.target_id] = cand.sub_level
                drained.append(cand)
                if used_bytes >= max_bytes_per_tick:
                    break
            else:
                idx += 1

        self.drained_bytes_last_tick = used_bytes
        self.streamed_bytes_total += used_bytes
        self.streamed_chunks_last_tick = len(drained)
        return drained^

    def tick_streaming_budget(mut self, base_tex_dim: Int = 1024, byte_budget: Int = 512 * 1024) -> Int:
        """Streams missing high-res BC7 mip levels within `byte_budget` so frame time never hitches!"""
        if len(self.upload_queue) > 0:
            var q_drained = self.drain_uploads_within_budget(byte_budget)
            if len(q_drained) > 0:
                return len(q_drained)
        var used_bytes = 0
        var chunks = 0
        for tid in range(len(self.resident_min_mip)):
            while self.resident_min_mip[tid] > self.requested_min_mip[tid]:
                var next_mip = self.resident_min_mip[tid] - 1
                var mip_dim = max(4, base_tex_dim >> next_mip)
                var bc7_bytes = ((mip_dim + 3) // 4) * ((mip_dim + 3) // 4) * 16
                if used_bytes + bc7_bytes > byte_budget and chunks > 0:
                    self.streamed_chunks_last_tick = chunks
                    self.streamed_bytes_total += used_bytes
                    return chunks
                self.resident_min_mip[tid] = next_mip
                used_bytes += bc7_bytes
                chunks += 1
        self.streamed_chunks_last_tick = chunks
        self.streamed_bytes_total += used_bytes
        return chunks

    def tick_streaming(mut self, base_tex_dim: Int = 1024, byte_budget: Int = 512 * 1024) -> Int:
        return self.tick_streaming_budget(base_tex_dim, byte_budget)
