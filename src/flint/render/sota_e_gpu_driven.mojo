"""Area E (`sota:E`): GPU-driven packing of the alpha-mask (Bucket 2) and cel-outline draw lists.

The v5 renderer walked every instance on the CPU each frame (`_pack_alphamask_visbuffer_commands`,
`_pack_outline_visbuffer_commands`: frustum test, discrete LOD pick, per-meshlet command writes) and
drew the result with a CPU-known count. With `SotaEGpuDriven` active both lists are produced by Area A's
cluster culler (`sota_cluster_cull_shader(route)`) on the GPU, in the same command buffer as the main
phase-1 cull:

  route 1  alpha-masked instances only (word 39 bit 17), into `ring_alphamask_indirect[slot]`
  route 2  cel-outlined instances only (word 39 bit 19, set by the renderer), into `ring_outline_indirect[slot]`

Both run with cone culling off (alpha-masked meshlets are double-sided; the outline pass draws back faces
of the inflated hull, which cone culling would reject) and occlusion off (the occlusion candidate queue
belongs to the main pass), so they keep the per-meshlet frustum test + continuous DAG LOD cut. The draws use
`vkCmdDrawIndexedIndirectCount` with the GPU-written counter (word 0), so the CPU only zeroes two words.
Needs `drawIndirectCount` (Vulkan 1.2 core feature); otherwise the CPU packers run as before.

The VSM page packer (`_pack_vsm_indirect_commands`) stays on the CPU: its per-page offsets drive per-page
viewport draws, so it needs per-page GPU slabs with their own count words first.

cvar: `r.sota_e.gpu_driven` = `SotaEGpuDriven.enabled` (default on); env `FLINT_SOTA_E_GPU_DRIVEN=0` turns it off.

Sources: Haar & Aaltonen "GPU-Driven Rendering Pipelines" (SIGGRAPH 2015); Wihlidal "Optimizing the
Graphics Pipeline with Compute" (GDC 2016).
"""
from std.collections import List
from std.memory import bitcast
from flint.platform.amdgpu import GpuBuffer
from flint.platform.os import env_value
from flint.platform.vulkan import VulkanGpu, put32
from flint.render.sota_geometry_cull import (
    SotaGeometryGpu,
    sota_cluster_cull_shader,
    SOTA_BIND_MESHLETS,
    SOTA_BIND_INSTANCES,
    SOTA_BIND_INDIRECT,
    SOTA_BIND_GLOBALS,
    SOTA_BIND_CANDIDATES,
    SOTA_BIND_DAG,
    SOTA_BIND_HIZ,
    SOTA_BIND_STATS,
    SOTA_BIND_DEPTH,
    SOTA_PUSH_WORDS,
    SOTA_SAMPLER_COUNT,
    SOTA_STORAGE_COUNT,
)

comptime SOTA_E_INSTANCE_OUTLINE_BIT: UInt32 = 0x00080000   # instance word 39, bit 19: cel-outlined material
comptime SOTA_E_ROUTE_MAX_DRAWS: Int = 16384               # capacity of the alpha-mask / outline indirect rings


struct SotaEGpuDriven(Movable):
    var enabled: Bool        # cvar r.sota_e.gpu_driven
    var active: Bool
    var pipe_alpha: Int
    var pipe_outline: Int
    var pool: Int
    var sets_alpha: List[Int]
    var sets_outline: List[Int]
    var frame_alpha_instances: Int     # filled by the renderer's instance upload loop
    var frame_outline_instances: Int

    def __init__(out self):
        self.enabled = env_value("FLINT_SOTA_E_GPU_DRIVEN") != "0"
        self.active = False
        self.pipe_alpha = 0
        self.pipe_outline = 0
        self.pool = 0
        self.sets_alpha = List[Int]()
        self.sets_outline = List[Int]()
        self.frame_alpha_instances = 0
        self.frame_outline_instances = 0

    def init(
        mut self,
        mut vk: VulkanGpu,
        geom: SotaGeometryGpu,
        frames: Int,
        buf_meshlets: GpuBuffer,
        ring_instances: List[GpuBuffer],
        ring_alpha: List[GpuBuffer],
        ring_outline: List[GpuBuffer],
        ring_globals: List[GpuBuffer],
        buf_candidates: GpuBuffer,
        depth_view: Int,
        sampler: Int,
    ) -> Bool:
        self.active = False
        if not geom.active or not vk.has_draw_indirect_count:
            return False
        if len(ring_alpha) < frames or len(ring_outline) < frames or len(geom.buf_stats) < frames:
            return False
        var ps = vk.allocate_aaa_descriptor_sets(geom.dsl, frames * 2, SOTA_STORAGE_COUNT, SOTA_SAMPLER_COUNT, 0, False)
        self.pool = ps[0]
        var sets = ps[1].copy()
        if len(sets) < frames * 2:
            return False
        self.pipe_alpha = vk.create_aaa_compute_pipeline(geom.layout, sota_cluster_cull_shader(1))
        self.pipe_outline = vk.create_aaa_compute_pipeline(geom.layout, sota_cluster_cull_shader(2))
        if self.pipe_alpha == 0 or self.pipe_outline == 0:
            return False
        self.sets_alpha = List[Int]()
        self.sets_outline = List[Int]()
        for f in range(frames):
            for r in range(2):
                var ds = sets[f * 2 + r]
                vk.write_descriptor(ds, SOTA_BIND_MESHLETS, buf_meshlets)
                vk.write_descriptor(ds, SOTA_BIND_INSTANCES, ring_instances[f])
                if r == 0:
                    vk.write_descriptor(ds, SOTA_BIND_INDIRECT, ring_alpha[f])
                else:
                    vk.write_descriptor(ds, SOTA_BIND_INDIRECT, ring_outline[f])
                vk.write_descriptor(ds, SOTA_BIND_GLOBALS, ring_globals[f])
                vk.write_descriptor(ds, SOTA_BIND_CANDIDATES, buf_candidates)   # bound, never written (occlusion off)
                vk.write_descriptor(ds, SOTA_BIND_DAG, geom.buf_dag)
                vk.write_descriptor(ds, SOTA_BIND_HIZ, geom.buf_hiz)
                vk.write_descriptor(ds, SOTA_BIND_STATS, geom.buf_stats[f])   # bound, never written (stats off)
                vk.write_aaa_texture(ds, SOTA_BIND_DEPTH, 0, sampler, depth_view)
                if r == 0:
                    self.sets_alpha.append(ds)
                else:
                    self.sets_outline.append(ds)
        self.active = True
        return True

    def use(self, geom: SotaGeometryGpu) -> Bool:
        return self.enabled and self.active and geom.use_cull()

    def begin_frame(mut self):
        self.frame_alpha_instances = 0
        self.frame_outline_instances = 0

    def reset_counters(self, alpha_buf: GpuBuffer, outline_buf: GpuBuffer):
        """CPU: zero the GPU draw counters of this slot (its fence has signalled)."""
        if alpha_buf.cpu_address != 0:
            put32(alpha_buf.cpu_address, 0)
        if outline_buf.cpu_address != 0:
            put32(outline_buf.cpu_address, 0)

    def record(
        mut self, mut vk: VulkanGpu, geom: SotaGeometryGpu, slot: Int, n_inst: Int, max_ml_per_inst: Int,
        render_w: Int, render_h: Int, lod_px: Float32, do_alpha: Bool, do_outline: Bool,
        max_draws: Int = SOTA_E_ROUTE_MAX_DRAWS,  # --- sota:perf hook --- ring capacity (grows with the scene)
    ):
        """Records the route dispatches; the caller's compute -> DRAW_INDIRECT barrier covers them."""
        if n_inst <= 0 or max_ml_per_inst <= 0:
            return
        var push = List[UInt32](length=SOTA_PUSH_WORDS, fill=0)
        push[0] = UInt32(n_inst)
        push[1] = UInt32(max_draws)  # --- sota:perf hook ---
        push[4] = UInt32(render_w)
        push[5] = UInt32(render_h)
        push[6] = bitcast[DType.uint32](lod_px)
        var gx = (n_inst + 7) // 8
        var gy = (max_ml_per_inst + 7) // 8
        if do_alpha:
            vk.dispatch_aaa(slot, self.pipe_alpha, geom.layout, self.sets_alpha[slot], push, gx, gy)
        if do_outline:
            vk.dispatch_aaa(slot, self.pipe_outline, geom.layout, self.sets_outline[slot], push, gx, gy)

    def shutdown(mut self, mut vk: VulkanGpu):
        if self.pipe_alpha != 0:
            vk.destroy_pipeline(self.pipe_alpha)
        if self.pipe_outline != 0:
            vk.destroy_pipeline(self.pipe_outline)
        self.pipe_alpha = 0
        self.pipe_outline = 0
        if self.pool != 0:
            vk.destroy_aaa_descriptors(self.pool, 0, 0)
        self.pool = 0
        self.active = False
