"""GPU robustness helpers for `FlintAaaVulkanRenderer` (kept out of `aaa_renderer.mojo`).

  * `AaaAsyncUploadQueue`: uploads recorded into their own command buffers on the main
    queue and tracked by timeline value, so adding a mesh or texture never waits for the
    device to go idle. Staging and BLAS scratch buffers are retired when the GPU passes
    the upload's timeline value (`poll`), not at once.
  * `AaaDeferredMeshRemovals`: mesh ids (and their BLAS) whose geometry ranges may be reused
    only after every frame that could have drawn them has retired.
  * `AaaDynamicResolution`: a small controller that turns GPU frame times into a render scale.
  * `Bc7MipChain` / `encode_bc7_mip_chain*`: BC7 encoding as a pure function (optionally
    multi-threaded on a `JobPool`), callable from a loader thread, separate from the upload.
  * `AaaGpuRobustness`: the state the renderer keeps for all of the above plus device-loss
    recovery bookkeeping.
"""
from std.collections import List
from flint.assets.bc_compress import downsample_argb_2x, encode_bc7_rgba, encode_bc7_rgba_simd
from flint.core.jobs import JobPool
from flint.platform.amdgpu import GpuBuffer
from flint.platform.vulkan import VulkanAccelStruct, VulkanGpu

comptime AAA_ASYNC_UPLOAD_CMD_BUFFERS: Int = 4


struct _PendingUpload(Movable):
    var fence: UInt64
    var cmd_index: Int
    var buffers: List[GpuBuffer]

    def __init__(out self, fence: UInt64, cmd_index: Int, var buffers: List[GpuBuffer]):
        self.fence = fence
        self.cmd_index = cmd_index
        self.buffers = buffers^


struct AaaAsyncUploadQueue(Movable):
    var cmd_base: Int
    var cmd_count: Int
    var pending: List[_PendingUpload]
    var submitted: Int
    var retired: Int
    var forced_waits: Int  # times every upload command buffer was busy and the oldest had to be waited on

    def __init__(out self):
        self.cmd_base = -1
        self.cmd_count = 0
        self.pending = List[_PendingUpload]()
        self.submitted = 0
        self.retired = 0
        self.forced_waits = 0

    def ready(self) -> Bool:
        return self.cmd_base >= 0 and self.cmd_count > 0

    def init(mut self, mut vk: VulkanGpu, count: Int = AAA_ASYNC_UPLOAD_CMD_BUFFERS) -> Bool:
        if self.ready():
            return True
        if vk.device == 0:
            return False
        var base = vk.allocate_extra_command_buffers(count)
        if base < 0:
            return False
        self.cmd_base = base
        self.cmd_count = count
        return True

    def pending_count(self) -> Int:
        return len(self.pending)

    def _busy(self, cmd_index: Int) -> Bool:
        for i in range(len(self.pending)):
            if self.pending[i].cmd_index == cmd_index:
                return True
        return False

    def _retire_at(mut self, mut vk: VulkanGpu, i: Int):
        var p = self.pending.pop(i)
        for j in range(len(p.buffers)):
            vk.destroy_buffer(p.buffers[j])
        self.retired += 1

    def poll(mut self, mut vk: VulkanGpu):
        """Retires every upload the GPU has finished (non-blocking)."""
        if len(self.pending) == 0:
            return
        var done = vk.timeline_value()
        var i = 0
        while i < len(self.pending):
            if self.pending[i].fence != 0 and self.pending[i].fence <= done:
                self._retire_at(vk, i)
            else:
                i += 1

    def drain(mut self, mut vk: VulkanGpu):
        """Waits for and retires every pending upload."""
        while len(self.pending) > 0:
            if not vk.is_device_lost() and self.pending[0].fence != 0:
                _ = vk.wait(self.pending[0].fence)
            self._retire_at(vk, 0)

    def release_all(mut self, mut vk: VulkanGpu):
        """Drops every pending upload without waiting (the device is idle or lost)."""
        while len(self.pending) > 0:
            self._retire_at(vk, 0)

    def begin(mut self, mut vk: VulkanGpu) -> Int:
        """A free upload command buffer, begun; -1 on failure."""
        if not self.ready() and not self.init(vk):
            return -1
        self.poll(vk)
        var chosen = -1
        for k in range(self.cmd_count):
            if not self._busy(self.cmd_base + k):
                chosen = self.cmd_base + k
                break
        if chosen < 0:
            # All in flight: wait for the oldest (rare: needs more than `cmd_count` uploads per frame).
            self.forced_waits += 1
            chosen = self.pending[0].cmd_index
            if not vk.is_device_lost():
                _ = vk.wait(self.pending[0].fence)
            self._retire_at(vk, 0)
        if not vk.begin_commands(chosen):
            return -1
        return chosen

    def submit(mut self, mut vk: VulkanGpu, cmd_index: Int, var retire: List[GpuBuffer]) -> UInt64:
        """Submits the upload; `retire` buffers are destroyed once the GPU is past it. 0 on failure
        (the buffers are then destroyed at once)."""
        var f = vk.submit(cmd_index)
        if f == 0:
            if not vk.is_device_lost():
                vk.wait_idle()
            for j in range(len(retire)):
                vk.destroy_buffer(retire[j])
            return 0
        self.submitted += 1
        self.pending.append(_PendingUpload(f, cmd_index, retire^))
        return f


struct AaaDeferredMeshRemovals(Movable):
    """Mesh ids waiting for the frames that might still draw them to retire."""
    var mesh_ids: List[Int]
    var fences: List[UInt64]
    var blas: List[VulkanAccelStruct]
    var blas_fences: List[UInt64]

    def __init__(out self):
        self.mesh_ids = List[Int]()
        self.fences = List[UInt64]()
        self.blas = List[VulkanAccelStruct]()
        self.blas_fences = List[UInt64]()

    def pending_count(self) -> Int:
        return len(self.mesh_ids)

    def push(mut self, mesh_id: Int, fence: UInt64, var blas: VulkanAccelStruct):
        self.mesh_ids.append(mesh_id)
        self.fences.append(fence)
        self.blas.append(blas^)
        self.blas_fences.append(fence)

    def take_ready(mut self, mut vk: VulkanGpu, completed: UInt64, force: Bool) -> List[Int]:
        """Destroys retired BLAS and returns mesh ids whose ranges may now be reused."""
        var ready = List[Int]()
        var i = 0
        while i < len(self.mesh_ids):
            if force or self.fences[i] == 0 or self.fences[i] <= completed:
                ready.append(self.mesh_ids[i])
                _ = self.mesh_ids.pop(i)
                _ = self.fences.pop(i)
            else:
                i += 1
        var j = 0
        while j < len(self.blas):
            if force or self.blas_fences[j] == 0 or self.blas_fences[j] <= completed:
                var a = self.blas.pop(j)
                _ = self.blas_fences.pop(j)
                if a.handle != 0:
                    vk.destroy_accel_struct(a)
            else:
                j += 1
        return ready^


struct AaaDynamicResolution(Copyable, ImplicitlyCopyable, Movable):
    """Picks a render scale from GPU frame times: steps down quickly when over budget, up slowly when
    comfortably under it, and only moves after `settle_frames` frames so targets are not rebuilt every frame."""
    var target_ms: Float64
    var min_scale: Float32
    var max_scale: Float32
    var step: Float32
    var settle_frames: Int
    var scale: Float32
    var frames_since_change: Int
    var avg_ms: Float64

    def __init__(out self, target_ms: Float64 = 16.6, min_scale: Float32 = 0.5, max_scale: Float32 = 1.0):
        self.target_ms = target_ms
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.step = Float32(0.0625)
        self.settle_frames = 30
        self.scale = max_scale
        self.frames_since_change = 0
        self.avg_ms = 0.0

    def update(mut self, gpu_ms: Float64) -> Float32:
        if gpu_ms <= 0.0:
            return self.scale
        self.avg_ms = gpu_ms if self.avg_ms <= 0.0 else self.avg_ms * 0.85 + gpu_ms * 0.15
        self.frames_since_change += 1
        if self.frames_since_change < self.settle_frames:
            return self.scale
        if self.avg_ms > self.target_ms * 1.05 and self.scale > self.min_scale:
            self.scale = max(self.min_scale, self.scale - self.step)
            self.frames_since_change = 0
        elif self.avg_ms < self.target_ms * 0.75 and self.scale < self.max_scale:
            self.scale = min(self.max_scale, self.scale + self.step * Float32(0.5))
            self.frames_since_change = 0
        return self.scale


struct Bc7MipChain(Movable):
    """A BC7 mip pyramid ready to upload: all mips in one blob at 16-byte aligned offsets."""
    var width: Int
    var height: Int
    var data: List[UInt8]
    var offsets: List[Int]
    var widths: List[Int]
    var heights: List[Int]

    def __init__(out self):
        self.width = 0
        self.height = 0
        self.data = List[UInt8]()
        self.offsets = List[Int]()
        self.widths = List[Int]()
        self.heights = List[Int]()

    def mip_count(self) -> Int:
        return len(self.offsets)

    def valid(self) -> Bool:
        return len(self.offsets) > 0 and len(self.data) > 0

    def _append(mut self, bc7: List[UInt8], w: Int, h: Int):
        var off = len(self.data)
        self.offsets.append(off)
        self.widths.append(w)
        self.heights.append(h)
        for b in bc7:
            self.data.append(b)
        while len(self.data) % 16 != 0:
            self.data.append(0)


def bc7_mip_count(width: Int, height: Int, generate_mips: Bool) -> Int:
    """The mip count `upload_bc7_texture` has always used (stop at 4 px, at most 8 levels)."""
    var mips = 1
    if generate_mips:
        var d = min(width, height)
        while d > 4 and mips < 8:
            d >>= 1
            mips += 1
    return mips


def encode_bc7_mip_chain(pixels_argb: List[UInt32], width: Int, height: Int, generate_mips: Bool = True) -> Bc7MipChain:
    """Encodes `0xAARRGGBB` pixels into a BC7 pyramid. Pure: touches no GPU state, so a loader thread
    can run it while the renderer keeps drawing."""
    var chain = Bc7MipChain()
    chain.width = width
    chain.height = height
    var mips = bc7_mip_count(width, height, generate_mips)
    var cur = pixels_argb.copy()
    var cw = width
    var ch = height
    for m in range(mips):
        var bc7 = encode_bc7_rgba(cur, cw, ch)
        if len(bc7) == 0:
            break
        chain._append(bc7, cw, ch)
        if m + 1 < mips:
            var ds = downsample_argb_2x(cur, cw, ch)
            cur = ds[0].copy()
            cw = ds[1]
            ch = ds[2]
    return chain^


def encode_bc7_mip_chain_parallel(
    pixels_argb: List[UInt32], width: Int, height: Int, mut pool: JobPool, generate_mips: Bool = True
) -> Bc7MipChain:
    """As `encode_bc7_mip_chain`, with each mip's blocks encoded across the `JobPool` workers."""
    if not pool.ready():
        return encode_bc7_mip_chain(pixels_argb, width, height, generate_mips)
    var chain = Bc7MipChain()
    chain.width = width
    chain.height = height
    var mips = bc7_mip_count(width, height, generate_mips)
    var cur = pixels_argb.copy()
    var cw = width
    var ch = height
    for m in range(mips):
        var bc7 = encode_bc7_rgba_simd(cur, cw, ch, pool)
        if len(bc7) == 0:
            break
        chain._append(bc7, cw, ch)
        if m + 1 < mips:
            var ds = downsample_argb_2x(cur, cw, ch)
            cur = ds[0].copy()
            cw = ds[1]
            ch = ds[2]
    return chain^


struct AaaGpuRobustness(Movable):
    """Renderer-side robustness state (one field in `FlintAaaVulkanRenderer`)."""
    var uploads: AaaAsyncUploadQueue
    var removals: AaaDeferredMeshRemovals
    var device_generation: Int  # bumps on every successful device recreation: re-upload textures then
    var recoveries: Int
    var recovery_failures: Int
    var last_recovery_error: String
    var meshes_lost_on_recovery: Int  # skinned/splat meshes whose GPU state could not be rebuilt
    var textures_lost_on_recovery: Int  # bindless textures now showing the fallback white
    var oom_fallbacks: Int  # allocations that failed and were degraded (evicted mips, skipped BLAS)
    var last_upload_failed: Bool  # the last `_upload_mesh_ranges` could not upload the geometry
    var streamed_slots: List[Int]  # bindless slots from `allocate_bindless_texture_slot`: first to evict on OOM

    def __init__(out self):
        self.uploads = AaaAsyncUploadQueue()
        self.removals = AaaDeferredMeshRemovals()
        self.device_generation = 0
        self.recoveries = 0
        self.recovery_failures = 0
        self.last_recovery_error = ""
        self.meshes_lost_on_recovery = 0
        self.textures_lost_on_recovery = 0
        self.oom_fallbacks = 0
        self.last_upload_failed = False
        self.streamed_slots = List[Int]()
