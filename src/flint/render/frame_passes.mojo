"""Compute passes another module adds to a frame (package 2.7: the GPU
particles), run by GpuRaster.submit right after the raster pass and before
bloom, on whichever backend draws. Each pass is a kernel-builder kernel
(render/kernel.mojo) dispatched directly or from three group counts in a
buffer (indirect: the GPU decided how much work there is), with a barrier
after it. The owner fills `passes` before the frame is submitted; the frame
empties it.
"""
from flint.platform.amdgpu import GpuBuffer


struct ExtraPass(Copyable, Movable):
    var code_address: UInt64    # amdgpu: the kernel's code
    var vgprs: UInt32
    var push_words: Int
    var table_address: UInt64   # amdgpu: the kernel's buffer address table
    var pipeline: Int           # Vulkan: the kernel's pipeline
    var descriptor: Int         # Vulkan: its buffers
    var push: List[UInt32]
    var groups_x: Int
    var groups_y: Int
    var indirect: Bool          # take the group counts from `indirect_offset` in the frame's indirect buffer
    var indirect_offset: Int    # bytes
    var feeds_indirect: Bool    # its output is a later pass's group counts (amdgpu: write L2 back after it)

    def __init__(out self):
        self.code_address = 0
        self.vgprs = 0
        self.push_words = 0
        self.table_address = 0
        self.pipeline = -1
        self.descriptor = 0
        self.push = List[UInt32]()
        self.groups_x = 0
        self.groups_y = 1
        self.indirect = False
        self.indirect_offset = 0
        self.feeds_indirect = False


struct FramePasses(Copyable, Movable):
    var passes: List[ExtraPass]
    var handles: List[UInt32]   # amdgpu buffer objects the passes touch
    var indirect: GpuBuffer     # where indirect passes read their group counts
    var zone: Int               # the GPU profiler zone the passes are timed as
    var copies: Int             # frames that ran passes

    def __init__(out self):
        self.passes = List[ExtraPass]()
        self.handles = List[UInt32]()
        self.indirect = GpuBuffer()
        self.zone = 0
        self.copies = 0

    def clear(mut self):
        self.passes.clear()
        self.handles.clear()

    def has_work(self) -> Bool:
        return len(self.passes) > 0
