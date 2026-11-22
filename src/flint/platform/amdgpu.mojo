"""Direct amdgpu kernel interface for Radeon GFX12 and GFX10.3 GPUs.

Mojo owns this path the same way it owns Wayland: requests are encoded here and
go straight to the render node through ioctl. Vulkan, Mesa, and libdrm are not
used. Struct sizes and ioctl numbers match the kernel's amdgpu_drm.h.
"""
from std.ffi import external_call
from std.sys.info import size_of
from std.time import monotonic
from flint.platform.amd_isa import ISA_GFX12, ISA_GFX103
from flint.platform.os import close_fd, env_value, last_errno, read_stream

comptime IOCTL_PRIME_HANDLE_TO_FD: UInt64 = 0xC00C642D
comptime PRIME_FLAGS: UInt32 = 0x80002  # DRM_CLOEXEC | DRM_RDWR
comptime IOCTL_GEM_CLOSE: UInt64 = 0x40086409
comptime IOCTL_GEM_CREATE: UInt64 = 0xC0206440
comptime IOCTL_GEM_MMAP: UInt64 = 0xC0086441
comptime IOCTL_CTX: UInt64 = 0xC0106442
comptime IOCTL_CS: UInt64 = 0xC0186444
comptime IOCTL_INFO: UInt64 = 0x40206445
comptime IOCTL_GEM_VA: UInt64 = 0x40406448
comptime IOCTL_WAIT_CS: UInt64 = 0xC0206449

comptime INFO_TIMESTAMP: UInt32 = 0x05
comptime INFO_DEV_INFO: UInt32 = 0x16
comptime INFO_HW_IP_INFO: UInt32 = 0x02
comptime DOMAIN_GTT: UInt64 = 2
comptime DOMAIN_VRAM: UInt64 = 4
comptime HW_IP_COMPUTE: UInt32 = 1
comptime CHUNK_IB: UInt32 = 1
comptime CHUNK_BO_HANDLES: UInt32 = 6
comptime GEM_CREATE_CPU_ACCESS: UInt64 = 1

comptime VA_OP_MAP: UInt32 = 1
comptime VA_OP_UNMAP: UInt32 = 2
comptime PAGE_READ: UInt32 = 2
comptime PAGE_WRITE: UInt32 = 4
comptime PAGE_EXECUTE: UInt32 = 8

comptime PM4_NOP: UInt32 = 0xFFFF1000
# Kernel family ids: discrete GFX12, and the integrated GFX10.3 parts this code has
# encodings for. Discrete GFX10 shares a family id across two ISAs, so it is left out.
comptime FAMILY_GC_12_0_0: UInt32 = 152
comptime FAMILY_GC_10_3_3: UInt32 = 146
comptime FAMILY_GC_10_3_6: UInt32 = 149
comptime FAMILY_GC_10_3_7: UInt32 = 151
comptime WAIT_TIMEOUT_NS: UInt64 = 2000000000  # the default; SD_GPU_FENCE_TIMEOUT_MS or gpu.fence_timeout_ms changes it

# Context operations and what AMDGPU_CTX_OP_QUERY_STATE2 reports.
comptime CTX_OP_ALLOC: UInt64 = 1
comptime CTX_OP_FREE: UInt64 = 2
comptime CTX_OP_QUERY_STATE2: UInt64 = 4
comptime CTX_QUERY2_FLAGS_RESET: UInt64 = 1  # the GPU was reset since the context was made
comptime CTX_QUERY2_FLAGS_VRAMLOST: UInt64 = 2
comptime CTX_QUERY2_FLAGS_GUILTY: UInt64 = 4  # this context's work caused it
comptime CTX_QUERY2_FLAGS_RAS_CE: UInt64 = 8
comptime CTX_QUERY2_FLAGS_RAS_UE: UInt64 = 16
comptime ECANCELED: Int = 125  # a submission to a context that a reset made unusable
comptime ENODEV: Int = 19


struct GpuState(Copyable, Movable):
    """What the kernel says about a context: AMDGPU_CTX_OP_QUERY_STATE2."""
    var queried: Bool
    var flags: UInt64
    var hangs: UInt32
    var reset_status: UInt32

    def __init__(out self):
        self.queried = False
        self.flags = 0
        self.hangs = 0
        self.reset_status = 0

    def reset(self) -> Bool:
        return (self.flags & CTX_QUERY2_FLAGS_RESET) != 0

    def guilty(self) -> Bool:
        return (self.flags & CTX_QUERY2_FLAGS_GUILTY) != 0

    def vram_lost(self) -> Bool:
        return (self.flags & CTX_QUERY2_FLAGS_VRAMLOST) != 0

    def describe(self) -> String:
        if not self.queried:
            return "state unknown (query failed)"
        var text = "reset=" + String(self.reset()) + " guilty=" + String(self.guilty()) + " vram_lost=" + String(self.vram_lost())
        if (self.flags & (CTX_QUERY2_FLAGS_RAS_CE | CTX_QUERY2_FLAGS_RAS_UE)) != 0:
            text += " ras_errors=" + hex(self.flags)
        return text + " hangs=" + String(self.hangs)


def pkt3(opcode: UInt32, body_dwords: UInt32) -> UInt32:
    """PM4 type-3 header for a packet with the given number of body dwords."""
    return (UInt32(3) << 30) | ((body_dwords - 1) << 16) | (opcode << 8)


def gpu_ioctl(fd: Int, request: UInt64, mut args: List[UInt64]) -> Int:
    return Int(external_call["ioctl", Int32](Int32(fd), request, args.unsafe_ptr()))


comptime ReleaseVkBufferFn = def(Int, Int, Int, Int, Int, UInt32, Bool) thin abi("C") -> Int32


struct _AmdDeviceControlBlock:
    var ref_count: Int
    var gpu_ref_count: Int
    var fd: Int
    var alive: Bool


struct _GpuBufferControl:
    var ref_count: Int
    var destroyed: Bool
    var backend_kind: Int  # 0 = non-owning/empty, 1 = vulkan, 2 = amdgpu
    var ctrl_ptr: Int
    var release_fn: Int
    var vk_buffer: Int
    var vk_memory: Int
    var vk_memory_offset: Int
    var vk_suballoc_block: Int
    var vk_suballoc_meta: UInt32
    var amd_handle: UInt32
    var amd_gpu_address: UInt64
    var amd_cpu_address: Int
    var amd_size: Int


@always_inline
def _amd_ctrl_ptr(addr: Int) -> Pointer[_AmdDeviceControlBlock, MutAnyOrigin]:
    return Pointer[_AmdDeviceControlBlock, MutAnyOrigin](unsafe_from_address=addr)


@always_inline
def _buf_ctrl_ptr(addr: Int) -> Pointer[_GpuBufferControl, MutAnyOrigin]:
    return Pointer[_GpuBufferControl, MutAnyOrigin](unsafe_from_address=addr)


def _amd_release_buffer_handles(fd: Int, handle: UInt32, gpu_address: UInt64, cpu_address: Int, size: Int):
    if size <= 0 or fd < 0:
        return
    if cpu_address != 0:
        _ = external_call["munmap", Int32](cpu_address, size)
    var va = List[UInt64](length=8, fill=0)
    va[0] = UInt64(handle)
    va[1] = UInt64(VA_OP_UNMAP)
    va[2] = gpu_address
    va[4] = UInt64(size)
    _ = gpu_ioctl(fd, IOCTL_GEM_VA, va)
    var close_args = List[UInt64](length=1, fill=0)
    close_args[0] = UInt64(handle)
    _ = gpu_ioctl(fd, IOCTL_GEM_CLOSE, close_args)


struct GpuBuffer(Movable):
    var handle: UInt32
    var size: Int
    var gpu_address: UInt64
    var cpu_address: Int
    var vk_buffer: Int  # the Vulkan backend's two objects, which have no GPU address
    var vk_memory: Int
    var vk_memory_offset: Int
    var vk_suballoc_block: Int
    var vk_suballoc_meta: UInt32
    var tag: Int  # the memory tag it is charged to (core/memory.mojo), -1 when not charged
    var vram: Bool
    var state_ptr: Int

    def __init__(out self, handle: UInt32 = 0, size: Int = 0, gpu_address: UInt64 = 0, cpu_address: Int = 0):
        self.handle = handle
        self.size = size
        self.gpu_address = gpu_address
        self.cpu_address = cpu_address
        self.vk_buffer = 0
        self.vk_memory = 0
        self.vk_memory_offset = 0
        self.vk_suballoc_block = -1
        self.vk_suballoc_meta = 0xFFFFFFFF
        self.tag = -1
        self.vram = False
        self.state_ptr = 0

    def __moveinit__(out self, deinit existing: Self):
        self.handle = existing.handle
        self.size = existing.size
        self.gpu_address = existing.gpu_address
        self.cpu_address = existing.cpu_address
        self.vk_buffer = existing.vk_buffer
        self.vk_memory = existing.vk_memory
        self.vk_memory_offset = existing.vk_memory_offset
        self.vk_suballoc_block = existing.vk_suballoc_block
        self.vk_suballoc_meta = existing.vk_suballoc_meta
        self.tag = existing.tag
        self.vram = existing.vram
        self.state_ptr = existing.state_ptr
        existing.handle = 0
        existing.size = 0
        existing.gpu_address = 0
        existing.cpu_address = 0
        existing.vk_buffer = 0
        existing.vk_memory = 0
        existing.vk_memory_offset = 0
        existing.vk_suballoc_block = -1
        existing.vk_suballoc_meta = 0xFFFFFFFF
        existing.tag = -1
        existing.vram = False
        existing.state_ptr = 0

    def __deinit__(deinit self):
        if self.state_ptr == 0:
            return
        ref st = _buf_ctrl_ptr(self.state_ptr)[unsafe_offset=0]
        st.ref_count -= 1
        if st.ref_count > 0:
            return
        if st.backend_kind == 1 and st.release_fn != 0 and st.ctrl_ptr != 0:
            var fn_slot = st.release_fn
            var cb = Pointer(to=fn_slot).unsafe_bitcast[ReleaseVkBufferFn]()[unsafe_offset=0]
            _ = cb(
                st.ctrl_ptr,
                st.vk_buffer,
                st.vk_memory,
                st.vk_memory_offset,
                st.vk_suballoc_block,
                st.vk_suballoc_meta,
                st.destroyed,
            )
        elif st.backend_kind == 2 and st.ctrl_ptr != 0:
            ref actrl = _amd_ctrl_ptr(st.ctrl_ptr)[unsafe_offset=0]
            if not st.destroyed and actrl.alive:
                _amd_release_buffer_handles(
                    actrl.fd, st.amd_handle, st.amd_gpu_address, st.amd_cpu_address, st.amd_size
                )
            actrl.ref_count -= 1
            if actrl.ref_count <= 0:
                external_call["free", NoneType](st.ctrl_ptr)
        external_call["free", NoneType](self.state_ptr)

    def read32(self, index: Int) -> UInt32:
        return Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=self.cpu_address + index * 4)[unsafe_offset=0]

    def write32(self, index: Int, value: UInt32):
        var pointer = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=self.cpu_address + index * 4)
        pointer[unsafe_offset=0] = value


struct AmdGpu(Movable):
    var fd: Int
    var context: UInt32
    var device_id: UInt32
    var family: UInt32
    var shader_engines: UInt32
    var clock_khz: UInt32
    var active_compute_units: UInt32
    var va_start: UInt64
    var va_end: UInt64
    var next_va: UInt64
    var ib_start_alignment: Int
    var ib_size_alignment: Int
    var compute_rings: UInt32
    var error: String
    var timed_out: Bool  # the last failed wait ran out of time; the GPU may be hung
    var lost: Bool  # a submission was refused because a reset made the context unusable
    var minor: Int
    var ctrl_ptr: Int

    def __init__(out self):
        self.fd = -1
        self.context = 0
        self.device_id = 0
        self.family = 0
        self.shader_engines = 0
        self.clock_khz = 0
        self.active_compute_units = 0
        self.va_start = 0
        self.va_end = 0
        self.next_va = 0
        self.ib_start_alignment = 256
        self.ib_size_alignment = 256
        self.compute_rings = 0
        self.error = ""
        self.timed_out = False
        self.lost = False
        self.minor = -1
        self.ctrl_ptr = 0

    def __moveinit__(out self, deinit existing: Self):
        self.fd = existing.fd
        self.context = existing.context
        self.device_id = existing.device_id
        self.family = existing.family
        self.shader_engines = existing.shader_engines
        self.clock_khz = existing.clock_khz
        self.active_compute_units = existing.active_compute_units
        self.va_start = existing.va_start
        self.va_end = existing.va_end
        self.next_va = existing.next_va
        self.ib_start_alignment = existing.ib_start_alignment
        self.ib_size_alignment = existing.ib_size_alignment
        self.compute_rings = existing.compute_rings
        self.error = existing.error^
        self.timed_out = existing.timed_out
        self.lost = existing.lost
        self.minor = existing.minor
        self.ctrl_ptr = existing.ctrl_ptr
        existing.fd = -1
        existing.context = 0
        existing.minor = -1
        existing.ctrl_ptr = 0

    def __deinit__(deinit self):
        if self.ctrl_ptr != 0:
            ref ctrl = _amd_ctrl_ptr(self.ctrl_ptr)[unsafe_offset=0]
            if ctrl.gpu_ref_count > 1:
                ctrl.gpu_ref_count -= 1
                ctrl.ref_count -= 1
                return
        self.close()

    def fail(mut self, message: String) -> Bool:
        self.error = message
        return False

    def open_node(mut self, minor: Int) -> Bool:
        # Render nodes must be opened read/write for command submission.
        var path = "/dev/dri/renderD" + String(minor)
        var fd = Int(external_call["openat", Int32](Int32(-100), path.as_c_string_span(), Int32(2), Int32(0)))
        if fd < 0:
            return False
        if self.read_device_info(fd) and self.supported():
            if self.ctrl_ptr != 0:
                self.close()
            self.fd = fd
            self.minor = minor
            var c_raw = external_call["calloc", Int](Int(1), max(64, size_of[_AmdDeviceControlBlock]()))
            ref ctrl = _amd_ctrl_ptr(c_raw)[unsafe_offset=0]
            ctrl.ref_count = 1
            ctrl.gpu_ref_count = 1
            ctrl.fd = fd
            ctrl.alive = True
            self.ctrl_ptr = c_raw
            return True
        close_fd(fd)
        return False

    def supported(self) -> Bool:
        return self.family == FAMILY_GC_12_0_0 or self.integrated()

    def integrated(self) -> Bool:
        return self.family == FAMILY_GC_10_3_3 or self.family == FAMILY_GC_10_3_6 or self.family == FAMILY_GC_10_3_7

    def isa(self) -> Int:
        return ISA_GFX12 if self.family == FAMILY_GC_12_0_0 else ISA_GFX103

    def open_device(mut self) -> Bool:
        """Find a render node by asking each one who it is.

        SD_GPU_NODE names one to use. Otherwise the fastest supported GPU wins, which
        means a discrete GFX12 card ahead of any integrated part.
        """
        var requested = env_value("SD_GPU_NODE")
        if requested.byte_length() > 0:
            try:
                if self.open_node(Int(requested)):
                    return True
            except:
                pass
            return self.fail("SD_GPU_NODE does not name a supported render node")
        for prefer_discrete in [True, False]:
            for minor in range(128, 136):
                if self.open_node(minor):
                    if not prefer_discrete or not self.integrated():
                        return True
                    self.close()
        return self.fail("no supported GPU render node found")

    def read_device_info(mut self, fd: Int) -> Bool:
        var info = List[UInt32](length=110, fill=0)  # drm_amdgpu_info_device: 440 bytes
        var request = List[UInt64](length=4, fill=0)
        request[0] = UInt64(Int(info.unsafe_ptr()))
        request[1] = UInt64(440) | (UInt64(INFO_DEV_INFO) << 32)
        if gpu_ioctl(fd, IOCTL_INFO, request) != 0:
            return False
        _ = request^
        self.device_id = info[0]
        self.family = info[4]
        self.shader_engines = info[5]
        self.clock_khz = info[7]  # gpu_counter_freq: the timestamp clock
        self.active_compute_units = info[12]
        self.va_start = UInt64(info[36]) | (UInt64(info[37]) << 32)
        self.va_end = UInt64(info[38]) | (UInt64(info[39]) << 32)
        return True

    def read_clock(self) -> UInt64:
        """The GPU's timestamp counter now, the one PM4 timestamps read. 0 on failure."""
        var value = List[UInt64](length=1, fill=0)
        var request = List[UInt64](length=4, fill=0)
        request[0] = UInt64(Int(value.unsafe_ptr()))
        request[1] = UInt64(8) | (UInt64(INFO_TIMESTAMP) << 32)
        if gpu_ioctl(self.fd, IOCTL_INFO, request) != 0:
            return 0
        _ = request^
        return value[0]

    def query_compute_alignment(mut self) -> Bool:
        """Read the queue's required command-buffer start and size alignment."""
        var info = List[UInt32](length=10, fill=0)  # drm_amdgpu_info_hw_ip: 40 bytes
        var request = List[UInt64](length=4, fill=0)
        request[0] = UInt64(Int(info.unsafe_ptr()))
        request[1] = UInt64(40) | (UInt64(INFO_HW_IP_INFO) << 32)
        request[2] = UInt64(HW_IP_COMPUTE)
        if gpu_ioctl(self.fd, IOCTL_INFO, request) != 0:
            return self.fail("compute queue query failed")
        _ = request^
        self.ib_start_alignment = Int(info[4])
        self.ib_size_alignment = Int(info[5])
        self.compute_rings = info[6]
        return self.compute_rings != 0 or self.fail("no compute queue available")

    def create_context(mut self) -> Bool:
        var args = List[UInt64](length=2, fill=0)
        args[0] = CTX_OP_ALLOC
        if gpu_ioctl(self.fd, IOCTL_CTX, args) != 0:
            return self.fail("context allocation failed")
        self.context = UInt32(args[0] & 0xFFFFFFFF)
        # Leave the kernel's reserved low range and start on a 2 MiB boundary.
        self.next_va = (max(self.va_start, UInt64(0x1000000)) + 0x1FFFFF) & ~UInt64(0x1FFFFF)
        return True

    def query_state(self) -> GpuState:
        """Ask the kernel whether the GPU was reset under this context, and whether
        this context caused it (AMDGPU_CTX_OP_QUERY_STATE2)."""
        var state = GpuState()
        if self.fd < 0:
            return state^
        var args = List[UInt64](length=2, fill=0)  # drm_amdgpu_ctx_in, then _out over it
        args[0] = CTX_OP_QUERY_STATE2
        args[1] = UInt64(self.context)
        if gpu_ioctl(self.fd, IOCTL_CTX, args) != 0:
            return state^
        state.queried = True
        state.flags = args[0]
        state.hangs = UInt32(args[1] & 0xFFFFFFFF)
        state.reset_status = UInt32(args[1] >> 32)
        return state^

    def free_context(mut self):
        if self.fd < 0 or self.context == 0:
            return
        var args = List[UInt64](length=2, fill=0)
        args[0] = CTX_OP_FREE
        args[1] = UInt64(self.context)
        _ = gpu_ioctl(self.fd, IOCTL_CTX, args)
        self.context = 0

    def kernel_memory(self, key: String) -> Int:
        """What the kernel charges this process's GPU file for, in bytes:
        the `drm-<key>` line of /proc/self/fdinfo (`total-vram`,
        `resident-gtt`, ...). -1 when the file is closed or the line missing.
        The memory report checks its VRAM tags against it."""
        if self.fd < 0:
            return -1
        return drm_fdinfo_bytes(self.fd, key)

    def close(mut self):
        """Free the context and close the device. The kernel frees every buffer with
        the file, and stops what this file still had queued."""
        if self.ctrl_ptr != 0:
            ref ctrl = _amd_ctrl_ptr(self.ctrl_ptr)[unsafe_offset=0]
            if ctrl.alive:
                self.free_context()
                if self.fd >= 0:
                    close_fd(self.fd)
                ctrl.alive = False
                ctrl.fd = -1
            self.fd = -1
            self.context = 0
            self.minor = -1
            ctrl.gpu_ref_count -= 1
            ctrl.ref_count -= 1
            if ctrl.ref_count <= 0:
                external_call["free", NoneType](self.ctrl_ptr)
            self.ctrl_ptr = 0
        else:
            self.free_context()
            if self.fd >= 0:
                close_fd(self.fd)
            self.fd = -1
            self.context = 0
            self.minor = -1

    def create_buffer(mut self, size: Int, domain: UInt64, flags: UInt32) -> GpuBuffer:
        """Allocate, map for the GPU, and map for the CPU. Size 0 reports failure."""
        var rounded = (size + 4095) & ~4095
        var create = List[UInt64](length=4, fill=0)
        create[0] = UInt64(rounded)
        create[1] = 4096
        create[2] = domain
        create[3] = GEM_CREATE_CPU_ACCESS
        if gpu_ioctl(self.fd, IOCTL_GEM_CREATE, create) != 0:
            _ = self.fail("buffer allocation failed")
            return GpuBuffer()
        var handle = UInt32(create[0] & 0xFFFFFFFF)

        var address = self.next_va
        var va = List[UInt64](length=8, fill=0)
        va[0] = UInt64(handle)
        va[1] = UInt64(VA_OP_MAP) | (UInt64(flags) << 32)
        va[2] = address
        va[3] = 0
        va[4] = UInt64(rounded)
        if gpu_ioctl(self.fd, IOCTL_GEM_VA, va) != 0:
            _ = self.fail("GPU address mapping failed")
            return GpuBuffer()
        self.next_va = (address + UInt64(rounded) + 0x1FFFFF) & ~UInt64(0x1FFFFF)

        var map_request = List[UInt64](length=1, fill=0)
        map_request[0] = UInt64(handle)
        if gpu_ioctl(self.fd, IOCTL_GEM_MMAP, map_request) != 0:
            _ = self.fail("CPU mapping lookup failed")
            return GpuBuffer()
        var cpu = external_call["mmap", Int](Int(0), Int(rounded), Int32(3), Int32(1), Int32(self.fd), Int(map_request[0]))
        if cpu == -1:
            _ = self.fail("CPU mapping failed")
            return GpuBuffer()
        var out = GpuBuffer(handle, rounded, address, cpu)
        if self.ctrl_ptr != 0:
            var st_raw = external_call["calloc", Int](Int(1), max(128, size_of[_GpuBufferControl]()))
            ref st = _buf_ctrl_ptr(st_raw)[unsafe_offset=0]
            st.ref_count = 1
            st.destroyed = False
            st.backend_kind = 2
            st.ctrl_ptr = self.ctrl_ptr
            st.amd_handle = handle
            st.amd_gpu_address = address
            st.amd_cpu_address = cpu
            st.amd_size = rounded
            ref actrl = _amd_ctrl_ptr(self.ctrl_ptr)[unsafe_offset=0]
            actrl.ref_count += 1
            out.state_ptr = st_raw
        return out^

    def destroy_buffer(mut self, buffer: GpuBuffer):
        """Release a buffer the GPU has finished with. The empty value is ignored."""
        if buffer.size == 0:
            return
        if buffer.state_ptr != 0:
            ref st = _buf_ctrl_ptr(buffer.state_ptr)[unsafe_offset=0]
            if st.destroyed:
                return
            st.destroyed = True
        _amd_release_buffer_handles(self.fd, buffer.handle, buffer.gpu_address, buffer.cpu_address, buffer.size)

    def export_buffer(mut self, buffer: GpuBuffer) -> Int:
        """Share a buffer as a dmabuf file descriptor. Returns -1 on failure."""
        var args = List[UInt32](length=3, fill=0)  # drm_prime_handle
        args[0] = buffer.handle
        args[1] = PRIME_FLAGS
        var result = Int(external_call["ioctl", Int32](Int32(self.fd), IOCTL_PRIME_HANDLE_TO_FD, args.unsafe_ptr()))
        if result != 0:
            _ = self.fail("dmabuf export failed")
            return -1
        return Int(args[2])

    def pad_commands(self, commands: GpuBuffer, dwords: Int) -> Int:
        """Fill with PM4 NOPs up to the queue's size alignment. Returns the new length."""
        var alignment = max(1, self.ib_size_alignment // 4)
        var padded = dwords
        while padded % alignment != 0:
            commands.write32(padded, PM4_NOP)
            padded += 1
        return padded

    def submit(mut self, commands: GpuBuffer, dwords: Int, buffers: List[UInt32], ring: UInt32 = 0) -> UInt64:
        """Submit an indirect buffer to the compute queue. Returns the fence, 0 on failure.

        `ring` picks the context's scheduler entity: work on one entity runs in
        order, entities run side by side (async compute). Frames use 0; GPU
        meshing uses 1 (render/gpu_mesher.mojo) so it never waits behind them.
        A fence belongs to its ring, and `wait` must be told which."""
        var entries = List[UInt32](length=len(buffers) * 2, fill=0)
        for index in range(len(buffers)):
            entries[index * 2] = buffers[index]

        var bo_list = List[UInt64](length=3, fill=0)  # drm_amdgpu_bo_list_in
        bo_list[1] = UInt64(len(buffers)) | (UInt64(8) << 32)  # count, entry size
        bo_list[2] = UInt64(Int(entries.unsafe_ptr()))

        var ib = List[UInt64](length=4, fill=0)  # drm_amdgpu_cs_chunk_ib
        ib[1] = commands.gpu_address
        ib[2] = UInt64(dwords * 4) | (UInt64(HW_IP_COMPUTE) << 32)
        ib[3] = UInt64(ring) << 32  # ip_instance 0, then the ring

        var chunks = List[UInt64](length=4, fill=0)  # two drm_amdgpu_cs_chunk
        chunks[0] = UInt64(CHUNK_BO_HANDLES) | (UInt64(6) << 32)
        chunks[1] = UInt64(Int(bo_list.unsafe_ptr()))
        chunks[2] = UInt64(CHUNK_IB) | (UInt64(8) << 32)
        chunks[3] = UInt64(Int(ib.unsafe_ptr()))

        var pointers = List[UInt64](length=2, fill=0)
        pointers[0] = UInt64(Int(chunks.unsafe_ptr()))
        pointers[1] = UInt64(Int(chunks.unsafe_ptr()) + 16)

        var cs = List[UInt64](length=3, fill=0)  # drm_amdgpu_cs_in
        cs[0] = UInt64(self.context)
        cs[1] = UInt64(2)  # num_chunks; flags stay zero
        cs[2] = UInt64(Int(pointers.unsafe_ptr()))
        var result = gpu_ioctl(self.fd, IOCTL_CS, cs)
        var error = Int(last_errno()) if result != 0 else 0
        # The kernel reads these through raw addresses, so they must outlive the call.
        _ = entries^
        _ = bo_list^
        _ = ib^
        _ = chunks^
        _ = pointers^
        if result != 0:
            if error == ECANCELED or error == ENODEV:
                # The kernel refuses work on a context a GPU reset made unusable.
                self.lost = True
                _ = self.fail("command submission refused: the GPU was reset (errno " + String(error) + ")")
            else:
                _ = self.fail("command submission rejected (errno " + String(error) + ")")
            return 0
        return cs[0]

    def wait(mut self, fence: UInt64, timeout_ns: UInt64 = WAIT_TIMEOUT_NS, ring: UInt32 = 0) -> Bool:
        """Wait until the fence signals or `timeout_ns` passes. A timeout sets
        `timed_out`: the GPU may be hung, and the caller decides what to do."""
        var args = List[UInt64](length=4, fill=0)
        args[0] = fence
        args[1] = UInt64(monotonic()) + timeout_ns  # the kernel expects a deadline
        args[2] = UInt64(HW_IP_COMPUTE)
        args[3] = (UInt64(self.context) << 32) | UInt64(ring)
        if gpu_ioctl(self.fd, IOCTL_WAIT_CS, args) != 0:
            var error = Int(last_errno())
            if error == ECANCELED or error == ENODEV:
                self.lost = True
            return self.fail("fence wait failed (errno " + String(error) + ")")
        if args[0] != 0:
            self.timed_out = True
            return self.fail("fence wait timed out after " + String(Int(timeout_ns // 1000000)) + " ms")
        return True


def drm_fdinfo_bytes(fd: Int, key: String) -> Int:
    """`drm-<key>:\t12 KiB` from /proc/self/fdinfo/<fd>, in bytes; -1 if missing."""
    var data = read_stream("/proc/self/fdinfo/" + String(fd), 16384)
    if len(data) == 0:
        return -1
    var text = String(unsafe_from_utf8=data)
    var wanted = "drm-" + key + ":"
    for line in text.split("\n"):
        var piece = String(line)
        if not piece.startswith(wanted):
            continue
        var words = List[String]()
        for word in String(piece[byte=wanted.byte_length():]).replace("\t", " ").split(" "):
            if String(word).byte_length() > 0:
                words.append(String(word))
        if len(words) == 0:
            return -1
        var value: Int
        try:
            value = Int(words[0])
        except:
            return -1
        if len(words) > 1:
            if words[1] == "KiB":
                value *= 1024
            elif words[1] == "MiB":
                value *= 1048576
            elif words[1] == "GiB":
                value *= 1073741824
        return value
    return -1
