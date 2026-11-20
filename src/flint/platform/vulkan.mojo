"""Vulkan through the system loader, opened with dlopen at run time.

No SDK and no headers: entry points come from vkGetInstanceProcAddr and structs
are laid out by hand from the registry's definitions, the same way amdgpu.mojo
treats the kernel. Struct memory comes from a bump arena that lives as long as
this object, because Mojo frees a List after its last named use, and a pointer
inside a struct does not count as one.

The backend is compute only. Host-visible storage buffers play the part of
amdgpu's mapped buffers, so the CPU fills them the same way, and a timeline
semaphore plays the part of the kernel's fence: a submission returns a number
to wait for, and 0 means none.

On x86-64 Linux every Vulkan argument travels as a 64-bit register or stack
slot, so one family of call helpers taking Int fits all of them.
"""
from std.collections import Dict
from std.ffi import external_call
from flint.platform.dynlib import dl_open, dl_sym, dl_close
from flint.platform.amdgpu import GpuBuffer
from flint.platform.os import env_flag, env_value, read_bytes, write_bytes

comptime Call1 = def(Int) thin abi("C") -> Int32
comptime Call2 = def(Int, Int) thin abi("C") -> Int32
comptime Call3 = def(Int, Int, Int) thin abi("C") -> Int32
comptime Call4 = def(Int, Int, Int, Int) thin abi("C") -> Int32
comptime Call5 = def(Int, Int, Int, Int, Int) thin abi("C") -> Int32
comptime Call6 = def(Int, Int, Int, Int, Int, Int) thin abi("C") -> Int32
comptime Call7 = def(Int, Int, Int, Int, Int, Int, Int) thin abi("C") -> Int32
comptime Call8 = def(Int, Int, Int, Int, Int, Int, Int, Int) thin abi("C") -> Int32
comptime Call9 = def(Int, Int, Int, Int, Int, Int, Int, Int, Int) thin abi("C") -> Int32
comptime Call10 = def(Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) thin abi("C") -> Int32
comptime Lookup = def(Int, Int) thin abi("C") -> Int
comptime DebugCallback = def(UInt32, UInt32, Int, Int) thin abi("C") -> UInt32

comptime ARENA_BYTES: Int = 512 * 1024
comptime API_VERSION_1_2: UInt32 = (1 << 22) | (2 << 12)
comptime API_VERSION_1_3: UInt32 = (1 << 22) | (3 << 12)
comptime SUCCESS: Int32 = 0
comptime TIMEOUT: Int32 = 2
comptime WAIT_TIMEOUT_NS: Int = 2000000000
comptime WHOLE_SIZE: Int = -1
comptime VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation"

comptime TYPE_APPLICATION_INFO: UInt32 = 0
comptime TYPE_INSTANCE_CREATE_INFO: UInt32 = 1
comptime TYPE_DEVICE_QUEUE_CREATE_INFO: UInt32 = 2
comptime TYPE_DEVICE_CREATE_INFO: UInt32 = 3
comptime TYPE_SUBMIT_INFO: UInt32 = 4
comptime TYPE_MEMORY_ALLOCATE_INFO: UInt32 = 5
comptime TYPE_SEMAPHORE_CREATE_INFO: UInt32 = 9
comptime TYPE_QUERY_POOL_CREATE_INFO: UInt32 = 11
comptime TYPE_BUFFER_CREATE_INFO: UInt32 = 12
comptime TYPE_SHADER_MODULE_CREATE_INFO: UInt32 = 16
comptime TYPE_PIPELINE_CACHE_CREATE_INFO: UInt32 = 17
comptime TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: UInt32 = 18
comptime TYPE_COMPUTE_PIPELINE_CREATE_INFO: UInt32 = 29
comptime TYPE_PIPELINE_LAYOUT_CREATE_INFO: UInt32 = 30
comptime TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: UInt32 = 32
comptime TYPE_DESCRIPTOR_POOL_CREATE_INFO: UInt32 = 33
comptime TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: UInt32 = 34
comptime TYPE_WRITE_DESCRIPTOR_SET: UInt32 = 35
comptime TYPE_COMMAND_POOL_CREATE_INFO: UInt32 = 39
comptime TYPE_COMMAND_BUFFER_ALLOCATE_INFO: UInt32 = 40
comptime TYPE_COMMAND_BUFFER_BEGIN_INFO: UInt32 = 42
comptime TYPE_MEMORY_BARRIER: UInt32 = 46
comptime TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO: UInt32 = 1000128004
comptime TYPE_TIMELINE_SEMAPHORE_FEATURES: UInt32 = 1000207000
comptime TYPE_SEMAPHORE_TYPE_CREATE_INFO: UInt32 = 1000207002
comptime TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO: UInt32 = 1000207003
comptime TYPE_SEMAPHORE_WAIT_INFO: UInt32 = 1000207004
comptime TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO: UInt32 = 1000072000
comptime TYPE_EXPORT_MEMORY_ALLOCATE_INFO: UInt32 = 1000072002
comptime TYPE_MEMORY_GET_FD_INFO: UInt32 = 1000074002
comptime TYPE_CALIBRATED_TIMESTAMP_INFO: UInt32 = 1000184000

comptime DEVICE_TYPE_INTEGRATED: UInt32 = 1
comptime DEVICE_TYPE_DISCRETE: UInt32 = 2
comptime DEVICE_TYPE_VIRTUAL: UInt32 = 3
comptime QUEUE_GRAPHICS: UInt32 = 1
comptime QUEUE_COMPUTE: UInt32 = 2
comptime MEMORY_DEVICE_LOCAL: UInt32 = 1
comptime MEMORY_HOST_VISIBLE: UInt32 = 2
comptime MEMORY_HOST_COHERENT: UInt32 = 4
comptime MEMORY_HOST_CACHED: UInt32 = 8
comptime BUFFER_USAGE: UInt32 = 0x123  # storage, transfer source and destination, indirect arguments (2.7)
comptime DESCRIPTOR_STORAGE_BUFFER: UInt32 = 7
comptime STAGE_COMPUTE_SHADER: UInt32 = 0x20
comptime BIND_POINT_COMPUTE: Int = 1
comptime PIPELINE_TOP: Int = 0x1
comptime PIPELINE_COMPUTE: Int = 0x800
comptime PIPELINE_TRANSFER: Int = 0x1000
comptime PIPELINE_BOTTOM: Int = 0x2000
comptime PIPELINE_HOST: Int = 0x4000
comptime PIPELINE_DRAW_INDIRECT: Int = 0x2
comptime ACCESS_SHADER_READ: UInt32 = 0x20
comptime ACCESS_SHADER_WRITE: UInt32 = 0x40
comptime ACCESS_TRANSFER_READ: UInt32 = 0x800
comptime ACCESS_HOST_READ: UInt32 = 0x2000
comptime ACCESS_HOST_WRITE: UInt32 = 0x4000
comptime ACCESS_INDIRECT_READ: UInt32 = 0x1
comptime COMMAND_POOL_RESET_BUFFER: UInt32 = 2
comptime COMMAND_ONE_TIME: UInt32 = 1
comptime QUERY_TIMESTAMP: UInt32 = 2
comptime QUERY_RESULT_64: Int = 1
comptime TIME_DOMAIN_DEVICE: UInt32 = 0
comptime SEMAPHORE_TIMELINE: UInt32 = 1
comptime HANDLE_DMA_BUF: UInt32 = 0x200
comptime SEVERITY_WARNING: UInt32 = 0x100
comptime SEVERITY_ERROR: UInt32 = 0x1000
comptime MESSAGE_ALL: UInt32 = 0x7

comptime LIMITS_OFFSET: Int = 296
comptime LIMIT_MAX_STORAGE_RANGE: Int = LIMITS_OFFSET + 28
comptime LIMIT_MAX_PUSH: Int = LIMITS_OFFSET + 32
comptime LIMIT_TIMESTAMP_COMPUTE: Int = LIMITS_OFFSET + 420
comptime LIMIT_TIMESTAMP_PERIOD: Int = LIMITS_OFFSET + 424
comptime DEVICE_PROPERTIES_BYTES: Int = 824
comptime MAX_REPORTED_MESSAGES: Int = 12

comptime TYPE_XCB_SURFACE_CREATE_INFO: UInt32 = 1000005000
comptime TYPE_WAYLAND_SURFACE_CREATE_INFO: UInt32 = 1000006000
comptime TYPE_DISPLAY_SURFACE_CREATE_INFO: UInt32 = 1000002001
comptime DISPLAY_ALPHA_OPAQUE: UInt32 = 1
comptime DISPLAY_ALPHA_GLOBAL: UInt32 = 2
comptime SURFACE_TRANSFORM_IDENTITY: UInt32 = 1
comptime DISPLAY_MODE_PROPERTIES_BYTES: Int = 24  # VkDisplayModePropertiesKHR
comptime DISPLAY_PLANE_PROPERTIES_BYTES: Int = 16  # VkDisplayPlanePropertiesKHR
comptime TYPE_SWAPCHAIN_CREATE_INFO: UInt32 = 1000001000
comptime TYPE_PRESENT_INFO: UInt32 = 1000001001
comptime TYPE_IMAGE_MEMORY_BARRIER: UInt32 = 45
comptime PRESENT_IMMEDIATE: UInt32 = 0
comptime PRESENT_MAILBOX: UInt32 = 1
comptime PRESENT_FIFO: UInt32 = 2
comptime PRESENT_FIFO_RELAXED: UInt32 = 3
comptime FORMAT_R8_UNORM: UInt32 = 9
comptime FORMAT_B8G8R8A8_UNORM: UInt32 = 44
comptime FORMAT_B8G8R8A8_SRGB: UInt32 = 50
comptime FORMAT_A2B10G10R10_UNORM_PACK32: UInt32 = 64
comptime COLOR_SPACE_SRGB_NONLINEAR: UInt32 = 0
comptime COLOR_SPACE_HDR10_ST2084: UInt32 = 1000104008
comptime COLOR_SPACE_EXTENDED_SRGB_LINEAR: UInt32 = 1000104002
comptime IMAGE_USAGE_TRANSFER_TARGET: UInt32 = 2
comptime COMPOSITE_OPAQUE: UInt32 = 1
comptime LAYOUT_UNDEFINED: UInt32 = 0
comptime LAYOUT_TRANSFER_TARGET: UInt32 = 7
comptime LAYOUT_PRESENT: UInt32 = 1000001002
comptime ASPECT_COLOR: UInt32 = 1
comptime ACCESS_TRANSFER_WRITE: UInt32 = 0x1000
comptime ERROR_OUT_OF_DATE: Int32 = -1000001004
comptime SUBOPTIMAL: Int32 = 1000001003
comptime NOT_READY: Int32 = 1
comptime NO_IMAGE: Int = -1
# Long enough to pace to a 30 Hz display, short enough that a hidden window runs on.
comptime ACQUIRE_TIMEOUT_NS: Int = 34000000

comptime GLOBAL_LOAD_LAZY: Int32 = 1
comptime GLOBAL_LOAD_LOCAL: Int32 = 0

# ============================================================================
# Vulkan 1.3 AAA Graphics, Bindless BC7/BC5/BC6H, Sync2 & Ray-Query Constants
# ============================================================================
comptime TYPE_IMAGE_CREATE_INFO: UInt32 = 14
comptime TYPE_IMAGE_VIEW_CREATE_INFO: UInt32 = 15
comptime TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO: UInt32 = 19
comptime TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO: UInt32 = 20
comptime TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO: UInt32 = 22
comptime TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO: UInt32 = 23
comptime TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO: UInt32 = 24
comptime TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO: UInt32 = 25
comptime TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO: UInt32 = 26
comptime TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO: UInt32 = 27
comptime TYPE_GRAPHICS_PIPELINE_CREATE_INFO: UInt32 = 28
comptime TYPE_SAMPLER_CREATE_INFO: UInt32 = 31
comptime TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: UInt32 = 51
comptime TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: UInt32 = 53
comptime TYPE_PHYSICAL_DEVICE_FEATURES_2: UInt32 = 1000059000
comptime TYPE_MEMORY_ALLOCATE_FLAGS_INFO: UInt32 = 1000060000
comptime TYPE_RENDERING_INFO: UInt32 = 1000044000
comptime TYPE_RENDERING_ATTACHMENT_INFO: UInt32 = 1000044001
comptime TYPE_PIPELINE_RENDERING_CREATE_INFO: UInt32 = 1000044002
comptime TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO: UInt32 = 1000161000
comptime TYPE_BUFFER_DEVICE_ADDRESS_INFO: UInt32 = 1000244001
comptime TYPE_MEMORY_BARRIER_2: UInt32 = 1000314000
comptime TYPE_IMAGE_MEMORY_BARRIER_2: UInt32 = 1000314002
comptime TYPE_DEPENDENCY_INFO: UInt32 = 1000314003

comptime TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR: UInt32 = 1000150000
comptime TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR: UInt32 = 1000150002
comptime TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR: UInt32 = 1000150004
comptime TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR: UInt32 = 1000150005
comptime TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR: UInt32 = 1000150006
comptime TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR: UInt32 = 1000150007
comptime TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR: UInt32 = 1000150013
comptime TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR: UInt32 = 1000150017
comptime TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR: UInt32 = 1000150020
comptime TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR: UInt32 = 1000348013

comptime FORMAT_R8G8B8A8_UNORM: UInt32 = 37
comptime FORMAT_R16G16_SFLOAT: UInt32 = 83
comptime FORMAT_R16G16B16A16_SFLOAT: UInt32 = 97
comptime FORMAT_R32_UINT: UInt32 = 98


@always_inline
def swapchain_format_bytes_per_pixel(format: UInt32) -> Int:
    """Returns exact texel byte stride for Vulkan swapchain formats (`8` for `R16G16B16A16_SFLOAT`, `4` for 32-bit SDR/HDR10)."""
    if format == FORMAT_R16G16B16A16_SFLOAT:
        return 8
    return 4
comptime FORMAT_R32_SFLOAT: UInt32 = 100
comptime FORMAT_R32G32_UINT: UInt32 = 101
comptime FORMAT_R32G32B32_SFLOAT: UInt32 = 106
comptime FORMAT_R32G32B32A32_UINT: UInt32 = 107
comptime FORMAT_D32_SFLOAT: UInt32 = 126
comptime FORMAT_BC5_UNORM: UInt32 = 141
comptime FORMAT_BC6H_UFLOAT: UInt32 = 143
comptime FORMAT_BC7_UNORM: UInt32 = 145
comptime FORMAT_BC7_SRGB: UInt32 = 146

comptime IMAGE_USAGE_TRANSFER_SRC: UInt32 = 1
comptime IMAGE_USAGE_SAMPLED: UInt32 = 4
comptime IMAGE_USAGE_STORAGE: UInt32 = 8
comptime IMAGE_USAGE_COLOR_ATTACHMENT: UInt32 = 16
comptime IMAGE_USAGE_DEPTH_ATTACHMENT: UInt32 = 32
comptime IMAGE_CREATE_CUBE_COMPATIBLE: UInt32 = 16
comptime IMAGE_VIEW_TYPE_2D: UInt32 = 1
comptime IMAGE_VIEW_TYPE_CUBE: UInt32 = 3
comptime ASPECT_DEPTH: UInt32 = 2

comptime LAYOUT_GENERAL: UInt32 = 1
comptime LAYOUT_COLOR_ATTACHMENT: UInt32 = 2
comptime LAYOUT_DEPTH_ATTACHMENT: UInt32 = 3
comptime LAYOUT_SHADER_READ_ONLY: UInt32 = 5
comptime LAYOUT_TRANSFER_SRC: UInt32 = 6

comptime STAGE_VERTEX_SHADER: UInt32 = 0x1
comptime STAGE_FRAGMENT_SHADER: UInt32 = 0x10
comptime STAGE_ALL_GRAPHICS: UInt32 = 0x1F
comptime STAGE_ALL_SHADERS: UInt32 = 0x31
comptime BIND_POINT_GRAPHICS: Int = 0

comptime PIPELINE_COLOR_ATTACHMENT_OUTPUT: Int = 0x400
comptime PIPELINE_EARLY_FRAGMENT_TESTS: Int = 0x100
comptime PIPELINE_LATE_FRAGMENT_TESTS: Int = 0x200
comptime PIPELINE_FRAGMENT_SHADER: Int = 0x80
comptime PIPELINE_VERTEX_SHADER: Int = 0x8
comptime PIPELINE_AS_BUILD_KHR: Int = 0x2000000

comptime ACCESS_COLOR_ATTACHMENT_WRITE: UInt32 = 0x100
comptime ACCESS_DEPTH_ATTACHMENT_READ: UInt32 = 0x200
comptime ACCESS_DEPTH_ATTACHMENT_WRITE: UInt32 = 0x400
comptime ACCESS_AS_READ_KHR: UInt32 = 0x200000
comptime ACCESS_AS_WRITE_KHR: UInt32 = 0x400000

comptime DESCRIPTOR_COMBINED_IMAGE_SAMPLER: UInt32 = 1
comptime DESCRIPTOR_STORAGE_IMAGE: UInt32 = 3
comptime DESCRIPTOR_ACCELERATION_STRUCTURE_KHR: UInt32 = 1000150000
comptime DESCRIPTOR_BINDING_UPDATE_AFTER_BIND: UInt32 = 1
comptime DESCRIPTOR_BINDING_PARTIALLY_BOUND: UInt32 = 4
comptime DESCRIPTOR_LAYOUT_UPDATE_AFTER_BIND_POOL: UInt32 = 2
comptime DESCRIPTOR_POOL_UPDATE_AFTER_BIND: UInt32 = 2

comptime COMPARE_LESS_OR_EQUAL: UInt32 = 3
comptime COMPARE_GREATER_OR_EQUAL: UInt32 = 6
comptime COMPARE_ALWAYS: UInt32 = 7
comptime LOAD_OP_LOAD: UInt32 = 0
comptime LOAD_OP_CLEAR: UInt32 = 1
comptime STORE_OP_STORE: UInt32 = 0
comptime TOPOLOGY_TRIANGLE_LIST: UInt32 = 3
comptime CULL_MODE_NONE: UInt32 = 0
comptime CULL_MODE_FRONT: UInt32 = 1
comptime CULL_MODE_BACK: UInt32 = 2
comptime FRONT_FACE_CCW: UInt32 = 0
comptime FRONT_FACE_CW: UInt32 = 1
comptime SAMPLE_COUNT_1: UInt32 = 1
comptime FILTER_NEAREST: UInt32 = 0
comptime FILTER_LINEAR: UInt32 = 1
comptime SAMPLER_MIPMAP_LINEAR: UInt32 = 1
comptime ADDRESS_MODE_REPEAT: UInt32 = 0
comptime ADDRESS_MODE_CLAMP_TO_EDGE: UInt32 = 2
comptime DYNAMIC_STATE_VIEWPORT: UInt32 = 0
comptime DYNAMIC_STATE_SCISSOR: UInt32 = 1

comptime MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT: UInt32 = 2
comptime BUFFER_USAGE_INDEX: UInt32 = 64
comptime BUFFER_USAGE_VERTEX: UInt32 = 128
comptime BUFFER_USAGE_DEVICE_ADDRESS: UInt32 = 131072
comptime BUFFER_USAGE_AS_BUILD_INPUT_KHR: UInt32 = 524288
comptime BUFFER_USAGE_AS_STORAGE_KHR: UInt32 = 1048576

comptime AS_TYPE_TOP_LEVEL_KHR: UInt32 = 0
comptime AS_TYPE_BOTTOM_LEVEL_KHR: UInt32 = 1
comptime GEOMETRY_TYPE_TRIANGLES_KHR: UInt32 = 0
comptime GEOMETRY_TYPE_INSTANCES_KHR: UInt32 = 2
comptime GEOMETRY_OPAQUE_BIT_KHR: UInt32 = 1
comptime BUILD_AS_ALLOW_UPDATE_KHR: UInt32 = 1
comptime BUILD_AS_PREFER_FAST_TRACE_KHR: UInt32 = 4
comptime BUILD_AS_MODE_BUILD_KHR: UInt32 = 0
comptime BUILD_AS_MODE_UPDATE_KHR: UInt32 = 1
comptime BUILD_AS_TYPE_DEVICE_KHR: Int = 1
comptime INDEX_TYPE_UINT32: UInt32 = 1


def put_float(address: Int, value: Float32):
    Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=address)[unsafe_offset=0] = value


struct VulkanImage(Copyable, Movable):
    var image: Int
    var view: Int
    var memory: Int
    var format: UInt32
    var width: Int
    var height: Int
    var mip_levels: Int
    var layers: Int
    var layout: UInt32

    def __init__(out self):
        self.image = 0
        self.view = 0
        self.memory = 0
        self.format = 0
        self.width = 0
        self.height = 0
        self.mip_levels = 1
        self.layers = 1
        self.layout = LAYOUT_UNDEFINED


struct VulkanAccelStruct(Copyable, Movable):
    var handle: Int
    var buffer: GpuBuffer
    var device_address: UInt64

    def __init__(out self):
        self.handle = 0
        self.buffer = GpuBuffer()
        self.device_address = 0


struct SwapchainHandle(Copyable, Movable):
    var surface: Int
    var swapchain: Int
    var vk_swapchain: Int
    var width: Int
    var height: Int
    var format: UInt32
    var color_space: UInt32
    var present_mode: UInt32
    var image_count: Int
    var is_hdr: Bool

    def __init__(out self):
        self.surface = 0
        self.swapchain = 0
        self.vk_swapchain = 0
        self.width = 0
        self.height = 0
        self.format = FORMAT_B8G8R8A8_UNORM
        self.color_space = COLOR_SPACE_SRGB_NONLINEAR
        self.present_mode = PRESENT_FIFO
        self.image_count = 0
        self.is_hdr = False

    def __init__(
        out self,
        surface: Int,
        swapchain: Int,
        width: Int,
        height: Int,
        format: UInt32,
        color_space: UInt32,
        present_mode: UInt32,
        image_count: Int,
        is_hdr: Bool,
    ):
        self.surface = surface
        self.swapchain = swapchain
        self.vk_swapchain = swapchain
        self.width = width
        self.height = height
        self.format = format
        self.color_space = color_space
        self.present_mode = present_mode
        self.image_count = image_count
        self.is_hdr = is_hdr

    def active(self) -> Bool:
        return self.swapchain != 0


def put32(address: Int, value: UInt32):
    Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=address)[unsafe_offset=0] = value


def put64(address: Int, value: Int):
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=address)[unsafe_offset=0] = value


def get32(address: Int) -> UInt32:
    return Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=address)[unsafe_offset=0]


def get64(address: Int) -> Int:
    return Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=address)[unsafe_offset=0]


def get_float(address: Int) -> Float32:
    return Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=address)[unsafe_offset=0]


def call1(f: Int, a: Int) -> Int32:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Call1]()[unsafe_offset=0](a)


def call2(f: Int, a: Int, b: Int) -> Int32:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Call2]()[unsafe_offset=0](a, b)


def call3(f: Int, a: Int, b: Int, c: Int) -> Int32:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Call3]()[unsafe_offset=0](a, b, c)


def call4(f: Int, a: Int, b: Int, c: Int, d: Int) -> Int32:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Call4]()[unsafe_offset=0](a, b, c, d)


def call5(f: Int, a: Int, b: Int, c: Int, d: Int, e: Int) -> Int32:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Call5]()[unsafe_offset=0](a, b, c, d, e)


def call6(f: Int, a: Int, b: Int, c: Int, d: Int, e: Int, g: Int) -> Int32:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Call6]()[unsafe_offset=0](a, b, c, d, e, g)


def call7(f: Int, a: Int, b: Int, c: Int, d: Int, e: Int, g: Int, h: Int) -> Int32:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Call7]()[unsafe_offset=0](a, b, c, d, e, g, h)


def call8(f: Int, a: Int, b: Int, c: Int, d: Int, e: Int, g: Int, h: Int, i: Int) -> Int32:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Call8]()[unsafe_offset=0](a, b, c, d, e, g, h, i)


def call9(f: Int, a: Int, b: Int, c: Int, d: Int, e: Int, g: Int, h: Int, i: Int, j: Int) -> Int32:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Call9]()[unsafe_offset=0](a, b, c, d, e, g, h, i, j)


def call10(f: Int, a: Int, b: Int, c: Int, d: Int, e: Int, g: Int, h: Int, i: Int, j: Int, k: Int) -> Int32:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Call10]()[unsafe_offset=0](a, b, c, d, e, g, h, i, j, k)


def lookup(f: Int, a: Int, b: Int) -> Int:
    var slot = f
    return Pointer(to=slot).unsafe_bitcast[Lookup]()[unsafe_offset=0](a, b)


def choose_display_mode(widths: List[Int], heights: List[Int], refresh_mhz: List[Int], want_width: Int, want_height: Int, want_mhz: Int) -> Int:
    """The Vulkan display mode for a KMS mode: same size, closest refresh. Without
    a size match, the largest. -1 when there are none."""
    var best = -1
    for index in range(len(widths)):
        if widths[index] != want_width or heights[index] != want_height:
            continue
        if best < 0 or abs(refresh_mhz[index] - want_mhz) < abs(refresh_mhz[best] - want_mhz):
            best = index
    if best >= 0:
        return best
    for index in range(len(widths)):
        if best < 0 or widths[index] * heights[index] > widths[best] * heights[best]:
            best = index
    return best


def c_string(address: Int) -> String:
    if address == 0:
        return ""
    var bytes = List[UInt8]()
    var pointer = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=address)
    var index = 0
    while pointer[unsafe_offset=index] != 0:
        bytes.append(pointer[unsafe_offset=index])
        index += 1
    return String(unsafe_from_utf8=bytes)


def on_debug_message(severity: UInt32, types: UInt32, data: Int, user: Int) abi("C") -> UInt32:
    """The validation layers report here. Word 0 of `user` counts the errors, word 1
    the warnings, and tests fail on either."""
    var counter = user + (0 if (severity & SEVERITY_ERROR) != 0 else 8)
    put64(counter, get64(counter) + 1)
    if get64(user) + get64(user + 8) <= MAX_REPORTED_MESSAGES:
        print("VULKAN_VALIDATION " + c_string(get64(data + 40)))
    return 0  # never abort the call that was being checked


struct VulkanGpu(Copyable, Movable):
    var library: Int
    var instance: Int
    var physical: Int
    var device: Int
    var queue: Int
    var family: UInt32
    var messenger: Int
    var messages: Int  # two counters shared with the debug callback
    var arena: Int
    var arena_used: Int
    var device_name: String
    var device_type: UInt32
    var memory_properties: Int
    var timestamp_period: Float64
    var timestamps: Bool
    var max_buffer: Int
    var has_dma_buf: Bool
    var has_calibration: Bool
    var has_swapchain: Bool
    var has_wayland_surface: Bool
    var has_xcb_surface: Bool
    var has_xlib_surface: Bool
    var has_swapchain_colorspace: Bool
    var has_display_surface: Bool  # VK_KHR_display + VK_EXT_acquire_drm_display: present with no compositor
    var display: Int  # the VkDisplayKHR taken from a KMS connector, released with the swapchain
    var api_version: UInt32
    var has_vulkan_13: Bool
    var has_dynamic_rendering: Bool
    var has_descriptor_indexing: Bool
    var has_sync2: Bool
    var has_bda: Bool
    var has_draw_indirect_count: Bool
    var has_ray_query: Bool
    var timeline: Int
    # The side queue, for work beside the frames (GPU meshing): a compute-only
    # family if the driver has one, else a second queue of the main family, else
    # the main queue itself. Its own timeline counts its own fences.
    var side_family: UInt32
    var side_index: Int  # queue index in side_family
    var side_queue: Int
    var side_timeline: Int
    var side_next: UInt64
    var side_pool: Int
    var next_signal: UInt64
    var command_pool: Int
    var command_buffers: List[Int]
    var query_pool: Int
    var descriptor_layout: Int
    var pipeline_layout: Int
    var pipeline_cache: Int
    var descriptor_pool: Int
    var side_pools: List[Int]  # descriptor pools of passes outside the frame (GPU meshing)
    var layout_bindings: Int
    var pipelines: List[Int]
    var surface: Int
    var swapchain: Int
    var swapchain_width: Int
    var swapchain_height: Int
    var swapchain_format: UInt32
    var swapchain_color_space: UInt32
    var swapchain_present_mode: UInt32
    var is_hdr: Bool
    var images: List[Int]
    var image_views: List[Int]
    var acquire_semaphores: List[Int]
    var finished_semaphores: List[Int]
    var presented: Int
    var skipped: Int
    var error: String
    # Instance and device entry points.
    var proc_instance: Int
    var proc_device: Int
    var fn_create_buffer: Int
    var fn_destroy_buffer: Int
    var fn_get_buffer_requirements: Int
    var fn_allocate_memory: Int
    var fn_free_memory: Int
    var fn_bind_buffer_memory: Int
    var fn_map_memory: Int
    var fn_get_memory_fd: Int
    var fn_begin_commands: Int
    var fn_end_commands: Int
    var fn_reset_commands: Int
    var fn_bind_pipeline: Int
    var fn_bind_descriptor_sets: Int
    var fn_push_constants: Int
    var fn_dispatch: Int
    var fn_dispatch_indirect: Int
    var fn_pipeline_barrier: Int
    var fn_write_timestamp: Int
    var fn_reset_query_pool: Int
    var fn_get_query_results: Int
    var fn_queue_submit: Int
    var fn_wait_semaphores: Int
    var fn_update_descriptors: Int
    var fn_calibrate: Int
    # Per-device cache of `vkGetDeviceProcAddr` results (filled lazily by `device_call`; reset when the device closes).
    var device_call_cache: Dict[String, Int]
    var device_call_cache_device: Int
    var overflow_slabs: List[Int]

    def __init__(out self):
        self.device_call_cache = Dict[String, Int]()
        self.device_call_cache_device = 0
        self.overflow_slabs = List[Int]()
        self.library = 0
        self.instance = 0
        self.physical = 0
        self.device = 0
        self.queue = 0
        self.family = 0
        self.messenger = 0
        self.messages = 0
        self.arena = 0
        self.arena_used = 0
        self.device_name = ""
        self.device_type = 0
        self.memory_properties = 0
        self.timestamp_period = 0
        self.timestamps = False
        self.max_buffer = 0
        self.has_dma_buf = False
        self.has_calibration = False
        self.has_swapchain = False
        self.has_display_surface = False
        self.display = 0
        self.has_wayland_surface = False
        self.has_xcb_surface = False
        self.has_xlib_surface = False
        self.has_swapchain_colorspace = False
        self.api_version = API_VERSION_1_2
        self.has_vulkan_13 = False
        self.has_dynamic_rendering = False
        self.has_descriptor_indexing = False
        self.has_sync2 = False
        self.has_bda = False
        self.has_draw_indirect_count = False
        self.has_ray_query = False
        self.timeline = 0
        self.side_family = 0
        self.side_index = -1
        self.side_queue = 0
        self.side_timeline = 0
        self.side_next = 1
        self.side_pool = 0
        self.next_signal = 1
        self.command_pool = 0
        self.command_buffers = List[Int]()
        self.query_pool = 0
        self.descriptor_layout = 0
        self.pipeline_layout = 0
        self.pipeline_cache = 0
        self.descriptor_pool = 0
        self.side_pools = List[Int]()
        self.layout_bindings = 0
        self.pipelines = List[Int]()
        self.surface = 0
        self.swapchain = 0
        self.swapchain_width = 0
        self.swapchain_height = 0
        self.swapchain_format = FORMAT_B8G8R8A8_UNORM
        self.swapchain_color_space = COLOR_SPACE_SRGB_NONLINEAR
        self.swapchain_present_mode = PRESENT_FIFO
        self.is_hdr = False
        self.images = List[Int]()
        self.image_views = List[Int]()
        self.acquire_semaphores = List[Int]()
        self.finished_semaphores = List[Int]()
        self.presented = 0
        self.skipped = 0
        self.error = ""
        self.proc_instance = 0
        self.proc_device = 0
        self.fn_create_buffer = 0
        self.fn_destroy_buffer = 0
        self.fn_get_buffer_requirements = 0
        self.fn_allocate_memory = 0
        self.fn_free_memory = 0
        self.fn_bind_buffer_memory = 0
        self.fn_map_memory = 0
        self.fn_get_memory_fd = 0
        self.fn_begin_commands = 0
        self.fn_end_commands = 0
        self.fn_reset_commands = 0
        self.fn_bind_pipeline = 0
        self.fn_bind_descriptor_sets = 0
        self.fn_push_constants = 0
        self.fn_dispatch = 0
        self.fn_dispatch_indirect = 0
        self.fn_pipeline_barrier = 0
        self.fn_write_timestamp = 0
        self.fn_reset_query_pool = 0
        self.fn_get_query_results = 0
        self.fn_queue_submit = 0
        self.fn_wait_semaphores = 0
        self.fn_update_descriptors = 0
        self.fn_calibrate = 0

    def fail(mut self, message: String) -> Bool:
        self.error = message
        return False

    def check(mut self, result: Int32, what: String) -> Bool:
        if result != SUCCESS:
            return self.fail(what + " returned " + String(result))
        return True

    # Struct memory.

    def scratch(mut self, size: Int) -> Int:
        """Zeroed bytes, good until the next `rewind`."""
        var rounded = (max(1, size) + 15) & ~15
        if self.arena == 0 or self.arena_used + rounded > ARENA_BYTES:
            var slab = external_call["calloc", Int](Int(1), rounded)
            self.overflow_slabs.append(slab)
            return slab
        var address = self.arena + self.arena_used
        self.arena_used += rounded
        _ = external_call["memset", Int](address, Int32(0), rounded)
        return address

    def rewind(mut self):
        self.arena_used = 0
        if len(self.overflow_slabs) > 0:
            for slab in self.overflow_slabs:
                if slab != 0:
                    external_call["free", NoneType](slab)
            self.overflow_slabs.clear()

    def text(mut self, value: String) -> Int:
        var address = self.scratch(value.byte_length() + 1)
        _ = external_call["memcpy", Int](address, Int(value.unsafe_ptr()), value.byte_length())
        return address

    def name_list(mut self, names: List[String]) -> Int:
        var array = self.scratch(max(len(names), 1) * 8)
        for index in range(len(names)):
            put64(array + index * 8, self.text(names[index]))
        return array

    def instance_call(mut self, name: String) -> Int:
        var address = lookup(self.proc_instance, self.instance, self.text(name))
        return address

    def device_call(mut self, name: String) -> Int:
        """`vkGetDeviceProcAddr`, memoised per device: the hot recording helpers (barriers, rendering, draws,
        copies, AS builds) call this for every command, and a loader lookup costs a scratch memset + string copy."""
        if self.device_call_cache_device != self.device:
            self.device_call_cache.clear()
            self.device_call_cache_device = self.device
        var cached = self.device_call_cache.get(name)
        if cached:
            return cached.value()
        var address = lookup(self.proc_device, self.device, self.text(name))
        if address == 0:
            self.error = "the driver has no " + name
        elif self.device != 0:
            self.device_call_cache[name] = address
        return address

    # Bring-up.

    def _free_host_scratch(mut self):
        self.rewind()
        if self.arena != 0:
            external_call["free", NoneType](self.arena)
            self.arena = 0
            self.arena_used = 0
        if self.messages != 0:
            external_call["free", NoneType](self.messages)
            self.messages = 0
        if self.memory_properties != 0:
            external_call["free", NoneType](self.memory_properties)
            self.memory_properties = 0

    def _fail_init(mut self) -> Bool:
        var saved_err = self.error
        self.close()
        self.error = saved_err
        return False

    def initialize(mut self, validate: Bool = False, need_discrete: Bool = False) -> Bool:
        """Open the loader and the best GPU. False leaves `error` and nothing to close."""
        self._free_host_scratch()
        self.arena = external_call["malloc", Int](ARENA_BYTES)
        self.messages = external_call["calloc", Int](Int(2), Int(8))
        self.memory_properties = external_call["malloc", Int](520)
        var library_name = "libvulkan.so.1"
        self.library = dl_open(library_name, GLOBAL_LOAD_LAZY | GLOBAL_LOAD_LOCAL)
        if self.library == 0:
            _ = self.fail("no Vulkan loader on this system")
            return self._fail_init()
        var symbol = "vkGetInstanceProcAddr"
        self.proc_instance = dl_sym(self.library, symbol)
        if self.proc_instance == 0:
            _ = self.fail("the loader has no vkGetInstanceProcAddr")
            return self._fail_init()
        if not self.create_instance(validate):
            return self._fail_init()
        if not self.choose_device(need_discrete):
            return self._fail_init()
        if not self.create_device():
            return self._fail_init()
        if not (self.load_device_calls() and self.create_queue_objects()):
            return self._fail_init()
        return True

    def layer_available(mut self, wanted: String) -> Bool:
        self.rewind()
        var enumerate_layers = self.instance_call("vkEnumerateInstanceLayerProperties")
        var count_slot = self.scratch(8)
        if enumerate_layers == 0 or call2(enumerate_layers, count_slot, 0) != SUCCESS:
            return False
        var count = Int(get32(count_slot))
        var layers = external_call["malloc", Int](count * 520 + 8)
        put32(count_slot, UInt32(count))
        var found = False
        if call2(enumerate_layers, count_slot, layers) == SUCCESS:
            for index in range(count):
                if c_string(layers + index * 520) == wanted:
                    found = True
        external_call["free", NoneType](layers)
        return found

    def instance_extensions(mut self) -> List[String]:
        self.rewind()
        var enumerate_extensions = self.instance_call("vkEnumerateInstanceExtensionProperties")
        return self.read_extensions(call_instance=True, function=enumerate_extensions)

    def device_extensions(mut self) -> List[String]:
        self.rewind()
        var enumerate_extensions = self.instance_call("vkEnumerateDeviceExtensionProperties")
        return self.read_extensions(call_instance=False, function=enumerate_extensions)

    def read_extensions(mut self, call_instance: Bool, function: Int) -> List[String]:
        var names = List[String]()
        if function == 0:
            return names^
        var count_slot = self.scratch(8)
        var result: Int32
        if call_instance:
            result = call3(function, 0, count_slot, 0)
        else:
            result = call4(function, self.physical, 0, count_slot, 0)
        if result != SUCCESS:
            return names^
        var count = Int(get32(count_slot))
        var table = external_call["malloc", Int](count * 260 + 8)
        if call_instance:
            result = call3(function, 0, count_slot, table)
        else:
            result = call4(function, self.physical, 0, count_slot, table)
        if result == SUCCESS:
            for index in range(count):
                names.append(c_string(table + index * 260))
        external_call["free", NoneType](table)
        return names^

    def create_instance(mut self, validate: Bool) -> Bool:
        var offered = self.instance_extensions()
        var extensions = List[String]()
        for wanted in [
            "VK_KHR_surface",
            "VK_KHR_wayland_surface",
            "VK_KHR_xcb_surface",
            "VK_KHR_xlib_surface",
            "VK_EXT_swapchain_colorspace",
        ]:
            if String(wanted) in offered:
                extensions.append(String(wanted))
        var has_surf = False
        var has_wl = False
        var has_xcb = False
        var has_xlib = False
        var has_cs = False
        for ext in extensions:
            if ext == "VK_KHR_surface":
                has_surf = True
            elif ext == "VK_KHR_wayland_surface":
                has_wl = True
            elif ext == "VK_KHR_xcb_surface":
                has_xcb = True
            elif ext == "VK_KHR_xlib_surface":
                has_xlib = True
            elif ext == "VK_EXT_swapchain_colorspace":
                has_cs = True
        self.has_wayland_surface = has_surf and has_wl
        self.has_xcb_surface = has_surf and has_xcb
        self.has_xlib_surface = has_surf and has_xlib
        self.has_swapchain_colorspace = has_surf and has_cs
        # Presenting straight to a KMS connector (platform/kms.mojo, GPU mode): all three or none.
        if has_surf and "VK_KHR_display" in offered and "VK_EXT_direct_mode_display" in offered and "VK_EXT_acquire_drm_display" in offered:
            extensions.append("VK_KHR_display")
            extensions.append("VK_EXT_direct_mode_display")
            extensions.append("VK_EXT_acquire_drm_display")
            self.has_display_surface = True
        var layers = List[String]()
        var checked = validate or env_flag("SD_VULKAN_VALIDATION")
        if checked:
            if not self.layer_available(VALIDATION_LAYER):
                return self.fail("the validation layers are not installed: sudo apt install vulkan-validationlayers")
            layers.append(VALIDATION_LAYER)
            extensions.append("VK_EXT_debug_utils")
        self.rewind()
        var application = self.scratch(48)
        put32(application, TYPE_APPLICATION_INFO)
        put64(application + 16, self.text("FlintEngine"))
        put64(application + 32, self.text("FlintEngine"))
        put32(application + 44, API_VERSION_1_3)
        var info = self.scratch(64)
        put32(info, TYPE_INSTANCE_CREATE_INFO)
        put64(info + 24, application)
        put32(info + 32, UInt32(len(layers)))
        put64(info + 40, self.name_list(layers))
        put32(info + 48, UInt32(len(extensions)))
        put64(info + 56, self.name_list(extensions))
        var messenger_info = self.scratch(48)
        if checked:
            # Chained here too, so instance creation itself is checked.
            var callback: DebugCallback = on_debug_message
            put32(messenger_info, TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO)
            put32(messenger_info + 20, SEVERITY_WARNING | SEVERITY_ERROR)
            put32(messenger_info + 24, MESSAGE_ALL)
            put64(messenger_info + 32, Pointer(to=callback).unsafe_bitcast[Int]()[unsafe_offset=0])
            put64(messenger_info + 40, self.messages)
            put64(info + 8, messenger_info)
        var slot = self.scratch(8)
        var create = self.instance_call("vkCreateInstance")
        var result = call3(create, info, 0, slot)
        if result != SUCCESS:
            return self.fail("vkCreateInstance returned " + String(result))
        self.instance = get64(slot)
        self.proc_device = self.instance_call("vkGetDeviceProcAddr")
        if checked:
            var create_messenger = self.instance_call("vkCreateDebugUtilsMessengerEXT")
            if not self.check(call4(create_messenger, self.instance, messenger_info, 0, slot), "vkCreateDebugUtilsMessengerEXT"):
                return False
            self.messenger = get64(slot)
        return True

    def choose_device(mut self, need_discrete: Bool) -> Bool:
        """The discrete GPU if there is one: the monitor's GPU has a desktop to keep alive."""
        self.rewind()
        var enumerate_devices = self.instance_call("vkEnumeratePhysicalDevices")
        var get_properties = self.instance_call("vkGetPhysicalDeviceProperties")
        var count_slot = self.scratch(8)
        if not self.check(call3(enumerate_devices, self.instance, count_slot, 0), "vkEnumeratePhysicalDevices"):
            return False
        var count = Int(get32(count_slot))
        if count == 0:
            return self.fail("Vulkan sees no GPU")
        var devices = self.scratch(count * 8)
        if not self.check(call3(enumerate_devices, self.instance, count_slot, devices), "vkEnumeratePhysicalDevices"):
            return False
        var force_cpu = env_flag("FLINT_VULKAN_LLVMPIPE")
        var properties = self.scratch(DEVICE_PROPERTIES_BYTES)
        var best_rank = 0
        for index in range(count):
            _ = call2(get_properties, get64(devices + index * 8), properties)
            var api = get32(properties)
            var kind = get32(properties + 16)
            var rank = (
                4
                if kind == DEVICE_TYPE_DISCRETE
                else (3 if kind == DEVICE_TYPE_VIRTUAL else (2 if kind == DEVICE_TYPE_INTEGRATED else 1))
            )
            if force_cpu and kind == 4:
                rank = 10
            if api < API_VERSION_1_2:
                rank = 0
            if rank > best_rank:
                best_rank = rank
                self.physical = get64(devices + index * 8)
                self.api_version = api
                self.device_type = kind
                self.device_name = c_string(properties + 20)
                self.timestamps = get32(properties + LIMIT_TIMESTAMP_COMPUTE) != 0
                self.timestamp_period = Float64(get_float(properties + LIMIT_TIMESTAMP_PERIOD))
                self.max_buffer = Int(get32(properties + LIMIT_MAX_STORAGE_RANGE))
        if best_rank == 0:
            return self.fail("no Vulkan 1.2 GPU")
        if need_discrete and self.device_type != DEVICE_TYPE_DISCRETE:
            return self.fail("this run may only use a discrete GPU, and Vulkan found " + self.device_name)
        var get_memory = self.instance_call("vkGetPhysicalDeviceMemoryProperties")
        _ = call2(get_memory, self.physical, self.memory_properties)
        return self.choose_family()

    def choose_family(mut self) -> Bool:
        self.rewind()
        var get_families = self.instance_call("vkGetPhysicalDeviceQueueFamilyProperties")
        var count_slot = self.scratch(8)
        _ = call3(get_families, self.physical, count_slot, 0)
        var count = Int(get32(count_slot))
        var families = self.scratch(count * 24)
        _ = call3(get_families, self.physical, count_slot, families)
        var found = False
        for index in range(count):
            # The universal queue: every driver can present from it.
            var flags = get32(families + index * 24)
            if (flags & (QUEUE_GRAPHICS | QUEUE_COMPUTE)) == (QUEUE_GRAPHICS | QUEUE_COMPUTE):
                self.family = UInt32(index)
                if get32(families + index * 24 + 8) == 0:
                    self.timestamps = False
                found = True
                break
        if not found:
            return self.fail(self.device_name + " has no compute queue")
        # The side queue: async compute where the driver offers it.
        self.side_family = self.family
        self.side_index = -1
        for index in range(count):
            var flags = get32(families + index * 24)
            if (flags & QUEUE_COMPUTE) != 0 and (flags & QUEUE_GRAPHICS) == 0 and get32(families + index * 24 + 4) > 0:
                self.side_family = UInt32(index)
                self.side_index = 0
                return True
        if get32(families + Int(self.family) * 24 + 4) >= 2:
            self.side_index = 1
        return True

    def create_device(mut self) -> Bool:
        var offered = self.device_extensions()
        var extensions = List[String]()
        if "VK_KHR_external_memory_fd" in offered and "VK_EXT_external_memory_dma_buf" in offered:
            extensions.append("VK_KHR_external_memory_fd")
            extensions.append("VK_EXT_external_memory_dma_buf")
            self.has_dma_buf = True
        var calibration = "vkGetCalibratedTimestampsKHR"
        if "VK_KHR_calibrated_timestamps" in offered:
            extensions.append("VK_KHR_calibrated_timestamps")
            self.has_calibration = True
        elif "VK_EXT_calibrated_timestamps" in offered:
            extensions.append("VK_EXT_calibrated_timestamps")
            calibration = "vkGetCalibratedTimestampsEXT"
            self.has_calibration = True
        if "VK_KHR_swapchain" in offered:
            extensions.append("VK_KHR_swapchain")
            self.has_swapchain = True
        var want_rt = (
            self.device_type != 4
            and "VK_KHR_acceleration_structure" in offered
            and "VK_KHR_ray_query" in offered
            and "VK_KHR_deferred_host_operations" in offered
        )
        if want_rt:
            extensions.append("VK_KHR_deferred_host_operations")
            extensions.append("VK_KHR_acceleration_structure")
            extensions.append("VK_KHR_ray_query")
        self.rewind()
        var priority = self.scratch(8)
        put32(priority, 0x3F800000)  # 1.0
        put32(priority + 4, 0x3F800000)
        var separate_side = self.side_index >= 0 and self.side_family != self.family
        var queue_info = self.scratch(80)
        put32(queue_info, TYPE_DEVICE_QUEUE_CREATE_INFO)
        put32(queue_info + 20, self.family)
        put32(queue_info + 24, UInt32(2) if (self.side_index == 1 and not separate_side) else UInt32(1))
        put64(queue_info + 32, priority)
        if separate_side:
            put32(queue_info + 40, TYPE_DEVICE_QUEUE_CREATE_INFO)
            put32(queue_info + 60, self.side_family)
            put32(queue_info + 64, 1)
            put64(queue_info + 72, priority)

        var pnext_head: Int
        if self.api_version >= API_VERSION_1_3:
            self.has_vulkan_13 = True
            # Query exact supported Vulkan 1.0 + 1.2 + 1.3 + RayQuery features first
            var q_feat2 = self.scratch(240)
            var q_vk12 = self.scratch(208)
            var q_vk13 = self.scratch(80)
            var q_as = self.scratch(40)
            var q_rq = self.scratch(24)
            put32(q_feat2, TYPE_PHYSICAL_DEVICE_FEATURES_2)
            put64(q_feat2 + 8, q_vk12)
            put32(q_vk12, TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
            put64(q_vk12 + 8, q_vk13)
            put32(q_vk13, TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
            if want_rt:
                put64(q_vk13 + 8, q_as)
                put32(q_as, TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR)
                put64(q_as + 8, q_rq)
                put32(q_rq, TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR)
            var get_feat2 = self.instance_call("vkGetPhysicalDeviceFeatures2")
            if get_feat2 != 0:
                _ = call2(get_feat2, self.physical, q_feat2)

            # Build clean enabled-features chain containing only the exact AAA features supported
            var feat2 = self.scratch(240)
            var vk12 = self.scratch(208)
            var vk13 = self.scratch(80)
            put32(feat2, TYPE_PHYSICAL_DEVICE_FEATURES_2)
            put64(feat2 + 8, vk12)
            put32(vk12, TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
            put64(vk12 + 8, vk13)
            put32(vk13, TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)

            # Core 1.0 features: geometryShader (16+16=32), multiDrawIndirect (16+36=52), samplerAnisotropy (16+76=92), fragmentStoresAndAtomics (16+104=120)
            put32(feat2 + 16 + 16, get32(q_feat2 + 16 + 16))   # geometryShader (for gl_PrimitiveID in fragment shaders)
            put32(feat2 + 16 + 36, get32(q_feat2 + 16 + 36))
            put32(feat2 + 16 + 76, get32(q_feat2 + 16 + 76))
            put32(feat2 + 16 + 104, get32(q_feat2 + 16 + 104))
            put32(feat2 + 16 + 116, get32(q_feat2 + 16 + 116))  # shaderStorageImageExtendedFormats

            # Vulkan 1.2 features
            put32(vk12 + 20, get32(q_vk12 + 20))   # drawIndirectCount
            put32(vk12 + 52, get32(q_vk12 + 52))   # descriptorIndexing
            put32(vk12 + 72, get32(q_vk12 + 72))   # shaderSampledImageArrayNonUniformIndexing
            put32(vk12 + 100, get32(q_vk12 + 100)) # descriptorBindingSampledImageUpdateAfterBind
            put32(vk12 + 108, get32(q_vk12 + 108)) # descriptorBindingStorageBufferUpdateAfterBind
            put32(vk12 + 124, get32(q_vk12 + 124)) # descriptorBindingPartiallyBound
            put32(vk12 + 132, get32(q_vk12 + 132)) # runtimeDescriptorArray
            put32(vk12 + 140, get32(q_vk12 + 140)) # scalarBlockLayout
            put32(vk12 + 164, 1)                   # timelineSemaphore
            put32(vk12 + 168, get32(q_vk12 + 168)) # bufferDeviceAddress

            self.has_draw_indirect_count = get32(vk12 + 20) != 0
            self.has_descriptor_indexing = get32(vk12 + 52) != 0 and get32(vk12 + 72) != 0
            self.has_bda = get32(vk12 + 168) != 0

            # Vulkan 1.3 features
            put32(vk13 + 52, get32(q_vk13 + 52))   # synchronization2
            put32(vk13 + 64, get32(q_vk13 + 64))   # dynamicRendering
            put32(vk13 + 72, get32(q_vk13 + 72))   # maintenance4
            self.has_sync2 = get32(vk13 + 52) != 0
            self.has_dynamic_rendering = get32(vk13 + 64) != 0

            if want_rt and get32(q_as + 16) != 0 and get32(q_rq + 16) != 0 and self.has_bda:
                var as_f = self.scratch(40)
                var rq_f = self.scratch(24)
                put32(as_f, TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR)
                put32(as_f + 16, 1)
                put64(as_f + 8, rq_f)
                put32(rq_f, TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR)
                put32(rq_f + 16, 1)
                put64(vk13 + 8, as_f)
                self.has_ray_query = True
            pnext_head = feat2
        else:
            var timeline_feature = self.scratch(24)
            put32(timeline_feature, TYPE_TIMELINE_SEMAPHORE_FEATURES)
            put32(timeline_feature + 16, 1)
            pnext_head = timeline_feature

        var info = self.scratch(72)
        put32(info, TYPE_DEVICE_CREATE_INFO)
        put64(info + 8, pnext_head)
        put32(info + 20, UInt32(2) if separate_side else UInt32(1))
        put64(info + 24, queue_info)
        put32(info + 48, UInt32(len(extensions)))
        put64(info + 56, self.name_list(extensions))
        var slot = self.scratch(8)
        var create = self.instance_call("vkCreateDevice")
        if not self.check(call4(create, self.physical, info, 0, slot), "vkCreateDevice"):
            return False
        self.device = get64(slot)
        var get_queue = self.device_call("vkGetDeviceQueue")
        _ = call4(get_queue, self.device, Int(self.family), 0, slot)
        self.queue = get64(slot)
        self.side_queue = self.queue
        if self.side_index >= 0:
            _ = call4(get_queue, self.device, Int(self.side_family), self.side_index, slot)
            self.side_queue = get64(slot)
        if self.has_calibration:
            self.fn_calibrate = self.device_call(calibration)
        return True

    def load_device_calls(mut self) -> Bool:
        self.rewind()
        self.fn_create_buffer = self.device_call("vkCreateBuffer")
        self.fn_destroy_buffer = self.device_call("vkDestroyBuffer")
        self.fn_get_buffer_requirements = self.device_call("vkGetBufferMemoryRequirements")
        self.fn_allocate_memory = self.device_call("vkAllocateMemory")
        self.fn_free_memory = self.device_call("vkFreeMemory")
        self.fn_bind_buffer_memory = self.device_call("vkBindBufferMemory")
        self.fn_map_memory = self.device_call("vkMapMemory")
        self.fn_begin_commands = self.device_call("vkBeginCommandBuffer")
        self.fn_end_commands = self.device_call("vkEndCommandBuffer")
        self.fn_reset_commands = self.device_call("vkResetCommandBuffer")
        self.fn_bind_pipeline = self.device_call("vkCmdBindPipeline")
        self.fn_bind_descriptor_sets = self.device_call("vkCmdBindDescriptorSets")
        self.fn_push_constants = self.device_call("vkCmdPushConstants")
        self.fn_dispatch = self.device_call("vkCmdDispatch")
        self.fn_dispatch_indirect = self.device_call("vkCmdDispatchIndirect")
        self.fn_pipeline_barrier = self.device_call("vkCmdPipelineBarrier")
        self.fn_write_timestamp = self.device_call("vkCmdWriteTimestamp")
        self.fn_reset_query_pool = self.device_call("vkCmdResetQueryPool")
        self.fn_get_query_results = self.device_call("vkGetQueryPoolResults")
        self.fn_queue_submit = self.device_call("vkQueueSubmit")
        self.fn_wait_semaphores = self.device_call("vkWaitSemaphores")
        self.fn_update_descriptors = self.device_call("vkUpdateDescriptorSets")
        if self.has_dma_buf:
            self.fn_get_memory_fd = self.device_call("vkGetMemoryFdKHR")
        return self.error.byte_length() == 0

    def create_queue_objects(mut self) -> Bool:
        self.rewind()
        var slot = self.scratch(8)
        var kind = self.scratch(32)
        put32(kind, TYPE_SEMAPHORE_TYPE_CREATE_INFO)
        put32(kind + 16, SEMAPHORE_TIMELINE)
        var info = self.scratch(24)
        put32(info, TYPE_SEMAPHORE_CREATE_INFO)
        put64(info + 8, kind)
        if not self.check(call4(self.device_call("vkCreateSemaphore"), self.device, info, 0, slot), "vkCreateSemaphore"):
            return False
        self.timeline = get64(slot)
        var pool = self.scratch(24)
        put32(pool, TYPE_COMMAND_POOL_CREATE_INFO)
        put32(pool + 16, COMMAND_POOL_RESET_BUFFER)
        put32(pool + 20, self.family)
        if not self.check(call4(self.device_call("vkCreateCommandPool"), self.device, pool, 0, slot), "vkCreateCommandPool"):
            return False
        self.command_pool = get64(slot)
        self.side_pool = self.command_pool
        if self.side_index >= 0:
            if not self.check(call4(self.device_call("vkCreateSemaphore"), self.device, info, 0, slot), "vkCreateSemaphore"):
                return False
            self.side_timeline = get64(slot)
            if self.side_family != self.family:
                put32(pool + 20, self.side_family)
                if not self.check(call4(self.device_call("vkCreateCommandPool"), self.device, pool, 0, slot), "vkCreateCommandPool"):
                    return False
                self.side_pool = get64(slot)
        return True

    def allocate_command_buffers(mut self, count: Int) -> Bool:
        self.rewind()
        var info = self.scratch(32)
        put32(info, TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
        put64(info + 16, self.command_pool)
        put32(info + 28, UInt32(count))
        var list = self.scratch(count * 8)
        if not self.check(call3(self.device_call("vkAllocateCommandBuffers"), self.device, info, list), "vkAllocateCommandBuffers"):
            return False
        for index in range(count):
            self.command_buffers.append(get64(list + index * 8))
        return True

    def allocate_side_command_buffers(mut self, count: Int) -> Int:
        """Command buffers for the side queue, appended to `command_buffers`; the
        index of the first, -1 on failure. Record them as any other."""
        self.rewind()
        var info = self.scratch(32)
        put32(info, TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
        put64(info + 16, self.side_pool)
        put32(info + 28, UInt32(count))
        var list = self.scratch(count * 8)
        if not self.check(call3(self.device_call("vkAllocateCommandBuffers"), self.device, info, list), "vkAllocateCommandBuffers"):
            return -1
        var first = len(self.command_buffers)
        for index in range(count):
            self.command_buffers.append(get64(list + index * 8))
        return first

    def has_side_queue(self) -> Bool:
        return self.side_index >= 0

    def submit_side(mut self, frame: Int) -> UInt64:
        """End a side command buffer and queue it on the side queue. Its fence counts
        on the side timeline (wait with `wait_side`). Without a side queue it goes
        on the main queue, and the fence is a main one."""
        if not self.has_side_queue():
            return self.submit(frame)
        if not self.check(call1(self.fn_end_commands, self.command_buffers[frame]), "vkEndCommandBuffer"):
            return 0
        self.rewind()
        var value = self.side_next
        var commands = self.scratch(8)
        put64(commands, self.command_buffers[frame])
        var semaphores = self.scratch(8)
        put64(semaphores, self.side_timeline)
        var values = self.scratch(8)
        put64(values, Int(value))
        var timeline = self.scratch(48)
        put32(timeline, TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO)
        put32(timeline + 32, 1)
        put64(timeline + 40, values)
        var info = self.scratch(72)
        put32(info, TYPE_SUBMIT_INFO)
        put64(info + 8, timeline)
        put32(info + 40, 1)
        put64(info + 48, commands)
        put32(info + 56, 1)
        put64(info + 64, semaphores)
        if not self.check(call4(self.fn_queue_submit, self.side_queue, 1, info, 0), "vkQueueSubmit"):
            return 0
        self.side_next += 1
        return value

    def wait_side(mut self, fence: UInt64) -> Bool:
        if not self.has_side_queue():
            return self.wait(fence)
        self.rewind()
        var semaphores = self.scratch(8)
        put64(semaphores, self.side_timeline)
        var values = self.scratch(8)
        put64(values, Int(fence))
        var info = self.scratch(40)
        put32(info, TYPE_SEMAPHORE_WAIT_INFO)
        put32(info + 20, 1)
        put64(info + 24, semaphores)
        put64(info + 32, values)
        var result = call3(self.fn_wait_semaphores, self.device, info, WAIT_TIMEOUT_NS)
        if result == TIMEOUT:
            return self.fail("side queue wait timed out")
        return self.check(result, "vkWaitSemaphores")

    def create_query_pool(mut self, count: Int) -> Bool:
        self.rewind()
        var info = self.scratch(32)
        put32(info, TYPE_QUERY_POOL_CREATE_INFO)
        put32(info + 20, QUERY_TIMESTAMP)
        put32(info + 24, UInt32(count))
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateQueryPool"), self.device, info, 0, slot), "vkCreateQueryPool"):
            return False
        self.query_pool = get64(slot)
        return True

    # Memory.

    def memory_type(self, allowed: UInt32, required: UInt32, forbidden: UInt32) -> Int:
        var count = Int(get32(self.memory_properties))
        for index in range(count):
            var flags = get32(self.memory_properties + 4 + index * 8)
            # Bits past the four core ones need features this backend does not enable.
            if (allowed >> UInt32(index)) & 1 != 0 and (flags & required) == required and (flags & forbidden) == 0:
                return index
        return -1

    def allocation(mut self, size: Int, type_index: Int, next_struct: Int) -> Int:
        """Device memory of this type, or 0 if the heap says no."""
        var info = self.scratch(32)
        put32(info, TYPE_MEMORY_ALLOCATE_INFO)
        put64(info + 8, next_struct)
        put64(info + 16, size)
        put32(info + 24, UInt32(type_index))
        var slot = self.scratch(8)
        if call4(self.fn_allocate_memory, self.device, info, 0, slot) != SUCCESS:
            return 0
        return get64(slot)

    def create_buffer(mut self, size: Int, local: Bool, shared: Bool = False, cpu_reads: Bool = False) -> GpuBuffer:
        """A mapped storage buffer. Local memory is the GPU's own, reachable by the CPU
        through a resizable BAR; where that is too small, or `local` is False, it is
        the CPU's memory. With `cpu_reads` that memory is cached: reading a frame from
        write-combined memory takes seven times as long. Size 0 reports failure."""
        self.rewind()
        var rounded = (size + 4095) & ~4095
        var external = 0
        var export = 0
        if shared and self.has_dma_buf:
            external = self.scratch(24)
            put32(external, TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO)
            put32(external + 16, HANDLE_DMA_BUF)
            export = self.scratch(24)
            put32(export, TYPE_EXPORT_MEMORY_ALLOCATE_INFO)
            put32(export + 16, HANDLE_DMA_BUF)
        var info = self.scratch(56)
        put32(info, TYPE_BUFFER_CREATE_INFO)
        put64(info + 8, external)
        put64(info + 24, rounded)
        put32(info + 32, BUFFER_USAGE)
        var slot = self.scratch(8)
        if call4(self.fn_create_buffer, self.device, info, 0, slot) != SUCCESS:
            _ = self.fail("vkCreateBuffer failed for " + String(rounded) + " bytes")
            return GpuBuffer()
        var buffer = get64(slot)
        var requirements = self.scratch(24)
        _ = call3(self.fn_get_buffer_requirements, self.device, buffer, requirements)
        var needed = get64(requirements)
        var allowed = get32(requirements + 16)
        var host = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT
        var extras = ~(MEMORY_DEVICE_LOCAL | MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT | MEMORY_HOST_CACHED)
        var memory = 0
        if local:
            var bar = self.memory_type(allowed, host | MEMORY_DEVICE_LOCAL, extras)
            if bar >= 0:
                memory = self.allocation(needed, bar, export)
        if memory == 0:
            # Write-combined where the CPU only writes, which is the faster way to do that.
            var system = -1
            if cpu_reads:
                system = self.memory_type(allowed, host | MEMORY_HOST_CACHED, extras | MEMORY_DEVICE_LOCAL)
            if system < 0:
                system = self.memory_type(allowed, host, extras | MEMORY_DEVICE_LOCAL | MEMORY_HOST_CACHED)
            if system < 0:
                system = self.memory_type(allowed, host, extras)
            if system >= 0:
                memory = self.allocation(needed, system, export)
        if memory == 0:
            _ = call3(self.fn_destroy_buffer, self.device, buffer, 0)
            _ = self.fail("no memory for a buffer of " + String(rounded) + " bytes")
            return GpuBuffer()
        if call4(self.fn_bind_buffer_memory, self.device, buffer, memory, 0) != SUCCESS or call6(self.fn_map_memory, self.device, memory, 0, WHOLE_SIZE, 0, slot) != SUCCESS:
            _ = call3(self.fn_destroy_buffer, self.device, buffer, 0)
            _ = call3(self.fn_free_memory, self.device, memory, 0)
            _ = self.fail("buffer binding or mapping failed")
            return GpuBuffer()
        var mapped = GpuBuffer(0, rounded, 0, get64(slot))
        mapped.vk_buffer = buffer
        mapped.vk_memory = memory
        return mapped^

    def destroy_buffer(mut self, buffer: GpuBuffer):
        """Release a buffer the GPU has finished with. The empty value is ignored."""
        if buffer.size == 0:
            return
        _ = call3(self.fn_destroy_buffer, self.device, buffer.vk_buffer, 0)
        _ = call3(self.fn_free_memory, self.device, buffer.vk_memory, 0)

    def export_buffer(mut self, buffer: GpuBuffer) -> Int:
        """Share a buffer as a dmabuf file descriptor. Returns -1 on failure."""
        if not self.has_dma_buf:
            _ = self.fail("this driver cannot share buffers as dmabufs")
            return -1
        self.rewind()
        var info = self.scratch(32)
        put32(info, TYPE_MEMORY_GET_FD_INFO)
        put64(info + 16, buffer.vk_memory)
        put32(info + 24, HANDLE_DMA_BUF)
        var slot = self.scratch(8)
        if call3(self.fn_get_memory_fd, self.device, info, slot) != SUCCESS:
            _ = self.fail("vkGetMemoryFdKHR failed")
            return -1
        return Int(Int32(get32(slot)))

    # Pipelines.

    def create_layouts(mut self, bindings: Int, push_bytes: Int) -> Bool:
        self.rewind()
        self.layout_bindings = bindings
        var table = self.scratch(bindings * 24)
        for index in range(bindings):
            var entry = table + index * 24
            put32(entry, UInt32(index))
            put32(entry + 4, DESCRIPTOR_STORAGE_BUFFER)
            put32(entry + 8, 1)
            put32(entry + 12, STAGE_COMPUTE_SHADER)
        var info = self.scratch(32)
        put32(info, TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
        put32(info + 20, UInt32(bindings))
        put64(info + 24, table)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateDescriptorSetLayout"), self.device, info, 0, slot), "vkCreateDescriptorSetLayout"):
            return False
        self.descriptor_layout = get64(slot)
        var range_info = self.scratch(12)
        put32(range_info, STAGE_COMPUTE_SHADER)
        put32(range_info + 8, UInt32(push_bytes))
        var layouts = self.scratch(8)
        put64(layouts, self.descriptor_layout)
        var layout_info = self.scratch(48)
        put32(layout_info, TYPE_PIPELINE_LAYOUT_CREATE_INFO)
        put32(layout_info + 20, 1)
        put64(layout_info + 24, layouts)
        put32(layout_info + 32, 1)
        put64(layout_info + 40, range_info)
        if not self.check(call4(self.device_call("vkCreatePipelineLayout"), self.device, layout_info, 0, slot), "vkCreatePipelineLayout"):
            return False
        self.pipeline_layout = get64(slot)
        return True

    def create_pipeline(mut self, code: List[UInt32]) -> Bool:
        """Compile one compute shader and add it to `pipelines`."""
        self.rewind()
        var module_info = self.scratch(40)
        put32(module_info, TYPE_SHADER_MODULE_CREATE_INFO)
        put64(module_info + 24, len(code) * 4)
        put64(module_info + 32, Int(code.unsafe_ptr()))
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateShaderModule"), self.device, module_info, 0, slot), "vkCreateShaderModule"):
            return False
        var module = get64(slot)
        var info = self.scratch(96)
        put32(info, TYPE_COMPUTE_PIPELINE_CREATE_INFO)
        put32(info + 24, TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
        put32(info + 44, STAGE_COMPUTE_SHADER)
        put64(info + 48, module)
        put64(info + 56, self.text("main"))
        put64(info + 72, self.pipeline_layout)
        put32(info + 88, 0xFFFFFFFF)
        var result = call6(self.device_call("vkCreateComputePipelines"), self.device, self.pipeline_cache, 1, info, 0, slot)
        var pipeline = get64(slot)
        _ = call3(self.device_call("vkDestroyShaderModule"), self.device, module, 0)
        if not self.check(result, "vkCreateComputePipelines"):
            return False
        self.pipelines.append(pipeline)
        return True

    def create_descriptor_sets(mut self, count: Int, bindings: Int) -> List[Int]:
        """Replace the whole pool: sets last until the next resize."""
        var sets = List[Int]()
        self.rewind()
        var slot = self.scratch(8)
        if self.descriptor_pool == 0:
            var size = self.scratch(8)
            put32(size, DESCRIPTOR_STORAGE_BUFFER)
            put32(size + 4, UInt32(count * bindings))
            var pool = self.scratch(40)
            put32(pool, TYPE_DESCRIPTOR_POOL_CREATE_INFO)
            put32(pool + 20, UInt32(count))
            put32(pool + 24, 1)
            put64(pool + 32, size)
            if not self.check(call4(self.device_call("vkCreateDescriptorPool"), self.device, pool, 0, slot), "vkCreateDescriptorPool"):
                return sets^
            self.descriptor_pool = get64(slot)
        else:
            _ = call3(self.device_call("vkResetDescriptorPool"), self.device, self.descriptor_pool, 0)
        var layouts = self.scratch(count * 8)
        for index in range(count):
            put64(layouts + index * 8, self.descriptor_layout)
        var info = self.scratch(40)
        put32(info, TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
        put64(info + 16, self.descriptor_pool)
        put32(info + 24, UInt32(count))
        put64(info + 32, layouts)
        var list = self.scratch(count * 8)
        if not self.check(call3(self.device_call("vkAllocateDescriptorSets"), self.device, info, list), "vkAllocateDescriptorSets"):
            return sets^
        for index in range(count):
            sets.append(get64(list + index * 8))
        return sets^

    def create_side_descriptor_sets(mut self, count: Int) -> List[Int]:
        """Sets in a pool of their own, for work outside the frame (GPU meshing): a
        resize resets the frame's pool and would take them with it. Same layout, so
        the pool holds every binding of it; the user writes the ones it reads."""
        var sets = List[Int]()
        var bindings = self.layout_bindings
        self.rewind()
        var slot = self.scratch(8)
        var size = self.scratch(8)
        put32(size, DESCRIPTOR_STORAGE_BUFFER)
        put32(size + 4, UInt32(count * bindings))
        var pool_info = self.scratch(40)
        put32(pool_info, TYPE_DESCRIPTOR_POOL_CREATE_INFO)
        put32(pool_info + 20, UInt32(count))
        put32(pool_info + 24, 1)
        put64(pool_info + 32, size)
        if not self.check(call4(self.device_call("vkCreateDescriptorPool"), self.device, pool_info, 0, slot), "vkCreateDescriptorPool"):
            return sets^
        var pool = get64(slot)
        self.side_pools.append(pool)
        var layouts = self.scratch(count * 8)
        for index in range(count):
            put64(layouts + index * 8, self.descriptor_layout)
        var info = self.scratch(40)
        put32(info, TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
        put64(info + 16, pool)
        put32(info + 24, UInt32(count))
        put64(info + 32, layouts)
        var list = self.scratch(count * 8)
        if not self.check(call3(self.device_call("vkAllocateDescriptorSets"), self.device, info, list), "vkAllocateDescriptorSets"):
            return sets^
        for index in range(count):
            sets.append(get64(list + index * 8))
        return sets^

    def write_descriptor(mut self, descriptor_set: Int, binding: Int, buffer: GpuBuffer):
        self.rewind()
        var target = self.scratch(24)
        put64(target, buffer.vk_buffer)
        put64(target + 16, WHOLE_SIZE)
        var write = self.scratch(64)
        put32(write, TYPE_WRITE_DESCRIPTOR_SET)
        put64(write + 16, descriptor_set)
        put32(write + 24, UInt32(binding))
        put32(write + 32, 1)
        put32(write + 36, DESCRIPTOR_STORAGE_BUFFER)
        put64(write + 48, target)
        _ = call5(self.fn_update_descriptors, self.device, 1, write, 0, 0)

    # Commands. One command buffer per frame set, recorded afresh each frame.

    def begin_commands(mut self, frame: Int) -> Bool:
        self.rewind()
        var commands = self.command_buffers[frame]
        if not self.check(call2(self.fn_reset_commands, commands, 0), "vkResetCommandBuffer"):
            return False
        var info = self.scratch(32)
        put32(info, TYPE_COMMAND_BUFFER_BEGIN_INFO)
        put32(info + 16, COMMAND_ONE_TIME)
        return self.check(call2(self.fn_begin_commands, commands, info), "vkBeginCommandBuffer")

    def barrier(mut self, frame: Int, source_stage: Int, source_access: UInt32, target_stage: Int, target_access: UInt32):
        self.rewind()
        var info = self.scratch(24)
        put32(info, TYPE_MEMORY_BARRIER)
        put32(info + 16, source_access)
        put32(info + 20, target_access)
        _ = call10(self.fn_pipeline_barrier, self.command_buffers[frame], source_stage, target_stage, 0, 1, info, 0, 0, 0, 0)

    def pass_barrier(mut self, frame: Int):
        """One pass's writes reach the next pass's reads."""
        self.barrier(frame, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)

    def dispatch(mut self, frame: Int, pipeline: Int, descriptor_set: Int, push: List[UInt32], groups_x: Int, groups_y: Int):
        self.rewind()
        var commands = self.command_buffers[frame]
        var sets = self.scratch(8)
        put64(sets, descriptor_set)
        _ = call3(self.fn_bind_pipeline, commands, BIND_POINT_COMPUTE, self.pipelines[pipeline])
        _ = call8(self.fn_bind_descriptor_sets, commands, BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, sets, 0, 0)
        _ = call6(self.fn_push_constants, commands, self.pipeline_layout, Int(STAGE_COMPUTE_SHADER), 0, len(push) * 4, Int(push.unsafe_ptr()))
        _ = call4(self.fn_dispatch, commands, groups_x, groups_y, 1)

    def dispatch_indirect(mut self, frame: Int, pipeline: Int, descriptor_set: Int, push: List[UInt32], arguments: GpuBuffer, offset: Int):
        """The group counts come from `arguments` at byte `offset`, written by an
        earlier pass: an `indirect_barrier` must stand between them."""
        self.rewind()
        var commands = self.command_buffers[frame]
        var sets = self.scratch(8)
        put64(sets, descriptor_set)
        _ = call3(self.fn_bind_pipeline, commands, BIND_POINT_COMPUTE, self.pipelines[pipeline])
        _ = call8(self.fn_bind_descriptor_sets, commands, BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, sets, 0, 0)
        _ = call6(self.fn_push_constants, commands, self.pipeline_layout, Int(STAGE_COMPUTE_SHADER), 0, len(push) * 4, Int(push.unsafe_ptr()))
        _ = call3(self.fn_dispatch_indirect, commands, arguments.vk_buffer, offset)

    def indirect_barrier(mut self, frame: Int):
        """A pass's writes reach the next pass and the indirect dispatches' counts."""
        self.barrier(frame, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_COMPUTE | PIPELINE_DRAW_INDIRECT, ACCESS_SHADER_READ | ACCESS_SHADER_WRITE | ACCESS_INDIRECT_READ)

    def reset_queries(mut self, frame: Int, first: Int, count: Int):
        _ = call4(self.fn_reset_query_pool, self.command_buffers[frame], self.query_pool, first, count)

    def write_timestamp(mut self, frame: Int, query: Int, stage: Int = PIPELINE_BOTTOM):
        _ = call4(self.fn_write_timestamp, self.command_buffers[frame], stage, self.query_pool, query)

    def read_queries(mut self, first: Int, count: Int, target: Int) -> Bool:
        """Copy finished timestamps as 64-bit values. Unfinished ones stay as they were."""
        return call8(self.fn_get_query_results, self.device, self.query_pool, first, count, count * 8, target, 8, QUERY_RESULT_64) == SUCCESS

    def read_clock(mut self) -> UInt64:
        """The counter that timestamps read, now. 0 on failure."""
        if self.fn_calibrate == 0:
            return 0
        self.rewind()
        var info = self.scratch(24)
        put32(info, TYPE_CALIBRATED_TIMESTAMP_INFO)
        put32(info + 16, TIME_DOMAIN_DEVICE)
        var value = self.scratch(8)
        var deviation = self.scratch(8)
        if call5(self.fn_calibrate, self.device, 1, info, value, deviation) != SUCCESS:
            return 0
        return UInt64(get64(value))

    def submit(mut self, frame: Int) -> UInt64:
        """End the frame's commands and queue them. Returns the fence, 0 on failure."""
        if not self.check(call1(self.fn_end_commands, self.command_buffers[frame]), "vkEndCommandBuffer"):
            return 0
        self.rewind()
        var value = self.next_signal
        var commands = self.scratch(8)
        put64(commands, self.command_buffers[frame])
        var semaphores = self.scratch(8)
        put64(semaphores, self.timeline)
        var values = self.scratch(8)
        put64(values, Int(value))
        var timeline = self.scratch(48)
        put32(timeline, TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO)
        put32(timeline + 32, 1)
        put64(timeline + 40, values)
        var info = self.scratch(72)
        put32(info, TYPE_SUBMIT_INFO)
        put64(info + 8, timeline)
        put32(info + 40, 1)
        put64(info + 48, commands)
        put32(info + 56, 1)
        put64(info + 64, semaphores)
        if not self.check(call4(self.fn_queue_submit, self.queue, 1, info, 0), "vkQueueSubmit"):
            return 0
        self.next_signal += 1
        return value

    def wait(mut self, fence: UInt64) -> Bool:
        self.rewind()
        var semaphores = self.scratch(8)
        put64(semaphores, self.timeline)
        var values = self.scratch(8)
        put64(values, Int(fence))
        var info = self.scratch(40)
        put32(info, TYPE_SEMAPHORE_WAIT_INFO)
        put32(info + 20, 1)
        put64(info + 24, semaphores)
        put64(info + 32, values)
        var result = call3(self.fn_wait_semaphores, self.device, info, WAIT_TIMEOUT_NS)
        if result == TIMEOUT:
            return self.fail("fence wait timed out")
        return self.check(result, "vkWaitSemaphores")

    # Presentation through VK_KHR_wayland_surface.

    def create_binary_semaphore(mut self) -> Int:
        var info = self.scratch(24)
        put32(info, TYPE_SEMAPHORE_CREATE_INFO)
        var slot = self.scratch(8)
        if call4(self.device_call("vkCreateSemaphore"), self.device, info, 0, slot) != SUCCESS:
            return 0
        return get64(slot)

    def create_surface(mut self, display: Int, wayland_surface: Int, frames: Int) -> Bool:
        """Make a Vulkan surface of this libwayland surface, and check we can present to it."""
        if not self.has_wayland_surface or not self.has_swapchain:
            return self.fail("this driver has no Wayland swapchain")
        self.rewind()
        var info = self.scratch(40)
        put32(info, TYPE_WAYLAND_SURFACE_CREATE_INFO)
        put64(info + 24, display)
        put64(info + 32, wayland_surface)
        var slot = self.scratch(8)
        if not self.check(call4(self.instance_call("vkCreateWaylandSurfaceKHR"), self.instance, info, 0, slot), "vkCreateWaylandSurfaceKHR"):
            return False
        self.surface = get64(slot)
        var supported = self.scratch(8)
        if not self.check(call4(self.instance_call("vkGetPhysicalDeviceSurfaceSupportKHR"), self.physical, Int(self.family), self.surface, supported), "vkGetPhysicalDeviceSurfaceSupportKHR"):
            return False
        if get32(supported) == 0:
            return self.fail("this queue cannot present to the window")
        self.rewind()
        for _ in range(frames):
            self.acquire_semaphores.append(self.create_binary_semaphore())
        return True

    def create_xcb_surface(mut self, connection: Int, window: Int, frames: Int = 2) -> Bool:
        """Make a Vulkan surface of this X11 XCB connection/window, and check we can present to it."""
        if not self.has_xcb_surface or not self.has_swapchain:
            return self.fail("this driver has no XCB swapchain")
        self.rewind()
        var info = self.scratch(40)
        put32(info, TYPE_XCB_SURFACE_CREATE_INFO)
        put64(info + 8, 0)
        put32(info + 16, 0)
        put64(info + 24, connection)
        put64(info + 32, window)
        var slot = self.scratch(8)
        if not self.check(call4(self.instance_call("vkCreateXcbSurfaceKHR"), self.instance, info, 0, slot), "vkCreateXcbSurfaceKHR"):
            return False
        self.surface = get64(slot)
        var supported = self.scratch(8)
        if not self.check(call4(self.instance_call("vkGetPhysicalDeviceSurfaceSupportKHR"), self.physical, Int(self.family), self.surface, supported), "vkGetPhysicalDeviceSurfaceSupportKHR"):
            return False
        if get32(supported) == 0:
            return self.fail("this queue cannot present to the X11 window")
        self.rewind()
        for _ in range(frames):
            self.acquire_semaphores.append(self.create_binary_semaphore())
        return True

    def create_drm_display_surface(mut self, drm_fd: Int, connector: Int, width: Int, height: Int, refresh_mhz: Int, frames: Int = 2, exact_size: Bool = False) -> Bool:
        """Present straight to a KMS connector, no compositor: take the display through our
        DRM master fd (VK_EXT_acquire_drm_display), match the KMS mode, pick a plane that can
        show it, and make a display-plane surface. The swapchain is then the usual one.
        With `exact_size`, fail instead of taking another size when the display has no
        width x height mode (the swapchain must match a renderer that cannot scale)."""
        if not self.has_display_surface or not self.has_swapchain:
            return self.fail("this driver cannot present to a display directly (VK_KHR_display / VK_EXT_acquire_drm_display)")
        self.rewind()
        var slot = self.scratch(8)
        if not self.check(call4(self.instance_call("vkGetDrmDisplayEXT"), self.physical, drm_fd, connector, slot), "vkGetDrmDisplayEXT (is this the GPU behind the DRM card?)"):
            return False
        var display = get64(slot)
        if not self.check(call3(self.instance_call("vkAcquireDrmDisplayEXT"), self.physical, drm_fd, display), "vkAcquireDrmDisplayEXT (needs DRM master)"):
            return False
        self.display = display
        # The display's modes.
        self.rewind()
        var count_slot = self.scratch(8)
        var list_modes = self.instance_call("vkGetDisplayModePropertiesKHR")
        _ = call4(list_modes, self.physical, display, count_slot, 0)
        var mode_count = Int(get32(count_slot))
        var modes = self.scratch(max(1, mode_count) * DISPLAY_MODE_PROPERTIES_BYTES)
        if not self.check(call4(list_modes, self.physical, display, count_slot, modes), "vkGetDisplayModePropertiesKHR"):
            return False
        mode_count = Int(get32(count_slot))
        var widths = List[Int]()
        var heights = List[Int]()
        var rates = List[Int]()
        for index in range(mode_count):
            widths.append(Int(get32(modes + index * DISPLAY_MODE_PROPERTIES_BYTES + 8)))
            heights.append(Int(get32(modes + index * DISPLAY_MODE_PROPERTIES_BYTES + 12)))
            rates.append(Int(get32(modes + index * DISPLAY_MODE_PROPERTIES_BYTES + 16)))
        var chosen = choose_display_mode(widths, heights, rates, width, height, refresh_mhz)
        if chosen < 0:
            return self.fail("the display offers no modes")
        if exact_size and (widths[chosen] != width or heights[chosen] != height):
            return self.fail("the display has no " + String(width) + "x" + String(height) + " mode")
        var mode = get64(modes + chosen * DISPLAY_MODE_PROPERTIES_BYTES)
        var mode_w = widths[chosen]
        var mode_h = heights[chosen]
        # A plane that can show this display: unused or already on it.
        var list_planes = self.instance_call("vkGetPhysicalDeviceDisplayPlanePropertiesKHR")
        _ = call3(list_planes, self.physical, count_slot, 0)
        var plane_count = Int(get32(count_slot))
        var planes = self.scratch(max(1, plane_count) * DISPLAY_PLANE_PROPERTIES_BYTES)
        if not self.check(call3(list_planes, self.physical, count_slot, planes), "vkGetPhysicalDeviceDisplayPlanePropertiesKHR"):
            return False
        var supported_list = self.instance_call("vkGetDisplayPlaneSupportedDisplaysKHR")
        var plane = -1
        var stack = 0
        for index in range(Int(get32(count_slot))):
            var current = get64(planes + index * DISPLAY_PLANE_PROPERTIES_BYTES)
            if current != 0 and current != display:
                continue
            var display_count = self.scratch(8)
            _ = call4(supported_list, self.physical, index, display_count, 0)
            var displays = self.scratch(max(1, Int(get32(display_count))) * 8)
            _ = call4(supported_list, self.physical, index, display_count, displays)
            for d in range(Int(get32(display_count))):
                if get64(displays + d * 8) == display:
                    plane = index
                    stack = Int(get32(planes + index * DISPLAY_PLANE_PROPERTIES_BYTES + 8))
                    break
            if plane >= 0:
                break
        if plane < 0:
            return self.fail("no display plane can show this connector")
        var capabilities = self.scratch(80)
        _ = call4(self.instance_call("vkGetDisplayPlaneCapabilitiesKHR"), self.physical, mode, plane, capabilities)
        var alphas = get32(capabilities)
        var info = self.scratch(64)
        put32(info, TYPE_DISPLAY_SURFACE_CREATE_INFO)
        put64(info + 24, mode)
        put32(info + 32, UInt32(plane))
        put32(info + 36, UInt32(stack))
        put32(info + 40, SURFACE_TRANSFORM_IDENTITY)
        put_float(info + 44, 1.0)
        put32(info + 48, DISPLAY_ALPHA_OPAQUE if (alphas & DISPLAY_ALPHA_OPAQUE) != 0 or alphas == 0 else (alphas & (~alphas + 1)))
        put32(info + 52, UInt32(mode_w))
        put32(info + 56, UInt32(mode_h))
        var surface_slot = self.scratch(8)
        if not self.check(call4(self.instance_call("vkCreateDisplayPlaneSurfaceKHR"), self.instance, info, 0, surface_slot), "vkCreateDisplayPlaneSurfaceKHR"):
            return False
        self.surface = get64(surface_slot)
        var supported = self.scratch(8)
        if not self.check(call4(self.instance_call("vkGetPhysicalDeviceSurfaceSupportKHR"), self.physical, Int(self.family), self.surface, supported), "vkGetPhysicalDeviceSurfaceSupportKHR"):
            return False
        if get32(supported) == 0:
            return self.fail("this queue cannot present to the display")
        self.rewind()
        for _ in range(frames):
            self.acquire_semaphores.append(self.create_binary_semaphore())
        return True

    def release_display(mut self):
        """Hand the KMS connector back (VK_EXT_direct_mode_display)."""
        if self.display != 0 and self.instance != 0:
            self.rewind()
            _ = call2(self.instance_call("vkReleaseDisplayEXT"), self.physical, self.display)
        self.display = 0

    def attach_surface(
        mut self,
        wl_display: Int = 0,
        wl_surface: Int = 0,
        xcb_conn: Int = 0,
        xcb_win: Int = 0,
        width: Int = 0,
        height: Int = 0,
        prefer_mailbox: Bool = True,
        prefer_hdr10: Bool = False,
        drm_fd: Int = -1,
        drm_connector: Int = 0,
        drm_refresh_mhz: Int = 60000,
        drm_exact_size: Bool = False,
    ) -> SwapchainHandle:
        """Attaches a Wayland, XCB or KMS display surface and creates swapchain with FIFO/MAILBOX/HDR10.
        For KMS pass the window's DRM fd and connector (`KmsWindow.drm_fd`, `.connector_id`),
        its mode size as width/height, and `drm_refresh_mhz`."""
        if wl_display != 0 and wl_surface != 0:
            if not self.create_surface(wl_display, wl_surface, 2):
                return SwapchainHandle()
        elif xcb_conn != 0 and xcb_win != 0:
            if not self.create_xcb_surface(xcb_conn, xcb_win, 2):
                return SwapchainHandle()
        elif drm_fd >= 0 and drm_connector != 0:
            if not self.create_drm_display_surface(drm_fd, drm_connector, width, height, drm_refresh_mhz, 2, drm_exact_size):
                self.release_display()
                return SwapchainHandle()
        else:
            return SwapchainHandle()

        var sw = width if width > 0 else 1280
        var sh = height if height > 0 else 720
        if not self.create_swapchain(sw, sh, prefer_mailbox=prefer_mailbox, prefer_hdr10=prefer_hdr10):
            return SwapchainHandle()

        return SwapchainHandle(
            self.surface,
            self.swapchain,
            self.swapchain_width,
            self.swapchain_height,
            self.swapchain_format,
            self.swapchain_color_space,
            self.swapchain_present_mode,
            len(self.images),
            self.is_hdr,
        )

    def create_image_view(mut self, img: Int, format: UInt32, aspect: UInt32 = ASPECT_COLOR) -> Int:
        self.rewind()
        var view_info = self.scratch(80)
        put32(view_info, TYPE_IMAGE_VIEW_CREATE_INFO)
        put64(view_info + 24, img)
        put32(view_info + 32, IMAGE_VIEW_TYPE_2D)
        put32(view_info + 36, format)
        put32(view_info + 56, aspect)
        put32(view_info + 60, 0)
        put32(view_info + 64, 1)
        put32(view_info + 68, 0)
        put32(view_info + 72, 1)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateImageView"), self.device, view_info, 0, slot), "vkCreateImageView"):
            return 0
        return get64(slot)

    def destroy_image_view(mut self, view: Int):
        if view != 0:
            _ = call3(self.device_call("vkDestroyImageView"), self.device, view, 0)

    def create_swapchain(
        mut self,
        width: Int,
        height: Int,
        prefer_mailbox: Bool = True,
        prefer_hdr10: Bool = False,
        old_swapchain: UInt64 = 0,
        prefer_immediate: Bool = False,
    ) -> Bool:
        """Creates swapchain supporting FIFO/MAILBOX/IMMEDIATE and SDR/HDR10 PQ/Extended Linear sRGB."""
        self.rewind()
        var capabilities = self.scratch(52)
        if not self.check(call3(self.instance_call("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"), self.physical, self.surface, capabilities), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"):
            return False
        var sup_usages = get32(capabilities + 48)
        if (sup_usages & IMAGE_USAGE_TRANSFER_TARGET) == 0:
            return self.fail("swapchain images cannot be copied to")
        var list_formats = self.instance_call("vkGetPhysicalDeviceSurfaceFormatsKHR")
        var count_slot = self.scratch(8)
        _ = call4(list_formats, self.physical, self.surface, count_slot, 0)
        var count = Int(get32(count_slot))
        var formats = self.scratch(max(count, 1) * 8)
        _ = call4(list_formats, self.physical, self.surface, count_slot, formats)

        var chosen_format = FORMAT_B8G8R8A8_UNORM
        var chosen_color_space = COLOR_SPACE_SRGB_NONLINEAR
        var is_hdr = False

        if prefer_hdr10:
            for index in range(count):
                var fmt = get32(formats + index * 8)
                var cs = get32(formats + index * 8 + 4)
                if fmt == FORMAT_A2B10G10R10_UNORM_PACK32 and cs == COLOR_SPACE_HDR10_ST2084:
                    chosen_format = fmt
                    chosen_color_space = cs
                    is_hdr = True
                    break
            if not is_hdr:
                for index in range(count):
                    var fmt = get32(formats + index * 8)
                    var cs = get32(formats + index * 8 + 4)
                    if fmt == FORMAT_R16G16B16A16_SFLOAT and cs == COLOR_SPACE_EXTENDED_SRGB_LINEAR:
                        chosen_format = fmt
                        chosen_color_space = cs
                        is_hdr = True
                        break

        if not is_hdr:
            var found_sdr = False
            for index in range(count):
                var fmt = get32(formats + index * 8)
                var cs = get32(formats + index * 8 + 4)
                if (fmt == FORMAT_B8G8R8A8_UNORM or fmt == FORMAT_B8G8R8A8_SRGB or fmt == FORMAT_R8G8B8A8_UNORM) and cs == COLOR_SPACE_SRGB_NONLINEAR:
                    chosen_format = fmt
                    chosen_color_space = cs
                    found_sdr = True
                    break
            if not found_sdr and count > 0:
                chosen_format = get32(formats)
                chosen_color_space = get32(formats + 4)

        self.swapchain_format = chosen_format
        self.swapchain_color_space = chosen_color_space
        self.is_hdr = is_hdr

        var list_modes = self.instance_call("vkGetPhysicalDeviceSurfacePresentModesKHR")
        _ = call4(list_modes, self.physical, self.surface, count_slot, 0)
        var mode_count = Int(get32(count_slot))
        var modes = self.scratch(max(mode_count, 1) * 4)
        _ = call4(list_modes, self.physical, self.surface, count_slot, modes)

        var has_mailbox = False
        var has_immediate = False
        for index in range(mode_count):
            var pm = get32(modes + index * 4)
            if pm == PRESENT_MAILBOX:
                has_mailbox = True
            elif pm == PRESENT_IMMEDIATE:
                has_immediate = True

        var chosen_present_mode = PRESENT_FIFO
        if prefer_immediate and has_immediate:
            chosen_present_mode = PRESENT_IMMEDIATE
        elif prefer_mailbox and has_mailbox:
            chosen_present_mode = PRESENT_MAILBOX
        self.swapchain_present_mode = chosen_present_mode

        var minimum = get32(capabilities)
        var maximum = get32(capabilities + 4)
        var wanted = max(minimum, UInt32(3) if chosen_present_mode == PRESENT_MAILBOX else UInt32(2))
        var eff_images = min(wanted, maximum) if maximum != 0 else wanted

        var cur_w = get32(capabilities + 8)
        var cur_h = get32(capabilities + 12)
        var final_w = UInt32(width)
        var final_h = UInt32(height)
        if cur_w != 0xFFFFFFFF and cur_h != 0xFFFFFFFF:
            final_w = cur_w
            final_h = cur_h

        var usage_flags = IMAGE_USAGE_TRANSFER_TARGET | (sup_usages & IMAGE_USAGE_COLOR_ATTACHMENT)
        if (sup_usages & IMAGE_USAGE_STORAGE) != 0:
            usage_flags |= IMAGE_USAGE_STORAGE

        var prev_sc = Int(old_swapchain) if old_swapchain != 0 else self.swapchain
        var info = self.scratch(104)
        put32(info, TYPE_SWAPCHAIN_CREATE_INFO)
        put64(info + 24, self.surface)
        put32(info + 32, eff_images)
        put32(info + 36, chosen_format)
        put32(info + 40, chosen_color_space)
        put32(info + 44, final_w)
        put32(info + 48, final_h)
        put32(info + 52, 1)
        put32(info + 56, usage_flags)
        put32(info + 80, get32(capabilities + 40))
        var alphas = get32(capabilities + 44)
        put32(info + 84, COMPOSITE_OPAQUE if (alphas & COMPOSITE_OPAQUE) != 0 else (alphas & (~alphas + 1)))
        put32(info + 88, chosen_present_mode)
        put32(info + 92, 1)
        put64(info + 96, prev_sc)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateSwapchainKHR"), self.device, info, 0, slot), "vkCreateSwapchainKHR"):
            return False
        var replaced = self.swapchain
        self.swapchain = get64(slot)
        if len(self.image_views) > 0:
            for v_idx in range(len(self.image_views)):
                self.destroy_image_view(self.image_views[v_idx])
            self.image_views.clear()
        if prev_sc != 0 and prev_sc != self.swapchain:
            _ = call3(self.device_call("vkDestroySwapchainKHR"), self.device, prev_sc, 0)
        if replaced != 0 and replaced != prev_sc and replaced != self.swapchain:
            _ = call3(self.device_call("vkDestroySwapchainKHR"), self.device, replaced, 0)
        self.swapchain_width = Int(final_w)
        self.swapchain_height = Int(final_h)
        var list_images = self.device_call("vkGetSwapchainImagesKHR")
        _ = call4(list_images, self.device, self.swapchain, count_slot, 0)
        var image_count = Int(get32(count_slot))
        var images = self.scratch(image_count * 8)
        if not self.check(call4(list_images, self.device, self.swapchain, count_slot, images), "vkGetSwapchainImagesKHR"):
            return False
        self.images.clear()
        for index in range(image_count):
            self.images.append(get64(images + index * 8))
        while len(self.finished_semaphores) < image_count:
            self.finished_semaphores.append(self.create_binary_semaphore())
        for index in range(image_count):
            var view = self.create_image_view(self.images[index], chosen_format, ASPECT_COLOR)
            self.image_views.append(view)
        return True

    def acquire_image(mut self, frame: Int) -> Int:
        """The next image to fill, or NO_IMAGE to skip presenting this frame."""
        self.rewind()
        var slot = self.scratch(8)
        var result = call6(self.device_call("vkAcquireNextImageKHR"), self.device, self.swapchain, ACQUIRE_TIMEOUT_NS, self.acquire_semaphores[frame], 0, slot)
        if result == SUCCESS or result == SUBOPTIMAL:
            return Int(get32(slot))
        self.skipped += 1
        if result == ERROR_OUT_OF_DATE:
            self.wait_idle()
            _ = self.create_swapchain(self.swapchain_width, self.swapchain_height)
        elif result != TIMEOUT and result != NOT_READY:
            self.error = "vkAcquireNextImageKHR returned " + String(result)
        return NO_IMAGE

    def image_barrier(mut self, frame: Int, image: Int, old_layout: UInt32, new_layout: UInt32, source_stage: Int, source_access: UInt32, target_stage: Int, target_access: UInt32):
        self.rewind()
        var barrier = self.scratch(72)
        put32(barrier, TYPE_IMAGE_MEMORY_BARRIER)
        put32(barrier + 16, source_access)
        put32(barrier + 20, target_access)
        put32(barrier + 24, old_layout)
        put32(barrier + 28, new_layout)
        put32(barrier + 32, 0xFFFFFFFF)
        put32(barrier + 36, 0xFFFFFFFF)
        put64(barrier + 40, self.images[image])
        put32(barrier + 48, ASPECT_COLOR)
        put32(barrier + 56, 1)
        put32(barrier + 64, 1)
        _ = call10(self.fn_pipeline_barrier, self.command_buffers[frame], source_stage, target_stage, 0, 0, 0, 0, 0, 1, barrier)

    def copy_to_image(
        mut self,
        frame: Int,
        source: GpuBuffer,
        row_pixels: Int,
        image: Int,
        buffer_offset: UInt64 = 0,
        copy_width: Int = 0,
        copy_height: Int = 0,
    ):
        """The finished frame, into the swapchain image, ready to present."""
        var cw = min(copy_width, self.swapchain_width) if copy_width > 0 else self.swapchain_width
        var ch = min(copy_height, self.swapchain_height) if copy_height > 0 else self.swapchain_height
        cw = max(1, min(cw, row_pixels)) if row_pixels > 0 else max(1, cw)
        ch = max(1, ch)
        var bpp = swapchain_format_bytes_per_pixel(self.swapchain_format)
        var stride_px = max(cw, row_pixels)
        var needed_bytes = Int(buffer_offset) + ((ch - 1) * stride_px + cw) * bpp
        if source.vk_buffer == 0 or source.size < needed_bytes:
            self.error = "copy_to_image source buffer byte size is smaller than target swapchain format stride"
            return
        self.barrier(frame, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_TRANSFER, ACCESS_TRANSFER_READ)
        self.image_barrier(frame, image, LAYOUT_UNDEFINED, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self.rewind()
        var region = self.scratch(56)
        put64(region, Int(buffer_offset))
        put32(region + 8, UInt32(row_pixels))
        put32(region + 16, ASPECT_COLOR)
        put32(region + 28, 1)
        put32(region + 44, UInt32(cw))
        put32(region + 48, UInt32(ch))
        put32(region + 52, 1)
        _ = call6(self.device_call("vkCmdCopyBufferToImage"), self.command_buffers[frame], source.vk_buffer, self.images[image], Int(LAYOUT_TRANSFER_TARGET), 1, region)
        self.image_barrier(frame, image, LAYOUT_TRANSFER_TARGET, LAYOUT_PRESENT, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_BOTTOM, 0)

    def submit_and_present(mut self, frame: Int, image: Int) -> UInt64:
        """As `submit`, waiting for the acquired image and handing it to the compositor."""
        if not self.check(call1(self.fn_end_commands, self.command_buffers[frame]), "vkEndCommandBuffer"):
            return 0
        self.rewind()
        var value = self.next_signal
        var commands = self.scratch(8)
        put64(commands, self.command_buffers[frame])
        var waits = self.scratch(8)
        put64(waits, self.acquire_semaphores[frame])
        var stages = self.scratch(8)
        put32(stages, UInt32(PIPELINE_TRANSFER))
        var signals = self.scratch(16)
        put64(signals, self.timeline)
        put64(signals + 8, self.finished_semaphores[image])
        var values = self.scratch(16)
        put64(values, Int(value))
        var wait_values = self.scratch(8)
        var timeline = self.scratch(48)
        put32(timeline, TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO)
        put32(timeline + 16, 1)
        put64(timeline + 24, wait_values)
        put32(timeline + 32, 2)
        put64(timeline + 40, values)
        var info = self.scratch(72)
        put32(info, TYPE_SUBMIT_INFO)
        put64(info + 8, timeline)
        put32(info + 16, 1)
        put64(info + 24, waits)
        put64(info + 32, stages)
        put32(info + 40, 1)
        put64(info + 48, commands)
        put32(info + 56, 2)
        put64(info + 64, signals)
        if not self.check(call4(self.fn_queue_submit, self.queue, 1, info, 0), "vkQueueSubmit"):
            return 0
        self.next_signal += 1
        var swapchains = self.scratch(8)
        put64(swapchains, self.swapchain)
        var indices = self.scratch(8)
        put32(indices, UInt32(image))
        var present_waits = self.scratch(8)
        put64(present_waits, self.finished_semaphores[image])
        var present = self.scratch(64)
        put32(present, TYPE_PRESENT_INFO)
        put32(present + 16, 1)
        put64(present + 24, present_waits)
        put32(present + 32, 1)
        put64(present + 40, swapchains)
        put64(present + 48, indices)
        var result = call2(self.device_call("vkQueuePresentKHR"), self.queue, present)
        if result == SUCCESS or result == SUBOPTIMAL:
            self.presented += 1
        elif result == ERROR_OUT_OF_DATE:
            self.skipped += 1
        else:
            self.error = "vkQueuePresentKHR returned " + String(result)
        return value

    def close_swapchain(mut self):
        """Before the window goes: the driver holds buffers on its connection."""
        if self.surface == 0:
            return
        self.wait_idle()
        self.rewind()
        if len(self.image_views) > 0:
            for v_idx in range(len(self.image_views)):
                self.destroy_image_view(self.image_views[v_idx])
            self.image_views.clear()
        if self.swapchain != 0:
            _ = call3(self.device_call("vkDestroySwapchainKHR"), self.device, self.swapchain, 0)
            self.swapchain = 0
        self.rewind()
        var destroy = self.device_call("vkDestroySemaphore")
        for semaphore in self.acquire_semaphores:
            _ = call3(destroy, self.device, semaphore, 0)
        for semaphore in self.finished_semaphores:
            _ = call3(destroy, self.device, semaphore, 0)
        self.acquire_semaphores.clear()
        self.finished_semaphores.clear()
        self.images.clear()
        _ = call3(self.instance_call("vkDestroySurfaceKHR"), self.instance, self.surface, 0)
        self.surface = 0
        self.release_display()

    def validation_errors(self) -> Int:
        return get64(self.messages)

    def validation_warnings(self) -> Int:
        return get64(self.messages + 8)

    def wait_idle(mut self):
        if self.device != 0:
            self.rewind()
            _ = call1(self.device_call("vkDeviceWaitIdle"), self.device)

    def close(mut self):
        """Destroy everything but the buffers, which belong to their owner and must be
        gone by now. The validation layers report anything left over here."""
        if self.instance == 0:
            self._free_host_scratch()
            if self.library != 0:
                _ = dl_close(self.library)
                self.library = 0
                self.proc_instance = 0
                self.proc_device = 0
            return
        self.close_swapchain()
        self.rewind()
        if self.device != 0:
            _ = call1(self.device_call("vkDeviceWaitIdle"), self.device)
            for index in range(len(self.pipelines)):
                _ = call3(self.device_call("vkDestroyPipeline"), self.device, self.pipelines[index], 0)
            self.pipelines.clear()
            if self.pipeline_cache != 0:
                _ = call3(self.device_call("vkDestroyPipelineCache"), self.device, self.pipeline_cache, 0)
                self.pipeline_cache = 0
            self.rewind()
            if self.descriptor_pool != 0:
                _ = call3(self.device_call("vkDestroyDescriptorPool"), self.device, self.descriptor_pool, 0)
            for pool in self.side_pools:
                _ = call3(self.device_call("vkDestroyDescriptorPool"), self.device, pool, 0)
            self.side_pools.clear()
            if self.pipeline_layout != 0:
                _ = call3(self.device_call("vkDestroyPipelineLayout"), self.device, self.pipeline_layout, 0)
            if self.descriptor_layout != 0:
                _ = call3(self.device_call("vkDestroyDescriptorSetLayout"), self.device, self.descriptor_layout, 0)
            if self.query_pool != 0:
                _ = call3(self.device_call("vkDestroyQueryPool"), self.device, self.query_pool, 0)
            if self.side_pool != 0 and self.side_pool != self.command_pool:
                _ = call3(self.device_call("vkDestroyCommandPool"), self.device, self.side_pool, 0)
            self.side_pool = 0
            if self.side_timeline != 0:
                _ = call3(self.device_call("vkDestroySemaphore"), self.device, self.side_timeline, 0)
            self.side_timeline = 0
            if self.command_pool != 0:
                _ = call3(self.device_call("vkDestroyCommandPool"), self.device, self.command_pool, 0)
            if self.timeline != 0:
                _ = call3(self.device_call("vkDestroySemaphore"), self.device, self.timeline, 0)
            _ = call2(self.device_call("vkDestroyDevice"), self.device, 0)
            self.device = 0
            self.device_call_cache.clear()
            self.device_call_cache_device = 0
        self.rewind()
        if self.messenger != 0:
            _ = call3(self.instance_call("vkDestroyDebugUtilsMessengerEXT"), self.instance, self.messenger, 0)
        _ = call2(self.instance_call("vkDestroyInstance"), self.instance, 0)
        self.instance = 0
        self._free_host_scratch()
        if self.library != 0:
            _ = dl_close(self.library)
            self.library = 0
            self.proc_instance = 0
            self.proc_device = 0

    # ========================================================================
    # Vulkan 1.3 AAA Hardware Graphics, Bindless Images, Sync2 & Ray Tracing
    # ========================================================================

    def create_buffer_bda(
        mut self,
        size: Int,
        extra_usage: UInt32 = 0,
        host_visible: Bool = True,
        cpu_reads: Bool = False,
    ) -> GpuBuffer:
        """Allocates a buffer with optional `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` + `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`."""
        self.rewind()
        var rounded = max(4096, (size + 4095) & ~4095)
        var use_bda = self.has_bda and ((extra_usage & BUFFER_USAGE_DEVICE_ADDRESS) != 0)
        var info = self.scratch(56)
        put32(info, TYPE_BUFFER_CREATE_INFO)
        put64(info + 24, rounded)
        put32(info + 32, BUFFER_USAGE | extra_usage)
        var slot = self.scratch(8)
        if call4(self.fn_create_buffer, self.device, info, 0, slot) != SUCCESS:
            _ = self.fail("vkCreateBuffer (BDA) failed")
            return GpuBuffer()
        var buffer = get64(slot)
        var requirements = self.scratch(24)
        _ = call3(self.fn_get_buffer_requirements, self.device, buffer, requirements)
        var needed = get64(requirements)
        var allowed = get32(requirements + 16)

        var alloc_flags = 0
        if use_bda:
            alloc_flags = self.scratch(24)
            put32(alloc_flags, TYPE_MEMORY_ALLOCATE_FLAGS_INFO)
            put32(alloc_flags + 16, MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)

        var memory = 0
        var extras = ~(MEMORY_DEVICE_LOCAL | MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT | MEMORY_HOST_CACHED)
        if not host_visible:
            var dev_only = self.memory_type(allowed, MEMORY_DEVICE_LOCAL, extras & ~MEMORY_HOST_VISIBLE & ~MEMORY_HOST_COHERENT)
            if dev_only < 0:
                dev_only = self.memory_type(allowed, MEMORY_DEVICE_LOCAL, 0)
            if dev_only >= 0:
                memory = self.allocation(needed, dev_only, alloc_flags)
        if memory == 0:
            var host = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT
            var mt = self.memory_type(allowed, host | (MEMORY_HOST_CACHED if cpu_reads else MEMORY_DEVICE_LOCAL), extras)
            if mt < 0:
                mt = self.memory_type(allowed, host, extras)
            if mt < 0:
                mt = self.memory_type(allowed, host, 0)
            if mt >= 0:
                memory = self.allocation(needed, mt, alloc_flags)
        if memory == 0:
            _ = call3(self.fn_destroy_buffer, self.device, buffer, 0)
            _ = self.fail("no memory for BDA buffer")
            return GpuBuffer()
        if call4(self.fn_bind_buffer_memory, self.device, buffer, memory, 0) != SUCCESS:
            _ = call3(self.fn_destroy_buffer, self.device, buffer, 0)
            _ = call3(self.fn_free_memory, self.device, memory, 0)
            return GpuBuffer()
        var cpu_addr = 0
        if host_visible:
            if call6(self.fn_map_memory, self.device, memory, 0, WHOLE_SIZE, 0, slot) == SUCCESS:
                cpu_addr = get64(slot)
        var out = GpuBuffer(0, rounded, 0, cpu_addr)
        out.vk_buffer = buffer
        out.vk_memory = memory
        return out^

    def buffer_device_address(mut self, buffer: GpuBuffer) -> UInt64:
        if not self.has_bda or buffer.vk_buffer == 0:
            return 0
        self.rewind()
        var info = self.scratch(24)
        put32(info, TYPE_BUFFER_DEVICE_ADDRESS_INFO)
        put64(info + 16, buffer.vk_buffer)
        var fn_bda = self.device_call("vkGetBufferDeviceAddress")
        if fn_bda == 0:
            return 0
        # vkGetBufferDeviceAddress returns uint64_t in x0/rax!
        comptime CallBDA = def(Int, Int) thin abi("C") -> UInt64
        var slot = fn_bda
        return Pointer(to=slot).unsafe_bitcast[CallBDA]()[unsafe_offset=0](self.device, info)

    def create_image_2d(
        mut self,
        width: Int,
        height: Int,
        format: UInt32,
        usage: UInt32,
        mip_levels: Int = 1,
        cube: Bool = False,
    ) -> VulkanImage:
        """Creates an optimal-tiled 2D or 6-face Cubemap `VkImage` + `VkDeviceMemory` + `VkImageView`."""
        var out = VulkanImage()
        self.rewind()
        var layers = 6 if cube else 1
        var info = self.scratch(88)
        put32(info, TYPE_IMAGE_CREATE_INFO)
        if cube:
            put32(info + 16, IMAGE_CREATE_CUBE_COMPATIBLE)
        put32(info + 20, 1)  # VK_IMAGE_TYPE_2D
        put32(info + 24, format)
        put32(info + 28, UInt32(width))
        put32(info + 32, UInt32(height))
        put32(info + 36, 1)
        put32(info + 40, UInt32(max(1, mip_levels)))
        put32(info + 44, UInt32(layers))
        put32(info + 48, SAMPLE_COUNT_1)
        put32(info + 52, 0)  # VK_IMAGE_TILING_OPTIMAL
        put32(info + 56, usage)
        put32(info + 80, LAYOUT_UNDEFINED)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateImage"), self.device, info, 0, slot), "vkCreateImage"):
            return out^
        var img = get64(slot)
        var req = self.scratch(24)
        _ = call3(self.device_call("vkGetImageMemoryRequirements"), self.device, img, req)
        var needed = get64(req)
        var allowed = get32(req + 16)
        var mt = self.memory_type(allowed, MEMORY_DEVICE_LOCAL, 0)
        if mt < 0:
            mt = self.memory_type(allowed, 0, 0)
        if mt < 0:
            _ = call3(self.device_call("vkDestroyImage"), self.device, img, 0)
            _ = self.fail("no memory type for VkImage")
            return out^
        var mem = self.allocation(needed, mt, 0)
        if mem == 0 or call4(self.device_call("vkBindImageMemory"), self.device, img, mem, 0) != SUCCESS:
            _ = call3(self.device_call("vkDestroyImage"), self.device, img, 0)
            if mem != 0:
                _ = call3(self.fn_free_memory, self.device, mem, 0)
            _ = self.fail("VkImage allocation/bind failed")
            return out^
        var aspect = ASPECT_DEPTH if format == FORMAT_D32_SFLOAT else ASPECT_COLOR
        var view_info = self.scratch(80)
        put32(view_info, TYPE_IMAGE_VIEW_CREATE_INFO)
        put64(view_info + 24, img)
        put32(view_info + 32, IMAGE_VIEW_TYPE_CUBE if cube else IMAGE_VIEW_TYPE_2D)
        put32(view_info + 36, format)
        put32(view_info + 56, aspect)
        put32(view_info + 60, 0)
        put32(view_info + 64, UInt32(max(1, mip_levels)))
        put32(view_info + 68, 0)
        put32(view_info + 72, UInt32(layers))
        if not self.check(call4(self.device_call("vkCreateImageView"), self.device, view_info, 0, slot), "vkCreateImageView"):
            _ = call3(self.device_call("vkDestroyImage"), self.device, img, 0)
            _ = call3(self.fn_free_memory, self.device, mem, 0)
            return out^
        out.image = img
        out.view = get64(slot)
        out.memory = mem
        out.format = format
        out.width = width
        out.height = height
        out.mip_levels = max(1, mip_levels)
        out.layers = layers
        out.layout = LAYOUT_UNDEFINED
        return out^

    def destroy_image(mut self, mut img: VulkanImage):
        if img.view != 0:
            self.rewind()
            _ = call3(self.device_call("vkDestroyImageView"), self.device, img.view, 0)
            img.view = 0
        if img.image != 0:
            self.rewind()
            _ = call3(self.device_call("vkDestroyImage"), self.device, img.image, 0)
            img.image = 0
        if img.memory != 0:
            _ = call3(self.fn_free_memory, self.device, img.memory, 0)
            img.memory = 0

    def create_sampler(
        mut self,
        linear: Bool = True,
        max_anisotropy: Float32 = 16.0,
        clamp_to_edge: Bool = False,
        max_lod: Float32 = 16.0,
    ) -> Int:
        self.rewind()
        var info = self.scratch(80)
        put32(info, TYPE_SAMPLER_CREATE_INFO)
        var flt = FILTER_LINEAR if linear else FILTER_NEAREST
        put32(info + 20, flt)
        put32(info + 24, flt)
        put32(info + 28, SAMPLER_MIPMAP_LINEAR if linear else 0)
        var addr = ADDRESS_MODE_CLAMP_TO_EDGE if clamp_to_edge else ADDRESS_MODE_REPEAT
        put32(info + 32, addr)
        put32(info + 36, addr)
        put32(info + 40, addr)
        if linear and max_anisotropy > Float32(1.0):
            put32(info + 48, 1)
            put_float(info + 52, max_anisotropy)
        put_float(info + 64, Float32(0.0))
        put_float(info + 68, max_lod)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateSampler"), self.device, info, 0, slot), "vkCreateSampler"):
            return 0
        return get64(slot)

    def destroy_sampler(mut self, sampler: Int):
        if sampler != 0:
            self.rewind()
            _ = call3(self.device_call("vkDestroySampler"), self.device, sampler, 0)

    def transition_image(
        mut self,
        frame: Int,
        mut img: VulkanImage,
        new_layout: UInt32,
        src_stage: Int,
        src_access: UInt32,
        dst_stage: Int,
        dst_access: UInt32,
    ):
        var aspect = ASPECT_DEPTH if img.format == FORMAT_D32_SFLOAT else ASPECT_COLOR
        self.rewind()
        if self.has_sync2:
            var imb2 = self.scratch(96)
            put32(imb2, TYPE_IMAGE_MEMORY_BARRIER_2)
            put64(imb2 + 16, src_stage)
            put64(imb2 + 24, Int(src_access))
            put64(imb2 + 32, dst_stage)
            put64(imb2 + 40, Int(dst_access))
            put32(imb2 + 48, img.layout)
            put32(imb2 + 52, new_layout)
            put32(imb2 + 56, 0xFFFFFFFF)
            put32(imb2 + 60, 0xFFFFFFFF)
            put64(imb2 + 64, img.image)
            put32(imb2 + 72, aspect)
            put32(imb2 + 76, 0)
            put32(imb2 + 80, UInt32(img.mip_levels))
            put32(imb2 + 84, 0)
            put32(imb2 + 88, UInt32(img.layers))
            var dep = self.scratch(64)
            put32(dep, TYPE_DEPENDENCY_INFO)
            put32(dep + 48, 1)
            put64(dep + 56, imb2)
            _ = call2(self.device_call("vkCmdPipelineBarrier2"), self.command_buffers[frame], dep)
        else:
            var barrier = self.scratch(72)
            put32(barrier, TYPE_IMAGE_MEMORY_BARRIER)
            put32(barrier + 16, src_access)
            put32(barrier + 20, dst_access)
            put32(barrier + 24, img.layout)
            put32(barrier + 28, new_layout)
            put32(barrier + 32, 0xFFFFFFFF)
            put32(barrier + 36, 0xFFFFFFFF)
            put64(barrier + 40, img.image)
            put32(barrier + 48, aspect)
            put32(barrier + 52, 0)
            put32(barrier + 56, UInt32(img.mip_levels))
            put32(barrier + 60, 0)
            put32(barrier + 64, UInt32(img.layers))
            _ = call10(self.fn_pipeline_barrier, self.command_buffers[frame], src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barrier)
        img.layout = new_layout

    def memory_barrier2(mut self, frame: Int, src_stage: Int, src_access: Int, dst_stage: Int, dst_access: Int):
        if self.has_sync2:
            self.rewind()
            var mb2 = self.scratch(48)
            put32(mb2, TYPE_MEMORY_BARRIER_2)
            put64(mb2 + 16, src_stage)
            put64(mb2 + 24, src_access)
            put64(mb2 + 32, dst_stage)
            put64(mb2 + 40, dst_access)
            var dep = self.scratch(64)
            put32(dep, TYPE_DEPENDENCY_INFO)
            put32(dep + 20, 1)
            put64(dep + 24, mb2)
            _ = call2(self.device_call("vkCmdPipelineBarrier2"), self.command_buffers[frame], dep)
        else:
            self.barrier(frame, src_stage, UInt32(src_access), dst_stage, UInt32(dst_access))

    def upload_image_mip(
        mut self,
        frame: Int,
        img: VulkanImage,
        staging: GpuBuffer,
        buf_offset: Int,
        mip_level: Int,
        mip_w: Int,
        mip_h: Int,
        base_layer: Int = 0,
        layer_count: Int = 1,
    ):
        self.rewind()
        var region = self.scratch(56)
        put64(region, buf_offset)
        put32(region + 16, ASPECT_COLOR)
        put32(region + 20, UInt32(mip_level))
        put32(region + 24, UInt32(base_layer))
        put32(region + 28, UInt32(layer_count))
        put32(region + 44, UInt32(max(1, mip_w)))
        put32(region + 48, UInt32(max(1, mip_h)))
        put32(region + 52, 1)
        _ = call6(
            self.device_call("vkCmdCopyBufferToImage"),
            self.command_buffers[frame],
            staging.vk_buffer,
            img.image,
            Int(LAYOUT_TRANSFER_TARGET),
            1,
            region,
        )

    def upload_image_subregion(
        mut self,
        frame: Int,
        img: VulkanImage,
        staging: GpuBuffer,
        buf_offset: Int,
        offset_x: Int,
        offset_y: Int,
        width: Int,
        height: Int,
        mip_level: Int = 0,
        base_layer: Int = 0,
        layer_count: Int = 1,
    ):
        self.rewind()
        var region = self.scratch(56)
        put64(region, buf_offset)
        put32(region + 8, UInt32(width))
        put32(region + 12, UInt32(height))
        put32(region + 16, ASPECT_COLOR)
        put32(region + 20, UInt32(mip_level))
        put32(region + 24, UInt32(base_layer))
        put32(region + 28, UInt32(layer_count))
        put32(region + 32, UInt32(offset_x))
        put32(region + 36, UInt32(offset_y))
        put32(region + 40, 0)
        put32(region + 44, UInt32(max(1, width)))
        put32(region + 48, UInt32(max(1, height)))
        put32(region + 52, 1)
        _ = call6(
            self.device_call("vkCmdCopyBufferToImage"),
            self.command_buffers[frame],
            staging.vk_buffer,
            img.image,
            Int(LAYOUT_TRANSFER_TARGET),
            1,
            region,
        )

    def copy_image_to_buffer(
        mut self,
        frame: Int,
        img: VulkanImage,
        dst_buf: GpuBuffer,
        width: Int,
        height: Int,
        dst_offset: Int = 0,
    ):
        var aspect = ASPECT_DEPTH if img.format == FORMAT_D32_SFLOAT else ASPECT_COLOR
        self.rewind()
        var region = self.scratch(56)
        put64(region, dst_offset)
        put32(region + 16, aspect)
        put32(region + 20, 0)
        put32(region + 24, 0)
        put32(region + 28, 1)
        put32(region + 44, UInt32(width))
        put32(region + 48, UInt32(height))
        put32(region + 52, 1)
        _ = call6(
            self.device_call("vkCmdCopyImageToBuffer"),
            self.command_buffers[frame],
            img.image,
            Int(LAYOUT_TRANSFER_SRC),
            dst_buf.vk_buffer,
            1,
            region,
        )

    def create_aaa_layouts(
        mut self,
        storage_count: Int,
        single_sampler_count: Int,
        max_bindless_textures: Int,
        has_tlas: Bool,
        push_bytes: Int,
    ) -> Tuple[Int, Int]:
        """Builds a unified DescriptorSetLayout + PipelineLayout for AAA Graphics, Bindless & Ray-Query passes:
        - bindings `0 .. storage_count - 1`: Storage Buffers (`Vertex | Fragment | Compute`)
        - bindings `storage_count .. storage_count + single_sampler_count - 1`: Combined Image Samplers (VisBuffer, Depth, VSM, Cubemap, LUT)
        - binding `storage_count + single_sampler_count`: Bindless `sampler2D[max_bindless_textures]` (`PARTIALLY_BOUND | UPDATE_AFTER_BIND`)
        - binding `storage_count + single_sampler_count + 1` (if `has_tlas`): `VkAccelerationStructureKHR`
        """
        self.rewind()
        var total_bindings = storage_count + single_sampler_count + (1 if max_bindless_textures > 0 else 0) + (1 if has_tlas else 0)
        var table = self.scratch(total_bindings * 24)
        var flags_arr = self.scratch(total_bindings * 4)
        var all_stages = STAGE_VERTEX_SHADER | STAGE_FRAGMENT_SHADER | STAGE_COMPUTE_SHADER
        var frag_comp = STAGE_FRAGMENT_SHADER | STAGE_COMPUTE_SHADER

        var b = 0
        for _ in range(storage_count):
            var e = table + b * 24
            put32(e, UInt32(b))
            put32(e + 4, DESCRIPTOR_STORAGE_BUFFER)
            put32(e + 8, 1)
            put32(e + 12, all_stages)
            put32(flags_arr + b * 4, DESCRIPTOR_BINDING_PARTIALLY_BOUND)
            b += 1

        for _ in range(single_sampler_count):
            var e = table + b * 24
            put32(e, UInt32(b))
            put32(e + 4, DESCRIPTOR_COMBINED_IMAGE_SAMPLER)
            put32(e + 8, 1)
            put32(e + 12, frag_comp)
            put32(flags_arr + b * 4, DESCRIPTOR_BINDING_PARTIALLY_BOUND | DESCRIPTOR_BINDING_UPDATE_AFTER_BIND)
            b += 1

        if max_bindless_textures > 0:
            var e = table + b * 24
            put32(e, UInt32(b))
            put32(e + 4, DESCRIPTOR_COMBINED_IMAGE_SAMPLER)
            put32(e + 8, UInt32(max_bindless_textures))
            put32(e + 12, frag_comp)
            put32(flags_arr + b * 4, DESCRIPTOR_BINDING_PARTIALLY_BOUND | DESCRIPTOR_BINDING_UPDATE_AFTER_BIND)
            b += 1

        if has_tlas:
            var e = table + b * 24
            put32(e, UInt32(b))
            put32(e + 4, DESCRIPTOR_ACCELERATION_STRUCTURE_KHR)
            put32(e + 8, 1)
            put32(e + 12, frag_comp)
            put32(flags_arr + b * 4, DESCRIPTOR_BINDING_PARTIALLY_BOUND)
            b += 1

        var bind_flags_info = self.scratch(32)
        put32(bind_flags_info, TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO)
        put32(bind_flags_info + 16, UInt32(total_bindings))
        put64(bind_flags_info + 24, flags_arr)

        var dsl_info = self.scratch(32)
        put32(dsl_info, TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
        put64(dsl_info + 8, bind_flags_info)
        put32(dsl_info + 16, DESCRIPTOR_LAYOUT_UPDATE_AFTER_BIND_POOL)
        put32(dsl_info + 20, UInt32(total_bindings))
        put64(dsl_info + 24, table)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateDescriptorSetLayout"), self.device, dsl_info, 0, slot), "vkCreateDescriptorSetLayout(AAA)"):
            return (0, 0)
        var dsl = get64(slot)

        var range_info = self.scratch(12)
        put32(range_info, all_stages)
        put32(range_info + 8, UInt32(push_bytes))
        var layouts = self.scratch(8)
        put64(layouts, dsl)
        var pl_info = self.scratch(48)
        put32(pl_info, TYPE_PIPELINE_LAYOUT_CREATE_INFO)
        put32(pl_info + 20, 1)
        put64(pl_info + 24, layouts)
        put32(pl_info + 32, UInt32(1) if push_bytes > 0 else UInt32(0))
        put64(pl_info + 40, range_info if push_bytes > 0 else 0)
        if not self.check(call4(self.device_call("vkCreatePipelineLayout"), self.device, pl_info, 0, slot), "vkCreatePipelineLayout(AAA)"):
            _ = call3(self.device_call("vkDestroyDescriptorSetLayout"), self.device, dsl, 0)
            return (0, 0)
        return (dsl, get64(slot))

    def allocate_aaa_descriptor_set(
        mut self,
        dsl: Int,
        storage_count: Int,
        single_sampler_count: Int,
        max_bindless_textures: Int,
        has_tlas: Bool,
    ) -> Tuple[Int, Int]:
        self.rewind()
        var pool_types = 2 + (1 if has_tlas else 0)
        var sizes = self.scratch(pool_types * 8)
        put32(sizes, DESCRIPTOR_STORAGE_BUFFER)
        put32(sizes + 4, UInt32(max(1, storage_count)))
        put32(sizes + 8, DESCRIPTOR_COMBINED_IMAGE_SAMPLER)
        put32(sizes + 12, UInt32(max(1, single_sampler_count + max_bindless_textures)))
        if has_tlas:
            put32(sizes + 16, DESCRIPTOR_ACCELERATION_STRUCTURE_KHR)
            put32(sizes + 20, 1)

        var pool_info = self.scratch(40)
        put32(pool_info, TYPE_DESCRIPTOR_POOL_CREATE_INFO)
        put32(pool_info + 16, DESCRIPTOR_POOL_UPDATE_AFTER_BIND)
        put32(pool_info + 20, 1)
        put32(pool_info + 24, UInt32(pool_types))
        put64(pool_info + 32, sizes)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateDescriptorPool"), self.device, pool_info, 0, slot), "vkCreateDescriptorPool(AAA)"):
            return (0, 0)
        var pool = get64(slot)

        var layouts = self.scratch(8)
        put64(layouts, dsl)
        var alloc_info = self.scratch(40)
        put32(alloc_info, TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
        put64(alloc_info + 16, pool)
        put32(alloc_info + 24, 1)
        put64(alloc_info + 32, layouts)
        if not self.check(call3(self.device_call("vkAllocateDescriptorSets"), self.device, alloc_info, slot), "vkAllocateDescriptorSets(AAA)"):
            _ = call3(self.device_call("vkDestroyDescriptorPool"), self.device, pool, 0)
            return (0, 0)
        return (pool, get64(slot))

    def allocate_aaa_descriptor_sets(
        mut self,
        dsl: Int,
        count: Int,
        storage_count: Int,
        single_sampler_count: Int,
        max_bindless_textures: Int,
        has_tlas: Bool,
    ) -> Tuple[Int, List[Int]]:
        self.rewind()
        var pool_types = 2 + (1 if has_tlas else 0)
        var sizes = self.scratch(pool_types * 8)
        put32(sizes, DESCRIPTOR_STORAGE_BUFFER)
        put32(sizes + 4, UInt32(max(1, storage_count * count)))
        put32(sizes + 8, DESCRIPTOR_COMBINED_IMAGE_SAMPLER)
        put32(sizes + 12, UInt32(max(1, (single_sampler_count + max_bindless_textures) * count)))
        if has_tlas:
            put32(sizes + 16, DESCRIPTOR_ACCELERATION_STRUCTURE_KHR)
            put32(sizes + 20, UInt32(count))

        var pool_info = self.scratch(40)
        put32(pool_info, TYPE_DESCRIPTOR_POOL_CREATE_INFO)
        put32(pool_info + 16, DESCRIPTOR_POOL_UPDATE_AFTER_BIND)
        put32(pool_info + 20, UInt32(count))
        put32(pool_info + 24, UInt32(pool_types))
        put64(pool_info + 32, sizes)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateDescriptorPool"), self.device, pool_info, 0, slot), "vkCreateDescriptorPool(AAA)"):
            return (0, List[Int]())
        var pool = get64(slot)

        var layouts = self.scratch(count * 8)
        for i in range(count):
            put64(layouts + i * 8, dsl)
        var alloc_info = self.scratch(40)
        put32(alloc_info, TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
        put64(alloc_info + 16, pool)
        put32(alloc_info + 24, UInt32(count))
        put64(alloc_info + 32, layouts)
        var set_slots = self.scratch(count * 8)
        if not self.check(call3(self.device_call("vkAllocateDescriptorSets"), self.device, alloc_info, set_slots), "vkAllocateDescriptorSets(AAA)"):
            _ = call3(self.device_call("vkDestroyDescriptorPool"), self.device, pool, 0)
            return (0, List[Int]())

        var out_sets = List[Int]()
        for i in range(count):
            out_sets.append(get64(set_slots + i * 8))
        return (pool, out_sets^)

    def write_aaa_texture(
        mut self,
        descriptor_set: Int,
        binding: Int,
        array_element: Int,
        sampler: Int,
        image_view: Int,
        image_layout: UInt32 = LAYOUT_SHADER_READ_ONLY,
    ):
        self.rewind()
        var img_info = self.scratch(24)
        put64(img_info, sampler)
        put64(img_info + 8, image_view)
        put32(img_info + 16, image_layout)
        var write = self.scratch(64)
        put32(write, TYPE_WRITE_DESCRIPTOR_SET)
        put64(write + 16, descriptor_set)
        put32(write + 24, UInt32(binding))
        put32(write + 28, UInt32(array_element))
        put32(write + 32, 1)
        put32(write + 36, DESCRIPTOR_COMBINED_IMAGE_SAMPLER)
        put64(write + 40, img_info)
        _ = call5(self.fn_update_descriptors, self.device, 1, write, 0, 0)

    def write_aaa_tlas(mut self, descriptor_set: Int, binding: Int, tlas_handle: Int):
        self.rewind()
        var handles = self.scratch(8)
        put64(handles, tlas_handle)
        var as_write = self.scratch(32)
        put32(as_write, TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
        put32(as_write + 16, 1)
        put64(as_write + 24, handles)
        var write = self.scratch(64)
        put32(write, TYPE_WRITE_DESCRIPTOR_SET)
        put64(write + 8, as_write)
        put64(write + 16, descriptor_set)
        put32(write + 24, UInt32(binding))
        put32(write + 32, 1)
        put32(write + 36, DESCRIPTOR_ACCELERATION_STRUCTURE_KHR)
        _ = call5(self.fn_update_descriptors, self.device, 1, write, 0, 0)

    def create_aaa_compute_pipeline(mut self, pipeline_layout: Int, code: List[UInt32]) -> Int:
        self.rewind()
        var module_info = self.scratch(40)
        put32(module_info, TYPE_SHADER_MODULE_CREATE_INFO)
        put64(module_info + 24, len(code) * 4)
        put64(module_info + 32, Int(code.unsafe_ptr()))
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateShaderModule"), self.device, module_info, 0, slot), "vkCreateShaderModule(AAA-Comp)"):
            return 0
        var module = get64(slot)
        var info = self.scratch(96)
        put32(info, TYPE_COMPUTE_PIPELINE_CREATE_INFO)
        put32(info + 24, TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
        put32(info + 44, STAGE_COMPUTE_SHADER)
        put64(info + 48, module)
        put64(info + 56, self.text("main"))
        put64(info + 72, pipeline_layout)
        put32(info + 88, 0xFFFFFFFF)
        var result = call6(self.device_call("vkCreateComputePipelines"), self.device, self.pipeline_cache, 1, info, 0, slot)
        var pipeline = get64(slot)
        _ = call3(self.device_call("vkDestroyShaderModule"), self.device, module, 0)
        if not self.check(result, "vkCreateComputePipelines(AAA)"):
            return 0
        return pipeline

    def create_graphics_pipeline(
        mut self,
        pipeline_layout: Int,
        vert_spv: List[UInt32],
        frag_spv: List[UInt32],
        has_color: Bool,
        color_format: UInt32,
        depth_format: UInt32,
        depth_test: Bool = True,
        depth_write: Bool = True,
        depth_compare: UInt32 = COMPARE_GREATER_OR_EQUAL,
        cull_mode: UInt32 = CULL_MODE_NONE,
        front_face: UInt32 = FRONT_FACE_CCW,
    ) -> Int:
        """Creates a Vulkan 1.3 `VK_KHR_dynamic_rendering` hardware graphics pipeline (Vertex + optional Fragment)."""
        self.rewind()
        var slot = self.scratch(8)
        var v_mod_info = self.scratch(40)
        put32(v_mod_info, TYPE_SHADER_MODULE_CREATE_INFO)
        put64(v_mod_info + 24, len(vert_spv) * 4)
        put64(v_mod_info + 32, Int(vert_spv.unsafe_ptr()))
        if not self.check(call4(self.device_call("vkCreateShaderModule"), self.device, v_mod_info, 0, slot), "vkCreateShaderModule(Vert)"):
            return 0
        var v_mod = get64(slot)

        var f_mod = 0
        var use_frag = len(frag_spv) > 0
        if use_frag:
            var f_mod_info = self.scratch(40)
            put32(f_mod_info, TYPE_SHADER_MODULE_CREATE_INFO)
            put64(f_mod_info + 24, len(frag_spv) * 4)
            put64(f_mod_info + 32, Int(frag_spv.unsafe_ptr()))
            if not self.check(call4(self.device_call("vkCreateShaderModule"), self.device, f_mod_info, 0, slot), "vkCreateShaderModule(Frag)"):
                _ = call3(self.device_call("vkDestroyShaderModule"), self.device, v_mod, 0)
                return 0
            f_mod = get64(slot)

        var entry_main = self.text("main")
        var stage_count = 2 if use_frag else 1
        var stages = self.scratch(stage_count * 48)
        put32(stages, TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
        put32(stages + 20, STAGE_VERTEX_SHADER)
        put64(stages + 24, v_mod)
        put64(stages + 32, entry_main)
        if use_frag:
            put32(stages + 48, TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
            put32(stages + 48 + 20, STAGE_FRAGMENT_SHADER)
            put64(stages + 48 + 24, f_mod)
            put64(stages + 48 + 32, entry_main)

        var vi = self.scratch(48)
        put32(vi, TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO)

        var ia = self.scratch(32)
        put32(ia, TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO)
        put32(ia + 20, TOPOLOGY_TRIANGLE_LIST)

        var vp = self.scratch(48)
        put32(vp, TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO)
        put32(vp + 20, 1)
        put32(vp + 32, 1)

        var rs = self.scratch(64)
        put32(rs, TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO)
        put32(rs + 32, cull_mode)
        put32(rs + 36, front_face)
        put_float(rs + 56, Float32(1.0))

        var ms = self.scratch(48)
        put32(ms, TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO)
        put32(ms + 20, SAMPLE_COUNT_1)

        var ds = self.scratch(104)
        put32(ds, TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO)
        put32(ds + 20, UInt32(1) if depth_test else UInt32(0))
        put32(ds + 24, UInt32(1) if depth_write else UInt32(0))
        put32(ds + 28, depth_compare)

        var cb_att = self.scratch(32)
        put32(cb_att + 28, 0xF)  # RGBA write mask

        var cb = self.scratch(56)
        put32(cb, TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO)
        put32(cb + 28, UInt32(1) if has_color else UInt32(0))
        put64(cb + 32, cb_att if has_color else 0)

        var dyn_states = self.scratch(8)
        put32(dyn_states, DYNAMIC_STATE_VIEWPORT)
        put32(dyn_states + 4, DYNAMIC_STATE_SCISSOR)
        var dyn = self.scratch(32)
        put32(dyn, TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO)
        put32(dyn + 20, 2)
        put64(dyn + 24, dyn_states)

        var color_fmt_ptr = self.scratch(8)
        put32(color_fmt_ptr, color_format)
        var pr_info = self.scratch(40)
        put32(pr_info, TYPE_PIPELINE_RENDERING_CREATE_INFO)
        put32(pr_info + 20, UInt32(1) if has_color else UInt32(0))
        put64(pr_info + 24, color_fmt_ptr if has_color else 0)
        put32(pr_info + 32, depth_format)

        var gp = self.scratch(144)
        put32(gp, TYPE_GRAPHICS_PIPELINE_CREATE_INFO)
        put64(gp + 8, pr_info)
        put32(gp + 20, UInt32(stage_count))
        put64(gp + 24, stages)
        put64(gp + 32, vi)
        put64(gp + 40, ia)
        put64(gp + 56, vp)
        put64(gp + 64, rs)
        put64(gp + 72, ms)
        put64(gp + 80, ds)
        put64(gp + 88, cb)
        put64(gp + 96, dyn)
        put64(gp + 104, pipeline_layout)
        put32(gp + 128, 0xFFFFFFFF)

        var res = call6(self.device_call("vkCreateGraphicsPipelines"), self.device, self.pipeline_cache, 1, gp, 0, slot)
        var pipe = get64(slot)
        _ = call3(self.device_call("vkDestroyShaderModule"), self.device, v_mod, 0)
        if f_mod != 0:
            _ = call3(self.device_call("vkDestroyShaderModule"), self.device, f_mod, 0)
        if not self.check(res, "vkCreateGraphicsPipelines"):
            return 0
        return pipe

    def destroy_pipeline(mut self, pipeline: Int):
        if pipeline != 0:
            self.rewind()
            _ = call3(self.device_call("vkDestroyPipeline"), self.device, pipeline, 0)

    def init_pipeline_cache(mut self, path: String = "") -> Bool:
        if self.device == 0 or self.pipeline_cache != 0:
            return False
        self.rewind()
        var info = self.scratch(40)
        put32(info, TYPE_PIPELINE_CACHE_CREATE_INFO)
        put64(info + 8, 0)
        put32(info + 16, 0)
        var has_data = False
        var data = List[UInt8]()
        if path.byte_length() > 0:
            data = read_bytes(path)
            if len(data) >= 32:
                has_data = True
        if has_data:
            put64(info + 24, len(data))
            put64(info + 32, Int(data.unsafe_ptr()))
        else:
            put64(info + 24, 0)
            put64(info + 32, 0)
        var slot = self.scratch(8)
        var res = call4(self.device_call("vkCreatePipelineCache"), self.device, info, 0, slot)
        if not self.check(res, "vkCreatePipelineCache"):
            return False
        self.pipeline_cache = get64(slot)
        return True

    def save_pipeline_cache(mut self, path: String) -> Bool:
        if self.device == 0 or self.pipeline_cache == 0 or path.byte_length() == 0:
            return False
        self.rewind()
        var size_slot = self.scratch(8)
        put64(size_slot, 0)
        var res = call4(self.device_call("vkGetPipelineCacheData"), self.device, self.pipeline_cache, size_slot, 0)
        if not self.check(res, "vkGetPipelineCacheData(size)"):
            return False
        var sz = get64(size_slot)
        if sz <= 0:
            return False
        var data = List[UInt8](length=sz, fill=0)
        res = call4(self.device_call("vkGetPipelineCacheData"), self.device, self.pipeline_cache, size_slot, Int(data.unsafe_ptr()))
        if not self.check(res, "vkGetPipelineCacheData(data)"):
            return False
        return write_bytes(path, data)

    def destroy_aaa_descriptors(mut self, pool: Int, dsl: Int, pipeline_layout: Int):
        self.rewind()
        if pool != 0:
            _ = call3(self.device_call("vkDestroyDescriptorPool"), self.device, pool, 0)
        if pipeline_layout != 0:
            _ = call3(self.device_call("vkDestroyPipelineLayout"), self.device, pipeline_layout, 0)
        if dsl != 0:
            _ = call3(self.device_call("vkDestroyDescriptorSetLayout"), self.device, dsl, 0)

    def begin_rendering(
        mut self,
        frame: Int,
        rx: Int,
        ry: Int,
        rw: Int,
        rh: Int,
        has_color: Bool,
        color_view: Int,
        clear_color: Bool,
        clear_u0: UInt32,
        clear_u1: UInt32,
        has_depth: Bool,
        depth_view: Int,
        clear_depth: Bool,
        depth_clear_val: Float32 = 0.0,
    ):
        self.rewind()
        var color_att = self.scratch(72)
        if has_color:
            put32(color_att, TYPE_RENDERING_ATTACHMENT_INFO)
            put64(color_att + 16, color_view)
            put32(color_att + 24, LAYOUT_COLOR_ATTACHMENT)
            put32(color_att + 44, LOAD_OP_CLEAR if clear_color else LOAD_OP_LOAD)
            put32(color_att + 48, STORE_OP_STORE)
            put32(color_att + 52, clear_u0)
            put32(color_att + 56, clear_u1)

        var depth_att = self.scratch(72)
        if has_depth:
            put32(depth_att, TYPE_RENDERING_ATTACHMENT_INFO)
            put64(depth_att + 16, depth_view)
            put32(depth_att + 24, LAYOUT_DEPTH_ATTACHMENT)
            put32(depth_att + 44, LOAD_OP_CLEAR if clear_depth else LOAD_OP_LOAD)
            put32(depth_att + 48, STORE_OP_STORE)
            put_float(depth_att + 52, depth_clear_val)

        var r_info = self.scratch(72)
        put32(r_info, TYPE_RENDERING_INFO)
        put32(r_info + 20, UInt32(rx))
        put32(r_info + 24, UInt32(ry))
        put32(r_info + 28, UInt32(rw))
        put32(r_info + 32, UInt32(rh))
        put32(r_info + 36, 1)  # layerCount = 1
        put32(r_info + 44, UInt32(1) if has_color else UInt32(0))
        put64(r_info + 48, color_att if has_color else 0)
        put64(r_info + 56, depth_att if has_depth else 0)
        _ = call2(self.device_call("vkCmdBeginRendering"), self.command_buffers[frame], r_info)

    def end_rendering(mut self, frame: Int):
        self.rewind()
        _ = call1(self.device_call("vkCmdEndRendering"), self.command_buffers[frame])

    def set_viewport_and_scissor(mut self, frame: Int, x: Int, y: Int, w: Int, h: Int, flip_y: Bool = True):
        self.rewind()
        var vp = self.scratch(24)
        put_float(vp + 0, Float32(x))
        put_float(vp + 4, Float32(y + h) if flip_y else Float32(y))
        put_float(vp + 8, Float32(w))
        put_float(vp + 12, Float32(-h) if flip_y else Float32(h))
        put_float(vp + 16, Float32(0.0))
        put_float(vp + 20, Float32(1.0))
        _ = call4(self.device_call("vkCmdSetViewport"), self.command_buffers[frame], 0, 1, vp)

        var sc = self.scratch(16)
        put32(sc + 0, UInt32(x))
        put32(sc + 4, UInt32(y))
        put32(sc + 8, UInt32(w))
        put32(sc + 12, UInt32(h))
        _ = call4(self.device_call("vkCmdSetScissor"), self.command_buffers[frame], 0, 1, sc)

    def bind_graphics_pipeline(
        mut self,
        frame: Int,
        pipeline: Int,
        pipeline_layout: Int,
        descriptor_set: Int,
        push: List[UInt32],
    ):
        self.rewind()
        var commands = self.command_buffers[frame]
        var sets = self.scratch(8)
        put64(sets, descriptor_set)
        var all_stages = Int(STAGE_VERTEX_SHADER | STAGE_FRAGMENT_SHADER | STAGE_COMPUTE_SHADER)
        _ = call3(self.fn_bind_pipeline, commands, BIND_POINT_GRAPHICS, pipeline)
        _ = call8(self.fn_bind_descriptor_sets, commands, BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, sets, 0, 0)
        if len(push) > 0:
            _ = call6(self.fn_push_constants, commands, pipeline_layout, all_stages, 0, len(push) * 4, Int(push.unsafe_ptr()))

    def dispatch_aaa(
        mut self,
        frame: Int,
        pipeline: Int,
        pipeline_layout: Int,
        descriptor_set: Int,
        push: List[UInt32],
        groups_x: Int,
        groups_y: Int,
        groups_z: Int = 1,
    ):
        self.rewind()
        var commands = self.command_buffers[frame]
        var sets = self.scratch(8)
        put64(sets, descriptor_set)
        var all_stages = Int(STAGE_VERTEX_SHADER | STAGE_FRAGMENT_SHADER | STAGE_COMPUTE_SHADER)
        _ = call3(self.fn_bind_pipeline, commands, BIND_POINT_COMPUTE, pipeline)
        _ = call8(self.fn_bind_descriptor_sets, commands, BIND_POINT_COMPUTE, pipeline_layout, 0, 1, sets, 0, 0)
        if len(push) > 0:
            _ = call6(self.fn_push_constants, commands, pipeline_layout, all_stages, 0, len(push) * 4, Int(push.unsafe_ptr()))
        _ = call4(self.fn_dispatch, commands, max(1, groups_x), max(1, groups_y), max(1, groups_z))

    def draw(mut self, frame: Int, vertex_count: Int, instance_count: Int = 1, first_vertex: Int = 0, first_instance: Int = 0):
        self.rewind()
        _ = call5(
            self.device_call("vkCmdDraw"),
            self.command_buffers[frame],
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        )

    def draw_indirect(
        mut self,
        frame: Int,
        indirect_buf: GpuBuffer,
        indirect_offset: Int,
        draw_count: Int,
        stride: Int = 16,
    ):
        if draw_count <= 0:
            return
        self.rewind()
        _ = call5(
            self.device_call("vkCmdDrawIndirect"),
            self.command_buffers[frame],
            indirect_buf.vk_buffer,
            indirect_offset,
            draw_count,
            stride,
        )

    def bind_index_buffer(
        mut self,
        frame: Int,
        buffer: GpuBuffer,
        offset: Int = 0,
        index_type: UInt32 = INDEX_TYPE_UINT32,
    ):
        """Binds the GPU index buffer for hardware indexed rasterization."""
        self.rewind()
        _ = call4(
            self.device_call("vkCmdBindIndexBuffer"),
            self.command_buffers[frame],
            buffer.vk_buffer,
            offset,
            Int(index_type),
        )

    def draw_indexed_indirect(
        mut self,
        frame: Int,
        indirect_buf: GpuBuffer,
        indirect_offset: Int,
        draw_count: Int,
        stride: Int = 20,
    ):
        """Issues vkCmdDrawIndexedIndirect with 20-byte VkDrawIndexedIndirectCommand records."""
        if draw_count <= 0:
            return
        self.rewind()
        _ = call5(
            self.device_call("vkCmdDrawIndexedIndirect"),
            self.command_buffers[frame],
            indirect_buf.vk_buffer,
            indirect_offset,
            draw_count,
            stride,
        )

    def draw_indexed_indirect_count(
        mut self,
        frame: Int,
        indirect_buf: GpuBuffer,
        indirect_offset: Int,
        count_buf: GpuBuffer,
        count_offset: Int,
        max_draw_count: Int,
        stride: Int = 20,
    ):
        """Issues vkCmdDrawIndexedIndirectCount with hardware count buffer, falling back to vkCmdDrawIndexedIndirect."""
        self.rewind()
        if self.has_draw_indirect_count:
            _ = call7(
                self.device_call("vkCmdDrawIndexedIndirectCount"),
                self.command_buffers[frame],
                indirect_buf.vk_buffer,
                indirect_offset,
                count_buf.vk_buffer,
                count_offset,
                max_draw_count,
                stride,
            )
        else:
            _ = call5(
                self.device_call("vkCmdDrawIndexedIndirect"),
                self.command_buffers[frame],
                indirect_buf.vk_buffer,
                indirect_offset,
                max_draw_count,
                stride,
            )

    def copy_buffer(
        mut self,
        frame: Int,
        src: GpuBuffer,
        src_offset: Int,
        dst: GpuBuffer,
        dst_offset: Int,
        byte_count: Int,
    ):
        if byte_count <= 0 or src.vk_buffer == 0 or dst.vk_buffer == 0:
            return
        self.rewind()
        var region = self.scratch(24)
        put64(region + 0, src_offset)
        put64(region + 8, dst_offset)
        put64(region + 16, byte_count)
        _ = call5(
            self.device_call("vkCmdCopyBuffer"),
            self.command_buffers[frame],
            src.vk_buffer,
            dst.vk_buffer,
            1,
            region,
        )

    def draw_indirect_count(
        mut self,
        frame: Int,
        indirect_buf: GpuBuffer,
        indirect_offset: Int,
        count_buf: GpuBuffer,
        count_offset: Int,
        max_draw_count: Int,
        stride: Int = 16,
    ):
        self.rewind()
        if self.has_draw_indirect_count:
            _ = call7(
                self.device_call("vkCmdDrawIndirectCount"),
                self.command_buffers[frame],
                indirect_buf.vk_buffer,
                indirect_offset,
                count_buf.vk_buffer,
                count_offset,
                max_draw_count,
                stride,
            )
        else:
            _ = call5(
                self.device_call("vkCmdDrawIndirect"),
                self.command_buffers[frame],
                indirect_buf.vk_buffer,
                indirect_offset,
                max_draw_count,
                stride,
            )

    # Hardware Ray-Tracing Acceleration Structures (BLAS + TLAS)
    def build_blas_triangles(
        mut self,
        frame: Int,
        vertex_addr: UInt64,
        vertex_stride: Int,
        max_vertex: Int,
        index_addr: UInt64,
        triangle_count: Int,
        mut scratch_out: GpuBuffer,
        allow_update: Bool = False,
    ) -> VulkanAccelStruct:
        var out = VulkanAccelStruct()
        if not self.has_ray_query or triangle_count <= 0:
            return out^
        var as_flags = BUILD_AS_PREFER_FAST_TRACE_KHR | (BUILD_AS_ALLOW_UPDATE_KHR if allow_update else UInt32(0))
        self.rewind()
        var geom = self.scratch(96)
        put32(geom, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
        put32(geom + 16, GEOMETRY_TYPE_TRIANGLES_KHR)
        # Embedded VkAccelerationStructureGeometryTrianglesDataKHR at geom + 24
        put32(geom + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR)
        put32(geom + 24 + 16, FORMAT_R32G32B32_SFLOAT)
        put64(geom + 24 + 24, Int(vertex_addr))
        put64(geom + 24 + 32, vertex_stride)
        put32(geom + 24 + 40, UInt32(max_vertex))
        put32(geom + 24 + 44, INDEX_TYPE_UINT32)
        put64(geom + 24 + 48, Int(index_addr))
        put32(geom + 88, GEOMETRY_OPAQUE_BIT_KHR)

        var build_info = self.scratch(80)
        put32(build_info, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
        put32(build_info + 16, AS_TYPE_BOTTOM_LEVEL_KHR)
        put32(build_info + 20, as_flags)
        put32(build_info + 24, BUILD_AS_MODE_BUILD_KHR)
        put32(build_info + 48, 1)
        put64(build_info + 56, geom)

        var prim_cnt = self.scratch(8)
        put32(prim_cnt, UInt32(triangle_count))
        var sizes = self.scratch(40)
        put32(sizes, TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR)
        _ = call5(
            self.device_call("vkGetAccelerationStructureBuildSizesKHR"),
            self.device,
            BUILD_AS_TYPE_DEVICE_KHR,
            build_info,
            prim_cnt,
            sizes,
        )
        var as_bytes = max(4096, get64(sizes + 16))
        var scratch_bytes = max(4096, max(get64(sizes + 24), get64(sizes + 32)) + 256)

        var as_buf = self.create_buffer_bda(
            as_bytes,
            extra_usage=BUFFER_USAGE_AS_STORAGE_KHR | BUFFER_USAGE_DEVICE_ADDRESS,
            host_visible=False,
        )
        if scratch_out.size < scratch_bytes:
            if scratch_out.size > 0:
                self.destroy_buffer(scratch_out)
            scratch_out = self.create_buffer_bda(
                scratch_bytes,
                extra_usage=BUFFER_USAGE_DEVICE_ADDRESS,
                host_visible=False,
            )
        var scratch_addr = (self.buffer_device_address(scratch_out) + 255) & ~UInt64(255)

        self.rewind()
        var create_info = self.scratch(64)
        put32(create_info, TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR)
        put64(create_info + 24, as_buf.vk_buffer)
        put64(create_info + 40, as_bytes)
        put32(create_info + 48, AS_TYPE_BOTTOM_LEVEL_KHR)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateAccelerationStructureKHR"), self.device, create_info, 0, slot), "vkCreateAccelerationStructureKHR(BLAS)"):
            self.destroy_buffer(as_buf)
            return out^
        var handle = get64(slot)

        # Re-populate build_info after create_buffer_bda rewound scratch arena
        self.rewind()
        var geom2 = self.scratch(96)
        put32(geom2, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
        put32(geom2 + 16, GEOMETRY_TYPE_TRIANGLES_KHR)
        put32(geom2 + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR)
        put32(geom2 + 24 + 16, FORMAT_R32G32B32_SFLOAT)
        put64(geom2 + 24 + 24, Int(vertex_addr))
        put64(geom2 + 24 + 32, vertex_stride)
        put32(geom2 + 24 + 40, UInt32(max_vertex))
        put32(geom2 + 24 + 44, INDEX_TYPE_UINT32)
        put64(geom2 + 24 + 48, Int(index_addr))
        put32(geom2 + 88, GEOMETRY_OPAQUE_BIT_KHR)

        var binfo2 = self.scratch(80)
        put32(binfo2, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
        put32(binfo2 + 16, AS_TYPE_BOTTOM_LEVEL_KHR)
        put32(binfo2 + 20, as_flags)
        put32(binfo2 + 24, BUILD_AS_MODE_BUILD_KHR)
        put64(binfo2 + 40, handle)
        put32(binfo2 + 48, 1)
        put64(binfo2 + 56, geom2)
        put64(binfo2 + 72, Int(scratch_addr))

        var range_info = self.scratch(16)
        put32(range_info, UInt32(triangle_count))
        var pp_range = self.scratch(8)
        put64(pp_range, range_info)
        _ = call4(self.device_call("vkCmdBuildAccelerationStructuresKHR"), self.command_buffers[frame], 1, binfo2, pp_range)

        self.rewind()
        var addr_info = self.scratch(24)
        put32(addr_info, TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR)
        put64(addr_info + 16, handle)
        comptime CallASDA = def(Int, Int) thin abi("C") -> UInt64
        var fn_asda = self.device_call("vkGetAccelerationStructureDeviceAddressKHR")
        var as_da = Pointer(to=fn_asda).unsafe_bitcast[CallASDA]()[unsafe_offset=0](self.device, addr_info)

        out.handle = handle
        out.buffer = as_buf^
        out.device_address = as_da
        return out^

    def refit_blas_triangles(
        mut self,
        frame: Int,
        accel: VulkanAccelStruct,
        vertex_addr: UInt64,
        vertex_stride: Int,
        max_vertex: Int,
        index_addr: UInt64,
        triangle_count: Int,
        scratch_buf: GpuBuffer,
    ):
        """Records an in-place `VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR` refit of `accel`
        after GPU Dual-Quaternion skinning deforms `vertex_addr` (`0` host allocations)."""
        if not self.has_ray_query or accel.handle == 0 or triangle_count <= 0 or scratch_buf.size == 0:
            return
        var scratch_addr = (self.buffer_device_address(scratch_buf) + 255) & ~UInt64(255)
        self.rewind()
        var geom = self.scratch(96)
        put32(geom, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
        put32(geom + 16, GEOMETRY_TYPE_TRIANGLES_KHR)
        put32(geom + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR)
        put32(geom + 24 + 16, FORMAT_R32G32B32_SFLOAT)
        put64(geom + 24 + 24, Int(vertex_addr))
        put64(geom + 24 + 32, vertex_stride)
        put32(geom + 24 + 40, UInt32(max_vertex))
        put32(geom + 24 + 44, INDEX_TYPE_UINT32)
        put64(geom + 24 + 48, Int(index_addr))
        put32(geom + 88, GEOMETRY_OPAQUE_BIT_KHR)

        var binfo = self.scratch(80)
        put32(binfo, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
        put32(binfo + 16, AS_TYPE_BOTTOM_LEVEL_KHR)
        put32(binfo + 20, BUILD_AS_PREFER_FAST_TRACE_KHR | BUILD_AS_ALLOW_UPDATE_KHR)
        put32(binfo + 24, BUILD_AS_MODE_UPDATE_KHR)
        put64(binfo + 32, accel.handle)
        put64(binfo + 40, accel.handle)
        put32(binfo + 48, 1)
        put64(binfo + 56, geom)
        put64(binfo + 72, Int(scratch_addr))

        var range_info = self.scratch(16)
        put32(range_info, UInt32(triangle_count))
        var pp_range = self.scratch(8)
        put64(pp_range, range_info)
        _ = call4(self.device_call("vkCmdBuildAccelerationStructuresKHR"), self.command_buffers[frame], 1, binfo, pp_range)

    def build_tlas_instances(
        mut self,
        frame: Int,
        instances_addr: UInt64,
        instance_count: Int,
        mut scratch_out: GpuBuffer,
    ) -> VulkanAccelStruct:
        var out = VulkanAccelStruct()
        if not self.has_ray_query or instance_count < 0:
            return out^
        self.rewind()
        var geom = self.scratch(96)
        put32(geom, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
        put32(geom + 16, GEOMETRY_TYPE_INSTANCES_KHR)
        put32(geom + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR)
        put64(geom + 24 + 24, Int(instances_addr))
        put32(geom + 88, GEOMETRY_OPAQUE_BIT_KHR)

        var build_info = self.scratch(80)
        put32(build_info, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
        put32(build_info + 16, AS_TYPE_TOP_LEVEL_KHR)
        put32(build_info + 20, BUILD_AS_PREFER_FAST_TRACE_KHR)
        put32(build_info + 24, BUILD_AS_MODE_BUILD_KHR)
        put32(build_info + 48, 1)
        put64(build_info + 56, geom)

        var prim_cnt = self.scratch(8)
        put32(prim_cnt, UInt32(instance_count))
        var sizes = self.scratch(40)
        put32(sizes, TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR)
        _ = call5(
            self.device_call("vkGetAccelerationStructureBuildSizesKHR"),
            self.device,
            BUILD_AS_TYPE_DEVICE_KHR,
            build_info,
            prim_cnt,
            sizes,
        )
        var as_bytes = max(4096, get64(sizes + 16))
        var scratch_bytes = max(4096, get64(sizes + 32) + 256)

        var as_buf = self.create_buffer_bda(
            as_bytes,
            extra_usage=BUFFER_USAGE_AS_STORAGE_KHR | BUFFER_USAGE_DEVICE_ADDRESS,
            host_visible=False,
        )
        if scratch_out.size < scratch_bytes:
            if scratch_out.size > 0:
                self.destroy_buffer(scratch_out)
            scratch_out = self.create_buffer_bda(
                scratch_bytes,
                extra_usage=BUFFER_USAGE_DEVICE_ADDRESS,
                host_visible=False,
            )
        var scratch_addr = (self.buffer_device_address(scratch_out) + 255) & ~UInt64(255)

        self.rewind()
        var create_info = self.scratch(64)
        put32(create_info, TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR)
        put64(create_info + 24, as_buf.vk_buffer)
        put64(create_info + 40, as_bytes)
        put32(create_info + 48, AS_TYPE_TOP_LEVEL_KHR)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateAccelerationStructureKHR"), self.device, create_info, 0, slot), "vkCreateAccelerationStructureKHR(TLAS)"):
            self.destroy_buffer(as_buf)
            return out^
        var handle = get64(slot)

        self.rewind()
        var geom2 = self.scratch(96)
        put32(geom2, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
        put32(geom2 + 16, GEOMETRY_TYPE_INSTANCES_KHR)
        put32(geom2 + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR)
        put64(geom2 + 24 + 24, Int(instances_addr))
        put32(geom2 + 88, GEOMETRY_OPAQUE_BIT_KHR)

        var binfo2 = self.scratch(80)
        put32(binfo2, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
        put32(binfo2 + 16, AS_TYPE_TOP_LEVEL_KHR)
        put32(binfo2 + 20, BUILD_AS_PREFER_FAST_TRACE_KHR)
        put32(binfo2 + 24, BUILD_AS_MODE_BUILD_KHR)
        put64(binfo2 + 40, handle)
        put32(binfo2 + 48, 1)
        put64(binfo2 + 56, geom2)
        put64(binfo2 + 72, Int(scratch_addr))

        var range_info = self.scratch(16)
        put32(range_info, UInt32(instance_count))
        var pp_range = self.scratch(8)
        put64(pp_range, range_info)
        _ = call4(self.device_call("vkCmdBuildAccelerationStructuresKHR"), self.command_buffers[frame], 1, binfo2, pp_range)

        out.handle = handle
        out.buffer = as_buf^
        return out^

    def ensure_tlas_capacity(
        mut self,
        mut tlas: VulkanAccelStruct,
        instances_addr: UInt64,
        max_instances_cap: Int,
        mut scratch_out: GpuBuffer,
    ) -> Bool:
        """Ensures `tlas` and `scratch_out` have sufficient capacity for up to `max_instances_cap` instances.
        Returns `True` if a new `VkAccelerationStructureKHR` handle was allocated (requiring descriptor update)."""
        if not self.has_ray_query or max_instances_cap <= 0:
            return False
        self.rewind()
        var geom = self.scratch(96)
        put32(geom, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
        put32(geom + 16, GEOMETRY_TYPE_INSTANCES_KHR)
        put32(geom + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR)
        put64(geom + 24 + 24, Int(instances_addr))
        put32(geom + 88, GEOMETRY_OPAQUE_BIT_KHR)

        var build_info = self.scratch(80)
        put32(build_info, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
        put32(build_info + 16, AS_TYPE_TOP_LEVEL_KHR)
        put32(build_info + 20, BUILD_AS_PREFER_FAST_TRACE_KHR)
        put32(build_info + 24, BUILD_AS_MODE_BUILD_KHR)
        put32(build_info + 48, 1)
        put64(build_info + 56, geom)

        var prim_cnt = self.scratch(8)
        put32(prim_cnt, UInt32(max_instances_cap))
        var sizes = self.scratch(40)
        put32(sizes, TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR)
        _ = call5(
            self.device_call("vkGetAccelerationStructureBuildSizesKHR"),
            self.device,
            BUILD_AS_TYPE_DEVICE_KHR,
            build_info,
            prim_cnt,
            sizes,
        )
        var as_bytes = max(4096, get64(sizes + 16))
        var scratch_bytes = max(4096, get64(sizes + 32) + 256)

        if scratch_out.size < scratch_bytes:
            if scratch_out.size > 0:
                self.destroy_buffer(scratch_out)
            scratch_out = self.create_buffer_bda(
                scratch_bytes,
                extra_usage=BUFFER_USAGE_DEVICE_ADDRESS,
                host_visible=False,
            )

        if tlas.handle != 0 and tlas.buffer.size >= as_bytes:
            return False

        if tlas.handle != 0:
            self.destroy_accel_struct(tlas)

        var as_buf = self.create_buffer_bda(
            as_bytes,
            extra_usage=BUFFER_USAGE_AS_STORAGE_KHR | BUFFER_USAGE_DEVICE_ADDRESS,
            host_visible=False,
        )
        self.rewind()
        var create_info = self.scratch(64)
        put32(create_info, TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR)
        put64(create_info + 24, as_buf.vk_buffer)
        put64(create_info + 40, as_bytes)
        put32(create_info + 48, AS_TYPE_TOP_LEVEL_KHR)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateAccelerationStructureKHR"), self.device, create_info, 0, slot), "vkCreateAccelerationStructureKHR(TLAS)"):
            self.destroy_buffer(as_buf)
            return False
        tlas.handle = get64(slot)
        tlas.buffer = as_buf^
        return True

    def record_build_tlas(
        mut self,
        frame: Int,
        tlas: VulkanAccelStruct,
        instances_addr: UInt64,
        instance_count: Int,
        scratch_buf: GpuBuffer,
    ):
        """Records an inline `vkCmdBuildAccelerationStructuresKHR` into `tlas` (`0` host allocations, `0` extra submits!)."""
        if not self.has_ray_query or tlas.handle == 0 or instance_count < 0 or scratch_buf.size == 0:
            return
        var scratch_addr = (self.buffer_device_address(scratch_buf) + 255) & ~UInt64(255)
        self.rewind()
        var geom = self.scratch(96)
        put32(geom, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
        put32(geom + 16, GEOMETRY_TYPE_INSTANCES_KHR)
        put32(geom + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR)
        put64(geom + 24 + 24, Int(instances_addr))
        put32(geom + 88, GEOMETRY_OPAQUE_BIT_KHR)

        var binfo = self.scratch(80)
        put32(binfo, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
        put32(binfo + 16, AS_TYPE_TOP_LEVEL_KHR)
        put32(binfo + 20, BUILD_AS_PREFER_FAST_TRACE_KHR)
        put32(binfo + 24, BUILD_AS_MODE_BUILD_KHR)
        put64(binfo + 40, tlas.handle)
        put32(binfo + 48, 1)
        put64(binfo + 56, geom)
        put64(binfo + 72, Int(scratch_addr))

        var range_info = self.scratch(16)
        put32(range_info, UInt32(instance_count))
        var pp_range = self.scratch(8)
        put64(pp_range, range_info)
        _ = call4(self.device_call("vkCmdBuildAccelerationStructuresKHR"), self.command_buffers[frame], 1, binfo, pp_range)

    def destroy_accel_struct(mut self, mut accel: VulkanAccelStruct):
        if accel.handle != 0:
            self.rewind()
            _ = call3(self.device_call("vkDestroyAccelerationStructureKHR"), self.device, accel.handle, 0)
            accel.handle = 0
        if accel.buffer.size > 0:
            self.destroy_buffer(accel.buffer)
            accel.buffer = GpuBuffer()
        accel.device_address = 0
