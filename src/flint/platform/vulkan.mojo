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
from std.ffi import external_call
from flint.platform.amdgpu import GpuBuffer
from flint.platform.os import env_flag, env_value

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

comptime ARENA_BYTES: Int = 256 * 1024
comptime API_VERSION_1_2: UInt32 = (1 << 22) | (2 << 12)
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

comptime TYPE_WAYLAND_SURFACE_CREATE_INFO: UInt32 = 1000006000
comptime TYPE_SWAPCHAIN_CREATE_INFO: UInt32 = 1000001000
comptime TYPE_PRESENT_INFO: UInt32 = 1000001001
comptime TYPE_IMAGE_MEMORY_BARRIER: UInt32 = 45
comptime FORMAT_B8G8R8A8_UNORM: UInt32 = 44
comptime PRESENT_FIFO: UInt32 = 2
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
    var descriptor_pool: Int
    var side_pools: List[Int]  # descriptor pools of passes outside the frame (GPU meshing)
    var layout_bindings: Int
    var pipelines: List[Int]
    var surface: Int
    var swapchain: Int
    var swapchain_width: Int
    var swapchain_height: Int
    var images: List[Int]
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

    def __init__(out self):
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
        self.has_wayland_surface = False
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
        self.descriptor_pool = 0
        self.side_pools = List[Int]()
        self.layout_bindings = 0
        self.pipelines = List[Int]()
        self.surface = 0
        self.swapchain = 0
        self.swapchain_width = 0
        self.swapchain_height = 0
        self.images = List[Int]()
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
        var rounded = (size + 15) & ~15
        if self.arena_used + rounded > ARENA_BYTES:
            self.arena_used = 0  # only a bug gets here: every caller rewinds first
        var address = self.arena + self.arena_used
        self.arena_used += rounded
        _ = external_call["memset", Int](address, Int32(0), rounded)
        return address

    def rewind(mut self):
        self.arena_used = 0

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
        var address = lookup(self.proc_device, self.device, self.text(name))
        if address == 0:
            self.error = "the driver has no " + name
        return address

    # Bring-up.

    def initialize(mut self, validate: Bool = False, need_discrete: Bool = False) -> Bool:
        """Open the loader and the best GPU. False leaves `error` and nothing to close."""
        self.arena = external_call["malloc", Int](ARENA_BYTES)
        self.messages = external_call["calloc", Int](Int(2), Int(8))
        self.memory_properties = external_call["malloc", Int](520)
        var library_name = "libvulkan.so.1"
        self.library = external_call["dlopen", Int](library_name.as_c_string_span(), GLOBAL_LOAD_LAZY | GLOBAL_LOAD_LOCAL)
        if self.library == 0:
            return self.fail("no Vulkan loader on this system")
        var symbol = "vkGetInstanceProcAddr"
        self.proc_instance = external_call["dlsym", Int](self.library, symbol.as_c_string_span())
        if self.proc_instance == 0:
            return self.fail("the loader has no vkGetInstanceProcAddr")
        if not self.create_instance(validate):
            return False
        if not self.choose_device(need_discrete):
            return False
        if not self.create_device():
            return False
        return self.load_device_calls() and self.create_queue_objects()

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
        for wanted in ["VK_KHR_surface", "VK_KHR_wayland_surface"]:
            if String(wanted) in offered:
                extensions.append(String(wanted))
        self.has_wayland_surface = len(extensions) == 2
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
        put64(application + 16, self.text("Sharddepth"))
        put64(application + 32, self.text("Sharddepth"))
        put32(application + 44, API_VERSION_1_2)
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
        var properties = self.scratch(DEVICE_PROPERTIES_BYTES)
        var best_rank = 0
        for index in range(count):
            _ = call2(get_properties, get64(devices + index * 8), properties)
            var api = get32(properties)
            var kind = get32(properties + 16)
            # Software rasterizers rank nothing: this game already has a better one.
            var rank = 3 if kind == DEVICE_TYPE_DISCRETE else (2 if kind == DEVICE_TYPE_VIRTUAL else (1 if kind == DEVICE_TYPE_INTEGRATED else 0))
            if api < API_VERSION_1_2:
                rank = 0
            if rank > best_rank:
                best_rank = rank
                self.physical = get64(devices + index * 8)
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
        var timeline_feature = self.scratch(24)
        put32(timeline_feature, TYPE_TIMELINE_SEMAPHORE_FEATURES)
        put32(timeline_feature + 16, 1)
        var info = self.scratch(72)
        put32(info, TYPE_DEVICE_CREATE_INFO)
        put64(info + 8, timeline_feature)
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
            _ = call4(self.fn_destroy_buffer, self.device, buffer, 0, 0)
            _ = self.fail("no memory for a buffer of " + String(rounded) + " bytes")
            return GpuBuffer()
        if call4(self.fn_bind_buffer_memory, self.device, buffer, memory, 0) != SUCCESS or call6(self.fn_map_memory, self.device, memory, 0, WHOLE_SIZE, 0, slot) != SUCCESS:
            _ = call4(self.fn_destroy_buffer, self.device, buffer, 0, 0)
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
        var result = call6(self.device_call("vkCreateComputePipelines"), self.device, 0, 1, info, 0, slot)
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

    def write_timestamp(mut self, frame: Int, query: Int):
        _ = call4(self.fn_write_timestamp, self.command_buffers[frame], PIPELINE_BOTTOM, self.query_pool, query)

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

    def create_swapchain(mut self, width: Int, height: Int) -> Bool:
        """FIFO always exists and paces to the display. Size is ours to choose on Wayland,
        and the compositor scales it, so it matches the render size, not the window."""
        self.rewind()
        var capabilities = self.scratch(52)
        if not self.check(call3(self.instance_call("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"), self.physical, self.surface, capabilities), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"):
            return False
        if (get32(capabilities + 48) & IMAGE_USAGE_TRANSFER_TARGET) == 0:
            return self.fail("swapchain images cannot be copied to")
        var list_formats = self.instance_call("vkGetPhysicalDeviceSurfaceFormatsKHR")
        var count_slot = self.scratch(8)
        _ = call4(list_formats, self.physical, self.surface, count_slot, 0)
        var count = Int(get32(count_slot))
        var formats = self.scratch(max(count, 1) * 8)
        _ = call4(list_formats, self.physical, self.surface, count_slot, formats)
        var color_space = UInt32(0xFFFFFFFF)
        for index in range(count):
            if get32(formats + index * 8) == FORMAT_B8G8R8A8_UNORM and color_space == 0xFFFFFFFF:
                color_space = get32(formats + index * 8 + 4)
        if color_space == 0xFFFFFFFF:
            return self.fail("the window takes no B8G8R8A8_UNORM images")
        var minimum = get32(capabilities)
        var maximum = get32(capabilities + 4)
        var wanted = max(minimum, UInt32(3))
        var info = self.scratch(104)
        put32(info, TYPE_SWAPCHAIN_CREATE_INFO)
        put64(info + 24, self.surface)
        put32(info + 32, min(wanted, maximum) if maximum != 0 else wanted)
        put32(info + 36, FORMAT_B8G8R8A8_UNORM)
        put32(info + 40, color_space)
        put32(info + 44, UInt32(width))
        put32(info + 48, UInt32(height))
        put32(info + 52, 1)
        put32(info + 56, IMAGE_USAGE_TRANSFER_TARGET)
        put32(info + 80, get32(capabilities + 40))
        var alphas = get32(capabilities + 44)
        put32(info + 84, COMPOSITE_OPAQUE if (alphas & COMPOSITE_OPAQUE) != 0 else (alphas & (~alphas + 1)))
        put32(info + 88, PRESENT_FIFO)
        put32(info + 92, 1)
        put64(info + 96, self.swapchain)
        var slot = self.scratch(8)
        if not self.check(call4(self.device_call("vkCreateSwapchainKHR"), self.device, info, 0, slot), "vkCreateSwapchainKHR"):
            return False
        var replaced = self.swapchain
        self.swapchain = get64(slot)
        if replaced != 0:
            _ = call3(self.device_call("vkDestroySwapchainKHR"), self.device, replaced, 0)
        self.swapchain_width = width
        self.swapchain_height = height
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
        return True

    def acquire_image(mut self, frame: Int) -> Int:
        """The next image to fill, or NO_IMAGE to skip presenting this frame."""
        self.rewind()
        var slot = self.scratch(8)
        var result = call7(self.device_call("vkAcquireNextImageKHR"), self.device, self.swapchain, ACQUIRE_TIMEOUT_NS, self.acquire_semaphores[frame], 0, slot, 0)
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

    def copy_to_image(mut self, frame: Int, source: GpuBuffer, row_pixels: Int, image: Int):
        """The finished frame, into the swapchain image, ready to present."""
        self.barrier(frame, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_TRANSFER, ACCESS_TRANSFER_READ)
        self.image_barrier(frame, image, LAYOUT_UNDEFINED, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self.rewind()
        var region = self.scratch(56)
        put32(region + 8, UInt32(row_pixels))
        put32(region + 16, ASPECT_COLOR)
        put32(region + 28, 1)
        put32(region + 44, UInt32(self.swapchain_width))
        put32(region + 48, UInt32(self.swapchain_height))
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
            return
        self.close_swapchain()
        self.rewind()
        if self.device != 0:
            _ = call1(self.device_call("vkDeviceWaitIdle"), self.device)
            for index in range(len(self.pipelines)):
                _ = call3(self.device_call("vkDestroyPipeline"), self.device, self.pipelines[index], 0)
            self.pipelines.clear()
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
        self.rewind()
        if self.messenger != 0:
            _ = call3(self.instance_call("vkDestroyDebugUtilsMessengerEXT"), self.instance, self.messenger, 0)
        _ = call2(self.instance_call("vkDestroyInstance"), self.instance, 0)
        self.instance = 0
