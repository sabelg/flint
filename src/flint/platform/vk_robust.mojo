"""Vulkan robustness state: device loss, out-of-memory, VRAM budgets and fault injection.

`VulkanGpu` owns one `VkRobustState`. Every failing Vulkan result goes through
`note_result`, which latches `VK_ERROR_DEVICE_LOST` (after which submits, waits
and acquires refuse to touch the device) and counts out-of-memory results so
callers can degrade instead of crashing.

`VkFaultHook` is the test seam: it can declare the device lost on the Nth submit
or fail the next N `vkAllocateMemory` calls as `VK_ERROR_OUT_OF_DEVICE_MEMORY`,
so the recovery paths run on a healthy GPU without crashing the driver.

The state also remembers how the presentation surface was attached (Wayland,
XCB or KMS, plus the swapchain preferences) so a renderer recreating its device
after a loss can attach the same window again.
"""
from std.collections import List

comptime VK_ERROR_OUT_OF_HOST_MEMORY: Int32 = -1
comptime VK_ERROR_OUT_OF_DEVICE_MEMORY: Int32 = -2
comptime VK_ERROR_DEVICE_LOST: Int32 = -4
comptime VK_ERROR_SURFACE_LOST: Int32 = -1000000000
comptime VK_MAX_MEMORY_HEAPS: Int = 16
comptime VK_HEAP_DEVICE_LOCAL: UInt32 = 1

comptime VK_SURFACE_NONE: Int = 0
comptime VK_SURFACE_WAYLAND: Int = 1
comptime VK_SURFACE_XCB: Int = 2
comptime VK_SURFACE_DRM: Int = 3

# `VkPhysicalDeviceMemoryBudgetPropertiesEXT` (VK_EXT_memory_budget).
comptime TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES: UInt32 = 1000237000
comptime TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2: UInt32 = 1000059006
comptime MEMORY_PROPERTIES_2_BYTES: Int = 16 + 520
comptime MEMORY_BUDGET_PROPERTIES_BYTES: Int = 16 + 16 * 8 * 2


def vk_result_is_device_lost(result: Int32) -> Bool:
    return result == VK_ERROR_DEVICE_LOST


def vk_result_is_oom(result: Int32) -> Bool:
    return result == VK_ERROR_OUT_OF_HOST_MEMORY or result == VK_ERROR_OUT_OF_DEVICE_MEMORY


struct VkHeapBudget(Copyable, ImplicitlyCopyable, Movable):
    """One memory heap: its size, what the driver lets this process use, and what it uses now."""
    var heap_index: Int
    var size: Int
    var budget: Int
    var usage: Int
    var device_local: Bool
    var from_driver: Bool  # True: VK_EXT_memory_budget numbers; False: heap size and own slab bytes

    def __init__(out self):
        self.heap_index = 0
        self.size = 0
        self.budget = 0
        self.usage = 0
        self.device_local = False
        self.from_driver = False

    def headroom(self) -> Int:
        return max(0, self.budget - self.usage)


struct VkFaultHook(Copyable, ImplicitlyCopyable, Movable):
    """Injectable faults for tests. All off by default."""
    var lose_device_after_submits: Int  # -1 = off; 0 = the next submit reports VK_ERROR_DEVICE_LOST
    var fail_next_allocations: Int  # the next N vkAllocateMemory calls fail as VK_ERROR_OUT_OF_DEVICE_MEMORY
    var budget_override_bytes: Int  # > 0: every device-local heap reports this budget

    def __init__(out self):
        self.lose_device_after_submits = -1
        self.fail_next_allocations = 0
        self.budget_override_bytes = 0

    def clear(mut self):
        self.lose_device_after_submits = -1
        self.fail_next_allocations = 0
        self.budget_override_bytes = 0

    def tick_submit(mut self) -> Bool:
        """True when this submit must be reported as a lost device."""
        if self.lose_device_after_submits < 0:
            return False
        if self.lose_device_after_submits == 0:
            self.lose_device_after_submits = -1
            return True
        self.lose_device_after_submits -= 1
        return False

    def take_alloc_failure(mut self) -> Bool:
        if self.fail_next_allocations <= 0:
            return False
        self.fail_next_allocations -= 1
        return True


struct VkRobustState(Copyable, Movable):
    var device_lost: Bool
    var device_lost_reason: String
    var surface_lost: Bool
    var last_error_result: Int32
    var oom_events: Int
    var device_lost_events: Int
    var budget_refusals: Int
    var has_memory_budget: Bool
    var fault: VkFaultHook
    # Soft allocations (streamed mips, optional caches) are refused above this share of the heap budget.
    var soft_budget_percent: Int
    var soft_depth: Int
    # How the presentation surface was attached, for re-attaching after a device loss.
    var surface_kind: Int
    var surf_a: Int  # wl_display / xcb connection / drm fd
    var surf_b: Int  # wl_surface / xcb window / drm connector
    var surf_refresh_mhz: Int
    var surf_exact: Bool
    var surf_width: Int
    var surf_height: Int
    var prefer_mailbox: Bool
    var prefer_hdr10: Bool
    var prefer_immediate: Bool

    def __init__(out self):
        self.device_lost = False
        self.device_lost_reason = ""
        self.surface_lost = False
        self.last_error_result = 0
        self.oom_events = 0
        self.device_lost_events = 0
        self.budget_refusals = 0
        self.has_memory_budget = False
        self.fault = VkFaultHook()
        self.soft_budget_percent = 90
        self.soft_depth = 0
        self.surface_kind = VK_SURFACE_NONE
        self.surf_a = 0
        self.surf_b = 0
        self.surf_refresh_mhz = 60000
        self.surf_exact = False
        self.surf_width = 0
        self.surf_height = 0
        self.prefer_mailbox = True
        self.prefer_hdr10 = False
        self.prefer_immediate = False

    def note_result(mut self, result: Int32, what: String) -> Bool:
        """Records a failing result. True when it is fatal for the device (device lost)."""
        if result >= 0:
            return False
        self.last_error_result = result
        if vk_result_is_device_lost(result):
            if not self.device_lost:
                self.device_lost_events += 1
                self.device_lost_reason = what
            self.device_lost = True
            return True
        if vk_result_is_oom(result):
            self.oom_events += 1
        elif result == VK_ERROR_SURFACE_LOST:
            self.surface_lost = True
        return False

    def soft_allocations_active(self) -> Bool:
        return self.soft_depth > 0
