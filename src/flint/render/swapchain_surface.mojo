"""Flint v5 Zero-Copy Swapchain & Unified Display Surface (`swapchain_surface.mojo`).

Unifies all Linux presentation backends behind a pure-Mojo Vulkan WSI interface (zero C wrappers):
  1. `SURFACE_WAYLAND`: Windowed & fullscreen Wayland via pure-Mojo `WaylandBridge` (`VK_KHR_wayland_surface`).
  2. `SURFACE_X11_WIRE`: Pure-wire X11 (`flint.platform.x11`) offscreen Vulkan render + `MIT-SHM` / wire presentation
     (zero `libxcb` or `libX11` C library loading).
  3. `SURFACE_KMS_DISPLAY`: Direct bare-metal DRM/KMS scanout (`VK_KHR_display` + `VK_EXT_acquire_drm_display`).
  4. `SURFACE_HEADLESS`: Offscreen CI / benchmark virtual surface requiring zero display server.

Supports `PRESENT_MAILBOX`, `PRESENT_FIFO`, and `PRESENT_IMMEDIATE` pacing alongside:
  - `DISPLAY_SDR_SRGB` (`BGRA8`, `4` bytes/pixel)
  - `DISPLAY_HDR10_PQ` (`A2B10G10R10` `VK_COLOR_SPACE_HDR10_ST2084_EXT`, `4` bytes/pixel)
  - `DISPLAY_EXTENDED_SCRGB_FP16` (`R16G16B16A16_SFLOAT` `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`, `8` bytes/pixel)
"""
from std.math import max, min
from flint.platform.amdgpu import GpuBuffer
from flint.platform.os import env_flag
from flint.platform.vulkan import (
    COLOR_SPACE_EXTENDED_SRGB_LINEAR,
    COLOR_SPACE_HDR10_ST2084,
    COLOR_SPACE_SRGB_NONLINEAR,
    FORMAT_A2B10G10R10_UNORM_PACK32,
    FORMAT_B8G8R8A8_UNORM,
    FORMAT_R16G16B16A16_SFLOAT,
    NO_IMAGE,
    PRESENT_FIFO as VK_PRESENT_FIFO,
    PRESENT_IMMEDIATE as VK_PRESENT_IMMEDIATE,
    PRESENT_MAILBOX as VK_PRESENT_MAILBOX,
    SwapchainHandle,
    VulkanGpu,
    swapchain_format_bytes_per_pixel,
)
from flint.platform.wayland import WaylandWindow

# Surface Backend Kinds
comptime SURFACE_HEADLESS: UInt32 = 0
comptime SURFACE_WAYLAND: UInt32 = 1
comptime SURFACE_X11_XCB: UInt32 = 2
comptime SURFACE_X11_WIRE: UInt32 = 2
comptime SURFACE_KMS_DISPLAY: UInt32 = 3

# Present Pacing Modes
comptime PRESENT_IMMEDIATE: UInt32 = VK_PRESENT_IMMEDIATE
comptime PRESENT_MAILBOX: UInt32 = VK_PRESENT_MAILBOX
comptime PRESENT_FIFO: UInt32 = VK_PRESENT_FIFO

# Display Color & Transfer Modes (matches `FlintAaaVulkanRenderer.display_mode`!)
comptime DISPLAY_SDR_SRGB: UInt32 = 0
comptime DISPLAY_HDR10_PQ: UInt32 = 1
comptime DISPLAY_LINEAR_SRGB: UInt32 = 2
comptime DISPLAY_EXTENDED_SCRGB_FP16: UInt32 = 3


@always_inline
def resolve_swapchain_display_mode(format: UInt32, is_hdr: Bool) -> UInt32:
    """Maps negotiated `VkFormat` + HDR flag to exact renderer output encoding (`SDR`, `HDR10 PQ`, or `Extended scRGB FP16`)."""
    if format == FORMAT_R16G16B16A16_SFLOAT:
        return DISPLAY_EXTENDED_SCRGB_FP16
    if is_hdr:
        return DISPLAY_HDR10_PQ
    return DISPLAY_SDR_SRGB


@always_inline
def validate_present_source_bytes(
    format: UInt32,
    source_bytes: Int,
    source_width: Int,
    copy_width: Int,
    copy_height: Int,
    buffer_offset: UInt64 = 0,
) -> Bool:
    """Verifies that `source_bytes` satisfies the exact byte-per-pixel stride of `format` (`8` for `RGBA16F`, `4` for `BGRA8`/`RGB10A2`)."""
    if copy_width <= 0 or copy_height <= 0:
        return False
    var bpp = swapchain_format_bytes_per_pixel(format)
    var row_px = max(copy_width, source_width)
    var required = Int(buffer_offset) + ((copy_height - 1) * row_px + copy_width) * bpp
    return source_bytes >= required


@fieldwise_init
struct AcquiredFrame(Copyable, ImplicitlyCopyable, Movable):
    """Per-frame swapchain image acquisition token."""
    var image_index: Int
    var slot_index: Int
    var width: Int
    var height: Int
    var format: UInt32
    var display_mode: UInt32
    var is_headless: Bool
    var suboptimal: Bool
    var valid: Bool

    def __init__(out self):
        self.image_index = NO_IMAGE
        self.slot_index = 0
        self.width = 0
        self.height = 0
        self.format = FORMAT_B8G8R8A8_UNORM
        self.display_mode = DISPLAY_SDR_SRGB
        self.is_headless = True
        self.suboptimal = False
        self.valid = False

    @always_inline
    def active(self) -> Bool:
        return self.valid

    @always_inline
    def bytes_per_pixel(self) -> Int:
        return swapchain_format_bytes_per_pixel(self.format)


struct SwapchainSurface(Movable):
    """Unified zero-copy GPU WSI swapchain & headless CI surface manager (100% pure-Mojo, zero `libxcb`)."""
    var backend: UInt32
    var present_mode: UInt32
    var display_mode: UInt32
    var width: Int
    var height: Int
    var handle: SwapchainHandle
    var frames_presented: UInt64
    var resize_generations: Int
    var last_error: String

    def __init__(out self):
        self.backend = SURFACE_HEADLESS
        self.present_mode = PRESENT_FIFO
        self.display_mode = DISPLAY_SDR_SRGB
        self.width = 0
        self.height = 0
        self.handle = SwapchainHandle()
        self.frames_presented = 0
        self.resize_generations = 0
        self.last_error = ""

    def __moveinit__(out self, deinit existing: Self):
        self.backend = existing.backend
        self.present_mode = existing.present_mode
        self.display_mode = existing.display_mode
        self.width = existing.width
        self.height = existing.height
        self.handle = existing.handle^
        self.frames_presented = existing.frames_presented
        self.resize_generations = existing.resize_generations
        self.last_error = existing.last_error^
        existing.backend = SURFACE_HEADLESS
        existing.width = 0
        existing.height = 0
        existing.frames_presented = 0
        existing.resize_generations = 0

    def __deinit__(deinit self):
        pass

    def backend_name(self) -> String:
        if self.backend == SURFACE_WAYLAND:
            return "Wayland-Vulkan"
        if self.backend == SURFACE_X11_WIRE:
            return "X11-PureWire"
        if self.backend == SURFACE_KMS_DISPLAY:
            return "KMS-Display-Vulkan"
        return "Headless-Offscreen"

    @always_inline
    def is_zero_copy_gpu(self) -> Bool:
        return self.backend != SURFACE_HEADLESS and self.handle.vk_swapchain != 0

    @always_inline
    def is_hdr10(self) -> Bool:
        return self.display_mode == DISPLAY_HDR10_PQ

    @always_inline
    def is_scrgb_fp16(self) -> Bool:
        return self.display_mode == DISPLAY_EXTENDED_SCRGB_FP16

    @always_inline
    def bytes_per_pixel(self) -> Int:
        return swapchain_format_bytes_per_pixel(self.handle.format)

    def initialize_headless(
        mut self,
        width: Int,
        height: Int,
        display_mode: UInt32 = DISPLAY_SDR_SRGB,
        present_mode: UInt32 = PRESENT_FIFO,
    ) -> Bool:
        """Initializes an offscreen headless surface for CI, automated metrology, or server rendering."""
        self.backend = SURFACE_HEADLESS
        self.width = max(16, width)
        self.height = max(16, height)
        self.present_mode = present_mode
        self.display_mode = display_mode
        var is_hdr10 = display_mode == DISPLAY_HDR10_PQ
        var is_scrgb = display_mode == DISPLAY_EXTENDED_SCRGB_FP16
        var fmt = FORMAT_B8G8R8A8_UNORM
        var cs = COLOR_SPACE_SRGB_NONLINEAR
        if is_scrgb:
            fmt = FORMAT_R16G16B16A16_SFLOAT
            cs = COLOR_SPACE_EXTENDED_SRGB_LINEAR
        elif is_hdr10:
            fmt = FORMAT_A2B10G10R10_UNORM_PACK32
            cs = COLOR_SPACE_HDR10_ST2084
        self.handle = SwapchainHandle(
            0,
            0,
            self.width,
            self.height,
            fmt,
            cs,
            present_mode,
            2,
            is_hdr10 or is_scrgb,
        )
        self.last_error = ""
        return True

    def initialize(
        mut self,
        mut window: WaylandWindow,
        mut vk: VulkanGpu,
        width: Int,
        height: Int,
        present_mode: UInt32 = PRESENT_MAILBOX,
        display_mode: UInt32 = DISPLAY_SDR_SRGB,
    ) -> Bool:
        """Attaches a zero-copy Vulkan swapchain across Wayland (`WaylandBridge`) or KMS (`VK_KHR_display`),
        or pure-wire X11 / `SURFACE_HEADLESS` without loading any external C wrapper (`libxcb`)."""
        var w = max(16, width)
        var h = max(16, height)
        var want_mailbox = present_mode == PRESENT_MAILBOX
        var want_imm = present_mode == PRESENT_IMMEDIATE
        var want_hdr = (
            display_mode == DISPLAY_HDR10_PQ or display_mode == DISPLAY_EXTENDED_SCRGB_FP16
        ) and vk.has_swapchain_colorspace

        if env_flag("FLINT_HEADLESS") or vk.device == 0 or not vk.has_swapchain:
            return self.initialize_headless(w, h, display_mode, present_mode)

        # 1. Direct DRM/KMS Display Plane Surface (`VK_KHR_display`)
        if window.using_kms:
            var sc_kms = window.attach_vulkan(vk, w, h, prefer_mailbox=False, prefer_hdr10=want_hdr)
            if sc_kms.vk_swapchain != 0:
                self.backend = SURFACE_KMS_DISPLAY
                self.handle = sc_kms^
                self.width = self.handle.width
                self.height = self.handle.height
                self.present_mode = self.handle.present_mode
                self.display_mode = resolve_swapchain_display_mode(self.handle.format, self.handle.is_hdr)
                return True
            self.last_error = window.kms.last_error
            return self.initialize_headless(w, h, display_mode, present_mode)

        # 2. Pure-wire X11 (`x11.mojo` MIT-SHM / socket wire protocol — zero `libxcb` C wrapper!)
        if window.using_x11:
            var ok_hl = self.initialize_headless(w, h, display_mode, present_mode)
            if window.x11.window_id != 0:
                self.backend = SURFACE_X11_WIRE
            return ok_hl

        # 3. Wayland Zero-Copy `VK_KHR_wayland_surface` via `WaylandBridge`
        if window.fd >= 0 and window.bridge.active() and window.bridge.surface_proxy != 0:
            if vk.create_surface(window.bridge.display, window.bridge.surface_proxy, 2):
                if vk.create_swapchain(
                    w,
                    h,
                    prefer_mailbox=want_mailbox,
                    prefer_hdr10=want_hdr,
                    old_swapchain=0,
                    prefer_immediate=want_imm,
                ):
                    var sc_wl = SwapchainHandle(
                        vk.surface,
                        vk.swapchain,
                        vk.swapchain_width,
                        vk.swapchain_height,
                        vk.swapchain_format,
                        vk.swapchain_color_space,
                        vk.swapchain_present_mode,
                        len(vk.images),
                        vk.is_hdr,
                    )
                    window.swapchain = SwapchainHandle(
                        vk.surface,
                        vk.swapchain,
                        vk.swapchain_width,
                        vk.swapchain_height,
                        vk.swapchain_format,
                        vk.swapchain_color_space,
                        vk.swapchain_present_mode,
                        len(vk.images),
                        vk.is_hdr,
                    )
                    window.gpu_width = sc_wl.width
                    window.gpu_height = sc_wl.height
                    window.fit_swapchain_to_window()
                    self.backend = SURFACE_WAYLAND
                    self.handle = sc_wl^
                    self.width = self.handle.width
                    self.height = self.handle.height
                    self.present_mode = self.handle.present_mode
                    self.display_mode = resolve_swapchain_display_mode(self.handle.format, self.handle.is_hdr)
                    return True

        self.last_error = vk.error
        return self.initialize_headless(w, h, display_mode, present_mode)

    def recreate(
        mut self,
        mut vk: VulkanGpu,
        new_width: Int,
        new_height: Int,
    ) -> Bool:
        """Recreates the live Vulkan swapchain in-place passing `old_swapchain` (`VkSwapchainKHR`)
        so window resizes never drop the surface or leak driver swapchain allocations."""
        var nw = max(16, new_width)
        var nh = max(16, new_height)
        if self.backend == SURFACE_HEADLESS or self.backend == SURFACE_X11_WIRE or vk.surface == 0:
            self.width = nw
            self.height = nh
            self.handle.width = nw
            self.handle.height = nh
            self.resize_generations += 1
            return True

        vk.wait_idle()
        var old_sc = UInt64(vk.swapchain)
        var want_hdr = self.display_mode == DISPLAY_HDR10_PQ or self.display_mode == DISPLAY_EXTENDED_SCRGB_FP16
        var ok = vk.create_swapchain(
            nw,
            nh,
            prefer_mailbox=(self.present_mode == PRESENT_MAILBOX),
            prefer_hdr10=want_hdr,
            old_swapchain=old_sc,
            prefer_immediate=(self.present_mode == PRESENT_IMMEDIATE),
        )
        if not ok:
            self.last_error = vk.error
            return False

        self.handle = SwapchainHandle(
            vk.surface,
            vk.swapchain,
            vk.swapchain_width,
            vk.swapchain_height,
            vk.swapchain_format,
            vk.swapchain_color_space,
            vk.swapchain_present_mode,
            len(vk.images),
            vk.is_hdr,
        )
        self.width = self.handle.width
        self.height = self.handle.height
        self.display_mode = resolve_swapchain_display_mode(self.handle.format, self.handle.is_hdr)
        self.resize_generations += 1
        return True

    def acquire_next_frame(mut self, mut vk: VulkanGpu, slot_index: Int) -> AcquiredFrame:
        """Acquires the next swapchain image (or returns a virtual headless frame in CI mode)."""
        if self.backend == SURFACE_HEADLESS or self.backend == SURFACE_X11_WIRE or vk.swapchain == 0:
            return AcquiredFrame(
                0,
                slot_index,
                self.width,
                self.height,
                self.handle.format,
                self.display_mode,
                True,
                False,
                True,
            )

        var img_idx = vk.acquire_image(slot_index)
        if img_idx == NO_IMAGE:
            return AcquiredFrame(
                NO_IMAGE,
                slot_index,
                vk.swapchain_width,
                vk.swapchain_height,
                vk.swapchain_format,
                self.display_mode,
                False,
                True,
                False,
            )

        self.width = vk.swapchain_width
        self.height = vk.swapchain_height
        self.display_mode = resolve_swapchain_display_mode(vk.swapchain_format, vk.is_hdr)
        return AcquiredFrame(
            img_idx,
            slot_index,
            vk.swapchain_width,
            vk.swapchain_height,
            vk.swapchain_format,
            self.display_mode,
            False,
            False,
            True,
        )

    def present_frame(
        mut self,
        mut vk: VulkanGpu,
        acquired: AcquiredFrame,
        source_buf: GpuBuffer,
        source_width: Int,
        source_height: Int,
        buffer_offset: UInt64 = 0,
    ) -> UInt64:
        """Blits/copies `min(source_extent, swapchain_extent)` into the acquired swapchain image and presents."""
        if not acquired.valid:
            return 0
        if acquired.is_headless or vk.swapchain == 0:
            var copy_w_hl = min(source_width, max(1, acquired.width))
            var copy_h_hl = min(source_height, max(1, acquired.height))
            if not validate_present_source_bytes(
                acquired.format,
                source_buf.size,
                source_width,
                copy_w_hl,
                copy_h_hl,
                buffer_offset,
            ):
                self.last_error = "present_frame source buffer too small for target format byte stride"
                return 0
            var f_hl = vk.submit(acquired.slot_index)
            if f_hl != 0:
                self.frames_presented += 1
            return f_hl

        var copy_w = min(source_width, vk.swapchain_width)
        var copy_h = min(source_height, vk.swapchain_height)
        if copy_w > 0 and copy_h > 0:
            if not validate_present_source_bytes(
                vk.swapchain_format,
                source_buf.size,
                source_width,
                copy_w,
                copy_h,
                buffer_offset,
            ):
                self.last_error = "present_frame source buffer too small for swapchain format byte stride"
                return 0
            vk.copy_to_image(
                acquired.slot_index,
                source_buf,
                source_width,
                acquired.image_index,
                buffer_offset,
                copy_w,
                copy_h,
            )
        var fence = vk.submit_and_present(acquired.slot_index, acquired.image_index)
        if fence != 0:
            self.frames_presented += 1
        return fence

    def present_direct(
        mut self,
        mut vk: VulkanGpu,
        acquired: AcquiredFrame,
        extra_wait_sem: Int = 0,
        extra_wait_val: UInt64 = 0,
    ) -> UInt64:
        """Presents an already directly-rendered swapchain image without any `buf_history` -> `copy_to_image` bounce."""
        if not acquired.valid:
            return 0
        if acquired.is_headless or vk.swapchain == 0:
            var f_hl = vk.submit(acquired.slot_index)
            if f_hl != 0:
                self.frames_presented += 1
            return f_hl
        var fence = vk.submit_and_present(
            acquired.slot_index,
            acquired.image_index,
            extra_wait_sem=extra_wait_sem,
            extra_wait_val=extra_wait_val,
        )
        if fence != 0:
            self.frames_presented += 1
        return fence

    def shutdown(mut self, mut vk: VulkanGpu, mut window: WaylandWindow):
        """Cleanly tears down the Vulkan swapchain before disconnecting any Wayland/KMS bridge."""
        window.swapchain = SwapchainHandle()
        window.gpu_width = 0
        window.gpu_height = 0
        if vk.surface != 0:
            vk.close_swapchain()
        self.handle = SwapchainHandle()
        self.backend = SURFACE_HEADLESS

    def shutdown_vk(mut self, mut vk: VulkanGpu):
        """Standalone teardown when no `WaylandWindow` reference is held."""
        if vk.surface != 0:
            vk.close_swapchain()
        self.handle = SwapchainHandle()
        self.backend = SURFACE_HEADLESS
