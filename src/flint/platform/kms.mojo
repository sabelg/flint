"""Direct Linux DRM/KMS window backend: fullscreen frames with no display server.

Talks to `/dev/dri/card*` with raw ioctls (no libdrm): reads the mode resources,
picks the first connected connector and its preferred mode, finds an encoder and
CRTC, allocates two dumb buffers (CREATE_DUMB / MAP_DUMB + mmap), wraps them in
framebuffers and page-flips between them, waiting for the flip event on the fd.

The game's canvas is scaled to the display mode, keeping its aspect ratio
(black bars where the shapes differ).

Input, with no compositor to deliver it:
  * keyboard: the virtual terminal itself, in K_MEDIUMRAW mode (raw Linux key
    codes, no permissions beyond owning the TTY). When there is no VT, evdev
    keyboards from `/dev/input/event*` are used instead.
  * mouse: evdev devices with REL_X/REL_Y. Opened directly when readable (the
    `input` group); otherwise systemd-logind hands them over (platform/logind.mojo,
    `TakeControl` + `TakeDevice`), as it does for display servers. Then keys come
    from evdev too, since logind turns the VT keyboard off. A small software cursor
    is drawn while the pointer is not locked.
  * gamepads: `InputState.poll_gamepad()`, as on every backend.

The console is always given back. The VT is put in KD_GRAPHICS so the text
console stops drawing over the frames; on `close()`, at process exit
(`atexit`), on SIGINT/SIGTERM/SIGHUP/SIGQUIT and on a crash (SIGSEGV, SIGABRT,
...) the previous CRTC is restored, DRM master is dropped and the VT keyboard
mode, termios and KD_TEXT mode are put back. Mojo has no globals and a signal
handler gets no pointer, so that state lives in one page at a fixed address,
`KMS_RESCUE`, like the crash anchor (core/anchor.mojo).

Ctrl+C quits (the TTY no longer turns it into SIGINT); so does a SIGINT or
SIGTERM sent from elsewhere, through `close_requested`. A second signal
restores the console and exits at once.

Select it with FLINT_BACKEND=kms, or run from a text console (Ctrl+Alt+F3) with
neither WAYLAND_DISPLAY nor DISPLAY set.

GPU mode (`for_vulkan = True` before `initialize`, or `WaylandWindow.wants_kms_vulkan`):
everything above except the CPU buffers and the first modeset, so Vulkan can take the
connector for a swapchain (`VulkanGpu.attach_surface(drm_fd=..., drm_connector=...)`,
VK_EXT_acquire_drm_display). `present()` then does nothing; `start_cpu_present()`
falls back to the CPU path when Vulkan cannot have the display.
"""
from std.ffi import external_call
from flint.platform.input import InputState
from flint.platform.os import env_value, env_flag, close_fd, get_time, list_directory, last_errno
from flint.platform.wayland import evdev_key
from flint.platform.logind import LogindSession, device_numbers
from flint.platform.vulkan import VulkanGpu, SwapchainHandle
from flint.core.anchor import sys, IS_X86, SYS_GETPID, SYS_MSYNC

comptime SYS_IOCTL: Int = 16 if IS_X86 else 29

# ---------------------------------------------------------------------------
# DRM uapi (include/uapi/drm/drm.h, drm_mode.h). Same layout on x86-64 and aarch64.
# ---------------------------------------------------------------------------

comptime IOC_WRITE: Int = 1
comptime IOC_READ: Int = 2


def ioc(direction: Int, kind: Int, number: Int, size: Int) -> Int:
    """The Linux `_IOC` request number."""
    return (direction << 30) | (size << 16) | (kind << 8) | number


def drm_iowr(number: Int, size: Int) -> Int:
    return ioc(IOC_READ | IOC_WRITE, 0x64, number, size)


comptime DRM_MODEINFO_BYTES: Int = 68
comptime DRM_CARD_RES_BYTES: Int = 64
comptime DRM_CRTC_BYTES: Int = 104
comptime DRM_ENCODER_BYTES: Int = 20
comptime DRM_CONNECTOR_BYTES: Int = 80
comptime DRM_FB_CMD_BYTES: Int = 28
comptime DRM_PAGE_FLIP_BYTES: Int = 24
comptime DRM_CREATE_DUMB_BYTES: Int = 32
comptime DRM_MAP_DUMB_BYTES: Int = 16
comptime DRM_GET_CAP_BYTES: Int = 16

comptime DRM_IOCTL_SET_MASTER: Int = 0x641E
comptime DRM_IOCTL_DROP_MASTER: Int = 0x641F
comptime DRM_CAP_DUMB_BUFFER: Int = 0x1
comptime DRM_MODE_CONNECTED: Int = 1
comptime DRM_MODE_TYPE_PREFERRED: Int = 1 << 3
comptime DRM_MODE_PAGE_FLIP_EVENT: Int = 0x01
comptime DRM_EVENT_FLIP_COMPLETE: Int = 0x02


def drm_ioctl_get_cap() -> Int:
    return drm_iowr(0x0C, DRM_GET_CAP_BYTES)


def drm_ioctl_get_resources() -> Int:
    return drm_iowr(0xA0, DRM_CARD_RES_BYTES)


def drm_ioctl_get_crtc() -> Int:
    return drm_iowr(0xA1, DRM_CRTC_BYTES)


def drm_ioctl_set_crtc() -> Int:
    return drm_iowr(0xA2, DRM_CRTC_BYTES)


def drm_ioctl_get_encoder() -> Int:
    return drm_iowr(0xA6, DRM_ENCODER_BYTES)


def drm_ioctl_get_connector() -> Int:
    return drm_iowr(0xA7, DRM_CONNECTOR_BYTES)


def drm_ioctl_add_fb() -> Int:
    return drm_iowr(0xAE, DRM_FB_CMD_BYTES)


def drm_ioctl_rm_fb() -> Int:
    return drm_iowr(0xAF, 4)


def drm_ioctl_page_flip() -> Int:
    return drm_iowr(0xB0, DRM_PAGE_FLIP_BYTES)


def drm_ioctl_create_dumb() -> Int:
    return drm_iowr(0xB2, DRM_CREATE_DUMB_BYTES)


def drm_ioctl_map_dumb() -> Int:
    return drm_iowr(0xB3, DRM_MAP_DUMB_BYTES)


def drm_ioctl_destroy_dumb() -> Int:
    return drm_iowr(0xB4, 4)


# ---------------------------------------------------------------------------
# VT, termios and evdev uapi
# ---------------------------------------------------------------------------

comptime KDSETMODE: Int = 0x4B3A
comptime KDGETMODE: Int = 0x4B3B
comptime KDGKBMODE: Int = 0x4B44
comptime KDSKBMODE: Int = 0x4B45
comptime KD_TEXT: Int = 0
comptime KD_GRAPHICS: Int = 1
comptime K_MEDIUMRAW: Int = 2
comptime TCGETS: Int = 0x5401
comptime TCSETS: Int = 0x5402
comptime TERMIOS_BYTES: Int = 64  # struct termios is 36 bytes; room to spare

comptime EV_SYN: Int = 0
comptime EV_KEY: Int = 1
comptime EV_REL: Int = 2
comptime REL_X: Int = 0
comptime REL_Y: Int = 1
comptime REL_WHEEL: Int = 8
comptime BTN_LEFT: Int = 0x110
comptime BTN_RIGHT: Int = 0x111
comptime BTN_MIDDLE: Int = 0x112
comptime INPUT_EVENT_BYTES: Int = 24  # struct input_event on 64-bit Linux
comptime LINUX_KEY_A: Int = 30
comptime LINUX_KEY_Z: Int = 44
comptime LINUX_KEY_C: Int = 46
comptime LINUX_KEY_LEFTCTRL: Int = 29
comptime LINUX_KEY_RIGHTCTRL: Int = 97


def eviocgbit(event_type: Int, length: Int) -> Int:
    """EVIOCGBIT(ev, len): the capability bits of one event type."""
    return ioc(IOC_READ, 0x45, 0x20 + event_type, length)


comptime O_RDWR_CLOEXEC: Int = 2 | 0x80000
comptime O_RDONLY_NONBLOCK_CLOEXEC: Int = 0 | 2048 | 0x80000
comptime O_RDWR_NONBLOCK_CLOEXEC: Int = 2 | 2048 | 0x80000

comptime KMS_MOUSE_BUTTONS: Int = 3
comptime KMS_RELEASE_HELD: Int = -1

# ---------------------------------------------------------------------------
# Little-endian struct packing
# ---------------------------------------------------------------------------


def kput16(mut data: List[UInt8], offset: Int, value: Int):
    data[offset] = UInt8(value & 0xFF)
    data[offset + 1] = UInt8((value >> 8) & 0xFF)


def kput32(mut data: List[UInt8], offset: Int, value: Int):
    for index in range(4):
        data[offset + index] = UInt8((value >> (index * 8)) & 0xFF)


def kput64(mut data: List[UInt8], offset: Int, value: Int):
    for index in range(8):
        data[offset + index] = UInt8((value >> (index * 8)) & 0xFF)


def kget16(data: List[UInt8], offset: Int) -> Int:
    return Int(data[offset]) | (Int(data[offset + 1]) << 8)


def kget32(data: List[UInt8], offset: Int) -> Int:
    var value = 0
    for index in range(4):
        value |= Int(data[offset + index]) << (index * 8)
    return value


def kgets32(data: List[UInt8], offset: Int) -> Int:
    var value = kget32(data, offset)
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def kget64(data: List[UInt8], offset: Int) -> Int:
    var value = 0
    for index in range(8):
        value |= Int(data[offset + index]) << (index * 8)
    return value


def drm_ioctl(fd: Int, request: Int, mut data: List[UInt8]) -> Bool:
    """An ioctl on a packed struct, retried on EINTR/EAGAIN like drmIoctl does."""
    for _ in range(16):
        var result = sys(SYS_IOCTL, fd, request, Int(data.unsafe_ptr()))
        if result == 0:
            return True
        if result != -4 and result != -11:  # EINTR, EAGAIN
            return False
    return False


def ioctl_value(fd: Int, request: Int, value: Int) -> Bool:
    """An ioctl whose argument is a plain integer (KDSETMODE, KDSKBMODE, SET_MASTER)."""
    return sys(SYS_IOCTL, fd, request, Int(value)) == 0


def ioctl_read_int(fd: Int, request: Int) -> Int:
    """An ioctl that writes one int (KDGETMODE, KDGKBMODE). -1 on failure."""
    var out = List[UInt8](length=8, fill=0)
    if sys(SYS_IOCTL, fd, request, Int(out.unsafe_ptr())) != 0:
        return -1
    return kgets32(out, 0)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


struct KmsMode(Copyable, Movable):
    """One `struct drm_mode_modeinfo`, decoded, plus its raw bytes for SETCRTC."""
    var clock: Int
    var hdisplay: Int
    var vdisplay: Int
    var htotal: Int
    var vtotal: Int
    var vrefresh: Int
    var flags: Int
    var type: Int
    var name: String
    var raw: List[UInt8]

    def __init__(out self):
        self.clock = 0
        self.hdisplay = 0
        self.vdisplay = 0
        self.htotal = 0
        self.vtotal = 0
        self.vrefresh = 0
        self.flags = 0
        self.type = 0
        self.name = ""
        self.raw = List[UInt8](length=DRM_MODEINFO_BYTES, fill=0)

    def preferred(self) -> Bool:
        return (self.type & DRM_MODE_TYPE_PREFERRED) != 0

    def refresh_hz(self) -> Float64:
        """The refresh rate, from the timings when vrefresh is missing."""
        if self.vrefresh > 0:
            return Float64(self.vrefresh)
        if self.htotal > 0 and self.vtotal > 0 and self.clock > 0:
            return Float64(self.clock) * 1000.0 / Float64(self.htotal * self.vtotal)
        return 60.0


def parse_mode(data: List[UInt8], offset: Int) -> KmsMode:
    """Decode a drm_mode_modeinfo: u32 clock; u16 h{display,sync_start,sync_end,total,skew};
    u16 v{display,sync_start,sync_end,total,scan}; u32 vrefresh, flags, type; char name[32]."""
    var mode = KmsMode()
    for index in range(DRM_MODEINFO_BYTES):
        mode.raw[index] = data[offset + index]
    mode.clock = kget32(data, offset)
    mode.hdisplay = kget16(data, offset + 4)
    mode.htotal = kget16(data, offset + 10)
    mode.vdisplay = kget16(data, offset + 14)
    mode.vtotal = kget16(data, offset + 20)
    mode.vrefresh = kget32(data, offset + 24)
    mode.flags = kget32(data, offset + 28)
    mode.type = kget32(data, offset + 32)
    var name = List[UInt8]()
    for index in range(32):
        var byte = data[offset + 36 + index]
        if byte == 0:
            break
        if byte < 128:
            name.append(byte)
    mode.name = String(unsafe_from_utf8=name)
    return mode^


def pack_mode(mut data: List[UInt8], offset: Int, clock: Int, hdisplay: Int, vdisplay: Int, vrefresh: Int, type: Int, name: String):
    """Encode a minimal drm_mode_modeinfo (used by tests and by nothing that reaches a driver)."""
    kput32(data, offset, clock)
    kput16(data, offset + 4, hdisplay)
    kput16(data, offset + 6, hdisplay + 48)
    kput16(data, offset + 8, hdisplay + 80)
    kput16(data, offset + 10, hdisplay + 160)
    kput16(data, offset + 14, vdisplay)
    kput16(data, offset + 16, vdisplay + 3)
    kput16(data, offset + 18, vdisplay + 8)
    kput16(data, offset + 20, vdisplay + 30)
    kput32(data, offset + 24, vrefresh)
    kput32(data, offset + 32, type)
    var bytes = name.as_bytes()
    for index in range(min(31, len(bytes))):
        data[offset + 36 + index] = bytes[index]


def pick_mode(modes: List[KmsMode]) -> Int:
    """The connector's preferred resolution at its fastest refresh (TVs often flag a 30 Hz
    mode as preferred while also offering 60 Hz); without one, the largest, then the fastest.
    -1 if there are no modes."""
    var preferred = -1
    for index in range(len(modes)):
        if modes[index].preferred():
            preferred = index
            break
    var best = -1
    for index in range(len(modes)):
        if preferred >= 0:
            if modes[index].hdisplay != modes[preferred].hdisplay or modes[index].vdisplay != modes[preferred].vdisplay:
                continue
            if best < 0 or modes[index].refresh_hz() > modes[best].refresh_hz() + 0.5:
                best = index
            continue
        if best < 0:
            best = index
            continue
        var area = modes[index].hdisplay * modes[index].vdisplay
        var best_area = modes[best].hdisplay * modes[best].vdisplay
        if area > best_area or (area == best_area and modes[index].refresh_hz() > modes[best].refresh_hz() + 0.5):
            best = index
    return best


def fit_rect(source_width: Int, source_height: Int, target_width: Int, target_height: Int) -> Tuple[Int, Int, Int, Int]:
    """Where a canvas lands on the screen, scaled to fit with its aspect ratio: x, y, w, h."""
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        return (0, 0, max(0, target_width), max(0, target_height))
    var w = target_width
    var h = (source_height * target_width) // source_width
    if h > target_height:
        h = target_height
        w = (source_width * target_height) // source_height
    w = max(1, min(target_width, w))
    h = max(1, min(target_height, h))
    return ((target_width - w) // 2, (target_height - h) // 2, w, h)


def possible_crtc_ids(possible_mask: Int, crtc_ids: List[Int]) -> List[Int]:
    """An encoder's `possible_crtcs` bitmask indexes the resources' CRTC list."""
    var result = List[Int]()
    for index in range(len(crtc_ids)):
        if index < 32 and (possible_mask >> index) & 1 == 1:
            result.append(crtc_ids[index])
    return result^


# ---------------------------------------------------------------------------
# Keyboard decoding
# ---------------------------------------------------------------------------


struct MediumRawDecoder(Copyable, Movable):
    """Bytes from a VT in K_MEDIUMRAW mode to Linux key codes.

    A code below 128 is one byte, bit 7 set on release. A larger one is three:
    `0x00` or `0x80` (release), then the high and low 7 bits, each with bit 7 set.
    """
    var pending: Int
    var release: Bool
    var high: Int
    var codes: List[Int]
    var downs: List[Bool]

    def __init__(out self):
        self.pending = 0
        self.release = False
        self.high = 0
        self.codes = List[Int]()
        self.downs = List[Bool]()

    def feed(mut self, byte: Int):
        if self.pending == 2:
            self.high = byte & 0x7F
            self.pending = 1
            return
        if self.pending == 1:
            self.codes.append((self.high << 7) | (byte & 0x7F))
            self.downs.append(not self.release)
            self.pending = 0
            return
        var code = byte & 0x7F
        self.release = (byte & 0x80) != 0
        if code == 0:
            self.pending = 2
            return
        self.codes.append(code)
        self.downs.append(not self.release)

    def clear(mut self):
        self.codes.clear()
        self.downs.clear()


def parse_input_events(data: List[UInt8], count: Int) -> List[Int]:
    """Whole `struct input_event` records as flat (type, code, value) triples."""
    var out = List[Int]()
    var offset = 0
    while offset + INPUT_EVENT_BYTES <= count:
        out.append(kget16(data, offset + 16))
        out.append(kget16(data, offset + 18))
        out.append(kgets32(data, offset + 20))
        offset += INPUT_EVENT_BYTES
    return out^


def bit_set(bits: List[UInt8], index: Int) -> Bool:
    var byte = index // 8
    return byte < len(bits) and (Int(bits[byte]) >> (index % 8)) & 1 == 1


# ---------------------------------------------------------------------------
# The rescue page: what a signal handler or atexit needs to give the console back
# ---------------------------------------------------------------------------

# 38-bit ARM64-safe candidate slots (< 2^38 = 256 GiB, valid on both 39-bit ARM64 VA and 48-bit x86-64 VA).
comptime KMS_RESCUE_SLOT0: Int = 0x0000003D5A000000
comptime KMS_RESCUE_SLOT1: Int = 0x0000002D5A000000
comptime KMS_RESCUE_SLOT2: Int = 0x0000001D5A000000
comptime KMS_RESCUE: Int = KMS_RESCUE_SLOT0
comptime KMS_RESCUE_BYTES: Int = 4096
comptime KMS_MAGIC: Int = 0x31534D4B544C46  # "FLTKMS1"
comptime R_MAGIC: Int = 0
comptime R_PID: Int = 8
comptime R_TTY_FD: Int = 16
comptime R_OLD_KD_MODE: Int = 24
comptime R_OLD_KB_MODE: Int = 32
comptime R_DRM_FD: Int = 40
comptime R_CRTC_SAVED: Int = 48
comptime R_ACTIVE: Int = 56  # 1 while the console is ours
comptime R_STOP: Int = 64  # termination signals received
comptime R_TERMIOS_SAVED: Int = 72
comptime R_HANDLERS: Int = 80
comptime R_CONNECTOR: Int = 96  # u32, the connector list SETCRTC points at
comptime R_SAVED_CRTC: Int = 128  # DRM_CRTC_BYTES
comptime R_SAVED_TERMIOS: Int = 256  # TERMIOS_BYTES
comptime R_OLD_ACTIONS: Int = 512  # one struct sigaction (152 bytes) per signal slot
comptime SIGACTION_BYTES: Int = 152
comptime R_ZERO_ACTION: Int = 2560  # a zeroed struct sigaction: SIG_DFL
comptime SA_SIGINFO: Int = 4
comptime SA_ONSTACK: Int = 0x08000000

comptime KmsSignalHandler = def(Int32, Int, Int) abi("C") thin -> None
comptime KmsExitHandler = def() abi("C") thin -> None


def _rescue_slot_addr(index: Int) -> Int:
    if index == 0:
        return KMS_RESCUE_SLOT0
    if index == 1:
        return KMS_RESCUE_SLOT1
    return KMS_RESCUE_SLOT2


def _rescue_base() -> Int:
    """Async-signal-safe lookup of the active rescue page among 38-bit ARM64-safe slots."""
    var pid = Int(sys(SYS_GETPID))
    for i in range(3):
        var addr = _rescue_slot_addr(i)
        if sys(SYS_MSYNC, addr, KMS_RESCUE_BYTES, 1) == 0:
            var magic = Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=addr + R_MAGIC)[unsafe_offset=0]
            var owner = Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=addr + R_PID)[unsafe_offset=0]
            if magic == KMS_MAGIC and owner == pid:
                return addr
    return 0


def rload(offset: Int) -> Int:
    var base = _rescue_base()
    if base == 0:
        return 0
    return Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + offset)[unsafe_offset=0]


def rstore(offset: Int, value: Int):
    var base = _rescue_base()
    if base == 0:
        return
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + offset)[unsafe_offset=0] = value


def rstore_byte(offset: Int, value: UInt8):
    var base = _rescue_base()
    if base == 0:
        return
    Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=base + offset)[unsafe_offset=0] = value


def rescue_signals() -> List[Int]:
    """Slot order for saved actions: termination signals first, then crashes."""
    return [2, 15, 1, 3, 11, 7, 8, 4, 6, 5]  # INT TERM HUP QUIT SEGV BUS FPE ILL ABRT TRAP


def rescue_slot(signal: Int) -> Int:
    """The signal's index in `rescue_signals()`, without allocating (it runs in a handler)."""
    if signal == 2:
        return 0
    if signal == 15:
        return 1
    if signal == 1:
        return 2
    if signal == 3:
        return 3
    if signal == 11:
        return 4
    if signal == 7:
        return 5
    if signal == 8:
        return 6
    if signal == 4:
        return 7
    if signal == 6:
        return 8
    if signal == 5:
        return 9
    return -1


def is_stop_signal(signal: Int) -> Bool:
    return signal == 2 or signal == 15 or signal == 1 or signal == 3


def rescue_mapped() -> Bool:
    return _rescue_base() != 0


def rescue_ready() -> Bool:
    return _rescue_base() != 0


def install_rescue() -> Bool:
    if rescue_ready():
        return True
    var pid = Int(sys(SYS_GETPID))
    var chosen = 0
    for i in range(3):
        var candidate = _rescue_slot_addr(i)
        # MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE (38-bit ARM64-safe)
        var mapped = external_call["mmap", Int](candidate, KMS_RESCUE_BYTES, Int32(3), Int32(0x22 | 0x100000), Int32(-1), Int(0))
        if mapped == candidate:
            chosen = candidate
            break
        # Handle fork of a process that already mapped this slot with KMS_MAGIC
        if sys(SYS_MSYNC, candidate, KMS_RESCUE_BYTES, 1) == 0:
            var magic = Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=candidate + R_MAGIC)[unsafe_offset=0]
            if magic == KMS_MAGIC:
                chosen = candidate
                break
    if chosen == 0:
        return False
    for offset in range(0, KMS_RESCUE_BYTES, 8):
        Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=chosen + offset)[unsafe_offset=0] = 0
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=chosen + R_TTY_FD)[unsafe_offset=0] = -1
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=chosen + R_DRM_FD)[unsafe_offset=0] = -1
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=chosen + R_OLD_KD_MODE)[unsafe_offset=0] = -1
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=chosen + R_OLD_KB_MODE)[unsafe_offset=0] = -1
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=chosen + R_PID)[unsafe_offset=0] = pid
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=chosen + R_MAGIC)[unsafe_offset=0] = KMS_MAGIC
    return True


def kms_restore_console():
    """Give the screen and the keyboard back. Async-signal-safe: plain ioctls on saved
    bytes, no allocation. Runs once; later calls do nothing."""
    var base = _rescue_base()
    if base == 0 or Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_ACTIVE)[unsafe_offset=0] == 0:
        return
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_ACTIVE)[unsafe_offset=0] = 0
    var drm = Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_DRM_FD)[unsafe_offset=0]
    if drm >= 0:
        if Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_CRTC_SAVED)[unsafe_offset=0] == 1:
            # The CRTC as it was (usually the text console's framebuffer).
            _ = sys(SYS_IOCTL, drm, drm_ioctl_set_crtc(), base + R_SAVED_CRTC)
        _ = sys(SYS_IOCTL, drm, DRM_IOCTL_DROP_MASTER, Int(0))
    var tty = Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_TTY_FD)[unsafe_offset=0]
    if tty >= 0:
        if Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_TERMIOS_SAVED)[unsafe_offset=0] == 1:
            _ = sys(SYS_IOCTL, tty, TCSETS, base + R_SAVED_TERMIOS)
        var kb = Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_OLD_KB_MODE)[unsafe_offset=0]
        if kb >= 0:
            _ = sys(SYS_IOCTL, tty, KDSKBMODE, Int(kb))
        var kd = Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_OLD_KD_MODE)[unsafe_offset=0]
        _ = sys(SYS_IOCTL, tty, KDSETMODE, Int(kd if kd >= 0 else KD_TEXT))


def kms_on_signal(signal: Int32, info: Int, context: Int) abi("C"):
    var number = Int(signal)
    var base = _rescue_base()
    if is_stop_signal(number) and base != 0 and Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_ACTIVE)[unsafe_offset=0] == 1 and Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_STOP)[unsafe_offset=0] == 0:
        # First request: let the game leave its loop and close the window itself.
        Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_STOP)[unsafe_offset=0] = 1
        return
    kms_restore_console()
    # Put back whatever handled this signal before (the crash reporter, or SIG_DFL).
    var slot = rescue_slot(number)
    if slot >= 0 and base != 0:
        _ = external_call["sigaction", Int32](signal, base + R_OLD_ACTIONS + slot * SIGACTION_BYTES, Int(0))
    elif base != 0:
        _ = external_call["sigaction", Int32](signal, base + R_ZERO_ACTION, Int(0))
    var code = Int(Pointer[Int32, origin=MutAnyOrigin](unsafe_from_address=info + 8)[unsafe_offset=0]) if info != 0 else 0
    if code <= 0 or is_stop_signal(number) or number == 6:
        # Sent, not a fault: send it again so the earlier action runs.
        _ = external_call["raise", Int32](signal)
    # A fault re-runs its instruction on return and reaches the earlier action then.


def kms_at_exit() abi("C"):
    kms_restore_console()


def install_rescue_handlers() -> Bool:
    var base = _rescue_base()
    if base == 0:
        return False
    if Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_HANDLERS)[unsafe_offset=0] == 1:
        return True
    var action = List[Int](length=19, fill=0)
    Pointer[KmsSignalHandler, origin=MutAnyOrigin](unsafe_from_address=Int(action.unsafe_ptr()))[unsafe_offset=0] = kms_on_signal
    action[17] = SA_SIGINFO | SA_ONSTACK
    var signals = rescue_signals()
    var ok = True
    for slot in range(len(signals)):
        if external_call["sigaction", Int32](Int32(signals[slot]), Int(action.unsafe_ptr()), base + R_OLD_ACTIONS + slot * SIGACTION_BYTES) != 0:
            ok = False
    _ = action^
    var exit_slot = List[Int](length=1, fill=0)
    Pointer[KmsExitHandler, origin=MutAnyOrigin](unsafe_from_address=Int(exit_slot.unsafe_ptr()))[unsafe_offset=0] = kms_at_exit
    _ = external_call["atexit", Int32](exit_slot[0])
    _ = exit_slot^
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=base + R_HANDLERS)[unsafe_offset=0] = 1
    return ok


def kms_stop_requested() -> Bool:
    return rescue_ready() and rload(R_STOP) != 0


def open_controlling_vt() -> Int:
    """A read-write fd on this process's VT, or -1 when it is not on one (ssh, a
    terminal emulator, a pipe). KDGETMODE only succeeds on a virtual console."""
    var candidates = List[String]()
    candidates.append("/proc/self/fd/0")
    candidates.append("/dev/tty")
    for path in candidates:
        var owned = path
        var fd = Int(external_call["open", Int32](owned.as_c_string_span(), Int32(O_RDWR_NONBLOCK_CLOEXEC), Int32(0)))
        if fd < 0:
            continue
        if ioctl_read_int(fd, KDGETMODE) >= 0:
            return fd
        close_fd(fd)
    return -1


def on_virtual_console() -> Bool:
    """True when stdin or the controlling terminal is a Linux VT (a text console)."""
    var fd = open_controlling_vt()
    if fd < 0:
        return False
    close_fd(fd)
    return True


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


struct KmsWindow(Copyable, Movable):
    var drm_fd: Int
    var tty_fd: Int
    var card_path: String
    var connector_id: Int
    var encoder_id: Int
    var crtc_id: Int
    var mode: KmsMode
    var width: Int
    var height: Int
    var logical_width: Int
    var logical_height: Int
    var close_requested: Bool
    var focused: Bool
    var fullscreen: Bool
    var pointer_locked: Bool
    var cursor_visible: Bool
    var handles: List[Int]
    var fb_ids: List[Int]
    var maps: List[Int]
    var pitch: Int
    var buffer_size: Int
    var front: Int
    var flip_pending: Bool
    var flips: Int
    var last_flip_at: Float64
    var uncapped: Bool
    var x_map: List[Int]
    var x_map_source: Int
    var x_map_target: Int
    var keyboard_from_tty: Bool
    var decoder: MediumRawDecoder
    var input_fds: List[Int]
    var input_is_keyboard: List[Bool]
    var logind: LogindSession
    var using_logind: Bool
    var input_note: String
    var cursor_x: Float32
    var cursor_y: Float32
    var pending_rel_x: Float32
    var pending_rel_y: Float32
    var pending_moved: Bool
    var pending_wheel: Float32
    var pending_keys: List[Int]
    var pending_key_down: List[Bool]
    var pending_buttons: List[Int]
    var pending_button_down: List[Bool]
    var ctrl_down: Bool
    var for_vulkan: Bool
    var last_error: String

    def __init__(out self):
        self.drm_fd = -1
        self.tty_fd = -1
        self.card_path = ""
        self.connector_id = 0
        self.encoder_id = 0
        self.crtc_id = 0
        self.mode = KmsMode()
        self.width = 0
        self.height = 0
        self.logical_width = 1280
        self.logical_height = 720
        self.close_requested = False
        self.focused = True
        self.fullscreen = True
        self.pointer_locked = False
        self.cursor_visible = True
        self.handles = List[Int]()
        self.fb_ids = List[Int]()
        self.maps = List[Int]()
        self.pitch = 0
        self.buffer_size = 0
        self.front = 0
        self.flip_pending = False
        self.flips = 0
        self.last_flip_at = 0
        self.uncapped = False
        self.x_map = List[Int]()
        self.x_map_source = 0
        self.x_map_target = 0
        self.keyboard_from_tty = False
        self.decoder = MediumRawDecoder()
        self.input_fds = List[Int]()
        self.input_is_keyboard = List[Bool]()
        self.logind = LogindSession()
        self.using_logind = False
        self.input_note = ""
        self.cursor_x = 0
        self.cursor_y = 0
        self.pending_rel_x = 0
        self.pending_rel_y = 0
        self.pending_moved = False
        self.pending_wheel = 0
        self.pending_keys = List[Int]()
        self.pending_key_down = List[Bool]()
        self.pending_buttons = List[Int]()
        self.pending_button_down = List[Bool]()
        self.ctrl_down = False
        self.for_vulkan = False
        self.last_error = ""

    def active(self) -> Bool:
        return self.drm_fd >= 0 and (self.for_vulkan or len(self.fb_ids) == 2)

    def backend_name(self) -> String:
        if not self.active():
            return "Headless"
        return "KMS-Vulkan" if self.for_vulkan and len(self.fb_ids) == 0 else "KMS"

    def refresh_mhz(self) -> Int:
        """The mode's refresh rate in millihertz, as Vulkan display modes count it."""
        return Int(self.mode.refresh_hz() * 1000.0 + 0.5)

    def start_cpu_present(mut self) -> Bool:
        """Leave GPU mode: make the CPU buffers and light the screen with them, for when
        Vulkan could not take the display."""
        if self.drm_fd < 0:
            return False
        if len(self.fb_ids) == 2:
            self.for_vulkan = False
            return True
        if not self.create_buffers() or not self.set_crtc(0):
            self.destroy_buffers()
            return False
        self.front = 0
        self.last_flip_at = get_time()
        self.for_vulkan = False
        return True

    def attach_vulkan(mut self, mut vk: VulkanGpu, width: Int, height: Int, prefer_hdr10: Bool = False) -> SwapchainHandle:
        """Hand the screen to a Vulkan swapchain of width x height, the renderer's output
        size: Vulkan sets a display mode of exactly that size (the monitor or TV scales it)
        at the current refresh when the size is the current mode's, else the one closest to
        60 Hz. The CPU buffers go once it has the screen, and present() becomes a no-op.
        Without such a mode, or when Vulkan cannot take the display, the handle is empty
        and the CPU path carries on. Close the swapchain (`vk.close_swapchain()`) before
        close(): Vulkan gives the display back through our DRM fd."""
        if self.drm_fd < 0 or width <= 0 or height <= 0:
            return SwapchainHandle()
        self.wait_flip(100)
        var want_mhz = self.refresh_mhz() if width == self.width and height == self.height else 60000
        var sc = vk.attach_surface(
            width=width,
            height=height,
            prefer_mailbox=False,
            prefer_hdr10=prefer_hdr10,
            drm_fd=self.drm_fd,
            drm_connector=self.connector_id,
            drm_refresh_mhz=want_mhz,
            drm_exact_size=True,
        )
        if sc.swapchain == 0 or sc.width != width or sc.height != height:
            var why = vk.error
            vk.close_swapchain()
            vk.release_display()
            self.last_error = "Vulkan could not take the display: " + why
            # Light the screen again with our buffers, in case Vulkan touched the CRTC.
            if len(self.fb_ids) == 2:
                _ = self.set_crtc(self.front)
            elif self.for_vulkan:
                _ = self.start_cpu_present()
            return SwapchainHandle()
        self.destroy_buffers()
        self.for_vulkan = True
        self.flip_pending = False
        self.width = sc.width
        self.height = sc.height
        return sc^

    def detach_vulkan(mut self, mut vk: VulkanGpu) -> Bool:
        """Take the screen back from a swapchain made by attach_vulkan: close it, let Vulkan
        release the display, and present CPU frames again at this window's KMS mode (for a
        game switching to a CPU-drawn view). attach_vulkan may be called again later."""
        vk.close_swapchain()
        vk.release_display()
        if self.drm_fd < 0:
            return False
        if len(self.fb_ids) == 2:
            return True
        self.width = self.mode.hdisplay
        self.height = self.mode.vdisplay
        self.for_vulkan = True  # start_cpu_present: make the buffers and set our mode
        return self.start_cpu_present()

    def set_title(mut self, title: String):
        """No window title on a bare display."""
        pass

    # -- device discovery ---------------------------------------------------

    def card_candidates(self) -> List[String]:
        var explicit = env_value("FLINT_KMS_DEVICE")
        var out = List[String]()
        if explicit.byte_length() > 0:
            out.append(explicit)
            return out^
        var names = list_directory("/dev/dri")
        var numbers = List[Int]()
        for name in names:
            if name.byte_length() > 4 and name.startswith("card"):
                var number = 0
                var ok = True
                for index in range(4, name.byte_length()):
                    var byte = Int(name.as_bytes()[index])
                    if byte < 48 or byte > 57:
                        ok = False
                        break
                    number = number * 10 + byte - 48
                if ok:
                    numbers.append(number)
        for i in range(1, len(numbers)):
            var j = i
            while j > 0 and numbers[j - 1] > numbers[j]:
                var t = numbers[j]
                numbers[j] = numbers[j - 1]
                numbers[j - 1] = t
                j -= 1
        for number in numbers:
            out.append("/dev/dri/card" + String(number))
        return out^

    def get_resources(self, fd: Int, mut crtcs: List[Int], mut connectors: List[Int]) -> Bool:
        var res = List[UInt8](length=DRM_CARD_RES_BYTES, fill=0)
        if not drm_ioctl(fd, drm_ioctl_get_resources(), res):
            return False
        var count_crtcs = kget32(res, 36)
        var count_connectors = kget32(res, 40)
        var count_encoders = kget32(res, 44)
        if count_crtcs == 0 or count_connectors == 0:
            return False
        var crtc_buf = List[UInt8](length=count_crtcs * 4, fill=0)
        var conn_buf = List[UInt8](length=count_connectors * 4, fill=0)
        var enc_buf = List[UInt8](length=max(1, count_encoders) * 4, fill=0)
        var fb_buf = List[UInt8](length=max(1, kget32(res, 32)) * 4, fill=0)
        res = List[UInt8](length=DRM_CARD_RES_BYTES, fill=0)
        kput64(res, 0, Int(fb_buf.unsafe_ptr()))
        kput64(res, 8, Int(crtc_buf.unsafe_ptr()))
        kput64(res, 16, Int(conn_buf.unsafe_ptr()))
        kput64(res, 24, Int(enc_buf.unsafe_ptr()))
        kput32(res, 32, len(fb_buf) // 4)
        kput32(res, 36, count_crtcs)
        kput32(res, 40, count_connectors)
        kput32(res, 44, len(enc_buf) // 4)
        var ok = drm_ioctl(fd, drm_ioctl_get_resources(), res)
        if ok:
            for index in range(min(count_crtcs, kget32(res, 36))):
                crtcs.append(kget32(crtc_buf, index * 4))
            for index in range(min(count_connectors, kget32(res, 40))):
                connectors.append(kget32(conn_buf, index * 4))
        _ = fb_buf^
        _ = crtc_buf^
        _ = conn_buf^
        _ = enc_buf^
        return ok

    def probe_connector(self, fd: Int, connector: Int, mut modes: List[KmsMode], mut encoders: List[Int]) -> Tuple[Int, Int]:
        """Modes and encoders of a connector. Returns (connection, current encoder id)."""
        var buf = List[UInt8](length=DRM_CONNECTOR_BYTES, fill=0)
        kput32(buf, 48, connector)
        if not drm_ioctl(fd, drm_ioctl_get_connector(), buf):
            return (0, 0)
        # The counts can change between the two calls (hotplug); retry a few times.
        for _ in range(4):
            var count_modes = kget32(buf, 32)
            var count_props = kget32(buf, 36)
            var count_encoders = kget32(buf, 40)
            var mode_buf = List[UInt8](length=max(1, count_modes) * DRM_MODEINFO_BYTES, fill=0)
            var enc_buf = List[UInt8](length=max(1, count_encoders) * 4, fill=0)
            var prop_buf = List[UInt8](length=max(1, count_props) * 4, fill=0)
            var value_buf = List[UInt8](length=max(1, count_props) * 8, fill=0)
            var call = List[UInt8](length=DRM_CONNECTOR_BYTES, fill=0)
            kput64(call, 0, Int(enc_buf.unsafe_ptr()))
            kput64(call, 8, Int(mode_buf.unsafe_ptr()))
            kput64(call, 16, Int(prop_buf.unsafe_ptr()))
            kput64(call, 24, Int(value_buf.unsafe_ptr()))
            kput32(call, 32, count_modes)
            kput32(call, 36, count_props)
            kput32(call, 40, count_encoders)
            kput32(call, 48, connector)
            var ok = drm_ioctl(fd, drm_ioctl_get_connector(), call)
            _ = prop_buf^
            _ = value_buf^
            if not ok:
                return (0, 0)
            if kget32(call, 32) > count_modes or kget32(call, 40) > count_encoders:
                buf = call^
                continue
            modes.clear()
            encoders.clear()
            for index in range(kget32(call, 32)):
                modes.append(parse_mode(mode_buf, index * DRM_MODEINFO_BYTES))
            for index in range(kget32(call, 40)):
                encoders.append(kget32(enc_buf, index * 4))
            _ = mode_buf^
            _ = enc_buf^
            return (kget32(call, 60), kget32(call, 44))
        return (0, 0)

    def encoder_info(self, fd: Int, encoder: Int) -> Tuple[Int, Int]:
        """(current crtc id, possible_crtcs mask) of an encoder, or (-1, 0)."""
        var buf = List[UInt8](length=DRM_ENCODER_BYTES, fill=0)
        kput32(buf, 0, encoder)
        if not drm_ioctl(fd, drm_ioctl_get_encoder(), buf):
            return (-1, 0)
        return (kget32(buf, 8), kget32(buf, 12))

    def choose_output(mut self, fd: Int) -> Bool:
        """The first connected connector with a mode, an encoder and a CRTC."""
        var crtcs = List[Int]()
        var connectors = List[Int]()
        if not self.get_resources(fd, crtcs, connectors):
            self.last_error = "no KMS resources (is modesetting enabled for this driver?)"
            return False
        for connector in connectors:
            var modes = List[KmsMode]()
            var encoders = List[Int]()
            var state = self.probe_connector(fd, connector, modes, encoders)
            if state[0] != DRM_MODE_CONNECTED or len(modes) == 0:
                continue
            var chosen = pick_mode(modes)
            var override = env_value("FLINT_KMS_MODE")  # e.g. 1280x720
            if override.byte_length() > 0:
                var matched = -1
                for index in range(len(modes)):
                    if String(modes[index].hdisplay) + "x" + String(modes[index].vdisplay) == override:
                        if matched < 0 or modes[index].refresh_hz() > modes[matched].refresh_hz() + 0.5:
                            matched = index  # the fastest refresh at that size, not the first listed
                if matched >= 0:
                    chosen = matched
            # The encoder and CRTC already lit, then any free pairing.
            var crtc = 0
            var encoder = 0
            if state[1] != 0:
                var info = self.encoder_info(fd, state[1])
                if info[0] > 0:
                    crtc = info[0]
                    encoder = state[1]
            if crtc == 0:
                for candidate in encoders:
                    var info = self.encoder_info(fd, candidate)
                    if info[0] < 0:
                        continue
                    var options = possible_crtc_ids(info[1], crtcs)
                    if len(options) > 0:
                        crtc = info[0] if info[0] > 0 else options[0]
                        encoder = candidate
                        break
            if crtc == 0:
                continue
            self.connector_id = connector
            self.encoder_id = encoder
            self.crtc_id = crtc
            self.mode = modes[chosen].copy()
            self.width = self.mode.hdisplay
            self.height = self.mode.vdisplay
            return True
        self.last_error = "no connected display with a usable CRTC"
        return False

    # -- buffers --------------------------------------------------------------

    def create_buffers(mut self) -> Bool:
        for _ in range(2):
            var create = List[UInt8](length=DRM_CREATE_DUMB_BYTES, fill=0)
            kput32(create, 0, self.height)
            kput32(create, 4, self.width)
            kput32(create, 8, 32)
            if not drm_ioctl(self.drm_fd, drm_ioctl_create_dumb(), create):
                self.last_error = "DRM_IOCTL_MODE_CREATE_DUMB failed"
                return False
            var handle = kget32(create, 16)
            self.pitch = kget32(create, 20)
            self.buffer_size = kget64(create, 24)
            self.handles.append(handle)
            var fb = List[UInt8](length=DRM_FB_CMD_BYTES, fill=0)
            kput32(fb, 4, self.width)
            kput32(fb, 8, self.height)
            kput32(fb, 12, self.pitch)
            kput32(fb, 16, 32)
            kput32(fb, 20, 24)
            kput32(fb, 24, handle)
            if not drm_ioctl(self.drm_fd, drm_ioctl_add_fb(), fb):
                self.last_error = "DRM_IOCTL_MODE_ADDFB failed"
                return False
            self.fb_ids.append(kget32(fb, 0))
            var map_req = List[UInt8](length=DRM_MAP_DUMB_BYTES, fill=0)
            kput32(map_req, 0, handle)
            if not drm_ioctl(self.drm_fd, drm_ioctl_map_dumb(), map_req):
                self.last_error = "DRM_IOCTL_MODE_MAP_DUMB failed"
                return False
            var address = external_call["mmap", Int](Int(0), self.buffer_size, Int32(3), Int32(1), Int32(self.drm_fd), kget64(map_req, 8))
            if address == -1 or address == 0:
                self.last_error = "mmap of a dumb buffer failed"
                return False
            self.maps.append(address)
            _ = external_call["memset", Int](address, Int32(0), self.buffer_size)
        return True

    def destroy_buffers(mut self):
        for address in self.maps:
            _ = external_call["munmap", Int32](address, self.buffer_size)
        self.maps.clear()
        if self.drm_fd >= 0:
            for fb in self.fb_ids:
                var buf = List[UInt8](length=4, fill=0)
                kput32(buf, 0, fb)
                _ = drm_ioctl(self.drm_fd, drm_ioctl_rm_fb(), buf)
            for handle in self.handles:
                var buf = List[UInt8](length=4, fill=0)
                kput32(buf, 0, handle)
                _ = drm_ioctl(self.drm_fd, drm_ioctl_destroy_dumb(), buf)
        self.fb_ids.clear()
        self.handles.clear()

    def set_crtc(mut self, buffer: Int) -> Bool:
        var connectors = List[UInt8](length=4, fill=0)
        kput32(connectors, 0, self.connector_id)
        var crtc = List[UInt8](length=DRM_CRTC_BYTES, fill=0)
        kput64(crtc, 0, Int(connectors.unsafe_ptr()))
        kput32(crtc, 8, 1)
        kput32(crtc, 12, self.crtc_id)
        kput32(crtc, 16, self.fb_ids[buffer])
        kput32(crtc, 32, 1)  # mode_valid
        for index in range(DRM_MODEINFO_BYTES):
            crtc[36 + index] = self.mode.raw[index]
        var ok = drm_ioctl(self.drm_fd, drm_ioctl_set_crtc(), crtc)
        _ = connectors^
        return ok

    def save_crtc(mut self):
        """Keep the CRTC as it is now, to put back on exit (the text console's mode)."""
        var crtc = List[UInt8](length=DRM_CRTC_BYTES, fill=0)
        kput32(crtc, 12, self.crtc_id)
        if not drm_ioctl(self.drm_fd, drm_ioctl_get_crtc(), crtc):
            return
        if kget32(crtc, 16) == 0 or kget32(crtc, 32) == 0:
            return  # nothing was shown; dropping master lets the fbdev console restore itself
        var base = _rescue_base()
        if base == 0:
            return
        Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=base + R_CONNECTOR)[unsafe_offset=0] = UInt32(self.connector_id)
        kput64(crtc, 0, base + R_CONNECTOR)
        kput32(crtc, 8, 1)
        for index in range(DRM_CRTC_BYTES):
            rstore_byte(R_SAVED_CRTC + index, crtc[index])
        rstore(R_CRTC_SAVED, 1)

    # -- console --------------------------------------------------------------

    def save_console(mut self):
        """Remember the VT's display mode, keyboard mode and termios before anything changes them."""
        self.tty_fd = open_controlling_vt()
        rstore(R_TTY_FD, self.tty_fd)
        if self.tty_fd < 0:
            return
        rstore(R_OLD_KD_MODE, ioctl_read_int(self.tty_fd, KDGETMODE))
        rstore(R_OLD_KB_MODE, ioctl_read_int(self.tty_fd, KDGKBMODE))
        var termios = List[UInt8](length=TERMIOS_BYTES, fill=0)
        if drm_ioctl(self.tty_fd, TCGETS, termios):
            for index in range(TERMIOS_BYTES):
                rstore_byte(R_SAVED_TERMIOS + index, termios[index])
            rstore(R_TERMIOS_SAVED, 1)
        rstore(R_ACTIVE, 1)

    def claim_console(mut self):
        """KD_GRAPHICS so the text console stops drawing; raw key codes from the VT unless
        the keyboard comes from evdev (always the case under logind, which turns it off)."""
        if self.tty_fd < 0:
            return
        var termios = List[UInt8](length=TERMIOS_BYTES, fill=0)
        if drm_ioctl(self.tty_fd, TCGETS, termios):
            var raw = termios.copy()
            # cfmakeraw: no signals from Ctrl+C, no echo, no line editing, no CR mapping.
            kput32(raw, 0, kget32(raw, 0) & ~(0x1 | 0x2 | 0x8 | 0x20 | 0x40 | 0x80 | 0x100 | 0x400))
            kput32(raw, 4, kget32(raw, 4) & ~0x1)
            kput32(raw, 12, kget32(raw, 12) & ~(0x1 | 0x2 | 0x8 | 0x40 | 0x8000))
            raw[17 + 5] = 0  # VTIME
            raw[17 + 6] = 0  # VMIN
            _ = drm_ioctl(self.tty_fd, TCSETS, raw)
        var has_evdev_keyboard = False
        for index in range(len(self.input_fds)):
            if self.input_is_keyboard[index]:
                has_evdev_keyboard = True
        if not self.using_logind and not has_evdev_keyboard:
            if rload(R_OLD_KB_MODE) >= 0 and ioctl_value(self.tty_fd, KDSKBMODE, K_MEDIUMRAW):
                self.keyboard_from_tty = True
        elif not self.using_logind:
            # evdev keyboards are readable: keep the VT from also turning them into text.
            _ = ioctl_value(self.tty_fd, KDSKBMODE, 0x04)  # K_OFF
        _ = ioctl_value(self.tty_fd, KDSETMODE, KD_GRAPHICS)

    def keep_input_device(mut self, fd: Int) -> Bool:
        """Keep a mouse or keyboard; close anything else. True when kept."""
        var types = List[UInt8](length=8, fill=0)
        var keys = List[UInt8](length=96, fill=0)
        var rels = List[UInt8](length=8, fill=0)
        _ = sys(SYS_IOCTL, fd, eviocgbit(0, 8), Int(types.unsafe_ptr()))
        if bit_set(types, EV_KEY):
            _ = sys(SYS_IOCTL, fd, eviocgbit(EV_KEY, 96), Int(keys.unsafe_ptr()))
        if bit_set(types, EV_REL):
            _ = sys(SYS_IOCTL, fd, eviocgbit(EV_REL, 8), Int(rels.unsafe_ptr()))
        var is_mouse = bit_set(rels, REL_X) and bit_set(rels, REL_Y) and bit_set(keys, BTN_LEFT)
        var is_keyboard = bit_set(keys, LINUX_KEY_A) and bit_set(keys, LINUX_KEY_Z) and bit_set(keys, 57)
        if is_mouse or is_keyboard:
            self.input_fds.append(fd)
            self.input_is_keyboard.append(is_keyboard and not is_mouse)
            return True
        close_fd(fd)
        return False

    def open_input_devices(mut self):
        """Mice and keyboards: opened directly when readable, else handed over by logind."""
        var names = list_directory("/dev/input")
        var denied = List[String]()
        for name in names:
            if not name.startswith("event"):
                continue
            var path = "/dev/input/" + name
            var fd = Int(external_call["open", Int32](path.as_c_string_span(), Int32(O_RDONLY_NONBLOCK_CLOEXEC), Int32(0)))
            if fd < 0:
                var error = Int(last_errno())
                if error == 13 or error == 1:  # EACCES, EPERM
                    denied.append(name)
                continue
            _ = self.keep_input_device(fd)
        if len(denied) == 0 or env_flag("FLINT_KMS_NO_LOGIND"):
            if len(denied) > 0:
                self.input_note = "no access to " + String(len(denied)) + " input devices (FLINT_KMS_NO_LOGIND is set)"
            return
        if not self.logind.take_control():
            self.input_note = "mouse unavailable: /dev/input is not readable and logind refused (" + self.logind.last_error + "); run from the text console you logged in on, or join the input group"
            self.logind.close()
            return
        self.using_logind = True
        var kept = 0
        for name in denied:
            var numbers = device_numbers(name)
            if numbers[0] < 0:
                continue
            var fd = self.logind.take_device(numbers[0], numbers[1])
            if fd >= 0 and self.keep_input_device(fd):
                kept += 1
        self.input_note = "input via logind: " + String(kept) + " devices"

    # -- lifecycle --------------------------------------------------------------

    def initialize(mut self, width: Int, height: Int) -> Bool:
        if env_flag("FLINT_HEADLESS"):
            return False
        self.logical_width = max(1, width)
        self.logical_height = max(1, height)
        self.cursor_x = Float32(self.logical_width) * 0.5
        self.cursor_y = Float32(self.logical_height) * 0.5
        self.uncapped = env_flag("FLINT_UNCAPPED")
        if not install_rescue():
            self.last_error = "could not map the console rescue page"
            return False
        var opened = False
        var errors = String("")
        for candidate in self.card_candidates():
            var path = candidate
            var fd = Int(external_call["open", Int32](path.as_c_string_span(), Int32(O_RDWR_CLOEXEC), Int32(0)))
            if fd < 0:
                errors += path + ": cannot open (errno " + String(Int(last_errno())) + "); "
                continue
            var cap = List[UInt8](length=DRM_GET_CAP_BYTES, fill=0)
            kput64(cap, 0, DRM_CAP_DUMB_BUFFER)
            if not drm_ioctl(fd, drm_ioctl_get_cap(), cap) or kget64(cap, 8) == 0:
                errors += path + ": no dumb buffers; "
                close_fd(fd)
                continue
            if not self.choose_output(fd):
                errors += path + ": " + self.last_error + "; "
                close_fd(fd)
                continue
            # The first opener of a card is its master already; this checks it.
            if not ioctl_value(fd, DRM_IOCTL_SET_MASTER, 0):
                errors += path + ": DRM master is held by another client (a display server on this GPU?); "
                close_fd(fd)
                continue
            self.drm_fd = fd
            self.card_path = path
            opened = True
            break
        if not opened:
            self.last_error = "KMS unavailable: " + errors
            return False
        rstore(R_DRM_FD, self.drm_fd)
        rstore(R_STOP, 0)
        _ = install_rescue_handlers()
        self.save_crtc()
        self.save_console()
        rstore(R_ACTIVE, 1)
        self.open_input_devices()
        self.claim_console()
        if self.input_note.byte_length() > 0:
            print("FLINT_KMS_INPUT " + self.input_note)
        if self.for_vulkan:
            # Vulkan takes the connector itself; the first modeset is its swapchain's.
            self.last_flip_at = get_time()
            return True
        if not self.create_buffers():
            self.close()
            return False
        if not self.set_crtc(0):
            self.last_error = "DRM_IOCTL_MODE_SETCRTC failed (errno " + String(Int(last_errno())) + ")"
            self.close()
            return False
        self.front = 0
        self.last_flip_at = get_time()
        return True

    def close(mut self):
        if self.drm_fd >= 0:
            self.wait_flip(100)
        kms_restore_console()
        self.destroy_buffers()
        for fd in self.input_fds:
            close_fd(fd)
        self.input_fds.clear()
        self.input_is_keyboard.clear()
        self.logind.close()  # logind also puts the VT back when control is released
        self.using_logind = False
        close_fd(self.drm_fd)
        self.drm_fd = -1
        close_fd(self.tty_fd)
        self.tty_fd = -1
        if rescue_ready():
            rstore(R_DRM_FD, -1)
            rstore(R_TTY_FD, -1)
            rstore(R_CRTC_SAVED, 0)
            rstore(R_TERMIOS_SAVED, 0)
        self.flip_pending = False

    # -- presenting --------------------------------------------------------------

    def read_drm_events(mut self):
        var buf = List[UInt8](length=1024, fill=0)
        while True:
            var count = Int(external_call["read", Int64](Int32(self.drm_fd), buf.unsafe_ptr(), Int64(len(buf))))
            if count <= 0:
                return
            var offset = 0
            while offset + 8 <= count:
                var kind = kget32(buf, offset)
                var length = kget32(buf, offset + 4)
                if length < 8:
                    break
                if kind == DRM_EVENT_FLIP_COMPLETE:
                    self.flip_pending = False
                offset += length
            if not self.flip_pending:
                return

    def wait_flip(mut self, timeout_ms: Int):
        """Block until the pending page flip completes (the next vblank)."""
        var deadline = get_time() + Float64(timeout_ms) * 0.001
        while self.flip_pending:
            var remaining = Int((deadline - get_time()) * 1000.0)
            if remaining <= 0:
                self.flip_pending = False  # a lost event must not stall the game forever
                return
            var descriptor = List[UInt8](length=8, fill=0)
            kput32(descriptor, 0, self.drm_fd)
            kput16(descriptor, 4, 1)  # POLLIN
            if external_call["poll", Int32](descriptor.unsafe_ptr(), UInt64(1), Int32(remaining)) > 0:
                self.read_drm_events()

    def blit(mut self, mut pixels: List[UInt32], source_width: Int, source_height: Int, buffer: Int):
        var rect = fit_rect(source_width, source_height, self.width, self.height)
        var rx = rect[0]
        var ry = rect[1]
        var rw = rect[2]
        var rh = rect[3]
        var base = self.maps[buffer]
        var src = pixels.unsafe_ptr()
        if rw == source_width:
            for y in range(rh):
                var sy = min(source_height - 1, (y * source_height) // rh)
                _ = external_call["memcpy", Int](base + (ry + y) * self.pitch + rx * 4, Int(src.unsafe_offset(sy * source_width)), rw * 4)
            return
        if self.x_map_source != source_width or self.x_map_target != rw or len(self.x_map) != rw:
            self.x_map = List[Int](length=rw, fill=0)
            for x in range(rw):
                self.x_map[x] = min(source_width - 1, (x * source_width) // rw)
            self.x_map_source = source_width
            self.x_map_target = rw
        var xmap = self.x_map.unsafe_ptr()
        # Build each row in ordinary memory, then copy it out in one go: dumb buffers are
        # often write-combined, where scattered stores are slow.
        var row = List[UInt32](length=rw, fill=0)
        var row_ptr = row.unsafe_ptr()
        var last_sy = -1
        for y in range(rh):
            var sy = min(source_height - 1, (y * source_height) // rh)
            if sy != last_sy:
                var srow = sy * source_width
                for x in range(rw):
                    row_ptr[unsafe_offset=x] = src[unsafe_offset=srow + xmap[unsafe_offset=x]]
                last_sy = sy
            _ = external_call["memcpy", Int](base + (ry + y) * self.pitch + rx * 4, Int(row_ptr), rw * 4)
        _ = row^

    def draw_cursor(mut self, source_width: Int, source_height: Int, buffer: Int):
        """A small arrow at the pointer, since a bare display has no cursor of its own."""
        if self.pointer_locked or not self.cursor_visible or len(self.input_fds) == 0:
            return
        var has_mouse = False
        for index in range(len(self.input_fds)):
            if not self.input_is_keyboard[index]:
                has_mouse = True
        if not has_mouse:
            return
        var rect = fit_rect(source_width, source_height, self.width, self.height)
        var px = rect[0] + Int(self.cursor_x * Float32(rect[2]) / Float32(max(1, self.logical_width)))
        var py = rect[1] + Int(self.cursor_y * Float32(rect[3]) / Float32(max(1, self.logical_height)))
        var base = self.maps[buffer]
        var size = max(12, self.height // 60)
        for dy in range(size):
            var span = dy * 2 // 3 + 1
            for dx in range(span):
                var x = px + dx
                var y = py + dy
                if x < rect[0] or y < rect[1] or x >= rect[0] + rect[2] or y >= rect[1] + rect[3]:
                    continue
                var edge = dx == 0 or dx == span - 1 or dy == size - 1
                Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=base + y * self.pitch + x * 4)[unsafe_offset=0] = UInt32(0x101010) if edge else UInt32(0xF2F5F8)

    def present(mut self, mut pixels: List[UInt32], source_width: Int, source_height: Int):
        if len(self.fb_ids) != 2:
            if kms_stop_requested():
                self.close_requested = True
            return  # GPU mode: the Vulkan swapchain presents
        if not self.active() or source_width <= 0 or source_height <= 0 or len(pixels) < source_width * source_height:
            return
        if kms_stop_requested():
            self.close_requested = True
        if self.flip_pending:
            self.wait_flip(100)
        # Drivers without vblank interrupts (simpledrm) complete flips at once; keep to the refresh rate.
        if not self.uncapped:
            var interval = 1.0 / max(24.0, self.mode.refresh_hz())
            var wait_ms = Int((interval - (get_time() - self.last_flip_at)) * 1000.0) - 1
            if wait_ms > 0 and wait_ms < 100:
                var idle = List[UInt8](length=8, fill=0)
                kput32(idle, 0, -1)  # a negative fd is ignored: a plain sleep
                _ = external_call["poll", Int32](idle.unsafe_ptr(), UInt64(1), Int32(wait_ms))
        var back = 1 - self.front
        self.blit(pixels, source_width, source_height, back)
        self.draw_cursor(source_width, source_height, back)
        var flip = List[UInt8](length=DRM_PAGE_FLIP_BYTES, fill=0)
        kput32(flip, 0, self.crtc_id)
        kput32(flip, 4, self.fb_ids[back])
        kput32(flip, 8, DRM_MODE_PAGE_FLIP_EVENT)
        kput64(flip, 16, self.flips + 1)
        if drm_ioctl(self.drm_fd, drm_ioctl_page_flip(), flip):
            self.flip_pending = True
        elif not self.set_crtc(back):
            self.last_error = "page flip and SETCRTC both failed (errno " + String(Int(last_errno())) + ")"
            return
        self.front = back
        self.flips += 1
        self.last_flip_at = get_time()

    def wait_for_frame(mut self, max_wait_ms: Int = 20):
        if self.flip_pending and max_wait_ms > 0:
            self.wait_flip(max_wait_ms)

    # -- input ------------------------------------------------------------------

    def queue_key(mut self, linux_code: Int, down: Bool):
        if linux_code == LINUX_KEY_LEFTCTRL or linux_code == LINUX_KEY_RIGHTCTRL:
            self.ctrl_down = down
        if down and self.ctrl_down and linux_code == LINUX_KEY_C:
            self.close_requested = True  # what Ctrl+C would have done on the console
        var key = evdev_key(linux_code)
        if key >= 0:
            self.pending_keys.append(key)
            self.pending_key_down.append(down)

    def read_tty(mut self):
        if self.tty_fd < 0 or not self.keyboard_from_tty:
            return
        var buf = List[UInt8](length=256, fill=0)
        while True:
            var count = Int(external_call["read", Int64](Int32(self.tty_fd), buf.unsafe_ptr(), Int64(len(buf))))
            if count <= 0:
                break
            for index in range(count):
                self.decoder.feed(Int(buf[index]))
        for index in range(len(self.decoder.codes)):
            self.queue_key(self.decoder.codes[index], self.decoder.downs[index])
        self.decoder.clear()

    def read_evdev(mut self):
        var buf = List[UInt8](length=INPUT_EVENT_BYTES * 64, fill=0)
        var mouse_scale_x = Float32(self.logical_width) / Float32(max(1, fit_rect(self.logical_width, self.logical_height, self.width, self.height)[2]))
        for device in range(len(self.input_fds)):
            while True:
                var count = Int(external_call["read", Int64](Int32(self.input_fds[device]), buf.unsafe_ptr(), Int64(len(buf))))
                if count <= 0:
                    break
                var events = parse_input_events(buf, count)
                for index in range(0, len(events), 3):
                    var kind = events[index]
                    var code = events[index + 1]
                    var value = events[index + 2]
                    if kind == EV_REL:
                        if code == REL_X:
                            self.pending_rel_x += Float32(value)
                            self.cursor_x = max(Float32(0), min(Float32(self.logical_width - 1), self.cursor_x + Float32(value) * mouse_scale_x))
                            self.pending_moved = True
                        elif code == REL_Y:
                            self.pending_rel_y += Float32(value)
                            self.cursor_y = max(Float32(0), min(Float32(self.logical_height - 1), self.cursor_y + Float32(value) * mouse_scale_x))
                            self.pending_moved = True
                        elif code == REL_WHEEL:
                            self.pending_wheel += Float32(value)
                    elif kind == EV_KEY and value != 2:
                        if code >= BTN_LEFT and code <= BTN_MIDDLE:
                            self.pending_buttons.append(code - BTN_LEFT)
                            self.pending_button_down.append(value != 0)
                        elif not self.keyboard_from_tty:
                            self.queue_key(code, value != 0)

    def poll(mut self, mut input: InputState):
        input.poll_gamepad()
        if not self.active():
            return
        if kms_stop_requested():
            self.close_requested = True
        if self.flip_pending:
            self.read_drm_events()
        self.read_tty()
        self.read_evdev()
        if self.using_logind:
            self.logind.drain()
        if self.pending_moved:
            if self.pointer_locked:
                input.add_mouse_delta(self.pending_rel_x, self.pending_rel_y)
            else:
                input.move_mouse(self.cursor_x, self.cursor_y)
            self.pending_moved = False
        self.pending_rel_x = 0
        self.pending_rel_y = 0
        for index in range(len(self.pending_buttons)):
            if self.pending_buttons[index] == KMS_RELEASE_HELD:
                for button in range(KMS_MOUSE_BUTTONS):
                    if input.mouse_down(button):
                        input.set_mouse(button, False)
            else:
                input.set_mouse(self.pending_buttons[index], self.pending_button_down[index])
        self.pending_buttons.clear()
        self.pending_button_down.clear()
        input.wheel += self.pending_wheel
        self.pending_wheel = 0
        for index in range(len(self.pending_keys)):
            input.set_key(self.pending_keys[index], self.pending_key_down[index])
        self.pending_keys.clear()
        self.pending_key_down.clear()

    def set_pointer_lock(mut self, active: Bool):
        self.pointer_locked = active
        self.pending_rel_x = 0
        self.pending_rel_y = 0

    def hide_cursor(mut self):
        self.cursor_visible = False

    def toggle_fullscreen(mut self):
        """Always fullscreen: there is nothing else on a bare display."""
        pass


def kms_probe() -> String:
    """A report of every card's connectors and modes, read-only: no master, no modeset.
    Safe to run beside a display server."""
    var report = String("")
    var probe = KmsWindow()
    for candidate in probe.card_candidates():
        var path = candidate
        var fd = Int(external_call["open", Int32](path.as_c_string_span(), Int32(O_RDWR_CLOEXEC), Int32(0)))
        if fd < 0:
            report += path + ": cannot open (errno " + String(Int(last_errno())) + ")\n"
            continue
        var cap = List[UInt8](length=DRM_GET_CAP_BYTES, fill=0)
        kput64(cap, 0, DRM_CAP_DUMB_BUFFER)
        var dumb = drm_ioctl(fd, drm_ioctl_get_cap(), cap) and kget64(cap, 8) != 0
        var crtcs = List[Int]()
        var connectors = List[Int]()
        if not probe.get_resources(fd, crtcs, connectors):
            report += path + ": no KMS resources (modesetting off for this driver?)\n"
            close_fd(fd)
            continue
        report += path + ": " + String(len(crtcs)) + " CRTCs, " + String(len(connectors)) + " connectors, dumb buffers " + ("yes" if dumb else "no") + "\n"
        for connector in connectors:
            var modes = List[KmsMode]()
            var encoders = List[Int]()
            var state = probe.probe_connector(fd, connector, modes, encoders)
            var status = "connected" if state[0] == DRM_MODE_CONNECTED else ("disconnected" if state[0] == 2 else "unknown")
            report += "  connector " + String(connector) + ": " + status + ", " + String(len(modes)) + " modes"
            var chosen = pick_mode(modes)
            var override = env_value("FLINT_KMS_MODE")
            if override.byte_length() > 0:
                var matched = -1
                for index in range(len(modes)):
                    if String(modes[index].hdisplay) + "x" + String(modes[index].vdisplay) == override:
                        if matched < 0 or modes[index].refresh_hz() > modes[matched].refresh_hz() + 0.5:
                            matched = index
                if matched >= 0:
                    chosen = matched
            if chosen >= 0:
                report += ", would use " + String(modes[chosen].hdisplay) + "x" + String(modes[chosen].vdisplay) + "@" + String(Int(modes[chosen].refresh_hz() + 0.5))
            report += "\n"
        close_fd(fd)
    if report.byte_length() == 0:
        report = "no /dev/dri/card* devices\n"
    return report
