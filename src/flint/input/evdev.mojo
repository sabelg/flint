"""Pure-Mojo Linux `/dev/input/event*` Evdev Gamepad, `EVIOCGABS` Calibration, `inotify` Hotplug & Dual-Motor `EV_FF` Rumble (`flint.input.evdev`).

Eliminates legacy `/dev/input/js*` uncalibrated trigger bugs by reading exact kernel
`input_absinfo` (`min, max, fuzz, flat`) via `EVIOCGABS` so Xbox (`0..1023`) and Sony
DualSense (`0..255`) unpressed triggers always read `0.0`, while opening event nodes
`O_RDWR | O_NONBLOCK` to upload 48-byte `ff_effect` rumble envelopes via `EVIOCSFF`.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, sqrt
from flint.platform.os import close_fd, list_directory, last_errno, EINTR
from flint.platform.input import (
    InputState,
    PAD_A,
    PAD_B,
    PAD_X,
    PAD_Y,
    PAD_LB,
    PAD_RB,
    PAD_BACK,
    PAD_START,
    PAD_GUIDE,
    PAD_LEFT_STICK,
    PAD_RIGHT_STICK,
    PAD_UP,
    PAD_DOWN,
    PAD_LEFT,
    PAD_RIGHT,
    PAD_LT,
    PAD_RT,
    PAD_STICK_UP,
    PAD_STICK_DOWN,
    PAD_STICK_LEFT,
    PAD_STICK_RIGHT,
    PAD_CONTROLS,
    PAD_DIRECTION,
    PAD_AXIS_LEFT_X,
    PAD_AXIS_LEFT_Y,
    PAD_AXIS_LT,
    PAD_AXIS_RIGHT_X,
    PAD_AXIS_RIGHT_Y,
    PAD_AXIS_RT,
    PAD_AXIS_DPAD_X,
    PAD_AXIS_DPAD_Y,
)

# Linux open(2) & inotify(7) flags
comptime O_RDONLY_NONBLOCK_CLOEXEC: Int32 = 0 | 2048 | 0x80000
comptime O_RDWR_NONBLOCK_CLOEXEC: Int32 = 2 | 2048 | 0x80000
comptime IN_CREATE: UInt32 = 0x00000100
comptime IN_DELETE: UInt32 = 0x00000200
comptime IN_ATTRIB: UInt32 = 0x00000004
comptime IN_NONBLOCK_CLOEXEC: Int32 = 0x800 | 0x80000
comptime EAGAIN_ERR: Int32 = 11
comptime ENODEV_ERR: Int32 = 19

# Linux input.h event types
comptime EV_SYN: UInt16 = 0x00
comptime EV_KEY: UInt16 = 0x01
comptime EV_ABS: UInt16 = 0x03
comptime EV_FF: UInt16 = 0x15
comptime SYN_REPORT: UInt16 = 0x00

# Linux input-event-codes.h ABS axes
comptime ABS_X: Int = 0x00
comptime ABS_Y: Int = 0x01
comptime ABS_Z: Int = 0x02
comptime ABS_RX: Int = 0x03
comptime ABS_RY: Int = 0x04
comptime ABS_RZ: Int = 0x05
comptime ABS_GAS: Int = 0x09
comptime ABS_BRAKE: Int = 0x0A
comptime ABS_HAT0X: Int = 0x10
comptime ABS_HAT0Y: Int = 0x11
comptime EVDEV_ABS_SLOTS: Int = 0x40

# Linux input-event-codes.h Gamepad BTN_* codes
comptime BTN_SOUTH: Int = 0x130   # Xbox A / DualSense Cross
comptime BTN_EAST: Int = 0x131    # Xbox B / DualSense Circle
comptime BTN_NORTH: Int = 0x133   # Xbox X(xpad)/Y / DualSense Triangle
comptime BTN_WEST: Int = 0x134    # Xbox Y(xpad)/X / DualSense Square
comptime BTN_TL: Int = 0x136      # LB / L1
comptime BTN_TR: Int = 0x137      # RB / R1
comptime BTN_TL2: Int = 0x138     # Digital LT / L2
comptime BTN_TR2: Int = 0x139     # Digital RT / R2
comptime BTN_SELECT: Int = 0x13A  # View / Create
comptime BTN_START: Int = 0x13B   # Menu / Options
comptime BTN_MODE: Int = 0x13C    # Guide / PS
comptime BTN_THUMBL: Int = 0x13D  # L3
comptime BTN_THUMBR: Int = 0x13E  # R3
comptime BTN_DPAD_UP: Int = 0x220
comptime BTN_DPAD_DOWN: Int = 0x221
comptime BTN_DPAD_LEFT: Int = 0x222
comptime BTN_DPAD_RIGHT: Int = 0x223

# Linux evdev ioctl constants (64-bit Linux UAPI)
comptime EVIOCGID: UInt64 = 0x80084502
comptime EVIOCGNAME_128: UInt64 = 0x80804506
comptime EVIOCGBIT_EV_8: UInt64 = 0x80084520
comptime EVIOCGBIT_KEY_96: UInt64 = 0x80604521
comptime EVIOCGBIT_ABS_8: UInt64 = 0x80084523
comptime EVIOCGABS_BASE: UInt64 = 0x80184540
comptime EVIOCSFF_IOCTL: UInt64 = 0x40304580
comptime EVIOCRMFF_IOCTL: UInt64 = 0x40044581
comptime FF_RUMBLE_TYPE: UInt16 = 0x0050

# USB Vendor IDs
comptime VENDOR_MICROSOFT: UInt16 = 0x045E
comptime VENDOR_SONY: UInt16 = 0x054C
comptime VENDOR_NINTENDO: UInt16 = 0x057E
comptime VENDOR_VALVE: UInt16 = 0x28DE

comptime MAX_EVDEV_PADS: Int = 4


@always_inline
def _read_i32_le(buf: List[UInt8], off: Int) -> Int32:
    var u = (
        UInt32(buf[off])
        | (UInt32(buf[off + 1]) << 8)
        | (UInt32(buf[off + 2]) << 16)
        | (UInt32(buf[off + 3]) << 24)
    )
    return Int32(u)


@always_inline
def _read_u16_le(buf: List[UInt8], off: Int) -> UInt16:
    return UInt16(buf[off]) | (UInt16(buf[off + 1]) << 8)


def pack_ff_rumble_effect(
    effect_id: Int,
    strong_motor: Float32,
    weak_motor: Float32,
    duration_ms: Int = 1000,
    delay_ms: Int = 0,
) -> List[UInt8]:
    """Packs the exact 48-byte 64-bit Linux UAPI `struct ff_effect` for `FF_RUMBLE` (`0x0050`):
      - `[0..1]`: `type = 0x0050` (`FF_RUMBLE`)
      - `[2..3]`: `id` (`-1` = `0xFFFF` for new effect allocation)
      - `[4..5]`: `direction = 0`
      - `[6..9]`: `ff_trigger { button=0, interval=0 }`
      - `[10..13]`: `ff_replay { length=duration_ms, delay=delay_ms }`
      - `[14..15]`: 2-byte 64-bit union alignment padding
      - `[16..17]`: `u.rumble.strong_magnitude` (`0..65535`)
      - `[18..19]`: `u.rumble.weak_magnitude` (`0..65535`)
      - `[20..47]`: union tail zero-padded to 48 bytes.
    """
    var pkt = List[UInt8](length=48, fill=0)
    pkt[0] = UInt8(FF_RUMBLE_TYPE & 0xFF)
    pkt[1] = UInt8((FF_RUMBLE_TYPE >> 8) & 0xFF)

    var id_u16 = UInt16(Int16(effect_id))
    pkt[2] = UInt8(id_u16 & 0xFF)
    pkt[3] = UInt8((id_u16 >> 8) & 0xFF)

    var dur_u16 = UInt16(min(65535, max(0, duration_ms)))
    pkt[10] = UInt8(dur_u16 & 0xFF)
    pkt[11] = UInt8((dur_u16 >> 8) & 0xFF)

    var dly_u16 = UInt16(min(65535, max(0, delay_ms)))
    pkt[12] = UInt8(dly_u16 & 0xFF)
    pkt[13] = UInt8((dly_u16 >> 8) & 0xFF)

    var s_clamped = max(Float32(0.0), min(Float32(1.0), strong_motor))
    var w_clamped = max(Float32(0.0), min(Float32(1.0), weak_motor))
    var strong_u16 = UInt16(min(65535, max(0, Int(s_clamped * Float32(65535.0) + Float32(0.5)))))
    var weak_u16 = UInt16(min(65535, max(0, Int(w_clamped * Float32(65535.0) + Float32(0.5)))))

    pkt[16] = UInt8(strong_u16 & 0xFF)
    pkt[17] = UInt8((strong_u16 >> 8) & 0xFF)
    pkt[18] = UInt8(weak_u16 & 0xFF)
    pkt[19] = UInt8((weak_u16 >> 8) & 0xFF)
    return pkt^


def pack_ff_play_event(effect_id: Int, play: Bool) -> List[UInt8]:
    """Packs the exact 24-byte 64-bit Linux `struct input_event` triggering/stopping `EV_FF`."""
    var ev = List[UInt8](length=24, fill=0)
    ev[16] = UInt8(EV_FF & 0xFF)
    ev[17] = UInt8((EV_FF >> 8) & 0xFF)
    var code = UInt16(max(0, effect_id) & 0xFFFF)
    ev[18] = UInt8(code & 0xFF)
    ev[19] = UInt8((code >> 8) & 0xFF)
    ev[20] = UInt8(1 if play else 0)
    return ev^


@fieldwise_init
struct AbsAxisInfo(Copyable, ImplicitlyCopyable, Movable):
    """Per-axis calibration read from `EVIOCGABS` (`struct input_absinfo`)."""
    var configured: Bool
    var is_trigger: Bool
    var min_val: Int
    var max_val: Int
    var fuzz: Int
    var flat: Int
    var raw_value: Int

    def __init__(out self):
        self.configured = False
        self.is_trigger = False
        self.min_val = -32768
        self.max_val = 32767
        self.fuzz = 0
        self.flat = 0
        self.raw_value = 0


struct EvdevGamepad(Movable):
    """Single Linux `/dev/input/event*` gamepad with `EVIOCGABS` axis calibration & `EV_FF` rumble."""
    var fd: Int
    var writable: Bool
    var connected: Bool
    var path: String
    var name: String
    var bustype: UInt16
    var vendor_id: UInt16
    var product_id: UInt16
    var version: UInt16
    var is_sony: Bool
    var ff_effect_id: Int
    var abs_info: Array[AbsAxisInfo, EVDEV_ABS_SLOTS]
    var axes: Array[Float32, 8]       # Matches PAD_AXIS_* (LX, LY, LT, RX, RY, RT, DPAD_X, DPAD_Y)
    var down: Array[Bool, PAD_CONTROLS]
    var pressed: Array[Bool, PAD_CONTROLS]
    var released: Array[Bool, PAD_CONTROLS]
    var press_count: Array[Int, PAD_CONTROLS]
    var released_first: Array[Bool, PAD_CONTROLS]
    var events_processed: UInt64
    var device_gen: UInt64

    def __init__(out self):
        self.fd = -1
        self.writable = False
        self.connected = False
        self.path = ""
        self.name = ""
        self.bustype = 0
        self.vendor_id = 0
        self.product_id = 0
        self.version = 0
        self.is_sony = False
        self.ff_effect_id = -1
        self.abs_info = Array[AbsAxisInfo, EVDEV_ABS_SLOTS](fill=AbsAxisInfo())
        self.axes = Array[Float32, 8](fill=0.0)
        self.down = Array[Bool, PAD_CONTROLS](fill=False)
        self.pressed = Array[Bool, PAD_CONTROLS](fill=False)
        self.released = Array[Bool, PAD_CONTROLS](fill=False)
        self.press_count = Array[Int, PAD_CONTROLS](fill=0)
        self.released_first = Array[Bool, PAD_CONTROLS](fill=False)
        self.events_processed = 0
        self.device_gen = 0

    def __moveinit__(out self, deinit existing: Self):
        self.fd = existing.fd
        self.writable = existing.writable
        self.connected = existing.connected
        self.path = existing.path^
        self.name = existing.name^
        self.bustype = existing.bustype
        self.vendor_id = existing.vendor_id
        self.product_id = existing.product_id
        self.version = existing.version
        self.is_sony = existing.is_sony
        self.ff_effect_id = existing.ff_effect_id
        self.abs_info = existing.abs_info^
        self.axes = existing.axes^
        self.down = existing.down^
        self.pressed = existing.pressed^
        self.released = existing.released^
        self.press_count = existing.press_count^
        self.released_first = existing.released_first^
        self.events_processed = existing.events_processed
        self.device_gen = existing.device_gen
        existing.fd = -1
        existing.writable = False
        existing.connected = False
        existing.ff_effect_id = -1
        existing.device_gen = 0

    def __deinit__(deinit self):
        self.close()

    def begin_frame(mut self):
        for i in range(PAD_CONTROLS):
            self.pressed[i] = False
            self.released[i] = False
            self.press_count[i] = 0
            self.released_first[i] = False

    def set_vendor_product(mut self, vendor: UInt16, product: UInt16, name: String = ""):
        if not self.connected:
            self.device_gen += 1
        self.vendor_id = vendor
        self.product_id = product
        self.name = name
        self.is_sony = (
            vendor == VENDOR_SONY
            or "Sony" in name
            or "DualSense" in name
            or "DualShock" in name
        )
        self.connected = True

    @staticmethod
    @always_inline
    def _is_trigger_axis_code(code: Int) -> Bool:
        return code == ABS_Z or code == ABS_RZ or code == ABS_GAS or code == ABS_BRAKE

    def configure_abs_axis(
        mut self,
        axis_code: Int,
        min_val: Int,
        max_val: Int,
        fuzz: Int = 0,
        flat: Int = 0,
        initial_raw: Int = -999999999,
    ):
        """Configures `EVIOCGABS` calibration for `axis_code` (used both by real ioctl and headless tests)."""
        if axis_code < 0 or axis_code >= EVDEV_ABS_SLOTS:
            return
        var trig = Self._is_trigger_axis_code(axis_code)
        var hi = max(min_val + 1, max_val)
        var rest_raw = min_val if trig else (min_val + hi) // 2
        if initial_raw != -999999999:
            rest_raw = initial_raw
        self.abs_info[axis_code] = AbsAxisInfo(
            True,
            trig,
            min_val,
            hi,
            max(0, fuzz),
            max(0, flat),
            rest_raw,
        )
        self._apply_calibrated_abs(axis_code, rest_raw, bypass_fuzz=True)

    @always_inline
    def _set_button(mut self, ctrl: Int, is_down: Bool):
        if ctrl < 0 or ctrl >= PAD_CONTROLS:
            return
        if is_down and not self.down[ctrl]:
            self.pressed[ctrl] = True
            self.press_count[ctrl] += 1
        elif (not is_down) and self.down[ctrl]:
            if self.press_count[ctrl] == 0:
                self.released_first[ctrl] = True
            self.released[ctrl] = True
        self.down[ctrl] = is_down

    def _normalize_trigger(self, info: AbsAxisInfo, raw: Int) -> Float32:
        var span = max(1, info.max_val - info.min_val)
        var shifted = raw - info.min_val
        if shifted <= info.flat:
            return 0.0
        var eff_span = max(1, span - info.flat)
        var norm = Float32(shifted - info.flat) / Float32(eff_span)
        return max(Float32(0.0), min(Float32(1.0), norm))

    def _normalize_signed_axis(self, info: AbsAxisInfo, raw: Int) -> Float32:
        var mid = Float32(info.min_val + info.max_val) * Float32(0.5)
        var half_span = max(Float32(0.5), Float32(info.max_val - info.min_val) * Float32(0.5))
        var delta = Float32(raw) - mid
        var flat_f = Float32(info.flat)
        if abs(delta) <= flat_f:
            return 0.0
        var sign = Float32(1.0) if delta > Float32(0.0) else Float32(-1.0)
        var eff = (abs(delta) - flat_f) / max(Float32(1.0), half_span - flat_f)
        return sign * min(Float32(1.0), eff)

    def _apply_calibrated_abs(mut self, code: Int, raw: Int, bypass_fuzz: Bool = False):
        if code < 0 or code >= EVDEV_ABS_SLOTS:
            return
        if not self.abs_info[code].configured:
            # Default fallback if axis arrived without prior EVIOCGABS
            if Self._is_trigger_axis_code(code):
                self.abs_info[code] = AbsAxisInfo(True, True, 0, 255, 0, 0, 0)
            elif code == ABS_HAT0X or code == ABS_HAT0Y:
                self.abs_info[code] = AbsAxisInfo(True, False, -1, 1, 0, 0, 0)
            else:
                self.abs_info[code] = AbsAxisInfo(True, False, -32768, 32767, 0, 0, 0)

        var info = self.abs_info[code]
        if not bypass_fuzz and info.fuzz > 0:
            var diff = abs(raw - info.raw_value)
            if diff <= (info.fuzz >> 1) and raw != info.min_val and raw != info.max_val:
                return
        info.raw_value = raw
        self.abs_info[code] = info

        if code == ABS_Z or code == ABS_BRAKE:
            var v = self._normalize_trigger(info, raw)
            self.axes[PAD_AXIS_LT] = v
            self._set_button(PAD_LT, v > Float32(0.12))
        elif code == ABS_RZ or code == ABS_GAS:
            var v = self._normalize_trigger(info, raw)
            self.axes[PAD_AXIS_RT] = v
            self._set_button(PAD_RT, v > Float32(0.12))
        elif code == ABS_X:
            self.axes[PAD_AXIS_LEFT_X] = self._normalize_signed_axis(info, raw)
        elif code == ABS_Y:
            self.axes[PAD_AXIS_LEFT_Y] = self._normalize_signed_axis(info, raw)
        elif code == ABS_RX:
            self.axes[PAD_AXIS_RIGHT_X] = self._normalize_signed_axis(info, raw)
        elif code == ABS_RY:
            self.axes[PAD_AXIS_RIGHT_Y] = self._normalize_signed_axis(info, raw)
        elif code == ABS_HAT0X:
            var hv = self._normalize_signed_axis(info, raw)
            self.axes[PAD_AXIS_DPAD_X] = hv
            self._set_button(PAD_LEFT, hv < Float32(-0.5))
            self._set_button(PAD_RIGHT, hv > Float32(0.5))
        elif code == ABS_HAT0Y:
            var hv = self._normalize_signed_axis(info, raw)
            self.axes[PAD_AXIS_DPAD_Y] = hv
            self._set_button(PAD_UP, hv < Float32(-0.5))
            self._set_button(PAD_DOWN, hv > Float32(0.5))

    def feed_raw_evdev_event(mut self, ev_type: UInt16, ev_code: UInt16, ev_value: Int32):
        """Processes a single Linux `input_event` `(type, code, value)` deterministically."""
        self.events_processed += 1
        if ev_type == EV_KEY:
            var is_down = ev_value != 0
            var c = Int(ev_code)
            if c == BTN_SOUTH:
                self._set_button(PAD_A, is_down)
            elif c == BTN_EAST:
                self._set_button(PAD_B, is_down)
            elif c == BTN_WEST:
                self._set_button(PAD_X if self.is_sony else PAD_Y, is_down)
            elif c == BTN_NORTH:
                self._set_button(PAD_Y if self.is_sony else PAD_X, is_down)
            elif c == BTN_TL:
                self._set_button(PAD_LB, is_down)
            elif c == BTN_TR:
                self._set_button(PAD_RB, is_down)
            elif c == BTN_TL2:
                # Only update digital LT if analog trigger axis is not holding a higher value
                if not self.abs_info[ABS_Z].configured and not self.abs_info[ABS_BRAKE].configured:
                    self.axes[PAD_AXIS_LT] = Float32(1.0) if is_down else Float32(0.0)
                    self._set_button(PAD_LT, is_down)
            elif c == BTN_TR2:
                if not self.abs_info[ABS_RZ].configured and not self.abs_info[ABS_GAS].configured:
                    self.axes[PAD_AXIS_RT] = Float32(1.0) if is_down else Float32(0.0)
                    self._set_button(PAD_RT, is_down)
            elif c == BTN_SELECT:
                self._set_button(PAD_BACK, is_down)
            elif c == BTN_START:
                self._set_button(PAD_START, is_down)
            elif c == BTN_MODE:
                self._set_button(PAD_GUIDE, is_down)
            elif c == BTN_THUMBL:
                self._set_button(PAD_LEFT_STICK, is_down)
            elif c == BTN_THUMBR:
                self._set_button(PAD_RIGHT_STICK, is_down)
            elif c == BTN_DPAD_UP:
                self._set_button(PAD_UP, is_down)
            elif c == BTN_DPAD_DOWN:
                self._set_button(PAD_DOWN, is_down)
            elif c == BTN_DPAD_LEFT:
                self._set_button(PAD_LEFT, is_down)
            elif c == BTN_DPAD_RIGHT:
                self._set_button(PAD_RIGHT, is_down)
        elif ev_type == EV_ABS:
            self._apply_calibrated_abs(Int(ev_code), Int(ev_value))

    def feed_raw_packet24(mut self, pkt24: List[UInt8]) -> Bool:
        if len(pkt24) < 24:
            return False
        var ev_type = _read_u16_le(pkt24, 16)
        var ev_code = _read_u16_le(pkt24, 18)
        var ev_val = _read_i32_le(pkt24, 20)
        self.feed_raw_evdev_event(ev_type, ev_code, ev_val)
        return True

    def open_device(mut self, dev_path: String) -> Bool:
        """Opens `/dev/input/eventN` (`O_RDWR | O_NONBLOCK`, fallback `O_RDONLY | O_NONBLOCK`),
        verifies gamepad capability bits (`BTN_SOUTH` + `EV_ABS`), and reads `EVIOCGID` + `EVIOCGABS`."""
        self.close()
        var p_copy = dev_path
        var fd_rw = Int(external_call["open", Int32](p_copy.as_c_string_span(), O_RDWR_NONBLOCK_CLOEXEC, Int32(0)))
        var is_rw = True
        if fd_rw < 0:
            fd_rw = Int(external_call["open", Int32](p_copy.as_c_string_span(), O_RDONLY_NONBLOCK_CLOEXEC, Int32(0)))
            is_rw = False
        if fd_rw < 0:
            return False

        # Check EV_KEY + EV_ABS capability bits and BTN_GAMEPAD (0x130) bit
        var ev_bits = List[UInt8](length=8, fill=0)
        var key_bits = List[UInt8](length=96, fill=0)
        var abs_bits = List[UInt8](length=8, fill=0)
        if external_call["ioctl", Int32](Int32(fd_rw), EVIOCGBIT_EV_8, ev_bits.unsafe_ptr()) < 0:
            close_fd(fd_rw)
            return False
        var has_key = (Int(ev_bits[0]) & (1 << Int(EV_KEY))) != 0
        var has_abs = (Int(ev_bits[0]) & (1 << Int(EV_ABS))) != 0
        if not has_key or not has_abs:
            close_fd(fd_rw)
            return False

        _ = external_call["ioctl", Int32](Int32(fd_rw), EVIOCGBIT_KEY_96, key_bits.unsafe_ptr())
        _ = external_call["ioctl", Int32](Int32(fd_rw), EVIOCGBIT_ABS_8, abs_bits.unsafe_ptr())
        var btn_south_byte = BTN_SOUTH >> 3
        var btn_south_mask = 1 << (BTN_SOUTH & 7)
        if btn_south_byte >= len(key_bits) or (Int(key_bits[btn_south_byte]) & btn_south_mask) == 0:
            close_fd(fd_rw)
            return False

        # Read device name and filter out IMU/Motion Sensor & Touchpad sub-nodes
        var name_buf = List[UInt8](length=128, fill=0)
        _ = external_call["ioctl", Int32](Int32(fd_rw), EVIOCGNAME_128, name_buf.unsafe_ptr())
        var name_bytes = List[UInt8]()
        for b in name_buf:
            if b == 0:
                break
            name_bytes.append(b)
        var dev_name = String(unsafe_from_utf8=name_bytes)
        if "Motion Sensors" in dev_name or "Touchpad" in dev_name:
            close_fd(fd_rw)
            return False

        # Read EVIOCGID (8 bytes: bustype, vendor, product, version)
        var id_buf = List[UInt8](length=8, fill=0)
        _ = external_call["ioctl", Int32](Int32(fd_rw), EVIOCGID, id_buf.unsafe_ptr())
        self.bustype = _read_u16_le(id_buf, 0)
        var vid = _read_u16_le(id_buf, 2)
        var pid = _read_u16_le(id_buf, 4)
        self.version = _read_u16_le(id_buf, 6)

        self.fd = fd_rw
        self.writable = is_rw
        self.path = dev_path
        self.set_vendor_product(vid, pid, dev_name)

        # Query EVIOCGABS (24 bytes: value, min, max, fuzz, flat, resolution) for all supported ABS axes
        var abs_buf = List[UInt8](length=24, fill=0)
        for ax in range(EVDEV_ABS_SLOTS):
            var ax_byte = ax >> 3
            var ax_mask = 1 << (ax & 7)
            if ax_byte < len(abs_bits) and (Int(abs_bits[ax_byte]) & ax_mask) != 0:
                if external_call["ioctl", Int32](Int32(fd_rw), EVIOCGABS_BASE + UInt64(ax), abs_buf.unsafe_ptr()) == 0:
                    var cur_v = Int(_read_i32_le(abs_buf, 0))
                    var mn_v = Int(_read_i32_le(abs_buf, 4))
                    var mx_v = Int(_read_i32_le(abs_buf, 8))
                    var fz_v = Int(_read_i32_le(abs_buf, 12))
                    var fl_v = Int(_read_i32_le(abs_buf, 16))
                    self.configure_abs_axis(ax, mn_v, mx_v, fz_v, fl_v, cur_v)

        return True

    def close(mut self):
        if self.fd >= 0:
            self.stop_rumble()
            close_fd(self.fd)
            self.fd = -1
        if self.connected:
            self.device_gen += 1
        self.connected = False
        self.writable = False
        self.path = ""
        self.ff_effect_id = -1
        for i in range(8):
            self.axes[i] = 0.0
        for i in range(PAD_CONTROLS):
            self.down[i] = False
            self.pressed[i] = False
            self.released[i] = False
            self.press_count[i] = 0
            self.released_first[i] = False

    def poll(mut self) -> Int:
        """Non-blocking drain of 24-byte `input_event` packets from `/dev/input/event*`."""
        if self.fd < 0:
            return 0
        self.begin_frame()
        var buf = List[UInt8](length=24 * 16, fill=0)
        var count = 0
        while True:
            var n = Int(external_call["read", Int64](Int32(self.fd), buf.unsafe_ptr(), Int64(24 * 16)))
            if n < 24:
                if n < 0:
                    var err = last_errno()
                    if err == ENODEV_ERR:
                        self.close()
                    elif err == EINTR:
                        continue
                break
            var off = 0
            while off + 24 <= n:
                var ev_type = _read_u16_le(buf, off + 16)
                var ev_code = _read_u16_le(buf, off + 18)
                var ev_val = _read_i32_le(buf, off + 20)
                self.feed_raw_evdev_event(ev_type, ev_code, ev_val)
                count += 1
                off += 24
        return count

    def set_rumble(mut self, strong_motor: Float32, weak_motor: Float32, duration_ms: Int = 1000) -> Bool:
        """Uploads 48-byte `ff_effect` via `EVIOCSFF` and writes 24-byte `EV_FF` trigger packet."""
        if self.fd < 0 or not self.writable:
            return False
        var pkt = pack_ff_rumble_effect(self.ff_effect_id, strong_motor, weak_motor, duration_ms)
        var rc = external_call["ioctl", Int32](Int32(self.fd), EVIOCSFF_IOCTL, pkt.unsafe_ptr())
        if rc != 0:
            return False
        var assigned_id = Int(Int16(_read_u16_le(pkt, 2)))
        self.ff_effect_id = assigned_id
        var play_ev = pack_ff_play_event(assigned_id, True)
        var written = external_call["write", Int64](Int32(self.fd), play_ev.unsafe_ptr(), Int64(24))
        return written == 24

    def stop_rumble(mut self):
        if self.fd >= 0 and self.writable and self.ff_effect_id >= 0:
            var stop_ev = pack_ff_play_event(self.ff_effect_id, False)
            _ = external_call["write", Int64](Int32(self.fd), stop_ev.unsafe_ptr(), Int64(24))

    def sync_into_input_state(self, mut input: InputState, slot: Int = 0):
        """Copies calibrated evdev axes & buttons into `InputState.pads[slot]`, applying radial deadzones, `PAD_STICK_*`, and `pad_map`/`toggle_pads`."""
        if slot < 0 or slot >= len(input.pads):
            return
        if not self.connected:
            if input.pads[slot].fd < 0 and input.pads[slot].connected:
                input.pads[slot].connected = False
                input.pads[slot].ff_fd = -1
                input.pads[slot].ff_writable = False
                input.pads[slot].owns_ff_fd = False
                input.pads[slot].clear()
            return
        input.pads[slot].connected = True
        input.pads[slot].sony_layout = self.is_sony
        input.pads[slot].ff_fd = self.fd
        input.pads[slot].ff_writable = self.writable
        input.pads[slot].owns_ff_fd = False

        # Copy raw calibrated axes and apply scaled radial deadzone to left & right sticks
        for a in range(8):
            input.pads[slot].raw_axes[a] = self.axes[a]
        input.pads[slot].axes[PAD_AXIS_LT] = self.axes[PAD_AXIS_LT]
        input.pads[slot].axes[PAD_AXIS_RT] = self.axes[PAD_AXIS_RT]
        input.pads[slot].axes[PAD_AXIS_DPAD_X] = self.axes[PAD_AXIS_DPAD_X]
        input.pads[slot].axes[PAD_AXIS_DPAD_Y] = self.axes[PAD_AXIS_DPAD_Y]

        var dz_l = max(Float32(0.0), min(Float32(0.95), input.deadzone_left))
        var lrx = self.axes[PAD_AXIS_LEFT_X]
        var lry = self.axes[PAD_AXIS_LEFT_Y]
        var lr = sqrt(lrx * lrx + lry * lry)
        if lr < dz_l:
            input.pads[slot].axes[PAD_AXIS_LEFT_X] = 0.0
            input.pads[slot].axes[PAD_AXIS_LEFT_Y] = 0.0
        else:
            var lscale = min(Float32(1.0), (lr - dz_l) / max(Float32(0.0001), Float32(1.0) - dz_l)) / max(lr, Float32(1e-6))
            input.pads[slot].axes[PAD_AXIS_LEFT_X] = lrx * lscale
            input.pads[slot].axes[PAD_AXIS_LEFT_Y] = lry * lscale

        var dz_r = max(Float32(0.0), min(Float32(0.95), input.deadzone_right))
        var rrx = self.axes[PAD_AXIS_RIGHT_X]
        var rry = self.axes[PAD_AXIS_RIGHT_Y]
        var rr = sqrt(rrx * rrx + rry * rry)
        if rr < dz_r:
            input.pads[slot].axes[PAD_AXIS_RIGHT_X] = 0.0
            input.pads[slot].axes[PAD_AXIS_RIGHT_Y] = 0.0
        else:
            var rscale = min(Float32(1.0), (rr - dz_r) / max(Float32(0.0001), Float32(1.0) - dz_r)) / max(rr, Float32(1e-6))
            input.pads[slot].axes[PAD_AXIS_RIGHT_X] = rrx * rscale
            input.pads[slot].axes[PAD_AXIS_RIGHT_Y] = rry * rscale

        input.pads[slot].set_control(PAD_STICK_LEFT, input.pads[slot].axes[PAD_AXIS_LEFT_X] < -PAD_DIRECTION)
        input.pads[slot].set_control(PAD_STICK_RIGHT, input.pads[slot].axes[PAD_AXIS_LEFT_X] > PAD_DIRECTION)
        input.pads[slot].set_control(PAD_STICK_UP, input.pads[slot].axes[PAD_AXIS_LEFT_Y] < -PAD_DIRECTION)
        input.pads[slot].set_control(PAD_STICK_DOWN, input.pads[slot].axes[PAD_AXIS_LEFT_Y] > PAD_DIRECTION)

        for c in range(PAD_CONTROLS):
            if c >= PAD_STICK_UP and c <= PAD_STICK_RIGHT:
                continue
            if self.released_first[c]:
                input.pads[slot].set_mapped(c, False, input.pad_map, input.toggle_pads)
            if self.press_count[c] > 0:
                for p in range(self.press_count[c]):
                    input.pads[slot].set_mapped(c, True, input.pad_map, input.toggle_pads)
                    if p + 1 < self.press_count[c] or not self.down[c]:
                        input.pads[slot].set_mapped(c, False, input.pad_map, input.toggle_pads)
            elif not self.released_first[c]:
                input.pads[slot].set_mapped(c, self.down[c], input.pad_map, input.toggle_pads)


struct EvdevHub(Movable):
    """Multi-controller `/dev/input/event*` manager with non-blocking `inotify` hotplug."""
    var pads: List[EvdevGamepad]
    var synced_gen: Array[UInt64, MAX_EVDEV_PADS]
    var inotify_fd: Int32
    var watch_wd: Int32
    var scanned_once: Bool

    def __init__(out self, enable_hotplug: Bool = True):
        self.pads = List[EvdevGamepad](capacity=MAX_EVDEV_PADS)
        for _ in range(MAX_EVDEV_PADS):
            self.pads.append(EvdevGamepad())
        self.synced_gen = Array[UInt64, MAX_EVDEV_PADS](fill=0)
        self.inotify_fd = Int32(-1)
        self.watch_wd = Int32(-1)
        self.scanned_once = False
        if enable_hotplug:
            _ = self.enable_hotplug()

    def enable_hotplug(mut self) -> Bool:
        if self.inotify_fd >= 0:
            return True
        self.inotify_fd = external_call["inotify_init1", Int32](IN_NONBLOCK_CLOEXEC)
        if self.inotify_fd >= 0:
            var dir_path = String("/dev/input")
            self.watch_wd = external_call["inotify_add_watch", Int32](
                self.inotify_fd,
                dir_path.as_c_string_span(),
                IN_CREATE | IN_DELETE | IN_ATTRIB,
            )
            return True
        return False

    def __moveinit__(out self, deinit existing: Self):
        self.pads = existing.pads^
        self.synced_gen = existing.synced_gen^
        self.inotify_fd = existing.inotify_fd
        self.watch_wd = existing.watch_wd
        self.scanned_once = existing.scanned_once
        existing.inotify_fd = Int32(-1)
        existing.watch_wd = Int32(-1)
        existing.scanned_once = False

    def __deinit__(deinit self):
        self.close()

    def close(mut self):
        for i in range(len(self.pads)):
            self.pads[i].close()
            self.synced_gen[i] = 0
        if self.inotify_fd >= 0:
            close_fd(Int(self.inotify_fd))
            self.inotify_fd = Int32(-1)
            self.watch_wd = Int32(-1)
        self.scanned_once = False

    def _is_path_open(self, full_path: String) -> Bool:
        for i in range(len(self.pads)):
            if self.pads[i].connected and self.pads[i].path == full_path:
                return True
        return False

    def scan_devices(mut self) -> Int:
        var entries = list_directory("/dev/input")
        var opened = 0
        for name in entries:
            if not name.startswith("event"):
                continue
            var full = "/dev/input/" + name
            if self._is_path_open(full):
                continue
            var free_slot = -1
            for s in range(len(self.pads)):
                if not self.pads[s].connected:
                    free_slot = s
                    break
            if free_slot < 0:
                break
            if self.pads[free_slot].open_device(full):
                opened += 1
        self.scanned_once = True
        return opened

    def discover_initial(mut self) -> Int:
        return self.scan_devices()

    def connected_count(self) -> Int:
        var count = 0
        for s in range(len(self.pads)):
            if self.pads[s].connected:
                count += 1
        return count

    def set_rumble(mut self, slot: Int, strong_motor: Float32, weak_motor: Float32, duration_ms: Int = 1000) -> Bool:
        if slot < 0 or slot >= len(self.pads) or not self.pads[slot].connected:
            return False
        return self.pads[slot].set_rumble(strong_motor, weak_motor, duration_ms)

    def _drain_hotplug_inotify(mut self) -> Bool:
        if self.inotify_fd < 0:
            return False
        var buf = List[UInt8](length=1024, fill=0)
        var any_event = False
        while True:
            var n = Int(external_call["read", Int64](self.inotify_fd, buf.unsafe_ptr(), Int64(1024)))
            if n <= 0:
                break
            any_event = True
        return any_event

    def poll(mut self, mut input: InputState):
        self.poll_all(input)

    def poll_into(mut self, mut input: InputState):
        self.poll(input)

    def poll_all(mut self, mut input: InputState):
        if not self.scanned_once or self._drain_hotplug_inotify():
            _ = self.scan_devices()
        var n_slots = min(len(self.pads), len(input.pads))
        for s in range(n_slots):
            if self.pads[s].connected:
                _ = self.pads[s].poll()
            if self.pads[s].connected:
                if self.synced_gen[s] != self.pads[s].device_gen:
                    if input.pads[s].fd < 0:
                        input.pads[s].clear()
                    self.synced_gen[s] = self.pads[s].device_gen
                self.pads[s].sync_into_input_state(input, s)
                if self.pads[s].fd < 0:
                    self.pads[s].begin_frame()
            else:
                self.synced_gen[s] = 0
                self.pads[s].sync_into_input_state(input, s)
