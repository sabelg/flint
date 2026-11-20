"""Pure-Mojo Win32 audio backend: WASAPI (`ole32.dll`) + `waveOut` (`winmm.dll`) (`flint.platform.win32_audio`).

Zero C/C++ wrapper libraries: implements `AudioBackend` with non-`Copyable` RAII
ownership, pure-Mojo `WAVEFORMATEX` (18 B), `WAVEFORMATEXTENSIBLE` (40 B), `WAVEHDR`
(48 B Win64), WASAPI COM GUID packers and vtable indices, a 4-slot S16LE stereo PCM
ring buffer, and a deterministic polyphonic voice + ambient drone mixer.
"""
from std.collections import List
from std.ffi import external_call
from std.math import sin
from std.sys.info import CompilationTarget
from flint.platform.dsp import soft_clip_tanh
from flint.platform.traits import AudioBackend
from flint.platform.win32_window import (
    IS_WINDOWS,
    _win32_free_dll,
    _win32_get_proc,
    _win32_load_dll,
)

comptime WIN32_WAVEFORMATEX_BYTES: Int = 18
comptime WIN32_WAVEFORMATEXTENSIBLE_BYTES: Int = 40
comptime WIN32_WAVEHDR_BYTES: Int = 48
comptime WIN32_AUDIO_RING_SLOTS: Int = 4
comptime WIN32_AUDIO_PERIOD_FRAMES: Int = 1024
comptime WIN32_AUDIO_SAMPLE_RATE: Int = 48000
comptime WIN32_AUDIO_MAX_VOICES: Int = 8

comptime WAVE_FORMAT_PCM: UInt16 = 0x0001
comptime WAVE_FORMAT_IEEE_FLOAT: UInt16 = 0x0003
comptime WAVE_FORMAT_EXTENSIBLE: UInt16 = 0xFFFE

comptime COINIT_MULTITHREADED: UInt32 = 0x0
comptime RPC_E_CHANGED_MODE: Int32 = -2147417850  # 0x80010106
comptime CLSCTX_ALL: UInt32 = 23
comptime AUDCLNT_SHAREMODE_SHARED: UInt32 = 0
comptime WAVE_MAPPER: UInt32 = 0xFFFFFFFF
comptime CALLBACK_NULL: UInt32 = 0
comptime WHDR_DONE: UInt32 = 0x00000001
comptime WHDR_PREPARED: UInt32 = 0x00000002

# WASAPI COM vtable indices (64-bit)
comptime VTBL_IUNKNOWN_RELEASE: Int = 2
comptime VTBL_ENUM_GET_DEFAULT_ENDPOINT: Int = 4
comptime VTBL_DEVICE_ACTIVATE: Int = 3
comptime VTBL_CLIENT_INITIALIZE: Int = 3
comptime VTBL_CLIENT_GET_BUFFER_SIZE: Int = 4
comptime VTBL_CLIENT_GET_CURRENT_PADDING: Int = 6
comptime VTBL_CLIENT_START: Int = 10
comptime VTBL_CLIENT_STOP: Int = 11
comptime VTBL_CLIENT_RESET: Int = 12
comptime VTBL_CLIENT_GET_SERVICE: Int = 14
comptime VTBL_RENDER_GET_BUFFER: Int = 3
comptime VTBL_RENDER_RELEASE_BUFFER: Int = 4


@always_inline
def _wa_put16(mut buf: List[UInt8], off: Int, val: UInt16):
    buf[off] = UInt8(val & 0xFF)
    buf[off + 1] = UInt8((val >> 8) & 0xFF)


@always_inline
def _wa_put32(mut buf: List[UInt8], off: Int, val: UInt32):
    buf[off] = UInt8(val & 0xFF)
    buf[off + 1] = UInt8((val >> 8) & 0xFF)
    buf[off + 2] = UInt8((val >> 16) & 0xFF)
    buf[off + 3] = UInt8((val >> 24) & 0xFF)


@always_inline
def _wa_put64(mut buf: List[UInt8], off: Int, val: UInt64):
    for i in range(8):
        buf[off + i] = UInt8((val >> UInt64(i * 8)) & 0xFF)


@always_inline
def _wa_get16(buf: List[UInt8], off: Int) -> UInt16:
    return UInt16(buf[off]) | (UInt16(buf[off + 1]) << 8)


@always_inline
def _wa_get32(buf: List[UInt8], off: Int) -> UInt32:
    return (
        UInt32(buf[off])
        | (UInt32(buf[off + 1]) << 8)
        | (UInt32(buf[off + 2]) << 16)
        | (UInt32(buf[off + 3]) << 24)
    )


def win32_pack_guid(
    d1: UInt32,
    d2: UInt16,
    d3: UInt16,
    b0: UInt8,
    b1: UInt8,
    b2: UInt8,
    b3: UInt8,
    b4: UInt8,
    b5: UInt8,
    b6: UInt8,
    b7: UInt8,
) -> List[UInt8]:
    """Packs a 16-byte Windows GUID in mixed-endian COM binary layout."""
    var g = List[UInt8](length=16, fill=0)
    _wa_put32(g, 0, d1)
    _wa_put16(g, 4, d2)
    _wa_put16(g, 6, d3)
    g[8] = b0
    g[9] = b1
    g[10] = b2
    g[11] = b3
    g[12] = b4
    g[13] = b5
    g[14] = b6
    g[15] = b7
    return g^


def win32_guid_clsid_mm_device_enumerator() -> List[UInt8]:
    return win32_pack_guid(0xBCDE0395, 0xE52F, 0x467C, 0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E)


def win32_guid_iid_imm_device_enumerator() -> List[UInt8]:
    return win32_pack_guid(0xA95664D2, 0x9614, 0x4F35, 0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6)


def win32_guid_iid_iaudio_client() -> List[UInt8]:
    return win32_pack_guid(0x1CB9AD4C, 0xDBFA, 0x4C32, 0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2)


def win32_guid_iid_iaudio_render_client() -> List[UInt8]:
    return win32_pack_guid(0xF294ACFC, 0x3146, 0x4483, 0xA8, 0xBF, 0xAD, 0xDC, 0xA3, 0xC2, 0x60, 0xE2)


def win32_guid_ksdataformat_pcm() -> List[UInt8]:
    return win32_pack_guid(0x00000001, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71)


def win32_guid_ksdataformat_ieee_float() -> List[UInt8]:
    return win32_pack_guid(0x00000003, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71)


def win32_pack_waveformatex(
    sample_rate: UInt32 = 48000,
    channels: UInt16 = 2,
    bits_per_sample: UInt16 = 16,
    format_tag: UInt16 = WAVE_FORMAT_PCM,
    cb_size: UInt16 = 0,
) -> List[UInt8]:
    """Packs an 18-byte Win32 `WAVEFORMATEX` descriptor."""
    var block_align = UInt16(Int(channels) * (Int(bits_per_sample) // 8))
    var avg_bps = sample_rate * UInt32(block_align)
    var buf = List[UInt8](length=WIN32_WAVEFORMATEX_BYTES, fill=0)
    _wa_put16(buf, 0, format_tag)
    _wa_put16(buf, 2, channels)
    _wa_put32(buf, 4, sample_rate)
    _wa_put32(buf, 8, avg_bps)
    _wa_put16(buf, 12, block_align)
    _wa_put16(buf, 14, bits_per_sample)
    _wa_put16(buf, 16, cb_size)
    return buf^


def win32_pack_waveformatextensible(
    sample_rate: UInt32 = 48000,
    channels: UInt16 = 2,
    bits_per_sample: UInt16 = 16,
    channel_mask: UInt32 = 3,
    is_float: Bool = False,
) -> List[UInt8]:
    """Packs a 40-byte Win32 `WAVEFORMATEXTENSIBLE` descriptor."""
    var wfx = win32_pack_waveformatex(
        sample_rate, channels, bits_per_sample, WAVE_FORMAT_EXTENSIBLE, 22
    )
    var buf = List[UInt8](length=WIN32_WAVEFORMATEXTENSIBLE_BYTES, fill=0)
    for i in range(WIN32_WAVEFORMATEX_BYTES):
        buf[i] = wfx[i]
    _wa_put16(buf, 18, bits_per_sample)  # wValidBitsPerSample
    _wa_put32(buf, 20, channel_mask)     # dwChannelMask (3 = FL | FR)
    var sub = win32_guid_ksdataformat_ieee_float() if is_float else win32_guid_ksdataformat_pcm()
    for i in range(16):
        buf[24 + i] = sub[i]
    return buf^


def win32_pack_wavehdr(
    data_ptr: UInt64, buffer_bytes: UInt32, user_data: UInt64 = 0, flags: UInt32 = 0
) -> List[UInt8]:
    """Packs a 48-byte Win64 `WAVEHDR` header."""
    var buf = List[UInt8](length=WIN32_WAVEHDR_BYTES, fill=0)
    _wa_put64(buf, 0, data_ptr)
    _wa_put32(buf, 8, buffer_bytes)
    _wa_put32(buf, 12, 0)         # dwBytesRecorded
    _wa_put64(buf, 16, user_data)
    _wa_put32(buf, 24, flags)
    _wa_put32(buf, 28, 0)         # dwLoops
    _wa_put64(buf, 32, 0)         # lpNext
    _wa_put64(buf, 40, 0)         # reserved
    return buf^


@fieldwise_init
struct _Win32SynthVoice(Copyable, ImplicitlyCopyable, Movable):
    var active: Bool
    var clip_id: Int
    var phase: Float64
    var freq_hz: Float64
    var gain: Float32
    var age_samples: Int
    var duration_samples: Int

    def __init__(out self):
        self.active = False
        self.clip_id = 0
        self.phase = 0.0
        self.freq_hz = 220.0
        self.gain = 0.0
        self.age_samples = 0
        self.duration_samples = 0


def _com_vtable_fn(obj: Int, slot: Int) -> Int:
    if obj == 0:
        return 0
    var vtbl = Pointer[Int, ImmutAnyOrigin](unsafe_from_address=obj)[unsafe_offset=0]
    if vtbl == 0:
        return 0
    return Pointer[Int, ImmutAnyOrigin](unsafe_from_address=vtbl)[unsafe_offset=slot]


struct Win32Audio(AudioBackend):
    """Pure-Mojo Win32 `AudioBackend` (WASAPI COM + `winmm.dll` `waveOut` + 4-slot PCM ring)."""
    var sample_rate: Int
    var channels: Int
    var period_frames: Int
    var ready: Bool
    var using_wasapi: Bool
    var com_initialized: Bool
    var com_owns_uninit: Bool
    var dll_ole32: Int
    var dll_winmm: Int
    var wasapi_enumerator: Int
    var wasapi_device: Int
    var wasapi_client: Int
    var wasapi_render: Int
    var wasapi_buffer_frames: UInt32
    var h_wave_out: UInt64
    var ring_pcm: List[List[Int16]]
    var ring_hdrs: List[List[UInt8]]
    var ring_cursor: Int
    var mixed_blocks: Int
    var frames_submitted: UInt64
    var drone_level: Float32
    var drone_gain: Float32
    var drone_phase: Float64
    var voices: List[_Win32SynthVoice]
    var voice_cursor: Int

    def __init__(out self):
        self.sample_rate = WIN32_AUDIO_SAMPLE_RATE
        self.channels = 2
        self.period_frames = WIN32_AUDIO_PERIOD_FRAMES
        self.ready = False
        self.using_wasapi = False
        self.com_initialized = False
        self.com_owns_uninit = False
        self.dll_ole32 = 0
        self.dll_winmm = 0
        self.wasapi_enumerator = 0
        self.wasapi_device = 0
        self.wasapi_client = 0
        self.wasapi_render = 0
        self.wasapi_buffer_frames = 0
        self.h_wave_out = 0
        self.ring_pcm = List[List[Int16]](capacity=WIN32_AUDIO_RING_SLOTS)
        self.ring_hdrs = List[List[UInt8]](capacity=WIN32_AUDIO_RING_SLOTS)
        for s in range(WIN32_AUDIO_RING_SLOTS):
            var slot = List[Int16](length=WIN32_AUDIO_PERIOD_FRAMES * 2, fill=0)
            var ptr = UInt64(Int(slot.unsafe_ptr()))
            self.ring_hdrs.append(
                win32_pack_wavehdr(ptr, UInt32(WIN32_AUDIO_PERIOD_FRAMES * 4), UInt64(s), WHDR_DONE)
            )
            self.ring_pcm.append(slot^)
        self.ring_cursor = 0
        self.mixed_blocks = 0
        self.frames_submitted = 0
        self.drone_level = 0.0
        self.drone_gain = 0.0
        self.drone_phase = 0.0
        self.voices = List[_Win32SynthVoice](length=WIN32_AUDIO_MAX_VOICES, fill=_Win32SynthVoice())
        self.voice_cursor = 0

    def __moveinit__(out self, deinit existing: Self):
        self.sample_rate = existing.sample_rate
        self.channels = existing.channels
        self.period_frames = existing.period_frames
        self.ready = existing.ready
        self.using_wasapi = existing.using_wasapi
        self.com_initialized = existing.com_initialized
        self.com_owns_uninit = existing.com_owns_uninit
        self.dll_ole32 = existing.dll_ole32
        self.dll_winmm = existing.dll_winmm
        self.wasapi_enumerator = existing.wasapi_enumerator
        self.wasapi_device = existing.wasapi_device
        self.wasapi_client = existing.wasapi_client
        self.wasapi_render = existing.wasapi_render
        self.wasapi_buffer_frames = existing.wasapi_buffer_frames
        self.h_wave_out = existing.h_wave_out
        self.ring_pcm = existing.ring_pcm^
        self.ring_hdrs = existing.ring_hdrs^
        self.ring_cursor = existing.ring_cursor
        self.mixed_blocks = existing.mixed_blocks
        self.frames_submitted = existing.frames_submitted
        self.drone_level = existing.drone_level
        self.drone_gain = existing.drone_gain
        self.drone_phase = existing.drone_phase
        self.voices = existing.voices^
        self.voice_cursor = existing.voice_cursor
        existing.ready = False
        existing.using_wasapi = False
        existing.com_initialized = False
        existing.com_owns_uninit = False
        existing.dll_ole32 = 0
        existing.dll_winmm = 0
        existing.wasapi_enumerator = 0
        existing.wasapi_device = 0
        existing.wasapi_client = 0
        existing.wasapi_render = 0
        existing.h_wave_out = 0

    def __deinit__(deinit self):
        self.shutdown()

    def _release_wasapi(mut self):
        comptime if IS_WINDOWS:
            if self.using_wasapi and self.wasapi_client != 0:
                var fn_stop = _com_vtable_fn(self.wasapi_client, VTBL_CLIENT_STOP)
                if fn_stop != 0:
                    comptime StopFn = def(Int) thin abi("C") -> Int32
                    _ = Pointer(to=fn_stop).unsafe_bitcast[StopFn]()[unsafe_offset=0](self.wasapi_client)
            comptime RelComFn = def(Int) thin abi("C") -> UInt32
            if self.wasapi_render != 0:
                var fn_r = _com_vtable_fn(self.wasapi_render, VTBL_IUNKNOWN_RELEASE)
                if fn_r != 0:
                    _ = Pointer(to=fn_r).unsafe_bitcast[RelComFn]()[unsafe_offset=0](self.wasapi_render)
            if self.wasapi_client != 0:
                var fn_r = _com_vtable_fn(self.wasapi_client, VTBL_IUNKNOWN_RELEASE)
                if fn_r != 0:
                    _ = Pointer(to=fn_r).unsafe_bitcast[RelComFn]()[unsafe_offset=0](self.wasapi_client)
            if self.wasapi_device != 0:
                var fn_r = _com_vtable_fn(self.wasapi_device, VTBL_IUNKNOWN_RELEASE)
                if fn_r != 0:
                    _ = Pointer(to=fn_r).unsafe_bitcast[RelComFn]()[unsafe_offset=0](self.wasapi_device)
            if self.wasapi_enumerator != 0:
                var fn_r = _com_vtable_fn(self.wasapi_enumerator, VTBL_IUNKNOWN_RELEASE)
                if fn_r != 0:
                    _ = Pointer(to=fn_r).unsafe_bitcast[RelComFn]()[unsafe_offset=0](self.wasapi_enumerator)
            if self.com_owns_uninit and self.dll_ole32 != 0:
                var fn_uninit = _win32_get_proc(self.dll_ole32, "CoUninitialize")
                if fn_uninit != 0:
                    comptime UninitFn = def() thin abi("C") -> None
                    Pointer(to=fn_uninit).unsafe_bitcast[UninitFn]()[unsafe_offset=0]()
            _win32_free_dll(self.dll_ole32)
        self.using_wasapi = False
        self.com_initialized = False
        self.com_owns_uninit = False
        self.dll_ole32 = 0
        self.wasapi_enumerator = 0
        self.wasapi_device = 0
        self.wasapi_client = 0
        self.wasapi_render = 0
        self.wasapi_buffer_frames = 0

    def _try_init_wasapi(mut self) -> Bool:
        comptime if IS_WINDOWS:
            self.dll_ole32 = _win32_load_dll("ole32.dll")
            if self.dll_ole32 == 0:
                return False
            var fn_co_init = _win32_get_proc(self.dll_ole32, "CoInitializeEx")
            var fn_co_create = _win32_get_proc(self.dll_ole32, "CoCreateInstance")
            if fn_co_init == 0 or fn_co_create == 0:
                self._release_wasapi()
                return False

            comptime CoInitFn = def(Int, UInt32) thin abi("C") -> Int32
            var hr_init = Pointer(to=fn_co_init).unsafe_bitcast[CoInitFn]()[unsafe_offset=0](
                0, COINIT_MULTITHREADED
            )
            if hr_init == 0 or hr_init == 1:
                self.com_initialized = True
                self.com_owns_uninit = True
            elif hr_init == RPC_E_CHANGED_MODE:
                # Thread already initialized COM (e.g. STA); proceed without owning CoUninitialize.
                self.com_initialized = True
                self.com_owns_uninit = False
            else:
                self._release_wasapi()
                return False

            var clsid_enum = win32_guid_clsid_mm_device_enumerator()
            var iid_enum = win32_guid_iid_imm_device_enumerator()
            var p_enum = 0
            comptime CoCreateFn = def(Int, Int, UInt32, Int, Int) thin abi("C") -> Int32
            var hr = Pointer(to=fn_co_create).unsafe_bitcast[CoCreateFn]()[unsafe_offset=0](
                Int(clsid_enum.unsafe_ptr()),
                0,
                CLSCTX_ALL,
                Int(iid_enum.unsafe_ptr()),
                Int(Pointer(to=p_enum)),
            )
            _ = len(clsid_enum)
            _ = len(iid_enum)
            if hr < 0 or p_enum == 0:
                self._release_wasapi()
                return False
            self.wasapi_enumerator = p_enum

            var fn_ep = _com_vtable_fn(self.wasapi_enumerator, VTBL_ENUM_GET_DEFAULT_ENDPOINT)
            if fn_ep == 0:
                self._release_wasapi()
                return False
            var p_dev = 0
            comptime GetEpFn = def(Int, UInt32, UInt32, Int) thin abi("C") -> Int32
            hr = Pointer(to=fn_ep).unsafe_bitcast[GetEpFn]()[unsafe_offset=0](
                self.wasapi_enumerator, 0, 0, Int(Pointer(to=p_dev))
            )
            if hr < 0 or p_dev == 0:
                self._release_wasapi()
                return False
            self.wasapi_device = p_dev

            var iid_client = win32_guid_iid_iaudio_client()
            var fn_act = _com_vtable_fn(self.wasapi_device, VTBL_DEVICE_ACTIVATE)
            if fn_act == 0:
                self._release_wasapi()
                return False
            var p_client = 0
            comptime ActFn = def(Int, Int, UInt32, Int, Int) thin abi("C") -> Int32
            hr = Pointer(to=fn_act).unsafe_bitcast[ActFn]()[unsafe_offset=0](
                self.wasapi_device,
                Int(iid_client.unsafe_ptr()),
                CLSCTX_ALL,
                0,
                Int(Pointer(to=p_client)),
            )
            _ = len(iid_client)
            if hr < 0 or p_client == 0:
                self._release_wasapi()
                return False
            self.wasapi_client = p_client

            var wfx = win32_pack_waveformatex(
                UInt32(self.sample_rate), UInt16(self.channels), 16, WAVE_FORMAT_PCM
            )
            var fn_aud_init = _com_vtable_fn(self.wasapi_client, VTBL_CLIENT_INITIALIZE)
            if fn_aud_init == 0:
                _ = len(wfx)
                self._release_wasapi()
                return False
            comptime AudInitFn = def(Int, UInt32, UInt32, Int64, Int64, Int, Int) thin abi("C") -> Int32
            hr = Pointer(to=fn_aud_init).unsafe_bitcast[AudInitFn]()[unsafe_offset=0](
                self.wasapi_client,
                AUDCLNT_SHAREMODE_SHARED,
                0x80000000,  # AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
                Int64(500000),  # 50 ms in 100ns units
                Int64(0),
                Int(wfx.unsafe_ptr()),
                0,
            )
            _ = len(wfx)
            if hr < 0:
                self._release_wasapi()
                return False

            var fn_bufsz = _com_vtable_fn(self.wasapi_client, VTBL_CLIENT_GET_BUFFER_SIZE)
            var bsz: UInt32 = 0
            if fn_bufsz != 0:
                comptime BufSzFn = def(Int, Int) thin abi("C") -> Int32
                _ = Pointer(to=fn_bufsz).unsafe_bitcast[BufSzFn]()[unsafe_offset=0](
                    self.wasapi_client, Int(Pointer(to=bsz))
                )
            self.wasapi_buffer_frames = bsz

            var iid_render = win32_guid_iid_iaudio_render_client()
            var fn_svc = _com_vtable_fn(self.wasapi_client, VTBL_CLIENT_GET_SERVICE)
            if fn_svc == 0:
                _ = len(iid_render)
                self._release_wasapi()
                return False
            var p_render = 0
            comptime SvcFn = def(Int, Int, Int) thin abi("C") -> Int32
            hr = Pointer(to=fn_svc).unsafe_bitcast[SvcFn]()[unsafe_offset=0](
                self.wasapi_client, Int(iid_render.unsafe_ptr()), Int(Pointer(to=p_render))
            )
            _ = len(iid_render)
            if hr < 0 or p_render == 0:
                self._release_wasapi()
                return False
            self.wasapi_render = p_render

            var fn_start = _com_vtable_fn(self.wasapi_client, VTBL_CLIENT_START)
            if fn_start == 0:
                self._release_wasapi()
                return False
            comptime StartFn = def(Int) thin abi("C") -> Int32
            hr = Pointer(to=fn_start).unsafe_bitcast[StartFn]()[unsafe_offset=0](self.wasapi_client)
            if hr < 0:
                self._release_wasapi()
                return False
            self.using_wasapi = True
            return True
        else:
            return False

    def _try_init_waveout(mut self) -> Bool:
        comptime if IS_WINDOWS:
            self.dll_winmm = _win32_load_dll("winmm.dll")
            if self.dll_winmm == 0:
                return False
            var fn_open = _win32_get_proc(self.dll_winmm, "waveOutOpen")
            if fn_open == 0:
                return False
            var wfx = win32_pack_waveformatex(
                UInt32(self.sample_rate), UInt16(self.channels), 16, WAVE_FORMAT_PCM
            )
            var hwo: UInt64 = 0
            comptime WaveOpenFn = def(Int, UInt32, Int, UInt64, UInt64, UInt32) thin abi("C") -> UInt32
            var rc = Pointer(to=fn_open).unsafe_bitcast[WaveOpenFn]()[unsafe_offset=0](
                Int(Pointer(to=hwo)),
                WAVE_MAPPER,
                Int(wfx.unsafe_ptr()),
                0,
                0,
                CALLBACK_NULL,
            )
            _ = len(wfx)
            if rc != 0 or hwo == 0:
                return False
            self.h_wave_out = hwo
            self.using_wasapi = False
            return True
        else:
            return False

    def initialize(mut self) -> Bool:
        self.shutdown()
        comptime if IS_WINDOWS:
            if self._try_init_wasapi():
                self.ready = True
                return True
            if self._try_init_waveout():
                self.ready = True
                return True
            self.shutdown()
            return False
        else:
            return False

    def trigger(mut self, clip_id: Int, pitch: Float32, gain: Float32):
        var slot = self.voice_cursor % WIN32_AUDIO_MAX_VOICES
        self.voice_cursor += 1
        var base_hz = 196.0 + Float64(abs(clip_id) % 12) * 32.0
        var p = max(Float64(0.25), min(Float64(4.0), Float64(pitch)))
        self.voices[slot] = _Win32SynthVoice(
            True,
            clip_id,
            0.0,
            base_hz * p,
            max(Float32(0.0), min(Float32(2.0), gain)),
            0,
            self.sample_rate // 4,
        )

    def set_ambient_drone(mut self, level: Float32):
        self.drone_level = max(Float32(0.0), min(Float32(1.0), level))

    def mix_into_slot(mut self, slot_idx: Int, master_gain: Float32, frames: Int):
        var mg = max(Float32(0.0), min(Float32(2.0), master_gain))
        var count = max(1, min(self.period_frames, frames))
        var two_pi = 6.283185307179586
        var inv_sr = 1.0 / Float64(self.sample_rate)
        for f in range(count):
            var mix_l = Float32(0.0)
            var mix_r = Float32(0.0)
            # Drone smoothing + harmonic pair
            self.drone_gain += (self.drone_level - self.drone_gain) * Float32(0.002)
            if self.drone_gain > Float32(1e-5):
                var d = Float32(sin(self.drone_phase * two_pi)) * self.drone_gain * Float32(0.22)
                mix_l += d
                mix_r += d * Float32(0.96)
                self.drone_phase += 55.0 * inv_sr
                if self.drone_phase >= 1.0:
                    self.drone_phase -= 1.0
            # Polyphonic synth voices
            for v in range(WIN32_AUDIO_MAX_VOICES):
                if not self.voices[v].active:
                    continue
                var age = self.voices[v].age_samples
                var dur = max(1, self.voices[v].duration_samples)
                var env = Float32(1.0 - Float64(age) / Float64(dur))
                var sig = Float32(sin(self.voices[v].phase * two_pi)) * self.voices[v].gain * env * Float32(0.35)
                var pan = Float32(((self.voices[v].clip_id % 5) - 2)) * Float32(0.15)
                mix_l += sig * (Float32(1.0) - max(Float32(0.0), pan))
                mix_r += sig * (Float32(1.0) + min(Float32(0.0), pan))
                self.voices[v].phase += self.voices[v].freq_hz * inv_sr
                if self.voices[v].phase >= 1.0:
                    self.voices[v].phase -= 1.0
                self.voices[v].age_samples += 1
                if self.voices[v].age_samples >= dur:
                    self.voices[v].active = False

            mix_l = soft_clip_tanh(mix_l * mg)
            mix_r = soft_clip_tanh(mix_r * mg)
            self.ring_pcm[slot_idx][f * 2] = Int16(
                Int(max(Float32(-1.0), min(Float32(1.0), mix_l)) * Float32(32767.0))
            )
            self.ring_pcm[slot_idx][f * 2 + 1] = Int16(
                Int(max(Float32(-1.0), min(Float32(1.0), mix_r)) * Float32(32767.0))
            )

    def pump(mut self, master_gain: Float32):
        var slot = self.ring_cursor % WIN32_AUDIO_RING_SLOTS
        comptime if IS_WINDOWS:
            if self.ready and not self.using_wasapi and self.h_wave_out != 0 and self.dll_winmm != 0:
                var flags = _wa_get32(self.ring_hdrs[slot], 24)
                if (flags & WHDR_DONE) == 0:
                    return
                if (flags & WHDR_PREPARED) != 0:
                    var fn_unprep = _win32_get_proc(self.dll_winmm, "waveOutUnprepareHeader")
                    if fn_unprep != 0:
                        comptime WaveUnprepFn = def(UInt64, Int, UInt32) thin abi("C") -> UInt32
                        _ = Pointer(to=fn_unprep).unsafe_bitcast[WaveUnprepFn]()[unsafe_offset=0](
                            self.h_wave_out,
                            Int(self.ring_hdrs[slot].unsafe_ptr()),
                            UInt32(WIN32_WAVEHDR_BYTES),
                        )
        self.mix_into_slot(slot, master_gain, self.period_frames)
        self.ring_cursor = (slot + 1) % WIN32_AUDIO_RING_SLOTS
        self.mixed_blocks += 1

        comptime if IS_WINDOWS:
            if not self.ready:
                return
            if self.using_wasapi and self.wasapi_client != 0 and self.wasapi_render != 0:
                var fn_pad = _com_vtable_fn(self.wasapi_client, VTBL_CLIENT_GET_CURRENT_PADDING)
                var pad: UInt32 = 0
                comptime PadFn = def(Int, Int) thin abi("C") -> Int32
                if Pointer(to=fn_pad).unsafe_bitcast[PadFn]()[unsafe_offset=0](
                    self.wasapi_client, Int(Pointer(to=pad))
                ) >= 0:
                    var avail = Int(self.wasapi_buffer_frames) - Int(pad)
                    var to_write = min(avail, self.period_frames)
                    if to_write > 0:
                        var fn_get = _com_vtable_fn(self.wasapi_render, VTBL_RENDER_GET_BUFFER)
                        var fn_rel = _com_vtable_fn(self.wasapi_render, VTBL_RENDER_RELEASE_BUFFER)
                        var dst_ptr = 0
                        comptime GetBufFn = def(Int, UInt32, Int) thin abi("C") -> Int32
                        comptime RelBufFn = def(Int, UInt32, UInt32) thin abi("C") -> Int32
                        if Pointer(to=fn_get).unsafe_bitcast[GetBufFn]()[unsafe_offset=0](
                            self.wasapi_render, UInt32(to_write), Int(Pointer(to=dst_ptr))
                        ) >= 0 and dst_ptr != 0:
                            _ = external_call["memcpy", Int](
                                dst_ptr, Int(self.ring_pcm[slot].unsafe_ptr()), Int64(to_write * 4)
                            )
                            _ = Pointer(to=fn_rel).unsafe_bitcast[RelBufFn]()[unsafe_offset=0](
                                self.wasapi_render, UInt32(to_write), 0
                            )
                            self.frames_submitted += UInt64(to_write)
            elif self.h_wave_out != 0 and self.dll_winmm != 0:
                var fn_prep = _win32_get_proc(self.dll_winmm, "waveOutPrepareHeader")
                var fn_write = _win32_get_proc(self.dll_winmm, "waveOutWrite")
                var fn_unprep = _win32_get_proc(self.dll_winmm, "waveOutUnprepareHeader")
                if fn_prep != 0 and fn_write != 0:
                    var pcm_ptr = UInt64(Int(self.ring_pcm[slot].unsafe_ptr()))
                    _wa_put64(self.ring_hdrs[slot], 0, pcm_ptr)
                    _wa_put32(self.ring_hdrs[slot], 8, UInt32(self.period_frames * 4))
                    _wa_put32(self.ring_hdrs[slot], 24, 0)
                    comptime WaveHdrFn = def(UInt64, Int, UInt32) thin abi("C") -> UInt32
                    var hdr_ptr = Int(self.ring_hdrs[slot].unsafe_ptr())
                    if Pointer(to=fn_prep).unsafe_bitcast[WaveHdrFn]()[unsafe_offset=0](
                        self.h_wave_out, hdr_ptr, UInt32(WIN32_WAVEHDR_BYTES)
                    ) == 0:
                        if Pointer(to=fn_write).unsafe_bitcast[WaveHdrFn]()[unsafe_offset=0](
                            self.h_wave_out, hdr_ptr, UInt32(WIN32_WAVEHDR_BYTES)
                        ) == 0:
                            self.frames_submitted += UInt64(self.period_frames)
                        elif fn_unprep != 0:
                            _ = Pointer(to=fn_unprep).unsafe_bitcast[WaveHdrFn]()[unsafe_offset=0](
                                self.h_wave_out, hdr_ptr, UInt32(WIN32_WAVEHDR_BYTES)
                            )
                            _wa_put32(self.ring_hdrs[slot], 24, WHDR_DONE)

    def last_mixed_slot(self) -> List[Int16]:
        if self.mixed_blocks == 0:
            return List[Int16]()
        var prev = (self.ring_cursor + WIN32_AUDIO_RING_SLOTS - 1) % WIN32_AUDIO_RING_SLOTS
        return self.ring_pcm[prev].copy()

    def shutdown(mut self):
        self._release_wasapi()
        comptime if IS_WINDOWS:
            if self.h_wave_out != 0 and self.dll_winmm != 0:
                var fn_reset = _win32_get_proc(self.dll_winmm, "waveOutReset")
                var fn_unprep = _win32_get_proc(self.dll_winmm, "waveOutUnprepareHeader")
                var fn_close = _win32_get_proc(self.dll_winmm, "waveOutClose")
                comptime WaveOpFn = def(UInt64) thin abi("C") -> UInt32
                comptime WaveHdrFn = def(UInt64, Int, UInt32) thin abi("C") -> UInt32
                if fn_reset != 0:
                    _ = Pointer(to=fn_reset).unsafe_bitcast[WaveOpFn]()[unsafe_offset=0](self.h_wave_out)
                if fn_unprep != 0:
                    for s in range(len(self.ring_hdrs)):
                        if (_wa_get32(self.ring_hdrs[s], 24) & WHDR_PREPARED) != 0:
                            _ = Pointer(to=fn_unprep).unsafe_bitcast[WaveHdrFn]()[unsafe_offset=0](
                                self.h_wave_out,
                                Int(self.ring_hdrs[s].unsafe_ptr()),
                                UInt32(WIN32_WAVEHDR_BYTES),
                            )
                if fn_close != 0:
                    _ = Pointer(to=fn_close).unsafe_bitcast[WaveOpFn]()[unsafe_offset=0](self.h_wave_out)
            _win32_free_dll(self.dll_winmm)
        for s in range(len(self.ring_hdrs)):
            _wa_put32(self.ring_hdrs[s], 24, WHDR_DONE)
        self.ready = False
        self.dll_winmm = 0
        self.h_wave_out = 0
