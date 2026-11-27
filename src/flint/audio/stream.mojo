"""Bounded-Memory Disk Streaming for Long WAV & QOA Sounds (`flint.audio.stream`).

Long music / VO files are never read whole. `probe_stream_source(path)` parses only the
file header; playback claims one of `STREAM_MAX_CHANNELS` channels whose decoded float
ring (`STREAM_RING_FRAMES` stereo frames, ~0.34 s @ 48 kHz) is refilled from disk with
`pread(2)` in `STREAM_CHUNK_FRAMES` chunks by a background streamer pthread (or
synchronously by `service_streams()` in offline/headless renders).

Threading contract (lock-free, single producer / single consumer per channel):
  - Control block (64 B, heap-pinned, raw atomics) per channel:
      `[0] state`, `[8] write_pos`, `[16] read_pos`, `[24] owner_slot`, `[32] owner_gen`,
      `[40] total_frames`, `[48] ring_addr`, `[56] looping`.
  - States: `FREE -> CLAIMED` (main thread opens & prefills) `-> ARMED` (main thread,
    then sends ACMD_PLAY) `-> ACTIVE` (RT thread on ACMD_PLAY) `-> RELEASING` (RT thread
    when its emitter dies, or main thread when the PLAY command could not be queued)
    `-> FREE` (the servicing thread closes the file).
  - Positions are monotonic *stream* frame indices (they keep counting across loops);
    the producer maps them to file frames (`s mod total` when looping). The producer
    never writes past `read_pos + STREAM_RING_FRAMES - 8`; when the consumer jumps ahead
    (virtualized voice), the producer re-seeks the file to the new `read_pos`.
"""
from std.atomic import Atomic
from std.collections import List
from std.ffi import external_call
from flint.assets.flpk_v3 import AUDIO_STREAM_PAGE_BYTES, crc64_ecma_ptr, _ptr_u64
from flint.platform.os import open_read, close_fd, file_size
from flint.audio.qoa import (
    QOA_MAGIC,
    QOA_SLICE_LEN,
    QOA_FRAME_LEN,
    QoaLms,
    qoa_full_frame_bytes,
    _qoa_dequant,
    _clamp_i16,
)

comptime STREAM_MAX_CHANNELS: Int = 16
comptime STREAM_RING_FRAMES: Int = 16384
comptime STREAM_RING_MASK: Int = STREAM_RING_FRAMES - 1
comptime STREAM_CHUNK_FRAMES: Int = 2048
comptime STREAM_PREFILL_FRAMES: Int = 4096
comptime STREAM_MARGIN_FRAMES: Int = 8

comptime STREAM_FMT_NONE: Int = 0
comptime STREAM_FMT_WAV16: Int = 1
comptime STREAM_FMT_QOA: Int = 2

comptime STREAM_STATE_FREE: Int64 = 0
comptime STREAM_STATE_CLAIMED: Int64 = 1
comptime STREAM_STATE_ARMED: Int64 = 2
comptime STREAM_STATE_ACTIVE: Int64 = 3
comptime STREAM_STATE_RELEASING: Int64 = 4

comptime _SC_STATE: Int = 0
comptime _SC_WRITE: Int = 8
comptime _SC_READ: Int = 16
comptime _SC_OWNER_SLOT: Int = 24
comptime _SC_OWNER_GEN: Int = 32
comptime _SC_TOTAL: Int = 40
comptime _SC_RING: Int = 48
comptime _SC_LOOP: Int = 56
comptime STREAM_CTRL_STRIDE: Int = 64


@always_inline
def stream_ctrl_atomic(ctrl_base: Int, ch: Int, off: Int) -> Pointer[Atomic[Int64], MutAnyOrigin]:
    return Pointer[Atomic[Int64], MutAnyOrigin](
        unsafe_from_address=ctrl_base + ch * STREAM_CTRL_STRIDE + off
    )


@always_inline
def stream_load(ctrl_base: Int, ch: Int, off: Int) -> Int64:
    return stream_ctrl_atomic(ctrl_base, ch, off)[unsafe_offset=0].load()


@always_inline
def stream_store(ctrl_base: Int, ch: Int, off: Int, v: Int64):
    stream_ctrl_atomic(ctrl_base, ch, off)[unsafe_offset=0].store(v)


@always_inline
def stream_ring_sample(ctrl_base: Int, ch: Int, idx: Int, lr: Int) -> Float32:
    """Consumer-side read of decoded stream frame `idx` (0 if not yet produced / out of range)."""
    if idx < 0:
        return 0.0
    var total = Int(stream_load(ctrl_base, ch, _SC_TOTAL))
    var looping = stream_load(ctrl_base, ch, _SC_LOOP) != 0
    if not looping and idx >= total:
        return 0.0
    var wp = Int(stream_load(ctrl_base, ch, _SC_WRITE))
    if idx >= wp or idx < wp - STREAM_RING_FRAMES:
        return 0.0
    var ring = Int(stream_load(ctrl_base, ch, _SC_RING))
    if ring == 0:
        return 0.0
    return Pointer[Float32, MutAnyOrigin](unsafe_from_address=ring)[
        unsafe_offset=(idx & STREAM_RING_MASK) * 2 + lr
    ]


@always_inline
def stream_frame_ready(ctrl_base: Int, ch: Int, idx: Int) -> Bool:
    var wp = Int(stream_load(ctrl_base, ch, _SC_WRITE))
    return idx < wp


@always_inline
def stream_publish_read(ctrl_base: Int, ch: Int, consumer_pos: Float64):
    """Consumer publishes how far it has read (keeps 2 frames of interpolation history)."""
    var rp = Int(consumer_pos) - 2
    if rp < 0:
        rp = 0
    var cur = Int(stream_load(ctrl_base, ch, _SC_READ))
    if rp > cur:
        stream_store(ctrl_base, ch, _SC_READ, Int64(rp))


# ==============================================================================
# Header probing
# ==============================================================================

struct StreamSourceInfo(Copyable, Movable):
    """Header-only description of a streamable file (no sample data is read)."""
    var valid: Bool
    var format: Int
    var path: String
    var channels: Int
    var sample_rate: Int
    var total_frames: Int
    var data_offset: Int
    var block_align: Int
    var file_bytes: Int  # absolute end of the readable region
    var archive_fd: Int
    var region_start: Int
    var checksums_offset: Int
    var archive_key: String

    def __init__(out self):
        self.valid = False
        self.format = STREAM_FMT_NONE
        self.path = ""
        self.channels = 0
        self.sample_rate = 0
        self.total_frames = 0
        self.data_offset = 0
        self.block_align = 0
        self.file_bytes = 0
        self.archive_fd = -1
        self.region_start = 0
        self.checksums_offset = -1
        self.archive_key = ""


def _pread_fd(fd: Int, dst_addr: Int, count: Int, offset: Int) -> Int:
    """Reads up to `count` bytes at `offset`; returns bytes read (short only at EOF)."""
    var done = 0
    while done < count:
        var got = Int(
            external_call["pread", Int64](
                Int32(fd), dst_addr + done, Int64(count - done), Int64(offset + done)
            )
        )
        if got < 0:
            var err = Int(external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()[unsafe_offset=0])
            if err == 4:
                continue
            return done
        if got == 0:
            return done
        done += got
    return done


@always_inline
def _le16(b: List[UInt8], o: Int) -> Int:
    return Int(b[o]) | (Int(b[o + 1]) << 8)


@always_inline
def _le32(b: List[UInt8], o: Int) -> Int:
    return Int(b[o]) | (Int(b[o + 1]) << 8) | (Int(b[o + 2]) << 16) | (Int(b[o + 3]) << 24)


@always_inline
def _be32(b: List[UInt8], o: Int) -> Int:
    return (Int(b[o]) << 24) | (Int(b[o + 1]) << 16) | (Int(b[o + 2]) << 8) | Int(b[o + 3])


def probe_stream_source(path: String) -> StreamSourceInfo:
    """Parses a 16-bit PCM `.wav` (RIFF chunk walk) or `.qoa` header for streaming."""
    var size = file_size(path)
    if size < 16:
        return StreamSourceInfo()
    var fd = open_read(path)
    if fd < 0:
        return StreamSourceInfo()
    var info = probe_stream_region(path, fd, 0, size)
    close_fd(fd)
    return info^


def probe_stream_region(
    path: String, fd: Int, start: Int, size: Int, cooked_pcm_magic: UInt32 = 0,
) -> StreamSourceInfo:
    """Read only a stream header within a validated byte range of an open file."""
    var info = StreamSourceInfo()
    if fd < 0 or start < 0 or size < 16 or start > 0x7fffffffffffffff - size:
        return info^
    info.path = path
    info.file_bytes = start + size
    info.region_start = start
    var hdr = List[UInt8](length=16, fill=0)
    if _pread_fd(fd, Int(hdr.unsafe_ptr()), 16, start) < 16:
        return info^
    if cooked_pcm_magic != 0 and UInt32(_le32(hdr, 0)) == cooked_pcm_magic:
        var channels = _le32(hdr, 4)
        var rate = _le32(hdr, 8)
        var samples = _le32(hdr, 12)
        if (
            channels < 1 or channels > 8 or rate < 1 or rate > 768000
            or samples < channels or samples % channels != 0 or samples > (size - 16) // 2
        ):
            return info^
        info.valid = True
        info.format = STREAM_FMT_WAV16
        info.channels = channels
        info.sample_rate = rate
        info.total_frames = samples // channels
        info.block_align = channels * 2
        info.data_offset = start + 16
        return info^
    if UInt32(_be32(hdr, 0)) == QOA_MAGIC:
        var total = _be32(hdr, 4)
        var ch = Int(hdr[8])
        var sr = (Int(hdr[9]) << 16) | (Int(hdr[10]) << 8) | Int(hdr[11])
        if total <= 0 or ch < 1 or ch > 8 or sr <= 0:
            return info^
        info.valid = True
        info.format = STREAM_FMT_QOA
        info.channels = ch
        info.sample_rate = sr
        info.total_frames = total
        info.data_offset = start + 8
        return info^

    # RIFF/WAVE
    if not (hdr[0] == 82 and hdr[1] == 73 and hdr[2] == 70 and hdr[3] == 70
            and hdr[8] == 87 and hdr[9] == 65 and hdr[10] == 86 and hdr[11] == 69):
        return info^
    var off = 12
    var have_fmt = False
    var fmt_tag = 0
    var ch = 0
    var sr = 0
    var bits = 0
    var align = 0
    var chunk = List[UInt8](length=40, fill=0)
    var guard = 0
    while off + 8 <= size and guard < 256:
        guard += 1
        if _pread_fd(fd, Int(chunk.unsafe_ptr()), 8, start + off) < 8:
            break
        var csize = _le32(chunk, 4)
        var is_fmt = chunk[0] == 102 and chunk[1] == 109 and chunk[2] == 116 and chunk[3] == 32
        var is_data = chunk[0] == 100 and chunk[1] == 97 and chunk[2] == 116 and chunk[3] == 97
        if is_fmt and csize >= 16:
            if size - off < 24 or _pread_fd(fd, Int(chunk.unsafe_ptr()), 16, start + off + 8) < 16:
                break
            fmt_tag = _le16(chunk, 0)
            ch = _le16(chunk, 2)
            sr = _le32(chunk, 4)
            align = _le16(chunk, 12)
            bits = _le16(chunk, 14)
            have_fmt = True
        elif is_data and have_fmt:
            var data_size = min(csize, size - (off + 8))
            if (fmt_tag == 1 or fmt_tag == 0xFFFE) and bits == 16 and ch >= 1 and ch <= 8 and sr > 0 and align == ch * 2:
                info.valid = data_size >= align
                info.format = STREAM_FMT_WAV16
                info.channels = ch
                info.sample_rate = sr
                info.block_align = align
                info.data_offset = start + off + 8
                info.total_frames = data_size // align
            break
        off += 8 + csize + (csize & 1)
    return info^


# ==============================================================================
# Producer-side per-channel state + heap-pinned pool
# ==============================================================================

struct _StreamProducer(Movable):
    var fd: Int
    var format: Int
    var channels: Int
    var total_frames: Int
    var data_offset: Int
    var block_align: Int
    var file_bytes: Int
    var looping: Bool
    var byte_buf: List[UInt8]
    var qoa_cache: List[Int16]
    var qoa_cached_frame: Int
    var qoa_cached_len: Int
    var region_start: Int
    var checksums_offset: Int
    var cached_page: Int
    var page_buffer: List[UInt8]

    def __init__(out self):
        self.fd = -1
        self.format = STREAM_FMT_NONE
        self.channels = 0
        self.total_frames = 0
        self.data_offset = 0
        self.block_align = 0
        self.file_bytes = 0
        self.looping = False
        self.byte_buf = List[UInt8]()
        self.qoa_cache = List[Int16]()
        self.qoa_cached_frame = -1
        self.qoa_cached_len = 0
        self.region_start = 0
        self.checksums_offset = -1
        self.cached_page = -1
        self.page_buffer = List[UInt8]()

    def read_samples(mut self, dest: Int, count: Int, offset: Int) -> Int:
        if offset < self.region_start or count < 0 or offset > self.file_bytes - count:
            return 0
        if self.checksums_offset < 0:
            return _pread_fd(self.fd, dest, count, offset)
        var done = 0
        while done < count:
            var rel = offset + done - self.region_start
            var index = rel // AUDIO_STREAM_PAGE_BYTES
            var base = self.region_start + index * AUDIO_STREAM_PAGE_BYTES
            var size = min(AUDIO_STREAM_PAGE_BYTES, self.file_bytes - base)
            if self.cached_page != index:
                self.cached_page = -1
                var expected = Array[UInt8, 8](fill=0)
                var addr = Int(self.page_buffer.unsafe_ptr())
                if (
                    _pread_fd(self.fd, Int(Pointer(to=expected)), 8, self.checksums_offset + index * 8) != 8
                    or _pread_fd(self.fd, addr, size, base) != size
                    or crc64_ecma_ptr(addr, size, 0) != _ptr_u64(Int(Pointer(to=expected)), 0)
                ):
                    return done
                self.cached_page = index
            var page_pos = rel % AUDIO_STREAM_PAGE_BYTES
            var take = min(count - done, size - page_pos)
            _ = external_call["memcpy", Int](dest + done, Int(self.page_buffer.unsafe_ptr()) + page_pos, take)
            done += take
        return done


struct AudioStreamPool(Movable):
    """Heap-pinned pool of streaming channels (see module docs for the threading contract)."""
    var ctrl_addr: Int
    var running_addr: Int
    var producers: List[_StreamProducer]
    var chunks_read: Int

    def __init__(out self):
        self.ctrl_addr = external_call["calloc", Int](Int(1), STREAM_MAX_CHANNELS * STREAM_CTRL_STRIDE + 64)
        self.running_addr = external_call["calloc", Int](Int(1), 64)
        self.producers = List[_StreamProducer]()
        for _ in range(STREAM_MAX_CHANNELS):
            self.producers.append(_StreamProducer())
        self.chunks_read = 0

    def destroy(mut self):
        for c in range(STREAM_MAX_CHANNELS):
            if self.producers[c].fd >= 0:
                close_fd(self.producers[c].fd)
                self.producers[c].fd = -1
            if self.ctrl_addr != 0:
                var ring = Int(stream_load(self.ctrl_addr, c, _SC_RING))
                if ring != 0:
                    external_call["free", NoneType](ring)
                    stream_store(self.ctrl_addr, c, _SC_RING, 0)
        if self.ctrl_addr != 0:
            external_call["free", NoneType](self.ctrl_addr)
            self.ctrl_addr = 0
        if self.running_addr != 0:
            external_call["free", NoneType](self.running_addr)
            self.running_addr = 0

    @always_inline
    def state(self, ch: Int) -> Int64:
        return stream_load(self.ctrl_addr, ch, _SC_STATE)

    def active_count(self) -> Int:
        var n = 0
        for c in range(STREAM_MAX_CHANNELS):
            var s = self.state(c)
            if s != STREAM_STATE_FREE:
                n += 1
        return n

    def claim(mut self, info: StreamSourceInfo, looping: Bool, owner_slot: Int, owner_gen: UInt32, start_frame: Int = 0) -> Int:
        """MAIN THREAD: opens `info.path` on a free channel and prefills it. Returns the
        channel index (state ARMED) or `-1` when no channel/file is available."""
        if not info.valid or self.ctrl_addr == 0 or start_frame < 0 or start_frame >= info.total_frames:
            return -1
        var ch = -1
        for c in range(STREAM_MAX_CHANNELS):
            if self.state(c) == STREAM_STATE_FREE:
                ch = c
                break
        if ch < 0:
            return -1
        stream_store(self.ctrl_addr, ch, _SC_STATE, STREAM_STATE_CLAIMED)
        var fd = (
            Int(external_call["fcntl", Int32](Int32(info.archive_fd), Int32(1030), Int32(0)))
            if info.archive_fd >= 0 else open_read(info.path)
        )
        if fd < 0:
            stream_store(self.ctrl_addr, ch, _SC_STATE, STREAM_STATE_FREE)
            return -1
        if stream_load(self.ctrl_addr, ch, _SC_RING) == 0:
            var ring = external_call["calloc", Int](Int(1), STREAM_RING_FRAMES * 2 * 4)
            if ring == 0:
                close_fd(fd)
                stream_store(self.ctrl_addr, ch, _SC_STATE, STREAM_STATE_FREE)
                return -1
            stream_store(self.ctrl_addr, ch, _SC_RING, Int64(ring))
        ref p = self.producers[ch]
        p.fd = fd
        p.format = info.format
        p.channels = info.channels
        p.total_frames = info.total_frames
        p.data_offset = info.data_offset
        p.block_align = info.block_align
        p.file_bytes = info.file_bytes
        p.region_start = info.region_start
        p.checksums_offset = info.checksums_offset
        p.cached_page = -1
        if info.checksums_offset >= 0 and len(p.page_buffer) < AUDIO_STREAM_PAGE_BYTES:
            p.page_buffer.resize(AUDIO_STREAM_PAGE_BYTES, 0)
        p.looping = looping
        p.qoa_cached_frame = -1
        p.qoa_cached_len = 0
        if info.format == STREAM_FMT_WAV16:
            var need = STREAM_CHUNK_FRAMES * info.block_align
            if len(p.byte_buf) < need:
                p.byte_buf.resize(need, 0)
        else:
            var need_b = qoa_full_frame_bytes(info.channels)
            if len(p.byte_buf) < need_b:
                p.byte_buf.resize(need_b, 0)
            var need_s = QOA_FRAME_LEN * info.channels
            if len(p.qoa_cache) < need_s:
                p.qoa_cache.resize(need_s, 0)
        var first = max(0, start_frame - 2)  # include Hermite interpolation history
        stream_store(self.ctrl_addr, ch, _SC_WRITE, Int64(first))
        stream_store(self.ctrl_addr, ch, _SC_READ, Int64(first))
        stream_store(self.ctrl_addr, ch, _SC_OWNER_SLOT, Int64(owner_slot))
        stream_store(self.ctrl_addr, ch, _SC_OWNER_GEN, Int64(owner_gen))
        stream_store(self.ctrl_addr, ch, _SC_TOTAL, Int64(info.total_frames))
        stream_store(self.ctrl_addr, ch, _SC_LOOP, Int64(1) if looping else Int64(0))
        # Synchronous prefill so the first quantum never underruns.
        var guard = 0
        while guard < 8 and Int(stream_load(self.ctrl_addr, ch, _SC_WRITE)) < first + STREAM_PREFILL_FRAMES:
            if not self._service_channel(ch):
                break
            guard += 1
        stream_store(self.ctrl_addr, ch, _SC_STATE, STREAM_STATE_ARMED)
        return ch

    def release_from_main(mut self, ch: Int):
        """MAIN THREAD: abandons an ARMED channel whose PLAY command was never queued."""
        if ch >= 0 and ch < STREAM_MAX_CHANNELS:
            stream_store(self.ctrl_addr, ch, _SC_STATE, STREAM_STATE_RELEASING)

    def service_all(mut self) -> Bool:
        """STREAMER THREAD (or main thread when no RT thread runs): refills armed/active
        channels by one chunk each and frees released channels. Returns True if any work was done."""
        if self.ctrl_addr == 0:
            return False
        var did = False
        for c in range(STREAM_MAX_CHANNELS):
            var st = self.state(c)
            if st == STREAM_STATE_RELEASING:
                if self.producers[c].fd >= 0:
                    close_fd(self.producers[c].fd)
                    self.producers[c].fd = -1
                stream_store(self.ctrl_addr, c, _SC_WRITE, 0)
                stream_store(self.ctrl_addr, c, _SC_READ, 0)
                stream_store(self.ctrl_addr, c, _SC_STATE, STREAM_STATE_FREE)
                did = True
            elif st == STREAM_STATE_ARMED or st == STREAM_STATE_ACTIVE:
                if self._service_channel(c):
                    did = True
        return did

    def _service_channel(mut self, ch: Int) -> Bool:
        var fd = self.producers[ch].fd
        var total = self.producers[ch].total_frames
        var looping = self.producers[ch].looping
        var fmt = self.producers[ch].format
        if fd < 0 or total <= 0:
            return False
        var rp = Int(stream_load(self.ctrl_addr, ch, _SC_READ))
        var wp = Int(stream_load(self.ctrl_addr, ch, _SC_WRITE))
        if rp > wp:
            wp = rp  # consumer jumped ahead (virtualized/seek): restart production there
        var limit = rp + STREAM_RING_FRAMES - STREAM_MARGIN_FRAMES
        if not looping:
            limit = min(limit, total)
        var n = min(limit - wp, STREAM_CHUNK_FRAMES)
        if n <= 0:
            if wp != Int(stream_load(self.ctrl_addr, ch, _SC_WRITE)):
                stream_store(self.ctrl_addr, ch, _SC_WRITE, Int64(wp))
            return False
        var ring = Int(stream_load(self.ctrl_addr, ch, _SC_RING))
        if ring == 0:
            return False
        var produced = 0
        while produced < n:
            var s = wp + produced
            var f = s % total if looping else s
            if f >= total:
                break
            var run = min(n - produced, total - f)
            var got: Int
            if fmt == STREAM_FMT_WAV16:
                got = self._produce_wav(ch, ring, s, f, run)
            else:
                got = self._produce_qoa(ch, ring, s, f, run)
            if got <= 0:
                # Unreadable region: fill silence so playback keeps its clock.
                for k in range(run):
                    var slot = ((s + k) & STREAM_RING_MASK) * 2
                    Pointer[Float32, MutAnyOrigin](unsafe_from_address=ring)[unsafe_offset=slot] = 0.0
                    Pointer[Float32, MutAnyOrigin](unsafe_from_address=ring)[unsafe_offset=slot + 1] = 0.0
                got = run
            produced += got
        self.chunks_read += 1
        stream_store(self.ctrl_addr, ch, _SC_WRITE, Int64(wp + produced))
        return produced > 0

    def _produce_wav(mut self, ch: Int, ring: Int, s: Int, f: Int, run: Int) -> Int:
        ref p = self.producers[ch]
        var frames = min(run, len(p.byte_buf) // max(1, p.block_align))
        var want = frames * p.block_align
        var dest = Int(p.byte_buf.unsafe_ptr())
        var got_b = p.read_samples(dest, want, p.data_offset + f * p.block_align)
        var got_f = got_b // p.block_align
        var rptr = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ring)
        var inv = Float32(1.0 / 32768.0)
        for k in range(got_f):
            var bo = k * p.block_align
            var l = Int16(Int(p.byte_buf[bo]) | (Int(p.byte_buf[bo + 1]) << 8))
            var r = l
            if p.channels > 1:
                r = Int16(Int(p.byte_buf[bo + 2]) | (Int(p.byte_buf[bo + 3]) << 8))
            var slot = ((s + k) & STREAM_RING_MASK) * 2
            rptr[unsafe_offset=slot] = Float32(l) * inv
            rptr[unsafe_offset=slot + 1] = Float32(r) * inv
        return got_f

    def _load_qoa_frame(mut self, ch: Int, qf: Int) -> Bool:
        ref p = self.producers[ch]
        if p.qoa_cached_frame == qf:
            return True
        var c = p.channels
        var full = qoa_full_frame_bytes(c)
        var off = p.data_offset + qf * full
        if off + 8 > p.file_bytes:
            return False
        var want = min(full, p.file_bytes - off)
        var dest = Int(p.byte_buf.unsafe_ptr())
        var got = p.read_samples(dest, want, off)
        if got < 8 + 16 * c:
            return False
        var fch = Int(p.byte_buf[0])
        var fsamples = (Int(p.byte_buf[4]) << 8) | Int(p.byte_buf[5])
        var fsize = (Int(p.byte_buf[6]) << 8) | Int(p.byte_buf[7])
        if fch != c or fsamples <= 0 or fsamples > QOA_FRAME_LEN or fsize > got:
            return False
        var lms = Array[QoaLms, 8](fill=QoaLms())
        var hp = 8
        for cc in range(c):
            lms[cc].h0 = _be16s(p.byte_buf, hp + 0)
            lms[cc].h1 = _be16s(p.byte_buf, hp + 2)
            lms[cc].h2 = _be16s(p.byte_buf, hp + 4)
            lms[cc].h3 = _be16s(p.byte_buf, hp + 6)
            lms[cc].w0 = _be16s(p.byte_buf, hp + 8)
            lms[cc].w1 = _be16s(p.byte_buf, hp + 10)
            lms[cc].w2 = _be16s(p.byte_buf, hp + 12)
            lms[cc].w3 = _be16s(p.byte_buf, hp + 14)
            hp += 16
        var sp = hp
        var pos = 0
        while pos < fsamples:
            if sp + c * 8 > fsize:
                return False
            var slice_len = min(QOA_SLICE_LEN, fsamples - pos)
            for cc in range(c):
                var bits = UInt64(0)
                for b in range(8):
                    bits = (bits << 8) | UInt64(p.byte_buf[sp + b])
                sp += 8
                var sf = Int((bits >> 60) & 0x0F)
                var shift = 57
                for k in range(slice_len):
                    var qr = Int((bits >> UInt64(shift)) & 0x07)
                    shift -= 3
                    var predicted = lms[cc].predict()
                    var dq = _qoa_dequant(sf, qr)
                    var r = _clamp_i16(predicted + dq)
                    lms[cc].update(r, dq)
                    p.qoa_cache[(pos + k) * c + cc] = Int16(r)
            pos += slice_len
        p.qoa_cached_frame = qf
        p.qoa_cached_len = fsamples
        return True

    def _produce_qoa(mut self, ch: Int, ring: Int, s: Int, f: Int, run: Int) -> Int:
        var qf = f // QOA_FRAME_LEN
        var within = f % QOA_FRAME_LEN
        if not self._load_qoa_frame(ch, qf):
            return 0
        ref p = self.producers[ch]
        var avail = p.qoa_cached_len - within
        if avail <= 0:
            return 0
        var cnt = min(run, avail)
        var c = p.channels
        var rptr = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ring)
        var inv = Float32(1.0 / 32768.0)
        for k in range(cnt):
            var base = (within + k) * c
            var l = p.qoa_cache[base]
            var r = p.qoa_cache[base + 1] if c > 1 else l
            var slot = ((s + k) & STREAM_RING_MASK) * 2
            rptr[unsafe_offset=slot] = Float32(l) * inv
            rptr[unsafe_offset=slot + 1] = Float32(r) * inv
        return cnt


@always_inline
def _be16s(b: List[UInt8], o: Int) -> Int:
    var u = (Int(b[o]) << 8) | Int(b[o + 1])
    return u - 65536 if u >= 32768 else u


def _audio_stream_thread_entry(pool_addr: Int) abi("C") -> Int:
    ref pool = Pointer[AudioStreamPool, MutAnyOrigin](unsafe_from_address=pool_addr)[unsafe_offset=0]
    var running = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=pool.running_addr)
    while running[unsafe_offset=0].load() != 0:
        var did = pool.service_all()
        if not did:
            var ts = Array[Int64, 2](fill=0)
            ts[1] = 2_000_000  # 2 ms idle poll
            _ = external_call["nanosleep", Int32](Pointer(to=ts), Int(0))
            _ = ts[1]
    return 0
