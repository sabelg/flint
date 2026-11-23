"""Pure-Mojo QOA (Quite OK Audio) Lossy Audio Codec (`flint.audio.qoa`).

Spec-compliant implementation of the Quite OK Audio format (Dominic Szablewski):
  - Fixed 64-bit slices encoding 20 audio samples (exact 3.2 bits/sample slice payload,
    5:1 compression vs 16-bit PCM).
  - 4-tap Sign-Sign Least Mean Squares (LMS) adaptive predictor per channel (`QoaLms`).
  - Self-contained 256-slice frames (`5,120` samples/channel/frame) enabling O(1)
    arithmetic random-access frame seeking (`QoaStreamCursor.seek`).
  - Zero heap allocations per streaming slice via inline `Array` state.
"""
from std.collections import List

comptime QOA_MAGIC: UInt32 = 0x716F6166  # "qoaf"
comptime QOA_SLICE_LEN: Int = 20
comptime QOA_SLICES_PER_FRAME: Int = 256
comptime QOA_FRAME_LEN: Int = QOA_SLICES_PER_FRAME * QOA_SLICE_LEN  # 5120 samples/channel
comptime QOA_MAX_CHANNELS: Int = 8
comptime QOA_SLICE_BITS_PER_SAMPLE: Float32 = 3.2


@always_inline
def _qoa_scalefactor(sf: Int) -> Int:
    var tab = Array[Int, 16](fill=0)
    tab[0] = 1; tab[1] = 7; tab[2] = 21; tab[3] = 45
    tab[4] = 84; tab[5] = 138; tab[6] = 211; tab[7] = 304
    tab[8] = 421; tab[9] = 562; tab[10] = 731; tab[11] = 928
    tab[12] = 1157; tab[13] = 1419; tab[14] = 1715; tab[15] = 2048
    return tab[sf & 15]


@always_inline
def _qoa_reciprocal(sf: Int) -> Int:
    var tab = Array[Int, 16](fill=0)
    tab[0] = 65536; tab[1] = 9363; tab[2] = 3121; tab[3] = 1457
    tab[4] = 781; tab[5] = 475; tab[6] = 311; tab[7] = 216
    tab[8] = 156; tab[9] = 117; tab[10] = 90; tab[11] = 71
    tab[12] = 57; tab[13] = 47; tab[14] = 39; tab[15] = 32
    return tab[sf & 15]


@always_inline
def _qoa_quant(scaled_clamped_plus_8: Int) -> Int:
    var tab = Array[Int, 17](fill=0)
    tab[0] = 7; tab[1] = 7; tab[2] = 7; tab[3] = 5; tab[4] = 5
    tab[5] = 3; tab[6] = 3; tab[7] = 1; tab[8] = 0; tab[9] = 0
    tab[10] = 2; tab[11] = 2; tab[12] = 4; tab[13] = 4
    tab[14] = 6; tab[15] = 6; tab[16] = 6
    var idx = max(0, min(16, scaled_clamped_plus_8))
    return tab[idx]


@always_inline
def _qoa_dequant(sf: Int, qr: Int) -> Int:
    """Exact QOA spec `qoa_dequant_tab[sf][qr]`: `round(scalefactor * {0.75, -0.75, 2.5, -2.5, 4.5, -4.5, 7, -7})`."""
    var s = _qoa_scalefactor(sf)
    var q = qr & 7
    var mag_idx = q >> 1
    var d: Int
    if mag_idx == 0:
        # round(s * 0.75) = (s * 3 + 2) >> 2
        d = (s * 3 + 2) >> 2
    elif mag_idx == 1:
        # round(s * 2.5) = (s * 5 + 1) >> 1
        d = (s * 5 + 1) >> 1
    elif mag_idx == 2:
        # round(s * 4.5) = (s * 9 + 1) >> 1
        d = (s * 9 + 1) >> 1
    else:
        # s * 7
        d = s * 7
    return -d if (q & 1) != 0 else d


@always_inline
def _qoa_div(v: Int, scalefactor: Int) -> Int:
    var reciprocal = _qoa_reciprocal(scalefactor)
    var n = (v * reciprocal + (1 << 15)) >> 16
    var sgn_v = (1 if v > 0 else 0) - (1 if v < 0 else 0)
    var sgn_n = (1 if n > 0 else 0) - (1 if n < 0 else 0)
    return n + sgn_v - sgn_n


@always_inline
def _clamp_i16(v: Int) -> Int:
    if v < -32768:
        return -32768
    if v > 32767:
        return 32767
    return v


@always_inline
def _put_be16_s(mut out: List[UInt8], v: Int):
    var u = UInt16(Int16(_clamp_i16(v)))
    out.append(UInt8((u >> 8) & 0xFF))
    out.append(UInt8(u & 0xFF))


@always_inline
def _put_be16_u(mut out: List[UInt8], v: Int):
    var u = UInt16(v & 0xFFFF)
    out.append(UInt8((u >> 8) & 0xFF))
    out.append(UInt8(u & 0xFF))


@always_inline
def _put_be32(mut out: List[UInt8], v: UInt32):
    out.append(UInt8((v >> 24) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


@always_inline
def _put_be64(mut out: List[UInt8], v: UInt64):
    out.append(UInt8((v >> 56) & 0xFF))
    out.append(UInt8((v >> 48) & 0xFF))
    out.append(UInt8((v >> 40) & 0xFF))
    out.append(UInt8((v >> 32) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


@always_inline
def _get_be16_u(data: List[UInt8], off: Int) -> Int:
    return (Int(data[off]) << 8) | Int(data[off + 1])


@always_inline
def _get_be16_s(data: List[UInt8], off: Int) -> Int:
    var u = _get_be16_u(data, off)
    return u - 65536 if u >= 32768 else u


@always_inline
def _get_be32(data: List[UInt8], off: Int) -> UInt32:
    return (
        (UInt32(data[off]) << 24)
        | (UInt32(data[off + 1]) << 16)
        | (UInt32(data[off + 2]) << 8)
        | UInt32(data[off + 3])
    )


@always_inline
def _get_be64(data: List[UInt8], off: Int) -> UInt64:
    return (
        (UInt64(data[off]) << 56)
        | (UInt64(data[off + 1]) << 48)
        | (UInt64(data[off + 2]) << 40)
        | (UInt64(data[off + 3]) << 32)
        | (UInt64(data[off + 4]) << 24)
        | (UInt64(data[off + 5]) << 16)
        | (UInt64(data[off + 6]) << 8)
        | UInt64(data[off + 7])
    )


@fieldwise_init
struct QoaLms(Copyable, ImplicitlyCopyable, Movable):
    """4-tap Sign-Sign LMS predictor state for one QOA channel."""
    var h0: Int
    var h1: Int
    var h2: Int
    var h3: Int
    var w0: Int
    var w1: Int
    var w2: Int
    var w3: Int

    def __init__(out self):
        self.h0 = 0
        self.h1 = 0
        self.h2 = 0
        self.h3 = 0
        self.w0 = 0
        self.w1 = 0
        self.w2 = -(1 << 13)
        self.w3 = 1 << 14

    @always_inline
    def clamp_header_i16(mut self):
        self.h0 = _clamp_i16(self.h0)
        self.h1 = _clamp_i16(self.h1)
        self.h2 = _clamp_i16(self.h2)
        self.h3 = _clamp_i16(self.h3)
        self.w0 = _clamp_i16(self.w0)
        self.w1 = _clamp_i16(self.w1)
        self.w2 = _clamp_i16(self.w2)
        self.w3 = _clamp_i16(self.w3)

    @always_inline
    def predict(self) -> Int:
        return (
            self.w0 * self.h0
            + self.w1 * self.h1
            + self.w2 * self.h2
            + self.w3 * self.h3
        ) >> 13

    @always_inline
    def update(mut self, sample: Int, residual: Int):
        var delta = residual >> 4
        self.w0 += -delta if self.h0 < 0 else delta
        self.w1 += -delta if self.h1 < 0 else delta
        self.w2 += -delta if self.h2 < 0 else delta
        self.w3 += -delta if self.h3 < 0 else delta
        self.h0 = self.h1
        self.h1 = self.h2
        self.h2 = self.h3
        self.h3 = sample


struct QoaClip(Copyable, Movable):
    """Decoded PCM representation of a QOA audio clip."""
    var valid: Bool
    var channels: Int
    var sample_rate: Int
    var samples_per_channel: Int
    var samples: List[Int16]

    def __init__(out self):
        self.valid = False
        self.channels = 0
        self.sample_rate = 0
        self.samples_per_channel = 0
        self.samples = List[Int16]()

    def frame_count(self) -> Int:
        return self.samples_per_channel


struct QoaAsset(Copyable, Movable):
    """Compressed in-memory QOA asset streamed directly by `QoaStreamCursor` at 5:1 compression."""
    var valid: Bool
    var channels: Int
    var sample_rate: Int
    var samples_per_channel: Int
    var bytes: List[UInt8]

    def __init__(out self):
        self.valid = False
        self.channels = 0
        self.sample_rate = 0
        self.samples_per_channel = 0
        self.bytes = List[UInt8]()

    @staticmethod
    def from_bytes(var raw: List[UInt8]) -> Self:
        var a = Self()
        if len(raw) < 16 or _get_be32(raw, 0) != QOA_MAGIC:
            return a^
        var total_samples = Int(_get_be32(raw, 4))
        var ch = Int(raw[8])
        var sr = (Int(raw[9]) << 16) | (Int(raw[10]) << 8) | Int(raw[11])
        if total_samples <= 0 or ch < 1 or ch > QOA_MAX_CHANNELS or sr <= 0:
            return a^
        a.valid = True
        a.channels = ch
        a.sample_rate = sr
        a.samples_per_channel = total_samples
        a.bytes = raw^
        return a^


@always_inline
def qoa_full_frame_bytes(channels: Int) -> Int:
    """Exact byte size of a full 256-slice QOA frame (`8 + 16*C + 256*8*C`)."""
    return 8 + channels * 16 + QOA_SLICES_PER_FRAME * channels * 8


def encode_qoa(pcm: List[Int16], channels: Int, sample_rate: Int) -> List[UInt8]:
    """Encodes interleaved 16-bit PCM into a spec-compliant QOA byte stream."""
    var out = List[UInt8]()
    if channels < 1 or channels > QOA_MAX_CHANNELS or sample_rate < 1 or sample_rate > 0xFFFFFF:
        return out^
    var total_frames = len(pcm) // channels
    if total_frames <= 0:
        return out^

    var num_qoa_frames = (total_frames + QOA_FRAME_LEN - 1) // QOA_FRAME_LEN
    var total_slices = (total_frames + QOA_SLICE_LEN - 1) // QOA_SLICE_LEN
    var est_bytes = 8 + num_qoa_frames * (8 + channels * 16) + total_slices * channels * 8
    out.reserve(est_bytes)

    # 8-byte QOA file header
    _put_be32(out, QOA_MAGIC)
    _put_be32(out, UInt32(total_frames))

    var lms = Array[QoaLms, QOA_MAX_CHANNELS](fill=QoaLms())
    var prev_sf = Array[Int, QOA_MAX_CHANNELS](fill=0)

    var sample_index = 0
    while sample_index < total_frames:
        var frame_len = min(QOA_FRAME_LEN, total_frames - sample_index)
        var frame_slices = (frame_len + QOA_SLICE_LEN - 1) // QOA_SLICE_LEN
        var frame_size = 8 + channels * 16 + frame_slices * channels * 8

        # 8-byte Frame Header: channels(8) | samplerate(24) | fsamples(16) | fsize(16)
        out.append(UInt8(channels & 0xFF))
        out.append(UInt8((sample_rate >> 16) & 0xFF))
        out.append(UInt8((sample_rate >> 8) & 0xFF))
        out.append(UInt8(sample_rate & 0xFF))
        _put_be16_u(out, frame_len)
        _put_be16_u(out, frame_size)

        # Clamp LMS weights/history to Int16 so frame header & decoder state match bit-for-bit
        for c in range(channels):
            lms[c].clamp_header_i16()
            _put_be16_s(out, lms[c].h0)
            _put_be16_s(out, lms[c].h1)
            _put_be16_s(out, lms[c].h2)
            _put_be16_s(out, lms[c].h3)
            _put_be16_s(out, lms[c].w0)
            _put_be16_s(out, lms[c].w1)
            _put_be16_s(out, lms[c].w2)
            _put_be16_s(out, lms[c].w3)

        var s_off = sample_index
        var frame_end = sample_index + frame_len
        while s_off < frame_end:
            var slice_len = min(QOA_SLICE_LEN, frame_end - s_off)
            for c in range(channels):
                var best_rank = UInt64(0xFFFFFFFFFFFFFFFF)
                var best_slice = UInt64(0)
                var best_lms = lms[c]
                var best_sf = 0

                for sfi in range(16):
                    var sf = (sfi + prev_sf[c]) & 15
                    var cur_lms = lms[c]
                    var slice_bits = UInt64(sf)
                    var current_rank = UInt64(0)

                    for k in range(slice_len):
                        var sample = Int(pcm[(s_off + k) * channels + c])
                        var predicted = cur_lms.predict()
                        var residual = sample - predicted
                        var scaled = _qoa_div(residual, sf)
                        var clamped = max(-8, min(8, scaled))
                        var quantized = _qoa_quant(clamped + 8)
                        var dequantized = _qoa_dequant(sf, quantized)
                        var reconstructed = _clamp_i16(predicted + dequantized)

                        var weights_penalty = (
                            (
                                cur_lms.w0 * cur_lms.w0
                                + cur_lms.w1 * cur_lms.w1
                                + cur_lms.w2 * cur_lms.w2
                                + cur_lms.w3 * cur_lms.w3
                            )
                            >> 18
                        ) - 0x8FF
                        if weights_penalty < 0:
                            weights_penalty = 0

                        var err = Int64(sample - reconstructed)
                        var err_sq = UInt64(err * err)
                        current_rank += err_sq + UInt64(weights_penalty * weights_penalty)
                        if current_rank >= best_rank:
                            break

                        cur_lms.update(reconstructed, dequantized)
                        slice_bits = (slice_bits << 3) | UInt64(quantized)

                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_slice = slice_bits
                        best_lms = cur_lms
                        best_sf = sf
                        if current_rank == 0:
                            break

                prev_sf[c] = best_sf
                lms[c] = best_lms
                # Left-align final partial slice (< 20 samples) into MSB slots of 60-bit payload
                best_slice <<= UInt64((QOA_SLICE_LEN - slice_len) * 3)
                _put_be64(out, best_slice)

            s_off += QOA_SLICE_LEN

        sample_index += frame_len

    return out^


def decode_qoa(data: List[UInt8]) -> QoaClip:
    """Decodes an entire QOA byte buffer into interleaved 16-bit PCM."""
    var clip = QoaClip()
    if len(data) < 16 or _get_be32(data, 0) != QOA_MAGIC:
        return clip^
    var total_frames = Int(_get_be32(data, 4))
    if total_frames <= 0 or total_frames > 200_000_000:
        return clip^

    var first_ch = Int(data[8])
    var first_sr = (Int(data[9]) << 16) | (Int(data[10]) << 8) | Int(data[11])
    if first_ch < 1 or first_ch > QOA_MAX_CHANNELS or first_sr <= 0:
        return clip^

    # Validate minimum encoded byte count before allocating sample buffer
    var min_frames = (total_frames + 65534) // 65535
    var min_slices = (total_frames + QOA_SLICE_LEN - 1) // QOA_SLICE_LEN
    var min_bytes = 8 + min_frames * (8 + 16 * first_ch) + min_slices * (8 * first_ch)
    if len(data) < min_bytes:
        return clip^

    clip.channels = first_ch
    clip.sample_rate = first_sr
    clip.samples_per_channel = total_frames
    clip.samples = List[Int16](length=total_frames * first_ch, fill=0)

    var lms = Array[QoaLms, QOA_MAX_CHANNELS](fill=QoaLms())
    var p = 8
    var decoded_frames = 0

    while p + 8 <= len(data) and decoded_frames < total_frames:
        var ch = Int(data[p])
        var sr = (Int(data[p + 1]) << 16) | (Int(data[p + 2]) << 8) | Int(data[p + 3])
        var fsamples = _get_be16_u(data, p + 4)
        var fsize = _get_be16_u(data, p + 6)

        if ch != first_ch or sr != first_sr or fsamples <= 0:
            return QoaClip()
        if fsize < 8 + ch * 16 or p + fsize > len(data):
            return QoaClip()

        var hdr_p = p + 8
        for c in range(ch):
            lms[c].h0 = _get_be16_s(data, hdr_p + 0)
            lms[c].h1 = _get_be16_s(data, hdr_p + 2)
            lms[c].h2 = _get_be16_s(data, hdr_p + 4)
            lms[c].h3 = _get_be16_s(data, hdr_p + 6)
            lms[c].w0 = _get_be16_s(data, hdr_p + 8)
            lms[c].w1 = _get_be16_s(data, hdr_p + 10)
            lms[c].w2 = _get_be16_s(data, hdr_p + 12)
            lms[c].w3 = _get_be16_s(data, hdr_p + 14)
            hdr_p += 16

        var slice_p = hdr_p
        var frame_end = min(total_frames, decoded_frames + fsamples)
        var s_pos = decoded_frames
        while s_pos < frame_end:
            if slice_p + ch * 8 > p + fsize:
                return QoaClip()
            var slice_len = min(QOA_SLICE_LEN, frame_end - s_pos)
            for c in range(ch):
                var slice_bits = _get_be64(data, slice_p)
                slice_p += 8
                var sf = Int((slice_bits >> 60) & 0x0F)
                var shift = 57
                for k in range(slice_len):
                    var qr = Int((slice_bits >> UInt64(shift)) & 0x07)
                    shift -= 3
                    var predicted = lms[c].predict()
                    var dequantized = _qoa_dequant(sf, qr)
                    var r = _clamp_i16(predicted + dequantized)
                    lms[c].update(r, dequantized)
                    clip.samples[(s_pos + k) * ch + c] = Int16(r)
            s_pos += slice_len

        decoded_frames = frame_end
        p += fsize

    if decoded_frames != total_frames:
        return QoaClip()
    clip.valid = True
    return clip^


struct QoaStreamCursor(Copyable, Movable):
    """Zero-allocation streaming QOA decoder with O(1) random-access frame seeking."""
    var valid: Bool
    var channels: Int
    var sample_rate: Int
    var total_frames: Int
    var current_sample_frame: Int
    var frame_byte_offset: Int
    var frame_end_offset: Int
    var slice_byte_offset: Int
    var frame_samples_remaining: Int
    var slice_cache: Array[Int16, QOA_SLICE_LEN * QOA_MAX_CHANNELS]
    var slice_cache_pos: Int
    var slice_cache_len: Int
    var lms: Array[QoaLms, QOA_MAX_CHANNELS]

    def __init__(out self):
        self.valid = False
        self.channels = 0
        self.sample_rate = 0
        self.total_frames = 0
        self.current_sample_frame = 0
        self.frame_byte_offset = 8
        self.frame_end_offset = 8
        self.slice_byte_offset = 8
        self.frame_samples_remaining = 0
        self.slice_cache = Array[Int16, QOA_SLICE_LEN * QOA_MAX_CHANNELS](fill=0)
        self.slice_cache_pos = 0
        self.slice_cache_len = 0
        self.lms = Array[QoaLms, QOA_MAX_CHANNELS](fill=QoaLms())

    def _reset_stream_state(mut self):
        self.valid = False
        self.channels = 0
        self.sample_rate = 0
        self.total_frames = 0
        self.current_sample_frame = 0
        self.frame_byte_offset = 8
        self.frame_end_offset = 8
        self.slice_byte_offset = 8
        self.frame_samples_remaining = 0
        self.slice_cache_pos = 0
        self.slice_cache_len = 0
        for i in range(QOA_SLICE_LEN * QOA_MAX_CHANNELS):
            self.slice_cache[i] = 0
        for c in range(QOA_MAX_CHANNELS):
            self.lms[c] = QoaLms()

    def open(mut self, data: List[UInt8]) -> Bool:
        self._reset_stream_state()
        if len(data) < 16 or _get_be32(data, 0) != QOA_MAGIC:
            return False
        var total = Int(_get_be32(data, 4))
        var ch = Int(data[8])
        var sr = (Int(data[9]) << 16) | (Int(data[10]) << 8) | Int(data[11])
        if total <= 0 or ch < 1 or ch > QOA_MAX_CHANNELS or sr <= 0:
            return False
        self.channels = ch
        self.sample_rate = sr
        self.total_frames = total
        if not self._seek_loaded(data, 0):
            self._reset_stream_state()
            return False
        self.valid = True
        return True

    def _load_frame_header(mut self, data: List[UInt8], byte_off: Int) -> Bool:
        if byte_off + 8 + self.channels * 16 > len(data):
            return False
        var ch = Int(data[byte_off])
        var fsamples = _get_be16_u(data, byte_off + 4)
        var fsize = _get_be16_u(data, byte_off + 6)
        if ch != self.channels or fsamples <= 0 or fsize < 8 + ch * 16 or byte_off + fsize > len(data):
            return False
        var need_slices = (fsamples + QOA_SLICE_LEN - 1) // QOA_SLICE_LEN
        if fsize < 8 + ch * 16 + need_slices * ch * 8:
            return False
        var hp = byte_off + 8
        for c in range(ch):
            self.lms[c].h0 = _get_be16_s(data, hp + 0)
            self.lms[c].h1 = _get_be16_s(data, hp + 2)
            self.lms[c].h2 = _get_be16_s(data, hp + 4)
            self.lms[c].h3 = _get_be16_s(data, hp + 6)
            self.lms[c].w0 = _get_be16_s(data, hp + 8)
            self.lms[c].w1 = _get_be16_s(data, hp + 10)
            self.lms[c].w2 = _get_be16_s(data, hp + 12)
            self.lms[c].w3 = _get_be16_s(data, hp + 14)
            hp += 16
        self.frame_byte_offset = byte_off
        self.frame_end_offset = byte_off + fsize
        self.slice_byte_offset = hp
        self.frame_samples_remaining = fsamples
        self.slice_cache_pos = 0
        self.slice_cache_len = 0
        return True

    def _decode_one_slice(mut self, data: List[UInt8]) -> Bool:
        """Decodes up to 20 sample frames into `self.slice_cache` with zero heap allocations."""
        if self.channels <= 0:
            return False
        if self.frame_samples_remaining <= 0:
            if self.current_sample_frame - (self.slice_cache_len - self.slice_cache_pos) >= self.total_frames:
                return False
            if not self._load_frame_header(data, self.slice_byte_offset):
                return False

        var ch = self.channels
        if self.slice_byte_offset + ch * 8 > self.frame_end_offset or self.slice_byte_offset + ch * 8 > len(data):
            return False

        var slice_len = min(QOA_SLICE_LEN, self.frame_samples_remaining)
        var sp = self.slice_byte_offset
        for c in range(ch):
            var slice_bits = _get_be64(data, sp)
            sp += 8
            var sf = Int((slice_bits >> 60) & 0x0F)
            var shift = 57
            for k in range(slice_len):
                var qr = Int((slice_bits >> UInt64(shift)) & 0x07)
                shift -= 3
                var predicted = self.lms[c].predict()
                var dequantized = _qoa_dequant(sf, qr)
                var r = _clamp_i16(predicted + dequantized)
                self.lms[c].update(r, dequantized)
                self.slice_cache[k * ch + c] = Int16(r)

        self.slice_byte_offset = sp
        self.frame_samples_remaining -= slice_len
        self.slice_cache_pos = 0
        self.slice_cache_len = slice_len
        return True

    def _seek_loaded(mut self, data: List[UInt8], target_sample_frame: Int) -> Bool:
        var clamped = max(0, min(self.total_frames, target_sample_frame))
        if clamped == self.total_frames:
            self.current_sample_frame = self.total_frames
            self.slice_cache_pos = 0
            self.slice_cache_len = 0
            self.frame_samples_remaining = 0
            return True

        var qoa_frame_idx = clamped // QOA_FRAME_LEN
        var within_frame = clamped % QOA_FRAME_LEN
        var byte_off = 8 + qoa_frame_idx * qoa_full_frame_bytes(self.channels)
        if not self._load_frame_header(data, byte_off):
            return False

        # Advance LMS through slices preceding `within_frame` inside this single QOA frame (<= 255 slices)
        var skip_slices = within_frame // QOA_SLICE_LEN
        var within_slice = within_frame % QOA_SLICE_LEN
        for _ in range(skip_slices):
            if not self._decode_one_slice(data):
                return False
        if within_slice > 0:
            if not self._decode_one_slice(data):
                return False
            self.slice_cache_pos = within_slice
        else:
            self.slice_cache_pos = 0
            self.slice_cache_len = 0

        self.current_sample_frame = clamped
        return True

    def seek(mut self, data: List[UInt8], target_sample_frame: Int) -> Bool:
        """O(1) arithmetic frame seek to any sample index `[0, total_frames]`."""
        if not self.valid:
            if not self.open(data):
                return False
        return self._seek_loaded(data, target_sample_frame)

    @always_inline
    def read_next_frame_i16(mut self, data: List[UInt8], mut out_l: Int16, mut out_r: Int16) -> Bool:
        """Reads one mono/stereo sample frame with zero heap allocations."""
        if not self.valid or self.current_sample_frame >= self.total_frames:
            return False
        if self.slice_cache_pos >= self.slice_cache_len:
            if not self._decode_one_slice(data):
                return False
        var base = self.slice_cache_pos * self.channels
        out_l = self.slice_cache[base]
        out_r = self.slice_cache[base + 1] if self.channels > 1 else out_l
        self.slice_cache_pos += 1
        self.current_sample_frame += 1
        return True

    def decode_frames_into(mut self, data: List[UInt8], mut dst: List[Int16], dst_frame_offset: Int, frame_count: Int) -> Int:
        """Zero-allocation block decode into caller-owned `dst` buffer."""
        if not self.valid or frame_count <= 0 or dst_frame_offset < 0 or self.channels <= 0:
            return 0
        var ch = self.channels
        var max_frames_in_dst = (len(dst) // ch) - dst_frame_offset
        if max_frames_in_dst <= 0:
            return 0
        var target_frames = min(frame_count, max_frames_in_dst)
        var written = 0
        while written < target_frames and self.current_sample_frame < self.total_frames:
            if self.slice_cache_pos >= self.slice_cache_len:
                if not self._decode_one_slice(data):
                    break
            var avail = min(target_frames - written, self.slice_cache_len - self.slice_cache_pos)
            for k in range(avail):
                var src_base = (self.slice_cache_pos + k) * ch
                var dst_base = (dst_frame_offset + written + k) * ch
                for c in range(ch):
                    dst[dst_base + c] = self.slice_cache[src_base + c]
            self.slice_cache_pos += avail
            self.current_sample_frame += avail
            written += avail
        return written
