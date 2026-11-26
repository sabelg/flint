"""Pure-Mojo Ogg Vorbis I Decoder (`flint.audio.vorbis`).

Implements the Xiph Vorbis I specification for the configurations every libvorbis /
aoTuV encoder emits:
  - Ogg framing: page capture with CRC-32 (poly `0x04C11DB7`) verification, lacing /
    multi-page packet reassembly, first logical bitstream only, granule-position end trim.
  - Identification / comment / setup headers: Huffman codebooks (ordered, sparse and
    dense length lists; VQ lookup types 1 and 2), floor type 1, residue types 0/1/2,
    mapping type 0 (submaps, channel coupling), modes.
  - Audio packets: floor 1 decode + curve synthesis (integer Bresenham `render_line`,
    inverse-dB table), residue decode, inverse square-polar channel coupling,
    IMDCT (DCT-IV via an N/4-point complex radix-2 FFT), Vorbis power-sine windows with
    long/short transitions, and overlap-add.
  - Not supported (rejected with an error, never mis-decoded): floor type 0 (only emitted
    by pre-2000 beta encoders), chained streams after the first logical stream.

`decode_ogg_vorbis(bytes)` returns interleaved `Int16` PCM (`VorbisClip`), and
`decode_ogg_vorbis_f32` the raw float output.
"""
from std.collections import List
from std.math import sin, cos, exp, log, floor, abs

comptime _EMPTY: Int32 = -2147483647
comptime _PI: Float64 = 3.141592653589793


# ==============================================================================
# Ogg framing
# ==============================================================================

def _crc_table() -> List[UInt32]:
    var t = List[UInt32](length=256, fill=0)
    for i in range(256):
        var r = UInt32(i) << 24
        for _ in range(8):
            if (r & 0x80000000) != 0:
                r = (r << 1) ^ 0x04C11DB7
            else:
                r = r << 1
        t[i] = r
    return t^


struct _OggPackets(Movable):
    """All packets of the first logical stream, concatenated."""
    var data: List[UInt8]
    var starts: List[Int]
    var lens: List[Int]
    var granule: List[Int64]  # granule of the page where the packet completes (-1 if none)
    var last_granule: Int64
    var error: String

    def __init__(out self):
        self.data = List[UInt8]()
        self.starts = List[Int]()
        self.lens = List[Int]()
        self.granule = List[Int64]()
        self.last_granule = -1
        self.error = ""


def _le32u(b: List[UInt8], o: Int) -> UInt32:
    return UInt32(b[o]) | (UInt32(b[o + 1]) << 8) | (UInt32(b[o + 2]) << 16) | (UInt32(b[o + 3]) << 24)


def _read_ogg_packets(src: List[UInt8]) -> _OggPackets:
    var out = _OggPackets()
    var crc = _crc_table()
    var p = 0
    var n = len(src)
    var serial = UInt32(0)
    var have_serial = False
    var in_packet = False
    var cur_start = 0
    while p + 27 <= n:
        if not (src[p] == 79 and src[p + 1] == 103 and src[p + 2] == 103 and src[p + 3] == 83):
            p += 1  # resync to the next capture pattern
            continue
        var nseg = Int(src[p + 26])
        if p + 27 + nseg > n:
            break
        var body = 0
        for i in range(nseg):
            body += Int(src[p + 27 + i])
        var page_len = 27 + nseg + body
        if p + page_len > n:
            break
        # CRC over the page with the CRC field zeroed
        var want = _le32u(src, p + 22)
        var c = UInt32(0)
        for i in range(page_len):
            var byte = src[p + i]
            if i >= 22 and i < 26:
                byte = 0
            c = (c << 8) ^ crc[Int(((c >> 24) & 0xFF) ^ UInt32(byte))]
        if c != want or src[p + 4] != 0:
            p += 1
            continue
        var htype = src[p + 5]
        var gran = Int64(0)
        for i in range(8):
            gran |= Int64(src[p + 6 + i]) << Int64(8 * i)
        var sn = _le32u(src, p + 14)
        if not have_serial:
            if (htype & 2) == 0:
                p += page_len
                continue
            serial = sn
            have_serial = True
        if sn != serial:
            p += page_len
            continue
        if (htype & 1) == 0 and in_packet:
            # Fresh page without the continuation flag: drop the partial packet.
            out.data.resize(cur_start, 0)
            in_packet = False
        if (htype & 1) != 0 and not in_packet:
            # Continuation of a packet we never saw the start of: skip its first piece.
            pass
        var body_p = p + 27 + nseg
        var skip_first = (htype & 1) != 0 and not in_packet
        var completed_any = False
        for i in range(nseg):
            var lace = Int(src[p + 27 + i])
            if skip_first:
                body_p += lace
                if lace < 255:
                    skip_first = False
                continue
            if not in_packet:
                cur_start = len(out.data)
                in_packet = True
            for k in range(lace):
                out.data.append(src[body_p + k])
            body_p += lace
            if lace < 255:
                out.starts.append(cur_start)
                out.lens.append(len(out.data) - cur_start)
                out.granule.append(-1)
                in_packet = False
                completed_any = True
        if completed_any and gran != -1:
            out.granule[len(out.granule) - 1] = gran
            out.last_granule = gran
        p += page_len
        if (htype & 4) != 0:
            break  # end of the logical stream
    return out^


# ==============================================================================
# Bit reader (LSB-first, Vorbis packing)
# ==============================================================================

struct _BitReader(Movable):
    var base: Int
    var end_bits: Int
    var pos: Int
    var eop: Bool

    def __init__(out self, start: Int, length: Int):
        self.base = start
        self.end_bits = length * 8
        self.pos = 0
        self.eop = False

    @always_inline
    def read(mut self, data: List[UInt8], n: Int) -> Int:
        if n <= 0:
            return 0
        if self.pos + n > self.end_bits:
            self.eop = True
            self.pos = self.end_bits
            return 0
        var v = UInt64(0)
        var got = 0
        var p = self.pos
        while got < n:
            var byte = UInt64(data[self.base + (p >> 3)])
            var off = p & 7
            var take = min(8 - off, n - got)
            v |= ((byte >> UInt64(off)) & ((UInt64(1) << UInt64(take)) - 1)) << UInt64(got)
            got += take
            p += take
        self.pos = p
        return Int(v)

    @always_inline
    def read1(mut self, data: List[UInt8]) -> Int:
        if self.pos >= self.end_bits:
            self.eop = True
            return 0
        var byte = data[self.base + (self.pos >> 3)]
        var bit = Int((byte >> UInt8(self.pos & 7)) & 1)
        self.pos += 1
        return bit


@always_inline
def _ilog(v: Int) -> Int:
    var r = 0
    var x = v
    while x > 0:
        r += 1
        x >>= 1
    return r


def _float32_unpack(x: Int) -> Float64:
    var mantissa = Float64(x & 0x1FFFFF)
    var exponent = (x & 0x7FE00000) >> 21
    if (x & 0x80000000) != 0:
        mantissa = -mantissa
    var e = exponent - 788
    var r = mantissa
    if e > 0:
        for _ in range(e):
            r *= 2.0
    elif e < 0:
        for _ in range(-e):
            r *= 0.5
    return r


def _pow_le(base: Int, dims: Int, limit: Int) -> Bool:
    """True when base^dims <= limit (overflow-safe: stops as soon as the product exceeds limit)."""
    var acc = 1
    for _ in range(dims):
        acc *= base
        if acc > limit:
            return False
    return True


def _lookup1_values(entries: Int, dims: Int) -> Int:
    """Largest r with r^dims <= entries (spec 9.2.3)."""
    var r = Int(floor(exp(log(Float64(entries)) / Float64(dims))))
    if r < 1:
        r = 1
    while _pow_le(r + 1, dims, entries):
        r += 1
    while r > 1 and not _pow_le(r, dims, entries):
        r -= 1
    return r


# ==============================================================================
# Setup structures
# ==============================================================================

struct _Codebook(Copyable, Movable):
    var dims: Int
    var entries: Int
    var lengths: List[UInt8]
    var tree: List[Int32]  # child pairs; >0 internal node, <0 leaf -(entry+1)
    var single_entry: Int
    var lookup_type: Int
    var vq: List[Float32]

    def __init__(out self):
        self.dims = 0
        self.entries = 0
        self.lengths = List[UInt8]()
        self.tree = List[Int32]()
        self.single_entry = -1
        self.lookup_type = 0
        self.vq = List[Float32]()


struct _Floor1(Copyable, Movable):
    var partitions: Int
    var partition_class: List[Int]
    var class_dims: List[Int]
    var class_subclasses: List[Int]
    var class_master: List[Int]
    var subclass_books: List[Int]  # class * 8 + j
    var multiplier: Int
    var xlist: List[Int]
    var sorted_idx: List[Int]
    var low_n: List[Int]
    var high_n: List[Int]

    def __init__(out self):
        self.partitions = 0
        self.partition_class = List[Int]()
        self.class_dims = List[Int](length=16, fill=0)
        self.class_subclasses = List[Int](length=16, fill=0)
        self.class_master = List[Int](length=16, fill=0)
        self.subclass_books = List[Int](length=16 * 8, fill=-1)
        self.multiplier = 1
        self.xlist = List[Int]()
        self.sorted_idx = List[Int]()
        self.low_n = List[Int]()
        self.high_n = List[Int]()


struct _Residue(Copyable, Movable):
    var rtype: Int
    var begin: Int
    var end: Int
    var part_size: Int
    var classifications: Int
    var classbook: Int
    var books: List[Int]  # classification * 8 + pass

    def __init__(out self):
        self.rtype = 0
        self.begin = 0
        self.end = 0
        self.part_size = 1
        self.classifications = 1
        self.classbook = 0
        self.books = List[Int]()


struct _Mapping(Copyable, Movable):
    var submaps: Int
    var coupling_steps: Int
    var magnitude: List[Int]
    var angle: List[Int]
    var mux: List[Int]
    var submap_floor: List[Int]
    var submap_residue: List[Int]

    def __init__(out self):
        self.submaps = 1
        self.coupling_steps = 0
        self.magnitude = List[Int]()
        self.angle = List[Int]()
        self.mux = List[Int]()
        self.submap_floor = List[Int]()
        self.submap_residue = List[Int]()


@fieldwise_init
struct _Mode(Copyable, ImplicitlyCopyable, Movable):
    var blockflag: Int
    var mapping: Int


def _assign_codewords(lengths: List[UInt8], entries: Int, mut words: List[UInt32]) -> Bool:
    """libvorbis `_make_words`: codeword (MSB-first) for every used entry."""
    var marker = List[UInt32](length=33, fill=0)
    for i in range(entries):
        var length = Int(lengths[i])
        if length == 0:
            continue
        var entry = marker[length]
        if length < 32 and (entry >> UInt32(length)) != 0:
            return False
        words[i] = entry
        var j = length
        while j > 0:
            if (marker[j] & 1) != 0:
                if j == 1:
                    marker[1] += 1
                else:
                    marker[j] = marker[j - 1] << 1
                break
            marker[j] += 1
            j -= 1
        j = length + 1
        while j < 33:
            if (marker[j] >> 1) == entry:
                entry = marker[j]
                marker[j] = marker[j - 1] << 1
            else:
                break
            j += 1
    return True


def _build_tree(mut cb: _Codebook) -> Bool:
    var used = 0
    var last_used = -1
    for i in range(cb.entries):
        if cb.lengths[i] > 0:
            used += 1
            last_used = i
    if used == 0:
        return True
    if used == 1:
        cb.single_entry = last_used
        return True
    var words = List[UInt32](length=cb.entries, fill=0)
    if not _assign_codewords(cb.lengths, cb.entries, words):
        return False
    cb.tree = List[Int32](capacity=used * 4)
    cb.tree.append(_EMPTY)
    cb.tree.append(_EMPTY)
    for i in range(cb.entries):
        var length = Int(cb.lengths[i])
        if length == 0:
            continue
        var w = words[i]
        var node = 0
        for k in range(length - 1, -1, -1):
            var bit = Int((w >> UInt32(k)) & 1)
            var slot = node * 2 + bit
            if k == 0:
                if cb.tree[slot] != _EMPTY:
                    return False
                cb.tree[slot] = Int32(-(i + 1))
            else:
                var child = cb.tree[slot]
                if child == _EMPTY:
                    var new_node = len(cb.tree) // 2
                    cb.tree.append(_EMPTY)
                    cb.tree.append(_EMPTY)
                    cb.tree[slot] = Int32(new_node)
                    node = new_node
                elif child < 0:
                    return False  # prefix collision
                else:
                    node = Int(child)
    return True


@always_inline
def _decode_entry(cb: _Codebook, data: List[UInt8], mut br: _BitReader) -> Int:
    if cb.single_entry >= 0:
        _ = br.read(data, Int(cb.lengths[cb.single_entry]))
        return -1 if br.eop else cb.single_entry
    if len(cb.tree) == 0:
        return -1
    var node = 0
    while True:
        var bit = br.read1(data)
        if br.eop:
            return -1
        var child = cb.tree[node * 2 + bit]
        if child == _EMPTY:
            return -1
        if child < 0:
            return Int(-child) - 1
        node = Int(child)


# ==============================================================================
# Decoder
# ==============================================================================

struct VorbisClip(Copyable, Movable):
    var valid: Bool
    var error: String
    var channels: Int
    var sample_rate: Int
    var samples_per_channel: Int
    var samples: List[Int16]

    def __init__(out self):
        self.valid = False
        self.error = ""
        self.channels = 0
        self.sample_rate = 0
        self.samples_per_channel = 0
        self.samples = List[Int16]()


struct VorbisPcmF32(Movable):
    var valid: Bool
    var error: String
    var channels: Int
    var sample_rate: Int
    var samples_per_channel: Int
    var samples: List[Float32]  # interleaved

    def __init__(out self):
        self.valid = False
        self.error = ""
        self.channels = 0
        self.sample_rate = 0
        self.samples_per_channel = 0
        self.samples = List[Float32]()


struct _Imdct(Movable):
    """IMDCT of size `n` (n/2 coefficients -> n samples) via DCT-IV / (n/4)-point FFT."""
    var n: Int
    var m: Int
    var h: Int
    var pre_c: List[Float64]
    var pre_s: List[Float64]
    var post_c: List[Float64]
    var post_s: List[Float64]
    var tw_c: List[Float64]
    var tw_s: List[Float64]
    var bitrev: List[Int]
    var re: List[Float64]
    var im: List[Float64]
    var u: List[Float64]

    def __init__(out self, n: Int):
        self.n = n
        self.m = n // 2
        self.h = n // 4
        var m = self.m
        var h = self.h
        self.pre_c = List[Float64](length=h, fill=0.0)
        self.pre_s = List[Float64](length=h, fill=0.0)
        self.post_c = List[Float64](length=h, fill=0.0)
        self.post_s = List[Float64](length=h, fill=0.0)
        for k in range(h):
            var a = _PI * Float64(4 * k + 1) / Float64(4 * m)
            self.pre_c[k] = cos(a)
            self.pre_s[k] = -sin(a)
            var b = _PI * Float64(k) / Float64(m)
            self.post_c[k] = cos(b)
            self.post_s[k] = -sin(b)
        self.tw_c = List[Float64](length=max(1, h // 2), fill=0.0)
        self.tw_s = List[Float64](length=max(1, h // 2), fill=0.0)
        for k in range(h // 2):
            var t = -2.0 * _PI * Float64(k) / Float64(h)
            self.tw_c[k] = cos(t)
            self.tw_s[k] = sin(t)
        self.bitrev = List[Int](length=h, fill=0)
        var bits = _ilog(h) - 1
        for k in range(h):
            var r = 0
            var x = k
            for _ in range(bits):
                r = (r << 1) | (x & 1)
                x >>= 1
            self.bitrev[k] = r
        self.re = List[Float64](length=h, fill=0.0)
        self.im = List[Float64](length=h, fill=0.0)
        self.u = List[Float64](length=m, fill=0.0)

    def run(mut self, coeffs: List[Float32], coff: Int, mut out: List[Float32], ooff: Int):
        var m = self.m
        var h = self.h
        # pre-twiddle into bit-reversed order
        for k in range(h):
            var xr = Float64(coeffs[coff + 2 * k])
            var xi = Float64(coeffs[coff + m - 1 - 2 * k])
            var c = self.pre_c[k]
            var s = self.pre_s[k]
            var j = self.bitrev[k]
            self.re[j] = xr * c - xi * s
            self.im[j] = xr * s + xi * c
        # iterative radix-2 FFT
        var size = 2
        while size <= h:
            var half = size // 2
            var step = h // size
            var start = 0
            while start < h:
                for k in range(half):
                    var wc = self.tw_c[k * step]
                    var ws = self.tw_s[k * step]
                    var a = start + k
                    var b = a + half
                    var tr = self.re[b] * wc - self.im[b] * ws
                    var ti = self.re[b] * ws + self.im[b] * wc
                    self.re[b] = self.re[a] - tr
                    self.im[b] = self.im[a] - ti
                    self.re[a] = self.re[a] + tr
                    self.im[a] = self.im[a] + ti
                start += size
            size *= 2
        # post-twiddle -> DCT-IV
        for q in range(h):
            var c = self.post_c[q]
            var s = self.post_s[q]
            var zr = self.re[q] * c - self.im[q] * s
            var zi = self.re[q] * s + self.im[q] * c
            self.u[2 * q] = zr
            self.u[m - 1 - 2 * q] = -zi
        # DCT-IV -> IMDCT (n outputs) with the n0 = n/4 + 1/2 phase shift
        var n = self.n
        var half_m = m // 2
        for i in range(n):
            var mm = i + half_m
            var v: Float64
            if mm < m:
                v = self.u[mm]
            elif mm < 2 * m:
                v = -self.u[2 * m - 1 - mm]
            else:
                v = -self.u[mm - 2 * m]
            out[ooff + i] = Float32(v)


struct VorbisDecoder(Movable):
    var channels: Int
    var sample_rate: Int
    var bs0: Int
    var bs1: Int
    var books: List[_Codebook]
    var floors: List[_Floor1]
    var residues: List[_Residue]
    var mappings: List[_Mapping]
    var modes: List[_Mode]
    var error: String
    var imdct0: _Imdct
    var imdct1: _Imdct
    var win0: List[Float32]  # rising slope of length bs0/2
    var win1: List[Float32]  # rising slope of length bs1/2
    var inv_db: List[Float32]
    var prev: List[Float32]   # channels * bs1 windowed previous block
    var prev_n: Int
    var cur: List[Float32]    # channels * bs1
    var resid: List[Float32]  # channels * bs1/2
    var floor_y: List[Int]    # channels * 65
    var floor_buf: List[Float32]
    var nonzero: List[Bool]
    var no_resid: List[Bool]
    var cls: List[Int]
    var big: List[Float32]

    def __init__(out self):
        self.channels = 0
        self.sample_rate = 0
        self.bs0 = 0
        self.bs1 = 0
        self.books = List[_Codebook]()
        self.floors = List[_Floor1]()
        self.residues = List[_Residue]()
        self.mappings = List[_Mapping]()
        self.modes = List[_Mode]()
        self.error = ""
        self.imdct0 = _Imdct(64)
        self.imdct1 = _Imdct(64)
        self.win0 = List[Float32]()
        self.win1 = List[Float32]()
        self.inv_db = List[Float32](length=256, fill=0.0)
        var lnq = -log(1.0649863e-07) / 255.0
        for i in range(256):
            self.inv_db[i] = Float32(exp(Float64(i - 255) * lnq))
        self.prev = List[Float32]()
        self.prev_n = 0
        self.cur = List[Float32]()
        self.resid = List[Float32]()
        self.floor_y = List[Int]()
        self.floor_buf = List[Float32]()
        self.nonzero = List[Bool]()
        self.no_resid = List[Bool]()
        self.cls = List[Int]()
        self.big = List[Float32]()

    def _fail(mut self, msg: String) -> Bool:
        if self.error.byte_length() == 0:
            self.error = msg
        return False

    # ---- headers ----

    def parse_ident(mut self, data: List[UInt8], start: Int, length: Int) -> Bool:
        if length < 30 or data[start] != 1:
            return self._fail("vorbis: bad identification header")
        if not _is_vorbis_sig(data, start):
            return self._fail("vorbis: missing 'vorbis' signature")
        var version = _le32u(data, start + 7)
        if version != 0:
            return self._fail("vorbis: unsupported version")
        self.channels = Int(data[start + 11])
        self.sample_rate = Int(_le32u(data, start + 12))
        var bs = data[start + 28]
        self.bs0 = 1 << Int(bs & 0x0F)
        self.bs1 = 1 << Int((bs >> 4) & 0x0F)
        if self.channels < 1 or self.sample_rate <= 0:
            return self._fail("vorbis: bad channel count / rate")
        if self.bs0 < 64 or self.bs1 > 8192 or self.bs0 > self.bs1:
            return self._fail("vorbis: bad block sizes")
        if (data[start + 29] & 1) == 0:
            return self._fail("vorbis: identification framing bit")
        self.imdct0 = _Imdct(self.bs0)
        self.imdct1 = _Imdct(self.bs1)
        self.win0 = _vorbis_window(self.bs0 // 2)
        self.win1 = _vorbis_window(self.bs1 // 2)
        self.prev = List[Float32](length=self.channels * self.bs1, fill=0.0)
        self.cur = List[Float32](length=self.channels * self.bs1, fill=0.0)
        self.resid = List[Float32](length=self.channels * (self.bs1 // 2), fill=0.0)
        self.floor_y = List[Int](length=self.channels * 65, fill=0)
        self.floor_buf = List[Float32](length=self.bs1 // 2, fill=0.0)
        self.nonzero = List[Bool](length=self.channels, fill=False)
        self.no_resid = List[Bool](length=self.channels, fill=False)
        self.big = List[Float32](length=self.channels * (self.bs1 // 2), fill=0.0)
        return True

    def parse_setup(mut self, data: List[UInt8], start: Int, length: Int) -> Bool:
        if length < 7 or data[start] != 5 or not _is_vorbis_sig(data, start):
            return self._fail("vorbis: bad setup header")
        var br = _BitReader(start + 7, length - 7)
        # codebooks
        var nbooks = br.read(data, 8) + 1
        for _ in range(nbooks):
            var cb = _Codebook()
            if br.read(data, 24) != 0x564342:
                return self._fail("vorbis: codebook sync")
            cb.dims = br.read(data, 16)
            cb.entries = br.read(data, 24)
            if cb.dims < 1 or cb.entries < 1:
                return self._fail("vorbis: empty codebook")
            # Every entry costs at least one bit (sparse flag) unless ordered: reject sizes the
            # remaining setup packet cannot possibly describe (guards hostile allocations).
            if cb.entries > br.end_bits - br.pos + 64:
                return self._fail("vorbis: codebook larger than setup packet")
            cb.lengths = List[UInt8](length=cb.entries, fill=0)
            var ordered = br.read(data, 1)
            if ordered == 0:
                var sparse = br.read(data, 1)
                for i in range(cb.entries):
                    if sparse != 0:
                        if br.read(data, 1) != 0:
                            cb.lengths[i] = UInt8(br.read(data, 5) + 1)
                    else:
                        cb.lengths[i] = UInt8(br.read(data, 5) + 1)
            else:
                var cur_len = br.read(data, 5) + 1
                var cur_entry = 0
                while cur_entry < cb.entries:
                    var num = br.read(data, _ilog(cb.entries - cur_entry))
                    if cur_entry + num > cb.entries or cur_len > 32:
                        return self._fail("vorbis: ordered codebook overflow")
                    for k in range(num):
                        cb.lengths[cur_entry + k] = UInt8(cur_len)
                    cur_entry += num
                    cur_len += 1
            cb.lookup_type = br.read(data, 4)
            if cb.lookup_type == 1 or cb.lookup_type == 2:
                var minv = _float32_unpack(br.read(data, 32))
                var delta = _float32_unpack(br.read(data, 32))
                var value_bits = br.read(data, 4) + 1
                var seq_p = br.read(data, 1) != 0
                var nvals: Int
                if cb.lookup_type == 1:
                    nvals = _lookup1_values(cb.entries, cb.dims)
                else:
                    nvals = cb.entries * cb.dims
                if nvals * value_bits > br.end_bits - br.pos or cb.entries * cb.dims > (1 << 24):
                    return self._fail("vorbis: VQ table exceeds setup packet")
                var mult = List[Int](length=nvals, fill=0)
                for i in range(nvals):
                    mult[i] = br.read(data, value_bits)
                if br.eop:
                    return self._fail("vorbis: truncated codebook")
                cb.vq = List[Float32](length=cb.entries * cb.dims, fill=0.0)
                for e in range(cb.entries):
                    var last = Float64(0.0)
                    var div = 1
                    for d in range(cb.dims):
                        var off: Int
                        if cb.lookup_type == 1:
                            off = (e // div) % nvals
                            div *= nvals
                        else:
                            off = e * cb.dims + d
                        var v = Float64(mult[off]) * delta + minv + last
                        if seq_p:
                            last = v
                        cb.vq[e * cb.dims + d] = Float32(v)
            elif cb.lookup_type != 0:
                return self._fail("vorbis: bad lookup type")
            if br.eop:
                return self._fail("vorbis: truncated setup")
            if not _build_tree(cb):
                return self._fail("vorbis: invalid Huffman lengths")
            self.books.append(cb^)

        # time domain transforms (placeholders)
        var ntimes = br.read(data, 6) + 1
        for _ in range(ntimes):
            if br.read(data, 16) != 0:
                return self._fail("vorbis: bad time domain transform")

        # floors
        var nfloors = br.read(data, 6) + 1
        for _ in range(nfloors):
            var ftype = br.read(data, 16)
            if ftype == 0:
                return self._fail("vorbis: floor type 0 is not supported")
            if ftype != 1:
                return self._fail("vorbis: bad floor type")
            var f = _Floor1()
            f.partitions = br.read(data, 5)
            var max_class = -1
            for _ in range(f.partitions):
                var c = br.read(data, 4)
                f.partition_class.append(c)
                max_class = max(max_class, c)
            for c in range(max_class + 1):
                f.class_dims[c] = br.read(data, 3) + 1
                f.class_subclasses[c] = br.read(data, 2)
                if f.class_subclasses[c] != 0:
                    f.class_master[c] = br.read(data, 8)
                    if f.class_master[c] >= len(self.books):
                        return self._fail("vorbis: floor masterbook")
                for j in range(1 << f.class_subclasses[c]):
                    var b = br.read(data, 8) - 1
                    if b >= len(self.books):
                        return self._fail("vorbis: floor subclass book")
                    f.subclass_books[c * 8 + j] = b
            f.multiplier = br.read(data, 2) + 1
            var rangebits = br.read(data, 4)
            f.xlist.append(0)
            f.xlist.append(1 << rangebits)
            for p in range(f.partitions):
                var c = f.partition_class[p]
                for _ in range(f.class_dims[c]):
                    f.xlist.append(br.read(data, rangebits))
            var nv = len(f.xlist)
            if nv > 65:
                return self._fail("vorbis: too many floor points")
            # sorted order and neighbors
            f.sorted_idx = List[Int](length=nv, fill=0)
            for i in range(nv):
                f.sorted_idx[i] = i
            for i in range(1, nv):
                var j = i
                while j > 0 and f.xlist[f.sorted_idx[j - 1]] > f.xlist[f.sorted_idx[j]]:
                    var t = f.sorted_idx[j - 1]
                    f.sorted_idx[j - 1] = f.sorted_idx[j]
                    f.sorted_idx[j] = t
                    j -= 1
            for i in range(1, nv):
                if f.xlist[f.sorted_idx[i]] == f.xlist[f.sorted_idx[i - 1]]:
                    return self._fail("vorbis: duplicate floor X")
            f.low_n = List[Int](length=nv, fill=0)
            f.high_n = List[Int](length=nv, fill=0)
            for i in range(2, nv):
                var lo = -1
                var hi = -1
                var lx = -1
                var hx = 1 << 30
                for j in range(i):
                    var xj = f.xlist[j]
                    if xj < f.xlist[i] and xj > lx:
                        lx = xj
                        lo = j
                    if xj > f.xlist[i] and xj < hx:
                        hx = xj
                        hi = j
                f.low_n[i] = lo
                f.high_n[i] = hi
            self.floors.append(f^)

        # residues
        var nres = br.read(data, 6) + 1
        for _ in range(nres):
            var r = _Residue()
            r.rtype = br.read(data, 16)
            if r.rtype > 2:
                return self._fail("vorbis: bad residue type")
            r.begin = br.read(data, 24)
            r.end = br.read(data, 24)
            r.part_size = br.read(data, 24) + 1
            r.classifications = br.read(data, 6) + 1
            r.classbook = br.read(data, 8)
            if r.classbook >= len(self.books):
                return self._fail("vorbis: residue classbook")
            var cascade = List[Int](length=r.classifications, fill=0)
            for c in range(r.classifications):
                var low = br.read(data, 3)
                var high = 0
                if br.read(data, 1) != 0:
                    high = br.read(data, 5)
                cascade[c] = high * 8 + low
            r.books = List[Int](length=r.classifications * 8, fill=-1)
            for c in range(r.classifications):
                for j in range(8):
                    if (cascade[c] & (1 << j)) != 0:
                        var b = br.read(data, 8)
                        if b >= len(self.books):
                            return self._fail("vorbis: residue book")
                        if self.books[b].lookup_type == 0:
                            return self._fail("vorbis: residue book without VQ")
                        r.books[c * 8 + j] = b
            self.residues.append(r^)

        # mappings
        var nmaps = br.read(data, 6) + 1
        for _ in range(nmaps):
            if br.read(data, 16) != 0:
                return self._fail("vorbis: bad mapping type")
            var mp = _Mapping()
            if br.read(data, 1) != 0:
                mp.submaps = br.read(data, 4) + 1
            if br.read(data, 1) != 0:
                mp.coupling_steps = br.read(data, 8) + 1
                var cbits = _ilog(self.channels - 1)
                for _ in range(mp.coupling_steps):
                    var mg = br.read(data, cbits)
                    var an = br.read(data, cbits)
                    if mg == an or mg >= self.channels or an >= self.channels:
                        return self._fail("vorbis: bad coupling")
                    mp.magnitude.append(mg)
                    mp.angle.append(an)
            if br.read(data, 2) != 0:
                return self._fail("vorbis: mapping reserved bits")
            mp.mux = List[Int](length=self.channels, fill=0)
            if mp.submaps > 1:
                for c in range(self.channels):
                    mp.mux[c] = br.read(data, 4)
                    if mp.mux[c] >= mp.submaps:
                        return self._fail("vorbis: bad mux")
            for _ in range(mp.submaps):
                _ = br.read(data, 8)
                var fl = br.read(data, 8)
                var rs = br.read(data, 8)
                if fl >= len(self.floors) or rs >= len(self.residues):
                    return self._fail("vorbis: submap floor/residue")
                mp.submap_floor.append(fl)
                mp.submap_residue.append(rs)
            self.mappings.append(mp^)

        # modes
        var nmodes = br.read(data, 6) + 1
        for _ in range(nmodes):
            var bf = br.read(data, 1)
            var wt = br.read(data, 16)
            var tt = br.read(data, 16)
            var mapn = br.read(data, 8)
            if wt != 0 or tt != 0 or mapn >= len(self.mappings):
                return self._fail("vorbis: bad mode")
            self.modes.append(_Mode(bf, mapn))
        if br.read(data, 1) != 1 or br.eop:
            return self._fail("vorbis: setup framing bit")
        return True

    # ---- floor 1 ----

    def _floor1_read(mut self, fi: Int, data: List[UInt8], mut br: _BitReader, ch: Int) -> Bool:
        var mult = self.floors[fi].multiplier
        var range_v = 256
        if mult == 2:
            range_v = 128
        elif mult == 3:
            range_v = 86
        elif mult == 4:
            range_v = 64
        if br.read1(data) == 0:
            return False
        var ybase = ch * 65
        var rb = _ilog(range_v - 1)
        self.floor_y[ybase] = br.read(data, rb)
        self.floor_y[ybase + 1] = br.read(data, rb)
        var offset = 2
        for p in range(self.floors[fi].partitions):
            var c = self.floors[fi].partition_class[p]
            var cdim = self.floors[fi].class_dims[c]
            var cbits = self.floors[fi].class_subclasses[c]
            var csub = (1 << cbits) - 1
            var cval = 0
            if cbits > 0:
                cval = _decode_entry(self.books[self.floors[fi].class_master[c]], data, br)
                if cval < 0:
                    return False
            for j in range(cdim):
                var book = self.floors[fi].subclass_books[c * 8 + (cval & csub)]
                cval >>= cbits
                if book >= 0:
                    var v = _decode_entry(self.books[book], data, br)
                    if v < 0:
                        return False
                    self.floor_y[ybase + offset + j] = v
                else:
                    self.floor_y[ybase + offset + j] = 0
            offset += cdim
        return not br.eop

    def _floor1_curve(mut self, fi: Int, ch: Int, half: Int):
        """Synthesizes the floor curve into `floor_buf[0:half]`."""
        var mult = self.floors[fi].multiplier
        var range_v = 256
        if mult == 2:
            range_v = 128
        elif mult == 3:
            range_v = 86
        elif mult == 4:
            range_v = 64
        var nv = len(self.floors[fi].xlist)
        var ybase = ch * 65
        var final_y = List[Int](length=nv, fill=0)
        var step2 = List[Bool](length=nv, fill=False)
        step2[0] = True
        step2[1] = True
        final_y[0] = self.floor_y[ybase]
        final_y[1] = self.floor_y[ybase + 1]
        for i in range(2, nv):
            var lo = self.floors[fi].low_n[i]
            var hi = self.floors[fi].high_n[i]
            var predicted = _render_point(
                self.floors[fi].xlist[lo], final_y[lo], self.floors[fi].xlist[hi], final_y[hi], self.floors[fi].xlist[i]
            )
            var val = self.floor_y[ybase + i]
            var highroom = range_v - predicted
            var lowroom = predicted
            var room = (highroom if highroom < lowroom else lowroom) * 2
            if val != 0:
                step2[lo] = True
                step2[hi] = True
                step2[i] = True
                if val >= room:
                    if highroom > lowroom:
                        final_y[i] = val - lowroom + predicted
                    else:
                        final_y[i] = predicted - val + highroom - 1
                else:
                    if (val & 1) != 0:
                        final_y[i] = predicted - (val + 1) // 2
                    else:
                        final_y[i] = predicted + val // 2
            else:
                step2[i] = False
                final_y[i] = predicted
        var hx = 0
        var hy = 0
        var lx = 0
        var ly = final_y[self.floors[fi].sorted_idx[0]] * mult
        for k in range(1, nv):
            var j = self.floors[fi].sorted_idx[k]
            if step2[j]:
                hy = final_y[j] * mult
                hx = self.floors[fi].xlist[j]
                self._render_line(lx, ly, hx, hy, half)
                lx = hx
                ly = hy
        if hx < half:
            self._render_line(hx, hy, half, hy, half)

    def _render_line(mut self, x0: Int, y0: Int, x1: Int, y1: Int, limit: Int):
        var dy = y1 - y0
        var adx = x1 - x0
        if adx <= 0:
            return
        var ady = abs(dy)
        var base = dy // adx if dy >= 0 else -((-dy) // adx)
        var sy = base - 1 if dy < 0 else base + 1
        var x = x0
        var y = y0
        var err = 0
        ady = ady - abs(base) * adx
        if x < limit:
            self.floor_buf[x] = self.inv_db[max(0, min(255, y))]
        x += 1
        while x < x1:
            err += ady
            if err >= adx:
                err -= adx
                y += sy
            else:
                y += base
            if x < limit:
                self.floor_buf[x] = self.inv_db[max(0, min(255, y))]
            x += 1

    # ---- residue ----

    def _residue_partitions(
        mut self, ri: Int, data: List[UInt8], mut br: _BitReader,
        vecs: List[Int], dnd: List[Bool], size: Int, use_big: Bool, format0: Bool,
    ):
        """Decodes partitions of residue `ri` into the vectors at offsets `vecs` (into `resid`,
        or `big` when `use_big`), each of length `size`."""
        var begin = min(self.residues[ri].begin, size)
        var end = min(self.residues[ri].end, size)
        var n_to_read = end - begin
        if n_to_read <= 0:
            return
        var psize = self.residues[ri].part_size
        var parts = n_to_read // psize
        if parts <= 0:
            return
        var classbook = self.residues[ri].classbook
        var cpc = self.books[classbook].dims
        var nclass = self.residues[ri].classifications
        var nvec = len(vecs)
        var cstride = parts + cpc
        if len(self.cls) < nvec * cstride:
            self.cls.resize(nvec * cstride, 0)
        for pass_i in range(8):
            var pc = 0
            while pc < parts:
                if pass_i == 0:
                    for j in range(nvec):
                        if dnd[j]:
                            continue
                        var temp = _decode_entry(self.books[classbook], data, br)
                        if temp < 0:
                            return
                        for i in range(cpc - 1, -1, -1):
                            self.cls[j * cstride + pc + i] = temp % nclass
                            temp = temp // nclass
                var i2 = 0
                while i2 < cpc and pc < parts:
                    for j in range(nvec):
                        if dnd[j]:
                            continue
                        var vqclass = self.cls[j * cstride + pc]
                        var book = self.residues[ri].books[vqclass * 8 + pass_i]
                        if book < 0:
                            continue
                        var off = vecs[j] + begin + pc * psize
                        var ok: Bool
                        if format0:
                            ok = self._decode_format0(book, data, br, off, psize, use_big)
                        else:
                            ok = self._decode_format1(book, data, br, off, psize, use_big)
                        if not ok:
                            return
                    i2 += 1
                    pc += 1

    def _decode_format0(mut self, book: Int, data: List[UInt8], mut br: _BitReader, off: Int, psize: Int, use_big: Bool) -> Bool:
        var dims = self.books[book].dims
        var step = psize // dims
        for i in range(step):
            var e = _decode_entry(self.books[book], data, br)
            if e < 0:
                return False
            for j in range(dims):
                var v = self.books[book].vq[e * dims + j]
                if use_big:
                    self.big[off + i + j * step] += v
                else:
                    self.resid[off + i + j * step] += v
        return True

    def _decode_format1(mut self, book: Int, data: List[UInt8], mut br: _BitReader, off: Int, psize: Int, use_big: Bool) -> Bool:
        var dims = self.books[book].dims
        var i = 0
        while i < psize:
            var e = _decode_entry(self.books[book], data, br)
            if e < 0:
                return False
            for j in range(dims):
                if i >= psize:
                    break
                var v = self.books[book].vq[e * dims + j]
                if use_big:
                    self.big[off + i] += v
                else:
                    self.resid[off + i] += v
                i += 1
        return True

    # ---- audio packet ----

    def decode_packet(mut self, data: List[UInt8], start: Int, length: Int, mut out: List[Float32]) -> Int:
        """Decodes one audio packet, appending interleaved float frames to `out`. Returns frames."""
        if length <= 0:
            return 0
        var br = _BitReader(start, length)
        if br.read(data, 1) != 0:
            return 0
        var mode = br.read(data, _ilog(len(self.modes) - 1))
        if br.eop or mode >= len(self.modes):
            return 0
        var blockflag = self.modes[mode].blockflag
        var mi = self.modes[mode].mapping
        var n = self.bs1 if blockflag != 0 else self.bs0
        var half = n // 2
        var prev_flag = 0
        var next_flag = 0
        if blockflag != 0:
            prev_flag = br.read(data, 1)
            next_flag = br.read(data, 1)
        var nch = self.channels
        var rstride = self.bs1 // 2

        # 1. floors
        for ch in range(nch):
            var sm = self.mappings[mi].mux[ch]
            var fi = self.mappings[mi].submap_floor[sm]
            self.nonzero[ch] = self._floor1_read(fi, data, br, ch)
            self.no_resid[ch] = not self.nonzero[ch]
        # 2. nonzero propagation through coupling
        for s in range(self.mappings[mi].coupling_steps):
            var mg = self.mappings[mi].magnitude[s]
            var an = self.mappings[mi].angle[s]
            if not self.no_resid[mg] or not self.no_resid[an]:
                self.no_resid[mg] = False
                self.no_resid[an] = False
        # 3. residues per submap
        for ch in range(nch):
            for i in range(half):
                self.resid[ch * rstride + i] = 0.0
        for sm in range(self.mappings[mi].submaps):
            var vecs = List[Int]()
            var dnd = List[Bool]()
            var chans = List[Int]()
            for ch in range(nch):
                if self.mappings[mi].mux[ch] == sm:
                    vecs.append(ch * rstride)
                    dnd.append(self.no_resid[ch])
                    chans.append(ch)
            var ri = self.mappings[mi].submap_residue[sm]
            var rtype = self.residues[ri].rtype
            if rtype == 2:
                var any = False
                for j in range(len(dnd)):
                    if not dnd[j]:
                        any = True
                if not any:
                    continue
                var cnt = len(chans)
                var total = half * cnt
                for i in range(total):
                    self.big[i] = 0.0
                var bv = List[Int]()
                bv.append(0)
                var bd = List[Bool]()
                bd.append(False)
                self._residue_partitions(ri, data, br, bv, bd, total, True, False)
                for i in range(total):
                    var c = chans[i % cnt]
                    self.resid[c * rstride + i // cnt] += self.big[i]
            else:
                self._residue_partitions(ri, data, br, vecs, dnd, half, False, rtype == 0)
        # 4. inverse coupling
        for s in range(self.mappings[mi].coupling_steps - 1, -1, -1):
            var mb = self.mappings[mi].magnitude[s] * rstride
            var ab = self.mappings[mi].angle[s] * rstride
            for i in range(half):
                var m = self.resid[mb + i]
                var a = self.resid[ab + i]
                var nm: Float32
                var na: Float32
                if m > 0.0:
                    if a > 0.0:
                        nm = m
                        na = m - a
                    else:
                        na = m
                        nm = m + a
                else:
                    if a > 0.0:
                        nm = m
                        na = m + a
                    else:
                        na = m
                        nm = m - a
                self.resid[mb + i] = nm
                self.resid[ab + i] = na
        # 5. floor curve x residue, IMDCT, window
        var left_start: Int
        var left_end: Int
        var right_start: Int
        var right_end: Int
        var left_long: Bool
        var right_long: Bool
        if blockflag != 0 and prev_flag == 0:
            left_start = n // 4 - self.bs0 // 4
            left_end = n // 4 + self.bs0 // 4
            left_long = False
        else:
            left_start = 0
            left_end = half
            left_long = blockflag != 0
        if blockflag != 0 and next_flag == 0:
            right_start = (n * 3) // 4 - self.bs0 // 4
            right_end = (n * 3) // 4 + self.bs0 // 4
            right_long = False
        else:
            right_start = half
            right_end = n
            right_long = blockflag != 0
        for ch in range(nch):
            var cb = ch * self.bs1
            if not self.nonzero[ch]:
                for i in range(n):
                    self.cur[cb + i] = 0.0
                continue
            var sm = self.mappings[mi].mux[ch]
            var fi = self.mappings[mi].submap_floor[sm]
            self._floor1_curve(fi, ch, half)
            for i in range(half):
                self.resid[ch * rstride + i] *= self.floor_buf[i]
            if blockflag != 0:
                self.imdct1.run(self.resid, ch * rstride, self.cur, cb)
            else:
                self.imdct0.run(self.resid, ch * rstride, self.cur, cb)
            for i in range(n):
                var w: Float32
                if i < left_start or i >= right_end:
                    w = 0.0
                elif i < left_end:
                    w = self.win1[i - left_start] if left_long else self.win0[i - left_start]
                elif i < right_start:
                    w = 1.0
                else:
                    var k = right_end - 1 - i
                    w = self.win1[k] if right_long else self.win0[k]
                self.cur[cb + i] *= w
        # 6. overlap-add with the previous block
        var produced = 0
        if self.prev_n > 0:
            var pn = self.prev_n
            var count = pn // 4 + n // 4
            for p in range(count):
                var pi = p + pn // 2
                var ci = p - pn // 4 + n // 4
                for ch in range(nch):
                    var v = Float32(0.0)
                    if pi < pn:
                        v += self.prev[ch * self.bs1 + pi]
                    if ci >= 0 and ci < n:
                        v += self.cur[ch * self.bs1 + ci]
                    out.append(v)
            produced = count
        for ch in range(nch):
            for i in range(n):
                self.prev[ch * self.bs1 + i] = self.cur[ch * self.bs1 + i]
        self.prev_n = n
        return produced


def _is_vorbis_sig(data: List[UInt8], start: Int) -> Bool:
    return (
        data[start + 1] == 118 and data[start + 2] == 111 and data[start + 3] == 114
        and data[start + 4] == 98 and data[start + 5] == 105 and data[start + 6] == 115
    )


def _vorbis_window(n: Int) -> List[Float32]:
    """Rising Vorbis power-sine slope of length `n`: sin(pi/2 * sin^2((i + .5) / n * pi/2))."""
    var w = List[Float32](length=n, fill=0.0)
    for i in range(n):
        var s = sin((Float64(i) + 0.5) / Float64(n) * _PI * 0.5)
        w[i] = Float32(sin(_PI * 0.5 * s * s))
    return w^


@always_inline
def _render_point(x0: Int, y0: Int, x1: Int, y1: Int, x: Int) -> Int:
    var dy = y1 - y0
    var adx = x1 - x0
    var ady = abs(dy)
    var err = ady * (x - x0)
    var off = err // adx if adx != 0 else 0
    return y0 - off if dy < 0 else y0 + off


def is_ogg_vorbis(data: List[UInt8]) -> Bool:
    return len(data) >= 4 and data[0] == 79 and data[1] == 103 and data[2] == 103 and data[3] == 83


def decode_ogg_vorbis_f32(src: List[UInt8]) -> VorbisPcmF32:
    """Decodes a complete Ogg Vorbis file into interleaved float PCM."""
    var res = VorbisPcmF32()
    if not is_ogg_vorbis(src):
        res.error = "vorbis: not an Ogg stream"
        return res^
    var pk = _read_ogg_packets(src)
    if len(pk.starts) < 3:
        res.error = "vorbis: missing header packets"
        return res^
    var dec = VorbisDecoder()
    if not dec.parse_ident(pk.data, pk.starts[0], pk.lens[0]):
        res.error = dec.error
        return res^
    if pk.lens[1] < 7 or pk.data[pk.starts[1]] != 3 or not _is_vorbis_sig(pk.data, pk.starts[1]):
        res.error = "vorbis: bad comment header"
        return res^
    if not dec.parse_setup(pk.data, pk.starts[2], pk.lens[2]):
        res.error = dec.error
        return res^
    var out = List[Float32](capacity=max(0, Int(pk.last_granule)) * dec.channels + dec.bs1 * dec.channels)
    var total = 0
    var first_granule_seen = False
    for i in range(3, len(pk.starts)):
        total += dec.decode_packet(pk.data, pk.starts[i], pk.lens[i], out)
        if not first_granule_seen and pk.granule[i] >= 0:
            first_granule_seen = True
            # Start trim: a first audio page whose granule is smaller than the decoded
            # sample count marks leading samples to discard (spec A.2). Not applied on the
            # final page, where the end trim below governs.
            var g = Int(pk.granule[i])
            if pk.granule[i] != pk.last_granule and g < total:
                var drop = total - g
                var nch = dec.channels
                var kept = List[Float32](capacity=len(out))
                for k in range(drop * nch, len(out)):
                    kept.append(out[k])
                out = kept^
                total = g
    # End trim to the final granule position (the encoder pads the last block).
    if pk.last_granule >= 0 and Int(pk.last_granule) < total:
        total = Int(pk.last_granule)
        out.resize(total * dec.channels, 0.0)
    res.valid = True
    res.channels = dec.channels
    res.sample_rate = dec.sample_rate
    res.samples_per_channel = total
    res.samples = out^
    return res^


def decode_ogg_vorbis(src: List[UInt8]) -> VorbisClip:
    """Decodes a complete Ogg Vorbis file into interleaved 16-bit PCM (round-to-nearest, clipped)."""
    var clip = VorbisClip()
    var f = decode_ogg_vorbis_f32(src)
    if not f.valid:
        clip.error = f.error
        return clip^
    clip.channels = f.channels
    clip.sample_rate = f.sample_rate
    clip.samples_per_channel = f.samples_per_channel
    clip.samples = List[Int16](capacity=len(f.samples))
    for i in range(len(f.samples)):
        var v = f.samples[i] * Float32(32768.0)
        var r = Int(floor(v + Float32(0.5)))
        if r > 32767:
            r = 32767
        elif r < -32768:
            r = -32768
        clip.samples.append(Int16(r))
    clip.valid = True
    return clip^
