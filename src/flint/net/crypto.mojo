"""Pure-Mojo RFC 8439 ChaCha20-Poly1305 AEAD, 64-Bit Replay Window & Stateless Cookie Auth (`flint.net.crypto`).

Implements:
- `AeadKey`: 256-bit symmetric key (8 x `UInt32` little-endian words, zero heap allocations).
- `chacha20_block` & `chacha20_xor`: RFC 8439 Section 2.3 stream cipher.
- `poly1305_mac` & `poly1305_verify_tag`: RFC 8439 Section 2.5 130-bit MAC using 5x26-bit limbs
  and constant-time tag comparison.
- `aead_seal` & `aead_open`: RFC 8439 Section 2.8 AEAD with 16-byte authentication tag.
- `ReplayWindow64`: 65-packet sliding anti-replay window for 64-bit packet nonces.
- `StatelessCookieAuth`: Zero-state server handshake cookie generator/verifier using Poly1305 MAC
  over `(peer_endpoint, time_bucket)` to defeat IP-spoofed slot-exhaustion DoS.
"""
from std.collections import List
from flint.net.socket_udp import NetEndpoint

comptime AEAD_KEY_BYTES: Int = 32
comptime AEAD_NONCE_BYTES: Int = 8
comptime AEAD_TAG_BYTES: Int = 16
comptime AEAD_OVERHEAD_BYTES: Int = AEAD_NONCE_BYTES + AEAD_TAG_BYTES  # 24 bytes


@always_inline
def _rotl32(v: UInt32, n: UInt32) -> UInt32:
    return (v << n) | (v >> (UInt32(32) - n))


@always_inline
def _load_u32_le_ptr(ptr: Int, off: Int) -> UInt32:
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=ptr + off)
    return (
        UInt32(p[unsafe_offset=0])
        | (UInt32(p[unsafe_offset=1]) << 8)
        | (UInt32(p[unsafe_offset=2]) << 16)
        | (UInt32(p[unsafe_offset=3]) << 24)
    )


@always_inline
def _store_u32_le_arr(mut out: Array[UInt8, 64], off: Int, v: UInt32):
    out[off] = UInt8(v & 0xFF)
    out[off + 1] = UInt8((v >> 8) & 0xFF)
    out[off + 2] = UInt8((v >> 16) & 0xFF)
    out[off + 3] = UInt8((v >> 24) & 0xFF)


@fieldwise_init
struct AeadKey(Copyable, ImplicitlyCopyable, Movable):
    """256-bit ChaCha20-Poly1305 key stored as 8 little-endian `UInt32` words."""
    var w0: UInt32
    var w1: UInt32
    var w2: UInt32
    var w3: UInt32
    var w4: UInt32
    var w5: UInt32
    var w6: UInt32
    var w7: UInt32

    def __init__(out self):
        self.w0 = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.w4 = 0
        self.w5 = 0
        self.w6 = 0
        self.w7 = 0

    @staticmethod
    def from_bytes(key_bytes: List[UInt8]) -> Self:
        if len(key_bytes) < 32:
            return Self()
        var ptr = Int(key_bytes.unsafe_ptr())
        var k = Self(
            _load_u32_le_ptr(ptr, 0),
            _load_u32_le_ptr(ptr, 4),
            _load_u32_le_ptr(ptr, 8),
            _load_u32_le_ptr(ptr, 12),
            _load_u32_le_ptr(ptr, 16),
            _load_u32_le_ptr(ptr, 20),
            _load_u32_le_ptr(ptr, 24),
            _load_u32_le_ptr(ptr, 28),
        )
        _ = len(key_bytes)
        return k

    @staticmethod
    def from_u64x4(k0: UInt64, k1: UInt64, k2: UInt64, k3: UInt64) -> Self:
        return Self(
            UInt32(k0 & 0xFFFFFFFF),
            UInt32(k0 >> 32),
            UInt32(k1 & 0xFFFFFFFF),
            UInt32(k1 >> 32),
            UInt32(k2 & 0xFFFFFFFF),
            UInt32(k2 >> 32),
            UInt32(k3 & 0xFFFFFFFF),
            UInt32(k3 >> 32),
        )


@always_inline
def _qr(mut s: Array[UInt32, 16], a: Int, b: Int, c: Int, d: Int):
    s[a] += s[b]
    s[d] = _rotl32(s[d] ^ s[a], 16)
    s[c] += s[d]
    s[b] = _rotl32(s[b] ^ s[c], 12)
    s[a] += s[b]
    s[d] = _rotl32(s[d] ^ s[a], 8)
    s[c] += s[d]
    s[b] = _rotl32(s[b] ^ s[c], 7)


def chacha20_block(
    key: AeadKey,
    counter: UInt32,
    n0: UInt32,
    n1: UInt32,
    n2: UInt32,
) -> Array[UInt8, 64]:
    """Computes one 64-byte RFC 8439 ChaCha20 keystream block."""
    var init = Array[UInt32, 16](fill=UInt32(0))
    init[0] = UInt32(0x61707865)
    init[1] = UInt32(0x3320646E)
    init[2] = UInt32(0x79622D32)
    init[3] = UInt32(0x6B206574)
    init[4] = key.w0
    init[5] = key.w1
    init[6] = key.w2
    init[7] = key.w3
    init[8] = key.w4
    init[9] = key.w5
    init[10] = key.w6
    init[11] = key.w7
    init[12] = counter
    init[13] = n0
    init[14] = n1
    init[15] = n2

    var s = init.copy()
    for _ in range(10):
        # Column rounds
        _qr(s, 0, 4, 8, 12)
        _qr(s, 1, 5, 9, 13)
        _qr(s, 2, 6, 10, 14)
        _qr(s, 3, 7, 11, 15)
        # Diagonal rounds
        _qr(s, 0, 5, 10, 15)
        _qr(s, 1, 6, 11, 12)
        _qr(s, 2, 7, 8, 13)
        _qr(s, 3, 4, 9, 14)

    var out = Array[UInt8, 64](fill=UInt8(0))
    for i in range(16):
        _store_u32_le_arr(out, i * 4, s[i] + init[i])
    return out^


def chacha20_xor_ptr(
    key: AeadKey,
    initial_counter: UInt32,
    n0: UInt32,
    n1: UInt32,
    n2: UInt32,
    in_ptr: Int,
    length: Int,
    out_ptr: Int,
):
    """XORs `length` bytes from `in_ptr` into `out_ptr` with ChaCha20 keystream."""
    if length <= 0 or in_ptr == 0 or out_ptr == 0:
        return
    var src = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=in_ptr)
    var dst = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=out_ptr)
    var ctr = initial_counter
    var off = 0
    while off < length:
        var block = chacha20_block(key, ctr, n0, n1, n2)
        ctr += 1
        var chunk = min(64, length - off)
        for i in range(chunk):
            dst[unsafe_offset=off + i] = src[unsafe_offset=off + i] ^ block[i]
        off += chunk


struct _Poly1305State:
    """RFC 8439 Poly1305 incremental MAC using 5x26-bit limbs (zero 64-bit overflow)."""
    var r0: UInt64
    var r1: UInt64
    var r2: UInt64
    var r3: UInt64
    var r4: UInt64
    var s1: UInt64
    var s2: UInt64
    var s3: UInt64
    var s4: UInt64
    var h0: UInt64
    var h1: UInt64
    var h2: UInt64
    var h3: UInt64
    var h4: UInt64
    var pad0: UInt64
    var pad1: UInt64
    var pad2: UInt64
    var pad3: UInt64
    var buf: Array[UInt8, 16]
    var buf_len: Int

    def __init__(out self, otk_block: Array[UInt8, 64]):
        # Load r[0..16] little-endian and clamp per RFC 8439 Section 2.5
        var t0 = (
            UInt64(otk_block[0])
            | (UInt64(otk_block[1]) << 8)
            | (UInt64(otk_block[2]) << 16)
            | (UInt64(otk_block[3]) << 24)
        )
        var t1 = (
            UInt64(otk_block[4])
            | (UInt64(otk_block[5]) << 8)
            | (UInt64(otk_block[6]) << 16)
            | (UInt64(otk_block[7]) << 24)
        )
        var t2 = (
            UInt64(otk_block[8])
            | (UInt64(otk_block[9]) << 8)
            | (UInt64(otk_block[10]) << 16)
            | (UInt64(otk_block[11]) << 24)
        )
        var t3 = (
            UInt64(otk_block[12])
            | (UInt64(otk_block[13]) << 8)
            | (UInt64(otk_block[14]) << 16)
            | (UInt64(otk_block[15]) << 24)
        )

        self.r0 = t0 & UInt64(0x03FFFFFF)
        self.r1 = ((t0 >> 26) | (t1 << 6)) & UInt64(0x03FFFF03)
        self.r2 = ((t1 >> 20) | (t2 << 12)) & UInt64(0x03FFC0FF)
        self.r3 = ((t2 >> 14) | (t3 << 18)) & UInt64(0x03F03FFF)
        self.r4 = (t3 >> 8) & UInt64(0x000FFFFF)

        self.s1 = self.r1 * 5
        self.s2 = self.r2 * 5
        self.s3 = self.r3 * 5
        self.s4 = self.r4 * 5

        self.h0 = 0
        self.h1 = 0
        self.h2 = 0
        self.h3 = 0
        self.h4 = 0

        self.pad0 = (
            UInt64(otk_block[16])
            | (UInt64(otk_block[17]) << 8)
            | (UInt64(otk_block[18]) << 16)
            | (UInt64(otk_block[19]) << 24)
        )
        self.pad1 = (
            UInt64(otk_block[20])
            | (UInt64(otk_block[21]) << 8)
            | (UInt64(otk_block[22]) << 16)
            | (UInt64(otk_block[23]) << 24)
        )
        self.pad2 = (
            UInt64(otk_block[24])
            | (UInt64(otk_block[25]) << 8)
            | (UInt64(otk_block[26]) << 16)
            | (UInt64(otk_block[27]) << 24)
        )
        self.pad3 = (
            UInt64(otk_block[28])
            | (UInt64(otk_block[29]) << 8)
            | (UInt64(otk_block[30]) << 16)
            | (UInt64(otk_block[31]) << 24)
        )
        self.buf = Array[UInt8, 16](fill=UInt8(0))
        self.buf_len = 0

    def _process_buf(mut self, hibit: UInt64):
        var t0 = (
            UInt64(self.buf[0])
            | (UInt64(self.buf[1]) << 8)
            | (UInt64(self.buf[2]) << 16)
            | (UInt64(self.buf[3]) << 24)
        )
        var t1 = (
            UInt64(self.buf[4])
            | (UInt64(self.buf[5]) << 8)
            | (UInt64(self.buf[6]) << 16)
            | (UInt64(self.buf[7]) << 24)
        )
        var t2 = (
            UInt64(self.buf[8])
            | (UInt64(self.buf[9]) << 8)
            | (UInt64(self.buf[10]) << 16)
            | (UInt64(self.buf[11]) << 24)
        )
        var t3 = (
            UInt64(self.buf[12])
            | (UInt64(self.buf[13]) << 8)
            | (UInt64(self.buf[14]) << 16)
            | (UInt64(self.buf[15]) << 24)
        )

        var h0 = self.h0 + (t0 & UInt64(0x03FFFFFF))
        var h1 = self.h1 + (((t0 >> 26) | (t1 << 6)) & UInt64(0x03FFFFFF))
        var h2 = self.h2 + (((t1 >> 20) | (t2 << 12)) & UInt64(0x03FFFFFF))
        var h3 = self.h3 + (((t2 >> 14) | (t3 << 18)) & UInt64(0x03FFFFFF))
        var h4 = self.h4 + ((t3 >> 8) | hibit)

        var d0 = (
            h0 * self.r0
            + h1 * self.s4
            + h2 * self.s3
            + h3 * self.s2
            + h4 * self.s1
        )
        var d1 = (
            h0 * self.r1
            + h1 * self.r0
            + h2 * self.s4
            + h3 * self.s3
            + h4 * self.s2
        )
        var d2 = (
            h0 * self.r2
            + h1 * self.r1
            + h2 * self.r0
            + h3 * self.s4
            + h4 * self.s3
        )
        var d3 = (
            h0 * self.r3
            + h1 * self.r2
            + h2 * self.r1
            + h3 * self.r0
            + h4 * self.s4
        )
        var d4 = (
            h0 * self.r4
            + h1 * self.r3
            + h2 * self.r2
            + h3 * self.r1
            + h4 * self.r0
        )

        var c = d0 >> 26
        h0 = d0 & UInt64(0x03FFFFFF)
        d1 += c
        c = d1 >> 26
        h1 = d1 & UInt64(0x03FFFFFF)
        d2 += c
        c = d2 >> 26
        h2 = d2 & UInt64(0x03FFFFFF)
        d3 += c
        c = d3 >> 26
        h3 = d3 & UInt64(0x03FFFFFF)
        d4 += c
        c = d4 >> 26
        h4 = d4 & UInt64(0x03FFFFFF)
        h0 += c * 5
        c = h0 >> 26
        h0 &= UInt64(0x03FFFFFF)
        h1 += c

        self.h0 = h0
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.h4 = h4

    def update_ptr(mut self, data_ptr: Int, length: Int):
        if length <= 0 or data_ptr == 0:
            return
        var src = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=data_ptr)
        for i in range(length):
            self.buf[self.buf_len] = src[unsafe_offset=i]
            self.buf_len += 1
            if self.buf_len == 16:
                self._process_buf(UInt64(1) << 24)
                self.buf_len = 0

    def update_u64_le(mut self, v: UInt64):
        for s in range(8):
            self.buf[self.buf_len] = UInt8((v >> UInt64(s * 8)) & 0xFF)
            self.buf_len += 1
            if self.buf_len == 16:
                self._process_buf(UInt64(1) << 24)
                self.buf_len = 0

    def pad16(mut self):
        while self.buf_len != 0:
            self.buf[self.buf_len] = 0
            self.buf_len += 1
            if self.buf_len == 16:
                self._process_buf(UInt64(1) << 24)
                self.buf_len = 0

    def finish(mut self) -> Array[UInt8, 16]:
        if self.buf_len > 0:
            self.buf[self.buf_len] = 1
            for i in range(self.buf_len + 1, 16):
                self.buf[i] = 0
            self._process_buf(UInt64(0))
            self.buf_len = 0

        # Full carry chain
        var h0 = self.h0
        var h1 = self.h1
        var h2 = self.h2
        var h3 = self.h3
        var h4 = self.h4

        var c = h1 >> 26
        h1 &= UInt64(0x03FFFFFF)
        h2 += c
        c = h2 >> 26
        h2 &= UInt64(0x03FFFFFF)
        h3 += c
        c = h3 >> 26
        h3 &= UInt64(0x03FFFFFF)
        h4 += c
        c = h4 >> 26
        h4 &= UInt64(0x03FFFFFF)
        h0 += c * 5
        c = h0 >> 26
        h0 &= UInt64(0x03FFFFFF)
        h1 += c

        # Compute h + 5 - 2^130; select if >= 2^130
        var g0 = h0 + 5
        c = g0 >> 26
        g0 &= UInt64(0x03FFFFFF)
        var g1 = h1 + c
        c = g1 >> 26
        g1 &= UInt64(0x03FFFFFF)
        var g2 = h2 + c
        c = g2 >> 26
        g2 &= UInt64(0x03FFFFFF)
        var g3 = h3 + c
        c = g3 >> 26
        g3 &= UInt64(0x03FFFFFF)
        var g4 = (h4 + c) - (UInt64(1) << 26)

        var mask = (g4 >> 63) - UInt64(1)  # 0xFFFF.. if g4 >= 0 else 0
        g0 &= mask
        g1 &= mask
        g2 &= mask
        g3 &= mask
        g4 &= mask
        var inv_mask = ~mask
        h0 = (h0 & inv_mask) | g0
        h1 = (h1 & inv_mask) | g1
        h2 = (h2 & inv_mask) | g2
        h3 = (h3 & inv_mask) | g3
        h4 = (h4 & inv_mask) | g4

        # Repack into 4 x 32-bit words and add pad
        var f0 = ((h0) | (h1 << 26)) & UInt64(0xFFFFFFFF)
        var f1 = ((h1 >> 6) | (h2 << 20)) & UInt64(0xFFFFFFFF)
        var f2 = ((h2 >> 12) | (h3 << 14)) & UInt64(0xFFFFFFFF)
        var f3 = ((h3 >> 18) | (h4 << 8)) & UInt64(0xFFFFFFFF)

        f0 += self.pad0
        var carry = f0 >> 32
        f1 += self.pad1 + carry
        carry = f1 >> 32
        f2 += self.pad2 + carry
        carry = f2 >> 32
        f3 += self.pad3 + carry

        var tag = Array[UInt8, 16](fill=UInt8(0))
        var words = Array[UInt32, 4](fill=UInt32(0))
        words[0] = UInt32(f0 & 0xFFFFFFFF)
        words[1] = UInt32(f1 & 0xFFFFFFFF)
        words[2] = UInt32(f2 & 0xFFFFFFFF)
        words[3] = UInt32(f3 & 0xFFFFFFFF)
        for w in range(4):
            var v = words[w]
            tag[w * 4] = UInt8(v & 0xFF)
            tag[w * 4 + 1] = UInt8((v >> 8) & 0xFF)
            tag[w * 4 + 2] = UInt8((v >> 16) & 0xFF)
            tag[w * 4 + 3] = UInt8((v >> 24) & 0xFF)
        return tag^


@always_inline
def poly1305_verify_tag(
    expected: Array[UInt8, 16], actual_ptr: Int
) -> Bool:
    """Constant-time 16-byte Poly1305 tag comparison."""
    if actual_ptr == 0:
        return False
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=actual_ptr)
    var diff = UInt8(0)
    for i in range(16):
        diff |= expected[i] ^ p[unsafe_offset=i]
    return diff == 0


def aead_seal_raw(
    key: AeadKey,
    n0: UInt32,
    n1: UInt32,
    n2: UInt32,
    aad_ptr: Int,
    aad_len: Int,
    pt_ptr: Int,
    pt_len: Int,
    mut out_ct_tag: List[UInt8],
):
    """RFC 8439 ChaCha20-Poly1305 AEAD seal: appends `ciphertext (pt_len B) || tag (16 B)`."""
    var otk = chacha20_block(key, 0, n0, n1, n2)
    var base_off = len(out_ct_tag)
    var old_buf = Int(out_ct_tag.unsafe_ptr())
    var aad_aliases = old_buf != 0 and aad_len > 0 and aad_ptr >= old_buf and aad_ptr < old_buf + base_off
    var aad_rel = aad_ptr - old_buf if aad_aliases else 0
    var pt_aliases = old_buf != 0 and pt_len > 0 and pt_ptr >= old_buf and pt_ptr < old_buf + base_off
    var pt_rel = pt_ptr - old_buf if pt_aliases else 0
    var ct_len = max(0, pt_len)
    out_ct_tag.resize(base_off + ct_len + AEAD_TAG_BYTES, UInt8(0))
    var new_buf = Int(out_ct_tag.unsafe_ptr())
    var ct_ptr = new_buf + base_off
    var eff_aad_ptr = (new_buf + aad_rel) if aad_aliases else aad_ptr
    var eff_pt_ptr = (new_buf + pt_rel) if pt_aliases else pt_ptr
    if ct_len > 0:
        chacha20_xor_ptr(key, 1, n0, n1, n2, eff_pt_ptr, ct_len, ct_ptr)

    var mac = _Poly1305State(otk)
    if aad_len > 0:
        mac.update_ptr(eff_aad_ptr, aad_len)
        mac.pad16()
    if ct_len > 0:
        mac.update_ptr(ct_ptr, ct_len)
        mac.pad16()
    mac.update_u64_le(UInt64(max(0, aad_len)))
    mac.update_u64_le(UInt64(ct_len))
    var tag = mac.finish()
    var dst_tag = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=ct_ptr + ct_len
    )
    for i in range(16):
        dst_tag[unsafe_offset=i] = tag[i]
    _ = len(out_ct_tag)


def aead_open_raw(
    key: AeadKey,
    n0: UInt32,
    n1: UInt32,
    n2: UInt32,
    aad_ptr: Int,
    aad_len: Int,
    ct_ptr: Int,
    ct_len: Int,
    tag_ptr: Int,
    mut out_pt: List[UInt8],
) -> Bool:
    """RFC 8439 ChaCha20-Poly1305 AEAD open: verifies tag in constant time before decrypting."""
    if ct_len < 0 or tag_ptr == 0 or (ct_len > 0 and ct_ptr == 0):
        return False
    var otk = chacha20_block(key, 0, n0, n1, n2)
    var mac = _Poly1305State(otk)
    if aad_len > 0:
        mac.update_ptr(aad_ptr, aad_len)
        mac.pad16()
    if ct_len > 0:
        mac.update_ptr(ct_ptr, ct_len)
        mac.pad16()
    mac.update_u64_le(UInt64(max(0, aad_len)))
    mac.update_u64_le(UInt64(ct_len))
    var expected = mac.finish()
    if not poly1305_verify_tag(expected, tag_ptr):
        return False

    if ct_len == 0:
        out_pt.clear()
        return True
    var out_buf = Int(out_pt.unsafe_ptr())
    var out_cap = out_pt.capacity()
    var ct_aliases = (
        out_buf != 0
        and ct_ptr < out_buf + out_cap
        and ct_ptr + ct_len > out_buf
    )
    if ct_aliases:
        var ct_copy = List[UInt8](capacity=ct_len)
        var src = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=ct_ptr)
        for i in range(ct_len):
            ct_copy.append(src[unsafe_offset=i])
        out_pt.clear()
        out_pt.resize(ct_len, UInt8(0))
        chacha20_xor_ptr(
            key, 1, n0, n1, n2, Int(ct_copy.unsafe_ptr()), ct_len, Int(out_pt.unsafe_ptr())
        )
        _ = len(ct_copy)
        _ = len(out_pt)
    else:
        out_pt.clear()
        out_pt.resize(ct_len, UInt8(0))
        chacha20_xor_ptr(
            key, 1, n0, n1, n2, ct_ptr, ct_len, Int(out_pt.unsafe_ptr())
        )
        _ = len(out_pt)
    return True


def aead_seal(
    key: AeadKey,
    nonce64: UInt64,
    aad: List[UInt8],
    plaintext: List[UInt8],
    mut out_ct_tag: List[UInt8],
    salt: UInt32 = 0,
):
    """High-level 64-bit nonce ChaCha20-Poly1305 AEAD seal."""
    var n1 = UInt32(nonce64 & 0xFFFFFFFF)
    var n2 = UInt32(nonce64 >> 32)
    var aad_ptr = Int(aad.unsafe_ptr()) if len(aad) > 0 else 0
    var pt_ptr = Int(plaintext.unsafe_ptr()) if len(plaintext) > 0 else 0
    aead_seal_raw(
        key, salt, n1, n2, aad_ptr, len(aad), pt_ptr, len(plaintext), out_ct_tag
    )
    _ = len(aad)
    _ = len(plaintext)


def aead_open(
    key: AeadKey,
    nonce64: UInt64,
    aad: List[UInt8],
    ct_and_tag: List[UInt8],
    mut out_pt: List[UInt8],
    salt: UInt32 = 0,
) -> Bool:
    """High-level 64-bit nonce ChaCha20-Poly1305 AEAD open."""
    if len(ct_and_tag) < AEAD_TAG_BYTES:
        return False
    var ct_len = len(ct_and_tag) - AEAD_TAG_BYTES
    var n1 = UInt32(nonce64 & 0xFFFFFFFF)
    var n2 = UInt32(nonce64 >> 32)
    var aad_ptr = Int(aad.unsafe_ptr()) if len(aad) > 0 else 0
    var base_ptr = Int(ct_and_tag.unsafe_ptr())
    var ok = aead_open_raw(
        key,
        salt,
        n1,
        n2,
        aad_ptr,
        len(aad),
        base_ptr,
        ct_len,
        base_ptr + ct_len,
        out_pt,
    )
    _ = len(aad)
    _ = len(ct_and_tag)
    return ok


struct ReplayWindow64(Copyable, ImplicitlyCopyable, Movable):
    """65-packet sliding anti-replay window for 64-bit nonces (`nonce == 0` rejected)."""
    var initialized: Bool
    var max_nonce: UInt64
    var bitmap: UInt64

    def __init__(out self):
        self.initialized = False
        self.max_nonce = 0
        self.bitmap = 0

    def reset(mut self):
        self.initialized = False
        self.max_nonce = 0
        self.bitmap = 0

    @always_inline
    def would_accept(self, nonce: UInt64) -> Bool:
        """Pure pre-check before AEAD decryption so forged packets cannot mutate the window."""
        if nonce == 0:
            return False
        if not self.initialized:
            return True
        if nonce > self.max_nonce:
            return True
        if nonce == self.max_nonce:
            return False
        var age = self.max_nonce - nonce
        if age > 64:
            return False
        return (self.bitmap & (UInt64(1) << (age - 1))) == 0

    @always_inline
    def record(mut self, nonce: UInt64):
        """Commits an authenticated nonce into the replay window."""
        if not self.initialized:
            self.initialized = True
            self.max_nonce = nonce
            self.bitmap = 0
            return
        if nonce > self.max_nonce:
            var diff = nonce - self.max_nonce
            if diff >= 65:
                self.bitmap = 0
            elif diff == 64:
                self.bitmap = UInt64(1) << 63
            else:
                self.bitmap = (self.bitmap << diff) | (UInt64(1) << (diff - 1))
            self.max_nonce = nonce
        else:
            var age = self.max_nonce - nonce
            if age >= 1 and age <= 64:
                self.bitmap |= UInt64(1) << (age - 1)


struct StatelessCookieAuth(Copyable, ImplicitlyCopyable, Movable):
    """Stateless HMAC/Poly1305 handshake cookie generator & verifier (zero per-client server state)."""
    var secret: AeadKey
    var bucket_ms: Float64

    def __init__(
        out self,
        secret: AeadKey = AeadKey(),
        bucket_ms: Float64 = 10000.0,
    ):
        self.secret = secret
        self.bucket_ms = max(100.0, bucket_ms)

    def _cookie_for_bucket(
        self, ep: NetEndpoint, bucket: UInt64
    ) -> Array[UInt8, 16]:
        var otk = chacha20_block(
            self.secret,
            0,
            UInt32(0x434F4F4B),  # "COOK"
            UInt32(bucket & 0xFFFFFFFF),
            UInt32(bucket >> 32),
        )
        var mac = _Poly1305State(otk)
        mac.update_u64_le(ep.addr_hi)
        mac.update_u64_le(ep.addr_lo)
        mac.update_u64_le(UInt64(ep.port) | (UInt64(ep.scope_id) << 16))
        mac.update_u64_le(bucket)
        return mac.finish()

    def issue_cookie(
        self, ep: NetEndpoint, now_ms: Float64
    ) -> Array[UInt8, 16]:
        var bucket = UInt64(max(0, Int(now_ms / self.bucket_ms)))
        return self._cookie_for_bucket(ep, bucket)

    def verify_cookie(
        self, ep: NetEndpoint, cookie_ptr: Int, now_ms: Float64
    ) -> Bool:
        if cookie_ptr == 0:
            return False
        var bucket = UInt64(max(0, Int(now_ms / self.bucket_ms)))
        var cur = self._cookie_for_bucket(ep, bucket)
        if poly1305_verify_tag(cur, cookie_ptr):
            return True
        if bucket > 0:
            var prev = self._cookie_for_bucket(ep, bucket - 1)
            if poly1305_verify_tag(prev, cookie_ptr):
                return True
        return False
