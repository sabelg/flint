"""Pure-Mojo session key exchange primitives (`flint.net.kex`).

Implements:
- `sha256` / `Sha256`: FIPS 180-4 SHA-256.
- `hmac_sha256`: RFC 2104 HMAC over SHA-256.
- `hkdf_extract` / `hkdf_expand` / `hkdf_sha256`: RFC 5869 HKDF-SHA256.
- `x25519` / `x25519_public_key`: RFC 7748 Curve25519 Diffie-Hellman (16 x 16-bit limb field
  arithmetic, constant-time Montgomery ladder with conditional swaps).
- `derive_session_keys`: HKDF-SHA256 split of an X25519 shared secret into per-direction
  ChaCha20-Poly1305 `AeadKey`s (client->server, server->client).
- `issue_connect_token` / `verify_connect_token`: HMAC-SHA256 signed connect tokens a backend
  issues so a server admits only authorised clients during the cookie handshake.
"""
from std.collections import List
from std.ffi import external_call
from std.time import monotonic
from flint.core.detsort import mix64
from flint.net.crypto import AeadKey

comptime X25519_KEY_BYTES: Int = 32
comptime SHA256_BYTES: Int = 32
comptime CONNECT_TOKEN_BYTES: Int = 72  # 8 magic+protocol, 8 client_id, 8 expire, 8 user, 8 nonce, 32 mac
comptime _CONNECT_TOKEN_BODY: Int = 40
comptime _TOKEN_MAGIC: UInt64 = 0x31544B4354534C46  # "FLSTCKT1"


# ---------------------------------------------------------------------------
# SHA-256
# ---------------------------------------------------------------------------

def _sha256_k() -> Array[UInt32, 64]:
    var k = Array[UInt32, 64](fill=UInt32(0))
    k[0] = 0x428A2F98
    k[1] = 0x71374491
    k[2] = 0xB5C0FBCF
    k[3] = 0xE9B5DBA5
    k[4] = 0x3956C25B
    k[5] = 0x59F111F1
    k[6] = 0x923F82A4
    k[7] = 0xAB1C5ED5
    k[8] = 0xD807AA98
    k[9] = 0x12835B01
    k[10] = 0x243185BE
    k[11] = 0x550C7DC3
    k[12] = 0x72BE5D74
    k[13] = 0x80DEB1FE
    k[14] = 0x9BDC06A7
    k[15] = 0xC19BF174
    k[16] = 0xE49B69C1
    k[17] = 0xEFBE4786
    k[18] = 0x0FC19DC6
    k[19] = 0x240CA1CC
    k[20] = 0x2DE92C6F
    k[21] = 0x4A7484AA
    k[22] = 0x5CB0A9DC
    k[23] = 0x76F988DA
    k[24] = 0x983E5152
    k[25] = 0xA831C66D
    k[26] = 0xB00327C8
    k[27] = 0xBF597FC7
    k[28] = 0xC6E00BF3
    k[29] = 0xD5A79147
    k[30] = 0x06CA6351
    k[31] = 0x14292967
    k[32] = 0x27B70A85
    k[33] = 0x2E1B2138
    k[34] = 0x4D2C6DFC
    k[35] = 0x53380D13
    k[36] = 0x650A7354
    k[37] = 0x766A0ABB
    k[38] = 0x81C2C92E
    k[39] = 0x92722C85
    k[40] = 0xA2BFE8A1
    k[41] = 0xA81A664B
    k[42] = 0xC24B8B70
    k[43] = 0xC76C51A3
    k[44] = 0xD192E819
    k[45] = 0xD6990624
    k[46] = 0xF40E3585
    k[47] = 0x106AA070
    k[48] = 0x19A4C116
    k[49] = 0x1E376C08
    k[50] = 0x2748774C
    k[51] = 0x34B0BCB5
    k[52] = 0x391C0CB3
    k[53] = 0x4ED8AA4A
    k[54] = 0x5B9CCA4F
    k[55] = 0x682E6FF3
    k[56] = 0x748F82EE
    k[57] = 0x78A5636F
    k[58] = 0x84C87814
    k[59] = 0x8CC70208
    k[60] = 0x90BEFFFA
    k[61] = 0xA4506CEB
    k[62] = 0xBEF9A3F7
    k[63] = 0xC67178F2
    return k^


@always_inline
def _rotr32(x: UInt32, n: UInt32) -> UInt32:
    return (x >> n) | (x << (UInt32(32) - n))


struct Sha256(Copyable, Movable):
    """Streaming FIPS 180-4 SHA-256."""
    var h: Array[UInt32, 8]
    var k: Array[UInt32, 64]
    var buf: Array[UInt8, 64]
    var buf_len: Int
    var total_len: UInt64

    def __init__(out self):
        self.h = Array[UInt32, 8](fill=UInt32(0))
        self.h[0] = 0x6A09E667
        self.h[1] = 0xBB67AE85
        self.h[2] = 0x3C6EF372
        self.h[3] = 0xA54FF53A
        self.h[4] = 0x510E527F
        self.h[5] = 0x9B05688C
        self.h[6] = 0x1F83D9AB
        self.h[7] = 0x5BE0CD19
        self.k = _sha256_k()
        self.buf = Array[UInt8, 64](fill=UInt8(0))
        self.buf_len = 0
        self.total_len = 0

    def _compress(mut self):
        var w = Array[UInt32, 64](fill=UInt32(0))
        for i in range(16):
            w[i] = (
                (UInt32(self.buf[i * 4]) << 24)
                | (UInt32(self.buf[i * 4 + 1]) << 16)
                | (UInt32(self.buf[i * 4 + 2]) << 8)
                | UInt32(self.buf[i * 4 + 3])
            )
        for i in range(16, 64):
            var w15 = w[i - 15]
            var w2 = w[i - 2]
            var s0 = _rotr32(w15, 7) ^ _rotr32(w15, 18) ^ (w15 >> 3)
            var s1 = _rotr32(w2, 17) ^ _rotr32(w2, 19) ^ (w2 >> 10)
            w[i] = w[i - 16] + s0 + w[i - 7] + s1
        var a = self.h[0]
        var b = self.h[1]
        var c = self.h[2]
        var d = self.h[3]
        var e = self.h[4]
        var f = self.h[5]
        var g = self.h[6]
        var hh = self.h[7]
        for i in range(64):
            var s1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
            var ch = (e & f) ^ ((~e) & g)
            var t1 = hh + s1 + ch + self.k[i] + w[i]
            var s0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
            var maj = (a & b) ^ (a & c) ^ (b & c)
            var t2 = s0 + maj
            hh = g
            g = f
            f = e
            e = d + t1
            d = c
            c = b
            b = a
            a = t1 + t2
        self.h[0] += a
        self.h[1] += b
        self.h[2] += c
        self.h[3] += d
        self.h[4] += e
        self.h[5] += f
        self.h[6] += g
        self.h[7] += hh

    @always_inline
    def update_byte(mut self, v: UInt8):
        self.buf[self.buf_len] = v
        self.buf_len += 1
        self.total_len += 1
        if self.buf_len == 64:
            self._compress()
            self.buf_len = 0

    def update(mut self, data: List[UInt8]):
        for i in range(len(data)):
            self.update_byte(data[i])

    def finish(mut self) -> List[UInt8]:
        var bit_len = self.total_len * 8
        self.buf[self.buf_len] = 0x80
        self.buf_len += 1
        if self.buf_len > 56:
            while self.buf_len < 64:
                self.buf[self.buf_len] = 0
                self.buf_len += 1
            self._compress()
            self.buf_len = 0
        while self.buf_len < 56:
            self.buf[self.buf_len] = 0
            self.buf_len += 1
        for i in range(8):
            self.buf[56 + i] = UInt8((bit_len >> UInt64(56 - 8 * i)) & 0xFF)
        self._compress()
        self.buf_len = 0
        var out = List[UInt8](capacity=32)
        for i in range(8):
            var v = self.h[i]
            out.append(UInt8((v >> 24) & 0xFF))
            out.append(UInt8((v >> 16) & 0xFF))
            out.append(UInt8((v >> 8) & 0xFF))
            out.append(UInt8(v & 0xFF))
        return out^


def sha256(data: List[UInt8]) -> List[UInt8]:
    """One-shot SHA-256 digest (32 bytes)."""
    var s = Sha256()
    s.update(data)
    return s.finish()


def hmac_sha256(key: List[UInt8], msg: List[UInt8]) -> List[UInt8]:
    """RFC 2104 HMAC-SHA256 (32-byte tag)."""
    var k0 = List[UInt8](length=64, fill=UInt8(0))
    if len(key) > 64:
        var hk = sha256(key)
        for i in range(32):
            k0[i] = hk[i]
    else:
        for i in range(len(key)):
            k0[i] = key[i]
    var inner = Sha256()
    for i in range(64):
        inner.update_byte(k0[i] ^ 0x36)
    inner.update(msg)
    var ih = inner.finish()
    var outer = Sha256()
    for i in range(64):
        outer.update_byte(k0[i] ^ 0x5C)
    outer.update(ih)
    return outer.finish()


def hkdf_extract(salt: List[UInt8], ikm: List[UInt8]) -> List[UInt8]:
    """RFC 5869 HKDF-Extract (empty salt == 32 zero bytes)."""
    if len(salt) == 0:
        return hmac_sha256(List[UInt8](length=32, fill=UInt8(0)), ikm)
    return hmac_sha256(salt, ikm)


def hkdf_expand(prk: List[UInt8], info: List[UInt8], length: Int) -> List[UInt8]:
    """RFC 5869 HKDF-Expand (`length <= 255 * 32`)."""
    var out = List[UInt8](capacity=max(0, length))
    if length <= 0 or length > 255 * 32:
        return out^
    var t = List[UInt8]()
    var counter = 1
    while len(out) < length:
        var msg = List[UInt8](capacity=len(t) + len(info) + 1)
        for i in range(len(t)):
            msg.append(t[i])
        for i in range(len(info)):
            msg.append(info[i])
        msg.append(UInt8(counter))
        t = hmac_sha256(prk, msg)
        for i in range(len(t)):
            if len(out) < length:
                out.append(t[i])
        counter += 1
    return out^


def hkdf_sha256(
    salt: List[UInt8], ikm: List[UInt8], info: List[UInt8], length: Int
) -> List[UInt8]:
    """RFC 5869 HKDF-SHA256 (extract + expand)."""
    return hkdf_expand(hkdf_extract(salt, ikm), info, length)


def constant_time_eq(a: List[UInt8], b: List[UInt8]) -> Bool:
    """Length-checked constant-time byte comparison."""
    if len(a) != len(b):
        return False
    var diff = UInt8(0)
    for i in range(len(a)):
        diff |= a[i] ^ b[i]
    return diff == 0


# ---------------------------------------------------------------------------
# X25519 (RFC 7748) -- 16 x 16-bit signed limbs in Int64 (TweetNaCl layout)
# ---------------------------------------------------------------------------

comptime _Gf = Array[Int64, 16]


@always_inline
def _gf0() -> _Gf:
    return _Gf(fill=Int64(0))


def _car25519(mut o: _Gf):
    for i in range(16):
        o[i] += Int64(1) << 16
        var c = o[i] >> 16
        if i < 15:
            o[i + 1] += c - 1
        else:
            o[0] += 38 * (c - 1)
        o[i] -= c << 16


def _sel25519(mut p: _Gf, mut q: _Gf, b: Int64):
    var c = ~(b - 1)
    for i in range(16):
        var t = c & (p[i] ^ q[i])
        p[i] ^= t
        q[i] ^= t


def _pack25519(n: _Gf) -> List[UInt8]:
    var t = n.copy()
    _car25519(t)
    _car25519(t)
    _car25519(t)
    for _ in range(2):
        var m = _gf0()
        m[0] = t[0] - 0xFFED
        for i in range(1, 15):
            m[i] = t[i] - 0xFFFF - ((m[i - 1] >> 16) & 1)
            m[i - 1] &= 0xFFFF
        m[15] = t[15] - 0x7FFF - ((m[14] >> 16) & 1)
        var b = (m[15] >> 16) & 1
        m[14] &= 0xFFFF
        _sel25519(t, m, 1 - b)
    var out = List[UInt8](capacity=32)
    for i in range(16):
        out.append(UInt8(t[i] & 0xFF))
        out.append(UInt8((t[i] >> 8) & 0xFF))
    return out^


def _unpack25519(n: List[UInt8]) -> _Gf:
    var o = _gf0()
    for i in range(16):
        o[i] = Int64(n[2 * i]) + (Int64(n[2 * i + 1]) << 8)
    o[15] &= 0x7FFF
    return o^


@always_inline
def _fadd(a: _Gf, b: _Gf) -> _Gf:
    var o = _gf0()
    for i in range(16):
        o[i] = a[i] + b[i]
    return o^


@always_inline
def _fsub(a: _Gf, b: _Gf) -> _Gf:
    var o = _gf0()
    for i in range(16):
        o[i] = a[i] - b[i]
    return o^


def _fmul(a: _Gf, b: _Gf) -> _Gf:
    var t = Array[Int64, 31](fill=Int64(0))
    for i in range(16):
        for j in range(16):
            t[i + j] += a[i] * b[j]
    for i in range(15):
        t[i] += 38 * t[i + 16]
    var o = _gf0()
    for i in range(16):
        o[i] = t[i]
    _car25519(o)
    _car25519(o)
    return o^


def _finv(i: _Gf) -> _Gf:
    var c = i.copy()
    var a = 253
    while a >= 0:
        c = _fmul(c, c)
        if a != 2 and a != 4:
            c = _fmul(c, i)
        a -= 1
    return c^


def x25519(scalar: List[UInt8], u_point: List[UInt8]) -> List[UInt8]:
    """RFC 7748 X25519(k, u): 32-byte scalar x 32-byte u-coordinate -> 32-byte u-coordinate.
    Returns 32 zero bytes when either input is not 32 bytes."""
    if len(scalar) != 32 or len(u_point) != 32:
        return List[UInt8](length=32, fill=UInt8(0))
    var z = List[UInt8](capacity=32)
    for i in range(32):
        z.append(scalar[i])
    z[31] = (z[31] & 127) | 64
    z[0] &= 248
    var x = _unpack25519(u_point)
    var a = _gf0()
    var b = x.copy()
    var c = _gf0()
    var d = _gf0()
    a[0] = 1
    d[0] = 1
    var k121665 = _gf0()
    k121665[0] = 0xDB41
    k121665[1] = 1
    var i = 254
    while i >= 0:
        var r = Int64((z[i >> 3] >> UInt8(i & 7)) & 1)
        _sel25519(a, b, r)
        _sel25519(c, d, r)
        var e = _fadd(a, c)
        a = _fsub(a, c)
        c = _fadd(b, d)
        b = _fsub(b, d)
        d = _fmul(e, e)
        var f = _fmul(a, a)
        a = _fmul(c, a)
        c = _fmul(b, e)
        e = _fadd(a, c)
        a = _fsub(a, c)
        b = _fmul(a, a)
        c = _fsub(d, f)
        a = _fmul(c, k121665)
        a = _fadd(a, d)
        c = _fmul(c, a)
        a = _fmul(d, f)
        d = _fmul(b, x)
        b = _fmul(e, e)
        _sel25519(a, b, r)
        _sel25519(c, d, r)
        i -= 1
    var inv_c = _finv(c)
    return _pack25519(_fmul(a, inv_c))


def x25519_public_key(private_key: List[UInt8]) -> List[UInt8]:
    """X25519 public key = X25519(private, 9)."""
    var base = List[UInt8](length=32, fill=UInt8(0))
    base[0] = 9
    return x25519(private_key, base)


def is_all_zero(data: List[UInt8]) -> Bool:
    var acc = UInt8(0)
    for i in range(len(data)):
        acc |= data[i]
    return acc == 0


def secure_random_bytes(count: Int) -> List[UInt8]:
    """`count` bytes from the kernel CSPRNG (`getrandom(2)`); falls back to a monotonic-clock
    mix only if the syscall is unavailable."""
    var out = List[UInt8](length=max(0, count), fill=UInt8(0))
    if count <= 0:
        return out^
    var got = external_call["getrandom", Int](Int(out.unsafe_ptr()), count, Int32(0))
    if got != count:
        var s = UInt64(monotonic())
        for i in range(count):
            s = mix64(s + UInt64(i) * 0x9E3779B97F4A7C15)
            out[i] = UInt8(s & 0xFF)
    _ = len(out)
    return out^


def x25519_generate_private() -> List[UInt8]:
    """Fresh random 32-byte X25519 private key (clamped by `x25519`)."""
    return secure_random_bytes(32)


def _aead_key_from_slice(data: List[UInt8], off: Int) -> AeadKey:
    var tmp = List[UInt8](capacity=32)
    for i in range(32):
        tmp.append(data[off + i])
    return AeadKey.from_bytes(tmp)


def derive_session_keys(
    shared_secret: List[UInt8],
    client_public: List[UInt8],
    server_public: List[UInt8],
) -> Tuple[AeadKey, AeadKey]:
    """HKDF-SHA256(salt = client_pub || server_pub, ikm = shared) -> (client->server, server->client)."""
    var salt = List[UInt8](capacity=64)
    for i in range(len(client_public)):
        salt.append(client_public[i])
    for i in range(len(server_public)):
        salt.append(server_public[i])
    var info_s = String("flint.net.v5 session keys")
    var info = List[UInt8](capacity=info_s.byte_length())
    for i in range(info_s.byte_length()):
        info.append(info_s.as_bytes()[i])
    var okm = hkdf_sha256(salt, shared_secret, info, 64)
    return (_aead_key_from_slice(okm, 0), _aead_key_from_slice(okm, 32))


# ---------------------------------------------------------------------------
# HMAC-signed connect tokens
# ---------------------------------------------------------------------------

@always_inline
def _put_u64_le(mut buf: List[UInt8], v: UInt64):
    for i in range(8):
        buf.append(UInt8((v >> UInt64(8 * i)) & 0xFF))


@always_inline
def _get_u64_le(buf: List[UInt8], off: Int) -> UInt64:
    var v = UInt64(0)
    for i in range(8):
        v |= UInt64(buf[off + i]) << UInt64(8 * i)
    return v


@fieldwise_init
struct ConnectTokenInfo(Copyable, ImplicitlyCopyable, Movable):
    """Decoded fields of a verified connect token."""
    var valid: Bool
    var protocol_id: UInt32
    var client_id: UInt64
    var expire_ms: UInt64
    var user_data: UInt64


def issue_connect_token(
    token_key: List[UInt8],
    protocol_id: UInt32,
    client_id: UInt64,
    expire_ms: UInt64,
    user_data: UInt64 = 0,
    nonce: UInt64 = 0,
) -> List[UInt8]:
    """Backend side: a `CONNECT_TOKEN_BYTES` token = body || HMAC-SHA256(token_key, body)."""
    var body = List[UInt8](capacity=CONNECT_TOKEN_BYTES)
    _put_u64_le(body, (_TOKEN_MAGIC & 0xFFFFFFFF) | (UInt64(protocol_id) << 32))
    _put_u64_le(body, client_id)
    _put_u64_le(body, expire_ms)
    _put_u64_le(body, user_data)
    _put_u64_le(body, nonce)
    var mac = hmac_sha256(token_key, body)
    for i in range(len(mac)):
        body.append(mac[i])
    return body^


def verify_connect_token(
    token_key: List[UInt8],
    token: List[UInt8],
    protocol_id: UInt32,
    now_ms: Float64,
) -> ConnectTokenInfo:
    """Server side: checks length, MAC (constant time), protocol id and expiry (`now_ms < expire_ms`)."""
    var bad = ConnectTokenInfo(False, 0, 0, 0, 0)
    if len(token) != CONNECT_TOKEN_BYTES:
        return bad
    var body = List[UInt8](capacity=_CONNECT_TOKEN_BODY)
    var mac = List[UInt8](capacity=32)
    for i in range(_CONNECT_TOKEN_BODY):
        body.append(token[i])
    for i in range(_CONNECT_TOKEN_BODY, CONNECT_TOKEN_BYTES):
        mac.append(token[i])
    if not constant_time_eq(hmac_sha256(token_key, body), mac):
        return bad
    var w0 = _get_u64_le(body, 0)
    if (w0 & 0xFFFFFFFF) != (_TOKEN_MAGIC & 0xFFFFFFFF):
        return bad
    var proto = UInt32(w0 >> 32)
    if proto != protocol_id:
        return bad
    var expire = _get_u64_le(body, 16)
    if not (now_ms >= 0.0) or now_ms >= Float64(expire):
        return bad
    return ConnectTokenInfo(True, proto, _get_u64_le(body, 8), expire, _get_u64_le(body, 24))
