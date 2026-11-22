"""LZ4, our own: the block format, a fast greedy compressor and a checked decoder.

The block format is the standard one (lz4.org, "LZ4 Block Format
Description"), so any LZ4 block decoder reads what we write: sequences of a
token (literal length in the high nibble, match length minus 4 in the low),
the extra length bytes of 255s, the literals, a two-byte little-endian offset
back into what is already decoded, and the extra match length bytes. The last
sequence is literals only; the last five bytes of a block are always
literals and no match starts in the last twelve, as the format requires.

`lz4_pack` wraps a block in our own twenty-byte frame: magic, the raw size,
the block size and a 64-bit hash of the raw bytes, so a damaged file is
refused rather than decoded into something plausible. Brick snapshots
(world/snapshot.mojo) and cooked assets (assets/cooked_model.mojo) both use
it. The decoder never reads or writes outside its buffers, whatever it is
given (tests/test_lz4.mojo feeds it damaged blocks).
"""
from std.ffi import external_call

comptime LZ4_MIN_MATCH: Int = 4
comptime LZ4_LAST_LITERALS: Int = 5  # a block always ends with this many literals
comptime LZ4_MATCH_LIMIT: Int = 12  # no match may start this close to the end
comptime LZ4_MAX_OFFSET: Int = 65535
comptime LZ4_HASH_BITS: Int = 16
comptime LZ4_SKIP_TRIGGER: Int = 6  # how quickly the search speeds up over data that does not match

comptime PACK_MAGIC: UInt32 = 0x5A344453  # "SD4Z"
comptime PACK_HEADER_BYTES: Int = 24
# A packed buffer may claim at most this many raw bytes; anything larger is damage.
comptime PACK_MAX_RAW: Int = 1 << 31


# Buffers are passed as addresses (an Int, as elsewhere in the engine) and
# read through these, which never assume alignment.
@always_inline
def bytes_at(address: Int) -> Pointer[UInt8, origin=MutAnyOrigin]:
    return Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=address)


@always_inline
def load_u32(p: Int, at: Int) -> UInt32:
    return Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=p + at).unsafe_load[alignment=1]()


@always_inline
def load_u64(p: Int, at: Int) -> UInt64:
    return Pointer[UInt64, origin=MutAnyOrigin](unsafe_from_address=p + at).unsafe_load[alignment=1]()


@always_inline
def store_u64(p: Int, at: Int, value: UInt64):
    Pointer[UInt64, origin=MutAnyOrigin](unsafe_from_address=p + at).unsafe_store[alignment=1](value)


@always_inline
def address_of(data: List[UInt8]) -> Int:
    return Int(data.unsafe_ptr())


@always_inline
def hash4(word: UInt32) -> Int:
    return Int((word * 2654435761) >> UInt32(32 - LZ4_HASH_BITS))


def lz4_bound(size: Int) -> Int:
    """The most bytes a block of `size` raw bytes can take."""
    return size + size // 255 + 16


@always_inline
def put_length(p: Int, var at: Int, var extra: Int) -> Int:
    """The 255-run that follows a nibble of 15."""
    while extra >= 255:
        bytes_at(p)[unsafe_offset=at] = 255
        at += 1
        extra -= 255
    bytes_at(p)[unsafe_offset=at] = UInt8(extra)
    return at + 1


def lz4_compress_into(source: Int, size: Int, target: Int) -> Int:
    """Compresses `size` bytes into `out` (at least lz4_bound(size) bytes) and
    gives the block's length. Greedy, one hash probe per position, like the
    reference "fast" mode: the same input always gives the same block."""
    var table = List[Int32](length=1 << LZ4_HASH_BITS, fill=-1)
    var hashes = table.unsafe_ptr()
    var src = bytes_at(source)
    var dst = bytes_at(target)
    var op = 0
    var anchor = 0
    var ip = 0
    var match_limit = size - LZ4_MATCH_LIMIT
    var end_of_matches = size - LZ4_LAST_LITERALS
    if size >= LZ4_MATCH_LIMIT + 1:
        var misses = 1 << LZ4_SKIP_TRIGGER
        while ip <= match_limit:
            var word = load_u32(source, ip)
            var h = hash4(word)
            var candidate = Int(hashes[unsafe_offset=h])
            hashes[unsafe_offset=h] = Int32(ip)
            if candidate < 0 or ip - candidate > LZ4_MAX_OFFSET or load_u32(source, candidate) != word:
                ip += misses >> LZ4_SKIP_TRIGGER
                misses += 1
                continue
            misses = 1 << LZ4_SKIP_TRIGGER
            # Reach back over literals that also match.
            while ip > anchor and candidate > 0 and src[unsafe_offset=ip - 1] == src[unsafe_offset=candidate - 1]:
                ip -= 1
                candidate -= 1
            # And forward, eight bytes at a time while they agree.
            var length = LZ4_MIN_MATCH
            var stopped = False
            while ip + length + 8 <= end_of_matches:
                var difference = load_u64(source, ip + length) ^ load_u64(source, candidate + length)
                if difference != 0:
                    length += count_trailing_zero_bytes(difference)
                    stopped = True
                    break
                length += 8
            if not stopped:
                while ip + length < end_of_matches and src[unsafe_offset=ip + length] == src[unsafe_offset=candidate + length]:
                    length += 1
            if ip + length > end_of_matches:
                length = end_of_matches - ip
            var literals = ip - anchor
            var token_at = op
            op += 1
            if literals >= 15:
                dst[unsafe_offset=token_at] = 0xF0
                op = put_length(target, op, literals - 15)
            else:
                dst[unsafe_offset=token_at] = UInt8(literals << 4)
            if literals > 0:
                _ = external_call["memcpy", Int](target + op, source + anchor, literals)
                op += literals
            var offset = ip - candidate
            dst[unsafe_offset=op] = UInt8(offset & 255)
            dst[unsafe_offset=op + 1] = UInt8(offset >> 8)
            op += 2
            var extra = length - LZ4_MIN_MATCH
            if extra >= 15:
                dst[unsafe_offset=token_at] = dst[unsafe_offset=token_at] | 15
                op = put_length(target, op, extra - 15)
            else:
                dst[unsafe_offset=token_at] = dst[unsafe_offset=token_at] | UInt8(extra)
            ip += length
            anchor = ip
            # Seed the table inside the match, so the next match can be found
            # right after it.
            if ip - 2 >= 0 and ip - 2 <= match_limit:
                hashes[unsafe_offset=hash4(load_u32(source, ip - 2))] = Int32(ip - 2)
    var literals = size - anchor
    var token_at = op
    op += 1
    if literals >= 15:
        dst[unsafe_offset=token_at] = 0xF0
        op = put_length(target, op, literals - 15)
    else:
        dst[unsafe_offset=token_at] = UInt8(literals << 4)
    if literals > 0:
        _ = external_call["memcpy", Int](target + op, source + anchor, literals)
        op += literals
    _ = table^
    return op


@always_inline
def count_trailing_zero_bytes(value: UInt64) -> Int:
    var count = 0
    var v = value
    while (v & 0xFF) == 0:
        v >>= 8
        count += 1
    return count


def lz4_decompress_into(block: Int, size: Int, target: Int, capacity: Int) -> Int:
    """Decodes a block into `out` and gives the decoded length, or -1 for a
    block that is damaged: a length running past either buffer, an offset of
    zero or before the start, or a block that does not end on literals."""
    var b = bytes_at(block)
    var o = bytes_at(target)
    var ip = 0
    var op = 0
    while ip < size:
        var token = Int(b[unsafe_offset=ip])
        ip += 1
        var literals = token >> 4
        if literals == 15:
            while True:
                if ip >= size:
                    return -1
                var more = Int(b[unsafe_offset=ip])
                ip += 1
                literals += more
                if more != 255:
                    break
        if literals > size - ip or literals > capacity - op:
            return -1
        if literals > 0:
            _ = external_call["memcpy", Int](target + op, block + ip, literals)
            ip += literals
            op += literals
        if ip == size:
            return op  # the last sequence: literals only
        if ip + 2 > size:
            return -1
        var offset = Int(b[unsafe_offset=ip]) | (Int(b[unsafe_offset=ip + 1]) << 8)
        ip += 2
        if offset == 0 or offset > op:
            return -1
        var length = (token & 15) + LZ4_MIN_MATCH
        if (token & 15) == 15:
            while True:
                if ip >= size:
                    return -1
                var more = Int(b[unsafe_offset=ip])
                ip += 1
                length += more
                if more != 255:
                    break
        if length > capacity - op:
            return -1
        var from_at = op - offset
        if offset >= 8 and op + length + 8 <= capacity:
            # Eight at a time; a copy may run up to seven bytes past the match,
            # which the next sequence overwrites.
            var copied = 0
            while copied < length:
                store_u64(target, op + copied, load_u64(target, from_at + copied))
                copied += 8
        else:
            # Overlapping (a run) or at the very end: byte by byte.
            for index in range(length):
                o[unsafe_offset=op + index] = o[unsafe_offset=from_at + index]
        op += length
    return -1  # a block may not end on a match


def lz4_compress(data: List[UInt8]) -> List[UInt8]:
    """One block holding all of `data`."""
    var out = List[UInt8](capacity=lz4_bound(len(data)))
    out.resize(unsafe_uninit_length=lz4_bound(len(data)))
    var used = lz4_compress_into(address_of(data), len(data), address_of(out))
    out.resize(unsafe_uninit_length=used)
    return out^


def lz4_decompress(block: List[UInt8], raw_size: Int) -> List[UInt8]:
    """The decoded block, or an empty list if it is damaged or does not decode
    to exactly `raw_size` bytes."""
    if raw_size <= 0:
        return List[UInt8]()
    var out = List[UInt8](capacity=raw_size + 1)
    out.resize(unsafe_uninit_length=raw_size)
    var used = lz4_decompress_into(address_of(block), len(block), address_of(out), raw_size)
    if used != raw_size:
        return List[UInt8]()
    return out^


def block_hash(data: Int, size: Int) -> UInt64:
    """A 64-bit hash of bytes, eight at a time: for telling damage from data,
    not for security. Byte-at-a-time FNV is too slow for tens of megabytes."""
    var h = UInt64(0x9E3779B97F4A7C15) ^ UInt64(size)
    var at = 0
    while at + 8 <= size:
        h = mix_word(h ^ load_u64(data, at))
        at += 8
    var tail = UInt64(0)
    var shift = UInt64(0)
    while at < size:
        tail |= UInt64(bytes_at(data)[unsafe_offset=at]) << shift
        shift += 8
        at += 1
    return mix_word(h ^ tail ^ (UInt64(0xA5) << 56))


@always_inline
def mix_word(value: UInt64) -> UInt64:
    var h = value * 0xBF58476D1CE4E5B9
    h ^= h >> 31
    h = h * 0x94D049BB133111EB
    return h ^ (h >> 29)


def hash_list(data: List[UInt8]) -> UInt64:
    return block_hash(address_of(data), len(data))


@always_inline
def put32(mut out: List[UInt8], at: Int, value: UInt32):
    for shift in range(4):
        out[at + shift] = UInt8((value >> UInt32(shift * 8)) & 0xFF)


@always_inline
def get32(data: List[UInt8], at: Int) -> UInt32:
    return UInt32(data[at]) | (UInt32(data[at + 1]) << 8) | (UInt32(data[at + 2]) << 16) | (UInt32(data[at + 3]) << 24)


def lz4_pack(data: List[UInt8]) -> List[UInt8]:
    """Our frame: magic, raw size, block size, hash of the raw bytes, then the block."""
    var bound = lz4_bound(len(data))
    var out = List[UInt8](capacity=PACK_HEADER_BYTES + bound)
    out.resize(unsafe_uninit_length=PACK_HEADER_BYTES + bound)
    var used = lz4_compress_into(address_of(data), len(data), address_of(out) + PACK_HEADER_BYTES)
    out.resize(unsafe_uninit_length=PACK_HEADER_BYTES + used)
    var hash = hash_list(data)
    put32(out, 0, PACK_MAGIC)
    put32(out, 4, UInt32(len(data)))
    put32(out, 8, UInt32(used))
    put32(out, 12, 0)
    put32(out, 16, UInt32(hash & 0xFFFFFFFF))
    put32(out, 20, UInt32(hash >> 32))
    return out^


def lz4_packed_raw_size(packed: List[UInt8], offset: Int = 0) -> Int:
    """What a packed buffer decodes to, or -1 if it is not one."""
    if offset < 0 or len(packed) - offset < PACK_HEADER_BYTES or get32(packed, offset) != PACK_MAGIC:
        return -1
    return Int(get32(packed, offset + 4))


def lz4_unpack(packed: List[UInt8], offset: Int = 0) -> List[UInt8]:
    """The raw bytes of a packed buffer starting at `offset`, or an empty list
    for anything damaged: wrong magic, sizes that do not fit, a block that
    does not decode, or a hash that does not match."""
    if offset < 0 or len(packed) - offset < PACK_HEADER_BYTES or get32(packed, offset) != PACK_MAGIC or get32(packed, offset + 12) != 0:
        return List[UInt8]()
    var raw = Int(get32(packed, offset + 4))
    var block = Int(get32(packed, offset + 8))
    var hash = UInt64(get32(packed, offset + 16)) | (UInt64(get32(packed, offset + 20)) << 32)
    if raw >= PACK_MAX_RAW or block > len(packed) - offset - PACK_HEADER_BYTES:
        return List[UInt8]()
    # A block can only grow data by so much: refuse a size it could never reach
    # before allocating it.
    if raw > block * 255 + 16:
        return List[UInt8]()
    var out = List[UInt8](capacity=raw + 1)
    out.resize(unsafe_uninit_length=raw)
    var used = lz4_decompress_into(address_of(packed) + offset + PACK_HEADER_BYTES, block, address_of(out), raw)
    if used != raw or hash_list(out) != hash:
        return List[UInt8]()
    return out^


def lz4_packed_size(packed: List[UInt8], offset: Int = 0) -> Int:
    """Bytes the packed buffer at `offset` takes, header included, or -1."""
    if offset < 0 or len(packed) - offset < PACK_HEADER_BYTES or get32(packed, offset) != PACK_MAGIC:
        return -1
    return PACK_HEADER_BYTES + Int(get32(packed, offset + 8))
