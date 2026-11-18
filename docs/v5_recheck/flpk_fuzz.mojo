from std.sys import argv
from flint.assets.flpk_v3 import (
    FlpkBuildChunk, FlpkV3MappedArchive, AssetUuid, encode_flpk_v3, crc64_ecma_ptr, ASSET_KIND_RAW,
)
from flint.core.lz4 import address_of
from flint.platform.os import write_bytes

def put64(mut b: List[UInt8], at: Int, v: UInt64):
    for i in range(8):
        b[at + i] = UInt8((v >> UInt64(8 * i)) & 0xFF)

def main() raises:
    var iters = Int(String(argv()[1])) if len(argv()) > 1 else 20000
    var path = String(argv()[2]) if len(argv()) > 2 else "/tmp/fz.flpk"
    var names = List[String]()
    var chunks = List[FlpkBuildChunk]()
    for k in range(6):
        var p = List[UInt8]()
        for i in range(200 + k * 97):
            p.append(UInt8((i // (k + 1)) & 0xFF) if k % 2 == 0 else UInt8((i * 31 + k) & 0xFF))
        var n = "dir/asset_" + String(k) + ".bin"
        names.append(n)
        chunks.append(FlpkBuildChunk(AssetUuid.from_path(n), n, 0, ASSET_KIND_RAW, p^))
    var base = encode_flpk_v3(chunks, page_align=False)
    var size = len(base)
    var hdr_meta = 64 + 512
    var s: UInt64 = 0x9E3779B97F4A7C15
    var valid_count = 0
    for it in range(iters):
        var pack = base.copy()
        s ^= s << 13; s ^= s >> 7; s ^= s << 17
        var nmut = Int(s % 4) + 1
        for _ in range(nmut):
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            # 80% of mutations hit header+tables, the rest anywhere
            var region = hdr_meta if (s >> 32) % 5 != 0 else size
            var at = Int((s >> 8) % UInt64(min(region, size)))
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            var kind = Int(s % 4)
            if kind == 0:
                pack[at] = UInt8(s >> 24)
            elif kind == 1:
                pack[at] = pack[at] ^ UInt8(1 << Int((s >> 40) % 8))
            elif kind == 2:
                pack[at] = 0xFF
            else:
                pack[at] = 0
        # Re-seal CRC like the writer does, when the (possibly mutated) layout is computable.
        var hcap = Int(pack[40]) | (Int(pack[41]) << 8)
        var str_off = (Int(pack[32]) | (Int(pack[33]) << 8)) + hcap * 16
        var spool = Int(pack[44]) | (Int(pack[45]) << 8)
        var meta_end = ((str_off + spool + 63) // 64) * 64
        if meta_end > 64 and meta_end <= size and (it % 8) != 0:
            var c0 = crc64_ecma_ptr(address_of(pack), 56, 0)
            var c = crc64_ecma_ptr(address_of(pack) + 64, meta_end - 64, c0)
            put64(pack, 56, c)
        _ = write_bytes(path, pack)
        var arch = FlpkV3MappedArchive.open(path)
        if arch.valid:
            valid_count += 1
            for k in range(len(names)):
                _ = arch.find_by_path(names[k])
                var r = arch.read_by_uuid(AssetUuid.from_path(names[k]))
                _ = r[0]
        arch.close()
    print("flpk fuzz done iters=", iters, "opened_valid=", valid_count)
