from std.sys import argv
from flint.core.serial import BlobWriter, BlobReader
from flint.ecs import STORAGE_TABLE, STORAGE_SPARSE, comp_hash64, Entity, Component, World

@fieldwise_init
struct FPos(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("fz.Pos")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var x: Float32
    var y: Float32
    var z: Float32

@fieldwise_init
struct FVel(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("fz.Vel")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var v: Float32

@fieldwise_init
struct FTag(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("fz.Tag")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var t: Int32

def make_world() -> World:
    var w = World()
    _ = w.register_component[FPos]()
    _ = w.register_component[FVel]()
    _ = w.register_component[FTag]()
    return w^

def main() raises:
    var iters = Int(String(argv()[1])) if len(argv()) > 1 else 5000
    var skip = Int(String(argv()[2])) if len(argv()) > 2 else 0
    var w = make_world()
    var ents = List[Entity]()
    for i in range(300):
        var e = w.spawn_empty()
        _ = w.insert[FPos](e, FPos(Float32(i), 1.0, 2.0))
        if i % 3 == 0:
            _ = w.insert[FVel](e, FVel(Float32(i)))
        if i % 5 == 0:
            _ = w.insert[FTag](e, FTag(Int32(i)))
        ents.append(e)
    for i in range(0, 300, 7):
        _ = w.despawn(ents[i])
    var snap = BlobWriter()
    w.snapshot_blob(snap)
    var base = snap.data.copy()
    var size = len(base)
    print("snapshot bytes", size)
    var s: UInt64 = 0xD1B54A32D192ED03
    var ok = 0
    var rejected = 0
    for it in range(iters):
        var b = base.copy()
        s ^= s << 13; s ^= s >> 7; s ^= s << 17
        var nmut = Int(s % 4) + 1
        for _ in range(nmut):
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            var at = skip + Int((s >> 8) % UInt64(size - skip))
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            var k = Int(s % 4)
            if k == 0: b[at] = UInt8(s >> 24)
            elif k == 1: b[at] = b[at] ^ UInt8(1 << Int((s >> 40) % 8))
            elif k == 2: b[at] = 0xFF
            else: b[at] = 0
        var t = make_world()
        var rd = BlobReader(b^)
        try:
            t.restore_blob(rd)
            ok += 1
        except:
            rejected += 1
        # Use the world after restore (successful or not): read, mutate, checksum.
        _ = t.entity_count()
        _ = t.state_checksum()
        for i in range(0, 300, 13):
            if t.is_alive(ents[i]):
                var p = t.get_ptr[FPos](ents[i])
                if p:
                    _ = p.value()[unsafe_offset=0].x
                _ = t.insert[FVel](ents[i], FVel(1.0))
                _ = t.insert[FTag](ents[i], FTag(7))
        var e2 = t.spawn_empty()
        _ = t.insert[FPos](e2, FPos(0.0, 0.0, 0.0))
        for i in range(0, 300, 17):
            _ = t.despawn(ents[i])
        _ = t.state_checksum()
    print("ecs fuzz done iters=", iters, "restored=", ok, "rejected=", rejected)
