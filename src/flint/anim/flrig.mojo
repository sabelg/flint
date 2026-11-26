"""Binary `.flrig` (`FRIG` FourCC) Rig, Ragdoll Chain & Socket Definition Codec (`flint.anim.flrig`)."""
from flint.core.math import Vec3
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.core.transform import Quat
from flint.platform.os import read_bytes, write_bytes

comptime FRIG_FOURCC: UInt32 = fourcc("FRIG")
comptime FRIG_VERSION: Int = 1


@fieldwise_init
struct FrigJointDef(Copyable, Movable):
    var name: String
    var parent_index: Int
    var translation: Vec3
    var rotation: Quat
    var scale: Vec3

    def __init__(out self):
        self.name = String()
        self.parent_index = -1
        self.translation = Vec3()
        self.rotation = Quat.identity()
        self.scale = Vec3(1.0, 1.0, 1.0)


@fieldwise_init
struct RagdollBoneConfig(Copyable, Movable):
    var bone_name: String
    var radius: Float32
    var length: Float32
    var mass: Float32

    def __init__(out self):
        self.bone_name = String()
        self.radius = Float32(0.1)
        self.length = Float32(0.3)
        self.mass = Float32(1.0)


struct RagdollChain(Copyable, Movable):
    var name: String
    var bones: List[RagdollBoneConfig]

    def __init__(out self, name: String = ""):
        self.name = name
        self.bones = List[RagdollBoneConfig]()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.bones = copy.bones.copy()


@fieldwise_init
struct FrigSocketDef(Copyable, Movable):
    var name: String
    var bone_name: String
    var translation: Vec3
    var rotation: Quat
    var scale: Vec3

    def __init__(out self):
        self.name = String()
        self.bone_name = String()
        self.translation = Vec3()
        self.rotation = Quat.identity()
        self.scale = Vec3(1.0, 1.0, 1.0)


struct FrigRigDef(Copyable, Movable):
    var name: String
    var joints: List[FrigJointDef]
    var ragdoll_chains: List[RagdollChain]
    var sockets: List[FrigSocketDef]

    def __init__(out self, name: String = ""):
        self.name = name
        self.joints = List[FrigJointDef]()
        self.ragdoll_chains = List[RagdollChain]()
        self.sockets = List[FrigSocketDef]()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.joints = copy.joints.copy()
        self.ragdoll_chains = copy.ragdoll_chains.copy()
        self.sockets = copy.sockets.copy()

    def find_joint(self, name: String) -> Int:
        for i in range(len(self.joints)):
            if self.joints[i].name == name:
                return i
        return -1


def validate_rig_bindings(rig: FrigRigDef) raises:
    """Validates joint parent indices and verifies all socket and ragdoll bone names resolve."""
    var nj = len(rig.joints)
    for i in range(nj):
        var p = rig.joints[i].parent_index
        if p < -1 or p >= nj or p == i:
            raise Error(
                "FrigRigDef '" + rig.name + "': invalid parent_index " + String(p)
                + " on joint '" + rig.joints[i].name + "'"
            )
    for si in range(len(rig.sockets)):
        ref s = rig.sockets[si]
        if rig.find_joint(s.bone_name) < 0:
            raise Error(
                "FrigRigDef '" + rig.name + "': unresolved socket bone '" + s.bone_name
                + "' in socket '" + s.name + "'"
            )
    for ci in range(len(rig.ragdoll_chains)):
        ref ch = rig.ragdoll_chains[ci]
        for bi in range(len(ch.bones)):
            ref b = ch.bones[bi]
            if rig.find_joint(b.bone_name) < 0:
                raise Error(
                    "FrigRigDef '" + rig.name + "': unresolved ragdoll bone '" + b.bone_name
                    + "' in chain '" + ch.name + "'"
                )


def save_flrig_bytes(rig: FrigRigDef) -> List[UInt8]:
    var w = BlobWriter()
    w.begin(FRIG_FOURCC, FRIG_VERSION)
    w.text(rig.name)

    w.u32(UInt32(len(rig.joints)))
    for i in range(len(rig.joints)):
        ref j = rig.joints[i]
        w.text(j.name)
        w.integer(j.parent_index)
        w.vec3(j.translation)
        w.quat(j.rotation)
        w.vec3(j.scale)

    w.u32(UInt32(len(rig.ragdoll_chains)))
    for ci in range(len(rig.ragdoll_chains)):
        ref ch = rig.ragdoll_chains[ci]
        w.text(ch.name)
        w.u32(UInt32(len(ch.bones)))
        for bi in range(len(ch.bones)):
            ref b = ch.bones[bi]
            w.text(b.bone_name)
            w.f32(b.radius)
            w.f32(b.length)
            w.f32(b.mass)

    w.u32(UInt32(len(rig.sockets)))
    for si in range(len(rig.sockets)):
        ref s = rig.sockets[si]
        w.text(s.name)
        w.text(s.bone_name)
        w.vec3(s.translation)
        w.quat(s.rotation)
        w.vec3(s.scale)

    w.end()
    return w.data.copy()


def save_flrig_file(path: String, rig: FrigRigDef) -> Bool:
    var bytes = save_flrig_bytes(rig)
    return write_bytes(path, bytes)


def load_flrig_bytes(data: List[UInt8]) raises -> FrigRigDef:
    if len(data) < 12:
        raise Error("invalid or empty .flrig buffer")
    var r = BlobReader(data.copy())
    _ = r.enter(FRIG_FOURCC, FRIG_VERSION)
    var out = FrigRigDef(r.text())
    if r.failed:
        raise Error("corrupted .flrig header")

    var nj = Int(r.u32())
    if r.failed or nj < 0 or nj > 65536 or nj > max(0, r.limit() - r.offset) // 40:
        raise Error("corrupted .flrig joint count")
    for _ in range(nj):
        var jn = r.text()
        var pi = r.integer()
        var tr = r.vec3()
        var rt = r.quat()
        var sc = r.vec3()
        if r.failed:
            raise Error("truncated .flrig joint entry")
        out.joints.append(FrigJointDef(jn, pi, tr, rt, sc))

    var nc = Int(r.u32())
    if r.failed or nc < 0 or nc > 4096 or nc > max(0, r.limit() - r.offset) // 8:
        raise Error("corrupted .flrig ragdoll chain count")
    for _ in range(nc):
        var ch = RagdollChain(r.text())
        var nb = Int(r.u32())
        if r.failed or nb < 0 or nb > 4096 or nb > max(0, r.limit() - r.offset) // 16:
            raise Error("corrupted .flrig ragdoll bone count")
        for _ in range(nb):
            var bn = r.text()
            var rad = r.f32()
            var ln = r.f32()
            var ms = r.f32()
            if r.failed:
                raise Error("truncated .flrig ragdoll bone entry")
            ch.bones.append(RagdollBoneConfig(bn, rad, ln, ms))
        out.ragdoll_chains.append(ch^)

    var ns = Int(r.u32())
    if r.failed or ns < 0 or ns > 65536 or ns > max(0, r.limit() - r.offset) // 48:
        raise Error("corrupted .flrig socket count")
    for _ in range(ns):
        var sn = r.text()
        var bn = r.text()
        var tr = r.vec3()
        var rt = r.quat()
        var sc = r.vec3()
        if r.failed:
            raise Error("truncated .flrig socket entry")
        out.sockets.append(FrigSocketDef(sn, bn, tr, rt, sc))

    r.leave()
    if r.failed:
        raise Error("corrupted .flrig payload tail")
    return out^


def load_flrig_file(path: String) raises -> FrigRigDef:
    var raw = read_bytes(path)
    if len(raw) == 0:
        raise Error("failed to read .flrig file: " + path)
    return load_flrig_bytes(raw)
