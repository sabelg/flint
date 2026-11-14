"""Cooked models: a glTF file's NativeModel laid out as the engine holds it.

The pipeline (tools/cook.mojo, run by `tools/run.sh assets` and `cook`) reads
each .glb once, through the same reader the game falls back to
(assets/glb_native.mojo), and writes what came out: every mesh's arrays,
textures already decoded to pixels, nodes, skins and clips, each array as
its element bytes after its length. Loading one is a file read, an LZ4
decode (core/lz4.mojo) and a memcpy per array: no JSON, no accessors, no
PNG or JPEG decoding.

A cooked file sits at build/cooked/<the source's path>.sdmodel (or in a pack,
assets/vfs.mojo). It starts with the cooked header (assets/cooked.mojo, kind
COOKED_MODEL), then the source stamp: the .glb's size, modification time and
a hash of its bytes. A cooked file is used only while its source is the file
it was cooked from: same size and time, or failing that the same hash. So a
changed .glb is read as glTF until the pipeline cooks it again, and a hot
reload never sees an old model. COOKED_MODEL_VERSION is bumped whenever this
layout or the reader in glb_native.mojo changes what a model holds; the
version is part of the stamp, and an older cook is never read.
"""
from std.ffi import external_call
from std.sys import size_of
from flint.core.lz4 import address_of, block_hash, lz4_pack, lz4_unpack, hash_list
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.assets.cooked import COOKED_HEADER_BYTES, COOKED_MAGIC
from flint.assets.glb import GlbInfo
from flint.assets.manifest import ENGINE_FORMAT
from flint.assets.model import NativeAnimation, NativeChannel, NativeMesh, NativeModel, NativeNode, NativeSkin, NativeTexture

comptime COOKED_MODEL: Int = 2  # the cooked header's kind
comptime COOKED_MODEL_VERSION: Int = 2  # 2: centers of rotation per skinned mesh (package 3.4)
comptime MODEL_STAMP_BYTES: Int = 64
comptime COOKED_MODEL_SUFFIX = ".sdmodel"


@always_inline
def f32_bits(value: Float32) -> UInt32:
    var copy = value
    return Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0]


comptime STAMP_CHECKSUM_AT: Int = COOKED_HEADER_BYTES + MODEL_STAMP_BYTES - 8


def stamp_checksum(data: List[UInt8]) -> UInt64:
    """Covers the source stamp, which the payload hash does not: model_info
    reads only the stamp."""
    return block_hash(address_of(data) + COOKED_HEADER_BYTES, STAMP_CHECKSUM_AT - COOKED_HEADER_BYTES)


struct SourceStamp(Copyable, ImplicitlyCopyable, Movable):
    """Which .glb a cooked model was made from."""
    var size: Int
    var mtime: Int  # nanoseconds
    var hash: UInt64  # block_hash of its bytes

    def __init__(out self):
        self.size = -1
        self.mtime = 0
        self.hash = 0

    def __init__(out self, size: Int, mtime: Int, hash: UInt64):
        self.size = size
        self.mtime = mtime
        self.hash = hash


struct CookedModel(Movable):
    var valid: Bool
    var error: String
    var stamp: SourceStamp
    var info: GlbInfo
    var asset_id: UInt64
    var model: NativeModel

    def __init__(out self):
        self.valid = False
        self.error = String()
        self.stamp = SourceStamp()
        self.info = GlbInfo()
        self.asset_id = 0
        self.model = NativeModel()

    def take_model(mut self) -> NativeModel:
        var model = self.model^
        self.model = NativeModel()
        return model^


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

struct ModelWriter:
    var data: List[UInt8]

    def __init__(out self):
        self.data = List[UInt8]()

    def take(mut self) -> List[UInt8]:
        var out = self.data^
        self.data = List[UInt8]()
        return out^

    def raw(mut self, address: Int, count: Int):
        if count <= 0:
            return
        var at = len(self.data)
        self.data.resize(unsafe_uninit_length=at + count)
        _ = external_call["memcpy", Int](address_of(self.data) + at, address, count)

    def u64(mut self, value: UInt64):
        for shift in range(8):
            self.data.append(UInt8((value >> UInt64(shift * 8)) & 0xFF))

    def i64(mut self, value: Int):
        self.u64(UInt64(value))

    def u32(mut self, value: UInt32):
        for shift in range(4):
            self.data.append(UInt8((value >> UInt32(shift * 8)) & 0xFF))

    def f32(mut self, value: Float32):
        self.u32(f32_bits(value))

    def vec3(mut self, value: Vec3):
        self.f32(value.x)
        self.f32(value.y)
        self.f32(value.z)

    def quat(mut self, value: Quat):
        self.f32(value.x)
        self.f32(value.y)
        self.f32(value.z)
        self.f32(value.w)

    def text(mut self, value: String):
        var bytes = value.as_bytes()
        self.i64(len(bytes))
        for byte in bytes:
            self.data.append(byte)

    def floats(mut self, values: List[Float32]):
        self.i64(len(values))
        self.raw(Int(values.unsafe_ptr()), len(values) * 4)

    def bytes(mut self, values: List[UInt8]):
        self.i64(len(values))
        self.raw(Int(values.unsafe_ptr()), len(values))

    def halves(mut self, values: List[UInt16]):
        self.i64(len(values))
        self.raw(Int(values.unsafe_ptr()), len(values) * 2)

    def words(mut self, values: List[UInt32]):
        self.i64(len(values))
        self.raw(Int(values.unsafe_ptr()), len(values) * 4)

    def ints(mut self, values: List[Int]):
        self.i64(len(values))
        self.raw(Int(values.unsafe_ptr()), len(values) * size_of[Int]())


def model_bytes(model: NativeModel) -> List[UInt8]:
    """Everything glb_native.mojo fills in, in one buffer (before packing)."""
    var w = ModelWriter()
    w.u32(UInt32(COOKED_MODEL_VERSION))
    w.i64(model.max_influences)
    w.i64(len(model.textures))
    for texture in model.textures:
        w.i64(texture.width)
        w.i64(texture.height)
        w.words(texture.pixels)
    w.i64(len(model.meshes))
    for mesh in model.meshes:
        w.floats(mesh.positions)
        w.floats(mesh.normals)
        w.floats(mesh.texcoords)
        w.bytes(mesh.colors)
        w.bytes(mesh.ao)
        w.ints(mesh.indices)
        w.halves(mesh.joints)
        w.floats(mesh.weights)
        w.i64(mesh.texture)
        w.u32(mesh.base_color)
        w.vec3(mesh.emissive_color)
        w.f32(mesh.emissive_strength)
        w.i64(mesh.skin)
        w.i64(mesh.alpha_mode)
        w.i64(1 if mesh.double_sided else 0)
        w.i64(mesh.morph_count)
        w.floats(mesh.morph_positions)
        w.floats(mesh.morph_normals)
        w.floats(mesh.centers)
    w.i64(len(model.nodes))
    for node in model.nodes:
        w.text(node.name)
        w.i64(node.mesh)
        w.ints(node.mesh_indices)
        w.ints(node.children)
        w.vec3(node.translation)
        w.quat(node.rotation)
        w.vec3(node.scale)
        w.vec3(node.base_translation)
        w.quat(node.base_rotation)
        w.vec3(node.base_scale)
        for index in range(16):
            w.f32(node.matrix.values[index])
        w.i64(1 if node.has_matrix else 0)
    w.ints(model.roots)
    w.i64(len(model.skins))
    for skin in model.skins:
        w.ints(skin.joints)
        w.floats(skin.inverse_bind)
    w.i64(len(model.animations))
    for animation in model.animations:
        w.text(animation.name)
        w.f32(animation.duration)
        w.i64(len(animation.channels))
        for channel in animation.channels:
            w.i64(channel.node)
            w.i64(channel.path)
            w.i64(1 if channel.step else 0)
            w.floats(channel.times)
            w.floats(channel.values)
    w.i64(len(model.morph_names))
    for name in model.morph_names:
        w.text(name)
    return w.take()


def encode_cooked_model(model: NativeModel, info: GlbInfo, stamp: SourceStamp, asset_id: UInt64) -> List[UInt8]:
    var raw = model_bytes(model)
    var packed = lz4_pack(raw)
    _ = raw^
    var w = ModelWriter()
    w.u32(COOKED_MAGIC)
    w.u32(UInt32(ENGINE_FORMAT))
    w.u32(UInt32(COOKED_MODEL))
    w.u32(UInt32(MODEL_STAMP_BYTES + len(packed)))
    w.u64(asset_id)
    w.u64(hash_list(packed))
    # The source stamp and what inspect_glb says of the file, readable
    # without unpacking.
    w.u32(UInt32(COOKED_MODEL_VERSION))
    w.u32(UInt32(1) if info.valid else UInt32(0))
    w.i64(stamp.size)
    w.i64(stamp.mtime)
    w.u64(stamp.hash)
    w.u32(UInt32(info.animation_count))
    w.u32(UInt32(info.image_count))
    w.u32(UInt32(info.embedded_image_count))
    w.u32(UInt32(info.skin_count))
    w.u32(UInt32(info.mesh_count))
    w.u32(UInt32(info.json_size))
    w.u64(stamp_checksum(w.data))
    w.raw(Int(packed.unsafe_ptr()), len(packed))
    _ = packed^
    return w.take()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

struct ModelReader:
    var address: Int
    var size: Int
    var at: Int
    var failed: Bool

    def __init__(out self, data: List[UInt8], start: Int = 0):
        self.address = address_of(data)
        self.size = len(data)
        self.at = start
        self.failed = False

    def take(mut self, count: Int) -> Int:
        """The address of the next `count` bytes, or 0 (and failed) past the end."""
        if self.failed or count < 0 or self.at + count > self.size:
            self.failed = True
            return 0
        var address = self.address + self.at
        self.at += count
        return address

    def i64(mut self) -> Int:
        var address = self.take(8)
        if address == 0:
            return 0
        return Int(Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=address).unsafe_load[alignment=1]())

    def u32(mut self) -> UInt32:
        var address = self.take(4)
        if address == 0:
            return 0
        return Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=address).unsafe_load[alignment=1]()

    def u64(mut self) -> UInt64:
        var address = self.take(8)
        if address == 0:
            return 0
        return Pointer[UInt64, origin=MutAnyOrigin](unsafe_from_address=address).unsafe_load[alignment=1]()

    def f32(mut self) -> Float32:
        var address = self.take(4)
        if address == 0:
            return 0
        return Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=address).unsafe_load[alignment=1]()

    def vec3(mut self) -> Vec3:
        var x = self.f32()
        var y = self.f32()
        return Vec3(x, y, self.f32())

    def quat(mut self) -> Quat:
        var x = self.f32()
        var y = self.f32()
        var z = self.f32()
        return Quat(x, y, z, self.f32())

    def count(mut self, element: Int) -> Int:
        """An array's length, checked against what is left."""
        var n = self.i64()
        if n < 0 or n > (self.size - self.at) // element:
            self.failed = True
            return 0
        return n

    def text(mut self) -> String:
        var n = self.count(1)
        var address = self.take(n)
        if n == 0 or address == 0:
            return String()
        var bytes = List[UInt8](capacity=n)
        bytes.resize(unsafe_uninit_length=n)
        _ = external_call["memcpy", Int](address_of(bytes), address, n)
        return String(unsafe_from_utf8=bytes)

    def floats(mut self) -> List[Float32]:
        var n = self.count(4)
        var out = List[Float32](capacity=n)
        if n > 0:
            out.resize(unsafe_uninit_length=n)
            _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), self.take(n * 4), n * 4)
        return out^

    def bytes(mut self) -> List[UInt8]:
        var n = self.count(1)
        var out = List[UInt8](capacity=n)
        if n > 0:
            out.resize(unsafe_uninit_length=n)
            _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), self.take(n), n)
        return out^

    def halves(mut self) -> List[UInt16]:
        var n = self.count(2)
        var out = List[UInt16](capacity=n)
        if n > 0:
            out.resize(unsafe_uninit_length=n)
            _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), self.take(n * 2), n * 2)
        return out^

    def words(mut self) -> List[UInt32]:
        var n = self.count(4)
        var out = List[UInt32](capacity=n)
        if n > 0:
            out.resize(unsafe_uninit_length=n)
            _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), self.take(n * 4), n * 4)
        return out^

    def ints(mut self) -> List[Int]:
        var n = self.count(size_of[Int]())
        var out = List[Int](capacity=n)
        if n > 0:
            out.resize(unsafe_uninit_length=n)
            _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), self.take(n * size_of[Int]()), n * size_of[Int]())
        return out^


def read_model(raw: List[UInt8], mut error: String) -> NativeModel:
    """The model in an unpacked buffer; on damage an empty model and `error`."""
    var r = ModelReader(raw)
    var model = NativeModel()
    if Int(r.u32()) != COOKED_MODEL_VERSION:
        error = "cooked by another version"
        return NativeModel()
    model.max_influences = r.i64()
    var textures = r.count(24)
    for _ in range(textures):
        var texture = NativeTexture()
        texture.width = r.i64()
        texture.height = r.i64()
        texture.pixels = r.words()
        model.textures.append(texture^)
    var meshes = r.count(8)
    for _ in range(meshes):
        var mesh = NativeMesh()
        mesh.positions = r.floats()
        mesh.normals = r.floats()
        mesh.texcoords = r.floats()
        mesh.colors = r.bytes()
        mesh.ao = r.bytes()
        mesh.indices = r.ints()
        mesh.joints = r.halves()
        mesh.weights = r.floats()
        mesh.texture = r.i64()
        mesh.base_color = r.u32()
        mesh.emissive_color = r.vec3()
        mesh.emissive_strength = r.f32()
        mesh.skin = r.i64()
        mesh.alpha_mode = r.i64()
        mesh.double_sided = r.i64() != 0
        mesh.morph_count = r.i64()
        mesh.morph_positions = r.floats()
        mesh.morph_normals = r.floats()
        mesh.centers = r.floats()
        if r.failed:
            break
        model.meshes.append(mesh^)
    var nodes = r.count(8)
    for _ in range(nodes):
        var node = NativeNode()
        node.name = r.text()
        node.mesh = r.i64()
        node.mesh_indices = r.ints()
        node.children = r.ints()
        node.translation = r.vec3()
        node.rotation = r.quat()
        node.scale = r.vec3()
        node.base_translation = r.vec3()
        node.base_rotation = r.quat()
        node.base_scale = r.vec3()
        for index in range(16):
            node.matrix.values[index] = r.f32()
        node.has_matrix = r.i64() != 0
        if r.failed:
            break
        model.nodes.append(node^)
    model.roots = r.ints()
    var skins = r.count(16)
    for _ in range(skins):
        var skin = NativeSkin()
        skin.joints = r.ints()
        skin.inverse_bind = r.floats()
        model.skins.append(skin^)
    var animations = r.count(20)
    for _ in range(animations):
        var animation = NativeAnimation()
        animation.name = r.text()
        animation.duration = r.f32()
        var channels = r.count(40)
        for _ in range(channels):
            var channel = NativeChannel()
            channel.node = r.i64()
            channel.path = r.i64()
            channel.step = r.i64() != 0
            channel.times = r.floats()
            channel.values = r.floats()
            animation.channels.append(channel^)
        if r.failed:
            break
        model.animations.append(animation^)
    var names = r.count(8)
    for _ in range(names):
        model.morph_names.append(r.text())
    if r.failed or r.at != r.size:
        error = "damaged: the arrays do not add up to the file"
        return NativeModel()
    return model^


struct CookedStamp(Copyable, Movable):
    """The part of a cooked model readable without unpacking it."""
    var valid: Bool
    var error: String
    var asset_id: UInt64
    var stamp: SourceStamp
    var info: GlbInfo

    def __init__(out self):
        self.valid = False
        self.error = String()
        self.asset_id = 0
        self.stamp = SourceStamp()
        self.info = GlbInfo()


def read_cooked_stamp(data: List[UInt8], whole_size: Int = -1) -> CookedStamp:
    """The header and stamp of a cooked model. `data` may be just its first
    bytes when `whole_size` gives the file's size."""
    var out = CookedStamp()
    var total = whole_size if whole_size >= 0 else len(data)
    if len(data) < COOKED_HEADER_BYTES + MODEL_STAMP_BYTES:
        out.error = "too short"
        return out^
    var r = ModelReader(data)
    if r.u32() != COOKED_MAGIC:
        out.error = "not a cooked file"
        return out^
    var engine_format = Int(r.u32())
    var kind = Int(r.u32())
    var payload = Int(r.u32())
    out.asset_id = r.u64()
    _ = r.u64()
    if engine_format != ENGINE_FORMAT or kind != COOKED_MODEL:
        out.error = "another engine format or kind"
        return out^
    if payload != total - COOKED_HEADER_BYTES:
        out.error = "cut short"
        return out^
    var version = Int(r.u32())
    if version != COOKED_MODEL_VERSION:
        out.error = "cooked by version " + String(version) + ", this engine reads " + String(COOKED_MODEL_VERSION)
        return out^
    out.info.valid = r.u32() != 0
    out.stamp.size = r.i64()
    out.stamp.mtime = r.i64()
    out.stamp.hash = r.u64()
    out.info.animation_count = Int(r.u32())
    out.info.image_count = Int(r.u32())
    out.info.embedded_image_count = Int(r.u32())
    out.info.skin_count = Int(r.u32())
    out.info.mesh_count = Int(r.u32())
    out.info.json_size = Int(r.u32())
    if r.u64() != stamp_checksum(data):
        out.error = "damaged: the source stamp does not match its checksum"
        return out^
    out.valid = True
    return out^


def decode_cooked_model(data: List[UInt8]) -> CookedModel:
    """A cooked model file read back. `valid` False (with `error`) for any damage."""
    var out = CookedModel()
    var stamp = read_cooked_stamp(data)
    if not stamp.valid:
        out.error = stamp.error
        return out^
    var r = ModelReader(data, COOKED_HEADER_BYTES - 8)
    var payload_hash = r.u64()
    var packed_at = COOKED_HEADER_BYTES + MODEL_STAMP_BYTES
    if block_hash(address_of(data) + packed_at, len(data) - packed_at) != payload_hash:
        out.error = "damaged: the payload hash does not match"
        return out^
    var raw = lz4_unpack(data, packed_at)
    if len(raw) == 0:
        out.error = "damaged: the packed model does not unpack"
        return out^
    var error = String()
    out.model = read_model(raw, error)
    _ = raw^
    if error.byte_length() > 0:
        out.error = error
        return out^
    out.stamp = stamp.stamp
    out.info = stamp.info
    out.asset_id = stamp.asset_id
    out.valid = True
    return out^


# ---------------------------------------------------------------------------
# Comparing (the cook's own check, and the tests')
# ---------------------------------------------------------------------------

def same_floats(a: List[Float32], b: List[Float32]) -> Bool:
    """Bit for bit, so NaNs and negative zeros count."""
    if len(a) != len(b):
        return False
    return len(a) == 0 or external_call["memcmp", Int32](Int(a.unsafe_ptr()), Int(b.unsafe_ptr()), len(a) * 4) == 0


def same_bytes(a: List[UInt8], b: List[UInt8]) -> Bool:
    return len(a) == len(b) and (len(a) == 0 or external_call["memcmp", Int32](Int(a.unsafe_ptr()), Int(b.unsafe_ptr()), len(a)) == 0)


def same_ints(a: List[Int], b: List[Int]) -> Bool:
    return len(a) == len(b) and (len(a) == 0 or external_call["memcmp", Int32](Int(a.unsafe_ptr()), Int(b.unsafe_ptr()), len(a) * size_of[Int]()) == 0)


def same_f(a: Float32, b: Float32) -> Bool:
    return f32_bits(a) == f32_bits(b)


def same_vec(a: Vec3, b: Vec3) -> Bool:
    return same_f(a.x, b.x) and same_f(a.y, b.y) and same_f(a.z, b.z)


def same_quat(a: Quat, b: Quat) -> Bool:
    return same_f(a.x, b.x) and same_f(a.y, b.y) and same_f(a.z, b.z) and same_f(a.w, b.w)


def model_difference(a: NativeModel, b: NativeModel) -> String:
    """Empty when the two hold the same model, bit for bit, in everything a
    glTF read fills in; else the first thing that differs."""
    if a.max_influences != b.max_influences:
        return "max_influences"
    if len(a.textures) != len(b.textures):
        return "texture count"
    for i in range(len(a.textures)):
        if a.textures[i].width != b.textures[i].width or a.textures[i].height != b.textures[i].height:
            return "texture " + String(i) + " size"
        if len(a.textures[i].pixels) != len(b.textures[i].pixels) or (len(a.textures[i].pixels) > 0 and external_call["memcmp", Int32](Int(a.textures[i].pixels.unsafe_ptr()), Int(b.textures[i].pixels.unsafe_ptr()), len(a.textures[i].pixels) * 4) != 0):
            return "texture " + String(i) + " pixels"
    if len(a.meshes) != len(b.meshes):
        return "mesh count"
    for i in range(len(a.meshes)):
        ref x = a.meshes[i]
        ref y = b.meshes[i]
        var name = "mesh " + String(i) + " "
        if not same_floats(x.positions, y.positions):
            return name + "positions"
        if not same_floats(x.normals, y.normals):
            return name + "normals"
        if not same_floats(x.texcoords, y.texcoords):
            return name + "texcoords"
        if not same_bytes(x.colors, y.colors) or not same_bytes(x.ao, y.ao):
            return name + "colours"
        if not same_ints(x.indices, y.indices):
            return name + "indices"
        if len(x.joints) != len(y.joints) or (len(x.joints) > 0 and external_call["memcmp", Int32](Int(x.joints.unsafe_ptr()), Int(y.joints.unsafe_ptr()), len(x.joints) * 2) != 0):
            return name + "joints"
        if not same_floats(x.weights, y.weights):
            return name + "weights"
        if x.texture != y.texture or x.base_color != y.base_color or not same_vec(x.emissive_color, y.emissive_color) or not same_f(x.emissive_strength, y.emissive_strength):
            return name + "material"
        if x.skin != y.skin or x.alpha_mode != y.alpha_mode or x.double_sided != y.double_sided:
            return name + "skin or alpha"
        if x.morph_count != y.morph_count or not same_floats(x.morph_positions, y.morph_positions) or not same_floats(x.morph_normals, y.morph_normals):
            return name + "morphs"
        if not same_floats(x.centers, y.centers):
            return name + "centers of rotation"
    if len(a.nodes) != len(b.nodes):
        return "node count"
    for i in range(len(a.nodes)):
        ref x = a.nodes[i]
        ref y = b.nodes[i]
        var name = "node " + String(i) + " "
        if x.name != y.name or x.mesh != y.mesh or not same_ints(x.mesh_indices, y.mesh_indices) or not same_ints(x.children, y.children):
            return name + "name, mesh or children"
        if not same_vec(x.translation, y.translation) or not same_quat(x.rotation, y.rotation) or not same_vec(x.scale, y.scale):
            return name + "transform"
        if not same_vec(x.base_translation, y.base_translation) or not same_quat(x.base_rotation, y.base_rotation) or not same_vec(x.base_scale, y.base_scale):
            return name + "rest transform"
        for index in range(16):
            if not same_f(x.matrix.values[index], y.matrix.values[index]):
                return name + "matrix"
        if x.has_matrix != y.has_matrix:
            return name + "matrix flag"
    if not same_ints(a.roots, b.roots):
        return "roots"
    if len(a.skins) != len(b.skins):
        return "skin count"
    for i in range(len(a.skins)):
        if not same_ints(a.skins[i].joints, b.skins[i].joints) or not same_floats(a.skins[i].inverse_bind, b.skins[i].inverse_bind):
            return "skin " + String(i)
    if len(a.animations) != len(b.animations):
        return "clip count"
    for i in range(len(a.animations)):
        ref x = a.animations[i]
        ref y = b.animations[i]
        if x.name != y.name or not same_f(x.duration, y.duration) or len(x.channels) != len(y.channels):
            return "clip " + String(i)
        for c in range(len(x.channels)):
            ref p = x.channels[c]
            ref q = y.channels[c]
            if p.node != q.node or p.path != q.path or p.step != q.step or not same_floats(p.times, q.times) or not same_floats(p.values, q.values):
                return "clip " + String(i) + " channel " + String(c)
    if len(a.morph_names) != len(b.morph_names):
        return "morph name count"
    for i in range(len(a.morph_names)):
        if a.morph_names[i] != b.morph_names[i]:
            return "morph name " + String(i)
    return String()
