"""Where the game's files come from: three mounts, one name for each file.

A file is named by its path under the repository (assets/deepsong/characters/
Bront_Base_LOD0.glb), whichever mount serves it:

- cooked: build/cooked/<path>.sdmodel, a model as the engine holds it
  (assets/cooked_model.mojo), written by the pipeline;
- pack: build/deepsong.sdpack, one file holding cooked models and other files
  under their paths (`tools/run.sh cook --pack`), for a build that ships
  without loose files;
- loose: the file itself in assets/.

A model is taken from the first of cooked, pack and loose that has it and is
current: a cooked model only while its source .glb is the one it was cooked
from (same size and time, or the same bytes; with no source at all, as in a
shipped build, the cooked one stands). Anything else, a stale, damaged or
missing cook, falls back to reading the glTF, so a cook can only ever make
loading faster, never change what loads. SD_COOKED=0 reads every model as
glTF. Other files (manifests) are read loose first, then from the pack.

Everything here is plain functions over files, safe on the asset-stream
threads (platform/native_runtime.mojo).
"""
from std.ffi import external_call
from flint.core.lz4 import address_of, hash_list
from flint.assets.cooked_model import (
    COOKED_MODEL_SUFFIX,
    CookedStamp,
    SourceStamp,
    decode_cooked_model,
    read_cooked_stamp,
)
from flint.assets.glb import GlbInfo, inspect_glb
from flint.assets.glb_native import parse_native_glb
from flint.assets.model import NativeModel
from flint.platform.os import env_now, file_stamp, read_at, read_bytes, write_bytes, open_read, close_fd

comptime COOKED_ROOT = "build/cooked/"
comptime PACK_PATH = "build/assets.flpk"
comptime PACK_MAGIC: UInt32 = 0x4B504C46  # "FLPK"
comptime PACK_VERSION: UInt32 = 2  # v2 with 64-byte aligned entries & persistent mmap
comptime PACK_HEADER_BYTES: Int = 16

comptime MOUNT_NONE: Int = 0
comptime MOUNT_COOKED: Int = 1
comptime MOUNT_PACK: Int = 2
comptime MOUNT_LOOSE: Int = 3


def mount_name(mount: Int) -> String:
    if mount == MOUNT_COOKED:
        return "cooked"
    if mount == MOUNT_PACK:
        return "pack"
    if mount == MOUNT_LOOSE:
        return "loose"
    return "none"


def cooked_enabled() -> Bool:
    return env_now("FLINT_COOKED") != "0"


def cooked_model_path(source: String) -> String:
    return String(COOKED_ROOT) + source + COOKED_MODEL_SUFFIX


def pack_path() -> String:
    """SD_PACK names another pack (tests)."""
    var named = env_now("FLINT_PACK")
    return named if named.byte_length() > 0 else String(PACK_PATH)


def source_stamp(path: String) -> SourceStamp:
    """A .glb's size, time and hash, as a cook records it."""
    var found = file_stamp(path)
    if found[0] < 0:
        return SourceStamp()
    return SourceStamp(found[0], found[1], hash_list(read_bytes(path)))


def stamp_is_current(stamp: SourceStamp, path: String) -> Bool:
    """Whether a cook of `path` with this stamp may stand for it."""
    var found = file_stamp(path)
    if found[0] < 0:
        return True  # no source here: the cooked file is all there is
    if found[0] != stamp.size:
        return False
    if found[1] == stamp.mtime:
        return True
    # Touched but maybe not changed (a checkout, a copy): the bytes decide.
    return hash_list(read_bytes(path)) == stamp.hash


# ---------------------------------------------------------------------------
# The pack: a directory of paths, then each file's bytes
# ---------------------------------------------------------------------------

struct PackEntry(Copyable, Movable):
    var path: String
    var offset: Int
    var size: Int

    def __init__(out self, path: String, offset: Int, size: Int):
        self.path = path
        self.offset = offset
        self.size = size


def u32_at(data: List[UInt8], at: Int) -> Int:
    return Int(data[at]) | (Int(data[at + 1]) << 8) | (Int(data[at + 2]) << 16) | (Int(data[at + 3]) << 24)


def u64_at(data: List[UInt8], at: Int) -> Int:
    return u32_at(data, at) | (u32_at(data, at + 4) << 32)


def pack_directory(pack: String) -> List[PackEntry]:
    """The pack's entries, or none if it is missing or damaged."""
    var entries = List[PackEntry]()
    var header = read_at(pack, 0, PACK_HEADER_BYTES)
    if len(header) != PACK_HEADER_BYTES or UInt32(u32_at(header, 0)) != PACK_MAGIC:
        return entries^
    var ver = UInt32(u32_at(header, 4))
    if ver != 1 and ver != 2:
        return entries^
    var count = u32_at(header, 8)
    var directory_bytes = u32_at(header, 12)
    var directory = read_at(pack, PACK_HEADER_BYTES, directory_bytes)
    if len(directory) != directory_bytes:
        return entries^
    var total = file_stamp(pack)[0]
    var at = 0
    for _ in range(count):
        if at + 20 > len(directory):
            return List[PackEntry]()
        var length = u32_at(directory, at)
        if at + 20 + length > len(directory):
            return List[PackEntry]()
        var name = List[UInt8]()
        for index in range(length):
            name.append(directory[at + 4 + index])
        var offset = u64_at(directory, at + 4 + length)
        var size = u64_at(directory, at + 12 + length)
        if offset < PACK_HEADER_BYTES + directory_bytes or size < 0 or offset > total or size > total - offset:
            return List[PackEntry]()
        entries.append(PackEntry(String(unsafe_from_utf8=name), offset, size))
        at += 20 + length
    return entries^


@always_inline
def _ptr_u32(p: Pointer[UInt8, ImmutAnyOrigin], idx: Int) -> Int:
    return Int(p[unsafe_offset=idx]) | (Int(p[unsafe_offset=idx + 1]) << 8) | (Int(p[unsafe_offset=idx + 2]) << 16) | (Int(p[unsafe_offset=idx + 3]) << 24)


@always_inline
def _ptr_u64(p: Pointer[UInt8, ImmutAnyOrigin], idx: Int) -> Int:
    return _ptr_u32(p, idx) | (_ptr_u32(p, idx + 4) << 32)


struct FlpkMappedArchive(Movable):
    """Persistent zero-copy mmap .flpk v2 VFS with 64-byte alignment and zero directory allocations."""
    var path: String
    var fd: Int
    var addr: Int
    var size: Int
    var entries: List[PackEntry]

    def __init__(out self):
        self.path = String()
        self.fd = -1
        self.addr = 0
        self.size = 0
        self.entries = List[PackEntry]()

    def close(mut self):
        if self.addr != 0 and self.size > 0:
            _ = external_call["munmap", Int32](Int(self.addr), Int64(self.size))
            self.addr = 0
        if self.fd >= 0:
            close_fd(self.fd)
            self.fd = -1
        self.size = 0
        self.entries.clear()

    @staticmethod
    def open(pack_path: String) -> Self:
        var arch = Self()
        arch.path = pack_path
        var stamp = file_stamp(pack_path)
        if stamp[0] <= PACK_HEADER_BYTES:
            return arch^
        arch.size = stamp[0]
        arch.fd = open_read(pack_path)
        if arch.fd < 0:
            return arch^
        # PROT_READ = 1, MAP_PRIVATE = 2
        var ptr = external_call["mmap", Int](Int(0), Int64(arch.size), Int32(1), Int32(2), Int32(arch.fd), Int64(0))
        if ptr == 0 or ptr == -1:
            arch.addr = 0
            arch.entries = pack_directory(pack_path)
            return arch^
        arch.addr = ptr
        arch.parse_mapped_directory()
        return arch^

    def parse_mapped_directory(mut self):
        if self.addr == 0 or self.size <= PACK_HEADER_BYTES:
            return
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=self.addr)
        var magic = UInt32(_ptr_u32(p, 0))
        var ver = UInt32(_ptr_u32(p, 4))
        if magic != PACK_MAGIC or (ver != 1 and ver != 2):
            return
        var count = _ptr_u32(p, 8)
        var dir_bytes = _ptr_u32(p, 12)
        if count < 0 or dir_bytes < 0 or PACK_HEADER_BYTES + dir_bytes > self.size:
            return
        var at = PACK_HEADER_BYTES
        for _ in range(count):
            if at + 20 > PACK_HEADER_BYTES + dir_bytes:
                self.entries.clear()
                return
            var length = _ptr_u32(p, at)
            if length < 0 or at + 20 + length > PACK_HEADER_BYTES + dir_bytes:
                self.entries.clear()
                return
            var name_bytes = List[UInt8](capacity=length)
            for k in range(length):
                name_bytes.append(p[unsafe_offset=at + 4 + k])
            var name_str = String(unsafe_from_utf8=name_bytes)
            var off_at = at + 4 + length
            var offset = _ptr_u64(p, off_at)
            var sz_at = off_at + 8
            var sz = _ptr_u64(p, sz_at)
            if offset < PACK_HEADER_BYTES + dir_bytes or sz < 0 or offset > self.size or sz > self.size - offset:
                self.entries.clear()
                return
            self.entries.append(PackEntry(name_str, offset, sz))
            at += 20 + length

    def read_entry(self, name: String) -> List[UInt8]:
        for i in range(len(self.entries)):
            if self.entries[i].path == name:
                var off = self.entries[i].offset
                var sz = self.entries[i].size
                if off < 0 or sz < 0 or off > self.size or sz > self.size - off:
                    return List[UInt8]()
                if sz == 0:
                    return List[UInt8]()
                if self.addr != 0:
                    var res = List[UInt8](capacity=sz)
                    res.resize(unsafe_uninit_length=sz)
                    _ = external_call["memcpy", Int](Int(res.unsafe_ptr()), self.addr + off, Int64(sz))
                    return res^
                else:
                    return read_at(self.path, off, sz)
        return List[UInt8]()


def pack_read(pack: String, name: String) -> List[UInt8]:
    """A file from the pack by its path, using persistent mmap zero-copy when available."""
    var arch = FlpkMappedArchive.open(pack)
    if arch.addr != 0:
        var res = arch.read_entry(name)
        arch.close()
        return res^
    for entry in pack_directory(pack):
        if entry.path == name:
            return read_at(pack, entry.offset, entry.size)
    return List[UInt8]()


def encode_pack(names: List[String], files: List[List[UInt8]]) -> List[UInt8]:
    """Encodes loose files into an .flpk v2 container with 64-byte aligned payload entries."""
    var directory = List[UInt8]()
    var offset = PACK_HEADER_BYTES
    for name in names:
        offset += 20 + name.byte_length()
    for index in range(len(names)):
        # 64-byte alignment for .flpk v2 DMA & SIMD streaming
        offset = (offset + 63) & ~63
        var bytes = names[index].as_bytes()
        put_u32(directory, len(bytes))
        for byte in bytes:
            directory.append(byte)
        put_u32(directory, offset & 0xFFFFFFFF)
        put_u32(directory, offset >> 32)
        put_u32(directory, len(files[index]) & 0xFFFFFFFF)
        put_u32(directory, len(files[index]) >> 32)
        offset += len(files[index])
    var out = List[UInt8]()
    put_u32(out, Int(PACK_MAGIC))
    put_u32(out, Int(PACK_VERSION))
    put_u32(out, len(names))
    put_u32(out, len(directory))
    out.extend(directory^)
    for file in files:
        while len(out) % 64 != 0:
            out.append(0)
        out.extend(file.copy())
    return out^


def put_u32(mut out: List[UInt8], value: Int):
    for shift in range(4):
        out.append(UInt8((value >> (shift * 8)) & 255))


def write_flpk_pack(output_pack: String, asset_paths: List[String]) -> Int:
    """Packs loose files at `asset_paths` into a single `.flpk` VFS archive at `output_pack`.

    Returns the number of non-empty files written, or 0 on failure.
    """
    var names = List[String]()
    var payloads = List[List[UInt8]]()
    for i in range(len(asset_paths)):
        var p = asset_paths[i]
        var data = read_bytes(p)
        if len(data) > 0:
            names.append(p)
            payloads.append(data^)
    if len(names) == 0:
        return 0
    var archive = encode_pack(names, payloads)
    if not write_bytes(output_pack, archive):
        return 0
    return len(names)


# ---------------------------------------------------------------------------
# Reading through the mounts
# ---------------------------------------------------------------------------

def vfs_read(path: String) -> List[UInt8]:
    """Any file: loose first, then the pack."""
    var data = read_bytes(path)
    if len(data) > 0:
        return data^
    return pack_read(pack_path(), path)


struct ModelFile(Movable):
    var model: NativeModel
    var info: GlbInfo
    var mount: Int  # MOUNT_COOKED, MOUNT_PACK or MOUNT_LOOSE; MOUNT_NONE if nothing loaded
    var note: String  # why a cook present was not used, for the log

    def __init__(out self):
        self.model = NativeModel()
        self.info = GlbInfo()
        self.mount = MOUNT_NONE
        self.note = String()

    def take_model(mut self) -> NativeModel:
        var model = self.model^
        self.model = NativeModel()
        return model^


def take_cooked(mut out: ModelFile, data: List[UInt8], path: String, mount: Int) -> Bool:
    if len(data) == 0:
        return False
    var stamp = read_cooked_stamp(data)
    if not stamp.valid:
        out.note = mount_name(mount) + " " + stamp.error
        return False
    if not stamp_is_current(stamp.stamp, path):
        out.note = mount_name(mount) + " stale: the source changed since it was cooked"
        return False
    var cooked = decode_cooked_model(data)
    if not cooked.valid:
        out.note = mount_name(mount) + " " + cooked.error
        return False
    out.model = cooked.take_model()
    out.info = stamp.info
    out.mount = mount
    return True


def load_model_file(path: String) -> ModelFile:
    """The model at `path` from the first mount that has a current copy."""
    var out = ModelFile()
    if cooked_enabled():
        if take_cooked(out, read_bytes(cooked_model_path(path)), path, MOUNT_COOKED):
            return out^
        if take_cooked(out, pack_read(pack_path(), path), path, MOUNT_PACK):
            return out^
    var bytes = read_bytes(path)
    if len(bytes) > 0:
        out.info = inspect_glb(path)
        out.model = parse_native_glb(bytes)
        out.mount = MOUNT_LOOSE
        return out^
    var packed_glb = pack_read(pack_path(), path)
    if len(packed_glb) > 0:
        out.model = parse_native_glb(packed_glb)
        out.mount = MOUNT_PACK
        return out^
    return out^


def load_model(path: String) -> NativeModel:
    """`load_native_glb` through the mounts: what the game and the tools call."""
    var file = load_model_file(path)
    return file.take_model()


def model_info(path: String) -> GlbInfo:
    """`inspect_glb` through the mounts: a current cook knows it without the glTF."""
    if cooked_enabled():
        var cooked = cooked_model_path(path)
        var head = read_at(cooked, 0, 96)
        if len(head) > 0:
            var stamp = read_cooked_stamp(head, file_stamp(cooked)[0])
            if stamp.valid and stamp_is_current(stamp.stamp, path):
                return stamp.info
        var packed = pack_read(pack_path(), path)
        if len(packed) > 0:
            var stamp = read_cooked_stamp(packed)
            if stamp.valid and stamp_is_current(stamp.stamp, path):
                return stamp.info
    return inspect_glb(path)
