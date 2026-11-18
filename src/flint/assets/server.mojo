"""Flint v5 Async `AssetServer`: Persistent `.flpk` v3 Mounts, Generational Arenas, Budgeted Mip Streaming & Live Hot-Reload (`server.mojo`).

Architecture:
  - **Persistent `FlpkV3MappedArchive` Mounts**: Zero per-read `mmap`/`munmap` churn across mounted `.flpk` v3 packs.
  - **`TypedAssetStorage[T: AssetPayload]`**: Generational arenas (`Handle[T]`) rejecting stale handles in O(1),
    strong reference counting (`retain` / `release`), and tick-ordered LRU eviction of unreferenced slots.
  - **Progressive Byte-Budgeted Mip/LOD Streaming**: Backed by `AaaResidencyStreamer` (`stream_budget_bytes_per_tick`),
    loading `sub_id = 0` mip tails (`mip 4..N`) immediately with zero hitch and streaming high-res mips (`4 -> 3 -> 2 -> 1 -> 0`)
    within strict per-tick byte budgets.
  - **Live Recursive `AssetWatcher` Hot-Reload**: Watches loose workspace directories (including newly created subdirectories)
    and reloads modified `.png` / `.glb` / `.hdr` / `.ttf` / `.wav` / raw assets in-place under the exact same `Handle[T]`
    while bumping `reload_epoch++`.
"""
from std.collections import Optional
from flint.assets.cooker_v2 import (
    cook_hdr_texture_bundle,
    cook_png_texture_bundle,
    cook_single_source_bytes,
)
from flint.assets.flpk_v3 import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_FONT,
    ASSET_KIND_HDR,
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    ASSET_KIND_TEXTURE,
    SUB_ID_BVH8_COLLIDER,
    SUB_ID_PRIMARY_OR_TAIL,
    TEX_FMT_BC7_SRGB,
    AssetPayload,
    AssetUuid,
    FlpkV3MappedArchive,
    Handle,
    UntypedHandle,
    clone_geometry_pool,
    decode_audio_pcm_slab,
    decode_bvh8_collider_slab,
    decode_cooked_mesh_slab,
    decode_cooked_mesh_slab_ptr,
    decode_texture_tail_slab,
    load_or_init_flmeta,
    mip_to_sub_id,
)
from flint.assets.meshlet import AaaGeometryPool
from flint.assets.watcher import AssetWatcher
from flint.physics.bvh import Bvh8MeshCollider
from flint.platform.os import read_bytes
from flint.render.aaa_render_graph import AaaResidencyStreamer
from flint.render.font import SdfFont


# ============================================================================
# 1. Concrete `AssetPayload` Types
# ============================================================================

struct RawBytesAsset(AssetPayload):
    var bytes: List[UInt8]

    def __init__(out self):
        self.bytes = List[UInt8]()

    def __init__(out self, var bytes: List[UInt8]):
        self.bytes = bytes^

    @staticmethod
    def asset_kind() -> UInt8:
        return ASSET_KIND_RAW

    def byte_footprint(self) -> Int:
        return len(self.bytes)


struct CookedMeshAsset(AssetPayload):
    var geom: AaaGeometryPool
    var has_collider: Bool
    var collider: Bvh8MeshCollider
    var resident_lod: Int

    def __init__(out self):
        self.geom = AaaGeometryPool()
        self.has_collider = False
        self.collider = Bvh8MeshCollider()
        self.resident_lod = 0

    def __init__(out self, *, copy: Self):
        self.geom = clone_geometry_pool(copy.geom)
        self.has_collider = copy.has_collider
        self.collider = copy.collider.copy()
        self.resident_lod = copy.resident_lod

    @staticmethod
    def asset_kind() -> UInt8:
        return ASSET_KIND_MESH

    def byte_footprint(self) -> Int:
        return (
            (len(self.geom.vertex_words) + len(self.geom.index_words) + len(self.geom.meshlet_words)) * 4
            + len(self.collider.triangles) * 52
            + len(self.collider.nodes) * 256
        )


struct CookedTextureAsset(AssetPayload):
    var width: Int
    var height: Int
    var mip_count: Int
    var tail_first_mip: Int
    var resident_min_mip: Int
    var tex_format: UInt8
    var mip_sizes: Array[UInt32, 16]
    var mips: List[List[UInt8]]

    def __init__(out self):
        self.width = 0
        self.height = 0
        self.mip_count = 0
        self.tail_first_mip = 0
        self.resident_min_mip = 0
        self.tex_format = TEX_FMT_BC7_SRGB
        self.mip_sizes = Array[UInt32, 16](fill=0)
        self.mips = List[List[UInt8]]()

    @staticmethod
    def asset_kind() -> UInt8:
        return ASSET_KIND_TEXTURE

    def byte_footprint(self) -> Int:
        var sum = 0
        for i in range(len(self.mips)):
            sum += len(self.mips[i])
        return max(64, sum)


struct CookedHdrAsset(AssetPayload):
    var width: Int
    var height: Int
    var mip_count: Int
    var tail_first_mip: Int
    var resident_min_mip: Int
    var mip_sizes: Array[UInt32, 16]
    var mips: List[List[UInt8]]

    def __init__(out self):
        self.width = 0
        self.height = 0
        self.mip_count = 0
        self.tail_first_mip = 0
        self.resident_min_mip = 0
        self.mip_sizes = Array[UInt32, 16](fill=0)
        self.mips = List[List[UInt8]]()

    @staticmethod
    def asset_kind() -> UInt8:
        return ASSET_KIND_HDR

    def byte_footprint(self) -> Int:
        var sum = 0
        for i in range(len(self.mips)):
            sum += len(self.mips[i])
        return max(64, sum)


struct CookedFontAsset(AssetPayload):
    var font: SdfFont
    var dsf_data: List[UInt8]

    def __init__(out self):
        self.font = SdfFont()
        self.dsf_data = List[UInt8]()

    @staticmethod
    def asset_kind() -> UInt8:
        return ASSET_KIND_FONT

    def byte_footprint(self) -> Int:
        return len(self.dsf_data) + len(self.font.atlas)


struct CookedAudioAsset(AssetPayload):
    var channels: Int
    var sample_rate: Int
    var samples: List[Int16]

    def __init__(out self):
        self.channels = 0
        self.sample_rate = 0
        self.samples = List[Int16]()

    @staticmethod
    def asset_kind() -> UInt8:
        return ASSET_KIND_AUDIO

    def byte_footprint(self) -> Int:
        return len(self.samples) * 2


# ============================================================================
# 2. Generational Ref-Counted `TypedAssetStorage[T: AssetPayload]`
# ============================================================================

struct TypedAssetStorage[T: AssetPayload](Movable):
    """Generational arena with O(1) stale-handle rejection, `Optional[T]` slot residency,
    strong ref-counts, immediate payload deinit on LRU eviction/unload, and hot-reload epochs."""
    var payloads: List[Optional[Self.T]]
    var occupied: List[Bool]
    var generations: List[UInt32]
    var ref_counts: List[Int]
    var reload_epochs: List[UInt64]
    var last_used_tick: List[UInt64]
    var uuids: List[AssetUuid]
    var rel_paths: List[String]
    var slot_cpu_bytes: List[Int]
    var slot_vram_bytes: List[Int]
    var free_slots: List[UInt32]
    var live_count: Int
    var cpu_live_bytes: Int
    var vram_live_bytes: Int
    var live_bytes: Int

    def __init__(out self):
        self.payloads = List[Optional[Self.T]]()
        self.occupied = List[Bool]()
        self.generations = List[UInt32]()
        self.ref_counts = List[Int]()
        self.reload_epochs = List[UInt64]()
        self.last_used_tick = List[UInt64]()
        self.uuids = List[AssetUuid]()
        self.rel_paths = List[String]()
        self.slot_cpu_bytes = List[Int]()
        self.slot_vram_bytes = List[Int]()
        self.free_slots = List[UInt32]()
        self.live_count = 0
        self.cpu_live_bytes = 0
        self.vram_live_bytes = 0
        self.live_bytes = 0

    def allocate_queued(
        mut self,
        uuid: AssetUuid,
        rel_path: String,
        tick: UInt64,
    ) -> Handle[Self.T]:
        """Allocates a generational slot in queued (`None`) state prior to payload decoding."""
        if len(self.free_slots) > 0:
            var slot = Int(self.free_slots.pop())
            var gen = self.generations[slot]
            self.payloads[slot] = None
            self.occupied[slot] = True
            self.ref_counts[slot] = 1
            self.reload_epochs[slot] = 0
            self.last_used_tick[slot] = tick
            self.uuids[slot] = uuid
            self.rel_paths[slot] = rel_path
            self.slot_cpu_bytes[slot] = 0
            self.slot_vram_bytes[slot] = 0
            self.live_count += 1
            return Handle[Self.T](UInt32(slot), gen)

        var slot = len(self.payloads)
        var gen = UInt32(1)
        self.payloads.append(Optional[Self.T](None))
        self.occupied.append(True)
        self.generations.append(gen)
        self.ref_counts.append(1)
        self.reload_epochs.append(0)
        self.last_used_tick.append(tick)
        self.uuids.append(uuid)
        self.rel_paths.append(rel_path)
        self.slot_cpu_bytes.append(0)
        self.slot_vram_bytes.append(0)
        self.live_count += 1
        return Handle[Self.T](UInt32(slot), gen)

    def set_loaded(
        mut self,
        handle: Handle[Self.T],
        var payload: Self.T,
        tick: UInt64,
        vram_bytes: Int = 0,
    ) -> Bool:
        """Populates an allocated slot with `Optional[T](payload^)` and updates CPU/VRAM live byte counters."""
        var s = self.valid_slot(handle)
        if s < 0:
            return False
        var prev_cpu = self.slot_cpu_bytes[s]
        var prev_vram = self.slot_vram_bytes[s]
        var next_cpu = max(0, payload.byte_footprint())
        var next_vram = max(0, vram_bytes)
        self.payloads[s] = Optional[Self.T](payload^)
        self.slot_cpu_bytes[s] = next_cpu
        self.slot_vram_bytes[s] = next_vram
        self.reload_epochs[s] += 1
        self.last_used_tick[s] = tick
        self.cpu_live_bytes = max(0, self.cpu_live_bytes - prev_cpu + next_cpu)
        self.vram_live_bytes = max(0, self.vram_live_bytes - prev_vram + next_vram)
        self.live_bytes = self.cpu_live_bytes + self.vram_live_bytes
        return True

    def insert(
        mut self,
        var payload: Self.T,
        uuid: AssetUuid,
        rel_path: String,
        tick: UInt64,
        vram_bytes: Int = 0,
    ) -> Handle[Self.T]:
        var h = self.allocate_queued(uuid, rel_path, tick)
        _ = self.set_loaded(h, payload^, tick, vram_bytes)
        return h

    @always_inline
    def valid_slot(self, handle: Handle[Self.T]) -> Int:
        if not handle.is_valid():
            return -1
        var s = Int(handle.slot())
        if s < 0 or s >= len(self.occupied) or not self.occupied[s]:
            return -1
        if self.generations[s] != handle.generation():
            return -1
        return s

    def is_alive(self, handle: Handle[Self.T]) -> Bool:
        return self.valid_slot(handle) >= 0

    def is_loaded(self, handle: Handle[Self.T]) -> Bool:
        var s = self.valid_slot(handle)
        return s >= 0 and Bool(self.payloads[s])

    def get_ptr(self, handle: Handle[Self.T]) -> Optional[Pointer[Self.T, ImmutAnyOrigin]]:
        var s = self.valid_slot(handle)
        if s < 0 or not self.payloads[s]:
            return None
        return Pointer[Self.T, ImmutAnyOrigin](
            unsafe_from_address=Int(Pointer(to=self.payloads[s].unsafe_value()))
        )

    def get_ptr_mut(mut self, handle: Handle[Self.T], tick: UInt64 = 0) -> Optional[Pointer[Self.T, MutAnyOrigin]]:
        var s = self.valid_slot(handle)
        if s < 0 or not self.payloads[s]:
            return None
        if tick > 0:
            self.last_used_tick[s] = tick
        return Pointer[Self.T, MutAnyOrigin](
            unsafe_from_address=Int(Pointer(to=self.payloads[s].unsafe_value()))
        )

    def borrow(self, handle: Handle[Self.T]) raises -> Pointer[Self.T, ImmutAnyOrigin]:
        var ptr = self.get_ptr(handle)
        if not ptr:
            raise Error("AssetPayload handle is invalid, stale, or unloaded")
        return ptr.value()

    def touch_and_copy(mut self, handle: Handle[Self.T], tick: UInt64) -> Optional[Self.T]:
        var s = self.valid_slot(handle)
        if s < 0 or not self.payloads[s]:
            return None
        self.last_used_tick[s] = tick
        return Optional[Self.T](self.payloads[s].unsafe_value().copy())

    def retain(mut self, handle: Handle[Self.T]) -> Bool:
        var s = self.valid_slot(handle)
        if s < 0:
            return False
        self.ref_counts[s] += 1
        return True

    def release(mut self, handle: Handle[Self.T]) -> Bool:
        var s = self.valid_slot(handle)
        if s < 0:
            return False
        if self.ref_counts[s] > 0:
            self.ref_counts[s] -= 1
        return True

    def ref_count(self, handle: Handle[Self.T]) -> Int:
        var s = self.valid_slot(handle)
        return self.ref_counts[s] if s >= 0 else -1

    def reload_epoch(self, handle: Handle[Self.T]) -> UInt64:
        var s = self.valid_slot(handle)
        return self.reload_epochs[s] if s >= 0 else 0

    def replace_in_place(
        mut self,
        handle: Handle[Self.T],
        var new_payload: Self.T,
        tick: UInt64,
        vram_bytes: Int = 0,
    ) -> Bool:
        return self.set_loaded(handle, new_payload^, tick, vram_bytes)

    def unload(mut self, handle: Handle[Self.T]) -> Bool:
        var s = self.valid_slot(handle)
        if s < 0:
            return False
        self._evict_slot_index(s)
        return True

    def remove(mut self, handle: Handle[Self.T]) -> Bool:
        return self.unload(handle)

    def _evict_slot_index(mut self, s: Int):
        var cpu_b = self.slot_cpu_bytes[s]
        var vram_b = self.slot_vram_bytes[s]
        # Assigning None runs `T.__deinit__` immediately, freeing all underlying payload buffers!
        self.payloads[s] = None
        self.slot_cpu_bytes[s] = 0
        self.slot_vram_bytes[s] = 0
        self.occupied[s] = False
        self.ref_counts[s] = 0
        var next_gen = (self.generations[s] + 1) & 0xFFFFFF
        if next_gen == 0:
            next_gen = 1
        self.generations[s] = next_gen
        self.uuids[s] = AssetUuid()
        self.rel_paths[s] = String()
        self.free_slots.append(UInt32(s))
        self.live_count = max(0, self.live_count - 1)
        self.cpu_live_bytes = max(0, self.cpu_live_bytes - cpu_b)
        self.vram_live_bytes = max(0, self.vram_live_bytes - vram_b)
        self.live_bytes = self.cpu_live_bytes + self.vram_live_bytes

    def evict_lru_unreferenced(mut self, target_max_bytes: Int) -> List[AssetUuid]:
        """Evicts `ref_count == 0` slots in LRU (`oldest last_used_tick`) order until `live_bytes <= target_max_bytes`."""
        var evicted = List[AssetUuid]()
        while self.live_bytes > target_max_bytes:
            var oldest_s = -1
            var oldest_tick = UInt64(0xFFFFFFFFFFFFFFFF)
            for s in range(len(self.occupied)):
                if self.occupied[s] and self.ref_counts[s] <= 0:
                    if self.last_used_tick[s] < oldest_tick:
                        oldest_tick = self.last_used_tick[s]
                        oldest_s = s
            if oldest_s < 0:
                break
            evicted.append(self.uuids[oldest_s])
            self._evict_slot_index(oldest_s)
        return evicted^


# ============================================================================
# 3. Async `AssetServer`
# ============================================================================

struct AssetServer(Movable):
    """Unified Flint v5 Asset Server managing persistent `.flpk` v3 mounts, generational handles,
    progressive byte-budgeted mip streaming, LRU eviction, and live recursive `inotify` hot-reload."""
    var mounts: List[FlpkV3MappedArchive]
    var loose_root: String
    var watcher: AssetWatcher
    var streamer: AaaResidencyStreamer
    var stream_budget_bytes_per_tick: Int
    var lru_max_bytes: Int
    var current_tick: UInt64
    var hot_reloads_applied: Int

    var meshes: TypedAssetStorage[CookedMeshAsset]
    var textures: TypedAssetStorage[CookedTextureAsset]
    var hdrs: TypedAssetStorage[CookedHdrAsset]
    var fonts: TypedAssetStorage[CookedFontAsset]
    var audios: TypedAssetStorage[CookedAudioAsset]
    var raws: TypedAssetStorage[RawBytesAsset]

    def __init__(
        out self,
        stream_budget_bytes_per_tick: Int = 256 * 1024,
        lru_max_bytes: Int = 512 * 1024 * 1024,
        enable_watcher: Bool = True,
    ):
        self.mounts = List[FlpkV3MappedArchive]()
        self.loose_root = String()
        self.watcher = AssetWatcher(enabled=enable_watcher)
        self.streamer = AaaResidencyStreamer(max_textures=1024, initial_tail_mip=4)
        self.stream_budget_bytes_per_tick = max(256, stream_budget_bytes_per_tick)
        self.lru_max_bytes = max(65536, lru_max_bytes)
        self.current_tick = 1
        self.hot_reloads_applied = 0

        self.meshes = TypedAssetStorage[CookedMeshAsset]()
        self.textures = TypedAssetStorage[CookedTextureAsset]()
        self.hdrs = TypedAssetStorage[CookedHdrAsset]()
        self.fonts = TypedAssetStorage[CookedFontAsset]()
        self.audios = TypedAssetStorage[CookedAudioAsset]()
        self.raws = TypedAssetStorage[RawBytesAsset]()

    def close(mut self):
        for i in range(len(self.mounts)):
            self.mounts[i].close()
        self.mounts.clear()

    def mount_flpk(mut self, pack_path: String) -> Bool:
        """Mounts a `.flpk` v3 archive persistently via `mmap` (zero per-read `mmap`/`munmap` churn!)."""
        for i in range(len(self.mounts)):
            if self.mounts[i].path == pack_path and self.mounts[i].valid:
                return True
        var arch = FlpkV3MappedArchive.open(pack_path)
        if not arch.valid:
            return False
        self.mounts.append(arch^)
        return True

    def watch_loose_directory(mut self, root_dir: String) -> Int:
        """Registers `root_dir` for loose-asset fallback and recursive `inotify` live hot-reload."""
        var clean = root_dir
        while clean.byte_length() > 1 and clean.as_bytes()[clean.byte_length() - 1] == 47:
            var trimmed = String(clean[byte=0 : clean.byte_length() - 1])
            clean = trimmed^
        self.loose_root = clean
        return self.watcher.watch_recursive(clean)

    def _find_in_mounts_by_uuid(self, uuid: AssetUuid, sub_id: UInt16 = 0) -> Tuple[Int, Int]:
        var m_idx = len(self.mounts) - 1
        while m_idx >= 0:
            var toc = self.mounts[m_idx].find_by_uuid(uuid, sub_id)
            if toc >= 0:
                return (m_idx, toc)
            m_idx -= 1
        return (-1, -1)

    def _find_in_mounts_by_path(self, rel_path: String, sub_id: UInt16 = 0) -> Tuple[Int, Int]:
        var m_idx = len(self.mounts) - 1
        while m_idx >= 0:
            var toc = self.mounts[m_idx].find_by_path(rel_path, sub_id)
            if toc >= 0:
                return (m_idx, toc)
            m_idx -= 1
        return (-1, -1)

    # ---- Mesh Loading ----

    def load_mesh_by_uuid(mut self, uuid: AssetUuid) -> Handle[CookedMeshAsset]:
        for s in range(len(self.meshes.occupied)):
            if self.meshes.occupied[s] and self.meshes.uuids[s] == uuid:
                var h = Handle[CookedMeshAsset](UInt32(s), self.meshes.generations[s])
                _ = self.meshes.retain(h)
                self.meshes.last_used_tick[s] = self.current_tick
                return h
        var loc = self._find_in_mounts_by_uuid(uuid, SUB_ID_PRIMARY_OR_TAIL)
        if loc[0] < 0:
            return Handle[CookedMeshAsset]()
        return self._decode_and_store_mesh_from_mount(loc[0], loc[1])

    def load_mesh(mut self, rel_path: String) -> Handle[CookedMeshAsset]:
        for s in range(len(self.meshes.occupied)):
            if self.meshes.occupied[s] and self.meshes.rel_paths[s] == rel_path:
                var h = Handle[CookedMeshAsset](UInt32(s), self.meshes.generations[s])
                _ = self.meshes.retain(h)
                self.meshes.last_used_tick[s] = self.current_tick
                return h

        var loc = self._find_in_mounts_by_path(rel_path, SUB_ID_PRIMARY_OR_TAIL)
        if loc[0] >= 0:
            return self._decode_and_store_mesh_from_mount(loc[0], loc[1])

        if self.loose_root.byte_length() > 0:
            var full = self.loose_root + "/" + rel_path
            var raw = read_bytes(full)
            if len(raw) > 0:
                var meta = load_or_init_flmeta(full, rel_path, ASSET_KIND_MESH)
                var b = cook_single_source_bytes(raw, meta, rel_path)
                if b.valid and len(b.chunks) > 0:
                    var dec = decode_cooked_mesh_slab(b.chunks[0].raw_payload)
                    if dec[0]:
                        var asset = CookedMeshAsset()
                        asset.geom = clone_geometry_pool(dec[2])
                        if len(b.chunks) > 1 and b.chunks[1].sub_id == SUB_ID_BVH8_COLLIDER:
                            var cdec = decode_bvh8_collider_slab(b.chunks[1].raw_payload)
                            asset.has_collider = cdec[0]
                            asset.collider = cdec[1].copy()
                        return self.meshes.insert(asset^, meta.uuid, rel_path, self.current_tick)
        return Handle[CookedMeshAsset]()

    def _decode_and_store_mesh_from_mount(mut self, mount_idx: Int, toc_idx: Int) -> Handle[CookedMeshAsset]:
        var entry = self.mounts[mount_idx].toc_entry(toc_idx)
        var rpath = self.mounts[mount_idx].entry_path(toc_idx)
        var asset = CookedMeshAsset()

        # Zero-copy mmap decode when CHUNK_COMP_NONE!
        var zc = self.mounts[mount_idx].mmap_slice_by_toc(toc_idx, verify_crc=True)
        if zc.valid:
            var dec_z = decode_cooked_mesh_slab_ptr(zc.addr, zc.size)
            if not dec_z[0]:
                return Handle[CookedMeshAsset]()
            asset.geom = clone_geometry_pool(dec_z[2])
        else:
            var pay = self.mounts[mount_idx].read_payload_by_toc(toc_idx, verify_crc=True)
            if not pay[0]:
                return Handle[CookedMeshAsset]()
            var dec = decode_cooked_mesh_slab(pay[1])
            if not dec[0]:
                return Handle[CookedMeshAsset]()
            asset.geom = clone_geometry_pool(dec[2])

        var col_loc = self._find_in_mounts_by_uuid(entry.uuid, SUB_ID_BVH8_COLLIDER)
        if col_loc[0] >= 0:
            var cpay = self.mounts[col_loc[0]].read_payload_by_toc(col_loc[1], verify_crc=True)
            if cpay[0]:
                var cdec = decode_bvh8_collider_slab(cpay[1])
                if cdec[0]:
                    asset.has_collider = True
                    asset.collider = cdec[1].copy()

        return self.meshes.insert(asset^, entry.uuid, rpath, self.current_tick)

    # ---- Texture Loading & Progressive Mip Streaming ----

    def load_texture(mut self, rel_path: String, progressive: Bool = True) -> Handle[CookedTextureAsset]:
        for s in range(len(self.textures.occupied)):
            if self.textures.occupied[s] and self.textures.rel_paths[s] == rel_path:
                var h = Handle[CookedTextureAsset](UInt32(s), self.textures.generations[s])
                _ = self.textures.retain(h)
                self.textures.last_used_tick[s] = self.current_tick
                return h

        var loc = self._find_in_mounts_by_path(rel_path, SUB_ID_PRIMARY_OR_TAIL)
        if loc[0] >= 0:
            var entry = self.mounts[loc[0]].toc_entry(loc[1])
            var pay = self.mounts[loc[0]].read_payload_by_toc(loc[1], verify_crc=True)
            if not pay[0]:
                return Handle[CookedTextureAsset]()
            var dec = decode_texture_tail_slab(pay[1])
            if not dec[0]:
                return Handle[CookedTextureAsset]()
            var hdr = dec[1].copy()
            var asset = CookedTextureAsset()
            asset.width = Int(hdr.width)
            asset.height = Int(hdr.height)
            asset.mip_count = Int(hdr.mip_count)
            asset.tail_first_mip = Int(hdr.tail_first_mip)
            asset.resident_min_mip = Int(hdr.tail_first_mip)
            asset.tex_format = hdr.tex_format
            asset.mip_sizes = hdr.mip_sizes.copy()
            asset.mips = dec[2].copy()

            var h = self.textures.insert(asset^, entry.uuid, rel_path, self.current_tick)
            var slot_i = Int(h.slot())
            if slot_i >= 0 and slot_i < len(self.streamer.resident_min_mip):
                self.streamer.resident_min_mip[slot_i] = Int(hdr.tail_first_mip)
                self.streamer.requested_min_mip[slot_i] = Int(hdr.tail_first_mip)
            if progressive and Int(hdr.tail_first_mip) > 0:
                self.request_texture_mips(h, target_min_mip=0)
            else:
                # Eagerly load all high-res mips immediately
                for m in range(Int(hdr.tail_first_mip) - 1, -1, -1):
                    _ = self._stream_single_texture_mip(slot_i, m)
            return h

        if self.loose_root.byte_length() > 0:
            var full = self.loose_root + "/" + rel_path
            var raw = read_bytes(full)
            if len(raw) > 0:
                var meta = load_or_init_flmeta(full, rel_path, ASSET_KIND_TEXTURE)
                var cooked = cook_png_texture_bundle(raw, meta, rel_path)
                if cooked.valid and len(cooked.chunks) > 0:
                    var dec = decode_texture_tail_slab(cooked.chunks[0].raw_payload)
                    if dec[0]:
                        var hdr = dec[1].copy()
                        var asset = CookedTextureAsset()
                        asset.width = Int(hdr.width)
                        asset.height = Int(hdr.height)
                        asset.mip_count = Int(hdr.mip_count)
                        asset.tail_first_mip = Int(hdr.tail_first_mip)
                        asset.resident_min_mip = 0
                        asset.tex_format = hdr.tex_format
                        asset.mip_sizes = hdr.mip_sizes.copy()
                        asset.mips = dec[2].copy()
                        for ci in range(1, len(cooked.chunks)):
                            var sub_m = Int(cooked.chunks[ci].sub_id) - 1
                            if sub_m >= 0 and sub_m < len(asset.mips):
                                asset.mips[sub_m] = cooked.chunks[ci].raw_payload.copy()
                        return self.textures.insert(asset^, meta.uuid, rel_path, self.current_tick)
        return Handle[CookedTextureAsset]()

    def request_texture_mips(mut self, handle: Handle[CookedTextureAsset], target_min_mip: Int = 0, priority: Int = 0):
        """Queues coarse-to-fine high-res mip sub-chunks (`resident_min_mip - 1 .. target_min_mip`) into `AaaResidencyStreamer`."""
        var s = self.textures.valid_slot(handle)
        if s < 0 or not self.textures.payloads[s]:
            return
        ref tex = self.textures.payloads[s].unsafe_value()
        var cur_min = tex.resident_min_mip
        var tgt = max(0, min(cur_min, target_min_mip))
        var m = cur_min - 1
        while m >= tgt:
            var bsz = max(16, Int(tex.mip_sizes[m]))
            self.streamer.enqueue_mip_upload(s, m, bsz, priority=priority, is_mesh_lod=False)
            m -= 1

    def _stream_single_texture_mip(mut self, slot_idx: Int, mip_level: Int) -> Bool:
        if (
            slot_idx < 0
            or slot_idx >= len(self.textures.occupied)
            or not self.textures.occupied[slot_idx]
            or not self.textures.payloads[slot_idx]
        ):
            return False
        var uuid = self.textures.uuids[slot_idx]
        var loc = self._find_in_mounts_by_uuid(uuid, mip_to_sub_id(mip_level))
        if loc[0] < 0:
            var rpath = self.textures.rel_paths[slot_idx]
            loc = self._find_in_mounts_by_path(rpath, mip_to_sub_id(mip_level))
            if loc[0] < 0:
                return False
        var pay = self.mounts[loc[0]].read_payload_by_toc(loc[1], verify_crc=True)
        if not pay[0]:
            return False
        ref tex = self.textures.payloads[slot_idx].unsafe_value()
        if mip_level >= 0 and mip_level < len(tex.mips):
            var prev_len = len(tex.mips[mip_level])
            tex.mips[mip_level] = pay[1].copy()
            var delta = len(tex.mips[mip_level]) - prev_len
            self.textures.slot_cpu_bytes[slot_idx] = max(0, self.textures.slot_cpu_bytes[slot_idx] + delta)
            self.textures.cpu_live_bytes = max(0, self.textures.cpu_live_bytes + delta)
            self.textures.live_bytes = self.textures.cpu_live_bytes + self.textures.vram_live_bytes
            if mip_level < tex.resident_min_mip:
                tex.resident_min_mip = mip_level
            return True
        return False

    # ---- Font, Audio, HDR & Raw Loading ----

    def load_font(mut self, rel_path: String) -> Handle[CookedFontAsset]:
        for s in range(len(self.fonts.occupied)):
            if self.fonts.occupied[s] and self.fonts.rel_paths[s] == rel_path:
                var h = Handle[CookedFontAsset](UInt32(s), self.fonts.generations[s])
                _ = self.fonts.retain(h)
                return h
        var loc = self._find_in_mounts_by_path(rel_path, SUB_ID_PRIMARY_OR_TAIL)
        if loc[0] >= 0:
            var e = self.mounts[loc[0]].toc_entry(loc[1])
            var pay = self.mounts[loc[0]].read_payload_by_toc(loc[1], verify_crc=True)
            if pay[0]:
                var fa = CookedFontAsset()
                fa.dsf_data = pay[1].copy()
                _ = fa.font.load_bytes(fa.dsf_data)
                return self.fonts.insert(fa^, e.uuid, rel_path, self.current_tick)
        return Handle[CookedFontAsset]()

    def load_audio(mut self, rel_path: String) -> Handle[CookedAudioAsset]:
        for s in range(len(self.audios.occupied)):
            if self.audios.occupied[s] and self.audios.rel_paths[s] == rel_path:
                var h = Handle[CookedAudioAsset](UInt32(s), self.audios.generations[s])
                _ = self.audios.retain(h)
                return h
        var loc = self._find_in_mounts_by_path(rel_path, SUB_ID_PRIMARY_OR_TAIL)
        if loc[0] >= 0:
            var e = self.mounts[loc[0]].toc_entry(loc[1])
            var pay = self.mounts[loc[0]].read_payload_by_toc(loc[1], verify_crc=True)
            if pay[0]:
                var dec = decode_audio_pcm_slab(pay[1])
                if dec[0]:
                    var aa = CookedAudioAsset()
                    aa.channels = dec[1]
                    aa.sample_rate = dec[2]
                    aa.samples = dec[3].copy()
                    return self.audios.insert(aa^, e.uuid, rel_path, self.current_tick)
        return Handle[CookedAudioAsset]()

    def load_raw(mut self, rel_path: String) -> Handle[RawBytesAsset]:
        for s in range(len(self.raws.occupied)):
            if self.raws.occupied[s] and self.raws.rel_paths[s] == rel_path:
                var h = Handle[RawBytesAsset](UInt32(s), self.raws.generations[s])
                _ = self.raws.retain(h)
                self.raws.last_used_tick[s] = self.current_tick
                return h
        var loc = self._find_in_mounts_by_path(rel_path, SUB_ID_PRIMARY_OR_TAIL)
        if loc[0] >= 0:
            var e = self.mounts[loc[0]].toc_entry(loc[1])
            var pay = self.mounts[loc[0]].read_payload_by_toc(loc[1], verify_crc=True)
            if pay[0]:
                return self.raws.insert(RawBytesAsset(pay[1].copy()), e.uuid, rel_path, self.current_tick)
        if self.loose_root.byte_length() > 0:
            var full = self.loose_root + "/" + rel_path
            var raw = read_bytes(full)
            if len(raw) > 0:
                return self.raws.insert(RawBytesAsset(raw^), AssetUuid.from_path(rel_path), rel_path, self.current_tick)
        return Handle[RawBytesAsset]()

    # ---- Per-Frame Tick: Progressive Mip Streaming + Live `inotify` Hot-Reload ----

    def _hot_reload_path(mut self, rel_path: String) -> Bool:
        if self.loose_root.byte_length() == 0:
            return False
        var full = self.loose_root + "/" + rel_path
        var raw = read_bytes(full)
        if len(raw) == 0:
            return False

        # Check textures
        for s in range(len(self.textures.occupied)):
            if self.textures.occupied[s] and self.textures.rel_paths[s] == rel_path:
                var meta = load_or_init_flmeta(full, rel_path, ASSET_KIND_TEXTURE)
                var cooked = cook_png_texture_bundle(raw, meta, rel_path)
                if cooked.valid and len(cooked.chunks) > 0:
                    var dec = decode_texture_tail_slab(cooked.chunks[0].raw_payload)
                    if dec[0]:
                        var hdr = dec[1].copy()
                        var asset = CookedTextureAsset()
                        asset.width = Int(hdr.width)
                        asset.height = Int(hdr.height)
                        asset.mip_count = Int(hdr.mip_count)
                        asset.tail_first_mip = Int(hdr.tail_first_mip)
                        asset.resident_min_mip = 0
                        asset.tex_format = hdr.tex_format
                        asset.mip_sizes = hdr.mip_sizes.copy()
                        asset.mips = dec[2].copy()
                        for ci in range(1, len(cooked.chunks)):
                            var sm = Int(cooked.chunks[ci].sub_id) - 1
                            if sm >= 0 and sm < len(asset.mips):
                                asset.mips[sm] = cooked.chunks[ci].raw_payload.copy()
                        var h = Handle[CookedTextureAsset](UInt32(s), self.textures.generations[s])
                        if self.textures.replace_in_place(h, asset^, self.current_tick):
                            self.hot_reloads_applied += 1
                            return True

        # Check meshes
        for s in range(len(self.meshes.occupied)):
            if self.meshes.occupied[s] and self.meshes.rel_paths[s] == rel_path:
                var meta = load_or_init_flmeta(full, rel_path, ASSET_KIND_MESH)
                var b = cook_single_source_bytes(raw, meta, rel_path)
                if b.valid and len(b.chunks) > 0:
                    var dec = decode_cooked_mesh_slab(b.chunks[0].raw_payload)
                    if dec[0]:
                        var asset = CookedMeshAsset()
                        asset.geom = clone_geometry_pool(dec[2])
                        if len(b.chunks) > 1 and b.chunks[1].sub_id == SUB_ID_BVH8_COLLIDER:
                            var cdec = decode_bvh8_collider_slab(b.chunks[1].raw_payload)
                            asset.has_collider = cdec[0]
                            asset.collider = cdec[1].copy()
                        var h = Handle[CookedMeshAsset](UInt32(s), self.meshes.generations[s])
                        if self.meshes.replace_in_place(h, asset^, self.current_tick):
                            self.hot_reloads_applied += 1
                            return True

        # Check raw bytes
        for s in range(len(self.raws.occupied)):
            if self.raws.occupied[s] and self.raws.rel_paths[s] == rel_path:
                var h = Handle[RawBytesAsset](UInt32(s), self.raws.generations[s])
                if self.raws.replace_in_place(h, RawBytesAsset(raw.copy()), self.current_tick):
                    self.hot_reloads_applied += 1
                    return True

        return False

    def tick(mut self, now: Float64 = -1.0) -> Int:
        """Advances progressive mip/LOD streaming within `stream_budget_bytes_per_tick`, polls
        `AssetWatcher` for live hot-reloads, and enforces LRU memory budget. Returns chunks/reloads processed."""
        self.current_tick += 1
        var work_done = 0

        # 1. Drain byte-budgeted progressive mip uploads
        var drained = self.streamer.drain_uploads_within_budget(self.stream_budget_bytes_per_tick)
        for i in range(len(drained)):
            var u = drained[i]
            if not u.is_mesh_lod:
                if self._stream_single_texture_mip(u.target_id, u.sub_level):
                    work_done += 1

        # 2. Poll recursive `inotify` AssetWatcher for live hot-reload
        var changed = self.watcher.poll_ready(now)
        for rel_p in changed:
            if self._hot_reload_path(rel_p):
                work_done += 1

        # 3. Enforce LRU memory ceiling across unreferenced assets
        var total_live = (
            self.meshes.live_bytes
            + self.textures.live_bytes
            + self.hdrs.live_bytes
            + self.fonts.live_bytes
            + self.audios.live_bytes
            + self.raws.live_bytes
        )
        if total_live > self.lru_max_bytes:
            _ = self.textures.evict_lru_unreferenced(self.lru_max_bytes // 2)
            _ = self.meshes.evict_lru_unreferenced(self.lru_max_bytes // 2)
            _ = self.raws.evict_lru_unreferenced(self.lru_max_bytes // 2)

        return work_done

    def total_cpu_live_bytes(self) -> Int:
        return (
            self.meshes.cpu_live_bytes
            + self.textures.cpu_live_bytes
            + self.hdrs.cpu_live_bytes
            + self.fonts.cpu_live_bytes
            + self.audios.cpu_live_bytes
            + self.raws.cpu_live_bytes
        )

    def total_vram_live_bytes(self) -> Int:
        return (
            self.meshes.vram_live_bytes
            + self.textures.vram_live_bytes
            + self.hdrs.vram_live_bytes
            + self.fonts.vram_live_bytes
            + self.audios.vram_live_bytes
            + self.raws.vram_live_bytes
        )

    def total_resident_bytes(self) -> Int:
        return (
            self.meshes.live_bytes
            + self.textures.live_bytes
            + self.hdrs.live_bytes
            + self.fonts.live_bytes
            + self.audios.live_bytes
            + self.raws.live_bytes
        )
