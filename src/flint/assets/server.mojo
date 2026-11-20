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
from std.collections import Dict, Optional
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool
from flint.assets.cooker_v2 import (
    cook_hdr_texture_bundle,
    cook_png_texture_bundle,
    cook_single_source_bytes,
    cook_ttf_font_bundle,
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
    TEX_FMT_BC5_UNORM,
    TEX_FMT_BC6H_UFLOAT,
    TEX_FMT_BC7_SRGB,
    TEX_FMT_BC7_UNORM,
    AssetPayload,
    AssetUuid,
    FlpkV3MappedArchive,
    Handle,
    UntypedHandle,
    clone_geometry_pool,
    decode_audio_pcm_slab,
    decode_bvh8_collider_slab,
    decode_cooked_mesh_slab_into,
    decode_cooked_mesh_slab_ptr_into,
    decode_texture_tail_slab,
    load_or_init_flmeta,
    mip_to_sub_id,
)
from flint.assets.meshlet import AaaGeometryPool
from flint.assets.watcher import AssetWatcher
from flint.physics.bvh import Bvh8MeshCollider
from flint.platform.audio import decode_wav
from flint.platform.os import read_bytes
from flint.platform.vulkan import (
    FORMAT_BC5_UNORM,
    FORMAT_BC6H_UFLOAT,
    FORMAT_BC7_SRGB,
    FORMAT_BC7_UNORM,
    VulkanGpu,
)
from flint.render.aaa_render_graph import AaaResidencyStreamer, bc_format_mip_bytes
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.font import SdfFont


comptime ASSET_STATE_UNLOADED: Int = 0
comptime ASSET_STATE_LOADING: Int = 1
comptime ASSET_STATE_READY: Int = 2
comptime ASSET_STATE_FAILED: Int = 3


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

comptime AssetEvictCallback = def(Int, Int) thin -> None


struct TypedAssetStorage[T: AssetPayload](Movable):
    """Generational arena with O(1) stale-handle rejection, `Optional[T]` slot residency,
    strong ref-counts, immediate payload deinit on LRU eviction/unload, and hot-reload epochs."""
    var payloads: List[Optional[Self.T]]
    var occupied: List[Bool]
    var states: List[Int]
    var generations: List[UInt32]
    var ref_counts: List[Int]
    var reload_epochs: List[UInt64]
    var last_used_tick: List[UInt64]
    var uuids: List[AssetUuid]
    var rel_paths: List[String]
    var slot_cpu_bytes: List[Int]
    var slot_vram_bytes: List[Int]
    var free_slots: List[UInt32]
    var path_to_slot: Dict[String, Int]
    var uuid_to_slot: Dict[String, Int]
    var live_count: Int
    var cpu_live_bytes: Int
    var vram_live_bytes: Int
    var live_bytes: Int
    var evict_ctx: Int
    var evict_cb: Optional[AssetEvictCallback]

    def __init__(out self):
        self.payloads = List[Optional[Self.T]]()
        self.occupied = List[Bool]()
        self.states = List[Int]()
        self.generations = List[UInt32]()
        self.ref_counts = List[Int]()
        self.reload_epochs = List[UInt64]()
        self.last_used_tick = List[UInt64]()
        self.uuids = List[AssetUuid]()
        self.rel_paths = List[String]()
        self.slot_cpu_bytes = List[Int]()
        self.slot_vram_bytes = List[Int]()
        self.free_slots = List[UInt32]()
        self.path_to_slot = Dict[String, Int]()
        self.uuid_to_slot = Dict[String, Int]()
        self.live_count = 0
        self.cpu_live_bytes = 0
        self.vram_live_bytes = 0
        self.live_bytes = 0
        self.evict_ctx = 0
        self.evict_cb = None

    def set_evict_hook(mut self, ctx: Int, cb: AssetEvictCallback):
        self.evict_ctx = ctx
        self.evict_cb = Optional[AssetEvictCallback](cb)

    def clear_evict_hook(mut self):
        self.evict_ctx = 0
        self.evict_cb = None

    def set_slot_vram_bytes_at(mut self, s: Int, vram_bytes: Int) -> Bool:
        if s < 0 or s >= len(self.occupied) or not self.occupied[s]:
            return False
        var prev_vram = self.slot_vram_bytes[s]
        var next_vram = max(0, vram_bytes)
        self.slot_vram_bytes[s] = next_vram
        self.vram_live_bytes = max(0, self.vram_live_bytes - prev_vram + next_vram)
        self.live_bytes = self.cpu_live_bytes + self.vram_live_bytes
        return True

    def set_slot_vram_bytes(mut self, handle: Handle[Self.T], vram_bytes: Int) -> Bool:
        return self.set_slot_vram_bytes_at(self.valid_slot(handle), vram_bytes)

    @staticmethod
    @always_inline
    def _uuid_key(uuid: AssetUuid) -> String:
        return String(uuid.hi) + ":" + String(uuid.lo)

    def find_slot_by_path(self, rel_path: String) -> Int:
        var found = self.path_to_slot.get(rel_path)
        if found:
            var s = found.value()
            if s >= 0 and s < len(self.occupied) and self.occupied[s] and self.rel_paths[s] == rel_path:
                return s
        return -1

    def find_slot_by_uuid(self, uuid: AssetUuid) -> Int:
        if uuid.hi == 0 and uuid.lo == 0:
            return -1
        var found = self.uuid_to_slot.get(Self._uuid_key(uuid))
        if found:
            var s = found.value()
            if s >= 0 and s < len(self.occupied) and self.occupied[s] and self.uuids[s] == uuid:
                return s
        return -1

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
            self.states[slot] = ASSET_STATE_LOADING
            self.ref_counts[slot] = 1
            self.reload_epochs[slot] = 0
            self.last_used_tick[slot] = tick
            self.uuids[slot] = uuid
            self.rel_paths[slot] = rel_path
            self.slot_cpu_bytes[slot] = 0
            self.slot_vram_bytes[slot] = 0
            if rel_path.byte_length() > 0:
                self.path_to_slot[rel_path] = slot
            if uuid.hi != 0 or uuid.lo != 0:
                self.uuid_to_slot[Self._uuid_key(uuid)] = slot
            self.live_count += 1
            return Handle[Self.T](UInt32(slot), gen)

        var slot = len(self.payloads)
        var gen = UInt32(1)
        self.payloads.append(Optional[Self.T](None))
        self.occupied.append(True)
        self.states.append(ASSET_STATE_LOADING)
        self.generations.append(gen)
        self.ref_counts.append(1)
        self.reload_epochs.append(0)
        self.last_used_tick.append(tick)
        self.uuids.append(uuid)
        self.rel_paths.append(rel_path)
        self.slot_cpu_bytes.append(0)
        self.slot_vram_bytes.append(0)
        if rel_path.byte_length() > 0:
            self.path_to_slot[rel_path] = slot
        if uuid.hi != 0 or uuid.lo != 0:
            self.uuid_to_slot[Self._uuid_key(uuid)] = slot
        self.live_count += 1
        return Handle[Self.T](UInt32(slot), gen)

    def set_loaded(
        mut self,
        handle: Handle[Self.T],
        var payload: Self.T,
        tick: UInt64,
        vram_bytes: Int = -1,
    ) -> Bool:
        """Populates an allocated slot with `Optional[T](payload^)` and updates CPU/VRAM live byte counters."""
        var s = self.valid_slot(handle)
        if s < 0:
            return False
        var prev_cpu = self.slot_cpu_bytes[s]
        var prev_vram = self.slot_vram_bytes[s]
        var next_cpu = max(0, payload.byte_footprint())
        var next_vram = prev_vram if vram_bytes < 0 else max(0, vram_bytes)
        self.payloads[s] = Optional[Self.T](payload^)
        self.states[s] = ASSET_STATE_READY
        self.slot_cpu_bytes[s] = next_cpu
        self.slot_vram_bytes[s] = next_vram
        self.reload_epochs[s] += 1
        self.last_used_tick[s] = tick
        self.cpu_live_bytes = max(0, self.cpu_live_bytes - prev_cpu + next_cpu)
        self.vram_live_bytes = max(0, self.vram_live_bytes - prev_vram + next_vram)
        self.live_bytes = self.cpu_live_bytes + self.vram_live_bytes
        return True

    def set_failed(mut self, handle: Handle[Self.T]) -> Bool:
        """Marks an allocated slot as `ASSET_STATE_FAILED` when async loading/decoding fails."""
        var s = self.valid_slot(handle)
        if s < 0:
            return False
        self.payloads[s] = None
        self.states[s] = ASSET_STATE_FAILED
        return True

    def asset_state(self, handle: Handle[Self.T]) -> Int:
        var s = self.valid_slot(handle)
        return self.states[s] if s >= 0 else ASSET_STATE_UNLOADED

    def insert(
        mut self,
        var payload: Self.T,
        uuid: AssetUuid,
        rel_path: String,
        tick: UInt64,
        vram_bytes: Int = -1,
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
        vram_bytes: Int = -1,
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
        if self.evict_cb and self.evict_ctx != 0:
            self.evict_cb.value()(self.evict_ctx, s)
        var cpu_b = self.slot_cpu_bytes[s]
        var vram_b = self.slot_vram_bytes[s]
        var old_path = self.rel_paths[s]
        var old_uuid = self.uuids[s]
        if old_path.byte_length() > 0:
            var p_hit = self.path_to_slot.get(old_path)
            if p_hit and p_hit.value() == s:
                try:
                    _ = self.path_to_slot.pop(old_path)
                except:
                    pass
        if old_uuid.hi != 0 or old_uuid.lo != 0:
            var uk = Self._uuid_key(old_uuid)
            var u_hit = self.uuid_to_slot.get(uk)
            if u_hit and u_hit.value() == s:
                try:
                    _ = self.uuid_to_slot.pop(uk)
                except:
                    pass
        # Assigning None runs `T.__deinit__` immediately, freeing all underlying payload buffers!
        self.payloads[s] = None
        self.slot_cpu_bytes[s] = 0
        self.slot_vram_bytes[s] = 0
        self.occupied[s] = False
        self.states[s] = ASSET_STATE_UNLOADED
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

    def oldest_unreferenced(self) -> Tuple[Int, UInt64]:
        """Returns `(slot_idx, last_used_tick)` of the oldest occupied unreferenced slot, or `(-1, UINT64_MAX)`."""
        var oldest_s = -1
        var oldest_tick = UInt64(0xFFFFFFFFFFFFFFFF)
        for s in range(len(self.occupied)):
            if self.occupied[s] and self.ref_counts[s] <= 0:
                if self.last_used_tick[s] < oldest_tick:
                    oldest_tick = self.last_used_tick[s]
                    oldest_s = s
        return (oldest_s, oldest_tick)

    def evict_slot(mut self, s: Int) -> Int:
        if s < 0 or s >= len(self.occupied) or not self.occupied[s]:
            return 0
        var freed = self.slot_cpu_bytes[s] + self.slot_vram_bytes[s]
        self._evict_slot_index(s)
        return freed

    def evict_lru_unreferenced(mut self, target_max_bytes: Int) -> List[AssetUuid]:
        """Evicts `ref_count == 0` slots in LRU (`oldest last_used_tick`) order until `live_bytes <= target_max_bytes`."""
        var evicted = List[AssetUuid]()
        while self.live_bytes > target_max_bytes:
            var cand = self.oldest_unreferenced()
            if cand[0] < 0:
                break
            evicted.append(self.uuids[cand[0]])
            self._evict_slot_index(cand[0])
        return evicted^


# ============================================================================
# 3. `JobPool`-Backed Async Asset Load Jobs
# ============================================================================

struct _PendingMeshLoad(Movable):
    var handle: Handle[CookedMeshAsset]
    var rel_path: String
    var full_path: String
    var is_cooked_slab: Bool
    var slab_bytes: List[UInt8]
    var collider_bytes: List[UInt8]
    var has_source: Bool
    var ok: Bool
    var asset: CookedMeshAsset

    def __init__(out self, handle: Handle[CookedMeshAsset], rel_path: String):
        self.handle = handle
        self.rel_path = rel_path
        self.full_path = String()
        self.is_cooked_slab = False
        self.slab_bytes = List[UInt8]()
        self.collider_bytes = List[UInt8]()
        self.has_source = False
        self.ok = False
        self.asset = CookedMeshAsset()


struct _MeshAsyncBatchCtx(Movable):
    var jobs: List[_PendingMeshLoad]

    def __init__(out self, var jobs: List[_PendingMeshLoad]):
        self.jobs = jobs^

    @always_inline
    def keep_alive(self):
        var anchor = Int(self.jobs.unsafe_ptr())
        _ = anchor


def _mesh_async_worker(ctx_addr: Int, idx: Int):
    ref ctx = Pointer[_MeshAsyncBatchCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    ref job = ctx.jobs[idx]
    if not job.has_source:
        job.ok = False
        return
    if job.is_cooked_slab:
        var dec = decode_cooked_mesh_slab_into(job.slab_bytes, job.asset.geom)
        if not dec[0]:
            job.ok = False
            return
        if len(job.collider_bytes) > 0:
            var cdec = decode_bvh8_collider_slab(job.collider_bytes)
            if cdec[0]:
                job.asset.has_collider = True
                job.asset.collider = cdec[1].copy()
        job.ok = True
        return
    var raw = read_bytes(job.full_path)
    if len(raw) == 0:
        job.ok = False
        return
    var meta = load_or_init_flmeta(job.full_path, job.rel_path, ASSET_KIND_MESH, persist_if_missing=False)
    var b = cook_single_source_bytes(raw, meta, job.rel_path)
    if not b.valid or len(b.chunks) == 0:
        job.ok = False
        return
    var dec = decode_cooked_mesh_slab_into(b.chunks[0].raw_payload, job.asset.geom)
    if not dec[0]:
        job.ok = False
        return
    if len(b.chunks) > 1 and b.chunks[1].sub_id == SUB_ID_BVH8_COLLIDER:
        var cdec = decode_bvh8_collider_slab(b.chunks[1].raw_payload)
        job.asset.has_collider = cdec[0]
        job.asset.collider = cdec[1].copy()
    job.ok = True


struct _PendingTextureLoad(Movable):
    var handle: Handle[CookedTextureAsset]
    var rel_path: String
    var full_path: String
    var is_cooked_tail: Bool
    var progressive: Bool
    var tail_bytes: List[UInt8]
    var has_source: Bool
    var ok: Bool
    var asset: CookedTextureAsset

    def __init__(out self, handle: Handle[CookedTextureAsset], rel_path: String, progressive: Bool):
        self.handle = handle
        self.rel_path = rel_path
        self.full_path = String()
        self.is_cooked_tail = False
        self.progressive = progressive
        self.tail_bytes = List[UInt8]()
        self.has_source = False
        self.ok = False
        self.asset = CookedTextureAsset()


struct _TexAsyncBatchCtx(Movable):
    var jobs: List[_PendingTextureLoad]

    def __init__(out self, var jobs: List[_PendingTextureLoad]):
        self.jobs = jobs^

    @always_inline
    def keep_alive(self):
        var anchor = Int(self.jobs.unsafe_ptr())
        _ = anchor


def _tex_async_worker(ctx_addr: Int, idx: Int):
    ref ctx = Pointer[_TexAsyncBatchCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    ref job = ctx.jobs[idx]
    if not job.has_source:
        job.ok = False
        return
    if job.is_cooked_tail:
        var dec = decode_texture_tail_slab(job.tail_bytes)
        if not dec[0]:
            job.ok = False
            return
        var hdr = dec[1].copy()
        job.asset.width = Int(hdr.width)
        job.asset.height = Int(hdr.height)
        job.asset.mip_count = Int(hdr.mip_count)
        job.asset.tail_first_mip = Int(hdr.tail_first_mip)
        job.asset.resident_min_mip = Int(hdr.tail_first_mip)
        job.asset.tex_format = hdr.tex_format
        job.asset.mip_sizes = hdr.mip_sizes.copy()
        job.asset.mips = dec[2].copy()
        job.ok = True
        return
    var raw = read_bytes(job.full_path)
    if len(raw) == 0:
        job.ok = False
        return
    var meta = load_or_init_flmeta(job.full_path, job.rel_path, ASSET_KIND_TEXTURE, persist_if_missing=False)
    var cooked = cook_png_texture_bundle(raw, meta, job.rel_path)
    if not cooked.valid or len(cooked.chunks) == 0:
        job.ok = False
        return
    var dec = decode_texture_tail_slab(cooked.chunks[0].raw_payload)
    if not dec[0]:
        job.ok = False
        return
    var hdr = dec[1].copy()
    job.asset.width = Int(hdr.width)
    job.asset.height = Int(hdr.height)
    job.asset.mip_count = Int(hdr.mip_count)
    job.asset.tail_first_mip = Int(hdr.tail_first_mip)
    job.asset.resident_min_mip = 0
    job.asset.tex_format = hdr.tex_format
    job.asset.mip_sizes = hdr.mip_sizes.copy()
    job.asset.mips = dec[2].copy()
    for ci in range(1, len(cooked.chunks)):
        var sub_m = Int(cooked.chunks[ci].sub_id) - 1
        if sub_m >= 0 and sub_m < len(job.asset.mips):
            job.asset.mips[sub_m] = cooked.chunks[ci].raw_payload.copy()
    job.ok = True


# ============================================================================
# 4. Async `AssetServer`
# ============================================================================

def _asset_server_evict_texture_slot(ctx: Int, slot: Int) -> None:
    if ctx == 0:
        return
    ref srv = Pointer[AssetServer, MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0]
    if srv.renderer_ptr != 0 and slot >= 0 and slot < len(srv.streamer.slot_bindless_idx):
        var b_idx = srv.streamer.slot_bindless_idx[slot]
        if b_idx >= 0:
            ref ren = Pointer[FlintAaaVulkanRenderer, MutAnyOrigin](unsafe_from_address=srv.renderer_ptr)[unsafe_offset=0]
            if ren.vk.device != 0:
                ren.evict_bindless_texture_slot(b_idx)
            srv.streamer.slot_bindless_idx[slot] = -1
    _ = srv.streamer.evict_slot_gpu(slot)


@always_inline
def _tex_format_to_vk(tex_format: UInt8) -> UInt32:
    if tex_format == TEX_FMT_BC7_UNORM:
        return FORMAT_BC7_UNORM
    if tex_format == TEX_FMT_BC5_UNORM:
        return FORMAT_BC5_UNORM
    if tex_format == TEX_FMT_BC6H_UFLOAT:
        return FORMAT_BC6H_UFLOAT
    return FORMAT_BC7_SRGB


def _infer_queued_bc7_dims(byte_size: Int, mip_level: Int) -> Tuple[Int, Int, Int]:
    var eff_bytes = max(16, byte_size)
    var blocks = max(1, (eff_bytes + 15) // 16)
    var bw = 1
    while (bw * 2) * (bw * 2) <= blocks:
        bw *= 2
    var bh = max(1, (blocks + bw - 1) // bw)
    var mip_w = max(4, bw * 4)
    var mip_h = max(4, bh * 4)
    var shift = max(0, min(12, mip_level))
    return (max(4, mip_w << shift), max(4, mip_h << shift), max(1, shift + 1))


struct AssetServer(Movable):
    """Unified Flint v5 Asset Server managing persistent `.flpk` v3 mounts, generational handles,
    progressive byte-budgeted mip streaming, LRU eviction, and live recursive `inotify` hot-reload."""
    var mounts: List[FlpkV3MappedArchive]
    var loose_root: String
    var watcher: AssetWatcher
    var streamer: AaaResidencyStreamer
    var renderer_ptr: Int
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

    var pending_mesh_loads: List[_PendingMeshLoad]
    var pending_texture_loads: List[_PendingTextureLoad]

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
        self.renderer_ptr = 0
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

        self.pending_mesh_loads = List[_PendingMeshLoad]()
        self.pending_texture_loads = List[_PendingTextureLoad]()

    def __moveinit__(out self, deinit existing: Self):
        self.mounts = existing.mounts^
        self.loose_root = existing.loose_root^
        self.watcher = existing.watcher^
        self.streamer = existing.streamer^
        self.renderer_ptr = existing.renderer_ptr
        self.stream_budget_bytes_per_tick = existing.stream_budget_bytes_per_tick
        self.lru_max_bytes = existing.lru_max_bytes
        self.current_tick = existing.current_tick
        self.hot_reloads_applied = existing.hot_reloads_applied
        self.meshes = existing.meshes^
        self.textures = existing.textures^
        self.hdrs = existing.hdrs^
        self.fonts = existing.fonts^
        self.audios = existing.audios^
        self.raws = existing.raws^
        self.pending_mesh_loads = existing.pending_mesh_loads^
        self.pending_texture_loads = existing.pending_texture_loads^
        existing.renderer_ptr = 0
        if self.textures.evict_cb:
            self.textures.evict_ctx = Int(Pointer(to=self))

    def close(mut self):
        self.textures.clear_evict_hook()
        self.streamer.detach_vulkan()
        self.renderer_ptr = 0
        for i in range(len(self.mounts)):
            self.mounts[i].close()
        self.mounts.clear()

    def attach_vulkan_residency_bridge(mut self, mut vk: VulkanGpu):
        """Attaches a standalone `VulkanGpu` to `AaaResidencyStreamer` for real GPU mip uploads & VRAM tracking."""
        self.renderer_ptr = 0
        self.streamer.attach_vulkan(vk)
        self.textures.set_evict_hook(Int(Pointer(to=self)), _asset_server_evict_texture_slot)

    def attach_renderer_residency_bridge(mut self, mut ren: FlintAaaVulkanRenderer):
        """Attaches a live `FlintAaaVulkanRenderer` so streamed mips stage into bindless slots with `minLod` clamping."""
        self.renderer_ptr = Int(Pointer(to=ren))
        self.streamer.attach_vulkan(ren.vk)
        self.textures.set_evict_hook(Int(Pointer(to=self)), _asset_server_evict_texture_slot)

    def upload_mip_to_gpu(
        mut self,
        handle: Handle[CookedTextureAsset],
        mip_level: Int = 0,
        byte_size: Int = 0,
    ) -> Bool:
        """Uploads `mip_level` of `handle` (loaded or queued) to GPU VRAM and updates `slot_vram_bytes`."""
        var s = self.textures.valid_slot(handle)
        if s < 0:
            return False
        if self.renderer_ptr == 0 and self.streamer.vk_ptr == 0:
            return False
        self.textures.set_evict_hook(Int(Pointer(to=self)), _asset_server_evict_texture_slot)

        var width: Int
        var height: Int
        var mip_count: Int
        var vk_fmt = FORMAT_BC7_SRGB
        var data_ptr = 0
        var data_bytes = max(16, byte_size)
        var eff_mip = max(0, mip_level)

        if self.textures.payloads[s]:
            ref tex = self.textures.payloads[s].unsafe_value()
            width = max(1, tex.width)
            height = max(1, tex.height)
            mip_count = max(1, min(16, tex.mip_count))
            eff_mip = max(0, min(mip_count - 1, mip_level))
            vk_fmt = _tex_format_to_vk(tex.tex_format)
            if eff_mip < len(tex.mips) and len(tex.mips[eff_mip]) > 0:
                data_ptr = Int(tex.mips[eff_mip].unsafe_ptr())
                data_bytes = len(tex.mips[eff_mip])
            elif eff_mip < 16 and tex.mip_sizes[eff_mip] > 0:
                data_bytes = Int(tex.mip_sizes[eff_mip])
            if eff_mip < tex.resident_min_mip:
                tex.resident_min_mip = eff_mip
        else:
            var dims = _infer_queued_bc7_dims(byte_size, eff_mip)
            width = dims[0]
            height = dims[1]
            mip_count = dims[2]

        if self.renderer_ptr != 0:
            ref ren = Pointer[FlintAaaVulkanRenderer, MutAnyOrigin](unsafe_from_address=self.renderer_ptr)[unsafe_offset=0]
            if ren.vk.device == 0:
                return False
            self.streamer._ensure_slot(s)
            var b_idx = self.streamer.slot_bindless_idx[s]
            var need_alloc = b_idx < 0 or b_idx >= len(ren.bindless_images) or ren.bindless_images[b_idx].image == 0
            if not need_alloc:
                if (
                    ren.bindless_images[b_idx].width != width
                    or ren.bindless_images[b_idx].height != height
                    or ren.bindless_images[b_idx].format != vk_fmt
                    or ren.bindless_images[b_idx].mip_levels != mip_count
                ):
                    ren.evict_bindless_texture_slot(b_idx)
                    need_alloc = True
            if need_alloc:
                b_idx = ren.allocate_bindless_texture_slot(width, height, mip_count, vk_fmt)
                if b_idx < 0:
                    return False
                self.streamer.slot_bindless_idx[s] = b_idx
            var minfo = bc_format_mip_bytes(width, height, eff_mip, vk_fmt)
            var vram_b = ren.upload_bindless_mip_staged(
                b_idx,
                eff_mip,
                minfo[0],
                minfo[1],
                data_ptr,
                max(minfo[2], data_bytes),
            )
            if vram_b <= 0:
                return False
            var prev_sv = self.streamer.slot_vram_bytes[s]
            self.streamer.slot_vram_bytes[s] = vram_b
            self.streamer.vram_live_bytes = max(0, self.streamer.vram_live_bytes - prev_sv + vram_b)
            self.streamer.slot_mip_mask[s] |= UInt32(1) << UInt32(eff_mip)
            if eff_mip < self.streamer.resident_min_mip[s]:
                self.streamer.resident_min_mip[s] = eff_mip
            _ = self.textures.set_slot_vram_bytes_at(s, vram_b)
            return True

        var vram_b = self.streamer.upload_mip_to_slot(
            s,
            eff_mip,
            width,
            height,
            mip_count,
            vk_fmt,
            data_ptr,
            data_bytes,
        )
        if vram_b <= 0:
            return False
        _ = self.textures.set_slot_vram_bytes_at(s, vram_b)
        return True

    def _upload_resident_mips_to_gpu(mut self, handle: Handle[CookedTextureAsset]):
        if self.renderer_ptr == 0 and self.streamer.vk_ptr == 0:
            return
        var s = self.textures.valid_slot(handle)
        if s < 0 or not self.textures.payloads[s]:
            return
        var m_cnt = self.textures.payloads[s].unsafe_value().mip_count
        var res_min = self.textures.payloads[s].unsafe_value().resident_min_mip
        var m = m_cnt - 1
        while m >= max(0, res_min):
            if m < len(self.textures.payloads[s].unsafe_value().mips):
                var blen = len(self.textures.payloads[s].unsafe_value().mips[m])
                if blen > 0:
                    _ = self.upload_mip_to_gpu(handle, m, blen)
            m -= 1

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
        var s = self.meshes.find_slot_by_uuid(uuid)
        if s >= 0:
            var h = Handle[CookedMeshAsset](UInt32(s), self.meshes.generations[s])
            _ = self.meshes.retain(h)
            self.meshes.last_used_tick[s] = self.current_tick
            return h
        var loc = self._find_in_mounts_by_uuid(uuid, SUB_ID_PRIMARY_OR_TAIL)
        if loc[0] < 0:
            return Handle[CookedMeshAsset]()
        return self._decode_and_store_mesh_from_mount(loc[0], loc[1])

    def load_mesh(mut self, rel_path: String) -> Handle[CookedMeshAsset]:
        var s = self.meshes.find_slot_by_path(rel_path)
        if s >= 0:
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
                    var asset = CookedMeshAsset()
                    var dec = decode_cooked_mesh_slab_into(b.chunks[0].raw_payload, asset.geom)
                    if dec[0]:
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

        # Zero-copy mmap decode directly into `asset.geom` (zero `clone_geometry_pool` overhead!)
        var zc = self.mounts[mount_idx].mmap_slice_by_toc(toc_idx, verify_crc=True)
        if zc.valid:
            var dec_z = decode_cooked_mesh_slab_ptr_into(zc.addr, zc.size, asset.geom)
            if not dec_z[0]:
                return Handle[CookedMeshAsset]()
        else:
            var pay = self.mounts[mount_idx].read_payload_by_toc(toc_idx, verify_crc=True)
            if not pay[0]:
                return Handle[CookedMeshAsset]()
            var dec = decode_cooked_mesh_slab_into(pay[1], asset.geom)
            if not dec[0]:
                return Handle[CookedMeshAsset]()

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
        var s = self.textures.find_slot_by_path(rel_path)
        if s >= 0:
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
            self.streamer._ensure_slot(slot_i)
            self.streamer.resident_min_mip[slot_i] = Int(hdr.tail_first_mip)
            self.streamer.requested_min_mip[slot_i] = Int(hdr.tail_first_mip)
            self._upload_resident_mips_to_gpu(h)
            if progressive and Int(hdr.tail_first_mip) > 0:
                self.request_texture_mips(h, target_min_mip=0)
            else:
                # Eagerly load all high-res mips immediately
                for m in range(Int(hdr.tail_first_mip) - 1, -1, -1):
                    if self._stream_single_texture_mip(slot_i, m):
                        if self.renderer_ptr != 0 or self.streamer.vk_ptr != 0:
                            _ = self.upload_mip_to_gpu(h, m, 0)
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
                        var h = self.textures.insert(asset^, meta.uuid, rel_path, self.current_tick)
                        var slot_i = Int(h.slot())
                        self.streamer._ensure_slot(slot_i)
                        self.streamer.resident_min_mip[slot_i] = 0
                        self.streamer.requested_min_mip[slot_i] = 0
                        self._upload_resident_mips_to_gpu(h)
                        return h
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

    def load_hdr(mut self, rel_path: String) -> Handle[CookedHdrAsset]:
        var s = self.hdrs.find_slot_by_path(rel_path)
        if s >= 0:
            var h = Handle[CookedHdrAsset](UInt32(s), self.hdrs.generations[s])
            _ = self.hdrs.retain(h)
            self.hdrs.last_used_tick[s] = self.current_tick
            return h

        var loc = self._find_in_mounts_by_path(rel_path, SUB_ID_PRIMARY_OR_TAIL)
        if loc[0] >= 0:
            var entry = self.mounts[loc[0]].toc_entry(loc[1])
            var pay = self.mounts[loc[0]].read_payload_by_toc(loc[1], verify_crc=True)
            if not pay[0]:
                return Handle[CookedHdrAsset]()
            var dec = decode_texture_tail_slab(pay[1])
            if not dec[0]:
                return Handle[CookedHdrAsset]()
            var hdr = dec[1].copy()
            var asset = CookedHdrAsset()
            asset.width = Int(hdr.width)
            asset.height = Int(hdr.height)
            asset.mip_count = Int(hdr.mip_count)
            asset.tail_first_mip = Int(hdr.tail_first_mip)
            asset.resident_min_mip = Int(hdr.tail_first_mip)
            asset.mip_sizes = hdr.mip_sizes.copy()
            asset.mips = dec[2].copy()
            var m = Int(hdr.tail_first_mip) - 1
            while m >= 0:
                var mloc = self._find_in_mounts_by_uuid(entry.uuid, mip_to_sub_id(m))
                if mloc[0] < 0:
                    mloc = self._find_in_mounts_by_path(rel_path, mip_to_sub_id(m))
                if mloc[0] >= 0:
                    var mpay = self.mounts[mloc[0]].read_payload_by_toc(mloc[1], verify_crc=True)
                    if mpay[0] and m < len(asset.mips):
                        asset.mips[m] = mpay[1].copy()
                        asset.resident_min_mip = m
                m -= 1
            return self.hdrs.insert(asset^, entry.uuid, rel_path, self.current_tick)

        if self.loose_root.byte_length() > 0:
            var full = self.loose_root + "/" + rel_path
            var raw = read_bytes(full)
            if len(raw) > 0:
                var meta = load_or_init_flmeta(full, rel_path, ASSET_KIND_HDR)
                var cooked = cook_hdr_texture_bundle(raw, meta, rel_path)
                if cooked.valid and len(cooked.chunks) > 0:
                    var dec = decode_texture_tail_slab(cooked.chunks[0].raw_payload)
                    if dec[0]:
                        var hdr = dec[1].copy()
                        var asset = CookedHdrAsset()
                        asset.width = Int(hdr.width)
                        asset.height = Int(hdr.height)
                        asset.mip_count = Int(hdr.mip_count)
                        asset.tail_first_mip = Int(hdr.tail_first_mip)
                        asset.resident_min_mip = 0
                        asset.mip_sizes = hdr.mip_sizes.copy()
                        asset.mips = dec[2].copy()
                        for ci in range(1, len(cooked.chunks)):
                            var sub_m = Int(cooked.chunks[ci].sub_id) - 1
                            if sub_m >= 0 and sub_m < len(asset.mips):
                                asset.mips[sub_m] = cooked.chunks[ci].raw_payload.copy()
                        return self.hdrs.insert(asset^, meta.uuid, rel_path, self.current_tick)
        return Handle[CookedHdrAsset]()

    def load_font(mut self, rel_path: String) -> Handle[CookedFontAsset]:
        var s = self.fonts.find_slot_by_path(rel_path)
        if s >= 0:
            var h = Handle[CookedFontAsset](UInt32(s), self.fonts.generations[s])
            _ = self.fonts.retain(h)
            self.fonts.last_used_tick[s] = self.current_tick
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
        if self.loose_root.byte_length() > 0:
            var full = self.loose_root + "/" + rel_path
            var raw = read_bytes(full)
            if len(raw) > 0:
                var meta = load_or_init_flmeta(full, rel_path, ASSET_KIND_FONT)
                var fa = CookedFontAsset()
                if fa.font.load_bytes(raw):
                    fa.dsf_data = raw.copy()
                    return self.fonts.insert(fa^, meta.uuid, rel_path, self.current_tick)
                var cooked = cook_ttf_font_bundle(raw, meta, rel_path)
                if cooked.valid and len(cooked.chunks) > 0:
                    fa.dsf_data = cooked.chunks[0].raw_payload.copy()
                    if fa.font.load_bytes(fa.dsf_data):
                        return self.fonts.insert(fa^, meta.uuid, rel_path, self.current_tick)
        return Handle[CookedFontAsset]()

    def load_audio(mut self, rel_path: String) -> Handle[CookedAudioAsset]:
        var s = self.audios.find_slot_by_path(rel_path)
        if s >= 0:
            var h = Handle[CookedAudioAsset](UInt32(s), self.audios.generations[s])
            _ = self.audios.retain(h)
            self.audios.last_used_tick[s] = self.current_tick
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
        if self.loose_root.byte_length() > 0:
            var full = self.loose_root + "/" + rel_path
            var raw = read_bytes(full)
            if len(raw) > 0:
                var meta = load_or_init_flmeta(full, rel_path, ASSET_KIND_AUDIO)
                var dec = decode_audio_pcm_slab(raw)
                if dec[0]:
                    var aa = CookedAudioAsset()
                    aa.channels = dec[1]
                    aa.sample_rate = dec[2]
                    aa.samples = dec[3].copy()
                    return self.audios.insert(aa^, meta.uuid, rel_path, self.current_tick)
                var clip = decode_wav(raw)
                if clip.valid:
                    var aa = CookedAudioAsset()
                    aa.channels = clip.channels
                    aa.sample_rate = clip.sample_rate
                    aa.samples = clip.samples.copy()
                    return self.audios.insert(aa^, meta.uuid, rel_path, self.current_tick)
        return Handle[CookedAudioAsset]()

    def load_raw(mut self, rel_path: String) -> Handle[RawBytesAsset]:
        var s = self.raws.find_slot_by_path(rel_path)
        if s >= 0:
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

    # ---- Async `JobPool`-Backed Loading API ----

    def mesh_state(self, handle: Handle[CookedMeshAsset]) -> Int:
        return self.meshes.asset_state(handle)

    def texture_state(self, handle: Handle[CookedTextureAsset]) -> Int:
        return self.textures.asset_state(handle)

    def pending_async_count(self) -> Int:
        return len(self.pending_mesh_loads) + len(self.pending_texture_loads)

    def load_mesh_async(mut self, rel_path: String) -> Handle[CookedMeshAsset]:
        """Queues a mesh for `JobPool`-backed async decoding, immediately returning a handle in `ASSET_STATE_LOADING`."""
        var s = self.meshes.find_slot_by_path(rel_path)
        if s >= 0:
            var h = Handle[CookedMeshAsset](UInt32(s), self.meshes.generations[s])
            _ = self.meshes.retain(h)
            self.meshes.last_used_tick[s] = self.current_tick
            return h

        var loc = self._find_in_mounts_by_path(rel_path, SUB_ID_PRIMARY_OR_TAIL)
        if loc[0] >= 0:
            var entry = self.mounts[loc[0]].toc_entry(loc[1])
            var h = self.meshes.allocate_queued(entry.uuid, rel_path, self.current_tick)
            var job = _PendingMeshLoad(h, rel_path)
            var pay = self.mounts[loc[0]].read_payload_by_toc(loc[1], verify_crc=True)
            if pay[0]:
                job.is_cooked_slab = True
                job.slab_bytes = pay[1].copy()
                job.has_source = True
                var col_loc = self._find_in_mounts_by_uuid(entry.uuid, SUB_ID_BVH8_COLLIDER)
                if col_loc[0] >= 0:
                    var cpay = self.mounts[col_loc[0]].read_payload_by_toc(col_loc[1], verify_crc=True)
                    if cpay[0]:
                        job.collider_bytes = cpay[1].copy()
            self.pending_mesh_loads.append(job^)
            return h

        var h = self.meshes.allocate_queued(AssetUuid.from_path(rel_path), rel_path, self.current_tick)
        var job = _PendingMeshLoad(h, rel_path)
        if self.loose_root.byte_length() > 0:
            job.full_path = self.loose_root + "/" + rel_path
            job.has_source = True
        self.pending_mesh_loads.append(job^)
        return h

    def load_texture_async(mut self, rel_path: String, progressive: Bool = True) -> Handle[CookedTextureAsset]:
        """Queues a texture for `JobPool`-backed async cooking/decoding, immediately returning a handle in `ASSET_STATE_LOADING`."""
        var s = self.textures.find_slot_by_path(rel_path)
        if s >= 0:
            var h = Handle[CookedTextureAsset](UInt32(s), self.textures.generations[s])
            _ = self.textures.retain(h)
            self.textures.last_used_tick[s] = self.current_tick
            return h

        var loc = self._find_in_mounts_by_path(rel_path, SUB_ID_PRIMARY_OR_TAIL)
        if loc[0] >= 0:
            var entry = self.mounts[loc[0]].toc_entry(loc[1])
            var h = self.textures.allocate_queued(entry.uuid, rel_path, self.current_tick)
            var job = _PendingTextureLoad(h, rel_path, progressive)
            var pay = self.mounts[loc[0]].read_payload_by_toc(loc[1], verify_crc=True)
            if pay[0]:
                job.is_cooked_tail = True
                job.tail_bytes = pay[1].copy()
                job.has_source = True
            self.pending_texture_loads.append(job^)
            return h

        var h = self.textures.allocate_queued(AssetUuid.from_path(rel_path), rel_path, self.current_tick)
        var job = _PendingTextureLoad(h, rel_path, progressive)
        if self.loose_root.byte_length() > 0:
            job.full_path = self.loose_root + "/" + rel_path
            job.has_source = True
        self.pending_texture_loads.append(job^)
        return h

    def poll_async_loads(mut self, mut pool: JobPool) -> Int:
        """Executes all queued async mesh & texture loads across `pool` worker threads and commits
        each handle to `ASSET_STATE_READY` or `ASSET_STATE_FAILED`. Returns completed job count."""
        var completed = 0

        if len(self.pending_mesh_loads) > 0:
            var batch = List[_PendingMeshLoad]()
            while len(self.pending_mesh_loads) > 0:
                batch.append(self.pending_mesh_loads.pop())
            var mctx = _MeshAsyncBatchCtx(batch^)
            var n_m = len(mctx.jobs)
            if pool.allocated() and pool.workers > 0 and n_m > 1:
                run_parallel_ctx(pool, n_m, mctx, _mesh_async_worker)
                mctx.keep_alive()
            else:
                for i in range(n_m):
                    _mesh_async_worker(Int(Pointer(to=mctx)), i)
                mctx.keep_alive()
            for i in range(n_m):
                var h = mctx.jobs[i].handle
                if mctx.jobs[i].ok:
                    var a = mctx.jobs[i].asset.copy()
                    _ = self.meshes.set_loaded(h, a^, self.current_tick)
                else:
                    _ = self.meshes.set_failed(h)
                completed += 1

        if len(self.pending_texture_loads) > 0:
            var tbatch = List[_PendingTextureLoad]()
            while len(self.pending_texture_loads) > 0:
                tbatch.append(self.pending_texture_loads.pop())
            var tctx = _TexAsyncBatchCtx(tbatch^)
            var n_t = len(tctx.jobs)
            if pool.allocated() and pool.workers > 0 and n_t > 1:
                run_parallel_ctx(pool, n_t, tctx, _tex_async_worker)
                tctx.keep_alive()
            else:
                for i in range(n_t):
                    _tex_async_worker(Int(Pointer(to=tctx)), i)
                tctx.keep_alive()
            for i in range(n_t):
                var h = tctx.jobs[i].handle
                if tctx.jobs[i].ok:
                    var tail_mip = tctx.jobs[i].asset.tail_first_mip
                    var is_cooked = tctx.jobs[i].is_cooked_tail
                    var prog = tctx.jobs[i].progressive
                    var a = tctx.jobs[i].asset.copy()
                    _ = self.textures.set_loaded(h, a^, self.current_tick)
                    var slot_i = Int(h.slot())
                    self.streamer._ensure_slot(slot_i)
                    if is_cooked:
                        self.streamer.resident_min_mip[slot_i] = tail_mip
                        self.streamer.requested_min_mip[slot_i] = tail_mip
                        self._upload_resident_mips_to_gpu(h)
                        if prog and tail_mip > 0:
                            self.request_texture_mips(h, target_min_mip=0)
                        else:
                            for m in range(tail_mip - 1, -1, -1):
                                if self._stream_single_texture_mip(slot_i, m):
                                    if self.renderer_ptr != 0 or self.streamer.vk_ptr != 0:
                                        _ = self.upload_mip_to_gpu(h, m, 0)
                    else:
                        self.streamer.resident_min_mip[slot_i] = 0
                        self.streamer.requested_min_mip[slot_i] = 0
                        self._upload_resident_mips_to_gpu(h)
                else:
                    _ = self.textures.set_failed(h)
                completed += 1

        return completed

    # ---- Per-Frame Tick: Progressive Mip Streaming + Live `inotify` Hot-Reload ----

    def _hot_reload_path(mut self, rel_path: String) -> Bool:
        if self.loose_root.byte_length() == 0:
            return False
        var full = self.loose_root + "/" + rel_path
        var raw = read_bytes(full)
        if len(raw) == 0:
            return False

        # Check textures (O(1) lookup)
        var ts = self.textures.find_slot_by_path(rel_path)
        if ts >= 0:
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
                    var h = Handle[CookedTextureAsset](UInt32(ts), self.textures.generations[ts])
                    if self.textures.replace_in_place(h, asset^, self.current_tick):
                        self._upload_resident_mips_to_gpu(h)
                        self.hot_reloads_applied += 1
                        return True

        # Check meshes (O(1) lookup)
        var ms = self.meshes.find_slot_by_path(rel_path)
        if ms >= 0:
            var meta = load_or_init_flmeta(full, rel_path, ASSET_KIND_MESH)
            var b = cook_single_source_bytes(raw, meta, rel_path)
            if b.valid and len(b.chunks) > 0:
                var asset = CookedMeshAsset()
                var dec = decode_cooked_mesh_slab_into(b.chunks[0].raw_payload, asset.geom)
                if dec[0]:
                    if len(b.chunks) > 1 and b.chunks[1].sub_id == SUB_ID_BVH8_COLLIDER:
                        var cdec = decode_bvh8_collider_slab(b.chunks[1].raw_payload)
                        asset.has_collider = cdec[0]
                        asset.collider = cdec[1].copy()
                    var h = Handle[CookedMeshAsset](UInt32(ms), self.meshes.generations[ms])
                    if self.meshes.replace_in_place(h, asset^, self.current_tick):
                        self.hot_reloads_applied += 1
                        return True

        # Check HDRs (O(1) lookup)
        var hs = self.hdrs.find_slot_by_path(rel_path)
        if hs >= 0:
            var meta = load_or_init_flmeta(full, rel_path, ASSET_KIND_HDR)
            var cooked = cook_hdr_texture_bundle(raw, meta, rel_path)
            if cooked.valid and len(cooked.chunks) > 0:
                var dec = decode_texture_tail_slab(cooked.chunks[0].raw_payload)
                if dec[0]:
                    var hdr = dec[1].copy()
                    var asset = CookedHdrAsset()
                    asset.width = Int(hdr.width)
                    asset.height = Int(hdr.height)
                    asset.mip_count = Int(hdr.mip_count)
                    asset.tail_first_mip = Int(hdr.tail_first_mip)
                    asset.resident_min_mip = 0
                    asset.mip_sizes = hdr.mip_sizes.copy()
                    asset.mips = dec[2].copy()
                    for ci in range(1, len(cooked.chunks)):
                        var sm = Int(cooked.chunks[ci].sub_id) - 1
                        if sm >= 0 and sm < len(asset.mips):
                            asset.mips[sm] = cooked.chunks[ci].raw_payload.copy()
                    var h = Handle[CookedHdrAsset](UInt32(hs), self.hdrs.generations[hs])
                    if self.hdrs.replace_in_place(h, asset^, self.current_tick):
                        self.hot_reloads_applied += 1
                        return True

        # Check fonts (O(1) lookup)
        var fs = self.fonts.find_slot_by_path(rel_path)
        if fs >= 0:
            var fa = CookedFontAsset()
            var loaded_ok = fa.font.load_bytes(raw)
            if loaded_ok:
                fa.dsf_data = raw.copy()
            else:
                var meta = load_or_init_flmeta(full, rel_path, ASSET_KIND_FONT)
                var cooked = cook_ttf_font_bundle(raw, meta, rel_path)
                if cooked.valid and len(cooked.chunks) > 0:
                    fa.dsf_data = cooked.chunks[0].raw_payload.copy()
                    loaded_ok = fa.font.load_bytes(fa.dsf_data)
            if loaded_ok:
                var h = Handle[CookedFontAsset](UInt32(fs), self.fonts.generations[fs])
                if self.fonts.replace_in_place(h, fa^, self.current_tick):
                    self.hot_reloads_applied += 1
                    return True

        # Check audios (O(1) lookup)
        var aus = self.audios.find_slot_by_path(rel_path)
        if aus >= 0:
            var aa = CookedAudioAsset()
            var ok_aud = False
            var dec = decode_audio_pcm_slab(raw)
            if dec[0]:
                aa.channels = dec[1]
                aa.sample_rate = dec[2]
                aa.samples = dec[3].copy()
                ok_aud = True
            else:
                var clip = decode_wav(raw)
                if clip.valid:
                    aa.channels = clip.channels
                    aa.sample_rate = clip.sample_rate
                    aa.samples = clip.samples.copy()
                    ok_aud = True
            if ok_aud:
                var h = Handle[CookedAudioAsset](UInt32(aus), self.audios.generations[aus])
                if self.audios.replace_in_place(h, aa^, self.current_tick):
                    self.hot_reloads_applied += 1
                    return True

        # Check raw bytes (O(1) lookup)
        var rs = self.raws.find_slot_by_path(rel_path)
        if rs >= 0:
            var h = Handle[RawBytesAsset](UInt32(rs), self.raws.generations[rs])
            if self.raws.replace_in_place(h, RawBytesAsset(raw.copy()), self.current_tick):
                self.hot_reloads_applied += 1
                return True

        return False

    def enforce_lru_budget(mut self) -> Int:
        """Evicts globally oldest unreferenced (`ref_count == 0`) assets across ALL 6 pools
        until `total_resident_bytes() <= lru_max_bytes`. Returns count of evicted assets."""
        var evicted_count = 0
        while self.total_resident_bytes() > self.lru_max_bytes:
            var best_pool = -1
            var best_slot = -1
            var best_tick = UInt64(0xFFFFFFFFFFFFFFFF)

            var c0 = self.meshes.oldest_unreferenced()
            if c0[0] >= 0 and c0[1] < best_tick:
                best_pool = 0; best_slot = c0[0]; best_tick = c0[1]
            var c1 = self.textures.oldest_unreferenced()
            if c1[0] >= 0 and c1[1] < best_tick:
                best_pool = 1; best_slot = c1[0]; best_tick = c1[1]
            var c2 = self.hdrs.oldest_unreferenced()
            if c2[0] >= 0 and c2[1] < best_tick:
                best_pool = 2; best_slot = c2[0]; best_tick = c2[1]
            var c3 = self.fonts.oldest_unreferenced()
            if c3[0] >= 0 and c3[1] < best_tick:
                best_pool = 3; best_slot = c3[0]; best_tick = c3[1]
            var c4 = self.audios.oldest_unreferenced()
            if c4[0] >= 0 and c4[1] < best_tick:
                best_pool = 4; best_slot = c4[0]; best_tick = c4[1]
            var c5 = self.raws.oldest_unreferenced()
            if c5[0] >= 0 and c5[1] < best_tick:
                best_pool = 5; best_slot = c5[0]

            if best_pool < 0:
                break
            if best_pool == 0:
                _ = self.meshes.evict_slot(best_slot)
            elif best_pool == 1:
                _ = self.textures.evict_slot(best_slot)
            elif best_pool == 2:
                _ = self.hdrs.evict_slot(best_slot)
            elif best_pool == 3:
                _ = self.fonts.evict_slot(best_slot)
            elif best_pool == 4:
                _ = self.audios.evict_slot(best_slot)
            else:
                _ = self.raws.evict_slot(best_slot)
            evicted_count += 1
        return evicted_count

    def tick(mut self, now: Float64 = -1.0) -> Int:
        """Advances progressive mip/LOD streaming within `stream_budget_bytes_per_tick`, polls
        `AssetWatcher` for live hot-reloads, and enforces global cross-pool LRU memory budget."""
        self.current_tick += 1
        var work_done = 0

        # 1. Drain byte-budgeted progressive mip uploads
        var drained = self.streamer.drain_uploads_within_budget(self.stream_budget_bytes_per_tick)
        for i in range(len(drained)):
            var u = drained[i]
            if not u.is_mesh_lod:
                if self._stream_single_texture_mip(u.target_id, u.sub_level):
                    if (
                        (self.renderer_ptr != 0 or self.streamer.vk_ptr != 0)
                        and u.target_id >= 0
                        and u.target_id < len(self.textures.generations)
                    ):
                        var th = Handle[CookedTextureAsset](UInt32(u.target_id), self.textures.generations[u.target_id])
                        _ = self.upload_mip_to_gpu(th, u.sub_level, u.byte_size)
                    work_done += 1

        # 2. Poll recursive `inotify` AssetWatcher for live hot-reload
        var changed = self.watcher.poll_ready(now)
        for rel_p in changed:
            if self._hot_reload_path(rel_p):
                work_done += 1

        # 3. Enforce global cross-pool LRU memory ceiling across all 6 pools
        _ = self.enforce_lru_budget()

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
