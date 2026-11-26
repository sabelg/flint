"""Area E (`sota:E`): robust on-disk Vulkan pipeline cache keyed by device identity and SPIR-V hashes.

The engine used to load/save `build/vk_pso.cache` relative to the working directory: silently a no-op
when there is no `build/` dir, never validated (a blob from another driver / GPU / a torn write went
straight into `vkCreatePipelineCache`), and written in place (a crash mid-write left a torn file).

This module replaces that with:

* a stable location: `$FLINT_PSO_CACHE_DIR`, else `$XDG_CACHE_HOME/flint`, else `$HOME/.cache/flint`,
  else `build/` (`FLINT_PSO_CACHE=0` disables the cache, the setting this feature hangs off);
* a Flint container `FLPSOE01` carrying the device identity (vendor, device, driver version and the
  driver's `pipelineCacheUUID`), the payload length and a 64-bit content hash, and a manifest of the
  SPIR-V module hashes compiled when it was written. A blob is only handed to the driver when all of
  those match *and* the payload's own Vulkan header (`VkPipelineCacheHeaderVersionOne`) names the
  same device; anything else starts cold with a recorded reason;
* atomic replacement on save (`write_file_durable`: temp file + fsync + rename + dir fsync);
* warm-start accounting: every SPIR-V module created this run is hashed (`VulkanGpu.sota_e_spirv_hashes`)
  and compared with the manifest, so `hits / misses` tell how much of the compile was served warm.

The Flint-side SPIR-V *emission* cache from the brief is deliberately not built: emitting every module
costs ~4 ms in total (deferred shade 3.6 ms / 81k words, all others < 0.2 ms) against ~3 s of driver
compile, so the win lives entirely in the driver cache.
"""
from std.collections import List
from flint.core.lz4 import hash_list
from flint.platform.os import env_value, file_size, make_dirs, read_bytes, write_file_durable
from flint.platform.vulkan import VulkanGpu, call2, call4, get64, put32, put64, TYPE_PIPELINE_CACHE_CREATE_INFO

comptime SOTA_E_PSO_MAGIC: String = "FLPSOE01"
comptime SOTA_E_PSO_HEADER_BYTES: Int = 56
comptime SOTA_E_PSO_FILE: String = "vk_pso.flpso"

# `cache_status`
comptime SOTA_E_PSO_DISABLED: Int = 0
comptime SOTA_E_PSO_COLD_MISSING: Int = 1
comptime SOTA_E_PSO_COLD_REJECTED: Int = 2
comptime SOTA_E_PSO_WARM: Int = 3


@fieldwise_init
struct SotaEDeviceIdentity(Copyable, Movable):
    var vendor: UInt32
    var device: UInt32
    var driver_version: UInt32
    var uuid: List[UInt8]   # 16 bytes `pipelineCacheUUID`

    def matches(self, other: SotaEDeviceIdentity) -> Bool:
        if self.vendor != other.vendor or self.device != other.device or self.driver_version != other.driver_version:
            return False
        if len(self.uuid) != 16 or len(other.uuid) != 16:
            return False
        for i in range(16):
            if self.uuid[i] != other.uuid[i]:
                return False
        return True


struct SotaEPsoCache(Copyable, Movable):
    """What the cache did this run (renderer field `sota_e_pso`)."""
    var enabled: Bool
    var path: String
    var status: Int
    var reason: String
    var loaded_bytes: Int
    var manifest: List[UInt64]      # SPIR-V module hashes recorded by the file that was loaded
    var saved_bytes: Int

    def __init__(out self):
        self.enabled = False
        self.path = ""
        self.status = SOTA_E_PSO_DISABLED
        self.reason = "disabled"
        self.loaded_bytes = 0
        self.manifest = List[UInt64]()
        self.saved_bytes = 0

    def warm_hits(self, current: List[UInt64]) -> Int:
        """How many of the SPIR-V modules created this run were in the loaded manifest."""
        var hits = 0
        for i in range(len(current)):
            for j in range(len(self.manifest)):
                if self.manifest[j] == current[i]:
                    hits += 1
                    break
        return hits


def sota_e_pso_cache_path() -> String:
    """`$FLINT_PSO_CACHE_DIR` > `$XDG_CACHE_HOME/flint` > `$HOME/.cache/flint` > `build`; "" when disabled."""
    if env_value("FLINT_PSO_CACHE") == "0":
        return ""
    var d = env_value("FLINT_PSO_CACHE_DIR")
    if d.byte_length() == 0:
        var xdg = env_value("XDG_CACHE_HOME")
        if xdg.byte_length() > 0:
            d = xdg + "/flint"
        else:
            var home = env_value("HOME")
            d = home + "/.cache/flint" if home.byte_length() > 0 else String("build")
    return d + "/" + SOTA_E_PSO_FILE


def sota_e_device_identity(mut vk: VulkanGpu) -> SotaEDeviceIdentity:
    """`VkPhysicalDeviceProperties`: vendorID @8, deviceID @12, driverVersion @4, pipelineCacheUUID @276."""
    var props = List[UInt8](length=1024, fill=0)
    if vk.physical != 0:
        _ = call2(vk.instance_call("vkGetPhysicalDeviceProperties"), vk.physical, Int(props.unsafe_ptr()))
    var uuid = List[UInt8]()
    for i in range(16):
        uuid.append(props[276 + i])
    return SotaEDeviceIdentity(_get32(props, 8), _get32(props, 12), _get32(props, 4), uuid^)


def _get32(b: List[UInt8], at: Int) -> UInt32:
    return UInt32(b[at]) | (UInt32(b[at + 1]) << 8) | (UInt32(b[at + 2]) << 16) | (UInt32(b[at + 3]) << 24)


def _get64(b: List[UInt8], at: Int) -> UInt64:
    return UInt64(_get32(b, at)) | (UInt64(_get32(b, at + 4)) << 32)


def _put32(mut b: List[UInt8], v: UInt32):
    for k in range(4):
        b.append(UInt8((v >> UInt32(8 * k)) & 0xFF))


def _put64(mut b: List[UInt8], v: UInt64):
    for k in range(8):
        b.append(UInt8((v >> UInt64(8 * k)) & 0xFF))


def sota_e_pso_encode(id: SotaEDeviceIdentity, manifest: List[UInt64], payload: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    var magic = SOTA_E_PSO_MAGIC
    for i in range(8):
        out.append(magic.as_bytes()[i])
    _put32(out, id.vendor)
    _put32(out, id.device)
    _put32(out, id.driver_version)
    for i in range(16):
        out.append(id.uuid[i] if i < len(id.uuid) else UInt8(0))
    _put32(out, UInt32(len(manifest)))
    _put64(out, UInt64(len(payload)))
    _put64(out, hash_list(payload))
    for i in range(len(manifest)):
        _put64(out, manifest[i])
    for i in range(len(payload)):
        out.append(payload[i])
    return out^


@fieldwise_init
struct SotaEPsoDecoded(Copyable, Movable):
    var ok: Bool
    var reason: String
    var manifest: List[UInt64]
    var payload: List[UInt8]


def sota_e_pso_decode(file: List[UInt8], id: SotaEDeviceIdentity) -> SotaEPsoDecoded:
    """Validates a container against `id`; `ok == False` carries the rejection reason."""
    var none_m = List[UInt64]()
    var none_p = List[UInt8]()
    if len(file) < SOTA_E_PSO_HEADER_BYTES:
        return SotaEPsoDecoded(False, "truncated header", none_m^, none_p^)
    var magic = SOTA_E_PSO_MAGIC
    for i in range(8):
        if file[i] != magic.as_bytes()[i]:
            return SotaEPsoDecoded(False, "bad magic", none_m^, none_p^)
    var uuid = List[UInt8]()
    for i in range(16):
        uuid.append(file[20 + i])
    var fid = SotaEDeviceIdentity(_get32(file, 8), _get32(file, 12), _get32(file, 16), uuid^)
    if not fid.matches(id):
        return SotaEPsoDecoded(False, "different GPU / driver", none_m^, none_p^)
    var count = Int(_get32(file, 36))
    var plen = Int(_get64(file, 40))
    var phash = _get64(file, 48)
    var body = SOTA_E_PSO_HEADER_BYTES + count * 8
    if count < 0 or count > 1 << 20 or plen < 32 or body + plen != len(file):
        return SotaEPsoDecoded(False, "size mismatch (torn write?)", none_m^, none_p^)
    var manifest = List[UInt64]()
    for i in range(count):
        manifest.append(_get64(file, SOTA_E_PSO_HEADER_BYTES + i * 8))
    var payload = List[UInt8]()
    for i in range(plen):
        payload.append(file[body + i])
    if hash_list(payload) != phash:
        return SotaEPsoDecoded(False, "payload hash mismatch", none_m^, none_p^)
    # The driver's own `VkPipelineCacheHeaderVersionOne`: length >= 32, version 1, same device.
    if _get32(payload, 0) < 32 or _get32(payload, 4) != 1:
        return SotaEPsoDecoded(False, "bad Vulkan cache header", none_m^, none_p^)
    if _get32(payload, 8) != id.vendor or _get32(payload, 12) != id.device:
        return SotaEPsoDecoded(False, "Vulkan header names another device", none_m^, none_p^)
    for i in range(16):
        if payload[16 + i] != id.uuid[i]:
            return SotaEPsoDecoded(False, "pipelineCacheUUID changed", none_m^, none_p^)
    return SotaEPsoDecoded(True, "ok", manifest^, payload^)


def _create_vk_cache(mut vk: VulkanGpu, payload: List[UInt8]) -> Bool:
    vk.rewind()
    var info = vk.scratch(40)
    put32(info, TYPE_PIPELINE_CACHE_CREATE_INFO)
    put64(info + 8, 0)
    put32(info + 16, 0)
    put64(info + 24, len(payload))
    put64(info + 32, Int(payload.unsafe_ptr()) if len(payload) > 0 else 0)
    var slot = vk.scratch(8)
    var res = call4(vk.device_call("vkCreatePipelineCache"), vk.device, info, 0, slot)
    if res != 0:
        return False
    vk.pipeline_cache = get64(slot)
    return True


def sota_e_pso_cache_open(mut vk: VulkanGpu) -> SotaEPsoCache:
    """Creates `vk.pipeline_cache`, seeded from the validated on-disk container when there is one."""
    var st = SotaEPsoCache()
    if vk.device == 0 or vk.pipeline_cache != 0:
        st.reason = "no device or cache already open"
        return st^
    st.path = sota_e_pso_cache_path()
    if st.path.byte_length() == 0:
        return st^  # FLINT_PSO_CACHE=0
    st.enabled = True
    var id = sota_e_device_identity(vk)
    var empty = List[UInt8]()
    if file_size(st.path) <= 0:
        st.status = SOTA_E_PSO_COLD_MISSING
        st.reason = "no cache file"
        _ = _create_vk_cache(vk, empty)
        return st^
    var raw = read_bytes(st.path)
    var dec = sota_e_pso_decode(raw, id)
    if not dec.ok:
        st.status = SOTA_E_PSO_COLD_REJECTED
        st.reason = dec.reason
        _ = _create_vk_cache(vk, empty)
        return st^
    if not _create_vk_cache(vk, dec.payload):
        st.status = SOTA_E_PSO_COLD_REJECTED
        st.reason = "driver refused the blob"
        _ = _create_vk_cache(vk, empty)
        return st^
    st.status = SOTA_E_PSO_WARM
    st.reason = "ok"
    st.loaded_bytes = len(dec.payload)
    st.manifest = dec.manifest.copy()
    return st^


def sota_e_pso_cache_save(mut vk: VulkanGpu, mut st: SotaEPsoCache) -> Bool:
    """Atomically writes the cache container (call before the device is destroyed)."""
    if not st.enabled or vk.device == 0 or vk.pipeline_cache == 0:
        return False
    vk.rewind()
    var size_slot = vk.scratch(8)
    put64(size_slot, 0)
    var get_data = vk.device_call("vkGetPipelineCacheData")
    if call4(get_data, vk.device, vk.pipeline_cache, size_slot, 0) != 0:
        return False
    var sz = get64(size_slot)
    if sz < 32:
        return False
    var payload = List[UInt8](length=sz, fill=0)
    if call4(get_data, vk.device, vk.pipeline_cache, size_slot, Int(payload.unsafe_ptr())) != 0:
        return False
    var n = get64(size_slot)
    if n < 32 or n > sz:
        return False
    payload.resize(n, 0)
    # Manifest: union of what was loaded and what this run compiled (order-stable, deduplicated).
    var manifest = st.manifest.copy()
    for i in range(len(vk.sota_e_spirv_hashes)):
        var h = vk.sota_e_spirv_hashes[i]
        var seen = False
        for j in range(len(manifest)):
            if manifest[j] == h:
                seen = True
                break
        if not seen:
            manifest.append(h)
    var file = sota_e_pso_encode(sota_e_device_identity(vk), manifest, payload)
    var slash = -1
    var pb = st.path.as_bytes()
    for i in range(st.path.byte_length()):
        if pb[i] == 47:
            slash = i
    if slash > 0:
        var dir_bytes = List[UInt8]()
        for i in range(slash):
            dir_bytes.append(pb[i])
        _ = make_dirs(String(unsafe_from_utf8=dir_bytes))
    if not write_file_durable(st.path, file):
        return False
    st.saved_bytes = len(file)
    return True
