"""Flint v5 Audio Plugin & ECS Component Schemas (`flint.audio.plugin`).

Provides:
- `AudioSyncStateResource`: `Resource` holding headless/RT flags, previous listener
  velocity tracking, `AcousticMaterialTable`, and optional `Bvh8MeshCollider` for
  raycast emitter occlusion.
- `AudioListenerComponent` & `AudioEmitterComponent` implementing `SerializableComponent`
  with explicit `world.set_component_binary_hooks` for non-trivial structs.
- `register_audio_components(mut world: World)` & `register_audio_codecs(mut reg: ComponentRegistry)`.
- Optional physics occlusion provider: with `AudioPlugin(physics_occlusion=True)` every
  `occlusion_every_n_frames` frames each occlusion-enabled emitter is raycast from the
  listener through the `PhysicsWorld` (read-only `raycast_all`) and the accumulated
  per-material transmission loss feeds the emitter's occlusion gain / low-pass cutoff.
  A static `Bvh8MeshCollider` in `occlusion_bvh` (see `set_occlusion_geometry`) still wins.
- `AudioPlugin`: inserts `AudioServer` directly as a `Resource`, starts headless or
  interactive RT thread in `STAGE_STARTUP`, syncs `-Z` forward/`+X` right/`+Y` up
  listener and spatial emitters in `STAGE_LAST` (after `TransformPlugin`), and
  stops the RT thread cleanly in `STAGE_SHUTDOWN`.
"""
from std.collections import Dict, List
from std.math import max, exp, isfinite
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    FrameTime,
    Resource,
    Resources,
    STAGE_LAST,
    STAGE_SHUTDOWN,
    STAGE_STARTUP,
)
from flint.assets.asset_ref import AssetRef
from flint.assets.flpk_v3 import ASSET_KIND_AUDIO
from flint.assets.json import JsonDocument
from flint.assets.json_write import json_quote
from flint.assets.server import AssetServer
from flint.audio.acoustics import (
    AcousticMaterialTable,
    AcousticOcclusionResult,
    CLEAR_CUTOFF_HZ,
    evaluate_emitter_acoustics,
)
from flint.audio.server import AudioServer, BUS_SFX, SoundHandle
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.ecs.world import (
    Entity,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.plugin import RigidBodyComponent
from flint.physics.world import PhysicsWorld, SceneCastHit
from flint.scene.scene import ComponentRegistry
from flint.scene.transform_hierarchy import GlobalTransform, SerializableComponent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@always_inline
def _f32_s(v: Float32) -> String:
    return String(Float64(v))


def _vec3_json(v: Vec3) -> String:
    return "[" + _f32_s(v.x) + ", " + _f32_s(v.y) + ", " + _f32_s(v.z) + "]"


def _read_vec3_json(doc: JsonDocument, node: Int, fallback: Vec3 = Vec3()) -> Vec3:
    if doc.array_size(node) < 3:
        return fallback
    return Vec3(
        Float32(doc.number(doc.at(node, 0), Float64(fallback.x))),
        Float32(doc.number(doc.at(node, 1), Float64(fallback.y))),
        Float32(doc.number(doc.at(node, 2), Float64(fallback.z))),
    )


def _audio_bin_write[C: SerializableComponent](comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[C, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.begin(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
    comp.write_bin(out)
    out.end()


def _audio_bin_read[C: SerializableComponent](comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    try:
        var ver = reader.enter(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
        var val = C.read_bin(reader, ver)
        reader.leave()
        var dst = Pointer[C, MutAnyOrigin](unsafe_from_address=comp_addr)
        __get_address_as_uninit_lvalue(dst._mlir_value) = val^
        return True
    except:
        return False


def _register_non_trivial_audio[C: SerializableComponent](mut world: World):
    var tid = world.register_component[C]()
    comptime is_triv = C.__copy_ctor_is_trivial and C.__move_ctor_is_trivial and C.__del__is_trivial
    if not is_triv and tid >= 0 and not world.components[tid].has_custom_bin:
        world.set_component_binary_hooks(
            tid, _audio_bin_write[C], _audio_bin_read[C]
        )


def _register_audio_bin_hooks[C: SerializableComponent](mut world: World):
    var tid = world.register_component[C]()
    if tid >= 0 and not world.components[tid].has_custom_bin:
        world.set_component_binary_hooks(
            tid, _audio_bin_write[C], _audio_bin_read[C]
        )


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------

struct _TrackedAudioEmitter(Copyable, ImplicitlyCopyable, Movable):
    var entity_bits: UInt64
    var handle: SoundHandle
    var asset_id: Int

    def __init__(out self, entity_bits: UInt64 = 0, handle: SoundHandle = SoundHandle(), asset_id: Int = -1):
        self.entity_bits = entity_bits
        self.handle = handle
        self.asset_id = asset_id


struct AudioSyncStateResource(Resource, Movable):
    """Listener velocity tracking, headless mode flag, active emitters, and occlusion geometry."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.audio.AudioSyncStateResource")
    var headless: Bool
    var rt_started: Bool
    var prev_listener_pos: Vec3
    var prev_listener_bits: UInt64
    var has_prev_listener: Bool
    var material_table: AcousticMaterialTable
    var occlusion_bvh: List[Bvh8MeshCollider]
    var active_emitters: List[_TrackedAudioEmitter]
    var emitter_to_slot: Dict[UInt64, Int]
    var physics_occlusion: Bool
    var occlusion_every_n: Int
    var occlusion_frame: Int
    var occlusion_queries: Int

    def __init__(out self, headless: Bool = False):
        self.headless = headless
        self.rt_started = False
        self.prev_listener_pos = Vec3()
        self.prev_listener_bits = 0
        self.has_prev_listener = False
        self.material_table = AcousticMaterialTable()
        self.occlusion_bvh = List[Bvh8MeshCollider]()
        self.active_emitters = List[_TrackedAudioEmitter]()
        self.emitter_to_slot = Dict[UInt64, Int]()
        self.physics_occlusion = False
        self.occlusion_every_n = 4
        self.occlusion_frame = 0
        self.occlusion_queries = 0

    def set_occlusion_geometry(mut self, var bvh: Bvh8MeshCollider):
        """Uses a static acoustic mesh for emitter occlusion (takes precedence over physics)."""
        self.occlusion_bvh.clear()
        self.occlusion_bvh.append(bvh^)

    def enable_physics_occlusion(mut self, every_n_frames: Int = 4):
        self.physics_occlusion = True
        self.occlusion_every_n = max(1, every_n_frames)

    def record_emitter(mut self, entity: Entity, handle: SoundHandle, asset_id: Int = -1):
        var bits = entity.to_bits()
        var found = self.emitter_to_slot.get(bits)
        if found:
            self.active_emitters[found.value()].handle = handle
            self.active_emitters[found.value()].asset_id = asset_id
        else:
            var idx = len(self.active_emitters)
            self.emitter_to_slot[bits] = idx
            self.active_emitters.append(_TrackedAudioEmitter(bits, handle, asset_id))


# ---------------------------------------------------------------------------
# 2 Serializable ECS Audio Components
# ---------------------------------------------------------------------------

struct AudioListenerComponent(SerializableComponent, ImplicitlyCopyable):
    """Spatial audio listener marker attached to an entity with `GlobalTransform`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.audio.AudioListenerComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "AudioListenerComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("ALIS")
    comptime SCHEMA_VERSION: Int = 1

    var active: Bool
    var priority: Int
    var volume_scale: Float32
    var enable_room_probe: Bool
    var has_prev_pos: Bool

    def __init__(
        out self,
        active: Bool = True,
        priority: Int = 0,
        volume_scale: Float32 = 1.0,
        enable_room_probe: Bool = False,
    ):
        self.active = active
        self.priority = priority
        self.volume_scale = volume_scale
        self.enable_room_probe = enable_room_probe
        self.has_prev_pos = False

    def write_bin(self, mut out: BlobWriter):
        out.boolean(self.active)
        out.integer(self.priority)
        out.f32(self.volume_scale)
        out.boolean(self.enable_room_probe)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var act = reader.boolean()
        var prio = reader.integer()
        var vol = reader.f32()
        var rp = reader.boolean()
        if reader.failed:
            raise Error("AudioListenerComponent: truncated blob")
        return Self(act, prio, vol, rp)

    def write_json(self) -> String:
        return (
            "{\"active\": " + ("true" if self.active else "false")
            + ", \"priority\": " + String(self.priority)
            + ", \"volume_scale\": " + _f32_s(self.volume_scale)
            + ", \"enable_room_probe\": " + ("true" if self.enable_room_probe else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var act = doc.boolean(doc.object_get(node, "active"), True)
        var prio = doc.integer(doc.object_get(node, "priority"), 0)
        var vol = Float32(doc.number(doc.object_get(node, "volume_scale"), 1.0))
        var rp = doc.boolean(doc.object_get(node, "enable_room_probe"), False)
        return Self(act, prio, vol, rp)


struct AudioEmitterComponent(SerializableComponent):
    """Spatial or 2D audio emitter component synced against `AudioServer`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.audio.AudioEmitterComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "AudioEmitterComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("AEMT")
    comptime SCHEMA_VERSION: Int = 3

    var asset_id: Int
    var asset_path: String
    var volume: Float32
    var pitch: Float32
    var bus_id: Int
    var looping: Bool
    var spatial: Bool
    var priority: Int
    var ref_dist: Float32
    var max_dist: Float32
    var play_on_spawn: Bool
    var playing: Bool
    var auto_play_pending: Bool
    var enable_occlusion: Bool
    var occlusion_gain: Float32
    var cutoff_hz: Float32
    var obstruction: Float32
    var prev_pos: Vec3
    var has_prev_pos: Bool
    var handle: SoundHandle
    var audio_ref: AssetRef
    var streaming: Bool

    def __init__(
        out self,
        asset_id: Int = -1,
        asset_path: String = "",
        volume: Float32 = 1.0,
        pitch: Float32 = 1.0,
        bus_id: Int = BUS_SFX,
        looping: Bool = False,
        spatial: Bool = True,
        priority: Int = 1,
        ref_dist: Float32 = 15.0,
        max_dist: Float32 = 350.0,
        play_on_spawn: Bool = False,
        playing: Bool = False,
        auto_play_pending: Bool = False,
        enable_occlusion: Bool = False,
        occlusion_gain: Float32 = 1.0,
        cutoff_hz: Float32 = CLEAR_CUTOFF_HZ,
        obstruction: Float32 = 0.0,
        var audio_ref: AssetRef = AssetRef(),
        streaming: Bool = False,
    ):
        self.asset_id = asset_id
        self.asset_path = asset_path
        self.volume = volume
        self.pitch = pitch
        self.bus_id = bus_id
        self.looping = looping
        self.spatial = spatial
        self.priority = priority
        self.ref_dist = ref_dist
        self.max_dist = max_dist
        self.play_on_spawn = play_on_spawn
        self.playing = playing
        self.auto_play_pending = auto_play_pending or play_on_spawn
        self.enable_occlusion = enable_occlusion
        self.occlusion_gain = occlusion_gain
        self.cutoff_hz = cutoff_hz
        self.obstruction = obstruction
        self.prev_pos = Vec3()
        self.has_prev_pos = False
        self.handle = SoundHandle()
        self.audio_ref = audio_ref^
        self.streaming = streaming
        if self.audio_ref.is_empty() and self.asset_path.byte_length() > 0:
            self.audio_ref = AssetRef.from_path(self.asset_path, ASSET_KIND_AUDIO)
        elif self.asset_path.byte_length() == 0 and self.audio_ref.rel_path.byte_length() > 0:
            self.asset_path = self.audio_ref.rel_path

    def play(mut self):
        self.playing = True
        self.auto_play_pending = True

    def stop(mut self):
        self.playing = False
        self.auto_play_pending = False

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.asset_id)
        out.text(self.asset_path)
        out.f32(self.volume)
        out.f32(self.pitch)
        out.integer(self.bus_id)
        out.boolean(self.looping)
        out.boolean(self.spatial)
        out.integer(self.priority)
        out.f32(self.ref_dist)
        out.f32(self.max_dist)
        out.boolean(self.play_on_spawn)
        out.boolean(self.playing)
        out.boolean(self.auto_play_pending)
        out.boolean(self.enable_occlusion)
        out.f32(self.occlusion_gain)
        out.f32(self.cutoff_hz)
        out.f32(self.obstruction)
        out.vec3(self.prev_pos)
        out.boolean(self.has_prev_pos)
        self.audio_ref.write_bin(out)
        out.boolean(self.streaming)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        var aid = reader.integer()
        var apath = reader.text()
        var vol = reader.f32()
        var pit = reader.f32()
        var bid = reader.integer()
        var loop = reader.boolean()
        var spat = reader.boolean()
        var prio = reader.integer()
        var rdist = reader.f32()
        var mdist = reader.f32()
        var pos_spawn = reader.boolean()
        var pl = reader.boolean()
        var pending = reader.boolean()
        var occ = reader.boolean()
        var ogain = reader.f32()
        var cut = reader.f32()
        var obs = reader.f32()
        var pp = reader.vec3()
        var hpp = reader.boolean()
        var aref = AssetRef()
        if version >= 2:
            aref = AssetRef.read_bin(reader, 1)
        var streaming = reader.boolean() if version >= 3 else False
        if reader.failed:
            raise Error("AudioEmitterComponent: truncated blob")
        if version < 2:
            if aref.is_empty() and apath.byte_length() > 0:
                aref = AssetRef.from_path(apath, ASSET_KIND_AUDIO)
            elif apath.byte_length() == 0 and aref.rel_path.byte_length() > 0:
                apath = aref.rel_path
        var out = Self(
            aid, apath, vol, pit, bid, loop, spat, prio, rdist, mdist,
            pos_spawn, pl, pending, occ, ogain, cut, obs,
        )
        out.asset_path = apath
        out.audio_ref = aref^
        out.streaming = streaming
        out.auto_play_pending = pending
        out.prev_pos = pp
        out.has_prev_pos = hpp
        out.handle = SoundHandle()
        return out^

    def write_json(self) -> String:
        return (
            "{\"asset_id\": " + String(self.asset_id)
            + ", \"asset_path\": " + json_quote(self.asset_path)
            + ", \"streaming\": " + ("true" if self.streaming else "false")
            + ", \"volume\": " + _f32_s(self.volume)
            + ", \"pitch\": " + _f32_s(self.pitch)
            + ", \"bus_id\": " + String(self.bus_id)
            + ", \"looping\": " + ("true" if self.looping else "false")
            + ", \"spatial\": " + ("true" if self.spatial else "false")
            + ", \"priority\": " + String(self.priority)
            + ", \"ref_dist\": " + _f32_s(self.ref_dist)
            + ", \"max_dist\": " + _f32_s(self.max_dist)
            + ", \"play_on_spawn\": " + ("true" if self.play_on_spawn else "false")
            + ", \"playing\": " + ("true" if self.playing else "false")
            + ", \"auto_play_pending\": " + ("true" if self.auto_play_pending else "false")
            + ", \"enable_occlusion\": " + ("true" if self.enable_occlusion else "false")
            + ", \"occlusion_gain\": " + _f32_s(self.occlusion_gain)
            + ", \"cutoff_hz\": " + _f32_s(self.cutoff_hz)
            + ", \"obstruction\": " + _f32_s(self.obstruction)
            + ", \"prev_pos\": " + _vec3_json(self.prev_pos)
            + ", \"has_prev_pos\": " + ("true" if self.has_prev_pos else "false")
            + ", \"audio_ref\": " + self.audio_ref.write_json()
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var aid = doc.integer(doc.object_get(node, "asset_id"), -1)
        var apath = doc.text(doc.object_get(node, "asset_path"), "")
        var vol = Float32(doc.number(doc.object_get(node, "volume"), 1.0))
        var pit = Float32(doc.number(doc.object_get(node, "pitch"), 1.0))
        var bid = doc.integer(doc.object_get(node, "bus_id"), BUS_SFX)
        var loop = doc.boolean(doc.object_get(node, "looping"), False)
        var spat = doc.boolean(doc.object_get(node, "spatial"), True)
        var prio = doc.integer(doc.object_get(node, "priority"), 1)
        var rdist = Float32(doc.number(doc.object_get(node, "ref_dist"), 15.0))
        var mdist = Float32(doc.number(doc.object_get(node, "max_dist"), 350.0))
        var pos_spawn = doc.boolean(doc.object_get(node, "play_on_spawn"), False)
        var pl = doc.boolean(doc.object_get(node, "playing"), False)
        var pending = doc.boolean(doc.object_get(node, "auto_play_pending"), pos_spawn)
        var occ = doc.boolean(doc.object_get(node, "enable_occlusion"), False)
        var ogain = Float32(doc.number(doc.object_get(node, "occlusion_gain"), 1.0))
        var cut = Float32(doc.number(doc.object_get(node, "cutoff_hz"), Float64(CLEAR_CUTOFF_HZ)))
        var obs = Float32(doc.number(doc.object_get(node, "obstruction"), 0.0))
        var pp = _read_vec3_json(doc, doc.object_get(node, "prev_pos"), Vec3())
        var hpp = doc.boolean(doc.object_get(node, "has_prev_pos"), False)
        var anode = doc.object_get(node, "audio_ref")
        var aref = AssetRef.read_json(doc, anode) if anode >= 0 else AssetRef()
        if aref.is_empty() and apath.byte_length() > 0:
            aref = AssetRef.from_path(apath, ASSET_KIND_AUDIO)
        elif apath.byte_length() == 0 and aref.rel_path.byte_length() > 0:
            apath = aref.rel_path
        var out = Self(
            aid, apath, vol, pit, bid, loop, spat, prio, rdist, mdist,
            pos_spawn, pl, pending, occ, ogain, cut, obs, aref^,
            streaming=doc.boolean(doc.object_get(node, "streaming"), False),
        )
        out.auto_play_pending = pending
        out.prev_pos = pp
        out.has_prev_pos = hpp
        return out^


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_audio_components(mut world: World):
    """Idempotently registers `AudioListenerComponent` and `AudioEmitterComponent`."""
    _register_audio_bin_hooks[AudioListenerComponent](world)
    _register_non_trivial_audio[AudioEmitterComponent](world)


def register_audio_codecs(mut reg: ComponentRegistry):
    """Registers audio `SerializableComponent` codecs on `ComponentRegistry`."""
    reg.register[AudioListenerComponent]()
    reg.register[AudioEmitterComponent]()


# ---------------------------------------------------------------------------
# Physics occlusion provider
# ---------------------------------------------------------------------------

def evaluate_physics_occlusion(
    physics: PhysicsWorld,
    materials: AcousticMaterialTable,
    listener_pos: Vec3,
    emitter_pos: Vec3,
    emitter_body_id: Int = -1,
    listener_body_id: Int = -1,
    max_walls: Int = 4,
) -> AcousticOcclusionResult:
    """Read-only `PhysicsWorld.raycast_all` listener -> emitter. Every distinct blocking body
    (ignoring the emitter's and listener's own bodies) adds its material's transmission loss;
    gain = `10^(-loss/20)` (>= 0.01), low-pass cutoff = `20 kHz * exp(-0.165 * loss)` (>= 220 Hz)."""
    var res = AcousticOcclusionResult()
    var rel = emitter_pos - listener_pos
    var dist = rel.length()
    if not isfinite(dist) or dist < Float32(0.05):
        return res^
    var hits = physics.raycast_all(listener_pos, rel, dist)
    var total_db = Float32(0.0)
    var walls = 0
    var seen = List[Int]()
    for i in range(len(hits)):
        if walls >= max_walls:
            break
        var hb = hits[i].body_id
        if hb < 0 or hb == emitter_body_id or hb == listener_body_id:
            continue
        if hits[i].distance >= dist - Float32(0.01):
            continue
        var dup = False
        for k in range(len(seen)):
            if seen[k] == hb:
                dup = True
                break
        if dup:
            continue
        seen.append(hb)
        walls += 1
        var mat_id = hits[i].material_id
        var mat = materials.get(UInt16(mat_id) if mat_id >= 0 and mat_id < 65536 else UInt16(0))
        total_db += mat.transmission_loss_db
    res.direct_blocked = walls > 0
    res.wall_hits = walls
    if walls == 0:
        return res^
    total_db = min(total_db, Float32(80.0))
    res.transmission_loss_db = total_db
    res.room_occlusion_factor = 1.0
    res.fan_open_rays = 0
    res.portal_openness = 0.0
    res.occlusion_gain = max(Float32(0.01), exp(-total_db * Float32(0.11512925465)))
    res.occlusion_cutoff_hz = max(Float32(220.0), CLEAR_CUTOFF_HZ * exp(-total_db * Float32(0.165)))
    return res^


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def _sys_audio_startup(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var srv_opt = resources.get_ptr_mut[AudioServer]()
    var sync_opt = resources.get_ptr_mut[AudioSyncStateResource]()
    if not srv_opt or not sync_opt:
        return
    ref server = srv_opt.value()[unsafe_offset=0]
    ref sync = sync_opt.value()[unsafe_offset=0]
    if server.rt_running:
        sync.rt_started = True
        return
    if sync.headless:
        sync.rt_started = server.start_headless_rt_thread()
    else:
        if not server.start_realtime_thread():
            sync.rt_started = server.start_headless_rt_thread()
        else:
            sync.rt_started = True


def _sys_audio_sync(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var srv_opt = resources.get_ptr_mut[AudioServer]()
    var sync_opt = resources.get_ptr_mut[AudioSyncStateResource]()
    if not srv_opt or not sync_opt:
        return
    ref server = srv_opt.value()[unsafe_offset=0]
    ref sync = sync_opt.value()[unsafe_offset=0]

    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt and ft_opt.value()[unsafe_offset=0].delta_seconds > Float32(1.0e-5):
        dt = ft_opt.value()[unsafe_offset=0].delta_seconds
    var inv_dt = Float32(1.0) / dt

    # 0. Stop active emitters whose entity was despawned or AudioEmitterComponent removed
    var keep_emitters = List[_TrackedAudioEmitter]()
    for i in range(len(sync.active_emitters)):
        var rec = sync.active_emitters[i]
        var e = Entity.from_bits(rec.entity_bits)
        var em_opt = world.get_ptr[AudioEmitterComponent](e)
        var alive_em = world.is_alive(e) and Bool(em_opt)
        if not alive_em:
            if rec.handle.is_valid():
                _ = server.stop_sound(rec.handle)
        else:
            ref em = em_opt.value()[unsafe_offset=0]
            if (not em.handle.is_valid()) or (not em.handle.equals(rec.handle)):
                if rec.handle.is_valid():
                    _ = server.stop_sound(rec.handle)
                if em.handle.is_valid():
                    keep_emitters.append(_TrackedAudioEmitter(rec.entity_bits, em.handle, em.asset_id))
            else:
                keep_emitters.append(rec)
    sync.active_emitters = keep_emitters^
    sync.emitter_to_slot.clear()
    for i in range(len(sync.active_emitters)):
        sync.emitter_to_slot[sync.active_emitters[i].entity_bits] = i

    # 1. Select single highest-priority active AudioListenerComponent (`-Z` forward, `+X` right, `+Y` up)
    var listener_pos = Vec3()
    var has_listener = False
    var best_prio = -2147483648
    var best_listener = Entity()
    var l_mask = world.bit_for[AudioListenerComponent]().union(world.bit_for[GlobalTransform]())
    var l_entities = world.query_entities(l_mask)
    for i in range(len(l_entities)):
        var e = l_entities[i]
        var lc_opt = world.get_ptr[AudioListenerComponent](e)
        var gt_opt = world.get_ptr[GlobalTransform](e)
        if not lc_opt or not gt_opt:
            continue
        ref lc = lc_opt.value()[unsafe_offset=0]
        if not lc.active:
            continue
        if (not has_listener) or lc.priority > best_prio:
            best_prio = lc.priority
            has_listener = True
            best_listener = e

    for i in range(len(l_entities)):
        var e = l_entities[i]
        if (not has_listener) or e != best_listener:
            var lc_mut = world.get_ptr_mut[AudioListenerComponent](e)
            if lc_mut:
                lc_mut.value()[unsafe_offset=0].has_prev_pos = False

    if has_listener:
        var lc_mut = world.get_ptr_mut[AudioListenerComponent](best_listener)
        var gt_opt = world.get_ptr[GlobalTransform](best_listener)
        if lc_mut and gt_opt:
            ref best_lis = lc_mut.value()[unsafe_offset=0]
            ref gt = gt_opt.value()[unsafe_offset=0]
            listener_pos = gt.translation
            var listener_bits = best_listener.to_bits()
            var vel = Vec3()
            var rb_opt = world.get_ptr[RigidBodyComponent](best_listener)
            if rb_opt:
                vel = rb_opt.value()[unsafe_offset=0].linear_vel
            elif (
                sync.has_prev_listener
                and listener_bits == sync.prev_listener_bits
                and best_lis.has_prev_pos
            ):
                vel = (listener_pos - sync.prev_listener_pos) * inv_dt
            best_lis.has_prev_pos = True
            sync.prev_listener_pos = listener_pos
            sync.prev_listener_bits = listener_bits
            sync.has_prev_listener = True
            var fwd = gt.rotation.rotate(Vec3(0.0, 0.0, -1.0)).normalized()
            var right = gt.rotation.rotate(Vec3(1.0, 0.0, 0.0)).normalized()
            var up = gt.rotation.rotate(Vec3(0.0, 1.0, 0.0)).normalized()
            server.set_listener(listener_pos, vel, fwd, right, up)
    else:
        sync.has_prev_listener = False
        sync.prev_listener_bits = 0

    # 2. Sync AudioEmitterComponents (optionally refreshing physics occlusion this frame)
    var do_phys_occ = False
    var listener_body = -1
    if sync.physics_occlusion and has_listener and len(sync.occlusion_bvh) == 0:
        if sync.occlusion_frame % max(1, sync.occlusion_every_n) == 0 and resources.has[PhysicsWorld]():
            do_phys_occ = True
            var lrb = world.get_ptr[RigidBodyComponent](best_listener)
            if lrb:
                listener_body = lrb.value()[unsafe_offset=0].body_id
        sync.occlusion_frame += 1
    var phys_opt = resources.get_ptr[PhysicsWorld]()
    var e_mask = world.bit_for[AudioEmitterComponent]().union(world.bit_for[GlobalTransform]())
    var e_entities = world.query_entities(e_mask)
    for i in range(len(e_entities)):
        var e = e_entities[i]
        var em_opt = world.get_ptr_mut[AudioEmitterComponent](e)
        var gt_opt = world.get_ptr[GlobalTransform](e)
        if not em_opt or not gt_opt:
            continue
        ref em = em_opt.value()[unsafe_offset=0]
        var pos = gt_opt.value()[unsafe_offset=0].translation
        var vel = Vec3()
        var emitter_body = -1
        var rb_opt = world.get_ptr[RigidBodyComponent](e)
        if rb_opt:
            vel = rb_opt.value()[unsafe_offset=0].linear_vel
            emitter_body = rb_opt.value()[unsafe_offset=0].body_id
        elif em.has_prev_pos:
            vel = (pos - em.prev_pos) * inv_dt
        em.prev_pos = pos
        em.has_prev_pos = True

        if em.asset_id < 0 and (em.asset_path.byte_length() > 0 or not em.audio_ref.is_empty()):
            var resolved_aid = -1
            var cache_key = (
                em.asset_path
                if em.asset_path.byte_length() > 0
                else em.audio_ref.rel_path
            )
            if em.streaming:
                var asset_srv_opt = resources.get_ptr[AssetServer]()
                if asset_srv_opt:
                    var info = asset_srv_opt.value()[].resolve_audio_stream(em.audio_ref)
                    if info.valid:
                        resolved_aid = server.register_stream_source(info^)
                if resolved_aid < 0 and em.asset_path.byte_length() > 0:
                    resolved_aid = server.register_stream_asset(em.asset_path)
            elif cache_key.byte_length() > 0:
                var cached = server.path_cache.get(cache_key)
                if cached:
                    resolved_aid = cached.value()
            if resolved_aid < 0 and not em.streaming:
                var asset_srv_opt = resources.get_ptr_mut[AssetServer]()
                if asset_srv_opt:
                    ref asset_srv = asset_srv_opt.value()[unsafe_offset=0]
                    var auh = (
                        asset_srv.resolve_audio(em.audio_ref)
                        if not em.audio_ref.is_empty()
                        else asset_srv.load_audio(em.asset_path)
                    )
                    if asset_srv.audios.is_loaded(auh):
                        var dptr = asset_srv.audios.get_ptr(auh)
                        if dptr:
                            resolved_aid = server.register_pcm16_asset(
                                dptr.value()[].samples,
                                dptr.value()[].channels,
                                dptr.value()[].sample_rate,
                            )
                            if resolved_aid >= 0 and cache_key.byte_length() > 0:
                                server.path_cache[cache_key] = resolved_aid
                    if resolved_aid < 0 and cache_key.byte_length() > 0:
                        var rh = asset_srv.load_raw(cache_key)
                        if asset_srv.raws.is_loaded(rh):
                            var rptr = asset_srv.raws.get_ptr(rh)
                            if rptr and len(rptr.value()[].bytes) > 0:
                                resolved_aid = server.load_audio_bytes_cached(
                                    rptr.value()[].bytes.copy(), cache_key
                                )
            if resolved_aid < 0 and not em.streaming and em.asset_path.byte_length() > 0:
                resolved_aid = server.load_audio_cached(em.asset_path)
            if resolved_aid >= 0:
                em.asset_id = resolved_aid
                if em.asset_path.byte_length() == 0 and em.audio_ref.rel_path.byte_length() > 0:
                    em.asset_path = em.audio_ref.rel_path

        if em.playing and em.handle.is_valid():
            var tracked_slot = sync.emitter_to_slot.get(e.to_bits())
            if tracked_slot:
                var tracked_aid = sync.active_emitters[tracked_slot.value()].asset_id
                if tracked_aid >= 0 and em.asset_id >= 0 and tracked_aid != em.asset_id:
                    _ = server.stop_sound(em.handle)
                    em.handle = SoundHandle()
                    em.auto_play_pending = True

        if (em.auto_play_pending or (em.playing and not em.handle.is_valid())) and em.asset_id >= 0:
            if em.handle.is_valid():
                _ = server.stop_sound(em.handle)
            if em.spatial:
                em.handle = server.play_3d(
                    em.asset_id,
                    pos,
                    vel,
                    em.volume,
                    em.pitch,
                    em.bus_id,
                    em.looping,
                    em.priority,
                    em.ref_dist,
                    em.max_dist,
                )
            else:
                em.handle = server.play_2d(
                    em.asset_id,
                    em.volume,
                    em.pitch,
                    Float32(0.0),
                    em.bus_id,
                    em.looping,
                    False,
                )
            em.playing = True
            em.auto_play_pending = False

        if em.handle.is_valid():
            if em.playing:
                sync.record_emitter(e, em.handle, em.asset_id)
                if em.spatial:
                    _ = server.update_emitter(em.handle, pos, vel, em.volume, em.pitch)
                    if em.enable_occlusion:
                        if has_listener and len(sync.occlusion_bvh) > 0:
                            var res = evaluate_emitter_acoustics(
                                sync.occlusion_bvh[0],
                                sync.material_table,
                                listener_pos,
                                pos,
                            )
                            em.occlusion_gain = res.occlusion_gain
                            em.cutoff_hz = res.occlusion_cutoff_hz
                            em.obstruction = res.obstruction_factor
                        elif do_phys_occ and phys_opt:
                            var pres = evaluate_physics_occlusion(
                                phys_opt.value()[unsafe_offset=0],
                                sync.material_table,
                                listener_pos,
                                pos,
                                emitter_body,
                                listener_body,
                            )
                            sync.occlusion_queries += 1
                            em.occlusion_gain = pres.occlusion_gain
                            em.cutoff_hz = pres.occlusion_cutoff_hz
                            em.obstruction = pres.obstruction_factor
                        _ = server.set_emitter_occlusion(
                            em.handle, em.occlusion_gain, em.cutoff_hz, em.obstruction
                        )
            else:
                _ = server.stop_sound(em.handle)
                em.handle = SoundHandle()


def _sys_audio_shutdown(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var srv_opt = resources.get_ptr_mut[AudioServer]()
    if srv_opt:
        srv_opt.value()[unsafe_offset=0].stop_realtime_thread()
    var sync_opt = resources.get_ptr_mut[AudioSyncStateResource]()
    if sync_opt:
        sync_opt.value()[unsafe_offset=0].rt_started = False


def _has_audio_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


struct AudioPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers `AudioServer`, audio ECS components, and spatial sync/RT lifecycle."""
    var headless: Bool
    var sample_rate: Int
    var quantum_frames: Int
    var ring_capacity: Int
    var physics_occlusion: Bool
    var occlusion_every_n_frames: Int

    def __init__(
        out self,
        headless: Bool = False,
        sample_rate: Int = 48000,
        quantum_frames: Int = 256,
        ring_capacity: Int = 2048,
        physics_occlusion: Bool = False,
        occlusion_every_n_frames: Int = 4,
    ):
        self.headless = headless
        self.sample_rate = sample_rate
        self.quantum_frames = quantum_frames
        self.ring_capacity = ring_capacity
        self.physics_occlusion = physics_occlusion
        self.occlusion_every_n_frames = max(1, occlusion_every_n_frames)

    def build(self, mut app: EngineApp):
        register_audio_components(app.world)
        var eff_headless = self.headless or app.config.headless
        if not app.resources.has[AudioServer]():
            app.insert_resource[AudioServer](
                AudioServer(
                    sample_rate=self.sample_rate,
                    quantum_frames=self.quantum_frames,
                    ring_capacity=self.ring_capacity,
                )
            )
        if not app.resources.has[AudioSyncStateResource]():
            app.insert_resource[AudioSyncStateResource](AudioSyncStateResource(eff_headless))
        if self.physics_occlusion:
            var sync_opt = app.resources.get_ptr_mut[AudioSyncStateResource]()
            if sync_opt:
                sync_opt.value()[unsafe_offset=0].enable_physics_occlusion(self.occlusion_every_n_frames)

        var b_gt = app.world.bit_for[GlobalTransform]()
        var b_lis = app.world.bit_for[AudioListenerComponent]()
        var b_emt = app.world.bit_for[AudioEmitterComponent]()

        if not _has_audio_system(app, STAGE_STARTUP, "flint.audio.startup"):
            _ = app.add_system(
                STAGE_STARTUP,
                "flint.audio.startup",
                _sys_audio_startup,
                exclusive=True,
            )
        if not _has_audio_system(app, STAGE_LAST, "flint.audio.sync"):
            _ = app.add_system(
                STAGE_LAST,
                "flint.audio.sync",
                _sys_audio_sync,
                reads=b_gt,
                writes=b_lis.union(b_emt),
                exclusive=True,
            )
        if not _has_audio_system(app, STAGE_SHUTDOWN, "flint.audio.shutdown"):
            _ = app.add_system(
                STAGE_SHUTDOWN,
                "flint.audio.shutdown",
                _sys_audio_shutdown,
                exclusive=True,
            )
