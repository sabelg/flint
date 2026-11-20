"""Flint v5 Audio Plugin & ECS Component Schemas (`flint.audio.plugin`).

Provides:
- `AudioSyncStateResource`: `Resource` holding headless/RT flags, previous listener
  velocity tracking, `AcousticMaterialTable`, and optional `Bvh8MeshCollider` for
  raycast emitter occlusion.
- `AudioListenerComponent` & `AudioEmitterComponent` implementing `SerializableComponent`
  with explicit `world.set_component_binary_hooks` for non-trivial structs.
- `register_audio_components(mut world: World)` & `register_audio_codecs(mut reg: ComponentRegistry)`.
- `AudioPlugin`: inserts `AudioServer` directly as a `Resource`, starts headless or
  interactive RT thread in `STAGE_STARTUP`, syncs `-Z` forward/`+X` right/`+Y` up
  listener and spatial emitters in `STAGE_LAST` (after `TransformPlugin`), and
  stops the RT thread cleanly in `STAGE_SHUTDOWN`.
"""
from std.collections import Dict, List
from std.math import max
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
from flint.assets.json import JsonDocument
from flint.assets.json_write import json_quote
from flint.audio.acoustics import (
    AcousticMaterialTable,
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

    def __init__(out self, entity_bits: UInt64 = 0, handle: SoundHandle = SoundHandle()):
        self.entity_bits = entity_bits
        self.handle = handle


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

    def record_emitter(mut self, entity: Entity, handle: SoundHandle):
        var bits = entity.to_bits()
        var found = self.emitter_to_slot.get(bits)
        if found:
            self.active_emitters[found.value()].handle = handle
        else:
            var idx = len(self.active_emitters)
            self.emitter_to_slot[bits] = idx
            self.active_emitters.append(_TrackedAudioEmitter(bits, handle))


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
    comptime SCHEMA_VERSION: Int = 1

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

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
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
        var out = Self(
            aid, apath, vol, pit, bid, loop, spat, prio, rdist, mdist,
            pos_spawn, pl, pending or pl, occ, ogain, cut, obs,
        )
        out.auto_play_pending = pending or pl
        out.prev_pos = pp
        out.has_prev_pos = hpp
        out.handle = SoundHandle()
        return out^

    def write_json(self) -> String:
        return (
            "{\"asset_id\": " + String(self.asset_id)
            + ", \"asset_path\": " + json_quote(self.asset_path)
            + ", \"volume\": " + _f32_s(self.volume)
            + ", \"pitch\": " + _f32_s(self.pitch)
            + ", \"bus_id\": " + String(self.bus_id)
            + ", \"looping\": " + ("true" if self.looping else "false")
            + ", \"spatial\": " + ("true" if self.spatial else "false")
            + ", \"ref_dist\": " + _f32_s(self.ref_dist)
            + ", \"max_dist\": " + _f32_s(self.max_dist)
            + ", \"play_on_spawn\": " + ("true" if self.play_on_spawn else "false")
            + ", \"enable_occlusion\": " + ("true" if self.enable_occlusion else "false")
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
        var rdist = Float32(doc.number(doc.object_get(node, "ref_dist"), 15.0))
        var mdist = Float32(doc.number(doc.object_get(node, "max_dist"), 350.0))
        var pos_spawn = doc.boolean(doc.object_get(node, "play_on_spawn"), False)
        var occ = doc.boolean(doc.object_get(node, "enable_occlusion"), False)
        return Self(
            aid, apath, vol, pit, bid, loop, spat, 1, rdist, mdist,
            pos_spawn, pos_spawn, pos_spawn, occ,
        )


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
                    keep_emitters.append(_TrackedAudioEmitter(rec.entity_bits, em.handle))
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

    # 2. Sync AudioEmitterComponents
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
        var rb_opt = world.get_ptr[RigidBodyComponent](e)
        if rb_opt:
            vel = rb_opt.value()[unsafe_offset=0].linear_vel
        elif em.has_prev_pos:
            vel = (pos - em.prev_pos) * inv_dt
        em.prev_pos = pos
        em.has_prev_pos = True

        if em.auto_play_pending and em.asset_id >= 0:
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
                sync.record_emitter(e, em.handle)
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

    def __init__(
        out self,
        headless: Bool = False,
        sample_rate: Int = 48000,
        quantum_frames: Int = 256,
        ring_capacity: Int = 2048,
    ):
        self.headless = headless
        self.sample_rate = sample_rate
        self.quantum_frames = quantum_frames
        self.ring_capacity = ring_capacity

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
