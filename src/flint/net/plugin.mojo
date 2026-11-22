"""Flint v5 Networking Plugin & ECS Component Schemas (`flint.net.plugin`).

Provides:
- `NetConfigResource`: networking mode (`OFFLINE`, `SERVER`, `CLIENT`, `VIRTUAL`),
  tick rate, virtual/wall clock, and packet telemetry.
- `NetReplicationResource`: `Resource` owning `SnapshotReplicator`,
  `SnapshotClientReceiver`, `SnapshotInterpolationBuffer`, `LagCompHistory`, and
  `net_id <-> Entity` lookup map.
- `NetTransportResource`: RAII `Resource` holding lazy `List`-boxed `NetHub[UdpSocket]`
  and `NetHub[VirtualEndpoint]` transports (zero OS sockets opened unless bound).
- Generic `NetHubResource[T]`, `RollbackSessionResource[InputT]`, and
  `PredictionResource[InputT]` wrappers.
- 3 POD `SerializableComponent` ECS schemas:
  `NetworkIdentityComponent`, `ReplicationComponent`, `RollbackInputComponent`.
- `register_net_components(mut world: World)` & `register_net_codecs(mut reg: ComponentRegistry)`.
- `NetPlugin`: polls hubs & applies client snapshots in `STAGE_FIRST`, records
  `LagCompHistory` in `STAGE_FIXED_UPDATE`, syncs replicated entities & sends
  AOI snapshot packets in `STAGE_POST_UPDATE` (after `TransformPlugin`), and
  closes transports cleanly in `STAGE_SHUTDOWN`.
"""
from std.collections import Dict, List
from std.math import max
from std.reflection import reflect
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    FixedTime,
    FrameTime,
    Resource,
    Resources,
    STAGE_FIRST,
    STAGE_FIXED_UPDATE,
    STAGE_POST_UPDATE,
    STAGE_SHUTDOWN,
)
from flint.assets.json import JsonDocument
from flint.core.jobs import JobPool
from flint.core.math import Vec3, Quat
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.ecs.world import (
    Entity,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.net.hub import NetHub
from flint.net.lag_comp import HitboxPose, LagCompHistory
from flint.net.prediction import ClientPredictionBuffer
from flint.net.replicator import (
    SnapshotClientReceiver,
    SnapshotInterpolationBuffer,
    SnapshotReplicator,
)
from flint.net.rollback import RollbackInput, RollbackSession, VisualSmoother
from flint.net.socket_udp import NetEndpoint, NetTransport, UdpSocket
from flint.net.virtual_sim import VirtualEndpoint
from flint.physics.plugin import RigidBodyComponent
from flint.scene.scene import ComponentRegistry
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    LocalTransform,
    SerializableComponent,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

comptime NET_MODE_OFFLINE: Int = 0
comptime NET_MODE_SERVER: Int = 1
comptime NET_MODE_CLIENT: Int = 2
comptime NET_MODE_VIRTUAL: Int = 3

comptime NET_AUTHORITY_SERVER: UInt8 = 0
comptime NET_AUTHORITY_CLIENT: UInt8 = 1
comptime NET_AUTHORITY_SHARED: UInt8 = 2
comptime _NET_SNAP_ACK_MAGIC: UInt16 = 0x4135


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@always_inline
def _f32_s(v: Float32) -> String:
    return String(Float64(v))


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

struct NetConfigResource(Resource, Copyable, ImplicitlyCopyable, Movable):
    """Network mode, topology configuration, clock, and telemetry counters."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.net.NetConfigResource")
    var mode: Int
    var max_entities: Int
    var max_clients: Int
    var world_bounds: Float32
    var chunk_size: Float32
    var tick_hz: Float64
    var now_ms: Float64
    var local_peer_id: Int
    var last_server_tick: UInt16
    var packets_sent: UInt64
    var packets_received: UInt64

    def __init__(
        out self,
        mode: Int = NET_MODE_OFFLINE,
        max_entities: Int = 512,
        max_clients: Int = 16,
        world_bounds: Float32 = 512.0,
        chunk_size: Float32 = 64.0,
        tick_hz: Float64 = 60.0,
        local_peer_id: Int = 0,
    ):
        self.mode = mode
        self.max_entities = max_entities
        self.max_clients = max_clients
        self.world_bounds = world_bounds
        self.chunk_size = chunk_size
        self.tick_hz = tick_hz
        self.now_ms = 0.0
        self.local_peer_id = local_peer_id
        self.last_server_tick = 0
        self.packets_sent = 0
        self.packets_received = 0


struct NetReplicationResource(Resource, Movable):
    """Owns AOI `SnapshotReplicator`, client receiver, interpolation buffer, and `LagCompHistory`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.net.NetReplicationResource")
    var replicator: SnapshotReplicator
    var receiver: SnapshotClientReceiver
    var interp: SnapshotInterpolationBuffer
    var lag_comp: LagCompHistory
    var net_id_to_entity: Dict[Int, UInt64]

    def __init__(
        out self,
        max_entities: Int = 512,
        max_clients: Int = 16,
        world_bounds: Float32 = 512.0,
        chunk_size: Float32 = 64.0,
    ):
        self.replicator = SnapshotReplicator(
            max_entities=max_entities,
            max_clients=max_clients,
            world_bounds=world_bounds,
            chunk_size=chunk_size,
        )
        self.receiver = SnapshotClientReceiver(
            max_entities=max_entities,
            world_bounds=world_bounds,
            chunk_size=chunk_size,
        )
        self.interp = SnapshotInterpolationBuffer(max_entities=max_entities)
        self.lag_comp = LagCompHistory(max_entities=max_entities)
        self.net_id_to_entity = Dict[Int, UInt64]()


struct NetTransportResource(Resource, Movable):
    """RAII `Resource` lazily owning `NetHub[UdpSocket]` or `NetHub[VirtualEndpoint]`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.net.NetTransportResource")
    var udp_hubs: List[NetHub[UdpSocket]]
    var virtual_hubs: List[NetHub[VirtualEndpoint]]
    var peer_cids: List[Int]

    def __init__(out self):
        self.udp_hubs = List[NetHub[UdpSocket]]()
        self.virtual_hubs = List[NetHub[VirtualEndpoint]]()
        self.peer_cids = List[Int]()

    def bind_udp(
        mut self,
        endpoint: NetEndpoint = NetEndpoint.any_dual_stack(0),
        accept: Bool = True,
        max_channels: Int = 64,
    ) -> Bool:
        var sock = UdpSocket()
        if not sock.bind(endpoint):
            return False
        self.udp_hubs.clear()
        self.udp_hubs.append(NetHub[UdpSocket](sock^, accept=accept, max_channels=max_channels))
        return True

    def attach_virtual_hub(mut self, var hub: NetHub[VirtualEndpoint]):
        self.virtual_hubs.clear()
        self.virtual_hubs.append(hub^)

    def close(mut self):
        self.udp_hubs.clear()
        self.virtual_hubs.clear()
        self.peer_cids.clear()

    def __deinit__(deinit self):
        self.close()


struct NetHubResource[T: NetTransport & Deinitable](Resource, Movable):
    """Generic `Resource` wrapper for custom `NetHub[T]` instances."""
    comptime RESOURCE_ID: UInt64 = comp_hash64(reflect[Self].name())
    var hub: NetHub[Self.T]

    def __init__(out self, var hub: NetHub[Self.T]):
        self.hub = hub^


struct RollbackSessionResource[InputT: RollbackInput](Resource, Movable):
    """Generic `Resource` wrapper for GGPO `RollbackSession[InputT]`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64(reflect[Self].name())
    var session: RollbackSession[Self.InputT]

    def __init__(out self, var session: RollbackSession[Self.InputT]):
        self.session = session^


struct PredictionResource[InputT: RollbackInput](Resource, Movable):
    """Generic `Resource` wrapper for `ClientPredictionBuffer[InputT]` & `VisualSmoother`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64(reflect[Self].name())
    var buffer: ClientPredictionBuffer[Self.InputT]
    var smoother: VisualSmoother

    def __init__(out self, capacity: Int = 128):
        self.buffer = ClientPredictionBuffer[Self.InputT](capacity=capacity)
        self.smoother = VisualSmoother()


# ---------------------------------------------------------------------------
# 3 Serializable ECS Networking Components
# ---------------------------------------------------------------------------

struct NetworkIdentityComponent(SerializableComponent, ImplicitlyCopyable):
    """Network identity and authority ownership for a replicated ECS entity."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.net.NetworkIdentityComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "NetworkIdentityComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("NNID")
    comptime SCHEMA_VERSION: Int = 1

    var net_id: Int
    var owner_peer_id: Int
    var is_local: Bool
    var authority: UInt8

    def __init__(
        out self,
        net_id: Int = 0,
        owner_peer_id: Int = 0,
        is_local: Bool = False,
        authority: UInt8 = NET_AUTHORITY_SERVER,
    ):
        self.net_id = net_id
        self.owner_peer_id = owner_peer_id
        self.is_local = is_local
        self.authority = authority

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.net_id)
        out.integer(self.owner_peer_id)
        out.boolean(self.is_local)
        out.u8(self.authority)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var nid = reader.integer()
        var opid = reader.integer()
        var loc = reader.boolean()
        var auth = reader.u8()
        if reader.failed:
            raise Error("NetworkIdentityComponent: truncated blob")
        return Self(nid, opid, loc, auth)

    def write_json(self) -> String:
        return (
            "{\"net_id\": " + String(self.net_id)
            + ", \"owner_peer_id\": " + String(self.owner_peer_id)
            + ", \"is_local\": " + ("true" if self.is_local else "false")
            + ", \"authority\": " + String(Int(self.authority))
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var nid = doc.integer(doc.object_get(node, "net_id"), 0)
        var opid = doc.integer(doc.object_get(node, "owner_peer_id"), 0)
        var loc = doc.boolean(doc.object_get(node, "is_local"), False)
        var auth = UInt8(doc.integer(doc.object_get(node, "authority"), Int(NET_AUTHORITY_SERVER)))
        return Self(nid, opid, loc, auth)


struct ReplicationComponent(SerializableComponent, ImplicitlyCopyable):
    """Per-entity snapshot replication masks, AOI priority, and lag-comp hitbox settings."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.net.ReplicationComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "ReplicationComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("NREP")
    comptime SCHEMA_VERSION: Int = 1

    var replicate_pos: Bool
    var replicate_rot: Bool
    var replicate_vel: Bool
    var replicate_state: Bool
    var state_flags: UInt32
    var base_priority: Float32
    var interest_radius: Float32
    var hitbox_radius: Float32
    var record_lag_comp: Bool

    def __init__(
        out self,
        replicate_pos: Bool = True,
        replicate_rot: Bool = True,
        replicate_vel: Bool = True,
        replicate_state: Bool = True,
        state_flags: UInt32 = 0,
        base_priority: Float32 = 1.0,
        interest_radius: Float32 = 110.0,
        hitbox_radius: Float32 = 0.5,
        record_lag_comp: Bool = True,
    ):
        self.replicate_pos = replicate_pos
        self.replicate_rot = replicate_rot
        self.replicate_vel = replicate_vel
        self.replicate_state = replicate_state
        self.state_flags = state_flags
        self.base_priority = base_priority
        self.interest_radius = interest_radius
        self.hitbox_radius = hitbox_radius
        self.record_lag_comp = record_lag_comp

    def write_bin(self, mut out: BlobWriter):
        out.boolean(self.replicate_pos)
        out.boolean(self.replicate_rot)
        out.boolean(self.replicate_vel)
        out.boolean(self.replicate_state)
        out.u32(self.state_flags)
        out.f32(self.base_priority)
        out.f32(self.interest_radius)
        out.f32(self.hitbox_radius)
        out.boolean(self.record_lag_comp)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var rp = reader.boolean()
        var rr = reader.boolean()
        var rv = reader.boolean()
        var rs = reader.boolean()
        var sf = reader.u32()
        var bp = reader.f32()
        var ir = reader.f32()
        var hr = reader.f32()
        var lc = reader.boolean()
        if reader.failed:
            raise Error("ReplicationComponent: truncated blob")
        return Self(rp, rr, rv, rs, sf, bp, ir, hr, lc)

    def write_json(self) -> String:
        return (
            "{\"replicate_pos\": " + ("true" if self.replicate_pos else "false")
            + ", \"replicate_rot\": " + ("true" if self.replicate_rot else "false")
            + ", \"replicate_vel\": " + ("true" if self.replicate_vel else "false")
            + ", \"replicate_state\": " + ("true" if self.replicate_state else "false")
            + ", \"state_flags\": " + String(Int(self.state_flags))
            + ", \"base_priority\": " + _f32_s(self.base_priority)
            + ", \"interest_radius\": " + _f32_s(self.interest_radius)
            + ", \"hitbox_radius\": " + _f32_s(self.hitbox_radius)
            + ", \"record_lag_comp\": " + ("true" if self.record_lag_comp else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var rp = doc.boolean(doc.object_get(node, "replicate_pos"), True)
        var rr = doc.boolean(doc.object_get(node, "replicate_rot"), True)
        var rv = doc.boolean(doc.object_get(node, "replicate_vel"), True)
        var rs = doc.boolean(doc.object_get(node, "replicate_state"), True)
        var sf = UInt32(doc.integer(doc.object_get(node, "state_flags"), 0))
        var bp = Float32(doc.number(doc.object_get(node, "base_priority"), 1.0))
        var ir = Float32(doc.number(doc.object_get(node, "interest_radius"), 110.0))
        var hr = Float32(doc.number(doc.object_get(node, "hitbox_radius"), 0.5))
        var lc = doc.boolean(doc.object_get(node, "record_lag_comp"), True)
        return Self(rp, rr, rv, rs, sf, bp, ir, hr, lc)


struct RollbackInputComponent(SerializableComponent, ImplicitlyCopyable):
    """Quantizable player input state for deterministic rollback / client prediction."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.net.RollbackInputComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "RollbackInputComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("NRIN")
    comptime SCHEMA_VERSION: Int = 1

    var peer_id: Int
    var last_input_seq: UInt32
    var steer: Float32
    var throttle: Float32
    var action_bits: UInt32

    def __init__(
        out self,
        peer_id: Int = 0,
        last_input_seq: UInt32 = 0,
        steer: Float32 = 0.0,
        throttle: Float32 = 0.0,
        action_bits: UInt32 = 0,
    ):
        self.peer_id = peer_id
        self.last_input_seq = last_input_seq
        self.steer = steer
        self.throttle = throttle
        self.action_bits = action_bits

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.peer_id)
        out.u32(self.last_input_seq)
        out.f32(self.steer)
        out.f32(self.throttle)
        out.u32(self.action_bits)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var pid = reader.integer()
        var seq = reader.u32()
        var st = reader.f32()
        var th = reader.f32()
        var ab = reader.u32()
        if reader.failed:
            raise Error("RollbackInputComponent: truncated blob")
        return Self(pid, seq, st, th, ab)

    def write_json(self) -> String:
        return (
            "{\"peer_id\": " + String(self.peer_id)
            + ", \"last_input_seq\": " + String(Int(self.last_input_seq))
            + ", \"steer\": " + _f32_s(self.steer)
            + ", \"throttle\": " + _f32_s(self.throttle)
            + ", \"action_bits\": " + String(Int(self.action_bits))
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var pid = doc.integer(doc.object_get(node, "peer_id"), 0)
        var seq = UInt32(doc.integer(doc.object_get(node, "last_input_seq"), 0))
        var st = Float32(doc.number(doc.object_get(node, "steer"), 0.0))
        var th = Float32(doc.number(doc.object_get(node, "throttle"), 0.0))
        var ab = UInt32(doc.integer(doc.object_get(node, "action_bits"), 0))
        return Self(pid, seq, st, th, ab)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_net_components(mut world: World):
    """Idempotently registers all 3 networking components on `world`."""
    _ = world.register_component[NetworkIdentityComponent]()
    _ = world.register_component[ReplicationComponent]()
    _ = world.register_component[RollbackInputComponent]()


def register_net_codecs(mut reg: ComponentRegistry):
    """Registers all 3 networking `SerializableComponent` codecs on `ComponentRegistry`."""
    reg.register[NetworkIdentityComponent]()
    reg.register[ReplicationComponent]()
    reg.register[RollbackInputComponent]()


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def _encode_snap_ack(snap_seq: UInt16) -> List[UInt8]:
    var pkt = List[UInt8](capacity=4)
    pkt.append(UInt8(_NET_SNAP_ACK_MAGIC & 0xFF))
    pkt.append(UInt8((_NET_SNAP_ACK_MAGIC >> 8) & 0xFF))
    pkt.append(UInt8(snap_seq & 0xFF))
    pkt.append(UInt8((snap_seq >> 8) & 0xFF))
    return pkt^


def _decode_snap_ack(pkt: List[UInt8]) -> Tuple[Bool, UInt16]:
    if len(pkt) != 4:
        return (False, UInt16(0))
    var magic = UInt16(pkt[0]) | (UInt16(pkt[1]) << 8)
    if magic != _NET_SNAP_ACK_MAGIC:
        return (False, UInt16(0))
    var seq = UInt16(pkt[2]) | (UInt16(pkt[3]) << 8)
    return (True, seq)


def _apply_received_packet(
    mut world: World,
    mut cfg: NetConfigResource,
    mut rep_res: NetReplicationResource,
    pkt: List[UInt8],
) -> Bool:
    var res = rep_res.receiver.apply_packet(pkt)
    if not res[0]:
        return False
    cfg.last_server_tick = res[1]
    cfg.packets_received += 1
    rep_res.interp.push_snapshot(cfg.now_ms, rep_res.receiver.entities)

    for ri in range(len(rep_res.receiver._removed_eids)):
        var net_id = rep_res.receiver._removed_eids[ri]
        var found = rep_res.net_id_to_entity.get(net_id)
        if found:
            var e = Entity.from_bits(found.value())
            if world.is_alive(e):
                _ = world.despawn(e)
            _ = rep_res.net_id_to_entity.pop(net_id, 0)

    for di in range(len(rep_res.receiver._decoded_eids)):
        var net_id = rep_res.receiver._decoded_eids[di]
        var found = rep_res.net_id_to_entity.get(net_id)
        if not found:
            continue
        var e = Entity.from_bits(found.value())
        if not world.is_alive(e):
            continue
        var snap = rep_res.receiver.entities[net_id]
        var lt_opt = world.get_ptr_mut[LocalTransform](e)
        if lt_opt:
            var r_opt = world.get_ptr[ReplicationComponent](e)
            var do_pos = True
            var do_rot = True
            if r_opt:
                do_pos = r_opt.value()[unsafe_offset=0].replicate_pos
                do_rot = r_opt.value()[unsafe_offset=0].replicate_rot
            if do_pos:
                lt_opt.value()[unsafe_offset=0].translation = snap.pos
            if do_rot:
                lt_opt.value()[unsafe_offset=0].rotation = snap.rot
        var rc_opt = world.get_ptr_mut[ReplicationComponent](e)
        if rc_opt and rc_opt.value()[unsafe_offset=0].replicate_state:
            rc_opt.value()[unsafe_offset=0].state_flags = snap.state_flags
    return True


def _sys_net_poll(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var cfg_opt = resources.get_ptr_mut[NetConfigResource]()
    var rep_opt = resources.get_ptr_mut[NetReplicationResource]()
    var tr_opt = resources.get_ptr_mut[NetTransportResource]()
    if not cfg_opt or not rep_opt or not tr_opt:
        return
    ref cfg = cfg_opt.value()[unsafe_offset=0]
    ref rep = rep_opt.value()[unsafe_offset=0]
    ref tr = tr_opt.value()[unsafe_offset=0]

    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        cfg.now_ms += Float64(max(Float32(0.0), ft_opt.value()[unsafe_offset=0].delta_seconds)) * 1000.0

    # Refresh net_id -> entity map for local entities
    var nid_mask = world.bit_for[NetworkIdentityComponent]()
    var nid_entities = world.query_entities(nid_mask)
    for i in range(len(nid_entities)):
        var e = nid_entities[i]
        var nc_opt = world.get_ptr[NetworkIdentityComponent](e)
        if nc_opt:
            rep.net_id_to_entity[nc_opt.value()[unsafe_offset=0].net_id] = e.to_bits()

    for hi in range(len(tr.virtual_hubs)):
        ref hub = tr.virtual_hubs[hi]
        hub.poll(cfg.now_ms)
        var is_server_hub = (cfg.mode == NET_MODE_SERVER) or (cfg.mode == NET_MODE_VIRTUAL and hub.accept)
        var is_client_hub = (cfg.mode == NET_MODE_CLIENT) or (cfg.mode == NET_MODE_VIRTUAL and not hub.accept)
        if is_client_hub:
            for cid in range(hub.count):
                if not hub.is_open(cid):
                    continue
                var ack_needed = False
                var ack_seq = UInt16(0)
                while True:
                    var msg = hub.pop_unreliable(cid)
                    if not msg[0]:
                        break
                    if _apply_received_packet(world, cfg, rep, msg[1]):
                        ack_needed = True
                        ack_seq = rep.receiver.last_decoded_seq
                if ack_needed:
                    _ = hub.send_unreliable(cid, _encode_snap_ack(ack_seq), cfg.now_ms)
        elif is_server_hub:
            for cid in range(hub.count):
                if not hub.is_open(cid):
                    continue
                while True:
                    var msg = hub.pop_unreliable(cid)
                    if not msg[0]:
                        break
                    var ack = _decode_snap_ack(msg[1])
                    if ack[0]:
                        rep.replicator.notify_client_ack(cid, ack[1])

    for hi in range(len(tr.udp_hubs)):
        ref hub = tr.udp_hubs[hi]
        hub.poll(cfg.now_ms)
        var is_server_hub = (cfg.mode == NET_MODE_SERVER) or (cfg.mode == NET_MODE_VIRTUAL and hub.accept)
        var is_client_hub = (cfg.mode == NET_MODE_CLIENT) or (cfg.mode == NET_MODE_VIRTUAL and not hub.accept)
        if is_client_hub:
            for cid in range(hub.count):
                if not hub.is_open(cid):
                    continue
                var ack_needed = False
                var ack_seq = UInt16(0)
                while True:
                    var msg = hub.pop_unreliable(cid)
                    if not msg[0]:
                        break
                    if _apply_received_packet(world, cfg, rep, msg[1]):
                        ack_needed = True
                        ack_seq = rep.receiver.last_decoded_seq
                if ack_needed:
                    _ = hub.send_unreliable(cid, _encode_snap_ack(ack_seq), cfg.now_ms)
        elif is_server_hub:
            for cid in range(hub.count):
                if not hub.is_open(cid):
                    continue
                while True:
                    var msg = hub.pop_unreliable(cid)
                    if not msg[0]:
                        break
                    var ack = _decode_snap_ack(msg[1])
                    if ack[0]:
                        rep.replicator.notify_client_ack(cid, ack[1])


def _sys_net_fixed_step(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var rep_opt = resources.get_ptr_mut[NetReplicationResource]()
    var fix_opt = resources.get_ptr[FixedTime]()
    if not rep_opt or not fix_opt:
        return
    ref rep = rep_opt.value()[unsafe_offset=0]
    var tick = Int(fix_opt.value()[unsafe_offset=0].tick_count)

    var mask = world.bit_for[NetworkIdentityComponent]().union(world.bit_for[ReplicationComponent]())
    var entities = world.query_entities(mask)
    var poses = List[HitboxPose](capacity=len(entities))
    for i in range(len(entities)):
        var e = entities[i]
        var nc_opt = world.get_ptr[NetworkIdentityComponent](e)
        var rc_opt = world.get_ptr[ReplicationComponent](e)
        if not nc_opt or not rc_opt:
            continue
        if not rc_opt.value()[unsafe_offset=0].record_lag_comp:
            continue
        var pos = Vec3()
        var lt_opt = world.get_ptr[LocalTransform](e)
        if lt_opt:
            pos = lt_opt.value()[unsafe_offset=0].translation
        else:
            var gt_opt = world.get_ptr[GlobalTransform](e)
            if gt_opt:
                pos = gt_opt.value()[unsafe_offset=0].translation
        poses.append(
            HitboxPose.sphere(
                nc_opt.value()[unsafe_offset=0].net_id,
                pos,
                rc_opt.value()[unsafe_offset=0].hitbox_radius,
            )
        )
    if len(poses) > 0:
        rep.lag_comp.record_tick(tick, poses)


def _sys_net_replicate(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var cfg_opt = resources.get_ptr_mut[NetConfigResource]()
    var rep_opt = resources.get_ptr_mut[NetReplicationResource]()
    var tr_opt = resources.get_ptr_mut[NetTransportResource]()
    if not cfg_opt or not rep_opt or not tr_opt:
        return
    ref cfg = cfg_opt.value()[unsafe_offset=0]
    ref rep = rep_opt.value()[unsafe_offset=0]
    ref tr = tr_opt.value()[unsafe_offset=0]

    var any_server = False
    if cfg.mode == NET_MODE_SERVER:
        any_server = True
    elif cfg.mode == NET_MODE_VIRTUAL:
        for hi in range(len(tr.virtual_hubs)):
            if tr.virtual_hubs[hi].accept:
                any_server = True
                break
        if not any_server:
            for hi in range(len(tr.udp_hubs)):
                if tr.udp_hubs[hi].accept:
                    any_server = True
                    break
    if not any_server:
        return

    var live_ids = List[Bool](length=rep.replicator.max_entities, fill=False)
    var mask = world.bit_for[NetworkIdentityComponent]().union(world.bit_for[ReplicationComponent]())
    var entities = world.query_entities(mask)
    for i in range(len(entities)):
        var e = entities[i]
        var nc_opt = world.get_ptr[NetworkIdentityComponent](e)
        var rc_opt = world.get_ptr[ReplicationComponent](e)
        if not nc_opt or not rc_opt:
            continue
        ref nc = nc_opt.value()[unsafe_offset=0]
        ref rc = rc_opt.value()[unsafe_offset=0]
        rep.net_id_to_entity[nc.net_id] = e.to_bits()
        if nc.net_id < 0 or nc.net_id >= rep.replicator.max_entities:
            continue
        live_ids[nc.net_id] = True

        var pos = Vec3()
        var rot = Quat()
        var gt_opt = world.get_ptr[GlobalTransform](e)
        if gt_opt:
            pos = gt_opt.value()[unsafe_offset=0].translation
            rot = gt_opt.value()[unsafe_offset=0].rotation
        else:
            var lt_opt = world.get_ptr[LocalTransform](e)
            if lt_opt:
                pos = lt_opt.value()[unsafe_offset=0].translation
                rot = lt_opt.value()[unsafe_offset=0].rotation

        var vel = Vec3()
        if rc.replicate_vel:
            var rb_opt = world.get_ptr[RigidBodyComponent](e)
            if rb_opt:
                vel = rb_opt.value()[unsafe_offset=0].linear_vel

        rep.replicator.set_entity(
            nc.net_id,
            pos if rc.replicate_pos else Vec3(),
            rot if rc.replicate_rot else Quat.identity(),
            vel,
            rc.state_flags if rc.replicate_state else UInt32(0),
            rc.base_priority,
        )

    for nid in range(rep.replicator.max_entities):
        if rep.replicator.entities[nid].active and not live_ids[nid]:
            rep.replicator.destroy_entity(nid)
            _ = rep.net_id_to_entity.pop(nid, 0)

    rep.replicator.prepare_tick()

    for hi in range(len(tr.virtual_hubs)):
        ref hub = tr.virtual_hubs[hi]
        var is_server_hub = (cfg.mode == NET_MODE_SERVER) or (cfg.mode == NET_MODE_VIRTUAL and hub.accept)
        if not is_server_hub:
            continue
        for cid in range(hub.count):
            if cid >= rep.replicator.max_clients or not hub.is_open(cid):
                continue
            var pkt_res = rep.replicator.build_client_packet(cid)
            if len(pkt_res[1]) > 0:
                if hub.send_unreliable(cid, pkt_res[1], cfg.now_ms):
                    cfg.packets_sent += 1
        hub.poll(cfg.now_ms)

    for hi in range(len(tr.udp_hubs)):
        ref hub = tr.udp_hubs[hi]
        var is_server_hub = (cfg.mode == NET_MODE_SERVER) or (cfg.mode == NET_MODE_VIRTUAL and hub.accept)
        if not is_server_hub:
            continue
        for cid in range(hub.count):
            if cid >= rep.replicator.max_clients or not hub.is_open(cid):
                continue
            var pkt_res = rep.replicator.build_client_packet(cid)
            if len(pkt_res[1]) > 0:
                if hub.send_unreliable(cid, pkt_res[1], cfg.now_ms):
                    cfg.packets_sent += 1
        hub.poll(cfg.now_ms)


def _sys_net_shutdown(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var tr_opt = resources.get_ptr_mut[NetTransportResource]()
    if tr_opt:
        tr_opt.value()[unsafe_offset=0].close()


def _has_net_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


struct NetPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers networking resources, ECS schemas, and poll/lag-comp/replication systems."""
    var max_entities: Int
    var max_clients: Int
    var world_bounds: Float32
    var chunk_size: Float32

    def __init__(
        out self,
        max_entities: Int = 512,
        max_clients: Int = 16,
        world_bounds: Float32 = 512.0,
        chunk_size: Float32 = 64.0,
    ):
        self.max_entities = max_entities
        self.max_clients = max_clients
        self.world_bounds = world_bounds
        self.chunk_size = chunk_size

    def build(self, mut app: EngineApp):
        register_net_components(app.world)
        if not app.resources.has[NetConfigResource]():
            app.insert_resource[NetConfigResource](
                NetConfigResource(
                    max_entities=self.max_entities,
                    max_clients=self.max_clients,
                    world_bounds=self.world_bounds,
                    chunk_size=self.chunk_size,
                )
            )
        if not app.resources.has[NetReplicationResource]():
            app.insert_resource[NetReplicationResource](
                NetReplicationResource(
                    self.max_entities,
                    self.max_clients,
                    self.world_bounds,
                    self.chunk_size,
                )
            )
        if not app.resources.has[NetTransportResource]():
            app.insert_resource[NetTransportResource](NetTransportResource())

        var b_nid = app.world.bit_for[NetworkIdentityComponent]()
        var b_rep = app.world.bit_for[ReplicationComponent]()
        var b_lt = app.world.bit_for[LocalTransform]()
        var b_gt = app.world.bit_for[GlobalTransform]()

        if not _has_net_system(app, STAGE_FIRST, "flint.net.poll"):
            _ = app.add_system(
                STAGE_FIRST,
                "flint.net.poll",
                _sys_net_poll,
                reads=b_nid,
                writes=b_lt.union(b_rep),
                exclusive=True,
            )
        if not _has_net_system(app, STAGE_FIXED_UPDATE, "flint.net.fixed_step"):
            _ = app.add_system(
                STAGE_FIXED_UPDATE,
                "flint.net.fixed_step",
                _sys_net_fixed_step,
                reads=b_nid.union(b_rep).union(b_lt).union(b_gt),
                exclusive=True,
            )
        if not _has_net_system(app, STAGE_POST_UPDATE, "flint.net.replicate"):
            var after_trans_id = -1
            ref post_sys = app.schedule.stages[STAGE_POST_UPDATE].systems
            for i in range(len(post_sys)):
                if post_sys[i].name == "flint.scene.propagate_transforms":
                    after_trans_id = post_sys[i].id
                    break
            _ = app.add_system(
                STAGE_POST_UPDATE,
                "flint.net.replicate",
                _sys_net_replicate,
                reads=b_nid.union(b_rep).union(b_lt).union(b_gt),
                exclusive=True,
                after_system_id=after_trans_id,
            )
        if not _has_net_system(app, STAGE_SHUTDOWN, "flint.net.shutdown"):
            _ = app.add_system(
                STAGE_SHUTDOWN,
                "flint.net.shutdown",
                _sys_net_shutdown,
                exclusive=True,
            )
