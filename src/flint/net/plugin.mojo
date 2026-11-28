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

Client/server session features:
- Spawn-on-first-sight: a snapshot for an unknown `net_id` builds the entity through the
  `NetSpawnFn` registered for its replicated `prefab_id` (`register_spawn_hook`), or a default
  `LocalTransform + GlobalTransform + ReplicationComponent` entity; server despawns remove it.
- Interpolation: with `NetConfigResource.interp_delay_ms > 0`, remote entities render from the
  `SnapshotInterpolationBuffer` on the server clock (`0` keeps legacy snap-to-latest).
- Input upstream: `submit_local_input(tick, bytes)` sends tick-stamped input frames with
  `input_redundancy`-deep redundancy; the server drains them with `pop_client_input(cid)`, which
  acks the tick so `PredictionResource.reconcile_from_server` can rewind/replay.
- Authority: client-authoritative state (`NET_AUTHORITY_CLIENT`/`SHARED`, owned, `is_local`) is
  sent upstream and only accepted when the sender peer owns the entity; RPCs that target an
  entity are only accepted from its owner.
- Extra POD components: `register_net_pod_component[C](rep, slot)` replicates up to
  `REPL_EXTRA_SLOTS` `NetPodComponent`s per entity (bit-packed, sent only when changed vs. the
  client's acked baseline). Reliable RPCs: `send_rpc` / `drain_rpcs`.
- Peer ids: the server is peer `0`; the client on channel `cid` is peer `cid + 1`
  (`net_peer_id_for_client`), announced to it with a reliable welcome message.
"""
from std.collections import Dict, List
from std.math import isfinite, max, min
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
from flint.ecs.world import Component
from flint.net.bitpack import BitReader, BitWriter
from flint.net.hub import NetHub
from flint.net.lag_comp import HitboxPose, LagCompHistory
from flint.net.prediction import (
    ClientPredictionBuffer,
    PredictedState,
    PredictionStepper,
)
from flint.net.replicator import (
    REPL_EXTRA_SLOTS,
    ReplicatedEntity,
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
comptime _NET_INPUT_MAGIC: UInt16 = 0x4936  # client -> server tick-stamped input frames (raw)
comptime _NET_CSTATE_MAGIC: UInt16 = 0x5336  # client -> server client-authoritative state (raw)
comptime _NET_MSG_MAGIC: UInt16 = 0x4D36  # reliable envelope
comptime _NET_MSG_WELCOME: UInt8 = 1
comptime _NET_MSG_RPC: UInt8 = 2
comptime NET_INPUT_RING: Int = 64
comptime NET_SERVER_INPUT_QUEUE: Int = 128
comptime NET_INPUT_MAX_BYTES: Int = 255
comptime NET_RPC_TARGET_NONE: Int = -1
comptime NET_PEER_ALL: Int = -1

comptime NetSpawnFn = def(Pointer[World, MutAnyOrigin], UInt32, Int) thin -> UInt64
"""Client spawn hook: `(world, prefab_id, net_id) -> Entity.to_bits()`. `NetworkIdentityComponent`
is inserted/overwritten by the plugin afterwards."""
comptime NetPodEncodeFn = def(Pointer[World, MutAnyOrigin], UInt64, Int) thin -> Bool
comptime NetPodDecodeFn = def(Pointer[World, MutAnyOrigin], UInt64, Int) thin -> Bool


trait NetPodComponent(Component):
    """Plain-old-data ECS component replicated as an extra snapshot payload slot."""
    def net_encode(self, mut writer: BitWriter): ...

    @staticmethod
    def net_decode(mut reader: BitReader) -> Self: ...


def _pod_encode[C: NetPodComponent](
    world_ptr: Pointer[World, MutAnyOrigin], entity_bits: UInt64, writer_addr: Int
) -> Bool:
    ref world = world_ptr[unsafe_offset=0]
    var c_opt = world.get_ptr[C](Entity.from_bits(entity_bits))
    if not c_opt:
        return False
    var wp = Pointer[BitWriter, MutAnyOrigin](unsafe_from_address=writer_addr)
    c_opt.value()[unsafe_offset=0].net_encode(wp[unsafe_offset=0])
    return True


def _pod_decode[C: NetPodComponent](
    world_ptr: Pointer[World, MutAnyOrigin], entity_bits: UInt64, reader_addr: Int
) -> Bool:
    ref world = world_ptr[unsafe_offset=0]
    var e = Entity.from_bits(entity_bits)
    if not world.is_alive(e):
        return False
    var rp = Pointer[BitReader, MutAnyOrigin](unsafe_from_address=reader_addr)
    var v = C.net_decode(rp[unsafe_offset=0])
    if rp[unsafe_offset=0].overflowed:
        return False
    var c_opt = world.get_ptr_mut[C](e)
    if c_opt:
        c_opt.value()[unsafe_offset=0] = v^
        return True
    return world.insert[C](e, v^)


def _noop_pod_encode(
    _world_ptr: Pointer[World, MutAnyOrigin], _entity_bits: UInt64, _addr: Int
) -> Bool:
    return False


def _noop_pod_decode(
    _world_ptr: Pointer[World, MutAnyOrigin], _entity_bits: UInt64, _addr: Int
) -> Bool:
    return False


def default_net_spawn(
    world_ptr: Pointer[World, MutAnyOrigin], prefab_id: UInt32, net_id: Int
) -> UInt64:
    """Default client spawn: `LocalTransform + GlobalTransform + ReplicationComponent`."""
    _ = prefab_id
    _ = net_id
    ref world = world_ptr[unsafe_offset=0]
    var e = world.spawn()
    _ = world.insert[LocalTransform](e, LocalTransform(Vec3()))
    _ = world.insert[GlobalTransform](e, GlobalTransform())
    _ = world.insert[ReplicationComponent](e, ReplicationComponent())
    return e.to_bits()


@always_inline
def net_peer_id_for_client(cid: Int) -> Int:
    """Peer id the server assigns to the client on channel `cid` (server itself is peer 0)."""
    return cid + 1


@fieldwise_init
struct NetRpcMessage(Copyable, Movable):
    """Reliable remote procedure call / event (validated against entity ownership on the server)."""
    var from_peer: Int
    var rpc_id: UInt16
    var target_net_id: Int
    var payload: List[UInt8]


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
    var snapshot_phase: Float64
    var snapshot_due: Bool
    var now_ms: Float64
    var local_peer_id: Int
    var last_server_tick: UInt16
    var packets_sent: UInt64
    var packets_received: UInt64
    var interp_delay_ms: Float64
    var input_redundancy: Int
    var predict_owned_entities: Bool

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
        self.snapshot_phase = 0
        self.snapshot_due = False
        self.now_ms = 0.0
        self.local_peer_id = local_peer_id
        self.last_server_tick = 0
        self.packets_sent = 0
        self.packets_received = 0
        # `> 0`: remote entities render from the interpolation buffer this far behind the server.
        self.interp_delay_ms = 0.0
        # Input frames repeated per upstream packet (survives `input_redundancy - 1` lost packets).
        self.input_redundancy = 8
        # Owned, server-authoritative entities are marked `is_local` (predicted by the game via
        # `PredictionResource`) and are not overwritten by snapshots/interpolation.
        self.predict_owned_entities = False


struct NetReplicationResource(Resource, Movable):
    """Owns AOI `SnapshotReplicator`, client receiver, interpolation buffer, and `LagCompHistory`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.net.NetReplicationResource")
    var replicator: SnapshotReplicator
    var receiver: SnapshotClientReceiver
    var interp: SnapshotInterpolationBuffer
    var lag_comp: LagCompHistory
    var net_id_to_entity: Dict[Int, UInt64]
    # Spawn hooks (client)
    var spawn_hook_prefabs: List[UInt32]
    var spawn_hook_fns: List[NetSpawnFn]
    var default_spawn_fn: NetSpawnFn
    # Extra POD component slots
    var pod_slot_used: List[Bool]
    var pod_encode_fns: List[NetPodEncodeFn]
    var pod_decode_fns: List[NetPodDecodeFn]
    # Interpolation on the server clock (client)
    var interp_scratch: List[ReplicatedEntity]
    var has_clock_offset: Bool
    var clock_offset_ms: Float64
    var has_server_clock: Bool
    var last_raw_server_ms: UInt32
    var server_clock_ms: Float64
    # Input upstream (client)
    var local_input_ticks: List[Int]
    var local_input_data: List[List[UInt8]]
    var input_packets_sent: Int
    # Input downstream (server), indexed by channel id
    var client_input_ticks: List[List[Int]]
    var client_input_data: List[List[List[UInt8]]]
    var client_last_input_tick: List[Int]
    var peer_welcomed: List[Bool]
    # Reliable RPCs
    var rpc_out_peer: List[Int]
    var rpc_out_bytes: List[List[UInt8]]
    var rpc_in: List[NetRpcMessage]
    # Telemetry
    var rejected_writes: Int
    var accepted_writes: Int
    var rejected_rpcs: Int
    var spawned_remote: Int

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
        self.spawn_hook_prefabs = List[UInt32]()
        self.spawn_hook_fns = List[NetSpawnFn]()
        self.default_spawn_fn = default_net_spawn
        self.pod_slot_used = List[Bool](length=REPL_EXTRA_SLOTS, fill=False)
        self.pod_encode_fns = List[NetPodEncodeFn]()
        self.pod_decode_fns = List[NetPodDecodeFn]()
        for _ in range(REPL_EXTRA_SLOTS):
            self.pod_encode_fns.append(_noop_pod_encode)
            self.pod_decode_fns.append(_noop_pod_decode)
        self.interp_scratch = List[ReplicatedEntity]()
        self.has_clock_offset = False
        self.clock_offset_ms = 0.0
        self.has_server_clock = False
        self.last_raw_server_ms = 0
        self.server_clock_ms = 0.0
        self.local_input_ticks = List[Int]()
        self.local_input_data = List[List[UInt8]]()
        self.input_packets_sent = 0
        var mc = max(1, max_clients)
        self.client_input_ticks = List[List[Int]](capacity=mc)
        self.client_input_data = List[List[List[UInt8]]](capacity=mc)
        for _ in range(mc):
            self.client_input_ticks.append(List[Int]())
            self.client_input_data.append(List[List[UInt8]]())
        self.client_last_input_tick = List[Int](length=mc, fill=-1)
        self.peer_welcomed = List[Bool](length=mc, fill=False)
        self.rpc_out_peer = List[Int]()
        self.rpc_out_bytes = List[List[UInt8]]()
        self.rpc_in = List[NetRpcMessage]()
        self.rejected_writes = 0
        self.accepted_writes = 0
        self.rejected_rpcs = 0
        self.spawned_remote = 0

    def register_spawn_hook(mut self, prefab_id: UInt32, hook: NetSpawnFn):
        """Client: builds entities whose replicated `prefab_id` matches (replaces an existing hook)."""
        for i in range(len(self.spawn_hook_prefabs)):
            if self.spawn_hook_prefabs[i] == prefab_id:
                self.spawn_hook_fns[i] = hook
                return
        self.spawn_hook_prefabs.append(prefab_id)
        self.spawn_hook_fns.append(hook)

    def set_default_spawn_hook(mut self, hook: NetSpawnFn):
        """Client: spawn hook used for prefab ids without a registered hook."""
        self.default_spawn_fn = hook

    def _spawn_hook_for(self, prefab_id: UInt32) -> NetSpawnFn:
        for i in range(len(self.spawn_hook_prefabs)):
            if self.spawn_hook_prefabs[i] == prefab_id:
                return self.spawn_hook_fns[i]
        return self.default_spawn_fn

    def register_pod_slot(
        mut self, slot: Int, enc: NetPodEncodeFn, dec: NetPodDecodeFn
    ) -> Bool:
        if slot < 0 or slot >= REPL_EXTRA_SLOTS:
            return False
        self.pod_slot_used[slot] = True
        self.pod_encode_fns[slot] = enc
        self.pod_decode_fns[slot] = dec
        return True

    def submit_local_input(mut self, tick: Int, payload: List[UInt8]) -> Bool:
        """Client: queues one tick-stamped input frame (`<= 255` bytes, strictly increasing ticks)
        for redundant upstream delivery in `STAGE_POST_UPDATE`."""
        if tick < 0 or len(payload) > NET_INPUT_MAX_BYTES:
            return False
        var n = len(self.local_input_ticks)
        if n > 0 and tick <= self.local_input_ticks[n - 1]:
            return False
        if n >= NET_INPUT_RING:
            var nt = List[Int](capacity=NET_INPUT_RING)
            var nd = List[List[UInt8]](capacity=NET_INPUT_RING)
            for i in range(1, n):
                nt.append(self.local_input_ticks[i])
                nd.append(self.local_input_data[i].copy())
            self.local_input_ticks = nt^
            self.local_input_data = nd^
        self.local_input_ticks.append(tick)
        self.local_input_data.append(payload.copy())
        return True

    def pending_client_inputs(self, cid: Int) -> Int:
        if cid < 0 or cid >= len(self.client_input_ticks):
            return 0
        return len(self.client_input_ticks[cid])

    def pop_client_input(mut self, cid: Int) -> Tuple[Bool, Int, List[UInt8]]:
        """Server: oldest unconsumed input frame from client `cid`. Consuming it acks the tick, which
        the next snapshot carries back for client reconciliation."""
        if cid < 0 or cid >= len(self.client_input_ticks):
            return (False, -1, List[UInt8]())
        if len(self.client_input_ticks[cid]) == 0:
            return (False, -1, List[UInt8]())
        var tick = self.client_input_ticks[cid][0]
        var data = self.client_input_data[cid][0].copy()
        var nt = List[Int]()
        var nd = List[List[UInt8]]()
        for i in range(1, len(self.client_input_ticks[cid])):
            nt.append(self.client_input_ticks[cid][i])
            nd.append(self.client_input_data[cid][i].copy())
        self.client_input_ticks[cid] = nt^
        self.client_input_data[cid] = nd^
        self.replicator.set_client_acked_input_seq(cid, tick)
        return (True, tick, data^)

    def send_rpc(
        mut self,
        target_peer: Int,
        rpc_id: UInt16,
        target_net_id: Int,
        payload: List[UInt8],
    ):
        """Queues a reliable RPC. Server: `target_peer` = client peer id or `NET_PEER_ALL`.
        Client: always goes to the server. `target_net_id` (or `NET_RPC_TARGET_NONE`) is checked
        against entity ownership by the server."""
        var w = BitWriter(16 + len(payload))
        w.write_u16(_NET_MSG_MAGIC)
        w.write_u8(_NET_MSG_RPC)
        w.write_u16(rpc_id)
        w.write_u32(UInt32(target_net_id & 0xFFFFFFFF))
        w.write_u32(UInt32(len(payload)))
        w.write_bytes_aligned(payload)
        self.rpc_out_peer.append(target_peer)
        self.rpc_out_bytes.append(w.finish())

    def drain_rpcs(mut self) -> List[NetRpcMessage]:
        """Takes every RPC received (and authorised) since the last call."""
        var out = self.rpc_in^
        self.rpc_in = List[NetRpcMessage]()
        return out^

    def _reset_client_slot(mut self, cid: Int):
        if cid < 0 or cid >= len(self.client_input_ticks):
            return
        self.client_input_ticks[cid].clear()
        self.client_input_data[cid].clear()
        self.client_last_input_tick[cid] = -1
        self.peer_welcomed[cid] = False


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

    def predict_local(
        mut self,
        mut rep: NetReplicationResource,
        tick: Int,
        input: Self.InputT,
        predicted: PredictedState,
        dt: Float32 = Float32(1.0 / 60.0),
    ) -> Bool:
        """Client: records the locally simulated result of `input` at `tick` and queues the input
        frame upstream (redundantly) for the server."""
        var w = BitWriter(64)
        input.encode_bits(w)
        if not rep.submit_local_input(tick, w.finish()):
            return False
        self.buffer.push_predicted(tick, input, predicted, tick, dt)
        return True

    def _auth_state(
        self, rep: NetReplicationResource, net_id: Int
    ) -> Tuple[Bool, Int, PredictedState]:
        if not rep.receiver.has_acked_input_seq:
            return (False, -1, PredictedState())
        if net_id < 0 or net_id >= len(rep.receiver.entities):
            return (False, -1, PredictedState())
        var acked = rep.receiver.last_acked_input_seq
        if self.buffer.has_acked and acked <= self.buffer.last_acked_input_seq:
            return (False, acked, PredictedState())
        var e = rep.receiver.entities[net_id]
        if not e.active:
            return (False, acked, PredictedState())
        return (True, acked, PredictedState(e.pos, e.vel, e.rot, e.state_flags))

    def reconcile_from_server(
        mut self, rep: NetReplicationResource, net_id: Int
    ) -> Tuple[Bool, PredictedState]:
        """Client: when a snapshot acked a newer input tick, rewinds to the server state of `net_id`
        and replays the unacked inputs (delta replay + `VisualSmoother` correction)."""
        var a = self._auth_state(rep, net_id)
        if not a[0]:
            return (False, self.buffer.latest_predicted_state())
        var st = self.buffer.reconcile_smoothed(a[1], a[2], self.smoother)
        return (True, st)

    def reconcile_from_server_with[
        StepperT: PredictionStepper
    ](
        mut self,
        rep: NetReplicationResource,
        net_id: Int,
        mut stepper: StepperT,
        default_dt: Float32 = Float32(1.0 / 60.0),
    ) -> Tuple[Bool, PredictedState]:
        """As `reconcile_from_server`, but re-simulates unacked inputs with `stepper`."""
        var a = self._auth_state(rep, net_id)
        if not a[0]:
            return (False, self.buffer.latest_predicted_state())
        var prev = self.buffer.latest_predicted_state()
        var had_prev = self.buffer.has_latest_state
        var st = self.buffer.reconcile_with[StepperT](a[1], a[2], stepper, default_dt)
        if had_prev:
            self.smoother.record_correction(
                self.smoother.smooth_pos(prev.pos), st.pos,
                self.smoother.smooth_rot(prev.rot), st.rot,
            )
        return (True, st)

    def latest(self) -> PredictedState:
        return self.buffer.latest_predicted_state()

    def visual_pos(self) -> Vec3:
        """Presentation position: latest prediction plus the decaying correction offset."""
        return self.smoother.smooth_pos(self.buffer.latest_predicted_state().pos)


def pop_client_input_as[
    InputT: RollbackInput
](mut rep: NetReplicationResource, cid: Int) -> Tuple[Bool, Int, InputT]:
    """Server: `pop_client_input` decoded through `InputT.decode_bits`."""
    var res = rep.pop_client_input(cid)
    var inp = InputT()
    if not res[0]:
        return (False, -1, inp)
    var r = BitReader(res[2])
    inp.decode_bits(r)
    if r.overflowed:
        return (False, res[1], InputT())
    return (True, res[1], inp)


def register_net_pod_component[
    C: NetPodComponent
](mut rep: NetReplicationResource, slot: Int) -> Bool:
    """Replicates component `C` in extra payload `slot` (register identically on server and client)."""
    return rep.register_pod_slot(slot, _pod_encode[C], _pod_decode[C])


# ---------------------------------------------------------------------------
# 3 Serializable ECS Networking Components
# ---------------------------------------------------------------------------

struct NetworkIdentityComponent(SerializableComponent, ImplicitlyCopyable):
    """Network identity and authority ownership for a replicated ECS entity."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.net.NetworkIdentityComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "NetworkIdentityComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("NNID")
    comptime SCHEMA_VERSION: Int = 2

    var net_id: Int
    var owner_peer_id: Int
    var is_local: Bool
    var authority: UInt8
    var prefab_id: UInt32

    def __init__(
        out self,
        net_id: Int = 0,
        owner_peer_id: Int = 0,
        is_local: Bool = False,
        authority: UInt8 = NET_AUTHORITY_SERVER,
        prefab_id: UInt32 = 0,
    ):
        self.net_id = net_id
        self.owner_peer_id = owner_peer_id
        self.is_local = is_local
        self.authority = authority
        self.prefab_id = prefab_id

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.net_id)
        out.integer(self.owner_peer_id)
        out.boolean(self.is_local)
        out.u8(self.authority)
        out.u32(self.prefab_id)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        var nid = reader.integer()
        var opid = reader.integer()
        var loc = reader.boolean()
        var auth = reader.u8()
        var pf = UInt32(0)
        if version >= 2:
            pf = reader.u32()
        if reader.failed:
            raise Error("NetworkIdentityComponent: truncated blob")
        return Self(nid, opid, loc, auth, pf)

    def write_json(self) -> String:
        return (
            "{\"net_id\": " + String(self.net_id)
            + ", \"owner_peer_id\": " + String(self.owner_peer_id)
            + ", \"is_local\": " + ("true" if self.is_local else "false")
            + ", \"authority\": " + String(Int(self.authority))
            + ", \"prefab_id\": " + String(Int(self.prefab_id))
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var nid = doc.integer(doc.object_get(node, "net_id"), 0)
        var opid = doc.integer(doc.object_get(node, "owner_peer_id"), 0)
        var loc = doc.boolean(doc.object_get(node, "is_local"), False)
        var auth = UInt8(doc.integer(doc.object_get(node, "authority"), Int(NET_AUTHORITY_SERVER)))
        var pf = UInt32(doc.integer(doc.object_get(node, "prefab_id"), 0))
        return Self(nid, opid, loc, auth, pf)


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


@always_inline
def _u32_to_signed(v: UInt32) -> Int:
    var x = Int(v)
    if x >= 0x80000000:
        x -= 0x100000000
    return x


@always_inline
def _f32_ok(v: Float32) -> Bool:
    return (v - v) == Float32(0.0)


def _update_server_clock(mut rep: NetReplicationResource, now_ms: Float64) -> Float64:
    """Unwraps the 32-bit server millisecond clock and tracks the (min-latency) local offset."""
    var raw = rep.receiver.last_server_time_ms
    if not rep.has_server_clock:
        rep.server_clock_ms = Float64(Int(raw))
        rep.has_server_clock = True
    else:
        var diff = Int(raw) - Int(rep.last_raw_server_ms)
        if diff > 0x7FFFFFFF:
            diff -= 0x100000000
        elif diff < -0x80000000:
            diff += 0x100000000
        rep.server_clock_ms += Float64(diff)
    rep.last_raw_server_ms = raw
    var sample = now_ms - rep.server_clock_ms
    if not rep.has_clock_offset:
        rep.clock_offset_ms = sample
        rep.has_clock_offset = True
    elif sample < rep.clock_offset_ms:
        rep.clock_offset_ms = sample
    else:
        rep.clock_offset_ms += (sample - rep.clock_offset_ms) * 0.01
    return rep.server_clock_ms


def _local_flag(cfg: NetConfigResource, owner_peer_id: Int, authority: UInt8) -> Bool:
    if cfg.local_peer_id <= 0 or owner_peer_id != cfg.local_peer_id:
        return False
    if authority == NET_AUTHORITY_SERVER:
        return cfg.predict_owned_entities
    return True


def _sync_identity(
    mut world: World, e: Entity, cfg: NetConfigResource, snap: ReplicatedEntity
):
    var ident = NetworkIdentityComponent(
        snap.net_id,
        snap.owner_peer_id,
        _local_flag(cfg, snap.owner_peer_id, snap.authority),
        snap.authority,
        snap.prefab_id,
    )
    var nc_opt = world.get_ptr_mut[NetworkIdentityComponent](e)
    if nc_opt:
        nc_opt.value()[unsafe_offset=0] = ident
    else:
        _ = world.insert[NetworkIdentityComponent](e, ident)


def _is_local_entity(world: World, e: Entity) -> Bool:
    var nc_opt = world.get_ptr[NetworkIdentityComponent](e)
    if nc_opt:
        return nc_opt.value()[unsafe_offset=0].is_local
    return False


def _write_snap_transform(mut world: World, e: Entity, snap: ReplicatedEntity):
    var lt_opt = world.get_ptr_mut[LocalTransform](e)
    if not lt_opt:
        return
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


def _mapped_entity(world: World, rep: NetReplicationResource, net_id: Int) -> Tuple[Bool, Entity]:
    var found = rep.net_id_to_entity.get(net_id)
    if not found:
        return (False, Entity())
    var e = Entity.from_bits(found.value())
    if not world.is_alive(e):
        return (False, Entity())
    return (True, e)


def _apply_received_packet(
    world_ptr: Pointer[World, MutAnyOrigin],
    mut cfg: NetConfigResource,
    mut rep_res: NetReplicationResource,
    pkt: List[UInt8],
) -> Bool:
    ref world = world_ptr[unsafe_offset=0]
    var res = rep_res.receiver.apply_packet(pkt)
    if not res[0]:
        return False
    cfg.last_server_tick = res[1]
    cfg.packets_received += 1
    var snap_t = cfg.now_ms
    if rep_res.receiver.has_server_time:
        snap_t = _update_server_clock(rep_res, cfg.now_ms)
    rep_res.interp.push_snapshot(snap_t, rep_res.receiver.entities)

    for ri in range(len(rep_res.receiver._removed_eids)):
        var net_id = rep_res.receiver._removed_eids[ri]
        var found = rep_res.net_id_to_entity.get(net_id)
        if found:
            var e = Entity.from_bits(found.value())
            if world.is_alive(e):
                _ = world.despawn(e)
            _ = rep_res.net_id_to_entity.pop(net_id, 0)

    # Ownership / authority / prefab changes on entities the client already has
    for si in range(len(rep_res.receiver._spawn_updates)):
        var net_id = rep_res.receiver._spawn_updates[si]
        var m = _mapped_entity(world, rep_res, net_id)
        if m[0]:
            _sync_identity(world, m[1], cfg, rep_res.receiver.entities[net_id])

    var interp_on = cfg.interp_delay_ms > 0.0
    for di in range(len(rep_res.receiver._decoded_eids)):
        var net_id = rep_res.receiver._decoded_eids[di]
        var snap = rep_res.receiver.entities[net_id]
        var m = _mapped_entity(world, rep_res, net_id)
        var e = m[1]
        if not m[0]:
            # Spawn-on-first-sight through the prefab's registered hook
            var hook = rep_res._spawn_hook_for(snap.prefab_id)
            e = Entity.from_bits(hook(world_ptr, snap.prefab_id, net_id))
            if not world.is_alive(e):
                continue
            _sync_identity(world, e, cfg, snap)
            rep_res.net_id_to_entity[net_id] = e.to_bits()
            rep_res.spawned_remote += 1
            # A fresh entity starts at the snapshot pose even when interpolating.
            _write_snap_transform(world, e, snap)
        if _is_local_entity(world, e):
            continue
        if not interp_on:
            _write_snap_transform(world, e, snap)
        var rc_opt = world.get_ptr_mut[ReplicationComponent](e)
        if rc_opt and rc_opt.value()[unsafe_offset=0].replicate_state:
            rc_opt.value()[unsafe_offset=0].state_flags = snap.state_flags

    # Extra POD component payloads
    for xi in range(len(rep_res.receiver._extra_updates)):
        var idx = rep_res.receiver._extra_updates[xi]
        var eid = idx // REPL_EXTRA_SLOTS
        var slot = idx % REPL_EXTRA_SLOTS
        if not rep_res.pod_slot_used[slot]:
            continue
        if len(rep_res.receiver.extra_data[idx]) == 0:
            continue
        var m = _mapped_entity(world, rep_res, eid)
        if not m[0] or _is_local_entity(world, m[1]):
            continue
        var reader = BitReader(rep_res.receiver.extra_data[idx])
        var dec = rep_res.pod_decode_fns[slot]
        _ = dec(world_ptr, m[1].to_bits(), Int(Pointer(to=reader)))
        _ = reader.overflowed
    return True


def _apply_interpolation(
    mut world: World, cfg: NetConfigResource, mut rep: NetReplicationResource
):
    """Writes remote entity poses sampled `interp_delay_ms` behind the estimated server clock."""
    if cfg.interp_delay_ms <= 0.0 or rep.interp.count == 0:
        return
    rep.interp.interp_delay_ms = cfg.interp_delay_ms
    var render_t = cfg.now_ms - (rep.clock_offset_ms if rep.has_clock_offset else 0.0)
    var scratch = List[ReplicatedEntity]()
    if not rep.interp.sample(render_t, scratch):
        return
    for nid in range(len(scratch)):
        if not scratch[nid].active:
            continue
        var m = _mapped_entity(world, rep, nid)
        if not m[0] or _is_local_entity(world, m[1]):
            continue
        _write_snap_transform(world, m[1], scratch[nid])


def _client_handle_reliable(
    mut world: World,
    mut cfg: NetConfigResource,
    mut rep: NetReplicationResource,
    pkt: List[UInt8],
):
    var r = BitReader(pkt)
    if r.read_u16() != _NET_MSG_MAGIC:
        return
    var kind = r.read_u8()
    if kind == _NET_MSG_WELCOME:
        var peer = Int(r.read_u16())
        if r.overflowed:
            return
        cfg.local_peer_id = peer
        var nid_entities = world.query_entities(world.bit_for[NetworkIdentityComponent]())
        for i in range(len(nid_entities)):
            var nc_opt = world.get_ptr_mut[NetworkIdentityComponent](nid_entities[i])
            if nc_opt:
                ref nc = nc_opt.value()[unsafe_offset=0]
                if nc.owner_peer_id == peer:
                    nc.is_local = _local_flag(cfg, nc.owner_peer_id, nc.authority)
    elif kind == _NET_MSG_RPC:
        var rpc_id = r.read_u16()
        var target = _u32_to_signed(r.read_u32())
        var n = Int(r.read_u32())
        var payload = r.read_bytes_aligned(n)
        if r.overflowed:
            return
        # Server is the authority: every well-formed server RPC is accepted.
        rep.rpc_in.append(NetRpcMessage(0, rpc_id, target, payload^))


def _server_owner_check(
    world: World, rep: NetReplicationResource, net_id: Int, peer: Int, need_client_authority: Bool
) -> Bool:
    var m = _mapped_entity(world, rep, net_id)
    if not m[0]:
        return False
    var nc_opt = world.get_ptr[NetworkIdentityComponent](m[1])
    if not nc_opt:
        return False
    ref nc = nc_opt.value()[unsafe_offset=0]
    if nc.owner_peer_id != peer:
        return False
    if need_client_authority:
        return nc.authority == NET_AUTHORITY_CLIENT or nc.authority == NET_AUTHORITY_SHARED
    return True


def _server_handle_reliable(
    world: World, mut rep: NetReplicationResource, cid: Int, pkt: List[UInt8]
):
    var r = BitReader(pkt)
    if r.read_u16() != _NET_MSG_MAGIC:
        return
    var kind = r.read_u8()
    if kind != _NET_MSG_RPC:
        return
    var rpc_id = r.read_u16()
    var target = _u32_to_signed(r.read_u32())
    var n = Int(r.read_u32())
    var payload = r.read_bytes_aligned(n)
    if r.overflowed:
        return
    var peer = net_peer_id_for_client(cid)
    if target != NET_RPC_TARGET_NONE and not _server_owner_check(world, rep, target, peer, False):
        rep.rejected_rpcs += 1
        return
    rep.rpc_in.append(NetRpcMessage(peer, rpc_id, target, payload^))


def _server_ingest_inputs(mut rep: NetReplicationResource, cid: Int, pkt: List[UInt8]):
    if cid < 0 or cid >= len(rep.client_input_ticks):
        return
    var r = BitReader(pkt)
    _ = r.read_u16()
    var n = Int(r.read_u8())
    for _ in range(n):
        var tick = Int(r.read_u32())
        var ln = Int(r.read_u8())
        var data = r.read_bytes_aligned(ln)
        if r.overflowed:
            return
        if tick <= rep.client_last_input_tick[cid]:
            continue
        rep.client_last_input_tick[cid] = tick
        rep.client_input_ticks[cid].append(tick)
        rep.client_input_data[cid].append(data^)
    var q = len(rep.client_input_ticks[cid])
    if q > NET_SERVER_INPUT_QUEUE:
        var drop = q - NET_SERVER_INPUT_QUEUE
        var nt = List[Int](capacity=NET_SERVER_INPUT_QUEUE)
        var nd = List[List[UInt8]](capacity=NET_SERVER_INPUT_QUEUE)
        for i in range(drop, q):
            nt.append(rep.client_input_ticks[cid][i])
            nd.append(rep.client_input_data[cid][i].copy())
        rep.client_input_ticks[cid] = nt^
        rep.client_input_data[cid] = nd^


def _server_apply_client_state(
    mut world: World, mut rep: NetReplicationResource, cid: Int, pkt: List[UInt8]
):
    """Client-authoritative transforms: accepted only for entities the sender peer owns with
    `NET_AUTHORITY_CLIENT`/`SHARED`; everything else is counted in `rejected_writes`."""
    var r = BitReader(pkt)
    _ = r.read_u16()
    var n = Int(r.read_u8())
    var peer = net_peer_id_for_client(cid)
    for _ in range(n):
        var net_id = Int(r.read_u32())
        var px = r.read_f32_raw()
        var py = r.read_f32_raw()
        var pz = r.read_f32_raw()
        var qx = r.read_f32_raw()
        var qy = r.read_f32_raw()
        var qz = r.read_f32_raw()
        var qw = r.read_f32_raw()
        if r.overflowed:
            return
        var finite = (
            _f32_ok(px) and _f32_ok(py) and _f32_ok(pz)
            and _f32_ok(qx) and _f32_ok(qy) and _f32_ok(qz) and _f32_ok(qw)
        )
        if not finite or not _server_owner_check(world, rep, net_id, peer, True):
            rep.rejected_writes += 1
            continue
        var m = _mapped_entity(world, rep, net_id)
        var lt_opt = world.get_ptr_mut[LocalTransform](m[1])
        if not lt_opt:
            rep.rejected_writes += 1
            continue
        lt_opt.value()[unsafe_offset=0].translation = Vec3(px, py, pz)
        lt_opt.value()[unsafe_offset=0].rotation = Quat(qx, qy, qz, qw).normalized()
        rep.accepted_writes += 1


def _server_handle_unreliable(
    mut world: World, mut rep: NetReplicationResource, cid: Int, pkt: List[UInt8]
):
    if len(pkt) < 2:
        return
    var magic = UInt16(pkt[0]) | (UInt16(pkt[1]) << 8)
    if magic == _NET_SNAP_ACK_MAGIC:
        var ack = _decode_snap_ack(pkt)
        if ack[0]:
            rep.replicator.notify_client_ack(cid, ack[1])
        return
    # BitWriter u16 fields are little-endian on the wire
    if magic == _NET_INPUT_MAGIC:
        _server_ingest_inputs(rep, cid, pkt)
    elif magic == _NET_CSTATE_MAGIC:
        _server_apply_client_state(world, rep, cid, pkt)


def _encode_welcome(peer_id: Int) -> List[UInt8]:
    var w = BitWriter(8)
    w.write_u16(_NET_MSG_MAGIC)
    w.write_u8(_NET_MSG_WELCOME)
    w.write_u16(UInt16(peer_id & 0xFFFF))
    return w.finish()


def _server_after_hub_poll[T: NetTransport & Deinitable](
    hub: NetHub[T], mut rep: NetReplicationResource
):
    """Fresh / reused channel slots start a clean replication + input session."""
    for i in range(len(hub.accepted)):
        var cid = hub.accepted[i]
        rep._reset_client_slot(cid)
        rep.replicator.reset_client(cid)


def _poll_hub[T: NetTransport & Deinitable](
    mut hub: NetHub[T],
    world_ptr: Pointer[World, MutAnyOrigin],
    mut cfg: NetConfigResource,
    mut rep: NetReplicationResource,
):
    ref world = world_ptr[unsafe_offset=0]
    hub.poll(cfg.now_ms)
    var is_server_hub = (cfg.mode == NET_MODE_SERVER) or (cfg.mode == NET_MODE_VIRTUAL and hub.accept)
    var is_client_hub = (cfg.mode == NET_MODE_CLIENT) or (cfg.mode == NET_MODE_VIRTUAL and not hub.accept)
    if is_client_hub:
        for cid in range(hub.count):
            if not hub.is_open(cid):
                continue
            while True:
                var rel = hub.pop_reliable(cid)
                if not rel[0]:
                    break
                _client_handle_reliable(world, cfg, rep, rel[1])
            var ack_needed = False
            var ack_seq = UInt16(0)
            while True:
                var msg = hub.pop_unreliable(cid)
                if not msg[0]:
                    break
                if _apply_received_packet(world_ptr, cfg, rep, msg[1]):
                    ack_needed = True
                    ack_seq = rep.receiver.last_decoded_seq
            if ack_needed:
                _ = hub.send_unreliable(cid, _encode_snap_ack(ack_seq), cfg.now_ms)
    elif is_server_hub:
        _server_after_hub_poll(hub, rep)
        for cid in range(hub.count):
            if not hub.is_open(cid):
                if cid < len(rep.peer_welcomed) and (
                    rep.peer_welcomed[cid] or rep.client_last_input_tick[cid] >= 0
                ):
                    rep._reset_client_slot(cid)
                continue
            if cid < len(rep.peer_welcomed) and not rep.peer_welcomed[cid]:
                if hub.send_reliable(cid, _encode_welcome(net_peer_id_for_client(cid))):
                    rep.peer_welcomed[cid] = True
            while True:
                var msg = hub.pop_unreliable(cid)
                if not msg[0]:
                    break
                _server_handle_unreliable(world, rep, cid, msg[1])
            while True:
                var raw = hub.pop_unreliable_raw(cid)
                if not raw[0]:
                    break
                _server_handle_unreliable(world, rep, cid, raw[1])
            while True:
                var rel = hub.pop_reliable(cid)
                if not rel[0]:
                    break
                _server_handle_reliable(world, rep, cid, rel[1])


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

    var any_client = cfg.mode == NET_MODE_CLIENT
    for hi in range(len(tr.virtual_hubs)):
        _poll_hub(tr.virtual_hubs[hi], world_ptr, cfg, rep)
        if cfg.mode == NET_MODE_VIRTUAL and not tr.virtual_hubs[hi].accept:
            any_client = True
    for hi in range(len(tr.udp_hubs)):
        _poll_hub(tr.udp_hubs[hi], world_ptr, cfg, rep)
        if cfg.mode == NET_MODE_VIRTUAL and not tr.udp_hubs[hi].accept:
            any_client = True
    if any_client:
        _apply_interpolation(world, cfg, rep)


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
    ref fixed = fix_opt.value()[]
    var tick = Int(fixed.tick_count)
    var cfg_opt = resources.get_ptr_mut[NetConfigResource]()
    if cfg_opt:
        ref cfg = cfg_opt.value()[]
        if isfinite(cfg.tick_hz) and cfg.tick_hz > 0:
            cfg.snapshot_phase += min(Float64(1), cfg.tick_hz * fixed.dt_f64)
            if cfg.snapshot_phase >= 1.0 - 1.0e-12:
                cfg.snapshot_phase = max(Float64(0), cfg.snapshot_phase - 1)
                # After a long frame send only the newest propagated world state.
                cfg.snapshot_due = True
        else:
            cfg.snapshot_phase = 0
            cfg.snapshot_due = False

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


def _server_send_hub[T: NetTransport & Deinitable](
    mut hub: NetHub[T], mut cfg: NetConfigResource, mut rep: NetReplicationResource
):
    var is_server_hub = (cfg.mode == NET_MODE_SERVER) or (cfg.mode == NET_MODE_VIRTUAL and hub.accept)
    if not is_server_hub:
        return
    for cid in range(hub.count):
        if cid >= rep.replicator.max_clients or not hub.is_open(cid):
            continue
        var pkt_res = rep.replicator.build_client_packet(cid)
        if len(pkt_res[1]) > 0:
            if hub.send_unreliable(cid, pkt_res[1], cfg.now_ms):
                cfg.packets_sent += 1
        var peer = net_peer_id_for_client(cid)
        for ri in range(len(rep.rpc_out_peer)):
            var tp = rep.rpc_out_peer[ri]
            if tp == NET_PEER_ALL or tp == peer:
                _ = hub.send_reliable(cid, rep.rpc_out_bytes[ri])
    hub.poll(cfg.now_ms)
    _server_after_hub_poll(hub, rep)


def _build_input_packet(cfg: NetConfigResource, mut rep: NetReplicationResource) -> List[UInt8]:
    # Acked frames never need resending
    if rep.receiver.has_acked_input_seq and len(rep.local_input_ticks) > 0:
        var acked = rep.receiver.last_acked_input_seq
        if rep.local_input_ticks[0] <= acked:
            var nt = List[Int]()
            var nd = List[List[UInt8]]()
            for i in range(len(rep.local_input_ticks)):
                if rep.local_input_ticks[i] > acked:
                    nt.append(rep.local_input_ticks[i])
                    nd.append(rep.local_input_data[i].copy())
            rep.local_input_ticks = nt^
            rep.local_input_data = nd^
    var n = len(rep.local_input_ticks)
    if n == 0:
        return List[UInt8]()
    var red = max(1, min(32, cfg.input_redundancy))
    var start = max(0, n - red)
    var w = BitWriter(256)
    w.write_u16(_NET_INPUT_MAGIC)
    w.write_u8(UInt8(n - start))
    for i in range(start, n):
        w.write_u32(UInt32(rep.local_input_ticks[i] & 0xFFFFFFFF))
        w.write_u8(UInt8(len(rep.local_input_data[i])))
        w.write_bytes_aligned(rep.local_input_data[i])
    return w.finish()


def _build_client_state_packet(mut world: World, cfg: NetConfigResource) -> List[UInt8]:
    if cfg.local_peer_id <= 0:
        return List[UInt8]()
    var ids = List[Int]()
    var poses = List[Vec3]()
    var rots = List[Quat]()
    var entities = world.query_entities(world.bit_for[NetworkIdentityComponent]())
    for i in range(len(entities)):
        var nc_opt = world.get_ptr[NetworkIdentityComponent](entities[i])
        if not nc_opt:
            continue
        ref nc = nc_opt.value()[unsafe_offset=0]
        if not nc.is_local or nc.owner_peer_id != cfg.local_peer_id:
            continue
        if nc.authority != NET_AUTHORITY_CLIENT and nc.authority != NET_AUTHORITY_SHARED:
            continue
        var lt_opt = world.get_ptr[LocalTransform](entities[i])
        if not lt_opt:
            continue
        ids.append(nc.net_id)
        poses.append(lt_opt.value()[unsafe_offset=0].translation)
        rots.append(lt_opt.value()[unsafe_offset=0].rotation)
        if len(ids) >= 30:  # 32 B each: stays inside one 1180 B datagram
            break
    if len(ids) == 0:
        return List[UInt8]()
    var w = BitWriter(8 + 32 * len(ids))
    w.write_u16(_NET_CSTATE_MAGIC)
    w.write_u8(UInt8(len(ids)))
    for i in range(len(ids)):
        w.write_u32(UInt32(ids[i] & 0xFFFFFFFF))
        w.write_f32_raw(poses[i].x)
        w.write_f32_raw(poses[i].y)
        w.write_f32_raw(poses[i].z)
        w.write_f32_raw(rots[i].x)
        w.write_f32_raw(rots[i].y)
        w.write_f32_raw(rots[i].z)
        w.write_f32_raw(rots[i].w)
    return w.finish()


def _client_send_hub[T: NetTransport & Deinitable](
    mut hub: NetHub[T],
    cfg: NetConfigResource,
    rep: NetReplicationResource,
    input_pkt: List[UInt8],
    state_pkt: List[UInt8],
) -> Bool:
    var is_client_hub = (cfg.mode == NET_MODE_CLIENT) or (cfg.mode == NET_MODE_VIRTUAL and not hub.accept)
    if not is_client_hub:
        return False
    for cid in range(hub.count):
        if not hub.is_open(cid):
            continue
        if len(input_pkt) > 0:
            _ = hub.send_unreliable_raw(cid, input_pkt, cfg.now_ms)
        if len(state_pkt) > 0:
            _ = hub.send_unreliable_raw(cid, state_pkt, cfg.now_ms)
        for ri in range(len(rep.rpc_out_bytes)):
            _ = hub.send_reliable(cid, rep.rpc_out_bytes[ri])
    hub.poll(cfg.now_ms)
    return True


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

    if not cfg.snapshot_due:
        return
    cfg.snapshot_due = False

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
        if cfg.mode != NET_MODE_CLIENT and cfg.mode != NET_MODE_VIRTUAL:
            return
        # Client upstream: redundant input frames, client-authoritative state, queued RPCs
        var input_pkt = _build_input_packet(cfg, rep)
        var state_pkt = _build_client_state_packet(world, cfg)
        var sent_any = False
        for hi in range(len(tr.virtual_hubs)):
            if _client_send_hub(tr.virtual_hubs[hi], cfg, rep, input_pkt, state_pkt):
                sent_any = True
        for hi in range(len(tr.udp_hubs)):
            if _client_send_hub(tr.udp_hubs[hi], cfg, rep, input_pkt, state_pkt):
                sent_any = True
        if len(input_pkt) > 0 and sent_any:
            rep.input_packets_sent += 1
        if sent_any:
            rep.rpc_out_peer.clear()
            rep.rpc_out_bytes.clear()
        return

    rep.replicator.ext_enabled = True
    rep.replicator.server_time_ms = cfg.now_ms
    var live_ids = List[Bool](length=rep.replicator.max_entities, fill=False)
    var mask = world.bit_for[NetworkIdentityComponent]().union(world.bit_for[ReplicationComponent]())
    var entities = world.query_entities(mask)
    var pod_writer = BitWriter(64)
    for i in range(len(entities)):
        var e = entities[i]
        var nc_opt = world.get_ptr[NetworkIdentityComponent](e)
        var rc_opt = world.get_ptr[ReplicationComponent](e)
        if not nc_opt or not rc_opt:
            continue
        var nc = nc_opt.value()[unsafe_offset=0]
        var rc = rc_opt.value()[unsafe_offset=0]
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
            nc.prefab_id,
            nc.owner_peer_id,
            nc.authority,
        )
        for slot in range(REPL_EXTRA_SLOTS):
            if not rep.pod_slot_used[slot]:
                continue
            pod_writer.clear()
            var enc = rep.pod_encode_fns[slot]
            if enc(world_ptr, e.to_bits(), Int(Pointer(to=pod_writer))):
                _ = rep.replicator.set_extra(nc.net_id, slot, pod_writer.finish())
            else:
                _ = rep.replicator.set_extra(nc.net_id, slot, List[UInt8]())

    for nid in range(rep.replicator.max_entities):
        if rep.replicator.entities[nid].active and not live_ids[nid]:
            rep.replicator.destroy_entity(nid)
            _ = rep.net_id_to_entity.pop(nid, 0)

    rep.replicator.prepare_tick()

    for hi in range(len(tr.virtual_hubs)):
        _server_send_hub(tr.virtual_hubs[hi], cfg, rep)
    for hi in range(len(tr.udp_hubs)):
        _server_send_hub(tr.udp_hubs[hi], cfg, rep)
    rep.rpc_out_peer.clear()
    rep.rpc_out_bytes.clear()


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
