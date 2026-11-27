"""Client-Server Delta Snapshot Replicator & Spatial Interest Management (`flint.net.replicator`).

Implements:
- Dirty component field bitmasks (`DIRTY_POS | DIRTY_ROT | DIRTY_VEL | DIRTY_STATE`)
- 2D/3D spatial-grid interest management (`SpatialInterestGrid` hashed via `detsort.mix64`)
- Per-client 32-slot ACKed baseline ring for zero-redundancy delta compression
- Starvation-free priority accumulator ordered deterministically via `detsort.stable_order`
- Strict per-packet byte budget (`<= max_packet_bytes`) + `SnapshotClientView` decoder.
"""
from std.collections import List
from std.math import abs, floor, max, min
from flint.core.detsort import mix64, stable_order
from flint.core.math import Quat, Vec3
from flint.net.bitpack import (
    BitReader,
    BitWriter,
    pack_quat_smallest_three,
    quat_angular_error_deg,
)
from flint.net.channel import seq_gt

comptime DIRTY_NONE: UInt8 = 0
comptime DIRTY_DESTROY: UInt8 = 0  # 4-bit opcode 0 in wire stream = despawn / out-of-AOI removal
comptime DIRTY_POS: UInt8 = 1 << 0
comptime DIRTY_ROT: UInt8 = 1 << 1
comptime DIRTY_VEL: UInt8 = 1 << 2
comptime DIRTY_STATE: UInt8 = 1 << 3
comptime DIRTY_ALL: UInt8 = DIRTY_POS | DIRTY_ROT | DIRTY_VEL | DIRTY_STATE

comptime _BASELINE_RING: Int = 32
comptime _BASELINE_MASK: Int = 31
comptime _REPL_MAGIC: UInt16 = 0x5235  # "5R"
comptime _MAX_GRID_QUERY_CELLS: Int = 4096


@always_inline
def _rep_is_finite32(v: Float32) -> Bool:
    return (v - v) == Float32(0.0)


@always_inline
def _rep_is_finite64(v: Float64) -> Bool:
    return (v - v) == 0.0


@always_inline
def _rep_vec3_finite(v: Vec3) -> Bool:
    return (
        _rep_is_finite32(v.x)
        and _rep_is_finite32(v.y)
        and _rep_is_finite32(v.z)
    )


@always_inline
def _rep_cell_coord(v: Float64, inv_cell_size: Float64) -> Int:
    var cf = floor(v * inv_cell_size)
    if cf < -4.0e18:
        cf = -4.0e18
    elif cf > 4.0e18:
        cf = 4.0e18
    return Int(cf)


@fieldwise_init
struct ReplicatedEntity(Copyable, ImplicitlyCopyable, Movable):
    """Authoritative server entity state replicated to interested clients."""
    var active: Bool
    var net_id: Int
    var pos: Vec3
    var rot: Quat
    var vel: Vec3
    var state_flags: UInt32
    var base_priority: Float32

    def __init__(out self, net_id: Int = 0):
        self.active = False
        self.net_id = net_id
        self.pos = Vec3()
        self.rot = Quat.identity()
        self.vel = Vec3()
        self.state_flags = 0
        self.base_priority = 1.0


@fieldwise_init
struct _QuantizedEntitySnap(Copyable, ImplicitlyCopyable, Movable):
    var known: Bool
    var chunk_ix: Int32
    var chunk_iy: Int32
    var chunk_iz: Int32
    var qx: Int
    var qy: Int
    var qz: Int
    var rot: Quat
    var packed_rot: UInt32
    var vx: Int
    var vy: Int
    var vz: Int
    var state_flags: UInt32

    def __init__(out self):
        self.known = False
        self.chunk_ix = 0
        self.chunk_iy = 0
        self.chunk_iz = 0
        self.qx = 0
        self.qy = 0
        self.qz = 0
        self.rot = Quat.identity()
        self.packed_rot = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.state_flags = 0


struct _ClientBaselineSlot(Movable):
    var valid: Bool
    var seq: UInt16
    var entities: List[_QuantizedEntitySnap]
    var known_eids: List[Int]

    def __init__(out self, max_entities: Int):
        self.valid = False
        self.seq = 0
        self.entities = List[_QuantizedEntitySnap](
            length=max_entities, fill=_QuantizedEntitySnap()
        )
        self.known_eids = List[Int](capacity=min(64, max_entities))

    def clear_sparse(mut self):
        for i in range(len(self.known_eids)):
            var eid = self.known_eids[i]
            self.entities[eid] = _QuantizedEntitySnap()
        self.known_eids.clear()


struct _ClientRepState(Movable):
    var connected: Bool
    var observer_pos: Vec3
    var interest_radius: Float32
    var next_snap_seq: UInt16
    var has_acked_seq: Bool
    var last_acked_seq: UInt16
    var has_acked_input_seq: Bool
    var last_acked_input_seq: UInt32
    var priority_accum: List[Float32]
    var baselines: List[_ClientBaselineSlot]
    var bytes_sent_total: Int
    var packets_sent_total: Int

    def __init__(out self, max_entities: Int, interest_radius: Float32):
        self.connected = True
        self.observer_pos = Vec3()
        self.interest_radius = interest_radius
        self.next_snap_seq = 0
        self.has_acked_seq = False
        self.last_acked_seq = 0
        self.has_acked_input_seq = False
        self.last_acked_input_seq = 0
        self.priority_accum = List[Float32](length=max_entities, fill=0.0)
        self.baselines = List[_ClientBaselineSlot](capacity=_BASELINE_RING)
        for _ in range(_BASELINE_RING):
            self.baselines.append(_ClientBaselineSlot(max_entities))
        self.bytes_sent_total = 0
        self.packets_sent_total = 0


struct SpatialInterestGrid(Movable):
    """Deterministic 2D (XZ) or 3D (XYZ) fixed-bucket spatial grid for AOI queries."""
    var cell_size: Float32
    var inv_cell_size: Float32
    var use_3d: Bool
    var num_buckets: Int
    var bucket_heads: List[Int32]
    var next_entity: List[Int32]
    var cell_coords_x: List[Int]
    var cell_coords_y: List[Int]
    var cell_coords_z: List[Int]

    def __init__(
        out self,
        max_entities: Int = 1024,
        cell_size: Float32 = 32.0,
        use_3d: Bool = False,
        bucket_pow2: Int = 10,
    ):
        var cs = max(Float32(1.0), cell_size)
        self.cell_size = cs
        self.inv_cell_size = Float32(1.0) / cs
        self.use_3d = use_3d
        self.num_buckets = 1 << max(6, min(14, bucket_pow2))
        self.bucket_heads = List[Int32](length=self.num_buckets, fill=-1)
        self.next_entity = List[Int32](length=max_entities, fill=-1)
        self.cell_coords_x = List[Int](length=max_entities, fill=0)
        self.cell_coords_y = List[Int](length=max_entities, fill=0)
        self.cell_coords_z = List[Int](length=max_entities, fill=0)

    @always_inline
    def _bucket_idx(self, cx: Int, cy: Int, cz: Int) -> Int:
        var packed = (
            (UInt64(cx & 0x1FFFFF) << 42)
            | (UInt64(cy & 0x1FFFFF) << 21)
            | UInt64(cz & 0x1FFFFF)
        )
        return Int(mix64(packed) & UInt64(self.num_buckets - 1))

    def rebuild(mut self, entities: List[ReplicatedEntity]):
        var n = len(entities)
        if len(self.next_entity) < n:
            self.next_entity.resize(n, -1)
            self.cell_coords_x.resize(n, 0)
            self.cell_coords_y.resize(n, 0)
            self.cell_coords_z.resize(n, 0)

        for b in range(self.num_buckets):
            self.bucket_heads[b] = -1

        var inv_cs64 = Float64(self.inv_cell_size)
        # Insert in reverse entity index so forward linked-list walk visits ascending indices!
        var idx = n - 1
        while idx >= 0:
            if entities[idx].active and _rep_vec3_finite(entities[idx].pos):
                var cx = _rep_cell_coord(Float64(entities[idx].pos.x), inv_cs64)
                var cy = (
                    _rep_cell_coord(Float64(entities[idx].pos.y), inv_cs64)
                    if self.use_3d
                    else 0
                )
                var cz = _rep_cell_coord(Float64(entities[idx].pos.z), inv_cs64)
                self.cell_coords_x[idx] = cx
                self.cell_coords_y[idx] = cy
                self.cell_coords_z[idx] = cz
                var b = self._bucket_idx(cx, cy, cz)
                self.next_entity[idx] = self.bucket_heads[b]
                self.bucket_heads[b] = Int32(idx)
            else:
                self.next_entity[idx] = -1
            idx -= 1

    def query_sphere(
        self,
        entities: List[ReplicatedEntity],
        center: Vec3,
        radius: Float32,
        mut out_indices: List[Int],
    ):
        out_indices.clear()
        if not _rep_vec3_finite(center):
            return
        if not _rep_is_finite32(radius) or radius < Float32(0.0):
            return
        var r64 = Float64(radius)
        var r_sq64 = r64 * r64
        var inv_cs64 = Float64(self.inv_cell_size)
        var min_cx = _rep_cell_coord(Float64(center.x) - r64, inv_cs64)
        var max_cx = _rep_cell_coord(Float64(center.x) + r64, inv_cs64)
        var min_cy = (
            _rep_cell_coord(Float64(center.y) - r64, inv_cs64)
            if self.use_3d
            else 0
        )
        var max_cy = (
            _rep_cell_coord(Float64(center.y) + r64, inv_cs64)
            if self.use_3d
            else 0
        )
        var min_cz = _rep_cell_coord(Float64(center.z) - r64, inv_cs64)
        var max_cz = _rep_cell_coord(Float64(center.z) + r64, inv_cs64)

        var span_x = max_cx - min_cx + 1
        var span_y = (max_cy - min_cy + 1) if self.use_3d else 1
        var span_z = max_cz - min_cz + 1
        var use_linear = (
            span_x <= 0
            or span_y <= 0
            or span_z <= 0
            or span_x > _MAX_GRID_QUERY_CELLS
            or span_y > _MAX_GRID_QUERY_CELLS
            or span_z > _MAX_GRID_QUERY_CELLS
        )
        if not use_linear:
            var span_xy = span_x * span_y
            if (
                span_xy > _MAX_GRID_QUERY_CELLS
                or span_xy * span_z > _MAX_GRID_QUERY_CELLS
            ):
                use_linear = True

        if use_linear:
            for idx in range(len(entities)):
                if not entities[idx].active or not _rep_vec3_finite(entities[idx].pos):
                    continue
                var dx = Float64(entities[idx].pos.x) - Float64(center.x)
                var dz = Float64(entities[idx].pos.z) - Float64(center.z)
                var d_sq: Float64
                if self.use_3d:
                    var dy = Float64(entities[idx].pos.y) - Float64(center.y)
                    d_sq = dx * dx + dy * dy + dz * dz
                else:
                    d_sq = dx * dx + dz * dz
                if d_sq <= r_sq64:
                    out_indices.append(idx)
            return

        for cy in range(min_cy, max_cy + 1):
            for cz in range(min_cz, max_cz + 1):
                for cx in range(min_cx, max_cx + 1):
                    var b = self._bucket_idx(cx, cy, cz)
                    var cur = Int(self.bucket_heads[b])
                    while cur >= 0:
                        if (
                            self.cell_coords_x[cur] == cx
                            and self.cell_coords_y[cur] == cy
                            and self.cell_coords_z[cur] == cz
                            and entities[cur].active
                            and _rep_vec3_finite(entities[cur].pos)
                        ):
                            var dx = Float64(entities[cur].pos.x) - Float64(center.x)
                            var dz = Float64(entities[cur].pos.z) - Float64(center.z)
                            var d_sq: Float64
                            if self.use_3d:
                                var dy = Float64(entities[cur].pos.y) - Float64(center.y)
                                d_sq = dx * dx + dy * dy + dz * dz
                            else:
                                d_sq = dx * dx + dz * dz
                            if d_sq <= r_sq64:
                                out_indices.append(cur)
                        cur = Int(self.next_entity[cur])


@always_inline
def _quant_pos_axis(v: Float32, chunk_size: Float32) -> Tuple[Int32, Int]:
    var cs = Float64(chunk_size)
    var vf = Float64(v) if (v - v) == Float32(0.0) else 0.0
    var cf = floor(vf / cs)
    if cf < -2147483648.0:
        cf = -2147483648.0
    elif cf > 2147483647.0:
        cf = 2147483647.0
    var c = Int32(cf)
    var local = vf - cf * cs
    if local < 0.0:
        if c > Int32(-2147483648):
            c -= 1
            local += cs
        else:
            local = 0.0
    elif local >= cs:
        if c < Int32(2147483647):
            c += 1
            local -= cs
        else:
            local = cs
    var q = Int(floor((local / cs) * 65536.0))
    if q < 0:
        q = 0
    elif q > 65535:
        q = 65535
    return (c, q)


@always_inline
def _dequant_pos_axis(c: Int32, q: Int, chunk_size: Float32) -> Float32:
    var cs = Float64(chunk_size)
    var origin = Float64(c) * cs
    var local = ((Float64(q) + 0.5) / 65536.0) * cs
    return Float32(origin + local)


@always_inline
def _apply_axis_delta(mut chunk: Int32, mut q: Int, delta: Int):
    var total = q + delta
    while total >= 65536:
        total -= 65536
        chunk += 1
    while total < 0:
        total += 65536
        chunk -= 1
    q = total


@always_inline
def _sign_extend_i16(raw: UInt64) -> Int32:
    var u = Int(raw & 0xFFFF)
    if u >= 32768:
        u -= 65536
    return Int32(u)


@always_inline
def _u32_to_i32(raw: UInt32) -> Int32:
    var v = Int64(raw)
    if v >= 2147483648:
        v -= 4294967296
    return Int32(v)


struct SnapshotReplicator(Movable):
    """Authoritative server snapshot delta replicator with spatial AOI and priority budgeting."""
    var max_entities: Int
    var max_clients: Int
    var max_packet_bytes: Int
    var world_min: Float32
    var world_max: Float32
    var chunk_size: Float32
    var vel_limit: Float32

    var entities: List[ReplicatedEntity]
    var quantized_now: List[_QuantizedEntitySnap]
    var clients: List[_ClientRepState]
    var grid: SpatialInterestGrid

    var candidate_indices: List[Int]
    var candidate_masks: List[UInt8]
    var in_aoi_stamp: List[Int]
    var aoi_query_stamp: Int
    var remove_candidates: List[Int]
    var sort_keys: List[Int]
    var writer: BitWriter

    def __init__(
        out self,
        max_entities: Int = 512,
        max_clients: Int = 16,
        max_packet_bytes: Int = 1100,
        interest_radius: Float32 = 110.0,
        cell_size: Float32 = 32.0,
        use_3d_grid: Bool = False,
        world_bounds: Float32 = 512.0,
        vel_limit: Float32 = 120.0,
        chunk_size: Float32 = 0.0,
    ):
        var wb = abs(world_bounds) if _rep_is_finite32(world_bounds) else Float32(512.0)
        var vl = vel_limit if _rep_is_finite32(vel_limit) else Float32(120.0)
        var cs_in = chunk_size if _rep_is_finite32(chunk_size) else Float32(0.0)
        var ir_in = interest_radius if (_rep_is_finite32(interest_radius) and interest_radius > Float32(0.0)) else Float32(110.0)
        var cell_in = cell_size if (_rep_is_finite32(cell_size) and cell_size > Float32(0.0)) else Float32(32.0)
        self.max_entities = max(1, max_entities)
        self.max_clients = max(1, max_clients)
        self.max_packet_bytes = max(64, min(1180, max_packet_bytes))
        self.world_min = -wb
        self.world_max = wb
        self.chunk_size = (
            max(Float32(1.0), cs_in)
            if cs_in > Float32(0.0)
            else max(Float32(64.0), wb)
        )
        self.vel_limit = max(Float32(1.0), vl)

        self.entities = List[ReplicatedEntity](capacity=self.max_entities)
        for i in range(self.max_entities):
            self.entities.append(ReplicatedEntity(i))
        self.quantized_now = List[_QuantizedEntitySnap](
            length=self.max_entities, fill=_QuantizedEntitySnap()
        )

        self.clients = List[_ClientRepState](capacity=self.max_clients)
        for _ in range(self.max_clients):
            self.clients.append(
                _ClientRepState(self.max_entities, ir_in)
            )

        self.grid = SpatialInterestGrid(
            self.max_entities, cell_in, use_3d_grid
        )
        self.candidate_indices = List[Int](capacity=self.max_entities)
        self.candidate_masks = List[UInt8](
            length=self.max_entities, fill=DIRTY_NONE
        )
        self.in_aoi_stamp = List[Int](length=self.max_entities, fill=0)
        self.aoi_query_stamp = 0
        self.remove_candidates = List[Int](capacity=64)
        self.sort_keys = List[Int](capacity=self.max_entities)
        self.writer = BitWriter(1200)

    @always_inline
    def _quant_pos(self, v: Float32) -> Int:
        var vf = v if _rep_is_finite32(v) else Float32(0.0)
        var span = self.world_max - self.world_min
        if not _rep_is_finite32(span) or span <= Float32(0.0):
            return 0
        var norm = max(
            Float32(0.0), min(Float32(1.0), (vf - self.world_min) / span)
        )
        return Int(norm * Float32(65535.0) + Float32(0.5))

    @always_inline
    def _dequant_pos(self, q: Int) -> Float32:
        return self.world_min + (Float32(q) / Float32(65535.0)) * (
            self.world_max - self.world_min
        )

    @always_inline
    def _quant_vel(self, v: Float32) -> Int:
        var vf = v if _rep_is_finite32(v) else Float32(0.0)
        var span = Float32(2.0) * self.vel_limit
        if not _rep_is_finite32(span) or span <= Float32(0.0):
            return 2048
        var norm = max(
            Float32(0.0), min(Float32(1.0), (vf + self.vel_limit) / span)
        )
        return Int(norm * Float32(4095.0) + Float32(0.5))

    @always_inline
    def _dequant_vel(self, q: Int) -> Float32:
        return -self.vel_limit + (
            Float32(q) / Float32(4095.0)
        ) * Float32(2.0) * self.vel_limit

    def set_entity(
        mut self,
        entity_id: Int,
        pos: Vec3,
        rot: Quat = Quat.identity(),
        vel: Vec3 = Vec3(),
        state_flags: UInt32 = 0,
        base_priority: Float32 = 1.0,
    ):
        if entity_id < 0 or entity_id >= self.max_entities:
            return
        var safe_pos = Vec3(
            pos.x if _rep_is_finite32(pos.x) else Float32(0.0),
            pos.y if _rep_is_finite32(pos.y) else Float32(0.0),
            pos.z if _rep_is_finite32(pos.z) else Float32(0.0),
        )
        var safe_vel = Vec3(
            vel.x if _rep_is_finite32(vel.x) else Float32(0.0),
            vel.y if _rep_is_finite32(vel.y) else Float32(0.0),
            vel.z if _rep_is_finite32(vel.z) else Float32(0.0),
        )
        var safe_rot = rot.normalized() if (
            _rep_is_finite32(rot.x)
            and _rep_is_finite32(rot.y)
            and _rep_is_finite32(rot.z)
            and _rep_is_finite32(rot.w)
        ) else Quat.identity()
        var safe_prio = base_priority if _rep_is_finite32(base_priority) else Float32(1.0)
        self.entities[entity_id].active = True
        self.entities[entity_id].net_id = entity_id
        self.entities[entity_id].pos = safe_pos
        self.entities[entity_id].rot = safe_rot
        self.entities[entity_id].vel = safe_vel
        self.entities[entity_id].state_flags = state_flags
        self.entities[entity_id].base_priority = max(
            Float32(0.05), safe_prio
        )

    def destroy_entity(mut self, entity_id: Int):
        """Marks an entity inactive/despawned so interested clients receive a removal opcode."""
        if entity_id < 0 or entity_id >= self.max_entities:
            return
        self.entities[entity_id].active = False
        self.quantized_now[entity_id].known = False

    def set_client_view(
        mut self,
        client_id: Int,
        observer_pos: Vec3,
        interest_radius: Float32 = -1.0,
    ):
        if client_id < 0 or client_id >= self.max_clients:
            return
        self.clients[client_id].connected = True
        self.clients[client_id].observer_pos = Vec3(
            observer_pos.x if _rep_is_finite32(observer_pos.x) else Float32(0.0),
            observer_pos.y if _rep_is_finite32(observer_pos.y) else Float32(0.0),
            observer_pos.z if _rep_is_finite32(observer_pos.z) else Float32(0.0),
        )
        if _rep_is_finite32(interest_radius) and interest_radius > Float32(0.0):
            self.clients[client_id].interest_radius = interest_radius

    def reset_client_baselines(mut self, client_id: Int):
        """A newly accepted transport peer has no snapshot or input history for its reused slot."""
        if client_id < 0 or client_id >= self.max_clients:
            return
        ref client = self.clients[client_id]
        client.has_acked_seq = False
        client.last_acked_seq = 0
        client.has_acked_input_seq = False
        client.last_acked_input_seq = 0
        if client.packets_sent_total == 0:
            return
        for i in range(len(client.baselines)):
            client.baselines[i].clear_sparse()
            client.baselines[i].valid = False
        for i in range(len(client.priority_accum)):
            client.priority_accum[i] = 0
        # Slot sequence numbers keep advancing, so late ACKs from the old
        # connection cannot acknowledge the new connection's first snapshot.
        # Bandwidth counters and the configured AOI remain slot-level settings.

    def notify_client_ack(mut self, client_id: Int, acked_snap_seq: UInt16):
        """Records that `client_id` received and reconstructed `acked_snap_seq`."""
        if client_id < 0 or client_id >= self.max_clients:
            return
        if (
            self.clients[client_id].has_acked_seq
            and not seq_gt(acked_snap_seq, self.clients[client_id].last_acked_seq)
        ):
            return
        var slot = Int(acked_snap_seq) & _BASELINE_MASK
        if (
            self.clients[client_id].baselines[slot].valid
            and self.clients[client_id].baselines[slot].seq == acked_snap_seq
        ):
            self.clients[client_id].has_acked_seq = True
            self.clients[client_id].last_acked_seq = acked_snap_seq

    def set_client_acked_input_seq(mut self, client_id: Int, input_seq: Int):
        """Records the latest authoritative input sequence processed for `client_id`."""
        if client_id < 0 or client_id >= self.max_clients or input_seq < 0:
            return
        var u_seq = UInt32(input_seq & 0xFFFFFFFF)
        if (
            not self.clients[client_id].has_acked_input_seq
            or u_seq > self.clients[client_id].last_acked_input_seq
        ):
            self.clients[client_id].has_acked_input_seq = True
            self.clients[client_id].last_acked_input_seq = u_seq

    def notify_client_input_ack(mut self, client_id: Int, input_seq: Int):
        self.set_client_acked_input_seq(client_id, input_seq)

    def prepare_tick(mut self):
        """Quantizes current world state once (chunk-relative positions + 32-bit rotations) and rebuilds spatial grid."""
        for i in range(self.max_entities):
            if not self.entities[i].active:
                self.quantized_now[i].known = False
                continue
            var q = _QuantizedEntitySnap()
            q.known = True
            var px = _quant_pos_axis(self.entities[i].pos.x, self.chunk_size)
            var py = _quant_pos_axis(self.entities[i].pos.y, self.chunk_size)
            var pz = _quant_pos_axis(self.entities[i].pos.z, self.chunk_size)
            q.chunk_ix = px[0]
            q.qx = px[1]
            q.chunk_iy = py[0]
            q.qy = py[1]
            q.chunk_iz = pz[0]
            q.qz = pz[1]
            q.rot = self.entities[i].rot
            q.packed_rot = pack_quat_smallest_three(self.entities[i].rot)
            q.vx = self._quant_vel(self.entities[i].vel.x)
            q.vy = self._quant_vel(self.entities[i].vel.y)
            q.vz = self._quant_vel(self.entities[i].vel.z)
            q.state_flags = self.entities[i].state_flags
            self.quantized_now[i] = q
        self.grid.rebuild(self.entities)

    def _compute_dirty_mask(
        self, cur: _QuantizedEntitySnap, base: _QuantizedEntitySnap
    ) -> UInt8:
        if not base.known:
            return DIRTY_ALL
        var mask = DIRTY_NONE
        if (
            cur.chunk_ix != base.chunk_ix
            or cur.chunk_iy != base.chunk_iy
            or cur.chunk_iz != base.chunk_iz
            or cur.qx != base.qx
            or cur.qy != base.qy
            or cur.qz != base.qz
        ):
            mask |= DIRTY_POS
        if (
            cur.packed_rot != base.packed_rot
            and quat_angular_error_deg(cur.rot, base.rot) > Float32(0.12)
        ):
            mask |= DIRTY_ROT
        if cur.vx != base.vx or cur.vy != base.vy or cur.vz != base.vz:
            mask |= DIRTY_VEL
        if cur.state_flags != base.state_flags:
            mask |= DIRTY_STATE
        return mask

    def build_client_packet(
        mut self, client_id: Int
    ) -> Tuple[UInt16, List[UInt8]]:
        """Builds one delta-compressed, priority-budgeted snapshot packet (`<= max_packet_bytes`)."""
        if client_id < 0 or client_id >= self.max_clients:
            return (UInt16(0), List[UInt8]())

        var snap_seq = self.clients[client_id].next_snap_seq
        self.clients[client_id].next_snap_seq = UInt16(
            (Int(snap_seq) + 1) & 0xFFFF
        )

        var new_slot = Int(snap_seq) & _BASELINE_MASK
        var has_base = False
        var base_seq = UInt16(0)
        var base_slot = 0
        if self.clients[client_id].has_acked_seq:
            var try_seq = self.clients[client_id].last_acked_seq
            var try_slot = Int(try_seq) & _BASELINE_MASK
            if (
                try_slot != new_slot
                and self.clients[client_id].baselines[try_slot].valid
                and self.clients[client_id].baselines[try_slot].seq == try_seq
            ):
                has_base = True
                base_seq = try_seq
                base_slot = try_slot

        var obs = self.clients[client_id].observer_pos
        var rad = self.clients[client_id].interest_radius
        self.grid.query_sphere(self.entities, obs, rad, self.candidate_indices)

        # Ensure ascending entity index order before stable priority sort so ties are 100% deterministic
        stable_sort_entity_ids(self.candidate_indices)

        # Stamp active AOI entities in O(candidates) to detect despawns & out-of-AOI exits
        self.aoi_query_stamp += 1
        var stamp = self.aoi_query_stamp
        for k in range(len(self.candidate_indices)):
            self.in_aoi_stamp[self.candidate_indices[k]] = stamp

        self.remove_candidates.clear()
        if has_base:
            var n_known = len(self.clients[client_id].baselines[base_slot].known_eids)
            for k in range(n_known):
                var eid = self.clients[client_id].baselines[base_slot].known_eids[k]
                if self.in_aoi_stamp[eid] != stamp:
                    self.remove_candidates.append(eid)

        var dirty_candidates = List[Int](capacity=len(self.candidate_indices))
        self.sort_keys.clear()
        var inv_rad = Float32(1.0) / max(Float32(1.0), rad)

        for k in range(len(self.candidate_indices)):
            var eid = self.candidate_indices[k]
            var cur_q = self.quantized_now[eid]
            var base_q = (
                self.clients[client_id].baselines[base_slot].entities[eid]
                if has_base
                else _QuantizedEntitySnap()
            )
            var dmask = self._compute_dirty_mask(cur_q, base_q)
            self.candidate_masks[eid] = dmask
            if dmask == DIRTY_NONE:
                continue

            var dist = (self.entities[eid].pos - obs).length()
            var near_boost = Float32(1.0) + Float32(2.0) * max(
                Float32(0.0), Float32(1.0) - dist * inv_rad
            )
            var spawn_boost = (
                Float32(4.0) if not base_q.known else Float32(1.0)
            )
            var gain = (
                self.entities[eid].base_priority * near_boost * spawn_boost
            )
            self.clients[client_id].priority_accum[eid] += gain
            var acc_raw = self.clients[client_id].priority_accum[eid] * Float32(1024.0)
            var score = Int(max(Float32(-1.0e15), min(Float32(1.0e15), acc_raw))) if _rep_is_finite32(acc_raw) else 0
            dirty_candidates.append(eid)
            self.sort_keys.append(-score)  # ascending key => highest priority first

        var order = stable_order(self.sort_keys)

        # Sparse O(known_in_aoi) baseline copy into `new_slot` (instead of O(max_entities))
        self.clients[client_id].baselines[new_slot].clear_sparse()
        self.clients[client_id].baselines[new_slot].valid = True
        self.clients[client_id].baselines[new_slot].seq = snap_seq
        if has_base:
            var n_known = len(self.clients[client_id].baselines[base_slot].known_eids)
            for k in range(n_known):
                var eid = self.clients[client_id].baselines[base_slot].known_eids[k]
                self.clients[client_id].baselines[new_slot].entities[eid] = (
                    self.clients[client_id].baselines[base_slot].entities[eid]
                )
                self.clients[client_id].baselines[new_slot].known_eids.append(eid)

        # Encode header (`12` bytes + 2 flag bits = 13-byte min packet):
        # magic u16, snap_seq u16, base_seq u16, count_placeholder u16,
        # last_acked_input_seq u32, has_base bool, has_acked_input bool
        self.writer.clear()
        self.writer.write_u16(_REPL_MAGIC)
        self.writer.write_u16(snap_seq)
        self.writer.write_u16(base_seq)
        self.writer.write_u16(0)  # patched after packing budgeted entities
        self.writer.write_u32(self.clients[client_id].last_acked_input_seq)
        self.writer.write_bool(has_base)
        self.writer.write_bool(self.clients[client_id].has_acked_input_seq)

        var packed_entities = 0
        var any_removed = False
        # Pack despawn / out-of-AOI removals first (high priority, tiny 4-bit mask == 0 opcode)
        for r_i in range(len(self.remove_candidates)):
            var eid = self.remove_candidates[r_i]
            var saved_mark = self.writer.mark()
            self.writer.write_int_bounded(eid, 0, self.max_entities - 1)
            self.writer.write_bits(UInt64(DIRTY_DESTROY), 4)
            if self.writer.byte_length() > self.max_packet_bytes:
                self.writer.restore(saved_mark)
                break
            self.clients[client_id].baselines[new_slot].entities[
                eid
            ] = _QuantizedEntitySnap()
            self.clients[client_id].priority_accum[eid] = 0.0
            any_removed = True
            packed_entities += 1

        for ord_i in range(len(order)):
            var eid = dirty_candidates[order[ord_i]]
            var mask = self.candidate_masks[eid]
            var cur_q = self.quantized_now[eid]
            var base_q = (
                self.clients[client_id].baselines[base_slot].entities[eid]
                if has_base
                else _QuantizedEntitySnap()
            )

            # Write directly to `self.writer` with zero-alloc rollback on budget overflow
            var saved_mark = self.writer.mark()
            self.writer.write_int_bounded(eid, 0, self.max_entities - 1)
            self.writer.write_bits(UInt64(mask & 0x0F), 4)

            if (mask & DIRTY_POS) != 0:
                var dx = (
                    (Int(cur_q.chunk_ix) - Int(base_q.chunk_ix)) * 65536
                    + (cur_q.qx - base_q.qx)
                )
                var dy = (
                    (Int(cur_q.chunk_iy) - Int(base_q.chunk_iy)) * 65536
                    + (cur_q.qy - base_q.qy)
                )
                var dz = (
                    (Int(cur_q.chunk_iz) - Int(base_q.chunk_iz)) * 65536
                    + (cur_q.qz - base_q.qz)
                )
                var use_small_delta = (
                    base_q.known
                    and abs(dx) <= 127
                    and abs(dy) <= 127
                    and abs(dz) <= 127
                )
                self.writer.write_bool(use_small_delta)
                if use_small_delta:
                    self.writer.write_int_bounded(dx, -127, 127)
                    self.writer.write_int_bounded(dy, -127, 127)
                    self.writer.write_int_bounded(dz, -127, 127)
                else:
                    var fits_i16 = (
                        cur_q.chunk_ix >= -32768
                        and cur_q.chunk_ix <= 32767
                        and cur_q.chunk_iy >= -32768
                        and cur_q.chunk_iy <= 32767
                        and cur_q.chunk_iz >= -32768
                        and cur_q.chunk_iz <= 32767
                    )
                    self.writer.write_bool(fits_i16)
                    if fits_i16:
                        self.writer.write_bits(
                            UInt64(Int(cur_q.chunk_ix) & 0xFFFF), 16
                        )
                        self.writer.write_bits(
                            UInt64(Int(cur_q.chunk_iy) & 0xFFFF), 16
                        )
                        self.writer.write_bits(
                            UInt64(Int(cur_q.chunk_iz) & 0xFFFF), 16
                        )
                    else:
                        self.writer.write_u32(
                            UInt32(Int(cur_q.chunk_ix) & 0xFFFFFFFF)
                        )
                        self.writer.write_u32(
                            UInt32(Int(cur_q.chunk_iy) & 0xFFFFFFFF)
                        )
                        self.writer.write_u32(
                            UInt32(Int(cur_q.chunk_iz) & 0xFFFFFFFF)
                        )
                    self.writer.write_bits(UInt64(cur_q.qx), 16)
                    self.writer.write_bits(UInt64(cur_q.qy), 16)
                    self.writer.write_bits(UInt64(cur_q.qz), 16)

            if (mask & DIRTY_ROT) != 0:
                self.writer.write_u32(cur_q.packed_rot)

            if (mask & DIRTY_VEL) != 0:
                self.writer.write_bits(UInt64(cur_q.vx), 12)
                self.writer.write_bits(UInt64(cur_q.vy), 12)
                self.writer.write_bits(UInt64(cur_q.vz), 12)

            if (mask & DIRTY_STATE) != 0:
                self.writer.write_u32(cur_q.state_flags)

            if self.writer.byte_length() > self.max_packet_bytes:
                self.writer.restore(saved_mark)
                # Entity stays accumulated for priority promotion on the very next snapshot!
                continue

            # Commit updated quantized fields into this snapshot's baseline record
            var was_known = self.clients[client_id].baselines[new_slot].entities[
                eid
            ].known
            var updated_b = base_q
            updated_b.known = True
            if (mask & DIRTY_POS) != 0:
                updated_b.chunk_ix = cur_q.chunk_ix
                updated_b.chunk_iy = cur_q.chunk_iy
                updated_b.chunk_iz = cur_q.chunk_iz
                updated_b.qx = cur_q.qx
                updated_b.qy = cur_q.qy
                updated_b.qz = cur_q.qz
            if (mask & DIRTY_ROT) != 0:
                updated_b.rot = cur_q.rot
                updated_b.packed_rot = cur_q.packed_rot
            if (mask & DIRTY_VEL) != 0:
                updated_b.vx = cur_q.vx
                updated_b.vy = cur_q.vy
                updated_b.vz = cur_q.vz
            if (mask & DIRTY_STATE) != 0:
                updated_b.state_flags = cur_q.state_flags
            self.clients[client_id].baselines[new_slot].entities[
                eid
            ] = updated_b
            if not was_known:
                self.clients[client_id].baselines[new_slot].known_eids.append(eid)
            self.clients[client_id].priority_accum[eid] = 0.0
            packed_entities += 1

        if any_removed:
            # Compact `known_eids` in O(known) to drop removed entities
            var write_idx = 0
            var n_k = len(self.clients[client_id].baselines[new_slot].known_eids)
            for k in range(n_k):
                var eid = self.clients[client_id].baselines[new_slot].known_eids[k]
                if self.clients[client_id].baselines[new_slot].entities[eid].known:
                    self.clients[client_id].baselines[new_slot].known_eids[
                        write_idx
                    ] = eid
                    write_idx += 1
            while (
                len(self.clients[client_id].baselines[new_slot].known_eids)
                > write_idx
            ):
                _ = self.clients[client_id].baselines[new_slot].known_eids.pop()

        var out_bytes = self.writer.finish()
        out_bytes[6] = UInt8(packed_entities & 0xFF)
        out_bytes[7] = UInt8((packed_entities >> 8) & 0xFF)

        self.clients[client_id].bytes_sent_total += len(out_bytes)
        self.clients[client_id].packets_sent_total += 1
        return (snap_seq, out_bytes^)


def stable_sort_entity_ids(mut ids: List[Int]):
    var order = stable_order(ids)
    var sorted = List[Int](capacity=len(ids))
    for i in range(len(order)):
        sorted.append(ids[order[i]])
    ids = sorted^


struct SnapshotClientReceiver(Movable):
    """Client-side delta snapshot decoder maintaining the matching 32-slot sparse baseline ring."""
    var max_entities: Int
    var world_min: Float32
    var world_max: Float32
    var chunk_size: Float32
    var vel_limit: Float32
    var entities: List[ReplicatedEntity]
    var baselines: List[_ClientBaselineSlot]
    var _staging_baseline: List[_QuantizedEntitySnap]
    var _staging_known_eids: List[Int]
    var _decoded_eids: List[Int]
    var _removed_eids: List[Int]
    var has_decoded_seq: Bool
    var last_decoded_seq: UInt16
    var has_acked_input_seq: Bool
    var last_acked_input_seq: Int
    var decoded_packets: Int

    def __init__(
        out self,
        max_entities: Int = 512,
        world_bounds: Float32 = 512.0,
        vel_limit: Float32 = 120.0,
        chunk_size: Float32 = 0.0,
    ):
        self.max_entities = max(1, max_entities)
        self.world_min = -abs(world_bounds)
        self.world_max = abs(world_bounds)
        self.chunk_size = (
            max(Float32(1.0), chunk_size)
            if chunk_size > Float32(0.0)
            else max(Float32(64.0), abs(world_bounds))
        )
        self.vel_limit = max(Float32(1.0), vel_limit)
        self.entities = List[ReplicatedEntity](capacity=self.max_entities)
        for i in range(self.max_entities):
            self.entities.append(ReplicatedEntity(i))
        self.baselines = List[_ClientBaselineSlot](capacity=_BASELINE_RING)
        for _ in range(_BASELINE_RING):
            self.baselines.append(_ClientBaselineSlot(self.max_entities))
        self._staging_baseline = List[_QuantizedEntitySnap](
            length=self.max_entities, fill=_QuantizedEntitySnap()
        )
        self._staging_known_eids = List[Int](capacity=min(64, self.max_entities))
        self._decoded_eids = List[Int](capacity=self.max_entities)
        self._removed_eids = List[Int](capacity=64)
        self.has_decoded_seq = False
        self.last_decoded_seq = 0
        self.has_acked_input_seq = False
        self.last_acked_input_seq = 0
        self.decoded_packets = 0

    def _clear_staging_sparse(mut self):
        for i in range(len(self._staging_known_eids)):
            var eid = self._staging_known_eids[i]
            self._staging_baseline[eid] = _QuantizedEntitySnap()
        self._staging_known_eids.clear()

    def apply_packet(mut self, pkt: List[UInt8]) -> Tuple[Bool, UInt16]:
        if len(pkt) < 13:
            return (False, UInt16(0))
        var r = BitReader(pkt)
        var magic = r.read_u16()
        if magic != _REPL_MAGIC:
            return (False, UInt16(0))
        var snap_seq = r.read_u16()
        var base_seq = r.read_u16()
        var count = Int(r.read_u16())
        var pkt_acked_input_seq = Int(r.read_u32())
        var has_base = r.read_bool()
        var pkt_has_acked_input = r.read_bool()
        if r.overflowed:
            return (False, UInt16(0))

        # Monotonic RFC-1982 sequence guard: reject stale or duplicate snapshots
        if self.has_decoded_seq and not seq_gt(snap_seq, self.last_decoded_seq):
            return (False, UInt16(0))

        var base_slot = Int(base_seq) & _BASELINE_MASK
        if has_base and (
            not self.baselines[base_slot].valid
            or self.baselines[base_slot].seq != base_seq
        ):
            return (False, UInt16(0))

        # Sparse O(known) staging so truncated/malformed packets never destroy valid baselines
        self._clear_staging_sparse()
        if has_base:
            var n_known = len(self.baselines[base_slot].known_eids)
            for k in range(n_known):
                var eid = self.baselines[base_slot].known_eids[k]
                self._staging_baseline[eid] = self.baselines[base_slot].entities[eid]
                self._staging_known_eids.append(eid)
        self._decoded_eids.clear()
        self._removed_eids.clear()

        for _ in range(count):
            var eid = r.read_int_bounded(0, self.max_entities - 1)
            var mask = UInt8(r.read_bits(4))
            if r.overflowed or eid < 0 or eid >= self.max_entities:
                return (False, UInt16(0))

            if mask == DIRTY_DESTROY:
                self._staging_baseline[eid] = _QuantizedEntitySnap()
                self._removed_eids.append(eid)
                continue

            var was_known = self._staging_baseline[eid].known
            var q = self._staging_baseline[eid]
            q.known = True

            if (mask & DIRTY_POS) != 0:
                var is_delta = r.read_bool()
                if is_delta:
                    var dx = r.read_int_bounded(-127, 127)
                    var dy = r.read_int_bounded(-127, 127)
                    var dz = r.read_int_bounded(-127, 127)
                    _apply_axis_delta(q.chunk_ix, q.qx, dx)
                    _apply_axis_delta(q.chunk_iy, q.qy, dy)
                    _apply_axis_delta(q.chunk_iz, q.qz, dz)
                else:
                    var fits_i16 = r.read_bool()
                    if fits_i16:
                        q.chunk_ix = _sign_extend_i16(r.read_bits(16))
                        q.chunk_iy = _sign_extend_i16(r.read_bits(16))
                        q.chunk_iz = _sign_extend_i16(r.read_bits(16))
                    else:
                        q.chunk_ix = _u32_to_i32(r.read_u32())
                        q.chunk_iy = _u32_to_i32(r.read_u32())
                        q.chunk_iz = _u32_to_i32(r.read_u32())
                    q.qx = Int(r.read_bits(16))
                    q.qy = Int(r.read_bits(16))
                    q.qz = Int(r.read_bits(16))

            if (mask & DIRTY_ROT) != 0:
                q.rot = r.read_quat_smallest_three()

            if (mask & DIRTY_VEL) != 0:
                q.vx = Int(r.read_bits(12))
                q.vy = Int(r.read_bits(12))
                q.vz = Int(r.read_bits(12))

            if (mask & DIRTY_STATE) != 0:
                q.state_flags = r.read_u32()

            if r.overflowed:
                return (False, UInt16(0))

            self._staging_baseline[eid] = q
            if not was_known:
                self._staging_known_eids.append(eid)
            self._decoded_eids.append(eid)

        # Atomic sparse commit: packet decoded cleanly with zero overflow
        var new_slot = Int(snap_seq) & _BASELINE_MASK
        self.baselines[new_slot].clear_sparse()
        self.baselines[new_slot].valid = True
        self.baselines[new_slot].seq = snap_seq
        for i in range(len(self._staging_known_eids)):
            var eid = self._staging_known_eids[i]
            if self._staging_baseline[eid].known:
                self.baselines[new_slot].entities[eid] = self._staging_baseline[eid]
                self.baselines[new_slot].known_eids.append(eid)

        if not has_base:
            for eid in range(self.max_entities):
                if not self._staging_baseline[eid].known:
                    self.entities[eid].active = False

        for i in range(len(self._removed_eids)):
            var eid = self._removed_eids[i]
            self.entities[eid].active = False

        var span_vel = Float32(2.0) * self.vel_limit
        for i in range(len(self._decoded_eids)):
            var eid = self._decoded_eids[i]
            var q = self._staging_baseline[eid]
            self.entities[eid].active = True
            self.entities[eid].pos = Vec3(
                _dequant_pos_axis(q.chunk_ix, q.qx, self.chunk_size),
                _dequant_pos_axis(q.chunk_iy, q.qy, self.chunk_size),
                _dequant_pos_axis(q.chunk_iz, q.qz, self.chunk_size),
            )
            self.entities[eid].rot = q.rot
            self.entities[eid].vel = Vec3(
                -self.vel_limit + (Float32(q.vx) / Float32(4095.0)) * span_vel,
                -self.vel_limit + (Float32(q.vy) / Float32(4095.0)) * span_vel,
                -self.vel_limit + (Float32(q.vz) / Float32(4095.0)) * span_vel,
            )
            self.entities[eid].state_flags = q.state_flags

        self.has_decoded_seq = True
        self.last_decoded_seq = snap_seq
        if pkt_has_acked_input and (
            not self.has_acked_input_seq
            or pkt_acked_input_seq > self.last_acked_input_seq
        ):
            self.has_acked_input_seq = True
            self.last_acked_input_seq = pkt_acked_input_seq
        self.decoded_packets += 1
        return (True, snap_seq)


struct _InterpFrame(Movable):
    var valid: Bool
    var time_ms: Float64
    var entities: List[ReplicatedEntity]

    def __init__(out self, max_entities: Int):
        self.valid = False
        self.time_ms = 0.0
        self.entities = List[ReplicatedEntity](capacity=max_entities)
        for i in range(max_entities):
            self.entities.append(ReplicatedEntity(i))


struct SnapshotInterpolationBuffer(Movable):
    """Client-side jitter-delay snapshot interpolation buffer (linear pos/vel + quaternion slerp)."""
    var max_entities: Int
    var capacity: Int
    var interp_delay_ms: Float64
    var frames: List[_InterpFrame]
    var slot_order: List[Int]
    var count: Int
    var write_idx: Int

    def __init__(
        out self,
        max_entities: Int = 512,
        interp_delay_ms: Float64 = 100.0,
        ring_capacity: Int = 16,
    ):
        self.max_entities = max(1, max_entities)
        self.capacity = max(2, ring_capacity)
        self.interp_delay_ms = (
            max(0.0, interp_delay_ms)
            if _rep_is_finite64(interp_delay_ms)
            else 0.0
        )
        self.frames = List[_InterpFrame](capacity=self.capacity)
        self.slot_order = List[Int](capacity=self.capacity)
        for s in range(self.capacity):
            self.frames.append(_InterpFrame(self.max_entities))
            self.slot_order.append(s)
        self.count = 0
        self.write_idx = 0

    def push_snapshot(
        mut self, server_time_ms: Float64, entities: List[ReplicatedEntity]
    ):
        """Records an authoritative decoded snapshot at `server_time_ms` in chronological order."""
        if not _rep_is_finite64(server_time_ms):
            return

        if self.count == self.capacity:
            var oldest_slot = self.slot_order[0]
            if server_time_ms < self.frames[oldest_slot].time_ms:
                return
            for i in range(self.count - 1):
                self.slot_order[i] = self.slot_order[i + 1]
            self.slot_order[self.count - 1] = oldest_slot
            self.count -= 1

        var slot = self.slot_order[self.count]
        self.frames[slot].valid = True
        self.frames[slot].time_ms = server_time_ms
        var n = min(self.max_entities, len(entities))
        for i in range(n):
            self.frames[slot].entities[i] = entities[i]
        for i in range(n, self.max_entities):
            self.frames[slot].entities[i].active = False

        var ins = self.count
        while (
            ins > 0
            and self.frames[self.slot_order[ins - 1]].time_ms > server_time_ms
        ):
            self.slot_order[ins] = self.slot_order[ins - 1]
            ins -= 1
        self.slot_order[ins] = slot
        self.count += 1
        self.write_idx = (self.write_idx + 1) % self.capacity

    def sample(
        self, render_time_ms: Float64, mut out_entities: List[ReplicatedEntity]
    ) -> Bool:
        """Samples interpolated entity transforms at `render_time_ms - interp_delay_ms`."""
        if (
            self.count == 0
            or not _rep_is_finite64(render_time_ms)
            or not _rep_is_finite64(self.interp_delay_ms)
        ):
            return False
        var target_t = render_time_ms - self.interp_delay_ms
        if not _rep_is_finite64(target_t):
            return False

        if len(out_entities) < self.max_entities:
            out_entities.clear()
            for i in range(self.max_entities):
                out_entities.append(ReplicatedEntity(i))

        var oldest_idx = self.slot_order[0]
        var newest_idx = self.slot_order[self.count - 1]

        if self.count == 1 or target_t <= self.frames[oldest_idx].time_ms:
            for i in range(self.max_entities):
                out_entities[i] = self.frames[oldest_idx].entities[i]
            return True

        if target_t >= self.frames[newest_idx].time_ms:
            for i in range(self.max_entities):
                out_entities[i] = self.frames[newest_idx].entities[i]
            return True

        # Find bracketing pair (idx0, idx1) with time0 <= target_t <= time1
        var idx0 = oldest_idx
        var idx1 = oldest_idx
        for k in range(self.count - 1):
            var a = self.slot_order[k]
            var b = self.slot_order[k + 1]
            if (
                self.frames[a].time_ms <= target_t
                and target_t <= self.frames[b].time_ms
            ):
                idx0 = a
                idx1 = b
                break

        var dt = self.frames[idx1].time_ms - self.frames[idx0].time_ms
        var alpha = Float32(0.0)
        if dt > 1.0e-6 and _rep_is_finite64(dt):
            alpha = max(
                Float32(0.0),
                min(Float32(1.0), Float32((target_t - self.frames[idx0].time_ms) / dt)),
            )
        var inv_a = Float32(1.0) - alpha

        for i in range(self.max_entities):
            var e0 = self.frames[idx0].entities[i]
            var e1 = self.frames[idx1].entities[i]
            if e0.active and e1.active:
                var dst = e1
                dst.pos = e0.pos * inv_a + e1.pos * alpha
                dst.vel = e0.vel * inv_a + e1.vel * alpha
                dst.rot = e0.rot.slerp(e1.rot, alpha)
                dst.state_flags = e0.state_flags if alpha < Float32(0.5) else e1.state_flags
                out_entities[i] = dst
            elif e1.active:
                out_entities[i] = e1
            else:
                out_entities[i].active = False
        return True
