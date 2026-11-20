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
    var cell_coords_x: List[Int32]
    var cell_coords_y: List[Int32]
    var cell_coords_z: List[Int32]

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
        self.cell_coords_x = List[Int32](length=max_entities, fill=0)
        self.cell_coords_y = List[Int32](length=max_entities, fill=0)
        self.cell_coords_z = List[Int32](length=max_entities, fill=0)

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

        # Insert in reverse entity index so forward linked-list walk visits ascending indices!
        var idx = n - 1
        while idx >= 0:
            if entities[idx].active:
                var cx = Int(floor(entities[idx].pos.x * self.inv_cell_size))
                var cy = (
                    Int(floor(entities[idx].pos.y * self.inv_cell_size))
                    if self.use_3d
                    else 0
                )
                var cz = Int(floor(entities[idx].pos.z * self.inv_cell_size))
                self.cell_coords_x[idx] = Int32(cx)
                self.cell_coords_y[idx] = Int32(cy)
                self.cell_coords_z[idx] = Int32(cz)
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
        var r = max(Float32(0.0), radius)
        var r_sq = r * r
        var min_cx = Int(floor((center.x - r) * self.inv_cell_size))
        var max_cx = Int(floor((center.x + r) * self.inv_cell_size))
        var min_cy = (
            Int(floor((center.y - r) * self.inv_cell_size))
            if self.use_3d
            else 0
        )
        var max_cy = (
            Int(floor((center.y + r) * self.inv_cell_size))
            if self.use_3d
            else 0
        )
        var min_cz = Int(floor((center.z - r) * self.inv_cell_size))
        var max_cz = Int(floor((center.z + r) * self.inv_cell_size))

        for cy in range(min_cy, max_cy + 1):
            for cz in range(min_cz, max_cz + 1):
                for cx in range(min_cx, max_cx + 1):
                    var b = self._bucket_idx(cx, cy, cz)
                    var cur = Int(self.bucket_heads[b])
                    while cur >= 0:
                        if (
                            Int(self.cell_coords_x[cur]) == cx
                            and Int(self.cell_coords_y[cur]) == cy
                            and Int(self.cell_coords_z[cur]) == cz
                        ):
                            var d: Float32
                            if self.use_3d:
                                d = (entities[cur].pos - center).length_squared()
                            else:
                                var dx = entities[cur].pos.x - center.x
                                var dz = entities[cur].pos.z - center.z
                                d = dx * dx + dz * dz
                            if d <= r_sq:
                                out_indices.append(cur)
                        cur = Int(self.next_entity[cur])


struct SnapshotReplicator(Movable):
    """Authoritative server snapshot delta replicator with spatial AOI and priority budgeting."""
    var max_entities: Int
    var max_clients: Int
    var max_packet_bytes: Int
    var world_min: Float32
    var world_max: Float32
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
    ):
        self.max_entities = max(1, max_entities)
        self.max_clients = max(1, max_clients)
        self.max_packet_bytes = max(64, min(1180, max_packet_bytes))
        self.world_min = -abs(world_bounds)
        self.world_max = abs(world_bounds)
        self.vel_limit = max(Float32(1.0), vel_limit)

        self.entities = List[ReplicatedEntity](capacity=self.max_entities)
        for i in range(self.max_entities):
            self.entities.append(ReplicatedEntity(i))
        self.quantized_now = List[_QuantizedEntitySnap](
            length=self.max_entities, fill=_QuantizedEntitySnap()
        )

        self.clients = List[_ClientRepState](capacity=self.max_clients)
        for _ in range(self.max_clients):
            self.clients.append(
                _ClientRepState(self.max_entities, interest_radius)
            )

        self.grid = SpatialInterestGrid(
            self.max_entities, cell_size, use_3d_grid
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
        var span = self.world_max - self.world_min
        var norm = max(
            Float32(0.0), min(Float32(1.0), (v - self.world_min) / span)
        )
        return Int(norm * Float32(65535.0) + Float32(0.5))

    @always_inline
    def _dequant_pos(self, q: Int) -> Float32:
        return self.world_min + (Float32(q) / Float32(65535.0)) * (
            self.world_max - self.world_min
        )

    @always_inline
    def _quant_vel(self, v: Float32) -> Int:
        var span = Float32(2.0) * self.vel_limit
        var norm = max(
            Float32(0.0), min(Float32(1.0), (v + self.vel_limit) / span)
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
        self.entities[entity_id].active = True
        self.entities[entity_id].net_id = entity_id
        self.entities[entity_id].pos = pos
        self.entities[entity_id].rot = rot.normalized()
        self.entities[entity_id].vel = vel
        self.entities[entity_id].state_flags = state_flags
        self.entities[entity_id].base_priority = max(
            Float32(0.05), base_priority
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
        self.clients[client_id].observer_pos = observer_pos
        if interest_radius > Float32(0.0):
            self.clients[client_id].interest_radius = interest_radius

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

    def prepare_tick(mut self):
        """Quantizes current world state once (including 32-bit rotations) and rebuilds the spatial grid."""
        for i in range(self.max_entities):
            if not self.entities[i].active:
                self.quantized_now[i].known = False
                continue
            var q = _QuantizedEntitySnap()
            q.known = True
            q.qx = self._quant_pos(self.entities[i].pos.x)
            q.qy = self._quant_pos(self.entities[i].pos.y)
            q.qz = self._quant_pos(self.entities[i].pos.z)
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
        if cur.qx != base.qx or cur.qy != base.qy or cur.qz != base.qz:
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
            var score = Int(
                self.clients[client_id].priority_accum[eid] * Float32(1024.0)
            )
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

        # Encode header (`8` bytes + 1-bit `has_base` flag):
        # magic u16, snap_seq u16, base_seq u16, count_placeholder u16, has_base bool
        self.writer.clear()
        self.writer.write_u16(_REPL_MAGIC)
        self.writer.write_u16(snap_seq)
        self.writer.write_u16(base_seq)
        self.writer.write_u16(0)  # patched after packing budgeted entities
        self.writer.write_bool(has_base)

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
                var dx = cur_q.qx - base_q.qx
                var dy = cur_q.qy - base_q.qy
                var dz = cur_q.qz - base_q.qz
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
    var vel_limit: Float32
    var entities: List[ReplicatedEntity]
    var baselines: List[_ClientBaselineSlot]
    var _staging_baseline: List[_QuantizedEntitySnap]
    var _staging_known_eids: List[Int]
    var _decoded_eids: List[Int]
    var _removed_eids: List[Int]
    var has_decoded_seq: Bool
    var last_decoded_seq: UInt16
    var decoded_packets: Int

    def __init__(
        out self,
        max_entities: Int = 512,
        world_bounds: Float32 = 512.0,
        vel_limit: Float32 = 120.0,
    ):
        self.max_entities = max(1, max_entities)
        self.world_min = -abs(world_bounds)
        self.world_max = abs(world_bounds)
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
        self.decoded_packets = 0

    def _clear_staging_sparse(mut self):
        for i in range(len(self._staging_known_eids)):
            var eid = self._staging_known_eids[i]
            self._staging_baseline[eid] = _QuantizedEntitySnap()
        self._staging_known_eids.clear()

    def apply_packet(mut self, pkt: List[UInt8]) -> Tuple[Bool, UInt16]:
        if len(pkt) < 9:
            return (False, UInt16(0))
        var r = BitReader(pkt)
        var magic = r.read_u16()
        if magic != _REPL_MAGIC:
            return (False, UInt16(0))
        var snap_seq = r.read_u16()
        var base_seq = r.read_u16()
        var count = Int(r.read_u16())
        var has_base = r.read_bool()
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
                    q.qx += r.read_int_bounded(-127, 127)
                    q.qy += r.read_int_bounded(-127, 127)
                    q.qz += r.read_int_bounded(-127, 127)
                else:
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

        for i in range(len(self._removed_eids)):
            var eid = self._removed_eids[i]
            self.entities[eid].active = False

        var span_pos = self.world_max - self.world_min
        var span_vel = Float32(2.0) * self.vel_limit
        for i in range(len(self._decoded_eids)):
            var eid = self._decoded_eids[i]
            var q = self._staging_baseline[eid]
            self.entities[eid].active = True
            self.entities[eid].pos = Vec3(
                self.world_min + (Float32(q.qx) / Float32(65535.0)) * span_pos,
                self.world_min + (Float32(q.qy) / Float32(65535.0)) * span_pos,
                self.world_min + (Float32(q.qz) / Float32(65535.0)) * span_pos,
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
        self.interp_delay_ms = max(0.0, interp_delay_ms)
        self.frames = List[_InterpFrame](capacity=self.capacity)
        for _ in range(self.capacity):
            self.frames.append(_InterpFrame(self.max_entities))
        self.count = 0
        self.write_idx = 0

    def push_snapshot(
        mut self, server_time_ms: Float64, entities: List[ReplicatedEntity]
    ):
        """Records an authoritative decoded snapshot at `server_time_ms`."""
        var slot = self.write_idx
        self.frames[slot].valid = True
        self.frames[slot].time_ms = server_time_ms
        var n = min(self.max_entities, len(entities))
        for i in range(n):
            self.frames[slot].entities[i] = entities[i]
        for i in range(n, self.max_entities):
            self.frames[slot].entities[i].active = False
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def sample(
        self, render_time_ms: Float64, mut out_entities: List[ReplicatedEntity]
    ) -> Bool:
        """Samples interpolated entity transforms at `render_time_ms - interp_delay_ms`."""
        if self.count == 0:
            return False
        if len(out_entities) < self.max_entities:
            out_entities.clear()
            for i in range(self.max_entities):
                out_entities.append(ReplicatedEntity(i))

        var target_t = render_time_ms - self.interp_delay_ms
        var oldest_idx = (
            (self.write_idx - self.count + self.capacity) % self.capacity
        )
        var newest_idx = (self.write_idx - 1 + self.capacity) % self.capacity

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
            var a = (oldest_idx + k) % self.capacity
            var b = (oldest_idx + k + 1) % self.capacity
            if (
                self.frames[a].time_ms <= target_t
                and target_t <= self.frames[b].time_ms
            ):
                idx0 = a
                idx1 = b
                break

        var dt = self.frames[idx1].time_ms - self.frames[idx0].time_ms
        var alpha = Float32(0.0)
        if dt > 1.0e-6:
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
