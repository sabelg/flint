"""Flint v5 Transform Hierarchy & Parallel Depth-Bucketed Propagation (`flint.scene.transform_hierarchy`).

Provides:
- `trait SerializableComponent(Component)` (`SCHEMA_NAME`, `SCHEMA_FOURCC`, `SCHEMA_VERSION`,
  `write_bin`, `read_bin`, `write_json`, `read_json`)
- `LocalTransform`, `GlobalTransform`, `PrevGlobalTransform`
- `Parent`, `Children` (inline 4-child array + overflow list with binary snapshot hooks), `HierarchyDepth`
- Cycle-guarded reparenting (`set_parent`, `clear_parent`)
- Depth-bucketed level-synchronous parallel `JobPool` `GlobalTransform` propagation
  (`propagate_transforms_parallel`, `snapshot_prev_global_transforms`)
- `interpolate_global_transform(prev, curr, alpha)` (TRS lerp + `Quat.slerp`)
- `TransformPlugin` (`EnginePlugin` wiring `PrevGlobalTransform` capture and `STAGE_POST_UPDATE` propagation).
"""
from std.collections import List
from std.ffi import external_call
from std.math import isfinite, max, min
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool
from flint.core.math import Vec3, Quat, Mat4
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.core.transform import trs
from flint.assets.json import JsonDocument
from flint.ecs.world import (
    World,
    Entity,
    NULL_ENTITY,
    Component,
    BitSet256,
    ParallelCommands,
    STORAGE_TABLE,
    comp_hash64,
)
from flint.app.engine_app import (
    EnginePlugin,
    EngineApp,
    Resource,
    Resources,
    WorldTickSyncResource,
    STAGE_FIXED_UPDATE,
    STAGE_POST_UPDATE,
)


# ---------------------------------------------------------------------------
# SerializableComponent Trait
# ---------------------------------------------------------------------------

trait SerializableComponent(Component):
    """Component with versioned binary (`BlobWriter`/`BlobReader`) and deterministic JSON codecs."""
    comptime SCHEMA_NAME: StaticString
    comptime SCHEMA_FOURCC: UInt32
    comptime SCHEMA_VERSION: Int

    def write_bin(self, mut out: BlobWriter):
        ...

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        ...

    def write_json(self) -> String:
        ...

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        ...


@always_inline
def _f32_json(v: Float32) -> String:
    return String(Float64(v)) if isfinite(v) else String("0.0")


@always_inline
def _vec3_finite(v: Vec3) -> Bool:
    return isfinite(v.x) and isfinite(v.y) and isfinite(v.z)


@always_inline
def _quat_finite(q: Quat) -> Bool:
    return isfinite(q.x) and isfinite(q.y) and isfinite(q.z) and isfinite(q.w)


def _vec3_json(v: Vec3) -> String:
    return "[" + _f32_json(v.x) + ", " + _f32_json(v.y) + ", " + _f32_json(v.z) + "]"


def _quat_json(q: Quat) -> String:
    return (
        "["
        + _f32_json(q.x)
        + ", "
        + _f32_json(q.y)
        + ", "
        + _f32_json(q.z)
        + ", "
        + _f32_json(q.w)
        + "]"
    )


def _json_vec3_all_finite(doc: JsonDocument, node: Int) -> Bool:
    if node < 0 or doc.array_size(node) < 3:
        return True
    for i in range(3):
        var d = doc.number(doc.at(node, i), 0.0)
        if not isfinite(d) or not isfinite(Float32(d)):
            return False
    return True


def _json_quat_all_finite(doc: JsonDocument, node: Int) -> Bool:
    if node < 0 or doc.array_size(node) < 4:
        return True
    for i in range(4):
        var d = doc.number(doc.at(node, i), 0.0)
        if not isfinite(d) or not isfinite(Float32(d)):
            return False
    return True


def _read_vec3_json(doc: JsonDocument, node: Int, fallback: Vec3 = Vec3()) -> Vec3:
    if doc.array_size(node) < 3:
        return fallback
    var x = Float32(doc.number(doc.at(node, 0), Float64(fallback.x)))
    var y = Float32(doc.number(doc.at(node, 1), Float64(fallback.y)))
    var z = Float32(doc.number(doc.at(node, 2), Float64(fallback.z)))
    return Vec3(
        x if isfinite(x) else fallback.x,
        y if isfinite(y) else fallback.y,
        z if isfinite(z) else fallback.z,
    )


def _read_quat_json(doc: JsonDocument, node: Int) -> Quat:
    if doc.array_size(node) < 4:
        return Quat.identity()
    var x = Float32(doc.number(doc.at(node, 0), 0.0))
    var y = Float32(doc.number(doc.at(node, 1), 0.0))
    var z = Float32(doc.number(doc.at(node, 2), 0.0))
    var w = Float32(doc.number(doc.at(node, 3), 1.0))
    if not (isfinite(x) and isfinite(y) and isfinite(z) and isfinite(w)):
        return Quat.identity()
    return Quat(x, y, z, w)


# ---------------------------------------------------------------------------
# Transform & Hierarchy Components
# ---------------------------------------------------------------------------

struct LocalTransform(SerializableComponent, ImplicitlyCopyable):
    """Authoritative local-space TRS transform relative to `Parent` (or world root)."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.scene.LocalTransform")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "LocalTransform"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("LTRF")
    comptime SCHEMA_VERSION: Int = 1

    var translation: Vec3
    var rotation: Quat
    var scale: Vec3

    def __init__(
        out self,
        translation: Vec3 = Vec3(0.0, 0.0, 0.0),
        rotation: Quat = Quat(0.0, 0.0, 0.0, 1.0),
        scale: Vec3 = Vec3(1.0, 1.0, 1.0),
    ):
        self.translation = translation
        self.rotation = rotation
        self.scale = scale

    @staticmethod
    @always_inline
    def identity() -> Self:
        return Self()

    @always_inline
    def to_mat4(self) -> Mat4:
        return trs(self.translation, self.rotation, self.scale)

    def write_bin(self, mut out: BlobWriter):
        out.vec3(self.translation)
        out.quat(self.rotation)
        out.vec3(self.scale)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var t = reader.vec3()
        var r = reader.quat()
        var s = reader.vec3()
        if reader.failed:
            raise Error("LocalTransform: truncated blob")
        if not (_vec3_finite(t) and _quat_finite(r) and _vec3_finite(s)):
            raise Error("LocalTransform: non-finite transform component")
        return Self(t, r, s)

    def write_json(self) -> String:
        return (
            "{\"translation\": "
            + _vec3_json(self.translation)
            + ", \"rotation\": "
            + _quat_json(self.rotation)
            + ", \"scale\": "
            + _vec3_json(self.scale)
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var t_node = doc.object_get(node, "translation")
        if t_node < 0:
            t_node = doc.object_get(node, "position")
        var r_node = doc.object_get(node, "rotation")
        var s_node = doc.object_get(node, "scale")
        if (
            not _json_vec3_all_finite(doc, t_node)
            or not _json_quat_all_finite(doc, r_node)
            or not _json_vec3_all_finite(doc, s_node)
        ):
            raise Error("LocalTransform: non-finite transform component in JSON")
        var t = _read_vec3_json(doc, t_node, Vec3(0, 0, 0))
        var r = _read_quat_json(doc, r_node)
        var s = _read_vec3_json(doc, s_node, Vec3(1, 1, 1))
        return Self(t, r, s)


struct GlobalTransform(SerializableComponent):
    """World-space 4x4 matrix + TRS representation computed by `TransformPlugin`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.scene.GlobalTransform")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "GlobalTransform"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("GTRF")
    comptime SCHEMA_VERSION: Int = 1

    var matrix: Mat4
    var translation: Vec3
    var rotation: Quat
    var scale: Vec3

    def __init__(
        out self,
        var matrix: Mat4,
        translation: Vec3,
        rotation: Quat,
        scale: Vec3,
    ):
        self.matrix = matrix^
        self.translation = translation
        self.rotation = rotation
        self.scale = scale

    def __init__(out self):
        self.matrix = Mat4()
        self.translation = Vec3(0.0, 0.0, 0.0)
        self.rotation = Quat(0.0, 0.0, 0.0, 1.0)
        self.scale = Vec3(1.0, 1.0, 1.0)

    @staticmethod
    def from_local(local: LocalTransform) -> Self:
        return Self(local.to_mat4(), local.translation, local.rotation.normalized(), local.scale)

    def compose_child(self, local: LocalTransform) -> Self:
        var m = self.matrix.multiply(local.to_mat4())
        var world_pos = Vec3(m.at(0, 3), m.at(1, 3), m.at(2, 3))
        var world_rot = (self.rotation * local.rotation).normalized()
        var world_scale = Vec3(
            self.scale.x * local.scale.x,
            self.scale.y * local.scale.y,
            self.scale.z * local.scale.z,
        )
        return Self(m^, world_pos, world_rot, world_scale)

    @always_inline
    def transform_point(self, p: Vec3) -> Vec3:
        return self.matrix.transform_point(p)

    def write_bin(self, mut out: BlobWriter):
        for i in range(16):
            out.f32(self.matrix.values[i])
        out.vec3(self.translation)
        out.quat(self.rotation)
        out.vec3(self.scale)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var m = Mat4()
        for i in range(16):
            var mv = reader.f32()
            if not isfinite(mv):
                raise Error("GlobalTransform: non-finite matrix entry")
            m.values[i] = mv
        var t = reader.vec3()
        var r = reader.quat()
        var s = reader.vec3()
        if reader.failed:
            raise Error("GlobalTransform: truncated blob")
        if not (_vec3_finite(t) and _quat_finite(r) and _vec3_finite(s)):
            raise Error("GlobalTransform: non-finite transform component")
        return Self(m^, t, r, s)

    def write_json(self) -> String:
        return (
            "{\"translation\": "
            + _vec3_json(self.translation)
            + ", \"rotation\": "
            + _quat_json(self.rotation)
            + ", \"scale\": "
            + _vec3_json(self.scale)
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var t_node = doc.object_get(node, "translation")
        if t_node < 0:
            t_node = doc.object_get(node, "position")
        var r_node = doc.object_get(node, "rotation")
        var s_node = doc.object_get(node, "scale")
        if (
            not _json_vec3_all_finite(doc, t_node)
            or not _json_quat_all_finite(doc, r_node)
            or not _json_vec3_all_finite(doc, s_node)
        ):
            raise Error("GlobalTransform: non-finite transform component in JSON")
        var t = _read_vec3_json(doc, t_node, Vec3(0, 0, 0))
        var r = _read_quat_json(doc, r_node)
        var s = _read_vec3_json(doc, s_node, Vec3(1, 1, 1))
        return Self(trs(t, r, s), t, r, s)


struct PrevGlobalTransform(SerializableComponent):
    """Previous fixed-tick `GlobalTransform` for smooth `overstep_alpha` render interpolation."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.scene.PrevGlobalTransform")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "PrevGlobalTransform"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("PGTR")
    comptime SCHEMA_VERSION: Int = 1

    var value: GlobalTransform

    def __init__(out self, var value: GlobalTransform = GlobalTransform()):
        self.value = value^

    def write_bin(self, mut out: BlobWriter):
        self.value.write_bin(out)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        return Self(GlobalTransform.read_bin(reader, version))

    def write_json(self) -> String:
        return self.value.write_json()

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        return Self(GlobalTransform.read_json(doc, node))


@fieldwise_init
struct Parent(Component, ImplicitlyCopyable):
    """Parent entity reference in the transform hierarchy."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.scene.Parent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var entity: Entity


comptime INLINE_CHILD_CAP: Int = 4


struct Children(Component):
    """Child entity list with zero-alloc inline 4-child array + heap overflow list."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.scene.Children")
    comptime STORAGE_MODE: Int = STORAGE_TABLE

    var inline_items: Array[Entity, INLINE_CHILD_CAP]
    var total_count: Int32
    var overflow: List[Entity]

    def __init__(out self):
        self.inline_items = Array[Entity, INLINE_CHILD_CAP](fill=NULL_ENTITY)
        self.total_count = 0
        self.overflow = List[Entity]()

    def __init__(out self, *, copy: Self):
        self.inline_items = copy.inline_items.copy()
        self.total_count = copy.total_count
        self.overflow = copy.overflow.copy()

    @always_inline
    def len(self) -> Int:
        return Int(self.total_count)

    @always_inline
    def at(self, idx: Int) -> Entity:
        if idx < 0 or idx >= Int(self.total_count):
            return NULL_ENTITY
        if idx < INLINE_CHILD_CAP:
            return self.inline_items[idx]
        return self.overflow[idx - INLINE_CHILD_CAP]

    def contains(self, child: Entity) -> Bool:
        for i in range(self.len()):
            if self.at(i) == child:
                return True
        return False

    def push_unique(mut self, child: Entity):
        if self.contains(child):
            return
        var c = Int(self.total_count)
        if c < INLINE_CHILD_CAP:
            self.inline_items[c] = child
        else:
            self.overflow.append(child)
        self.total_count = Int32(c + 1)

    def remove(mut self, child: Entity) -> Bool:
        var n = Int(self.total_count)
        var found = -1
        for i in range(n):
            if self.at(i) == child:
                found = i
                break
        if found < 0:
            return False
        for i in range(found, n - 1):
            var next_val = self.at(i + 1)
            if i < INLINE_CHILD_CAP:
                self.inline_items[i] = next_val
            else:
                self.overflow[i - INLINE_CHILD_CAP] = next_val
        var last = n - 1
        if last < INLINE_CHILD_CAP:
            self.inline_items[last] = NULL_ENTITY
        else:
            _ = self.overflow.pop()
        self.total_count = Int32(last)
        return True


def _children_bin_write(comp_addr: Int, writer_addr: Int):
    ref ch = Pointer[Children, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref w = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    var n = ch.len()
    w.integer(n)
    for i in range(n):
        w.u64(ch.at(i).to_bits())


def _children_bin_read(comp_addr: Int, reader_addr: Int) -> Bool:
    ref r = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    var n = r.integer()
    if n < 0 or r.failed:
        return False
    var rem = r.limit() - r.offset
    if rem < 0 or n > rem // 8:
        return False
    var built = Children()
    for _ in range(n):
        var bits = r.u64()
        if r.failed:
            return False
        built.push_unique(Entity.from_bits(bits))
    if r.failed:
        return False
    var dst = Pointer[Children, MutAnyOrigin](unsafe_from_address=comp_addr)
    __get_address_as_uninit_lvalue(dst._mlir_value) = built^
    return True


@fieldwise_init
struct HierarchyDepth(Component, ImplicitlyCopyable):
    """Distance from root (`0` for root entities) used for level-synchronous parallel propagation."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.scene.HierarchyDepth")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var depth: Int32


def register_transform_components(mut world: World):
    """Idempotently registers all transform hierarchy components + custom `Children` snapshot hooks."""
    _ = world.register_component[LocalTransform]()
    _ = world.register_component[GlobalTransform]()
    _ = world.register_component[PrevGlobalTransform]()
    _ = world.register_component[Parent]()
    var ch_tid = world.register_component[Children]()
    if ch_tid >= 0 and not world.components[ch_tid].has_custom_bin:
        world.set_component_binary_hooks(ch_tid, _children_bin_write, _children_bin_read)
    _ = world.register_component[HierarchyDepth]()


# ---------------------------------------------------------------------------
# Cycle-Guarded Reparenting & Subtree Depth Maintenance
# ---------------------------------------------------------------------------

def _update_subtree_depths(mut world: World, root: Entity, root_depth: Int):
    var queue_ent = List[Entity]()
    var queue_dep = List[Int]()
    queue_ent.append(root)
    queue_dep.append(root_depth)
    var head = 0
    while head < len(queue_ent):
        var cur = queue_ent[head]
        var dep = queue_dep[head]
        head += 1
        _ = world.insert[HierarchyDepth](cur, HierarchyDepth(Int32(dep)))
        var ch_opt = world.get_ptr[Children](cur)
        if ch_opt:
            var ch_ptr = ch_opt.value()
            var n = ch_ptr[unsafe_offset=0].len()
            for i in range(n):
                var child_e = ch_ptr[unsafe_offset=0].at(i)
                if world.is_alive(child_e):
                    queue_ent.append(child_e)
                    queue_dep.append(dep + 1)


def set_parent(mut world: World, child: Entity, new_parent: Entity) -> Bool:
    """Cycle-guarded reparenting: links `child` under `new_parent`, updating `Parent`, `Children`,
    and `HierarchyDepth` across `child`'s subtree. Returns `False` if invalid or cyclic."""
    register_transform_components(world)
    if not world.is_alive(child) or not world.is_alive(new_parent):
        return False
    if child == new_parent:
        return False

    # Walk ancestors of new_parent to reject any cycle (child cannot be an ancestor of new_parent)
    var max_hops = world.entity_count() + 1
    var cursor = new_parent
    var hops = 0
    while hops < max_hops:
        var p_opt = world.get_ptr[Parent](cursor)
        if not p_opt:
            break
        var next_p = p_opt.value()[unsafe_offset=0].entity
        if next_p == child:
            return False
        if not world.is_alive(next_p):
            break
        cursor = next_p
        hops += 1
    if hops >= max_hops:
        return False

    # Detach from old parent if different
    var old_p_opt = world.get_ptr[Parent](child)
    if old_p_opt:
        var old_p = old_p_opt.value()[unsafe_offset=0].entity
        if old_p == new_parent:
            return True
        if world.is_alive(old_p):
            var old_ch = world.get_ptr_mut[Children](old_p)
            if old_ch:
                _ = old_ch.value()[unsafe_offset=0].remove(child)

    # Ensure new_parent has HierarchyDepth and Children
    var parent_depth = 0
    var pd_opt = world.get_ptr[HierarchyDepth](new_parent)
    if pd_opt:
        parent_depth = Int(pd_opt.value()[unsafe_offset=0].depth)
    else:
        _ = world.insert[HierarchyDepth](new_parent, HierarchyDepth(0))

    if not world.has[Children](new_parent):
        _ = world.insert[Children](new_parent, Children())
    var new_ch = world.get_ptr_mut[Children](new_parent)
    if new_ch:
        new_ch.value()[unsafe_offset=0].push_unique(child)

    _ = world.insert[Parent](child, Parent(new_parent))
    _update_subtree_depths(world, child, parent_depth + 1)
    return True


def clear_parent(mut world: World, child: Entity) -> Bool:
    """Detaches `child` into a root entity (`depth = 0`) and updates its subtree depths."""
    if not world.is_alive(child):
        return False
    var old_p_opt = world.get_ptr[Parent](child)
    if not old_p_opt:
        return False
    var old_p = old_p_opt.value()[unsafe_offset=0].entity
    if world.is_alive(old_p):
        var old_ch = world.get_ptr_mut[Children](old_p)
        if old_ch:
            _ = old_ch.value()[unsafe_offset=0].remove(child)
    _ = world.remove[Parent](child)
    _update_subtree_depths(world, child, 0)
    return True


# ---------------------------------------------------------------------------
# Depth-Bucketed Level-Synchronous Parallel GlobalTransform Propagation
# ---------------------------------------------------------------------------

comptime _PROP_BATCH_SIZE: Int = 32
comptime MAX_HIERARCHY_DEPTH: Int = 65536


@fieldwise_init
struct _PropNodeItem(Copyable, ImplicitlyCopyable, Movable):
    var local_addr: Int
    var parent_gt_addr: Int
    var dst_gt_addr: Int
    var prev_gt_addr: Int
    var entity_idx: Int32
    var parent_idx: Int32
    var self_dirty: Bool


struct TransformPropagationCache(Resource):
    """Persistent depth buckets and change-tracking state for zero-realloc parallel transform propagation."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.scene.TransformPropagationCache")
    var buckets: List[List[_PropNodeItem]]
    var dirty_flags: List[UInt8]
    var last_propagated_tick: UInt32
    var last_restore_epoch: UInt64
    var last_propagated_count: Int
    var last_skipped_count: Int

    def __init__(out self):
        self.buckets = List[List[_PropNodeItem]]()
        self.dirty_flags = List[UInt8]()
        self.last_propagated_tick = 0
        self.last_restore_epoch = 0
        self.last_propagated_count = 0
        self.last_skipped_count = 0

    def invalidate(mut self):
        """Forces the next propagation pass to recompute all entities."""
        self.last_propagated_tick = 0


@fieldwise_init
struct _PropLevelCtx(Copyable, ImplicitlyCopyable, Movable):
    var items_ptr: Int
    var item_count: Int
    var copy_to_prev: Bool


def _prop_level_worker(ctx_addr: Int, batch_idx: Int):
    var ctx = Pointer[_PropLevelCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var start = batch_idx * _PROP_BATCH_SIZE
    var stop = min(ctx.item_count, start + _PROP_BATCH_SIZE)
    var items = Pointer[_PropNodeItem, ImmutAnyOrigin](unsafe_from_address=ctx.items_ptr)
    for i in range(start, stop):
        var it = items[unsafe_offset=i]
        var loc = Pointer[LocalTransform, ImmutAnyOrigin](unsafe_from_address=it.local_addr)[unsafe_offset=0]
        var dst = Pointer[GlobalTransform, MutAnyOrigin](unsafe_from_address=it.dst_gt_addr)
        if ctx.copy_to_prev and it.prev_gt_addr != 0:
            var prev = Pointer[PrevGlobalTransform, MutAnyOrigin](unsafe_from_address=it.prev_gt_addr)
            prev[unsafe_offset=0].value = dst[unsafe_offset=0].copy()
        if it.parent_gt_addr == 0:
            dst[unsafe_offset=0] = GlobalTransform.from_local(loc)
        else:
            ref pgt = Pointer[GlobalTransform, ImmutAnyOrigin](unsafe_from_address=it.parent_gt_addr)[unsafe_offset=0]
            dst[unsafe_offset=0] = pgt.compose_child(loc)


def snapshot_prev_global_transforms(mut world: World, mut pool: JobPool):
    """Copies current `GlobalTransform` into `PrevGlobalTransform` in parallel prior to a fixed tick."""
    world.par_for_each2[GlobalTransform, PrevGlobalTransform](pool, _copy_gt_to_prev_kernel)


def _copy_gt_to_prev_kernel(
    _lane: Int,
    _order: Int,
    _e: Entity,
    gt: Pointer[GlobalTransform, MutAnyOrigin],
    prev: Pointer[PrevGlobalTransform, MutAnyOrigin],
    _ctx: Int,
):
    prev[unsafe_offset=0].value = gt[unsafe_offset=0].copy()


def propagate_transforms_parallel_cached(
    mut world: World,
    mut pool: JobPool,
    mut cache: TransformPropagationCache,
    copy_to_prev: Bool = False,
    allow_insert_missing: Bool = True,
    cmds_addr: Int = 0,
):
    """Dirty-tracked, depth-bucketed level-synchronous parallel `GlobalTransform` propagation.
    Reuses `cache.buckets` across frames (zero per-tick bucket reallocation) and skips static
    root chunks and static subtrees whose `LocalTransform` / `Parent` / `HierarchyDepth`
    `changed_ticks` have not advanced since `cache.last_propagated_tick`."""
    if allow_insert_missing:
        register_transform_components(world)
    var tid_local = world.component_id[LocalTransform]()
    var tid_global = world.component_id[GlobalTransform]()
    var tid_prev = world.component_id[PrevGlobalTransform]()
    var tid_parent = world.component_id[Parent]()
    var tid_depth = world.component_id[HierarchyDepth]()
    if tid_local < 0 or tid_global < 0:
        return

    # Auto-attach missing GlobalTransform on any newly spawned LocalTransform entities.
    # In non-exclusive scheduled mode (`allow_insert_missing=False`), defer via ParallelCommands.
    var missing_gt = world.query_entities(
        BitSet256().with_bit(tid_local),
        BitSet256().with_bit(tid_global),
    )
    if len(missing_gt) > 0:
        if allow_insert_missing:
            for i in range(len(missing_gt)):
                var e = missing_gt[i]
                var loc = world.get[LocalTransform](e)
                _ = world.insert[GlobalTransform](e, GlobalTransform.from_local(loc))
        elif cmds_addr != 0:
            ref cmds = Pointer[ParallelCommands, MutAnyOrigin](unsafe_from_address=cmds_addr)[unsafe_offset=0]
            for i in range(len(missing_gt)):
                var e = missing_gt[i]
                var loc = world.get[LocalTransform](e)
                cmds.insert[GlobalTransform](0, e, GlobalTransform.from_local(loc))

    # Clear persistent depth buckets without freeing their backing allocations
    for d in range(len(cache.buckets)):
        cache.buckets[d].clear()

    var ent_cap = len(world.entities)
    if len(cache.dirty_flags) < ent_cap:
        cache.dirty_flags.resize(ent_cap, UInt8(0))
    if ent_cap > 0:
        _ = external_call["memset", Int](Int(cache.dirty_flags.unsafe_ptr()), Int32(0), ent_cap)

    var last_tick = cache.last_propagated_tick
    var force_all = (
        (last_tick == 0)
        or (world.current_tick <= last_tick)
        or (world.restore_epoch != cache.last_restore_epoch)
        or copy_to_prev
    )
    var skipped = 0
    var propagated = 0
    var with_lt_gt = BitSet256().with_bit(tid_local).with_bit(tid_global)

    for a in range(len(world.archetypes)):
        if not world.archetypes[a].table_mask.contains_all(with_lt_gt):
            continue
        var col_loc = world.archetypes[a].col_index(tid_local)
        var col_gt = world.archetypes[a].col_index(tid_global)
        var col_prev = world.archetypes[a].col_index(tid_prev)
        var col_par = world.archetypes[a].col_index(tid_parent)
        var col_dep = world.archetypes[a].col_index(tid_depth)

        for ch in range(len(world.archetypes[a].chunks)):
            var cnt = world.archetypes[a].chunks[ch].count
            if cnt <= 0:
                continue

            # Root-only chunk fast skip: no Parent component and no LocalTransform/HierarchyDepth changes
            if not force_all and col_par < 0:
                var loc_chg = world.archetypes[a].chunks[ch].any_group_changed_since(col_loc, last_tick)
                var dep_chg = (
                    col_dep >= 0
                    and world.archetypes[a].chunks[ch].any_group_changed_since(col_dep, last_tick)
                )
                if not loc_chg and not dep_chg:
                    skipped += cnt
                    continue

            var chunk_any_dirty = False
            for r in range(cnt):
                var e = world.archetypes[a].chunks[ch].entities[r]
                var e_idx = Int(e.index)
                var self_dirty = force_all
                if not self_dirty:
                    if world.archetypes[a].chunks[ch].row_changed_since(col_loc, r, last_tick):
                        self_dirty = True
                    elif col_par >= 0 and world.archetypes[a].chunks[ch].row_changed_since(col_par, r, last_tick):
                        self_dirty = True
                    elif col_dep >= 0 and world.archetypes[a].chunks[ch].row_changed_since(col_dep, r, last_tick):
                        self_dirty = True

                var depth = 0
                var parent_gt_addr = 0
                var parent_idx = Int32(-1)
                if col_par >= 0:
                    var par_e = Pointer[Parent, ImmutAnyOrigin](
                        unsafe_from_address=world.archetypes[a].chunks[ch].cell_ptr(col_par, r)
                    )[unsafe_offset=0].entity
                    if world.is_alive(par_e):
                        parent_gt_addr = world.get_raw_ptr(par_e, tid_global)
                        parent_idx = Int32(par_e.index)
                        if col_dep >= 0:
                            var max_depth_cap = min(MAX_HIERARCHY_DEPTH, max(1, ent_cap))
                            depth = min(
                                max_depth_cap,
                                max(
                                    1,
                                    Int(
                                        Pointer[HierarchyDepth, ImmutAnyOrigin](
                                            unsafe_from_address=world.archetypes[a].chunks[ch].cell_ptr(col_dep, r)
                                        )[unsafe_offset=0].depth
                                    ),
                                ),
                            )
                        else:
                            depth = 1

                if depth == 0 and not self_dirty:
                    skipped += 1
                    continue

                if self_dirty:
                    chunk_any_dirty = True
                    if e_idx >= 0 and e_idx < len(cache.dirty_flags):
                        cache.dirty_flags[e_idx] = UInt8(1)

                var loc_addr = world.archetypes[a].chunks[ch].cell_ptr(col_loc, r)
                var gt_addr = world.archetypes[a].chunks[ch].cell_ptr(col_gt, r)
                var prev_addr = world.archetypes[a].chunks[ch].cell_ptr(col_prev, r) if col_prev >= 0 else 0
                while len(cache.buckets) <= depth:
                    cache.buckets.append(List[_PropNodeItem]())
                cache.buckets[depth].append(
                    _PropNodeItem(loc_addr, parent_gt_addr, gt_addr, prev_addr, Int32(e_idx), parent_idx, self_dirty)
                )

            if chunk_any_dirty or col_par >= 0:
                if copy_to_prev and col_prev >= 0:
                    world.archetypes[a].chunks[ch].mark_all_groups_changed(col_prev, world.current_tick)

    # Execute level-by-level across JobPool with strict barrier between depth d and d + 1.
    # For depth >= 1, filter in-place to entities that are self-dirty or have a dirty parent.
    for d in range(len(cache.buckets)):
        var count_d = len(cache.buckets[d])
        if count_d <= 0:
            continue
        var active_d = count_d
        if d > 0 and not force_all:
            var write_i = 0
            for i in range(count_d):
                var it = cache.buckets[d][i]
                var parent_dirty = (
                    it.parent_idx >= 0
                    and Int(it.parent_idx) < len(cache.dirty_flags)
                    and cache.dirty_flags[Int(it.parent_idx)] != 0
                )
                if it.self_dirty or parent_dirty:
                    if it.entity_idx >= 0 and Int(it.entity_idx) < len(cache.dirty_flags):
                        cache.dirty_flags[Int(it.entity_idx)] = UInt8(1)
                    if write_i != i:
                        cache.buckets[d][write_i] = it
                    write_i += 1
                else:
                    skipped += 1
            active_d = write_i
        else:
            for i in range(count_d):
                var e_idx = Int(cache.buckets[d][i].entity_idx)
                if e_idx >= 0 and e_idx < len(cache.dirty_flags):
                    cache.dirty_flags[e_idx] = UInt8(1)

        if active_d <= 0:
            continue
        propagated += active_d
        var batches = (active_d + _PROP_BATCH_SIZE - 1) // _PROP_BATCH_SIZE
        var level_ctx = _PropLevelCtx(Int(cache.buckets[d].unsafe_ptr()), active_d, copy_to_prev)
        run_parallel_ctx(pool, batches, level_ctx, _prop_level_worker)
        var anchor_b = len(cache.buckets[d])
        _ = anchor_b

    if propagated > 0:
        for a in range(len(world.archetypes)):
            if not world.archetypes[a].table_mask.contains_all(with_lt_gt):
                continue
            var col_gt = world.archetypes[a].col_index(tid_global)
            for ch in range(len(world.archetypes[a].chunks)):
                var cnt = world.archetypes[a].chunks[ch].count
                for r in range(cnt):
                    var e_idx = Int(world.archetypes[a].chunks[ch].entities[r].index)
                    if e_idx >= 0 and e_idx < len(cache.dirty_flags) and cache.dirty_flags[e_idx] != 0:
                        world.archetypes[a].chunks[ch].mark_changed(col_gt, r, world.current_tick)

    cache.last_propagated_tick = world.current_tick
    cache.last_restore_epoch = world.restore_epoch
    cache.last_propagated_count = propagated
    cache.last_skipped_count = skipped

    for a in range(len(world.archetypes)):
        for ch in range(len(world.archetypes[a].chunks)):
            world.archetypes[a].chunks[ch].keep_alive()
    world.keep_alive()
    var anchor_buckets = len(cache.buckets)
    _ = anchor_buckets


def propagate_transforms_parallel(
    mut world: World,
    mut pool: JobPool,
    copy_to_prev: Bool = False,
):
    """Level-synchronous depth-bucketed parallel `GlobalTransform` propagation across `JobPool`.
    Ensures every `LocalTransform` entity has `GlobalTransform`, then dispatches each hierarchy
    depth `0 .. max_depth` across worker threads with a strict level barrier between depths."""
    var local_cache = TransformPropagationCache()
    propagate_transforms_parallel_cached(
        world, pool, local_cache, copy_to_prev=copy_to_prev, allow_insert_missing=True
    )


# ---------------------------------------------------------------------------
# Render Interpolation (PrevGlobalTransform -> GlobalTransform via overstep_alpha)
# ---------------------------------------------------------------------------

def interpolate_global_transform(
    prev: GlobalTransform,
    curr: GlobalTransform,
    alpha: Float32,
) -> GlobalTransform:
    """Smoothly interpolates `prev` and `curr` world transforms using `alpha` in `[0, 1]`
    (linear translation/scale + shortest-arc `Quat.slerp` rotation, free of matrix-lerp shear)."""
    var a = max(Float32(0.0), min(Float32(1.0), alpha)) if isfinite(alpha) else Float32(0.0)
    var pos = prev.translation + (curr.translation - prev.translation) * a
    var rot = prev.rotation.slerp(curr.rotation, a)
    var scl = prev.scale + (curr.scale - prev.scale) * a
    return GlobalTransform(trs(pos, rot, scl), pos, rot, scl)


def interpolate_entity_transform(world: World, entity: Entity, alpha: Float32) -> GlobalTransform:
    var curr_opt = world.get_ptr[GlobalTransform](entity)
    if not curr_opt:
        return GlobalTransform()
    var prev_opt = world.get_ptr[PrevGlobalTransform](entity)
    if not prev_opt:
        return curr_opt.value()[unsafe_offset=0].copy()
    return interpolate_global_transform(
        prev_opt.value()[unsafe_offset=0].value,
        curr_opt.value()[unsafe_offset=0],
        alpha,
    )


# ---------------------------------------------------------------------------
# TransformPlugin (EnginePlugin)
# ---------------------------------------------------------------------------

def _sys_snapshot_prev_transforms(
    world_ptr: Pointer[World, MutAnyOrigin],
    _res_ptr: Pointer[Resources, MutAnyOrigin],
    pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    snapshot_prev_global_transforms(world_ptr[unsafe_offset=0], pool_ptr[unsafe_offset=0])


def _sys_propagate_transforms(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    pool_ptr: Pointer[JobPool, MutAnyOrigin],
    cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var cache_opt = res_ptr[unsafe_offset=0].get_ptr_mut[TransformPropagationCache]()
    if cache_opt:
        var sync_opt = res_ptr[unsafe_offset=0].get_ptr[WorldTickSyncResource]()
        if sync_opt:
            ref sync = sync_opt.value()[unsafe_offset=0]
            if (
                sync.has_observed
                and sync.min_tick_since_post_update <= cache_opt.value()[unsafe_offset=0].last_propagated_tick
            ):
                cache_opt.value()[unsafe_offset=0].invalidate()
        propagate_transforms_parallel_cached(
            world_ptr[unsafe_offset=0],
            pool_ptr[unsafe_offset=0],
            cache_opt.value()[unsafe_offset=0],
            copy_to_prev=False,
            allow_insert_missing=False,
            cmds_addr=Int(cmds_ptr),
        )
    else:
        var local_cache = TransformPropagationCache()
        propagate_transforms_parallel_cached(
            world_ptr[unsafe_offset=0],
            pool_ptr[unsafe_offset=0],
            local_cache,
            copy_to_prev=False,
            allow_insert_missing=False,
            cmds_addr=Int(cmds_ptr),
        )


def _has_transform_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


struct TransformPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers transform hierarchy components and parallel propagation systems in `EngineApp`."""
    var snapshot_prev_in_fixed: Bool

    def __init__(out self, snapshot_prev_in_fixed: Bool = True):
        self.snapshot_prev_in_fixed = snapshot_prev_in_fixed

    def build(self, mut app: EngineApp):
        register_transform_components(app.world)
        if not app.resources.has[TransformPropagationCache]():
            app.insert_resource[TransformPropagationCache](TransformPropagationCache())
        var b_loc = app.world.bit_for[LocalTransform]()
        var b_gt = app.world.bit_for[GlobalTransform]()
        var b_prev = app.world.bit_for[PrevGlobalTransform]()
        var b_par = app.world.bit_for[Parent]()
        var b_dep = app.world.bit_for[HierarchyDepth]()

        if self.snapshot_prev_in_fixed and not _has_transform_system(
            app, STAGE_FIXED_UPDATE, "flint.scene.snapshot_prev_transforms"
        ):
            _ = app.add_system(
                STAGE_FIXED_UPDATE,
                "flint.scene.snapshot_prev_transforms",
                _sys_snapshot_prev_transforms,
                reads=b_gt,
                writes=b_prev,
                exclusive=False,
            )

        # Non-exclusive parallel propagation: defers missing GlobalTransform insertion via
        # ParallelCommands so unrelated STAGE_POST_UPDATE systems can run concurrently.
        if not _has_transform_system(app, STAGE_POST_UPDATE, "flint.scene.propagate_transforms"):
            var prop_id = app.add_system(
                STAGE_POST_UPDATE,
                "flint.scene.propagate_transforms",
                _sys_propagate_transforms,
                reads=b_loc.union(b_par).union(b_dep),
                writes=b_gt,
                exclusive=False,
            )
            # Ensure `flint.scene.propagate_transforms` precedes `flint.net.replicate` in
            # `STAGE_POST_UPDATE` even if `NetPlugin` was registered before `TransformPlugin`.
            ref st_post = app.schedule.stages[STAGE_POST_UPDATE]
            var found_net_idx = -1
            for i in range(len(st_post.systems) - 1):
                if st_post.systems[i].name == "flint.net.replicate":
                    found_net_idx = i
                    break
            if found_net_idx >= 0:
                var last_idx = len(st_post.systems) - 1
                var prop_desc = st_post.systems[last_idx].copy()
                var k = last_idx
                while k > found_net_idx:
                    st_post.systems[k] = st_post.systems[k - 1].copy()
                    k -= 1
                st_post.systems[found_net_idx] = prop_desc^
                st_post.systems[found_net_idx + 1].explicit_after.append(prop_id)
                st_post.dirty = True
