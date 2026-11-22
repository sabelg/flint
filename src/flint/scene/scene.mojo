"""Flint v5 Scene Graph, ComponentRegistry, Nested Prefabs & `.flscn` / `.flscnb` Persistence (`flint.scene.scene`).

Provides:
- Re-exports `trait SerializableComponent(Component)` (`SCHEMA_NAME`, `SCHEMA_FOURCC`,
  `SCHEMA_VERSION`, `write_bin`, `read_bin`, `write_json`, `read_json`)
- `ComponentRegistry`: type-erased binary + JSON component codec vtable
- `Prefab`, `PrefabNode`, `PrefabOverride`, `NestedPrefabRef`: composable nested prefabs with
  per-instance component overrides and cycle-guarded instantiation
- `Scene`: deterministic `.flscn` JSON (`assets/json.mojo` + `json_write.mojo`), compressed `.flscnb`
  LZ4 binary (`BlobWriter` + `lz4_pack`), and `<0.2 ms` Play-In-Editor (`enter_pie` / `exit_pie`)
  bit-exact world snapshot & restoration.
"""
from std.collections import List
from flint.core.detsort import stable_order
from flint.core.lz4 import lz4_pack, lz4_unpack
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote, json_text
from flint.platform.os import read_bytes, write_bytes
from flint.ecs.world import World, Entity, NULL_ENTITY, BitSet256
from flint.scene.transform_hierarchy import (
    SerializableComponent,
    LocalTransform,
    GlobalTransform,
    PrevGlobalTransform,
    Parent,
    register_transform_components,
    set_parent,
)

comptime SCENE_BIN_MAGIC: UInt32 = 0x4E435346  # "FSCN"
comptime SCENE_SCHEMA_VERSION: Int = 1
comptime MAX_PREFAB_NEST_DEPTH: Int = 16


# ---------------------------------------------------------------------------
# ComponentRegistry: Type-Erased Binary + JSON Codec Vtable
# ---------------------------------------------------------------------------

comptime _CodecRegisterWorldFn = def(Int) thin -> Int
comptime _CodecHasFn = def(Int, Entity) thin -> Bool
comptime _CodecWriteBinFn = def(Int, Entity, Int) thin -> Bool
comptime _CodecReadBinFn = def(Int, Entity, Int, Int) thin -> Bool
comptime _CodecWriteJsonFn = def(Int, Entity, Int) thin -> Bool
comptime _CodecReadJsonFn = def(Int, Entity, Int, Int) thin -> Bool


def _serializable_bin_write[C: SerializableComponent](comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[C, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.begin(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
    comp.write_bin(out)
    out.end()


def _serializable_bin_read[C: SerializableComponent](comp_addr: Int, reader_addr: Int) -> Bool:
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


def _codec_register_world[C: SerializableComponent](world_addr: Int) -> Int:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    var tid = world.register_component[C]()
    comptime is_triv = C.__copy_ctor_is_trivial and C.__move_ctor_is_trivial and C.__del__is_trivial
    if not is_triv and tid >= 0 and not world.components[tid].has_custom_bin:
        world.set_component_binary_hooks(
            tid, _serializable_bin_write[C], _serializable_bin_read[C]
        )
    return tid


def _codec_has[C: SerializableComponent](world_addr: Int, entity: Entity) -> Bool:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return world.has[C](entity)


def _codec_write_bin[C: SerializableComponent](world_addr: Int, entity: Entity, writer_addr: Int) -> Bool:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    var ptr_opt = world.get_ptr[C](entity)
    if not ptr_opt:
        return False
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.begin(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
    ptr_opt.value()[unsafe_offset=0].write_bin(out)
    out.end()
    return True


def _codec_read_bin[C: SerializableComponent](
    world_addr: Int, entity: Entity, reader_addr: Int, newest_ver: Int
) -> Bool:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    try:
        var ver = reader.enter(C.SCHEMA_FOURCC, newest_ver)
        var val = C.read_bin(reader, ver)
        reader.leave()
        return world.insert[C](entity, val^)
    except:
        return False


def _codec_write_json[C: SerializableComponent](world_addr: Int, entity: Entity, str_out_addr: Int) -> Bool:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    var ptr_opt = world.get_ptr[C](entity)
    if not ptr_opt:
        return False
    ref s_out = Pointer[String, MutAnyOrigin](unsafe_from_address=str_out_addr)[unsafe_offset=0]
    s_out = ptr_opt.value()[unsafe_offset=0].write_json()
    return True


def _codec_read_json[C: SerializableComponent](
    world_addr: Int, entity: Entity, doc_addr: Int, node: Int
) -> Bool:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    ref doc = Pointer[JsonDocument, ImmutAnyOrigin](unsafe_from_address=doc_addr)[unsafe_offset=0]
    try:
        var val = C.read_json(doc, node)
        return world.insert[C](entity, val^)
    except:
        return False


@fieldwise_init
struct ComponentCodecEntry(Copyable, Movable):
    var component_hash: UInt64
    var schema_name: String
    var schema_fourcc: UInt32
    var schema_version: Int
    var reg_world_fn: _CodecRegisterWorldFn
    var has_fn: _CodecHasFn
    var write_bin_fn: _CodecWriteBinFn
    var read_bin_fn: _CodecReadBinFn
    var write_json_fn: _CodecWriteJsonFn
    var read_json_fn: _CodecReadJsonFn


struct ComponentRegistry(Movable):
    """Type-erased registry of `SerializableComponent` binary & JSON codecs."""
    var entries: List[ComponentCodecEntry]

    def __init__(out self):
        self.entries = List[ComponentCodecEntry]()

    @staticmethod
    def with_builtins() -> Self:
        var reg = Self()
        reg.register[LocalTransform]()
        reg.register[GlobalTransform]()
        reg.register[PrevGlobalTransform]()
        return reg^

    def find_by_hash(self, comp_hash: UInt64) -> Int:
        for i in range(len(self.entries)):
            if self.entries[i].component_hash == comp_hash:
                return i
        return -1

    def find_by_name(self, name: String) -> Int:
        for i in range(len(self.entries)):
            if self.entries[i].schema_name == name:
                return i
        return -1

    def find_by_fourcc(self, tag: UInt32) -> Int:
        for i in range(len(self.entries)):
            if self.entries[i].schema_fourcc == tag:
                return i
        return -1

    def register[C: SerializableComponent](mut self):
        if self.find_by_hash(C.COMPONENT_ID) >= 0:
            return
        self.entries.append(
            ComponentCodecEntry(
                C.COMPONENT_ID,
                String(C.SCHEMA_NAME),
                C.SCHEMA_FOURCC,
                C.SCHEMA_VERSION,
                _codec_register_world[C],
                _codec_has[C],
                _codec_write_bin[C],
                _codec_read_bin[C],
                _codec_write_json[C],
                _codec_read_json[C],
            )
        )

    def ensure_registered_in_world(self, mut world: World):
        register_transform_components(world)
        var w_addr = Int(Pointer(to=world))
        for i in range(len(self.entries)):
            var fn_ptr = self.entries[i].reg_world_fn
            _ = fn_ptr(w_addr)

    def sorted_entry_indices(self) -> List[Int]:
        """Returns deterministic schema-hash-sorted indices across `self.entries`."""
        var keys = List[Int](capacity=len(self.entries))
        for i in range(len(self.entries)):
            keys.append(Int(self.entries[i].schema_fourcc))
        return stable_order(keys)

    def has_on_entity(self, idx: Int, world: World, entity: Entity) -> Bool:
        var fn_ptr = self.entries[idx].has_fn
        return fn_ptr(Int(Pointer(to=world)), entity)

    def write_entity_bin(self, idx: Int, world: World, entity: Entity, mut out: BlobWriter) -> Bool:
        var fn_ptr = self.entries[idx].write_bin_fn
        return fn_ptr(Int(Pointer(to=world)), entity, Int(Pointer(to=out)))

    def read_entity_bin(self, idx: Int, mut world: World, entity: Entity, mut reader: BlobReader) -> Bool:
        var fn_ptr = self.entries[idx].read_bin_fn
        return fn_ptr(
            Int(Pointer(to=world)),
            entity,
            Int(Pointer(to=reader)),
            self.entries[idx].schema_version,
        )

    def write_entity_json(self, idx: Int, world: World, entity: Entity, mut out_json: String) -> Bool:
        var fn_ptr = self.entries[idx].write_json_fn
        return fn_ptr(Int(Pointer(to=world)), entity, Int(Pointer(to=out_json)))

    def read_entity_json(
        self, idx: Int, mut world: World, entity: Entity, doc: JsonDocument, node: Int
    ) -> Bool:
        var fn_ptr = self.entries[idx].read_json_fn
        return fn_ptr(Int(Pointer(to=world)), entity, Int(Pointer(to=doc)), node)


# ---------------------------------------------------------------------------
# Prefab, PrefabOverride & NestedPrefabRef
# ---------------------------------------------------------------------------

@fieldwise_init
struct PrefabOverride(Copyable, Movable):
    """Per-instance component override targeting a prefab node by slot index or node name."""
    var target_slot: Int
    var target_name: String
    var schema_name: String
    var json_payload: String

    @staticmethod
    def for_slot[C: SerializableComponent](slot: Int, comp: C) -> Self:
        return Self(slot, "", String(C.SCHEMA_NAME), comp.write_json())

    @staticmethod
    def for_name[C: SerializableComponent](node_name: String, comp: C) -> Self:
        return Self(-1, node_name, String(C.SCHEMA_NAME), comp.write_json())


@fieldwise_init
struct NestedPrefabRef(Copyable, Movable):
    """Embeds a child `Prefab` under `attach_parent_slot` with local root offset and overrides."""
    var prefab_name: String
    var attach_parent_slot: Int
    var local_offset: LocalTransform
    var overrides: List[PrefabOverride]

    def __init__(
        out self,
        prefab_name: String,
        attach_parent_slot: Int = -1,
        local_offset: LocalTransform = LocalTransform.identity(),
    ):
        self.prefab_name = prefab_name
        self.attach_parent_slot = attach_parent_slot
        self.local_offset = local_offset
        self.overrides = List[PrefabOverride]()

    def add_override(mut self, var ov: PrefabOverride):
        self.overrides.append(ov^)


struct PrefabNode(Copyable, Movable):
    var name: String
    var parent_slot: Int
    var schema_names: List[String]
    var json_payloads: List[String]

    def __init__(out self, name: String, parent_slot: Int = -1):
        self.name = name
        self.parent_slot = parent_slot
        self.schema_names = List[String]()
        self.json_payloads = List[String]()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.parent_slot = copy.parent_slot
        self.schema_names = copy.schema_names.copy()
        self.json_payloads = copy.json_payloads.copy()

    def set_raw_component(mut self, schema_name: String, payload: String):
        for i in range(len(self.schema_names)):
            if self.schema_names[i] == schema_name:
                self.json_payloads[i] = payload
                return
        self.schema_names.append(schema_name)
        self.json_payloads.append(payload)


def _apply_json_component_str(
    mut world: World,
    registry: ComponentRegistry,
    entity: Entity,
    schema_name: String,
    json_str: String,
) raises:
    var cidx = registry.find_by_name(schema_name)
    if cidx < 0:
        raise Error("unregistered component in prefab/scene: " + schema_name)
    var bytes = List[UInt8]()
    for b in json_str.as_bytes():
        bytes.append(b)
    var doc = parse_json(bytes)
    if not doc.valid:
        raise Error("invalid component JSON payload for " + schema_name)
    if not registry.read_entity_json(cidx, world, entity, doc, doc.root):
        raise Error("failed to decode component JSON for " + schema_name)


struct Prefab(Copyable, Movable):
    """Reusable hierarchical entity template supporting nested prefabs and overrides."""
    var name: String
    var nodes: List[PrefabNode]
    var nested_refs: List[NestedPrefabRef]

    def __init__(out self, name: String):
        self.name = name
        self.nodes = List[PrefabNode]()
        self.nested_refs = List[NestedPrefabRef]()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.nodes = copy.nodes.copy()
        self.nested_refs = copy.nested_refs.copy()

    def add_node(mut self, name: String, parent_slot: Int = -1) -> Int:
        var idx = len(self.nodes)
        self.nodes.append(PrefabNode(name, parent_slot))
        return idx

    def set_component[C: SerializableComponent](mut self, node_slot: Int, comp: C):
        if node_slot >= 0 and node_slot < len(self.nodes):
            self.nodes[node_slot].set_raw_component(String(C.SCHEMA_NAME), comp.write_json())

    def add_nested(mut self, var nested_ref: NestedPrefabRef):
        self.nested_refs.append(nested_ref^)

    def instantiate(
        self,
        mut world: World,
        registry: ComponentRegistry,
        library: List[Prefab] = List[Prefab](),
        overrides: List[PrefabOverride] = List[PrefabOverride](),
        root_parent: Entity = NULL_ENTITY,
    ) raises -> List[Entity]:
        registry.ensure_registered_in_world(world)
        return self._instantiate_impl(world, registry, library, overrides, root_parent, 0)

    def _instantiate_impl(
        self,
        mut world: World,
        registry: ComponentRegistry,
        library: List[Prefab],
        overrides: List[PrefabOverride],
        root_parent: Entity,
        depth: Int,
    ) raises -> List[Entity]:
        if depth > MAX_PREFAB_NEST_DEPTH:
            raise Error("nested prefab recursion exceeded MAX_PREFAB_NEST_DEPTH")

        var spawned = List[Entity](capacity=len(self.nodes))
        var spawned_names = List[String](capacity=len(self.nodes))
        for i in range(len(self.nodes)):
            var e = world.spawn_empty()
            spawned.append(e)
            spawned_names.append(self.nodes[i].name)
            for c in range(len(self.nodes[i].schema_names)):
                _apply_json_component_str(
                    world,
                    registry,
                    e,
                    self.nodes[i].schema_names[c],
                    self.nodes[i].json_payloads[c],
                )

        # Wire internal hierarchy + optional attachment to root_parent
        for i in range(len(self.nodes)):
            var p_slot = self.nodes[i].parent_slot
            if p_slot >= 0 and p_slot < len(spawned):
                _ = set_parent(world, spawned[i], spawned[p_slot])
            elif root_parent.valid() and world.is_alive(root_parent):
                _ = set_parent(world, spawned[i], root_parent)

        # Instantiate nested prefabs
        for n_idx in range(len(self.nested_refs)):
            ref nref = self.nested_refs[n_idx]
            var found_idx = -1
            for k in range(len(library)):
                if library[k].name == nref.prefab_name:
                    found_idx = k
                    break
            if found_idx < 0:
                raise Error("nested prefab not found in library: " + nref.prefab_name)
            var attach_ent = root_parent
            if nref.attach_parent_slot >= 0 and nref.attach_parent_slot < len(spawned):
                attach_ent = spawned[nref.attach_parent_slot]

            var child_base = len(spawned)
            var child_ents = library[found_idx]._instantiate_impl(
                world, registry, library, nref.overrides, attach_ent, depth + 1
            )
            # Apply local_offset to nested prefab root nodes (nodes whose parent_slot < 0)
            for ci in range(len(library[found_idx].nodes)):
                if library[found_idx].nodes[ci].parent_slot < 0 and ci < len(child_ents):
                    var re = child_ents[ci]
                    var lt_opt = world.get_ptr_mut[LocalTransform](re)
                    if lt_opt:
                        var lt_ptr = lt_opt.value()
                        var cur = lt_ptr[unsafe_offset=0]
                        lt_ptr[unsafe_offset=0] = LocalTransform(
                            nref.local_offset.translation + nref.local_offset.rotation.rotate(cur.translation),
                            (nref.local_offset.rotation * cur.rotation).normalized(),
                            cur.scale,
                        )
                    else:
                        _ = world.insert[LocalTransform](re, nref.local_offset)
                if ci < len(child_ents):
                    spawned.append(child_ents[ci])
                    spawned_names.append(library[found_idx].nodes[ci].name)
            for extra in range(len(library[found_idx].nodes), len(child_ents)):
                spawned.append(child_ents[extra])
                spawned_names.append("")
            _ = child_base

        # Apply caller overrides (overrides win over base and nested defaults)
        for ov_i in range(len(overrides)):
            ref ov = overrides[ov_i]
            var target_e = NULL_ENTITY
            if ov.target_slot >= 0 and ov.target_slot < len(spawned):
                target_e = spawned[ov.target_slot]
            elif ov.target_name.byte_length() > 0:
                for si in range(len(spawned_names)):
                    if spawned_names[si] == ov.target_name:
                        target_e = spawned[si]
                        break
            if target_e.valid():
                _apply_json_component_str(world, registry, target_e, ov.schema_name, ov.json_payload)

        return spawned^


# ---------------------------------------------------------------------------
# Scene: .flscn Deterministic JSON + .flscnb LZ4 Binary + PIE Snapshot/Restore
# ---------------------------------------------------------------------------

def _string_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _collect_sorted_world_entities(world: World) -> List[Entity]:
    """Returns all live entities ordered deterministically by ascending `Entity.index`."""
    var out = List[Entity](capacity=world.entity_count())
    for idx in range(len(world.entities)):
        if world.entities[idx].alive:
            out.append(Entity(UInt32(idx), world.entities[idx].generation))
    return out^


struct Scene(Movable):
    """Flint v5 Scene container supporting `.flscn` JSON, `.flscnb` LZ4 binary, and PIE snapshots."""
    var name: String
    var pie_snapshot: List[UInt8]
    var has_pie_snapshot: Bool

    def __init__(out self, name: String = "UntitledScene"):
        self.name = name
        self.pie_snapshot = List[UInt8]()
        self.has_pie_snapshot = False

    # ---- Instant Play-In-Editor (PIE) World Snapshot / Restore (<0.2 ms) ----

    def enter_pie(mut self, world: World) raises:
        """Captures an instant binary snapshot of `world` prior to Play-In-Editor simulation."""
        var writer = BlobWriter()
        world.snapshot_blob(writer)
        self.pie_snapshot = List[UInt8]()
        swap(self.pie_snapshot, writer.data)
        self.has_pie_snapshot = True

    def exit_pie(self, mut world: World) raises:
        """Restores `world` bit-for-bit to its pre-PIE state in `<0.2 ms`."""
        if not self.has_pie_snapshot or len(self.pie_snapshot) == 0:
            raise Error("Scene.exit_pie called without an active PIE snapshot")
        var reader = BlobReader(self.pie_snapshot.copy())
        world.restore_blob(reader)

    # ---- .flscn Deterministic JSON Format (via assets/json.mojo + json_write.mojo) ----

    def save_flscn_string(self, world: World, registry: ComponentRegistry) raises -> String:
        """Serializes `world` into canonical, deterministic `.flscn` JSON text."""
        var ents = _collect_sorted_world_entities(world)
        var max_idx = len(world.entities)
        var ent_to_slot = List[Int](length=max(1, max_idx), fill=-1)
        for slot in range(len(ents)):
            ent_to_slot[Int(ents[slot].index)] = slot

        var sorted_codecs = registry.sorted_entry_indices()

        var raw = String("{\"format\": \"flscn\", \"version\": ") + String(SCENE_SCHEMA_VERSION)
        raw += ", \"name\": " + json_quote(self.name) + ", \"entities\": ["

        for slot in range(len(ents)):
            if slot > 0:
                raw += ", "
            var e = ents[slot]
            var parent_slot = -1
            var p_opt = world.get_ptr[Parent](e)
            if p_opt:
                var pe = p_opt.value()[unsafe_offset=0].entity
                if world.is_alive(pe) and Int(pe.index) < len(ent_to_slot):
                    parent_slot = ent_to_slot[Int(pe.index)]

            raw += "{\"slot\": " + String(slot) + ", \"parent\": " + String(parent_slot) + ", \"components\": {"
            var wrote_comp = 0
            for ci in range(len(sorted_codecs)):
                var cidx = sorted_codecs[ci]
                if registry.has_on_entity(cidx, world, e):
                    var comp_json = String()
                    if registry.write_entity_json(cidx, world, e, comp_json):
                        if wrote_comp > 0:
                            raw += ", "
                        raw += json_quote(registry.entries[cidx].schema_name) + ": " + comp_json
                        wrote_comp += 1
            raw += "}}"
        raw += "]}"

        # Canonicalize & validate through assets/json.mojo + assets/json_write.mojo
        var doc = parse_json(_string_to_bytes(raw))
        if not doc.valid:
            raise Error("generated .flscn JSON failed strict validation")
        return json_text(doc, doc.root) + "\n"

    def save_flscn_bytes(self, world: World, registry: ComponentRegistry) raises -> List[UInt8]:
        return _string_to_bytes(self.save_flscn_string(world, registry))

    def save_flscn_file(self, path: String, world: World, registry: ComponentRegistry) raises -> Bool:
        var data = self.save_flscn_bytes(world, registry)
        return write_bytes(path, data)

    def load_flscn_bytes(
        mut self,
        mut world: World,
        registry: ComponentRegistry,
        data: List[UInt8],
        clear_existing: Bool = True,
    ) raises -> List[Entity]:
        """Parses `.flscn` JSON and populates `world`, reconstructing components and hierarchy."""
        var doc = parse_json(data)
        if not doc.valid:
            raise Error("invalid .flscn JSON document")
        var fmt = doc.text(doc.object_get(doc.root, "format"), "")
        var ver = doc.integer(doc.object_get(doc.root, "version"), -1)
        if fmt != "flscn" or ver < 1 or ver > SCENE_SCHEMA_VERSION:
            raise Error("unsupported .flscn header or version")

        self.name = doc.text(doc.object_get(doc.root, "name"), self.name)
        registry.ensure_registered_in_world(world)
        if clear_existing:
            world.clear_entities()

        var ents_arr = doc.object_get(doc.root, "entities")
        var count = doc.array_size(ents_arr)
        var spawned = List[Entity](capacity=count)
        var parent_slots = List[Int](length=count, fill=-1)

        for _ in range(count):
            spawned.append(world.spawn_empty())

        for i in range(count):
            var enode = doc.at(ents_arr, i)
            parent_slots[i] = doc.integer(doc.object_get(enode, "parent"), -1)
            var comps_obj = doc.object_get(enode, "components")
            if comps_obj >= 0 and doc.nodes[comps_obj].kind == 5:
                var num_keys = len(doc.nodes[comps_obj].keys)
                for k in range(num_keys):
                    var sname = doc.nodes[comps_obj].keys[k]
                    var cnode = doc.nodes[comps_obj].children[k]
                    var cidx = registry.find_by_name(sname)
                    if cidx < 0:
                        raise Error("unknown component in .flscn: " + sname)
                    if not registry.read_entity_json(cidx, world, spawned[i], doc, cnode):
                        raise Error("failed to decode .flscn component: " + sname)

        for i in range(count):
            var p = parent_slots[i]
            if p >= 0 and p < count:
                if not set_parent(world, spawned[i], spawned[p]):
                    raise Error("cycle or invalid parent in .flscn scene")

        return spawned^

    def load_flscn_file(
        mut self,
        path: String,
        mut world: World,
        registry: ComponentRegistry,
        clear_existing: Bool = True,
    ) raises -> List[Entity]:
        var data = read_bytes(path)
        if len(data) == 0:
            raise Error("failed to read .flscn file: " + path)
        return self.load_flscn_bytes(world, registry, data, clear_existing)

    # ---- .flscnb LZ4 Binary Format (BlobWriter + lz4_pack) ----

    def save_flscnb_bytes(self, world: World, registry: ComponentRegistry) -> List[UInt8]:
        """Serializes `world` into versioned `BlobWriter` chunks and compresses with `lz4_pack`."""
        var ents = _collect_sorted_world_entities(world)
        var max_idx = len(world.entities)
        var ent_to_slot = List[Int](length=max(1, max_idx), fill=-1)
        for slot in range(len(ents)):
            ent_to_slot[Int(ents[slot].index)] = slot

        var sorted_codecs = registry.sorted_entry_indices()
        var writer = BlobWriter()
        writer.begin(SCENE_BIN_MAGIC, SCENE_SCHEMA_VERSION)
        writer.text(self.name)
        writer.integer(len(ents))

        for slot in range(len(ents)):
            var e = ents[slot]
            var parent_slot = -1
            var p_opt = world.get_ptr[Parent](e)
            if p_opt:
                var pe = p_opt.value()[unsafe_offset=0].entity
                if world.is_alive(pe) and Int(pe.index) < len(ent_to_slot):
                    parent_slot = ent_to_slot[Int(pe.index)]
            writer.integer(parent_slot)

            var present_indices = List[Int]()
            for ci in range(len(sorted_codecs)):
                var cidx = sorted_codecs[ci]
                if registry.has_on_entity(cidx, world, e):
                    present_indices.append(cidx)

            writer.integer(len(present_indices))
            for pi in range(len(present_indices)):
                _ = registry.write_entity_bin(present_indices[pi], world, e, writer)

        writer.end()
        return lz4_pack(writer.data)

    def save_flscnb_file(self, path: String, world: World, registry: ComponentRegistry) -> Bool:
        var packed = self.save_flscnb_bytes(world, registry)
        return write_bytes(path, packed)

    def load_flscnb_bytes(
        mut self,
        mut world: World,
        registry: ComponentRegistry,
        packed: List[UInt8],
        clear_existing: Bool = True,
    ) raises -> List[Entity]:
        """Decompresses and validates `.flscnb` LZ4 binary scene into `world`."""
        var raw = lz4_unpack(packed)
        if len(raw) == 0:
            raise Error("corrupted or invalid LZ4 frame in .flscnb")
        var reader = BlobReader(raw^)
        _ = reader.enter(SCENE_BIN_MAGIC, SCENE_SCHEMA_VERSION)
        self.name = reader.text()
        var count = reader.integer()
        if (
            reader.failed
            or count < 0
            or count > 1 << 20
            or count > max(0, reader.limit() - reader.offset) // 16
        ):
            raise Error("corrupted entity count in .flscnb")

        registry.ensure_registered_in_world(world)
        if clear_existing:
            world.clear_entities()

        var spawned = List[Entity](capacity=count)
        var parent_slots = List[Int](length=count, fill=-1)
        for _ in range(count):
            spawned.append(world.spawn_empty())

        for slot in range(count):
            parent_slots[slot] = reader.integer()
            var num_comps = reader.integer()
            if (
                reader.failed
                or num_comps < 0
                or num_comps > 65536
                or num_comps > max(0, reader.limit() - reader.offset) // 12
            ):
                raise Error("corrupted component count in .flscnb")
            for _ in range(num_comps):
                var tag = reader.peek_tag()
                var cidx = registry.find_by_fourcc(tag)
                if cidx < 0:
                    raise Error("unregistered component FourCC in .flscnb")
                if not registry.read_entity_bin(cidx, world, spawned[slot], reader):
                    raise Error("failed to decode binary component in .flscnb")

        reader.leave()

        for slot in range(count):
            var p = parent_slots[slot]
            if p >= 0 and p < count:
                if not set_parent(world, spawned[slot], spawned[p]):
                    raise Error("cycle or invalid parent in .flscnb")

        return spawned^

    def load_flscnb_file(
        mut self,
        path: String,
        mut world: World,
        registry: ComponentRegistry,
        clear_existing: Bool = True,
    ) raises -> List[Entity]:
        var packed = read_bytes(path)
        if len(packed) == 0:
            raise Error("failed to read .flscnb file: " + path)
        return self.load_flscnb_bytes(world, registry, packed, clear_existing)
