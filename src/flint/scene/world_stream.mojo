"""Stream authored .flscn/.flscnb cells from an AssetServer into a live ECS world.

The manifest lists a cell ID, integer grid coordinates and scene path per cell.
CellActor IDs are local to their cell. Links store cell-id/actor-id and resolve
against the current entity generation whenever the target cell is resident.
With a save directory, changed cell contents are written atomically on eviction
or checkpoint. Scene parent handles remain local to a cell.
"""
from std.collections import Dict, List
from std.math import abs, floor, isfinite
from std.time import monotonic
from flint.assets.asset_ref import normalize_asset_rel_path
from flint.assets.flpk_v3 import AssetUuid, Handle
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote
from flint.assets.server import AssetServer, RawBytesAsset
from flint.core.math import Vec3
from flint.core.serial import BlobReader, BlobWriter, fnv64, fourcc
from flint.ecs.world import Entity, NULL_ENTITY, STORAGE_TABLE, World, comp_hash64
from flint.platform.os import file_size, make_dirs, read_bytes, write_file_durable
from flint.scene.scene import ComponentRegistry, Scene
from flint.scene.transform_hierarchy import SerializableComponent


struct CellActor(SerializableComponent):
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.scene.CellActor")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "CellActor"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("CACT")
    comptime SCHEMA_VERSION: Int = 1
    var id: String
    var target: String

    def __init__(out self, id: String = "", target: String = ""):
        self.id = id
        self.target = target

    def __init__(out self, *, copy: Self):
        self.id = copy.id
        self.target = copy.target

    def write_bin(self, mut writer: BlobWriter):
        writer.text(self.id)
        writer.text(self.target)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        var id = reader.text()
        var target = reader.text()
        if version != 1 or reader.failed or id.byte_length() == 0:
            raise Error("invalid CellActor")
        return Self(id, target)

    def write_json(self) -> String:
        return "{\"id\":" + json_quote(self.id) + ",\"target\":" + json_quote(self.target) + "}"

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var id = doc.text(doc.object_get(node, "id"), "")
        if id.byte_length() == 0:
            raise Error("CellActor requires an ID")
        return Self(id, doc.text(doc.object_get(node, "target"), ""))


@fieldwise_init
struct WorldCell(Copyable, Movable):
    var id: String
    var x: Int
    var z: Int
    var scene_path: String


@fieldwise_init
struct _ResidentCell(Copyable, Movable):
    var index: Int
    var entities: List[Entity]
    var actor_ids: List[String]
    var baseline: UInt64


def _release_cell_source(mut assets: AssetServer, handle: Handle[RawBytesAsset]):
    _ = assets.raws.release(handle)
    if assets.raws.ref_count(handle) == 0:
        _ = assets.raws.unload(handle)


struct WorldCellStreamer(Movable):
    var cell_size: Float32
    var radius: Int
    var save_directory: String
    var cells: List[WorldCell]
    var _grid: Dict[UInt64, Int]
    var _cell_ids: Dict[String, Int]
    var _resident: List[_ResidentCell]
    var _actors: Dict[String, Entity]
    var loads: Int
    var evictions: Int
    var saves: Int
    var peak_cell_source_bytes: Int

    def __init__(out self, cell_size: Float32 = 32, radius: Int = 1, save_directory: String = ""):
        self.cell_size = cell_size
        self.radius = radius
        self.save_directory = save_directory
        self.cells = List[WorldCell]()
        self._grid = Dict[UInt64, Int]()
        self._cell_ids = Dict[String, Int]()
        self._resident = List[_ResidentCell]()
        self._actors = Dict[String, Entity]()
        self.loads = 0
        self.evictions = 0
        self.saves = 0
        self.peak_cell_source_bytes = 0

    @staticmethod
    def _key(x: Int, z: Int) -> UInt64:
        return (UInt64(UInt32(x)) << 32) | UInt64(UInt32(z))

    def define_cell(mut self, var cell: WorldCell) raises:
        var norm = normalize_asset_rel_path(cell.scene_path)
        if (
            cell.id.byte_length() == 0 or "/" in cell.id or not norm[0] or norm[1].byte_length() == 0
            or not (norm[1].endswith(".flscn") or norm[1].endswith(".flscnb"))
            or cell.x < -2147483648 or cell.x > 2147483647 or cell.z < -2147483648 or cell.z > 2147483647
        ):
            raise Error("invalid world cell ID, coordinates or scene path")
        var key = Self._key(cell.x, cell.z)
        if key in self._grid or cell.id in self._cell_ids:
            raise Error("duplicate world cell ID or coordinates")
        cell.scene_path = norm[1]
        self._grid[key] = len(self.cells)
        self._cell_ids[cell.id] = len(self.cells)
        self.cells.append(cell^)

    @staticmethod
    def from_manifest(
        mut assets: AssetServer, path: String, radius: Int = 1, save_directory: String = "",
    ) raises -> Self:
        var handle = assets.load_raw(path)
        var data = assets.raws.get_ptr(handle)
        if not data:
            raise Error("missing world manifest: " + path)
        var doc = parse_json(data.value()[].bytes)
        _release_cell_source(assets, handle)
        var size = Float32(doc.number(doc.object_get(doc.root, "cell_size"), 0))
        var cells = doc.object_get(doc.root, "cells")
        if (
            not doc.valid or doc.text(doc.object_get(doc.root, "format"), "") != "flint-world-grid"
            or doc.integer(doc.object_get(doc.root, "version"), 0) != 1
            or not isfinite(size) or size <= 0 or radius < 0 or cells < 0 or doc.nodes[cells].kind != 4
        ):
            raise Error("invalid world grid manifest")
        var out = Self(size, radius, save_directory)
        for i in range(doc.array_size(cells)):
            var c = doc.at(cells, i)
            out.define_cell(WorldCell(
                doc.text(doc.object_get(c, "id"), ""), doc.integer(doc.object_get(c, "x"), 2147483648),
                doc.integer(doc.object_get(c, "z"), 2147483648), doc.text(doc.object_get(c, "scene"), ""),
            ))
        return out^

    def resident_count(self) -> Int:
        return len(self._resident)

    def resident_entity_count(self, world: World) -> Int:
        var count = 0
        for cell in self._resident:
            for entity in cell.entities:
                count += Int(world.is_alive(entity))
        return count

    def resolve(self, world: World, actor_id: String) -> Entity:
        var found = self._actors.get(actor_id)
        return found.value() if found and world.is_alive(found.value()) else NULL_ENTITY

    def adopt(mut self, world: World, cell_id: String, entity: Entity) raises:
        """Associate a runtime spawn with a resident cell for eviction and save."""
        if not world.is_alive(entity):
            raise Error("cannot adopt a dead entity")
        var index = self._cell_ids.get(cell_id)
        if not index:
            raise Error("unknown world cell: " + cell_id)
        var slot = -1
        for i in range(len(self._resident)):
            if self._resident[i].index == index.value():
                slot = i
            for other in self._resident[i].entities:
                if other == entity:
                    raise Error("entity already belongs to a world cell")
        if slot < 0:
            raise Error("cell is not resident: " + cell_id)
        var actor = world.get_ptr[CellActor](entity)
        if actor:
            var id = cell_id + "/" + actor.value()[].id
            if actor.value()[].id.byte_length() == 0 or self.resolve(world, id).valid():
                raise Error("invalid or duplicate actor ID in cell: " + id)
            self._actors[id] = entity
            self._resident[slot].actor_ids.append(id)
        self._resident[slot].entities.append(entity)

    def _save_path(self, index: Int) -> String:
        return self.save_directory + "/" + AssetUuid.from_path(self.cells[index].id).format() + ".flscnb"

    def _save(mut self, world: World, registry: ComponentRegistry, i: Int) raises:
        if self.save_directory.byte_length() == 0:
            return
        ref cell = self._resident[i]
        var scene = Scene(self.cells[cell.index].id)
        var data = scene.save_flscnb_entities(world, registry, cell.entities)
        var digest = fnv64(data)
        if digest != cell.baseline:
            if not make_dirs(self.save_directory) or not write_file_durable(self._save_path(cell.index), data):
                raise Error("could not save streamed cell: " + self.cells[cell.index].id)
            cell.baseline = digest
            self.saves += 1

    def checkpoint(mut self, world: World, registry: ComponentRegistry) raises:
        """Persist changes in the current window, including a partially completed transition."""
        for i in range(len(self._resident)):
            self._save(world, registry, i)

    def _unload(mut self, mut world: World, i: Int):
        ref cell = self._resident[i]
        for id in cell.actor_ids:
            _ = self._actors.pop(id, NULL_ENTITY)
        for j in range(len(cell.entities) - 1, -1, -1):
            if world.is_alive(cell.entities[j]):
                _ = world.despawn(cell.entities[j])
        _ = self._resident.pop(i)

    def unload_all(mut self, mut world: World, registry: ComponentRegistry) raises:
        self.checkpoint(world, registry)
        while len(self._resident) > 0:
            self._unload(world, len(self._resident) - 1)
            self.evictions += 1

    def _load(mut self, mut world: World, registry: ComponentRegistry, mut assets: AssetServer, idx: Int) raises:
        var scene = Scene(self.cells[idx].id)
        var members = List[Entity]()
        var saved = self._save_path(idx)
        if self.save_directory.byte_length() > 0 and file_size(saved) >= 0:
            var data = read_bytes(saved)
            self.peak_cell_source_bytes = max(self.peak_cell_source_bytes, len(data))
            members = scene.load_flscnb_bytes(world, registry, data, clear_existing=False)
        else:
            var path = self.cells[idx].scene_path
            var handle = assets.load_raw(path)
            var data = assets.raws.get_ptr(handle)
            if not data:
                raise Error("missing world cell scene: " + path)
            self.peak_cell_source_bytes = max(self.peak_cell_source_bytes, len(data.value()[].bytes))
            try:
                if path.endswith(".flscnb"):
                    members = scene.load_flscnb_bytes(world, registry, data.value()[].bytes, clear_existing=False)
                else:
                    members = scene.load_flscn_bytes(world, registry, data.value()[].bytes, clear_existing=False)
            except err:
                _release_cell_source(assets, handle)
                raise err^
            _release_cell_source(assets, handle)
        var ids = List[String]()
        for entity in members:
            var actor = world.get_ptr[CellActor](entity)
            if actor:
                var id = self.cells[idx].id + "/" + actor.value()[].id
                if id in self._actors:
                    for old in ids:
                        _ = self._actors.pop(old, NULL_ENTITY)
                    for j in range(len(members) - 1, -1, -1):
                        _ = world.despawn(members[j])
                    raise Error("duplicate live world actor: " + id)
                self._actors[id] = entity
                ids.append(id)
        var baseline = UInt64(0)
        if self.save_directory.byte_length() > 0:
            scene.name = self.cells[idx].id
            baseline = fnv64(scene.save_flscnb_entities(world, registry, members))
        self._resident.append(_ResidentCell(idx, members^, ids^, baseline))

    def update(
        mut self, mut world: World, mut registry: ComponentRegistry, mut assets: AssetServer,
        focus: Vec3, load_budget: Int = 1, time_budget_ms: Float64 = 0,
    ) raises -> Bool:
        """Advance the window by at most load_budget synchronous loads. True means ready.
        A positive time budget stops scheduling additional cells once elapsed. The first load
        and final eviction/save work can exceed the budget; each individual scene is atomic.
        """
        if self.radius < 0 or not isfinite(self.cell_size) or self.cell_size <= 0 or load_budget <= 0 or not isfinite(time_budget_ms) or time_budget_ms < 0:
            raise Error("invalid world streaming window or budget")
        var fx = floor(Float64(focus.x) / Float64(self.cell_size))
        var fz = floor(Float64(focus.z) / Float64(self.cell_size))
        if not (fx >= -2147483648 and fx <= 2147483647 and fz >= -2147483648 and fz <= 2147483647):
            raise Error("world streaming focus exceeds grid coordinates")
        var x = Int(fx)
        var z = Int(fz)
        var targets = List[Int]()
        # Wide windows on sparse maps scan the manifest. Typical windows use the coordinate index.
        if self.radius > 1024 or (2 * self.radius + 1) ** 2 > len(self.cells):
            for i in range(len(self.cells)):
                if abs(self.cells[i].x - x) <= self.radius and abs(self.cells[i].z - z) <= self.radius:
                    targets.append(i)
        else:
            for cz in range(max(-2147483648, z - self.radius), min(2147483647, z + self.radius) + 1):
                for cx in range(max(-2147483648, x - self.radius), min(2147483647, x + self.radius) + 1):
                    var hit = self._grid.get(Self._key(cx, cz))
                    if hit:
                        targets.append(hit.value())
        if registry.find_by_fourcc(CellActor.SCHEMA_FOURCC) < 0:
            registry.register[CellActor]()
        var started = monotonic() if time_budget_ms > 0 else 0
        var previous = len(self._resident)
        var pending = 0
        var added = 0
        try:
            for idx in targets:
                var present = False
                for i in range(previous):
                    if self._resident[i].index == idx:
                        present = True
                        break
                if present:
                    continue
                if added == load_budget or (
                    added > 0 and time_budget_ms > 0 and Float64(monotonic() - started) >= time_budget_ms * 1e6
                ):
                    pending += 1
                    continue
                self._load(world, registry, assets, idx)
                added += 1
            # Make outgoing edits durable before removing a single previous cell.
            for i in range(previous):
                ref cell = self.cells[self._resident[i].index]
                if abs(cell.x - x) > self.radius or abs(cell.z - z) > self.radius:
                    self._save(world, registry, i)
        except err:
            while len(self._resident) > previous:
                self._unload(world, len(self._resident) - 1)
            raise err^
        for i in range(previous - 1, -1, -1):
            ref cell = self.cells[self._resident[i].index]
            if abs(cell.x - x) > self.radius or abs(cell.z - z) > self.radius:
                self._unload(world, i)
                self.evictions += 1
        self.loads += added
        return pending == 0
