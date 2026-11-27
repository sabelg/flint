"""Stream Sunspire's lanterns and guide as the player crosses the bridge."""
from std.math import floor
from std.time import monotonic
from flint.app import EngineApp, Resource, Resources
from flint.assets.server import AssetServer
from flint.core.math import Vec3
from flint.ecs import Entity, World, comp_hash64
from flint.scene import ComponentRegistry
from flint.scene.world_stream import CellActor, WorldCellStreamer


struct BridgeCells(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("example.sunspire.BridgeCells")
    var grid: WorldCellStreamer
    var registry: ComponentRegistry
    var guide: Entity
    var last_z: Int
    var last_x: Int
    var ready: Bool
    var started_ns: Int
    var usable_ms: Float64
    var peak_cells: Int
    var peak_entities: Int
    var pending_frames: Int
    var resolved_transitions: Int
    var update_max_ms: Float64
    var failure: String

    def __init__(out self, var grid: WorldCellStreamer, var registry: ComponentRegistry):
        self.grid = grid^
        self.registry = registry^
        self.guide = Entity()
        self.last_z = 0
        self.last_x = 0
        self.ready = False
        self.started_ns = -1
        self.usable_ms = -1
        self.peak_cells = 0
        self.peak_entities = 0
        self.pending_frames = 0
        self.resolved_transitions = 0
        self.update_max_ms = 0
        self.failure = String()

    def advance(mut self, mut world: World, mut assets: AssetServer, focus: Vec3) -> Entity:
        if self.failure:
            return Entity()
        var x = Int(floor(focus.x / self.grid.cell_size))
        var z = Int(floor(focus.z / self.grid.cell_size))
        if self.ready and self.last_x == x and self.last_z == z:
            return self.guide
        var start = Int(monotonic())
        if self.started_ns < 0:
            self.started_ns = start
        self.last_x = x
        self.last_z = z
        try:
            self.ready = self.grid.update(world, self.registry, assets, focus, load_budget=1, time_budget_ms=1)
        except err:
            self.failure = String(err)
            print("[Bridge] failed to stream cell:", self.failure)
            return Entity()
        self.peak_cells = max(self.peak_cells, self.grid.resident_count())
        self.peak_entities = max(self.peak_entities, self.grid.resident_entity_count(world))
        if not self.ready:
            self.pending_frames += 1
        self.guide = Entity()
        for marker in ["south/lantern", "keystone/lantern", "watch/lantern"]:
            var entity = self.grid.resolve(world, marker)
            if entity.valid():
                var target = world.get[CellActor](entity).target
                self.guide = self.grid.resolve(world, target)
                if self.guide.valid():
                    break
        if self.ready:
            self.resolved_transitions += 1
            if self.usable_ms < 0:
                self.usable_ms = Float64(Int(monotonic()) - self.started_ns) / 1e6
        self.update_max_ms = max(self.update_max_ms, Float64(Int(monotonic()) - start) / 1e6)
        return self.guide


def setup_bridge_cells(mut app: EngineApp, var registry: ComponentRegistry) raises:
    ref assets = app.resources.get_ptr_mut[AssetServer]().value()[]
    var grid = WorldCellStreamer.from_manifest(assets, "bridge_cells/grid.json", radius=1)
    app.insert_resource(BridgeCells(grid^, registry^))


def advance_bridge_cells(mut world: World, mut resources: Resources, focus: Vec3) -> Entity:
    ref cells = resources.get_ptr_mut[BridgeCells]().value()[]
    return cells.advance(world, resources.get_ptr_mut[AssetServer]().value()[], focus)


def report_bridge_cells(app: EngineApp) raises:
    var state = app.resources.get_ptr[BridgeCells]()
    if state:
        ref cells = state.value()[]
        if cells.failure:
            raise Error(cells.failure)
        print("[Bridge] streamed cell loads/evictions", cells.grid.loads, cells.grid.evictions,
              "peak resident cells/entities", cells.peak_cells, cells.peak_entities)
        print("[Bridge] settled windows", cells.resolved_transitions, "incomplete frames", cells.pending_frames,
              "first settled window ms", cells.usable_ms, "max streaming update ms", cells.update_max_ms)
