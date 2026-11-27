"""Flint v5 Navigation Plugin & ECS Component Schemas (`flint.nav.plugin`).

Provides:
- `NavPluginStateResource`: `Resource` owning `NavMesh`, `CrowdSimulator`,
  `CrowdNavBridge`, `baked`, `draw_debug`, `last_poly_count`, and entity-to-bridge map.
- 2 POD `SerializableComponent` ECS schemas:
  `NavMeshComponent` (`"NAVM"`) and `NavAgentComponent` (`"NAVA"`).
- `register_nav_components(mut world: World)` & `register_nav_codecs(mut reg: ComponentRegistry)`.
- `bake_from_physics(mut world: World, cfg: NavMeshConfig = NavMeshConfig()) -> Int`.
- `NavPlugin(draw_debug=True)`: registers nav components, `NavPluginStateResource`,
  `STAGE_FIXED_UPDATE` ORCA/NavMesh crowd stepping (`flint.nav.crowd_step`), and
  `STAGE_POST_UPDATE` wireframe polygon extraction to `RenderWorld.overlay_lines`
  (`flint.nav.debug_extract`) before `RenderPlugin` draws `STAGE_LAST`.
"""
from std.collections import Dict, List
from std.math import isfinite, max
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    FixedTime,
    Resource,
    Resources,
    STAGE_FIXED_UPDATE,
    STAGE_POST_UPDATE,
)
from flint.assets.json import JsonDocument
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.ecs.world import (
    Entity,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.nav.navmesh import NavMesh, NavMeshConfig
from flint.nav.orca import CrowdNavBridge, CrowdSimulator
from flint.physics.narrowphase import ConvexHull
from flint.physics.plugin import ColliderComponent, RigidBodyComponent
from flint.physics.world import (
    BODY_MOTION_DYNAMIC,
    BODY_SHAPE_BOX,
    BODY_SHAPE_CONVEX_HULL,
    PhysicsWorld,
)
from flint.render.render_world import RenderWorld
from flint.scene.scene import ComponentRegistry
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    LocalTransform,
    Parent,
    SerializableComponent,
)


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


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------

struct NavPluginStateResource(Resource, Movable):
    """Owns runtime `NavMesh`, `CrowdSimulator`, `CrowdNavBridge`, and debug overlay state."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.nav.NavPluginStateResource")
    var navmesh: NavMesh
    var crowd: CrowdSimulator
    var bridge: CrowdNavBridge
    var baked: Bool
    var draw_debug: Bool
    var last_poly_count: Int
    var entity_to_bridge: Dict[UInt64, Int]

    def __init__(out self, draw_debug: Bool = True):
        self.navmesh = NavMesh()
        self.crowd = CrowdSimulator()
        self.bridge = CrowdNavBridge()
        self.baked = False
        self.draw_debug = draw_debug
        self.last_poly_count = 0
        self.entity_to_bridge = Dict[UInt64, Int]()

    def bake_from_physics(
        mut self,
        pw: PhysicsWorld,
        cfg: NavMeshConfig = NavMeshConfig(),
    ) -> Int:
        self.navmesh = NavMesh.build_from_physics(pw, cfg)
        self.crowd = CrowdSimulator()
        self.bridge = CrowdNavBridge()
        self.entity_to_bridge.clear()
        self.last_poly_count = self.navmesh.poly_count()
        self.baked = self.last_poly_count > 0
        return self.last_poly_count


# ---------------------------------------------------------------------------
# 2 Serializable ECS Nav Components
# ---------------------------------------------------------------------------

struct NavMeshComponent(SerializableComponent, ImplicitlyCopyable):
    """NavMesh bake settings and debug visualization toggle."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.nav.NavMeshComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "NavMeshComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("NAVM")
    comptime SCHEMA_VERSION: Int = 1

    var cell_size: Float32
    var cell_height: Float32
    var walkable_slope_deg: Float32
    var walkable_height: Float32
    var walkable_climb: Float32
    var walkable_radius: Float32
    var max_edge_len: Float32
    var max_simplification_error: Float32
    var min_region_cells: Int
    var tile_size: Int
    var max_verts_per_poly: Int
    var enabled: Bool
    var draw_debug: Bool
    var auto_bake: Bool

    def __init__(
        out self,
        cell_size: Float32 = 0.25,
        cell_height: Float32 = 0.10,
        walkable_slope_deg: Float32 = 46.0,
        walkable_height: Float32 = 1.80,
        walkable_climb: Float32 = 0.40,
        walkable_radius: Float32 = 0.35,
        max_edge_len: Float32 = 6.0,
        max_simplification_error: Float32 = 0.35,
        min_region_cells: Int = 3,
        tile_size: Int = 16,
        max_verts_per_poly: Int = 6,
        enabled: Bool = True,
        draw_debug: Bool = True,
        auto_bake: Bool = False,
    ):
        self.cell_size = cell_size
        self.cell_height = cell_height
        self.walkable_slope_deg = walkable_slope_deg
        self.walkable_height = walkable_height
        self.walkable_climb = walkable_climb
        self.walkable_radius = walkable_radius
        self.max_edge_len = max_edge_len
        self.max_simplification_error = max_simplification_error
        self.min_region_cells = min_region_cells
        self.tile_size = tile_size
        self.max_verts_per_poly = max_verts_per_poly
        self.enabled = enabled
        self.draw_debug = draw_debug
        self.auto_bake = auto_bake

    def to_config(self) -> NavMeshConfig:
        var c = NavMeshConfig()
        c.cell_size = max(Float32(0.05), self.cell_size) if isfinite(self.cell_size) else Float32(0.25)
        c.cell_height = max(Float32(0.02), self.cell_height) if isfinite(self.cell_height) else Float32(0.10)
        c.walkable_slope_deg = self.walkable_slope_deg if isfinite(self.walkable_slope_deg) else Float32(46.0)
        c.walkable_height = self.walkable_height if isfinite(self.walkable_height) else Float32(1.80)
        c.walkable_climb = self.walkable_climb if isfinite(self.walkable_climb) else Float32(0.40)
        c.walkable_radius = self.walkable_radius if isfinite(self.walkable_radius) else Float32(0.35)
        c.max_edge_len = self.max_edge_len if isfinite(self.max_edge_len) else Float32(6.0)
        c.max_simplification_error = (
            self.max_simplification_error if isfinite(self.max_simplification_error) else Float32(0.35)
        )
        c.min_region_cells = max(1, self.min_region_cells)
        c.tile_size = max(8, self.tile_size)
        c.max_verts_per_poly = max(3, self.max_verts_per_poly)
        return c

    def write_bin(self, mut out: BlobWriter):
        out.f32(self.cell_size)
        out.f32(self.cell_height)
        out.f32(self.walkable_slope_deg)
        out.f32(self.walkable_height)
        out.f32(self.walkable_climb)
        out.f32(self.walkable_radius)
        out.f32(self.max_edge_len)
        out.f32(self.max_simplification_error)
        out.integer(self.min_region_cells)
        out.integer(self.tile_size)
        out.integer(self.max_verts_per_poly)
        out.boolean(self.enabled)
        out.boolean(self.draw_debug)
        out.boolean(self.auto_bake)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var cs = reader.f32()
        var ch = reader.f32()
        var ws = reader.f32()
        var wh = reader.f32()
        var wc = reader.f32()
        var wr = reader.f32()
        var mel = reader.f32()
        var mse = reader.f32()
        var mrc = reader.integer()
        var ts = reader.integer()
        var mvp = reader.integer()
        var en = reader.boolean()
        var dd = reader.boolean()
        var ab = reader.boolean()
        if reader.failed:
            raise Error("NavMeshComponent: truncated blob")
        return Self(cs, ch, ws, wh, wc, wr, mel, mse, mrc, ts, mvp, en, dd, ab)

    def write_json(self) -> String:
        return (
            "{\"cell_size\": " + _f32_s(self.cell_size)
            + ", \"cell_height\": " + _f32_s(self.cell_height)
            + ", \"walkable_slope_deg\": " + _f32_s(self.walkable_slope_deg)
            + ", \"walkable_height\": " + _f32_s(self.walkable_height)
            + ", \"walkable_climb\": " + _f32_s(self.walkable_climb)
            + ", \"walkable_radius\": " + _f32_s(self.walkable_radius)
            + ", \"max_edge_len\": " + _f32_s(self.max_edge_len)
            + ", \"max_simplification_error\": " + _f32_s(self.max_simplification_error)
            + ", \"min_region_cells\": " + String(self.min_region_cells)
            + ", \"tile_size\": " + String(self.tile_size)
            + ", \"max_verts_per_poly\": " + String(self.max_verts_per_poly)
            + ", \"enabled\": " + ("true" if self.enabled else "false")
            + ", \"draw_debug\": " + ("true" if self.draw_debug else "false")
            + ", \"auto_bake\": " + ("true" if self.auto_bake else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var cs = Float32(doc.number(doc.object_get(node, "cell_size"), 0.25))
        var ch = Float32(doc.number(doc.object_get(node, "cell_height"), 0.10))
        var ws = Float32(doc.number(doc.object_get(node, "walkable_slope_deg"), 46.0))
        var wh = Float32(doc.number(doc.object_get(node, "walkable_height"), 1.80))
        var wc = Float32(doc.number(doc.object_get(node, "walkable_climb"), 0.40))
        var wr = Float32(doc.number(doc.object_get(node, "walkable_radius"), 0.35))
        var mel = Float32(doc.number(doc.object_get(node, "max_edge_len"), 6.0))
        var mse = Float32(doc.number(doc.object_get(node, "max_simplification_error"), 0.35))
        var mrc = doc.integer(doc.object_get(node, "min_region_cells"), 3)
        var ts = doc.integer(doc.object_get(node, "tile_size"), 16)
        var mvp = doc.integer(doc.object_get(node, "max_verts_per_poly"), 6)
        var en = doc.boolean(doc.object_get(node, "enabled"), True)
        var dd = doc.boolean(doc.object_get(node, "draw_debug"), True)
        var ab = doc.boolean(doc.object_get(node, "auto_bake"), False)
        return Self(cs, ch, ws, wh, wc, wr, mel, mse, mrc, ts, mvp, en, dd, ab)


struct NavAgentComponent(SerializableComponent, ImplicitlyCopyable):
    """Autonomous crowd steering agent synchronized via `CrowdNavBridge`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.nav.NavAgentComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "NavAgentComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("NAVA")
    comptime SCHEMA_VERSION: Int = 1

    var goal: Vec3
    var has_goal: Bool
    var max_speed: Float32
    var radius: Float32
    var arrival_radius: Float32
    var waypoint_radius: Float32
    var enabled: Bool
    var agent_id: Int
    var bridge_idx: Int

    def __init__(
        out self,
        goal: Vec3 = Vec3(),
        has_goal: Bool = False,
        max_speed: Float32 = 3.5,
        radius: Float32 = 0.35,
        arrival_radius: Float32 = 0.45,
        waypoint_radius: Float32 = 0.65,
        enabled: Bool = True,
        agent_id: Int = -1,
        bridge_idx: Int = -1,
    ):
        self.goal = goal
        self.has_goal = has_goal
        self.max_speed = max_speed
        self.radius = radius
        self.arrival_radius = arrival_radius
        self.waypoint_radius = waypoint_radius
        self.enabled = enabled
        self.agent_id = agent_id
        self.bridge_idx = bridge_idx

    def write_bin(self, mut out: BlobWriter):
        out.vec3(self.goal)
        out.boolean(self.has_goal)
        out.f32(self.max_speed)
        out.f32(self.radius)
        out.f32(self.arrival_radius)
        out.f32(self.waypoint_radius)
        out.boolean(self.enabled)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var gl = reader.vec3()
        var hg = reader.boolean()
        var ms = reader.f32()
        var rad = reader.f32()
        var ar = reader.f32()
        var wr = reader.f32()
        var en = reader.boolean()
        if reader.failed:
            raise Error("NavAgentComponent: truncated blob")
        return Self(gl, hg, ms, rad, ar, wr, en, -1, -1)

    def write_json(self) -> String:
        return (
            "{\"goal\": " + _vec3_json(self.goal)
            + ", \"has_goal\": " + ("true" if self.has_goal else "false")
            + ", \"max_speed\": " + _f32_s(self.max_speed)
            + ", \"radius\": " + _f32_s(self.radius)
            + ", \"arrival_radius\": " + _f32_s(self.arrival_radius)
            + ", \"waypoint_radius\": " + _f32_s(self.waypoint_radius)
            + ", \"enabled\": " + ("true" if self.enabled else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var gl = _read_vec3_json(doc, doc.object_get(node, "goal"), Vec3())
        var hg = doc.boolean(doc.object_get(node, "has_goal"), False)
        var ms = Float32(doc.number(doc.object_get(node, "max_speed"), 3.5))
        var rad = Float32(doc.number(doc.object_get(node, "radius"), 0.35))
        var ar = Float32(doc.number(doc.object_get(node, "arrival_radius"), 0.45))
        var wr = Float32(doc.number(doc.object_get(node, "waypoint_radius"), 0.65))
        var en = doc.boolean(doc.object_get(node, "enabled"), True)
        return Self(gl, hg, ms, rad, ar, wr, en, -1, -1)


# ---------------------------------------------------------------------------
# Registration & Bake Helpers
# ---------------------------------------------------------------------------

def register_nav_components(mut world: World):
    """Idempotently registers `NavMeshComponent` and `NavAgentComponent` on `world`."""
    _ = world.register_component[NavMeshComponent]()
    _ = world.register_component[NavAgentComponent]()


def register_nav_codecs(mut reg: ComponentRegistry):
    """Registers `NavMeshComponent` and `NavAgentComponent` codecs on `ComponentRegistry`."""
    reg.register[NavMeshComponent]()
    reg.register[NavAgentComponent]()


def _build_physics_world_from_ecs(mut world: World) -> PhysicsWorld:
    var pw = PhysicsWorld(gravity=0.0)
    pw.begin_body_batch()
    var col_mask = world.bit_for[ColliderComponent]()
    var col_ents = world.query_entities(col_mask)
    for i in range(len(col_ents)):
        var e = col_ents[i]
        var col_opt = world.get_ptr[ColliderComponent](e)
        if not col_opt:
            continue
        ref col = col_opt.value()[unsafe_offset=0]
        if col.is_trigger:
            continue
        var rb_opt = world.get_ptr[RigidBodyComponent](e)
        if rb_opt:
            ref rb = rb_opt.value()[unsafe_offset=0]
            if not rb.active or rb.motion_mode == BODY_MOTION_DYNAMIC:
                continue
        var pos = Vec3()
        var rot = Quat.identity()
        var gt_opt = world.get_ptr[GlobalTransform](e)
        if gt_opt:
            pos = gt_opt.value()[unsafe_offset=0].translation
            rot = gt_opt.value()[unsafe_offset=0].rotation
        else:
            var lt_opt = world.get_ptr[LocalTransform](e)
            if lt_opt:
                pos = lt_opt.value()[unsafe_offset=0].translation
                rot = lt_opt.value()[unsafe_offset=0].rotation
        var w_pos = pos + rot.rotate(col.local_offset)
        var w_rot = (rot * col.local_rot).normalized()
        if col.shape == BODY_SHAPE_CONVEX_HULL and len(col.points) >= 4:
            var hull = ConvexHull.build_quickhull(col.points)
            _ = pw.add_convex_hull(w_pos, hull, Float32(0.0), w_rot)
        elif col.shape == BODY_SHAPE_BOX:
            _ = pw.add_box(w_pos, col.half_extents, Float32(0.0), w_rot)
    pw.end_body_batch()
    return pw^


def bake_from_physics(
    mut world: World,
    cfg: NavMeshConfig = NavMeshConfig(),
) -> Int:
    """Builds a `NavMesh` from static/kinematic `ColliderComponent` entities in `world`
    and returns the resulting polygon count."""
    var pw = _build_physics_world_from_ecs(world)
    var nm = NavMesh.build_from_physics(pw, cfg)
    return nm.poly_count()


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def _sys_nav_crowd_step(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[NavPluginStateResource]()
    if not st_opt:
        return
    ref state = st_opt.value()[unsafe_offset=0]

    if not state.baked:
        var nm_mask = world.bit_for[NavMeshComponent]()
        var nm_ents = world.query_entities(nm_mask)
        for i in range(len(nm_ents)):
            var nm_opt = world.get_ptr[NavMeshComponent](nm_ents[i])
            if not nm_opt:
                continue
            ref nmc = nm_opt.value()[unsafe_offset=0]
            if nmc.enabled and nmc.auto_bake:
                var pw_opt = resources.get_ptr[PhysicsWorld]()
                if pw_opt:
                    _ = state.bake_from_physics(pw_opt.value()[unsafe_offset=0], nmc.to_config())
                else:
                    var ecs_pw = _build_physics_world_from_ecs(world)
                    _ = state.bake_from_physics(ecs_pw, nmc.to_config())
                break

    # Reconcile the registered generation before stepping or allocating new slots.
    var released = List[UInt64]()
    for bits in state.entity_to_bridge.keys():
        var e = Entity.from_bits(bits)
        var ac = world.get_ptr_mut[NavAgentComponent](e)
        if ac and ac.value()[].enabled and world.has[LocalTransform](e):
            continue
        if ac:
            ac.value()[].bridge_idx = -1
            ac.value()[].agent_id = -1
        released.append(bits)
    for bits in released:
        var idx = state.entity_to_bridge.pop(bits, -1)
        state.bridge.remove_agent(idx, state.crowd)

    if not state.baked or state.navmesh.poly_count() == 0:
        return

    var dt = Float32(1.0 / 60.0)
    var fix_opt = resources.get_ptr[FixedTime]()
    if fix_opt and fix_opt.value()[unsafe_offset=0].dt > Float32(1.0e-5):
        dt = fix_opt.value()[unsafe_offset=0].dt

    var ag_mask = world.bit_for[NavAgentComponent]().union(world.bit_for[LocalTransform]())
    var ag_ents = world.query_entities(ag_mask)
    for i in range(len(ag_ents)):
        var e = ag_ents[i]
        var ac_opt = world.get_ptr_mut[NavAgentComponent](e)
        var lt_opt = world.get_ptr[LocalTransform](e)
        if not ac_opt or not lt_opt:
            continue
        ref ac = ac_opt.value()[unsafe_offset=0]
        if not ac.enabled:
            continue
        var bits = e.to_bits()
        var pos = lt_opt.value()[unsafe_offset=0].translation
        var gt_opt = world.get_ptr[GlobalTransform](e)
        if gt_opt:
            pos = gt_opt.value()[unsafe_offset=0].translation

        var b_found = state.entity_to_bridge.get(bits)
        if b_found:
            ac.bridge_idx = b_found.value()
        else:
            ac.bridge_idx = -1

        if ac.bridge_idx < 0 or ac.bridge_idx >= len(state.bridge.agents):
            var tgt = ac.goal if ac.has_goal else pos
            var b_idx = state.bridge.add_agent(
                state.crowd,
                state.navmesh,
                pos,
                tgt,
                ac.radius,
                ac.max_speed,
                ac.arrival_radius,
                ac.waypoint_radius,
            )
            ac.bridge_idx = b_idx
            ac.agent_id = state.bridge.agents[b_idx].agent_id
            state.entity_to_bridge[bits] = b_idx
        else:
            ac.agent_id = state.bridge.agents[ac.bridge_idx].agent_id
            if ac.has_goal:
                ref bag = state.bridge.agents[ac.bridge_idx]
                if (bag.goal - ac.goal).length_squared() > Float32(1.0e-6):
                    state.bridge.set_goal(ac.bridge_idx, state.crowd, state.navmesh, ac.goal)

    if len(state.bridge.agents) > 0:
        state.bridge.update_and_step(state.crowd, state.navmesh, dt)
        for i in range(len(ag_ents)):
            var e = ag_ents[i]
            var ac_opt = world.get_ptr[NavAgentComponent](e)
            if not ac_opt or not ac_opt.value()[unsafe_offset=0].enabled:
                continue
            var aid = ac_opt.value()[unsafe_offset=0].agent_id
            if aid < 0 or aid >= state.crowd.count:
                continue
            var new_pos = state.crowd.agent_position(aid)
            var lt_mut = world.get_ptr_mut[LocalTransform](e)
            if lt_mut and not world.has[Parent](e):
                lt_mut.value()[unsafe_offset=0].translation = new_pos
                var gt_mut = world.get_ptr_mut[GlobalTransform](e)
                if gt_mut:
                    gt_mut.value()[unsafe_offset=0].translation = new_pos


def _sys_nav_debug_extract(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr[NavPluginStateResource]()
    var rw_opt = resources.get_ptr_mut[RenderWorld]()
    if not st_opt or not rw_opt:
        return
    ref state = st_opt.value()[unsafe_offset=0]
    var should_draw = state.draw_debug
    var nm_mask = world.bit_for[NavMeshComponent]()
    var nm_ents = world.query_entities(nm_mask)
    for i in range(len(nm_ents)):
        var nm_opt = world.get_ptr[NavMeshComponent](nm_ents[i])
        if nm_opt and nm_opt.value()[unsafe_offset=0].enabled:
            should_draw = should_draw and nm_opt.value()[unsafe_offset=0].draw_debug
            break
    if not should_draw or len(state.navmesh.polys) == 0:
        return

    ref ol = rw_opt.value()[unsafe_offset=0].overlay_lines
    var lift = Vec3(0.0, 0.03, 0.0)
    var edge_col = UInt32(0xFF38E898)
    for pi in range(len(state.navmesh.polys)):
        ref poly = state.navmesh.polys[pi]
        var nv = len(poly.verts)
        if nv < 2:
            continue
        for vi in range(nv):
            var a = poly.verts[vi] + lift
            var b = poly.verts[(vi + 1) % nv] + lift
            ol.add_line(a, b, edge_col, Float32(1.5))


def _has_nav_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

struct NavPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers `NavMeshComponent`, `NavAgentComponent`, `NavPluginStateResource`,
    crowd stepping in `STAGE_FIXED_UPDATE`, and NavMesh debug overlay extraction."""
    var draw_debug: Bool

    def __init__(out self, draw_debug: Bool = True):
        self.draw_debug = draw_debug

    def build(self, mut app: EngineApp):
        register_nav_components(app.world)
        if not app.resources.has[NavPluginStateResource]():
            app.insert_resource[NavPluginStateResource](NavPluginStateResource(self.draw_debug))

        var b_lt = app.world.bit_for[LocalTransform]()
        var b_gt = app.world.bit_for[GlobalTransform]()
        var b_nm = app.world.bit_for[NavMeshComponent]()
        var b_ag = app.world.bit_for[NavAgentComponent]()

        if not _has_nav_system(app, STAGE_FIXED_UPDATE, "flint.nav.crowd_step"):
            _ = app.add_system(
                STAGE_FIXED_UPDATE,
                "flint.nav.crowd_step",
                _sys_nav_crowd_step,
                reads=b_nm,
                writes=b_ag.union(b_lt).union(b_gt),
                exclusive=True,
            )
        if not _has_nav_system(app, STAGE_POST_UPDATE, "flint.nav.debug_extract"):
            _ = app.add_system(
                STAGE_POST_UPDATE,
                "flint.nav.debug_extract",
                _sys_nav_debug_extract,
                reads=b_nm,
                exclusive=True,
            )
