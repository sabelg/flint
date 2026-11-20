"""Flint v5 Render Plugin & ECS Component Schemas (`flint.render.plugin`).

Provides:
- `RenderPluginStateResource`: `Resource` owning `RenderExtractor`, `UiDrawList`,
  optional `TerrainClipmapPacket`, and frame render telemetry.
- 5 POD `SerializableComponent` ECS schemas:
  `MeshRendererComponent`, `CameraComponent`, `PointLightComponent`,
  `MaterialOverrideComponent`, and `VatCrowdComponent`.
- `register_render_components(mut world: World)` & `register_render_codecs(mut reg: ComponentRegistry)`.
- `RenderPlugin`: inserts `RenderWorld` directly as a `Resource`, `ScalableRenderer`
  directly as a `Resource`, and `RenderPluginStateResource`; extracts interpolated
  `PrevGlobalTransform -> GlobalTransform` poses, cameras, lights, materials, and
  VAT crowd instances in `STAGE_LAST`, calls `renderer.render_world(...)` and
  clears `ui_draw_list`; calls `renderer.shutdown()` deterministically in `STAGE_SHUTDOWN`.
"""
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    FixedTime,
    FrameTime,
    Resource,
    Resources,
    STAGE_LAST,
    STAGE_SHUTDOWN,
)
from flint.assets.json import JsonDocument
from flint.core.jobs import JobPool
from flint.core.math import Mat4, Vec3
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.ecs.world import (
    Entity,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.render_world import (
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    RenderCameraPacket,
    RenderExtractor,
    RenderWorld,
    ScalableRenderer,
    TerrainClipmapPacket,
)
from flint.scene.scene import ComponentRegistry
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    PrevGlobalTransform,
    SerializableComponent,
    interpolate_global_transform,
)
from flint.ui.draw_list import UiDrawList


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

struct RenderPluginStateResource(Resource, Movable):
    """Owns `RenderExtractor`, `UiDrawList`, optional `TerrainClipmapPacket`, and stats."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.render.RenderPluginStateResource")
    var extractor: RenderExtractor
    var ui_draw_list: UiDrawList
    var terrain_clipmap: TerrainClipmapPacket
    var rendered_frames: UInt64
    var last_render_ok: Bool

    def __init__(out self):
        self.extractor = RenderExtractor()
        self.ui_draw_list = UiDrawList()
        self.terrain_clipmap = TerrainClipmapPacket()
        self.rendered_frames = 0
        self.last_render_ok = True


# ---------------------------------------------------------------------------
# 5 Serializable ECS Render Components
# ---------------------------------------------------------------------------

struct MeshRendererComponent(SerializableComponent, ImplicitlyCopyable):
    """Renderable mesh instance referencing registered `mesh_id` & `material_id`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.MeshRendererComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "MeshRendererComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RMSH")
    comptime SCHEMA_VERSION: Int = 1

    var mesh_id: Int
    var material_id: Int
    var cast_shadows: Bool
    var visible: Bool

    def __init__(
        out self,
        mesh_id: Int = 0,
        material_id: Int = -1,
        cast_shadows: Bool = True,
        visible: Bool = True,
    ):
        self.mesh_id = mesh_id
        self.material_id = material_id
        self.cast_shadows = cast_shadows
        self.visible = visible

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.mesh_id)
        out.integer(self.material_id)
        out.boolean(self.cast_shadows)
        out.boolean(self.visible)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var mid = reader.integer()
        var mat = reader.integer()
        var cs = reader.boolean()
        var vis = reader.boolean()
        return Self(mid, mat, cs, vis)

    def write_json(self) -> String:
        return (
            "{\"mesh_id\": " + String(self.mesh_id)
            + ", \"material_id\": " + String(self.material_id)
            + ", \"cast_shadows\": " + ("true" if self.cast_shadows else "false")
            + ", \"visible\": " + ("true" if self.visible else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var mid = doc.integer(doc.object_get(node, "mesh_id"), 0)
        var mat = doc.integer(doc.object_get(node, "material_id"), -1)
        var cs = doc.boolean(doc.object_get(node, "cast_shadows"), True)
        var vis = doc.boolean(doc.object_get(node, "visible"), True)
        return Self(mid, mat, cs, vis)


struct CameraComponent(SerializableComponent, ImplicitlyCopyable):
    """Perspective camera component driven by entity `GlobalTransform` (`-Z` forward)."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.CameraComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "CameraComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RCAM")
    comptime SCHEMA_VERSION: Int = 1

    var fov_y_deg: Float32
    var near_z: Float32
    var far_z: Float32
    var exposure: Float32
    var active: Bool
    var look_target_offset: Vec3

    def __init__(
        out self,
        fov_y_deg: Float32 = 55.0,
        near_z: Float32 = 0.10,
        far_z: Float32 = 1000.0,
        exposure: Float32 = 1.0,
        active: Bool = True,
        look_target_offset: Vec3 = Vec3(0.0, 0.0, -1.0),
    ):
        self.fov_y_deg = fov_y_deg
        self.near_z = near_z
        self.far_z = far_z
        self.exposure = exposure
        self.active = active
        self.look_target_offset = look_target_offset

    def write_bin(self, mut out: BlobWriter):
        out.f32(self.fov_y_deg)
        out.f32(self.near_z)
        out.f32(self.far_z)
        out.f32(self.exposure)
        out.boolean(self.active)
        out.vec3(self.look_target_offset)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var fov = reader.f32()
        var nz = reader.f32()
        var fz = reader.f32()
        var exp = reader.f32()
        var act = reader.boolean()
        var off = reader.vec3()
        return Self(fov, nz, fz, exp, act, off)

    def write_json(self) -> String:
        return (
            "{\"fov_y_deg\": " + _f32_s(self.fov_y_deg)
            + ", \"near_z\": " + _f32_s(self.near_z)
            + ", \"far_z\": " + _f32_s(self.far_z)
            + ", \"exposure\": " + _f32_s(self.exposure)
            + ", \"active\": " + ("true" if self.active else "false")
            + ", \"look_target_offset\": " + _vec3_json(self.look_target_offset)
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var fov = Float32(doc.number(doc.object_get(node, "fov_y_deg"), 55.0))
        var nz = Float32(doc.number(doc.object_get(node, "near_z"), 0.10))
        var fz = Float32(doc.number(doc.object_get(node, "far_z"), 1000.0))
        var exp = Float32(doc.number(doc.object_get(node, "exposure"), 1.0))
        var act = doc.boolean(doc.object_get(node, "active"), True)
        var off = _read_vec3_json(doc, doc.object_get(node, "look_target_offset"), Vec3(0.0, 0.0, -1.0))
        return Self(fov, nz, fz, exp, act, off)


struct PointLightComponent(SerializableComponent, ImplicitlyCopyable):
    """Clustered point light extracted as `AaaClusterLight.point`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.PointLightComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "PointLightComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RLGT")
    comptime SCHEMA_VERSION: Int = 1

    var radiance: Vec3
    var radius: Float32
    var enabled: Bool

    def __init__(
        out self,
        radiance: Vec3 = Vec3(4.0, 4.0, 4.0),
        radius: Float32 = 12.0,
        enabled: Bool = True,
    ):
        self.radiance = radiance
        self.radius = radius
        self.enabled = enabled

    def write_bin(self, mut out: BlobWriter):
        out.vec3(self.radiance)
        out.f32(self.radius)
        out.boolean(self.enabled)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var rad = reader.vec3()
        var r = reader.f32()
        var en = reader.boolean()
        return Self(rad, r, en)

    def write_json(self) -> String:
        return (
            "{\"radiance\": " + _vec3_json(self.radiance)
            + ", \"radius\": " + _f32_s(self.radius)
            + ", \"enabled\": " + ("true" if self.enabled else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var rad = _read_vec3_json(doc, doc.object_get(node, "radiance"), Vec3(4.0, 4.0, 4.0))
        var r = Float32(doc.number(doc.object_get(node, "radius"), 12.0))
        var en = doc.boolean(doc.object_get(node, "enabled"), True)
        return Self(rad, r, en)


struct MaterialOverrideComponent(SerializableComponent, ImplicitlyCopyable):
    """Per-entity material ID and PBR parameter override."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.MaterialOverrideComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "MaterialOverrideComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RMAT")
    comptime SCHEMA_VERSION: Int = 1

    var material_id: Int
    var base_color: Vec3
    var roughness: Float32
    var metallic: Float32
    var emissive: Vec3
    var enabled: Bool

    def __init__(
        out self,
        material_id: Int = -1,
        base_color: Vec3 = Vec3(0.8, 0.8, 0.8),
        roughness: Float32 = 0.35,
        metallic: Float32 = 0.0,
        emissive: Vec3 = Vec3(),
        enabled: Bool = True,
    ):
        self.material_id = material_id
        self.base_color = base_color
        self.roughness = roughness
        self.metallic = metallic
        self.emissive = emissive
        self.enabled = enabled

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.material_id)
        out.vec3(self.base_color)
        out.f32(self.roughness)
        out.f32(self.metallic)
        out.vec3(self.emissive)
        out.boolean(self.enabled)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var mid = reader.integer()
        var bc = reader.vec3()
        var rg = reader.f32()
        var mt = reader.f32()
        var em = reader.vec3()
        var en = reader.boolean()
        return Self(mid, bc, rg, mt, em, en)

    def write_json(self) -> String:
        return (
            "{\"material_id\": " + String(self.material_id)
            + ", \"base_color\": " + _vec3_json(self.base_color)
            + ", \"roughness\": " + _f32_s(self.roughness)
            + ", \"metallic\": " + _f32_s(self.metallic)
            + ", \"emissive\": " + _vec3_json(self.emissive)
            + ", \"enabled\": " + ("true" if self.enabled else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var mid = doc.integer(doc.object_get(node, "material_id"), -1)
        var bc = _read_vec3_json(doc, doc.object_get(node, "base_color"), Vec3(0.8, 0.8, 0.8))
        var rg = Float32(doc.number(doc.object_get(node, "roughness"), 0.35))
        var mt = Float32(doc.number(doc.object_get(node, "metallic"), 0.0))
        var em = _read_vec3_json(doc, doc.object_get(node, "emissive"), Vec3())
        var en = doc.boolean(doc.object_get(node, "enabled"), True)
        return Self(mid, bc, rg, mt, em, en)


struct VatCrowdComponent(SerializableComponent, ImplicitlyCopyable):
    """Baked Vertex-Animation-Texture crowd instance component."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.VatCrowdComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "VatCrowdComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RVAT")
    comptime SCHEMA_VERSION: Int = 1

    var clip_id: Int
    var phase_sec: Float32
    var prev_phase_sec: Float32
    var material_id: Int
    var visible: Bool

    def __init__(
        out self,
        clip_id: Int = 0,
        phase_sec: Float32 = 0.0,
        prev_phase_sec: Float32 = 0.0,
        material_id: Int = -1,
        visible: Bool = True,
    ):
        self.clip_id = clip_id
        self.phase_sec = phase_sec
        self.prev_phase_sec = prev_phase_sec
        self.material_id = material_id
        self.visible = visible

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.clip_id)
        out.f32(self.phase_sec)
        out.f32(self.prev_phase_sec)
        out.integer(self.material_id)
        out.boolean(self.visible)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var cid = reader.integer()
        var ph = reader.f32()
        var pph = reader.f32()
        var mat = reader.integer()
        var vis = reader.boolean()
        return Self(cid, ph, pph, mat, vis)

    def write_json(self) -> String:
        return (
            "{\"clip_id\": " + String(self.clip_id)
            + ", \"phase_sec\": " + _f32_s(self.phase_sec)
            + ", \"material_id\": " + String(self.material_id)
            + ", \"visible\": " + ("true" if self.visible else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var cid = doc.integer(doc.object_get(node, "clip_id"), 0)
        var ph = Float32(doc.number(doc.object_get(node, "phase_sec"), 0.0))
        var mat = doc.integer(doc.object_get(node, "material_id"), -1)
        var vis = doc.boolean(doc.object_get(node, "visible"), True)
        return Self(cid, ph, ph, mat, vis)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_render_components(mut world: World):
    """Idempotently registers all 5 render components on `world`."""
    _ = world.register_component[MeshRendererComponent]()
    _ = world.register_component[CameraComponent]()
    _ = world.register_component[PointLightComponent]()
    _ = world.register_component[MaterialOverrideComponent]()
    _ = world.register_component[VatCrowdComponent]()


def register_render_codecs(mut reg: ComponentRegistry):
    """Registers all 5 render `SerializableComponent` codecs on `ComponentRegistry`."""
    reg.register[MeshRendererComponent]()
    reg.register[CameraComponent]()
    reg.register[PointLightComponent]()
    reg.register[MaterialOverrideComponent]()
    reg.register[VatCrowdComponent]()


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def _interpolated_gt(world: World, entity: Entity, alpha: Float32) -> GlobalTransform:
    var gt_opt = world.get_ptr[GlobalTransform](entity)
    if not gt_opt:
        return GlobalTransform()
    ref curr = gt_opt.value()[unsafe_offset=0]
    var prev_opt = world.get_ptr[PrevGlobalTransform](entity)
    if prev_opt:
        return interpolate_global_transform(prev_opt.value()[unsafe_offset=0].value, curr, alpha)
    return curr.copy()


def _sys_render_extract_and_draw(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var rw_opt = resources.get_ptr_mut[RenderWorld]()
    var rend_opt = resources.get_ptr_mut[ScalableRenderer]()
    var st_opt = resources.get_ptr_mut[RenderPluginStateResource]()
    if not rw_opt or not rend_opt or not st_opt:
        return

    ref render_world = rw_opt.value()[unsafe_offset=0]
    ref renderer = rend_opt.value()[unsafe_offset=0]
    ref state = st_opt.value()[unsafe_offset=0]

    var frame_idx = UInt64(0)
    var frame_dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        frame_idx = ft_opt.value()[unsafe_offset=0].frame_index
        frame_dt = ft_opt.value()[unsafe_offset=0].delta_seconds

    var alpha = Float32(1.0)
    var fix_opt = resources.get_ptr[FixedTime]()
    if fix_opt:
        alpha = fix_opt.value()[unsafe_offset=0].overstep_alpha

    state.extractor.begin_extraction(render_world, frame_idx, frame_dt)
    if state.terrain_clipmap.active:
        render_world.terrain_clipmap = state.terrain_clipmap.copy()

    # 1. Extract active CameraComponent
    var cam_mask = world.bit_for[CameraComponent]().union(world.bit_for[GlobalTransform]())
    var cam_entities = world.query_entities(cam_mask)
    for i in range(len(cam_entities)):
        var e = cam_entities[i]
        var cam_opt = world.get_ptr[CameraComponent](e)
        if not cam_opt or not cam_opt.value()[unsafe_offset=0].active:
            continue
        ref cam = cam_opt.value()[unsafe_offset=0]
        var gt = _interpolated_gt(world, e, alpha)
        var eye = gt.translation
        var off = cam.look_target_offset
        if off.length_squared() <= Float32(1.0e-8):
            off = Vec3(0.0, 0.0, -1.0)
        var target = eye + gt.rotation.rotate(off)
        var up = gt.rotation.rotate(Vec3(0.0, 1.0, 0.0)).normalized()
        render_world.camera = RenderCameraPacket(
            eye, target, up, cam.fov_y_deg, cam.near_z, cam.far_z, cam.exposure
        )
        break

    # 2. Extract MeshRendererComponents (+ MaterialOverrideComponent)
    var mesh_mask = world.bit_for[MeshRendererComponent]().union(world.bit_for[GlobalTransform]())
    var mesh_entities = world.query_entities(mesh_mask)
    for i in range(len(mesh_entities)):
        var e = mesh_entities[i]
        var mc_opt = world.get_ptr[MeshRendererComponent](e)
        if not mc_opt or not mc_opt.value()[unsafe_offset=0].visible:
            continue
        ref mc = mc_opt.value()[unsafe_offset=0]
        var eff_mat = mc.material_id
        var ov_opt = world.get_ptr[MaterialOverrideComponent](e)
        if ov_opt and ov_opt.value()[unsafe_offset=0].enabled:
            ref ov = ov_opt.value()[unsafe_offset=0]
            if ov.material_id >= 0:
                eff_mat = ov.material_id
            if eff_mat >= 0 and eff_mat < len(renderer.materials):
                renderer.materials[eff_mat].base_color = ov.base_color
                renderer.materials[eff_mat].roughness = ov.roughness
                renderer.materials[eff_mat].metallic = ov.metallic
                renderer.materials[eff_mat].emissive = ov.emissive
                if renderer.uses_vulkan():
                    renderer.gpu_slot[0].update_material_asset(eff_mat, renderer.materials[eff_mat])

        var gt = _interpolated_gt(world, e, alpha)
        state.extractor.extract_entity(
            render_world,
            e.to_bits(),
            mc.mesh_id,
            gt.matrix.copy(),
            eff_mat,
            mc.cast_shadows,
        )

    # 3. Extract VatCrowdComponents
    var vat_mask = world.bit_for[VatCrowdComponent]().union(world.bit_for[GlobalTransform]())
    var vat_entities = world.query_entities(vat_mask)
    for i in range(len(vat_entities)):
        var e = vat_entities[i]
        var vc_opt = world.get_ptr[VatCrowdComponent](e)
        if not vc_opt or not vc_opt.value()[unsafe_offset=0].visible:
            continue
        ref vc = vc_opt.value()[unsafe_offset=0]
        var gt = _interpolated_gt(world, e, alpha)
        state.extractor.extract_vat_crowd_entity(
            render_world,
            e.to_bits(),
            vc.clip_id,
            vc.phase_sec,
            vc.prev_phase_sec,
            gt.matrix.copy(),
            vc.material_id,
        )

    # 4. Extract PointLightComponents
    var lgt_mask = world.bit_for[PointLightComponent]().union(world.bit_for[GlobalTransform]())
    var lgt_entities = world.query_entities(lgt_mask)
    for i in range(len(lgt_entities)):
        var e = lgt_entities[i]
        var lc_opt = world.get_ptr[PointLightComponent](e)
        if not lc_opt or not lc_opt.value()[unsafe_offset=0].enabled:
            continue
        ref lc = lc_opt.value()[unsafe_offset=0]
        var gt = _interpolated_gt(world, e, alpha)
        render_world.cluster_lights.append(
            AaaClusterLight.point(gt.translation, lc.radiance, lc.radius)
        )

    state.extractor.end_extraction()
    state.last_render_ok = renderer.render_world(render_world, state.ui_draw_list)
    state.rendered_frames += 1
    state.ui_draw_list.clear()


def _sys_render_shutdown(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var rend_opt = resources.get_ptr_mut[ScalableRenderer]()
    if rend_opt:
        rend_opt.value()[unsafe_offset=0].shutdown()


def _has_render_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


struct RenderPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers `RenderWorld`, `ScalableRenderer`, render ECS schemas, and `STAGE_LAST` drawing."""
    var tier: UInt32
    var width: Int
    var height: Int
    var tsr_scale: Float32

    def __init__(
        out self,
        tier: UInt32 = RENDER_TIER_2_AAA_VISBUFFER_RT,
        width: Int = 640,
        height: Int = 360,
        tsr_scale: Float32 = 1.0,
    ):
        self.tier = tier
        self.width = width
        self.height = height
        self.tsr_scale = tsr_scale

    def build(self, mut app: EngineApp):
        register_render_components(app.world)
        if not app.resources.has[RenderWorld]():
            app.insert_resource[RenderWorld](RenderWorld())
        if not app.resources.has[ScalableRenderer]():
            app.insert_resource[ScalableRenderer](
                ScalableRenderer(
                    tier=self.tier,
                    width=self.width,
                    height=self.height,
                    tsr_scale=self.tsr_scale,
                )
            )
        if not app.resources.has[RenderPluginStateResource]():
            app.insert_resource[RenderPluginStateResource](RenderPluginStateResource())

        var b_gt = app.world.bit_for[GlobalTransform]()
        var b_prev = app.world.bit_for[PrevGlobalTransform]()
        var b_mesh = app.world.bit_for[MeshRendererComponent]()
        var b_cam = app.world.bit_for[CameraComponent]()
        var b_lgt = app.world.bit_for[PointLightComponent]()
        var b_mat = app.world.bit_for[MaterialOverrideComponent]()
        var b_vat = app.world.bit_for[VatCrowdComponent]()
        var all_reads = (
            b_gt.union(b_prev).union(b_mesh).union(b_cam)
            .union(b_lgt).union(b_mat).union(b_vat)
        )

        if not _has_render_system(app, STAGE_LAST, "flint.render.extract_and_draw"):
            _ = app.add_system(
                STAGE_LAST,
                "flint.render.extract_and_draw",
                _sys_render_extract_and_draw,
                reads=all_reads,
                exclusive=True,
            )
        if not _has_render_system(app, STAGE_SHUTDOWN, "flint.render.shutdown"):
            _ = app.add_system(
                STAGE_SHUTDOWN,
                "flint.render.shutdown",
                _sys_render_shutdown,
                exclusive=True,
            )
